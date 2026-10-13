"""Belief-aware play (§3.1); evolution is deliberately outside act()."""

from __future__ import annotations

from copy import deepcopy
import re

from base.interfaces import BaseMethod
from base.model_output import OutputTruncated
from .policy import PolicyState, RESPONSE_PROTOCOL, canonical, string_schema
from .fallback import (EmptyModelResponse, KnownModelFailure, check_policy,
                       failed_attempt, failure_reason, fallback_payload, selection_context)
from .retry import (DecisionInvalid, RETRY_PROTOCOL, RETRY_STAGES, attempt_record,
                    correction_prompt, parse_decision, retry_count)

RETURN_FORMAT = ('return format:\n'
                 '{"self_belief": "your belief here",\n'
                 '"world_belief": "your belief here",\n'
                 '"action": "valid action"}')
OUTPUT_INSTRUCTIONS = (
    "Return exactly one JSON object with all three fields: self_belief, world_belief, action. "
    "Write your actual beliefs in the first two fields (nonempty strings, each <=6000 characters). "
    "Replace valid action with exactly one STRING from valid_actions. "
    "A decision mentioned inside a belief does not replace the separate action field. "
    "Do not omit action. No extra fields, Markdown fences or surrounding prose. "
    "For Leduc use the listed action word. For dice, bid_Q_F means quantity Q of face F; "
    "challenge means challenge. For Goofspiel, bid_N means play card N. "
    "Choose from the listed valid_actions; do not output placeholder text."
)

PRIVATE = {"leduc": ("card",), "liars_dice": ("my_dice",), "goofspiel": ("hand", "my_bids")}
PUBLIC = {
    "leduc": ("community_card", "pot", "street", "phase", "total_bets", "round_bets", "to_call", "round_raises"),
    "liars_dice": ("total_dice_remaining", "my_dice_count", "opponent_dice_count", "last_bid", "round"),
    "goofspiel": ("round", "prize", "prize_sequence", "scores", "outcome_history"),
}
HISTORY = {
    "leduc": ("player", "action", "amount", "street"),
    "liars_dice": ("player", "action", "quantity", "face", "round", "result", "bid_quantity", "bid_face", "actual_count", "loser", "revealed_dice"),
    "goofspiel": ("round", "prize", "p0_card", "p1_card", "outcome", "player", "action"),
}


def visible_view(game: str, obs) -> dict:
    """Whitelist structured player observations, not state_text or diagnostics."""
    return deepcopy({"player_id": obs.player_id,
                     "private_info": {k: obs.private_info[k] for k in PRIVATE[game] if k in obs.private_info},
                     "public_info": {k: obs.public_info[k] for k in PUBLIC[game] if k in obs.public_info},
                     "history": [{k: row[k] for k in HISTORY[game] if k in row} for row in obs.history]})


def invoke(model, stage: str, prompt: str) -> str:
    """Direct model injection remains safe even when a journal is not supplied."""
    fn = getattr(model, "generate_stage", None)
    raw = fn(stage, prompt) if callable(fn) else model.generate(prompt)
    if getattr(model, "last_finish_reason", None) == "length":
        raise OutputTruncated(raw if isinstance(raw, str) else "")
    if type(raw) is not str or not raw.strip():
        raise EmptyModelResponse(raw)
    return raw


def action_token(game, action):
    """Exact action label with all decision parameters, not a guessed default."""
    if game == "liars_dice" and action.type == "bid":
        return f"bid_{action.data['quantity']}_{action.data['face']}"
    if game == "goofspiel" and action.type == "bid":
        return f"bid_{action.amount}"
    return action.type


def parse_action(game, value, legal_actions):
    """Resolve only an explicit, unique legal choice; never infer from beliefs.

    Full objects remain exact (no ignored fields, amounts or player coercions).
    Leduc types encode a unique legal amount supplied by the engine; dice bids
    still require quantity/face, and a Goofspiel alias must name its exact card.
    """
    if type(value) is dict:
        matches = [a for a in legal_actions if canonical(a.model_dump()) == canonical(value)]
    elif type(value) is str:
        token = value.strip().lower()
        if game == "leduc" and token in {"fold", "check", "call", "bet", "raise"}:
            matches = [a for a in legal_actions if a.type == token]
        elif game == "liars_dice" and token == "challenge":
            matches = [a for a in legal_actions if a.type == "challenge"]
        elif game == "liars_dice" and re.fullmatch(r"bid_[1-9][0-9]*_[1-9][0-9]*", token):
            matches = [a for a in legal_actions if a.type == "bid" and action_token(game, a) == token]
        elif game == "goofspiel" and re.fullmatch(r"bid_[1-9][0-9]*", token):
            matches = [a for a in legal_actions if a.type == "bid" and str(a.amount) == token[4:]]
        else:
            matches = []
    else:
        matches = []
    if len(matches) != 1:
        raise ValueError("Action must identify exactly one current legal action; no guessing or repair")
    return matches[0]


def render_decision_prompt(policy, observation, legal_actions, belief):
    tokens = [action_token(policy.game, a) for a in legal_actions]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Legal action labels must be unique")
    schema = string_schema(("self_belief", "world_belief"), limit=6000)
    schema["required"].append("action")
    schema["properties"]["action"] = {"type": "string", "enum": tokens}
    context = {
        "task": "Agent-Pro belief-aware decision. Update self-belief (own resources, plan and risk), then world-belief (opponent hypotheses, intentions and their possible view of you), then choose one legal action. Distinguish observations from conjectures. Never assume hidden information is known. In simultaneous bidding you cannot observe the opponent's current bid.",
        "game": policy.game, "authoritative_rules": observation.rules_text,
        "policy": policy.instructions(), "observation": visible_view(policy.game, observation),
        "previous_belief": belief, "legal_actions": [a.model_dump() for a in legal_actions],
        "valid_actions": tokens, "response_protocol": RESPONSE_PROTOCOL, "output_schema": schema,
    }
    # Keep structured state, but NEVER serialize output instructions into sorted
    # context fields again. The exact user-requested template is the final text.
    return canonical(context) + "\n\n" + OUTPUT_INSTRUCTIONS + "\n\n" + RETURN_FORMAT


class AgentProMethod(BaseMethod):
    name = "Agent-Pro"
    method_type = "agent_pro"

    def __init__(self, policy: PolicyState, player_id: int, parse_retries: int = 3,
                 *, action_failure_policy="strict", fallback_seed=0):
        if type(player_id) is not int or player_id not in (0, 1):
            raise ValueError("Expected P0 or P1")
        self.parse_retries = retry_count(parse_retries)
        self.action_failure_policy = check_policy(action_failure_policy)
        if type(fallback_seed) is not int:
            raise ValueError("Fallback seed must be an integer")
        self.fallback_seed = fallback_seed
        self.policy = policy
        self.player_id = player_id
        self.reset_game()

    def reset_game(self):
        self.belief = {"self_belief": "", "world_belief": ""}
        self.records: list[dict] = []

    def act(self, observation, legal_actions, model):
        if observation.player_id != self.player_id or not legal_actions or any(a.player_id != self.player_id for a in legal_actions):
            raise ValueError("Agent-Pro requires its own nonempty legal action set")
        view = visible_view(self.policy.game, observation)
        prompt = render_decision_prompt(self.policy, observation, legal_actions, self.belief)
        configure_failure = getattr(model, "configure_action_failure", None)
        if callable(configure_failure):
            configure_failure(self.action_failure_policy)
        configure = getattr(model, "configure_decision_retries", None)
        if callable(configure):
            configure(self.parse_retries)
        original = prompt
        attempts = []
        context = selection_context(self.policy.game, self.player_id, self.fallback_seed,
                                    len(self.records), original, legal_actions, self.parse_retries)

        def finish_fallback(payload, persist=True):
            # Persist BEFORE exposing the selected action to the real engine.
            save = getattr(model, "record_action_fallback", None)
            if persist and callable(save):
                save(payload)
            action = parse_action(self.policy.game, payload["action"], legal_actions)
            self.records.append({"observation": view, "self_belief": None, "world_belief": None,
                                 "action": action.model_dump(), "source": "random_fallback"})
            # Keep previous genuine model beliefs; fabricate no replacement.
            last = payload["decision_attempts"][-1]
            return action, last["prompt"], last["response"], deepcopy(payload)

        restore = getattr(model, "restore_action_fallback", None)
        if self.action_failure_policy != "strict" and callable(restore):
            saved = restore(context, self.policy.policy_id)
            if saved is not None:
                return finish_fallback(saved, persist=False)
        for index in range(self.parse_retries + 1):
            stage = "decision" if index == 0 else RETRY_STAGES[index - 1]
            # Invocation failures never become parsing corrections. Only typed
            # known output/transport failures can finalize a random action.
            try:
                raw = invoke(model, stage, prompt)
            except Exception as exc:
                reason = (exc.attempt["error_category"] if isinstance(exc, KnownModelFailure)
                          else failure_reason(exc))
                if self.action_failure_policy == "strict" or reason is None:
                    raise
                attempt = (exc.attempt if isinstance(exc, KnownModelFailure) else
                    failed_attempt(stage, prompt, getattr(exc, "response", None), reason,
                        "length" if reason == "length" else getattr(model, "last_finish_reason", None),
                        getattr(model, "last_usage", {})))
                attempts.append(attempt)
                return finish_fallback(fallback_payload(context, attempts, reason, self.policy.policy_id))
            reason = getattr(model, "last_finish_reason", None)
            attempt = attempt_record(stage, prompt, raw, self.parse_retries, reason)
            attempts.append(attempt)
            persist = getattr(model, "record_output_validation", None)
            if callable(persist):
                persist(attempt)
            try:
                result, belief, action = parse_decision(raw, self.policy.game, legal_actions)
            except DecisionInvalid as exc:
                # Metadata-less direct offline models are supported. A ledger or
                # provider that exposes finish metadata must explicitly say stop.
                eligible = not hasattr(model, "last_finish_reason") or reason == "stop"
                if index == self.parse_retries or not eligible:
                    if self.action_failure_policy != "strict":
                        return finish_fallback(fallback_payload(context, attempts,
                            "parse_exhausted" if index == self.parse_retries else "unconfirmed_output",
                            self.policy.policy_id))
                    raise
                prompt = correction_prompt(original, raw, exc.args[0], index + 1)
            else:
                break
        self.belief = belief
        self.records.append({"observation": view, **belief, "action": action.model_dump()})
        return action, prompt, raw, {"policy_id": self.policy.policy_id, **deepcopy(belief),
            "decision_attempts": attempts, "decision_parse_retries": self.parse_retries,
            "decision_protocol": RETRY_PROTOCOL, "decision_attempt_count": len(attempts),
            "decision_correction_count": len(attempts) - 1,
            "response_protocol": RESPONSE_PROTOCOL,
            "response_envelope": "fenced_json" if raw.strip().startswith("```") else "json",
            "action_encoding": "legal_alias" if type(result["action"]) is str else "canonical_object"}