"""Training-only, player-visible calibration and prompt-policy revision.

The caller owns verification, search, budgets and retry identities. A loss is
not ground truth about an opponent's private information or a belief's accuracy.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re

from base.types import Observation
from . import method
from .policy import (POLICY_FIELDS, RESPONSE_PROTOCOL, PolicyState, canonical,
                     digest, string_schema, text_field)
from .reflection_retry import (CALIBRATION_FIELDS, ReflectionInvalid, attempt_record,
                               correction_prompt, parse_reflection)
from .retry import retry_count
from .fallback import KnownModelFailure, contaminated

CALIBRATION_REUSE_PROTOCOL = "agent_pro_exact_calibration_reuse_v1"


class DuplicatePolicyInstructions(ValueError):
    """The model proposed no novel policy for this DFS branch.

    This is a semantic rejection, not a corrupt run or a provider failure.
    Online search may record the branch and continue with its remaining budget.
    """


def _profit(value) -> float:
    if type(value) not in (int, float):
        raise ValueError("Expected finite net profit")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("Expected finite net profit") from None
    if not math.isfinite(number):
        raise ValueError("Expected finite net profit")
    return number


def _seat(value, player: int) -> None:
    if type(value) is not int or value != player:
        raise ValueError("Reflection requires the policy player's own records")


def _public_values(view: dict) -> dict:
    """Reject nested diagnostics inside otherwise allowed observation fields."""
    sequences = {"my_dice", "hand", "my_bids", "prize_sequence", "outcome_history"}
    seat_maps = {"total_bets", "round_bets", "scores", "revealed_dice"}

    def scalar(value):
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        raise ValueError("Malformed visible observation value")

    def clean(key, value):
        if key == "last_bid":
            if value is None:
                return None
            if (type(value) is not dict or set(value) != {"quantity", "face"}
                    or any(type(value[k]) is not int or value[k] <= 0 for k in ("quantity", "face"))):
                raise ValueError("Malformed public dice bid")
            return {k: value[k] for k in ("quantity", "face")}
        if key in seat_maps:
            if type(value) is not dict:
                raise ValueError("Malformed public player mapping")
            result = {}
            for seat in (0, 1):
                if seat in value and str(seat) in value:
                    raise ValueError("Duplicate player mapping")
                k = seat if seat in value else str(seat)
                if k in value:
                    item = value[k]
                    if key == "revealed_dice":
                        if type(item) not in (list, tuple) or any(type(n) is not int for n in item):
                            raise ValueError("Malformed historical dice reveal")
                        item = list(item)
                    else:
                        item = scalar(item)
                    result[str(seat)] = item
            return result
        if key in sequences:
            if type(value) not in (list, tuple):
                raise ValueError("Malformed visible sequence")
            return [scalar(item) for item in value]
        return scalar(value)

    for name in ("private_info", "public_info"):
        view[name] = {k: clean(k, v) for k, v in view[name].items()}
    view["history"] = [{k: clean(k, v) for k, v in row.items()} for row in view["history"]]
    return view


def _view(game: str, raw, player: int, *, terminal: bool = False) -> dict:
    if isinstance(raw, Observation):
        raw = {"player_id": raw.player_id, "private_info": raw.private_info,
               "public_info": raw.public_info, "history": raw.history}
    if type(raw) is not dict:
        raise ValueError("Expected structured observation")
    _seat(raw.get("player_id", player if terminal else None), player)
    if (type(raw.get("private_info")) is not dict
            or type(raw.get("public_info")) is not dict
            or type(raw.get("history")) is not list
            or any(type(row) is not dict for row in raw["history"])):
        raise ValueError("Malformed structured observation")
    # Do not construct an Observation from the complete input (state_text etc.).
    obs = Observation(player_id=player, private_info=raw["private_info"],
                      public_info=raw["public_info"], history=raw["history"])
    view = method.visible_view(game, obs)
    for row in view["history"]:
        if game == "liars_dice" and row.get("action") != "challenge":
            for key in ("result", "bid_quantity", "bid_face", "actual_count", "loser", "revealed_dice"):
                row.pop(key, None)
        if game == "goofspiel" and row.get("action") != "round_result":
            for key in ("p0_card", "p1_card", "outcome"):
                row.pop(key, None)
    return _public_values(view)


def _action(game: str, raw, player: int) -> dict:
    if type(raw) is not dict:
        raise ValueError("Expected own structured action")
    _seat(raw.get("player_id"), player)
    kinds = {"leduc": {"fold", "check", "call", "bet", "raise"},
             "liars_dice": {"bid", "challenge"}, "goofspiel": {"bid"}}
    if type(raw.get("type")) is not str or raw["type"] not in kinds[game]:
        raise ValueError("Invalid action type")
    amount, data = raw.get("amount"), raw.get("data")
    if type(amount) is not int or amount < 0 or type(data) is not dict:
        raise ValueError("Malformed action")
    keys = ("quantity", "face") if game == "liars_dice" and raw["type"] == "bid" else ()
    if any(type(data.get(k)) is not int or data[k] <= 0 for k in keys):
        raise ValueError("Malformed dice bid")
    return {"player_id": player, "type": raw["type"], "amount": amount,
            "data": {k: data[k] for k in keys}}


def _evidence(policy: PolicyState, episode: dict, *, allowed_split="train") -> dict:
    if type(episode) is not dict or episode.get("game") != policy.game:
        raise ValueError("Episode game does not match policy")
    if contaminated(episode):
        raise ValueError("Random fallback episode is not model learning evidence")
    if allowed_split not in ("train", "online") or episode.get("split") != allowed_split:
        raise ValueError("Reflection is restricted to training episodes")
    if episode.get("policy_id") != policy.policy_id:
        raise ValueError("Episode policy does not match parent")
    source = episode.get("scenario_id")
    if type(source) is not str or re.fullmatch(r"[0-9a-f]{64}", source) is None:
        raise ValueError("Expected scenario fingerprint")
    player = episode.get("player_id")
    if type(player) is not int or player not in (0, 1):
        raise ValueError("Expected P0 or P1")
    net = _profit(episode.get("net_profit"))
    if net >= 0:
        raise ValueError("Reflection requires a negative-net training episode")
    records = episode.get("records")
    if type(records) is not list or not records:
        raise ValueError("Expected nonempty own decision records")
    visible = []
    for record in records:
        if type(record) is not dict:
            raise ValueError("Malformed decision record")
        if "player_id" in record:
            _seat(record["player_id"], player)
        visible.append({
            "observation": _view(policy.game, record.get("observation"), player),
            **{k: text_field(record.get(k), limit=6000) for k in ("self_belief", "world_belief")},
            "action": _action(policy.game, record.get("action"), player),
        })
    feedback = episode.get("feedback")
    if type(feedback) is not dict or _profit(feedback.get("net_profit")) != net:
        raise ValueError("Missing or inconsistent player feedback")
    if "player_id" in feedback:
        _seat(feedback["player_id"], player)
    if "game" in feedback and feedback["game"] != policy.game:
        raise ValueError("Feedback game mismatch")
    history = _view(policy.game, {"player_id": player, "private_info": {},
                    "public_info": {}, "history": feedback.get("public_history")}, player)["history"]
    reveal = feedback.get("public_reveal")
    if type(reveal) is not dict:
        raise ValueError("Expected structured public reveal")
    safe_reveal = {}
    # Terminal Leduc call/check closes a showdown; missing/unknown/fold history
    # never establishes a reveal. Do not use omniscient episode diagnostics.
    if policy.game == "leduc" and history and history[-1].get("action") in ("call", "check"):
        cards = reveal.get("showdown_cards")
        if cards is not None:
            if type(cards) is not dict:
                raise ValueError("Malformed showdown cards")
            safe_cards = {}
            for seat in (0, 1):
                if seat in cards and str(seat) in cards:
                    raise ValueError("Duplicate showdown player")
                value = cards.get(str(seat), cards.get(seat))
                safe_cards[str(seat)] = text_field(value, limit=32)
            safe_reveal["showdown_cards"] = safe_cards
    return {"records": visible, "feedback": {
        "rules_text": text_field(feedback.get("rules_text")),
        "terminal_observation": _view(policy.game, feedback.get("terminal_observation"), player, terminal=True),
        "public_history": history, "public_reveal": safe_reveal, "net_profit": net,
    }}


def _instructions(game: str, data: dict) -> dict:
    if type(data) is not dict:
        raise ValueError("Expected complete candidate instructions")
    if set(data) == set(POLICY_FIELDS):
        return PolicyState(game=game, **data).instructions()
    candidate = PolicyState.from_dict(data)
    if candidate.game != game:
        raise ValueError("Previous candidate game mismatch")
    return candidate.instructions()


class PolicyReflector:
    """Two learning stages with separate bounded format corrections, no mutation."""

    def __init__(self, model, parse_retries=3, *, online=False):
        self.model = model
        self.parse_retries = retry_count(parse_retries)
        self.allowed_split = "online" if online else "train"
        self.attempts = []
        # Direct backends have no durable journal. Only confirmed valid stop
        # completions are reusable here; runner recovery uses ledger pins.
        self._calibrations = {}

    def _generate(self, stage, original, game):
        prompt = original
        for index in range(self.parse_retries + 1):
            current = stage if index == 0 else f"{stage}_retry_{index}"
            # Invocation errors (including empty output) are never parser errors.
            if hasattr(self.model, "last_finish_reason"):
                # Direct providers must report THIS completion, not a stale stop
                # from calibration or a prior proposal. The journal also resets it.
                self.model.last_finish_reason = None
            try:
                raw = method.invoke(self.model, current, prompt)
            except KnownModelFailure as exc:
                self.attempts.append(deepcopy(exc.attempt))
                raise
            reason = getattr(self.model, "last_finish_reason", None)
            attempt = attempt_record(current, prompt, raw, self.parse_retries, reason)
            self.attempts.append(attempt)
            persist = getattr(self.model, "record_reflection_validation", None)
            if callable(persist):
                persist(attempt)
            try:
                return parse_reflection(raw, stage, game)
            except ReflectionInvalid as exc:
                # Unknown metadata is not an explicit successful stop, even for
                # direct models without bind()/a journal. No speculative retry.
                if index == self.parse_retries or reason != "stop":
                    raise
                prompt = correction_prompt(original, raw, exc.args[0], index + 1, stage)

    def propose(self, policy: PolicyState, episode: dict, identity: dict,
                previous_candidates: list[dict] | None = None) -> PolicyState:
        evidence = _evidence(policy, episode, allowed_split=self.allowed_split)
        if type(identity) is not dict or not identity:
            raise ValueError("Expected nonempty semantic identity")
        scope = deepcopy(identity)
        for key, value in (("policy_id", policy.policy_id), ("source_id", episode["scenario_id"])):
            if key in scope and scope[key] != value:
                raise ValueError("Conflicting reflection identity")
            scope[key] = value
        canonical(scope)
        if previous_candidates is not None and type(previous_candidates) is not list:
            raise ValueError("Expected previous candidate list")
        previous = [_instructions(policy.game, item) for item in (previous_candidates or [])]
        context = {"game": policy.game, "policy": policy.instructions(), **evidence}
        calibration_prompt = canonical({**context,
            "task": "Calibrate self-belief and world-belief using only this player's visible evidence. Assess correctness, consistency and rationality and explain reasons. Distinguish observed facts from hypotheses; unavailable hidden truth remains unknown. A loss does not prove a belief incorrect or an action irrational. Do not invent opponent private information or probability calibration.",
            "response_protocol": RESPONSE_PROTOCOL,
            "output_schema": string_schema(CALIBRATION_FIELDS, limit=6000),
            "output": "Return exactly one JSON object with four nonempty strings: correctness, consistency, rationality, reasons. Each <=6000 characters. No Markdown fences, surrounding prose, other fields or text. Put all reasoning inside these four strings.",
        })
        self.attempts = []
        configure = getattr(self.model, "configure_reflection_retries", None)
        if callable(configure):
            configure(self.parse_retries)
        bind = getattr(self.model, "bind", None)
        if callable(bind):
            bind(scope)
        pin = getattr(self.model, "bind_reflection_context", None)
        reuse_key = digest({"protocol": CALIBRATION_REUSE_PROTOCOL,
                            "policy": policy.to_dict(), "episode": episode,
                            "prompt": calibration_prompt, "parse_retries": self.parse_retries,
                            "split": self.allowed_split})
        reuse_pin = getattr(self.model, "bind_reusable_reflection_context", None)
        calibration_scope = scope
        if callable(reuse_pin):
            calibration_scope = reuse_pin(calibration_prompt, previous, reuse_key)
        elif callable(pin):
            pin(calibration_prompt, previous)
        if callable(reuse_pin):
            # Replaying the first owner's journal revalidates corrections and
            # finish metadata, without another physical request or usage charge.
            if calibration_scope != scope:
                bind(calibration_scope)
            reflection = self._generate("calibration", calibration_prompt, policy.game)
            if calibration_scope != scope:
                bind(scope)
        elif not callable(pin) and reuse_key in self._calibrations:
            reflection = deepcopy(self._calibrations[reuse_key])
        else:
            reflection = self._generate("calibration", calibration_prompt, policy.game)
            if not callable(pin) and getattr(self.model, "last_finish_reason", None) == "stop":
                self._calibrations[reuse_key] = deepcopy(reflection)
        proposal_prompt = canonical({**context, "reflection": reflection,
            "previous_candidates": previous,
            "task": "Propose a full revised Agent-Pro policy from the calibration and visible experience: behavioral guidelines, demonstrations and cautious world/opponent modeling. Preserve authoritative game rules; do not treat unavailable hidden truth as known. Diversify from the parent and previous rejected candidate instructions. Do not encode scenario identifiers or oracle lookups. Return a complete policy, not a patch; verification is performed separately.",
            "response_protocol": RESPONSE_PROTOCOL,
            "output_schema": string_schema(POLICY_FIELDS, limit=12000,
                                            optional_empty=("demonstrations", "opponent_model")),
            "output": "Return exactly one JSON object with string fields goal, strategy, demonstrations, world_rules, opponent_model. Goal, strategy and world_rules must be nonempty. Each field <=12000 characters and their total <=12000. No Markdown fences, surrounding prose, other fields or text.",
        })
        revised = self._generate("proposal", proposal_prompt, policy.game)
        candidate = PolicyState(game=policy.game, parent_id=policy.policy_id,
                                source_id=episode["scenario_id"], **revised)
        if candidate.instructions() in [policy.instructions(), *previous]:
            raise DuplicatePolicyInstructions("Duplicate policy instructions; no candidate")
        return candidate
