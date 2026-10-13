"""EMO Algorithm 1/2, equations (1)-(6), with Appendix-A two-round limits.

The only environment-specific substitutions are the private-information target,
rules/observations and CoT decision tasks. No heuristic opponent update occurs.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from typing import Any, Protocol

from base.interfaces import BaseMethod, BaseModel
from base.model_output import OutputTruncated
from base.types import Action, Observation
from methods.emo.adapters import BINDINGS, GAMES, TARGETS, opponent_events, own_view, scope

SELF_ROUNDS = 2
GLOBAL_ROUNDS = 2
UPDATE_STAGES = ("init", "self_feedback_1", "self_refine_1", "self_feedback_2",
                 "self_refine_2", "global_feedback_1", "global_refine_1",
                 "global_feedback_2", "global_refine_2")
DECISION_STAGES = ("summary", "cot_1", "cot_2", "cot_3")


class OutputProtocolError(ValueError):
    """Malformed model output; do not silently repair, retry, or choose a fallback."""


class Caller(Protocol):
    def call(self, stage: str, prompt: str) -> str: ...


def unfence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0].lower() not in ("```", "```json", "```python") or lines[-1] != "```":
            raise OutputProtocolError("Incomplete/unsupported output fence")
        value = "\n".join(lines[1:-1]).strip()
    return value


def parse_hypothesis(text: str) -> list[str] | None:
    """Appendix-D candidate list/None; Python quotes and JSON are envelopes only.

    No rank/dice/held-card truth checks or pruning: wrong but well-formed
    hypotheses must remain available for the paper's LLM refinement process.
    """
    value = unfence(text)
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, RecursionError):
            raise OutputProtocolError("Expected candidate list or None") from None
    if parsed is None:
        return None
    if (type(parsed) is not list or not 1 <= len(parsed) <= 3
            or any(type(x) is not str or not x.strip() for x in parsed)):
        raise OutputProtocolError("Expected one to three textual candidates or None")
    return parsed


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutputProtocolError("Duplicate action field")
        result[key] = value
    return result


def parse_action(text: str, legal: list[Action]) -> Action:
    """Transport format only: match a complete canonical legal action exactly."""
    try:
        raw = json.loads(unfence(text), object_pairs_hook=_unique)
    except (ValueError, RecursionError):
        raise OutputProtocolError("Expected one complete legal action JSON object") from None
    # Canonical JSON comparison also rejects bool/float coercions (True != 1 here).
    if type(raw) is dict:
        token = json.dumps(raw, sort_keys=True, allow_nan=False)
        matches = [a for a in legal if json.dumps(a.model_dump(), sort_keys=True) == token]
        if len(matches) == 1:
            return matches[0]
    raise OutputProtocolError("Action does not exactly match a legal action")


class EMOMethod(BaseMethod):
    name = "CoT+EMO"
    method_type = "emo"

    def __init__(self, game: str, player_id: int, caller: Caller, output_retries: int = 3):
        if game not in GAMES or player_id not in (0, 1):
            raise ValueError("Unsupported game/player")
        if type(output_retries) is not int or not 0 <= output_retries <= 3:
            raise ValueError("output_retries must be an integer in [0,3]")
        self.game = game
        self.player_id = player_id
        self.caller = caller
        self.output_retries = output_retries
        self.atomic_history: list[dict] = []
        self.central_history: list[dict] = []
        self.agent_history: list[dict] = []
        self.prediction: dict | None = None
        self.processed_events: list[int] = []
        self.public_history: list[dict] = []
        self.censored: list[dict] = []

    def begin_game(self, game_index: int):
        """Reset engine-local observation state at a repeated-match boundary.

        The benchmark keeps one EMO instance across the 10-game series so its
        long-term inference records remain available.  Public histories and
        event ids, however, belong to one game and restart from zero/empty.
        Keeping them across games makes the next game's first observation look
        like a missing or duplicate transition.
        """
        if type(game_index) is not int or game_index < 1:
            raise ValueError("game_index must be a positive integer")
        self.prediction = None
        self.processed_events = []
        self.public_history = []
        self.agent_history = []

    def _ask(self, stage: str, role: str, task: str, rules: str, context: dict) -> tuple[str, str]:
        payload = {"method": "EMO", "stage": stage, "role": role,
                   "observer": self.player_id, "opponent": 1-self.player_id,
                   "game": self.game, "rules": rules, "environment_binding": BINDINGS[self.game],
                   "target": TARGETS[self.game], "task": task, "context": context}
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        response = self.caller.call(stage, prompt)
        if getattr(self.caller, "last_finish_reason", None) == "length":
            raise OutputTruncated(response)
        if not isinstance(response, str) or not response.strip():
            raise OutputProtocolError("Empty EMO response")
        return prompt, response

    def _ask_with_retries(self, stage: str, role: str, task: str, rules: str,
                          context: dict, validator=None):
        """Retry an EMO semantic output without committing partial state.

        The provider can return a syntactically valid response that violates
        EMO's candidate-list protocol.  Those are format failures, not game
        failures.  Each correction is a new model call and is persisted by
        the enclosing caller; transport/auth/budget errors still propagate.
        """
        current_context = context
        last_error = None
        for attempt in range(self.output_retries + 1):
            try:
                _, answer = self._ask(stage, role, task, rules, current_context)
                return validator(answer) if validator is not None else answer
            except (OutputProtocolError, OutputTruncated) as exc:
                last_error = exc
                if attempt >= self.output_retries:
                    raise
                previous = exc.response if isinstance(exc, OutputTruncated) else "<invalid protocol output>"
                current_context = {
                    **context,
                    "format_correction": {
                        "attempt": attempt + 1,
                        "previous_response": previous,
                        "instruction": (
                            "Return only the required EMO output. For a hypothesis stage, "
                            "return JSON null or a JSON list of one to three nonempty strings; "
                            "do not include reasoning, Markdown or extra fields."
                        ),
                    },
                }
        raise last_error or OutputProtocolError("EMO output retry failed")

    def observe_transition(self, before: Observation, after: Observation) -> None:
        """Algorithm 2: update only after an opponent action becomes observable."""
        if before.history != self.public_history:
            raise ValueError("Missing or duplicate observed transition")
        for event in opponent_events(self.game, self.player_id, before, after):
            try:
                self._update(event, before.rules_text)
            except (OutputProtocolError, OutputTruncated) as exc:
                # No partial hypothesis/update is committed.  Mark this
                # observed event as censored so public-history alignment
                # continues and the next legal action can be selected.
                self.censored.append({
                    "event_id": event["event_id"],
                    "reason": "output_retries_exhausted",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                self.prediction = None
                self.processed_events.append(event["event_id"])
            # Algorithm 2 line 15: H_i stores observations/actions, not LLM
            # summaries or hidden chain-of-thought transcripts.
            self.agent_history.append({
                "public_information": {
                    "observer": event["after"]["observer"],
                    "public_info": deepcopy(event["after"]["public_info"]),
                },
                "private_information": deepcopy(after.private_info),
                "observed_action": deepcopy(event["action"]),
            })
        self.public_history = deepcopy(after.history)

    def _update(self, event: dict, rules: str) -> None:
        if event["event_id"] in self.processed_events:
            raise ValueError("Duplicate opponent event")
        output_format = (
            "Output only a list of at most three textual candidate descriptions for the target "
            "private information. If no information is available, output None."
        )

        def ask(stage: str, role: str, task: str, context: dict) -> str:
            return self._ask_with_retries(stage, role, task, rules, context)

        def ask_hypothesis(stage: str, role: str, task: str, context: dict):
            return self._ask_with_retries(stage, role, task, rules, context,
                                           validator=parse_hypothesis)

        # Eq. (1): current action absent; all previous atomic interactions retained.
        hypothesis = ask_hypothesis(
            "init", "atomic_opponent_model",
            "Use the prior observations and record to infer your possible private information. " + output_format,
            {"history": self.atomic_history, "public_observation": event["before"],
             "target_scope": event["scope"]})
        for iteration in range(1, SELF_ROUNDS + 1):
            context = {"history": self.atomic_history, "public_observation": event["after"],
                       "actual_action": event["action"], "target_scope": event["scope"],
                       "hypothesis": hypothesis}
            feedback = ask(f"self_feedback_{iteration}", "atomic_opponent_model",
                           "Based on the actual observed action and your private-information estimate, "
                           "analyze step by step and provide feedback for adjusting the estimate.", context)
            hypothesis = ask_hypothesis(
                f"self_refine_{iteration}", "atomic_opponent_model",
                "Adjust your private-information estimate based on the feedback. " + output_format,
                {**context, "feedback": feedback})
        for iteration in range(1, GLOBAL_ROUNDS + 1):
            context = {"public_observation": event["after"], "actual_action": event["action"],
                       "target_scope": event["scope"],
                       "all_opponent_predictions": {str(1-self.player_id): hypothesis},
                       "global_history": self.central_history,
                       "agent_history": self.agent_history}
            feedback = ask(f"global_feedback_{iteration}", "central_validator",
                           "Evaluate the opponent model's private-information estimate against the global "
                           "public record, other players' observed actions and the game rules. "
                           "Analyze step by step and give feedback for adjusting the estimate.", context)
            hypothesis = ask_hypothesis(
                f"global_refine_{iteration}", "atomic_opponent_model",
                "Adjust your private-information estimate based on the global feedback. " + output_format,
                {"history": self.atomic_history, "public_observation": event["after"],
                 "actual_action": event["action"], "target_scope": event["scope"],
                 "hypothesis": hypothesis, "feedback": feedback})
        prediction = {"opponent": event["opponent"], "scope": event["scope"],
                      "event_id": event["event_id"], "candidates": hypothesis}
        # Commit only after all paper phases finish. On failure the runner replays
        # the episode from durable responses rather than continuing partial memory.
        # Algorithm 1 line 15: H_j <- H_j U {p_tilde, q_hat, a_j}.
        # Keep only that finalized tuple; all raw calls remain in the journal for
        # audit but are not recursively injected into later model contexts.
        history_item = {
            "public_information": {
                "observer": event["after"]["observer"],
                "public_info": deepcopy(event["after"]["public_info"]),
            },
            "predicted_private_information": deepcopy(prediction),
            "observed_action": deepcopy(event["action"]),
        }
        self.atomic_history.append(history_item)
        self.central_history.append(deepcopy(history_item))
        self.prediction = prediction
        self.processed_events.append(event["event_id"])

    def act(self, observation: Observation, legal_actions: list[Action],
            model: BaseModel | None = None) -> tuple[Action, str, str]:
        if observation.player_id != self.player_id or observation.history != self.public_history:
            raise ValueError("Observe all preceding transitions before acting")
        if not legal_actions or any(a.player_id != self.player_id for a in legal_actions):
            raise ValueError("Expected learner legal actions")
        prediction = deepcopy(self.prediction)
        # Environment lifetime, not an extra inference: reroll invalidates the old
        # dice as current hidden information. Historical hypotheses remain stored.
        if self.game == "liars_dice" and prediction is not None and prediction["scope"] != scope(self.game, observation):
            prediction = None
        _, summary = self._ask("summary", "central_validator",
                               "Summarize the possible private information from the opponent model's "
                               "inference. Preserve the recorded scope and lack of information.",
                               observation.rules_text,
                               {"all_opponent_predictions": {str(1-self.player_id): prediction}})
        context: dict[str, Any] = {"history": self.agent_history, "observation": own_view(observation),
                                  "opponent_summary": summary,
                                  "legal_actions": [a.model_dump() for a in legal_actions]}
        _, first = self._ask("cot_1", "reasoning_agent",
                             "Using the opponent-model summary and your own private information, "
                             "analyze your current situation in this game step by step.", observation.rules_text, context)
        _, second = self._ask("cot_2", "reasoning_agent",
                              "Compare the available legal actions in this situation, using the preceding "
                              "analysis and the opponent-model summary to reason about their consequences.",
                              observation.rules_text, {**context, "previous_reasoning": first})
        prompt, response = self._ask("cot_3", "reasoning_agent",
                                     "Choose your action based on the preceding reasoning. Output only one "
                                     "complete JSON object copied from legal_actions; do not add fields.",
                                     observation.rules_text, {**context, "previous_reasoning": second})
        action = parse_action(response, legal_actions)
        # Algorithm 2 line 15: H_i <- H_i U {p_i, q_i, a}. The reasoning states
        # s_k are transient inputs to the next CoT step, not persistent history.
        self.agent_history.append({
            "public_information": {"observer": observation.player_id,
                                   "public_info": deepcopy(observation.public_info)},
            "private_information": deepcopy(observation.private_info),
            "action": action.model_dump(),
        })
        return action, prompt, response

    def snapshot(self) -> dict:
        return deepcopy({"game": self.game, "player_id": self.player_id,
            "atomic_history": self.atomic_history, "central_history": self.central_history,
            "agent_history": self.agent_history, "prediction": self.prediction,
            "processed_events": self.processed_events, "public_history": self.public_history,
            "censored": self.censored, "output_retries": self.output_retries})
