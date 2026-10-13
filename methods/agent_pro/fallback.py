"""Agent-Pro-only failure types and deterministic, visible-context legal sampling.

No global RNG, oracle state, API retry, token-cap change or invented belief.
Budget exhaustion and ambiguous requests are deliberately NOT eligible failures.
"""

from copy import deepcopy
import hashlib
import random

import httpx

from base.model_output import ModelRequestError, OutputTruncated
from .policy import canonical
from .retry import RETRY_PROTOCOL

PROTOCOL = "agent_pro_random_legal_final_action_v1"
POLICIES = ("strict", "random_legal")


class EmptyModelResponse(ValueError):
    """Only raised at the model output boundary, not for arbitrary ValueError."""

    def __init__(self, raw):
        self.response = raw if type(raw) is str else None
        super().__init__("Empty Agent-Pro response")


class KnownModelFailure(RuntimeError):
    """Audited received output / known transport failure, never an unknown call."""

    def __init__(self, attempt):
        self.attempt = deepcopy(attempt)
        super().__init__(attempt["error_category"])


def check_policy(value):
    if value not in POLICIES:
        raise ValueError("Expected strict or random_legal action failure policy")
    return value


def failure_reason(exc):
    if isinstance(exc, OutputTruncated):
        return "length"
    if isinstance(exc, EmptyModelResponse):
        return "empty_response"
    if isinstance(exc, ModelRequestError):
        return exc.category
    if isinstance(exc, httpx.TransportError):
        return "transport_error"
    return None


def failed_attempt(stage, prompt, response, reason, finish_reason=None, usage=None):
    from base.baseline_calls import USAGE_FIELDS
    usage = usage if type(usage) is dict else {}
    return {"stage": stage, "prompt": prompt, "response": response,
            "validation": "failed", "error_category": reason,
            "finish_reason": finish_reason, "usage": {k: v for k, v in usage.items()
                if k in USAGE_FIELDS and type(v) is int and v >= 0}}


def selection_context(game, player, seed, index, original, legal, retries):
    if type(seed) is not int or type(index) is not int or index < 0:
        raise ValueError("Fallback seed/index must be integers")
    actions = sorted((a.model_dump() for a in legal), key=canonical)
    if not actions or len({canonical(a) for a in actions}) != len(actions):
        raise ValueError("Fallback requires unique nonempty current legal actions")
    return {"protocol": PROTOCOL, "game": game, "player_id": player,
            "seed": seed, "decision_index": index, "parse_retries": retries,
            "original_prompt_sha256": hashlib.sha256(original.encode("utf-8", errors="surrogatepass")).hexdigest(),
            "legal_actions": actions}


def fallback_payload(context, attempts, reason, policy_id):
    if not attempts:
        raise ValueError("Fallback requires attempted model evidence")
    seed = int(hashlib.sha256(canonical(context).encode()).hexdigest(), 16)
    actions = context["legal_actions"]
    index = random.Random(seed).randrange(len(actions))
    return {"source": "random_fallback", "fallback_protocol": PROTOCOL,
            "fallback_context": context, "fallback_reason": reason,
            "fallback_selected_index": index, "fallback_probability": 1 / len(actions),
            "action": deepcopy(actions[index]), "policy_id": policy_id,
            "self_belief": None, "world_belief": None,
            "decision_attempts": deepcopy(attempts),
            "decision_parse_retries": context["parse_retries"],
            "decision_protocol": RETRY_PROTOCOL, "decision_attempt_count": len(attempts),
            "decision_correction_count": len(attempts) - 1}


def contaminated(episode):
    # Inspect provenance as well as aggregate flags: callers cannot silently
    # make random actions into model evidence by forgetting the aggregate.
    return bool(episode.get("action_fallback_count", 0) or any(
        r.get("source") == "random_fallback" for r in episode.get("records", [])) or any(
        s.get("decision_metadata", {}).get("source") == "random_fallback"
        or s.get("opponent_metadata", {}).get("source") == "random_fallback"
        for s in episode.get("trajectory", {}).get("steps", [])))


def learning_failure(exc):
    from .reflection_retry import ReflectionInvalid
    return isinstance(exc, (KnownModelFailure, ReflectionInvalid)) or failure_reason(exc) is not None
