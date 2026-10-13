"""Strict, immutable prompt policies; no game engine or provider dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

GAMES = ("leduc", "liars_dice", "goofspiel")
POLICY_FIELDS = ("goal", "strategy", "demonstrations", "world_rules", "opponent_model")
MAX_POLICY_CHARS = 12000
RESPONSE_PROTOCOL = "agent_pro_terminal_action_format_v3"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON value")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if type(result) is not dict:
        raise ValueError("Expected one complete JSON object")
    return result


def model_json(text: str) -> dict:
    """Parse one model JSON object, optionally in one complete Markdown fence.

    Artifact loading deliberately continues to use strict_json. No extraction
    from prose, brace repair, eval, field coercion, or repeated model requests.
    Callers must check finish_reason before this function and validate schemas
    afterwards. The original response remains untouched for journal/audit.
    """
    if type(text) is not str or not text.strip():
        raise ValueError("Expected nonempty model JSON text")
    payload = text.strip()
    if payload.startswith("```"):
        fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*", payload,
                              flags=re.DOTALL | re.IGNORECASE)
        if fenced is None:
            raise ValueError("Expected exactly one complete JSON code block")
        payload = fenced.group(1)
    result = strict_json(payload)
    canonical(result)  # Also reject numeric overflow such as 1e999.
    return result


def string_schema(fields, *, limit: int, optional_empty=()) -> dict:
    """Prompt-level contract, not a claim of provider-enforced JSON Schema."""
    return {"type": "object", "required": list(fields), "additionalProperties": False,
            "properties": {k: {"type": "string", "minLength": 0 if k in optional_empty else 1,
                               "maxLength": limit} for k in fields}}


def text_field(value: Any, *, empty: bool = False, limit: int = 12000) -> str:
    if type(value) is not str or len(value) > limit or (not empty and not value.strip()):
        raise ValueError("Invalid or overlong text field; no automatic clipping")
    return value


@dataclass(frozen=True)
class PolicyState:
    game: str
    goal: str
    strategy: str = "Use current observations and update beliefs before choosing a legal action."
    demonstrations: str = ""
    world_rules: str = "The authoritative rules are provided by the game, not by these notes."
    opponent_model: str = "Unknown; infer cautiously from observed actions."
    parent_id: str | None = None
    source_id: str | None = None

    def __post_init__(self):
        if self.game not in GAMES:
            raise ValueError("Unsupported Agent-Pro game")
        for name in POLICY_FIELDS:
            text_field(getattr(self, name), empty=name in {"demonstrations", "opponent_model"})
        if sum(len(getattr(self, k)) for k in POLICY_FIELDS) > MAX_POLICY_CHARS:
            raise ValueError("Policy exceeds total character limit")
        for value in (self.parent_id, self.source_id):
            if value is not None:
                text_field(value, limit=128)

    @property
    def policy_id(self) -> str:
        return digest(asdict(self))

    def instructions(self) -> dict:
        # No parent/source/scenario identifiers enter decision prompts.
        return {k: getattr(self, k) for k in POLICY_FIELDS}

    def to_dict(self) -> dict:
        return {"schema_version": 1, "policy_id": self.policy_id, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> PolicyState:
        expected = {"schema_version", "policy_id", "game", "parent_id", "source_id", *POLICY_FIELDS}
        if type(data) is not dict or set(data) != expected or type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Invalid policy schema")
        obj = cls(**{k: v for k, v in data.items() if k not in {"schema_version", "policy_id"}})
        if data["policy_id"] != obj.policy_id:
            raise ValueError("Policy content fingerprint mismatch")
        return obj

    @classmethod
    def initial(cls, game: str) -> PolicyState:
        goals = {"leduc": "Maximize expected net chips, not gross pot payout.",
                 "liars_dice": "Maximize the probability of winning the complete multi-round match.",
                 "goofspiel": "Maximize final own score minus opponent score across all rounds."}
        if game not in goals:
            raise ValueError("Unsupported Agent-Pro game")
        return cls(game=game, goal=goals[game])