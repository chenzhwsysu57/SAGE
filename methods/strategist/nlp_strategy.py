"""Structured natural-language Strategist policy and fixed local executor.

The model never supplies executable code. It supplies a short natural-language
guide plus six bounded numeric coefficients. ``compile_policy`` is trusted,
static repository code that maps those data to the existing ValueHeuristic ABI
so the already-tested information-set PUCT implementation can be reused.
"""
from __future__ import annotations

import base64
import json
import math
import re

from methods.strategist.evolution import canonical, digest
from methods.strategist.runner import strict_json
from methods.strategist.value import ValueHeuristic

SCHEMA = "strategist.nlp_policy.v1"
WEIGHTS = ("score", "resources", "pressure", "progress", "risk")


def parse_policy(text: str, game: str) -> dict:
    if type(text) is not str or len(text) > 8000:
        raise ValueError("Invalid NLP policy response")
    value = strict_json(text.strip())
    if (set(value) != {"schema", "game", "strategy", "weights", "bias"}
            or value["schema"] != SCHEMA or value["game"] != game
            or type(value["strategy"]) is not str
            or not 1 <= len(value["strategy"].strip()) <= 400
            or any(ord(ch) < 32 and ch not in "\n\t" for ch in value["strategy"])
            or type(value["weights"]) is not dict
            or tuple(sorted(value["weights"])) != tuple(sorted(WEIGHTS))):
        raise ValueError("Invalid NLP policy schema")
    for name in (*WEIGHTS, "bias"):
        number = value["bias"] if name == "bias" else value["weights"][name]
        if type(number) not in (int, float) or not math.isfinite(number) or not -2 <= number <= 2:
            raise ValueError("NLP policy coefficients must be finite in [-2,2]")
    value["strategy"] = value["strategy"].strip()
    value["weights"] = {name: float(value["weights"][name]) for name in WEIGHTS}
    value["bias"] = float(value["bias"])
    return value


def policy_contract(game: str) -> dict:
    return {
        "representation": SCHEMA,
        "instruction": "Return exactly one JSON object, no Markdown and no Python code. The strategy is an operational natural-language guide. Coefficients tune a fixed trusted local evaluator and must each be in [-2,2].",
        "schema": {"schema": SCHEMA, "game": game, "strategy": "1..400 chars",
                   "weights": {name: "finite number [-2,2]" for name in WEIGHTS},
                   "bias": "finite number [-2,2]"},
        "semantics": {
            "score": "current public score/pot advantage",
            "resources": "private hand/dice/card resources",
            "pressure": "cost, current bid, or current prize pressure",
            "progress": "round/street/history progression",
            "risk": "uncertainty and exposure penalty",
            "bias": "small baseline preference; fixed seat coordinates are handled locally",
        },
        "executor": "Fixed repository template; strategy prose is logged but never executed. MCTS and game rules remain local.",
    }


def extract_policy(source: str) -> dict:
    first = source.splitlines()[0] if type(source) is str and source else ""
    match = re.fullmatch(r"# NLP_POLICY_V1 ([A-Za-z0-9_-]+)", first)
    if not match:
        raise ValueError("Missing structured policy marker")
    raw = base64.urlsafe_b64decode(match.group(1) + "=" * (-len(match.group(1)) % 4)).decode()
    value = strict_json(raw)
    return parse_policy(canonical(value), value.get("game") if type(value) is dict else "")


def compile_policy(policy: dict) -> str:
    policy = parse_policy(canonical(policy), policy.get("game", ""))
    game, w, bias = policy["game"], policy["weights"], policy["bias"]
    marker = base64.urlsafe_b64encode(canonical(policy).encode()).decode().rstrip("=")
    # All emitted syntax is repository-owned. Numeric literals are canonical
    # validated floats; model text appears only in the base64 comment marker.
    if game == "leduc":
        body = f'''    p = state["player_id"]
    private = state["private_info"]
    public = state["public_info"]
    card = private["card"]
    strength = 0.0
    if card in ["KH", "KD", "KS", "KC"]:
        strength = 1.0
    elif card in ["QH", "QD", "QS", "QC"]:
        strength = 0.5
    score = (public["opponent_bet_total"] - public["my_bet_total"]) / 19.0
    pressure = public["to_call"] / 12.0
    progress = len(state["history"]) / 10.0
    risk = pressure * (1.0 - strength)
    v = {bias!r} + {w['score']!r} * score + {w['resources']!r} * strength + {w['pressure']!r} * pressure + {w['progress']!r} * progress - {w['risk']!r} * risk
'''
    elif game == "liars_dice":
        body = f'''    p = state["player_id"]
    private = state["private_info"]
    public = state["public_info"]
    dice = private["my_dice"]
    ones = 0
    for die in dice:
        if die == 1:
            ones += 1
    resources = (sum(dice) / (3.0 * len(dice)) + ones / len(dice)) / 2.0
    score = (public["my_dice_count"] - public["opponent_dice_count"]) / 2.0
    pressure = 0.0
    if public["last_bid"] is not None:
        pressure = public["last_bid"]["quantity"] / public["total_dice_remaining"]
    progress = (3.0 - public["total_dice_remaining"]) / 3.0
    risk = pressure * (1.0 - resources)
    v = {bias!r} + {w['score']!r} * score + {w['resources']!r} * resources + {w['pressure']!r} * pressure + {w['progress']!r} * progress - {w['risk']!r} * risk
'''
    elif game == "goofspiel":
        body = f'''    p = state["player_id"]
    private = state["private_info"]
    public = state["public_info"]
    hand = private["hand"]
    resources = sum(hand) / 15.0
    score = (public["scores"]["0"] - public["scores"]["1"]) / 15.0
    if p == 1:
        score = -score
    pressure = public["prize"] / 5.0
    progress = public["round"] / 5.0
    risk = pressure * resources
    v = {bias!r} + {w['score']!r} * score + {w['resources']!r} * resources + {w['pressure']!r} * pressure + {w['progress']!r} * progress - {w['risk']!r} * risk
'''
    else:
        raise ValueError("Unsupported game")
    source = f'''# NLP_POLICY_V1 {marker}
def evaluate_state(state):
{body}    v = max(-1.0, min(1.0, v))
    if p == 1:
        v = -v
    return (v, -v), {{"representation": "nlp_structured_v1", "policy_hash": "{digest(policy)[:16]}"}}
'''
    ValueHeuristic(source)  # trusted-template regression assertion
    return source
