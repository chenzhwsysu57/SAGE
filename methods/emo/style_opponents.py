"""Existing benchmark prompts with strict legal-action selection, no rule fallback."""

import json
import re

from base.types import Action

STYLES = {
    "leduc": ("LAG", "TAG", "CallingStation", "Nit", "Maniac", "GTO"),
    "liars_dice": ("Bluffer", "Honest", "AggressiveChallenger", "ConservativeChallenger", "Random", "GTO"),
    "goofspiel": ("Greedy", "Conservative", "Sacrificial", "Random", "GTO", "Mirror"),
}


def is_llm_style(game, name):
    return name in STYLES.get(game, ()) and name not in ("Random", "Mirror", "GTO")


def renderer(game, name):
    if not is_llm_style(game, name):
        raise ValueError("Expected an existing LLM archetype")
    if game == "leduc":
        from modeling.leduc.archetypes import ArchetypeAgent
        return ArchetypeAgent(name)
    if game == "liars_dice":
        from modeling.liars_dice.archetypes import ArchetypeAgent
        return ArchetypeAgent(name)
    from modeling.goofspiel.archetypes import GoofspielArchetypeAgent
    return GoofspielArchetypeAgent(name)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def parse_style_action(raw: str, game: str, legal: list[Action]) -> Action:
    """Accept one JSON envelope or fenced JSON; never infer an action from prose."""
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1)
    obj = json.loads(text, object_pairs_hook=_unique_object)
    if type(obj) is not dict or "action" not in obj or not legal:
        raise ValueError("Opponent requires an explicit JSON action")
    value = obj["action"]
    if type(value) is str:
        token = value.strip().lower()
        matches = []
        for action in legal:
            labels = {action.type}
            if game == "goofspiel":
                labels = {f"bid_{action.amount}", f"bid {action.amount}"}
            elif game == "liars_dice" and action.type == "bid":
                q, f = action.data["quantity"], action.data["face"]
                labels = {f"bid_{q}_{f}", f"bid {q} {f}"}
            if token in labels:
                matches.append(action)
    else:
        if type(value) is not dict or type(value.get("type")) is not str:
            raise ValueError("Opponent requires action.type")
        if set(value) - {"type", "player_id", "amount", "quantity", "face", "data"}:
            raise ValueError("Unexpected opponent action fields")
        if "player_id" in value and (type(value["player_id"]) is not int or value["player_id"] != legal[0].player_id):
            raise ValueError("Wrong opponent player")
        for key in ("amount", "quantity", "face"):
            if key in value and type(value[key]) is not int:
                raise ValueError("Opponent parameters must be integers")
        matches = [a for a in legal if a.type == value["type"].lower()]
        if "amount" in value:
            matches = [a for a in matches if a.amount == value["amount"]]
        if game == "goofspiel" and "amount" not in value:
            raise ValueError("Goofspiel bid requires amount")
        if game == "liars_dice" and value["type"].lower() == "bid":
            data = value.get("data", {k: value[k] for k in ("quantity", "face") if k in value})
            if (type(data) is not dict or set(data) != {"quantity", "face"}
                    or any(type(v) is not int for v in data.values())
                    or any(k in value and value[k] != data[k] for k in ("quantity", "face"))):
                raise ValueError("Dice bid requires complete quantity/face")
            matches = [a for a in matches if a.data == data]
        elif value.get("data") not in (None, {}) or "quantity" in value or "face" in value:
            raise ValueError("Unexpected opponent action data")
    if len(matches) != 1:
        raise ValueError("Illegal or ambiguous opponent action; no fallback")
    return matches[0]