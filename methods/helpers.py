"""Shared utilities for method prompt rendering and response parsing.

All methods (SimplePrompt, CoT, ReAct, Reflexion, ToM) use these
to avoid duplicating ~74 lines of identical code.
"""

from __future__ import annotations

import json
import re

from base.types import Action


def render_action_descs(legal_actions: list[Action]) -> list[str]:
    """Render legal actions as human-readable description lines."""
    lines = []
    for a in legal_actions:
        desc = f"  {a.type}"
        if a.amount > 0:
            desc += f" (amount={a.amount})"
        if a.data:
            desc += f" → provide: {json.dumps(a.data)}"
        lines.append(desc)
    return lines


def build_example(legal_actions: list[Action]) -> dict:
    """Return a placeholder example showing the JSON structure with data fields."""
    example_action: dict = {"type": "<action_type>"}
    for a in legal_actions:
        if a.amount > 0:
            example_action["amount"] = "<amount:int>"
        if a.data:
            for k, v in a.data.items():
                # Keep the schema but avoid leaking a concrete answer-like value.
                if isinstance(v, bool):
                    placeholder = f"<{k}:true/false>"
                elif isinstance(v, int):
                    placeholder = f"<{k}:int>"
                elif isinstance(v, float):
                    placeholder = f"<{k}:number>"
                elif isinstance(v, str):
                    placeholder = f"<{k}:text>"
                else:
                    placeholder = f"<{k}>"
                example_action[k] = placeholder
            break
    return {
        "reasoning": "step by step analysis of the current situation",
        "action": example_action,
    }


def build_rule_section(rule_text: str | list[str]) -> list[str]:
    """Build a standalone prompt section for persona/role guidance."""
    if isinstance(rule_text, str):
        body = [line.rstrip() for line in rule_text.strip().splitlines() if line.strip()]
    else:
        body = [str(line).rstrip() for line in rule_text if str(line).strip()]
    if not body:
        return []
    return ["", "#### YOUR ROLE ####", "", *body]


def extract_json(text: str) -> str | None:
    """Extract a balanced JSON object from text using brace counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_action_response(
    raw_response: str,
    player_id: int,
    legal_actions: list[Action],
) -> Action:
    """Parse LLM response into an Action. Tries JSON first, then regex fallback."""
    text = raw_response.strip()
    legal_types = {a.type for a in legal_actions}
    action_type = ""
    action_data: dict = {}
    amount = 0

    json_str = extract_json(text)
    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                # New format: {"reasoning": "...", "action": {"type": "...", ...}}
                if "action" in data and isinstance(data["action"], dict):
                    inner = data["action"]
                    action_type = str(inner.get("type", inner.get("action", ""))).lower()
                    action_data = {
                        k: v for k, v in inner.items()
                        if k not in {"type", "action", "move", "choice"}
                    }
                else:
                    # Fallback: flat format {"action": "...", ...}
                    action_key = next(
                        (k for k in ("action", "move", "choice", "type") if k in data),
                        "",
                    )
                    action_type = str(data.get(action_key, "")).lower() if action_key else ""
                    action_data = {
                        k: v for k, v in data.items()
                        if k not in {"action", "move", "choice", "type"}
                    }
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass

    if not action_type:
        types_pattern = "|".join(re.escape(t) for t in legal_types)
        if types_pattern:
            match = re.search(rf"\b({types_pattern})\b", text.lower())
            action_type = match.group(1) if match else ""

    if action_type == "bid":
        # Goofspiel: bid with amount=card_value
        if "amount" in action_data and action_data["amount"] is not None:
            amount = int(action_data["amount"])
        # Liar's Dice: bid with quantity/face
        elif "quantity" not in action_data or "face" not in action_data:
            bid_match = re.search(
                r"\bbid[_\s:-]*(\d+)[_\s:-]+(\d+)\b",
                text.lower(),
            )
            if bid_match:
                action_data = {
                    "quantity": int(bid_match.group(1)),
                    "face": int(bid_match.group(2)),
                }
            else:
                pair_match = re.search(
                    r"\b(\d+)\s*(?:of\s*)?(?:face\s*)?(\d+)\b",
                    text.lower(),
                )
                if pair_match:
                    action_data = {
                        "quantity": int(pair_match.group(1)),
                        "face": int(pair_match.group(2)),
                    }

    if amount == 0:
        for a in legal_actions:
            if a.type == action_type:
                amount = a.amount
                break

    return Action(player_id=player_id, type=action_type, amount=amount, data=action_data)


def history_to_text(history: list[dict], max_entries: int = 20) -> str:
    """Convert structured history list into a concise text summary."""
    if not history:
        return ""
    recent = history[-max_entries:]
    lines = ["Recent events:"]
    for entry in recent:
        player = entry.get("player", "?")
        action = entry.get("action", "?")
        phase = entry.get("phase", "")
        line = ""
        if player == -1:
            tally = entry.get("tally", {})
            eliminated = entry.get("eliminated")
            tie = entry.get("tie", False)
            if action == "vote_result":
                tally_str = ", ".join(f"P{p}:{c}" for p, c in sorted(tally.items()))
                elim_str = "tie" if tie else f"P{eliminated} eliminated"
                line = f"  Vote result: {tally_str} ({elim_str})"
            elif action == "round_result":
                line = f"  {entry.get('text', '')}"
        if not line:
            data = entry.get("data", {})
            result = entry.get("result", "")
            line = f"  P{player}: {action}"
            if data:
                line += f" {data}"
            if result:
                line += f" → {result}"
        if phase:
            line += f" [{phase}]"
        lines.append(line)
    return "\n".join(lines)
