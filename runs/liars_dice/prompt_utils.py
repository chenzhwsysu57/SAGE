from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _action_summary(action) -> dict[str, Any] | None:
    if action is None:
        return None
    summary: dict[str, Any] = {
        "player_id": getattr(action, "player_id", None),
        "type": getattr(action, "type", None),
        "amount": getattr(action, "amount", 0),
        "data": getattr(action, "data", {}) or {},
        "display": str(action),
    }
    if summary["type"] == "bid":
        data = summary["data"]
        summary["quantity"] = data.get("quantity")
        summary["face"] = data.get("face")
    return summary


def save_step_artifacts(
    prompts_dir: Path,
    prefix: str,
    step,
    method_name: str,
) -> None:
    """Save prompt/response plus a compact decision sidecar."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / f"{prefix}_prompt.txt").write_text(step.rendered_prompt or "")
    (prompts_dir / f"{prefix}_response.txt").write_text(step.raw_llm_response or "")

    decision = {
        "step": step.step,
        "player_id": step.player_id,
        "method_name": method_name,
        "action_valid": step.action_valid,
        "retry_count": step.retry_count,
        "rejection_errors": step.rejection_errors,
        "selected_action": _action_summary(step.parsed_action),
    }
    (prompts_dir / f"{prefix}_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=str)
    )
