"""Case-based self-evolve memory helpers."""

from __future__ import annotations

from typing import Any


CASE_MEMORY_BANK_KEY = "case_based_self_evolve_memory_bank"
CASE_MEMORY_ACTIVE_IDS_KEY = "case_based_self_evolve_active_memory_ids"
CASE_MEMORY_REFRESH_KEYS = "case_based_self_evolve_refresh_keys"
CASE_MEMORY_REQUIRED_TOP_K = 3
CASE_MEMORY_MIN_TOP1 = 0.7
CASE_MEMORY_MIN_TOP3 = 0.35
CASE_MEMORY_MIN_TOP_GAP = 0.05
CASE_MEMORY_MIN_STEP = 3


def _memory_get(memory: Any, key: str, default: Any = None) -> Any:
    if memory is None:
        return default
    getter = getattr(memory, "get", None)
    if not callable(getter):
        return default
    return getter(key, default)


def _source_similarity(memory_item: dict[str, Any]) -> float:
    source = memory_item.get("source_case") or {}
    try:
        similarity = source.get("similarity")
        if similarity is None:
            similarity = memory_item.get("similarity")
        if similarity is None:
            public_prefix = memory_item.get("public_prefix") or {}
            similarity = public_prefix.get("similarity")
        if isinstance(similarity, dict):
            similarity = similarity.get("overall") or 0.0
        return float(similarity or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _trigger_case_memory(selected: list[dict[str, Any]]) -> bool:
    if len(selected) < CASE_MEMORY_REQUIRED_TOP_K:
        return False
    top1 = _source_similarity(selected[0])
    top3 = _source_similarity(selected[CASE_MEMORY_REQUIRED_TOP_K - 1])
    if top3 >= 0.999999:
        return False
    if top1 < CASE_MEMORY_MIN_TOP1:
        return False
    if top3 < CASE_MEMORY_MIN_TOP3:
        return False
    if top1 - top3 < CASE_MEMORY_MIN_TOP_GAP:
        return False
    return True


def case_memory_has_gradient(selected: list[dict[str, Any]]) -> bool:
    """Return whether the top cases have enough similarity gradient to justify a memory."""
    return _trigger_case_memory(selected)


def select_case_memories(
    memory: Any,
    *,
    game: str,
    position: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return memories of the cases matched by the latest runtime refresh.

    Rendering is driven by ``CASE_MEMORY_ACTIVE_IDS_KEY``, which the refresh
    hook writes before the prompt is built, so the render chain uses exactly
    the same matched cases as the generation chain. No quality/cleanliness
    filtering is applied; only entries that actually carry memory text are
    returned.
    """
    if memory is None or CASE_MEMORY_ACTIVE_IDS_KEY not in memory:
        return []
    ids = memory.get(CASE_MEMORY_ACTIVE_IDS_KEY, []) or []
    if not ids:
        return []
    raw = _memory_get(memory, CASE_MEMORY_BANK_KEY, [])
    if not isinstance(raw, list):
        return []
    by_id = {
        item.get("memory_id"): item
        for item in raw
        if isinstance(item, dict)
        and item.get("memory_id")
        and item.get("game") == game
        and item.get("position") == position
        and item.get("memory_status") == "memory"
        and item.get("memory")
    }
    try:
        top_k = max(0, int(limit)) if limit is not None else CASE_MEMORY_REQUIRED_TOP_K
    except (TypeError, ValueError):
        top_k = CASE_MEMORY_REQUIRED_TOP_K
    if top_k <= 0:
        return []

    selected = []
    for memory_id in ids:
        if len(selected) >= top_k:
            break
        item = by_id.get(memory_id)
        if item is not None:
            selected.append(item)
    return selected[:top_k]


def render_case_memory_block(
    memory: Any,
    *,
    game: str,
    position: str,
    limit: int | None = None,
) -> str:
    selected = select_case_memories(
        memory,
        game=game,
        position=position,
        limit=limit,
    )
    if not selected:
        return ""

    lines = [
        "#### SELF EVOLVE MEMORY ####",
        "",
        "Each entry is a complete battle case extracted from a state similar to the current one, along with possible opponent action patterns generated from it. History, hand distribution, and opponent style may differ from the current situation — compare them before relying on an entry. Reference a case's future projection only when its value is high; treat entries as evidence for forming or adjusting hypotheses, not as action instructions.",
        "",
    ]
    for index, item in enumerate(selected, start=1):
        lines.extend([
            f"case {index}:",
            str(item.get("memory") or ""),
            "",
        ])
    return "\n".join(lines).rstrip()
