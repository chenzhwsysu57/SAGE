"""Compute hand-strength-conditioned action statistics from game logs.

Used by LLMBeliefMethod to add a readable summary to the classification prompt,
so the LLM doesn't have to count actions from raw sequences.
"""

from __future__ import annotations

from collections import defaultdict

RANK_ORDER = {"J": "weak", "Q": "medium", "K": "strong"}
RANK_DISPLAY = {"weak": "Jack (weak)", "medium": "Queen (medium)", "strong": "King (strong)"}


def compute(game_logs: list[dict]) -> dict:
    """Compute per-rank action counts from a list of game logs.

    Each game_log is: {"card": "JS", "community": "QH", "actions": ["preflop: bet 2 (pot=3)", ...]}

    Returns:
        {"J": {"bet": 5, "raise": 3, "call": 1, "fold": 0, "check": 0}, ...}
    """
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gl in game_logs:
        card = gl.get("card", "?")
        if not card:
            continue
        rank = card[0]  # "JS" → "J"
        for action_str in gl.get("actions", []):
            action_type = _extract_action(action_str)
            if action_type:
                stats[rank][action_type] += 1
    return {r: dict(stats[r]) for r in RANK_ORDER}


def format_text(stats: dict, total_games: int) -> str:
    """Format computed stats into a human-readable summary for the LLM.

    Args:
        stats: Output of compute() — keyed by card rank ("J", "Q", "K")
        total_games: Number of games (for context)

    Returns:
        Multi-line string ready to inject into a classification prompt.
    """
    lines = [f"Opponent played {total_games} hands. Action breakdown by card rank:"]
    for rank in ("J", "Q", "K"):
        label = RANK_DISPLAY.get(RANK_ORDER.get(rank, ""), rank)
        counts = stats.get(rank, {})
        if not counts:
            lines.append(f"  {label}: no data")
            continue
        total = sum(counts.values())
        parts = [f"{a}: {c}" for a, c in sorted(counts.items(), key=lambda x: -x[1])]
        agg_pct = _aggression_pct(counts)
        fold_pct = counts.get("fold", 0) / total * 100 if total > 0 else 0
        lines.append(
            f"  {label}: {total} actions — {', '.join(parts)} "
            f"(aggression: {agg_pct:.0f}%, fold: {fold_pct:.0f}%)"
        )
    return "\n".join(lines)


def _extract_action(action_str: str) -> str:
    """Parse action type from a formatted action string.

    Handles both formats:
      "preflop: bet 2 (pot=3)"  → "bet"
      "preflop: P1 bet 2 (pot=3)" → "bet"
    """
    import re
    # Strip street prefix
    if ":" in action_str:
        action_str = action_str.split(":", 1)[1].strip()
    # Extract action word: either "bet" or "P1 bet" → "bet"
    m = re.search(r'(?:P\d+\s+)?(\w+)', action_str)
    if not m:
        return ""
    raw = m.group(1).lower()
    return {"raises": "raise", "bets": "bet", "calls": "call",
            "folds": "fold", "checks": "check"}.get(raw, raw)


def _aggression_pct(counts: dict) -> float:
    """Fraction of aggressive actions (bet + raise)."""
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    total = sum(counts.values())
    return aggressive / total * 100 if total > 0 else 0
