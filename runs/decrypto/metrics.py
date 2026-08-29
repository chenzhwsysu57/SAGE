"""Metrics helpers for Decrypto experiment runs."""

from __future__ import annotations

from typing import Any


def summarize_trajectory(
    trajectory: Any,
    *,
    max_turns: int,
    interception_threshold: int,
    miscommunication_threshold: int,
) -> dict[str, Any]:
    """Summarize one Decrypto trajectory from its recorded final state."""
    data = _as_mapping(trajectory)
    steps = data.get("steps", [])
    final_state = _step_field(steps[-1], "state_after") if steps else {}
    guess_history = final_state.get("guess_history", [])

    eve_interceptions = 0
    bob_miscommunications = 0
    first_hit_round: int | None = None
    per_round: list[dict[str, Any]] = []

    for round_index, rec in enumerate(guess_history, start=1):
        code = rec.get("code")
        bob = rec.get("bob")
        eve = rec.get("eve")
        bob_wrong = bob != code
        eve_hit = eve == code

        if bob_wrong:
            bob_miscommunications += 1
        if eve_hit:
            eve_interceptions += 1
            if first_hit_round is None:
                first_hit_round = round_index

        per_round.append({
            "round": round_index,
            "bob_wrong": bob_wrong,
            "eve_hit": eve_hit,
        })

    payout = data.get("payout", {})
    p2_reward = _reward_for_player(payout, 2)
    eve_won = p2_reward > 0
    ab_win = p2_reward < 0

    end_by_interception = eve_interceptions >= interception_threshold
    end_by_miscommunication = bob_miscommunications >= miscommunication_threshold
    if end_by_interception and end_by_miscommunication:
        end_cause = "interception_and_miscommunication"
    elif end_by_interception:
        end_cause = "interception"
    elif end_by_miscommunication:
        end_cause = "miscommunication"
    elif len(guess_history) >= max_turns:
        end_cause = "survival"
    else:
        end_cause = "unknown"

    return {
        "rounds": len(guess_history),
        "eve_interceptions": eve_interceptions,
        "bob_miscommunications": bob_miscommunications,
        "eve_first_hit_round": first_hit_round,
        "eve_won": eve_won,
        "ab_win": ab_win,
        "end_cause": end_cause,
        "per_round": per_round,
    }


def aggregate_metrics(
    game_metrics: list[dict[str, Any]],
    *,
    max_turns: int,
) -> dict[str, Any]:
    """Aggregate per-game Decrypto metrics into run-level metrics."""
    games = len(game_metrics)
    total_rounds = sum(g["rounds"] for g in game_metrics)
    total_eve_interceptions = sum(g["eve_interceptions"] for g in game_metrics)
    total_bob_miscommunications = sum(g["bob_miscommunications"] for g in game_metrics)
    first_hits = [
        g["eve_first_hit_round"]
        for g in game_metrics
        if g["eve_first_hit_round"] is not None
    ]

    per_round = []
    for round_num in range(1, max_turns + 1):
        rows = [
            item
            for game in game_metrics
            for item in game["per_round"]
            if item["round"] == round_num
        ]
        attempts = len(rows)
        eve_hits = sum(1 for row in rows if row["eve_hit"])
        bob_misses = sum(1 for row in rows if row["bob_wrong"])
        per_round.append({
            "round": round_num,
            "attempts": attempts,
            "eve_interceptions": eve_hits,
            "bob_miscommunications": bob_misses,
            "eve_int_rate": _rate(eve_hits, attempts),
            "bob_mis_rate": _rate(bob_misses, attempts),
        })

    return {
        "games": games,
        "total_rounds": total_rounds,
        "ab_wins": sum(1 for g in game_metrics if g["ab_win"]),
        "eve_wins": sum(1 for g in game_metrics if g["eve_won"]),
        "avg_survival_rounds": (
            total_rounds / games if games else 0.0
        ),
        "eve_interceptions": total_eve_interceptions,
        "bob_miscommunications": total_bob_miscommunications,
        "eve_int_rate": _rate(total_eve_interceptions, total_rounds),
        "bob_mis_rate": _rate(total_bob_miscommunications, total_rounds),
        "eve_first_hit_round_avg": (
            sum(first_hits) / len(first_hits) if first_hits else None
        ),
        "eve_first_hit_round_n": len(first_hits),
        "end_causes": _count_end_causes(game_metrics),
        "per_round": per_round,
        "stages": {
            "early": _stage_metrics(per_round, 1, 4),
            "mid": _stage_metrics(per_round, 5, 8),
            "late": _stage_metrics(per_round, 9, max_turns),
        },
    }


def _stage_metrics(
    per_round: list[dict[str, Any]],
    start_round: int,
    end_round: int,
) -> dict[str, Any]:
    rows = [
        row for row in per_round
        if start_round <= row["round"] <= end_round
    ]
    attempts = sum(row["attempts"] for row in rows)
    eve_hits = sum(row["eve_interceptions"] for row in rows)
    bob_misses = sum(row["bob_miscommunications"] for row in rows)
    return {
        "rounds": [start_round, end_round],
        "attempts": attempts,
        "eve_interceptions": eve_hits,
        "bob_miscommunications": bob_misses,
        "eve_int_rate": _rate(eve_hits, attempts),
        "bob_mis_rate": _rate(bob_misses, attempts),
    }


def _count_end_causes(game_metrics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in game_metrics:
        cause = metric["end_cause"]
        counts[cause] = counts.get(cause, 0) + 1
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Unsupported trajectory type: {type(value)!r}")


def _step_field(step: Any, field: str) -> Any:
    if isinstance(step, dict):
        return step.get(field, {})
    return getattr(step, field)


def _reward_for_player(payout: dict[Any, Any], player_id: int) -> float:
    if player_id in payout:
        return float(payout[player_id])
    return float(payout.get(str(player_id), 0.0))
