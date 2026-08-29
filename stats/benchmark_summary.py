#!/usr/bin/env python3
"""Print compact performance summaries for completed benchmark runs.

Usage:
    python stats/benchmark_summary.py results/leduc/my_run
    python stats/benchmark_summary.py results/leduc/run_a results/leduc/run_b \
        --csv results/benchmark_summary.csv

The script reads only ``summary.json`` files. It does not inspect prompts,
trajectories, model weights, or produce figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


FIELDS = [
    "run",
    "game",
    "condition",
    "opponent",
    "position",
    "games",
    "wins",
    "losses",
    "draws",
    "winrate_pct",
    "total_chips",
    "avg_chips",
]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("label") or value.get("key") or "")
    return "" if value is None else str(value)


def _infer_game(summary: dict[str, Any], path: Path) -> str:
    explicit = summary.get("game") or summary.get("game_name")
    if explicit:
        return str(explicit)
    parts = {part.lower() for part in path.parts}
    for game in ("leduc", "liars_dice", "goofspiel"):
        if game in parts:
            return game
    return ""


def _rows_for_result(summary: dict[str, Any], result: dict[str, Any], run: str) -> list[dict[str, Any]]:
    condition = result.get("condition_name") or result.get("condition_key")
    opponent = result.get("archetype") or result.get("persona")
    if not condition:
        condition = _label(summary.get("agent_condition"))
    if not opponent:
        opponent = _label(summary.get("opponent_condition"))

    position_rows = []
    for key in ("p0_results", "p1_results"):
        if isinstance(result.get(key), dict):
            position_rows.append(result[key])
    if not position_rows and result.get("position") is not None:
        position_rows = [result]
    if not position_rows:
        position_rows = [result]

    rows = []
    for item in position_rows:
        games = int(_number(item.get("games", result.get("games", 0))))
        rows.append({
            "run": run,
            "game": _infer_game(summary, Path(run)),
            "condition": condition or "",
            "opponent": opponent or "",
            "position": item.get("position", "all"),
            "games": games,
            "wins": int(_number(item.get("wins", 0))),
            "losses": int(_number(item.get("losses", 0))),
            "draws": int(_number(item.get("draws", 0))),
            "winrate_pct": round(_number(item.get("winrate", 0.0)), 4),
            "total_chips": round(_number(item.get("total_chips", item.get("method_net", 0.0))), 4),
            "avg_chips": round(_number(item.get("avg_chips", 0.0)), 4),
        })
    return rows


def read_run(path: Path) -> list[dict[str, Any]]:
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"summary has no results list: {summary_path}")
    run_name = str(summary_path.parent)
    rows = []
    for result in results:
        if isinstance(result, dict):
            rows.extend(_rows_for_result(summary, result, run_name))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark summary.json files.")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or summary.json files")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    failed = 0
    for run in args.runs:
        try:
            rows.extend(read_run(run))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"warning: {exc}", file=sys.stderr)
            failed += 1

    if args.csv:
        write_csv(rows, args.csv)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return 1 if failed and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
