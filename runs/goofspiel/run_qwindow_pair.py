"""Run paired Goofspiel Full vs w/o QWindow experiments.

This launcher starts Full ToM-Bayes and the q-window ablation concurrently for
each persona within a seed, then waits before moving to the next seed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = (2003, 2007)
DEFAULT_PERSONAS = ("Conservative", "Sacrificial", "Random", "Greedy", "GTO")


def _parse_csv_ints(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one persona is required")
    return items


def _build_command(
    *,
    seed: int,
    persona: str,
    games: int,
    n_cards: int,
    ablate_qwindow: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "runs/goofspiel/goofspiel_bench.py",
        "--condition",
        "tom_bayes",
        "--games",
        str(games),
        "--start-seed",
        str(seed),
        "--n",
        str(n_cards),
        "--personas",
        persona,
    ]
    if ablate_qwindow:
        cmd.extend(["--ablate", "qwindow"])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Goofspiel Full and w/o QWindow paired experiments."
    )
    parser.add_argument(
        "--seeds",
        type=_parse_csv_ints,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated start seeds. Default: 2003,2007.",
    )
    parser.add_argument(
        "--personas",
        type=_parse_csv_strings,
        default=list(DEFAULT_PERSONAS),
        help=(
            "Comma-separated personas. Default: "
            "Conservative,Sacrificial,Random,Greedy,GTO."
        ),
    )
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--n", type=int, default=5, help="Goofspiel card count.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/goof_qwindow_pair_logs"),
    )
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    any_failed = False

    for seed in args.seeds:
        print(f"===== launching Goofspiel Full + w/o QWindow seed={seed} =====")
        processes: list[tuple[str, subprocess.Popen[bytes], object]] = []

        for persona in args.personas:
            for label, ablate_qwindow in (
                ("full", False),
                ("wo_qwindow", True),
            ):
                logfile = args.log_dir / f"{label}_{persona}_seed{seed}.log"
                cmd = _build_command(
                    seed=seed,
                    persona=persona,
                    games=args.games,
                    n_cards=args.n,
                    ablate_qwindow=ablate_qwindow,
                )
                print(f"  launching {label} {persona} -> {logfile}")
                handle = logfile.open("wb")
                proc = subprocess.Popen(
                    cmd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                processes.append((f"{label}_{persona}_seed{seed}", proc, handle))

        failed = 0
        for name, proc, handle in processes:
            returncode = proc.wait()
            handle.close()
            if returncode == 0:
                print(f"  done: {name}")
            else:
                print(f"  failed: {name} returncode={returncode}")
                failed += 1
                any_failed = True

        print(f"===== finished seed={seed}; failed={failed} =====")

    print(f"Logs: {args.log_dir}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
