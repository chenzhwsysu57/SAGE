"""Train CFR+ to low exploitability and save the strategy.

Usage:
    python runs/leduc/train_cfr.py [--iterations 500000] [--target-exp 0.01] [--output-dir results/leduc/cfr_model]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from methods.cfr import CFRPlusTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CFR+ to low exploitability.")
    parser.add_argument(
        "--iterations", type=int, default=500_000,
        help="Max training iterations (default 500k).",
    )
    parser.add_argument(
        "--target-exp", type=float, default=0.01,
        help="Target exploitability for early stopping (default 0.01).",
    )
    parser.add_argument(
        "--log-interval", type=int, default=5_000,
        help="Iterations between exploitability checks (default 5000).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/leduc/cfr_model",
        help="Directory to save the trained strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trainer = CFRPlusTrainer(iterations=args.iterations, target_exp=args.target_exp)
    trainer.log_interval = args.log_interval

    t0 = time.perf_counter()
    trainer.train()
    dt = time.perf_counter() - t0

    strategy = trainer.strategy
    exp = getattr(trainer, "_last_exp", None)
    iterations_run = getattr(trainer, "_iterations_run", None) or args.iterations

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"cfr_plus_{iterations_run}"
    path = os.path.join(args.output_dir, f"{tag}.json")

    payload = {
        "iterations": iterations_run,
        "max_iterations": args.iterations,
        "num_infosets": len(strategy),
        "exploitability": exp,
        "training_time_s": round(dt, 1),
        "target_exp": args.target_exp,
        "log_interval": args.log_interval,
        "strategy": {k: dict(v) for k, v in strategy.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nStrategy saved to {path}")
    print(f"  Iterations:     {iterations_run}")
    print(f"  Info sets:      {len(strategy)}")
    print(f"  Exploitability: {exp:.6f}" if exp is not None else "  Exploitability: N/A")
    print(f"  Training time:  {dt:.0f}s ({dt/60:.1f}min)")


if __name__ == "__main__":
    main()
