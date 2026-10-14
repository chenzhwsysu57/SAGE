"""Train a tabular CFR+ policy for Liar's Dice."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.liars_dice_cfr import LiarsDiceCFRTrainer
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFR+ for Liar's Dice")
    parser.add_argument("--num-dice", type=int, default=4)
    parser.add_argument("--num-faces", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument(
        "--traverser-action-sample-size",
        type=int,
        default=1,
        help="Top-k traverser action sampling; set to 0 or a negative value for full traversal.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/liars_dice/cfr")
    parser.add_argument("--run-name", default="cfr_4x4")
    parser.add_argument("--run-phase", default="dev_debug")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir, args.run_name)
    config = {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "liars_dice_cfr_train",
        "run_phase": args.run_phase,
    }
    write_json(run_dir / "config.json", config)

    trainer = LiarsDiceCFRTrainer(
        num_dice=args.num_dice,
        num_faces=args.num_faces,
        iterations=args.iterations,
        seed=args.seed,
        traverser_action_sample_size=args.traverser_action_sample_size,
    )
    trainer.train(verbose=not args.no_verbose)
    trainer.save(
        str(run_dir / "model"),
        metadata={
            "output_dir": str(run_dir),
            "phase": args.run_phase,
        },
    )
    write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "iterations": args.iterations,
            "num_infosets": len(trainer.strategy),
            "traverser_action_sample_size": args.traverser_action_sample_size,
        },
    )
    print(f"Saved CFR model to: {run_dir / 'model'}")


if __name__ == "__main__":
    main()
