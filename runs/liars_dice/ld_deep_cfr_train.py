"""Train a Deep CFR policy for Liar's Dice."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.liars_dice_deep_cfr import LiarsDiceDeepCFRTrainer
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep CFR for Liar's Dice")
    parser.add_argument("--num-dice", type=int, default=4)
    parser.add_argument("--num-faces", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--train-freq", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/liars_dice/deep_cfr")
    parser.add_argument("--run-name", default="deep_cfr_4x4")
    parser.add_argument("--run-phase", default="dev_debug")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir, args.run_name)
    config = {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "liars_dice_deep_cfr_train",
        "run_phase": args.run_phase,
    }
    write_json(run_dir / "config.json", config)

    trainer = LiarsDiceDeepCFRTrainer(
        num_dice=args.num_dice,
        num_faces=args.num_faces,
        iterations=args.iterations,
        warmup_iters=args.warmup_iters,
        train_freq=args.train_freq,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        seed=args.seed,
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
            "train_updates": trainer._train_step_count,
            "seen_infosets": trainer.infoset_coverage(),
            "network_ever_trained": trainer._network_ever_trained,
        },
    )
    print(f"Saved Deep CFR model to: {run_dir / 'model'}")


if __name__ == "__main__":
    main()
