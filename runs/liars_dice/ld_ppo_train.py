"""Train a minimal PPO policy for Liar's Dice."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.liars_dice_ppo import LiarsDicePPOTrainer
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for Liar's Dice")
    parser.add_argument("--num-dice", type=int, default=4)
    parser.add_argument("--num-faces", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/liars_dice/ppo")
    parser.add_argument("--run-name", default="ppo_4x4")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir, args.run_name)
    config = {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "liars_dice_ppo_train",
        "run_phase": "dev_debug",
    }
    write_json(run_dir / "config.json", config)

    trainer = LiarsDicePPOTrainer(
        num_dice=args.num_dice,
        num_faces=args.num_faces,
        episodes=args.episodes,
        rollout_batch_size=args.rollout_batch_size,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    trainer.train(verbose=not args.no_verbose)
    trainer.save(str(run_dir / "model"), metadata={
        "output_dir": str(run_dir),
        "phase": "dev_debug",
    })
    write_json(run_dir / "summary.json", {
        "run_dir": str(run_dir),
        "episodes": args.episodes,
        "train_updates": trainer._train_updates,
        "avg_return": sum(trainer._episode_returns) / max(1, len(trainer._episode_returns)),
        "avg_length": sum(trainer._episode_lengths) / max(1, len(trainer._episode_lengths)),
    })
    print(f"Saved PPO model to: {run_dir / 'model'}")


if __name__ == "__main__":
    main()
