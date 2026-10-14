"""Pilot experiment: extract behavioral features from Werewolf games.

Baseline: all agents use SimplePromptMethod, no ToM.
Saves trajectories, per-game prompts (optional), config, and feature data.

Usage:
    python runs/werewolf/werewolf_pilot.py --games 10 --save-prompts
    python runs/werewolf/werewolf_pilot.py --games 60 --resume --output-dir results/werewolf/pilot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from tqdm import tqdm

from games.werewolf import Werewolf, ROLE_SPECS
from methods.baseline.simple_prompt import SimplePromptMethod
from apis.openai_wrapper import OpenAIModel
from rollout import RolloutRunner, save_trajectory
from modeling.werewolf.features import (
    PlayerFeatures,
    extract_player_features,
    aggregate_features,
    format_aggregate_summary,
)
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json

DEFAULT_ROLES = {"werewolf": 3, "villager": 6, "seer": 1, "witch": 1, "hunter": 1}
DEFAULT_GAMES = 10


def save_prompts(trajectory, run_dir, game_idx, seed):
    """Write per-step prompt and response to prompts/ directory."""
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    game_dir = prompts_dir / f"game_{game_idx:03d}_seed{seed}"
    game_dir.mkdir(exist_ok=True)

    for step in trajectory.steps:
        if not step.rendered_prompt:
            continue
        pid = step.player_id
        prefix = f"step{step.step:03d}_P{pid}"

        # Save original prompt (before any retry rejection feedback)
        if step.retry_attempts and step.retry_attempts[0].get("prompt"):
            original_prompt = step.retry_attempts[0]["prompt"]
        else:
            original_prompt = step.rendered_prompt
        with open(game_dir / f"{prefix}_prompt.txt", "w") as f:
            f.write(original_prompt)

        if step.raw_llm_response:
            with open(game_dir / f"{prefix}_response.txt", "w") as f:
                f.write(step.raw_llm_response)


def main():
    parser = argparse.ArgumentParser(description="Werewolf pilot experiment")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"Number of games to run (default: {DEFAULT_GAMES})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash",
                        help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="results/werewolf/pilot",
                        help="Output root directory")
    parser.add_argument("--save-prompts", action="store_true",
                        help="Save full prompts/responses to prompts/ directory")
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing trajectory files")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print step-by-step game progress")
    args = parser.parse_args()

    # Create timestamped run directory
    slug = f"{args.games}games_seed{args.seed}"
    run_dir = make_run_dir(args.output_dir, slug)
    trajectories_dir = run_dir / "trajectories"
    metrics_dir = run_dir / "metrics"

    # Save config
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "phase": "dev_debug",
        "roles": DEFAULT_ROLES,
    })

    print(f"Werewolf Pilot Experiment")
    print(f"Config: {DEFAULT_ROLES}")
    print(f"Model: {args.model} (T={args.temperature})")
    print(f"Output: {run_dir}")
    print(f"Save prompts: {args.save_prompts}")
    print(f"Resume: {args.resume}")
    print()

    model = OpenAIModel(model=args.model, temperature=args.temperature)
    all_features: list[dict[int, PlayerFeatures]] = []
    payout_history: list[dict] = []
    failed_games = 0
    skipped_games = 0

    t_start = time.time()

    pbar = tqdm(range(args.games), desc="Games", unit="game")
    for game_idx in pbar:
        game_seed = args.seed + game_idx
        game = Werewolf(seed=game_seed, roles=DEFAULT_ROLES)
        players = game.num_players()
        methods = {i: SimplePromptMethod(name=f"P{i}", lang="en")
                   for i in range(players)}

        # Resume: skip if trajectory already exists
        traj_path = trajectories_dir / f"game_{game_idx:03d}_seed{game_seed}.json"
        if args.resume and traj_path.exists():
            skipped_games += 1
            pbar.set_postfix_str(f"seed={game_seed} SKIP")
            continue

        runner = RolloutRunner(game, methods, model, verbose=args.verbose)
        t_game = time.time()

        try:
            trajectory = runner.run()
        except Exception as e:
            elapsed = time.time() - t_game
            tqdm.write(f"Game {game_idx + 1}/{args.games} (seed={game_seed}) FAILED ({elapsed:.0f}s): {e}")
            failed_games += 1
            continue

        game_time = time.time() - t_game
        roles = trajectory.static_state.get("roles", {})
        # Normalize string keys from JSON
        if roles and all(isinstance(k, str) for k in roles):
            roles = {int(k): v for k, v in roles.items()}

        # Save trajectory
        save_trajectory(trajectory, traj_path)

        # Save prompts (optional)
        if args.save_prompts:
            save_prompts(trajectory, run_dir, game_idx, game_seed)

        # Extract features
        feats = extract_player_features(trajectory, roles, ROLE_SPECS)
        all_features.append(feats)
        payout_history.append({
            "seed": game_seed,
            "winner": trajectory.winner,
            "payout": trajectory.payout,
        })

        # Quick per-role summary
        by_role = {}
        for pid, f in feats.items():
            by_role.setdefault(f.role, []).append(f)
        parts = []
        for role in sorted(by_role, key=lambda r: ROLE_SPECS[r]["side"]):
            f_list = by_role[role]
            avg_h = sum(f.h_vote for f in f_list) / len(f_list)
            avg_c = sum(f.c_accuse_vote for f in f_list) / len(f_list)
            parts.append(f"{role}: H={avg_h:.2f} C={avg_c:.2f}")
        pbar.set_postfix_str(f"seed={game_seed} {game_time:.0f}s | " + " | ".join(parts))

    elapsed = time.time() - t_start
    n_ok = args.games - failed_games - skipped_games

    print(f"\n{'='*50}")
    print(f"Completed: {n_ok}/{args.games} games ({failed_games} failed, "
          f"{skipped_games} skipped) in {elapsed:.0f}s")
    if n_ok > 0:
        print(f"Average: {elapsed/n_ok:.1f}s/game")

    if all_features:
        agg = aggregate_features(all_features)
        print(format_aggregate_summary(agg))

        # Save summary
        summary = {
            "config": dict(DEFAULT_ROLES),
            "model": args.model,
            "temperature": args.temperature,
            "num_games_requested": args.games,
            "num_games_completed": n_ok,
            "failed_games": failed_games,
            "skipped_games": skipped_games,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "aggregate_features": {
                role: {
                    feat: {
                        "mean": sum(vals) / len(vals) if vals else 0,
                        "stdev": (sum((x - sum(vals)/len(vals))**2 for x in vals) / len(vals))**0.5
                        if len(vals) > 1 else 0,
                        "n": len(vals),
                    }
                    for feat, vals in features.items()
                }
                for role, features in agg.items()
            },
            "payout_history": payout_history,
        }
        write_json(run_dir / "summary.json", summary)
        print(f"\nSummary: {run_dir / 'summary.json'}")

        # CSV export
        csv_path = metrics_dir / "features.csv"
        with open(csv_path, "w") as f:
            f.write("game_idx,player_id,role,side,h_vote,h_speak,c_accuse_vote,"
                    "vote_accuracy,mention_accuracy,majority_alignment,speak_count,total_words\n")
            for gi, game_feats in enumerate(all_features):
                for pid, feat in sorted(game_feats.items()):
                    f.write(f"{gi},{pid},{feat.role},{feat.side},"
                            f"{feat.h_vote:.4f},{feat.h_speak:.4f},"
                            f"{feat.c_accuse_vote:.4f},{feat.vote_accuracy:.4f},"
                            f"{feat.mention_accuracy:.4f},{feat.majority_alignment:.4f},"
                            f"{feat.speak_count},{feat.total_words}\n")
        print(f"Features CSV: {csv_path}")


if __name__ == "__main__":
    main()
