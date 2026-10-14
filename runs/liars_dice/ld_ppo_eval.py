"""Evaluate a trained Liar's Dice PPO policy against random play."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from games.liars_dice import LiarsDice
from methods.sage.liars_dice_ppo import LiarsDicePPOMethod
from modeling.liars_dice.archetypes import ArchetypeAgent
from rollout import RolloutRunner
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json


def _make_ppo(model_path: str, name: str, seed: int) -> LiarsDicePPOMethod:
    return LiarsDicePPOMethod(name=name, model_path=model_path, seed=seed)


def _make_random(name: str, seed: int = 0):
    return ArchetypeAgent("Random", name=name, seed=seed)


def _run_matchup(
    model_path: str,
    games: int,
    start_seed: int,
    run_dir: str | None,
    verbose: bool,
) -> dict:
    results = []
    wins = losses = draws = 0
    total_chips = 0.0

    pbar = tqdm(range(games), desc="  PPO vs Random", unit="game", leave=False)
    for g in pbar:
        seed = start_seed + g
        game = LiarsDice(num_dice=4, num_faces=4, seed=seed)

        # Seat 0: PPO
        p0 = _make_ppo(model_path, name="PPO-P0", seed=start_seed)
        p1 = _make_random(name="Random-P1", seed=start_seed)
        t0 = time.time()
        trajectory = RolloutRunner(game, {0: p0, 1: p1}, None, verbose=verbose).run()
        dt = time.time() - t0
        method_chips = trajectory.payout.get(0, 0)
        opp_chips = trajectory.payout.get(1, 0)
        total_chips += method_chips
        if method_chips > opp_chips:
            wins += 1
        elif method_chips < opp_chips:
            losses += 1
        else:
            draws += 1
        results.append({
            "game": g + 1,
            "seed": seed,
            "position": "P0",
            "method_chips": method_chips,
            "opp_chips": opp_chips,
            "method_win": method_chips > opp_chips,
            "steps": len(trajectory.steps),
            "time": round(dt, 2),
        })

        # Seat 1: PPO
        game = LiarsDice(num_dice=4, num_faces=4, seed=seed + games)
        p0 = _make_random(name="Random-P0", seed=start_seed)
        p1 = _make_ppo(model_path, name="PPO-P1", seed=start_seed + 1)
        t0 = time.time()
        trajectory = RolloutRunner(game, {0: p0, 1: p1}, None, verbose=verbose).run()
        dt = time.time() - t0
        method_chips = trajectory.payout.get(1, 0)
        opp_chips = trajectory.payout.get(0, 0)
        total_chips += method_chips
        if method_chips > opp_chips:
            wins += 1
        elif method_chips < opp_chips:
            losses += 1
        else:
            draws += 1
        results.append({
            "game": g + 1,
            "seed": seed + games,
            "position": "P1",
            "method_chips": method_chips,
            "opp_chips": opp_chips,
            "method_win": method_chips > opp_chips,
            "steps": len(trajectory.steps),
            "time": round(dt, 2),
        })
        pbar.set_postfix({"chips": f"{total_chips:+.0f}", "wr": f"{wins/len(results)*100:.0f}%"})

    return {
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / len(results) * 100 if results else 0.0,
        "total_chips": total_chips,
        "avg_chips": total_chips / len(results) if results else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO vs Random")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--model-path", default="results/liars_dice/ppo")
    parser.add_argument("--output-dir", default="results/liars_dice/ppo_eval")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir, f"g{args.games}_ppo_vs_random")
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "ppo_vs_random",
        "run_phase": "dev_debug",
    })

    print(f"Run dir: {run_dir}")
    print(f"Model: {args.model_path}")
    row = _run_matchup(args.model_path, args.games, args.start_seed, str(run_dir), args.verbose)
    print(
        f"Result: wr={row['winrate']:.1f}% chips={row['total_chips']:+.0f} "
        f"W{row['wins']}/L{row['losses']}/D{row['draws']}"
    )

    write_json(run_dir / "summary.json", {
        "benchmark": "ppo_vs_random",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "model_path": args.model_path,
        "row": row,
    })
    print(f"Summary saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
