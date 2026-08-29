"""Benchmark a trained PPO policy against all Liar's Dice archetypes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from apis.openai_wrapper import OpenAIModel
from games.liars_dice import LiarsDice
from methods.liars_dice_ppo import LiarsDicePPOMethod
from modeling.liars_dice.archetypes import ARCHETYPE_NAMES, ArchetypeAgent
from rollout import RolloutRunner
from runs.liars_dice.prompt_utils import save_step_artifacts
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json


def _make_ppo_method(name: str, model_path: str, seed: int) -> LiarsDicePPOMethod:
    return LiarsDicePPOMethod(name=name, model_path=model_path, seed=seed)


def _make_opponent(archetype: str, name: str, seed: int = 0):
    if archetype == "Random":
        return ArchetypeAgent(archetype, name=name, seed=seed)
    return ArchetypeAgent(archetype, name=name)


def _save_trajectory_prompts(
    run_dir: Path,
    archetype: str,
    seat: str,
    game_index: int,
    trajectory,
    methods: dict[int, object],
) -> None:
    prompts_dir = run_dir / "prompts" / archetype / seat / f"game_{game_index:02d}"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for step in trajectory.steps:
        if not step.rendered_prompt:
            continue
        method = methods.get(step.player_id)
        method_name = getattr(method, "name", f"P{step.player_id}")
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in method_name
        )
        prefix = f"step{step.step:03d}_P{step.player_id}_{safe_name}"
        save_step_artifacts(prompts_dir, prefix, step, method_name)


def _run_matchup(
    model,
    archetype: str,
    games: int,
    start_seed: int,
    model_path: str,
    run_dir: Path | None,
    save_prompts: bool,
    verbose: bool,
) -> dict:
    results = []

    # Seat 0: PPO vs archetype
    p0_method = _make_ppo_method(
        name="PPO-P0",
        model_path=model_path,
        seed=start_seed,
    )
    p1_method = _make_opponent(archetype, name=f"{archetype}-P1", seed=start_seed)
    p0_wins = p0_losses = p0_draws = 0
    p0_chips = 0.0

    pbar = tqdm(range(games), desc=f"  {archetype} P0", unit="game", leave=False)
    for g in pbar:
        seed = start_seed + g
        game = LiarsDice(num_dice=4, num_faces=4, seed=seed)
        methods = {0: p0_method, 1: p1_method}
        runner = RolloutRunner(game, methods, model, verbose=verbose)
        t0 = time.time()
        trajectory = runner.run()
        dt = time.time() - t0

        if save_prompts and run_dir:
            _save_trajectory_prompts(run_dir, archetype, "P0", g + 1, trajectory, methods)

        method_chips = trajectory.payout.get(0, 0)
        opp_chips = trajectory.payout.get(1, 0)
        p0_chips += method_chips
        if method_chips > opp_chips:
            p0_wins += 1
        elif method_chips < opp_chips:
            p0_losses += 1
        else:
            p0_draws += 1

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
        pbar.set_postfix({"chips": f"{p0_chips:+.0f}", "wr": f"{p0_wins/(g+1)*100:.0f}%"})

    # Seat 1: archetype vs PPO
    p0_method = _make_opponent(archetype, name=f"{archetype}-P0", seed=start_seed)
    p1_method = _make_ppo_method(
        name="PPO-P1",
        model_path=model_path,
        seed=start_seed + 1,
    )
    p1_wins = p1_losses = p1_draws = 0
    p1_chips = 0.0

    pbar = tqdm(range(games), desc=f"  {archetype} P1", unit="game", leave=False)
    for g in pbar:
        seed = start_seed + games + g
        game = LiarsDice(num_dice=4, num_faces=4, seed=seed)
        methods = {0: p0_method, 1: p1_method}
        runner = RolloutRunner(game, methods, model, verbose=verbose)
        t0 = time.time()
        trajectory = runner.run()
        dt = time.time() - t0

        if save_prompts and run_dir:
            _save_trajectory_prompts(run_dir, archetype, "P1", g + 1, trajectory, methods)

        method_chips = trajectory.payout.get(1, 0)
        opp_chips = trajectory.payout.get(0, 0)
        p1_chips += method_chips
        if method_chips > opp_chips:
            p1_wins += 1
        elif method_chips < opp_chips:
            p1_losses += 1
        else:
            p1_draws += 1

        results.append({
            "game": g + 1,
            "seed": seed,
            "position": "P1",
            "method_chips": method_chips,
            "opp_chips": opp_chips,
            "method_win": method_chips > opp_chips,
            "steps": len(trajectory.steps),
            "time": round(dt, 2),
        })
        pbar.set_postfix({"chips": f"{p1_chips:+.0f}", "wr": f"{p1_wins/(g+1)*100:.0f}%"})

    total_games = len(results)
    total_wins = sum(1 for r in results if r["method_win"] is True)
    total_losses = sum(1 for r in results if r["method_win"] is False)
    total_draws = sum(1 for r in results if r["method_win"] is None)
    total_chips = sum(r["method_chips"] for r in results)

    return {
        "archetype": archetype,
        "games": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "winrate": total_wins / total_games * 100 if total_games else 0,
        "total_chips": total_chips,
        "avg_chips": total_chips / total_games if total_games else 0,
        "p0": {
            "wins": p0_wins,
            "losses": p0_losses,
            "draws": p0_draws,
            "winrate": p0_wins / games * 100 if games else 0,
            "total_chips": p0_chips,
        },
        "p1": {
            "wins": p1_wins,
            "losses": p1_losses,
            "draws": p1_draws,
            "winrate": p1_wins / games * 100 if games else 0,
            "total_chips": p1_chips,
        },
        "results": results,
    }


def print_results(rows: list[dict]) -> None:
    print(f"\n{'=' * 90}")
    print("  PPO vs archetype benchmark")
    print(f"{'=' * 90}")
    print(f"  {'Archetype':<24} {'WinRate':>8} {'Wins':>5} {'Losses':>6} {'Draws':>5} {'NetChips':>10} {'Avg/Game':>9}")
    print(f"  {'-' * 78}")
    for row in sorted(rows, key=lambda r: r["total_chips"], reverse=True):
        print(
            f"  {row['archetype']:<24} {row['winrate']:7.1f}% "
            f"{row['wins']:>5} {row['losses']:>6} {row['draws']:>5} "
            f"{row['total_chips']:>+10.0f} {row['avg_chips']:>+9.1f}"
        )
        p0 = row["p0"]
        p1 = row["p1"]
        print(
            f"    P0: {p0['wins']}-{p0['losses']}-{p0['draws']} "
            f"({p0['winrate']:.1f}%), chips={p0['total_chips']:+.0f} | "
            f"P1: {p1['wins']}-{p1['losses']}-{p1['draws']} "
            f"({p1['winrate']:.1f}%), chips={p1['total_chips']:+.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PPO vs all archetypes")
    parser.add_argument("--games", type=int, default=5, help="Games per seat per archetype")
    parser.add_argument("--archetypes", nargs="*", default=ARCHETYPE_NAMES)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument(
        "--model-path",
        default="results/liars_dice/ppo/20260620_235617_ppo_4x4_main/model",
        help="Path to trained PPO model directory or model.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="results/liars_dice/ppo_benchmark",
        help="Benchmark artifact root",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        default=True,
        help="Save rendered prompts and raw responses to the run directory",
    )
    parser.add_argument(
        "--no-save-prompts",
        dest="save_prompts",
        action="store_false",
        help="Disable prompt saving",
    )
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    slug = f"g{args.games}_{len(args.archetypes)}arch"
    run_dir = make_run_dir(args.output_dir, slug)
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "ppo_vs_archetypes",
        "run_phase": "real_measurement",
    })
    print(f"Run dir: {run_dir}")

    model = OpenAIModel(
        model=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    print(f"Benchmarking {len(args.archetypes)} archetypes × {args.games * 2} games each")
    print(f"Model: {args.model_path}")
    print(f"Prompts: {'on' if args.save_prompts else 'off'}")

    rows = []
    for archetype in args.archetypes:
        print(f"\n--- {archetype} vs PPO ---")
        row = _run_matchup(
            model=model,
            archetype=archetype,
            games=args.games,
            start_seed=args.start_seed,
            model_path=args.model_path,
            run_dir=run_dir,
            save_prompts=args.save_prompts,
            verbose=args.verbose,
        )
        rows.append(row)
        print(
            f"  Result: wr={row['winrate']:.1f}% chips={row['total_chips']:+.0f} "
            f"W{row['wins']}/L{row['losses']}/D{row['draws']}"
        )

    print_results(rows)

    write_json(run_dir / "summary.json", {
        "benchmark": "ppo_vs_archetypes",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": args.model_name,
        "model_path": args.model_path,
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "archetypes": rows,
    })
    print(f"\nResults saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
