"""Style-shift benchmark across Leduc, Liar's Dice, and Goofspiel.

Runs one continuous method instance per seat while switching the opponent
persona after a fixed number of games.  This measures adaptation to a true
online style change rather than splicing independent fixed-persona runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from apis.openai_wrapper import OpenAIModel
from base.ablation import AblationConfig
from base.memory import BaseMemory
from base.types import Trajectory
from games.goofspiel import Goofspiel
from games.leduc import LeducHoldem
from games.liars_dice import LiarsDice
from methods.sage.goofspiel_cfr_method import GoofspielCFRMethod
from methods.sage.goofspiel_tom import GoofspielToMMethod
from methods.sage.liars_dice_cfr import LiarsDiceCFRMethod
from methods.sage.liars_dice_tom import LiarsDiceToMMethod
from methods.baseline.simple_prompt import SimplePromptMethod
from methods.suspicion import SuspicionPromptMethod
from modeling.base import BaseModelingMethod
from modeling.goofspiel.archetypes import GoofspielArchetypeAgent
from modeling.leduc import (
    ArchetypeAgent as LeducArchetypeAgent,
    CFRGTOAgent,
    StatisticalProfileMethod,
)
from modeling.liars_dice.archetypes import ArchetypeAgent as LiarsDiceArchetypeAgent
from rollout import RolloutRunner, save_trajectory
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
from runs.leduc.leduc_tom_ablation import build_empirical_signature_bundle


STYLE_SHIFTS: dict[str, tuple[str, str]] = {
    "leduc": ("TAG", "CallingStation"),
    "liars_dice": ("Bluffer", "ConservativeChallenger"),
    "goofspiel": ("Greedy", "Conservative"),
}

DEFAULT_CONDITIONS = ["full", "cfr", "simple"]
GAME_ORDER = ["leduc", "liars_dice", "goofspiel"]


def _game_label(game_key: str) -> str:
    return {
        "leduc": "Leduc",
        "liars_dice": "Liar's Dice",
        "goofspiel": "Goofspiel",
    }[game_key]


def _condition_label(condition_key: str) -> str:
    return {
        "full": "Full",
        "cfr": "CFR",
        "simple": "Simple",
        "suspicion": "SuspicionAgent",
    }[condition_key]


def _make_game(game_key: str, seed: int, args: argparse.Namespace):
    if game_key == "leduc":
        return LeducHoldem(seed=seed)
    if game_key == "liars_dice":
        return LiarsDice(
            num_dice=args.liars_dice_num_dice,
            num_faces=args.liars_dice_num_faces,
            seed=seed,
        )
    if game_key == "goofspiel":
        return Goofspiel(n=args.goofspiel_n, seed=seed)
    raise ValueError(f"Unknown game: {game_key}")


def _make_opponent(game_key: str, style: str, name: str, seed: int, args: argparse.Namespace):
    if game_key == "leduc":
        if style == "GTO":
            return CFRGTOAgent(name=name, seed=seed)
        return LeducArchetypeAgent(style, name=name)
    if game_key == "liars_dice":
        if style == "GTO":
            return LiarsDiceCFRMethod(
                name=name,
                seed=seed,
                strategy_path=args.liars_dice_cfr_model_path,
            )
        return LiarsDiceArchetypeAgent(style, name=name, seed=seed)
    if game_key == "goofspiel":
        if style == "GTO":
            return GoofspielCFRMethod(
                name=name,
                model_path=args.goofspiel_cfr_model,
                seed=seed,
            )
        return GoofspielArchetypeAgent(style, name=name, seed=seed)
    raise ValueError(f"Unknown game: {game_key}")


def _make_method(
    game_key: str,
    condition_key: str,
    name: str,
    args: argparse.Namespace,
    ablation: AblationConfig,
):
    if condition_key == "simple":
        return SimplePromptMethod(name=name)

    if condition_key == "suspicion":
        if game_key != "leduc":
            raise ValueError("SuspicionAgent is supported only for Leduc in this benchmark.")
        return SuspicionPromptMethod(name=name, lang="en")

    if game_key == "leduc":
        if condition_key == "cfr":
            return CFRGTOAgent(name=name)
        signatures, metadata = build_empirical_signature_bundle(
            args.leduc_signature_manifest,
            max_games_per_archetype=args.leduc_signature_games,
        )
        effective_mode = args.leduc_stat_mode
        modeling_method = StatisticalProfileMethod(
            memory=BaseMemory(),
            signatures=signatures,
            signature_source=str(metadata.get("source", "historical_manifest")),
            signature_metadata=metadata,
            classification_mode=effective_mode,
            q_window=args.q_window,
            q_alpha=args.q_alpha,
            q_tau=args.q_tau,
        )
        from methods.sage.leduc_tom import LeducToMMethod

        return LeducToMMethod(
            name=name,
            memory=BaseMemory(),
            modeling_method=modeling_method,
            verbose=args.verbose,
            use_cfr=True,
            ablation=ablation,
        )

    if game_key == "liars_dice":
        if condition_key == "cfr":
            return LiarsDiceCFRMethod(
                name=name,
                strategy_path=args.liars_dice_cfr_model_path,
            )
        return LiarsDiceToMMethod(
            name=name,
            memory=BaseMemory(),
            verbose=args.verbose,
            lang="en",
            strategy_path=args.liars_dice_cfr_model_path,
            signature_bank_path=args.liars_dice_signature_bank_path,
            q_window=args.q_window,
            q_alpha=args.q_alpha,
            ablation=ablation,
        )

    if game_key == "goofspiel":
        if condition_key == "cfr":
            return GoofspielCFRMethod(
                name=name,
                model_path=args.goofspiel_cfr_model,
                seed=args.start_seed,
            )
        return GoofspielToMMethod(
            name=name,
            memory=BaseMemory(),
            lang="en",
            verbose=args.verbose,
            use_cfr=True,
            cfr_model_path=args.goofspiel_cfr_model,
            q_window=args.q_window,
            q_alpha=args.q_alpha,
            ablation=ablation,
        )

    raise ValueError(f"Unknown game: {game_key}")


def _method_belief_snapshot(method: Any, opponent_id: int) -> dict[str, Any] | None:
    if hasattr(method, "_posterior_cache"):
        cache = getattr(method, "_posterior_cache")
        if isinstance(cache, dict) and cache:
            if all(isinstance(v, (int, float)) for v in cache.values()):
                return {str(k): round(float(v), 4) for k, v in cache.items()}
            nested = cache.get(opponent_id)
            if isinstance(nested, dict):
                return {str(k): round(float(v), 4) for k, v in nested.items()}

    modeling = getattr(method, "modeling_method", None)
    cache = getattr(modeling, "_posterior_cache", None)
    if isinstance(cache, dict):
        nested = cache.get(opponent_id)
        if isinstance(nested, dict):
            return {str(k): round(float(v), 4) for k, v in nested.items()}
    return None


def _latest_step_beliefs(trajectory: Trajectory, method_position: int) -> dict[str, Any] | None:
    for step in reversed(trajectory.steps):
        if step.player_id != method_position or not isinstance(step.method_data, dict):
            continue
        data = step.method_data
        if isinstance(data.get("beliefs"), dict):
            return data["beliefs"]
        for key, value in data.items():
            if key.startswith("opp_") and isinstance(value, dict):
                dist = value.get("distribution")
                if isinstance(dist, dict):
                    return dist
    return None


def _save_prompts(
    trajectory: Trajectory,
    run_dir: Path,
    game_key: str,
    condition_key: str,
    seat_label: str,
    game_index: int,
    methods: dict[int, Any],
) -> None:
    prompt_dir = (
        run_dir / "prompts" / game_key / condition_key / seat_label
        / f"game_{game_index:03d}"
    )
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for step in trajectory.steps:
        method = methods.get(step.player_id)
        method_name = getattr(method, "name", f"P{step.player_id}")
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in method_name
        )
        prefix = f"step{step.step:02d}_P{step.player_id}_{safe_name}"
        if step.rendered_prompt:
            (prompt_dir / f"{prefix}_prompt.txt").write_text(step.rendered_prompt)
        if step.raw_llm_response:
            (prompt_dir / f"{prefix}_response.txt").write_text(step.raw_llm_response)

        md = step.method_data or {}
        suspicion = md.get("suspicion") if isinstance(md, dict) else None
        if isinstance(suspicion, dict):
            for stage in ("pattern", "belief", "planning"):
                prompt = str(suspicion.get(f"{stage}_prompt", ""))
                response = str(suspicion.get(f"{stage}_response", ""))
                if prompt:
                    (prompt_dir / f"{prefix}_{stage}_prompt.txt").write_text(prompt)
                if response:
                    (prompt_dir / f"{prefix}_{stage}_response.txt").write_text(response)


def _reflect_methods(trajectory: Trajectory, methods: dict[int, Any], model, game_name: str) -> None:
    for method in methods.values():
        reflect = getattr(method, "reflect", None)
        if callable(reflect):
            reflect(trajectory, model, game_name)


def run_style_shift_condition(
    game_key: str,
    condition_key: str,
    args: argparse.Namespace,
    model,
    run_dir: Path,
) -> dict[str, Any]:
    before_style, after_style = STYLE_SHIFTS[game_key]
    ablation = AblationConfig()
    display = _condition_label(condition_key)

    seat_summaries: list[dict[str, Any]] = []
    for method_position in (0, 1):
        seat_label = f"P{method_position}"
        method = _make_method(
            game_key,
            condition_key,
            name=f"{display}-{seat_label}",
            args=args,
            ablation=ablation,
        )
        wins = losses = draws = 0
        total_net = 0.0
        results: list[dict[str, Any]] = []

        desc = f"{_game_label(game_key)} {display} {seat_label}"
        pbar = tqdm(range(args.games), desc=desc, unit="game", leave=False)
        for zero_index in pbar:
            game_index = zero_index + 1
            phase = "pre_shift" if game_index <= args.switch_after else "post_shift"
            opponent_style = before_style if phase == "pre_shift" else after_style
            seed = args.start_seed + zero_index + (args.games if method_position == 1 else 0)
            game = _make_game(game_key, seed, args)
            opponent = _make_opponent(
                game_key,
                opponent_style,
                name=f"{opponent_style}-P{1 - method_position}",
                seed=seed + 1000,
                args=args,
            )
            methods = (
                {0: method, 1: opponent}
                if method_position == 0
                else {0: opponent, 1: method}
            )
            runner = RolloutRunner(game, methods, model, verbose=args.verbose)

            t0 = time.time()
            trajectory = runner.run()
            dt = time.time() - t0

            _reflect_methods(trajectory, methods, model, game.__class__.__name__)

            if args.save_prompts:
                _save_prompts(
                    trajectory,
                    run_dir,
                    game_key,
                    condition_key,
                    seat_label,
                    game_index,
                    methods,
                )
            traj_path = (
                run_dir / "trajectories" / game_key / condition_key / seat_label
                / f"game_{game_index:03d}.json"
            )
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            save_trajectory(trajectory, str(traj_path))

            method_chips = trajectory.payout.get(method_position, 0)
            opp_chips = trajectory.payout.get(1 - method_position, 0)
            net = trajectory.net_profit.get(method_position, 0.0)
            total_net += net
            if method_chips > opp_chips:
                wins += 1
                method_win = True
            elif method_chips < opp_chips:
                losses += 1
                method_win = False
            else:
                draws += 1
                method_win = None

            beliefs = (
                _method_belief_snapshot(method, 1 - method_position)
                or _latest_step_beliefs(trajectory, method_position)
            )
            row = {
                "game": game_index,
                "seed": seed,
                "phase": phase,
                "opponent_style": opponent_style,
                "method_position": method_position,
                "method_chips": method_chips,
                "opp_chips": opp_chips,
                "method_net": net,
                "method_win": method_win,
                "steps": len(trajectory.steps),
                "time": round(dt, 2),
                "beliefs": beliefs or {},
                "trajectory": str(traj_path),
            }
            results.append(row)
            pbar.set_postfix({"chips": f"{total_net:+.0f}", "wr": f"{wins / game_index * 100:.0f}%"})

        seat_summaries.append({
            "position": method_position,
            "games": args.games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": wins / args.games * 100 if args.games else 0,
            "total_chips": total_net,
            "avg_chips": total_net / args.games if args.games else 0,
            "results": results,
        })

    all_rows = [row for seat in seat_summaries for row in seat["results"]]
    total_games = len(all_rows)
    return {
        "game": game_key,
        "condition_key": condition_key,
        "condition_name": display,
        "style_shift": {
            "before": before_style,
            "after": after_style,
            "switch_after": args.switch_after,
            "first_post_shift_game": args.switch_after + 1,
        },
        "games": total_games,
        "wins": sum(1 for row in all_rows if row["method_win"] is True),
        "losses": sum(1 for row in all_rows if row["method_win"] is False),
        "draws": sum(1 for row in all_rows if row["method_win"] is None),
        "winrate": (
            sum(1 for row in all_rows if row["method_win"] is True) / total_games * 100
            if total_games else 0
        ),
        "total_chips": sum(float(row.get("method_net", 0.0)) for row in all_rows),
        "avg_chips": (
            sum(float(row.get("method_net", 0.0)) for row in all_rows) / total_games
            if total_games else 0
        ),
        "p0_results": seat_summaries[0],
        "p1_results": seat_summaries[1],
    }


def _resolve_games(raw: str) -> list[str]:
    if raw == "all":
        return list(GAME_ORDER)
    games = [g.strip() for g in raw.split(",") if g.strip()]
    bad = [g for g in games if g not in GAME_ORDER]
    if bad:
        raise ValueError(f"Unknown game(s): {bad}. Valid: {GAME_ORDER + ['all']}")
    return games


def _resolve_conditions(raw: str, games: list[str]) -> list[str]:
    if raw == "default":
        conditions = list(DEFAULT_CONDITIONS)
        if games == ["leduc"] or "leduc" in games:
            conditions.append("suspicion")
        return conditions
    conditions = [c.strip() for c in raw.split(",") if c.strip()]
    valid = set(DEFAULT_CONDITIONS + ["suspicion"])
    bad = [c for c in conditions if c not in valid]
    if bad:
        raise ValueError(f"Unknown condition(s): {bad}. Valid: {sorted(valid)}")
    if "suspicion" in conditions:
        unsupported = [g for g in games if g != "leduc"]
        if unsupported:
            raise ValueError(
                "SuspicionAgent is Leduc-only; run it with --game leduc or omit suspicion "
                f"for {unsupported}."
            )
    return conditions


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print("  STYLE-SHIFT RESULTS")
    print("=" * 88)
    print(f"  {'Game':<14} {'Condition':<15} {'WinRate':>8} {'Wins':>5} "
          f"{'Losses':>6} {'Draws':>5} {'Net':>9} {'Avg':>8}")
    print("  " + "-" * 76)
    for row in results:
        print(
            f"  {_game_label(row['game']):<14} {row['condition_name']:<15} "
            f"{row['winrate']:7.1f}% {row['wins']:>5} {row['losses']:>6} "
            f"{row['draws']:>5} {row['total_chips']:>+9.1f} "
            f"{row['avg_chips']:>+8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run continuous opponent style-shift benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", default="all",
                        help="all, leduc, liars_dice, goofspiel, or comma-separated list")
    parser.add_argument("--condition", default="default",
                        help="default or comma-separated: full,cfr,simple,suspicion")
    parser.add_argument("--games", type=int, default=40,
                        help="Total games per seat; opponent switches after --switch-after")
    parser.add_argument("--switch-after", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts",
                        default=True)
    parser.add_argument("--phase", default="dev_debug",
                        choices=["dev_debug", "real_measurement", "analysis_only"])
    parser.add_argument("--q-window", type=int, default=10)
    parser.add_argument("--q-alpha", type=float, default=0.85)
    parser.add_argument("--q-tau", type=float, default=1.5)
    parser.add_argument("--leduc-stat-mode", default="q_window_log",
                        choices=["q_window_ema", "q_window_log", "cumulative"])
    parser.add_argument("--leduc-signature-manifest",
                        default="runs/leduc/manifests/classify_all6_legacy.json")
    parser.add_argument("--leduc-signature-games", type=int, default=0)
    parser.add_argument("--liars-dice-num-dice", type=int, default=2)
    parser.add_argument("--liars-dice-num-faces", type=int, default=3)
    parser.add_argument("--liars-dice-cfr-model-path",
                        default="results/liars_dice/cfr/cfr_2x3_evalevery/model")
    parser.add_argument("--liars-dice-signature-bank-path",
                        default=(
                            "results/liars_dice/persona_signature_bank/"
                            "20260626_125343_signature_bank_3f_12r_flat/"
                            "metrics/signature_bank.json"
                        ))
    parser.add_argument("--goofspiel-n", type=int, default=5)
    parser.add_argument("--goofspiel-cfr-model", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.switch_after < 0:
        parser.error("--switch-after must be non-negative")
    if args.switch_after >= args.games:
        parser.error("--switch-after must be less than --games")

    try:
        games = _resolve_games(args.game)
        conditions = _resolve_conditions(args.condition, games)
    except ValueError as exc:
        parser.error(str(exc))

    output_root = Path(args.output_root)
    shared_slug = (
        f"style_shift_{'_'.join(games)}_{'_'.join(conditions)}_"
        f"{args.games}g_switch{args.switch_after}"
    )
    run_dirs = {
        game_key: make_run_dir(
            str(output_root / game_key / "style_shift"),
            shared_slug,
        )
        for game_key in games
    }

    model = OpenAIModel(
        model=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    all_results: list[dict[str, Any]] = []
    for game_key in games:
        game_conditions = [
            c for c in conditions
            if c != "suspicion" or game_key == "leduc"
        ]
        run_dir = run_dirs[game_key]
        write_json(run_dir / "config.json", {
            "experiment": "style_shift",
            "args": serializable_args(args),
            "game": game_key,
            "conditions": game_conditions,
            "style_shift": {
                "before": STYLE_SHIFTS[game_key][0],
                "after": STYLE_SHIFTS[game_key][1],
                "switch_after": args.switch_after,
                "first_post_shift_game": args.switch_after + 1,
            },
            "phase": args.phase,
            "live_api": True,
            "git_commit": git_commit(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        })
        print(f"\n{_game_label(game_key)} run dir: {run_dir}")
        print(
            f"Style shift: {STYLE_SHIFTS[game_key][0]} -> "
            f"{STYLE_SHIFTS[game_key][1]} after game {args.switch_after}"
        )
        for condition_key in game_conditions:
            print(f"--- {_game_label(game_key)} / {_condition_label(condition_key)} ---")
            result = run_style_shift_condition(
                game_key,
                condition_key,
                args,
                model,
                run_dir,
            )
            all_results.append(result)
            write_json(
                run_dir / "metrics" / f"{condition_key}_summary.json",
                result,
            )
            print(
                f"  wr={result['winrate']:.1f}% chips={result['total_chips']:+.1f} "
                f"avg={result['avg_chips']:+.2f}/game"
            )

        game_results = [r for r in all_results if r["game"] == game_key]
        write_json(run_dir / "summary.json", {
            "experiment": "style_shift",
            "game": game_key,
            "phase": args.phase,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model": args.model_name,
            "games_per_position": args.games,
            "start_seed": args.start_seed,
            "save_prompts": args.save_prompts,
            "style_shift": {
                "before": STYLE_SHIFTS[game_key][0],
                "after": STYLE_SHIFTS[game_key][1],
                "switch_after": args.switch_after,
                "first_post_shift_game": args.switch_after + 1,
            },
            "conditions": game_results,
        })

    print_summary(all_results)


if __name__ == "__main__":
    main()
