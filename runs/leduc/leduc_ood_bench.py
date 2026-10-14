"""Leduc OOD opponent benchmark (drift + counter-modeling).

Two OOD opponent agents (implemented in ``modeling/leduc/parametric_style.py``):

- ``drift``: parametric theta table interpolated linearly across games (soft
  style switch).  Compares the additive component ladder: simple / llm_nash /
  llm_om / full.
- ``counter_modeling``: "cheater" that reads the method's style posterior from
  the previous game (information channel provided by the harness) and, once
  the argmax confidence stays above a threshold for a consecutive window,
  jumps to the farthest style by signature distance.  Only llm_om / full are
  meaningful here (both expose structured beliefs via md["beliefs"]).

Reuses ``leduc_tom_ablation._run_games`` (memory refresh / prompt saving /
trajectory reflection) via two lightweight hooks:

- ``before_game_hook`` updates the opponent theta for the upcoming game;
- ``post_game_hook`` reads the method's belief snapshot (after reflect),
  feeds it to the counter-modeling controller, and saves the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from apis.openai_wrapper import OpenAIModel
from base.ablation import AblationConfig
from base.memory import BaseMemory
from methods.baseline.simple_prompt import SimplePromptMethod
from methods.sage.leduc_tom import LeducToMMethod
from modeling.leduc import StatisticalProfileMethod
from modeling.leduc.archetypes import ARCHETYPE_NAMES
from modeling.leduc.parametric_style import (
    CounterModelingController,
    ParametricDriftController,
    ParametricStyleAgent,
    estimate_theta_tables,
    make_theta,
)
from rollout import save_trajectory
from runs.additive import FirstLegalMockModel, is_dirty_worktree, parse_csv
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
import runs.leduc.leduc_tom_ablation as leduc_ablation


# condition key -> (label, use_nash, use_opponent_model, use_memory)
OOD_CONDITIONS: dict[str, tuple[str, bool, bool, bool]] = {
    "simple": ("Simple", False, False, False),
    "llm_nash": ("LLM+Nash", True, False, False),
    "llm_om": ("LLM+OM", False, True, False),
    "full": ("Full", True, True, True),
}

DEFAULT_DRIFT_CONDITIONS = "simple,llm_nash,llm_om,full"
DEFAULT_COUNTER_CONDITIONS = "llm_om,full"


def make_method(args: argparse.Namespace, cond_key: str, name: str,
                signatures: dict, sig_meta: dict) -> Any:
    _, use_nash, use_om, _ = OOD_CONDITIONS[cond_key]
    if cond_key == "simple":
        return SimplePromptMethod(name=name)
    mm = StatisticalProfileMethod(
        memory=BaseMemory(),
        true_archetype="",
        signatures=signatures,
        signature_source=str(sig_meta.get("source", "historical_manifest")),
        signature_metadata=sig_meta,
        classification_mode=args.leduc_stat_mode,
        q_window=args.leduc_q_window,
        q_alpha=args.leduc_q_alpha,
        q_tau=args.leduc_q_tau,
    )
    return LeducToMMethod(
        name=name,
        memory=BaseMemory(),
        modeling_method=mm,
        verbose=args.verbose,
        use_cfr=True,
        ablation=AblationConfig(no_nash=not use_nash, no_belief=not use_om),
    )


def make_controller(args: argparse.Namespace,
                    theta_tables: dict[str, dict],
                    signatures: dict,
                    seed: int = 0) -> Any:
    if args.ood_opponent == "drift":
        endpoints = parse_csv(
            args.drift_endpoints,
            valid=set(ARCHETYPE_NAMES),
            label="drift endpoint",
        )
        if len(endpoints) != 2:
            raise ValueError("--drift-endpoints requires exactly two archetypes")
        return ParametricDriftController(
            theta_tables[endpoints[0]],
            theta_tables[endpoints[1]],
            drift_ramp=args.drift_ramp,
            drift_pre_stable=args.drift_pre_stable,
            drift_post_stable=args.drift_post_stable,
        )
    controller = CounterModelingController(
        theta_tables,
        signatures,
        threshold=args.switch_threshold,
        min_consecutive=args.switch_min_games,
        second_prob=args.switch_second_prob,
        seed=seed,
    )
    controller.set_start_style(args.start_style)
    return controller


def belief_snapshot(method: Any, opp_id: int) -> dict[str, float] | None:
    """Style posterior of the method's modeling module (harness-level channel)."""
    mm = getattr(method, "modeling_method", None)
    fn = getattr(mm, "style_posterior", None)
    if not callable(fn):
        return None
    try:
        posterior = fn(opp_id)
    except Exception:
        return None
    if not posterior:
        return None
    return {str(k): round(float(v), 4) for k, v in posterior.items()}


def run_ood_seat(args: argparse.Namespace, model, run_dir: Path,
                 cond_key: str, method_position: int,
                 signatures: dict, sig_meta: dict,
                 theta_tables: dict) -> dict[str, Any]:
    label, _, _, use_memory = OOD_CONDITIONS[cond_key]
    controller = make_controller(
        args, theta_tables, signatures,
        seed=args.start_seed + 7919 * (method_position + 1),
    )
    opponent = ParametricStyleAgent(
        theta=make_theta(),  # actual theta is set by before_game_hook per game
        name=f"OOD-P{1 - method_position}",
        seed=args.start_seed + 7919 * (method_position + 1),
    )
    method = make_method(args, cond_key, f"{label}-P{method_position}",
                         signatures, sig_meta)

    game_counter: list[int] = [0]
    controller_log: list[dict[str, Any]] = []
    beliefs_log: list[dict[str, float] | None] = []

    def before_game_hook(ctx: dict[str, Any]) -> None:
        game_index = ctx["game_index"]
        opponent.set_theta(controller.theta_for_game(game_index))
        opponent.theta_label = controller.label_for_game(game_index)

    def post_game_hook(trajectory) -> None:
        game_counter[0] += 1
        game_index = game_counter[0]
        seat_dir = run_dir / "trajectories" / cond_key / f"P{method_position}"
        seat_dir.mkdir(parents=True, exist_ok=True)
        save_trajectory(
            trajectory,
            str(seat_dir / f"game_{game_index:03d}.json"),
        )
        belief = belief_snapshot(method, 1 - method_position)
        beliefs_log.append(belief)
        if args.ood_opponent == "counter_modeling":
            controller.observe_belief(belief, game_index)

    t0 = time.time()
    p0, p1 = (method, opponent) if method_position == 0 else (opponent, method)
    stats = leduc_ablation._run_games(
        model,
        p0,
        p1,
        method_position,
        args.games,
        args.start_seed,
        args.verbose,
        condition_name=cond_key,
        archetype_name=args.ood_opponent,
        save_prompts=args.save_prompts,
        run_dir=run_dir,
        suffix=f"P{method_position}",
        refresh_memory_bank=use_memory,
        refresh_parallel=False,
        refresh_max_workers=4,
        refresh_model_name=args.model_name,
        refresh_temperature=0.2,
        refresh_base_url=args.base_url,
        refresh_api_key=args.api_key,
        refresh_timeout=120.0,
        before_game_hook=before_game_hook,
        post_game_hook=post_game_hook,
    )
    elapsed = round(time.time() - t0, 2)

    controller_log = controller.log
    rows = stats.get("results", [])
    for i, row in enumerate(rows):
        if i < len(controller_log):
            row["theta"] = controller_log[i]
        if i < len(beliefs_log):
            row["beliefs"] = beliefs_log[i]
    return {
        "condition_key": cond_key,
        "condition_label": label,
        "ood_opponent": args.ood_opponent,
        "position": method_position,
        "games": stats["games"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "winrate": stats["winrate"],
        "total_chips": stats["total_chips"],
        "avg_chips": stats["avg_chips"],
        "elapsed_sec": elapsed,
        "results": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leduc OOD opponent benchmark (drift / counter-modeling).",
    )
    parser.add_argument(
        "--ood-opponent", choices=["drift", "counter_modeling"], default="drift",
        help="OOD opponent type: drift (soft theta interpolation) or "
             "counter_modeling (cheater switching to farthest style).",
    )
    parser.add_argument(
        "--conditions", default=DEFAULT_DRIFT_CONDITIONS,
        help=f"Comma-separated conditions. Valid: {list(OOD_CONDITIONS)} "
             f"(counter_modeling only supports llm_om,full).",
    )
    parser.add_argument("--run-phase", default="dev_debug",
                        choices=["dev_debug", "real_measurement"],
                        help="Experiment phase recorded in config; real_measurement "
                             "requires approval.")
    parser.add_argument("--approved-real-measurement", action="store_true")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--mock-model", action="store_true",
                        help="Use a no-live first-legal-action model for dev_debug only.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts",
                        default=True)
    parser.add_argument("--verbose", "-v", action="store_true")

    # drift params
    parser.add_argument("--drift-endpoints", default="LAG,Nit",
                        help="Two archetype names for the drift endpoints.")
    parser.add_argument("--drift-pre-stable", type=int, default=10,
                        help="Games at style A (lambda=0) before the ramp.")
    parser.add_argument("--drift-ramp", type=int, default=30,
                        help="Games over which lambda goes 0 -> 1.")
    parser.add_argument("--drift-post-stable", type=int, default=10,
                        help="Games at style B (lambda=1) after the ramp.")

    # counter_modeling params
    parser.add_argument("--start-style", default="LAG",
                        choices=list(ARCHETYPE_NAMES),
                        help="Cheater's initial style (counter_modeling).")
    parser.add_argument("--switch-threshold", type=float, default=0.5,
                        help="Argmax confidence required to count a 'modeled' game.")
    parser.add_argument("--switch-min-games", type=int, default=3,
                        help="Consecutive modeled games before switching.")
    parser.add_argument("--switch-second-prob", type=float, default=0.5,
                        help="Probability of switching to the second-farthest "
                             "style (otherwise farthest).")

    # opponent-modeling settings (same defaults as additive.py)
    parser.add_argument("--leduc-signature-manifest",
                        default="runs/leduc/manifests/classify_all6_legacy.json")
    parser.add_argument("--leduc-signature-games", type=int, default=0)
    parser.add_argument("--leduc-stat-mode", default="q_window_log",
                        choices=["q_window_ema", "q_window_log", "cumulative"])
    parser.add_argument("--leduc-q-window", type=int, default=10)
    parser.add_argument("--leduc-q-alpha", type=float, default=0.85)
    parser.add_argument("--leduc-q-tau", type=float, default=1.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    condition_keys = parse_csv(
        args.conditions,
        valid=set(OOD_CONDITIONS),
        label="OOD condition",
    )
    if args.run_phase == "real_measurement" and not args.approved_real_measurement:
        parser.error("--run-phase real_measurement requires --approved-real-measurement")
    if args.ood_opponent == "counter_modeling":
        invalid = [k for k in condition_keys if k not in ("llm_om", "full")]
        if invalid:
            parser.error(
                f"counter_modeling only supports llm_om,full; got {invalid}"
            )
    if args.ood_opponent == "drift":
        endpoints = parse_csv(
            args.drift_endpoints,
            valid=set(ARCHETYPE_NAMES),
            label="drift endpoint",
        )
        if len(endpoints) != 2:
            parser.error("--drift-endpoints requires exactly two archetypes")

    signatures, sig_meta = leduc_ablation.build_empirical_signature_bundle(
        args.leduc_signature_manifest,
        max_games_per_archetype=args.leduc_signature_games,
    )
    theta_tables, theta_meta = estimate_theta_tables(
        args.leduc_signature_manifest,
        max_games_per_archetype=args.leduc_signature_games,
    )

    output_root = args.output_dir or f"results/leduc/leduc_ood_{args.ood_opponent}"
    slug = f"leduc_{args.games}g_{args.ood_opponent}_{'_'.join(condition_keys)}"
    run_dir = make_run_dir(output_root, slug)

    if args.ood_opponent == "drift":
        ood_spec = {
            "type": "drift",
            "endpoints": endpoints,
            "drift_pre_stable": args.drift_pre_stable,
            "drift_ramp": args.drift_ramp,
            "drift_post_stable": args.drift_post_stable,
        }
    else:
        ood_spec = {
            "type": "counter_modeling",
            "start_style": args.start_style,
            "threshold": args.switch_threshold,
            "min_consecutive": args.switch_min_games,
            "second_farthest_prob": args.switch_second_prob,
            "info_channel": "method_style_posterior_t_minus_1",
            "mode": "continuous_top2_random",
        }

    config = {
        "experiment": "leduc_ood",
        "phase": args.run_phase,
        "ood_opponent": ood_spec,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "dirty_worktree": is_dirty_worktree(),
        "live_api": not args.mock_model,
        "mock_model": args.mock_model,
        "conditions": [
            {"key": k, "label": OOD_CONDITIONS[k][0],
             "use_nash": OOD_CONDITIONS[k][1],
             "use_opponent_model": OOD_CONDITIONS[k][2],
             "use_memory": OOD_CONDITIONS[k][3]}
            for k in condition_keys
        ],
        "theta_metadata": theta_meta,
    }
    write_json(run_dir / "config.json", config)
    print(f"Run dir: {run_dir}")
    print(f"OOD opponent: {args.ood_opponent}  ({ood_spec})")
    print(f"Conditions: {', '.join(condition_keys)}  phase={args.run_phase}")

    if args.mock_model:
        model = FirstLegalMockModel()
    else:
        model = OpenAIModel(
            model=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    all_results: list[dict[str, Any]] = []
    for cond_key in condition_keys:
        for method_position in (0, 1):
            stats = run_ood_seat(
                args, model, run_dir, cond_key, method_position,
                signatures, sig_meta, theta_tables,
            )
            all_results.append(stats)
            print(
                f"{cond_key} P{method_position}: "
                f"wr={stats['winrate']:.0f}% chips={stats['total_chips']:+.0f} "
                f"W{stats['wins']}/L{stats['losses']}/D{stats['draws']} "
                f"({stats['elapsed_sec']}s)"
            )

    # Per-game metrics CSV (theta / trigger / outcome per row).
    csv_rows = []
    for seat in all_results:
        for row in seat["results"]:
            theta = row.get("theta") or {}
            csv_rows.append({
                "game": row["game"],
                "condition": seat["condition_key"],
                "position": seat["position"],
                "seed": row["seed"],
                "method_win": row["method_win"],
                "method_net": row["method_net"],
                "theta_lambda": theta.get("lambda"),
                "theta_style": theta.get("style"),
                "switched": theta.get("switched"),
            })
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    with open(metrics_dir / "ood_per_game.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    summary = {
        "experiment": "leduc_ood",
        "phase": args.run_phase,
        "ood_opponent": ood_spec,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "conditions": config["conditions"],
        "live_api": not args.mock_model,
        "mock_model": args.mock_model,
        "results": all_results,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Results saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
