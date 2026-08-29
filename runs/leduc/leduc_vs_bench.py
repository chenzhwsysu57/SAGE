"""Leduc method-vs-method benchmark.

Pits one agent method (--agent-condition) against another full LLM method
(--opponent-condition) over N games per position (both positions).  The
opponent is a full agent (SimplePrompt / LLM+Nash / LLM+OM / Suspicion /
Full), so its behavior is emergent and out-of-distribution relative to the
six parametric archetypes used for opponent-modeling signatures.

Usage:
    python runs/leduc/leduc_vs_bench.py --agent-condition full \
        --opponent-condition suspicion --games 30 --start-seed 42
    python runs/leduc/leduc_vs_bench.py --mock-model --games 2   # dev_debug

Real-measurement (live API) requires --run-phase real_measurement together
with --approved-real-measurement.
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
from methods.leduc_tom import LeducToMMethod
from methods.simple_prompt import SimplePromptMethod
from methods.suspicion_prompt import SuspicionPromptMethod
from modeling.leduc import StatisticalProfileMethod
from rollout import save_trajectory
from runs.additive import FirstLegalMockModel, is_dirty_worktree
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
import runs.leduc.leduc_tom_ablation as leduc_ablation


# condition -> (label, use_nash, use_opponent_model, use_memory)
VS_CONDITIONS: dict[str, tuple[str, bool, bool, bool]] = {
    "simple": ("Simple", False, False, False),
    "llm_nash": ("LLM+Nash", True, False, False),
    "llm_om": ("LLM+OM", False, True, False),
    "full": ("Full", True, True, True),
    "suspicion": ("Suspicion", False, False, False),
}


def make_method(args: argparse.Namespace, cond_key: str, name: str,
                signatures: dict, sig_meta: dict) -> Any:
    """Build an agent method for a vs-bench condition."""
    label, use_nash, use_om, _ = VS_CONDITIONS[cond_key]
    if cond_key == "simple":
        return SimplePromptMethod(name=name)
    if cond_key == "suspicion":
        return SuspicionPromptMethod(name=name, lang="en")
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


def run_vs_seat(args: argparse.Namespace, model, run_dir: Path,
                agent_key: str, opp_key: str, method_position: int,
                signatures: dict, sig_meta: dict) -> dict[str, Any]:
    """Run one seat (agent at method_position vs opponent method)."""
    agent_label, _, _, agent_memory = VS_CONDITIONS[agent_key]
    opp_label, _, _, _ = VS_CONDITIONS[opp_key]
    agent = make_method(args, agent_key, f"{agent_label}-P{method_position}",
                        signatures, sig_meta)
    opponent = make_method(args, opp_key, f"{opp_label}-P{1 - method_position}",
                           signatures, sig_meta)

    game_counter: list[int] = [0]

    def post_game_hook(trajectory) -> None:
        game_counter[0] += 1
        seat_dir = run_dir / "trajectories" / agent_key / f"P{method_position}"
        seat_dir.mkdir(parents=True, exist_ok=True)
        save_trajectory(trajectory, str(seat_dir / f"game_{game_counter[0]:03d}.json"))

    t0 = time.time()
    p0, p1 = (agent, opponent) if method_position == 0 else (opponent, agent)
    stats = leduc_ablation._run_games(
        model,
        p0,
        p1,
        method_position,
        args.games,
        args.start_seed,
        args.verbose,
        condition_name=agent_key,
        archetype_name=opp_key,
        save_prompts=args.save_prompts,
        run_dir=run_dir,
        suffix=f"P{method_position}",
        refresh_memory_bank=agent_memory,
        refresh_parallel=False,
        refresh_max_workers=4,
        refresh_model_name=args.model_name,
        refresh_temperature=0.2,
        refresh_base_url=args.base_url,
        refresh_api_key=args.api_key,
        refresh_timeout=120.0,
        post_game_hook=post_game_hook,
    )
    elapsed = round(time.time() - t0, 2)

    return {
        "agent_condition": agent_key,
        "agent_label": agent_label,
        "opponent_condition": opp_key,
        "opponent_label": opp_label,
        "position": method_position,
        "games": stats["games"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "winrate": stats["winrate"],
        "total_chips": stats["total_chips"],
        "avg_chips": stats["avg_chips"],
        "elapsed_sec": elapsed,
        "results": stats.get("results", []),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leduc method-vs-method benchmark (agent vs opponent method)."
    )
    parser.add_argument("--agent-condition", default="full",
                        choices=list(VS_CONDITIONS),
                        help="Agent method (default: full).")
    parser.add_argument("--opponent-condition", default="simple",
                        choices=list(VS_CONDITIONS),
                        help="Opponent method (default: simple).")
    parser.add_argument("--run-phase", default="dev_debug",
                        choices=["dev_debug", "real_measurement", "analysis_only"])
    parser.add_argument("--approved-real-measurement", action="store_true")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--mock-model", action="store_true",
                        help="Use a deterministic mock model (dev_debug only).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-prompts", action="store_true", default=True,
                        help="Save per-step prompt/response files (default).")
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts")
    parser.add_argument("--verbose", "-v", action="store_true")

    # opponent-modeling settings (same defaults as additive.py / ood bench)
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

    if args.run_phase == "real_measurement" and not args.approved_real_measurement:
        parser.error("--run-phase real_measurement requires --approved-real-measurement")

    signatures, sig_meta = leduc_ablation.build_empirical_signature_bundle(
        args.leduc_signature_manifest,
        max_games_per_archetype=args.leduc_signature_games,
    )

    output_root = args.output_dir or "results/leduc/leduc_vs_bench"
    slug = f"leduc_{args.games}g_{args.agent_condition}_vs_{args.opponent_condition}"
    run_dir = make_run_dir(output_root, slug)

    def cond_spec(key: str) -> dict[str, Any]:
        label, use_nash, use_om, use_memory = VS_CONDITIONS[key]
        return {
            "key": key,
            "label": label,
            "use_nash": use_nash,
            "use_opponent_model": use_om,
            "use_memory": use_memory,
        }

    config = {
        "experiment": "leduc_vs_bench",
        "phase": args.run_phase,
        "agent_condition": cond_spec(args.agent_condition),
        "opponent_condition": cond_spec(args.opponent_condition),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "dirty_worktree": is_dirty_worktree(),
        "live_api": not args.mock_model,
        "mock_model": args.mock_model,
        "signature_metadata": sig_meta,
        "note": (
            "method-vs-method: both players are full LLM methods; the "
            "opponent's behavior is emergent and out-of-distribution relative "
            "to the six parametric archetypes."
        ),
    }
    write_json(run_dir / "config.json", config)

    agent_label = VS_CONDITIONS[args.agent_condition][0]
    opp_label = VS_CONDITIONS[args.opponent_condition][0]
    print(f"Run dir: {run_dir}")
    print(f"Matchup: {agent_label} (agent) vs {opp_label} (opponent)  "
          f"phase={args.run_phase}  games/position={args.games}")

    if args.mock_model:
        model = FirstLegalMockModel()
    else:
        model = OpenAIModel(
            model=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    all_results: list[dict[str, Any]] = []
    for method_position in (0, 1):
        seat = run_vs_seat(
            args, model, run_dir,
            args.agent_condition, args.opponent_condition, method_position,
            signatures, sig_meta,
        )
        all_results.append(seat)
        print(
            f"{agent_label} P{method_position}: wr={seat['winrate']:.0f}% "
            f"chips={seat['total_chips']:+.0f} "
            f"W{seat['wins']}/L{seat['losses']}/D{seat['draws']} "
            f"({seat['elapsed_sec']}s)"
        )

    csv_rows = []
    for seat in all_results:
        for row in seat["results"]:
            csv_rows.append({
                "game": row["game"],
                "agent_condition": seat["agent_condition"],
                "opponent_condition": seat["opponent_condition"],
                "position": seat["position"],
                "seed": row["seed"],
                "method_win": row["method_win"],
                "method_net": row["method_net"],
            })
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    fieldnames = ["game", "agent_condition", "opponent_condition", "position",
                  "seed", "method_win", "method_net"]
    with open(metrics_dir / "vs_per_game.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "experiment": "leduc_vs_bench",
        "phase": args.run_phase,
        "agent_condition": config["agent_condition"],
        "opponent_condition": config["opponent_condition"],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "live_api": not args.mock_model,
        "mock_model": args.mock_model,
        "results": all_results,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Results saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
