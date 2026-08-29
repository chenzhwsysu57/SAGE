"""Liar's Dice ToM exploit ablation against fixed archetype opponents.

Compares ToM vs Simple vs pure CFR baseline to isolate the effect of
opponent modeling and exploit reasoning in the ToM decision prompt.

Usage:
    # Phase 1: small batch with full prompt saving
    python runs/liars_dice/ld_tom_ablation.py --phase 1 --games 5 --save-prompts

    # Phase 2: 30-game comparison (per position)
    python runs/liars_dice/ld_tom_ablation.py --phase 2 --games 30

    # Run a single condition
    python runs/liars_dice/ld_tom_ablation.py --condition tom_cfr --games 10
    python runs/liars_dice/ld_tom_ablation.py --condition cfr --games 5
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

from games.liars_dice import LiarsDice
from apis.openai_wrapper import OpenAIModel
from methods.simple_prompt import SimplePlaceboPromptMethod, SimplePromptMethod
from methods.tom import ToMMethod
from methods.liars_dice_cfr import LiarsDiceCFRMethod
from methods.liars_dice_deep_cfr import LiarsDiceDeepCFRMethod
from methods.liars_dice_tom import LiarsDiceToMMethod
from base.case_memory_runtime import (
    append_trajectory_cases_to_archive,
    append_step_case_to_archive,
    refresh_case_memory_bank_for_step,
)
from base.memory import BaseMemory
from modeling.base import BaseModelingMethod
from modeling.liars_dice.archetypes import ArchetypeAgent, ARCHETYPE_NAMES
from rollout import RolloutRunner
from runs.liars_dice.prompt_utils import save_step_artifacts
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
from base.ablation import AblationConfig


# ── Condition definitions ────────────────────────────────────────────────

CONDITIONS = {
    "tom": (
        "ToM",
        lambda name, memory, modeling_method, verbose, **kw: ToMMethod(
            name=name, memory=memory, modeling_method=modeling_method,
            verbose=verbose, use_cfr=False,
        ),
    ),
    "simple": (
        "Simple",
        lambda name, memory, modeling_method, verbose, **kw: SimplePromptMethod(
            name=name,
        ),
    ),
    "simple_placebo": (
        "Simple-Placebo",
        lambda name, memory, modeling_method, verbose, **kw: SimplePlaceboPromptMethod(
            name=name,
            style_labels=list(ARCHETYPE_NAMES),
            target_chars=5200,
        ),
    ),
    "cfr": (
        "CFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceCFRMethod(
            name=name,
        ),
    ),
    "deep_cfr": (
        "DeepCFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceDeepCFRMethod(
            name=name,
            model_path="results/liars_dice/deep_cfr/deep_cfr_4x4",
        ),
    ),
    "tom_cfr": (
        "ToM+CFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceToMMethod(
            name=name, memory=memory, verbose=verbose, lang="en",
        ),
    ),
}

DEFAULT_CONDITIONS = ["tom", "simple", "cfr", "deep_cfr", "tom_cfr"]


def _make_opponent(archetype: str, name: str, cfr_model_path: str):
    if archetype == "GTO":
        return LiarsDiceCFRMethod(
            name=name,
            seed=hash(name) % 10000,
            strategy_path=cfr_model_path,
        )
    return ArchetypeAgent(archetype, name=name)


# ── Game runner ───────────────────────────────────────────────────────────

def _run_games(model, p0_method, p1_method, method_position: int,
               games: int, start_seed: int, verbose: bool,
               num_dice: int, num_faces: int,
               condition_name: str, archetype_name: str,
               save_prompts: bool = False, run_dir: Path | None = None,
               suffix: str = "",
               refresh_memory_bank: bool = True,
               refresh_parallel: bool = False,
               refresh_max_workers: int = 4,
               refresh_model_name: str = "deepseek-v4-flash",
               refresh_temperature: float = 0.2,
               refresh_base_url: str | None = None,
               refresh_api_key: str | None = None,
               refresh_timeout: float = 120.0) -> dict:
    """Run N games with fixed positions."""
    results = []
    wins = losses = draws = 0
    total_method_net = 0.0
    case_archive: list[dict[str, Any]] = []

    pbar = tqdm(range(games), desc=f"  {condition_name} {suffix}", unit="game",
                leave=False)
    for g in pbar:
        seed = start_seed + g
        game = LiarsDice(num_dice=num_dice, num_faces=num_faces, seed=seed)
        methods = {0: p0_method, 1: p1_method}

        def before_act_hook(ctx: dict[str, Any]) -> None:
            if not refresh_memory_bank:
                return
            if ctx.get("player") != method_position:
                return
            method = ctx.get("method")
            if method is None or not hasattr(method, "memory"):
                return
            refresh_case_memory_bank_for_step(
                game="liars_dice",
                state_before=ctx["state_before_raw"],
                legal_actions=ctx["legal_actions"],
                position=method_position,
                current_game_id=g + 1,
                current_step=int(ctx["step_num"]),
                memory=getattr(method, "memory", None),
                archive=case_archive,
                run_dir=run_dir,
                condition=condition_name,
                live_api=True,
                model_name=refresh_model_name,
                temperature=refresh_temperature,
                base_url=refresh_base_url,
                api_key=refresh_api_key,
                timeout=refresh_timeout,
                parallel=refresh_parallel,
                max_workers=refresh_max_workers,
            )

        def after_step_hook(ctx: dict[str, Any]) -> None:
            if ctx.get("player") != method_position:
                return
            append_step_case_to_archive(
                game="liars_dice",
                state_before=ctx["state_before_raw"],
                legal_actions=ctx["legal_actions"],
                position=method_position,
                current_game_id=g + 1,
                current_step=int(ctx["step_num"]),
                archive=case_archive,
                condition=condition_name,
            )

        def after_game_hook(trajectory) -> None:
            append_trajectory_cases_to_archive(
                game="liars_dice",
                trajectory=trajectory,
                position=method_position,
                current_game_id=g + 1,
                archive=case_archive,
                condition=condition_name,
            )

        runner = RolloutRunner(
            game,
            methods,
            model,
            verbose=verbose,
            before_act_hook=before_act_hook,
            after_step_hook=after_step_hook,
            after_game_hook=after_game_hook,
        )

        t0 = time.time()
        trajectory = runner.run()
        dt = time.time() - t0

        for m in [p0_method, p1_method]:
            if hasattr(m, "reflect") and callable(m.reflect):
                m.reflect(trajectory, model, str(g + 1))

        # Save prompts for Phase 1 review
        if save_prompts and run_dir:
            prompts_dir = run_dir / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            cond_dir = prompts_dir / f"{condition_name}_{suffix}"
            cond_dir.mkdir(exist_ok=True)
            for step in trajectory.steps:
                if not step.rendered_prompt:
                    continue
                method = methods.get(step.player_id)
                method_name = getattr(method, "name", f"P{step.player_id}")
                safe_name = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_"
                    for ch in method_name
                )
                prefix = f"g{g+1:02d}_step{step.step}_P{step.player_id}_{safe_name}"
                save_step_artifacts(cond_dir, prefix, step, method_name)

        method_chips = trajectory.payout.get(method_position, 0)
        opp_chips = trajectory.payout.get(1 - method_position, 0)
        net = trajectory.net_profit.get(method_position, 0.0)
        total_method_net += net

        if method_chips > opp_chips:
            wins += 1
        elif method_chips < opp_chips:
            losses += 1
        else:
            draws += 1

        results.append({
            "game": g + 1, "seed": seed,
            "method_chips": method_chips, "opp_chips": opp_chips,
            "method_net": net,
            "method_win": method_chips > opp_chips,
            "steps": len(trajectory.steps), "time": round(dt, 2),
        })

        pbar.set_postfix({"chips": f"{total_method_net:+.0f}",
                          "wr": f"{wins/(g+1)*100:.0f}%"})

    return {
        "position": method_position, "games": games,
        "wins": wins, "losses": losses, "draws": draws,
        "winrate": wins / games * 100 if games else 0,
        "total_chips": total_method_net,
        "avg_chips": total_method_net / games if games else 0,
        "results": results,
    }


# ── Main runner ───────────────────────────────────────────────────────────

def run_condition(model, condition_key: str, archetype_name: str,
                  games: int, start_seed: int, num_dice: int, num_faces: int,
                  cfr_model_path: str,
                  verbose: bool = False,
                  save_prompts: bool = False, run_dir: Path | None = None,
                  ablation: AblationConfig | None = None,
                  refresh_memory_bank: bool = True,
                  refresh_parallel: bool = False,
                  refresh_max_workers: int = 4,
                  refresh_model_name: str = "deepseek-v4-flash",
                  refresh_temperature: float = 0.2,
                  refresh_base_url: str | None = None,
                  refresh_api_key: str | None = None,
                  refresh_timeout: float = 120.0,
                  ) -> dict:
    """Run full matchup for one condition: N games P0 + N games P1."""
    display_name, make_method = CONDITIONS[condition_key]

    # --- P0: method vs archetype ---
    mm_p0 = BaseModelingMethod()
    memory_p0 = BaseMemory()
    method_p0 = make_method(
        name=f"{display_name}-P0", memory=memory_p0,
        modeling_method=mm_p0, verbose=verbose, ablation=ablation,
    )
    opp_p1 = _make_opponent(
        archetype_name, name=f"{archetype_name}-P1",
        cfr_model_path=cfr_model_path,
    )
    p0_results = _run_games(
        model, method_p0, opp_p1, 0, games, start_seed, verbose,
        num_dice, num_faces,
        display_name, archetype_name,
        save_prompts=save_prompts, run_dir=run_dir, suffix="P0",
        refresh_memory_bank=refresh_memory_bank,
        refresh_parallel=refresh_parallel,
        refresh_max_workers=refresh_max_workers,
        refresh_model_name=refresh_model_name,
        refresh_temperature=refresh_temperature,
        refresh_base_url=refresh_base_url,
        refresh_api_key=refresh_api_key,
        refresh_timeout=refresh_timeout,
    )

    # --- P1: archetype vs method ---
    opp_p0 = _make_opponent(
        archetype_name, name=f"{archetype_name}-P0",
        cfr_model_path=cfr_model_path,
    )
    mm_p1 = BaseModelingMethod()
    memory_p1 = BaseMemory()
    method_p1 = make_method(
        name=f"{display_name}-P1", memory=memory_p1,
        modeling_method=mm_p1, verbose=verbose, ablation=ablation,
    )
    p1_results = _run_games(
        model, opp_p0, method_p1, 1, games, start_seed + games, verbose,
        num_dice, num_faces,
        display_name, archetype_name,
        save_prompts=save_prompts, run_dir=run_dir, suffix="P1",
        refresh_memory_bank=refresh_memory_bank,
        refresh_parallel=refresh_parallel,
        refresh_max_workers=refresh_max_workers,
        refresh_model_name=refresh_model_name,
        refresh_temperature=refresh_temperature,
        refresh_base_url=refresh_base_url,
        refresh_api_key=refresh_api_key,
        refresh_timeout=refresh_timeout,
    )

    all_results = p0_results["results"] + p1_results["results"]
    total_games = len(all_results)
    method_wins = sum(1 for g in all_results if g["method_win"] is True)
    method_losses = sum(1 for g in all_results if g["method_win"] is False)
    method_draws = sum(1 for g in all_results if g["method_win"] is None)
    net_chips = sum(g.get("method_net", 0) for g in all_results)

    return {
        "condition_key": condition_key,
        "condition_name": display_name,
        "archetype": archetype_name,
        "games": total_games,
        "wins": method_wins, "losses": method_losses, "draws": method_draws,
        "winrate": method_wins / total_games * 100 if total_games else 0,
        "total_chips": net_chips,
        "avg_chips": net_chips / total_games if total_games else 0,
        "p0_results": p0_results,
        "p1_results": p1_results,
    }


# ── Summary printing ──────────────────────────────────────────────────────

def print_results(all_conditions: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"  RESULTS across {len(all_conditions)} conditions")
    print(f"{'='*70}")
    print(f"  {'Condition':<14} {'WinRate':>8} {'Wins':>5} {'Losses':>6} "
          f"{'Draws':>5} {'NetChips':>10} {'Avg/Game':>9}")
    print(f"  {'-'*60}")

    all_conditions.sort(key=lambda c: c["total_chips"], reverse=True)
    best = all_conditions[0]["total_chips"]

    for c in all_conditions:
        delta = ""
        if c["total_chips"] != best and best != 0:
            gap = best - c["total_chips"]
            delta = f"  (-{gap:.0f} vs best)"
        print(f"  {c['condition_name']:<14} {c['winrate']:7.1f}% "
              f"{c['wins']:>5} {c['losses']:>6} {c['draws']:>5} "
              f"{c['total_chips']:>+10.0f} {c['avg_chips']:>+9.1f}{delta}")

    print(f"\n{'='*70}")
    print(f"  POSITION BREAKDOWN")
    print(f"{'='*70}")
    print(f"  {'Condition':<14} {'P0 W-L-D':>12} {'P0 Chips':>9}  "
          f"{'P1 W-L-D':>12} {'P1 Chips':>9}")
    print(f"  {'-'*60}")
    for c in all_conditions:
        p0 = c["p0_results"]
        p1 = c["p1_results"]
        print(f"  {c['condition_name']:<14} "
              f"{p0['wins']:>2}-{p0['losses']}-{p0['draws']}  "
              f"{p0['total_chips']:>+9.0f}  "
              f"{p1['wins']:>2}-{p1['losses']}-{p1['draws']}  "
              f"{p1['total_chips']:>+9.0f}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Liar's Dice ToM exploit ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s --phase 1 --games 5\n"
               "  %(prog)s --phase 2 --games 30\n"
               "  %(prog)s --condition tom --games 10")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1=validation, Phase 2=full comparison")
    parser.add_argument("--games", type=int, default=10,
                        help="Games per position per condition")
    parser.add_argument("--num-dice", type=int, default=4)
    parser.add_argument("--num-faces", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--archetype", default="Bluffer",
                        choices=ARCHETYPE_NAMES)
    parser.add_argument("--condition", default=None,
                        choices=list(CONDITIONS.keys()),
                        help="Run only one condition (default: all)")
    parser.add_argument(
        "--cfr-model-path",
        default="results/liars_dice/cfr/20260622_142853_cfr_3x3_1m/model",
        help="Path to the trained tabular CFR model directory or file",
    )
    parser.add_argument(
        "--signature-bank-path",
        default="results/liars_dice/persona_signature_bank/20260624_190658_signature_bank_wild1/metrics/signature_bank.json",
        help="Path to the persona signature bank JSON",
    )
    parser.add_argument(
        "--deep-cfr-model-path",
        default="results/liars_dice/deep_cfr/20260621_174108_deep_cfr_4x4_10m/model",
        help="Path to the trained Deep CFR model directory or model.pt",
    )
    parser.add_argument("--no-save-prompts", action="store_true",
                        help="Skip saving full prompts and responses")
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-dir",
                        default="results/liars_dice/ld_tom_ablation")
    parser.add_argument("--ablate", default="",
                        help="Comma-separated ablation flags: nash,belief")
    parser.add_argument("--no-refresh-memory-bank", action="store_false",
                        dest="refresh_memory_bank", default=True,
                        help="Disable case-based self-evolve memory refresh (default: enabled).")
    parser.add_argument("--refresh-memory-parallel", action="store_true",
                        help="Generate refreshed memories in parallel when live API is enabled.")
    parser.add_argument("--refresh-memory-workers", type=int, default=4,
                        help="Worker count for refresh-memory parallel generation.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-game details")
    args = parser.parse_args()

    ablation_config = AblationConfig.from_comma_separated(args.ablate)
    CONDITIONS["cfr"] = (
        "CFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceCFRMethod(
            name=name,
            strategy_path=args.cfr_model_path,
        ),
    )
    CONDITIONS["deep_cfr"] = (
        "DeepCFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceDeepCFRMethod(
            name=name,
            model_path=args.deep_cfr_model_path,
        ),
    )
    CONDITIONS["tom_cfr"] = (
        "ToM+CFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceToMMethod(
            name=name, memory=memory, verbose=verbose, lang="en",
            strategy_path=args.cfr_model_path,
            signature_bank_path=args.signature_bank_path,
            ablation=kw.get("ablation"),
        ),
    )

    condition_keys = (
        [args.condition] if args.condition
        else DEFAULT_CONDITIONS
    )

    slug = f"{args.archetype}_p{args.phase}_{args.games}g_{args.num_dice}x{args.num_faces}"
    if args.ablate:
        slug += f"_ablate-{args.ablate}"
    run_dir = make_run_dir(args.output_dir, slug)
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "ablation": args.ablate or "none",
        "phase": args.phase,
    })
    print(f"Run dir: {run_dir}")

    model = OpenAIModel(
        model=args.model_name, api_key=args.api_key, base_url=args.base_url,
    )

    print(f"{'='*70}")
    print(f"  Liar's Dice ToM Exploit Ablation — Phase {args.phase}")
    print(f"  Archetype: {args.archetype}")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Games/position: {args.games}, Start seed: {args.start_seed}")
    total = len(condition_keys) * args.games * 2
    print(f"  Total: {len(condition_keys)} x {args.games * 2} = {total} games")
    print(f"  Model: {args.model_name}")
    print(f"  Save prompts: {not args.no_save_prompts}")
    print(f"{'='*70}\n")

    all_conditions = []
    for ck in condition_keys:
        display_name = CONDITIONS[ck][0]
        print(f"--- {display_name} vs {args.archetype} ---")
        t0 = time.time()
        stats = run_condition(
            model, ck, args.archetype, args.games, args.start_seed,
            args.num_dice, args.num_faces, args.cfr_model_path,
            verbose=args.verbose, save_prompts=not args.no_save_prompts,
            run_dir=run_dir, ablation=ablation_config,
            refresh_memory_bank=args.refresh_memory_bank,
            refresh_parallel=args.refresh_memory_parallel,
            refresh_max_workers=args.refresh_memory_workers,
            refresh_model_name=args.model_name,
            refresh_temperature=0.2,
            refresh_base_url=args.base_url,
            refresh_api_key=args.api_key,
        )
        dt = time.time() - t0
        print(f"  Result: wr={stats['winrate']:.0f}%  "
              f"chips={stats['total_chips']:+.0f}  "
              f"avg={stats['avg_chips']:+.1f}/game  "
              f"W{stats['wins']}/L{stats['losses']}/D{stats['draws']}  "
              f"({dt:.0f}s)\n")
        all_conditions.append(stats)

    print_results(all_conditions)

    write_json(run_dir / "summary.json", {
        "experiment": "ld_tom_ablation",
        "phase": args.phase,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": args.model_name,
        "archetype": args.archetype,
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "conditions": all_conditions,
    })
    print(f"\nResults saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
