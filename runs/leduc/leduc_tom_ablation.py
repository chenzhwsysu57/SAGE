"""ToM exploit ablation against a fixed archetype.

Compares ToM-CFR, ToM-NoCFR, Simple, and optional pure CFR baseline to
isolate the effect of CFR Nash analysis + exploit reasoning in the ToM
decision prompt.

Usage:
    # Default: saves prompts (use --no-save-prompts to disable)
    python runs/leduc/leduc_tom_ablation.py --phase 1 --games 10

    # Phase 2: 100-game comparison (disable prompt saving for speed)
    python runs/leduc/leduc_tom_ablation.py --phase 2 --games 100 --no-save-prompts

    # Run a single condition
    python runs/leduc/leduc_tom_ablation.py --condition tom_cfr --games 50
    python runs/leduc/leduc_tom_ablation.py --condition cfr --games 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from games.leduc import LeducHoldem
from apis.openai_wrapper import OpenAIModel
from methods.simple_prompt import SimplePlaceboPromptMethod, SimplePromptMethod
from methods.suspicion_prompt import SuspicionPromptMethod
from methods.leduc_tom import LeducToMMethod
from base.case_memory_runtime import (
    append_trajectory_cases_to_archive,
    append_step_case_to_archive,
    refresh_case_memory_bank_for_step,
)
from base.memory import BaseMemory
from modeling.leduc import LLMBeliefMethod, StatisticalProfileMethod, ArchetypeAgent, CFRGTOAgent
from modeling.leduc.archetypes import ARCHETYPE_NAMES
import modeling.leduc.statistical_profile as _stat_profile
from rollout import RolloutRunner
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
from base.ablation import AblationConfig


# ── Condition definitions ────────────────────────────────────────────────

CONDITIONS = {
    "tom_cfr": (
        "ToM-CFR",
        lambda name, memory, modeling_method, verbose, **kw: LeducToMMethod(
            name=name, memory=memory, modeling_method=modeling_method,
            verbose=verbose, use_cfr=True, ablation=kw.get("ablation"),
        ),
    ),
    "tom_nocfr": (
        "ToM-NoCFR",
        lambda name, memory, modeling_method, verbose, **kw: LeducToMMethod(
            name=name, memory=memory, modeling_method=modeling_method,
            verbose=verbose, use_cfr=False, ablation=kw.get("ablation"),
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
    "suspicion": (
        "SuspicionPrompt",
        lambda name, memory, modeling_method, verbose, **kw: SuspicionPromptMethod(
            name=name, lang="en",
        ),
    ),
    "cfr": (
        "CFR",
        lambda name, memory, modeling_method, verbose, **kw: CFRGTOAgent(
            name=name,
        ),
    ),
}


def _make_opponent(archetype: str, name: str, seed: int = 0):
    if archetype == "GTO":
        return CFRGTOAgent(name=name, seed=hash(name) % 10000)
    return ArchetypeAgent(archetype, name=name)


def _load_signature_manifest(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "entries" not in data or not isinstance(data["entries"], dict):
        raise ValueError(f"Signature manifest missing entries: {path}")
    return data


def build_empirical_signature_bundle(
    manifest_path: str,
    max_games_per_archetype: int = 0,
) -> tuple[dict[str, dict[str, float]], dict]:
    """Build data-driven signatures from all labeled P0/P1 historical batches."""
    manifest = _load_signature_manifest(manifest_path)
    profiles = {}
    metadata = {
        "mode": "data",
        "source": "historical_manifest",
        "manifest_path": manifest_path,
        "max_games_per_archetype": max_games_per_archetype or None,
        "fallback_to_persona": False,
        "position_aggregation": "p0_p1_merged",
        "archetypes": {},
    }

    for arch in ARCHETYPE_NAMES:
        item = manifest["entries"].get(arch)
        if item is None:
            raise ValueError(f"Signature manifest has no entry for {arch}: {manifest_path}")

        sp = StatisticalProfileMethod(memory=BaseMemory())
        arch_meta = {
            "source_files": [],
            "batch_labels": [],
            "legacy_train_json": item.get("train_json"),
            "legacy_train_batch_idx": item.get("train_batch_idx"),
            "legacy_opp_id": item.get("opp_id"),
            "p0_games": 0,
            "p1_games": 0,
            "total_games": 0,
            "p0_actions": 0,
            "p1_actions": 0,
            "total_actions": 0,
        }

        source_files = []
        for key in ("train_json", "test_json"):
            path = item.get(key)
            if path and path not in source_files:
                source_files.append(path)

        for source_path in source_files:
            with open(source_path) as f:
                data = json.load(f)
            for batch_idx, batch in enumerate(data.get("batches", [])):
                label = str(batch.get("label", ""))
                player_id = _archetype_player_from_label(label, arch)
                if player_id is None:
                    continue
                trajectories = batch.get("sample_trajectories", [])
                if max_games_per_archetype > 0:
                    remaining = max_games_per_archetype - arch_meta["total_games"]
                    if remaining <= 0:
                        break
                    trajectories = trajectories[:remaining]
                if not trajectories:
                    continue

                games_loaded = 0
                actions_loaded = 0
                for traj in trajectories:
                    n_actions = sp._accumulate_game(player_id, traj)
                    if n_actions > 0:
                        games_loaded += 1
                        actions_loaded += n_actions
                if games_loaded == 0:
                    continue

                pos_key = "p0" if player_id == 0 else "p1"
                arch_meta[f"{pos_key}_games"] += games_loaded
                arch_meta[f"{pos_key}_actions"] += actions_loaded
                arch_meta["total_games"] += games_loaded
                arch_meta["total_actions"] += actions_loaded
                arch_meta["batch_labels"].append({
                    "source_file": source_path,
                    "batch_idx": batch_idx,
                    "label": label,
                    "player_id": player_id,
                    "games": games_loaded,
                    "actions": actions_loaded,
                })
                if source_path not in arch_meta["source_files"]:
                    arch_meta["source_files"].append(source_path)

        profile = _stat_profile._build_profile(
            sp._load_actions(0) + sp._load_actions(1),
            sp._load_strengths(0) + sp._load_strengths(1),
        )
        profile.n_games = arch_meta["total_games"]
        if arch_meta["total_games"] == 0 or profile.n_actions == 0:
            raise ValueError(
                f"No usable P0/P1 historical actions for {arch} in {source_files}"
            )
        profiles[arch] = profile
        arch_meta["games"] = arch_meta["total_games"]
        arch_meta["actions"] = profile.n_actions
        metadata["archetypes"][arch] = arch_meta

    signatures = StatisticalProfileMethod.estimate_signatures(profiles)
    return signatures, metadata


def _archetype_player_from_label(label: str, archetype: str) -> int | None:
    if not label:
        return None
    if f"_vs_{archetype}_" in label:
        return 1
    if label.startswith(f"{archetype}_vs_"):
        return 0
    return None


# ── Game runner ───────────────────────────────────────────────────────────

def _run_games(model, p0_method, p1_method, method_position: int,
               games: int, start_seed: int, verbose: bool,
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
               refresh_timeout: float = 120.0,
               before_game_hook: Callable[[dict], None] | None = None,
               post_game_hook: Callable[[Any], None] | None = None) -> dict:
    """Run N games with fixed positions."""
    results = []
    wins = losses = draws = 0
    total_method_net = 0.0
    case_archive: list[dict[str, Any]] = []

    pbar = tqdm(range(games), desc=f"  {condition_name} {suffix}", unit="game",
                leave=False)
    for g in pbar:
        seed = start_seed + g
        game = LeducHoldem(seed=seed)
        methods = {0: p0_method, 1: p1_method}
        target_method = methods.get(method_position)
        if before_game_hook:
            before_game_hook({
                "game_index": g + 1,
                "method_position": method_position,
                "method": target_method,
                "opponent": methods.get(1 - method_position),
                "methods": methods,
            })

        def before_act_hook(ctx: dict[str, Any]) -> None:
            if not refresh_memory_bank or target_method is None:
                return
            if ctx.get("player") != method_position:
                return
            method = ctx.get("method")
            if method is None or not hasattr(method, "memory"):
                return
            refresh_case_memory_bank_for_step(
                game="leduc",
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
                game="leduc",
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
                game="leduc",
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

        if post_game_hook:
            post_game_hook(trajectory)

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
                p_path = cond_dir / f"{prefix}_prompt.txt"
                with open(p_path, "w") as f:
                    f.write(step.rendered_prompt)
                r_path = cond_dir / f"{prefix}_response.txt"
                with open(r_path, "w") as f:
                    f.write(step.raw_llm_response or "")
                method_data = step.method_data or {}
                if isinstance(method_data, dict):
                    suspicion = method_data.get("suspicion")
                    if isinstance(suspicion, dict):
                        for stage in ("pattern", "belief"):
                            sp = cond_dir / f"{prefix}_{stage}_prompt.txt"
                            with open(sp, "w") as f:
                                f.write(str(suspicion.get(f"{stage}_prompt", "")))
                            sr = cond_dir / f"{prefix}_{stage}_response.txt"
                            with open(sr, "w") as f:
                                f.write(str(suspicion.get(f"{stage}_response", "")))

        method_payout = trajectory.payout.get(method_position, 0)
        opp_payout = trajectory.payout.get(1 - method_position, 0)
        net = trajectory.net_profit.get(method_position, 0.0)
        total_method_net += net

        if method_payout > opp_payout:
            wins += 1; method_win = True
        elif method_payout < opp_payout:
            losses += 1; method_win = False
        else:
            draws += 1; method_win = None

        results.append({
            "game": g + 1, "seed": seed,
            "method_chips": method_payout, "opp_chips": opp_payout,
            "method_net": net,
            "method_win": method_win,
            "steps": len(trajectory.steps), "time": round(dt, 2),
        })

        pbar.set_postfix({"chips": f"{total_method_net:+.0f}", "wr": f"{wins/(g+1)*100:.0f}%"})

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
                  games: int, start_seed: int, verbose: bool = False,
                  save_prompts: bool = False, run_dir: Path | None = None,
                  modeling_method: str = "llm", decay: float = 0.85,
                  signatures: dict[str, dict[str, float]] | None = None,
                  signature_metadata: dict | None = None,
                  stat_mode: str = "q_window_log",
                  q_window: int = 10,
                  q_alpha: float = 0.85,
                  q_tau: float = 1.5,
                  ablation: AblationConfig | None = None,
                  refresh_memory_bank: bool = True,
                  refresh_parallel: bool = False,
                  refresh_max_workers: int = 4,
                  refresh_model_name: str = "deepseek-v4-flash",
                  refresh_temperature: float = 0.2,
                  refresh_base_url: str | None = None,
                  refresh_api_key: str | None = None,
                  refresh_timeout: float = 120.0) -> dict:
    """Run full matchup for one condition: N games P0 + N games P1."""
    display_name, make_method = CONDITIONS[condition_key]

    effective_stat_mode = stat_mode

    # --- P0: method vs archetype ---
    memory_p0 = BaseMemory()
    if modeling_method == "statistical":
        mm_p0 = StatisticalProfileMethod(memory=BaseMemory(),
                                          true_archetype=archetype_name,
                                          signatures=signatures,
                                          signature_source=str((signature_metadata or {}).get("source", "historical_manifest")),
                                          signature_metadata=signature_metadata,
                                          classification_mode=effective_stat_mode,
                                          q_window=q_window,
                                          q_alpha=q_alpha,
                                          q_tau=q_tau)
    else:
        prompts_dir = run_dir / "prompts" if save_prompts and run_dir else None
        mm_p0 = LLMBeliefMethod(memory=BaseMemory(), decay=decay,
                                 prompts_dir=prompts_dir)
    method_p0 = make_method(
        name=f"{display_name}-P0", memory=memory_p0,
        modeling_method=mm_p0, verbose=verbose, ablation=ablation,
    )
    opp_p1 = _make_opponent(archetype_name, name=f"{archetype_name}-P1")
    p0_results = _run_games(
        model, method_p0, opp_p1, 0, games, start_seed, verbose,
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
    opp_p0 = _make_opponent(archetype_name, name=f"{archetype_name}-P0")
    memory_p1 = BaseMemory()
    if modeling_method == "statistical":
        mm_p1 = StatisticalProfileMethod(memory=BaseMemory(),
                                          true_archetype=archetype_name,
                                          signatures=signatures,
                                          signature_source=str((signature_metadata or {}).get("source", "historical_manifest")),
                                          signature_metadata=signature_metadata,
                                          classification_mode=effective_stat_mode,
                                          q_window=q_window,
                                          q_alpha=q_alpha,
                                          q_tau=q_tau)
    else:
        mm_p1 = LLMBeliefMethod(memory=BaseMemory(), decay=decay,
                                 prompts_dir=prompts_dir)
    method_p1 = make_method(
        name=f"{display_name}-P1", memory=memory_p1,
        modeling_method=mm_p1, verbose=verbose, ablation=ablation,
    )
    p1_results = _run_games(
        model, opp_p0, method_p1, 1, games, start_seed + games, verbose,
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
        description="ToM exploit ablation: tom_cfr vs tom_nocfr vs simple, with optional cfr baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s --phase 1 --games 10\n"
               "  %(prog)s --phase 2 --games 100\n"
               "  %(prog)s --condition tom_cfr --games 50\n"
               "  %(prog)s --condition cfr --games 5")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1=validation, Phase 2=full comparison")
    parser.add_argument("--games", type=int, default=10,
                        help="Games per position per condition")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--archetype", default="CallingStation",
                        choices=["Maniac", "CallingStation", "LAG", "TAG", "Nit", "GTO"])
    parser.add_argument("--condition", default=None,
                        choices=list(CONDITIONS.keys()),
                        help="Run only one condition (default: ToM-CFR, ToM-NoCFR, Simple, SuspicionPrompt)")
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts",
                        default=True,
                        help="Don't save prompts and responses to disk")
    parser.add_argument("--method", default="statistical",
                        choices=["llm", "statistical", "sage"],
                        help="Opponent modeling method; sage selects the full ToM-CFR agent")
    parser.add_argument("--signature-manifest",
                        default="runs/leduc/manifests/classify_all6_legacy.json",
                        help="Manifest with labeled historical games for data-driven signatures")
    parser.add_argument("--signature-games", type=int, default=0,
                        help="Max historical games per archetype for signatures; 0=all")
    parser.add_argument("--stat-mode", default="q_window_log",
                        choices=["q_window_ema", "q_window_log", "cumulative"],
                        help="Statistical classifier update mode (default: q_window_log)")
    parser.add_argument("--q-window", type=int, default=10,
                        help="Recent-game window for q-window statistical modes")
    parser.add_argument("--q-alpha", type=float, default=0.85,
                        help="EMA retention alpha for --stat-mode q_window_ema")
    parser.add_argument("--q-tau", type=float, default=1.5,
                        help="Log update strength for --stat-mode q_window_log")
    parser.add_argument("--decay", type=float, default=0.85,
                        help="Decay factor for LLMBeliefMethod (default: 0.85)")
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-dir", default="results/leduc/leduc_tom_ablation")
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
    requested_method = args.method
    if args.method == "sage":
        if args.condition not in (None, "tom_cfr"):
            parser.error("--method sage only supports the tom_cfr condition")
        args.condition = "tom_cfr"
        args.method = "statistical"

    ablation_config = AblationConfig.from_comma_separated(args.ablate)

    condition_keys = (
        [args.condition] if args.condition
        else ["tom_cfr", "tom_nocfr", "simple"]
    )

    signatures = None
    signature_metadata = None
    if args.method == "statistical":
        signatures, signature_metadata = build_empirical_signature_bundle(
            args.signature_manifest,
            max_games_per_archetype=args.signature_games,
        )
    slug = f"{args.archetype}_p{args.phase}_{args.games}g"
    if args.ablate:
        slug += f"_ablate-{args.ablate}"
    run_dir = make_run_dir(args.output_dir, slug)
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "requested_method": requested_method,
        "git_commit": git_commit(),
        "ablation": args.ablate or "none",
        "phase": args.phase,
        "signature_metadata": signature_metadata,
    })
    print(f"Run dir: {run_dir}")

    model = OpenAIModel(
        model=args.model_name, api_key=args.api_key, base_url=args.base_url,
    )

    print(f"{'='*70}")
    print(f"  ToM Exploit Ablation — Phase {args.phase}")
    print(f"  Archetype: {args.archetype}")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Games/position: {args.games}, Start seed: {args.start_seed}")
    total = len(condition_keys) * args.games * 2
    print(f"  Total: {len(condition_keys)} x {args.games * 2} = {total} games")
    print(f"  Model: {args.model_name}")
    print(f"  Save prompts: {args.save_prompts}")
    print(f"{'='*70}\n")

    all_conditions = []
    for ck in condition_keys:
        display_name = CONDITIONS[ck][0]
        print(f"--- {display_name} vs {args.archetype} ---")
        t0 = time.time()
        stats = run_condition(
            model, ck, args.archetype, args.games, args.start_seed,
            verbose=args.verbose, save_prompts=args.save_prompts,
            run_dir=run_dir, modeling_method=args.method, decay=args.decay,
            signatures=signatures, signature_metadata=signature_metadata,
            stat_mode=args.stat_mode, q_window=args.q_window,
            q_alpha=args.q_alpha, q_tau=args.q_tau,
            ablation=ablation_config,
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
        "experiment": "leduc_tom_ablation",
        "phase": args.phase,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": args.model_name,
        "archetype": args.archetype,
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "ablation": args.ablate or "none",
        "requested_method": requested_method,
        "modeling_method": args.method,
        "stat_mode": args.stat_mode,
        "q_window": args.q_window,
        "q_alpha": args.q_alpha,
        "q_tau": args.q_tau,
        "signature_metadata": signature_metadata,
        "conditions": all_conditions,
    })
    print(f"\nResults saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
