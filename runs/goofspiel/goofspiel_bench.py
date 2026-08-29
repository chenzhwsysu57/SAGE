"""Goofspiel benchmark: LLM methods vs fixed persona opponents.

Runs N games of each method vs each persona (both positions),
collects per-game and aggregate statistics.

Usage:
    # Phase 1: validate all personas and LLM pipeline
    python runs/goofspiel/goofspiel_bench.py --phase 1 --games 4 --save-prompts

    # Phase 2: N-game comparison
    python runs/goofspiel/goofspiel_bench.py --phase 2 --games 30

    # Only specific personas
    python runs/goofspiel/goofspiel_bench.py --personas Greedy,Conservative,Random

    # Dry-run: no LLM, just personas vs personas
    python runs/goofspiel/goofspiel_bench.py --dry-run --games 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from games.goofspiel import Goofspiel
from apis.openai_wrapper import OpenAIModel
from methods.simple_prompt import SimplePlaceboPromptMethod, SimplePromptMethod
from methods.tom import ToMMethod
from methods.goofspiel_cfr_method import GoofspielCFRMethod
from base.case_memory_runtime import (
    append_trajectory_cases_to_archive,
    append_step_case_to_archive,
    refresh_case_memory_bank_for_step,
)
from base.memory import BaseMemory
from rollout import RolloutRunner
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
from base.ablation import AblationConfig
from modeling.goofspiel import (
    PERSONAS, PERSONA_NAMES, PERSONA_DESCRIPTIONS_TEXT,
    GoofspielArchetypeAgent,
)
from modeling.goofspiel.classifier import PersonaModelingMethod
from methods.goofspiel_tom import GoofspielToMMethod

# Lazy-loaded Goofspiel CFR renderer for ToM prompt
_goofspiel_cfr_renderer_instance = None

def _goofspiel_cfr_renderer(observation, legal_actions) -> str:
    global _goofspiel_cfr_renderer_instance
    if _goofspiel_cfr_renderer_instance is None:
        _goofspiel_cfr_renderer_instance = GoofspielCFRMethod(model_path=None)
    return _goofspiel_cfr_renderer_instance.format_for_prompt(observation, legal_actions)


# ── Conditions ──────────────────────────────────────────────────────────

CONDITIONS = {
    "simple": (
        "Simple",
        lambda name, **kw: SimplePromptMethod(name=name),
    ),
    "simple_placebo": (
        "Simple-Placebo",
        lambda name, **kw: SimplePlaceboPromptMethod(
            name=name,
            style_labels=list(PERSONA_NAMES),
            target_chars=4000,
        ),
    ),
    "tom": (
        "ToM",
        lambda name, **kw: ToMMethod(
            name=name,
            memory=BaseMemory(),
            verbose=kw.get("verbose", False),
            use_cfr=True,
            cfr_renderer=_goofspiel_cfr_renderer,
            modeling_method=PersonaModelingMethod(),
        ),
    ),
    "tom_bayes": (
        "ToM-Bayes",
        lambda name, **kw: GoofspielToMMethod(
            name=name,
            memory=BaseMemory(),
            lang=kw.get("lang", "en"),
            verbose=kw.get("verbose", False),
            use_cfr=True,
            cfr_model_path=kw.get("cfr_model", None),
            ablation=kw.get("ablation"),
        ),
    ),
    "cfr": (
        "CFR",
        lambda name, **kw: GoofspielCFRMethod(
            name=name,
            model_path=kw.get("cfr_model", None),
            seed=kw.get("seed", 0),
        ),
    ),
}

DEFAULT_CONDITIONS = ["simple", "tom", "cfr"]

# ── Opponent factory ────────────────────────────────────────────────────

def _make_opponent(persona_name: str, *, suffix: str = "",
                   cfr_model: str | None = None, seed: int = 0):
    """Create opponent agent. GTO uses CFR method; others use ArchetypeAgent."""
    if persona_name == "GTO":
        return GoofspielCFRMethod(
            model_path=cfr_model, name=f"GTO-{suffix}", seed=seed,
        )
    return GoofspielArchetypeAgent(persona_name, name=f"{persona_name}-{suffix}",
                                    seed=seed)


# ── Game runner ─────────────────────────────────────────────────────────

def _run_games(model, p0_method, p1_method, method_position: int,
               games: int, start_seed: int, verbose: bool,
               condition_name: str, persona_name: str,
               dry_run: bool = False,
               save_prompts: bool = False, run_dir: Path | None = None,
               suffix: str = "", n: int = 6,
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

    pbar = tqdm(total=games, desc=f"{condition_name} vs {persona_name} ({suffix})",
                 unit="game", bar_format="{desc}  {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
    for g in range(games):
        seed = start_seed + g
        game = Goofspiel(n=n, seed=seed)
        methods = {0: p0_method, 1: p1_method}

        def before_act_hook(ctx: dict[str, Any]) -> None:
            if not refresh_memory_bank or dry_run:
                return
            if ctx.get("player") != method_position:
                return
            method = ctx.get("method")
            if method is None or not hasattr(method, "memory"):
                return
            refresh_case_memory_bank_for_step(
                game="goofspiel",
                state_before=ctx["state_before_raw"],
                legal_actions=ctx["legal_actions"],
                position=method_position,
                current_game_id=g + 1,
                current_step=int(ctx["step_num"]),
                memory=getattr(method, "memory", None),
                archive=case_archive,
                run_dir=run_dir,
                condition=condition_name,
                live_api=bool(model is not None),
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
                game="goofspiel",
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
                game="goofspiel",
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

        # Post-game reflection (ToM updates its opponent model)
        for m in [p0_method, p1_method]:
            if hasattr(m, "reflect") and callable(m.reflect):
                m.reflect(trajectory, model, str(g + 1))

        # Save prompts if requested
        if save_prompts and run_dir:
            prompts_dir = run_dir / "prompts" / persona_name
            prompts_dir.mkdir(parents=True, exist_ok=True)
            cond_dir = prompts_dir / f"{condition_name}_{suffix}"
            cond_dir.mkdir(exist_ok=True)
            for step in trajectory.steps:
                if not step.rendered_prompt:
                    continue
                safe_name = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_"
                    for ch in getattr(methods.get(step.player_id), "name",
                                      f"P{step.player_id}")
                )
                prefix = f"g{g+1:02d}_step{step.step}_P{step.player_id}_{safe_name}"
                with open(cond_dir / f"{prefix}_prompt.txt", "w") as f:
                    f.write(step.rendered_prompt)
                if step.raw_llm_response:
                    with open(cond_dir / f"{prefix}_response.txt", "w") as f:
                        f.write(step.raw_llm_response)

        # Track classifier posterior (ToM condition only)
        posterior = None
        for m in [p0_method, p1_method]:
            if hasattr(m, "modeling_method") and hasattr(m.modeling_method, "classifier"):
                clf = m.modeling_method.classifier
                posterior = {k: round(v, 3) for k, v in clf.posterior(temperature=0.5).items()}
                break

        # Track statistics
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

        # Update progress bar
        played = g + 1
        pbar.set_description_str(
            f"{condition_name} vs {persona_name} ({suffix})  "
            f"{condition_name}_wr={wins/played*100:.0f}%  {condition_name}_chips={total_method_net:+.0f}"
        )
        pbar.update(1)

        # Record per-round persona actions for behavioral analysis
        persona_actions = _extract_persona_actions(trajectory, 1 - method_position)

        results.append({
            "game": g + 1, "seed": seed,
            "method_chips": method_chips, "opp_chips": opp_chips,
            "method_net": net,
            "method_win": True if method_chips > opp_chips else (False if method_chips < opp_chips else None),
            "steps": len(trajectory.steps), "time": round(dt, 2),
            "persona_actions": persona_actions,
            "posterior": posterior,
        })

    pbar.close()

    return {
        "position": method_position, "games": games,
        "wins": wins, "losses": losses, "draws": draws,
        "winrate": wins / games * 100 if games else 0,
        "total_chips": total_method_net,
        "avg_chips": total_method_net / games if games else 0,
        "results": results,
    }


def _extract_persona_actions(trajectory, persona_player: int) -> list[dict]:
    """Extract the persona's actions round by round from a trajectory."""
    extracted = []
    for step in trajectory.steps:
        if step.player_id == persona_player and step.parsed_action:
            obs = step.observation
            pub = obs.public_info if obs else {}
            a = step.parsed_action
            game_round = pub.get("round", step.step // 2 + 1)
            entry = {
                "round": game_round,
                "card": a.amount,
                "prize": pub.get("prize", 0),
                "score_me": pub.get("scores", {}).get(persona_player, 0),
                "score_opp": pub.get("scores", {}).get(1 - persona_player, 0),
                "outcome_history": list(pub.get("outcome_history", [])),
            }
            if step.state_before:
                hands_dict = step.state_before.get("hands", {})
                str_pid = str(persona_player)
                if str_pid in hands_dict:
                    entry["hand_before"] = sorted(hands_dict[str_pid])
            extracted.append(entry)
    return extracted


# ── Main runner ─────────────────────────────────────────────────────────

def run_condition(model, condition_key: str, persona_name: str,
                  games: int, start_seed: int, verbose: bool = False,
                  dry_run: bool = False,
                  save_prompts: bool = False, run_dir: Path | None = None,
                  cfr_model: str | None = None,
                  n: int = 6,
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

    # --- P0: method vs persona ---
    method_p0 = make_method(name=f"{display_name}-P0", verbose=verbose,
                            cfr_model=cfr_model, seed=start_seed,
                            ablation=ablation)
    opp_p1 = _make_opponent(persona_name, suffix="P1", cfr_model=cfr_model,
                             seed=start_seed + 1000)
    p0_results = _run_games(
        model, method_p0, opp_p1, 0, games, start_seed, verbose,
        display_name, persona_name,
        dry_run=dry_run,
        save_prompts=save_prompts, run_dir=run_dir, suffix="P0",
        n=n,
        refresh_memory_bank=refresh_memory_bank,
        refresh_parallel=refresh_parallel,
        refresh_max_workers=refresh_max_workers,
        refresh_model_name=refresh_model_name,
        refresh_temperature=refresh_temperature,
        refresh_base_url=refresh_base_url,
        refresh_api_key=refresh_api_key,
        refresh_timeout=refresh_timeout,
    )

    # --- P1: persona vs method ---
    opp_p0 = _make_opponent(persona_name, suffix="P0", cfr_model=cfr_model,
                             seed=start_seed + 2000)
    method_p1 = make_method(name=f"{display_name}-P1", verbose=verbose,
                            cfr_model=cfr_model, seed=start_seed + games,
                            ablation=ablation)
    p1_results = _run_games(
        model, opp_p0, method_p1, 1, games, start_seed + games, verbose,
        display_name, persona_name,
        dry_run=dry_run,
        save_prompts=save_prompts, run_dir=run_dir, suffix="P1",
        n=n,
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
    method_chips = sum(g["method_chips"] for g in all_results)
    net_chips = method_chips  # zero-sum: opp_chips = -method_chips

    return {
        "condition_key": condition_key,
        "condition_name": display_name,
        "persona": persona_name,
        "games": total_games,
        "wins": method_wins, "losses": method_losses, "draws": method_draws,
        "winrate": method_wins / total_games * 100 if total_games else 0,
        "total_chips": net_chips,
        "avg_chips": net_chips / total_games if total_games else 0,
        "p0_results": p0_results,
        "p1_results": p1_results,
    }


# ── Persona profiling (dry-run mode) ────────────────────────────────────

def profile_persona(persona_name: str, games: int, start_seed: int,
                    n: int = 6, cfr_model: str | None = None) -> dict:
    """Run personas against each other to generate behavioral profile."""
    profile = {"persona": persona_name, "games": []}
    for g in range(games):
        seed = start_seed + g
        p0 = _make_opponent(persona_name, suffix="P0", cfr_model=cfr_model,
                             seed=seed)
        p1 = _make_opponent("Random", suffix="P1", seed=seed + 1000)
        game = Goofspiel(n=n)
        methods = {0: p0, 1: p1}
        runner = RolloutRunner(game, methods, model=None, verbose=False)
        trajectory = runner.run()
        actions = _extract_persona_actions(trajectory, 0)
        profile["games"].append({
            "game": g + 1, "seed": seed,
            "payout": trajectory.payout,
            "persona_actions": actions,
        })
    return profile


# ── Summary printing ────────────────────────────────────────────────────

def print_results(all_results: list[dict]) -> None:
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  {'Condition':<14} {'Persona':<14} {'WinRate':>8} {'Wins':>5} "
          f"{'Losses':>6} {'Draws':>5} {'NetChips':>10} {'Avg/Game':>9}")
    print(f"  {'-'*68}")

    for c in all_results:
        print(f"  {c['condition_name']:<14} {c['persona']:<14} "
              f"{c['winrate']:7.1f}% {c['wins']:>5} {c['losses']:>6} "
              f"{c['draws']:>5} {c['total_chips']:>+10.0f} "
              f"{c['avg_chips']:>+9.1f}")

    print(f"\n{'='*80}")
    print(f"  POSITION BREAKDOWN")
    print(f"{'='*80}")
    print(f"  {'Condition':<14} {'Persona':<14} {'P0 W-L-D':>12} {'P0 Chips':>9}  "
          f"{'P1 W-L-D':>12} {'P1 Chips':>9}")
    print(f"  {'-'*68}")
    for c in all_results:
        p0, p1 = c["p0_results"], c["p1_results"]
        print(f"  {c['condition_name']:<14} {c['persona']:<14} "
              f"{p0['wins']:>2}-{p0['losses']}-{p0['draws']}  "
              f"{p0['total_chips']:>+9.0f}  "
              f"{p1['wins']:>2}-{p1['losses']}-{p1['draws']}  "
              f"{p1['total_chips']:>+9.0f}")

    # Convergence summary for ToM condition
    tom_results = [c for c in all_results if c["condition_key"] == "tom"]
    if tom_results:
        print_convergence_summary(tom_results)


def print_convergence_summary(tom_results: list[dict]) -> None:
    """Print classifier posterior convergence for ToM condition."""
    print(f"\n{'='*80}")
    print(f"  CLASSIFIER CONVERGENCE (ToM condition only)")
    print(f"{'='*80}")

    for c in tom_results:
        pn = c["persona"]
        print(f"\n  True persona: {pn}")

        for label, pos_key in [("P0 (method=P0)", "p0_results"), ("P1 (method=P1)", "p1_results")]:
            pos_data = c[pos_key]
            posteriors = [g.get("posterior") for g in pos_data["results"] if g.get("posterior")]
            if not posteriors:
                continue

            checkpoints = [0, 1, 2, 4, 9, 14, 19, 29]
            checkpoints = [cp for cp in checkpoints if cp < len(posteriors)]

            print(f"    {label}:")
            print(f"      {'Game':>5}  {'P(correct)':>10}  {'Rank':>5}  {'Top-3'}")

            for cp in checkpoints:
                post = posteriors[cp]
                correct_prob = post.get(pn, 0.0)
                ranked = sorted(post.items(), key=lambda x: -x[1])
                rank = next(i + 1 for i, (p, _) in enumerate(ranked) if p == pn)
                top3 = ", ".join(f"{p}={v:.2f}" for p, v in ranked[:3])
                print(f"      {cp+1:>5}  {correct_prob:>10.3f}  {rank:>5}  [{top3}]")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Goofspiel benchmark: LLM methods vs persona opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1=validation, Phase 2=full comparison")
    parser.add_argument("--games", type=int, default=10,
                        help="Games per position per condition")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--personas", default="all",
                        help="Comma-separated persona list, or 'all'")
    parser.add_argument("--condition", default=None,
                        choices=list(CONDITIONS),
                        help="Run only one condition")
    parser.add_argument("--dry-run", action="store_true",
                        help="No LLM: personas vs personas only (profile mode)")
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts",
                        default=True,
                        help="Don't save prompts and responses to disk")
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--cfr-model", default=None,
                        help="Path to CFR model pickle (default: auto-detect)")
    parser.add_argument("--output-dir", default="results/goofspiel/goofspiel_bench")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of cards/prizes (default: 5)")
    parser.add_argument("--ablate", default="",
                        help="Comma-separated ablation flags: nash,belief")
    parser.add_argument("--no-refresh-memory-bank", action="store_false",
                        dest="refresh_memory_bank", default=True,
                        help="Disable case-based self-evolve memory refresh (default: enabled).")
    parser.add_argument("--refresh-memory-parallel", action="store_true",
                        help="Generate refreshed memories in parallel when live API is enabled.")
    parser.add_argument("--refresh-memory-workers", type=int, default=4,
                        help="Worker count for refresh-memory parallel generation.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    ablation_config = AblationConfig.from_comma_separated(args.ablate)
    # Resolve persona list
    if args.personas == "all":
        persona_list = PERSONA_NAMES
    else:
        persona_list = [p.strip() for p in args.personas.split(",")]
        unknown = [p for p in persona_list if p not in PERSONA_NAMES]
        if unknown:
            parser.error(f"Unknown personas: {unknown}. Available: {PERSONA_NAMES}")

    # Resolve conditions
    if args.dry_run:
        condition_keys = []
    elif args.condition:
        condition_keys = [args.condition]
    else:
        condition_keys = DEFAULT_CONDITIONS

    # Create run directory
    slug = f"p{args.phase}_{args.games}g_{'_'.join(persona_list)}"
    if args.ablate:
        slug += f"_ablate-{args.ablate}"
    if args.dry_run:
        slug = f"dryrun_{slug}"
    run_dir = make_run_dir(args.output_dir, slug)
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "ablation": args.ablate or "none",
        "phase": args.phase,
    })
    print(f"Run dir: {run_dir}")

    # Initialize model (unless dry-run)
    model = None
    if not args.dry_run:
        model = OpenAIModel(
            model=args.model_name, api_key=args.api_key,
            base_url=args.base_url,
        )

    total_games = 0
    all_results = []

    print(f"\n{'='*70}")
    print(f"  Goofspiel Benchmark — Phase {args.phase}")
    print(f"  Personas: {', '.join(persona_list)}")
    if not args.dry_run:
        print(f"  Conditions: {', '.join(condition_keys)}")
    else:
        print(f"  Mode: DRY RUN (persona profiling, no LLM)")
    print(f"  Games/position: {args.games}, Start seed: {args.start_seed}")
    if not args.dry_run:
        total = len(condition_keys) * len(persona_list) * args.games * 2
        print(f"  Total: {len(condition_keys)} conditions x {len(persona_list)} personas "
              f"x {args.games * 2} = {total} games")
        print(f"  Model: {args.model_name}")
    print(f"{'='*70}\n")

    # Dry-run: persona profiling
    if args.dry_run:
        print("Dry-run mode: profiling persona behavior...")
        profiles = {}
        for persona_name in persona_list:
            print(f"  Profiling {persona_name}...")
            profile = profile_persona(persona_name, args.games, args.start_seed,
                                          n=args.n, cfr_model=args.cfr_model)
            profiles[persona_name] = profile

        profile_path = run_dir / "persona_profiles.json"
        write_json(profile_path, {
            "description": "Persona behavioral profiles (dry-run, no LLM)",
            "persona_descriptions": PERSONA_DESCRIPTIONS_TEXT,
            "profiles": profiles,
        })
        print(f"\nProfiles saved to: {profile_path}")
        # Print summary
        for pn, profile in profiles.items():
            all_cards = []
            for g in profile["games"]:
                for a in g["persona_actions"]:
                    all_cards.append(a["card"])
            avg_payout = sum(g["payout"].get(0, 0) for g in profile["games"]) / len(profile["games"])
            print(f"  {pn}: avg_payout={avg_payout:+.1f}, "
                  f"actions={all_cards}")
        return

    # Normal: run conditions
    for persona_name in persona_list:
        print(f"\n{'='*60}")
        print(f"  Opponent: {persona_name}")
        print(f"{'='*60}")
        for ck in condition_keys:
            display_name = CONDITIONS[ck][0]
            print(f"  --- {display_name} vs {persona_name} ---")
            t0 = time.time()
            stats = run_condition(
                model, ck, persona_name,
                args.games, args.start_seed,
                verbose=args.verbose,
                save_prompts=args.save_prompts,
                run_dir=run_dir,
                cfr_model=args.cfr_model,
                n=args.n,
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
            total_games += stats["games"]
            all_results.append(stats)
            print(f"    Result: wr={stats['winrate']:.0f}%  "
                  f"chips={stats['total_chips']:+.0f}  "
                  f"avg={stats['avg_chips']:+.1f}/game  "
                  f"W{stats['wins']}/L{stats['losses']}/D{stats['draws']}  "
                  f"({dt:.0f}s)")

    print_results(all_results)

    # Save summary
    summary = {
        "experiment": "goofspiel_bench",
        "n": args.n,
        "phase": args.phase,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": args.model_name,
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "personas": persona_list,
        "conditions": condition_keys,
        "results": all_results,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"\nResults saved to: {run_dir / 'summary.json'}")
    print(f"Total games: {total_games}")


if __name__ == "__main__":
    main()
