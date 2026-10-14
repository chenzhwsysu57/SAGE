"""Decrypto persona comparison experiment.

Usage:
    python runs/decrypto/decrypto_bench.py --games 10 --seed 100
    python runs/decrypto/decrypto_bench.py --persona theme_loyalist --games 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from games.decrypto import Decrypto
from apis.openai_wrapper import OpenAIModel
from methods.baseline.decrypto import DecryptoMethod
from modeling.decrypto import LockedThemeState
from rollout import RolloutRunner
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
from runs.decrypto.metrics import aggregate_metrics, summarize_trajectory

_PROJECT = Path(__file__).resolve().parent.parent.parent
WORD_PATH = _PROJECT / "games" / "decrypto_words.txt"
PERSONAS = [
    "default",
    "anti_convergence",
    "tom",
    "literal",
    "theme_loyalist",
    "adaptive",
]


def main():
    parser = argparse.ArgumentParser(description="Decrypto persona comparison")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=12,
                        help="Turns to survive for AB win")
    parser.add_argument("--interceptions", type=int, default=3,
                        help="Interceptions needed for Eve to win")
    parser.add_argument("--miscommunications", type=int, default=3,
                        help="Miscommunications needed for Eve to win")
    parser.add_argument("--persona", default="default",
                        choices=PERSONAS,
                        help="Default persona for all roles")
    parser.add_argument("--alice-persona", default=None,
                        choices=PERSONAS,
                        help="Override persona for Alice (Encoder)")
    parser.add_argument("--bob-persona", default=None,
                        choices=PERSONAS,
                        help="Override persona for Bob (Decoder)")
    parser.add_argument("--eve-persona", default=None,
                        choices=PERSONAS,
                        help="Override persona for Eve (Interceptor)")
    parser.add_argument("--no-save-prompts", action="store_true",
                        help="Skip saving prompts (saves by default)")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-dir", default="results/decrypto/decrypto_bench")
    parser.add_argument("--keyword-manifest", default=None,
                        help="Path to keyword_manifest.json (enables fixed keyword sets)")
    parser.add_argument("--keyword-tier", default="all_abstract",
                        choices=["all_abstract", "3_abstract", "2_abstract",
                                 "all_wide", "wide_3", "wide_2"],
                        help="Which difficulty tier to use from manifest")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-game details")
    args = parser.parse_args()

    alice_persona = args.alice_persona or args.persona
    bob_persona = args.bob_persona or args.persona
    eve_persona = args.eve_persona or args.persona
    condition_name = f"alice_{alice_persona}__bob_{bob_persona}__eve_{eve_persona}"

    slug = f"{condition_name}_{args.games}g"
    run_dir = make_run_dir(args.output_dir, slug)
    config = {
        "args": serializable_args(args),
        "condition_name": condition_name,
        "role_personas": {
            "alice": alice_persona,
            "bob": bob_persona,
            "eve": eve_persona,
        },
        "git_commit": git_commit(),
    }
    write_json(run_dir / "config.json", config)
    print(f"Run dir: {run_dir}")

    model = OpenAIModel(
        model=args.model, temperature=args.temperature,
        api_key=args.api_key or None, base_url=args.base_url or None,
    )

    print(f"\n{'='*60}")
    print(f"  Decrypto — {condition_name}")
    print(f"  {args.games} games, seed {args.seed}")
    print(f"{'='*60}\n")

    seeds = list(range(args.seed, args.seed + args.games))
    results = []
    game_metrics = []
    role_names = {0: "alice", 1: "bob", 2: "eve"}

    # Load keyword manifest if provided
    kw_manifest = None
    if args.keyword_manifest:
        with open(args.keyword_manifest) as f:
            kw_manifest = json.load(f)[args.keyword_tier]
        if args.games > len(kw_manifest):
            print(f"WARNING: {args.games} games requested but only {len(kw_manifest)} "
                  f"keyword sets in tier '{args.keyword_tier}'. Using {len(kw_manifest)}.")
            args.games = len(kw_manifest)
            seeds = seeds[:args.games]

    pbar = tqdm(range(args.games), desc=f"  {condition_name}", unit="game")

    for gi in pbar:
        seed = seeds[gi]

        eve_locked = LockedThemeState() if eve_persona == "theme_loyalist" else None
        methods = {
            0: DecryptoMethod(
                name="Alice",
                persona=alice_persona,
            ),
            1: DecryptoMethod(name="Bob", persona=bob_persona),
            2: DecryptoMethod(name="Eve", persona=eve_persona,
                              locked_theme_state=eve_locked),
        }

        kw_args = {}
        if kw_manifest:
            kw_args["keywords"] = tuple(kw_manifest[gi])

        game = Decrypto(seed=seed, max_turns=args.max_turns,
                        interception_threshold=args.interceptions,
                        miscommunication_threshold=args.miscommunications,
                        **kw_args)
        runner = RolloutRunner(game, methods, model, verbose=args.verbose)
        trajectory = runner.run()
        metrics = summarize_trajectory(
            trajectory,
            max_turns=args.max_turns,
            interception_threshold=args.interceptions,
            miscommunication_threshold=args.miscommunications,
        )
        game_metrics.append(metrics)

        # Score
        payout = trajectory.payout
        eve_won = payout.get(2, 0) > 0
        result = {
            "game": gi + 1, "seed": seed,
            "steps": len(trajectory.steps),
            "eve_won": eve_won,
            "payout": {str(k): v for k, v in payout.items()},
            "winner": trajectory.winner,
            "metrics": {
                k: v for k, v in metrics.items()
                if k != "per_round"
            },
        }
        results.append(result)

        # Save prompts (by default)
        if not args.no_save_prompts:
            prompts_dir = run_dir / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            cond_dir = prompts_dir / condition_name
            cond_dir.mkdir(exist_ok=True)
            for step in trajectory.steps:
                pid = step.player_id
                name = f"g{gi+1:02d}_s{step.step:02d}_P{pid}_{role_names[pid]}"
                (cond_dir / f"{name}_prompt.txt").write_text(step.rendered_prompt)
                (cond_dir / f"{name}_response.txt").write_text(
                    step.raw_llm_response or "")
                if (
                    pid == 2
                    and isinstance(step.method_data, dict)
                    and step.method_data.get("literal_policy")
                ):
                    for literal_step in step.method_data["literal_policy"]:
                        word_idx = literal_step.get("word_index", 0)
                        if word_idx <= 0 or literal_step.get("bootstrap"):
                            continue
                        (cond_dir / f"{name}_word{word_idx}_prompt.txt").write_text(
                            literal_step.get("prompt", "")
                        )
                        (cond_dir / f"{name}_word{word_idx}_response.txt").write_text(
                            literal_step.get("response", "")
                        )
                    (cond_dir / f"{name}_final_response.txt").write_text(
                        step.method_data.get("literal_final_response", step.raw_llm_response or "")
                    )

        # Save trajectory
        traj_dir = run_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        traj_path = traj_dir / f"g{gi+1:02d}_seed{seed}.json"
        traj_path.write_text(
            json.dumps(trajectory.model_dump(), indent=2,
                       default=str, ensure_ascii=False))

        pbar.set_postfix({"result": "EVE" if eve_won else "A+B"})

    eve_wins = sum(1 for r in results if r["eve_won"])
    avg_steps = sum(r["steps"] for r in results) / len(results) if results else 0
    aggregate = aggregate_metrics(game_metrics, max_turns=args.max_turns)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Condition:   {condition_name}")
    print(f"  Eve wins:    {eve_wins}/{args.games} ({eve_wins/args.games:.0%})")
    print(f"  A+B wins:    {args.games - eve_wins}/{args.games}")
    print(f"  Avg steps:   {avg_steps:.1f}")
    print(f"  Eve int rate:{aggregate['eve_int_rate']:.3f}")
    print(f"  Bob mis rate:{aggregate['bob_mis_rate']:.3f}")
    print(f"\nSaved to: {run_dir}")

    write_json(run_dir / "metrics" / "decrypto_metrics.json", {
        "condition_name": condition_name,
        "role_personas": config["role_personas"],
        "aggregate": aggregate,
        "games": game_metrics,
    })
    write_json(run_dir / "summary.json", {
        "experiment": "decrypto_persona",
        "condition_name": condition_name,
        "persona": args.persona,
        "role_personas": config["role_personas"],
        "model": args.model,
        "games": args.games,
        "start_seed": args.seed,
        "eve_wins": eve_wins,
        "alice_bob_wins": args.games - eve_wins,
        "avg_steps": avg_steps,
        "metrics": aggregate,
        "results": results,
    })


if __name__ == "__main__":
    main()
