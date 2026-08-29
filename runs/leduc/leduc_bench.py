"""Leduc Hold'em benchmark: multiple agents vs multiple methods across seeds.

Usage:
    python runs/leduc/leduc_bench.py --games 20 --start-seed 42
    python runs/leduc/leduc_bench.py --agent risk --opponents simple,cot,react --games 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from games.leduc import LeducHoldem
from apis.openai_wrapper import OpenAIModel
from methods.simple_prompt import SimplePromptMethod
from methods.cot import CoTMethod
from methods.react import ReActMethod
from methods.reflexion import ReflexionMethod
from methods.tom import ToMMethod
from rollout import RolloutRunner
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json

OPPONENTS = {
    "simple": ("Simple", SimplePromptMethod),
    "cot": ("CoT", CoTMethod),
    "react": ("ReAct", ReActMethod),
    "reflexion": ("Reflexion", ReflexionMethod),
    "tom": ("ToM", ToMMethod),
}

AGENTS = {
    "risk": (
        "RiskSeeking",
        "You are a risk-seeking player. You prefer aggressive plays: "
        "betting and raising over checking and calling. You are willing "
        "to bluff and take risks to win big pots.\n\n",
    ),
    "safe": (
        "SafePlay",
        "You are a conservative player. You prefer safe plays: "
        "check and call rather than bet and raise. You avoid big losses. "
        "Only bet when you have a strong hand. Protect your chips.\n\n",
    ),
}


def make_agent(agent_type: str) -> SimplePromptMethod:
    """Create an agent with a specific play-style bias."""
    name, prompt_prefix = AGENTS[agent_type]

    class BiasedMethod(SimplePromptMethod):
        def __init__(self):
            super().__init__(name=name)
        def _render_prompt(self, obs, legal_actions):
            raw = super()._render_prompt(obs, legal_actions)
            return prompt_prefix + raw

    return BiasedMethod()


def run_matchup(
    model: OpenAIModel, agent_name: str, agent_method,
    opp_name: str, opp_cls, games: int, start_seed: int,
    output_dir: str,
) -> dict:
    """Run N games between agent and opponent, return stats."""
    results = []
    wins = 0
    losses = 0
    draws = 0
    total_agent_reward = 0.0

    for g in range(games):
        seed = start_seed + g
        game = LeducHoldem(seed=seed)
        agent = agent_method()
        opp = opp_cls(name=opp_name)
        methods = {0: agent, 1: opp}
        runner = RolloutRunner(game, methods, model)

        t0 = time.time()
        trajectory = runner.run()
        dt = time.time() - t0

        agent_reward = trajectory.payout[0]
        total_agent_reward += agent_reward
        if agent_reward > 0:
            wins += 1
        elif trajectory.payout[1] > 0:
            losses += 1
        else:
            draws += 1

        results.append({
            "game": g + 1,
            "seed": seed,
            "winner": "agent" if agent_reward > 0 else (
                "opponent" if trajectory.payout[1] > 0 else "draw"),
            "agent_reward": agent_reward,
            "opponent_reward": trajectory.payout[1],
            "steps": len(trajectory.steps),
            "time": round(dt, 1),
        })

        # Save trajectory
        if output_dir:
            ts = datetime.now().strftime("%H%M%S")
            path = os.path.join(
                output_dir, f"leduc_{agent_name}_vs_{opp_name}_g{g+1:02d}_{ts}.json")
            from rollout import save_trajectory
            save_trajectory(trajectory, path)

        wr = wins / (g + 1) * 100
        print(f"  Game {g+1:2d} (s={seed}): "
              f"{agent_name}={trajectory.payout[0]:4.0f} "
              f"{opp_name}={trajectory.payout[1]:4.0f} "
              f"wr={wr:.0f}% ({dt:.0f}s)", flush=True)

    n = games
    return {
        "agent": agent_name,
        "opponent": opp_name,
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / n * 100,
        "avg_agent_reward": total_agent_reward / n,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Leduc Hold'em benchmark")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--agents", default="risk,safe",
                       help="Agent types: risk,safe")
    parser.add_argument("--opponents", default="simple,cot,react,reflexion,tom",
                       help="Opponent methods")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-dir", default="results/leduc/leduc_bench")
    args = parser.parse_args()
    run_dir = make_run_dir(args.output_dir, f"{args.agents}_vs_{args.opponents}")
    write_json(run_dir / "config.json", {
        "args": serializable_args(args),
        "git_commit": git_commit(),
    })

    model = OpenAIModel(model=args.model, api_key=args.api_key,
                        base_url=args.base_url)

    agent_types = [a.strip() for a in args.agents.split(",")]
    opp_ids = [o.strip() for o in args.opponents.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment": "leduc_bench",
        "timestamp": timestamp,
        "model": args.model,
        "games_per_matchup": args.games,
        "start_seed": args.start_seed,
        "matchups": [],
    }

    print(f"Leduc Benchmark: {len(agent_types)} agents x {len(opp_ids)} opponents")
    print(f"Games per matchup: {args.games}, Start seed: {args.start_seed}")
    print(f"Model: {args.model}")
    print()

    for agent_type in agent_types:
        agent_name = AGENTS[agent_type][0]
        for opp_id in opp_ids:
            opp_name, opp_cls = OPPONENTS[opp_id]
            print(f"{'='*50}")
            print(f"  {agent_name} vs {opp_name}")
            print(f"{'='*50}")
            stats = run_matchup(
                model, agent_name, lambda: make_agent(agent_type),
                opp_name, opp_cls, args.games, args.start_seed, None,
            )
            summary["matchups"].append(stats)
            print(f"  Result: winrate={stats['winrate']:.0f}%, "
                  f"avg_reward={stats['avg_agent_reward']:.1f}")
            print()

    # Save summary
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"Summary saved to: {summary_path}")

    # Print final table
    print()
    print("=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for m in summary["matchups"]:
        print(f"  {m['agent']:>12s} vs {m['opponent']:<10s}: "
              f"wr={m['winrate']:5.1f}%  "
              f"avg_reward={m['avg_agent_reward']:6.1f}")


if __name__ == "__main__":
    main()
