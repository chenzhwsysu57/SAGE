"""Goofspiel strategic reasoning benchmark.

Compares vanilla LLM prompting, LLM+equilibrium guidance, single-call
Tree-of-Thought prompting, and LLM-driven partially observed MCTS planning.
This entry point is independent from goofspiel_bench.py so existing experiments
keep their behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from apis.openai_wrapper import OpenAIModel
from base.interfaces import BaseModel
from games.goofspiel import Goofspiel
from methods.sage.goofspiel_cfr_method import GoofspielCFRMethod
from methods.baseline.strategic_reasoning_goofspiel import SearchConfig, StrategicGoofspielMethod
from modeling.goofspiel import GoofspielArchetypeAgent, PERSONA_NAMES
from rollout import RolloutRunner, save_trajectory
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json


CONDITIONS = ("vanilla", "llm_eq", "llm_tot", "llm_pomcts")


class PromptLegalActionModel(BaseModel):
    """Offline model for schema/dev tests; chooses the first legal bid in prompt."""

    def generate(self, prompt: str) -> str:
        amounts = [int(x) for x in re.findall(r"bid \(amount=(\d+)\)", prompt)]
        amount = amounts[0] if amounts else 1
        return json.dumps({
            "reasoning": "mock-model legal action for development",
            "action": {"type": "bid", "amount": amount},
        })


def make_method(condition: str, name: str, args: argparse.Namespace, seed: int) -> StrategicGoofspielMethod:
    return StrategicGoofspielMethod(
        name=name,
        mode=condition,
        cfr_model_path=args.cfr_model,
        tot_branches=args.tot_branches,
        search_config=SearchConfig(
            width=args.mcts_width,
            depth=args.mcts_depth,
            sims=args.mcts_sims,
            seed=seed,
        ),
    )


def make_opponent(persona_name: str, suffix: str, cfr_model: str | None, seed: int):
    if persona_name == "GTO":
        return GoofspielCFRMethod(model_path=cfr_model, name=f"GTO-{suffix}", seed=seed)
    return GoofspielArchetypeAgent(persona_name, name=f"{persona_name}-{suffix}", seed=seed)


def run_games(
    model: BaseModel,
    method,
    opponent,
    method_position: int,
    *,
    games: int,
    start_seed: int,
    n: int,
    condition: str,
    persona: str,
    suffix: str,
    run_dir: Path,
    save_prompts: bool,
    save_trajectories: bool,
    verbose: bool,
) -> dict[str, Any]:
    results = []
    wins = losses = draws = 0
    total_net = 0.0

    pbar = tqdm(range(games), desc=f"{condition} {persona} {suffix}", unit="game")
    for g in pbar:
        seed = start_seed + g
        game = Goofspiel(n=n, seed=seed)
        runner = RolloutRunner(
            game,
            {method_position: method, 1 - method_position: opponent},
            model,
            verbose=verbose,
        )
        t0 = time.time()
        trajectory = runner.run()
        dt = time.time() - t0

        if save_prompts:
            save_step_texts(
                run_dir,
                persona,
                condition,
                suffix,
                g + 1,
                trajectory,
                method_position,
            )
        if save_trajectories:
            traj_dir = run_dir / "trajectories" / persona / condition / suffix
            traj_dir.mkdir(parents=True, exist_ok=True)
            save_trajectory(trajectory, str(traj_dir / f"game_{g+1:03d}.json"))

        method_chips = trajectory.payout.get(method_position, 0.0)
        opp_chips = trajectory.payout.get(1 - method_position, 0.0)
        net = trajectory.net_profit.get(method_position, method_chips)
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

        results.append({
            "game": g + 1,
            "seed": seed,
            "method_net": net,
            "method_chips": method_chips,
            "opp_chips": opp_chips,
            "method_win": method_win,
            "steps": len(trajectory.steps),
            "time": round(dt, 3),
        })
        pbar.set_postfix({"net": f"{total_net:+.1f}", "wr": f"{wins/(g+1)*100:.0f}%"})

    return {
        "position": method_position,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / games * 100 if games else 0.0,
        "total_chips": total_net,
        "avg_chips": total_net / games if games else 0.0,
        "results": results,
    }


def run_condition(
    model: BaseModel,
    condition: str,
    persona: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    method_p0 = make_method(condition, f"{condition}-P0", args, args.start_seed)
    opp_p1 = make_opponent(persona, "P1", args.cfr_model, args.start_seed + 1000)
    p0 = run_games(
        model,
        method_p0,
        opp_p1,
        0,
        games=args.games,
        start_seed=args.start_seed,
        n=args.n,
        condition=condition,
        persona=persona,
        suffix="P0",
        run_dir=run_dir,
        save_prompts=args.save_prompts,
        save_trajectories=args.save_trajectories,
        verbose=args.verbose,
    )

    opp_p0 = make_opponent(persona, "P0", args.cfr_model, args.start_seed + 2000)
    method_p1 = make_method(condition, f"{condition}-P1", args, args.start_seed + args.games)
    p1 = run_games(
        model,
        method_p1,
        opp_p0,
        1,
        games=args.games,
        start_seed=args.start_seed + args.games,
        n=args.n,
        condition=condition,
        persona=persona,
        suffix="P1",
        run_dir=run_dir,
        save_prompts=args.save_prompts,
        save_trajectories=args.save_trajectories,
        verbose=args.verbose,
    )

    all_results = p0["results"] + p1["results"]
    total_games = len(all_results)
    wins = sum(1 for r in all_results if r["method_win"] is True)
    losses = sum(1 for r in all_results if r["method_win"] is False)
    draws = sum(1 for r in all_results if r["method_win"] is None)
    total_chips = sum(float(r["method_net"]) for r in all_results)
    return {
        "condition": condition,
        "persona": persona,
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / total_games * 100 if total_games else 0.0,
        "total_chips": total_chips,
        "avg_chips": total_chips / total_games if total_games else 0.0,
        "p0_results": p0,
        "p1_results": p1,
    }


def save_step_texts(
    run_dir: Path,
    persona: str,
    condition: str,
    suffix: str,
    game_idx: int,
    trajectory,
    method_position: int,
) -> None:
    out_dir = run_dir / "prompts" / persona / condition / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in trajectory.steps:
        if step.player_id != method_position:
            continue
        if not step.rendered_prompt:
            continue
        prefix = f"g{game_idx:03d}_step{step.step:02d}_P{step.player_id}"
        (out_dir / f"{prefix}_prompt.txt").write_text(step.rendered_prompt)
        if step.raw_llm_response:
            (out_dir / f"{prefix}_response.txt").write_text(step.raw_llm_response)
        save_planning_texts(out_dir, prefix, step.method_data)


def save_planning_texts(out_dir: Path, step_prefix: str, method_data: dict[str, Any] | None) -> None:
    if not method_data:
        return
    pomcts = method_data.get("pomcts")
    if not isinstance(pomcts, dict):
        return
    records = pomcts.get("planning_records", [])
    if not isinstance(records, list):
        return
    planning_dir = out_dir / "planning"
    planning_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        phase = sanitize_slug(str(record.get("phase", "planning")))
        sim = record.get("sim")
        depth = record.get("depth")
        bid = record.get("selected_bid", record.get("root_bid"))
        bid_part = "bidNA" if bid is None else f"bid{bid}"
        if isinstance(sim, int) and isinstance(depth, int):
            stem = f"{step_prefix}_sim{sim:02d}_depth{depth:02d}_{phase}_{bid_part}"
        else:
            stem = f"{step_prefix}_plan{idx:02d}_{phase}_{bid_part}"
        prompt = record.get("prompt")
        response = record.get("response")
        if isinstance(prompt, str) and prompt:
            (planning_dir / f"{stem}_prompt.txt").write_text(prompt)
        if isinstance(response, str) and response:
            (planning_dir / f"{stem}_response.txt").write_text(response)


def sanitize_slug(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return cleaned or "planning"


def summarize_token_usage(run_dir: Path) -> dict[str, Any]:
    prompt_root = run_dir / "prompts"
    by_condition: dict[str, dict[str, int]] = {}
    totals = {"prompt_chars": 0, "response_chars": 0, "prompt_files": 0, "response_files": 0}
    if not prompt_root.exists():
        return {"total": {**totals, "approx_tokens": 0}, "by_condition": {}}

    for path in prompt_root.rglob("*.txt"):
        parts = path.relative_to(prompt_root).parts
        if len(parts) < 3:
            continue
        condition = parts[1]
        row = by_condition.setdefault(condition, {
            "prompt_chars": 0,
            "response_chars": 0,
            "prompt_files": 0,
            "response_files": 0,
        })
        text_len = len(path.read_text(errors="replace"))
        if path.name.endswith("_prompt.txt"):
            row["prompt_chars"] += text_len
            row["prompt_files"] += 1
            totals["prompt_chars"] += text_len
            totals["prompt_files"] += 1
        elif path.name.endswith("_response.txt"):
            row["response_chars"] += text_len
            row["response_files"] += 1
            totals["response_chars"] += text_len
            totals["response_files"] += 1

    for row in by_condition.values():
        row["approx_tokens"] = approx_tokens(row["prompt_chars"] + row["response_chars"])
    total = dict(totals)
    total["approx_tokens"] = approx_tokens(total["prompt_chars"] + total["response_chars"])
    return {"total": total, "by_condition": by_condition}


def approx_tokens(chars: int) -> int:
    return int((chars + 3) // 4)


def redacted_args(args: argparse.Namespace) -> dict[str, Any]:
    data = serializable_args(args)
    if "api_key" in data:
        data["api_key"] = "<redacted>"
    return data


def build_metrics(results: list[dict[str, Any]], token_usage: dict[str, Any]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Any]] = {}
    for row in results:
        cond = row["condition"]
        agg = by_condition.setdefault(cond, {
            "condition": cond,
            "games": 0,
            "total_chips": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "personas": {},
        })
        agg["games"] += row["games"]
        agg["total_chips"] += row["total_chips"]
        agg["wins"] += row["wins"]
        agg["losses"] += row["losses"]
        agg["draws"] += row["draws"]
        agg["personas"][row["persona"]] = {
            "games": row["games"],
            "total_chips": row["total_chips"],
            "avg_chips": row["avg_chips"],
            "winrate": row["winrate"],
        }
    for cond, agg in by_condition.items():
        agg["avg_chips"] = agg["total_chips"] / agg["games"] if agg["games"] else 0.0
        agg["winrate"] = agg["wins"] / agg["games"] * 100 if agg["games"] else 0.0
        usage = token_usage.get("by_condition", {}).get(cond, {})
        agg["prompt_chars"] = usage.get("prompt_chars", 0)
        agg["response_chars"] = usage.get("response_chars", 0)
        agg["approx_tokens"] = usage.get("approx_tokens", 0)
    return {
        "by_condition": by_condition,
        "points": list(by_condition.values()),
        "token_usage": token_usage,
    }


def parse_csv(raw: str, valid: tuple[str, ...] | list[str]) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in values if x not in valid]
    if unknown:
        raise ValueError(f"Unknown values: {unknown}; valid={list(valid)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Goofspiel strategic reasoning analysis")
    parser.add_argument("--conditions", default="vanilla,llm_eq,llm_tot,llm_pomcts")
    parser.add_argument("--personas", default="Greedy,Conservative,Random,GTO")
    parser.add_argument("--games", type=int, default=10, help="Games per position")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--phase", default="dev_debug",
                        choices=["dev_debug", "real_measurement", "analysis_only"])
    parser.add_argument("--cfr-model", default=None)
    parser.add_argument("--output-dir", default="results/goofspiel/strategic_reasoning")
    parser.add_argument("--tot-branches", type=int, default=3)
    parser.add_argument("--mcts-width", type=int, default=1)
    parser.add_argument("--mcts-depth", type=int, default=5)
    parser.add_argument("--mcts-sims", type=int, default=3)
    parser.add_argument("--mcts-rollout-policy", choices=["uniform", "heuristic"], default="uniform",
                        help="Deprecated compatibility option; LLM-POMCTS no longer uses local rollouts.")
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts", default=True)
    parser.add_argument("--save-trajectories", action="store_true", default=False)
    parser.add_argument("--mock-model", action="store_true", help="Use local legal-action mock; no API call")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    conditions = parse_csv(args.conditions, CONDITIONS)
    personas = parse_csv(args.personas, list(PERSONA_NAMES))
    slug = f"{args.games}g_{'_'.join(conditions)}"
    if args.mock_model:
        slug = f"mock_{slug}"
    run_dir = make_run_dir(args.output_dir, slug)
    write_json(run_dir / "config.json", {
        "experiment": "strategic_reasoning_goofspiel",
        "args": redacted_args(args),
        "conditions": conditions,
        "personas": personas,
        "git_commit": git_commit(),
        "phase": args.phase,
        "live_api": not args.mock_model,
    })
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    model: BaseModel
    if args.mock_model:
        model = PromptLegalActionModel()
    else:
        model = OpenAIModel(model=args.model_name, api_key=args.api_key, base_url=args.base_url)

    all_results = []
    for persona in personas:
        print(f"\n=== Opponent: {persona} ===")
        for condition in conditions:
            print(f"--- {condition} ---")
            stats = run_condition(model, condition, persona, args, run_dir)
            all_results.append(stats)
            print(
                f"{condition} vs {persona}: chips={stats['total_chips']:+.1f}, "
                f"avg={stats['avg_chips']:+.2f}, wr={stats['winrate']:.1f}%"
            )

    token_usage = summarize_token_usage(run_dir)
    metrics = build_metrics(all_results, token_usage)
    write_json(run_dir / "metrics" / "strategic_reasoning_metrics.json", metrics)
    summary = {
        "experiment": "strategic_reasoning_goofspiel",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n": args.n,
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "conditions": conditions,
        "personas": personas,
        "results": all_results,
        "metrics_path": str(run_dir / "metrics" / "strategic_reasoning_metrics.json"),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"\nSummary saved to: {run_dir / 'summary.json'}")
    print(f"Metrics saved to: {run_dir / 'metrics' / 'strategic_reasoning_metrics.json'}")


if __name__ == "__main__":
    main()
