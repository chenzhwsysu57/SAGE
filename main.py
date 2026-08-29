"""CLI entrypoint for running ToM Research experiments.

Usage:
    python main.py leduc --p1 Alice --p2 Bob --seed 42
    python main.py leduc --p1 human --p2 Bob --seed 42
    python main.py decrypto --encoder Alice --decoder Bob --eve Eve --seed 42
    python main.py werewolf --roles werewolf:2,villager:3,seer:1,witch:1,hunter:1 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from games import LeducHoldem
from methods import (SimplePromptMethod, HumanMethod, ReActMethod,
                        CoTMethod, ReflexionMethod, ToMMethod, CFRMethod,
                        DecryptoMethod)
from apis import OpenAIModel
from rollout import RolloutRunner, save_run_log
from base.types import RunLog, RunSummary, GameResult

METHOD_REGISTRY = {
    "simple": SimplePromptMethod,
    "react": ReActMethod,
    "cot": CoTMethod,
    "reflexion": ReflexionMethod,
    "tom": ToMMethod,
    "cfr": CFRMethod,
    "human": HumanMethod,
    "decrypto": DecryptoMethod,
}


def _make_method(name: str, method_type: str, lang: str = "en"):
    if name.lower() == "human" or method_type == "human":
        return HumanMethod(name="Human")
    cls = METHOD_REGISTRY.get(method_type, SimplePromptMethod)
    if method_type in ("cfr",):
        return cls(name=name)
    return cls(name=name, lang=lang)


def _build_model(args: argparse.Namespace):
    try:
        return OpenAIModel(
            model=args.model,
            temperature=args.temperature,
            base_url=args.base_url or None,
            api_key=args.api_key or None,
        )
    except ValueError:
        # No API key — return dummy model for non-LLM methods
        class _DummyModel:
            def generate(self, prompt):
                return ""
        return _DummyModel()


def run_game(game_cls, methods: dict, model, args,
             game_kwargs: dict | None = None) -> None:
    """Run one or more games. If args.games > 1, loop and report stats."""
    num_games = getattr(args, "games", 1)
    game_kwargs = game_kwargs or {}
    stats = {p: {"wins": 0, "total_reward": 0.0} for p in methods}
    total_steps = 0
    trajectories: list[tuple] = []  # (seed, trajectory)

    for g in range(num_games):
        game_seed = (args.seed + g) if args.seed is not None else None
        game = game_cls(seed=game_seed, **game_kwargs)
        runner = RolloutRunner(game=game, methods=methods, model=model)
        trajectory = runner.run()
        trajectories.append((game_seed, trajectory))
        total_steps += len(trajectory.steps)

        winner = trajectory.winner
        for p, r in trajectory.payout.items():
            stats[p]["total_reward"] += r
        if winner is not None:
            stats[winner]["wins"] += 1
        else:
            # Tie: award half a win to each player
            rewards = list(trajectory.payout.values())
            if rewards and all(r == rewards[0] for r in rewards):
                for p in trajectory.payout:
                    stats[p]["wins"] += 0.5

        # Call reflect on methods that support it
        for m in methods.values():
            if hasattr(m, "reflect") and callable(m.reflect):
                m.reflect(trajectory, model, game.__class__.__name__)

        if num_games == 1:
            print(f"Game over after {len(trajectory.steps)} steps")
            for p, r in trajectory.payout.items():
                print(f"  {methods[p].name} (P{p}): {r}")
            if trajectory.winner is not None:
                print(f"  Winner: {methods[trajectory.winner].name}")
        else:
            if g == num_games - 1:
                print(f"\n{'='*50}")
                print(f"  {num_games} games completed")
                print(f"  Avg steps/game: {total_steps/num_games:.1f}")
                for p in sorted(stats):
                    s = stats[p]
                    wr = s["wins"] / num_games * 100
                    avg_r = s["total_reward"] / num_games
                    print(f"  {methods[p].name} (P{p}): "
                          f"winrate={wr:.0f}% avg_reward={avg_r:.1f}")
                print(f"{'='*50}")

        if args.verbose and num_games == 1:
            print("\n--- Trajectory ---")
            for step in trajectory.steps:
                print(
                    f"Step {step.step}: P{step.player_id} "
                    f"→ {step.parsed_action} "
                    f"(valid={step.action_valid})"
                )
                if args.show_prompts:
                    print(f"  Prompt:\n{step.rendered_prompt}")
                    print(f"  Response: {step.raw_llm_response}")

    # Save consolidated run log
    if args.output_dir and trajectories:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        first = trajectories[0][1]
        game_name = first.game_name.lower()
        path = os.path.join(args.output_dir, f"{game_name}_{ts}.json")

        game_results = []
        summary_payout: dict[int, float] = {}
        summary_net: dict[int, float] = {}
        for i, (gs, traj) in enumerate(trajectories, 1):
            game_results.append(GameResult(
                game_index=i,
                seed=gs,
                static_state=traj.static_state,
                steps=traj.steps,
                payout=traj.payout,
                winner=traj.winner,
            ))
            for p, r in traj.payout.items():
                p_int = int(p)
                summary_payout[p_int] = summary_payout.get(p_int, 0.0) + r
                # Net = payout - total invested (from final step's state_after)
                if traj.steps:
                    invested = traj.steps[-1].state_after.get("total_bets", {}).get(p, 0)
                else:
                    invested = 0
                summary_net[p_int] = summary_net.get(p_int, 0.0) + (r - invested)

        summary = RunSummary(
            total_payout=summary_payout,
            net_chips=summary_net,
            wins={p: stats[p]["wins"] for p in stats},
            winrate={p: stats[p]["wins"] / num_games for p in stats},
        )

        run_log = RunLog(
            game_name=first.game_name,
            players=first.players,
            player_methods=first.player_methods,
            rules_text=first.rules_text,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            num_games=num_games,
            timestamp=ts,
            games=game_results,
            summary=summary,
        )
        save_run_log(run_log, path)
        print(f"Trajectory saved to: {path}")


def run_leduc(args: argparse.Namespace) -> None:
    model = _build_model(args)
    p1 = _make_method(args.p1, args.p1_method)
    p2 = _make_method(args.p2, args.p2_method)
    methods = {0: p1, 1: p2}

    print(f"=== Leduc Hold'em (seed={args.seed}) ===")
    print(f"Model: {args.model}, Methods: {args.p1_method}/{args.p2_method}")
    print(f"Players: {p1.name} (P0) vs {p2.name} (P1)")
    print()

    run_game(LeducHoldem, methods, model, args)


def run_leduc_cn(args: argparse.Namespace) -> None:
    from games.leduc_cn import LeducHoldemCN

    model = _build_model(args)
    p1 = _make_method(args.p1, args.p1_method, lang="cn")
    p2 = _make_method(args.p2, args.p2_method, lang="cn")
    methods = {0: p1, 1: p2}

    print(f"=== Leduc Hold'em 中文版 (seed={args.seed}) ===")
    print(f"模型: {args.model}, 策略: {args.p1_method}/{args.p2_method}")
    print(f"玩家: {p1.name} (P0) vs {p2.name} (P1)")
    print()

    run_game(LeducHoldemCN, methods, model, args)


def run_decrypto(args: argparse.Namespace) -> None:
    from games.decrypto import Decrypto

    model = _build_model(args)
    encoder = _make_method(args.encoder, args.encoder_method)
    decoder = _make_method(args.decoder, args.decoder_method)
    eve = _make_method(args.eve, args.eve_method)
    methods = {0: encoder, 1: decoder, 2: eve}

    print(f"=== Decrypto (seed={args.seed}, max_turns={args.max_turns}) ===")
    print(f"Model: {args.model}, Methods: {args.encoder_method}/{args.decoder_method}/{args.eve_method}")
    print(f"Encoder: {encoder.name}, Decoder: {decoder.name}, Eve: {eve.name}")
    print()

    run_game(Decrypto, methods, model, args, game_kwargs={"max_turns": args.max_turns})


def run_werewolf(args: argparse.Namespace) -> None:
    from games.werewolf import Werewolf

    role_config: dict[str, int] = {}
    for part in args.roles.split(","):
        name, count = part.strip().split(":")
        role_config[name.strip()] = int(count.strip())

    # Use a fresh game to peek at role assignments for display
    # The runner will use its own fresh game with the same seed
    display_game = Werewolf(seed=args.seed, roles=role_config)
    display_state = display_game.reset()
    num_players = display_game.num_players()

    if args.role_methods:
        # Parse into per-role queues: first werewolf gets first method, etc.
        role_method_queues: dict[str, list[str]] = {}
        for part in args.role_methods.split(","):
            role, m = part.strip().split(":")
            role_method_queues.setdefault(role.strip(), []).append(m.strip())
        # Track how many players of each role we've assigned
        role_counters: dict[str, int] = {}
        methods_list = []
        for i in range(num_players):
            role = display_state.roles[i]
            idx = role_counters.get(role, 0)
            role_counters[role] = idx + 1
            queue = role_method_queues.get(role, ["simple"])
            methods_list.append(queue[idx] if idx < len(queue) else queue[-1])
    else:
        methods_list = list(args.methods) if args.methods else ["simple"]
        while len(methods_list) < num_players:
            methods_list.append(methods_list[-1])
        methods_list = methods_list[:num_players]

    names = (args.names if args.names
             else [f"P{i}" for i in range(num_players)])
    names = names[:num_players]

    model = _build_model(args)

    methods = {
        i: _make_method(names[i], methods_list[i])
        for i in range(num_players)
    }

    print(f"=== Werewolf (seed={args.seed}) ===")
    print(f"Model: {args.model}")
    print(f"Roles: {role_config}")
    print(f"Players:")
    for i in range(num_players):
        print(f"  P{i} ({display_state.roles[i]}): {methods[i].name} [{methods[i].method_type}]")
    print()

    run_game(Werewolf, methods, model, args,
             game_kwargs={"roles": role_config})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ToM Research - Multi-agent LLM Game Framework"
    )
    subparsers = parser.add_subparsers(dest="game", help="Game to run")

    def add_common_args(p):
        p.add_argument("--seed", type=int, default=None, help="Random seed")
        p.add_argument("--games", type=int, default=1, help="Number of games to play")
        p.add_argument("--model", default="deepseek-v4-flash", help="Model name")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--api-key", default=None, help="API key")
        p.add_argument("--base-url", default="https://api.deepseek.com", help="API base URL")
        p.add_argument("--output-dir", default="results/cli", help="Trajectory output dir")
        p.add_argument("--verbose", "-v", action="store_true")
        p.add_argument("--show-prompts", action="store_true")

    leduc_parser = subparsers.add_parser("leduc", help="Run Leduc Hold'em")
    leduc_parser.add_argument("--p1", default="P1", help="P1 name, or 'human'")
    leduc_parser.add_argument("--p2", default="P2", help="P2 name, or 'human'")
    leduc_parser.add_argument("--p1-method", default="simple", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human"])
    leduc_parser.add_argument("--p2-method", default="simple", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human"])
    add_common_args(leduc_parser)

    decrypto_parser = subparsers.add_parser("decrypto", help="Run Decrypto")
    decrypto_parser.add_argument("--encoder", default="Encoder", help="Encoder agent")
    decrypto_parser.add_argument("--decoder", default="Decoder", help="Decoder agent")
    decrypto_parser.add_argument("--eve", default="Eve", help="Interceptor agent")
    decrypto_parser.add_argument("--encoder-method", default="decrypto", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human", "decrypto"])
    decrypto_parser.add_argument("--decoder-method", default="decrypto", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human", "decrypto"])
    decrypto_parser.add_argument("--eve-method", default="decrypto", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human", "decrypto"])
    decrypto_parser.add_argument("--max-turns", type=int, default=8)
    add_common_args(decrypto_parser)

    # Leduc Chinese subcommand
    leduc_cn_parser = subparsers.add_parser("leduc-cn", help="Leduc Hold'em (Chinese)")
    leduc_cn_parser.add_argument("--p1", default="P1", help="P1 name or 'human'")
    leduc_cn_parser.add_argument("--p2", default="P2", help="P2 name or 'human'")
    leduc_cn_parser.add_argument("--p1-method", default="simple", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human"])
    leduc_cn_parser.add_argument("--p2-method", default="simple", choices=["simple", "react", "cot", "reflexion", "tom", "cfr", "human"])
    add_common_args(leduc_cn_parser)

    werewolf_parser = subparsers.add_parser("werewolf", help="Run Werewolf (狼人杀)")
    werewolf_parser.add_argument(
        "--roles", default="werewolf:2,villager:2,seer:1",
        help="Role config, e.g. 'werewolf:2,villager:2,seer:1'")
    werewolf_parser.add_argument(
        "--names", nargs="*", default=None,
        help="Player names in order (one per role slot)")
    werewolf_parser.add_argument(
        "--methods", nargs="*", default=["simple"],
        help="Method types per player position")
    werewolf_parser.add_argument(
        "--role-methods", default=None,
        help="Method types per role, e.g. 'werewolf:simple,villager:react,seer:react'")
    add_common_args(werewolf_parser)

    args = parser.parse_args()

    game_runners = {
        "leduc": run_leduc,
        "leduc-cn": run_leduc_cn,
        "decrypto": run_decrypto,
        "werewolf": run_werewolf,
    }

    if args.game in game_runners:
        game_runners[args.game](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
