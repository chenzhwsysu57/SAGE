import argparse
import numpy as np

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import exploitability


def load_goofspiel(n: int):
    # fixed descending public point cards.
    # imp_info=True: imperfect-information Goofspiel.
    # turn_based_simultaneous_game: converts simultaneous bidding into sequential imperfect-information form.
    game_str = (
        "turn_based_simultaneous_game("
        f"game=goofspiel("
        f"imp_info=True,"
        f"num_cards={n},"
        f"points_order=descending,"
        f"returns_type=point_difference"
        f")"
        ")"
    )
    return pyspiel.load_game(game_str), game_str


def format_policy_at_state(game, policy, state, player_id=None):
    if player_id is None:
        player_id = state.current_player()

    probs = policy.action_probabilities(state, player_id)
    out = {}
    for action, prob in sorted(probs.items()):
        try:
            action_name = state.action_to_string(player_id, action)
        except Exception:
            action_name = str(action)
        out[action_name] = f"{prob:.3f}"
    return out


def root_policies(game, avg_policy):
    # P0 root policy: initial state.
    s0 = game.new_initial_state()
    p0_dist = format_policy_at_state(game, avg_policy, s0, player_id=0)

    # P1 root-equivalent policy:
    # In the turn-based simultaneous transform, P0 acts first in the tree,
    # then P1 acts, but P1's information state should not reveal P0's pending bid.
    # We choose any legal P0 action to move to P1's first decision.
    s1 = game.new_initial_state()
    first_action = s1.legal_actions()[0]
    s1.apply_action(first_action)
    p1_dist = format_policy_at_state(game, avg_policy, s1, player_id=1)

    return p0_dist, p1_dist


def evaluate(game, avg_policy, use_cpp_br=True):
    # exploitability = NashConv / 2 for 2p zero-sum / constant-sum games.
    # nash_conv(..., return_only_nash_conv=False) also returns per-player improvements.
    nc = exploitability.nash_conv(
        game,
        avg_policy,
        return_only_nash_conv=False,
        use_cpp_br=use_cpp_br,
    )
    expl = nc.nash_conv / game.num_players()
    return expl, nc.player_improvements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--iters", type=int, default=500_000)
    parser.add_argument("--eval_every", type=int, default=10_000)
    parser.add_argument(
        "--average_type",
        choices=["simple", "full"],
        default="simple",
        help=(
            "OpenSpiel ExternalSamplingSolver average type. "
            "simple is faster/default; full does a full average-policy update pass."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python_br", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    game, game_str = load_goofspiel(args.n)
    print("Game:", game_str)
    print("Game type:", game.get_type())
    print("Num players:", game.num_players())
    print()

    if args.average_type == "simple":
        average_type = external_sampling_mccfr.AverageType.SIMPLE
    else:
        average_type = external_sampling_mccfr.AverageType.FULL

    solver = external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=average_type,
    )

    for it in range(1, args.iters + 1):
        solver.iteration()

        if it % args.eval_every == 0:
            avg_policy = solver.average_policy()

            expl, improvements = evaluate(
                game,
                avg_policy,
                use_cpp_br=not args.python_br,
            )

            p0_root, p1_root = root_policies(game, avg_policy)

            br0 = improvements[0]
            br1 = improvements[1]
            ratio = br1 / max(br0, 1e-12)

            print(
                f"{it:8d}: "
                f"BR_0={br0:.6f} "
                f"BR_1={br1:.6f} "
                f"expl={expl:.6f} "
                f"ratio={ratio:.2f} | "
                f"P0={p0_root} "
                f"P1={p1_root}"
            )


if __name__ == "__main__":
    main()