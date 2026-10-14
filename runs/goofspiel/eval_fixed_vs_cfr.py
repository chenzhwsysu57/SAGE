"""Evaluate a fixed card sequence vs trained CFR strategy.

Usage:
  python runs/goofspiel/eval_fixed_vs_cfr.py --n 6 --seq 1 6 5 4 3 2 --games 10000
  python runs/goofspiel/eval_fixed_vs_cfr.py --n 6 --load model.pkl --seq 1 6 5 4 3 2
"""

import argparse
import math
import os
import pickle
import random
import sys
import time


# ── helpers (copied from methods/sage/goofspiel_cfr.py, no numpy dependency) ──

def hand_to_mask(hand: set[int]) -> int:
    m = 0
    for c in hand:
        m |= 1 << c
    return m


def mask_to_list(mask: int, n: int) -> list[int]:
    return [c for c in range(1, n + 1) if mask & (1 << c)]


# ── game simulation ────────────────────────────────────────────────────

InfosetKey = tuple  # (player, hand_mask, outcome_history, action_history)


def play_game_fixed_as_p0(fixed_seq: tuple[int, ...],
                          cfr_snapshot: dict,
                          n: int,
                          rng: random.Random) -> float:
    """Fixed strategy as P0 vs CFR as P1. Returns point difference (P0 - P1)."""
    p0_hand = set(range(1, n + 1))
    p1_hand = set(range(1, n + 1))
    p0_actions: tuple[int, ...] = ()
    p1_actions: tuple[int, ...] = ()
    outcome_history: tuple[int, ...] = ()
    scores = [0.0, 0.0]

    for round_idx in range(n):
        prize = n - round_idx

        # P0: fixed card
        p0_card = fixed_seq[round_idx]
        p0_hand.remove(p0_card)
        p0_actions = p0_actions + (p0_card,)

        # P1: sample from CFR snapshot
        p1_hand_mask = hand_to_mask(p1_hand)
        ik1 = (1, p1_hand_mask, outcome_history, p1_actions)
        strat = cfr_snapshot.get(ik1, None)
        cards = sorted(p1_hand)
        if strat is None:
            probs = [1.0 / len(cards)] * len(cards)
        else:
            probs = [strat.get(c, 0.0) for c in cards]
            total = sum(probs)
            if total < 1e-12:
                probs = [1.0 / len(cards)] * len(cards)
            else:
                probs = [p / total for p in probs]

        p1_card = rng.choices(cards, weights=probs)[0]
        p1_hand.remove(p1_card)
        p1_actions = p1_actions + (p1_card,)

        if p0_card > p1_card:
            scores[0] += prize
            outcome = 1
        elif p0_card < p1_card:
            scores[1] += prize
            outcome = -1
        else:
            scores[0] += prize / 2.0
            scores[1] += prize / 2.0
            outcome = 0
        outcome_history = outcome_history + (outcome,)

    return scores[0] - scores[1]


def play_game_cfr_as_p0(cfr_snapshot: dict,
                         fixed_seq: tuple[int, ...],
                         n: int,
                         rng: random.Random) -> float:
    """CFR as P0 vs fixed strategy as P1. Returns point difference (P0 - P1)."""
    p0_hand = set(range(1, n + 1))
    p1_hand = set(range(1, n + 1))
    p0_actions: tuple[int, ...] = ()
    p1_actions: tuple[int, ...] = ()
    outcome_history: tuple[int, ...] = ()
    scores = [0.0, 0.0]

    for round_idx in range(n):
        prize = n - round_idx

        # P0: sample from CFR snapshot
        p0_hand_mask = hand_to_mask(p0_hand)
        ik0 = (0, p0_hand_mask, outcome_history, p0_actions)
        strat = cfr_snapshot.get(ik0, None)
        cards = sorted(p0_hand)
        if strat is None:
            probs = [1.0 / len(cards)] * len(cards)
        else:
            probs = [strat.get(c, 0.0) for c in cards]
            total = sum(probs)
            if total < 1e-12:
                probs = [1.0 / len(cards)] * len(cards)
            else:
                probs = [p / total for p in probs]

        p0_card = rng.choices(cards, weights=probs)[0]
        p0_hand.remove(p0_card)
        p0_actions = p0_actions + (p0_card,)

        # P1: fixed card
        p1_card = fixed_seq[round_idx]
        p1_hand.remove(p1_card)
        p1_actions = p1_actions + (p1_card,)

        if p0_card > p1_card:
            scores[0] += prize
            outcome = 1
        elif p0_card < p1_card:
            scores[1] += prize
            outcome = -1
        else:
            scores[0] += prize / 2.0
            scores[1] += prize / 2.0
            outcome = 0
        outcome_history = outcome_history + (outcome,)

    return scores[0] - scores[1]


# ── evaluation ────────────────────────────────────────────────────────

def evaluate(fixed_seq: tuple[int, ...],
             cfr_snapshot: dict,
             n: int,
             num_games: int,
             seed: int) -> tuple:
    rng = random.Random(seed)
    results_p0 = [play_game_fixed_as_p0(fixed_seq, cfr_snapshot, n, rng)
                  for _ in range(num_games)]
    results_p1 = [play_game_cfr_as_p0(cfr_snapshot, fixed_seq, n, rng)
                  for _ in range(num_games)]

    def stats(data, label):
        n_g = len(data)
        mean = sum(data) / n_g
        wins = sum(1 for d in data if d > 0)
        losses = sum(1 for d in data if d < 0)
        win_rate = wins / n_g
        var = sum((d - mean) ** 2 for d in data) / n_g
        std = math.sqrt(var)
        ci = 1.96 * std / math.sqrt(n_g)
        net = sum(data)
        return {
            "label": label,
            "win_rate": win_rate,
            "net_chips": net,
            "avg_per_game": mean,
            "std": std,
            "ci_95": ci,
            "loss_rate": losses / n_g,
            "tie_rate": 1.0 - win_rate - losses / n_g,
        }

    s_p0 = stats(results_p0, "固定策略(P0) vs CFR(P1)")
    s_p1 = stats(results_p1, "CFR(P0) vs 固定策略(P1)")

    # Combined: fixed player's perspective in both positions
    combined = results_p0 + [-d for d in results_p1]
    s_comb = stats(combined, "综合(消去位置偏差)")

    return s_p0, s_p1, s_comb


def print_result(s):
    print(f"\n  {s['label']}:")
    print(f"    WinRate:    {s['win_rate']*100:6.2f}%  "
          f"(输 {s['loss_rate']*100:.2f}%  平 {s['tie_rate']*100:.2f}%)")
    print(f"    NetChips:   {s['net_chips']:+.1f}")
    print(f"    Avg/Game:   {s['avg_per_game']:+.3f}")
    print(f"    Std:        {s['std']:.3f}")
    print(f"    95% CI:     ±{s['ci_95']:.3f}  "
          f"[{s['avg_per_game']-s['ci_95']:.3f}, {s['avg_per_game']+s['ci_95']:.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="牌数")
    parser.add_argument("--seq", type=int, nargs="+", default=[1, 6, 5, 4, 3, 2],
                        help="固定出牌顺序")
    parser.add_argument("--games", type=int, default=10000, help="每方局数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--load", type=str,
                        default=os.path.join(
                            os.path.dirname(__file__), "..", "..",
                            "results", "goofspiel", "models",
                            "goofspiel_n6_mccfr_cfr_seed42.pkl"),
                        help="加载 CFR 模型路径")
    args = parser.parse_args()

    fixed_seq = tuple(args.seq)
    assert len(fixed_seq) == args.n, f"序列长度 {len(fixed_seq)} != N={args.n}"

    # Load trained CFR snapshot
    print(f"Loading CFR from {args.load}...")
    with open(args.load, "rb") as f:
        data = pickle.load(f)
    cfr_snapshot = {}
    for ik, sums in data["strategy_sum"].items():
        tot = sum(sums.values())
        if tot > 0:
            cfr_snapshot[ik] = {k: v / tot for k, v in sums.items()}
    print(f"  loaded {len(cfr_snapshot)} infosets")

    print(f"\nFixed sequence: {fixed_seq}")
    print(f"Games per position: {args.games}")

    start = time.time()
    s_p0, s_p1, s_comb = evaluate(fixed_seq, cfr_snapshot, args.n,
                                    args.games, args.seed + 999)
    elapsed = time.time() - start

    print(f"\n{'='*55}")
    print_result(s_p0)
    print_result(s_p1)
    print_result(s_comb)
    print(f"\nTime: {elapsed:.1f}s  局/秒: {2*args.games/elapsed:.0f}")


if __name__ == "__main__":
    main()
