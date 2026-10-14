"""Train II-Goofspiel MCCFR/MCCFR+ with detailed per-eval output.

Usage:
  python runs/goofspiel/train_goofspiel.py --n 4 --iters 20000 --eval 5000
  python runs/goofspiel/train_goofspiel.py --n 5 --iters 50000 --cfr-plus
"""

from collections import defaultdict
import argparse
import pickle
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from methods.sage.goofspiel_cfr import (
    GoofspielCFRTrainer,
    mask_to_list,
    hand_to_mask,
    infoset_key_from_hand,
)
from base.types import Action
from tqdm import tqdm

def root_policies(t, snapshot):
    """Return (P0_root_strat, P1_root_strat) from the snapshot."""
    s = t._initial_state()
    ik0 = infoset_key_from_hand(
        0, s.hands[0], s.outcome_history, (),
        opp_actions=(), prize_sequence=s.prize_sequence,
    )
    p0 = dict(sorted(snapshot.get(ik0, {}).items()))

    s1 = t._step(s, 1)  # dummy P0 action to reach P1's root info set
    ik1 = infoset_key_from_hand(
        1, s1.hands[1], s1.outcome_history, (),
        opp_actions=(), prize_sequence=s1.prize_sequence,
    )
    p1 = dict(sorted(snapshot.get(ik1, {}).items()))

    return p0, p1


def build_snapshot(t):
    snap = {}
    for ik, sums in t.strategy_sum.items():
        tot = sum(sums.values())
        if tot > 0:
            snap[ik] = {k: v / tot for k, v in sums.items()}
    return snap


def detail(t):
    snap = build_snapshot(t)
    br0 = t._br_player_value(0, snap)
    br1 = t._br_player_value(1, snap)
    exp = (br0 + br1) / 2.0
    p0s, p1s = root_policies(t, snap)
    return br0, br1, exp, p0s, p1s, len(t.strategy_sum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--eval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfr-plus", action="store_true", default=True)
    parser.add_argument("--no-cfr-plus", dest="cfr_plus", action="store_false")
    parser.add_argument("--variant", choices=["mccfr", "full"], default="mccfr")
    parser.add_argument("--stop-exp", type=float, default=0.01,
                        help="早停阈值：exp_avg 低于此值停止（默认 0.01）")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="关闭早停，跑满 iterations")
    parser.add_argument("--save", type=str, default=None,
                        help="保存路径；不指定则自动生成到 results/goofspiel/models/")
    parser.add_argument("--load", type=str, default=None,
                        help="加载已有模型继续训练 (*.pkl)")
    args = parser.parse_args()

    plus_tag = "cfrplus" if args.cfr_plus else "cfr"
    auto_name = f"goofspiel_n{args.n}_{args.variant}_{plus_tag}_seed{args.seed}.pkl"
    save_path = args.save or os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "goofspiel", "models", auto_name
    )

    t = GoofspielCFRTrainer(
        n=args.n,
        iterations=args.iters,
        seed=args.seed,
        cfr_plus=args.cfr_plus,
        cfr_variant=args.variant,
    )

    mode = "CFRfull" if args.variant == "full" else ("MCCFR+" if args.cfr_plus else "MCCFR")
    print(f"Game: II-Goofspiel N={args.n}  {mode}  seed={args.seed}")
    print(f"Iterations: {args.iters}  eval_every={args.eval}")

    if args.load:
        print(f"Loading: {args.load}")
        with open(args.load, "rb") as f:
            data = pickle.load(f)
        for ik, actions in data["regret_sum"].items():
            t.regret_sum[ik].update(actions)
        for ik, actions in data["strategy_sum"].items():
            t.strategy_sum[ik].update(actions)
        print(f"  loaded {len(data['strategy_sum'])} infosets")
    print()

    start = time.perf_counter()
    for i in tqdm(range(args.iters)):
        state = t._initial_state()
        if args.variant == "full":
            t._walk_cfr(state, 1.0, 1.0)
        else:
            t._walk(state, 1.0, 1.0, i % 2)

        if (i + 1) % args.eval == 0:
            br0, br1, exp, p0s, p1s, n_infosets = detail(t)
            elapsed = time.perf_counter() - start
            print(f"Step {i+1:6d}:  BR0={br0:.4f}  BR1={br1:.4f}  exp={exp:.4f}")
            print(f"  P0 root: {p0s}")
            print(f"  P1 root: {p1s}")
            print(f"  infosets={n_infosets}  elapsed={elapsed:.1f}s")
            print()

            if not args.no_early_stop and exp < args.stop_exp:
                print(f"🎯 exp={exp:.4f} < {args.stop_exp}，提前停止")
                break

    br0, br1, exp, p0s, p1s, n_infosets = detail(t)
    elapsed = time.perf_counter() - start
    print(f"{'='*50}")
    print(f"Final ({args.iters} iters, {elapsed:.1f}s)")
    print(f"  BR0={br0:.4f}  BR1={br1:.4f}  exp={exp:.4f}  infosets={n_infosets}")
    print(f"  P0 root: {p0s}")
    print(f"  P1 root: {p1s}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "regret_sum": dict(t.regret_sum),
            "strategy_sum": dict(t.strategy_sum),
            "n": args.n,
            "args": args,
        }, f)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
