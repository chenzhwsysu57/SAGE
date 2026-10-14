"""Check BR asymmetry at different iteration counts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer, hand_to_mask

for n_iter in [10000, 50000, 100000]:
    t = GoofspielCFRTrainer(n=5, iterations=n_iter, seed=42, cfr_plus=False)
    t.train(eval_every=0)

    snapshot = {}
    for infoset, sums in t.strategy_sum.items():
        total = sum(sums.values())
        if total > 0:
            snapshot[infoset] = {k: v / total for k, v in sums.items()}

    br0 = t._br_player_value(0, snapshot)
    br1 = t._br_player_value(1, snapshot)
    expl = (br0 + br1) / 2.0

    print(f"\n{n_iter} iterations: BR_0={br0:.4f}  BR_1={br1:.4f}  "
          f"expl={expl:.4f}  ratio={br1/max(br0,1e-10):.2f}")

    # Check round-1 strategies
    mask_all = hand_to_mask(set(range(1, 6)))
    s0 = t.strategy.get((0, mask_all, ()), {})
    s1 = t.strategy.get((1, mask_all, ()), {})
    print(f"  P0 strat: {dict(sorted(s0.items()))}")
    print(f"  P1 strat: {dict(sorted(s1.items()))}")
