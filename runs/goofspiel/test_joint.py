"""Test joint round walk for N=5 — verify symmetry and convergence."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer, hand_to_mask, regret_matching_strategy

t = GoofspielCFRTrainer(n=5, iterations=1, seed=42, cfr_plus=False)

print("Training with joint round walk...")
for i in range(5):
    for _ in range(10000):
        traverser = t.rng.randint(0, 1)
        state = t._initial_state()
        t._walk(state, 1.0, 1.0, traverser)

    # Build snapshot
    snapshot = {}
    for infoset, sums in t.strategy_sum.items():
        total = sum(sums.values())
        if total > 0:
            snapshot[infoset] = {k: v / total for k, v in sums.items()}

    br0 = t._br_player_value(0, snapshot)
    br1 = t._br_player_value(1, snapshot)
    expl = (br0 + br1) / 2.0

    # Round 1 strategies from current regrets
    mask_all = hand_to_mask(set(range(1, 6)))
    r0 = t.regret_sum.get((0, mask_all, ()), {})
    r1 = t.regret_sum.get((1, mask_all, ()), {})
    keys0 = sorted(r0.keys())
    keys1 = sorted(r1.keys())
    if keys0:
        probs0 = regret_matching_strategy([max(r0[k], 0.0) for k in keys0])
        s0_disp = dict(zip(keys0, [f"{p:.4f}" for p in probs0]))
    else:
        s0_disp = {}
    if keys1:
        probs1 = regret_matching_strategy([max(r1[k], 0.0) for k in keys1])
        s1_disp = dict(zip(keys1, [f"{p:.4f}" for p in probs1]))
    else:
        s1_disp = {}

    elapsed = (i+1)*10000
    print(f"\n{elapsed}: BR_0={br0:.4f}  BR_1={br1:.4f}  expl={expl:.4f}  "
          f"ratio={br1/max(br0,1e-10):.2f}  infosets={len(t.strategy_sum)}")
    print(f"  P0 r1: {s0_disp}")
    print(f"  P1 r1: {s1_disp}")
