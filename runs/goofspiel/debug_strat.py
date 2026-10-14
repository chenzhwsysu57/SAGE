"""Inspect the average strategy after training."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer, mask_to_list, infoset_key_from_hand

trainer = GoofspielCFRTrainer(n=6, iterations=20_000, seed=42)
trainer.train(eval_every=0)  # no exploitability eval during training (slow)

# Round 1 strategy for P0
mask_all = (1 << 7) - 2  # bits 1..6 = 126
ik0_r1 = (0, mask_all, ())
if ik0_r1 in trainer.strategy:
    print("\n── P0 round 1 strategy ──")
    for k, v in sorted(trainer.strategy[ik0_r1].items()):
        print(f"  bid {k}: {v:.4f}")

# Round 1 strategy for P1
ik1_r1 = (1, mask_all, ())
if ik1_r1 in trainer.strategy:
    print("\n── P1 round 1 strategy ──")
    for k, v in sorted(trainer.strategy[ik1_r1].items()):
        print(f"  bid {k}: {v:.4f}")

# Expected game value under this strategy profile
def expected_game_value(t):
    def traverse(state):
        if state.phase == "terminal":
            return state.rewards[0]
        p0c = mask_to_list(state.hands[0], t.n)
        p1c = mask_to_list(state.hands[1], t.n)
        ik0 = infoset_key_from_hand(0, state.hands[0], state.outcome_history)
        ik1 = infoset_key_from_hand(1, state.hands[1], state.outcome_history)
        s0 = t.strategy.get(ik0, None)
        s1 = t.strategy.get(ik1, None)
        if s0 is None: p0u, s0 = 1.0/len(p0c), {c: 1.0/len(p0c) for c in p0c}
        if s1 is None: p1u, s1 = 1.0/len(p1c), {c: 1.0/len(p1c) for c in p1c}
        val = 0.0
        for c0 in p0c:
            pr0 = s0.get(c0, 0.0)
            if pr0 == 0: continue
            for c1 in p1c:
                pr1 = s1.get(c1, 0.0)
                if pr1 == 0: continue
                ns = t._resolve_round(state, c0, c1)
                val += pr0 * pr1 * traverse(ns)
        return val
    return traverse(t._initial_state())

ev = expected_game_value(trainer)
print(f"\nExpected game value (P0 reward) under avg strategy: {ev:.4f}")
print(f"  (should be ≈ 0 for symmetric equilibrium)")
