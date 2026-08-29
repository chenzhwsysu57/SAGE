"""Inspect trained strategy and compare MCCFR vs MCCFR+."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer, mask_to_list, infoset_key_from_hand

# Train MCCFR (standard)
print("=== MCCFR (standard) ===")
t1 = GoofspielCFRTrainer(n=5, iterations=50_000, seed=42, cfr_plus=False)
t1.train(eval_every=0)

mask_all = (1 << 6) - 2  # bits 1..5 = 62

# P0 round 1
ik0 = (0, mask_all, ())
strat0 = t1.strategy.get(ik0, {})
print("\nP0 round 1 strategy:")
for k, v in sorted(strat0.items()):
    print(f"  bid {k}: {v:.4f}")

# P1 round 1
ik1 = (1, mask_all, ())
strat1 = t1.strategy.get(ik1, {})
print("\nP1 round 1 strategy:")
for k, v in sorted(strat1.items()):
    print(f"  bid {k}: {v:.4f}")

# Expected game value
def expected_value(t):
    def traverse(state):
        if state.phase == "terminal":
            return state.rewards[0]
        p0c = mask_to_list(state.hands[0], t.n)
        p1c = mask_to_list(state.hands[1], t.n)
        ik0 = infoset_key_from_hand(0, state.hands[0], state.outcome_history)
        ik1 = infoset_key_from_hand(1, state.hands[1], state.outcome_history)
        s0 = t.strategy.get(ik0)
        s1 = t.strategy.get(ik1)
        if s0 is None:
            s0 = {c: 1.0/len(p0c) for c in p0c}
        if s1 is None:
            s1 = {c: 1.0/len(p1c) for c in p1c}
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

ev = expected_value(t1)
print(f"\nGame value (P0 reward) under avg strategy: {ev:.4f}")

print("\n=== MCCFR+ ===")
t2 = GoofspielCFRTrainer(n=5, iterations=50_000, seed=42, cfr_plus=True)
t2.train(eval_every=0)

strat0p = t2.strategy.get(ik0, {})
print("\nP0 round 1 strategy:")
for k, v in sorted(strat0p.items()):
    print(f"  bid {k}: {v:.4f}")

strat1p = t2.strategy.get(ik1, {})
print("\nP1 round 1 strategy:")
for k, v in sorted(strat1p.items()):
    print(f"  bid {k}: {v:.4f}")

ev2 = expected_value(t2)
print(f"\nGame value (P0 reward) under avg strategy: {ev2:.4f}")
