"""Check if the game value under average strategy is reasonable.
If the average strategy is close to Nash, the game value should be ≈ 0
and the BR should not find large deviations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer, mask_to_list, infoset_key_from_hand

# N=5 with few iterations - quick check
t = GoofspielCFRTrainer(n=5, iterations=10_000, seed=42, cfr_plus=False)
t.train(eval_every=0)

# Compute expected game value
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

# Also compute BR for both players
from methods.goofspiel_cfr import regret_matching_strategy, hand_to_mask

# Build snapshot from strategy_sum
snapshot = {}
for infoset, sums in t.strategy_sum.items():
    total = sum(sums.values())
    if total > 0:
        snapshot[infoset] = {k: v / total for k, v in sums.items()}

print(f"Info sets: {len(snapshot)}")

# Quick sanity: check if P0 and P1 strategies are symmetric at round 1
mask_all = hand_to_mask(set(range(1,6)))
ik0 = (0, mask_all, ())
ik1 = (1, mask_all, ())

print("\nP0 round 1 strategy:")
for k, v in sorted(t.strategy.get(ik0, {}).items()):
    print(f"  {k}: {v:.4f}")
print("P1 round 1 strategy:")
for k, v in sorted(t.strategy.get(ik1, {}).items()):
    print(f"  {k}: {v:.4f}")

ev = expected_value(t)
print(f"\nGame value (P0 reward) under avg strategy: {ev:.4f}")

# Compute BR
br0 = t._br_player_value(0, snapshot)
br1 = t._br_player_value(1, snapshot)
print(f"\nBR_0 = {br0:.4f}, BR_1 = {br1:.4f}")
print(f"Exploitability = {(br0 + br1) / 2:.4f}")
