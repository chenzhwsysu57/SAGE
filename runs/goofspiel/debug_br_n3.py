"""Debug: compare BR against brute-force for N=3 uniform strategy.
For N=3 with uniform strategy, the BR should be computable analytically.
Prizes: 3, 2, 1. Total pool = 6."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer, hand_to_mask, mask_to_list

t = GoofspielCFRTrainer(n=3, iterations=1, seed=0)

# Build uniform snapshot: all infosets with uniform strategy
mask_all = hand_to_mask({1, 2, 3})
uniform_snapshot = {}

# Round 1
uniform_snapshot[(0, mask_all, ())] = {1: 1/3, 2: 1/3, 3: 1/3}
uniform_snapshot[(1, mask_all, ())] = {1: 1/3, 2: 1/3, 3: 1/3}

# Round 2: after outcome, each player has 2 cards left
# Generate all 2-card subsets
from itertools import combinations
for outcome in (1, -1, 0):
    for hand_mask in [hand_to_mask(set(combo)) for combo in combinations([1, 2, 3], 2)]:
        uniform_snapshot[(0, hand_mask, (outcome,))] = {
            c: 1.0/len(mask_to_list(hand_mask, 3)) for c in mask_to_list(hand_mask, 3)
        }
        uniform_snapshot[(1, hand_mask, (outcome,))] = {
            c: 1.0/len(mask_to_list(hand_mask, 3)) for c in mask_to_list(hand_mask, 3)
        }

# Round 3: after two outcomes, each player has 1 card left
for outcome1 in (1, -1, 0):
    for outcome2 in (1, -1, 0):
        outcome_hist = (outcome1, outcome2)
        for hand_mask in [hand_to_mask({c}) for c in [1, 2, 3]]:
            uniform_snapshot[(0, hand_mask, outcome_hist)] = {
                c: 1.0 for c in mask_to_list(hand_mask, 3)
            }
            uniform_snapshot[(1, hand_mask, outcome_hist)] = {
                c: 1.0 for c in mask_to_list(hand_mask, 3)
            }

br0 = t._br_player_value(0, uniform_snapshot)
br1 = t._br_player_value(1, uniform_snapshot)
expl = (br0 + br1) / 2.0

print(f"Uniform strategy N=3:")
print(f"  BR_0 = {br0:.4f}")
print(f"  BR_1 = {br1:.4f}")
print(f"  Exploitability = {expl:.4f}")

# Also compute game value under uniform strategy (should be 0 by symmetry)
def uniform_traverse(state):
    if state.phase == "terminal":
        return state.rewards[0]
    h0 = mask_to_list(state.hands[0], 3)
    h1 = mask_to_list(state.hands[1], 3)
    val = 0.0
    for c0 in h0:
        for c1 in h1:
            ns = t._resolve_round(state, c0, c1)
            val += (1.0/len(h0)) * (1.0/len(h1)) * uniform_traverse(ns)
    return val

game_val = uniform_traverse(t._initial_state())
print(f"  Game value (uniform vs uniform) = {game_val:.4f}")
