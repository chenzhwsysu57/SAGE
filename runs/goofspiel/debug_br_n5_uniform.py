"""Check BR for uniform strategy on N=5."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer, hand_to_mask, mask_to_list
from itertools import combinations

t = GoofspielCFRTrainer(n=5, iterations=1, seed=0)

# Build uniform snapshot
uniform_snapshot = {}

# All possible hands for each round
all_cards = set(range(1, 6))

def build_uniform_for_round(snapshot, player, round_num):
    n = 5
    n_cards_remaining = n - round_num + 1
    for combo in combinations(range(1, n+1), n_cards_remaining):
        mask = hand_to_mask(set(combo))
        # All possible outcome histories for this round
        # (This is a simplification — we'd need all histories, but let's just do all)
        ...

# Simpler: just generate all possible infosets
for r in range(1, 6):  # rounds 1-5
    n_cards = 6 - r
    for outcome_hist_len in range(r):
        for outcome_hist in [[]]:  # placeholder
            ...

# Even simpler: just compute BR for a very limited snapshot
# Uniform strategy means every legal action is equally likely
# For each info set, the uniform strategy is 1/(num_legal_actions)

# Pre-generate all possible info sets
all_masks = set(range(1, 32))  # all 5-bit patterns representing subsets of {1,2,3,4,5}

for r in range(1, 6):  # round r (1-indexed)
    n_cards_remaining = 6 - r
    # Generate all possible hands of size n_cards_remaining
    hands = [hand_to_mask(set(combo)) for combo in combinations(range(1, 6), n_cards_remaining)]
    # Generate all possible outcome histories of length r-1
    # This is 3^(r-1) possibilities — manageable for N=5
    # Total: 5 * 3^4 * C(5,n_cards) = 5 * 81 * C(5,n_cards) ~ 405 * [5+10+10+5+1] = 12,555 infosets
    # That's a lot but manageable

# OK let's be smarter. Let's just use the simplified BR value.
# Actually, let me just check if the BR computation itself has errors by comparing
# with a direct EV computation for a specific BR action at a specific node.

# For round 1, BR_0 uniform strategy:
# P0 picks br_card, P1 has all 5 cards and picks uniformly
# E[reward|br_card] = sum_{opp_card} (1/5) * value(br_card, opp_card, future)

# Compute exact BR for round 1 (lookahead to future rounds)
# For N=5 with small tree (5! × 5! = 14,400 nodes), we can compute exact BR by DP

from functools import lru_cache

n = 5

def exact_br_value(p0_hand_mask, p1_hand_mask, outcome_history, br_player):
    """Exact BR value by enumeration."""
    # If terminal (all cards used)
    p0_cards = mask_to_list(p0_hand_mask, n)
    p1_cards = mask_to_list(p1_hand_mask, n)

    if not p0_cards and not p1_cards:
        # Terminal: compute reward from score
        # But we don't track scores in this function...
        # We need scores to be passed as parameter
        pass

# This approach is getting complex. Let me instead verify against the N=3 uniform case
# and then use debug_strat.py to inspect the trained strategy.

# N=3 should give BR_0 ≈ 1.333 (as we verified earlier)
# Let me verify N=5 gives something reasonable

print("Computing BR for uniform N=5 (may take a moment)...")

# Use a complete uniform snapshot
uniform_snapshot = {}

# Round 1: both have all 5 cards
mask_all = hand_to_mask({1, 2, 3, 4, 5})
uniform_snapshot[(0, mask_all, ())] = {c: 0.2 for c in range(1, 6)}
uniform_snapshot[(1, mask_all, ())] = {c: 0.2 for c in range(1, 6)}

# Round 2-5: all possible hands with uniform strategy
for r in range(2, 6):
    n_cards = 6 - r
    hands = [hand_to_mask(set(combo)) for combo in combinations(range(1, 6), n_cards)]
    # All outcome histories of length r-1
    # 3^(r-1) possibilities
    n_histories = 3**(r-1)
    hist_values = []
    for h in range(n_histories):
        hist = []
        temp = h
        for _ in range(r-1):
            hist.append((temp % 3) - 1)  # -1, 0, or 1
            temp //= 3
        hist_values.append(tuple(hist))

    for hand in hands:
        cards = mask_to_list(hand, n)
        n_cards = len(cards)
        uniform_strat = {c: 1.0/n_cards for c in cards}
        for hist in hist_values:
            uniform_snapshot[(0, hand, hist)] = uniform_strat
            uniform_snapshot[(1, hand, hist)] = uniform_strat

print(f"Built uniform snapshot with {len(uniform_snapshot)} infosets")

br0 = t._br_player_value(0, uniform_snapshot)
br1 = t._br_player_value(1, uniform_snapshot)
expl = (br0 + br1) / 2.0

print(f"N=5 uniform strategy:")
print(f"  BR_0 = {br0:.4f}")
print(f"  BR_1 = {br1:.4f}")
print(f"  Exploitability = {expl:.4f}")
print(f"  Total prize pool = {sum(range(1, n+1))}")
