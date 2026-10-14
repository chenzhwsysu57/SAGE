"""Unit test: verify BR computation against known values.

N=2 uniform策略下，exploitability应为0.5。这验证BR没有系统性错误。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer, hand_to_mask

def test_uniform_exploitability():
    """Verify that uniform strategy has the correct exploitability for N=2."""
    t = GoofspielCFRTrainer(n=2, iterations=1, seed=0)

    # Build a uniform snapshot: all possible info sets with uniform strategy
    # For N=2, the info sets are:
    # r1: (0, mask_all, ()), (1, mask_all, ())
    # r2: (0, mask_without_X, (+1)), (0, mask_without_X, (-1)), (0, mask_without_X, (0))
    #     (1, mask_without_Y, (+1)), (1, mask_without_Y, (-1)), (1, mask_without_Y, (0))

    mask_all = hand_to_mask({1, 2})
    mask_1 = hand_to_mask({2})
    mask_2 = hand_to_mask({1})

    uniform_snapshot = {}

    # Round 1
    uniform_snapshot[(0, mask_all, ())] = {1: 0.5, 2: 0.5}
    uniform_snapshot[(1, mask_all, ())] = {1: 0.5, 2: 0.5}

    # Round 2, P0 after each outcome
    # mask_1 = hand_to_mask({2}) = only card 2 remaining → legal action is {2: 1.0}
    # mask_2 = hand_to_mask({1}) = only card 1 remaining → legal action is {1: 1.0}
    for outcome in (1, -1, 0):
        uniform_snapshot[(0, mask_1, (outcome,))] = {2: 1.0}
        uniform_snapshot[(0, mask_2, (outcome,))] = {1: 1.0}
        uniform_snapshot[(1, mask_1, (outcome,))] = {2: 1.0}
        uniform_snapshot[(1, mask_2, (outcome,))] = {1: 1.0}

    br0 = t._br_player_value(0, uniform_snapshot)
    br1 = t._br_player_value(1, uniform_snapshot)
    expl = (br0 + br1) / 2.0

    print(f"BR_0 = {br0:.4f}  BR_1 = {br1:.4f}  exploitability = {expl:.4f}")
    print(f"Expected: BR_0=0.5 BR_1=0.5 exploitability=0.5")

    # Allow small tolerance — sampling in MCCFR creates slight numerical differences
    # but on the BR (full enumeration, deterministic) it should be exact
    assert abs(expl - 0.5) < 1e-10, f"Expected 0.5, got {expl}"

test_uniform_exploitability()
print("\nN=2 uniform test PASSED")
