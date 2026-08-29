"""Compute exploitability of uniform random strategy for baseline."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from methods.cfr import CFRPlusTrainer, ANTE, DECK
from runs.leduc.compute_exploitability import best_response_value
import itertools

# Uniform strategy: equal probs for all actions at every infoset
# Build it by creating uniform distributions for a typical infoset
# Actually, just use a trainer with 0 iterations (pure uniform random)

trainer = CFRPlusTrainer(iterations=1)
# Don't train, just use the initial uniform strategy from regret matching

# Manually build uniform strategy - all infosets have equal probs
# After 1 iteration, the strategy should be close to uniform but may have some noise
# Better: build a pure uniform strategy dict

class FakeTrainer:
    def __init__(self):
        self.strategy = {}

# Create fake uniform strategy for reachable infosets
# Train 1 iteration just to populate infosets and get uniform probs
trainer = CFRPlusTrainer(iterations=1)
trainer.train()

# The strategy after 1 iteration uses regret matching, which starts uniform
# Let's verify it's uniform
for info, probs in list(trainer.strategy.items())[:5]:
    print(f"{info}: {probs}")

br0 = best_response_value(trainer.strategy, br_player=0)
br1 = best_response_value(trainer.strategy, br_player=1)
exp = (br0 + br1) / 2.0
print(f"\n=== After 1 iteration (~uniform) ===")
print(f"BR(P0) = {br0:+.4f}  BR(P1) = {br1:+.4f}")
print(f"Exploitability = {exp:.4f}")
