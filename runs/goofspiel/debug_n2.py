"""Debug: N=2 CFR convergence check. Known equilibrium: 50-50 in round 1."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer

trainer = GoofspielCFRTrainer(n=2, iterations=5000, seed=0)
trainer.train(eval_every=500)

# Print strategy for P0 at round 1 (all cards, no history)
print("\n── P0 round-1 strategy (all cards, no history) ──")
ik = (0, 0b110, ())  # mask for cards 1,2 = bits 1,2 set = 6
if ik in trainer.strategy:
    for k, v in sorted(trainer.strategy[ik].items()):
        print(f"  bid {k}: {v:.4f}")
else:
    print("  (not found)")

# Print strategy for P1 at round 1 same info set
ik1 = (1, 0b110, ())
if ik1 in trainer.strategy:
    for k, v in sorted(trainer.strategy[ik1].items()):
        print(f"  bid {k}: {v:.4f}")
else:
    print("  (not found)")

# Check exploitability trends
import re
with open("/dev/stdout", "w") as f:
    pass  # suppress further output
