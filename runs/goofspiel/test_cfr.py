"""Goofspiel CFR convergence test — N=6."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer

trainer = GoofspielCFRTrainer(n=5, iterations=30_000, seed=42)
trainer.train(eval_every=1000)  # no mid-training eval (BR is slow for N=6)
print(f"\nFinal exploitability: {trainer.compute_exploitability():.6f}")
