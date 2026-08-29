"""Quick N=4 convergence check."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer

trainer = GoofspielCFRTrainer(n=4, iterations=30_000, seed=42)
trainer.train(eval_every=2000)
