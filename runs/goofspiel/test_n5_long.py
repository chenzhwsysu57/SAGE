"""N=5 long run to check convergence trend."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer

trainer = GoofspielCFRTrainer(n=5, iterations=150_000, seed=42)
trainer.train(eval_every=10000)
