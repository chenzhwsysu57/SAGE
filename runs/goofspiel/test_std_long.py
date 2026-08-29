"""Standard MCCFR N=5 long run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer

t = GoofspielCFRTrainer(n=5, iterations=200_000, seed=42, cfr_plus=False)
t.train(eval_every=25000)
