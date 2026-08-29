"""Standard MCCFR N=5 for 500K iterations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer

t = GoofspielCFRTrainer(n=5, iterations=500_000, seed=42, cfr_plus=False)
t.train(eval_every=100000)
