"""Full opponent enumeration (vanilla CFR at opponent nodes) for N=5."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer

t = GoofspielCFRTrainer(n=5, iterations=30_000, seed=42,
                         cfr_plus=False, full_opponent=True)
t.train(eval_every=5000)
