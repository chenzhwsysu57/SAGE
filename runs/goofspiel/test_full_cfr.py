"""Full CFR for N=5 — exact counterfactual values, zero sampling variance."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer

t = GoofspielCFRTrainer(n=5, iterations=20000, seed=42,
                         cfr_plus=False, cfr_variant="full")
t.train(eval_every=1000)
