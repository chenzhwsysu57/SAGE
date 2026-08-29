"""Compare MCCFR vs MCCFR+ for N=5."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.goofspiel_cfr import GoofspielCFRTrainer

print("=" * 60)
print("MCCFR (standard) N=5, 50K iterations")
print("=" * 60)
t1 = GoofspielCFRTrainer(n=5, iterations=50_000, seed=42, cfr_plus=False)
t1.train(eval_every=10000)

print("\n" + "=" * 60)
print("MCCFR+ (cfr_plus) N=5, 50K iterations")
print("=" * 60)
t2 = GoofspielCFRTrainer(n=5, iterations=50_000, seed=42, cfr_plus=True)
t2.train(eval_every=10000)
