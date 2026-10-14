"""Check exploitability at 500k iterations (~3 hours)."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")

from methods.sage.cfr import CFRPlusTrainer, ANTE
from runs.leduc.compute_exploitability import best_response_value

t0 = time.perf_counter()
trainer = CFRPlusTrainer(iterations=500_000)
trainer.train()
dt = time.perf_counter() - t0
print(f"Training: {dt:.0f}s")

t1 = time.perf_counter()
br0 = best_response_value(trainer.strategy, br_player=0)
print(f"BR(P0) took {time.perf_counter()-t1:.0f}s")

t1 = time.perf_counter()
br1 = best_response_value(trainer.strategy, br_player=1)
print(f"BR(P1) took {time.perf_counter()-t1:.0f}s")

exp = (br0 + br1) / 2.0
print(f"\n=== 500k iterations ===")
print(f"BR(P0) = {br0:+.4f}  BR(P1) = {br1:+.4f}")
print(f"Exploitability = {exp:.4f} chips/hand ({exp/ANTE*100:.2f}% of ante)")
