"""Quick test of CFR+ exploitability."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
from tqdm import tqdm
from methods.sage.cfr import CFRPlusTrainer, ANTE

trainer = CFRPlusTrainer(iterations=50_000)
t0 = time.perf_counter()
trainer.train()
dt = time.perf_counter() - t0
print(f"\nTraining: {dt:.0f}s")

from runs.leduc.compute_exploitability import best_response_value
import time
t1 = time.perf_counter()
br0 = best_response_value(trainer.strategy, 0)
print(f"BR(P0)={br0:+.4f}  ({time.perf_counter()-t1:.0f}s)")

t1 = time.perf_counter()
br1 = best_response_value(trainer.strategy, 1)
print(f"BR(P1)={br1:+.4f}  ({time.perf_counter()-t1:.0f}s)")
print(f"Exploitability = {(br0+br1)/2:.4f}")
print(f"Sum = {br0+br1:.4f}")

# Show preflop strategy
print("\n=== Preflop strategy ===")
for r in "JQK":
    info = f"{r}|?||0"
    if info in trainer.strategy:
        s = trainer.strategy[info]
        items = ", ".join(f"{a}: {p:.4f}" for a,p in sorted(s.items()))
        print(f"  {info}: {items}")
    info = f"{r}|?|c|0"
    if info in trainer.strategy:
        s = trainer.strategy[info]
        items = ", ".join(f"{a}: {p:.4f}" for a,p in sorted(s.items()))
        print(f"  {info}: {items}")
