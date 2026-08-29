"""Check if regrets are accumulating over time."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from methods.cfr import CFRPlusTrainer

trainer = CFRPlusTrainer(iterations=20_000)

# Run first iteration
trainer._walk(("J","H"), ("Q","S"), ("K","H"), "", 0, 0, 1.0, 1.0, 1, 1)

# Check regret values
print("=== After 1 iteration ===")
regrets_snapshot = dict(trainer.regrets)
for k in sorted(regrets_snapshot.keys())[:10]:
    print(f"  {k}: {dict(regrets_snapshot[k])}")

# Run full training
trainer = CFRPlusTrainer(iterations=20_000)
trainer.train()

print("\n=== After 20k iterations ===")
for k in sorted(trainer.strategy.keys())[:10]:
    print(f"  {k}: {dict(trainer.strategy[k])}")

print(f"\nTotal info sets: {len(trainer.strategy)}")
print(f"Total regret entries: {sum(len(v) for v in trainer.regrets.values())}")

# Check if non-uniform strategies exist
uniform_count = sum(1 for v in trainer.strategy.values()
                    if len(v) > 0 and all(abs(p - 1.0/len(v)) < 0.05 for p in v.values()))
print(f"Near-uniform strategies: {uniform_count}/{len(trainer.strategy)}")
