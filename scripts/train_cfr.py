"""Train CFR+ for Leduc Hold'em and save to disk. Run once, offline."""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from methods.cfr import CFRPlusTrainer

os.makedirs("results/leduc/cfr_model", exist_ok=True)
save_path = "results/leduc/cfr_model/cfr_plus_100k.json"

print(f"[{time.strftime('%H:%M:%S')}] Training CFR+ 100k iterations...")
t0 = time.time()
trainer = CFRPlusTrainer(iterations=100000)
trainer.train()
dt = time.time() - t0

data = {
    "iterations": 100000,
    "training_time_s": round(dt, 1),
    "num_infosets": len(trainer.strategy),
    "strategy": {k: dict(v) for k, v in trainer.strategy.items()},
}
with open(save_path, "w") as f:
    json.dump(data, f)

print(f"[{time.strftime('%H:%M:%S')}] Done in {dt:.0f}s ({dt/60:.1f}min)")
print(f"Saved: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB, {len(trainer.strategy)} infosets)")
