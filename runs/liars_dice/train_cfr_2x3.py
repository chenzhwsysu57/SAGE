#!/usr/bin/env python3
"""Train CFR for 2-dice x 3-face Liar's Dice with early stopping.

Saves model to results/liars_dice/cfr/cfr_2x3_evalevery/model when
exploitability drops below 0.01, or after max iterations.
"""
import json, os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from methods.liars_dice_cfr import LiarsDiceCFRTrainer, converged_strategy

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/liars_dice/cfr/cfr_2x3_evalevery")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_DICE = 2; NUM_FACES = 3
MAX_ITERATIONS = 500_000
SEED = 42
EVAL_EVERY = 1000
THRESHOLD = 0.01

LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")

def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

trainer = LiarsDiceCFRTrainer(
    num_dice=NUM_DICE, num_faces=NUM_FACES,
    iterations=MAX_ITERATIONS, seed=SEED,
    traverser_action_sample_size=None,
)

log(f"Liar's Dice CFR ({NUM_DICE}d×{NUM_FACES}f)")
log(f"  max_iters={MAX_ITERATIONS} eval_every={EVAL_EVERY} threshold={THRESHOLD}")

# ── Estimate iteration speed ──────────────────────────────────────────
t0 = time.time()
for i in range(1000):
    traverser = i % 2
    state = trainer._reset_state()
    trainer._walk(state, 1.0, 1.0, traverser)
t1 = time.time()
log(f"  1k iters in {t1-t0:.3f}s ({(t1-t0)/1000*1e6:.0f} us/iter)")

# ── Estimate first exploitability cost ───────────────────────────────
log("  Computing initial exploitability...")
te0 = time.time()
exp = trainer.compute_exploitability()
te1 = time.time()
log(f"  Initial exploitability = {exp:.6g} (took {te1-te0:.1f}s)")

# ── Main training loop ───────────────────────────────────────────────
converged_at = None
curve = []
t_start = time.time()
next_eval = EVAL_EVERY
i = 1000

while i < MAX_ITERATIONS:
    if i >= next_eval:
        exp = trainer.compute_exploitability()
        elapsed = time.time() - t_start
        it_per_sec = i / max(elapsed, 1e-9)
        infosets = len(trainer.strategy_sum)
        curve.append({"iteration": i, "exploitability": exp,
                       "infosets": infosets, "time": elapsed})
        log(f"  iter={i:,}  exp={exp:.6g}  infosets={infosets}  "
              f"it/s={it_per_sec:.0f}  elapsed={elapsed:.1f}s")
        next_eval += EVAL_EVERY

        if exp < THRESHOLD:
            converged_at = i
            log(f"\n*** CONVERGED at iter {i:,}: exp={exp:.6g} ***")
            break

    traverser = i % 2
    state = trainer._reset_state()
    trainer._walk(state, 1.0, 1.0, traverser)
    i += 1

total_time = time.time() - t_start

# ── Finalize strategy ────────────────────────────────────────────────
trainer.strategy = {}
for infoset, sums in trainer.strategy_sum.items():
    avg = converged_strategy(dict(sums))
    if avg:
        trainer.strategy[infoset] = avg
trainer._num_infosets = len(trainer.strategy)
trainer._trained = True

final_exp = trainer.compute_exploitability()
log(f"  Final verification: exploitability = {final_exp:.6g}")

# ── Save ─────────────────────────────────────────────────────────────
json.dump(curve, open(os.path.join(OUTPUT_DIR, "exploitability_curve.json"), "w"), indent=2)
json.dump({
    "num_dice": NUM_DICE, "num_faces": NUM_FACES,
    "max_iterations": MAX_ITERATIONS, "seed": SEED,
    "eval_every": EVAL_EVERY, "threshold": THRESHOLD,
    "run_phase": "dev_debug",
}, open(os.path.join(OUTPUT_DIR, "config.json"), "w"), indent=2)

trainer.save(MODEL_PATH, metadata={
    "training_time_seconds": round(total_time, 2),
    "final_exploitability": final_exp,
    "num_infosets": trainer._num_infosets,
    "converged_at_iteration": converged_at,
})

log(f"\n{'='*60}")
log(f"  Results: {MODEL_PATH}")
log(f"  Iterations: {converged_at or MAX_ITERATIONS:,}")
log(f"  Final exp:  {final_exp:.8f}")
log(f"  Infosets:   {trainer._num_infosets:,}")
log(f"  Time:       {total_time:.1f}s ({total_time/60:.1f}m)")
log(f"{'='*60}")
