"""Debug one 2×3 game with forced stdout flush and 120s per-game timeout."""
from __future__ import annotations

import os, sys, signal, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Unbuffered
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

from apis.openai_wrapper import OpenAIModel
from games.liars_dice import LiarsDice
from methods.baseline.simple_prompt import SimplePromptMethod
from modeling.liars_dice.archetypes import ArchetypeAgent
from rollout import RolloutRunner

# ── Global timeout via SIGALRM ────────────────────────────────────────
class TimeoutError_(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutError_("Game timed out")

# ── Test one game ──────────────────────────────────────────────────────
arch = sys.argv[1] if len(sys.argv) > 1 else "Honest"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
timeout_s = int(sys.argv[3]) if len(sys.argv) > 3 else 120

print(f"[debug] archetype={arch} seed={seed} timeout={timeout_s}s", flush=True)

model = OpenAIModel(model="deepseek-v4-flash", base_url="https://api.deepseek.com")
game = LiarsDice(num_dice=2, num_faces=3, seed=seed)

p0 = SimplePromptMethod(name="SimplePrompt-P0")
p1 = ArchetypeAgent(arch, name=f"{arch}-P1", seed=seed)

runner = RolloutRunner(game, {0: p0, 1: p1}, model, verbose=True)

signal.signal(signal.SIGALRM, _handler)
signal.alarm(timeout_s)

t0 = time.time()
try:
    traj = runner.run()
    dt = time.time() - t0
    signal.alarm(0)
    print(f"\n[debug] DONE in {dt:.1f}s | {len(traj.steps)} steps | payout={traj.payout}", flush=True)
except TimeoutError_:
    dt = time.time() - t0
    print(f"\n[debug] TIMEOUT after {dt:.1f}s", flush=True)
except Exception as e:
    dt = time.time() - t0
    print(f"\n[debug] ERROR after {dt:.1f}s: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
