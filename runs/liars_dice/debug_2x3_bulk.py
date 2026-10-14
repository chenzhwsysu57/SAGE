"""Debug: run multiple 2x3 games with Honest archetype, check for hangs."""
from __future__ import annotations

import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from apis.openai_wrapper import OpenAIModel
from games.liars_dice import LiarsDice
from methods.baseline.simple_prompt import SimplePromptMethod
from modeling.liars_dice.archetypes import ArchetypeAgent
from rollout import RolloutRunner

import openai
# Client with 30s timeout
base_kwargs = {"api_key": os.environ.get("OPENAI_API_KEY"), "base_url": os.environ.get("OPENAI_BASE_URL"), "timeout": 30.0}
client = openai.OpenAI(**base_kwargs)

# Monkey-patch the model
orig_init = OpenAIModel.__init__
def patched_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.client = client
OpenAIModel.__init__ = patched_init

model = OpenAIModel(model="deepseek-v4-flash", base_url="https://api.deepseek.com")

# Test all archetypes with 2x3
from modeling.liars_dice.archetypes import ARCHETYPE_NAMES
import random

for arch in ARCHETYPE_NAMES:
    for seed in [42, 43, 44, 100, 101]:
        game = LiarsDice(num_dice=2, num_faces=3, seed=seed)
        p0 = SimplePromptMethod(name="SimplePrompt-P0")
        p1 = ArchetypeAgent(arch, name=f"{arch}-P1", seed=seed)
        runner = RolloutRunner(game, {0: p0, 1: p1}, model, verbose=False)
        t0 = time.time()
        try:
            traj = runner.run()
            elapsed = time.time() - t0
            print(f"  {arch:<25} seed={seed:<3} OK  {elapsed:5.1f}s {len(traj.steps):>2} steps payout={traj.payout}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {arch:<25} seed={seed:<3} FAIL after {elapsed:.0f}s: {type(e).__name__}: {str(e)[:80]}")
