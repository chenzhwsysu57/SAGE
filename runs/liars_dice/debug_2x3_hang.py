"""Debug: run one 2x3 game with Honest archetype, verbose + timeout."""
from __future__ import annotations

import os, sys, time, signal
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from apis.openai_wrapper import OpenAIModel
from games.liars_dice import LiarsDice
from methods.simple_prompt import SimplePromptMethod
from modeling.liars_dice.archetypes import ArchetypeAgent
from rollout import RolloutRunner

class TimeoutOpenAIModel(OpenAIModel):
    """Wrapper with 30s HTTP request timeout."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Recreate client with timeout
        import openai
        resolved_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        resolved_url = kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        client_kwargs = {"api_key": resolved_key, "timeout": 30.0}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url
        self.client = openai.OpenAI(**client_kwargs)

model = TimeoutOpenAIModel(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
)

# 2x3 config: 2 dice/player, faces 1-3
game = LiarsDice(num_dice=2, num_faces=3, seed=42)

p0 = SimplePromptMethod(name="SimplePrompt-P0")
p1 = ArchetypeAgent("Honest", name="Honest-P1")

methods = {0: p0, 1: p1}

print("=== Starting 2x3 Honest debug game ===")
print(f"Game: {game.num_dice}dice x {game.num_faces}faces")
print(f"P0: {p0.name}")
print(f"P1: {p1.name}")
print(f"Config: LiarsDice(num_dice={game.num_dice}, num_faces={game.num_faces})")
print()

runner = RolloutRunner(game, methods, model, verbose=True)

t0 = time.time()
try:
    traj = runner.run()
    elapsed = time.time() - t0
    print(f"\n=== Game finished in {elapsed:.1f}s ({len(traj.steps)} steps) ===")
    print(f"Payout: {traj.payout}")
    print(f"Winner: P{traj.winner}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"\n=== FAILED after {elapsed:.1f}s ===")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
