"""Run Decrypto with Eve using semantic keyword belief model.

Usage:
    # Dry-run: preview prompt structure without calling LLM
    python runs/decrypto/decrypto_belief.py --dry-run

    # Real run with DeepSeek
    python runs/decrypto/decrypto_belief.py --model deepseek-v4-flash --seed 42

    # Save prompts to disk
    python runs/decrypto/decrypto_belief.py --model deepseek-v4-flash --save-prompts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT))

from games.decrypto import Decrypto
from methods.baseline.decrypto import DecryptoMethod
from methods.helpers import build_example
from modeling.decrypto.semantic_belief import DecryptoBeliefState, make_embed_fn
from rollout import RolloutRunner, save_trajectory


WORD_PATH = _PROJECT / "games" / "decrypto_words.txt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Decrypto with semantic belief model")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--output-dir", default="results/decrypto/decrypto_belief")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview prompts without calling LLM")
    p.add_argument("--save-prompts", action="store_true",
                   help="Save each prompt/response to disk")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _dry_run_preview(game, methods, args) -> None:
    """Simulate one turn to show prompt structure for each role."""
    from base.types import Action

    state = game.reset()
    print(f"Keywords: {state.keywords}\n")

    # ── Turn 1: Alice, Bob, Eve (no belief data yet) ──
    for step_idx, (pid, label) in enumerate(
        [(0, "ALICE (Encoder)"), (1, "BOB (Decoder)"), (2, "EVE (Interceptor)")]
    ):
        obs = game.get_observation(state, pid)
        legal = game.get_legal_actions(state)
        prompt = methods[pid]._render_prompt(obs, legal)
        _print_prompt_section(label, prompt, pid)

        # Step the game to advance state
        if pid == 0:
            action = Action(player_id=0, type="hint",
                          data={"code": [1, 3, 4], "hints": ["water_hint", "air_hint", "ice_hint"]})
        elif pid == 1:
            action = Action(player_id=1, type="guess", data={"code": [1, 3, 4]})
        else:
            action = Action(player_id=2, type="guess", data={"code": [1, 3, 4]})
        state = game.step(state, action)

    # ── Turn 2: Eve now has belief data from Turn 1 ──
    state = game.step(state, Action(player_id=0, type="hint",
        data={"code": [4, 2, 1], "hints": ["skating", "talk", "drink"]}))
    state = game.step(state, Action(player_id=1, type="guess", data={"code": [4, 2, 1]}))
    obs = game.get_observation(state, 2)
    legal = game.get_legal_actions(state)
    prompt = methods[2]._render_prompt(obs, legal)
    _print_prompt_section("EVE Turn 2 (with belief data)", prompt, 2)

    if args.save_prompts:
        prompts_dir = Path(args.output_dir) / "dry_run_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for pid, label in [(0, "alice"), (1, "bob"), (2, "eve")]:
            state2 = game.reset()
            if pid > 0:
                state2 = game.step(state2, Action(player_id=0, type="hint",
                    data={"code": [1,3,4], "hints": ["x","y","z"]}))
            if pid > 1:
                state2 = game.step(state2, Action(player_id=1, type="guess", data={"code": [1,3,4]}))
            obs = game.get_observation(state2, pid)
            legal = game.get_legal_actions(state2)
            prompt = methods[pid]._render_prompt(obs, legal)
            (prompts_dir / f"{label}_turn1_prompt.txt").write_text(prompt)
        print(f"\nPrompts saved to: {prompts_dir}")


def _print_prompt_section(label: str, prompt: str, player_id: int) -> None:
    """Print a condensed view of the prompt: sections and length."""
    sections = [l for l in prompt.split("\n") if l.startswith("####")]
    print(f"=== {label} ===  ({len(prompt)} chars)")
    for s in sections:
        print(f"  {s.strip()}")
    # Show first lines of reasoning steps
    move_start = prompt.find("#### YOUR MOVE")
    if move_start > 0:
        step_lines = [
            l for l in prompt[move_start:].split("\n")
            if l.strip().startswith("Step")
        ][:4]
        for s in step_lines:
            print(f"    {s.strip()}")
    has_belief = "KEYWORD CANDIDATES" in prompt
    print(f"  Belief section: {'YES' if has_belief else 'no (no hint_history yet)'}")
    print()


def main() -> None:
    args = build_parser().parse_args()

    # ── Load wordlist ──
    with open(WORD_PATH) as f:
        wordlist = [w.strip() for w in f.readlines()]
    print(f"Loaded {len(wordlist)} words")

    # ── Build embedding model ──
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_fn = make_embed_fn("all-MiniLM-L6-v2")
    print("Embedding model ready")

    # ── Build belief state for Eve ──
    belief_state = DecryptoBeliefState(wordlist, embed_fn, tau=0.15)

    # ── Build methods ──
    alice = DecryptoMethod(name="Alice", lang="en")
    bob = DecryptoMethod(name="Bob", lang="en")
    eve = DecryptoMethod(name="Eve (belief)", lang="en", belief_state=belief_state)

    methods = {0: alice, 1: bob, 2: eve}

    game = Decrypto(seed=args.seed, max_turns=args.max_turns)

    if args.dry_run:
        _dry_run_preview(game, methods, args)
        return

    from apis.openai_wrapper import OpenAIModel
    model = OpenAIModel(
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
    )

    # ── Run ──
    runner = RolloutRunner(game=game, methods=methods, model=model, verbose=args.verbose)
    trajectory = runner.run()

    # ── Print results ──
    print(f"\nGame over after {len(trajectory.steps)} steps")
    state = game.reset()  # just to access keywords
    print(f"Keywords: {state.keywords}")
    for pid, reward in trajectory.payout.items():
        name = methods[pid].name
        print(f"  {name} (P{pid}): {reward:+.0f}")
    if trajectory.winner is not None:
        print(f"  Winner: {methods[trajectory.winner].name}")

    # ── Print belief state at end ──
    print("\n--- Eve's final keyword beliefs ---")
    for pos in (1, 2, 3, 4):
        kb = belief_state.beliefs[pos]
        top = kb.top_k(3)
        print(f"Position {pos} (hints: {kb.hints_seen}): entropy={kb.entropy():.2f}")
        for word, prob in top:
            print(f"  {word:<16s} {prob:.1%}")

    # ── Save trajectory ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"trajectory_{ts}.json"
    save_trajectory(trajectory, str(traj_path))
    print(f"\nTrajectory saved to: {traj_path}")
    print(f"(each step includes 'rendered_prompt' and 'raw_llm_response')")

    # ── Save prompts separately if requested ──
    if args.save_prompts:
        prompts_dir = out_dir / f"prompts_{ts}"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for step in trajectory.steps:
            role = {0: "alice", 1: "bob", 2: "eve"}[step.player_id]
            step_dir = prompts_dir / role
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / f"step{step.step:02d}_prompt.txt").write_text(
                step.rendered_prompt)
            (step_dir / f"step{step.step:02d}_response.txt").write_text(
                step.raw_llm_response)
        print(f"Prompts saved to: {prompts_dir}")


if __name__ == "__main__":
    main()
