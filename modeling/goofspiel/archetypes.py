"""Goofspiel persona archetype definitions.

Each persona is an LLM-driven ArchetypeAgent with behavior prompts
injected into the LLM prompt. Used as opponents in ToM+CFR experiments.
When model=None (dry-run/testing), falls back to rule-based behavior.
"""

from __future__ import annotations

import random
from typing import Any

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
from methods.baseline.simple_prompt import SimplePromptMethod


# ── common helpers ──────────────────────────────────────────────────────

def _remaining_prizes(obs: Observation) -> list[int]:
    """Remaining prize sequence from current round onward."""
    seq = obs.public_info.get("prize_sequence", [])
    rnd = obs.public_info.get("round", 1)
    return list(seq[rnd - 1:])


def _prize_is_top_tier(obs: Observation) -> bool:
    """True if current prize is in top third of remaining prizes."""
    remaining = _remaining_prizes(obs)
    if not remaining:
        return True
    prize = obs.public_info.get("prize", 1)
    sorted_rem = sorted(remaining, reverse=True)
    cutoff = sorted_rem[len(sorted_rem) // 3]
    return prize >= cutoff


def _pick_max(actions: list[Action]) -> Action:
    return max(actions, key=lambda a: a.amount)


def _pick_min(actions: list[Action]) -> Action:
    return min(actions, key=lambda a: a.amount)


def _pick_closest_to(actions: list[Action], target: int) -> Action:
    return min(actions, key=lambda a: abs(a.amount - target))


# ── Persona definitions ─────────────────────────────────────────────────
# Each entry: (display_name, behavior_prompt)

PERSONAS: dict[str, tuple[str, str]] = {
    "Greedy": (
        "Greedy",
        "You are playing Goofspiel and you are driven by pure greed. "
        "You don't care about small prizes — they're beneath you. "
        "But when a really valuable prize is on the table, you want it "
        "and you want it now. You reach for your biggest card and "
        "slam it down, determined to crush anyone who gets in your way. "
        "It doesn't matter if you waste high cards on overkill — "
        "what matters is sending a message. "
        "Small prizes get the minimum effort. Big prizes get everything "
        "you've got. You play to dominate.\n",
    ),
    "Conservative": (
        "Conservative",
        "You are playing Goofspiel and you believe in getting value. "
        "You never pay more than something is worth. "
        "Each round you look at the prize and ask yourself: "
        "\"What's the smallest card I can spend to have a shot at this?\" "
        "You are happy to win a prize for less than it's worth — "
        "that feels like profit. Winning is nice, but winning at a "
        "discount is what gets you excited. "
        "You bid just no more than the prize value, looking for bargains. "
        "Overpaying feels like losing. You're not afraid to walk away "
        "from an overpriced prize.\n",
    ),
    "Sacrificial": (
        "Sacrificial",
        "You are playing Goofspiel with a measured strategy: "
        "sacrifice the biggest prize to win everything else. "
        "Each round you look at the current prize and compare it "
        "to what's remaining. "
        "If this prize is the largest remaining — you deliberately "
        "bid low, just below the prize value, so your opponent "
        "has to spend a real card to take it. You lose the prize "
        "but drain their hand. "
        "For every other prize — any prize that isn't the largest — "
        "you bid just enough to secure it, matching the prize value. "
        "No wasteful overkill, no reckless max cards. "
        "Let them overpay for the crown jewel. You take the rest "
        "efficiently.\n",
    ),
    "Random": (
        "Random",
        "",  # no prompt needed — uses random action directly
    ),
    "GTO": (
        "GTO",
        "",  # uses pre-trained CFR strategy, not LLM
    ),
    "Mirror": (
        "Mirror",
        "",  # rule-based only, no LLM needed
    ),
}

PERSONA_NAMES: list[str] = list(PERSONAS.keys())

PERSONA_DESCRIPTIONS_TEXT = """
1. Greedy: Fights hard for big prizes, folds for small ones.
   Top-third prizes → max card. Others → min card.

2. Conservative: Seeks discounts. Targets prize-1, never
   spends more than the prize is worth.

3. Sacrificial: Sacrifices the largest remaining prize (bids prize-2),
   bids the prize value for all others.

4. Random: Picks cards uniformly at random. No structure or pattern.
5. GTO: Plays a pre-trained Nash equilibrium (CFR) strategy.
6. Mirror: Always bids exactly the prize value. Prize-driven fixed point.
""".strip()

PERSONA_STYLE_NOTES: dict[str, str] = {
    "Greedy": (
        "Prize-driven binary behaviour. Top-third prizes → max card; "
        "others → min card. Detectable by win distribution across prize values."
    ),
    "Conservative": (
        "Discount-seeker. Targets prize-1; picks closest available card. "
        "Detectable by opponent's bid ≤ prize-1 pattern."
    ),
    "Sacrificial": (
        "Prize-driven conditional behaviour. Sacrifices the largest remaining prize "
        "(bids prize-2), bids prize value for all others. "
        "Detectable by bid pattern: prize-2 only on max prize, prize otherwise."
    ),
    "Random": (
        "No stable pattern. Uniform random card selection each round. "
        "Maximum entropy behaviour."
    ),
    "GTO": (
        "Pre-computed Nash equilibrium from MCCFR. "
        "Balanced play across all infosets. Not driven by LLM."
    ),
    "Mirror": (
        "Bids exactly the prize value every round. "
        "Pure prize-driven determinism. "
        "Detectable by bid == prize pattern."
    ),
}


# ── Archetype Agent (LLM-based) ─────────────────────────────────────────

class GoofspielArchetypeAgent(SimplePromptMethod):
    """SimplePromptMethod with persona behavior prefix injected into prompt.

    Used as LLM-driven opponent in experiments.
    When model=None (dry-run/testing), falls back to rule-based behavior.
    """

    def __init__(self, persona_name: str, name: str = "", seed: int = 0):
        display_name, prompt = PERSONAS[persona_name]
        super().__init__(name=name or display_name)
        self._persona_prefix = prompt
        self._persona_name = persona_name
        self._rng = random.Random(seed)

    def _render_prompt(self, obs, legal_actions):
        prompt = super()._render_prompt(obs, legal_actions)
        marker = "\nLegal actions:"
        if marker in prompt:
            idx = prompt.index(marker)
            return prompt[:idx] + "\n--- Your Playing Style ---\n" + self._persona_prefix + prompt[idx:]
        return prompt

    def act(self, observation, legal_actions, model=None):
        # Random and Mirror — no LLM call needed
        if self._persona_name in ("Random", "Mirror"):
            action = self._rng.choice(legal_actions) if self._persona_name == "Random" \
                else _pick_closest_to(legal_actions, observation.public_info.get("prize", 1))
            return action, "persona_fallback", f"Persona fallback: {self.name}"
        if model is None:
            return self._fallback_act(observation, legal_actions)
        return super().act(observation, legal_actions, model)

    def _fallback_act(self, observation, legal_actions):
        """Rule-based fallback when no LLM model is available."""
        pid = self._persona_name
        if pid == "Greedy":
            if _prize_is_top_tier(observation):
                action = _pick_max(legal_actions)
            else:
                action = _pick_min(legal_actions)
        elif pid == "Conservative":
            prize = observation.public_info.get("prize", 1)
            target = max(1, prize - 1)
            action = _pick_closest_to(legal_actions, target)
        elif pid == "Sacrificial":
            prize = observation.public_info.get("prize", 1)
            remaining = _remaining_prizes(observation)
            is_largest = prize >= max(remaining) if remaining else True
            target = max(1, prize - 2) if is_largest else prize
            action = _pick_closest_to(legal_actions, target)
        elif pid == "Mirror":
            prize = observation.public_info.get("prize", 1)
            action = _pick_closest_to(legal_actions, prize)
        else:  # Random (shouldn't reach here)
            action = self._rng.choice(legal_actions)
        return action, "persona_fallback", f"Persona fallback: {self.name}"

    def reflect(self, trajectory, model, game_name=""):
        return []


# ── Backward-compatible alias ────────────────────────────────────────────

GoofspielPersonaAgent = GoofspielArchetypeAgent
