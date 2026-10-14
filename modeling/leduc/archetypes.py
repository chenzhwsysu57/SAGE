"""Leduc Hold'em archetype definitions.

Central location for:
- Archetype descriptions (for LLM classification)
- Behavior prompts (for ArchetypeAgent opponents)
- ArchetypeAgent class (for experiments)
"""

from methods.baseline.simple_prompt import SimplePromptMethod

# ── Archetype definitions ──
# Each entry: (display_name, behavior_prompt_prefix)
# Used by ArchetypeAgent to create fixed-strategy opponents.

ARCHETYPES = {
    "LAG": (
        "Loose-Aggressive",
        "You are role-playing as a LAG (Loose-Aggressive). "
        "You play loose and aggressive poker — bet and raise with a wide "
        "range. Bluff frequently. Pressure opponents constantly. "
        "Never slow-play; always push the action.\n\n",
    ),
    "TAG": (
        "Tight-Aggressive",
        "You are role-playing as a TAG (Tight-Aggressive). "
        "You only play strong cards (Kings and Queens), but "
        "bet/raise hard when you do. Fold Jacks quickly. "
        "Discipline is your strength.\n\n",
    ),
    "CallingStation": (
        "Calling Station",
        "You are role-playing as a Calling Station. "
        "You love to call and rarely fold. When facing any bet, you call. "
        "You want to see showdowns. Folding means missing something. "
        "Never raise or bluff — just call.\n\n",
    ),
    "Nit": (
        "Nit",
        "You are role-playing as a Nit (extreme tight player). "
        "You are extremely cautious. Only play Kings. Fold everything else. "
        "Never bluff. Protect chips at all costs. "
        "Patience is your only weapon.\n\n",
    ),
    "Maniac": (
        "Maniac",
        "You are role-playing as a Maniac — a reckless, hyper-aggressive player. "
        "You ALWAYS bet or raise — NEVER check, NEVER call. Maximum "
        "aggression with any hand. Intimidate opponents into folding. "
        "If you can bet, you bet. If you can raise, you raise.\n\n",
    ),
    "GTO": (
        "GTO-Balanced",
        "You are role-playing as a GTO (Game Theory Optimal) player. "
        "You play balanced poker. Mix up your actions — sometimes bet, "
        "sometimes check with the same hand. Be unpredictable. "
        "Balance your range to remain unexploitable.\n\n",
    ),
}

ARCHETYPE_NAMES: list[str] = list(ARCHETYPES.keys())

ARCHETYPE_DESCRIPTIONS_TEXT = """
1. LAG (Loose-Aggressive): Bets and raises with a wide range of hands.
   Bluffs frequently. Pressures opponents constantly. High aggression even
   with weak cards like Jack-high.

2. TAG (Tight-Aggressive): Only plays strong cards (Kings and Queens).
   Bets/raises hard when strong, folds quickly when weak (Jacks).

3. CallingStation: Loves to call. Rarely folds. When facing any bet,
   calls regardless of hand strength. Never raises — just checks or calls.
   Low aggression, very low fold rate.

4. Nit: Extremely cautious. Only plays Kings. Folds everything else
   (Jacks and Queens). Never bluffs. Very low aggression, very high fold rate
   when holding anything but Kings.

5. Maniac: Always bets or raises — never checks, never calls. Maximum
   aggression with any hand. Tries to intimidate opponents into folding.

6. GTO-Balanced: Plays balanced poker. Mixes up actions — sometimes bets,
   sometimes checks with the same hand. Unpredictable. Moderate aggression.
""".strip()

# ── Archetype style notes ──────────────────────────────────────────────
# Purely descriptive — how each archetype typically plays.
# Used in ToM opponent analysis prompts. No strategy advice, no "you should".

ARCHETYPE_STYLE_NOTES: dict[str, str] = {
    "LAG": (
        "Plays a wide range of hands aggressively. "
        "Bets and raises frequently, even with weak holdings. "
        "Puts constant pressure on opponents. Occasionally traps with strong hands."
    ),
    "TAG": (
        "Plays selectively: only strong cards (Kings, Queens). "
        "Bets and raises hard with good hands. Folds weak hands quickly. "
        "Disciplined about pot selection."
    ),
    "CallingStation": (
        "Prefers to check and call. Rarely bets or raises. "
        "Seldom folds, even with weak hands. "
        "Wants to see showdowns. Passive across all streets."
    ),
    "Nit": (
        "Extremely cautious. Only plays premium hands (Kings). "
        "Folds everything else. Checks when unsure, folds to pressure. "
        "Very low aggression, very high fold rate when not holding the nuts."
    ),
    "Maniac": (
        "Always bets or raises — never checks, never calls. "
        "Maximum aggression with any hand, regardless of strength. "
        "Tries to intimidate opponents into folding."
    ),
    "GTO": (
        "Plays balanced and unpredictable. Mixes bets with checks, "
        "value bets with bluffs at optimal frequencies. "
        "Does not have easily exploitable patterns."
    ),
}


class ArchetypeAgent(SimplePromptMethod):
    """SimplePromptMethod with archetype behavior prefix injected into prompt.

    Used in experiments to create fixed-strategy opponents.
    """

    def __init__(self, archetype_name: str, name: str = ""):
        display_name, prompt = ARCHETYPES[archetype_name]
        super().__init__(name=name or display_name)
        self._archetype_prefix = prompt

    def _render_prompt(self, obs, legal_actions):
        prompt = super()._render_prompt(obs, legal_actions)
        # Inject archetype style instruction right before legal actions,
        # so the model sees "never check" immediately alongside the "check" option.
        marker = "\nLegal actions:"
        if marker in prompt:
            idx = prompt.index(marker)
            return prompt[:idx] + "\n--- Your Playing Style ---\n" + self._archetype_prefix + prompt[idx:]
        return prompt

    def reflect(self, trajectory, model, game_name=""):
        return []  # archetypes don't learn


# ── CFR-based GTO agent ──────────────────────────────────────────────────

class CFRGTOAgent:
    """GTO agent backed by a pre-computed CFR+ Nash equilibrium strategy.

    Loads strategy from disk (no LLM calls, no re-training needed after first
    training run). Falls back to on-the-fly training only if the saved file
    is missing.
    """

    _DEFAULT_PATH = "results/leduc/cfr_model/cfr_plus_21000.json"

    def __init__(self, name: str = "GTO-CFR+",
                 strategy_path: str = "",
                 seed: int = 0):
        import json as _json
        import os as _os
        import random as _random

        self.name = name
        self.method_type = "cfr"
        self._rng = _random.Random(seed)

        path = strategy_path or self._DEFAULT_PATH

        if _os.path.exists(path):
            with open(path) as f:
                data = _json.load(f)
            self._strategy = data["strategy"]
        else:
            # Train on the fly (one-time, then save to disk for future use)
            from methods.sage.cfr import CFRPlusTrainer
            print(f"CFR+ model not found at {path}, training from scratch...")
            trainer = CFRPlusTrainer(iterations=100000)
            trainer.train()
            self._strategy = {k: dict(v) for k, v in trainer.strategy.items()}

    def act(self, observation, legal_actions, model=None):
        """Look up CFR strategy for current info set, sample an action."""
        from methods.sage.cfr import (
            encode_history_for_cfr,
            sample_action_from_probs,
            strategy_probs_for_legal_actions,
        )

        card = observation.private_info.get("card", "??")
        rank = card[0] if len(card) > 0 else "?"
        comm_val = observation.public_info.get("community_card", "")
        comm_rank = comm_val[0] if comm_val else "?"  # CFR uses "?" for no community
        street_str = observation.public_info.get("street", "preflop")
        street = 0 if street_str == "preflop" else 1

        cfr_hist = encode_history_for_cfr(observation.history, street)
        info = f"{rank}|{comm_rank}|{cfr_hist}|{street}"
        lookup_info = info
        if not self._strategy.get(lookup_info, {}):
            # CFR trainer uses deterministic community card (deck order), but
            # the real game deals randomly. Fallback: try other community
            # ranks for the same hand+history+street pattern.
            for fallback_comm in ("J", "Q", "K", "?"):
                if fallback_comm == comm_rank:
                    continue
                fb_key = f"{rank}|{fallback_comm}|{cfr_hist}|{street}"
                if self._strategy.get(fb_key, {}):
                    lookup_info = fb_key
                    break

        action_probs, used_fallback = strategy_probs_for_legal_actions(
            self._strategy, lookup_info, legal_actions,
        )
        action = sample_action_from_probs(legal_actions, action_probs, self._rng)
        fallback_note = " fallback=uniform" if used_fallback else ""
        return action, f"CFR-GTO: {lookup_info}{fallback_note}", action.type

    def reflect(self, trajectory, model=None, game_name=""):
        return []  # CFR doesn't learn from experience
