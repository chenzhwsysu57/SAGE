"""Liar's Dice archetype definitions.

Central location for:
- Archetype descriptions (for LLM classification)
- Behavior prompts (for ArchetypeAgent opponents)
- ArchetypeAgent class (for experiments)
"""

from methods.baseline.simple_prompt import SimplePromptMethod

# ── Archetype definitions ──────────────────────────────────────────────
# Each entry: (display_name, behavior_prompt_prefix)
# Used by ArchetypeAgent to create fixed-strategy opponents.

ARCHETYPES = {
    "Bluffer": (
        "Bluffer",
        "You are role-playing as a Bluffer in Liar's Dice. You are bold, slippery, "
        "and confident at the table. You like to keep pressure on the opponent and "
        "make your bids sound stronger than your actual hand. You prefer to look "
        "aggressive and hard to pin down, even when the current situation is not "
        "fully supported by what you hold. Your style is to keep the bidding alive, "
        "stay one step ahead in tempo, and make your table presence feel forceful "
        "and somewhat overcommitted.\n\n",
    ),
    "Honest": (
        "Honest",
        "You are role-playing as an Honest player in Liar's Dice. You always bid "
        "based on what your own dice show — never bluff, never inflate a count. "
        "Look at your actual dice including wild-1s and bid the maximum honest "
        "quantity you personally hold for a given face.\n\n"
        "When it's your turn to raise, consider: can you truthfully make a higher "
        "bid? If yes, bid honestly. If no — meaning every legal raise would be a "
        "bluff — then you challenge instead. You never resort to bluffing; you "
        "challenge when pushed past your honest limit. You do not challenge "
        "proactively or suspiciously — only when the numbers leave you no honest "
        "option.\n\n"
        "Your table persona is: transparent, predictable, and backed-into-a-corner "
        "rather than confrontational. You bid what you have; when that's no longer "
        "possible, you call it out.\n\n",
    ),
    "AggressiveChallenger": (
        "Aggressive Challenger",
        "You are role-playing as an Aggressive Challenger in Liar's Dice. You are "
        "tense, suspicious, and quick to pressure the other player. You do not give "
        "bids much room to breathe, and you are comfortable forcing a showdown when "
        "the bidding starts to feel shaky. You like to make the table prove itself. "
        "Your style is confrontational and probing: you want the other side to feel "
        "watched, tested, and slightly uncomfortable. You may still bid, but your "
        "overall presence is sharp, impatient, and ready to challenge the moment the "
        "round starts to smell weak.\n\n",
    ),
    "ConservativeChallenger": (
        "Conservative Challenger",
        "You are role-playing as a Conservative Challenger in Liar's Dice. You are "
        "patient, careful, and reluctant to escalate. You prefer to keep the round "
        "moving rather than end it with a challenge, and you do not like making a "
        "scene unless the bidding has clearly become extreme. Your style is restrained "
        "and cautious: you tend to stay out of trouble, make modest bids, and avoid "
        "unnecessary confrontation. You are not passive, but you are slow to accuse. "
        "You come across as someone who watches first, reacts later, and only becomes "
        "forceful when the situation has become obviously too far gone.\n\n",
    ),
    "Random": (
        "Random",
        "You are role-playing as a Random player in Liar's Dice. You do not have a "
        "stable pattern, a consistent rhythm, or a predictable preference. Sometimes "
        "you sound bold, sometimes cautious, sometimes suspicious, and sometimes "
        "indifferent. Your choices are not meant to be easy to read or easy to "
        "summarize. You are meant to feel erratic, unstable, and hard to classify, "
        "as if your decisions are driven by no clear long-term style. The important "
        "thing is not consistency, but unpredictability.\n\n",
    ),
    "GTO": (
        "GTO-Balanced",
        "You are role-playing as a balanced, hard-to-read player in Liar's Dice. "
        "You do not lean too heavily into one visible habit, and you avoid becoming "
        "obvious or easy to exploit. Your style is controlled, adaptive, and difficult "
        "to pin down. Sometimes you look aggressive, sometimes cautious, sometimes "
        "willing to challenge, sometimes willing to continue the bidding. You are "
        "trying to feel like a player whose behavior does not collapse into a single "
        "simple pattern. The goal is not to be flashy or extreme, but to stay varied, "
        "natural, and difficult to predict.\n\n",
    ),
}

ARCHETYPE_NAMES: list[str] = list(ARCHETYPES.keys())

ARCHETYPE_DESCRIPTIONS_TEXT = """
1. Bluffer: Habitual liar. Constantly overbids — claims more dice of a face
   than they actually have. Rarely challenges. Prefers raising even when
   suspicious of a bluff.

2. Honest: Always bids truthfully based on actual dice. Rarely exaggerates.
   Will challenge when a bid is clearly impossible.

3. Aggressive Challenger: Challenges early and often. Frequently calls
   "Liar!" instead of raising. Suspicious by nature. Bids are moderately
   aggressive.

4. Conservative Challenger: Almost never challenges. Prefers raising over
   challenging. Only challenges outrageously impossible bids. Patient.

5. Random: Picks actions uniformly at random. No strategy or pattern.
   Unpredictable by definition.

6. GTO-Balanced: Plays balanced Liar's Dice. Mixes bluffing, honest bidding,
   challenging, and raising at optimal frequencies. No exploitable pattern.
""".strip()

# ── Archetype style notes ──────────────────────────────────────────────

ARCHETYPE_STYLE_NOTES: dict[str, str] = {
    "Bluffer": (
        "Bold and slippery; prefers aggressive pressure and overcommitted bids. "
        "Keeps tempo high and makes the table react. Low caution, high confidence."
    ),
    "Honest": (
        "Calm, sincere, and grounded. Prefers believable bids and restrained pushback. "
        "Anchors decisions in what the hand can reasonably support."
    ),
    "AggressiveChallenger": (
        "Tense, suspicious, and confrontational. Likes to pressure the table and force "
        "showdowns when bidding starts to feel shaky."
    ),
    "ConservativeChallenger": (
        "Patient, careful, and reluctant to escalate. Stays out of trouble, keeps the "
        "round moving, and only challenges when things look extreme."
    ),
    "Random": (
        "No stable pattern or consistent rhythm. Hard to summarize, hard to read, "
        "and deliberately unstable across situations."
    ),
    "GTO": (
        "Controlled, adaptive, and difficult to pin down. Keeps behavior varied without "
        "collapsing into a single obvious pattern."
    ),
}


class ArchetypeAgent(SimplePromptMethod):
    """SimplePromptMethod with archetype behavior prefix injected into prompt.

    Random uses direct random.choice(legal_actions) — no LLM call.
    Other archetypes use LLM with behavior prompt injection.

    Used in experiments to create fixed-strategy opponents.
    """

    def __init__(self, archetype_name: str, name: str = "", seed: int = 0):
        self._is_random = archetype_name == "Random"
        if self._is_random:
            self._rng = __import__("random").Random(seed)
            display_name = ARCHETYPES[archetype_name][0]
            super().__init__(name=name or display_name)
            self._archetype_prefix = ""
        else:
            display_name, prompt = ARCHETYPES[archetype_name]
            super().__init__(name=name or display_name)
            self._archetype_prefix = prompt

    def act(self, observation, legal_actions, model):
        if self._is_random:
            idx = self._rng.randint(0, len(legal_actions) - 1)
            action = legal_actions[idx]
            return action, f"Random choice: {action.type}", action.type
        action, prompt, raw = super().act(observation, legal_actions, model)
        # Safety net: if the parsed action is illegal, fall back to closest match.
        from base.liars_dice_core import match_legal_action
        matched = match_legal_action(action, legal_actions)
        if matched is None:
            matched = self._fallback_action(action, legal_actions)
        return matched, prompt, raw

    @staticmethod
    def _fallback_action(action, legal_actions):
        """Fallback: same-type action, or first legal if nothing matches."""
        same_type = [a for a in legal_actions if a.type == action.type]
        if same_type:
            return same_type[0]
        return legal_actions[0]

    def _render_prompt(self, obs, legal_actions):
        if self._is_random:
            return super()._render_prompt(obs, legal_actions)
        prompt = super()._render_prompt(obs, legal_actions)
        marker = "\n#### YOUR MOVE ####\n"
        if marker in prompt:
            idx = prompt.index(marker)
            rule_block = "\n#### YOUR ROLE ####\n\n" + self._archetype_prefix.rstrip() + "\n"
            return prompt[:idx] + rule_block + prompt[idx:]
        return prompt

    def reflect(self, trajectory, model, game_name=""):
        return []  # archetypes don't learn
