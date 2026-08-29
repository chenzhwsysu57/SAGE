"""Locked-theme state for Theme-Loyalist Interceptor.

LockedThemeState: on first hint for a keyword, calls LLM to form a locked theme.
Once locked, never updated. Injects locked themes into the Interceptor's prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from base.interfaces import BaseModel
    from base.types import Observation


class LockedThemeState:
    """Program-level theme locking for a Theme-Loyalist Interceptor.

    After each round, checks if any keyword has received its first hint.
    If so, calls the LLM to form a one-sentence theme, then stores it.
    Once all 4 keywords have locked themes, stops calling the LLM.

    Usage::

        state = LockedThemeState()
        # in _render_prompt:
        state.update_from_observation(obs, model)
        prompt_section = state.themes  # injected into prompt
    """

    def __init__(self):
        self._locked: dict[int, str] = {}  # keyword_num → theme description
        self._all_locked = False
        self._events: list[dict] = []
        self._last_update_events: list[dict] = []

    @property
    def themes(self) -> dict[int, str]:
        return dict(self._locked)

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    @property
    def last_update_events(self) -> list[dict]:
        return list(self._last_update_events)

    def update_from_observation(self, obs, model):
        """Check hint_history for first hints. Lock themes for new keywords."""
        self._last_update_events = []
        if self._all_locked:
            return []

        hint_history = obs.public_info.get("hint_history", {})
        new_locks = {}

        for pos in (1, 2, 3, 4):
            key = str(pos)
            if pos in self._locked:
                continue
            hints = hint_history.get(key, [])
            if not hints or len(hints) == 0:
                continue

            first_hint = hints[0]
            theme, prompt, raw = self._form_theme(first_hint, model)
            if theme:
                new_locks[pos] = theme
                self._last_update_events.append({
                    "turn": obs.public_info.get("turn"),
                    "keyword": pos,
                    "first_hint": first_hint,
                    "theme": theme,
                    "prompt": prompt,
                    "raw_response": raw,
                })

        self._locked.update(new_locks)
        self._events.extend(self._last_update_events)
        if len(self._locked) == 4:
            self._all_locked = True
        return self.last_update_events

    def _form_theme(self, first_hint: str, model) -> tuple[str | None, str, str]:
        """Call LLM to infer a single coherent theme from one hint word."""
        prompt = (
            f'A player gave the hint "{first_hint}" for a secret keyword.\n'
            "Based ONLY on this one hint, what is the SINGLE most likely theme "
            "of the keyword?\n\n"
            "Rules:\n"
            "- Answer in one short phrase (2-5 words).\n"
            "- Describe a SINGLE coherent idea. Do NOT list alternatives with "
            '"or", "/", or commas.\n'
            "- The theme should be a tight semantic category, not a broad umbrella.\n\n"
            'Good: "bodies of water"  (tight, singular)\n'
            'Bad:  "water, liquid, or ocean"  (list of alternatives)\n'
            'Good: "ice hockey sport"  (tight, singular)\n'
            'Bad:  "sports, competitions, or games"  (too broad)\n\n'
            "One short phrase, no explanation:"
        )
        raw = model.generate(prompt)
        theme = raw.strip().strip('"').strip("'")
        theme = theme.split("\n")[0].strip()
        # Reject if it contains "or", "/", or is too long
        if " or " in theme.lower() or " / " in theme or len(theme) > 60:
            theme = theme.replace(" or ", ", ").replace(" / ", ", ")
            if len(theme) > 60:
                theme = theme[:60]
        return (theme if theme else None), prompt, raw

    def to_dict(self) -> dict:
        """Serializable snapshot for trajectory method_data."""
        return {
            "themes": self.themes,
            "events": self.events,
            "all_locked": self._all_locked,
        }

    def to_prompt(self) -> str:
        """Format locked themes for injection into the Interceptor's prompt."""
        if not self._locked:
            return ""

        lines = ["\n#### YOUR LOCKED THEMES (never change) ####\n"]
        for pos in sorted(self._locked.keys()):
            lines.append(f"  Keyword {pos}: {self._locked[pos]}")
        lines.append("")
        return "\n".join(lines)
