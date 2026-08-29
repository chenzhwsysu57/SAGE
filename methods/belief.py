"""Generic opponent belief tracking.

DirichletBelief — Bayesian posterior over action types (purely statistical).
OpponentProfile — simple action frequency counter.

No game-specific knowledge here. Game-specific analysis (archetype
classification, exploitation advice) lives in modeling/{game}/.
"""

from __future__ import annotations

from collections import Counter


class OpponentProfile:
    """Tracks opponent action frequencies.

    Generic: just counts actions by type.
    Game-specific analysis (aggression, patterns) belongs in modeling methods.
    """

    def __init__(self):
        self.actions: Counter = Counter()

    def record(self, action_type: str) -> None:
        self.actions[action_type] += 1

    @property
    def total(self) -> int:
        return sum(self.actions.values())

    def summary(self) -> str:
        if self.total == 0:
            return ""
        items = sorted(self.actions.items(), key=lambda x: -x[1])
        parts = [f"{a}:{c}" for a, c in items]
        return f"{', '.join(parts)} (n={self.total})"

    def to_dict(self) -> dict:
        return dict(self.actions)

    @classmethod
    def from_dict(cls, data: dict) -> OpponentProfile:
        p = cls()
        p.actions = Counter(data)
        return p


class DirichletBelief:
    """Dirichlet prior/posterior over action types.

    Purely statistical — no game semantics.
    P(action | history) ∝ count(action) + alpha
    """

    def __init__(self, action_types: list[str], alpha: float = 1.0):
        self.counts: dict[str, float] = {a: alpha for a in action_types}

    def observe(self, action_type: str) -> None:
        """Update with one observed action."""
        self.counts[action_type] = self.counts.get(action_type, 0.0) + 1.0

    def predict(self, action_type: str) -> float:
        """Predictive probability P(next action == action_type)."""
        total = sum(self.counts.values())
        return self.counts.get(action_type, 0) / total if total > 0 else 0

    @property
    def n_observations(self) -> int:
        """Number of observations beyond the prior."""
        return sum(self.counts.values()) - len(self.counts)

    def to_text(self) -> str:
        """Render distribution as compact text for LLM prompts."""
        if self.n_observations == 0:
            return "(no data)"
        items = sorted(self.counts.items(), key=lambda x: -x[1])
        total = sum(self.counts.values())
        parts = [f"{a}: {c/total:.0%}" for a, c in items if c > 0]
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {"counts": self.counts}

    @classmethod
    def from_dict(cls, data: dict) -> DirichletBelief:
        b = cls([])
        b.counts = dict(data.get("counts", {}))
        return b
