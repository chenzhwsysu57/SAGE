"""Opponent modeling method interface for ToMMethod.

ToMMethod holds a BaseModelingMethod and calls update() / statement().
Each game (Leduc, Werewolf, Decrypto) provides its own subclasses.
"""

from __future__ import annotations

from base.types import Trajectory
from base.interfaces import BaseModel


class BaseModelingMethod:
    """Pluggable opponent modeling algorithm.

    All historical data is passed via update(). Each subclass decides
    internally how to process, store, and query it.

    Default implementations are no-ops so this class can be used as a
    dummy/default modeling method in experiments.
    """

    def update(
        self,
        opp_id: int,
        trajectory: Trajectory,
        model: BaseModel | None = None,
    ) -> None:
        """Process a completed game. May call LLM via model, or not."""
        pass

    def statement(self, opp_id: int, **kwargs) -> str:
        """Human-readable description of this method, target, and current effect.

        Used for both logging and LLM prompt injection.
        Returns '' if insufficient data.
        """
        return ""

    def exploit_hint(self, opp_id: int) -> str:
        """Optional: exploitation advice based on current model."""
        return ""

    def reset(self, opp_id: int) -> None:
        """Clear accumulated data for this opponent."""
        pass
