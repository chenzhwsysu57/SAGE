"""Suspicion-Agent methods for imperfect-information games."""

from .suspicion_prompt import SuspicionPromptMethod
from .suspicion_cross_game import CrossGameSuspicionMethod

__all__ = ["SuspicionPromptMethod", "CrossGameSuspicionMethod"]