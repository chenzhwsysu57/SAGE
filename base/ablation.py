from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AblationConfig:
    """Controls removal of specific ToM-CFR components for ablation experiments.

    All flags default to False (all features enabled).
    """
    no_nash: bool = False
    no_belief: bool = False

    _FLAG_MAP: ClassVar[dict[str, str]] = {
        "nash": "no_nash",
        "belief": "no_belief",
    }

    @classmethod
    def from_comma_separated(cls, raw: str) -> AblationConfig:
        """Parse 'nash,belief' into AblationConfig(no_nash=True, no_belief=True)."""
        if not raw or not raw.strip():
            return cls()
        flags = [f.strip() for f in raw.split(",") if f.strip()]
        invalid = [f for f in flags if f not in cls._FLAG_MAP]
        if invalid:
            raise ValueError(
                f"Unknown ablation flag(s): {invalid}. "
                f"Valid: {list(cls._FLAG_MAP.keys())}"
            )
        kwargs = {cls._FLAG_MAP[f]: True for f in flags}
        return cls(**kwargs)
