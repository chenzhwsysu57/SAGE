"""Local first-visit Monte Carlo values, not the LLM Strategist optimizer.

Each exact, player-visible information set stores a P0 terminal-return mean.
Only real decision roots receive labels. Hypothetical MCTS states do not.
Unseen states have order-independent seeded initialization, without insertion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from methods.strategist.adapters import GAMES, PAYOFF_BOUNDS, PAYOFF_SCALES, _json, _seed


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_key(features: dict) -> str:
    """Exact observation key; no seed, omniscient state, text, or search data."""
    if (type(features) is not dict or set(features) != {
            "game", "player_id", "private_info", "public_info", "history"}
            or features["game"] not in GAMES
            or type(features["player_id"]) is not int or features["player_id"] not in (0, 1)):
        raise ValueError("Expected exact Strategist visible-feature schema")
    return canonical(_json(features))


@dataclass(frozen=True)
class Entry:
    count: int
    total: float

    @property
    def value(self) -> float:
        return self.total / self.count


class FrozenValueTable:
    """Immutable policy parameters. evaluate() never creates or updates entries."""

    def __init__(self, game: str, seed: int = 9100, initial_scale: float = 0.05,
                 entries: dict[str, Entry] | None = None):
        if game not in GAMES:
            raise ValueError("Unsupported game")
        _seed(seed)
        bound = PAYOFF_BOUNDS[game] / PAYOFF_SCALES[game]
        if (type(initial_scale) not in (int, float) or not math.isfinite(initial_scale)
                or not 0 <= initial_scale <= bound):
            raise ValueError("Invalid initialization scale")
        checked = dict(entries or {})
        for key, entry in checked.items():
            features = json.loads(key)
            if state_key(features) != key or features["game"] != game:
                raise ValueError("Invalid information-set key")
            if (not isinstance(entry, Entry) or type(entry.count) is not int or entry.count < 1
                    or type(entry.total) not in (int, float) or not math.isfinite(entry.total)
                    or abs(entry.value) > bound + 1e-12):
                raise ValueError("Invalid value-table entry")
        self._game, self._seed, self._initial_scale = game, seed, float(initial_scale)
        self._entries = MappingProxyType(checked)

    @property
    def game(self):
        return self._game

    @property
    def entries(self):
        return self._entries

    def initial_value(self, key: str) -> float:
        # 53-bit unit interval: reproducible across processes and lookup order.
        integer = int(digest(["tabular_initial_v1", self._seed, self.game, key])[:16], 16) >> 11
        return (2 * integer / (2**53 - 1) - 1) * self._initial_scale

    def evaluate(self, features: dict) -> tuple[tuple[float, float], dict]:
        key = state_key(features)
        if features["game"] != self.game:
            raise ValueError("Value table belongs to another game")
        entry = self.entries.get(key)
        value = entry.value if entry else self.initial_value(key)
        return (value, -value), {
            "source": "mc_table" if entry else "seeded_initial_value",
            "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "count": entry.count if entry else 0,
        }

    def payload(self) -> dict:
        return {
            "schema": "strategist.tabular_value.v1", "game": self.game,
            "seed": self._seed, "initial_scale": self._initial_scale,
            "perspective": "P0", "scale": PAYOFF_SCALES[self.game],
            "update_rule": "cumulative_first_visit_terminal_return_no_pseudocount",
            "entries": {key: {"count": e.count, "total": e.total, "value": e.value}
                        for key, e in sorted(self.entries.items())},
        }

    @property
    def fingerprint(self) -> str:
        return digest(self.payload())

    @classmethod
    def from_payload(cls, payload: dict) -> FrozenValueTable:
        expected = {"schema", "game", "seed", "initial_scale", "perspective", "scale",
                    "update_rule", "entries"}
        if set(payload) != expected or payload["schema"] != "strategist.tabular_value.v1":
            raise ValueError("Unsupported table schema")
        entries = {}
        for key, row in payload["entries"].items():
            if set(row) != {"count", "total", "value"}:
                raise ValueError("Invalid stored entry")
            entries[key] = Entry(row["count"], row["total"])
        table = cls(payload["game"], payload["seed"], payload["initial_scale"], entries)
        if canonical(table.payload()) != canonical(payload):
            raise ValueError("Table metadata/mean mismatch")
        return table

    def updated(self, episodes: list[dict]) -> FrozenValueTable:
        """Return a NEW table; one terminal label per distinct real key/episode.

        All games in a batch use the old table. The random initializer has no
        pseudo-count. Historical samples remain in the cumulative mean even
        though the self-play search policy changes across batches.
        """
        entries = dict(self.entries)
        bound = PAYOFF_BOUNDS[self.game] / PAYOFF_SCALES[self.game]
        for episode in episodes:
            values = episode["normalized_outcome"]
            if (len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                                      or abs(v) > bound + 1e-12 for v in values)
                    or abs(sum(values)) > 1e-10):
                raise ValueError("Invalid terminal target")
            seen = set()
            for step in episode["steps"]:
                features = step["features"]
                key = state_key(features)
                if features["game"] != self.game:
                    raise ValueError("Mixed game training labels")
                if key in seen:
                    continue
                seen.add(key)
                old = entries.get(key, Entry(0, 0.0))
                entries[key] = Entry(old.count + 1, old.total + values[0])
        return FrozenValueTable(self.game, self._seed, self._initial_scale, entries)