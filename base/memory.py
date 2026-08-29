"""Cross-game memory for methods that learn across episodes."""

from __future__ import annotations


class BaseMemory:
    """Simple key-value memory that persists across games within one process.

    Methods use this to store and retrieve learned information:
    - ReflexionMethod stores lesson strings
    - ToMMethod stores opponent behavior profiles
    - Custom methods store arbitrary data

    Usage:
        memory = BaseMemory()
        memory.set("lessons", ["Don't fold with K", "Bluff more on flop"])
        lessons = memory.get("lessons", [])
    """

    def __init__(self):
        self._data: dict = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value

    def append(self, key: str, item) -> None:
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(item)

    def clear(self) -> None:
        self._data = {}

    def to_dict(self) -> dict:
        return dict(self._data)

    def from_dict(self, data: dict) -> None:
        self._data = dict(data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"BaseMemory({self._data})"
