"""Paper-inspired Strategist genealogy/idea queue, independently implemented.

This module contains no upstream source and no executable-code evaluation. The
scripted backend is a deliberately weak *mock*, not evidence of LLM learning.
Search values and outcomes always retain the fixed (P0, P1) coordinate system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random

from methods.strategist.value import ValueHeuristic


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value: object, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def seed_list(values: object, name: str) -> tuple[int, ...]:
    if type(values) not in (tuple, list) or not 1 <= len(values) <= 8:
        raise ValueError(f"{name} requires 1..8 seeds")
    for value in values:
        integer(value, 0, 2**31 - 1, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate seeds")
    return tuple(values)


def vector(values: object) -> list[float]:
    if (type(values) not in (tuple, list) or len(values) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in values)
            or abs(sum(values)) > 1e-8):
        raise ValueError("Expected normalized zero-sum P0/P1 vector")
    return [float(v) for v in values]


@dataclass
class Strategy:
    id: str
    source: str
    sha256: str
    parent_id: str | None = None
    idea_id: str | None = None
    score: float | None = None
    score_round: int | None = None
    dev_score: float | None = None
    status: str = "validated"

    @classmethod
    def create(cls, identity: str, source: str, parent_id=None, idea_id=None):
        heuristic = ValueHeuristic(source)
        return cls(identity, heuristic.source, heuristic.sha256, parent_id, idea_id)


@dataclass
class Idea:
    id: str
    parent_id: str
    text: str
    implementations: int = 0
    improvements: list[float] = field(default_factory=list)


class IdeaQueue:
    """Softmax UCB over average child-parent improvement.

    The exploration clock counts completed implementations globally, NEVER
    insertions. All logits/probabilities are recomputed before every selection.
    Failed implementations stop the run and do not masquerade as observations.
    """

    def __init__(self, temperature: float = 0.5, exploration: float = 1.0):
        for value in (temperature, exploration):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 10:
                raise ValueError("Queue parameters must be finite in (0,10]")
        self.temperature, self.exploration = temperature, exploration
        self.ideas: list[Idea] = []
        self.implementation_count = 0
        self.selections: list[dict] = []

    def add(self, idea: Idea) -> None:
        if (not isinstance(idea, Idea) or any(i.id == idea.id for i in self.ideas)
                or type(idea.text) is not str or not idea.text.strip() or len(idea.text) > 4000
                or idea.implementations != 0 or idea.improvements):
            raise ValueError("Invalid new idea")
        self.ideas.append(idea)

    def select(self, seed: int, eligible: set[str] | None = None) -> Idea:
        integer(seed, 0, 2**128 - 1, "selection seed")
        choices = [i for i in self.ideas if eligible is None or i.parent_id in eligible]
        if not choices:
            raise ValueError("No eligible ideas")
        scores = [(sum(i.improvements) / i.implementations if i.implementations else 0.0)
                  + self.exploration * math.sqrt(math.log(self.implementation_count + 2)
                                                / (i.implementations + 1)) for i in choices]
        top = max(scores)
        weights = [math.exp((s - top) / self.temperature) for s in scores]
        probabilities = [w / sum(weights) for w in weights]
        selected = random.Random(seed).choices(choices, weights=probabilities, k=1)[0]
        self.selections.append({"implementation_count": self.implementation_count,
                                "ideas": [i.id for i in choices], "ucb": scores,
                                "probabilities": probabilities, "selected": selected.id})
        return selected

    def observe(self, idea_id: str, child_score: float, parent_score: float) -> None:
        for value in (child_score, parent_score):
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1:
                raise ValueError("Invalid paired score")
        idea = next(i for i in self.ideas if i.id == idea_id)
        idea.improvements.append(child_score - parent_score)
        idea.implementations += 1
        self.implementation_count += 1

    def snapshot(self) -> dict:
        return {"implementation_count": self.implementation_count,
                "ideas": [asdict(i) for i in self.ideas], "selections": self.selections,
                "temperature": self.temperature, "exploration": self.exploration}


def key_feedback(games: list[dict], strategy_id: str, limit: int = 3) -> list[dict]:
    """Rank genuinely visited decision roots, not hypothetical descendants.

    State in reflection feedback is public-only. Private inputs and free-form
    heuristic explanations are intentionally not forwarded (they can echo a
    private hand). Keep a fixed explanatory reason, both aligned value vectors,
    the squared discrepancy and the eventual normalized outcome instead.
    """
    integer(limit, 1, 20, "feedback limit")
    rows = []
    for game in games:
        outcome = vector(game["normalized_outcome"])
        for index, step in enumerate(game["steps"]):
            if game["players"][step["player_id"]] != strategy_id or step["trace"] is None:
                continue
            trace = step["trace"]
            if sum(a["visits"] for a in trace["actions"]) <= 0:
                continue
            initial = vector(trace["root_initial_heuristic"]["values"])
            searched = vector(trace["root_search_value"])
            features = trace["root_features"]
            rows.append({"game_id": game["item_id"], "step": index,
                         "player_id": step["player_id"],
                         "state": {k: features[k] for k in ("game", "player_id", "public_info", "history")},
                         "heuristic_values": initial, "search_values": searched,
                         "squared_discrepancy": sum((a-b)**2 for a, b in zip(initial, searched)),
                         "normalized_outcome": outcome,
                         "reason": "Visited root: squared P0/P1 heuristic-search discrepancy; not causal error."})
    return sorted(rows, key=lambda r: (-r["squared_discrepancy"], r["game_id"], r["step"]))[:limit]


ZERO_SOURCE = 'def evaluate_state(state):\n    return (0, 0), {"mock": "initial zero"}\n'


class ScriptedModel:
    """Deterministic, weak mock generator. No API, tokens, training or weights.

    Responses depend on prompt fields, not an implicit cursor, so replaying
    cached calls cannot change subsequent mock responses. The small branch is
    a protocol fixture, deliberately NOT a handcrafted competitive strategy.
    """
    model_name = "strategist-scripted-mock-v1"
    provenance = {"kind": "scripted", "live_api": False, "real_learning": False}

    def __init__(self):
        self.calls = 0
        self.last_usage = {}
        self.last_finish_reason = None

    def generate(self, prompt: str) -> str:
        request = json.loads(prompt)
        self.calls += 1
        self.last_usage = {}
        self.last_finish_reason = "stop"
        stage = request["true_stage"]
        if stage == "seed_code":
            return ZERO_SOURCE
        if stage == "reflection":
            return "MOCK reflection: compare the public visited roots and aligned search values; no learning claim."
        if stage == "idea":
            return "MOCK idea: exercise a weak bounded game/seat/history conditional, not a competitive policy."
        if stage == "code":
            sign = 1 if request["iteration"] % 2 else -1
            return ('def evaluate_state(state):\n'
                    '    v = 0\n'
                    '    if state["game"] == "leduc":\n'
                    f'        v = {sign} * 0.02\n'
                    '    elif len(state["history"]) > 0:\n'
                    f'        v = {sign} * 0.01\n'
                    '    if state["player_id"] == 1:\n'
                    '        v = -v\n'
                    '    return (v, -v), {"mock": "weak branch fixture"}\n')
        raise ValueError("Unknown scripted stage")