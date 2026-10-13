"""Bounded two-player, observation-keyed PUCT with vector-value backups.

One transposition table per decide(), shared across ALL determinizations; its
cumulative node count never exceeds max_nodes. Each simulation resamples hidden
information and future chance. All selections maximize the acting seat's value,
not a hero-only minimax backup. Uniform first-visit exploration avoids legal-list
order bias, then Q + c P sqrt(N)/(1+n) drives selection (P is uniform).

This is an information-set MCTS adaptation, not equilibrium inference or exact
paper Appendix F. Uniform hidden priors ignore action likelihood. Sharing nodes
prevents direct hidden-state-conditioned policies but does not remove all
determinization/non-locality or imperfect-information search biases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random

from base.types import Action, Observation
from methods.strategist.adapters import GAMES, World, _json, _seed, sample_world, visible
from methods.strategist.value import InvalidHeuristic, ValueHeuristic


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 32
    max_depth: int = 12
    max_nodes: int = 512
    exploration: float = 1.4

    def __post_init__(self) -> None:
        for name, upper in (("simulations", 4096), ("max_depth", 64), ("max_nodes", 16384)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1, {upper}]")
        if (type(self.exploration) not in (int, float)
            or not 0 < self.exploration <= 100 or not math.isfinite(self.exploration)):
            raise ValueError("exploration must be finite in (0, 100]")


@dataclass
class SearchDecision:
    action: Action
    trace: dict


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> int:
    return int.from_bytes(hashlib.sha256(_canonical(value).encode()).digest()[:16], "big")


def action_key(action: Action) -> str:
    return _canonical(action.model_dump())


def information_key(world: World) -> str:
    """No simulator seed, deck, hidden dice, pending bid, or removed card.

    Sorting legal action objects also makes a key insensitive to enumeration
    order. In Goofspiel, the second selector shares this key across ALL first
    bids, including when root hero is P1 (engine resolution remains P0 then P1).
    """
    if world.terminal:
        raise ValueError("Terminal worlds have no decision key")
    return _canonical({"features": world.features(world.actor),
                       "legal": sorted(action_key(a) for a in world.legal())})


@dataclass
class _Edge:
    action: Action
    prior: float
    visits: int = 0
    sums: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def mean(self) -> list[float]:
        return [v / self.visits for v in self.sums] if self.visits else [0.0, 0.0]


@dataclass
class _Node:
    actor: int
    edges: list[_Edge]
    visits: int = 0


class StrategistSearch:
    def __init__(self, game: str, heuristic: ValueHeuristic,
                 config: SearchConfig, seed: int = 0):
        if game not in GAMES:
            raise ValueError("Unsupported Strategist game")
        if not isinstance(config, SearchConfig):
            raise TypeError("Expected SearchConfig")
        config.__post_init__()
        _seed(seed)
        if not callable(getattr(heuristic, "evaluate", None)):
            raise TypeError("Heuristic must expose evaluate(features)")
        self.game, self.heuristic, self.config, self.seed = game, heuristic, config, seed

    def _evaluate(self, features: dict) -> tuple[tuple[float, float], dict]:
        # Explicit failure: never substitute zero, random policy, or old values.
        try:
            values, explanation = self.heuristic.evaluate(features)
            if (type(values) not in (tuple, list) or len(values) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in values)
                    or abs(sum(values)) > 1e-8 or type(explanation) is not dict):
                raise InvalidHeuristic("Invalid heuristic value vector or explanation")
            return (float(values[0]), float(values[1])), _json(explanation)
        except Exception as exc:
            raise InvalidHeuristic("Strategist heuristic evaluation failed") from exc

    def _node(self, world: World) -> _Node:
        actions = sorted(world.legal(), key=action_key)
        if not actions:
            raise ValueError("Nonterminal decision has no legal actions")
        return _Node(world.actor, [_Edge(a, 1.0 / len(actions)) for a in actions])

    def _select(self, node: _Node, rng: random.Random) -> _Edge:
        unvisited = [edge for edge in node.edges if not edge.visits]
        if unvisited:
            return rng.choice(unvisited)
        scores = [edge.mean()[node.actor] + self.config.exploration * edge.prior
                  * math.sqrt(node.visits) / (1 + edge.visits) for edge in node.edges]
        best = max(scores)
        return rng.choice([edge for edge, score in zip(node.edges, scores) if score == best])

    def decide(self, obs: Observation) -> SearchDecision:
        # This seed is a planner seed derived ONLY from visible structured input.
        # It is not the real environment seed. No persistent RNG/table can make
        # repeated calls depend on earlier decisions or unrelated simulations.
        if (not isinstance(obs, Observation) or not 1 <= len(obs.legal_actions) <= 13
            or any(type(token) is not str or len(token) > 32 for token in obs.legal_actions)):
            raise ValueError("Expected bounded nonempty legal action labels")
        decision_seed = _hash([self.seed, self.game, visible(obs), sorted(obs.legal_actions)])
        root = sample_world(self.game, obs, _hash([decision_seed, "root"]))
        features = root.features(root.hero)
        initial, explanation = self._evaluate(features)
        root_key = information_key(root)
        table = {root_key: self._node(root)}
        transitions = backups = heuristic_leaves = terminal_leaves = capacity_leaves = 0
        deepest = 0
        simulation_depths = []
        for simulation in range(self.config.simulations):
            world = sample_world(self.game, obs, _hash([decision_seed, "world", simulation]))
            path: list[tuple[_Node, _Edge]] = []
            depth = 0
            while True:
                if world.terminal:
                    values = world.terminal_values()
                    terminal_leaves += 1
                    break
                if depth >= self.config.max_depth:
                    values, _ = self._evaluate(world.features(world.actor))
                    heuristic_leaves += 1
                    break
                key = information_key(world)
                node = table.get(key)
                if node is None:
                    if len(table) < self.config.max_nodes:
                        table[key] = self._node(world)
                    else:
                        capacity_leaves += 1
                    values, _ = self._evaluate(world.features(world.actor))
                    heuristic_leaves += 1
                    break
                # Node-local randomness cannot condition a simultaneous second
                # choice on how many random draws a first bid happened to use.
                rng = random.Random(_hash([decision_seed, "select", simulation, depth, key]))
                edge = self._select(node, rng)
                path.append((node, edge))
                world = world.advance(edge.action)
                depth += 1
                transitions += 1
            for node, edge in path:
                node.visits += 1
                edge.visits += 1
                for player in (0, 1):
                    edge.sums[player] += values[player]
                backups += 1
            deepest = max(deepest, depth)
            simulation_depths.append(depth)
        root_node = table[root_key]
        best_visits = max(e.visits for e in root_node.edges)
        candidates = [e for e in root_node.edges if e.visits == best_visits]
        best_mean = max(e.mean()[root.hero] for e in candidates)
        candidates = [e for e in candidates if e.mean()[root.hero] == best_mean]
        chosen = random.Random(_hash([decision_seed, "final"])).choice(candidates)
        # Return the exact canonical engine action for the INPUT information set,
        # not a hypothetical descendant action or an amount parsed from text.
        action = next(a for a in root.legal() if a == chosen.action)
        root_value = [sum(e.sums[p] for e in root_node.edges) / root_node.visits for p in (0, 1)]
        trace = {
            "root_features": features,
            "root_initial_heuristic": {"values": list(initial), "explanation": explanation},
            "root_search_value": root_value,
            "actions": [{"action": e.action.model_dump(), "visits": e.visits,
                         "mean_values": e.mean(), "prior": e.prior} for e in root_node.edges],
            "nodes": len(table), "transitions": transitions,
            "simulations": self.config.simulations, "backups": backups,
            "max_depth_reached": deepest, "simulation_depths": simulation_depths,
            "heuristic_leaves": heuristic_leaves, "terminal_leaves": terminal_leaves,
            "capacity_leaves": capacity_leaves, "seed": decision_seed,
            "caveats": [
                "Uniform hidden prior ignores action likelihood; not a posterior or equilibrium solver.",
                "Observation-keyed PUCT adaptation, not exact paper Appendix F; determinization biases remain.",
                "Both seats maximize their own component; all backups retain P0/P1 values.",
                "Goofspiel uses precommit paired bids and public full sequence; second key excludes first bid.",
                "max_nodes is cumulative per decision; tables reset between decisions.",
                "Depth/transitions count selected actions, including unresolved simultaneous commitments.",
                "Leduc exact net bound is 19; conservative scale 32. Dice scale 1; Goofspiel scale 15.",
                "Seed is observation-derived planner randomness, never the real environment seed.",
            ],
        }
        # Keep even oversized (but bounded-config) traces JSON, without applying
        # the smaller heuristic-input item budget to per-simulation diagnostics.
        json.dumps(trace, allow_nan=False)
        return SearchDecision(action, trace)