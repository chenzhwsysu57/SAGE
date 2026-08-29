"""Parametric action-frequency opponent for Leduc OOD experiments.

Theta (theta) parameterizes the opponent's conditional action distribution in
the same coarse feature space that ``StatisticalProfileMethod`` classifies on:

    strength in {weak, medium, strong} x context in {no_bet, facing_bet}

``no_bet`` actions are {check, bet}; ``facing_bet`` actions are
{fold, call, raise}.  Action amounts are taken from the legal Action objects,
so the bot is fully playable in ``LeducHoldem``.

Two controllers drive theta across games:

- ``ParametricDriftController``: theta stays at A for ``drift_pre_stable``
  games, interpolates linearly to B over ``drift_ramp`` games, then stays at B
  for ``drift_post_stable`` games (soft style switch with stable windows).
- ``CounterModelingController``: reads the method's style posterior every game
  (the "cheating" information channel, provided by the harness) and, once the
  argmax confidence stays above a threshold for a consecutive window, switches
  to one of the two farthest styles (farthest with prob ``1-second_prob``,
  second-farthest with prob ``second_prob``).  Switching is continuous: the new
  theta applies from the next game onward, and the window restarts so the
  cheater can switch again once the method re-models the new style.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from base.types import Action, Observation

STRENGTHS: tuple[str, ...] = ("weak", "medium", "strong")
NO_BET_ACTIONS: tuple[str, ...] = ("check", "bet")
FACING_BET_ACTIONS: tuple[str, ...] = ("fold", "call", "raise")
CONTEXTS: tuple[str, ...] = ("no_bet", "facing_bet")
MAX_ENTROPY: float = math.log2(5.0)  # matches StatisticalProfileMethod

# theta shape: {strength: {context: {action: probability}}}
Theta = dict[str, dict[str, dict[str, float]]]


def make_theta() -> Theta:
    """Uniform theta table (default before any estimation/controller)."""
    return {
        strength: {
            "no_bet": {a: 1.0 / len(NO_BET_ACTIONS) for a in NO_BET_ACTIONS},
            "facing_bet": {a: 1.0 / len(FACING_BET_ACTIONS) for a in FACING_BET_ACTIONS},
        }
        for strength in STRENGTHS
    }


def _hand_strength(card: str) -> str | None:
    if not card:
        return None
    rank = str(card)[0].upper()
    if rank == "J":
        return "weak"
    if rank == "Q":
        return "medium"
    if rank == "K":
        return "strong"
    return None


def _context_of_action(action_type: str) -> str | None:
    if action_type in NO_BET_ACTIONS:
        return "no_bet"
    if action_type in FACING_BET_ACTIONS:
        return "facing_bet"
    return None


def _normalize(counts: dict[str, float], actions: tuple[str, ...],
               smoothing: float) -> dict[str, float]:
    total = sum(counts.get(a, 0.0) for a in actions)
    denom = total + smoothing * len(actions)
    if denom <= 0:
        return {a: 1.0 / len(actions) for a in actions}
    return {
        a: (counts.get(a, 0.0) + smoothing) / denom
        for a in actions
    }


def _new_counts() -> dict[str, dict[str, dict[str, float]]]:
    return {s: {c: {} for c in CONTEXTS} for s in STRENGTHS}


def _add_step_counts(counts: dict[str, dict[str, dict[str, float]]],
                     step: Any, opp_id: int) -> None:
    pid = step.get("player_id", -1) if isinstance(step, dict) else getattr(step, "player_id", -1)
    if pid != opp_id:
        return
    action_type = _action_type_from_step(step)
    context = _context_of_action(action_type)
    strength = _strength_from_step(step, opp_id)
    if context is None or strength is None:
        return
    counts[strength][context][action_type] = (
        counts[strength][context].get(action_type, 0.0) + 1.0
    )


def _counts_to_theta(counts: dict[str, dict[str, dict[str, float]]],
                     smoothing: float) -> Theta:
    theta: Theta = {}
    for strength in STRENGTHS:
        theta[strength] = {
            "no_bet": _normalize(counts[strength]["no_bet"], NO_BET_ACTIONS, smoothing),
            "facing_bet": _normalize(counts[strength]["facing_bet"], FACING_BET_ACTIONS, smoothing),
        }
    return theta


def interpolate_theta(theta_a: Theta, theta_b: Theta, lam: float) -> Theta:
    """Linear interpolation in probability space, renormalized per context."""
    lam = min(1.0, max(0.0, float(lam)))
    out: Theta = {}
    for strength in STRENGTHS:
        out[strength] = {}
        for context, actions in (
            ("no_bet", NO_BET_ACTIONS),
            ("facing_bet", FACING_BET_ACTIONS),
        ):
            counts = {}
            for action in actions:
                p_a = theta_a.get(strength, {}).get(context, {}).get(action, 0.0)
                p_b = theta_b.get(strength, {}).get(context, {}).get(action, 0.0)
                counts[action] = (1.0 - lam) * p_a + lam * p_b
            total = sum(counts.values())
            if total <= 0:
                out[strength][context] = {
                    a: 1.0 / len(actions) for a in actions
                }
            else:
                out[strength][context] = {
                    a: counts[a] / total for a in actions
                }
    return out


def _action_type_from_step(step: Any) -> str:
    """Extract normalized action type from a dict or pydantic trajectory step."""
    if isinstance(step, dict):
        parsed = step.get("parsed_action")
        if isinstance(parsed, dict) and parsed.get("type"):
            return str(parsed["type"]).lower()
        raw = str(step.get("action", ""))
        m = re.search(r"P\d+\s+(\w+)", raw)
        if not m:
            return ""
        raw_type = m.group(1).lower()
    else:
        parsed = getattr(step, "parsed_action", None)
        if parsed is not None and getattr(parsed, "type", None):
            return str(parsed.type).lower()
        return ""
    return {
        "raises": "raise", "bets": "bet", "calls": "call",
        "folds": "fold", "checks": "check",
    }.get(raw_type, raw_type)


def _strength_from_step(step: Any, opp_id: int) -> str | None:
    if isinstance(step, dict):
        cards = step.get("private_cards")
        if not isinstance(cards, dict):
            cards = (step.get("state_before") or {}).get("private_cards", {})
        return _hand_strength(str(cards.get(str(opp_id), "")))
    state_before = getattr(step, "state_before", None) or {}
    cards = state_before.get("private_cards", {})
    return _hand_strength(str(cards.get(str(opp_id), "")))


def estimate_theta_from_steps(steps: list[Any], opp_id: int,
                              smoothing: float = 0.25) -> Theta:
    """Estimate a theta table from labeled trajectory steps of ``opp_id``.

    Actions whose private card is unavailable are skipped (the profile's
    selectivity feature has the same behavior).
    """
    counts = _new_counts()
    for step in steps:
        _add_step_counts(counts, step, opp_id)
    return _counts_to_theta(counts, smoothing)


def _label_player(label: str, archetype: str) -> int | None:
    """Mirror of leduc_tom_ablation._archetype_player_from_label."""
    if not label:
        return None
    if f"_vs_{archetype}_" in label:
        return 1
    if label.startswith(f"{archetype}_vs_"):
        return 0
    return None


def estimate_theta_tables(manifest_path: str,
                          max_games_per_archetype: int = 0,
                          smoothing: float = 0.25) -> tuple[dict[str, Theta], dict]:
    """Estimate one theta table per archetype from the signature manifest.

    Mirrors ``build_empirical_signature_bundle`` manifest traversal (same
    label convention, merged P0/P1) so drift endpoints / cheater targets live
    in the same data space as the signatures.
    """
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest.get("entries", {})
    theta_tables: dict[str, Theta] = {}
    metadata: dict[str, Any] = {"source": "historical_manifest", "archetypes": {}}

    for archetype, item in entries.items():
        source_files = []
        for key in ("train_json", "test_json"):
            path = item.get(key)
            if path and path not in source_files:
                source_files.append(path)
        counts = _new_counts()
        meta = {"source_files": list(source_files), "games": 0, "actions": 0}
        for source_path in source_files:
            with open(source_path) as f:
                data = json.load(f)
            for batch in data.get("batches", []):
                label = str(batch.get("label", ""))
                player_id = _label_player(label, archetype)
                if player_id is None:
                    continue
                trajectories = batch.get("sample_trajectories", [])
                if max_games_per_archetype > 0:
                    remaining = max_games_per_archetype - meta["games"]
                    if remaining <= 0:
                        break
                    trajectories = trajectories[:remaining]
                for traj in trajectories:
                    steps = traj.get("steps", []) if isinstance(traj, dict) else getattr(traj, "steps", [])
                    if not steps:
                        continue
                    game_actions = 0
                    for step in steps:
                        pid = step.get("player_id", -1) if isinstance(step, dict) else getattr(step, "player_id", -1)
                        if pid == player_id and _action_type_from_step(step):
                            game_actions += 1
                    if game_actions == 0:
                        continue
                    for step in steps:
                        _add_step_counts(counts, step, player_id)
                    meta["games"] += 1
                    meta["actions"] += game_actions
        if meta["games"] == 0:
            raise ValueError(
                f"No usable P0/P1 historical actions for {archetype} in {source_files}"
            )
        theta_tables[archetype] = _counts_to_theta(counts, smoothing)
        metadata["archetypes"][archetype] = meta
    return theta_tables, metadata


class ParametricStyleAgent:
    """Fixed-theta opponent that samples actions from the theta table."""

    def __init__(self, theta: Theta | None = None, name: str = "Parametric",
                 seed: int = 0):
        self.theta = theta or make_theta()
        self.name = name
        self.method_type = "parametric"
        self.theta_label = ""
        self._rng = random.Random(seed)

    def set_theta(self, theta: Theta) -> None:
        self.theta = theta

    def act(self, observation: Observation, legal_actions: list[Action],
            model=None):
        context = (
            "no_bet"
            if any(a.type == "check" for a in legal_actions)
            else "facing_bet"
        )
        strength = _hand_strength(str(observation.private_info.get("card", ""))) or "medium"
        legal_by_type = {a.type: a for a in legal_actions}
        probs = self.theta.get(strength, {}).get(context, {})
        candidates = [t for t in NO_BET_ACTIONS + FACING_BET_ACTIONS
                      if t in legal_by_type and float(probs.get(t, 0.0)) > 0.0]
        if not candidates:
            candidates = [a.type for a in legal_actions]
        weights = [max(float(probs.get(t, 0.0)), 0.0) for t in candidates]
        if sum(weights) <= 0:
            weights = [1.0] * len(candidates)
        chosen = self._rng.choices(candidates, weights=weights, k=1)[0]
        action = legal_by_type[chosen]
        md: dict[str, Any] = {
            "theta_label": self.theta_label,
            "context": context,
            "strength": strength,
            "theta": {
                s: {
                    c: {a: round(float(p), 4) for a, p in d.items()}
                    for c, d in ctxs.items()
                }
                for s, ctxs in self.theta.items()
            },
        }
        return action, "", "", md

    def reflect(self, trajectory, model, game_name: str = ""):
        return []


class ParametricDriftController:
    """Soft switch with stable windows.

    lambda(g) = 0 for g <= drift_pre_stable,
                1 for g > drift_pre_stable + drift_ramp,
                and ramps linearly in between.  After the ramp the opponent
                stays at style B for drift_post_stable games (or until the run
                ends, whichever comes first).
    """

    def __init__(self, theta_a: Theta, theta_b: Theta,
                 drift_ramp: int, drift_pre_stable: int = 0,
                 drift_post_stable: int = 0):
        if drift_ramp < 1:
            raise ValueError("drift_ramp must be >= 1")
        if drift_pre_stable < 0 or drift_post_stable < 0:
            raise ValueError("drift_pre_stable / drift_post_stable must be >= 0")
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.drift_ramp = int(drift_ramp)
        self.drift_pre_stable = int(drift_pre_stable)
        self.drift_post_stable = int(drift_post_stable)
        self.log: list[dict[str, Any]] = []

    def lambda_for_game(self, game_index: int) -> float:
        g = int(game_index)
        if g <= self.drift_pre_stable:
            return 0.0
        if g <= self.drift_pre_stable + self.drift_ramp:
            return (g - self.drift_pre_stable) / self.drift_ramp
        return 1.0

    def theta_for_game(self, game_index: int) -> Theta:
        lam = self.lambda_for_game(game_index)
        self.log.append({"game": game_index, "lambda": round(lam, 4)})
        return interpolate_theta(self.theta_a, self.theta_b, lam)

    def label_for_game(self, game_index: int) -> str:
        return f"lambda={self.lambda_for_game(game_index):.4f}"


def signature_feature_vector(sig: dict[str, float]) -> list[float]:
    """8-dim signature vector in the same space as _profile_likelihood."""
    return (
        [float(sig.get(f"no_bet_{a}", 0.0)) for a in NO_BET_ACTIONS]
        + [float(sig.get(f"fb_{a}", 0.0)) for a in FACING_BET_ACTIONS]
        + [min(1.0, max(0.0, float(sig.get("selectivity", 0.0))))]
        + [min(1.0, max(0.0, float(sig.get("entropy", 0.0)) / MAX_ENTROPY))]
    )


def farthest_two_style_map(signatures: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """For each style, the [farthest, second-farthest] styles by signature distance."""
    styles = list(signatures)
    vectors = {s: signature_feature_vector(signatures[s]) for s in styles}
    out: dict[str, list[str]] = {}
    for s in styles:
        dists = []
        for t in styles:
            if t == s:
                continue
            dists.append((t, math.sqrt(
                sum((a - b) ** 2 for a, b in zip(vectors[s], vectors[t]))
            )))
        dists.sort(key=lambda item: -item[1])
        out[s] = [name for name, _ in dists[:2]]
    return out


def farthest_style_map(signatures: dict[str, dict[str, float]]) -> dict[str, str]:
    """For each style, the style with the largest signature distance."""
    return {s: two[0] for s, two in farthest_two_style_map(signatures).items()}


class CounterModelingController:
    """Cheater: reads the method's style posterior and switches to one of the
    two farthest styles once confidence >= threshold for a consecutive window.
    Continuous: the window restarts after each switch."""

    def __init__(self, theta_by_style: dict[str, Theta],
                 signatures: dict[str, dict[str, float]],
                 threshold: float = 0.5, min_consecutive: int = 3,
                 second_prob: float = 0.5, seed: int = 0):
        self.theta_by_style = theta_by_style
        self.threshold = float(threshold)
        self.min_consecutive = max(1, int(min_consecutive))
        self.second_prob = min(1.0, max(0.0, float(second_prob)))
        self.farthest_two = farthest_two_style_map(signatures)
        self._rng = random.Random(seed)
        self._style: str | None = None
        self._consecutive = 0
        self._switched_at_game: int | None = None
        self.log: list[dict[str, Any]] = []

    def set_start_style(self, style: str) -> None:
        self._style = style

    @property
    def current_style(self) -> str | None:
        return self._style

    def observe_belief(self, belief: dict[str, float] | None,
                       game_index: int) -> None:
        if belief:
            top = max(belief, key=lambda k: float(belief[k]))
            conf = float(belief[top])
        else:
            top, conf = None, 0.0

        switched_now = False
        if top is None or conf < self.threshold:
            self._consecutive = 0
        else:
            self._consecutive += 1
            if self._consecutive >= self.min_consecutive:
                self._consecutive = 0
                target = self._pick_target(top)
                if target and target != self._style:
                    self._style = target
                    self._switched_at_game = game_index + 1
                    switched_now = True

        self.log.append({
            "game": game_index,
            "belief": belief,
            "style": self._style,
            "switched": switched_now,
            "switched_at_game": self._switched_at_game,
            "candidates": self.farthest_two.get(top, []) if top else [],
        })

    def theta_for_game(self, game_index: int) -> Theta:
        return self.theta_by_style.get(self._style) or make_theta()

    def label_for_game(self, game_index: int) -> str:
        label = f"style={self._style}"
        if self._switched_at_game and game_index >= self._switched_at_game:
            label += " (switched)"
        return label

    def _pick_target(self, modeled_style: str) -> str | None:
        candidates = self.farthest_two.get(modeled_style) or []
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # second-farthest with probability second_prob, farthest otherwise
        idx = 1 if self._rng.random() < self.second_prob else 0
        if candidates[idx] == self._style:
            idx = 1 - idx
        return candidates[idx]
