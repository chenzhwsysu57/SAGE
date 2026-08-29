"""StatisticalProfileMethod: opponent modeling via Euclidean distance to archetype signatures.

Self-contained: includes profile construction, classification, and convergence plotting.
No LLM dependency — pure statistical matching in conditional action space.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from base.memory import BaseMemory
from base.types import Trajectory
from modeling.base import BaseModelingMethod
from modeling.leduc.archetypes import ARCHETYPE_NAMES


# ── Constants ───────────────────────────────────────────────────────────────

LEDUC_ACTION_TYPES = ["check", "call", "bet", "raise", "fold"]
AGGRESSIVE_ACTIONS = {"bet", "raise"}
NO_BET_ACTIONS = ["check", "bet"]
FACING_BET_ACTIONS = ["call", "raise", "fold"]


def _parse_action_type(action_str: str) -> str:
    """Parse and normalize action type from a string like 'P1 raises 4'."""
    import re
    m = re.search(r'P\d+\s+(\w+)', action_str)
    if not m:
        return ""
    raw = m.group(1).lower()
    # Normalize: "raises" → "raise", "bets" → "bet", "calls" → "call", etc.
    return {"raises": "raise", "bets": "bet", "calls": "call",
            "folds": "fold", "checks": "check"}.get(raw, raw)


def _hand_strength(card: str) -> str:
    if not card:
        return "unknown"
    rank = card[0]
    if rank == "K":
        return "strong"
    elif rank == "Q":
        return "medium"
    return "weak"


# ── Profile dataclass ───────────────────────────────────────────────────────

@dataclass
class OpponentBehaviorProfile:
    archetype_label: str = ""
    n_games: int = 0
    n_actions: int = 0
    action_counts: Counter = field(default_factory=Counter)
    no_bet_fractions: dict[str, float] = field(default_factory=dict)
    facing_bet_fractions: dict[str, float] = field(default_factory=dict)
    never_observed: list[str] = field(default_factory=list)
    selectivity: float = 0.0
    entropy: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Profile ({self.n_games} games, {self.n_actions} actions):",
        ]
        if self.archetype_label:
            lines.append(f"  Label: {self.archetype_label}")
        parts = [f"{a}={self.no_bet_fractions.get(a, 0) * 100:.0f}%"
                 for a in NO_BET_ACTIONS]
        lines.append("  No bet:  " + " ".join(parts))
        parts = [f"{a}={self.facing_bet_fractions.get(a, 0) * 100:.0f}%"
                 for a in FACING_BET_ACTIONS]
        lines.append("  Facing:  " + " ".join(parts))
        lines.append(f"  Selectivity: {self.selectivity:+.0%}")
        if self.never_observed:
            lines.append(f"  Never observed: {self.never_observed}")
        lines.append(f"  Entropy: {self.entropy:.2f}")
        return "\n".join(lines)


# ── Profile construction ────────────────────────────────────────────────────

def _build_profile(action_types: list[str], strength_labels: list[str],
                   label: str = "") -> OpponentBehaviorProfile:
    counts = Counter(action_types)
    total = len(action_types)

    no_bet_counts = Counter()
    facing_bet_counts = Counter()
    for at in action_types:
        if at in ("check", "bet"):
            no_bet_counts[at] += 1
        elif at in ("call", "raise", "fold"):
            facing_bet_counts[at] += 1
    nb_total = sum(no_bet_counts.values())
    fb_total = sum(facing_bet_counts.values())
    nb_f = {a: no_bet_counts.get(a, 0) / nb_total for a in NO_BET_ACTIONS} if nb_total else {}
    fb_f = {a: facing_bet_counts.get(a, 0) / fb_total for a in FACING_BET_ACTIONS} if fb_total else {}

    never = [a for a in LEDUC_ACTION_TYPES if counts.get(a, 0) == 0]

    strong_aggr = weak_aggr = strong_total = weak_total = 0
    for at, st in zip(action_types, strength_labels):
        if st == "strong":
            strong_total += 1
            if at in AGGRESSIVE_ACTIONS:
                strong_aggr += 1
        elif st == "weak":
            weak_total += 1
            if at in AGGRESSIVE_ACTIONS:
                weak_aggr += 1
    selectivity = ((strong_aggr / strong_total) if strong_total else 0) - \
                  ((weak_aggr / weak_total) if weak_total else 0)

    entropy = -sum(c / total * math.log2(c / total)
                   for c in counts.values() if c > 0) if total else 0.0

    return OpponentBehaviorProfile(
        archetype_label=label,
        n_actions=total,
        action_counts=counts,
        no_bet_fractions=nb_f,
        facing_bet_fractions=fb_f,
        never_observed=never,
        selectivity=selectivity,
        entropy=entropy,
    )


# ── Classification ──────────────────────────────────────────────────────────

def _classify_profile(profile: OpponentBehaviorProfile,
                      signatures: dict | None = None) -> dict[str, float]:
    return _profile_likelihood(profile, signatures)


def _profile_likelihood(profile: OpponentBehaviorProfile,
                        signatures: dict | None = None) -> dict[str, float]:
    if not signatures:
        raise ValueError(
            "signatures must be provided; use statistical estimation from historical profiles"
        )
    sigs = signatures
    MAX_ENTROPY = math.log2(len(LEDUC_ACTION_TYPES))  # ~2.32 for 5 actions
    p_vec = (
        [profile.no_bet_fractions.get(a, 0) for a in NO_BET_ACTIONS] +
        [profile.facing_bet_fractions.get(a, 0) for a in FACING_BET_ACTIONS] +
        [max(0.0, min(1.0, profile.selectivity))] +
        [max(0.0, min(1.0, profile.entropy / MAX_ENTROPY))]
    )
    scores: dict[str, float] = {}
    for name, sig in sigs.items():
        eliminated = False
        for a in profile.never_observed:
            if sig.get(a, 0) > 0.15:
                eliminated = True
                break
        if eliminated:
            scores[name] = 0.0
            continue
        sig_entropy = sig.get("entropy", 0.0)
        s_vec = (
            [sig.get(f"no_bet_{a}", 0) for a in NO_BET_ACTIONS] +
            [sig.get(f"fb_{a}", 0) for a in FACING_BET_ACTIONS] +
            [max(0.0, min(1.0, sig.get("selectivity", 0.0)))] +
            [max(0.0, min(1.0, sig_entropy / MAX_ENTROPY))]
        )
        dist = math.sqrt(sum((p_vec[i] - s_vec[i]) ** 2 for i in range(len(p_vec))))
        scores[name] = 1.0 / (1.0 + dist)
    total = sum(scores.values())
    if total > 0:
        return {k: v / total for k, v in scores.items()}
    n = len(scores)
    return {k: 1.0 / n for k in scores}


def _softmax_from_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    """Convert unnormalized log-scores into a normalized posterior."""
    if not log_scores:
        return {}
    max_log = max(log_scores.values())
    exp_scores = {k: math.exp(v - max_log) for k, v in log_scores.items()}
    total = sum(exp_scores.values())
    if total <= 0:
        n = len(exp_scores)
        return {k: 1.0 / n for k in exp_scores}
    return {k: v / total for k, v in exp_scores.items()}


def _dirichlet_multinomial_log_score(counts: Counter,
                                     alpha: dict[str, float],
                                     support: list[str]) -> float:
    """Log score under a categorical model with Dirichlet prior."""
    total = sum(counts.get(k, 0) for k in support)
    alpha_total = sum(alpha.get(k, 1.0) for k in support)
    score = math.lgamma(alpha_total) - math.lgamma(alpha_total + total)
    for k in support:
        a = alpha.get(k, 1.0)
        c = counts.get(k, 0)
        score += math.lgamma(a + c) - math.lgamma(a)
    return score


# ── Modeling method ─────────────────────────────────────────────────────────

class StatisticalProfileMethod(BaseModelingMethod):
    """Opponent modeling via Euclidean distance to archetype signatures.

    Accumulates all opponent actions across games, builds a conditional
    frequency profile, and classifies by nearest archetype signature.

    Parameters:
        memory: persistence backend
        plot_convergence: if True, track classification after each game
        chart_dir: where to save convergence plots
    """

    def __init__(self, memory: BaseMemory | None = None,
                 plot_convergence: bool = False,
                 true_archetype: str = "",
                 chart_dir: str = "results/leduc/pilot_tom/figures",
                 signatures: dict[str, dict[str, float]] | None = None,
                 signature_source: str = "data",
                 signature_metadata: dict[str, Any] | None = None,
                 bayes_mode: bool = False,
                 classification_mode: str = "q_window_ema",
                 q_window: int = 10,
                 q_alpha: float = 0.85,
                 q_tau: float = 1.0):
        self.memory = memory or BaseMemory()
        self.plot_convergence = plot_convergence
        self.true_archetype = true_archetype
        self.chart_dir = chart_dir
        self.signatures = signatures or {}
        self.signature_source = signature_source
        self.signature_metadata = signature_metadata or {
            "mode": "data",
            "source": "historical_actions",
            "archetypes": {},
        }
        self.bayes_mode = bayes_mode
        self.classification_mode = classification_mode
        self.q_window = max(1, int(q_window))
        self.q_alpha = max(0.0, min(1.0, float(q_alpha)))
        self.q_tau = float(q_tau)
        self._game_count: dict[int, int] = {}
        self._posterior_logps: dict[int, dict[str, float]] = {}
        self._posterior_cache: dict[int, dict[str, float]] = {}
        self._bayes_pseudo_count = 0.5

    # ── Persistence helpers ────────────────────────────────────────────

    def _actions_key(self, opp_id: int) -> str:
        return f"stat_actions_{opp_id}"

    def _strengths_key(self, opp_id: int) -> str:
        return f"stat_strengths_{opp_id}"

    def _game_actions_key(self, opp_id: int) -> str:
        return f"stat_game_actions_{opp_id}"

    def _game_strengths_key(self, opp_id: int) -> str:
        return f"stat_game_strengths_{opp_id}"

    def _convergence_key(self, opp_id: int) -> str:
        return f"stat_convergence_{opp_id}"

    def _load_actions(self, opp_id: int) -> list[str]:
        return self.memory.get(self._actions_key(opp_id), [])

    def _load_strengths(self, opp_id: int) -> list[str]:
        return self.memory.get(self._strengths_key(opp_id), [])

    def _save_actions(self, opp_id: int, data: list[str]) -> None:
        self.memory.set(self._actions_key(opp_id), data)

    def _save_strengths(self, opp_id: int, data: list[str]) -> None:
        self.memory.set(self._strengths_key(opp_id), data)

    def _load_game_actions(self, opp_id: int) -> list[list[str]]:
        data = self.memory.get(self._game_actions_key(opp_id), [])
        return [list(item) for item in data if isinstance(item, list)]

    def _load_game_strengths(self, opp_id: int) -> list[list[str]]:
        data = self.memory.get(self._game_strengths_key(opp_id), [])
        return [list(item) for item in data if isinstance(item, list)]

    def _save_game_actions(self, opp_id: int, data: list[list[str]]) -> None:
        self.memory.set(self._game_actions_key(opp_id), data)

    def _save_game_strengths(self, opp_id: int, data: list[list[str]]) -> None:
        self.memory.set(self._game_strengths_key(opp_id), data)

    def _initial_log_prior(self) -> dict[str, float]:
        if not self.signatures:
            return {}
        n = len(self.signatures)
        return {name: math.log(1.0 / n) for name in self.signatures}

    def _initial_posterior(self) -> dict[str, float]:
        if not self.signatures:
            return {}
        n = len(self.signatures)
        return {name: 1.0 / n for name in self.signatures}

    def _bayes_likelihood(self, actions: list[str],
                          strengths: list[str]) -> dict[str, float]:
        counts = Counter(actions)
        support = LEDUC_ACTION_TYPES
        log_scores = {}
        for name, sig in self.signatures.items():
            action_probs = sig.get("action_probs", {})
            concentration = float(sig.get("action_concentration", 20.0))
            alpha = {
                a: max(self._bayes_pseudo_count, concentration * action_probs.get(a, 1.0 / len(support)))
                for a in support
            }
            log_scores[name] = _dirichlet_multinomial_log_score(counts, alpha, support)
        return _softmax_from_log_scores(log_scores)

    def _posterior_from_current_profile(self, opp_id: int) -> dict[str, float]:
        """Return posterior for the current accumulated profile.

        If no cumulative state has been built yet, this falls back to a
        one-shot likelihood from the current profile.
        """
        if not self.signatures:
            raise ValueError("StatisticalProfileMethod requires data-driven signatures")
        actions = self._load_actions(opp_id)
        strengths = self._load_strengths(opp_id)
        if not actions:
            n = len(self.signatures)
            return {name: 1.0 / n for name in self.signatures}

        if self.classification_mode in {"q_window_ema", "q_window_log"}:
            cached = self._posterior_cache.get(opp_id)
            if cached:
                return cached
            window_actions, window_strengths = self._window_data(opp_id)
            return self._classify_actions(window_actions, window_strengths)

        if self.bayes_mode:
            return self._bayes_likelihood(actions, strengths)

        profile = _build_profile(actions, strengths)
        if opp_id not in self._posterior_logps:
            return _profile_likelihood(profile, self.signatures)

        return self._posterior_cache.get(opp_id) or _softmax_from_log_scores(
            self._posterior_logps[opp_id]
        )

    def _update_cumulative_posterior(self, opp_id: int,
                                     likelihood: dict[str, float]) -> dict[str, float]:
        """Update and return the cumulative posterior after the latest game."""
        if not self.signatures:
            raise ValueError("StatisticalProfileMethod requires data-driven signatures")

        if opp_id not in self._posterior_logps:
            self._posterior_logps[opp_id] = self._initial_log_prior()

        logps = self._posterior_logps[opp_id]
        eps = 1e-12
        for name, prob in likelihood.items():
            logps[name] = logps.get(name, math.log(1.0 / max(len(likelihood), 1))) + math.log(max(prob, eps))

        posterior = _softmax_from_log_scores(logps)
        self._posterior_cache[opp_id] = posterior
        return posterior

    def _window_data(self, opp_id: int) -> tuple[list[str], list[str]]:
        game_actions = self._load_game_actions(opp_id)
        game_strengths = self._load_game_strengths(opp_id)
        if not game_actions:
            return self._load_actions(opp_id), self._load_strengths(opp_id)
        selected_actions = game_actions[-self.q_window:]
        selected_strengths = game_strengths[-self.q_window:]
        actions = [a for game in selected_actions for a in game]
        strengths = [s for game in selected_strengths for s in game]
        return actions, strengths

    def _classify_actions(self, actions: list[str],
                          strengths: list[str]) -> dict[str, float]:
        if not actions:
            return self._initial_posterior()
        if self.bayes_mode:
            return self._bayes_likelihood(actions, strengths)
        profile = _build_profile(actions, strengths)
        return _profile_likelihood(profile, self.signatures)

    def _update_window_ema_posterior(self, opp_id: int,
                                     likelihood: dict[str, float]) -> dict[str, float]:
        prior = self._posterior_cache.get(opp_id) or self._initial_posterior()
        posterior = {
            name: self.q_alpha * prior.get(name, 0.0)
            + (1.0 - self.q_alpha) * likelihood.get(name, 0.0)
            for name in self.signatures
        }
        total = sum(posterior.values())
        if total > 0:
            posterior = {name: value / total for name, value in posterior.items()}
        self._posterior_cache[opp_id] = posterior
        return posterior

    def _update_window_log_posterior(self, opp_id: int,
                                     likelihood: dict[str, float]) -> dict[str, float]:
        if opp_id not in self._posterior_logps:
            self._posterior_logps[opp_id] = self._initial_log_prior()
        logps = self._posterior_logps[opp_id]
        eps = 1e-12
        for name, prob in likelihood.items():
            logps[name] = logps.get(name, math.log(1.0 / max(len(likelihood), 1))) + (
                self.q_tau * math.log(max(prob, eps))
            )
        posterior = _softmax_from_log_scores(logps)
        self._posterior_cache[opp_id] = posterior
        return posterior

    # ── Core: data ingestion (online + offline share this) ──────────────

    def _accumulate_game(self, opp_id: int,
                         trajectory: Trajectory | dict) -> int:
        """Extract opponent actions from trajectory and append to accumulated lists.
        Pure data accumulation — no classification. Shared by online and offline.
        Accepts both Pydantic Trajectory objects and raw dict trajectories.
        Returns number of new actions ingested.
        """
        steps = (trajectory.steps if hasattr(trajectory, 'steps')
                 else trajectory.get("steps", []))
        if not steps:
            return 0

        new_actions, new_strengths = [], []
        for s in steps:
            pid = s.player_id if hasattr(s, 'player_id') else s.get("player_id", -1)
            if pid != opp_id:
                continue
            # Handle both Pydantic Action objects and dict actions
            if hasattr(s, 'parsed_action') and s.parsed_action is not None:
                a_type = s.parsed_action.type
            else:
                action_str = s.get("action", "") if isinstance(s, dict) else ""
                a_type = _parse_action_type(action_str)
                if not a_type:
                    continue
            new_actions.append(a_type)
            # Hand strength from card
            card = ""
            if hasattr(s, 'state_before'):
                card = s.state_before.get("private_cards", {}).get(str(opp_id), "")
            else:
                card = (s.get("private_cards") or {}).get(str(opp_id), "")
            new_strengths.append(_hand_strength(card))

        if not new_actions:
            return 0

        all_actions = self._load_actions(opp_id) + new_actions
        all_strengths = self._load_strengths(opp_id) + new_strengths
        self._save_actions(opp_id, all_actions)
        self._save_strengths(opp_id, all_strengths)
        self._save_game_actions(
            opp_id,
            self._load_game_actions(opp_id) + [new_actions],
        )
        self._save_game_strengths(
            opp_id,
            self._load_game_strengths(opp_id) + [new_strengths],
        )
        self._game_count[opp_id] = self._game_count.get(opp_id, 0) + 1
        return len(new_actions)

    def load_from_dir(self, opp_id: int, dir_path: str) -> int:
        """Load all trajectory JSON files from a directory. No LLM calls.
        Returns number of games loaded.
        """
        import glob as _glob, json as _json
        paths = sorted(_glob.glob(f"{dir_path}/*.json"))
        count = 0
        for path in paths:
            traj = Trajectory.model_validate(_json.loads(open(path).read()))
            if self.update(opp_id, traj):
                count += 1
        return count

    # ── BaseModelingMethod interface ───────────────────────────────────

    def update(self, opp_id: int, trajectory: Trajectory,
               model=None) -> bool:
        """Online: ingest one game then update convergence tracking."""
        ingested = self._accumulate_game(opp_id, trajectory)
        if ingested == 0:
            return False

        if self.classification_mode == "cumulative":
            actions = self._load_actions(opp_id)
            strengths = self._load_strengths(opp_id)
            current_actions = actions[-ingested:]
            current_strengths = strengths[-ingested:]
            likelihood = self._classify_actions(current_actions, current_strengths)
            posterior = self._update_cumulative_posterior(opp_id, likelihood)
            evidence_actions = current_actions
        elif self.classification_mode == "q_window_log":
            window_actions, window_strengths = self._window_data(opp_id)
            likelihood = self._classify_actions(window_actions, window_strengths)
            posterior = self._update_window_log_posterior(opp_id, likelihood)
            evidence_actions = window_actions
        elif self.classification_mode == "q_window_ema":
            window_actions, window_strengths = self._window_data(opp_id)
            likelihood = self._classify_actions(window_actions, window_strengths)
            posterior = self._update_window_ema_posterior(opp_id, likelihood)
            evidence_actions = window_actions
        else:
            raise ValueError(f"Unknown classification_mode: {self.classification_mode}")
        hist = self.memory.get(self._convergence_key(opp_id), [])
        hist.append({
            "game": self._game_count[opp_id],
            "n_actions": len(evidence_actions),
            "total_actions": len(self._load_actions(opp_id)),
            "classification_mode": self.classification_mode,
            "q_window": self.q_window,
            "q_alpha": self.q_alpha,
            "q_tau": self.q_tau,
            "dist": posterior,
            "posterior": posterior,
            "likelihood": likelihood,
        })
        self.memory.set(self._convergence_key(opp_id), hist)
        return True

    def style_posterior(self, opp_id: int) -> dict[str, float]:
        """Current posterior over opponent styles (public accessor for records)."""
        return dict(self._posterior_from_current_profile(opp_id))

    def statement(self, opp_id: int, **kwargs) -> str:
        actions = self._load_actions(opp_id)
        if not actions:
            return ""
        if not self.signatures:
            raise ValueError("StatisticalProfileMethod requires data-driven signatures")
        dist = self._posterior_from_current_profile(opp_id)
        n_games = self._game_count.get(opp_id, 0)
        history = self.convergence_history(opp_id)
        lines = [
            "Statistical Profile",
            f"Classification mode: {self.classification_mode}",
            f"Opponent P{opp_id}: {n_games} games, {len(actions)} total actions, "
            f"{history[-1].get('n_actions', len(actions)) if history else len(actions)} evidence actions",
            "Recent style probabilities:",
        ]
        recent = history[-5:]
        if recent:
            for rec in reversed(recent):
                rec_dist = rec.get("posterior", rec.get("dist", {}))
                names = [name for name, _ in sorted(
                    rec_dist.items(), key=lambda item: -item[1]
                )]
                dist_text = ", ".join(
                    f"{name} {rec_dist.get(name, 0.0):.1%}"
                    for name in names
                )
                lines.append(f"  game {rec.get('game', '?')}: {dist_text}")
        else:
            names = [name for name, _ in sorted(
                dist.items(), key=lambda item: -item[1]
            )]
            dist_text = ", ".join(
                f"{name} {dist.get(name, 0.0):.1%}"
                for name in names
            )
            lines.append(f"  current: {dist_text}")
        lines.append(
"""These are some style legend that you might refer to, but not strictly applicable:
Maniac: over-aggressive, frequent bets/raises/bluffs.
LAG: loose-aggressive, wide range + pressure.
TAG: tight-aggressive, selective + value/aggression.
Nit: very tight, over-folds, bets strong.
CallingStatio: over-calls, under-folds, low bluff.
GTO: balanced Nash baseline.
"""
        )
        if self.classification_mode in {"q_window_ema", "q_window_log"}:
            lines.insert(3, f"Q-window: last {self.q_window} games")
        if self.classification_mode == "q_window_ema":
            lines.insert(4, f"EMA alpha: {self.q_alpha:.2f}")
        elif self.classification_mode == "q_window_log":
            lines.insert(4, f"Log update tau: {self.q_tau:.2f}")
        return "\n".join(lines)

    def exploit_hint(self, opp_id: int) -> str:
        actions = self._load_actions(opp_id)
        if not actions:
            return ""
        if not self.signatures:
            raise ValueError("StatisticalProfileMethod requires data-driven signatures")
        strengths = self._load_strengths(opp_id)
        profile = _build_profile(actions, strengths)
        dist = self._posterior_from_current_profile(opp_id)
        top = max(dist, key=dist.get)
        hints = {
            "CallingStation": "Never bluffs, never folds. Value bet thinly, never bluff.",
            "Maniac": "Always aggressive regardless of hand. Call down with medium hands, trap with strong ones.",
            "Nit": "Only plays strong hands, folds everything else. Bluff frequently, fold to their aggression.",
            "TAG": "Solid, slightly passive. Respect their bets, steal when they check.",
            "LAG": "Aggressive and unpredictable. Trap with strong hands, call down lighter.",
            "GTO": "Balanced play. No obvious exploit — focus on your own hand strength.",
        }
        hint = hints.get(top, "")
        if hint:
            return f"Exploitation vs P{opp_id} ({top}): {hint}"
        return ""

    def reset(self, opp_id: int) -> None:
        self._save_actions(opp_id, [])
        self._save_strengths(opp_id, [])
        self._save_game_actions(opp_id, [])
        self._save_game_strengths(opp_id, [])
        self.memory.set(self._convergence_key(opp_id), [])
        self._game_count.pop(opp_id, None)
        self._posterior_logps.pop(opp_id, None)
        self._posterior_cache.pop(opp_id, None)

    # ── Convergence analysis ───────────────────────────────────────────

    def convergence_history(self, opp_id: int) -> list[dict]:
        """Return per-game classification history for convergence plotting."""
        return self.memory.get(self._convergence_key(opp_id), [])

    def plot_convergence_chart(self, opp_id: int,
                               output_path: str | None = None) -> str | None:
        """Generate convergence plot for this opponent. Returns path or None."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        history = self.convergence_history(opp_id)
        if not history:
            return None

        true_arch = self.true_archetype
        archetype_order = ["CallingStation", "Maniac", "Nit", "TAG", "GTO", "LAG"]
        cmap = plt.cm.tab10
        colors = {a: cmap(i % 10) for i, a in enumerate(archetype_order)}

        fig, ax = plt.subplots(figsize=(10, 6))
        games = [h["game"] for h in history]
        for arch in archetype_order:
            probs = [h["dist"].get(arch, 0) for h in history]
            lw = 2.0 if arch == true_arch else 0.8
            alpha = 1.0 if arch == true_arch else 0.5
            ax.plot(games, probs, label=arch, color=colors[arch],
                    linewidth=lw, alpha=alpha)

        title = f"Convergence — P{opp_id}"
        if true_arch:
            title += f" (true: {true_arch})"
        ax.set_title(title)
        ax.set_xlabel("Games")
        ax.set_ylabel("Predicted Probability")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = f"_{self.true_archetype}" if self.true_archetype else ""
        path = output_path or str(
            Path(self.chart_dir) / f"stat{arch}_P{opp_id}_{ts}.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Offline analysis ───────────────────────────────────────────────

    def load_from_json(self, path: str, opp_id: int = 1) -> int:
        """Replay all games from a pilot_tom JSON to build model state offline.
        Returns number of trajectories loaded."""
        with open(path) as f:
            data = json.load(f)
        count = 0
        for batch in data.get("batches", []):
            for traj_dict in batch.get("sample_trajectories", []):
                traj = Trajectory.model_validate(traj_dict)
                if self.update(opp_id, traj):
                    count += 1
        return count

    def get_profile(self, opp_id: int) -> OpponentBehaviorProfile:
        """Return the current behavior profile for an opponent."""
        actions = self._load_actions(opp_id)
        strengths = self._load_strengths(opp_id)
        n_games = self._game_count.get(opp_id, 0)
        p = _build_profile(actions, strengths)
        p.n_games = n_games
        return p

    @staticmethod
    def estimate_signatures(profiles: dict[str, OpponentBehaviorProfile],
                            ) -> dict[str, dict[str, float]]:
        """Build archetype signature dict from labeled profiles."""
        sigs = {}
        for name, profile in profiles.items():
            sig = {}
            for a in NO_BET_ACTIONS:
                sig[f"no_bet_{a}"] = profile.no_bet_fractions.get(a, 0)
            for a in FACING_BET_ACTIONS:
                sig[f"fb_{a}"] = profile.facing_bet_fractions.get(a, 0)
            sig["selectivity"] = profile.selectivity
            sig["entropy"] = profile.entropy
            sigs[name] = sig
        return sigs

    def _signature_sample_text(self) -> str:
        archetypes = self.signature_metadata.get("archetypes", {})
        if not isinstance(archetypes, dict) or not archetypes:
            return ""
        parts = []
        for arch in ARCHETYPE_NAMES:
            item = archetypes.get(arch, {})
            if not isinstance(item, dict):
                continue
            games = item.get("games", item.get("trajectories"))
            actions = item.get("actions")
            p0_games = item.get("p0_games")
            p1_games = item.get("p1_games")
            if games is None and actions is None:
                continue
            pos = ""
            if p0_games is not None or p1_games is not None:
                pos = f" ({p0_games or 0} P0, {p1_games or 0} P1)"
            if actions is None:
                parts.append(f"{arch}: {games} games{pos}")
            else:
                parts.append(f"{arch}: {games} games/{actions} actions{pos}")
        if not parts:
            return ""
        return "Signature training samples: " + "; ".join(parts) + "."
