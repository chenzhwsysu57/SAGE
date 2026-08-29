"""Bayesian persona classifier for online ToM opponent modeling.

Loads pre-trained P(bid | prize, persona) from 60-game benchmark data.
Maintains per-opponent posterior via cumulative likelihood update.

Usage:
    model = BayesPersonaModel()
    model.observe_round(opp_id=1, prize=4, bid=5)
    post = model.posterior(1)       # np.ndarray(6,)
    stmt = model.statement(1)       # formatted string for prompt injection
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

PERSONAS = ["Greedy", "Conservative", "Sacrificial", "Random", "GTO", "Mirror"]
N = 5  # cards/prizes

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "results/goofspiel/models/nb_persona_models.json",
)

TRAIN_RUNS = {
    "Greedy":       "results/goofspiel/goofspiel_bench/20260701_201428_p1_30g_greedy",
    "Conservative": "results/goofspiel/goofspiel_bench/20260701_203256_p1_30g_conservative",
    "Sacrificial":  "results/goofspiel/goofspiel_bench/20260701_201641_p1_30g_sacrificial",
    "Random":       "results/goofspiel/goofspiel_bench/20260701_200410_p1_30g_random",
    "GTO":          "results/goofspiel/goofspiel_bench/20260701_200546_p1_30g_gto",
    "Mirror":       "results/goofspiel/goofspiel_bench/20260701_200234_p1_30g_mirror",
}


def train_and_save(path: str | None = None) -> str:
    """Train P(bid|prize,persona) from benchmark data and save to JSON."""
    if path is None:
        path = DEFAULT_MODEL_PATH

    from analysis.goofspiel.persona_bayes_classifier import load_games  # noqa: E402

    models = {}
    for pn in PERSONAS:
        games = load_games(TRAIN_RUNS[pn])
        count: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        total: dict[int, int] = defaultdict(int)
        for game in games:
            for rnd in game:
                count[rnd["prize"]][rnd["bid"]] += 1
                total[rnd["prize"]] += 1
        model = {}
        for prize in range(1, N + 1):
            t = total.get(prize, 0)
            model[str(prize)] = {}
            for bid in range(1, N + 1):
                model[str(prize)][str(bid)] = (count[prize].get(bid, 0) + 1) / (t + N)
        models[pn] = model

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"personas": PERSONAS, "N": N, "models": models}, f, indent=2)
    print(f"NB models saved to {path}")
    return path


class BayesPersonaModel:
    """Online Bayesian persona classifier for Goofspiel.

    Updates posterior after each observed opponent bid.
    Uses adaptive temperature to prevent probability saturation.

    Thread-safe for multiple opponents (keyed by opp_id).
    """

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            model_path = train_and_save(model_path)

        with open(model_path) as f:
            data = json.load(f)

        self.personas: list[str] = data["personas"]
        self.N: int = data["N"]

        # models[pn][prize][bid] = probability
        raw = data["models"]
        self.models: dict[str, dict[int, dict[int, float]]] = {}
        for pn in self.personas:
            self.models[pn] = {}
            for prize_str, bids in raw[pn].items():
                prize = int(prize_str)
                self.models[pn][prize] = {}
                for bid_str, prob in bids.items():
                    self.models[pn][prize][int(bid_str)] = prob

        # Per-opponent state
        self._log_probs: dict[int, np.ndarray] = {}
        self._n_rounds: dict[int, int] = {}
        self._ranking_history: dict[int, list[tuple[int, np.ndarray]]] = {}

    def observe_round(self, opp_id: int, prize: int, bid: int) -> None:
        """Update posterior after observing one opponent bid."""
        if opp_id not in self._log_probs:
            n_p = len(self.personas)
            self._log_probs[opp_id] = np.full(n_p, np.log(1.0 / n_p))
            self._n_rounds[opp_id] = 0
            self._ranking_history[opp_id] = []

        for pi, pn in enumerate(self.personas):
            self._log_probs[opp_id][pi] += np.log(self.models[pn][prize][bid])
        self._n_rounds[opp_id] += 1

        # Record posterior at this round
        post = self.posterior(opp_id)
        self._ranking_history[opp_id].append((self._n_rounds[opp_id], post.copy()))

    def posterior(self, opp_id: int, temperature: float = 5.0) -> np.ndarray:
        """Current posterior with adaptive temperature.

        Adaptive temperature = base_temp * sqrt(total_rounds / 5.0)
        prevents probability saturation as more data accumulates.
        """
        if opp_id not in self._log_probs:
            n_p = len(self.personas)
            return np.full(n_p, 1.0 / n_p)

        lp = self._log_probs[opp_id]
        n = self._n_rounds.get(opp_id, 1)
        t = temperature * np.sqrt(n / 5.0)

        scaled = (lp - lp.max()) / t
        exp_vals = np.exp(scaled)
        return exp_vals / exp_vals.sum()

    def statement(self, opp_id: int, temperature: float = 5.0) -> str:
        """Human-readable posterior for LLM prompt injection."""
        if opp_id not in self._log_probs:
            return ""

        post = self.posterior(opp_id, temperature)
        n = self._n_rounds[opp_id]
        indices = np.argsort(post)[::-1]

        lines = [
            f"Opponent persona estimate (based on {n} observed bids):",
        ]
        for i in indices:
            pct = post[i] * 100
            marker = " ← most likely" if i == indices[0] else ""
            lines.append(f"  {self.personas[i]:<15}: {pct:>5.1f}%{marker}")

        # Recent posterior history — last 5 rounds (newest first)
        hist = self._ranking_history.get(opp_id, [])
        if len(hist) > 1:
            lines.append(f"\nPersona posterior history (last 5 rounds):")
            for rnd, post_vec in reversed(hist[-5:]):
                idx = np.argsort(post_vec)[::-1]
                entries = [f"{self.personas[i]}({post_vec[i]*100:.1f}%)"
                           for i in idx]
                lines.append(f"  Round {rnd:>2}: {'  '.join(entries)}")

        return "\n".join(lines)

    def exploit_hint(self, opp_id: int, post: np.ndarray | None = None,
                     indices: np.ndarray | None = None) -> str:
        """Generate persona-specific exploitation advice."""
        if opp_id not in self._log_probs:
            return ""
        if post is None:
            post = self.posterior(opp_id)
        if indices is None:
            indices = np.argsort(post)[::-1]

        top_pn = self.personas[indices[0]]
        top_prob = post[indices[0]]

        if top_prob < 0.35:
            return "Low confidence — play balanced GTO until clearer pattern emerges."

        hints = {
            "Greedy": "Overbids for high prizes, underbids for low prizes. "
                      "Win cheap prizes with moderate bids; let them waste high cards.",
            "Conservative": "Bids ~prize-1 (seeks discount). "
                            "Bid at prize level to win consistently with modest cards.",
            "Sacrificial": "Sacrifices one round with a high card, bids prize for the rest. "
                           "Identify the sacrifice round and win it cheaply.",
            "Random": "Uniform random — no pattern. Stick to GTO baseline.",
            "GTO": "Plays Nash equilibrium. Stick to GTO baseline.",
            "Mirror": "Bids exactly the prize value. "
                      "Bid prize+1 to win every round at minimal card cost.",
        }
        return hints.get(top_pn, "Play GTO.")

    def reset(self, opp_id: int | None = None) -> None:
        """Clear accumulated data for one opponent (or all if None)."""
        if opp_id is not None:
            self._log_probs.pop(opp_id, None)
            self._n_rounds.pop(opp_id, None)
            self._ranking_history.pop(opp_id, None)
        else:
            self._log_probs.clear()
            self._n_rounds.clear()
            self._ranking_history.clear()

    @property
    def n_rounds_by_opp(self) -> dict[int, int]:
        return dict(self._n_rounds)


if __name__ == "__main__":
    m = BayesPersonaModel()
    print(f"Loaded {len(m.personas)} personas, N={m.N}")
    print(f"  Mirror P(bid=3|prize=3) = {m.models['Mirror'][3][3]:.3f}")
    print(f"  Greedy P(bid=5|prize=5) = {m.models['Greedy'][5][5]:.3f}")

    # Simulate observing a Greedy opponent
    m.observe_round(1, prize=5, bid=5)
    m.observe_round(1, prize=1, bid=1)
    print(f"\nAfter 2 Greedy-like rounds:")
    print(m.statement(1))
