"""CFR-based GTO method for II-Goofspiel.

Loads a pre-trained MCCFR/MCCFR+ strategy and uses it to bid.
Falls back to uniform random when an info set is not found.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Any

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
from methods.sage.goofspiel_cfr import (
    hand_to_mask,
    regret_matching_strategy,
)

# Default model paths relative to project root (tried in order)
_DEFAULT_MODEL_CANDIDATES = [
    "results/goofspiel/models/goofspiel_n5_mccfr_cfrplus_seed42_continued.pkl",
    "results/goofspiel/models/goofspiel_n5_mccfr_cfr_seed42.pkl",
    "results/goofspiel/models/goofspiel_n5_mccfr_cfrplus_seed42.pkl",
]


class GoofspielCFRMethod(BaseMethod):
    """CFR-based GTO strategy for Goofspiel.

    Loads a trained MCCFR/MCCFR+ strategy and samples actions according
    to the averaged Nash-equilibrium policy.

    Args:
        model_path: Path to a .pkl file saved by train_goofspiel.py.
        name: Display name.
        seed: Random seed for action sampling.
        fallback_uniform: If True, fall back to uniform when infoset not found.
    """

    def __init__(
        self,
        model_path: str | None = None,
        name: str = "CFR",
        seed: int = 0,
        fallback_uniform: bool = True,
    ):
        self.name = name
        self.method_type = "cfr"
        self._rng = random.Random(seed)
        self._fallback_uniform = fallback_uniform
        self._strategy: dict[tuple, dict[int, float]] = {}
        self._n_infosets = 0
        self._n = 5  # default, overridden from loaded model

        # Resolve model path relative to project root
        if model_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            for candidate in _DEFAULT_MODEL_CANDIDATES:
                candidate_path = os.path.join(base, candidate)
                if os.path.exists(candidate_path):
                    model_path = candidate_path
                    break
            if model_path is None:
                model_path = os.path.join(base, _DEFAULT_MODEL_CANDIDATES[0])
        else:
            model_path = os.path.abspath(model_path)

        if os.path.exists(model_path):
            self._load(model_path)
        else:
            print(f"[GoofspielCFRMethod] Model not found at {model_path}, "
                  f"will fall back to uniform")

    def _load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        strategy_sum: dict = data.get("strategy_sum", data)
        self._n = data.get("n", 5)
        if isinstance(data, dict) and "n" in data:
            self._n = data["n"]

        # Average the strategy sums
        for infoset, sums in strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                self._strategy[infoset] = {k: v / total for k, v in sums.items()}

        self._n_infosets = len(self._strategy)
        print(f"[GoofspielCFRMethod] Loaded {self._n_infosets} infosets "
              f"from {os.path.basename(path)} (N={self._n})")

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel | None = None,
    ) -> tuple[Action, str, str]:
        # Build infoset key from observation
        player_id = observation.player_id
        hand = observation.private_info.get("hand", [])
        hand_mask = hand_to_mask(set(hand))
        outcome_history = tuple(
            observation.public_info.get("outcome_history", [])
        )
        my_bids = tuple(observation.private_info.get("my_bids", []))
        opp_bids = tuple(
            entry["p1_card"] if player_id == 0 else entry["p0_card"]
            for entry in observation.history
            if entry.get("p0_card") is not None
        )
        prize_sequence = tuple(
            observation.public_info.get("prize_sequence", [])
        )
        key = (player_id, hand_mask, outcome_history, my_bids, opp_bids, prize_sequence)

        # Look up strategy
        probs = self._strategy.get(key, None)
        if probs is None:
            # Fall back to uniform
            idx = self._rng.randint(0, len(legal_actions) - 1)
            action = legal_actions[idx]
            return action, "cfr_uniform", (
                f"CFR GTO (uniform fallback): bid {action.amount}"
            )

        # Sample according to strategy
        legal_cards = sorted(a.amount for a in legal_actions)
        strat_probs = [probs.get(c, 0.0) for c in legal_cards]
        total = sum(strat_probs)
        if total < 1e-12:
            idx = self._rng.randint(0, len(legal_cards) - 1)
        else:
            r = self._rng.random() * total
            cum = 0.0
            idx = 0
            for i, p in enumerate(strat_probs):
                cum += p
                if r <= cum:
                    idx = i
                    break

        card = legal_cards[idx]
        action = Action(
            player_id=player_id, type="bid", amount=card
        )
        return action, "cfr_gto", (
            f"CFR GTO: sampled card {card} from {dict(zip(legal_cards, [f'{p:.3f}' for p in strat_probs]))}"
        )

    def format_for_prompt(self, observation, legal_actions) -> str:
        """Format CFR GTO strategy as a prompt section."""
        player_id = observation.player_id
        hand = observation.private_info.get("hand", [])
        hand_mask = hand_to_mask(set(hand))
        outcome_history = tuple(
            observation.public_info.get("outcome_history", [])
        )
        my_bids = tuple(observation.private_info.get("my_bids", []))
        opp_bids = tuple(
            entry["p1_card"] if player_id == 0 else entry["p0_card"]
            for entry in observation.history
            if entry.get("p0_card") is not None
        )
        prize_sequence = tuple(
            observation.public_info.get("prize_sequence", [])
        )
        key = (player_id, hand_mask, outcome_history, my_bids, opp_bids, prize_sequence)

        probs = self._strategy.get(key, None)
        if probs is None:
            return ""

        legal_cards = sorted(a.amount for a in legal_actions)
        rnd = observation.public_info.get("round", "?")
        prize = observation.public_info.get("prize", "?")

        lines = [
            f"Current infoset — P{player_id}, hand={sorted(hand)}, "
            f"round {rnd}, prize {prize}",
            "GTO bid distribution:",
        ]
        for c in legal_cards:
            p = probs.get(c, 0.0)
            if p > 0.001:
                lines.append(f"  bid {c}: {p*100:.0f}%")

        lines.append(
            "Note: this is the Nash equilibrium baseline. "
            "Deviations by opponent are your edge."
        )
        return "\n".join(lines)

    def reflect(self, trajectory, model, game_name=""):
        return []  # CFR strategy is fixed

    def get_strategy_size(self) -> int:
        return self._n_infosets
