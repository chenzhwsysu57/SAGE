"""Agent-Pro-private local opponents; no provider, training or shared cache.

Dice GTO and Mirror delegate to the unchanged shared adapters. Leduc loads only
an existing JSON CFR+ table and reuses the real CFR encoding/normalization/sampler.
The historical model has no explicit game-config manifest: compatibility is
checked against the entire fixed-variant information-set/action space, not
inferred from its filename. GTO is approximate, never an exact-equilibrium claim.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random

from methods import cfr
from methods.baseline_opponents import (
    LocalSpecifiedOpponent as SharedOpponent,
    opponent_config as shared_opponent_config,
)
from .policy import strict_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDUC_MODEL = ROOT / "results/leduc/cfr_model/cfr_plus_21000.json"
LEDUC_CONFIG = {"game": "leduc", "num_players": 2, "deck": "JJQQKK",
                "ante": 1, "preflop_bet": 2, "flop_bet": 4,
                "max_raises_per_round": 2}


def _expected_rows():
    """Enumerate all public betting histories, not sampled wins/losses/deals."""
    rows = {}

    def walk(history, street):
        codes = set(cfr.get_legals(history, street))
        for rank in cfr.RANKS:
            for board in (cfr.RANKS if street else ("?",)):
                rows[f"{rank}|{board}|{history}|{street}"] = codes
        for code in codes:
            terminal, new_street, _, new_history = cfr.next_round(
                history + code, street, ("J", "H"), ("Q", "S"), known_comm=("K", "S"))
            if not terminal:
                walk(new_history, new_street)

    walk("", 0)
    return rows


def _validate_row(row, codes):
    if (type(row) is not dict or set(row) != set(codes)
            or any(type(p) not in (int, float) or p < 0 or not math.isfinite(p)
                   for p in row.values())):
        raise ValueError("Missing or incompatible Leduc CFR information set/distribution; fallback forbidden")
    total = sum(row.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid Leduc CFR probability mass; fallback forbidden")


def _load_leduc(model_path):
    path = Path(model_path or DEFAULT_LEDUC_MODEL).resolve()
    if not path.is_file():
        raise ValueError("Leduc GTO requires an existing strategy JSON model; never train")
    raw = path.read_bytes()
    data = strict_json(raw.decode("utf-8"))
    if type(data) is not dict or type(data.get("strategy")) is not dict or not data["strategy"]:
        raise ValueError("Empty or invalid Leduc CFR strategy")
    # Accept the existing manifest-less checkpoint, but reject explicit conflicts.
    for key, expected in LEDUC_CONFIG.items():
        if key in data and (type(data[key]) is not type(expected) or data[key] != expected):
            raise ValueError("Leduc CFR model configuration mismatch")
    if "game_config" in data and data["game_config"] != LEDUC_CONFIG:
        raise ValueError("Leduc CFR model configuration mismatch")
    strategy = data["strategy"]
    expected = _expected_rows()
    if set(strategy) != set(expected):
        raise ValueError("Leduc CFR information-set coverage mismatch; fallback forbidden")
    for info, codes in expected.items():
        _validate_row(strategy[info], codes)
    for key in ("iterations", "num_infosets"):
        if key in data and (type(data[key]) is not int or data[key] <= 0):
            raise ValueError("Invalid Leduc CFR model metadata")
    if data.get("num_infosets", len(strategy)) != len(strategy):
        raise ValueError("Leduc CFR information-set count mismatch")
    config = {"name": "GTO", "execution": "local_cfr", "live_api": False,
              "policy": "existing_leduc_cfr_plus_approximation", "fallback": "error",
              "model_path": str(path), "model_sha256": hashlib.sha256(raw).hexdigest(),
              "strategy_rows": len(strategy), "iterations": data.get("iterations"),
              "game_config": dict(LEDUC_CONFIG), "exact_equilibrium_claim": False,
              "compatibility": "complete_fixed_variant_infoset_action_coverage",
              "model_game_config_explicit": "game_config" in data}
    return config, strategy


def opponent_config(game: str, opponent: str, model_path: str | Path | None = None) -> dict:
    if (game, opponent) == ("leduc", "GTO"):
        return _load_leduc(model_path)[0]
    if (game, opponent) not in (("liars_dice", "GTO"), ("goofspiel", "Mirror"), (game, "uniform")):
        from .style_opponents import style_config
        return style_config(game, opponent, model_path)
    return shared_opponent_config(game, opponent, model_path)


class LocalSpecifiedOpponent:
    """Same four-value episode interface; private diagnostics are never prompts."""

    def __init__(self, game: str, config: dict, seed: int):
        self._delegate = None
        if config.get("execution") in ("llm_style", "local_random") or (game == "goofspiel" and config.get("name") == "GTO"):
            from .style_opponents import StyleOpponent
            if opponent_config(game,config["name"],config.get("model_path")) != config:
                raise ValueError("Style opponent configuration changed")
            self._delegate = StyleOpponent(game, config, seed)
            return
        if (game, config.get("name")) != ("leduc", "GTO"):
            self._delegate = SharedOpponent(game, config, seed)
            return
        actual, self.strategy = _load_leduc(config.get("model_path"))
        if actual != config:
            raise ValueError("Opponent configuration/model fingerprint mismatch")
        self.config = actual
        self.name = "LocalGTO"
        self.method_type = "local_cfr"
        self.rng_seed = seed
        self.rng = random.Random(seed)
        self._check_hash()

    def _check_hash(self):
        path = Path(self.config["model_path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != self.config["model_sha256"]:
            raise ValueError("Leduc CFR model fingerprint changed")

    def act(self, observation, legal_actions, model=None):
        if self._delegate is not None:
            return self._delegate.act(observation, legal_actions, model)
        self._check_hash()
        if not legal_actions or any(a.player_id != observation.player_id for a in legal_actions):
            raise ValueError("Opponent needs its own observation and legal actions")
        street_name = observation.public_info.get("street")
        card = observation.private_info.get("card")
        board = observation.public_info.get("community_card")
        if (street_name not in ("preflop", "flop") or not isinstance(card, str)
                or len(card) != 2 or card[0] not in cfr.RANKS or card[1] not in "HS"
                or (street_name == "preflop" and board)
                or (street_name == "flop" and (not isinstance(board, str) or len(board) != 2
                    or board[0] not in cfr.RANKS or board[1] not in "HS"))):
            raise ValueError("Invalid Leduc CFR observation/configuration")
        street = 0 if street_name == "preflop" else 1
        history = cfr.encode_history_for_cfr(observation.history, street)
        info = f"{card[0]}|{board[0] if board else '?'}|{history}|{street}"
        codes = [cfr.ACTION_TO_CODE.get(a.type) for a in legal_actions]
        if None in codes or len(set(codes)) != len(codes):
            raise ValueError("Invalid Leduc legal action set")
        _validate_row(self.strategy.get(info), codes)
        probabilities, fallback = cfr.strategy_probs_for_legal_actions(self.strategy, info, legal_actions)
        if fallback:
            raise ValueError("Leduc CFR uniform fallback forbidden")
        _validate_row(probabilities, [a.type for a in legal_actions])
        action = cfr.sample_action_from_probs(legal_actions, probabilities, self.rng)
        if action not in legal_actions:
            raise ValueError("Local Leduc opponent returned an illegal action")
        self._check_hash()
        # Never expose CFR lookup text (contains the actor's private rank).
        return action, "", "", {"opponent": "GTO", "execution": self.method_type,
                                "fallback_used": False, "rng_seed": self.rng_seed,
                                "model_sha256": self.config["model_sha256"]}