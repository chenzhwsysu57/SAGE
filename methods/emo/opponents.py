"""EMO-local adapters for existing fixed evaluation opponents."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import random

from base.types import Action
from methods.baseline_opponents import LocalSpecifiedOpponent, opponent_config as existing_opponent_config
from methods.emo.style_opponents import is_llm_style

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDUC_CFR = ROOT / "results/leduc/cfr_model/cfr_plus_21000.json"
DEFAULT_GOOFSPIEL_CFR = ROOT / "results/goofspiel/models/goofspiel_n5_mccfr_cfrplus_seed42_continued.pkl"


class _CheckpointUnpickler(pickle.Unpickler):
    """Load only container/metadata classes used by the existing checkpoint."""
    def find_class(self, module, name):
        allowed = {("collections", "defaultdict"): defaultdict,
                   ("builtins", "float"): float,
                   ("argparse", "Namespace"): argparse.Namespace}
        if (module, name) not in allowed:
            raise ValueError("Unsupported class in Goofspiel GTO checkpoint")
        return allowed[(module, name)]


def emo_opponent_config(game: str, opponent: str, model_path: str | Path | None = None) -> dict:
    if game == "leduc" and is_llm_style(game, opponent):
        if model_path is not None:
            raise ValueError("LLM archetypes do not use a CFR model")
        return {"name": opponent, "execution": "llm_archetype", "live_api": True,
                "policy": "existing_archetype_prompt", "fallback": "error", "game": game}
    if game == "leduc" and opponent == "GTO":
        path = Path(model_path or DEFAULT_LEDUC_CFR).resolve()
        if not path.is_file():
            raise ValueError("Leduc GTO requires the existing CFR+ strategy file")
        content = path.read_bytes()
        return {"name": "GTO", "execution": "local_cfr_plus", "live_api": False,
            "policy": "existing_leduc_cfr_plus_approximation", "fallback": "error",
                "model_path": str(path), "model_sha256": hashlib.sha256(content).hexdigest()}
    if game == "goofspiel" and opponent == "GTO":
        path = Path(model_path or DEFAULT_GOOFSPIEL_CFR).resolve()
        if not path.is_file():
            raise ValueError("Goofspiel GTO requires the existing n5 MCCFR+ checkpoint")
        return {"name": "GTO", "execution": "local_mccfr_plus", "live_api": False,
                "policy": "existing_goofspiel_mccfr_plus_average", "fallback": "error",
                "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n": 5}
    return existing_opponent_config(game, opponent, model_path)


class EMOLocalOpponent:
    """Use existing opponent methods without forwarding their private diagnostics."""

    def __init__(self, game: str, config: dict, seed: int):
        actual = emo_opponent_config(game, config.get("name"), config.get("model_path"))
        if actual != config:
            raise ValueError("Opponent configuration/model fingerprint mismatch")
        self.config = dict(config)
        self.name = f"Local{config['name']}"
        self.method_type = config["execution"]
        self.seed = seed
        if game == "leduc":
            from methods.sage.cfr import CFRMethod
            self.actor = CFRMethod(name=self.name, seed=seed, strategy_path=config["model_path"])
            if emo_opponent_config(game, config["name"], config["model_path"]) != config:
                raise ValueError("Leduc CFR file changed during loading")
            if self.actor._trainer.strategy != json.loads(Path(config["model_path"]).read_text())["strategy"]:
                raise ValueError("Leduc cached strategy differs from checkpoint")
        elif game == "goofspiel" and config["name"] == "GTO":
            with Path(config["model_path"]).open("rb") as stream:
                checkpoint = _CheckpointUnpickler(stream).load()
            if type(checkpoint) is not dict or checkpoint.get("n") != 5:
                raise ValueError("Goofspiel GTO checkpoint must declare n=5")
            self.strategy = checkpoint.get("strategy_sum")
            if not isinstance(self.strategy, dict) or not self.strategy:
                raise ValueError("Goofspiel GTO strategy is empty")
            if emo_opponent_config(game, config["name"], config["model_path"]) != config:
                raise ValueError("Goofspiel CFR file changed during loading")
            self.rng = random.Random(seed)
        else:
            self.actor = LocalSpecifiedOpponent(game, config, seed)

    def act(self, observation, legal_actions: list[Action], model=None):
        if not legal_actions or any(a.player_id != observation.player_id for a in legal_actions):
            raise ValueError("Opponent needs its own observation and legal actions")
        if self.config["execution"] == "local_mccfr_plus":
            from methods.sage.goofspiel_cfr import hand_to_mask
            player = observation.player_id
            key = (player, hand_to_mask(set(observation.private_info["hand"])),
                   tuple(observation.public_info["outcome_history"]),
                   tuple(observation.private_info["my_bids"]),
                   tuple(row["p1_card"] if player == 0 else row["p0_card"]
                         for row in observation.history if row.get("p0_card") is not None),
                   tuple(observation.public_info["prize_sequence"]))
            cards = sorted(a.amount for a in legal_actions)
            row = self.strategy.get(key)
            if (not isinstance(row, dict) or set(row) != set(cards)
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in row.values())):
                raise ValueError("Missing or invalid Goofspiel GTO infoset; fallback forbidden")
            total = sum(row.values())
            if not math.isfinite(total) or total <= 0:
                raise ValueError("Invalid Goofspiel GTO probability mass")
            draw, cumulative, chosen = self.rng.random() * total, 0.0, cards[-1]
            for card in cards:
                cumulative += row[card]
                if draw < cumulative:
                    chosen = card
                    break
            action = next(a for a in legal_actions if a.amount == chosen)
            return action, "", "", {"opponent": "GTO", "execution": "local_mccfr_plus",
                                      "fallback_used": False,
                                      "model_sha256": self.config["model_sha256"], "rng_seed": self.seed}
        if self.config["execution"] != "local_cfr_plus":
            return self.actor.act(observation, legal_actions, None)
        from methods.sage.cfr import ACTION_TO_CODE, encode_history_for_cfr
        street_name = observation.public_info.get("street")
        if street_name not in ("preflop", "flop"):
            raise ValueError("Invalid Leduc street")
        street = 0 if street_name == "preflop" else 1
        card = observation.private_info["card"]
        board = observation.public_info.get("community_card")
        history = encode_history_for_cfr(observation.history, street)
        key = f"{card[0]}|{board[0] if board else '?'}|{history}|{street}"
        row = self.actor._trainer.strategy.get(key)
        codes = {ACTION_TO_CODE.get(a.type) for a in legal_actions}
        if (not isinstance(row, dict) or None in codes or set(row) != codes
                or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in row.values())
                or not math.isfinite(sum(row.values())) or sum(row.values()) <= 0):
            raise ValueError("Missing or invalid Leduc GTO infoset; fallback forbidden")
        action, _, _ = self.actor.act(observation, legal_actions, None)
        if action not in legal_actions:
            raise ValueError("Leduc CFR returned an illegal action")
        return action, "", "", {"opponent": "GTO", "execution": "local_cfr_plus",
                                  "fallback_used": False,
                                  "model_sha256": self.config["model_sha256"], "rng_seed": self.seed}