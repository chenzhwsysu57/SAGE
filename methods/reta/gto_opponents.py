"""ReTA-only fail-closed adapters for local pre-trained GTO approximations."""

import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import pickle
import random

from methods.baseline_opponents import LocalSpecifiedOpponent, opponent_config
from methods.sage.goofspiel_cfr import hand_to_mask
from methods.reta.leduc_opponent import LeducGTOOpponent

ROOT = Path(__file__).resolve().parents[2]
GOOFSPIEL_MODEL = ROOT / "results/goofspiel/models/goofspiel_n5_mccfr_cfrplus_seed42_continued.pkl"


class _ModelUnpickler(pickle.Unpickler):
    """Only the simple containers/metadata in the repository trainer checkpoint."""
    def find_class(self, module, name):
        allowed = {("collections", "defaultdict"): defaultdict,
                   ("builtins", "float"): float, ("argparse", "Namespace"): argparse.Namespace}
        if (module, name) not in allowed:
            raise ValueError("Unsupported class in GTO checkpoint")
        return allowed[(module, name)]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def goofspiel_key(obs):
    p = obs.player_id
    return (p, hand_to_mask(set(obs.private_info["hand"])),
            tuple(obs.public_info["outcome_history"]), tuple(obs.private_info["my_bids"]),
            tuple(e["p1_card"] if p == 0 else e["p0_card"] for e in obs.history
                  if e.get("p0_card") is not None), tuple(obs.public_info["prize_sequence"]))


class GoofspielGTOOpponent:
    """Same infoset and averaged strategy as GoofspielCFRMethod, no fallback.

    Retain sums and normalize only the queried row instead of duplicating the
    entire 170MB checkpoint's strategy. No training or full-tree inference.
    """
    def __init__(self, seed, path=GOOFSPIEL_MODEL):
        self.path = Path(path).resolve()
        self.sha256 = file_hash(self.path)
        with self.path.open("rb") as stream:
            data = _ModelUnpickler(stream).load()
        if type(data) is not dict or data.get("n") != 5 or not data.get("strategy_sum"):
            raise ValueError("Goofspiel requires an existing n5 strategy checkpoint")
        self.strategy = data["strategy_sum"]
        if not isinstance(self.strategy, dict) or file_hash(self.path) != self.sha256:
            raise ValueError("Invalid or changed GTO checkpoint")
        self.rng = random.Random(seed)

    def metadata(self):
        return {"name": "GTO", "implementation": "pretrained_goofspiel_MCCFR_plus_average",
                "path": str(self.path), "sha256": self.sha256, "n": 5,
                "infosets": len(self.strategy), "exact_equilibrium_claim": False,
                "fallback": "forbidden", "live_api": False}

    def act(self, obs, legal):
        if not legal or any(a.player_id != obs.player_id or a.type != "bid" for a in legal):
            raise ValueError("Wrong GTO seat or legal actions")
        cards = sorted(a.amount for a in legal)
        row = self.strategy.get(goofspiel_key(obs))
        if (not isinstance(row, dict) or set(row) != set(cards)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row.values())):
            raise ValueError("Missing or incompatible Goofspiel GTO infoset; no fallback")
        total = sum(row.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Invalid Goofspiel GTO probability mass")
        draw, cumulative, chosen = self.rng.random() * total, 0.0, cards[-1]
        for card in cards:
            cumulative += row[card]
            if draw < cumulative:
                chosen = card
                break
        return next(a for a in legal if a.amount == chosen), {"policy": "MCCFR_plus_average", "fallback": False}


class DiceGTOOpponent:
    def __init__(self, seed):
        self.config = opponent_config("liars_dice", "GTO")
        self.actor = LocalSpecifiedOpponent("liars_dice", self.config, seed)

    def metadata(self):
        return {**self.config, "exact_equilibrium_claim": False}

    def act(self, obs, legal):
        action, _, _, data = self.actor.act(obs, legal)
        return action, {"policy": "pretrained_dice_CFR", "fallback": data["fallback_used"]}


def make_opponent(game, seed):
    classes = {"leduc": LeducGTOOpponent, "liars_dice": DiceGTOOpponent, "goofspiel": GoofspielGTOOpponent}
    return classes[game](seed)