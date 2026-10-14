"""Read-only pre-trained Leduc CFR+ opponent for ReTA's isolated test harness.

Reuse existing infoset/probability/sampling helpers, but never instantiate the
legacy auto-training loader. Missing/incompatible strategy rows fail closed.
"""

import hashlib
import json
import math
from pathlib import Path
import random

from methods.sage.cfr import ACTION_TO_CODE, encode_history_for_cfr, sample_action_from_probs, strategy_probs_for_legal_actions

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "results/leduc/cfr_model/cfr_plus_21000.json"


class LeducGTOOpponent:
    def __init__(self, seed: int, path: Path = DEFAULT_MODEL):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()  # Missing file fails; NEVER train or write it.
        self.sha256 = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        self.strategy = data["strategy"]
        if not isinstance(self.strategy, dict) or not self.strategy:
            raise ValueError("Empty CFR model")
        self.iterations = data.get("iterations")
        self.rng = random.Random(seed)

    def metadata(self):
        return {"name": "GTO", "implementation": "pretrained_leduc_CFR_plus",
                "path": str(self.path), "sha256": self.sha256,
                "iterations": self.iterations, "infosets": len(self.strategy),
                "exact_equilibrium_claim": False, "fallback": "forbidden", "live_api": False}

    def act(self, observation, legal_actions):
        if not legal_actions or any(a.player_id != observation.player_id for a in legal_actions):
            raise ValueError("Invalid opponent seat/legal actions")
        rank = observation.private_info["card"][0]
        board = observation.public_info["community_card"]
        street = 0 if observation.public_info["street"] == "preflop" else 1
        history = encode_history_for_cfr(observation.history, street)
        info = f"{rank}|{board[0] if board else '?'}|{history}|{street}"
        row = self.strategy.get(info)
        codes = {ACTION_TO_CODE[a.type] for a in legal_actions}
        if (type(row) is not dict or set(row) != codes
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row.values())
                or not math.isfinite(sum(row.values())) or sum(row.values()) <= 0):
            raise ValueError("Missing or incompatible CFR information set; no fallback")
        probabilities, fallback = strategy_probs_for_legal_actions(self.strategy, info, legal_actions)
        if fallback:
            raise ValueError("CFR fallback forbidden")
        action = sample_action_from_probs(legal_actions, probabilities, self.rng)
        if action not in legal_actions:
            raise ValueError("CFR produced noncanonical action")
        return action, {"policy": "pretrained_CFR_plus", "fallback": False}