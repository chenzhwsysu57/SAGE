"""Original benchmark style prompts with strict actions.

Leduc five LLM + CFR; dice four LLM + Random/CFR; Goofspiel three LLM +
Random/Mirror/CFR. Only original prompt renderers are reused, not permissive
parsers. Any LLM action parsing failure is explicitly recorded and replaced by
one random legal action; transport failures still stop rather than guess.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import pickle
import random

from .policy import model_json
from .method import parse_action

STYLES = {
    "leduc": ("LAG", "TAG", "CallingStation", "Nit", "Maniac", "GTO"),
    "liars_dice": ("Bluffer", "Honest", "AggressiveChallenger", "ConservativeChallenger", "Random", "GTO"),
    "goofspiel": ("Greedy", "Conservative", "Sacrificial", "Random", "GTO", "Mirror"),
}
ROOT = Path(__file__).resolve().parents[2]
GOOF_MODEL = ROOT / "results/goofspiel/models/goofspiel_n5_mccfr_cfrplus_seed42_continued.pkl"


def is_llm_style(game, name):
    return name in STYLES[game] and name not in ("GTO", "Random", "Mirror")


def style_config(game, name, model_path=None):
    if name not in STYLES.get(game, ()):
        raise ValueError("Unknown style for this game")
    if model_path and not (game == "goofspiel" and name == "GTO"):
        raise ValueError("Only GTO accepts a model")
    if game == "goofspiel" and name == "GTO":
        path = Path(model_path or GOOF_MODEL).resolve()
        if not path.is_file():
            raise ValueError("Existing Goofspiel GTO checkpoint required; no training")
        with path.open("rb") as f:
            fingerprint = hashlib.file_digest(f, "sha256").hexdigest()
        return {"name": name, "execution": "local_cfr", "live_api": False,
                "model_path": str(path), "model_sha256": fingerprint,
                "policy": "n5_MCCFR_plus_average_approximation", "fallback": "error",
                "exact_equilibrium_claim": False}
    return {"name": name, "execution": "llm_style" if is_llm_style(game,name) else "local_random",
            "live_api": is_llm_style(game,name), "fallback": "error",
            "prompt_protocol": "original_benchmark_style_strict_parse_v1"}


def renderer(game, name):
    if game == "leduc":
        from modeling.leduc.archetypes import ArchetypeAgent
        return ArchetypeAgent(name)
    if game == "liars_dice":
        from modeling.liars_dice.archetypes import ArchetypeAgent
        return ArchetypeAgent(name)
    from modeling.goofspiel.archetypes import GoofspielArchetypeAgent
    return GoofspielArchetypeAgent(name)


def parse_style_action(raw, game, legal):
    obj = model_json(raw)
    if type(obj) is not dict or "action" not in obj:
        raise ValueError("Style opponent must return an explicit JSON action")
    action = obj["action"]
    if type(action) is str:
        return parse_action(game, action, legal)
    if type(action) is not dict or "type" not in action:
        raise ValueError("Style opponent requires action.type")
    allowed = {"type", "amount", "quantity", "face", "player_id", "data"}
    if set(action) - allowed:
        raise ValueError("Unknown opponent action fields")
    if "player_id" in action and (type(action["player_id"]) is not int or action["player_id"] != legal[0].player_id):
        raise ValueError("Opponent action has wrong player")
    for key in ("amount", "quantity", "face"):
        if key in action and type(action[key]) is not int:
            raise ValueError("Opponent action parameters must be integers")
    matches = [a for a in legal if a.type == action["type"]]
    if "amount" in action:
        matches = [a for a in matches if a.amount == action["amount"]]
    if game == "goofspiel" and "amount" not in action:
        raise ValueError("Opponent bid requires an explicit amount")
    if game == "liars_dice" and action["type"] == "bid":
        data = action.get("data", {k:action[k] for k in ("quantity","face") if k in action})
        if (type(data) is not dict or set(data) != {"quantity","face"}
                or any(type(x) is not int for x in data.values())
                or any(k in action and action[k] != data[k] for k in ("quantity","face"))):
            raise ValueError("Opponent bid requires complete quantity/face")
        matches = [a for a in matches if a.data == data]
    elif action.get("data") not in (None, {}):
        raise ValueError("Unexpected opponent action data")
    if len(matches) != 1:
        raise ValueError("Illegal or ambiguous opponent action; fallback forbidden")
    return matches[0]


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("collections", "defaultdict"): defaultdict,
                   ("builtins", "float"): float, ("argparse", "Namespace"): argparse.Namespace}
        if (module,name) not in allowed:
            raise ValueError("Unsupported Goofspiel checkpoint class")
        return allowed[(module,name)]


@lru_cache(maxsize=1)
def _goof_strategy(path, fingerprint):
    with Path(path).open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest() != fingerprint:
            raise ValueError("Goofspiel model changed")
        f.seek(0)
        data = _SafeUnpickler(f).load()
    if type(data) is not dict or data.get("n") != 5 or not isinstance(data.get("strategy_sum"), dict):
        raise ValueError("Expected n5 Goofspiel strategy sums")
    return data["strategy_sum"]


class StyleOpponent:
    def __init__(self, game, config, seed):
        self.game, self.config, self.rng = game, config, random.Random(seed)
        self.is_llm = config["execution"] == "llm_style"
        self.render = renderer(game,config["name"]) if self.is_llm else None
        self.strategy = (_goof_strategy(config["model_path"], config["model_sha256"])
                         if game == "goofspiel" and config["name"] == "GTO" else None)

    def act(self, observation, legal_actions, model=None):
        if self.is_llm:
            if model is None:
                raise ValueError("LLM style requires a logged backend; never rule fallback")
            prompt = self.render._render_prompt(observation,legal_actions)
            raw = model.generate_stage("opponent",prompt)
            if type(raw) is not str or not raw.strip():
                if not legal_actions:
                    raise ValueError("Empty opponent response with no legal action")
                action = self.rng.choice(legal_actions)
                return action, prompt, raw if type(raw) is str else "", {
                    "execution": "llm_style", "fallback_used": True,
                    "source": "random_fallback", "fallback_reason": "empty_response",
                }
            try:
                action = parse_style_action(raw,self.game,legal_actions)
            except ValueError as exc:
                if not legal_actions:
                    raise
                return self.rng.choice(legal_actions), prompt, raw, {
                    "execution": "llm_style", "fallback_used": True,
                    "source": "random_fallback", "fallback_reason": "invalid_action_response",
                    "error_type": type(exc).__name__,
                }
            return action,prompt,raw,{"execution":"llm_style","fallback_used":False}
        if self.strategy is None:
            return self.rng.choice(legal_actions),"","",{"execution":"local_random","fallback_used":False}
        from methods.goofspiel_cfr import hand_to_mask
        p=observation.player_id
        key=(p,hand_to_mask(set(observation.private_info["hand"])),
             tuple(observation.public_info["outcome_history"]),tuple(observation.private_info["my_bids"]),
             tuple(e["p1_card"] if p==0 else e["p0_card"] for e in observation.history if e.get("p0_card") is not None),
             tuple(observation.public_info["prize_sequence"]))
        row=self.strategy.get(key);cards=sorted(a.amount for a in legal_actions)
        if (not isinstance(row,dict) or set(row)!=set(cards)
                or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in row.values())):
            raise ValueError("Missing or invalid Goofspiel GTO row; fallback forbidden")
        total=sum(row.values())
        if not math.isfinite(total) or total<=0:
            raise ValueError("Invalid Goofspiel GTO probability mass")
        draw=self.rng.random()*total;running=0.0;chosen=cards[-1]
        for card in cards:
            running+=row[card]
            if draw<running:
                chosen=card;break
        return next(a for a in legal_actions if a.amount==chosen),"","",{"execution":"local_cfr","fallback_used":False}
