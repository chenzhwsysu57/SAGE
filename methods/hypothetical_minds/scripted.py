"""Deterministic offline protocol fixture. NOT a real LLM or efficacy baseline."""

import json

from base.interfaces import BaseModel
from .protocol import canonical


class ScriptedHMModel(BaseModel):
    def __init__(self):
        self.calls = 0
        self.last_usage = {}
        self.last_finish_reason = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_finish_reason = "stop"
        self.last_usage = {}
        row = json.loads(prompt)
        stage = row["stage"]
        if stage == "hypothesis":
            return canonical({"hypothesis": "Offline fixture: opponent may repeat low-cost legal moves; revise this hypothesis when predictions fail."})
        if stage == "predict":
            return canonical({"predictions": {k: v[0] if v else None
                                               for k, v in row["response_options"].items()}})
        if stage == "plan":
            return canonical({"strategy": "Offline fixture: choose a legal conservative move while collecting public evidence."})
        if stage == "action":
            legal = row["valid_actions"]
            choice = next((v for v in ("check", "call", "challenge") if v in legal), legal[0])
            return canonical({"action": choice})
        if stage == "opponent_action":
            legal = row["valid_actions"]
            if not legal:
                raise ValueError("Opponent fixture requires legal actions")
            return canonical({"action": legal[0]})
        raise ValueError("Unknown HM stage")