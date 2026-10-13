"""Stateless offline flow fixtures, not an implementation of an opponent model."""

import json

from base.interfaces import BaseModel


class ScriptedEMOModel(BaseModel):
    def __init__(self):
        self.calls = 0
        self.last_usage = {}
        self.last_finish_reason = None
        self.opponent_fixture = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_finish_reason = "stop"
        self.last_usage = {}  # No fabricated provider usage.
        if self.opponent_fixture is not None and prompt == self.opponent_fixture[0]:
            return json.dumps(self.opponent_fixture[1])
        value = json.loads(prompt)
        stage = value["stage"]
        if stage == "init" or "refine" in stage:
            return "None"
        if stage == "cot_3":
            priority = {"challenge": 0, "call": 1, "check": 2, "bid": 3,
                        "bet": 4, "raise": 5, "fold": 6}
            legal = value["context"]["legal_actions"]
            action = min(legal, key=lambda a: (priority.get(a["type"], 9), a["amount"],
                         a["data"].get("quantity", 0), a["data"].get("face", 0)))
            return json.dumps(action)
        return f"Offline {stage} fixture; no inferred private information."