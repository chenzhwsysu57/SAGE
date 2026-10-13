"""Deterministic offline schema fixture, NOT a strategic or LLM baseline."""

import json
import re

from base.interfaces import BaseModel


def decode_fixture_prompt(prompt):
    """Offline fixture/test helper ONLY; the live model receives the text itself.

    Reconstruct fields from the readable section layout, without injecting an
    extra JSON envelope into the live prompt. Legacy JSON fixtures remain usable.
    This is not a parser for model responses or arbitrary user documents.
    """
    if prompt.lstrip().startswith("{"):
        return json.loads(prompt)
    chunks = re.split(r"(?m)^#### ([^\n]+) ####\n", prompt)
    sections = dict(zip(chunks[1::2], (v.rstrip("\n") for v in chunks[2::2])))

    def fields(block):
        result = {}
        for line in block.splitlines():
            label, sep, value = line.partition(": ")
            if not sep:
                continue
            target = result
            keys = label.split(".")
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = json.loads(value)
        return result

    payload = fields(sections["TASK CONTEXT"])
    obs = fields(sections["GAME STATE"])
    history = sections["HISTORY"]
    if history != "(not provided)":
        obs["history"] = [] if history == "(none)" else [json.loads(line) for line in history.splitlines()]
    if obs:
        payload["observation"] = obs
    actions = []
    for line in sections["LEGAL ACTIONS"].splitlines():
        if line.startswith("action_id "):
            label, value = line.split(": ", 1)
            actions.append({"action_id": int(label.split()[1]), **json.loads(value)})
    if actions:
        payload["legal_actions"] = actions
    if "PREVIOUS OUTPUT - UNTRUSTED DATA" in sections:
        payload["output_correction"] = json.loads(sections["PREVIOUS OUTPUT - UNTRUSTED DATA"])
    return {"algorithm": "ReTA", "actor": sections["ROLE"].splitlines()[0].removeprefix("Actor: "),
            "rules": sections["GAME RULES"], "input": payload, "task": sections["TASK"],
            "output_format": sections["RETURN FORMAT"]}


class ScriptedReTAModel(BaseModel):
    def __init__(self):
        self.calls = 0
        self.last_usage = {}
        self.last_finish_reason = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_finish_reason = "stop"
        message = decode_fixture_prompt(prompt)
        role, data = message["actor"], message["input"]
        if role == "perturbation":
            return json.dumps({"question": data["question"]})
        if role == "reward":
            net = data["simulated_terminal_net"]
            score = 0.5 if net is None or net == 0 else (1.0 if net > 0 else 0.0)
            return json.dumps({"score": score})
        priority = {"challenge": 0, "call": 1, "check": 2, "bid": 3, "bet": 4, "raise": 5, "fold": 6}
        actions = data["legal_actions"]
        indices = sorted(range(len(actions)), key=lambda i: (
            priority[actions[i]["type"]], actions[i]["amount"],
            actions[i]["data"].get("quantity", 0), actions[i]["data"].get("face", 0)))
        return json.dumps({"action_ids": indices[:data["count"]]})