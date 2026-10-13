"""Model adapter: structured NLP policy in, trusted fixed evaluator source out."""
from __future__ import annotations

import json

from methods.strategist.evolution import canonical
from methods.strategist.nlp_strategy import compile_policy, extract_policy, parse_policy, policy_contract


def _replace_sources(value):
    if type(value) is dict:
        return {k: _replace_sources(v) for k, v in value.items()}
    if type(value) is list:
        return [_replace_sources(v) for v in value]
    if type(value) is str and value.startswith("# NLP_POLICY_V1 "):
        return {"nlp_policy": extract_policy(value)}
    return value


class NLPPolicyBackend:
    """Wrap any existing text backend without executing model-authored code."""
    cache_fingerprint = "strategist-nlp-structured-fixed-executor-v1"

    def __init__(self, backend):
        self.backend = backend
        self.last_usage, self.last_finish_reason = {}, None

    @staticmethod
    def outbound_prompt(prompt: str) -> str:
        request = json.loads(prompt)
        stage, game = request["true_stage"], request["game"]
        outbound = _replace_sources(request)
        outbound.pop("contract", None)
        if stage in ("seed_code", "code"):
            outbound["representation_change"] = policy_contract(game)
            outbound["instruction"] = (
                "Generate the initial structured natural-language strategy." if stage == "seed_code" else
                "Apply the selected idea to the parent NLP policy. Return a complete replacement policy."
            )
        else:
            outbound["representation_change"] = {
                "instruction": "Reason about the parent natural-language policy and training feedback. Never propose or discuss Python code.",
                "policy_schema": policy_contract(game),
            }
        return canonical(outbound)

    def generate(self, prompt: str) -> str:
        request = json.loads(prompt)
        stage, game = request["true_stage"], request["game"]
        raw = self.backend.generate(self.outbound_prompt(prompt))
        self.last_usage = getattr(self.backend, "last_usage", {})
        self.last_finish_reason = getattr(self.backend, "last_finish_reason", None)
        if stage not in ("seed_code", "code"):
            return raw
        try:
            return compile_policy(parse_policy(raw, game))
        except ValueError:
            # Preserve the exact invalid model text for the existing durable
            # correction loop. It cannot execute: ValueHeuristic rejects it.
            return raw

    def redact(self, text):
        redact = getattr(self.backend, "redact", None)
        return redact(text) if callable(redact) else text

    def bind_run_directory(self, directory):
        bind = getattr(self.backend, "bind_run_directory", None)
        if callable(bind):
            bind(directory)

    def close(self):
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


class ScriptedNLPModel:
    """Deterministic mechanics fixture; not evidence of policy quality."""
    model_name = "strategist-nlp-scripted-v1"
    cache_fingerprint = "strategist-nlp-scripted-v1"

    def __init__(self):
        self.calls, self.last_usage, self.last_finish_reason = 0, {}, None

    def generate(self, prompt):
        req = json.loads(prompt)
        self.calls += 1
        self.last_usage, self.last_finish_reason = {}, "stop"
        stage = req["true_stage"]
        if stage == "reflection":
            return "The current strategy should balance public score, private resources, pressure, progression, and uncertainty."
        if stage == "idea":
            return "Increase resource preservation under high pressure while retaining a small public-score adjustment."
        game = req["game"]
        iteration = req.get("iteration", 0)
        return canonical({"schema": "strategist.nlp_policy.v1", "game": game,
                          "strategy": f"Use public score and private resources conservatively; revision {iteration}.",
                          "weights": {"score": 0.6, "resources": 0.4 + 0.1*iteration,
                                      "pressure": -0.2, "progress": 0.1, "risk": 0.5},
                          "bias": 0.0})
