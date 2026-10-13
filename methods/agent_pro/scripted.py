"""Stateless legal fixtures, not a learned model or evidence of effectiveness."""

import json

from base.interfaces import BaseModel
from base.types import Action
from .method import OUTPUT_INSTRUCTIONS, RETURN_FORMAT, action_token
from .policy import canonical, digest, strict_json
from .retry import RETRY_STAGES, read_correction_context
from .reflection_retry import LEARNING_RETRY_STAGES, read_context, stage_parts


def read_prompt_context(prompt):
    """Offline fixture/inspection helper, never used to parse model responses.

    Decision requests now have a structured context plus a fixed terminal output
    instruction/template. Reflection requests remain whole JSON documents.
    Reject an unexpected suffix rather than silently ignoring prompt content.
    """
    text = prompt.lstrip()
    _, end = json.JSONDecoder().raw_decode(text)
    suffix = text[end:]
    if suffix and suffix != "\n\n" + OUTPUT_INSTRUCTIONS + "\n\n" + RETURN_FORMAT:
        raise ValueError("Unexpected prompt suffix")
    return strict_json(text[:end])


class ScriptedAgentProModel(BaseModel):
    cache_fingerprint = {"fixture": "agent_pro_v3_string_action"}

    def set_opponent_fixture(self, game, observation, legal):
        self._opponent_action = legal[0]

    def generate(self, prompt):
        return self.generate_stage("decision", prompt)

    def generate_stage(self, stage, prompt):
        self.last_finish_reason = "stop"
        if stage == "opponent":
            a = self._opponent_action
            return canonical({"reasoning": "Offline legal fixture, not a style effectiveness sample",
                              "action": {"type": a.type, "amount": a.amount, **a.data}})
        if stage in LEARNING_RETRY_STAGES:
            data, _ = read_context(prompt, stage)
            stage, _ = stage_parts(stage)
        else:
            data = (read_correction_context(prompt, stage) if stage in RETRY_STAGES
                    else read_prompt_context(prompt))
        if stage == "decision" or stage in RETRY_STAGES:
            actions = data["legal_actions"]
            revised = "Offline candidate" in data["policy"]["strategy"]
            priority = ({"call": 0, "check": 1, "challenge": 2, "bid": 3,
                         "bet": 4, "raise": 5, "fold": 6} if revised else
                        {"fold": 0, "challenge": 1, "check": 2, "call": 3,
                         "bid": 4, "bet": 5, "raise": 6})
            action = min(actions, key=lambda a: (priority[a["type"]],
                         -a["amount"] if revised else a["amount"],
                         a["data"].get("quantity", 0), a["data"].get("face", 0)))
            return canonical({"self_belief": "Offline fixture: consider own legal resources.",
                              "world_belief": "Opponent intent is unknown; visible evidence only.",
                              "action": action_token(data["game"], Action(**action))})
        if stage == "calibration":
            return canonical({k: "Offline fixture: a loss alone does not establish a belief error."
                              for k in ("correctness", "consistency", "rationality", "reasons")})
        if stage == "proposal":
            policy = dict(data["policy"])
            policy["strategy"] = "Offline candidate " + digest({
                "parent": policy, "previous": data["previous_candidates"]}) + ". Prefer a legal passive response."
            return canonical(policy)
        raise ValueError("Unknown Agent-Pro fixture stage")