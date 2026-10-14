"""ReAct method: structured reasoning before action."""

from __future__ import annotations

import json

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
from methods.helpers import (
    build_example, build_rule_section, history_to_text, parse_action_response,
    render_action_descs,
)


class ReActMethod(BaseMethod):
    """Structured reasoning: think step by step, then act.

    Args:
        name: Display name.
        lang: 'en' or 'cn' for prompt language.
    """

    def __init__(self, name: str = "ReAct", lang: str = "en"):
        self.name = name
        self.method_type = "react"
        self.lang = lang

    def act(
        self, observation: Observation, legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple[Action, str, str]:
        prompt = self._render_prompt(observation, legal_actions)
        raw_response = model.generate(prompt)
        action = parse_action_response(raw_response, observation.player_id, legal_actions)
        return action, prompt, raw_response

    def _render_prompt(
        self, obs: Observation, legal_actions: list[Action],
    ) -> str:
        action_descs = render_action_descs(legal_actions)
        hist_text = history_to_text(obs.history, max_entries=15)
        example = json.dumps(build_example(legal_actions))

        parts = [obs.rules_text, "", "--- Game State ---", obs.state_text]

        if hist_text:
            parts.extend(["", "--- Recent Events ---", hist_text])

        parts.extend(["", "Legal actions:", "\n".join(action_descs)])

        if obs.role_guidance:
            parts.extend(build_rule_section(obs.role_guidance))

        if self.lang == "cn":
            parts.extend([
                "",
                "请先按照你的角色/规则思考:",
                "1. 评估当前局势和你的处境",
                "2. 推测其他玩家可能的行为或意图",
                "3. 基于分析决定最佳动作",
                "",
                "然后用JSON格式输出你的动作，包含'action'字段",
                f"示例: {example}",
                "",
                "现在先思考，再行动:",
            ])
        else:
            parts.extend([
                "",
                "First think according to your role/rule:",
                "1. Evaluate your position given the current state",
                "2. Infer what your opponent might have or be planning",
                "3. Decide the best action based on your analysis",
                "",
                "Output your action as a JSON object with an 'action' field",
                f"Example: {example}",
                "",
                "Now think, then act:",
            ])
        return "\n".join(parts)
