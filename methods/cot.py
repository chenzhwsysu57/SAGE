"""Chain of Thought method: think step by step, then act."""

from __future__ import annotations

import json

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
from methods.helpers import (
    build_example, build_rule_section, history_to_text, parse_action_response,
    render_action_descs,
)


class CoTMethod(BaseMethod):
    """Chain of Thought: reason freely, then output your action.

    Args:
        name: Display name.
        lang: 'en' or 'cn' for prompt language.
    """

    def __init__(self, name: str = "CoT", lang: str = "en"):
        self.name = name
        self.method_type = "cot"
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
                "请先按照你的角色/规则思考当前局势，再给出你的动作。",
                f"用JSON格式输出，示例: {example}",
                "",
                "现在先思考，再输出动作:",
            ])
        else:
            parts.extend([
                "",
                "First think according to your role/rule and the current situation.",
                "Then reason about the best strategy and output your action.",
                f"Use JSON format, example: {example}",
                "",
                "Now think, then act:",
            ])
        return "\n".join(parts)
