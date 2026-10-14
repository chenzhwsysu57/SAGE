"""Reflexion method: learn from past game outcomes."""

from __future__ import annotations

import json
import re

from base.interfaces import BaseMethod, BaseModel
from base.memory import BaseMemory
from base.types import Action, Observation, Trajectory
from methods.helpers import (
    build_example, build_rule_section, history_to_text, parse_action_response,
    render_action_descs,
)


class ReflexionMethod(BaseMethod):
    """Learn from past games through reflection.

    During play, reflections from previous games are prepended to the
    observation. After each game, the trajectory is analyzed to generate
    new lessons.

    Args:
        name: Display name.
        lang: 'en' or 'cn'.
        memory: Shared cross-game memory.
    """

    def __init__(
        self, name: str = "Reflexion", lang: str = "en",
        memory: BaseMemory | None = None,
    ):
        self.name = name
        self.method_type = "reflexion"
        self.lang = lang
        self.memory = memory or BaseMemory()
        self.buffer: list[dict] = []
        self._last_role: str = ""

    @property
    def reflections(self) -> list[str]:
        return self.memory.get("reflections", [])

    @reflections.setter
    def reflections(self, value: list[str]):
        self.memory.set("reflections", value)

    def clear(self) -> None:
        self.memory.set("reflections", [])
        self.buffer = []

    def act(
        self, observation: Observation, legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple:
        role = observation.private_info.get("role", "")
        if role:
            self._last_role = role
        self.buffer.append({
            "player_id": observation.player_id,
            "phase": observation.public_info.get("phase", ""),
        })

        modified_obs = self._inject_reflections(observation)
        prompt = self._render_prompt(modified_obs, legal_actions)
        raw_response = model.generate(prompt)
        action = parse_action_response(raw_response, observation.player_id, legal_actions)
        md = {"reflections": self.reflections, "count": len(self.reflections)}
        return action, prompt, raw_response, md

    def reflect(
        self, trajectory: Trajectory, model: BaseModel, game_name: str = "",
    ) -> list[str]:
        prompt = self._build_reflection_prompt(trajectory, game_name)
        raw = model.generate(prompt)
        lessons = self._parse_reflections(raw)
        self.reflections.extend(lessons)
        self.buffer = []
        return lessons

    def _inject_reflections(self, obs: Observation) -> Observation:
        if not self.reflections:
            return obs
        prefix = "[Lessons from past games]\n"
        for i, r in enumerate(self.reflections, 1):
            prefix += f"  {i}. {r}\n"
        new_obs = obs.model_copy()
        new_obs.state_text = prefix + "\n" + obs.state_text
        return new_obs

    def _build_reflection_prompt(
        self, trajectory: Trajectory, game_name: str,
    ) -> str:
        my_pid = trajectory.steps[0].player_id if trajectory.steps else 0
        won = trajectory.payout.get(my_pid, 0) > 0

        steps_summary = []
        for s in trajectory.steps:
            action_str = str(s.parsed_action)
            valid_str = "" if s.action_valid else " [INVALID]"
            steps_summary.append(
                f"  Step {s.step} (P{s.player_id}): {action_str}{valid_str}")

        result = "WON" if won else "LOST"
        return (
            f"You just finished a game of {game_name}.\n"
            f"Your role: {self._last_role}.\n"
            f"Result: {result}.\n\n"
            f"Here is what happened:\n" + "\n".join(steps_summary) + "\n\n"
            f"What lessons can you learn from this game?\n"
            f"Output 1-3 concise lessons, one per line."
        )

    def _parse_reflections(self, raw: str) -> list[str]:
        lessons = []
        for line in raw.strip().split("\n"):
            line = re.sub(r"^[\d\-\*\.\s]+", "", line.strip())
            if len(line) > 15:
                lessons.append(line)
        return lessons[:5]

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
                "请先基于以上信息、过去的教训和你的角色/规则思考，再给出动作。",
                f"用JSON格式输出，示例: {example}",
                "",
                "现在先思考，再输出动作:",
            ])
        else:
            parts.extend([
                "",
                "First think according to the above information, past lessons,",
                "and your role/rule before outputting your action.",
                f"Use JSON format, example: {example}",
                "",
                "Now think, then act:",
            ])
        return "\n".join(parts)
