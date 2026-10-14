"""Leduc-specific ToM method with poker prompt and CFR tooling."""

from __future__ import annotations

import json
from typing import Any

from base.case_memory import render_case_memory_block, select_case_memories
from base.types import Action, Observation
from methods.helpers import (
    build_example, build_rule_section, extract_json, history_to_text,
    parse_action_response, render_action_descs,
)
from methods.sage.tom import ToMMethod
from tools.leduc.cfr_tools import CFRTools
from base.ablation import AblationConfig


class LeducToMMethod(ToMMethod):
    """Leduc Hold'em prompt specialization for ToM."""

    def __init__(self, *args, ablation: AblationConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation = ablation or AblationConfig()
        self._cfr_tools: CFRTools | None = None

    def _render_prompt_context(
        self,
        observation: Observation,
        legal_actions: list[Action],
        extra_sections: list[str],
    ) -> tuple[list[str], list[str], str]:
        action_descs = render_action_descs(legal_actions)
        hist_text = history_to_text(observation.history, max_entries=15)
        example = json.dumps(build_example(legal_actions))

        parts: list[str] = []

        parts.append("\n#### GAME RULES ####\n")
        parts.append(observation.rules_text)

        parts.append("\n#### GAME STATE ####\n")
        parts.append(observation.state_text)
        if hist_text:
            parts.append(f"\n{hist_text}")

        opp_sections: list[str] = []
        opponent_ids = self._opponent_ids(observation)
        if opponent_ids:
            for opp_id in opponent_ids:
                opp_sections.extend(self._render_opponent_status(opp_id))
        else:
            opp_sections.extend([
                "Opponent model status:",
                "  Opponents detected: none yet.",
                "  Observed actions: 0",
                "  Guidance: no opponent-specific exploit is justified yet; use robust baseline play.",
            ])

        for section in extra_sections:
            if section.startswith("--- Opponent P"):
                opp_sections.append(f"\n{section[4:]}")
            else:
                opp_sections.append(section)

        if not self.ablation.no_nash:
            cfr_section = self._render_cfr_section(observation, legal_actions)
            if cfr_section:
                parts.append("\n#### GTO BASELINE (CFR Nash) ####\n")
                parts.append(cfr_section)

        if observation.role_guidance:
            parts.extend(build_rule_section(observation.role_guidance))

        parts.append("\n#### OPPONENT MODELS ####\n")
        parts.extend(opp_sections)

        return parts, action_descs, example

    def _self_evolve_memory_text(
        self,
        player_id: int | None = None,
        limit: int = 3,
    ) -> str:
        return render_case_memory_block(
            self.memory,
            game="leduc",
            position=f"P{player_id}" if player_id is not None else "",
            limit=limit,
        )

    def _self_evolve_output_example(
        self,
        legal_actions: list[Action],
    ) -> str:
        example = build_example(legal_actions)
        example = {
            "reasoning": example.get("reasoning", "step by step analysis of the current situation"),
            "action": example.get("action", {"type": "<action_type>"}),
        }
        return json.dumps(example, ensure_ascii=False)

    def _build_self_evolve_memory_block(
        self,
        player_id: int,
    ) -> list[str]:
        memory_text = self._self_evolve_memory_text(player_id)
        if not memory_text:
            return []
        return ["", memory_text]

    def _trajectory_before_predict(self, step) -> str:
        """Render the available action/state order before opponent prediction."""
        history = step.observation.history
        current_street = str(step.observation.public_info.get("street", "preflop"))

        def street_name(raw: Any) -> str:
            if raw in {1, "1", "flop"}:
                return "flop"
            return "preflop"

        def action_token(entry: dict[str, Any]) -> str:
            player = entry.get("player", "?")
            action = str(entry.get("action", "?"))
            amount = entry.get("amount", None)
            amount_text = f"{amount}" if amount not in {None, 0, "0"} else ""
            return f"P{player}:{action}{amount_text}"

        def state_text(text: str) -> str:
            return " ".join(str(text).split())

        tokens: list[str] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            street = street_name(entry.get("street", 0))
            tokens.append(f"action: {action_token(entry)}@{street}")
            tokens.append(f"state: after {street} action")

        current = state_text(step.observation.state_text)
        if current:
            tokens.append(f"state: {current}")
        else:
            tokens.append(f"state: current:P{step.player_id}@{street_name(current_street)}")
        return " | ".join(tokens)

    def _render_prompt(
        self,
        observation: Observation,
        legal_actions: list[Action],
        extra_sections: list[str],
    ) -> str:
        parts, action_descs, example = self._render_prompt_context(
            observation, legal_actions, extra_sections,
        )

        memory_block = self._build_self_evolve_memory_block(observation.player_id)
        if memory_block:
            parts.extend(["", *memory_block, ""])

        parts.append("\n#### YOUR MOVE ####\n")

        if self.use_cfr:
            if self.lang == "cn":
                parts.extend([
                    "目标：针对当前对手，最大化筹码收益或最小化筹码损失。",
                    "",
                    "如果上方提供了任何人格/角色信息，先按照那个角色去思考。",
                    "再结合你的角色/规则和上方模型分析思考，决定最终动作。",
                    "",
                    "GTO 是理论最优，但对手会偏离 GTO，这些偏差就是你获利的机会。",
                    "目标是找到比 GTO baseline 更好的剥削策略。",
                    "",
                    "推理顺序：",
                    "1. 先根据上方 CFR/GTO 策略指导，建立当前局面的 baseline 行动倾向。",
                    "2. 结合你的手牌、公共牌、底池、主动权和合法动作，判断 baseline 下自然的行动线。",
                    "3. 再结合对手模型进行推演：对手风格概率、近期动作和可能偏离。",
                    "4. 如果出现 self-evolve memory 块，把它们当成来自相似局面的完整",
                    "   对战 case 与可能的对手行动方式；使用前先核对 history、手牌分布、",
                    "   对手风格是否一致，不要照抄其中的动作——只在相关时把该 case 的",
                    "   未来推演当作生成/修正假设的证据。",
                    "5. 选择期望结果最优的动作。",
                ])
            else:
                parts.extend([
                    "Goal: maximize expected chips or minimize chips lost against",
                    "this specific opponent.",
                    "",
                    "If any persona guidance is provided, think in character",
                    "according to it first.",
                    "Then think according to your role/rule and the opponent models",
                    "above, and decide your final action.",
                    "",
                    "GTO/CFR policy is optimal against a perfect opponent, but your opponent",
                    "may deviate from GTO — reason about them wisely. Opponent tendencies is our opportunity to",
                    "outperform the GTO baseline.",
                    "",
                    "Reason in this order:",
                    "1. Start from the CFR/GTO strategy guidance above as the",
                    "   baseline action tendency for this state.",
                    "2. Use your private card, board, pot, initiative, and legal",
                    "   actions to identify the natural baseline line.",
                    "3. Then adjust from that baseline using the opponent model:",
                    "   style probabilities, recent actions, and likely deviations.",
                    "4. If a self-evolve memory block is present, treat entries as",
                    "   complete cases from similar states with possible opponent",
                    "   action patterns; compare history, hand distribution, and",
                    "   opponent style before relying on them, and never copy their",
                    "   actions — use a case's future projection as evidence for",
                    "   your own hypotheses.",
                    "5. Choose the action with the best expected outcome.",
                ])
        else:
            if self.lang == "cn":
                parts.extend([
                    "目标：针对当前对手，最大化筹码收益或最小化筹码损失。",
                    "",
                    "推理顺序：",
                    "1. 先判断当前手牌、公共牌、底池、主动权和合法动作下的自然行动线。",
                    "2. 再结合对手模型进行推演：对手风格概率、近期动作和可能偏离。",
                    "3. 如果出现 self-evolve memory 块，把它们当成来自相似局面的完整",
                    "   对战 case 与可能的对手行动方式；使用前先核对 history、手牌分布、",
                    "   对手风格是否一致，不要照抄其中的动作——只在相关时把该 case 的",
                    "   未来推演当作生成/修正假设的证据。",
                    "4. 选择期望结果最优的动作。",
                ])
            else:
                parts.extend([
                    "Goal: maximize expected chips or minimize chips lost against",
                    "this specific opponent.",
                    "",
                    "Reason in this order:",
                    "1. Use your private card, board, pot, initiative, and legal",
                    "   actions to identify the natural line for this state.",
                    "2. Then adjust using the opponent model: style probabilities,",
                    "   recent actions, and likely deviations.",
                    "3. If a self-evolve memory block is present, treat entries as",
                    "   complete cases from similar states with possible opponent",
                    "   action patterns; compare history, hand distribution, and",
                    "   opponent style before relying on them, and never copy their",
                    "   actions — use a case's future projection as evidence for",
                    "   your own hypotheses.",
                    "4. Choose the action with the best expected outcome.",
                ])
        parts.extend([
            "",
            "Legal actions:",
            "\n".join(action_descs),
            "",
            "Output as JSON with these fields:",
            self._self_evolve_output_example(legal_actions),
        ])

        return "\n".join(parts)

    def _render_cfr_section(self, observation, legal_actions) -> str:
        """Build CFR Nash equilibrium analysis section for Leduc."""
        if not self.use_cfr:
            return ""
        if self._cfr_renderer is not None:
            return self._cfr_renderer(observation, legal_actions)
        if self._cfr_tools is None:
            self._cfr_tools = CFRTools()
        my_card = observation.private_info.get("card", "")
        community_card = observation.public_info.get("community_card", "")
        street = observation.public_info.get("street", "preflop")
        history = observation.history
        return self._cfr_tools.format_for_prompt(
            my_card, community_card, history, street,
            observation.player_id,
        )

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model,
    ) -> tuple:
        self._dbg(f"Act P{observation.player_id} ({self.name})")
        self._my_id = observation.player_id

        opp_ids = self._opponent_ids(observation)
        self._dbg(f"Opponents from history: {opp_ids}", indent=1)
        for opp_id in opp_ids:
            self._update_from_history(observation, opp_id)
            self._dbg(f"Profile P{opp_id}: {self._load_profile(opp_id).summary()}", indent=2)
            self._dbg(f"Belief P{opp_id}: {self._load_belief(opp_id).to_text()}", indent=2)

        beliefs = {oid: self._load_belief(oid) for oid in opp_ids}
        profiles = {oid: self._load_profile(oid) for oid in opp_ids}

        extra_sections: list[str] = []
        for opp_id in opp_ids:
            if profiles[opp_id].total == 0:
                continue
            if self.ablation.no_belief:
                continue
            stmt = self.modeling_method.statement(opp_id, n_actions=profiles[opp_id].total)
            if stmt:
                extra_sections.append(f"--- Opponent P{opp_id} ---")
                extra_sections.append(stmt)
        if extra_sections:
            self._dbg(f"Modeling method added {len(extra_sections)} line(s)", indent=1)

        prompt = self._render_prompt(
            observation, legal_actions, extra_sections,
        )
        self._dbg(f"Prompt ({len(prompt)} chars):", indent=1)
        if self.verbose:
            for line in prompt.split("\n"):
                print(f"    | {line}")

        raw_response = model.generate(prompt)
        self._dbg(f"LLM response: {raw_response[:200]}", indent=1)

        action = parse_action_response(raw_response, observation.player_id, legal_actions)
        self._dbg(f"Parsed action: {action}", indent=1)

        md: dict[str, Any] = {}
        for opp_id in opp_ids:
            p = profiles[opp_id]
            b = beliefs[opp_id]
            md[f"opp_{opp_id}"] = {
                "n_actions": p.total,
                "actions": dict(p.actions),
                "distribution": {a: round(b.predict(a), 3) for a in b.counts},
            }
            if p.total > 0:
                style_posterior_fn = getattr(self.modeling_method, "style_posterior", None)
                if callable(style_posterior_fn):
                    posterior = style_posterior_fn(opp_id)
                    if posterior:
                        md["beliefs"] = {
                            str(k): round(float(v), 4) for k, v in posterior.items()
                        }
                        top = max(posterior, key=lambda k: float(posterior[k]))
                        md["persona"] = str(top)
                        md["confidence"] = round(float(posterior[top]), 4)
                        md["n_actions"] = p.total
        md["self_evolve"] = {
            "memory_records": len(select_case_memories(
                self.memory,
                game="leduc",
                position=f"P{observation.player_id}",
            )),
        }

        return action, prompt, raw_response, md

    def reflect(
        self, trajectory, model, game_name: str = "",
    ) -> list[str]:
        return super().reflect(trajectory, model, game_name)
