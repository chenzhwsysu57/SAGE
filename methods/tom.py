"""Theory of Mind method: model opponents through a pluggable modeling method.

Generic ToM that tracks:
  - OpponentProfile: per-opponent action counts
  - DirichletBelief: per-opponent action distribution (Bayesian posterior)

Opponent modeling is delegated to a BaseModelingMethod instance.

Usage:
    method = StatisticalProfileMethod(memory=BaseMemory())
    tom = ToMMethod(name="ToM", modeling_method=method)
"""

from __future__ import annotations

import json

from base.interfaces import BaseMethod, BaseModel
from base.memory import BaseMemory
from base.types import Action, Observation, Trajectory
from methods.helpers import (
    build_example, build_rule_section, history_to_text, parse_action_response,
    render_action_descs,
)
from methods.belief import OpponentProfile, DirichletBelief
from modeling.base import BaseModelingMethod


class ToMMethod(BaseMethod):
    """Opponent modeling through a pluggable modeling method.

    Generic: no game-specific knowledge.
    Inject a BaseModelingMethod for the modeling algorithm.

    Per-opponent tracking (maintained in memory):
      - profile_{opp_id}: OpponentProfile (action counts)
      - belief_{opp_id}: DirichletBelief (action distribution)
    """

    def __init__(
        self,
        name: str = "ToM",
        lang: str = "en",
        memory: BaseMemory | None = None,
        modeling_method: BaseModelingMethod | None = None,
        verbose: bool = False,
        use_cfr: bool = True,
        cfr_renderer: callable | None = None,
    ):
        self.name = name
        self.method_type = "tom"
        self.lang = lang
        self.verbose = verbose
        self.memory = memory or BaseMemory()
        self.modeling_method = modeling_method or BaseModelingMethod()
        self._seen_hist_len: dict[int, int] = {}
        self._my_id: int | None = None
        self.use_cfr = use_cfr
        self._cfr_renderer = cfr_renderer

    def _dbg(self, msg: str, indent: int = 0) -> None:
        if self.verbose:
            prefix = "  " * indent
            print(f"[ToM] {prefix}{msg}")

    # ── Opponent detection ──────────────────────────────────────────

    def _opponent_ids(self, observation: Observation) -> list[int]:
        my_id = observation.player_id
        opps: set[int] = set()
        for h in observation.history:
            p = h.get("player")
            if p is not None and p != my_id:
                opps.add(int(p))
        if not opps and my_id in (0, 1):
            opps.add(1 - my_id)
        return sorted(opps)

    # ── Memory persistence ──────────────────────────────────────────

    def _profile_key(self, opp_id: int) -> str:
        return f"profile_{opp_id}"

    def _belief_key(self, opp_id: int) -> str:
        return f"belief_{opp_id}"

    def _load_profile(self, opp_id: int) -> OpponentProfile:
        key = self._profile_key(opp_id)
        if key not in self.memory:
            self.memory.set(key, OpponentProfile().to_dict())
        raw = self.memory.get(key)
        return OpponentProfile.from_dict(raw) if isinstance(raw, dict) else OpponentProfile()

    def _save_profile(self, opp_id: int, profile: OpponentProfile) -> None:
        self.memory.set(self._profile_key(opp_id), profile.to_dict())

    def _load_belief(self, opp_id: int) -> DirichletBelief:
        key = self._belief_key(opp_id)
        if key not in self.memory:
            self.memory.set(key, DirichletBelief([]).to_dict())
        raw = self.memory.get(key)
        return DirichletBelief.from_dict(raw) if isinstance(raw, dict) else DirichletBelief([])

    def _save_belief(self, opp_id: int, belief: DirichletBelief) -> None:
        self.memory.set(self._belief_key(opp_id), belief.to_dict())

    # ── History scanning ────────────────────────────────────────────

    def _update_from_history(self, observation: Observation, opp_id: int) -> None:
        profile = self._load_profile(opp_id)
        belief = self._load_belief(opp_id)

        seen = self._seen_hist_len.get(opp_id, 0)
        history = observation.history
        if len(history) < seen:
            seen = 0

        for i in range(seen, len(history)):
            entry = history[i]
            if not isinstance(entry, dict):
                continue
            if entry.get("player") != opp_id:
                continue
            atype = entry.get("action", "")
            if atype:
                profile.record(atype)
                belief.observe(atype)

        self._seen_hist_len[opp_id] = len(history)
        self._save_profile(opp_id, profile)
        self._save_belief(opp_id, belief)

    # ── Main interface ──────────────────────────────────────────────

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple:
        self._dbg(f"Act P{observation.player_id} ({self.name})")
        self._my_id = observation.player_id

        # 1. Update models from any new opponent actions
        opp_ids = self._opponent_ids(observation)
        self._dbg(f"Opponents from history: {opp_ids}", indent=1)
        for opp_id in opp_ids:
            self._update_from_history(observation, opp_id)
            self._dbg(f"Profile P{opp_id}: {self._load_profile(opp_id).summary()}", indent=2)
            self._dbg(f"Belief P{opp_id}: {self._load_belief(opp_id).to_text()}", indent=2)

        # 2. Gather beliefs and profiles
        beliefs = {oid: self._load_belief(oid) for oid in opp_ids}
        profiles = {oid: self._load_profile(oid) for oid in opp_ids}

        # 3. Query modeling method for prompt sections
        extra_sections: list[str] = []
        for opp_id in opp_ids:
            if profiles[opp_id].total == 0:
                continue
            stmt = self.modeling_method.statement(opp_id, n_actions=profiles[opp_id].total)
            if stmt:
                extra_sections.append(f"--- Opponent P{opp_id} ---")
                extra_sections.append(stmt)
        if extra_sections:
            self._dbg(f"Modeling method added {len(extra_sections)} line(s)", indent=1)

        # 4. Render prompt and call model
        prompt = self._render_prompt(observation, legal_actions, extra_sections)
        self._dbg(f"Prompt ({len(prompt)} chars):", indent=1)
        if self.verbose:
            for line in prompt.split("\n"):
                print(f"    | {line}")

        raw_response = model.generate(prompt)
        self._dbg(f"LLM response: {raw_response[:200]}", indent=1)

        action = parse_action_response(raw_response, observation.player_id, legal_actions)
        self._dbg(f"Parsed action: {action}", indent=1)

        # 5. Build method_data
        md: dict = {}
        for opp_id in opp_ids:
            p = profiles[opp_id]
            b = beliefs[opp_id]
            md[f"opp_{opp_id}"] = {
                "n_actions": p.total,
                "actions": dict(p.actions),
                "distribution": {a: round(b.predict(a), 3) for a in b.counts},
            }

        return action, prompt, raw_response, md

    # ── Post-game reflection ────────────────────────────────────────

    def reflect(
        self, trajectory: Trajectory, model: BaseModel, game_name: str = "",
    ) -> list[str]:
        self._dbg(f"Reflect — {game_name}, {len(trajectory.steps)} steps")

        player_ids: set[int] = set()
        for s in trajectory.steps:
            if s.player_id >= 0:
                player_ids.add(s.player_id)

        for pid in sorted(player_ids):
            if pid == self._my_id:
                continue  # skip self — only model opponents
            profile = self._load_profile(pid)
            if profile.total == 0:
                self._dbg(f"  P{pid}: no data, skip", indent=1)
                continue
            self._dbg(f"  P{pid}: {profile.total} actions tracked", indent=1)

            self.modeling_method.update(pid, trajectory, model)
            self._dbg(f"  Updated model for P{pid}", indent=1)

            if hasattr(self.modeling_method, "plot_convergence_chart"):
                chart_path = self.modeling_method.plot_convergence_chart(pid)
                if chart_path:
                    self._dbg(f"  Convergence chart saved to {chart_path}", indent=1)

        return []

    # ── Prompt rendering ────────────────────────────────────────────

    def _render_prompt(
        self,
        observation: Observation,
        legal_actions: list[Action],
        extra_sections: list[str],
    ) -> str:
        action_descs = render_action_descs(legal_actions)
        hist_text = history_to_text(observation.history, max_entries=15)
        example = json.dumps(build_example(legal_actions))

        parts: list[str] = []

        # SECTION 1: GAME RULES
        parts.append("\n#### GAME RULES ####\n")
        parts.append(observation.rules_text)

        # SECTION 2: GAME STATE
        parts.append("\n#### GAME STATE ####\n")
        parts.append(observation.state_text)
        if hist_text:
            parts.append(f"\n{hist_text}")

        # SECTION 3: OPPONENT MODELS
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

        cfr_section = self._render_cfr_section(observation, legal_actions)
        if cfr_section:
            parts.append("\n#### GTO BASELINE (CFR Nash) ####\n")
            parts.append(cfr_section)

        if observation.role_guidance:
            parts.extend(build_rule_section(observation.role_guidance))

        parts.append("\n#### OPPONENT MODELS ####\n")
        parts.extend(opp_sections)

        # SECTION 4: YOUR MOVE
        parts.append("\n#### YOUR MOVE ####\n")

        if self.use_cfr:
            parts.extend(self._decision_guidance())
        else:
            parts.extend(self._decision_guidance_no_cfr())

        parts.extend([
            "",
            "Legal actions:",
            "\n".join(action_descs),
            "",
            f"Output as JSON: {example}",
        ])

        return "\n".join(parts)

    def _decision_guidance(self) -> list[str]:
        if self.lang == "cn":
            return [
                "目标：结合对手模型与 GTO 参考，选择期望结果最优的动作。",
                "",
                "对每个合法动作，分别分析预期结果：",
                "1. GTO 基准：这个动作对均衡对手的期望收益是多少？",
                "2. 对手反应：根据对手已观察到的风格和行动模式，",
                "   他们会如何回应这个动作？",
                "3. 综合结果：考虑不同场景后，这个动作期望收益更高吗？",
                "",
                "比较所有合法动作，选择期望结果最优的。",
            ]
        return [
            "Goal: combine the opponent model with the GTO reference and",
            "choose the action with the best expected outcome.",
            "",
            "For each legal action, analyze the expected outcome:",
            "1. GTO baseline: what is the equilibrium EV of this action?",
            "2. Opponent response: how would this opponent likely respond",
            "   to this action, given their observed tendencies?",
            "3. Net outcome: across different scenarios, will this action",
            "   gain more or lose less?",
            "",
            "Compare all legal actions and choose the one with the best",
            "expected outcome.",
        ]

    def _decision_guidance_no_cfr(self) -> list[str]:
        if self.lang == "cn":
            return [
                "目标：针对当前对手，最大化筹码收益或最小化筹码损失。",
                "",
                "对每个合法动作，分别分析预期结果：",
                "1. 对手反应：根据对手已观察到的风格和行动模式，",
                "   他们会如何回应这个动作？",
                "2. 综合结果：考虑各种场景后，这个动作期望收益更高吗？",
                "",
                "比较所有合法动作，选择期望结果最优的。",
            ]
        return [
            "Goal: maximize expected chips or minimize chips lost against",
            "this specific opponent.",
            "",
            "For each legal action, analyze the expected outcome:",
            "1. Opponent response: how would this opponent likely respond",
            "   to this action, given their observed tendencies?",
            "2. Net outcome: across different scenarios, will this action",
            "   gain more chips or lose fewer chips?",
            "",
            "Compare all legal actions and choose the one with the",
            "best expected outcome.",
        ]

    def _render_opponent_status(self, opp_id: int) -> list[str]:
        profile = self._load_profile(opp_id)
        if profile.total > 0:
            return []  # covered by statement() in extra_sections
        return [
            f"Opponent P{opp_id}: no actions observed yet — no exploit justified.",
        ]

    def _render_cfr_section(self, observation, legal_actions) -> str:
        """Build CFR analysis section for the prompt."""
        if not self.use_cfr or self._cfr_renderer is None:
            return ""
        try:
            return self._cfr_renderer(observation, legal_actions)
        except Exception:
            return ""
