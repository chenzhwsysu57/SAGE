"""Goofspiel ToM method — q-window EMA persona classification + self-evolve.

This method:
1. Tracks opponent's bids per round via observation history
2. Maintains q-window records for per-window posterior q_n
3. EMA-smooths: p_n = alpha * p_{n-1} + (1-alpha) * q_n
4. Prompt includes opponent model and self-evolve memory when available
5. Post-game reflect is a no-op for self-evolve memory

Usage:
    method = GoofspielToMMethod(name="ToM", lang="en", memory=BaseMemory())
"""

from __future__ import annotations

import json
import math
from typing import Any

from base.case_memory import render_case_memory_block, select_case_memories
from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation, Trajectory
from methods.helpers import (
    build_example, history_to_text, parse_action_response,
    render_action_descs,
)
from methods.goofspiel_cfr_method import GoofspielCFRMethod
from modeling.goofspiel.bayes_persona import BayesPersonaModel
from base.ablation import AblationConfig


class GoofspielToMMethod(BaseMethod):
    """Theory of Mind method for Goofspiel with q-window EMA + self-evolve.

    Per-round belief updates from observation history via window-Bayes + EMA.
    Key-node two-phase: separate judgment call → branched main prompt.
    Self-evolve memory persists across games when memory is provided.
    """

    _SELF_EVOLVE_KEY = "goofspiel_self_evolve_records"

    def __init__(
        self,
        name: str = "ToM",
        lang: str = "en",
        verbose: bool = False,
        use_cfr: bool = True,
        cfr_model_path: str | None = None,
        bayes_model_path: str | None = None,
        memory=None,
        classification_mode: str = "q_window_ema",
        q_window: int = 10,
        q_alpha: float = 0.85,
        ablation: AblationConfig | None = None,
    ):
        self.name = name
        self.method_type = "tom"
        self.lang = lang
        self.verbose = verbose
        self.use_cfr = use_cfr
        self.memory = memory
        self.ablation = ablation or AblationConfig()
        self.classification_mode = classification_mode
        self.q_window = max(1, int(q_window))
        self.q_alpha = max(0.0, min(1.0, float(q_alpha)))

        # Bayes persona model (likelihood tables P(bid|prize,persona))
        self.bayes = BayesPersonaModel(bayes_model_path)

        # CFR baseline for GTO analysis section
        self._cfr_method = None
        if use_cfr:
            self._cfr_method = GoofspielCFRMethod(
                model_path=cfr_model_path, name="CFR",
            )

        # Per-opponent round records for q-window computation
        self._all_records: dict[int, list[dict]] = {}
        self._current_game: int = 0
        self._prev_hist_len: dict[int, int] = {}
        self._posterior_cache: dict[int, dict[str, float]] = {}
        self._belief_history: dict[int, list[dict]] = {}

        # Length-based incremental tracking
        self._seen_hist_len: dict[int, int] = {}

    def _opponent_id(self, observation: Observation) -> int:
        return 1 - observation.player_id

    def _dbg(self, msg: str) -> None:
        if self.verbose:
            print(f"[GoofspielToM] {msg}")

    # ── Memory helpers ─────────────────────────────────────────────────

    def _memory_get(self, key: str, default: Any = None) -> Any:
        if self.memory is None:
            return default
        getter = getattr(self.memory, "get", None)
        if callable(getter):
            return getter(key, default)
        return default

    def _memory_set(self, key: str, value: Any) -> None:
        if self.memory is None:
            return
        setter = getattr(self.memory, "set", None)
        if callable(setter):
            setter(key, value)

    def _load_self_evolve_records(self) -> list[dict[str, Any]]:
        raw = self._memory_get(self._SELF_EVOLVE_KEY, [])
        if not isinstance(raw, list):
            return []
        return [r for r in raw if isinstance(r, dict)]

    def _save_self_evolve_records(self, records: list[dict[str, Any]]) -> None:
        self._memory_set(self._SELF_EVOLVE_KEY, records)

    @staticmethod
    def _self_evolve_game_label(record: dict[str, Any], fallback: int) -> str:
        raw = record.get("game_number", record.get("game", ""))
        if isinstance(raw, int):
            return str(raw)
        if isinstance(raw, str) and raw.strip().isdigit():
            return str(int(raw.strip()))
        return str(fallback)

    # ── Round record tracking ──────────────────────────────────────────

    def _update_records(self, observation: Observation, opp_id: int) -> None:
        """Extract round_result entries from history for q-window computation."""
        history = observation.history or []
        prev_len = self._prev_hist_len.get(opp_id, 0)

        # New game detection: history got shorter
        if len(history) < prev_len:
            self._current_game += 1
            self._prev_hist_len[opp_id] = 0
            prev_len = 0

        my_id = observation.player_id
        for i in range(prev_len, len(history)):
            entry = history[i]
            if not isinstance(entry, dict) or entry.get("action") != "round_result":
                continue
            opp_bid = entry["p1_card"] if my_id == 0 else entry["p0_card"]
            prize = entry["prize"]
            if opp_id not in self._all_records:
                self._all_records[opp_id] = []
            self._all_records[opp_id].append({
                "game": self._current_game,
                "prize": prize,
                "opp_bid": opp_bid,
                "round": entry.get("round", 0),
            })

        self._prev_hist_len[opp_id] = len(history)

    def _process_new_rounds(self, observation: Observation, opp_id: int) -> None:
        """Process newly completed rounds for BayesPersonaModel round count."""
        history = observation.history
        my_id = observation.player_id
        seen = self._seen_hist_len.get(opp_id, 0)

        if len(history) < seen:
            seen = 0

        for i in range(seen, len(history)):
            entry = history[i]
            if entry.get("action") != "round_result":
                continue
            prize = entry["prize"]
            opp_bid = entry["p1_card"] if my_id == 0 else entry["p0_card"]
            self.bayes.observe_round(opp_id, prize, opp_bid)

        self._seen_hist_len[opp_id] = len(history)

    def _update_beliefs(self, observation: Observation, opp_id: int) -> None:
        """Combined update: Bayes round count + q-window records."""
        self._process_new_rounds(observation, opp_id)
        self._update_records(observation, opp_id)

    # ── Q-window EMA classification ────────────────────────────────────

    def _initial_posterior(self, opp_id: int) -> dict[str, float]:
        n_p = len(self.bayes.personas)
        return {pn: 1.0 / n_p for pn in self.bayes.personas}

    def _compute_window_posterior(self, opp_id: int) -> dict[str, float]:
        """Compute q_n from last q_window records via Bayes likelihood."""
        records = self._all_records.get(opp_id, [])
        if len(records) < 2:
            return {}

        window = records[-self.q_window:]
        log_scores: dict[str, float] = {pn: 0.0 for pn in self.bayes.personas}

        for r in window:
            prize = r["prize"]
            opp_bid = r["opp_bid"]
            for pn in self.bayes.personas:
                prob = self.bayes.models[pn].get(prize, {}).get(opp_bid, 1e-10)
                log_scores[pn] += math.log(max(float(prob), 1e-10))

        # Softmax
        vals = list(log_scores.values())
        mx = max(vals) if vals else 0.0
        exps = {k: math.exp(v - mx) for k, v in log_scores.items()}
        total = sum(exps.values()) or 1.0
        return {k: v / total for k, v in exps.items()}

    def _update_persona_posterior(
        self, opp_id: int, likelihood: dict[str, float],
    ) -> dict[str, float]:
        """EMA: p_n = alpha * p_{n-1} + (1-alpha) * q_n."""
        if not likelihood:
            return {}
        if self.classification_mode != "q_window_ema":
            self._posterior_cache[opp_id] = dict(likelihood)
            return self._posterior_cache[opp_id]

        prior = self._posterior_cache.get(opp_id) or self._initial_posterior(opp_id)
        posterior = {
            name: self.q_alpha * prior.get(name, 0.0)
            + (1.0 - self.q_alpha) * likelihood.get(name, 0.0)
            for name in likelihood
        }
        total = sum(posterior.values())
        if total > 0:
            posterior = {name: v / total for name, v in posterior.items()}
        self._posterior_cache[opp_id] = posterior
        return posterior

    def _classify_opponent(
        self, opp_id: int,
    ) -> tuple[str | None, float, dict[str, float]]:
        """Return (top_persona, confidence, all_beliefs) via q-window EMA."""
        records = self._all_records.get(opp_id, [])
        if len(records) < 2:
            return None, 0.0, {}

        q_n = self._compute_window_posterior(opp_id)
        beliefs = self._update_persona_posterior(opp_id, q_n)
        top = max(beliefs, key=beliefs.get) if beliefs else None
        return top, beliefs.get(top, 0.0) if top else 0.0, beliefs

    # ── Belief history for prompt display ──────────────────────────────

    def _record_belief_snapshot(self, opp_id: int, beliefs: dict[str, float]) -> None:
        if not beliefs:
            return
        if opp_id not in self._belief_history:
            self._belief_history[opp_id] = []
        n_rounds = len(self._all_records.get(opp_id, []))
        snapshot = {"game": self._current_game, "rounds": n_rounds, "beliefs": dict(beliefs)}
        prev = self._belief_history[opp_id][-1] if self._belief_history[opp_id] else None
        if prev and prev.get("game") == snapshot["game"]:
            self._belief_history[opp_id][-1] = snapshot
        else:
            self._belief_history[opp_id].append(snapshot)

    def _recent_belief_lines(self, opp_id: int, limit: int = 5) -> list[str]:
        lines: list[str] = []
        for snapshot in self._belief_history.get(opp_id, [])[-limit:][::-1]:
            beliefs = snapshot.get("beliefs", {})
            if not isinstance(beliefs, dict) or not beliefs:
                continue
            ordered = sorted(beliefs, key=lambda k: beliefs[k], reverse=True)
            probs = ", ".join(
                f"{name} {float(beliefs.get(name, 0.0)):.1%}"
                for name in ordered
            )
            lines.append(f"game {snapshot.get('game', '?')}: {probs}")
        return lines

    # ── Self-evolve memory ─────────────────────────────────────────────

    def _self_evolve_memory_text(
        self,
        player_id: int | None = None,
        limit: int = 3,
    ) -> str:
        return render_case_memory_block(
            self.memory,
            game="goofspiel",
            position=f"P{player_id}" if player_id is not None else "",
            limit=limit,
        )

    def _self_evolve_output_example(
        self, legal_actions: list[Action],
    ) -> str:
        example = build_example(legal_actions)
        example = {
            "reasoning": example.get("reasoning", "step by step analysis"),
            "action": example.get("action", {"type": "<action_type>"}),
        }
        return json.dumps(example, ensure_ascii=False)

    def _infer_predicted_action_label(self, predicted_text: str) -> str:
        text = predicted_text.strip().lower()
        text = text.strip("\"'.,;:! ")
        if text.isdigit():
            return str(int(text))
        if text.startswith("bid "):
            parts = text.split()
            for part in parts[1:]:
                cleaned = part.strip("\"'.,;:! ")
                if cleaned.isdigit():
                    return str(int(cleaned))
        return ""

    def _find_observed_opponent_bid(
        self,
        trajectory: Trajectory,
        source_step: Any,
    ) -> tuple[str, str]:
        """Return (observed_text, observed_amount) for the same-round opponent bid."""
        obs = source_step.observation
        if obs is None:
            return "", ""
        pub = obs.public_info or {}
        round_no = pub.get("round")
        opp_id = 1 - source_step.player_id

        for candidate in trajectory.steps:
            if candidate.player_id != opp_id:
                continue
            cand_obs = candidate.observation
            cand_round = (cand_obs.public_info or {}).get("round") if cand_obs else None
            if cand_round != round_no:
                continue
            action = candidate.parsed_action
            if action and action.type == "bid" and action.amount > 0:
                return f"P{opp_id} bids {action.amount}", str(int(action.amount))
        return "", ""

    def _trajectory_before_predict(self, observation: Observation) -> str:
        tokens: list[str] = []
        for entry in observation.history or []:
            if not isinstance(entry, dict) or entry.get("action") != "round_result":
                continue
            r = entry.get("round", "?")
            prize = entry.get("prize", "?")
            p0 = entry.get("p0_card", "?")
            p1 = entry.get("p1_card", "?")
            outcome = {1: "P0w", -1: "P1w", 0: "tie"}.get(entry.get("outcome"), "?")
            tokens.append(f"action: round {r} prize {prize}, P0 bid {p0}, P1 bid {p1}")
            tokens.append(f"state: round {r} result {outcome}")
        pub = observation.public_info or {}
        current = " ".join(str(observation.state_text).split())
        if current:
            tokens.append(f"state: {current}")
        else:
            tokens.append(f"state: current:P{observation.player_id}@R{pub.get('round', '?')}")
        return " | ".join(tokens)

    # ── Prompt rendering ──────────────────────────────────────────────

    def _render_prompt(
        self,
        observation: Observation,
        legal_actions: list[Action],
        opp_id: int,
    ) -> str:
        action_descs = render_action_descs(legal_actions)

        parts: list[str] = []

        # SECTION 1: GAME RULES
        parts.append("#### GAME RULES ####\n")
        parts.append(observation.rules_text)

        # SECTION 2: GAME STATE
        parts.append("\n#### GAME STATE ####\n")
        parts.append(observation.state_text)
        hist_text = history_to_text(observation.history, max_entries=10)
        if hist_text:
            parts.append(f"\n{hist_text}")

        # SECTION 3: GTO BASELINE (CFR Nash)
        if self.use_cfr and self._cfr_method and not self.ablation.no_nash:
            parts.append("\n#### GTO BASELINE (CFR Nash) ####\n")
            try:
                cfr_text = self._cfr_method.format_for_prompt(
                    observation, legal_actions,
                )
                if cfr_text:
                    parts.append(cfr_text)
            except Exception as e:
                self._dbg(f"CFR error: {e}")

        # SECTION 4: OPPONENT MODEL
        if not self.ablation.no_belief:
            parts.append("\n#### OPPONENT MODEL ####\n")
            parts.extend(self._opponent_model_section(opp_id))

        memory_text = self._self_evolve_memory_text(observation.player_id)
        if memory_text:
            parts.extend(["", memory_text, ""])

        # SECTION 5: YOUR MOVE
        parts.append("\n#### YOUR MOVE ####\n")
        if self.lang == "cn":
            parts.extend(self._decision_guidance_cn())
        else:
            parts.extend(self._decision_guidance_en())

        example = self._self_evolve_output_example(legal_actions)
        parts.extend([
            "",
            "Legal actions:",
            "\n".join(action_descs),
            "",
            f"Output as JSON: {example}",
        ])
        return "\n".join(parts)

    def _opponent_model_section(self, opp_id: int) -> list[str]:
        n_rounds = len(self._all_records.get(opp_id, []))
        if n_rounds == 0:
            return ["No opponent bids observed yet. Play GTO until you have data."]

        lines: list[str] = []
        beliefs = self._posterior_cache.get(opp_id)
        recent_lines = self._recent_belief_lines(opp_id)
        if recent_lines:
            lines.append("Recent belief history (newest first):")
            lines.extend(recent_lines)
        else:
            lines.append("Recent belief history unavailable yet.")

        lines.append("")
        lines.append("Persona descriptions:")
        descriptions = {
            "Greedy": "Bids high for big prizes, low for small ones",
            "Conservative": "Seeks discounts; targets prize-1",
            "Sacrificial": "Sacrifices largest prize, bids prize for the rest",
            "Random": "Uniform random; no exploitable pattern",
            "GTO": "Nash equilibrium (pre-computed CFR mixed strategy)",
            "Mirror": "Bids exactly the prize value",
        }
        for pn in self.bayes.personas:
            desc = descriptions.get(pn, "")
            if desc:
                lines.append(f"  {pn}: {desc}")

        # Exploit hint
        if beliefs:
            top = max(beliefs, key=beliefs.get)
            top_prob = beliefs[top]
            hint = self._exploit_hint(top, top_prob)
            if hint:
                lines.append("")
                lines.append(hint)

        return lines

    def _exploit_hint(self, top_pn: str, top_prob: float) -> str:
        if top_prob < 0.35:
            return "Low confidence — play balanced GTO until clearer pattern emerges."
        hints = {
            "Greedy": "Suggestion: win cheap prizes with moderate bids; let them waste high cards.",
            "Conservative": "Suggestion: bid at prize level to win consistently with modest cards.",
            "Sacrificial": "Suggestion: identify the sacrifice round (largest prize) and win it cheaply.",
            "Random": "Suggestion: no exploitable pattern — stick to GTO baseline.",
            "GTO": "Suggestion: opponent plays Nash equilibrium — stick to GTO baseline.",
            "Mirror": "Suggestion: bid prize+1 to win every round at minimal card cost.",
        }
        return hints.get(top_pn, "")

    def _decision_guidance_en(self) -> list[str]:
        lines = [
            "Goal: Maximize your final total score (not just this round).",
            "",
            "Each card can only be used once — card management matters.",
            "",
            "Reason in this order:",
            "1. Start from the CFR/GTO strategy above as the baseline action tendency.",
            "2. Use your remaining hand, current prize, and round number to",
            "   identify the natural baseline bid.",
            "3. Adjust using the opponent model and recent behavior.",
            "4. If a self-evolve memory block is present, treat entries as",
            "   complete cases from similar states with possible opponent",
            "   action patterns; compare history, hand distribution, and",
            "   opponent style before relying on them, and never copy their",
            "   actions — use a case's future projection as evidence for",
            "   your own hypotheses.",
            "5. Choose the bid with the best expected outcome across ALL remaining rounds.",
        ]
        return lines

    def _decision_guidance_cn(self) -> list[str]:
        lines = [
            "目标：最大化最终总分（不是只看本轮）。",
            "",
            "每张牌只能用一次——出牌管理很重要。",
            "",
            "推理顺序：",
            "1. 以上方 CFR/GTO 策略为 baseline 行动倾向。",
            "2. 结合剩余手牌、当前 prize 和轮次确定自然 baseline 出牌。",
            "3. 使用对手模型和近期行为进行调整。",
            "4. 如果出现 self-evolve memory 块，把它们当成来自相似局面的完整",
            "   对战 case 与可能的对手行动方式；使用前先核对 history、手牌分布、",
            "   对手风格是否一致，不要照抄其中的动作——只在相关时把该 case 的",
            "   未来推演当作生成/修正假设的证据。",
            "5. 选择跨所有剩余轮次期望总分最优的出牌。",
        ]
        return lines

    # ── Bayesian statement (for backward compat / external use) ────────

    def bayes_statement(self, opp_id: int) -> str:
        """Delegate to underlying BayesPersonaModel.statement()."""
        return self.bayes.statement(opp_id)

    # ── Main interface ─────────────────────────────────────────────────

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple:
        self._dbg(f"Act P{observation.player_id} ({self.name})")
        opp_id = self._opponent_id(observation)

        # 1. Update beliefs from newly completed rounds
        self._update_beliefs(observation, opp_id)

        # 2. Classify opponent with q-window EMA
        persona, confidence, beliefs = self._classify_opponent(opp_id)
        self._record_belief_snapshot(opp_id, beliefs)
        if self.verbose and persona:
            self._dbg(f"Opponent: {persona} ({confidence:.0%})")

        prompt = self._render_prompt(
            observation, legal_actions, opp_id,
        )
        self._dbg(f"Main prompt ({len(prompt)} chars)")
        raw_response = model.generate(prompt)
        action = parse_action_response(raw_response, observation.player_id, legal_actions)
        self._dbg(f"Action: bid {action.amount}")

        # 5. Build metadata
        md: dict[str, Any] = {
            "persona": persona,
            "confidence": round(confidence, 4),
            "beliefs": {k: round(v, 4) for k, v in beliefs.items()} if beliefs else {},
            "n_rounds": len(self._all_records.get(opp_id, [])),
            "self_evolve": {
                "memory_records": len(select_case_memories(
                    self.memory,
                    game="goofspiel",
                    position=f"P{observation.player_id}",
                )),
            },
        }
        return action, prompt, raw_response, md

    def reflect(
        self, trajectory: Trajectory, model: BaseModel, game_name: str = "",
    ) -> list[str]:
        self._dbg(f"Reflect — {game_name}, {len(trajectory.steps)} steps")
        return []
