"""Theory of Mind method for Liar's Dice with persona classification + CFR guidance.

Combines:
  - Data-driven opponent persona classification (via signature bank)
  - Pre-computed CFR Nash equilibrium strategy for the current infoset
  - LLM reasoning to exploit opponent deviations from GTO

Usage:
    method = LiarsDiceToMMethod(
        name="ToM+CFR",
        strategy_path="results/liars_dice/cfr/20260622_180205_cfr_1x4_wild1_1m/model",
        signature_bank_path="results/liars_dice/persona_signature_bank/.../metrics/signature_bank.json",
    )
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from base.case_memory import render_case_memory_block, select_case_memories
from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation, Trajectory
from methods.helpers import build_example, parse_action_response, render_action_descs
from methods.liars_dice_cfr import (
    LiarsDiceCFRTrainer,
    _action_to_key,
    _infoset_key_from_observation_internal,
)
from analysis.liars_dice.signature_features import FEATURE_ORDER, get_feature_order, game_features
from base.ablation import AblationConfig

PERSONA_ORDER = [
    "Bluffer",
    "Honest",
    "AggressiveChallenger",
    "ConservativeChallenger",
    "Random",
    "GTO",
]

PERSONA_DESCRIPTIONS = {
    "Bluffer": (
        "Habitual liar. Constantly overbids — claims more dice of a face than "
        "they actually have. Rarely challenges. Prefers raising even when suspicious."
    ),
    "Honest": (
        "Always bids truthfully based on actual dice. Rarely exaggerates. "
        "Will challenge when a bid is clearly impossible."
    ),
    "AggressiveChallenger": (
        "Challenges early and often. Frequently calls \"Liar!\" instead of raising. "
        "Suspicious by nature."
    ),
    "ConservativeChallenger": (
        "Almost never challenges. Prefers raising over challenging. "
        "Only challenges outrageously impossible bids. Patient."
    ),
    "Random": (
        "Picks actions uniformly at random. No consistent strategy or pattern. "
        "Unpredictable by definition."
    ),
    "GTO": (
        "Balanced, hard-to-read player. No strong visible habits. "
        "Plays close to game-theoretic optimal."
    ),
}


class LiarsDiceToMMethod(BaseMethod):
    """Liar's Dice ToM: classify opponent persona + inject CFR GTO strategy."""

    _SELF_EVOLVE_KEY = "liars_dice_self_evolve_records"

    def __init__(
        self,
        name: str = "ToM+CFR",
        strategy_path: str = "",
        signature_bank_path: str = "",
        lang: str = "en",
        verbose: bool = False,
        memory=None,
        modeling_method=None,
        classification_mode: str = "q_window_ema",
        q_window: int = 10,
        q_alpha: float = 0.85,
        ablation: AblationConfig | None = None,
    ):
        self.name = name
        self.method_type = "tom"
        self.lang = lang
        self.verbose = verbose
        self.memory = memory
        self.modeling_method = modeling_method
        self.ablation = ablation or AblationConfig()
        self.classification_mode = classification_mode
        self.q_window = max(1, int(q_window))
        self.q_alpha = max(0.0, min(1.0, float(q_alpha)))

        # Load pre-computed CFR Nash equilibrium strategy
        self._cfr_strategy: dict = {}
        self._cfr_trainer: LiarsDiceCFRTrainer | None = None
        if strategy_path and Path(strategy_path).exists():
            trainer = LiarsDiceCFRTrainer()
            trainer.load(strategy_path)
            self._cfr_strategy = trainer.strategy
            self._cfr_trainer = trainer
            if self.verbose:
                print(
                    f"[ToM] Loaded CFR strategy: "
                    f"{len(self._cfr_strategy)} infosets"
                )

        # Load persona signature bank for classification
        self._bank: dict = {}
        self._source_stats: dict = {}
        self._conv_stats: dict = {}
        self._feature_order: list[str] = FEATURE_ORDER
        self._num_faces: int = 4
        self._rank_count: int = 8
        self._thresholds: dict[str, float] = {}
        self._rank_mode: str = "flat"
        if signature_bank_path and Path(signature_bank_path).exists():
            with open(signature_bank_path) as f:
                self._bank = json.load(f)
            self._feature_order = self._bank.get("feature_order", FEATURE_ORDER)
            self._num_faces = self._bank.get("num_faces", 4)
            self._rank_count = self._bank.get("rank_count", 8)
            self._thresholds = self._bank.get("thresholds", {})
            self._rank_mode = self._bank.get("rank_mode", "flat")
            self._source_stats = self._bank.get("source_stats", {})
            self._conv_stats = self._bank.get("conv_stats", {})
            if self.verbose:
                print(
                    f"[ToM] Loaded signature bank: "
                    f"{len(self._bank.get('personas', {}))} personas"
                )

        # Cross-game action accumulation
        self._all_records: list[dict[str, Any]] = []
        self._belief_history: list[dict[str, Any]] = []
        self._posterior_cache: dict[str, float] = {}
        self._prev_hist_len: int = 0
        self._current_game: int = 0

    # ── Self-evolve memory ─────────────────────────────────────────────

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

    # ── History accumulation ────────────────────────────────────────────

    def _update_records(self, observation: Observation) -> None:
        """Extract opponent action records from observation history."""
        history = observation.history or []
        my_id = observation.player_id

        # New game: history resets (length drops)
        if len(history) < self._prev_hist_len:
            self._current_game += 1
            self._prev_hist_len = 0

        for i in range(self._prev_hist_len, len(history)):
            entry = history[i]
            if not isinstance(entry, dict):
                continue
            atype = entry.get("action", "")
            if atype not in ("bid", "challenge"):
                continue
            # Only record opponent actions (signature bank only has persona actions)
            if entry.get("player") != 1 - my_id:
                continue
            self._all_records.append({
                "game": self._current_game,
                "step": len(self._all_records),
                "type": atype,
                "qty": entry.get("quantity"),
                "face": entry.get("face"),
            })

        self._prev_hist_len = len(history)

    # ── Persona classification ─────────────────────────────────────────

    def _initial_posterior(self) -> dict[str, float]:
        personas = self._bank.get("personas", {})
        if not personas:
            return {}
        p = 1.0 / len(personas)
        return {name: p for name in personas}

    def _records_for_window(self) -> list[dict[str, Any]]:
        if not self._all_records:
            return []
        game_ids = sorted({int(r.get("game", 0)) for r in self._all_records})
        selected = set(game_ids[-self.q_window:])
        return [r for r in self._all_records if int(r.get("game", 0)) in selected]

    def _classify_records(
        self, records: list[dict[str, Any]], my_id: int,
    ) -> dict[str, float]:
        """Return persona likelihood from a selected evidence window."""
        # Use the opponent's position signature
        opp_position = "p1" if my_id == 0 else "p0"

        if len(records) < 2:
            return {}

        features = game_features(
            records,
            self._thresholds,
            num_faces=self._num_faces,
            rank_count=self._rank_count,
            rank_mode=self._rank_mode,
        )
        personas = self._bank.get("personas", {})
        scores: dict[str, float] = {}

        for pname, entry in personas.items():
            sig = entry.get(opp_position) or entry.get("combined") or {}
            sig_std = (
                entry.get(f"{opp_position}_std") or entry.get("combined_std") or {}
            )
            d2 = 0.0
            for feat in self._feature_order:
                std = max(float(sig_std.get(feat, 0.0) or 0.0), 0.02)
                fv = float(features.get(feat, 0.0))
                sv = float(sig.get(feat, 0.0))
                z = (fv - sv) / std
                d2 += z * z
            scores[pname] = -math.sqrt(d2)

        # Softmax
        vals = list(scores.values())
        mx = max(vals) if vals else 0.0
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        total = sum(exps.values()) or 1.0
        return {k: v / total for k, v in exps.items()}

    def _update_persona_posterior(self, likelihood: dict[str, float]) -> dict[str, float]:
        if not likelihood:
            return {}
        if self.classification_mode == "cumulative":
            self._posterior_cache = dict(likelihood)
            return self._posterior_cache
        if self.classification_mode != "q_window_ema":
            raise ValueError(f"Unknown classification_mode: {self.classification_mode}")

        prior = self._posterior_cache or self._initial_posterior()
        posterior = {
            name: self.q_alpha * prior.get(name, 0.0)
            + (1.0 - self.q_alpha) * likelihood.get(name, 0.0)
            for name in likelihood
        }
        total = sum(posterior.values())
        if total > 0:
            posterior = {name: value / total for name, value in posterior.items()}
        self._posterior_cache = posterior
        return posterior

    def _classify_opponent(
        self, my_id: int
    ) -> tuple[str | None, float, dict[str, float]]:
        """Classify opponent persona with q-window EMA by default."""
        if len(self._all_records) < 2:
            return None, 0.0, {}

        if self.classification_mode == "cumulative":
            evidence = self._all_records
        elif self.classification_mode == "q_window_ema":
            evidence = self._records_for_window()
        else:
            raise ValueError(f"Unknown classification_mode: {self.classification_mode}")

        likelihood = self._classify_records(evidence, my_id)
        beliefs = self._update_persona_posterior(likelihood)

        top = max(beliefs, key=beliefs.get) if beliefs else None
        return top, beliefs.get(top, 0.0) if top else 0.0, beliefs

    def _record_belief_snapshot(self, beliefs: dict[str, float]) -> None:
        if not beliefs:
            return
        n_actions = sum(
            1 for r in self._all_records if r.get("type") in ("bid", "challenge")
        )
        snapshot = {
            "game": self._current_game + 1,
            "actions": n_actions,
            "classification_mode": self.classification_mode,
            "q_window": self.q_window,
            "q_alpha": self.q_alpha,
            "beliefs": dict(beliefs),
        }
        prev = self._belief_history[-1] if self._belief_history else None
        if prev and prev.get("game") == snapshot["game"]:
            self._belief_history[-1] = snapshot
        else:
            self._belief_history.append(snapshot)

    def _recent_belief_lines(self, limit: int = 5) -> list[str]:
        lines: list[str] = []
        for snapshot in self._belief_history[-limit:][::-1]:
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

    # ── CFR strategy query ──────────────────────────────────────────────

    def _query_cfr_strategy(
        self, observation: Observation, legal_actions: list[Action]
    ) -> list[tuple[str, float]]:
        """Get CFR strategy for current infoset. Returns [(label, prob), ...]."""
        if not self._cfr_strategy:
            return []
        info_key = _infoset_key_from_observation_internal(observation)
        strategy = self._cfr_strategy.get(info_key, {})
        if not strategy:
            return []
        result: list[tuple[str, float]] = []
        for action in legal_actions:
            key = _action_to_key(action)
            prob = strategy.get(key, 0.0)
            if prob > 0.001:
                if action.type == "challenge":
                    result.append(("challenge", prob))
                else:
                    qty = action.data.get("quantity", "?")
                    face = action.data.get("face", "?")
                    result.append((f"bid {qty}x{face}", prob))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ── Prompt rendering ────────────────────────────────────────────────

    def _action_label(self, action: Action) -> str:
        if action.type == "challenge":
            return "challenge"
        if action.type == "bid":
            qty = action.data.get("quantity", "?")
            face = action.data.get("face", "?")
            return f"bid {qty}x{face}"
        return action.type

    def _predicted_action_label_set(self, legal_actions: list[Action]) -> set[str]:
        labels = {a.type for a in legal_actions}
        if any(a.type == "bid" for a in legal_actions):
            labels.add("bid")
            labels.add("raise_bid")
        if any(a.type == "challenge" for a in legal_actions):
            labels.add("challenge")
        return labels

    def _infer_predicted_action_label(self, predicted_text: str) -> str:
        text = predicted_text.strip().lower()
        text = text.strip("\"'.,;:! ")
        if text.startswith("bid "):
            return "bid"
        if text in {"bid", "raise", "raise_bid", "raise bid"}:
            return "bid"
        if text in {"challenge", "call liar", "liar", "call"}:
            return "challenge"
        return ""

    def _history_to_text(self, history: list[dict]) -> str:
        if not history:
            return ""
        lines = ["Recent events:"]
        for entry in history[-10:]:
            if not isinstance(entry, dict):
                continue
            player = entry.get("player", "?")
            action = entry.get("action", "?")
            if action == "bid":
                qty = entry.get("quantity", "?")
                face = entry.get("face", "?")
                lines.append(f"  P{player}: bid {qty}x{face}")
            elif action == "challenge":
                lines.append(f"  P{player}: challenge!")
            else:
                result = entry.get("result", "")
                data = {k: v for k, v in entry.items()
                        if k not in ("player", "action", "result", "round")}
                line = f"  P{player}: {action}"
                if data:
                    line += f" {data}"
                if result:
                    line += f" → {result}"
                lines.append(line)
        return "\n".join(lines)

    def _trajectory_before_predict(self, observation: Observation) -> str:
        tokens: list[str] = []
        for entry in observation.history or []:
            if not isinstance(entry, dict):
                continue
            player = entry.get("player", "?")
            if entry.get("action") == "bid":
                tokens.append(
                    "action: "
                    f"P{player} bid {entry.get('quantity', '?')}x{entry.get('face', '?')}"
                )
                tokens.append("state: public bid history updated")
            elif entry.get("action") == "challenge":
                tokens.append(f"action: P{player} challenge")
                tokens.append("state: challenge resolved")
        current = " ".join(str(observation.state_text).split())
        if current:
            tokens.append(f"state: {current}")
        else:
            tokens.append(f"state: current:P{observation.player_id}")
        return " | ".join(tokens)

    def _self_evolve_memory_text(
        self,
        player_id: int | None = None,
        limit: int = 3,
    ) -> str:
        return render_case_memory_block(
            self.memory,
            game="liars_dice",
            position=f"P{player_id}" if player_id is not None else "",
            limit=limit,
        )

    def _output_example(self, legal_actions: list[Action]) -> str:
        example = build_example(legal_actions)
        example = {
            "reasoning": example.get("reasoning", "step by step analysis of the current situation"),
            "action": example.get("action", {"type": "<action_type>"}),
        }
        return json.dumps(example, ensure_ascii=False)

    def _render_prompt(
        self,
        observation: Observation,
        legal_actions: list[Action],
        persona: str | None,
        confidence: float,
        beliefs: dict[str, float],
        cfr_strategy: list[tuple[str, float]],
    ) -> str:
        action_descs = render_action_descs(legal_actions)
        hist_text = self._history_to_text(observation.history)
        example = self._output_example(legal_actions)

        parts: list[str] = []
        parts.append("#### GAME RULES ####\n")
        parts.append(observation.rules_text)
        parts.append(
            "\nNote: If a bid is the maximum legal bid (quantity equals total dice), "
            "the opponent cannot raise and must challenge.\n"
        )

        parts.append("\n#### GAME STATE ####\n")
        parts.append(observation.state_text)
        parts.append(
            "\nNote: Current dice are freshly rolled for this round; "
            "history dice are irrelevant.\n"
        )
        if hist_text:
            parts.append(f"\n{hist_text}")

        if cfr_strategy and not self.ablation.no_nash:
            parts.append("\n#### GTO BASELINE (CFR Nash) ####\n")
            parts.append("Nash mixed strategy for this situation (pre-computed):")
            for text, prob in cfr_strategy[:6]:
                parts.append(f"  {text}: {prob*100:.1f}%")
            parts.append("")

        if not self.ablation.no_belief:
            parts.append("\n#### OPPONENT MODEL ####\n")
            n_actions = sum(
                1 for r in self._all_records if r.get("type") in ("bid", "challenge")
            )
            n_games = len(set(
                r.get("game") for r in self._all_records
                if r.get("type") in ("bid", "challenge")
            ))
            if n_actions == 0:
                parts.append("No opponent actions observed yet.")
            else:
                parts.append(f"Observed: {n_games} game(s), {n_actions} opponent actions")
                parts.append(f"Classification mode: {self.classification_mode}")
                if self.classification_mode == "q_window_ema":
                    parts.append(f"Q-window: last {self.q_window} games")
                    parts.append(f"EMA alpha: {self.q_alpha:.2f}")
                recent_lines = self._recent_belief_lines()
                if recent_lines:
                    parts.append("\nRecent persona probabilities:")
                    parts.extend(recent_lines)
                parts.append("\nPersona style reference:")
                for pname in PERSONA_ORDER:
                    desc = PERSONA_DESCRIPTIONS.get(pname, "")
                    parts.append(f"  {pname}: {desc}")
                parts.append(
                    "\nNote: Persona tendencies cannot override legal action "
                    "constraints — even a Bluffer can challenge an impossible bid.\n"
                )
            parts.append("")

        if observation.role_guidance:
            if isinstance(observation.role_guidance, list):
                parts.extend(observation.role_guidance)
            else:
                parts.append(str(observation.role_guidance))

        memory_text = self._self_evolve_memory_text(observation.player_id)
        if memory_text:
            parts.extend(["", memory_text, ""])

        parts.append("\n#### YOUR MOVE ####\n")
        if self.lang == "cn":
            parts.extend([
                "目标：最大化筹码收益或最小化筹码损失。",
                "",
                "先用 CFR/GTO 混合策略作为 baseline，再结合对手模型寻找剥削机会。",
                "",
                "推理顺序：",
                "1. 先根据 CFR/GTO 混合策略判断 baseline 行动倾向。",
                "2. 结合你的骰子、当前 bid、总骰数和合法动作判断自然行动线。",
                "3. 再结合对手 persona 概率和近期动作进行调整。",
                "4. 如果出现 self-evolve memory 块，把它们当成来自相似局面的完整对战",
                "   case 与可能的对手行动方式；使用前先核对 history、骰子分布、",
                "   对手风格是否一致，不要照抄其中的动作——只在相关时把该 case 的",
                "   未来推演当作生成/修正假设的证据。",
                "5. 选择期望结果最优的动作。",
            ])
        else:
            parts.extend([
                "Goal: maximize chips won or minimize chips lost against",
                "this specific opponent.",
                "",
                "Use the CFR/GTO mixed strategy as the baseline, then adjust",
                "for this opponent's persona and recent actions.",
                "",
                "Reason in this order:",
                "1. Start from the CFR/GTO mixed strategy above as the",
                "   baseline action tendency.",
                "2. Use your dice, current bid, total dice, and legal actions",
                "   to identify the natural line.",
                "3. Then adjust using the opponent model: persona probabilities,",
                "   recent bids/challenges, and likely deviations.",
                "4. If a self-evolve memory block is present, treat entries as",
                "   complete cases from similar states with possible opponent",
                "   action patterns; compare history, dice distribution, and",
                "   opponent style before relying on them, and never copy their",
                "   actions — use a case's future projection as evidence for",
                "   your own hypotheses.",
                "5. Choose the action with the best expected outcome.",
            ])

        parts.extend([
            "",
            "Legal actions:",
            "\n".join(action_descs),
            "",
            f"Output as JSON: {example}",
        ])
        return "\n".join(parts)

    # ── Main interface ──────────────────────────────────────────────────

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple[Action, str, str, dict]:
        if self.verbose:
            print(f"[ToM] Act P{observation.player_id} ({self.name})")

        # 1. Accumulate action records from history
        self._update_records(observation)

        # 2. Classify opponent persona
        my_id = observation.player_id
        persona, confidence, beliefs = self._classify_opponent(my_id)
        self._record_belief_snapshot(beliefs)
        if self.verbose and persona:
            print(f"[ToM] Classified opponent: {persona} ({confidence:.0%})")

        # 3. Query CFR strategy and action EVs
        cfr_strategy = self._query_cfr_strategy(observation, legal_actions)

        prompt = self._render_prompt(
            observation, legal_actions, persona, confidence, beliefs,
            cfr_strategy,
        )
        if self.verbose:
            print(f"[ToM] Prompt ({len(prompt)} chars)")
            for line in prompt.split("\n"):
                print(f"    | {line}")
        raw_response = model.generate(prompt)
        action = parse_action_response(raw_response, my_id, legal_actions)

        if self.verbose:
            print(f"[ToM] Action: {action}")

        # 5. Build metadata
        md: dict[str, Any] = {
            "persona": persona,
            "confidence": round(confidence, 4),
            "beliefs": {k: round(v, 4) for k, v in beliefs.items()},
            "n_actions": len(self._all_records),
            "cfr_strategy": [[t, round(p, 4)] for t, p in cfr_strategy],
            "self_evolve": {
                "memory_records": len(select_case_memories(
                    self.memory,
                    game="liars_dice",
                    position=f"P{observation.player_id}",
                )),
            },
        }
        return action, prompt, raw_response, md

    def _find_observed_opponent_action(
        self, trajectory, source_step,
    ) -> tuple[str, str]:
        """Return (observed_text, observed_label) of first opponent action after source_step."""
        opp_id = 1 - source_step.player_id
        found = False
        for candidate in trajectory.steps:
            if candidate is source_step:
                found = True
                continue
            if not found:
                continue
            if candidate.player_id != opp_id:
                continue
            action = candidate.parsed_action
            if action and action.type:
                normalized = "bid" if action.type == "bid" else action.type
                return f"P{opp_id} {action.type}", normalized
        return "game_ended", ""

    def reflect(
        self,
        trajectory: Trajectory,
        model: BaseModel,
        game_name: str = "",
    ) -> list[str]:
        return []
