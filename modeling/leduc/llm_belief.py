"""LLMBeliefMethod: opponent modeling via LLM classification with exponential decay.

Classification uses ALL accumulated game history, not just the current game.
Belief vector is updated with exponential decay, giving more weight to recent games.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from base.memory import BaseMemory
from base.types import Trajectory
from base.interfaces import BaseModel
from modeling.base import BaseModelingMethod
from modeling.leduc.archetypes import (
    ARCHETYPE_NAMES, ARCHETYPE_DESCRIPTIONS_TEXT, ARCHETYPE_STYLE_NOTES,
)
from tools.leduc.hand_stats import (
    compute as compute_hand_stats,
    format_text as format_hand_stats,
)

BELIEF_ARCHETYPES = list(ARCHETYPE_NAMES)
SCORE_METHODS = ("llm", "linear", "exp", "plurality")


# ── OpponentBelief ──────────────────────────────────────────────────────────

class OpponentBelief:
    """Bayesian belief vector over archetype categories with exponential decay.

    beliefs[i] = P(opponent == archetype_i)

    Update rule:
      1. Decay: beliefs *= decay
      2. Inject: beliefs += (1 - decay) * normalized_scores
      3. Normalize
    """

    def __init__(self, decay: float = 0.85):
        self.archetypes = list(BELIEF_ARCHETYPES)
        n = len(self.archetypes)
        self.beliefs = [1.0 / n] * n
        self.decay = decay
        self.history: list[dict] = []

    def update(self, scores: dict[str, float], reasoning: str = "") -> None:
        n = len(self.beliefs)
        for i in range(n):
            self.beliefs[i] *= self.decay
        total_score = sum(scores.values()) or 1.0
        for i, arch in enumerate(self.archetypes):
            w = scores.get(arch, 0.0) / total_score
            self.beliefs[i] += (1.0 - self.decay) * w
        total = sum(self.beliefs)
        if total > 0:
            for i in range(n):
                self.beliefs[i] /= total
        self.history.append({
            "scores": scores,
            "reasoning": reasoning,
            "beliefs_after": [round(b, 4) for b in self.beliefs],
        })

    def most_likely(self) -> tuple[str, float]:
        idx = max(range(len(self.beliefs)), key=lambda i: self.beliefs[i])
        return self.archetypes[idx], self.beliefs[idx]

    def to_prompt(self) -> str:
        """LLM-friendly posterior summary for action prompts."""
        ranked = sorted(zip(self.archetypes, self.beliefs), key=lambda x: -x[1])
        parts = [f"{name}: {prob:.1%}" for name, prob in ranked]
        return "Current beliefs:\n  " + ", ".join(parts)

    def to_display(self) -> str:
        """Human-friendly: with bar chart."""
        ranked = sorted(zip(self.archetypes, self.beliefs), key=lambda x: -x[1])
        lines = [f"Games played: {len(self.history)}"]
        lines.append(f"Archetype belief (decay={self.decay:.2f}):\n")
        for name, prob in ranked:
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            lines.append(f"  {name:<16s} {prob:5.1%} |{bar}|")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"archetypes": list(self.archetypes), "beliefs": list(self.beliefs),
                "decay": self.decay, "history": list(self.history)}

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentBelief":
        b = cls(decay=data.get("decay", 0.85))
        stored_archetypes = list(data.get("archetypes", BELIEF_ARCHETYPES))
        stored_beliefs = list(data.get("beliefs", [1.0 / len(stored_archetypes)] * len(stored_archetypes)))
        # Migrate old 7-dim data (includes "unknown") to 6-dim
        if len(stored_archetypes) == 7 and stored_archetypes[-1] == "unknown":
            stored_archetypes = stored_archetypes[:-1]
            stored_beliefs = stored_beliefs[:-1]
            total = sum(stored_beliefs)
            if total > 0:
                stored_beliefs = [v / total for v in stored_beliefs]
            else:
                stored_beliefs = [1.0 / len(stored_archetypes)] * len(stored_archetypes)
        b.archetypes = stored_archetypes
        b.beliefs = stored_beliefs
        b.history = list(data.get("history", []))
        # Migrate history entries that may have 7-dim belief arrays
        for entry in b.history:
            if "beliefs_after" in entry and len(entry["beliefs_after"]) == 7:
                entry["beliefs_after"] = entry["beliefs_after"][:-1]
        return b


# ── LLMBeliefMethod ─────────────────────────────────────────────────────────

class LLMBeliefMethod(BaseModelingMethod):
    """Opponent modeling via LLM classification over accumulated history.

    After each game, the opponent's actions (with cards and street info) are
    appended to a cumulative log. The classification prompt shows ALL games,
    giving the LLM a complete picture of the opponent's play style.

    The belief vector is updated with exponential decay, giving more weight
    to recent classifications.

    Parameters:
        memory: persistence backend
        decay: exponential decay factor (0-1). Lower = forget faster.
        score_method: how to derive belief update weights from LLM output
            "llm":       use LLM-provided 0-1 scores directly (LLM启发式)
            "linear":    convert ranking to scores with linear weights [n, n-1, ..., 1]
            "exp":       convert ranking to scores with exp decay weights [e^0, e^(-λ), ...]
            "plurality": winner-take-all, only #1 gets weight
        exp_lam: lambda for "exp" score method
    """

    def __init__(self, memory: BaseMemory | None = None, decay: float = 0.5,
                 score_method: str = "exp", exp_lam: float = 0.7,
                 include_hand_stats: bool = True,
                 prompts_dir: str | Path | None = None):
        self.memory = memory or BaseMemory()
        self.decay = decay
        if score_method not in SCORE_METHODS:
            raise ValueError(
                f"score_method must be one of {SCORE_METHODS}, got '{score_method}'")
        self.score_method = score_method
        self.exp_lam = exp_lam
        self.include_hand_stats = include_hand_stats
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None

    # ── Persistence ────────────────────────────────────────────────────

    def _belief_key(self, opp_id: int) -> str:
        return f"leduc_archetype_belief_{opp_id}"

    def _game_logs_key(self, opp_id: int) -> str:
        return f"llm_belief_logs_{opp_id}"

    def _load_belief(self, opp_id: int) -> OpponentBelief:
        key = self._belief_key(opp_id)
        if key not in self.memory:
            self.memory.set(key, OpponentBelief(decay=self.decay).to_dict())
        return OpponentBelief.from_dict(self.memory.get(key))

    def _save_belief(self, opp_id: int, b: OpponentBelief) -> None:
        self.memory.set(self._belief_key(opp_id), b.to_dict())

    def _load_game_logs(self, opp_id: int) -> list[dict]:
        return self.memory.get(self._game_logs_key(opp_id), [])

    def _save_game_logs(self, opp_id: int, logs: list[dict]) -> None:
        self.memory.set(self._game_logs_key(opp_id), logs)

    # ── Core: data ingestion (online + offline share this) ──────────────

    def _accumulate_game(self, opp_id: int,
                         trajectory: Trajectory | dict) -> dict | None:
        """Extract game log from trajectory and append to accumulated list.
        Pure data accumulation — no LLM calls. Shared by online and offline paths.
        Accepts both Pydantic Trajectory objects and raw dict trajectories
        (from pilot_tom JSON output).
        Returns the game log dict, or None if no opponent actions found.
        """
        game_log = self._extract_game_log(opp_id, trajectory)
        if game_log is None:
            return None
        logs = self._load_game_logs(opp_id)
        logs.append(game_log)
        self._save_game_logs(opp_id, logs)
        return game_log

    def classify(self, opp_id: int, model: BaseModel) -> dict | None:
        """Run LLM classification on ALL accumulated game logs.
        Updates the belief vector. Returns scores dict + reasoning, or None.
        """
        logs = self._load_game_logs(opp_id)
        if not logs:
            return None
        prompt = self._build_classify_prompt(opp_id, logs)
        if not prompt:
            return None
        raw = model.generate(prompt)
        scores, reasoning = self._parse_classification(raw)
        if not scores:
            return None

        if self.prompts_dir is not None:
            game_idx = len(logs)  # 1-based
            classify_dir = self.prompts_dir / f"classify_P{opp_id}"
            classify_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(ch if ch.isalnum() or ch in ("-_") else "_" for ch in self.score_method)
            (classify_dir / f"g{game_idx:02d}_{safe}_prompt.txt").write_text(prompt)
            (classify_dir / f"g{game_idx:02d}_{safe}_response.txt").write_text(raw)

        b = self._load_belief(opp_id)
        b.update(scores, reasoning)
        self._save_belief(opp_id, b)
        return {"scores": scores, "reasoning": reasoning}

    def load_from_dir(self, opp_id: int, dir_path: str) -> int:
        """Load all trajectory JSON files from a directory. No LLM calls.
        Returns number of games loaded.
        """
        import glob as _glob
        paths = sorted(_glob.glob(f"{dir_path}/*.json"))
        count = 0
        for path in paths:
            traj = Trajectory.model_validate(json.loads(open(path).read()))
            if self._accumulate_game(opp_id, traj):
                count += 1
        return count

    # ── BaseModelingMethod interface ───────────────────────────────────

    def update(self, opp_id: int, trajectory: Trajectory,
               model: BaseModel | None = None) -> None:
        """Online: ingest one game then classify. Convenience wrapper."""
        if model is None:
            return
        if self._accumulate_game(opp_id, trajectory) is None:
            return
        self.classify(opp_id, model)

    def statement(self, opp_id: int, n_actions: int = 0) -> str:
        b = self._load_belief(opp_id)
        if not b.history:
            return ""
        n_acts = f", {n_actions} actions" if n_actions > 0 else ""
        ml, prob = b.most_likely()
        lines = [
            f"LLM Belief ({len(b.history)} games{n_acts}, "
            f"decay={self.decay:.2f}, score={self.score_method})",
            b.to_prompt(),
        ]
        style = ARCHETYPE_STYLE_NOTES.get(ml, "")
        if style:
            lines.append(f"\nTypical {ml} style: {style}")
        return "\n".join(lines)

    def display(self, opp_id: int) -> str:
        """Human-friendly debug display with bar charts."""
        b = self._load_belief(opp_id)
        if not b.history:
            return f"No data for P{opp_id} yet."
        logs = self._load_game_logs(opp_id)
        lines = [
            f"LLM Belief — P{opp_id} ({len(b.history)} classifications, "
            f"{len(logs)} games logged, decay={self.decay:.2f})",
            b.to_display(),
        ]
        return "\n".join(lines)

    def exploit_hint(self, opp_id: int) -> str:
        b = self._load_belief(opp_id)
        if not b.history:
            return ""
        ml, prob = b.most_likely()
        hints = {
            "CallingStation": "Never bluffs, never folds. Value bet thinly, never bluff.",
            "Maniac": "Always aggressive. Call down with medium hands, trap with strong ones.",
            "Nit": "Only plays strong hands. Bluff frequently, fold to their aggression.",
            "TAG": "Solid, slightly passive. Respect their bets, steal when they check.",
            "LAG": "Aggressive and unpredictable. Trap with strong hands, call down lighter.",
            "GTO": "Balanced play. No obvious exploit.",
        }
        hint = hints.get(ml, "")
        if hint:
            return f"Exploitation vs P{opp_id} ({ml}, {prob:.0%} confidence): {hint}"
        return ""

    def reset(self, opp_id: int) -> None:
        self._save_belief(opp_id, OpponentBelief(decay=self.decay))
        self._save_game_logs(opp_id, [])

    # ── Game log extraction ────────────────────────────────────────────

    def _extract_game_log(self, player_id: int,
                          trajectory: Trajectory | dict) -> dict | None:
        """Extract one game's opponent actions into a compact log entry.

        Works with both new-format Trajectory objects and old-format dict
        trajectories from pilot_tom JSON output.
        """
        steps = (trajectory.steps if hasattr(trajectory, 'steps')
                 else trajectory.get("steps", []))
        if steps and isinstance(steps[0], dict):
            return self._extract_game_log_dict(player_id, steps)

        actions: list[str] = []
        card = ""
        community = ""
        for s in steps:
            if s.player_id != player_id:
                continue
            sb = s.state_before
            if not card and "private_cards" in sb:
                pc = sb["private_cards"]
                card = pc.get(player_id) or pc.get(str(player_id)) or ""
            cc = sb.get("community_card", "")
            if cc:
                community = cc
            street = "preflop" if sb.get("street", 0) == 0 else "flop"
            a_str = str(s.parsed_action) if s.parsed_action else "?"
            actions.append(f"{street}: {a_str} (pot={sb.get('pot', '?')})")

        if not actions:
            return None
        return {"card": card, "community": community, "actions": actions}

    def _extract_game_log_dict(self, player_id: int,
                               steps: list[dict]) -> dict | None:
        """Extract game log from old-format pilot_tom dict steps."""
        import re
        actions: list[str] = []
        card = ""
        community = ""
        street = "preflop"
        for s in steps:
            if s.get("player_id") != player_id:
                continue
            if not card:
                card = (s.get("private_cards") or {}).get(str(player_id), "")
            cc = s.get("community_card", "")
            if cc:
                community = cc
                street = "flop"
            raw_action = s.get("action", "")
            # Parse "P1 raises 4" → type="raise", amount=4
            m = re.search(r'P\d+\s+(\w+)', raw_action)
            raw_type = m.group(1).lower() if m else "?"
            # Normalize: "raises" → "raise", etc.
            action_type = {"raises": "raise", "bets": "bet", "calls": "call",
                           "folds": "fold", "checks": "check"}.get(raw_type, raw_type)
            # Try to extract amount
            am = re.search(r'(\d+)$', raw_action)
            amount = int(am.group(1)) if am else 0
            pot = s.get("rendered_prompt", "")
            # Try to extract pot from rendered_prompt
            pm = re.search(r'Pot:\s*(\d+)', pot)
            pot_val = int(pm.group(1)) if pm else "?"
            actions.append(f"{street}: {action_type} {amount} (pot={pot_val})")

        if not actions:
            return None
        return {"card": card, "community": community, "actions": actions}

    # ── Classification prompt ──────────────────────────────────────────

    def _build_classify_prompt(self, player_id: int,
                               game_logs: list[dict]) -> str:
        """Build a structured classification prompt."""
        n = len(game_logs)

        # ── Historical data (4 cards per game → compact rows) ──
        hist_lines = []
        for i, gl in enumerate(game_logs):
            card = gl.get("card") or "?"
            community = gl.get("community") or "none"
            actions_str = " → ".join(
                a.split(":", 1)[1].strip().split("(")[0].strip()
                for a in gl.get("actions", [])
            )
            hist_lines.append(
                f"  G{i+1:2d}  card={card:3s}  board={community:4s}  actions: {actions_str}")

        # ── Hand-strength-conditioned statistics (optional) ──
        stats_block = ""
        if self.include_hand_stats:
            stats = compute_hand_stats(game_logs)
            stats_block = format_hand_stats(stats, n) + "\n"

        archetype_keys = '", "'.join(ARCHETYPE_NAMES)
        scores_example = ", ".join(f'"{a}": 0.0' for a in ARCHETYPE_NAMES)

        prompt = (
            "### TASK ###\n"
            f"Classify opponent P{player_id}'s poker play style into one of 6 archetypes.\n"
            f"Use the historical data below ({n} hands) to identify behavioral patterns.\n"
            "\n"
            "Key diagnostic signals to look for:\n"
            "  - Does the opponent play weak cards (Jack) aggressively?  →  LAG or Maniac\n"
            "  - Does the opponent fold when holding weak cards?         →  TAG or Nit\n"
            "  - Does the opponent ever call or check?                  →  NOT a Maniac\n"
            "  - Does the opponent ever play cards weaker than Kings?   →  NOT a Nit\n"
            "  - Are actions unpredictable, mixing checks and bets?     →  GTO\n"
            "\n"
            "### HISTORICAL DATA ###\n"
            + "\n".join(hist_lines) + "\n"
            "\n" + stats_block +
            "### REFERENCE: ARCHETYPE DEFINITIONS ###\n"
            + ARCHETYPE_DESCRIPTIONS_TEXT + "\n"
            "\n"
            "### OUTPUT REQUIREMENTS ###\n"
            f"1. Rank all 6 archetypes from most to least likely for P{player_id}.\n"
            "2. Assign each a score 0.0–1.0 (how well it matches the observed behavior).\n"
            "3. Write a 2-3 sentence reasoning that cites specific patterns in the data.\n"
            "\n"
            "Output strictly as JSON:\n"
            f'{{"ranking": ["{archetype_keys}"], '
            f'"scores": {{{scores_example}}}, '
            f'"reasoning": "<2-3 sentences>"}}'
        )
        return prompt

    # ── Classification parsing ─────────────────────────────────────────

    def _parse_classification(self, raw: str) -> tuple[dict[str, float], str]:
        try:
            text = raw.strip()
            if "{" in text:
                text = text[text.find("{"):text.rfind("}") + 1]
            data = json.loads(text)

            if self.score_method == "llm":
                # LLM启发式: use scores directly from LLM output
                scores = {}
                raw_scores = data.get("scores", {})
                if isinstance(raw_scores, dict):
                    for k, v in raw_scores.items():
                        if k in BELIEF_ARCHETYPES:
                            scores[k] = float(v)

            else:
                # 排序结果 → 用不同权重转换为scores
                ranking = data.get("ranking", [])
                scores = (self._ranking_to_scores(ranking)
                          if isinstance(ranking, list) else {})

            # Fallback: if scores is empty, try ranking → linear conversion
            if not scores:
                ranking = data.get("ranking", [])
                if isinstance(ranking, list) and ranking:
                    scores = self._ranking_to_scores(ranking, method="linear")
            if not scores:
                arch = str(data.get("archetype", "")).strip()
                if arch in BELIEF_ARCHETYPES:
                    scores = {arch: 1.0}

            return scores, str(data.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            return {}, "classification parse error"

    def _ranking_to_scores(self, ranking: list[str],
                           method: str | None = None) -> dict[str, float]:
        """Convert ranking to scores using one of three weighting schemes.
        linear: weights = [n, n-1, ..., 1]
        exp:    weights = [e^0, e^(-λ), e^(-2λ), ...]
        plurality: only #1 gets weight.
        """
        m = method or self.score_method
        valid = [a for a in ranking if a in BELIEF_ARCHETYPES]
        if not valid:
            return {}
        n = len(valid)
        if m == "linear":
            weights = [float(n - i) for i in range(n)]
        elif m == "exp":
            weights = [math.exp(-self.exp_lam * i) for i in range(n)]
        elif m == "plurality":
            weights = [1.0] + [0.0] * (n - 1)
        else:
            return {}
        total = sum(weights)
        if total == 0:
            return {}
        return {arch: w / total for arch, w in zip(valid, weights)}
