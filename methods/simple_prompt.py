from __future__ import annotations

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
import json

from methods.helpers import (
    build_example, build_rule_section, history_to_text, parse_action_response,
    render_action_descs,
)


class SimplePromptMethod(BaseMethod):
    """Direct action output — no reasoning, no memory.

    Args:
        name: Display name.
        lang: 'en' or 'cn' for prompt language.
    """

    def __init__(self, name: str = "SimplePrompt", lang: str = "en"):
        self.name = name
        self.method_type = "simple"
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

        parts = []

        # 1. GAME RULES
        parts.append("\n#### GAME RULES ####\n")
        parts.append(obs.rules_text)

        # 2. GAME STATE + history (like Leduc ToM format)
        parts.append("\n#### GAME STATE ####\n")
        parts.append(obs.state_text)
        hist_text = history_to_text(obs.history, max_entries=10)
        if hist_text:
            parts.append(f"\n{hist_text}")

        # 3. Full history (only Werewolf)
        if obs.full_history_text:
            parts.append(f"\n#### GAME FULL HISTORY ####\n")
            parts.append(obs.full_history_text)

        # 4. YOUR ROLE (only Werewolf)
        role_lines = []
        role = obs.private_info.get("role")
        if role:
            role_lines.append(f"You are P{obs.player_id}. Your role: {role}.")
        if "other_werewolves" in obs.private_info:
            others = obs.private_info["other_werewolves"]
            if others:
                role_lines.append(f"Fellow werewolves: {', '.join(f'P{p}' for p in others)}.")
        if "seer_results" in obs.private_info:
            results = obs.private_info["seer_results"]
            if results:
                results_str = ", ".join(f"P{p} -> {r}" for p, r in sorted(results.items()))
                role_lines.append(f"Your investigation results: {results_str}.")
        if "save_available" in obs.private_info or "poison_available" in obs.private_info:
            save = "available" if obs.private_info.get("save_available") else "ALREADY USED"
            poison = "available" if obs.private_info.get("poison_available") else "ALREADY USED"
            role_lines.append(f"Save potion: {save}. Poison potion: {poison}.")
        if role_lines:
            parts.append("\n#### YOUR ROLE ####\n")
            parts.extend(role_lines)

        if obs.role_guidance:
            parts.extend(build_rule_section(obs.role_guidance))

        # 5. YOUR MOVE
        parts.append("\n#### YOUR MOVE ####\n")
        parts.append(
            "If any persona guidance is provided above, think in character according "
            "to it before choosing your action. Then think step by step about the "
            "current situation and your role/rule, and output your action in JSON "
            "format."
        )
        parts.append("Legal actions:")
        parts.append("\n".join(action_descs))

        example = json.dumps(build_example(legal_actions))
        parts.extend(["", f"Output as JSON: {example}"])

        return "\n".join(parts)


class SimplePlaceboPromptMethod(SimplePromptMethod):
    """Simple prompt with length-matched, non-informative PLSE-style context."""

    def __init__(
        self,
        name: str = "SimplePlacebo",
        lang: str = "en",
        style_labels: list[str] | None = None,
        target_chars: int = 0,
    ):
        super().__init__(name=name, lang=lang)
        self.method_type = "simple_placebo"
        self.style_labels = style_labels or []
        self.target_chars = target_chars

    def _render_prompt(
        self, obs: Observation, legal_actions: list[Action],
    ) -> str:
        action_descs = render_action_descs(legal_actions)

        parts = []

        parts.append("\n#### GAME RULES ####\n")
        parts.append(obs.rules_text)

        parts.append("\n#### GAME STATE ####\n")
        parts.append(obs.state_text)
        hist_text = history_to_text(obs.history, max_entries=10)
        if hist_text:
            parts.append(f"\n{hist_text}")

        if obs.full_history_text:
            parts.append(f"\n#### GAME FULL HISTORY ####\n")
            parts.append(obs.full_history_text)

        role_lines = []
        role = obs.private_info.get("role")
        if role:
            role_lines.append(f"You are P{obs.player_id}. Your role: {role}.")
        if "other_werewolves" in obs.private_info:
            others = obs.private_info["other_werewolves"]
            if others:
                role_lines.append(f"Fellow werewolves: {', '.join(f'P{p}' for p in others)}.")
        if "seer_results" in obs.private_info:
            results = obs.private_info["seer_results"]
            if results:
                results_str = ", ".join(f"P{p} -> {r}" for p, r in sorted(results.items()))
                role_lines.append(f"Your investigation results: {results_str}.")
        if "save_available" in obs.private_info or "poison_available" in obs.private_info:
            save = "available" if obs.private_info.get("save_available") else "ALREADY USED"
            poison = "available" if obs.private_info.get("poison_available") else "ALREADY USED"
            role_lines.append(f"Save potion: {save}. Poison potion: {poison}.")
        if role_lines:
            parts.append("\n#### YOUR ROLE ####\n")
            parts.extend(role_lines)

        if obs.role_guidance:
            parts.extend(build_rule_section(obs.role_guidance))

        parts.extend(self._placebo_context_sections(legal_actions))
        padding_index = len(parts)

        parts.append("\n#### YOUR MOVE ####\n")
        parts.append(
            "Use the real game rules, real game state, and legal actions above. "
            "The placebo context is intentionally non-diagnostic: do not treat it "
            "as evidence about the opponent, a Nash solution, recent trend, or "
            "memory of past mistakes. Think step by step and choose one legal "
            "action."
        )
        parts.append("Legal actions:")
        parts.append("\n".join(action_descs))

        example = json.dumps(build_example(legal_actions))
        parts.extend(["", f"Output as JSON: {example}"])
        self._pad_to_target_chars(parts, insert_index=padding_index)

        return "\n".join(parts)

    def _placebo_context_sections(self, legal_actions: list[Action]) -> list[str]:
        legal_types = [a.type for a in legal_actions]
        action_prior = ", ".join(f"{action}: equal placeholder weight" for action in legal_types)
        if not action_prior:
            action_prior = "No legal action prior is available."

        styles = self.style_labels or ["StyleA", "StyleB", "StyleC", "StyleD", "StyleE", "StyleF"]
        style_prior = ", ".join(f"{style}: uniform placeholder prior" for style in styles)
        style_notes = self._style_space_notes(styles)

        return [
            "\n#### PLACEBO STRATEGIC REFERENCE ####\n",
            "This control prompt matches the full method's structure, but it does "
            "not provide a calibrated Nash, CFR, EV, or action-value estimate.",
            f"Placeholder action prior: {action_prior}.",
            "Treat every placeholder weight as non-diagnostic.",
            "Generic play checklist: identify legal actions, compare risk and "
            "reward, follow the public rules, and use only the visible state.",
            "\n#### PLACEBO OPPONENT BELIEF ####\n",
            f"Candidate styles: {style_prior}.",
            "These labels define the possible style space, not an inferred belief.",
            "No observed action has been mapped to a style, and no posterior "
            "probability has been estimated.",
            "General style reminders:",
            *style_notes,
            "These reminders are generic and do not identify the current opponent.",
            "\n#### PLACEBO RECENT WINDOW ####\n",
            "Recent-window evidence is intentionally withheld. This control does "
            "not compute a q-window, moving average, likelihood update, or "
            "style-switch detector.",
            "Use visible history only as ordinary game context, not as a hidden "
            "summary of opponent tendencies.",
            "\n#### PLACEBO MEMORY ####\n",
            "Memory slot 1: neutral placeholder; no failed prediction is recorded.",
            "Memory slot 2: neutral placeholder; no corrected model is recorded.",
            "Memory slot 3: neutral placeholder; no learned exploit is recorded.",
            "\n#### PLACEBO DECISION SCAFFOLD ####\n",
            "Think internally through the rules, legal actions, and payoff objective. "
            "Keep the final JSON reasoning concise.",
            "Do not claim that a Nash solver, opponent classifier, q-window estimator, "
            "or memory module recommended the final move.",
        ]

    def _style_space_notes(self, styles: list[str]) -> list[str]:
        notes: list[str] = []
        for style in styles:
            lower = style.lower()
            if "gto" in lower:
                desc = "balanced candidate; no equilibrium policy supplied"
            elif "random" in lower:
                desc = "unstable candidate; no randomness estimate supplied"
            elif "conservative" in lower or "nit" in lower:
                desc = "risk-averse candidate; no passivity evidence supplied"
            elif "aggressive" in lower or "maniac" in lower or lower == "lag":
                desc = "pressure candidate; no aggression evidence supplied"
            elif "calling" in lower or "honest" in lower:
                desc = "direct-play candidate; no call evidence supplied"
            elif "bluffer" in lower:
                desc = "bluffing candidate; no bluff evidence supplied"
            elif "greedy" in lower:
                desc = "high-reward candidate; no greed evidence supplied"
            elif "sacrificial" in lower:
                desc = "future-payoff candidate; no sacrifice evidence supplied"
            elif "mirror" in lower:
                desc = "reactive candidate; no mirroring evidence supplied"
            elif lower == "tag":
                desc = "selective-pressure candidate; no style evidence supplied"
            else:
                desc = "candidate label only; no style evidence supplied"
            notes.append(f"- {style}: {desc}.")
        return notes

    def _pad_to_target_chars(self, parts: list[str], insert_index: int | None = None) -> None:
        if self.target_chars <= 0:
            return

        current = len("\n".join(parts))
        remaining = self.target_chars - current
        if remaining <= 0:
            return

        padding = ["\n#### PLACEBO LENGTH CONTROL ####\n"]
        filler = [
            "Neutral control note: this sentence adds task-format context without "
            "strategic evidence, opponent evidence, Nash information, recent-window "
            "statistics, or memory-derived advice.",
            "Neutral control note: evaluate only the real rules, state, history, and "
            "legal actions; this placeholder text is present for prompt-length "
            "matching and should not alter beliefs.",
            "Neutral control note: keep candidate opponent styles possible but "
            "uniform; no label is favored by this block and no action is endorsed.",
        ]
        idx = 0
        while len("\n".join(parts[:insert_index] + padding + parts[insert_index:]) if insert_index is not None else "\n".join(parts + padding)) < self.target_chars:
            combined = (
                parts[:insert_index] + padding + parts[insert_index:]
                if insert_index is not None
                else parts + padding
            )
            remaining = self.target_chars - len("\n".join(combined))
            line = filler[idx % len(filler)]
            if remaining <= len(line) + 1:
                padding.append(line[:max(0, remaining - 1)])
                break
            padding.append(line)
            idx += 1
        if insert_index is None:
            parts.extend(padding)
        else:
            parts[insert_index:insert_index] = padding
