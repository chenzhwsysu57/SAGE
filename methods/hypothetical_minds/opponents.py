"""Canonical six-style HM opponents with strict local/LLM execution."""

from methods.agent_pro.opponents import LocalSpecifiedOpponent, opponent_config
from methods.agent_pro.style_opponents import (
    STYLES, is_llm_style, parse_style_action, renderer,
)

from .protocol import canonical, request

SUPPORTED = {game: tuple(styles) for game, styles in STYLES.items()}


class CanonicalOpponent:
    def __init__(self, game: str, name: str, seed: int, output_retries: int = 3):
        if name not in SUPPORTED.get(game, ()):
            raise ValueError(f"Unsupported HM opponent {game}/{name}")
        self.game, self.name = game, name
        self.config = opponent_config(game, name)
        self.output_retries = output_retries
        self.is_llm = is_llm_style(game, name)
        self.render = renderer(game, name) if self.is_llm else None
        self.actor = None if self.is_llm else LocalSpecifiedOpponent(game, self.config, seed)

    def metadata(self):
        return {**self.config, "hm_opponent_protocol": (
            "original_benchmark_style_strict_parse_v1" if self.is_llm
            else "local_canonical_policy_v1")}

    def act(self, observation, legal, model=None):
        if not legal or any(action.player_id != observation.player_id for action in legal):
            raise ValueError("Opponent requires its own nonempty legal-action set")
        if not self.is_llm:
            action, _, _, data = self.actor.act(observation, legal, None)
            return action, {"opponent": self.name, "execution": self.config["execution"],
                            "policy": self.config.get("policy"),
                            "fallback": data["fallback_used"], "calls": []}
        if model is None:
            raise ValueError("LLM style opponent requires the logged HM journal")
        trace = []
        original = self.render._render_prompt(observation, legal)

        def validate(row):
            return parse_style_action(canonical(row), self.game, legal)

        action = request(model, {
            "protocol": "hm_canonical_opponent_v1", "stage": "opponent_action",
            "game": self.game, "opponent_style": self.name,
            "task": "Follow the original benchmark style prompt below. Return one strict legal action as the final JSON object. No fallback or aliases.",
            "original_style_prompt": original,
            "valid_actions": [action.model_dump() for action in legal],
            "output_schema": {"action": "one complete action object from valid_actions"},
        }, validate, self.output_retries, trace)
        return action, {"opponent": self.name, "execution": "llm_style",
                        "policy": self.config["prompt_protocol"], "fallback": False,
                        "calls": trace}


def make_opponent(game: str, name: str, seed: int):
    return CanonicalOpponent(game, name, seed)