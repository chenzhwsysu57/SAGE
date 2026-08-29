from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    player_id: int
    type: str
    amount: int = 0
    data: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        if self.amount > 0:
            return f"P{self.player_id} {self.type}s {self.amount}"
        if self.data:
            return f"P{self.player_id} {self.type}: {self.data}"
        return f"P{self.player_id} {self.type}s"


class Observation(BaseModel):
    player_id: int
    private_info: dict[str, Any] = Field(default_factory=dict)
    public_info: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    legal_actions: list[str] = Field(default_factory=list)
    rules_text: str = ""
    state_text: str = ""
    full_history_text: str = ""
    role_guidance: str = ""


class TrajectoryStep(BaseModel):
    step: int
    player_id: int
    observation: Observation
    rendered_prompt: str
    raw_llm_response: str
    parsed_action: Action | None = None
    action_valid: bool = False
    retry_count: int = 0
    rejection_errors: list[str] = Field(default_factory=list)
    retry_attempts: list[dict[str, Any]] = Field(default_factory=list)
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    method_data: dict[str, Any] | None = None


class AnalysisCall(BaseModel):
    """LLM call log for analysis/reflection phases (outside main game loop)."""
    phase: str  # "classify" or "reflect"
    player_id: int
    prompt: str
    response: str
    parsed: str = ""


class Trajectory(BaseModel):
    game_name: str
    players: list[str]
    player_methods: dict[int, str] = Field(default_factory=dict)
    rules_text: str = ""
    static_state: dict[str, Any] = Field(default_factory=dict)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    payout: dict[int, float] = Field(default_factory=dict)
    net_profit: dict[int, float] = Field(default_factory=dict)
    winner: int | None = None
    analysis: list[AnalysisCall] = Field(default_factory=list)


class GameResult(BaseModel):
    """A single game's result within a consolidated run log."""
    game_index: int
    seed: int | None = None
    static_state: dict[str, Any] = Field(default_factory=dict)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    payout: dict[int, float] = Field(default_factory=dict)
    winner: int | None = None


class RunSummary(BaseModel):
    """Aggregated stats across all games in a run."""
    total_payout: dict[int, float] = Field(default_factory=dict)
    net_chips: dict[int, float] = Field(default_factory=dict)
    wins: dict[int, float] = Field(default_factory=dict)
    winrate: dict[int, float] = Field(default_factory=dict)


class RunLog(BaseModel):
    """Consolidated log for a multi-game run, sharing rules/metadata across games."""
    game_name: str
    players: list[str]
    player_methods: dict[int, str] = Field(default_factory=dict)
    rules_text: str = ""
    model: str = ""
    temperature: float = 0.7
    seed: int | None = None
    num_games: int = 0
    timestamp: str = ""
    games: list[GameResult] = Field(default_factory=list)
    summary: RunSummary | None = None
