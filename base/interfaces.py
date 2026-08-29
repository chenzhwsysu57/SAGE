from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from base.types import Action, Observation

StateType = TypeVar("StateType")


class BaseGame(ABC, Generic[StateType]):
    """Abstract game interface.

    Games are responsible ONLY for:
    - state transition
    - legality checking
    - rewards
    - observations
    """

    @abstractmethod
    def reset(self) -> StateType:
        """Return the initial game state."""
        ...

    @abstractmethod
    def step(self, state: StateType, action: Action) -> StateType:
        """Apply action to state and return new state (immutable)."""
        ...

    @abstractmethod
    def get_observation(self, state: StateType, player_id: int) -> Observation:
        """Return the structured observation for a player."""
        ...

    @abstractmethod
    def get_legal_actions(self, state: StateType) -> list[Action]:
        """Return all legal actions from the current state."""
        ...

    @abstractmethod
    def is_terminal(self, state: StateType) -> bool:
        """Check if the game has ended."""
        ...

    @abstractmethod
    def get_rewards(self, state: StateType) -> dict[int, float]:
        """Return rewards for all players. Only valid at terminal states."""
        ...

    @abstractmethod
    def get_current_player(self, state: StateType) -> int:
        """Return the player whose turn it is."""
        ...

    @abstractmethod
    def state_to_dict(self, state: StateType) -> dict[str, Any]:
        """Serialize state to a plain dict for logging."""
        ...

    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players."""
        ...

    @abstractmethod
    def get_rules_text(self) -> str:
        """Return the full rules of the game as text."""
        ...

    @abstractmethod
    def state_to_text(self, state: StateType, player_id: int) -> str:
        """Return a text description of the game state from a player's perspective."""
        ...

    def get_static_state(self, state: StateType) -> dict[str, Any]:
        """Return unchanging parts of state stored once per trajectory."""
        return {}

    def get_net_profit(self, state: StateType, player_id: int) -> float:
        """Return the net profit for a player at a terminal state.

        Default: use get_rewards directly (works for net zero-sum encodings).
        Override for games where get_rewards returns gross pot allocation
        (e.g., Leduc) to compute payout - total_bets.
        """
        return self.get_rewards(state).get(player_id, 0.0)

    def get_state_summary(self, state: StateType) -> str:
        """Return a one-line summary of the current state for verbose debugging.

        Override in game-specific subclasses to show relevant info
        (e.g., street, community card). Default returns empty string.
        """
        return ""


class BaseModel(ABC):
    """Abstract model interface.

    Models are thin wrappers around LLM APIs.
    They receive text prompts and return text responses.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the model and return raw response text."""
        ...


class BaseMethod(ABC):
    """Abstract method (strategy/reasoning) interface.

    Methods are responsible for:
    - rendering prompts from observations
    - calling models
    - parsing outputs into actions

    Methods are decoupled from both games and models.
    """

    @abstractmethod
    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple[Action, str, str]:
        """Return (action, rendered_prompt, raw_response)."""
        ...
