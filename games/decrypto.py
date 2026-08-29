"""Decrypto game implementation.

3 players: Alice (encoder), Bob (decoder), Eve (interceptor).
4 keywords numbered 1-4.
Each turn the game assigns a 3-digit code to Alice, who gives 3 hints.
Bob and Eve guess.

Win conditions:
- {self.miscommunication_threshold} miscommunications (Bob wrong) → Eve wins
- {self.interception_threshold} interceptions (Eve right) → Eve wins
- Survive {self.max_turns} turns → Alice & Bob win
"""

from __future__ import annotations

import os
import random
from typing import Any

from base.interfaces import BaseGame
from base.types import Action, Observation

_word_path = os.path.join(os.path.dirname(__file__), "decrypto_words.txt")
with open(_word_path) as f:
    WORDLIST = [w.strip() for w in f.readlines()]

ALLOWED_DIGITS = {1, 2, 3, 4}
ALL_PERMUTATIONS = [
    (a, b, c) for a in (1, 2, 3, 4)
    for b in (1, 2, 3, 4)
    for c in (1, 2, 3, 4)
    if len({a, b, c}) == 3
]


class DecryptoState:
    def __init__(
        self,
        *,
        keywords: tuple[str, str, str, str],
        current_code: tuple[int, int, int] | None,
        code_history: list[tuple[int, int, int]],
        hint_history: dict[int, list[str]],
        turn: int,
        max_turns: int,
        miscommunications: int,
        interceptions: int,
        phase: str,
        acting_player: int,
        current_hints: tuple[str, str, str] | None,
        bob_guess: tuple[int, int, int] | None,
        eve_guess: tuple[int, int, int] | None,
        guess_history: list[dict] | None = None,
        rewards: dict[int, float] | None,
    ):
        self.keywords = keywords
        self.current_code = current_code
        self.code_history = code_history
        self.hint_history = hint_history
        self.turn = turn
        self.max_turns = max_turns
        self.miscommunications = miscommunications
        self.interceptions = interceptions
        self.phase = phase
        self.acting_player = acting_player
        self.current_hints = current_hints
        self.bob_guess = bob_guess
        self.eve_guess = eve_guess
        self.guess_history = guess_history if guess_history is not None else []
        self.rewards = rewards

    def clone(self) -> DecryptoState:
        return DecryptoState(
            keywords=self.keywords,
            current_code=self.current_code,
            code_history=list(self.code_history),
            hint_history={k: list(v) for k, v in self.hint_history.items()},
            turn=self.turn,
            max_turns=self.max_turns,
            miscommunications=self.miscommunications,
            interceptions=self.interceptions,
            phase=self.phase,
            acting_player=self.acting_player,
            current_hints=self.current_hints,
            bob_guess=self.bob_guess,
            eve_guess=self.eve_guess,
            guess_history=list(self.guess_history),
            rewards=dict(self.rewards) if self.rewards else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "current_code": list(self.current_code) if self.current_code else None,
            "code_history": [list(c) for c in self.code_history],
            "hint_history": {str(k): list(v) for k, v in self.hint_history.items()},
            "turn": self.turn,
            "max_turns": self.max_turns,
            "miscommunications": self.miscommunications,
            "interceptions": self.interceptions,
            "phase": self.phase,
            "acting_player": self.acting_player,
            "current_hints": list(self.current_hints) if self.current_hints else None,
            "bob_guess": list(self.bob_guess) if self.bob_guess else None,
            "eve_guess": list(self.eve_guess) if self.eve_guess else None,
            "guess_history": list(self.guess_history),
            "rewards": dict(self.rewards) if self.rewards else None,
        }


class Decrypto(BaseGame[DecryptoState]):
    NUM_PLAYERS = 3

    def __init__(self, seed: int | None = None, max_turns: int = 12,
                 interception_threshold: int = 3,
                 miscommunication_threshold: int = 3,
                 keywords: tuple[str, str, str, str] | None = None):
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.interception_threshold = interception_threshold
        self.miscommunication_threshold = miscommunication_threshold
        self._keywords = keywords

    def num_players(self) -> int:
        return self.NUM_PLAYERS

    def get_rules_text(self) -> str:
        return f"""Decrypto Rules:
- 3 players: Encoder (Alice), Decoder (Bob), Interceptor (Eve).
- 4 keywords numbered 1-4. Alice and Bob know them; Eve does NOT.
- Each turn:
  1. A 3-digit secret code (digits 1-4, no repeats) is assigned to Alice.
     She gives 3 hints, one for each code digit.
  2. Bob sees the hints + keywords and guesses the 3-digit code.
  3. Eve sees the hints only and guesses the 3-digit code.
- Win conditions:
  - Bob guesses wrong {self.miscommunication_threshold} times → Eve wins.
  - Eve guesses right {self.interception_threshold} times → Eve wins.
  - Survive {self.max_turns} turns without either → Alice & Bob win."""

    def _generate_code(self, state: DecryptoState) -> tuple[int, int, int]:
        """Pick a random code permutation not in code_history."""
        used = set(state.code_history)
        available = [p for p in ALL_PERMUTATIONS if p not in used]
        if not available:
            raise RuntimeError("No available codes — all 24 permutations used")
        return self.rng.choice(available)

    def state_to_text(self, state: DecryptoState, player_id: int) -> str:
        lines = [
            f"Turn: {state.turn + 1}/{state.max_turns}",
            (
                "Miscommunications (Bob wrong): "
                f"{state.miscommunications}/{self.miscommunication_threshold}"
            ),
            (
                "Interceptions (Eve right): "
                f"{state.interceptions}/{self.interception_threshold}"
            ),
        ]

        if player_id in (0, 1):  # Alice and Bob know keywords
            keywords_str = ", ".join(
                f"{i + 1}: {state.keywords[i]}" for i in range(4)
            )
            lines.append(f"Keywords: {keywords_str}")

        if player_id == 0 and state.current_code is not None:
            code_str = "-".join(str(d) for d in state.current_code)
            lines.append(f"Current code: {code_str}")

        if state.current_hints and player_id != 0:
            lines.append(f"Alice's hints this turn: {', '.join(state.current_hints)}")

        if state.hint_history:
            lines.append("Hint history (by keyword number):")
            for digit in sorted(state.hint_history.keys()):
                if state.hint_history[digit]:
                    hints_str = ", ".join(state.hint_history[digit])
                    lines.append(f"  Keyword {digit}: {hints_str}")

        if state.code_history:
            codes_str = ", ".join(
                "-".join(str(d) for d in c) for c in state.code_history
            )
            lines.append(f"Past codes: {codes_str}")

        return "\n".join(lines)

    def reset(self) -> DecryptoState:
        keywords = self._keywords if self._keywords else tuple(self.rng.sample(WORDLIST, 4))
        state = DecryptoState(
            keywords=keywords,
            current_code=None,
            code_history=[],
            hint_history={1: [], 2: [], 3: [], 4: []},
            turn=0,
            max_turns=self.max_turns,
            miscommunications=0,
            interceptions=0,
            phase="alice_hint",
            acting_player=0,
            current_hints=None,
            bob_guess=None,
            eve_guess=None,
            guess_history=[],
            rewards=None,
        )
        state.current_code = self._generate_code(state)
        return state

    def get_current_player(self, state: DecryptoState) -> int:
        return state.acting_player

    def get_legal_actions(self, state: DecryptoState) -> list[Action]:
        if self.is_terminal(state):
            return []

        p = state.acting_player
        if state.phase == "alice_hint":
            return [Action(player_id=p, type="hint")]
        elif state.phase in ("bob_guess", "eve_guess"):
            return [Action(player_id=p, type="guess")]
        return []

    def step(self, state: DecryptoState, action: Action) -> DecryptoState:
        legal_actions = self.get_legal_actions(state)
        legal_types = {a.type for a in legal_actions}

        if action.type not in legal_types:
            raise ValueError(
                f"Illegal action '{action.type}' for P{action.player_id}. "
                f"Legal: {list(legal_types)}"
            )
        if action.player_id != state.acting_player:
            raise ValueError(
                f"Wrong player P{action.player_id}. Current: P{state.acting_player}"
            )

        state = state.clone()

        if action.type == "hint":
            hints = action.data.get("hints", [])
            if len(hints) != 3:
                raise ValueError(f"Expected 3 hints, got {len(hints)}")
            state.current_hints = (str(hints[0]), str(hints[1]), str(hints[2]))
            state.phase = "bob_guess"
            state.acting_player = 1
            return state

        if action.type == "guess":
            code = _parse_guess_code(action)
            if len(code) != 3:
                raise ValueError(f"Expected 3-digit code, got {len(code)}")

            if state.phase == "bob_guess":
                state.bob_guess = tuple(int(d) for d in code)
                state.phase = "eve_guess"
                state.acting_player = 2
                return state

            if state.phase == "eve_guess":
                state.eve_guess = tuple(int(d) for d in code)
                state = self._evaluate_turn(state)
                return state

        return state

    def _evaluate_turn(self, state: DecryptoState) -> DecryptoState:
        hints = state.current_hints
        code = state.current_code
        assert hints is not None and code is not None

        state.code_history.append(code)

        state.guess_history.append({
            "code": list(code),
            "bob": list(state.bob_guess) if state.bob_guess else None,
            "eve": list(state.eve_guess) if state.eve_guess else None,
        })

        for i, digit in enumerate(code):
            state.hint_history[digit].append(hints[i])

        if state.bob_guess != code:
            state.miscommunications += 1

        if state.eve_guess == code:
            state.interceptions += 1

        if (state.miscommunications >= self.miscommunication_threshold or
                state.interceptions >= self.interception_threshold):
            state.rewards = {0: -1.0, 1: -1.0, 2: 1.0}
            state.phase = "terminal"
            state.acting_player = -1
            return state

        state.turn += 1
        if state.turn >= state.max_turns:
            state.rewards = {0: 1.0, 1: 1.0, 2: -1.0}
            state.phase = "terminal"
            state.acting_player = -1
            return state

        state.current_code = self._generate_code(state)
        state.current_hints = None
        state.bob_guess = None
        state.eve_guess = None
        state.phase = "alice_hint"
        state.acting_player = 0
        return state

    def get_observation(self, state: DecryptoState, player_id: int) -> Observation:
        legal_actions = self.get_legal_actions(state)

        return Observation(
            player_id=player_id,
            private_info=self._private_info(state, player_id),
            public_info=self._public_info(state, player_id),
            history=self._build_history(state, player_id),
            legal_actions=[a.type for a in legal_actions],
            rules_text=self.get_rules_text(),
            state_text=self.state_to_text(state, player_id),
        )

    def _build_history(self, state: DecryptoState, player_id: int) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for turn_idx in range(len(state.code_history)):
            code = state.code_history[turn_idx]
            gh = state.guess_history[turn_idx]
            history.append({
                "turn": turn_idx + 1,
                "player": 0,
                "phase": "encoder_hint",
                "action": "hint",
                "data": {"code": list(code)},
            })
            history.append({
                "turn": turn_idx + 1,
                "player": 1,
                "phase": "decoder_guess",
                "action": "guess",
                "data": {"code": list(code), "guess": gh["bob"]},
            })
            history.append({
                "turn": turn_idx + 1,
                "player": 2,
                "phase": "interceptor_guess",
                "action": "guess",
                "data": {"code": list(code), "guess": gh["eve"]},
            })
        return history

    def _private_info(self, state: DecryptoState, player_id: int) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if player_id in (0, 1):
            info["keywords"] = list(state.keywords)
        if player_id == 0 and state.current_code is not None:
            info["current_code"] = list(state.current_code)
        return info

    def _public_info(self, state: DecryptoState, player_id: int) -> dict[str, Any]:
        info: dict[str, Any] = {
            "turn": state.turn,
            "max_turns": self.max_turns,
            "miscommunications": state.miscommunications,
            "interceptions": state.interceptions,
            "interception_threshold": self.interception_threshold,
            "miscommunication_threshold": self.miscommunication_threshold,
            "phase": state.phase,
        }
        info["hint_history"] = {str(k): list(v) for k, v in state.hint_history.items()}
        # code_history: Encoder/Decoder only (not Interceptor)
        if player_id != 2:
            info["code_history"] = [list(c) for c in state.code_history]
        # guess_history: Encoder sees Bob+Eve, Decoder sees self
        if state.guess_history and player_id != 2:
            info["guess_history"] = [
                {
                    "code": list(gh["code"]),
                    "bob": list(gh["bob"]) if gh["bob"] else None,
                    "eve": list(gh["eve"]) if gh["eve"] else None,
                }
                for gh in state.guess_history
            ]
        if state.current_hints and player_id != 0:
            info["current_hints"] = list(state.current_hints)
        return info

    def is_terminal(self, state: DecryptoState) -> bool:
        return state.phase == "terminal"

    def get_rewards(self, state: DecryptoState) -> dict[int, float]:
        if state.rewards is None:
            return {0: 0.0, 1: 0.0, 2: 0.0}
        return dict(state.rewards)

    def state_to_dict(self, state: DecryptoState) -> dict[str, Any]:
        return state.to_dict()


def _parse_guess_code(action: Action) -> list[int]:
    """Parse guess code from action data. Supports:

    - ``"X-Y-Z"`` string
    - ``[X, Y, Z]`` list (or tuple) of ints/strings
    """
    code_data = action.data.get("code")
    if code_data is None:
        return []

    if isinstance(code_data, str):
        parts = code_data.split("-")
        return [int(p.strip()) for p in parts if p.strip()]

    if isinstance(code_data, (list, tuple)):
        return [int(x) for x in code_data]

    return []
