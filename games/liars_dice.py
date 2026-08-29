"""Two-player Liar's Dice (parametric M dice × N faces).

This module is intentionally thin:
  - it owns game state transition and observation rendering
  - it delegates action/infoset ordering to ``methods.liars_dice_core``

The previous implementation mixed multiple encodings and fallback paths.
This rewrite keeps the game engine as the single source of truth for rules
and makes the game-theoretic layer consume the same legal-action ordering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from base.interfaces import BaseGame
from base.types import Action, Observation
from base.liars_dice_core import (
    count_bid_matches,
    legal_action_keys,
    legal_actions_from_keys,
    match_legal_action,
)


RULES_TEXT_TEMPLATE = """Liar's Dice 2-Player Rules:

Setup:
  Each player rolls {num_dice} dice (faces 1–{num_faces}) secretly. Only you see your own dice.

Turns:
  Player P0 starts by making a bid.
  A bid is "M of face N" — "there are at least M dice showing N across ALL dice."
  The first bid can be any M >= 1 and N in [1..{num_faces}].

On your turn, you may:
  1. RAISE: make a higher bid.
     - Increase the quantity (M) to any face, OR
     - Keep the same quantity with a strictly higher face (N).
  2. CHALLENGE: call "Liar!" to end the round and reveal all dice.

Challenge resolution:
  All dice are revealed. Count how many show face N (the challenged face).
  - If N > 1, 1s are wild and also count toward the bid.
  - If N = 1, only 1s count.
  - Count >= M: the bid was TRUE → the CHALLENGER loses 1 die.
  - Count < M:  the bid was FALSE → the BIDDER loses 1 die.
  The player who lost the die starts the next round.

Winning:
  The first player to lose all their dice loses. The remaining player wins.
"""


@dataclass(slots=True)
class LiarsDiceState:
    """State container for the two-player game."""

    dice: dict[int, tuple[int, ...]]
    dice_counts: dict[int, int]
    current_player: int
    last_bid: tuple[int, int] | None
    phase: str
    history: list[dict[str, Any]]
    round_num: int
    num_faces: int
    rewards: dict[int, float] | None = None

    def clone(self) -> "LiarsDiceState":
        return LiarsDiceState(
            dice={p: tuple(v) for p, v in self.dice.items()},
            dice_counts=dict(self.dice_counts),
            current_player=self.current_player,
            last_bid=self.last_bid,
            phase=self.phase,
            history=list(self.history),
            round_num=self.round_num,
            num_faces=self.num_faces,
            rewards=dict(self.rewards) if self.rewards else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dice": {p: list(v) for p, v in self.dice.items()},
            "dice_counts": dict(self.dice_counts),
            "current_player": self.current_player,
            "last_bid": list(self.last_bid) if self.last_bid else None,
            "phase": self.phase,
            "history": list(self.history),
            "round_num": self.round_num,
            "num_faces": self.num_faces,
            "rewards": dict(self.rewards) if self.rewards else None,
        }


def _raise_bids(
    last_bid: tuple[int, int],
    total_dice: int,
    num_faces: int,
    player: int,
) -> list[Action]:
    """Generate raise actions in the canonical game order."""
    keys = legal_action_keys(
        total_dice=total_dice,
        num_faces=num_faces,
        last_bid=last_bid,
    )
    return [
        action
        for action in legal_actions_from_keys(keys, player)
        if action.type == "bid"
    ]


class LiarsDice(BaseGame[LiarsDiceState]):
    """Parametric Liar's Dice (2p, M dice × N faces)."""

    def __init__(self, num_dice: int = 4, num_faces: int = 4,
                 seed: int | None = None):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.rng = random.Random(seed)

    def num_players(self) -> int:
        return 2

    def _roll_dice(self, count: int) -> tuple[int, ...]:
        return tuple(self.rng.randint(1, self.num_faces) for _ in range(count))

    def reset(self) -> LiarsDiceState:
        dice_p0 = self._roll_dice(self.num_dice)
        dice_p1 = self._roll_dice(self.num_dice)
        return LiarsDiceState(
            dice={0: dice_p0, 1: dice_p1},
            dice_counts={0: self.num_dice, 1: self.num_dice},
            current_player=0,
            last_bid=None,
            phase="bidding",
            history=[],
            round_num=1,
            num_faces=self.num_faces,
        )

    def get_current_player(self, state: LiarsDiceState) -> int:
        return state.current_player

    def get_legal_actions(self, state: LiarsDiceState) -> list[Action]:
        if self.is_terminal(state):
            return []
        p = state.current_player
        total_dice = state.dice_counts[0] + state.dice_counts[1]
        keys = legal_action_keys(
            total_dice=total_dice,
            num_faces=state.num_faces,
            last_bid=state.last_bid,
        )
        return legal_actions_from_keys(keys, p)

    def step(self, state: LiarsDiceState, action: Action) -> LiarsDiceState:
        legal = self.get_legal_actions(state)
        match = match_legal_action(action, legal)
        if match is None:
            raise ValueError(
                f"Illegal action {action.model_dump()} for P{action.player_id}. "
                f"Legal actions: {[a.model_dump() for a in legal]}"
            )
        if action.player_id != state.current_player:
            raise ValueError(
                f"Wrong player P{action.player_id}. Current: P{state.current_player}"
            )
        # Use matched legal action (which has correct data fields)
        action = match

        state = state.clone()
        p = state.current_player

        if action.type == "bid":
            q = action.data["quantity"]
            f = action.data["face"]
            state.history.append({
                "player": p, "action": "bid",
                "quantity": q, "face": f,
                "round": state.round_num,
            })
            state.last_bid = (q, f)
            state.current_player = 1 - p
            return state

        # Challenge
        if action.type != "challenge":
            raise ValueError(f"expected challenge, got {action.type}")
        if state.last_bid is None:
            raise ValueError("no bid to challenge")
        q, f = state.last_bid

        # Reveal all dice and count
        all_dice = list(state.dice[0]) + list(state.dice[1])
        actual = count_bid_matches(all_dice, f)

        bid_true = actual >= q
        if bid_true:
            loser = p  # challenger loses
        else:
            loser = 1 - p  # bidder loses

        state.dice_counts[loser] -= 1

        state.history.append({
            "player": p, "action": "challenge",
            "round": state.round_num,
            "result": "correct" if bid_true else "false",
            "bid_quantity": q,
            "bid_face": f,
            "actual_count": actual,
            "loser": loser,
            "revealed_dice": {0: list(state.dice[0]), 1: list(state.dice[1])},
        })

        # Check terminal
        if state.dice_counts[loser] == 0:
            winner = 1 - loser
            state.phase = "terminal"
            state.rewards = {winner: 1.0, 1 - winner: -1.0}
            state.current_player = -1
            return state

        # New round: re-roll remaining dice, loser starts
        state.last_bid = None
        state.round_num += 1
        state.dice[0] = self._roll_dice(state.dice_counts[0])
        state.dice[1] = self._roll_dice(state.dice_counts[1])
        state.current_player = loser  # loser starts next round
        return state

    def get_observation(self, state: LiarsDiceState,
                        player_id: int) -> Observation:
        opp = 1 - player_id
        legal_actions = self.get_legal_actions(state)

        total_dice = state.dice_counts[0] + state.dice_counts[1]

        bid_info = None
        if state.last_bid:
            bid_info = {"quantity": state.last_bid[0], "face": state.last_bid[1]}

        public_info: dict[str, Any] = {
            "total_dice_remaining": total_dice,
            "my_dice_count": state.dice_counts[player_id],
            "opponent_dice_count": state.dice_counts[opp],
            "last_bid": bid_info,
            "round": state.round_num,
        }

        return Observation(
            player_id=player_id,
            private_info={
                "my_dice": list(state.dice[player_id]),
            },
            public_info=public_info,
            history=list(state.history),
            legal_actions=[a.type for a in legal_actions],
            rules_text=self.get_rules_text(),
            state_text=self.state_to_text(state, player_id),
        )

    def get_rules_text(self) -> str:
        return RULES_TEXT_TEMPLATE.format(
            num_dice=self.num_dice,
            num_faces=self.num_faces,
        )

    def state_to_text(self, state: LiarsDiceState, player_id: int) -> str:
        p = player_id
        opp = 1 - p
        dice_str = ", ".join(str(d) for d in state.dice[p])
        parts = [
            f"Your dice ({state.dice_counts[p]} left): {dice_str}",
            f"Opponent dice remaining: {state.dice_counts[opp]}",
            f"Total dice in play: {state.dice_counts[0] + state.dice_counts[1]}",
            f"Round: {state.round_num}",
        ]
        if state.last_bid:
            parts.append(
                f"Current bid: {state.last_bid[0]} of face {state.last_bid[1]}")
        else:
            parts.append("No bid yet — you must open the bidding.")

        if state.history:
            parts.append("History:")
            for h in state.history:
                if h["action"] == "bid":
                    parts.append(
                        f"  P{h['player']} bids {h['quantity']} of face {h['face']}")
                elif h["action"] == "challenge":
                    result = "correct" if h.get("result") == "correct" else "false"
                    parts.append(
                        f"  P{h['player']} challenges! "
                        f"Bid was {result}. "
                        f"P{h['loser']} loses a die.")

        return "\n".join(parts)

    def is_terminal(self, state: LiarsDiceState) -> bool:
        return state.phase == "terminal"

    def get_rewards(self, state: LiarsDiceState) -> dict[int, float]:
        if state.rewards is None:
            return {0: 0.0, 1: 0.0}
        return dict(state.rewards)

    def state_to_dict(self, state: LiarsDiceState) -> dict[str, Any]:
        return state.to_dict()

    def get_static_state(self, state: LiarsDiceState) -> dict[str, Any]:
        return {
            "num_dice": self.num_dice,
            "num_faces": self.num_faces,
        }

    def get_state_summary(self, state: LiarsDiceState) -> str:
        parts = [
            f"R{state.round_num}",
            f"dice P0={state.dice_counts[0]} P1={state.dice_counts[1]}",
        ]
        if state.last_bid:
            parts.append(f"bid={state.last_bid[0]}x{state.last_bid[1]}")
        if state.phase == "terminal":
            parts.append("TERMINAL")
        return " | ".join(parts)
