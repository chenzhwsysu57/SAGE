from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from enum import StrEnum

from base.interfaces import BaseGame
from base.types import Action, Observation

class Suit(StrEnum):
    SPADES = "S"
    HEARTS = "H"


class Card:
    def __init__(self, rank: str, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.value}"

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit


class LeducState:
    """Immutable-ish state for Leduc Hold'em."""

    def __init__(
        self,
        *,
        deck: list[Card],
        private_cards: dict[int, Card | None],
        community_card: Card | None,
        pot: int,
        street: int,
        phase: str,
        acting_player: int,
        total_bets: dict[int, int],
        round_bets: dict[int, int],
        round_raises: int,
        history: list[dict[str, Any]],
        rewards: dict[int, float] | None,
        max_raises_per_round: int = 2,
    ):
        self.deck = deck
        self.private_cards = private_cards
        self.community_card = community_card
        self.pot = pot
        self.street = street
        self.phase = phase
        self.acting_player = acting_player
        self.total_bets = total_bets
        self.round_bets = round_bets
        self.round_raises = round_raises
        self.history = history
        self.rewards = rewards
        self.max_raises_per_round = max_raises_per_round

    def clone(self) -> LeducState:
        return LeducState(
            deck=list(self.deck),
            private_cards=dict(self.private_cards),
            community_card=self.community_card,
            pot=self.pot,
            street=self.street,
            phase=self.phase,
            acting_player=self.acting_player,
            total_bets=dict(self.total_bets),
            round_bets=dict(self.round_bets),
            round_raises=self.round_raises,
            history=list(self.history),
            rewards=dict(self.rewards) if self.rewards else None,
            max_raises_per_round=self.max_raises_per_round,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deck": [str(c) for c in self.deck],
            "private_cards": {p: str(c) if c else None for p, c in self.private_cards.items()},
            "community_card": str(self.community_card) if self.community_card else None,
            "pot": self.pot,
            "street": self.street,
            "phase": self.phase,
            "acting_player": self.acting_player,
            "total_bets": dict(self.total_bets),
            "round_bets": dict(self.round_bets),
            "round_raises": self.round_raises,
            "history": list(self.history),
            "rewards": dict(self.rewards) if self.rewards else None,
        }


LEDUC_DECK = [
    Card(rank="J", suit=Suit.SPADES),
    Card(rank="J", suit=Suit.HEARTS),
    Card(rank="Q", suit=Suit.SPADES),
    Card(rank="Q", suit=Suit.HEARTS),
    Card(rank="K", suit=Suit.SPADES),
    Card(rank="K", suit=Suit.HEARTS),
]

RANK_ORDER = {"J": 0, "Q": 1, "K": 2}
LEDUC_RULES_TEXT = """Leduc Hold'em Poker Rules:

Players:
- 2 players, each antes 1 chip.

Deck:
- J♠ J♥ Q♠ Q♥ K♠ K♥ (6 cards)
- Each player is dealt 1 private card.
- 1 community card is revealed before Round 2.

Card Ranking:
- J < Q < K

Game Structure:
- Round 1 (preflop)
- Round 2 (flop after community card is revealed)

Betting Rules:
- Legal actions: fold, check, bet, call, raise
- bet = first wager in a round when no bet exists
- call = match current current bet exactly
- raise = increase current bet by the round bet size

Betting State:
- At any time, there is a current bet amount (0 if no bet has been made)
- A player who bets becomes the current aggressor
- A raise increases the current bet and resets opponent response
- Each raise counts toward the raise limit
- Each round resets current bet and raise counter

Round 1:
- Base bet size = 2 chips
- Max raises = 2 per player per round

Round 2:
- Base bet size = 4 chips
- Max raises = 2 per player per round

Turn Rules:
- If current bet = 0: player may check or bet
- If current bet > 0: player may call, raise, or fold

Important:
- bet and raise both set/modify the current bet level
- call always matches the current bet exactly

Showdown:
- Each player forms best hand from:
  - private card + community card (if available)
  - otherwise private card only

Hand Evaluation:
- Pair if private card matches community card
- Otherwise high card

Hand Strength:
- Pair always beats any high card
- Pair ranking: K pair > Q pair > J pair
- High card ranking: K > Q > J

Tie Rule:
- If same hand type and same rank → split pot

End Conditions:
- If a player folds → opponent wins pot immediately
- After Round 2 betting ends → showdown occurs
"""

class LeducHoldem(BaseGame[LeducState]):
    """Leduc Hold'em poker variant.

    2 players, 6-card deck (J/Q/K × 2 suits).
    1 private card each, 1 community card.
    2 betting rounds with max 2 raises each.
    """

    NUM_PLAYERS = 2
    ANTE = 1
    PREFLOP_BET = 2
    FLOP_BET = 4

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def num_players(self) -> int:
        return self.NUM_PLAYERS

    def _bet_size(self, state: LeducState) -> int:
        return self.PREFLOP_BET if state.street == 0 else self.FLOP_BET

    def reset(self) -> LeducState:
        deck = self.rng.sample(LEDUC_DECK, len(LEDUC_DECK))
        return LeducState(
            deck=deck,
            private_cards={0: deck[0], 1: deck[1]},
            community_card=None,
            pot=self.ANTE * self.NUM_PLAYERS,
            street=0,
            phase="preflop",
            acting_player=0,
            total_bets={0: self.ANTE, 1: self.ANTE},
            round_bets={0: 0, 1: 0},
            round_raises=0,
            history=[],
            rewards=None,
        )

    def get_current_player(self, state: LeducState) -> int:
        return state.acting_player

    def get_legal_actions(self, state: LeducState) -> list[Action]:
        if self.is_terminal(state):
            return []

        p = state.acting_player
        to_call = state.round_bets[1 - p] - state.round_bets[p]
        bet_size = self._bet_size(state)

        actions: list[Action] = []

        if to_call == 0:
            actions.append(Action(player_id=p, type="check"))
            actions.append(Action(player_id=p, type="bet", amount=bet_size))
        else:
            actions.append(Action(player_id=p, type="fold"))
            actions.append(Action(player_id=p, type="call", amount=to_call))
            if state.round_raises < state.max_raises_per_round:
                raise_amount = to_call + bet_size
                actions.append(Action(player_id=p, type="raise", amount=raise_amount))

        return actions

    def step(self, state: LeducState, action: Action) -> LeducState:
        # Validate legality before applying
        legal_actions = self.get_legal_actions(state)
        legal_by_type = {a.type: a for a in legal_actions}

        if action.type not in legal_by_type:
            raise ValueError(
                f"Illegal action '{action.type}' for P{action.player_id}. "
                f"Legal: {list(legal_by_type.keys())}"
            )
        expected = legal_by_type[action.type]
        if action.amount != expected.amount:
            raise ValueError(
                f"Illegal amount {action.amount} for '{action.type}'. "
                f"Expected: {expected.amount}"
            )
        if action.player_id != state.acting_player:
            raise ValueError(
                f"Wrong player P{action.player_id}. Current: P{state.acting_player}"
            )

        state = state.clone()
        p = state.acting_player

        state.history.append({
            "player": p,
            "action": action.type,
            "amount": action.amount,
            "street": state.street,
        })

        if action.type == "fold":
            winner = 1 - p
            state.rewards = {winner: state.pot, p: 0}
            state.phase = "terminal"
            state.acting_player = -1
            return state

        if action.type in ("check", "call", "bet", "raise"):
            state.pot += action.amount
            state.total_bets[p] += action.amount
            state.round_bets[p] += action.amount
            if action.type == "raise":
                state.round_raises += 1

        # Determine next actor and phase transitions
        if action.type == "call":
            # Call ends the betting round
            if state.street == 0:
                # Deal community card and move to flop
                state.community_card = state.deck[2]
                state.street = 1
                state.phase = "flop"
                state.round_bets = {0: 0, 1: 0}
                state.round_raises = 0
                # Player who acted first in preflop acts first in flop
                state.acting_player = 0
            else:
                # Showdown
                state.phase = "showdown"
                state.rewards = self._compute_showdown(state)
                state.phase = "terminal"
                state.acting_player = -1
            return state

        if action.type == "check":
            # Round ends if both players have checked consecutively
            checks_this_street = sum(
                1 for h in state.history
                if h.get("action") == "check" and h.get("street") == state.street
            )
            if checks_this_street >= 2:
                if state.street == 0:
                    state.community_card = state.deck[2]
                    state.street = 1
                    state.phase = "flop"
                    state.round_bets = {0: 0, 1: 0}
                    state.round_raises = 0
                    state.acting_player = 0
                else:
                    state.phase = "showdown"
                    state.rewards = self._compute_showdown(state)
                    state.phase = "terminal"
                    state.acting_player = -1
                return state

            state.acting_player = 1 - p
            return state

        # bet or raise: action continues to next player
        state.acting_player = 1 - p
        return state

    def get_observation(self, state: LeducState, player_id: int) -> Observation:
        p = player_id
        opponent = 1 - p
        legal_actions = self.get_legal_actions(state)

        public_info: dict[str, Any] = {
            "community_card": str(state.community_card) if state.community_card else None,
            "pot": state.pot,
            "street": "preflop" if state.street == 0 else "flop",
            "my_bet_total": state.total_bets[p],
            "opponent_bet_total": state.total_bets[opponent],
            "my_bet_this_round": state.round_bets[p],
            "opponent_bet_this_round": state.round_bets[opponent],
            "round_raises": state.round_raises,
            "to_call": state.round_bets[opponent] - state.round_bets[p],
        }

        return Observation(
            player_id=p,
            private_info={"card": str(state.private_cards[p])} if state.private_cards[p] else {},
            public_info=public_info,
            history=state.history,
            legal_actions=[a.type for a in legal_actions],
            rules_text=self.get_rules_text(),
            state_text=self.state_to_text(state, p),
        )

    def get_static_state(self, state: LeducState) -> dict[str, Any]:
        return {"deck": [str(c) for c in state.deck]}

    def get_state_summary(self, state: LeducState) -> str:
        street = "preflop" if state.street == 0 else "flop"
        parts = [f"pot={state.pot}", street]

        # Show private cards (visible to reader for debugging)
        cards = ", ".join(
            f"P{p}={str(state.private_cards[p])}"
            for p in sorted(state.private_cards)
            if state.private_cards[p]
        )
        if cards:
            parts.append(cards)

        if state.community_card:
            parts.append(f"board={state.community_card}")

        if state.phase == "terminal":
            parts.append("TERMINAL")

        return " | ".join(parts)

    def get_rules_text(self) -> str:
        return LEDUC_RULES_TEXT

    def state_to_text(self, state: LeducState, player_id: int) -> str:
        p = player_id
        opponent = 1 - p
        parts = [
            f"Your card: {state.private_cards[p]}",
            f"Community card: {state.community_card or 'not revealed yet'}",
            f"Street: {'preflop' if state.street == 0 else 'flop'}",
            f"Pot: {state.pot} chips",
            f"Your total bet: {state.total_bets[p]}",
            f"Opponent total bet: {state.total_bets[opponent]}",
            f"Amount to call: {state.round_bets[opponent] - state.round_bets[p]}",
        ]
        if state.history:
            parts.append("History:")
            for h in state.history:
                action_str = f"  P{h['player']} {h['action']}s"
                if h.get("amount"):
                    action_str += f" {h['amount']}"
                parts.append(action_str)

        return "\n".join(parts)

    def is_terminal(self, state: LeducState) -> bool:
        return state.phase == "terminal"

    def get_rewards(self, state: LeducState) -> dict[int, float]:
        if state.rewards is None:
            return {0: 0.0, 1: 0.0}
        return dict(state.rewards)

    def get_net_profit(self, state: LeducState, player_id: int) -> float:
        """Net profit: pot allocation minus player's total bets."""
        payout = self.get_rewards(state).get(player_id, 0.0)
        contribution = state.total_bets.get(player_id, 0)
        return payout - contribution

    def state_to_dict(self, state: LeducState) -> dict[str, Any]:
        return state.to_dict()

    def _compute_showdown(self, state: LeducState) -> dict[int, float]:
        """Compare hands and award pot to winner."""
        c0 = state.private_cards[0]
        c1 = state.private_cards[1]
        cc = state.community_card
        assert c0 is not None and c1 is not None and cc is not None

        def hand_strength(pc: Card) -> int:
            if pc.rank == cc.rank:
                # Pair: base 100 + rank value
                return 100 + RANK_ORDER[pc.rank]
            # High card
            return RANK_ORDER[pc.rank]

        s0 = hand_strength(c0)
        s1 = hand_strength(c1)

        if s0 > s1:
            return {0: state.pot, 1: 0}
        elif s1 > s0:
            return {0: 0, 1: state.pot}
        else:
            return {0: state.pot / 2, 1: state.pot / 2}
