"""Goofspiel (Game of Pure Strategy) — Simultaneous Bidding, Post-Round Reveal.

Rules:
  - N cards (1..N) per player, default 6
  - Prize sequence is drawn from a fixed pool of 48 permutations,
    ensuring large cards appear roughly evenly across round positions
  - Each round: P0 bids first (hidden from P1), then P1 bids (blind)
  - After both have bid, the round resolves: higher bid wins the prize (tie → split)
  - Both players' bids are revealed after resolution
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from typing import Any

from base.interfaces import BaseGame
from base.types import Action, Observation


RULES_TEXT_TEMPLATE = """Goofspiel — Simultaneous Bidding variant

Setup:
  The deck has cards 1 to {n}. Each player receives one full set.
  The prize sequence is drawn from a fixed pool of 48 permutations.

Each round:
  1. The current prize is announced.
  2. P0 secretly chooses a card (P1 does NOT see it).
  3. P1 secretly chooses a card (does NOT see P0's choice).
  4. Both bids are revealed simultaneously.
  5. Higher bid wins the prize (score = prize value).
     On a tie the prize is split equally between both players.
  6. Used cards are discarded. Both players' bids are now known.

Winning:
  After all {n} rounds the player with the highest total score wins.
"""


def load_prize_pool(n: int) -> list[tuple[int, ...]]:
    """Load the fixed prize pool from games/prize_pool_n{N}.txt."""
    path = os.path.join(os.path.dirname(__file__), f"prize_pool_n{n}.txt")
    pool = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pool.append(tuple(int(x) for x in line.split()))
    if not pool:
        raise FileNotFoundError(f"No prize sequences found in {path}")
    return pool


@dataclass(slots=True)
class GoofspielState:
    n: int
    round: int  # 1..n
    prize: int
    prize_sequence: tuple[int, ...]  # random permutation of n..1
    hands: dict[int, set[int]]
    scores: dict[int, float]
    outcome_history: list[int]         # per completed round: +1 P0 win, -1 P1 win, 0 tie
    pending_bid: int | None            # P0's bid (stored until P1 bids, then revealed)
    phase: str                         # "bid_p0" | "bid_p1" | "terminal"
    current_player: int
    history: list[dict[str, Any]] = field(default_factory=list)
    rewards: dict[int, float] | None = None

    def clone(self) -> "GoofspielState":
        return GoofspielState(
            n=self.n,
            round=self.round,
            prize=self.prize,
            prize_sequence=self.prize_sequence,
            hands={p: set(h) for p, h in self.hands.items()},
            scores=dict(self.scores),
            outcome_history=list(self.outcome_history),
            pending_bid=self.pending_bid,
            phase=self.phase,
            current_player=self.current_player,
            history=list(self.history),
            rewards=dict(self.rewards) if self.rewards is not None else None,
        )


class Goofspiel(BaseGame[GoofspielState]):
    """Goofspiel — perfect information, fixed prize pool."""

    def __init__(self, n: int = 5, seed: int | None = None) -> None:
        assert n >= 2, f"n must be >= 2, got {n}"
        self.n = n
        self._seed = seed
        self._prize_pool = load_prize_pool(n)
        self.rules_text = RULES_TEXT_TEMPLATE.format(n=n)

    def reset(self, seed: int | None = None) -> GoofspielState:
        rng_seed = seed if seed is not None else self._seed
        rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        prize_seq = rng.choice(self._prize_pool)
        hands = {0: set(range(1, self.n + 1)), 1: set(range(1, self.n + 1))}
        return GoofspielState(
            n=self.n,
            round=1,
            prize=prize_seq[0],
            prize_sequence=prize_seq,
            hands=hands,
            scores={0: 0.0, 1: 0.0},
            outcome_history=[],
            pending_bid=None,
            phase="bid_p0",
            current_player=0,
        )

    def step(self, state: GoofspielState, action: Action) -> GoofspielState:
        s = state.clone()

        # ── validate ──
        if action.player_id != s.current_player:
            raise ValueError(
                f"expected P{state.current_player}, got P{action.player_id}"
            )
        if action.type != "bid":
            raise ValueError(f"expected bid, got {action.type}")
        card = action.amount
        if card not in s.hands[action.player_id]:
            raise ValueError(
                f"card {card} not in P{action.player_id}'s hand {s.hands[action.player_id]}"
            )

        # ── P0 bids (stored, not revealed until P1 bids) ──
        if s.phase == "bid_p0":
            s.hands[action.player_id].discard(card)
            s.pending_bid = card
            s.phase = "bid_p1"
            s.current_player = 1
            # Don't append to history — P1 must not see P0's bid (simultaneous bidding)
            return s

        # ── P1 bids (resolve round — both bids revealed simultaneously) ──
        if s.phase == "bid_p1":
            p0_card = s.pending_bid
            p1_card = card
            s.hands[action.player_id].discard(card)

            # Resolve
            if p0_card > p1_card:
                outcome = 1   # P0 wins
                s.scores[0] += s.prize
            elif p0_card < p1_card:
                outcome = -1  # P1 wins
                s.scores[1] += s.prize
            else:
                outcome = 0   # tie — split
                s.scores[0] += s.prize / 2.0
                s.scores[1] += s.prize / 2.0

            s.outcome_history.append(outcome)

            # Append complete round info (both bids revealed simultaneously)
            outcome_str = {1: "P0 win", -1: "P1 win", 0: "Tie"}.get(outcome, "?")
            s.history.append({
                "round": s.round,
                "prize": s.prize,
                "p0_card": p0_card,
                "p1_card": p1_card,
                "outcome": outcome,
                "player": -1,
                "action": "round_result",
                "text": f"Round {s.round}: prize={s.prize}  "
                        f"P0={p0_card} vs P1={p1_card}  → {outcome_str}",
            })
            s.pending_bid = None

            # Check terminal
            if s.round >= s.n:
                s.phase = "terminal"
                score_diff = s.scores[0] - s.scores[1]
                s.rewards = {0: score_diff, 1: -score_diff}
                return s

            # Advance to next round
            s.round += 1
            s.prize = s.prize_sequence[s.round - 1]
            s.phase = "bid_p0"
            s.current_player = 0
            return s

        raise RuntimeError(f"Unexpected phase: {s.phase}")

    def legal_actions(self, state: GoofspielState) -> list[Action]:
        return [
            Action(player_id=state.current_player, type="bid", amount=c)
            for c in sorted(state.hands[state.current_player])
        ]

    def is_terminal(self, state: GoofspielState) -> bool:
        return state.phase == "terminal"

    def get_observation(self, state: GoofspielState, player_id: int | None = None) -> Observation:
        pid = player_id if player_id is not None else state.current_player
        hand = sorted(state.hands[pid])
        past_bids = [
            entry["p0_card"] if pid == 0 else entry["p1_card"]
            for entry in state.history
            if entry.get("p0_card") is not None
        ]

        return Observation(
            player_id=pid,
            history=list(state.history),
            private_info={
                "hand": hand,
                "my_bids": past_bids,
            },
            public_info={
                "round": state.round,
                "prize": state.prize,
                "prize_sequence": list(state.prize_sequence),
                "scores": dict(state.scores),
                "outcome_history": list(state.outcome_history),
            },
            legal_actions=[f"bid_{c}" for c in hand],
            rules_text=self.rules_text,
            state_text=self._render_state(state, pid),
            full_history_text=self._render_history(state, pid),
        )

    # ── BaseGame abstract methods ──

    def get_legal_actions(self, state: GoofspielState) -> list[Action]:
        return self.legal_actions(state)

    def get_rewards(self, state: GoofspielState) -> dict[int, float]:
        if state.rewards is not None:
            return dict(state.rewards)
        return {0: 0.0, 1: 0.0}

    def get_current_player(self, state: GoofspielState) -> int:
        return state.current_player

    def state_to_dict(self, state: GoofspielState) -> dict[str, Any]:
        return {
            "n": state.n,
            "round": state.round,
            "prize": state.prize,
            "prize_sequence": list(state.prize_sequence),
            "hands": {str(p): sorted(h) for p, h in state.hands.items()},
            "scores": state.scores,
            "outcome_history": state.outcome_history,
            "phase": state.phase,
            "current_player": state.current_player,
            "rewards": state.rewards,
        }

    def num_players(self) -> int:
        return 2

    def get_rules_text(self) -> str:
        return self.rules_text

    def state_to_text(self, state: GoofspielState, player_id: int) -> str:
        return self._render_state(state, player_id)

    # ── internal rendering hooks ──

    def _render_state(self, state: GoofspielState, pid: int) -> str:
        hand = sorted(state.hands[pid])
        scores = state.scores
        return (
            f"Round {state.round}/{self.n}  |  Prize = {state.prize}\n"
            f"Remaining prizes: {list(state.prize_sequence[state.round - 1:])}\n"
            f"Your hand: {hand}\n"
            f"Score — You: {scores[pid]:g}  Opp: {scores[1 - pid]:g}"
        )

    def _render_history(self, state: GoofspielState, pid: int) -> str:
        lines = ["=== Completed rounds ==="]
        for k in range(1, state.round):
            prize = state.prize_sequence[k - 1]
            outcome = state.outcome_history[k - 1]

            # Find the bids from history
            p0_card = None
            p1_card = None
            for entry in state.history:
                if entry.get("round") == k:
                    if entry.get("p0_card") is not None:
                        p0_card = entry["p0_card"]
                        p1_card = entry["p1_card"]

            if outcome == 1:
                who = "YOU" if pid == 0 else "OPPONENT"
            elif outcome == -1:
                who = "OPPONENT" if pid == 0 else "YOU"
            else:
                who = "TIE"

            if p0_card is not None and p1_card is not None:
                your_bid = p0_card if pid == 0 else p1_card
                opp_bid = p1_card if pid == 0 else p0_card
                lines.append(
                    f"  Round {k}: prize={prize} → {who} won "
                    f"(you: {your_bid}, opp: {opp_bid})"
                )
            else:
                lines.append(f"  Round {k}: prize={prize} → {who} won")
        return "\n".join(lines) if lines else "(no completed rounds)"
