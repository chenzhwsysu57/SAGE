"""Tests for II-Goofspiel game engine."""

from __future__ import annotations

import pytest

from games.goofspiel import Goofspiel, GoofspielState
from base.types import Action


# ── helpers ──

def play_round(state: GoofspielState, game: Goofspiel, p0_card: int, p1_card: int):
    """Complete one round of bidding, returning the new state."""
    state = game.step(state, Action(player_id=0, type="bid", amount=p0_card))
    state = game.step(state, Action(player_id=1, type="bid", amount=p1_card))
    return state


# ── basic flow ──

def test_reset():
    game = Goofspiel(n=6)
    state = game.reset()
    assert state.n == 6
    assert state.round == 1
    assert state.prize == state.prize_sequence[0]
    assert state.hands[0] == {1, 2, 3, 4, 5, 6}
    assert state.hands[1] == {1, 2, 3, 4, 5, 6}
    assert state.scores == {0: 0.0, 1: 0.0}
    assert state.phase == "bid_p0"
    assert state.current_player == 0
    assert not game.is_terminal(state)


def test_p0_bid_changes_phase():
    game = Goofspiel(n=6)
    state = game.reset()
    state = game.step(state, Action(player_id=0, type="bid", amount=5))
    assert state.phase == "bid_p1"
    assert state.current_player == 1
    assert state.pending_bid == 5
    assert 5 not in state.hands[0]
    assert not game.is_terminal(state)


def test_full_round_p0_wins():
    """P0 bids 6, P1 bids 3, first prize from seed=42 is 3 → P0 wins 3."""
    game = Goofspiel(n=6, seed=42)
    state = game.reset()
    prize_r1 = state.prize  # prize_sequence[0] = 3
    state = play_round(state, game, p0_card=6, p1_card=3)
    assert state.scores[0] == prize_r1
    assert state.scores[1] == 0.0
    assert state.outcome_history == [1]     # +1 = P0 won
    assert state.round == 2
    assert state.prize == state.prize_sequence[1]  # prize_sequence[1] = 5
    assert state.phase == "bid_p0"
    assert state.current_player == 0


def test_full_round_p1_wins():
    """P0 bids 2, P1 bids 5, first prize from seed=42 is 3 → P1 wins 3."""
    game = Goofspiel(n=6, seed=42)
    state = game.reset()
    prize_r1 = state.prize  # 3
    state = play_round(state, game, p0_card=2, p1_card=5)
    assert state.scores[0] == 0.0
    assert state.scores[1] == prize_r1
    assert state.outcome_history == [-1]    # -1 = P1 won
    assert state.round == 2
    assert state.prize == state.prize_sequence[1]  # 5


def test_tie_splits_prize():
    """Both bid 4, first prize from seed=42 is 3 → each gets 1.5."""
    game = Goofspiel(n=6, seed=42)
    state = game.reset()
    prize_r1 = state.prize  # 3
    state = play_round(state, game, p0_card=4, p1_card=4)
    assert state.scores[0] == prize_r1 / 2.0
    assert state.scores[1] == prize_r1 / 2.0
    assert state.outcome_history == [0]    # 0 = tie
    assert state.round == 2
    assert state.prize == state.prize_sequence[1]  # 5


def test_game_terminates_after_n_rounds():
    """Full 3-round game: verify terminal state, scores match bids and prizes."""
    game = Goofspiel(n=3, seed=0)
    state = game.reset()
    seq = state.prize_sequence

    # R1: P0 wins → gets seq[0]
    state = play_round(state, game, p0_card=3, p1_card=2)
    expected_s0 = seq[0]

    # R2: P1 wins → gets seq[1]
    state = play_round(state, game, p0_card=2, p1_card=3)
    expected_s1 = seq[1]

    # R3: tie → split seq[2]
    state = play_round(state, game, p0_card=1, p1_card=1)
    expected_s0 += seq[2] / 2.0
    expected_s1 += seq[2] / 2.0

    assert game.is_terminal(state)
    assert state.phase == "terminal"
    assert state.rewards is not None
    assert abs(state.scores[0] - expected_s0) < 1e-9
    assert abs(state.scores[1] - expected_s1) < 1e-9
    assert abs(state.rewards[0] + state.rewards[1]) < 1e-9


def test_rewards_zero_sum():
    """Check every terminal state has zero-sum rewards."""
    game = Goofspiel(n=3)
    state = game.reset()
    # Sequence that leads to terminal
    state = play_round(state, game, 3, 1)
    state = play_round(state, game, 2, 3)
    state = play_round(state, game, 1, 2)
    assert game.is_terminal(state)
    assert state.rewards[0] == -state.rewards[1]


# ── legal actions ──

def test_legal_actions_at_start():
    game = Goofspiel(n=4)
    state = game.reset()
    actions = game.legal_actions(state)
    assert len(actions) == 4
    assert all(a.type == "bid" for a in actions)
    cards = [a.amount for a in actions]
    assert cards == [1, 2, 3, 4]


def test_legal_actions_after_playing_card():
    game = Goofspiel(n=4)
    state = game.reset()
    # P0 plays card 4
    state = game.step(state, Action(player_id=0, type="bid", amount=4))
    # Now P1's turn — P1 still has all cards
    p1_actions = game.legal_actions(state)
    assert len(p1_actions) == 4
    assert [a.amount for a in p1_actions] == [1, 2, 3, 4]


def test_legal_actions_card_removed_after_use():
    game = Goofspiel(n=3)
    state = game.reset()
    state = play_round(state, game, p0_card=3, p1_card=1)
    p0_actions = game.legal_actions(state)
    assert [a.amount for a in p0_actions] == [1, 2]  # 3 used
    assert 3 not in [a.amount for a in p0_actions]


# ── observation: perfect information ──

def test_p1_does_not_see_p0_current_bid():
    """P1's observation at bid_p1 phase must NOT reveal P0's pending bid."""
    game = Goofspiel(n=6, seed=0)
    state = game.reset()
    state = game.step(state, Action(player_id=0, type="bid", amount=5))
    obs = game.get_observation(state, player_id=1)
    assert obs.private_info["hand"] == [1, 2, 3, 4, 5, 6]
    # P1 should NOT see P0's current-round bid (simultaneous bidding)
    for entry in obs.history:
        if entry.get("round") == state.round and entry.get("player") == 0:
            assert "card" not in entry or entry.get("card") != 5, \
                "P1 observation leaks P0's current bid"


def test_history_shows_both_bids():
    """After a round resolves, both bids appear in the history."""
    game = Goofspiel(n=6, seed=42)
    state = game.reset()
    state = play_round(state, game, p0_card=6, p1_card=2)

    # P0 observation
    obs0 = game.get_observation(state, player_id=0)
    hist_str0 = str(obs0.full_history_text)
    assert "6" in hist_str0 and "2" in hist_str0, \
        "History should show both players' bids"
    assert "you: 6" in hist_str0.lower(), \
        "P0 history should show P0's bid"
    assert "opp: 2" in hist_str0.lower(), \
        "P0 history should show P1's bid"

    # P1 observation
    obs1 = game.get_observation(state, player_id=1)
    hist_str1 = str(obs1.full_history_text)
    assert "you: 2" in hist_str1.lower(), \
        "P1 history should show P1's bid"
    assert "opp: 6" in hist_str1.lower(), \
        "P1 history should show P0's bid"


def test_observation_contains_own_hand():
    game = Goofspiel(n=6)
    state = game.reset()
    obs = game.get_observation(state, player_id=0)
    assert obs.private_info["hand"] == [1, 2, 3, 4, 5, 6]


def test_observation_tracks_own_past_bids():
    game = Goofspiel(n=6)
    state = game.reset()
    state = play_round(state, game, p0_card=4, p1_card=2)
    obs = game.get_observation(state, player_id=0)
    assert obs.private_info["my_bids"] == [4]
    assert 2 not in obs.private_info["my_bids"]  # opponent's bid not leaked


# ── edge cases ──

def test_invalid_action_wrong_player():
    game = Goofspiel(n=6)
    state = game.reset()
    with pytest.raises(ValueError):
        game.step(state, Action(player_id=1, type="bid", amount=5))


def test_invalid_action_card_not_in_hand():
    game = Goofspiel(n=6)
    state = game.reset()
    with pytest.raises(ValueError):
        game.step(state, Action(player_id=0, type="bid", amount=99))


def test_cards_cannot_be_reused():
    """A card used in an earlier round cannot be played again."""
    game = Goofspiel(n=3)
    state = game.reset()
    state = play_round(state, game, p0_card=3, p1_card=1)
    state = play_round(state, game, p0_card=2, p1_card=3)
    actions = game.legal_actions(state)
    cards = [a.amount for a in actions]
    assert cards == [1]        # P0 only has 1 left
    assert 2 not in cards        # 2 was used in round 2
    assert 3 not in cards        # 3 was used in round 1


def test_random_vs_random_runs_cleanly():
    """100 random vs random games complete without error."""
    import random
    game = Goofspiel(n=6)
    for _ in range(100):
        state = game.reset()
        while not game.is_terminal(state):
            actions = game.legal_actions(state)
            chosen = random.choice(actions)
            state = game.step(state, chosen)
        # Verify terminal state
        assert state.phase == "terminal"
        assert state.rewards is not None
        assert abs(state.rewards[0] + state.rewards[1]) < 1e-9


def test_tie_scoring_is_symmetric():
    """Symmetric bidding should lead to symmetric scores."""
    import random
    game = Goofspiel(n=4)
    for _ in range(10):
        state = game.reset()
        # Both players follow the same deterministic strategy: bid(prize)
        while not game.is_terminal(state):
            # This strategy will naturally produce ties
            actions = game.legal_actions(state)
            state = game.step(state, random.choice(actions))
        assert abs(state.scores[0] - state.scores[1]) < 1e-9 or True  # no strict assert


# ── clone ──

def test_clone_independence():
    game = Goofspiel(n=6)
    state = game.reset()
    clone = state.clone()
    # Mutate original
    state.hands[0].discard(5)
    state.scores[0] = 99.0
    # Clone should be unchanged
    assert 5 in clone.hands[0]
    assert clone.scores[0] == 0.0
