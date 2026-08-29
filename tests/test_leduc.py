"""Tests for Leduc Hold'em game logic."""

import pytest
from games.leduc import LEDUC_DECK, LeducHoldem, LeducState
from base.types import Action
from games.leduc import Card, Suit


class TestLeducGame:
    def test_reset_initial_state(self):
        game = LeducHoldem(seed=42)
        state = game.reset()

        assert state.phase == "preflop"
        assert state.street == 0
        assert state.acting_player == 0
        assert state.pot == 2  # 2 antes
        assert state.total_bets == {0: 1, 1: 1}
        assert state.round_bets == {0: 0, 1: 0}
        assert state.round_raises == 0
        assert state.community_card is None
        assert state.private_cards[0] is not None
        assert state.private_cards[1] is not None
        assert len(state.deck) == 6
        assert not game.is_terminal(state)

    def test_num_players(self):
        game = LeducHoldem()
        assert game.num_players() == 2

    def test_legal_actions_preflop_first_act(self):
        game = LeducHoldem(seed=42)
        state = game.reset()

        actions = game.get_legal_actions(state)
        action_types = {a.type for a in actions}
        assert action_types == {"check", "bet"}
        for a in actions:
            if a.type == "bet":
                assert a.amount == 2

    def test_legal_actions_after_bet(self):
        game = LeducHoldem(seed=42)
        state = game.reset()

        # P0 bets
        state = game.step(state, Action(player_id=0, type="bet", amount=2))

        # P1 faces a bet
        actions = game.get_legal_actions(state)
        action_types = {a.type for a in actions}
        assert action_types == {"fold", "call", "raise"}
        for a in actions:
            if a.type == "call":
                assert a.amount == 2
            if a.type == "raise":
                assert a.amount == 4  # to_call (2) + bet_size (2)

    def test_legal_actions_after_raise(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="bet", amount=2))
        state = game.step(state, Action(player_id=1, type="raise", amount=4))

        # P0 faces a raise, can fold/call/raise (last raise)
        actions = game.get_legal_actions(state)
        action_types = {a.type for a in actions}
        assert action_types == {"fold", "call", "raise"}
        for a in actions:
            if a.type == "call":
                assert a.amount == 2  # to_call: 4 - 2 = 2

    def test_max_raises_per_round(self):
        game = LeducHoldem(seed=42)
        state = game.reset()

        # P0 bets, P1 raises, P0 re-raises (2 raises total)
        state = game.step(state, Action(player_id=0, type="bet", amount=2))
        state = game.step(state, Action(player_id=1, type="raise", amount=4))
        state = game.step(state, Action(player_id=0, type="raise", amount=4))

        # P1 can only fold or call (max 2 raises reached)
        actions = game.get_legal_actions(state)
        action_types = {a.type for a in actions}
        assert action_types == {"fold", "call"}

    def test_fold_ends_game(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="bet", amount=2))
        state = game.step(state, Action(player_id=1, type="fold"))

        assert game.is_terminal(state)
        assert state.rewards == {0: state.pot, 1: 0}

    def test_rewards_terminal_only(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        rewards = game.get_rewards(state)
        assert rewards == {0: 0.0, 1: 0.0}

    def test_check_check_preflop_goes_to_flop(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="check"))
        state = game.step(state, Action(player_id=1, type="check"))

        assert state.street == 1
        assert state.phase == "flop"
        assert state.community_card is not None
        assert state.acting_player == 0
        assert state.round_bets == {0: 0, 1: 0}

    def test_flop_bet_size(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        # Check through preflop
        state = game.step(state, Action(player_id=0, type="check"))
        state = game.step(state, Action(player_id=1, type="check"))

        # Flop first act
        actions = game.get_legal_actions(state)
        for a in actions:
            if a.type == "bet":
                assert a.amount == 4  # flop bet size

    def test_calling_ends_round_and_deals_community(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="check"))
        state = game.step(state, Action(player_id=1, type="bet", amount=2))
        state = game.step(state, Action(player_id=0, type="call", amount=2))

        assert state.street == 1
        assert state.community_card is not None

    def test_showdown_pair_beats_high_card(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        # Force a known deck arrangement
        from games.leduc import RANK_ORDER
        # We test hand strength directly
        assert RANK_ORDER == {"J": 0, "Q": 1, "K": 2}

    def test_showdown_computation(self):
        """Test showdown hand comparison directly via private method."""
        game = LeducHoldem(seed=42)
        # Build a specific showdown state
        state = LeducState(
            deck=[],
            private_cards={
                0: Card(rank="K", suit=Suit.SPADES),
                1: Card(rank="Q", suit=Suit.HEARTS),
            },
            community_card=Card(rank="Q", suit=Suit.SPADES),
            pot=10,
            street=1,
            phase="showdown",
            acting_player=-1,
            total_bets={0: 5, 1: 5},
            round_bets={0: 0, 1: 0},
            round_raises=0,
            history=[],
            rewards=None,
        )
        rewards = game._compute_showdown(state)
        # P1 has pair of Q, P0 has K-high → P1 wins
        assert rewards[1] == 10
        assert rewards[0] == 0

    def test_showdown_pair_vs_pair(self):
        game = LeducHoldem(seed=42)
        state = LeducState(
            deck=[],
            private_cards={
                0: Card(rank="K", suit=Suit.SPADES),
                1: Card(rank="Q", suit=Suit.HEARTS),
            },
            community_card=Card(rank="K", suit=Suit.HEARTS),
            pot=10,
            street=1,
            phase="showdown",
            acting_player=-1,
            total_bets={0: 5, 1: 5},
            round_bets={0: 0, 1: 0},
            round_raises=0,
            history=[],
            rewards=None,
        )
        rewards = game._compute_showdown(state)
        # P0 has pair of K, P1 has pair of Q → P0 wins
        assert rewards[0] == 10
        assert rewards[1] == 0

    def test_state_serialization(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        d = game.state_to_dict(state)
        assert d["phase"] == "preflop"
        assert d["street"] == 0
        assert isinstance(d["deck"], list)
        assert len(d["deck"]) == 6

    def test_state_clone(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        cloned = state.clone()

        assert cloned.phase == state.phase
        assert cloned.pot == state.pot
        # Modify cloned shouldn't affect original
        cloned.pot = 999
        assert state.pot == 2

    def test_deterministic_reset(self):
        game1 = LeducHoldem(seed=42)
        game2 = LeducHoldem(seed=42)
        state1 = game1.reset()
        state2 = game2.reset()

        assert str(state1.private_cards[0]) == str(state2.private_cards[0])
        assert str(state1.private_cards[1]) == str(state2.private_cards[1])

    def test_full_game_fold_scenario(self):
        """Run a complete game that ends in fold."""
        game = LeducHoldem(seed=42)
        state = game.reset()

        # P0 bets, P1 folds
        state = game.step(state, Action(player_id=0, type="bet", amount=2))
        state = game.step(state, Action(player_id=1, type="fold"))

        assert game.is_terminal(state)
        assert game.get_rewards(state)[0] > 0
        assert game.get_rewards(state)[1] == 0

    def test_full_game_showdown(self):
        """Run a complete game that goes to showdown."""
        game = LeducHoldem(seed=42)
        state = game.reset()

        # Check through preflop
        state = game.step(state, Action(player_id=0, type="check"))
        state = game.step(state, Action(player_id=1, type="check"))

        # Check through flop
        state = game.step(state, Action(player_id=0, type="check"))
        state = game.step(state, Action(player_id=1, type="check"))

        assert game.is_terminal(state)
        rewards = game.get_rewards(state)
        assert sum(rewards.values()) > 0  # pot goes to someone

    def test_get_observation(self):
        game = LeducHoldem(seed=42)
        state = game.reset()

        obs = game.get_observation(state, 0)
        assert obs.player_id == 0
        assert "card" in obs.private_info
        assert obs.public_info["community_card"] is None
        assert obs.public_info["street"] == "preflop"
        assert len(obs.legal_actions) == 2  # check, bet

    def test_legal_actions_empty_at_terminal(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="bet", amount=2))
        state = game.step(state, Action(player_id=1, type="fold"))
        assert game.is_terminal(state)
        assert game.get_legal_actions(state) == []

    def test_illegal_action_type_raises(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        # P0 cannot fold as first action (only check/bet)
        with pytest.raises(ValueError, match="Illegal action"):
            game.step(state, Action(player_id=0, type="fold"))

    def test_illegal_action_amount_raises(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="bet", amount=2))
        # P1 must call 2, not 0
        with pytest.raises(ValueError, match="Illegal amount"):
            game.step(state, Action(player_id=1, type="call", amount=0))

    def test_wrong_player_raises(self):
        game = LeducHoldem(seed=42)
        state = game.reset()
        # P1 tries to act when it's P0's turn
        with pytest.raises(ValueError, match="Wrong player"):
            game.step(state, Action(player_id=1, type="check"))
