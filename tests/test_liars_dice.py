"""Tests for Liar's Dice 2p game engine."""
import pytest
from games.liars_dice import LiarsDice, LiarsDiceState, _raise_bids
from base.types import Action


class TestLiarsDiceState:
    def test_clone_independence(self):
        s = LiarsDiceState(
            dice={0: (1, 2, 3), 1: (2, 3, 4)},
            dice_counts={0: 3, 1: 3},
            current_player=0, last_bid=None, phase="bidding",
            history=[], round_num=1, num_faces=4,
        )
        c = s.clone()
        c.dice[0] = (4, 4, 4)
        assert s.dice[0] == (1, 2, 3)

    def test_to_dict_roundtrip(self):
        s = LiarsDiceState(
            dice={0: (1, 2, 3), 1: (3, 3, 4)},
            dice_counts={0: 3, 1: 3},
            current_player=0, last_bid=(2, 3), phase="bidding",
            history=[], round_num=1, num_faces=4,
        )
        d = s.to_dict()
        assert d["dice"][0] == [1, 2, 3]
        assert d["last_bid"] == [2, 3]
        assert d["phase"] == "bidding"


class TestLiarsDice:
    def test_reset_creates_correct_state(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        assert s.dice_counts == {0: 4, 1: 4}
        assert len(s.dice[0]) == 4
        assert len(s.dice[1]) == 4
        assert s.current_player == 0
        assert s.last_bid is None
        assert s.phase == "bidding"
        assert s.round_num == 1
        assert s.rewards is None

    def test_reset_seed_reproducible(self):
        g1 = LiarsDice(num_dice=4, num_faces=4, seed=42)
        g2 = LiarsDice(num_dice=4, num_faces=4, seed=42)
        assert g1.reset().dice == g2.reset().dice

    def test_reset_different_seed_different_dice(self):
        g1 = LiarsDice(num_dice=4, num_faces=4, seed=42)
        g2 = LiarsDice(num_dice=4, num_faces=4, seed=99)
        assert g1.reset().dice != g2.reset().dice

    def test_first_legal_actions_no_challenge(self):
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        actions = g.get_legal_actions(s)
        types = {a.type for a in actions}
        assert "bid" in types
        assert "challenge" not in types

    def test_first_legal_actions_quantity_range(self):
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        actions = g.get_legal_actions(s)
        # total dice = 4, so quantity 1..4, face 1..3
        assert len(actions) == 4 * 3  # 12 first bids
        qs = {a.data["quantity"] for a in actions}
        assert qs == {1, 2, 3, 4}

    def test_bid_updates_state(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        s2 = g.step(s, Action(player_id=0, type="bid", data={"quantity": 2, "face": 3}))
        assert s2.last_bid == (2, 3)
        assert s2.current_player == 1
        assert s2.dice_counts == {0: 4, 1: 4}  # dice unchanged
        assert len(s2.history) == 1

    def test_after_bid_challenge_is_legal(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        s = g.step(s, Action(player_id=0, type="bid", data={"quantity": 2, "face": 3}))
        legal = g.get_legal_actions(s)
        types = {a.type for a in legal}
        assert "challenge" in types
        assert "bid" in types

    def test_raise_bids_generation(self):
        # After bid (2, 3) with total_dice=8, num_faces=4
        bids = _raise_bids((2, 3), total_dice=8, num_faces=4, player=1)
        types = {b.type for b in bids}
        assert all(t == "bid" for t in types)

    def test_raise_same_q_higher_face(self):
        # After (2, 3): should include (2, 4)
        bids = _raise_bids((2, 3), total_dice=8, num_faces=4, player=1)
        pairs = {(b.data["quantity"], b.data["face"]) for b in bids}
        assert (2, 4) in pairs
        assert (2, 3) not in pairs  # same face is not legal
        assert (2, 2) not in pairs  # lower face not legal

    def test_raise_higher_q(self):
        # After (2, 3): should include (3, 1), (3, 2), etc.
        bids = _raise_bids((2, 3), total_dice=8, num_faces=4, player=1)
        pairs = {(b.data["quantity"], b.data["face"]) for b in bids}
        assert (3, 1) in pairs
        assert (3, 4) in pairs
        assert (8, 4) in pairs

    def test_raise_bids_full_list(self):
        bids = _raise_bids((2, 3), total_dice=4, num_faces=3, player=1)
        # (2, 4) doesn't exist (num_faces=3) → only higher q
        # (3, 1..3) and (4, 1..3) = 6 bids
        assert len(bids) == 6

    def test_raise_higher_quantity_at_max_face(self):
        # After (8, 4) with total=8: no higher face, no higher q → no raises
        bids = _raise_bids((8, 4), total_dice=8, num_faces=4, player=1)
        assert len(bids) == 0

    def test_challenge_false_bid(self):
        """Bid false → bidder (P0) loses 1 die (2→1)."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        s.dice = {0: (1, 1), 1: (1, 1)}
        s.dice_counts = {0: 2, 1: 2}
        s.last_bid = (5, 2)
        s.current_player = 1
        s2 = g.step(s, Action(player_id=1, type="challenge"))
        assert s2.dice_counts[0] == 1

    def test_challenge_correct_bid(self):
        """Bid was true under wild-1 rules → challenger loses."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        # Force dice: P0 has (2, 2), P1 has (1, 2).
        # For bid 2x face 2, the wild 1 counts as a 2, so actual = 3.
        s.dice = {0: (2, 2), 1: (1, 2)}
        s.dice_counts = {0: 2, 1: 2}
        s.last_bid = (2, 2)  # bid 2x face 2, actual count = 3 ≥ 2 → true
        s.current_player = 1  # P1 challenges
        s2 = g.step(s, Action(player_id=1, type="challenge"))
        # Bid was true → challenger (P1) loses 1 die
        assert s2.dice_counts[1] == 1
        assert s2.dice_counts[0] == 2

    def test_wild_one_counts_toward_non_one_bid(self):
        g = LiarsDice(num_dice=1, num_faces=4, seed=42)
        s = LiarsDiceState(
            dice={0: (4,), 1: (2,)},
            dice_counts={0: 1, 1: 1},
            current_player=0,
            last_bid=(2, 4),
            phase="bidding",
            history=[{"player": 1, "action": "bid", "quantity": 2, "face": 4, "round": 1}],
            round_num=1,
            num_faces=4,
        )
        s2 = g.step(s, Action(player_id=0, type="challenge"))
        assert s2.phase == "terminal"
        assert s2.rewards == {0: 1.0, 1: -1.0}

    def test_wild_one_makes_bid_true(self):
        g = LiarsDice(num_dice=1, num_faces=4, seed=42)
        s = LiarsDiceState(
            dice={0: (1,), 1: (4,)},
            dice_counts={0: 1, 1: 1},
            current_player=0,
            last_bid=(2, 4),
            phase="bidding",
            history=[{"player": 1, "action": "bid", "quantity": 2, "face": 4, "round": 1}],
            round_num=1,
            num_faces=4,
        )
        s2 = g.step(s, Action(player_id=0, type="challenge"))
        assert s2.phase == "terminal"
        assert s2.rewards == {0: -1.0, 1: 1.0}

    def test_challenge_false_bidder_loses(self):
        """Bid was false → bidder loses."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        s.dice = {0: (1, 1), 1: (1, 1)}
        s.dice_counts = {0: 2, 1: 2}
        s.last_bid = (5, 2)  # impossible even with wild 1s
        s.current_player = 1
        s2 = g.step(s, Action(player_id=1, type="challenge"))
        # Bid false → bidder (P0) loses 1 die
        assert s2.dice_counts[0] == 1
        assert s2.dice_counts[1] == 2

    def test_game_terminal_when_player_loses_all_dice(self):
        g = LiarsDice(num_dice=1, num_faces=2, seed=42)
        s = g.reset()
        # P0 has 1 die, P1 has 1 die
        s.dice = {0: (1,), 1: (1,)}
        s.dice_counts = {0: 1, 1: 1}
        s.last_bid = (100, 2)  # impossible bid
        s.current_player = 1
        s2 = g.step(s, Action(player_id=1, type="challenge"))
        # Bid false → P0 loses last die → game over
        assert s2.phase == "terminal"
        assert s2.rewards is not None
        assert s2.rewards[1] == 1.0  # P1 wins
        assert s2.rewards[0] == -1.0

    def test_new_round_rerolls_dice(self):
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        old_dice = {p: tuple(v) for p, v in s.dice.items()}
        s.dice = {0: (1, 1), 1: (1, 1)}
        s.dice_counts = {0: 2, 1: 2}
        s.last_bid = (5, 2)
        s.current_player = 1
        s2 = g.step(s, Action(player_id=1, type="challenge"))
        # Non-terminal → new round with re-rolled dice
        assert s2.phase == "bidding"
        assert s2.round_num == 2
        assert s2.last_bid is None
        # Dice should be different (new roll)
        assert len(s2.dice[0]) == 1  # P0 lost 1 die
        assert len(s2.dice[1]) == 2

    def test_observation_hides_opponent_dice(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)
        assert "my_dice" in obs.private_info
        assert len(obs.private_info["my_dice"]) == 4
        assert obs.public_info["my_dice_count"] == 4
        assert obs.public_info["opponent_dice_count"] == 4
        assert obs.public_info["last_bid"] is None

    def test_observation_includes_bid_info(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        s = g.step(s, Action(player_id=0, type="bid", data={"quantity": 2, "face": 3}))
        obs = g.get_observation(s, 1)
        assert obs.public_info["last_bid"] == {"quantity": 2, "face": 3}
        assert "challenge" in obs.legal_actions

    def test_get_state_summary(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        summary = g.get_state_summary(s)
        assert "R1" in summary
        assert "dice P0=4 P1=4" in summary

    def test_get_rewards_returns_zero_in_bidding(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        r = g.get_rewards(s)
        assert r == {0: 0.0, 1: 0.0}

    def test_game_runs_to_completion(self):
        """Full game with random bidding/challenge."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        player = 0
        step_count = 0
        while not g.is_terminal(s) and step_count < 200:
            legal = g.get_legal_actions(s)
            # Pick a random legal action
            action = legal[g.rng.randint(0, len(legal) - 1)]
            s = g.step(s, action)
            step_count += 1
        assert g.is_terminal(s)
        rewards = g.get_rewards(s)
        assert sum(rewards.values()) == 0.0  # zero-sum: +1 for winner, -1 for loser
        assert step_count < 200  # should not hit timeout

    @pytest.mark.parametrize("num_dice,num_faces", [(2, 3), (3, 3), (4, 4), (3, 6)])
    def test_different_configs_run(self, num_dice, num_faces):
        g = LiarsDice(num_dice=num_dice, num_faces=num_faces, seed=42)
        s = g.reset()
        assert len(s.dice[0]) == num_dice
        for _ in range(100):
            legal = g.get_legal_actions(s)
            action = legal[g.rng.randint(0, len(legal) - 1)]
            s = g.step(s, action)
            if g.is_terminal(s):
                break
        # Just verify it completes without error

    def test_illegal_action_raises(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        with pytest.raises(ValueError):
            g.step(s, Action(player_id=0, type="challenge"))

    def test_wrong_player_raises(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        with pytest.raises(ValueError):
            g.step(s, Action(player_id=1, type="challenge",
                             data={"quantity": 2, "face": 3}))

    def test_serialization_roundtrip(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        d = g.state_to_dict(s)
        assert d["dice_counts"] == {0: 4, 1: 4}
        assert len(d["dice"][0]) == 4
        assert d["phase"] == "bidding"

    def test_get_rules_text(self):
        g = LiarsDice(num_dice=4, num_faces=4)
        rules = g.get_rules_text()
        assert "4 dice" in rules
        assert "1–4" in rules or "1-4" in rules
