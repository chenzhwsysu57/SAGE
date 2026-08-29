"""Tests for Liar's Dice CFR+ trainer."""
import pytest
import json
import os
import tempfile
from types import SimpleNamespace

from methods.liars_dice_cfr import (
    LiarsDiceCFRTrainer, LiarsDiceCFRMethod, CFRState,
    encode_infoset, encode_action, decode_action,
    regret_matching_strategy, converged_strategy,
    opponent_reach,
)
from base.liars_dice_core import infoset_key_from_observation, legal_action_keys
from base.liars_dice_core import match_legal_action
from games.liars_dice import LiarsDice
from base.types import Action, Observation
from base.interfaces import BaseModel


def _write_minimal_strategy(tmpdir, state, player_id, legal_actions):
    path = os.path.join(tmpdir, "cfr_strategy.json")
    probs = {}
    weight = 1.0 / len(legal_actions)
    for action in legal_actions:
        probs[encode_action(action)] = weight
    data = {
        "num_dice": 2,
        "num_faces": 3,
        "iterations": 1,
        "traverser_action_sample_size": 1,
        "strategy": {
            encode_infoset(state, player_id): probs,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestEncoding:
    def test_encode_infoset_first_bid(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        key = encode_infoset(s, 0)
        assert "dice=" in key
        assert "bid=None" in key
        assert "counts=4,4" in key
        assert "player=0" in key

    def test_encode_infoset_with_bid(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        s2 = g.step(s, Action(player_id=0, type="bid",
                               data={"quantity": 2, "face": 3}))
        key = encode_infoset(s2, 1)
        assert "bid=2,3" in key
        assert "counts=4,4" in key
        assert "player=1" in key

    def test_infoset_from_observation_matches_state(self):
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        s = g.step(s, Action(player_id=0, type="bid",
                              data={"quantity": 2, "face": 3}))
        obs = g.get_observation(s, 1)
        assert infoset_key_from_observation(obs) == encode_infoset(s, 1)

    def test_encode_action_challenge(self):
        a = Action(player_id=0, type="challenge")
        assert encode_action(a) == "challenge"

    def test_encode_action_bid(self):
        a = Action(player_id=0, type="bid",
                   data={"quantity": 3, "face": 4})
        assert encode_action(a) == "bid_3_4"

    def test_decode_challenge(self):
        legals = [
            Action(player_id=1, type="challenge"),
            Action(player_id=1, type="bid", data={"quantity": 3, "face": 1}),
        ]
        a = decode_action("challenge", 1, legals)
        assert a.type == "challenge"

    def test_decode_bid(self):
        legals = [
            Action(player_id=1, type="challenge"),
            Action(player_id=1, type="bid", data={"quantity": 3, "face": 1}),
        ]
        a = decode_action("bid_3_1", 1, legals)
        assert a.type == "bid"
        assert a.data["quantity"] == 3
        assert a.data["face"] == 1

    def test_match_legal_action_rejects_unknown_type(self):
        legal = [
            Action(player_id=0, type="challenge"),
            Action(player_id=0, type="bid", data={"quantity": 1, "face": 1}),
        ]
        bogus = Action(player_id=0, type="your_move", data={"quantity": 1, "face": 1})
        assert match_legal_action(bogus, legal) is None

    def test_canonical_legal_action_order(self):
        first = legal_action_keys(total_dice=4, num_faces=3, last_bid=None)
        assert first[:3] == ("bid_1_1", "bid_1_2", "bid_1_3")
        assert first[-3:] == ("bid_4_1", "bid_4_2", "bid_4_3")

        raised = legal_action_keys(total_dice=4, num_faces=4, last_bid=(2, 3))
        assert raised[0] == "challenge"
        assert raised[1] == "bid_2_4"
        assert raised[-1] == "bid_4_4"


class TestRegretMatching:
    def test_positive_regrets(self):
        strat = regret_matching_strategy([5, 0, 0], 3)
        assert strat[0] == pytest.approx(1.0)
        assert strat[1] == pytest.approx(0.0)

    def test_all_positive(self):
        strat = regret_matching_strategy([3, 1, 0], 3)
        assert strat[0] == pytest.approx(0.75)
        assert strat[1] == pytest.approx(0.25)
        assert strat[2] == pytest.approx(0.0)

    def test_all_negative_uniform(self):
        strat = regret_matching_strategy([-1, -2, -3], 3)
        assert all(s == pytest.approx(1/3) for s in strat)

    def test_converged_strategy(self):
        res = converged_strategy({"bid_2_3": 10, "bid_3_1": 30, "challenge": 60})
        assert res["bid_2_3"] == pytest.approx(0.1)
        assert res["bid_3_1"] == pytest.approx(0.3)
        assert res["challenge"] == pytest.approx(0.6)

    def test_opponent_reach(self):
        assert opponent_reach(0.25, 0.6, 0) == pytest.approx(0.6)
        assert opponent_reach(0.25, 0.6, 1) == pytest.approx(0.25)

    def test_wild_one_counts_in_trainer_step(self):
        trainer = LiarsDiceCFRTrainer(num_dice=1, num_faces=4, iterations=1, seed=0)
        state = CFRState(
            dice0=(1,),
            dice1=(4,),
            count0=1,
            count1=1,
            current_player=0,
            last_bid=(2, 4),
            phase="bidding",
            round_num=1,
            rewards=None,
        )
        next_state = trainer._step(state, (0,))
        assert next_state.phase == "terminal"
        assert next_state.rewards == (-1.0, 1.0)


class TestCFRTrainer:
    def test_train_small_converges(self):
        """Quick training test: 100 iterations on 2×3."""
        trainer = LiarsDiceCFRTrainer(
            num_dice=2, num_faces=3, iterations=100, seed=42)
        trainer.train(verbose=False)
        assert trainer._trained
        assert len(trainer.strategy) > 0

    def test_train_4x4_few_iterations(self):
        """Verify 4×4 runs without error for a small number of iterations."""
        trainer = LiarsDiceCFRTrainer(
            num_dice=4, num_faces=4, iterations=100, seed=42,
            traverser_action_sample_size=1)
        trainer.train(verbose=False)
        assert trainer._trained
        assert len(trainer.strategy) > 0

    def test_save_load_roundtrip(self):
        trainer = LiarsDiceCFRTrainer(
            num_dice=2, num_faces=3, iterations=100, seed=42)
        trainer.train(verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            trainer.save(path)

        trainer2 = LiarsDiceCFRTrainer()
        trainer2.load(path)
        assert trainer2._trained
        assert trainer2._num_infosets == trainer._num_infosets
        os.unlink(path)

    def test_strategy_probs_sum_to_one(self):
        trainer = LiarsDiceCFRTrainer(
            num_dice=2, num_faces=3, iterations=200, seed=42)
        trainer.train(verbose=False)

        for infoset, strat in trainer.strategy.items():
            total = sum(strat.values())
            assert abs(total - 1.0) < 1e-6, f"Infoset {infoset} sums to {total}"

    def test_regret_update_uses_opponent_reach(self):
        trainer = LiarsDiceCFRTrainer(
            num_dice=1, num_faces=2, iterations=1, seed=0,
            traverser_action_sample_size=2)

        state = CFRState(
            dice0=(1,),
            dice1=(1,),
            count0=1,
            count1=1,
            current_player=0,
            last_bid=None,
            phase="bidding",
            round_num=1,
            rewards=None,
        )
        terminal_win = CFRState(
            dice0=(1,),
            dice1=(1,),
            count0=1,
            count1=1,
            current_player=-1,
            last_bid=None,
            phase="terminal",
            round_num=1,
            rewards=(1.0, 0.0),
        )
        terminal_loss = CFRState(
            dice0=(1,),
            dice1=(1,),
            count0=1,
            count1=1,
            current_player=-1,
            last_bid=None,
            phase="terminal",
            round_num=1,
            rewards=(0.0, 1.0),
        )

        trainer._legal_keys = lambda _state: ((1, 1, 1), (1, 1, 2))
        trainer._step = (
            lambda _state, key:
            terminal_win if key == (1, 1, 1) else terminal_loss
        )

        trainer._walk(state, p0_reach=0.25, p1_reach=0.6, traverser=0)

        infoset = (0, (1,), 1, 1, None)
        regrets = trainer.regret_sum[infoset]
        assert regrets[(1, 1, 1)] == pytest.approx(0.3)
        assert regrets[(1, 1, 2)] == pytest.approx(0.0)
        assert trainer.strategy_sum[infoset][(1, 1, 1)] == pytest.approx(0.3)
        assert trainer.strategy_sum[infoset][(1, 1, 2)] == pytest.approx(0.3)


class TestCFRMethod:
    def test_act_returns_valid_action(self):
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)
        legal = g.get_legal_actions(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_minimal_strategy(tmpdir, s, 0, legal)
            method = LiarsDiceCFRMethod(
                name="CFR+", seed=99,
                strategy_path=path)

            action, info, action_type = method.act(obs, legal)
            assert action is not None
            assert action.player_id == 0
            assert action.type in {a.type for a in legal}
            assert "selected=" in info
            assert action_type == action.type

    def test_act_consistent_with_legal(self):
        """Ensure the CFR strategy only selects actions that are legal."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        s = g.step(s, Action(player_id=0, type="bid",
                              data={"quantity": 2, "face": 2}))
        obs = g.get_observation(s, 1)
        legal = g.get_legal_actions(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_minimal_strategy(tmpdir, s, 1, legal)
            method = LiarsDiceCFRMethod(
                name="CFR+", seed=99,
                strategy_path=path)

            action, info, _ = method.act(obs, legal)
            # Bid 2 of face 1 is legal (higher quantity, lower face is ok)
            assert action.type in {a.type for a in legal}
            assert "selected=" in info
            if action.type == "bid":
                pair = (action.data["quantity"], action.data["face"])
                # Should be a valid raise
                assert any(
                    a.data.get("quantity") == pair[0] and a.data.get("face") == pair[1]
                    for a in legal if a.type == "bid"
                )

    def test_reflect_noop(self):
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        legal = g.get_legal_actions(s)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_minimal_strategy(tmpdir, s, 0, legal)
            method = LiarsDiceCFRMethod(strategy_path=path)
            assert method.reflect(None) == []
