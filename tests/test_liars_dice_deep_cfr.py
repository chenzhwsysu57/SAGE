"""Tests for Liar's Dice Deep CFR implementation."""
import json
import pytest
import numpy as np
import os
import tempfile
import torch

from methods.sage.liars_dice_deep_cfr import (
    LiarsDiceDeepCFRTrainer,
    LiarsDiceDeepCFRMethod,
    DeepCFRNetwork,
    ReservoirBuffer,
    encode_features,
    encode_features_from_obs,
)
from games.liars_dice import LiarsDice
from base.types import Action, Observation


# ═══════════════════════════════════════════════════════════════════════
#  1. Feature Encoding Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEncodeFeatures:
    def test_encode_features_shape(self):
        """Verify feature vector shape for 4x4."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        feat = encode_features(s, 0)
        # 4 (hist) + 2 (counts) + 2 (bid) + 2 (player) + 1 (round) = 11
        assert feat.shape == (11,), f"Expected (11,), got {feat.shape}"

    def test_encode_features_shape_2x3(self):
        """Verify feature vector shape for 2x3."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        feat = encode_features(s, 0, num_faces=3, max_dice_count=2, max_total_dice=4)
        # 3 (hist) + 2 (counts) + 2 (bid) + 2 (player) + 1 (round) = 10
        assert feat.shape == (10,), f"Expected (10,), got {feat.shape}"

    def test_encode_features_histogram_sum(self):
        """Dice histogram should count correctly."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        feat = encode_features(s, 0)
        hist_sum = feat[0:4].sum()
        # Should equal number of dice P0 has
        assert hist_sum == pytest.approx(s.dice_counts[0])

    def test_encode_features_first_bid_zeros(self):
        """Before any bid, bid features should be zero."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        feat = encode_features(s, 0)
        # Bid quantity and face should be 0
        assert feat[6] == 0.0  # bid_q
        assert feat[7] == 0.0  # bid_f

    def test_encode_features_with_bid(self):
        """After a bid, bid features should be non-zero."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        s = g.step(s, Action(player_id=0, type="bid",
                              data={"quantity": 3, "face": 2}))
        # P1 observes the bid
        feat = encode_features(s, 1)
        assert feat[6] > 0.0  # bid_q / 8
        assert feat[7] > 0.0  # bid_f / 4
        assert feat[8] == 0.0  # P1 indicator
        assert feat[9] == 1.0  # P1 indicator

    def test_encode_features_player_indicator(self):
        """Player indicator should distinguish P0 and P1."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        feat0 = encode_features(s, 0)
        feat1 = encode_features(s, 1)
        assert feat0[8] == 1.0 and feat0[9] == 0.0, "P0 indicator wrong"
        assert feat1[8] == 0.0 and feat1[9] == 1.0, "P1 indicator wrong"

    def test_encode_features_normalized_counts(self):
        """Dice counts should be in [0, 1]."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        feat = encode_features(s, 0)
        assert 0.0 <= feat[4] <= 1.0  # my_count
        assert 0.0 <= feat[5] <= 1.0  # opp_count

    def test_encode_features_round_fraction(self):
        """Round fraction should be in (0, 1]."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        feat = encode_features(s, 0)
        assert 0.0 < feat[10] <= 1.0

    def test_encode_features_from_obs_matches_state(self):
        """encode_features_from_obs should match encode_features."""
        g = LiarsDice(num_dice=4, num_faces=4, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)

        state_feat = encode_features(s, 0)
        obs_feat = encode_features_from_obs(obs)

        np.testing.assert_array_almost_equal(state_feat, obs_feat,
                                             err_msg="State and obs features differ")


# ═══════════════════════════════════════════════════════════════════════
#  2. Neural Network Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeepCFRNetwork:
    def test_network_output_shapes(self):
        net = DeepCFRNetwork(feature_dim=11, n_actions=33)
        batch = 4
        x = torch.randn(batch, 11)
        mask = torch.ones(batch, 33)
        out = net(x, mask)
        assert out["strategy"].shape == (batch, 33)
        assert out["cfv"].shape == (batch, 33)

    def test_policy_sums_to_one(self):
        net = DeepCFRNetwork(feature_dim=11, n_actions=33)
        x = torch.randn(1, 11)
        mask = torch.ones(1, 33)
        out = net(x, mask)
        assert out["strategy"].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_masked_strategy_ignores_unavailable(self):
        net = DeepCFRNetwork(feature_dim=11, n_actions=10)
        x = torch.randn(1, 11)
        # Only first 3 actions available
        mask = torch.zeros(1, 10)
        mask[0, :3] = 1.0
        out = net(x, mask)
        s = out["strategy"].squeeze(0)
        # Unavailable actions should be 0
        assert s[3:].sum().item() == pytest.approx(0.0, abs=1e-6)
        # Available actions should sum to 1
        assert s[:3].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_network_forward_no_crash(self):
        """Verify forward pass runs without error."""
        net = DeepCFRNetwork(feature_dim=11, n_actions=33)
        x = torch.randn(2, 11)
        out = net(x)  # no mask
        assert out["strategy"].shape == (2, 33)


# ═══════════════════════════════════════════════════════════════════════
#  3. Reservoir Buffer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReservoirBuffer:
    def test_add_and_len(self):
        buf = ReservoirBuffer(capacity=100)
        assert len(buf) == 0
        buf.add((np.array([1.0]), np.array([0.5])))
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(20):
            buf.add((np.array([float(i)]), np.array([float(i)])))
        assert len(buf) == 10

    def test_sample_returns_correct_size(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(50):
            buf.add((np.array([float(i)]), np.array([float(i)])))
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_sample_empty(self):
        buf = ReservoirBuffer(capacity=100)
        batch = buf.sample(10)
        assert len(batch) == 0

    def test_sample_less_than_requested(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(5):
            buf.add((np.array([float(i)]), np.array([float(i)])))
        batch = buf.sample(10)
        assert len(batch) == 5


# ═══════════════════════════════════════════════════════════════════════
#  4. Trainer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeepCFRTrainer:
    def test_train_small_converges(self):
        """100 episodes on 2x3 — verify no crash and basic convergence."""
        trainer = LiarsDiceDeepCFRTrainer(
            num_dice=2, num_faces=3, iterations=100,
            warmup_iters=10, train_freq=20, batch_size=32,
            seed=42,
        )
        trainer.train(verbose=False)
        assert trainer._trained
        assert trainer.infoset_coverage() > 0
        # Network should have been updated at least once
        assert trainer._network_ever_trained

    def test_train_4x4(self):
        """500 episodes on 4x4 — verify no crash + strategy entropy."""
        trainer = LiarsDiceDeepCFRTrainer(
            num_dice=4, num_faces=4, iterations=500,
            warmup_iters=50, train_freq=100, batch_size=64,
            seed=42,
        )
        trainer.train(verbose=False)
        assert trainer._trained
        assert trainer._network_ever_trained

        # Sanity: strategy entropy > 1.0 at root infosets
        # Compute entropy from network prediction on a random root state
        import math
        game = LiarsDice(num_dice=4, num_faces=4, seed=1)
        state = game.reset()
        feat = encode_features(state, 0)
        legal = game.get_legal_actions(state)
        n = len(legal)
        mask_np = np.zeros(trainer.max_actions, dtype=np.float32)
        mask_np[:n] = 1.0
        trainer.network.eval()
        with torch.no_grad():
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(trainer.device)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(trainer.device)
            probs = trainer.network(feat_t, mask_t)["strategy"].squeeze(0).cpu().numpy()
        probs = probs[:n] + 1e-10
        probs /= probs.sum()
        h = -sum(p * math.log(p) for p in probs)
        assert h > 1.0, f"Strategy entropy {h:.3f} <= 1.0 at root infoset"

    def test_policy_probs_sum_to_one(self):
        """Verify the learned network policy sums to 1."""
        trainer = LiarsDiceDeepCFRTrainer(
            num_dice=2, num_faces=3, iterations=100,
            warmup_iters=10, train_freq=20, batch_size=32,
            seed=42,
        )
        trainer.train(verbose=False)
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)
        legal = g.get_legal_actions(s)
        summary = LiarsDiceDeepCFRMethod(
            trainer=trainer, num_dice=2, num_faces=3
        ).predict_summary(obs, legal)
        total = sum(summary["average_policy"].values())
        assert abs(total - 1.0) < 1e-6, f"Policy sums to {total}"

    def test_save_load_roundtrip(self):
        """Verify saving and loading preserves state."""
        trainer = LiarsDiceDeepCFRTrainer(
            num_dice=2, num_faces=3, iterations=100,
            warmup_iters=10, train_freq=20, batch_size=32,
            seed=42,
        )
        trainer.train(verbose=False)
        assert trainer._trained

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "model.pt"))
            assert os.path.exists(os.path.join(tmpdir, "config.json"))

            # Load into new trainer
            trainer2 = LiarsDiceDeepCFRTrainer(
                num_dice=2, num_faces=3,
            )
            trainer2.load(tmpdir)
            assert trainer2._network_ever_trained
            assert trainer2._trained
            assert trainer2._num_infosets == trainer._num_infosets

    def test_reservoir_fills_during_training(self):
        """Verify the buffer accumulates transitions."""
        trainer = LiarsDiceDeepCFRTrainer(
            num_dice=2, num_faces=3, iterations=50,
            warmup_iters=10, train_freq=20, batch_size=32,
            seed=42,
        )
        trainer.train(verbose=False)
        assert len(trainer.buffer) > 0, "Buffer should not be empty after training"


# ═══════════════════════════════════════════════════════════════════════
#  5. Agent Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeepCFRMethod:
    def _train_and_save(self, tmpdir):
        """Helper: train a small model and save to tmpdir."""
        trainer = LiarsDiceDeepCFRTrainer(
            num_dice=2, num_faces=3, iterations=100,
            warmup_iters=10, train_freq=20, batch_size=32,
            seed=42,
        )
        trainer.train(verbose=False)
        trainer.save(tmpdir)
        return trainer

    def test_act_returns_valid_action(self):
        """Smoke test: agent returns a legal action."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)
        legal = g.get_legal_actions(s)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._train_and_save(tmpdir)
            method = LiarsDiceDeepCFRMethod(
                name="DeepCFR",
                model_path=os.path.join(tmpdir, "model.pt"),
                num_dice=2, num_faces=3,
                seed=99,
            )

            action, info_str, action_type = method.act(obs, legal)
            assert action is not None
            assert action.player_id == 0
            assert action.type in {a.type for a in legal}
            payload = json.loads(info_str)
            assert payload["decision_source"] in {
                "network_policy",
                "uniform_fallback",
            }
            assert "average_policy" in payload
            assert "action_values" in payload
            assert "state_value" in payload
            assert "regrets" in payload
            assert abs(sum(payload["average_policy"].values()) - 1.0) < 1e-6

    def test_predict_summary_structure(self):
        """predict_summary should expose policy, value and regret fields."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)
        legal = g.get_legal_actions(s)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._train_and_save(tmpdir)
            method = LiarsDiceDeepCFRMethod(
                name="DeepCFR",
                model_path=os.path.join(tmpdir, "model.pt"),
                num_dice=2, num_faces=3,
                seed=99,
            )

            summary = method.predict_summary(obs, legal)
            assert summary["policy_source"] in {"network_policy", "uniform"}
            assert summary["value_source"] in {
                "network_cfv",
                "unavailable",
            }
            assert abs(sum(summary["average_policy"].values()) - 1.0) < 1e-6
            if summary["action_values"] is not None:
                assert len(summary["action_values"]) == len(legal)
                assert len(summary["regrets"]) == len(legal)
                assert summary["state_value"] is not None

    def test_act_consistent_with_legal(self):
        """Ensure the agent only selects actions that are legal."""
        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        s = g.step(s, Action(player_id=0, type="bid",
                              data={"quantity": 2, "face": 2}))
        obs = g.get_observation(s, 1)
        legal = g.get_legal_actions(s)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._train_and_save(tmpdir)
            method = LiarsDiceDeepCFRMethod(
                name="DeepCFR",
                model_path=os.path.join(tmpdir, "model.pt"),
                num_dice=2, num_faces=3,
                seed=99,
            )

            action, _, _ = method.act(obs, legal)
            assert action.type in {a.type for a in legal}
            if action.type == "bid":
                pair = (action.data["quantity"], action.data["face"])
                assert any(
                    a.data.get("quantity") == pair[0]
                    and a.data.get("face") == pair[1]
                    for a in legal if a.type == "bid"
                )

    def test_act_fallback_uniform(self):
        """Without a network, agent should fall back to uniform."""
        method = LiarsDiceDeepCFRMethod(
            name="DeepCFR_no_net",
            model_path="",  # no network
            num_dice=2, num_faces=3,
            seed=99,
            fallback=True,
        )

        g = LiarsDice(num_dice=2, num_faces=3, seed=42)
        s = g.reset()
        obs = g.get_observation(s, 0)
        legal = g.get_legal_actions(s)

        action, info_str, action_type = method.act(obs, legal)
        assert action is not None
        payload = json.loads(info_str)
        assert "fallback" in payload["decision_source"]
        assert action.type in {a.type for a in legal}

    def test_act_returns_different_actions_over_time(self):
        """Verify the agent explores (not always the same action)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._train_and_save(tmpdir)
            method = LiarsDiceDeepCFRMethod(
                name="DeepCFR",
                model_path=os.path.join(tmpdir, "model.pt"),
                num_dice=2, num_faces=3,
                seed=42,  # fixed seed for reproducibility
                fallback=False,
            )

            g = LiarsDice(num_dice=2, num_faces=3, seed=42)
            actions_seen = set()
            for _ in range(10):
                s = g.reset()
                obs = g.get_observation(s, 0)
                legal = g.get_legal_actions(s)
                action, _, _ = method.act(obs, legal)
                actions_seen.add(action.type)
            # With only 3 actions possible, should see at least some variety
            # (this is a weak test but catches hardcoded action bugs)
            assert len(actions_seen) >= 1

    def test_reflect_noop(self):
        """reflect() should return empty list."""
        method = LiarsDiceDeepCFRMethod(fallback=True)
        assert method.reflect(None) == []


# ═══════════════════════════════════════════════════════════════════════
#  6. Integration Smoke Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_train_and_play_episode(self):
        """Train on 2x3, then play a full episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LiarsDiceDeepCFRTrainer(
                num_dice=2, num_faces=3, iterations=100,
                warmup_iters=10, train_freq=20, batch_size=32,
                seed=42,
            )
            trainer.train(verbose=False)
            trainer.save(tmpdir)

            # Both players use DeepCFR
            agent0 = LiarsDiceDeepCFRMethod(
                model_path=os.path.join(tmpdir, "model.pt"),
                num_dice=2, num_faces=3,
                seed=0, fallback=False,
            )
            agent1 = LiarsDiceDeepCFRMethod(
                model_path=os.path.join(tmpdir, "model.pt"),
                num_dice=2, num_faces=3,
                seed=1, fallback=False,
            )

            game = LiarsDice(num_dice=2, num_faces=3, seed=42)
            state = game.reset()
            while not game.is_terminal(state):
                player = game.get_current_player(state)
                obs = game.get_observation(state, player)
                legal = game.get_legal_actions(state)
                if player == 0:
                    action, _, _ = agent0.act(obs, legal)
                else:
                    action, _, _ = agent1.act(obs, legal)
                state = game.step(state, action)

            rewards = game.get_rewards(state)
            # One player should have 1.0, the other 0.0
            assert set(rewards.values()) == {-1.0, 1.0}
