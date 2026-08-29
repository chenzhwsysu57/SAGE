"""Smoke tests for the Liar's Dice PPO baseline."""

from __future__ import annotations

import tempfile

from games.liars_dice import LiarsDice
from base.types import Action
from methods.liars_dice_ppo import LiarsDicePPOTrainer, LiarsDicePPOMethod


def test_ppo_trainer_smoke():
    trainer = LiarsDicePPOTrainer(
        num_dice=2,
        num_faces=3,
        episodes=8,
        rollout_batch_size=2,
        update_epochs=1,
        minibatch_size=4,
        hidden_size=32,
        seed=42,
    )
    trainer.train(verbose=False)
    assert trainer._trained is True
    assert trainer._train_updates >= 1


def test_ppo_save_load_and_act():
    trainer = LiarsDicePPOTrainer(
        num_dice=2,
        num_faces=3,
        episodes=4,
        rollout_batch_size=2,
        update_epochs=1,
        minibatch_size=4,
        hidden_size=32,
        seed=7,
    )
    trainer.train(verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = f"{tmpdir}/model"
        trainer.save(model_dir)

        method = LiarsDicePPOMethod(model_path=model_dir, num_dice=2, num_faces=3, seed=7)
        game = LiarsDice(num_dice=2, num_faces=3, seed=7)
        state = game.reset()
        obs = game.get_observation(state, 0)
        legal = game.get_legal_actions(state)
        action, info, action_type = method.act(obs, legal)

        assert action.type in {a.type for a in legal}
        assert action_type == action.type
        assert "policy" in info


def test_ppo_act_respects_specific_legal_actions():
    trainer = LiarsDicePPOTrainer(
        num_dice=2,
        num_faces=3,
        episodes=4,
        rollout_batch_size=2,
        update_epochs=1,
        minibatch_size=4,
        hidden_size=32,
        seed=11,
    )
    trainer.train(verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = f"{tmpdir}/model"
        trainer.save(model_dir)
        method = LiarsDicePPOMethod(model_path=model_dir, num_dice=2, num_faces=3, seed=11)

        game = LiarsDice(num_dice=2, num_faces=3, seed=11)
        state = game.reset()
        state = game.step(state, Action(player_id=0, type="bid", data={"quantity": 1, "face": 1}))
        obs = game.get_observation(state, 1)
        legal = game.get_legal_actions(state)
        action, _, _ = method.act(obs, legal)

        assert any(
            action.type == a.type
            and action.data.get("quantity") == a.data.get("quantity")
            and action.data.get("face") == a.data.get("face")
            for a in legal
        )
