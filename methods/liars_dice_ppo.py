"""PPO policy/value learner for Liar's Dice.

This is a minimal discrete-action PPO baseline that reuses the existing
LiarsDice observation encoding. It trains a shared policy by self-play and
can be loaded as a non-LLM method at inference time.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.interfaces import BaseModel
from base.types import Action, Observation
from games.liars_dice import LiarsDice
from methods.liars_dice_deep_cfr import encode_features_from_obs
from runs.artifacts import git_commit


def _action_index(num_faces: int, action: Action) -> int:
    if action.type == "challenge":
        return 0
    q = int(action.data.get("quantity", 0))
    f = int(action.data.get("face", 0))
    return 1 + (q - 1) * num_faces + (f - 1)


def _index_action(num_faces: int, idx: int, player_id: int) -> Action:
    if idx == 0:
        return Action(player_id=player_id, type="challenge")
    idx -= 1
    q = idx // num_faces + 1
    f = idx % num_faces + 1
    return Action(
        player_id=player_id,
        type="bid",
        data={"quantity": q, "face": f},
    )


def _sample_legal_action(
    probs: np.ndarray,
    legal_actions: list[Action],
    num_faces: int,
    rng: random.Random,
) -> tuple[Action, int, float]:
    """Sample directly from the legal action list.

    This avoids returning an action that is only valid in the full action
    space but not in the current state-specific legal set.
    """
    if not legal_actions:
        raise ValueError("No legal actions available.")

    legal_probs: list[float] = []
    legal_indices: list[int] = []
    for i, action in enumerate(legal_actions):
        idx = _action_index(num_faces, action)
        legal_indices.append(idx)
        p = float(probs[idx]) if 0 <= idx < len(probs) else 0.0
        legal_probs.append(max(p, 0.0))

    total = sum(legal_probs)
    if total <= 0.0:
        choice = rng.randint(0, len(legal_actions) - 1)
        action = legal_actions[choice]
        return action, legal_indices[choice], 1.0 / len(legal_actions)

    norm_probs = [p / total for p in legal_probs]
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(norm_probs):
        cum += p
        if r <= cum:
            return legal_actions[i], legal_indices[i], p
    last = len(legal_actions) - 1
    return legal_actions[last], legal_indices[last], norm_probs[last]


def _legal_mask(num_dice: int, num_faces: int, legal_actions: list[Action]) -> np.ndarray:
    mask = np.zeros(1 + num_dice * 2 * num_faces, dtype=np.float32)
    for action in legal_actions:
        idx = _action_index(num_faces, action)
        if 0 <= idx < mask.shape[0]:
            mask[idx] = 1.0
    return mask


class PPOPolicyNet(nn.Module):
    def __init__(self, feature_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        value = self.value_head(h).squeeze(-1)
        return logits, value


@dataclass
class RolloutStep:
    features: np.ndarray
    mask: np.ndarray
    action_idx: int
    logprob: float
    value: float
    player_id: int


class LiarsDicePPOTrainer:
    """Minimal discrete-action PPO trainer for Liar's Dice."""

    def __init__(
        self,
        num_dice: int = 4,
        num_faces: int = 4,
        episodes: int = 2_000,
        rollout_batch_size: int = 32,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        clip_range: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        hidden_size: int = 128,
        seed: int = 0,
        device: torch.device | None = None,
    ):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.episodes = episodes
        self.rollout_batch_size = rollout_batch_size
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.hidden_size = hidden_size
        self.max_total_dice = 2 * num_dice
        self.max_actions = 1 + self.max_total_dice * num_faces
        self.feature_dim = num_faces + 7

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rng = random.Random(seed)
        self.game = LiarsDice(num_dice=num_dice, num_faces=num_faces, seed=seed)
        self.policy = PPOPolicyNet(
            feature_dim=self.feature_dim,
            n_actions=self.max_actions,
            hidden_size=hidden_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self._trained = False
        self._train_updates = 0
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._action_histogram: dict[str, int] = {}

    def _features(self, observation: Observation) -> np.ndarray:
        return encode_features_from_obs(
            observation,
            num_faces=self.num_faces,
            max_dice_count=self.num_dice,
            max_total_dice=self.max_total_dice,
        )

    def _mask(self, legal_actions: list[Action]) -> np.ndarray:
        return _legal_mask(self.num_dice, self.num_faces, legal_actions)

    def _sample_action(
        self, observation: Observation, legal_actions: list[Action]
    ) -> tuple[Action, float, float, int, np.ndarray, np.ndarray]:
        feat = self._features(observation)
        mask_np = self._mask(legal_actions)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(feat_t, mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(dist.sample().item())
            logprob = float(
                dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
            )
            value_f = float(value.squeeze(0).item())
        action = _index_action(self.num_faces, action_idx, observation.player_id)
        return action, logprob, value_f, action_idx, feat, mask_np

    def _run_episode(self) -> tuple[list[RolloutStep], dict[int, float]]:
        state = self.game.reset()
        steps: list[RolloutStep] = []
        while not self.game.is_terminal(state):
            player = self.game.get_current_player(state)
            obs = self.game.get_observation(state, player)
            legal = self.game.get_legal_actions(state)
            action, logprob, value, action_idx, feat, mask_np = self._sample_action(
                obs, legal
            )
            steps.append(
                RolloutStep(
                    features=feat,
                    mask=mask_np,
                    action_idx=action_idx,
                    logprob=logprob,
                    value=value,
                    player_id=player,
                )
            )
            state = self.game.step(state, action)
        returns = self.game.get_rewards(state)
        return steps, returns

    def _action_hist(self, steps: list[RolloutStep]) -> dict[str, int]:
        hist: dict[str, int] = {}
        for step in steps:
            key = str(step.action_idx)
            hist[key] = hist.get(key, 0) + 1
        return hist

    def train(self, verbose: bool = True) -> None:
        if verbose:
            print(
                f"Training PPO ({self.episodes} episodes, "
                f"{self.num_dice}d×{self.num_faces}f) on {self.device}..."
            )

        rollout_buffer: list[tuple[list[RolloutStep], dict[int, float]]] = []
        log_interval = max(1, self.episodes // 10)

        for episode in range(1, self.episodes + 1):
            steps, returns = self._run_episode()
            rollout_buffer.append((steps, returns))
            self._episode_lengths.append(len(steps))
            self._episode_returns.append(sum(returns.values()) / max(1, len(returns)))
            self._action_histogram.update(self._action_hist(steps))

            if len(rollout_buffer) >= self.rollout_batch_size:
                self._update(rollout_buffer)
                rollout_buffer.clear()

            if verbose and episode % log_interval == 0:
                avg_ret = sum(self._episode_returns[-log_interval:]) / min(
                    log_interval, len(self._episode_returns)
                )
                avg_len = sum(self._episode_lengths[-log_interval:]) / min(
                    log_interval, len(self._episode_lengths)
                )
                print(
                    f"  Episode {episode}/{self.episodes}: "
                    f"avg_return={avg_ret:+.2f} avg_len={avg_len:.1f} "
                    f"updates={self._train_updates}"
                )

        if rollout_buffer:
            self._update(rollout_buffer)

        self._trained = True
        if verbose:
            print(f"PPO training complete. Updates: {self._train_updates}")

    def _update(self, rollout_buffer: list[tuple[list[RolloutStep], dict[int, float]]]) -> None:
        steps: list[RolloutStep] = []
        returns: list[float] = []
        old_logprobs: list[float] = []
        values: list[float] = []

        for episode_steps, payout in rollout_buffer:
            for step in episode_steps:
                steps.append(step)
                returns.append(float(payout.get(step.player_id, 0.0)))
                old_logprobs.append(step.logprob)
                values.append(step.value)

        if not steps:
            return

        feat_np = np.stack([s.features for s in steps]).astype(np.float32)
        mask_np = np.stack([s.mask for s in steps]).astype(np.float32)
        action_idx_np = np.array([s.action_idx for s in steps], dtype=np.int64)
        return_np = np.array(returns, dtype=np.float32)
        old_logprob_np = np.array(old_logprobs, dtype=np.float32)
        value_np = np.array(values, dtype=np.float32)

        adv_np = return_np - value_np
        if len(adv_np) > 1:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        feat_t = torch.from_numpy(feat_np).to(self.device)
        mask_t = torch.from_numpy(mask_np).to(self.device)
        action_t = torch.from_numpy(action_idx_np).to(self.device)
        return_t = torch.from_numpy(return_np).to(self.device)
        old_logprob_t = torch.from_numpy(old_logprob_np).to(self.device)
        adv_t = torch.from_numpy(adv_np.astype(np.float32)).to(self.device)

        n = feat_t.shape[0]
        idxs = list(range(n))
        for _ in range(self.update_epochs):
            self.rng.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                batch_idx = idxs[start:start + self.minibatch_size]
                if not batch_idx:
                    continue
                b_feat = feat_t[batch_idx]
                b_mask = mask_t[batch_idx]
                b_action = action_t[batch_idx]
                b_return = return_t[batch_idx]
                b_old_logprob = old_logprob_t[batch_idx]
                b_adv = adv_t[batch_idx]

                logits, values = self.policy(b_feat, b_mask)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprob = dist.log_prob(b_action)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprob - b_old_logprob)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = -torch.min(ratio * b_adv, clipped_ratio * b_adv).mean()
                value_loss = F.mse_loss(values, b_return)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

        self._train_updates += 1

    def save(self, path_dir: str, metadata: dict[str, Any] | None = None) -> None:
        os.makedirs(path_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path_dir, "model.pt"))
        config = {
            "num_dice": self.num_dice,
            "num_faces": self.num_faces,
            "episodes": self.episodes,
            "rollout_batch_size": self.rollout_batch_size,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "clip_range": self.clip_range,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "hidden_size": self.hidden_size,
            "feature_dim": self.feature_dim,
            "max_actions": self.max_actions,
            "device": str(self.device),
            "train_updates": self._train_updates,
            "git_commit": git_commit(),
        }
        if metadata:
            config["metadata"] = metadata
        with open(os.path.join(path_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        summary = {
            "trained": self._trained,
            "episodes": self.episodes,
            "train_updates": self._train_updates,
            "avg_return": float(np.mean(self._episode_returns)) if self._episode_returns else 0.0,
            "avg_length": float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0,
            "action_histogram": self._action_histogram,
        }
        with open(os.path.join(path_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def load(self, path_dir: str) -> None:
        model_path = path_dir
        if os.path.isdir(path_dir):
            model_path = os.path.join(path_dir, "model.pt")
            config_path = os.path.join(path_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                self.num_dice = int(cfg.get("num_dice", self.num_dice))
                self.num_faces = int(cfg.get("num_faces", self.num_faces))
                self.episodes = int(cfg.get("episodes", self.episodes))
                self.rollout_batch_size = int(cfg.get("rollout_batch_size", self.rollout_batch_size))
                self.update_epochs = int(cfg.get("update_epochs", self.update_epochs))
                self.minibatch_size = int(cfg.get("minibatch_size", self.minibatch_size))
                self.clip_range = float(cfg.get("clip_range", self.clip_range))
                self.gamma = float(cfg.get("gamma", self.gamma))
                self.gae_lambda = float(cfg.get("gae_lambda", self.gae_lambda))
                self.lr = float(cfg.get("lr", self.lr))
                self.entropy_coef = float(cfg.get("entropy_coef", self.entropy_coef))
                self.value_coef = float(cfg.get("value_coef", self.value_coef))
                self.hidden_size = int(cfg.get("hidden_size", self.hidden_size))
                self.feature_dim = int(cfg.get("feature_dim", self.feature_dim))
                self.max_actions = int(cfg.get("max_actions", self.max_actions))

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(state)
            self._trained = True


class LiarsDicePPOMethod:
    """Load a trained PPO policy and pick legal actions."""

    _DEFAULT_DIR = "results/liars_dice/ppo"

    def __init__(
        self,
        name: str = "PPO",
        model_path: str = "",
        num_dice: int = 4,
        num_faces: int = 4,
        seed: int = 0,
    ):
        self.name = name
        self.method_type = "ppo"
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.max_total_dice = 2 * num_dice
        self.max_actions = 1 + self.max_total_dice * num_faces
        self.feature_dim = num_faces + 7
        self._rng = random.Random(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = model_path or self._DEFAULT_DIR
        config = None
        if os.path.isdir(path):
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
            model_file = os.path.join(path, "model.pt")
        else:
            model_file = path

        hidden_size = int((config or {}).get("hidden_size", 128))
        self.policy = PPOPolicyNet(
            self.feature_dim, self.max_actions, hidden_size=hidden_size
        ).to(self.device)

        if os.path.exists(model_file):
            state = torch.load(model_file, map_location=self.device)
            self.policy.load_state_dict(state)
            self.policy.eval()

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel | None = None,
    ) -> tuple[Action, str, str]:
        feat = encode_features_from_obs(
            observation,
            num_faces=self.num_faces,
            max_dice_count=self.num_dice,
            max_total_dice=self.max_total_dice,
        )
        mask_np = _legal_mask(self.num_dice, self.num_faces, legal_actions)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.policy(feat_t, mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        action, idx, prob = _sample_legal_action(
            probs=probs,
            legal_actions=legal_actions,
            num_faces=self.num_faces,
            rng=self._rng,
        )
        info = {
            "policy": "ppo",
            "value": float(value.squeeze(0).item()),
            "action_prob": float(prob),
            "action_idx": idx,
        }
        return action, json.dumps(info, ensure_ascii=False), action.type

    def reflect(self, trajectory, model=None, game_name=""):
        return []
