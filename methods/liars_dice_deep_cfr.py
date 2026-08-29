"""Deep CFR (Deep Counterfactual Regret Minimization) for Liar's Dice (MxN).

Implements Deep CFR (Brown et al. 2019) with outcome-sampling for
tractable tree walk and bootstrapped regret updates.

Architecture:
  - Neural network with strategy head (masked softmax) and value head.
  - Reservoir buffer (2M capacity) for experience replay.
  - Outcome-sampling MCCFR: 1 action sampled per node (both traverser
    and opponent), giving a single path per iteration for efficiency.
  - Local regret targets are bootstrapped from the network on the fly.
  - No infoset-level strategy table is persisted; the public interface
    is the neural policy/value summary used by ToM.

The pipeline:
  1. Phase A — Data Collection: outcome-sampling tree walk.
     Traverser and opponent both sample from the current network
     policy when available, otherwise from a uniform fallback.
  2. Phase B — Network Update: sample minibatch from reservoir, train
     strategy head (cross-entropy) and value head (MSE) simultaneously.

Usage:
    trainer = LiarsDiceDeepCFRTrainer(iterations=50000)
    trainer.train()
    trainer.save("results/liars_dice/deep_cfr/")

    agent = LiarsDiceDeepCFRMethod(model_path="results/liars_dice/deep_cfr/model.pt")
    action, info, action_type = agent.act(observation, legal_actions)
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional in dev envs
    tqdm = None

from games.liars_dice import LiarsDice
from methods.liars_dice_cfr import (
    encode_infoset,
    encode_action,
    regret_matching_strategy,
    sample_index,
    opponent_reach,
)
from base.interfaces import BaseModel
from base.types import Action, Observation


# ═══════════════════════════════════════════════════════════════════════
#  1. Feature Encoding
# ═══════════════════════════════════════════════════════════════════════

def encode_features(
    state,
    player: int,
    num_faces: int = 4,
    max_dice_count: int = 4,
    max_total_dice: int = 8,
    max_rounds: int = 100,
) -> np.ndarray:
    """Convert Liar's Dice infoset into a fixed-size feature vector.

    Feature layout (length = num_faces + 7):
      [0..num_faces-1]   Dice histogram: count of each face value
      [num_faces+0]      my_dice_count / max_dice_count
      [num_faces+1]      opp_dice_count / max_dice_count
      [num_faces+2]      bid_quantity / max_total_dice  (0 if no bid)
      [num_faces+3]      bid_face / num_faces           (0 if no bid)
      [num_faces+4..+5]  Player indicator: [1,0] for P0, [0,1] for P1
      [num_faces+6]      round_num / max_rounds

    For 4×4 the feature vector is 11-dimensional; for M×N it scales to
    num_faces + 7.
    """
    # -- Dice histogram -------------------------------------------------
    my_dice = state.dice[player]
    hist = np.zeros(num_faces, dtype=np.float32)
    for d in my_dice:
        if 1 <= d <= num_faces:
            hist[d - 1] += 1.0

    # -- Normalized dice counts -----------------------------------------
    my_count = state.dice_counts[player] / max_dice_count
    opp_count = state.dice_counts[1 - player] / max_dice_count

    # -- Current bid encoding -------------------------------------------
    if state.last_bid is not None:
        bid_q, bid_f_val = state.last_bid
        bid_q_enc = bid_q / max_total_dice
        bid_f_enc = bid_f_val / num_faces
    else:
        bid_q_enc = 0.0
        bid_f_enc = 0.0

    # -- Player indicator -----------------------------------------------
    if player == 0:
        p_ind = np.array([1.0, 0.0], dtype=np.float32)
    else:
        p_ind = np.array([0.0, 1.0], dtype=np.float32)

    # -- Round fraction -------------------------------------------------
    round_frac = state.round_num / max_rounds

    return np.concatenate([
        hist,
        np.array([my_count, opp_count, bid_q_enc, bid_f_enc], dtype=np.float32),
        p_ind,
        np.array([round_frac], dtype=np.float32),
    ])


def encode_features_from_obs(
    observation: Observation,
    num_faces: int = 4,
    max_dice_count: int = 4,
    max_total_dice: int = 8,
    max_rounds: int = 100,
) -> np.ndarray:
    """Encode an Observation into the same fixed-size feature vector.

    Mirror of ``encode_features`` but accepts an Observation (as seen by
    the agent at inference time) instead of a raw game state.
    """
    my_dice = observation.private_info.get("my_dice", [])
    player = observation.player_id

    hist = np.zeros(num_faces, dtype=np.float32)
    for d in my_dice:
        if 1 <= d <= num_faces:
            hist[d - 1] += 1.0

    my_count = observation.public_info.get("my_dice_count", max_dice_count) / max_dice_count
    opp_count = observation.public_info.get("opponent_dice_count", max_dice_count) / max_dice_count

    last_bid = observation.public_info.get("last_bid")
    if last_bid is not None:
        bid_q_enc = last_bid["quantity"] / max_total_dice
        bid_f_enc = last_bid["face"] / num_faces
    else:
        bid_q_enc = 0.0
        bid_f_enc = 0.0

    if player == 0:
        p_ind = np.array([1.0, 0.0], dtype=np.float32)
    else:
        p_ind = np.array([0.0, 1.0], dtype=np.float32)

    round_frac = observation.public_info.get("round", 1) / max_rounds

    return np.concatenate([
        hist,
        np.array([my_count, opp_count, bid_q_enc, bid_f_enc], dtype=np.float32),
        p_ind,
        np.array([round_frac], dtype=np.float32),
    ])


# ═══════════════════════════════════════════════════════════════════════
#  2. Neural Network
# ═══════════════════════════════════════════════════════════════════════

class DeepCFRNetwork(nn.Module):
    """PyTorch network with strategy and value heads.

    Architecture:
      Input (feature_dim) → Linear(128) → LayerNorm → ReLU
                           → Linear(128) → LayerNorm → ReLU
                           → Linear(128) → LayerNorm → ReLU
                           → strategy_head (Linear → masked softmax)
                           → value_head   (Linear → linear output)

    The strategy head outputs a probability vector over ``n_actions``
    actions (masked softmax: unavailable action logits are set to -inf).
    The value head outputs raw counterfactual values for each action.
    """

    def __init__(self, feature_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(feature_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.strategy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, n_actions)

    def forward(self, x: torch.Tensor,
                action_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, feature_dim) input features.
            action_mask: (batch, n_actions) — 1 for available, 0 for unavailable.

        Returns:
            dict with keys:
              "strategy": (batch, n_actions) masked-softmax probabilities.
              "cfv":      (batch, n_actions) raw counterfactual values.
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        logits = self.strategy_head(x)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        strategy = F.softmax(logits, dim=-1)

        cfv = self.value_head(x)

        return {"strategy": strategy, "cfv": cfv}


# ═══════════════════════════════════════════════════════════════════════
#  3. Reservoir Buffer
# ═══════════════════════════════════════════════════════════════════════

class ReservoirBuffer:
    """Uniform reservoir sampling buffer for experience replay.

    Each transition is ``(features, action_mask, strategy_target,
    cfv_target, cfv_mask, sample_weight)`` as numpy arrays.  When the
    buffer reaches capacity, new items replace old ones with uniform
    probability (reservoir sampling).
    """

    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self._buffer: list[tuple[np.ndarray, ...]] = []
        self._n_seen = 0

    def add(self, transition: tuple[np.ndarray, ...]) -> None:
        """Insert a transition via reservoir sampling."""
        self._n_seen += 1
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            idx = random.randint(0, self._n_seen - 1)
            if idx < self.capacity:
                self._buffer[idx] = transition

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, ...]]:
        """Return a random minibatch (without replacement)."""
        k = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, k)

    def __len__(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════
#  4. Deep CFR Decision Summary
# ═══════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class DeepCFRDecisionSummary:
    """Structured Deep CFR output for downstream ToM consumption."""

    model: str
    infoset: str
    decision_source: str
    policy_source: str
    value_source: str
    action_keys: list[str]
    average_policy: dict[str, float]
    action_values: dict[str, float] | None
    regrets: dict[str, float] | None
    state_value: float | None
    entropy: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
#  4. Deep CFR Trainer
# ═══════════════════════════════════════════════════════════════════════

class LiarsDiceDeepCFRTrainer:
    """Deep CFR trainer for Liar's Dice.

    Training alternates between:
      Phase A — Outcome-sampling tree walk with neural policy/value
                targets and uniform fallback before warmup.
      Phase B — Network update via reservoir sampling minibatches.

    Args:
        num_dice: Initial dice per player.
        num_faces: Face values 1..num_faces.
        iterations: Total MCCFR iterations (tree walks).
        warmup_iters: Network-free iterations at start.
        train_freq: Update network every *train_freq* iterations.
        batch_size: Minibatch size for network updates.
        buffer_capacity: Reservoir buffer capacity.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        hidden_size: Network hidden layer size.
        seed: RNG seed.
        device: Torch device (auto-detect MPS / CUDA / CPU if None).
    """

    def __init__(
        self,
        num_dice: int = 4,
        num_faces: int = 4,
        iterations: int = 50_000,
        warmup_iters: int = 100,
        train_freq: int = 100,
        batch_size: int = 512,
        buffer_capacity: int = 2_000_000,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        hidden_size: int = 128,
        seed: int = 0,
        device: torch.device | None = None,
    ):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.iterations = iterations
        self.warmup_iters = warmup_iters
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # Derived constants
        self.max_total_dice = 2 * num_dice
        self.max_actions = self.max_total_dice * num_faces + 1  # safe upper bound
        self.feature_dim = num_faces + 7  # hist + 2counts + 2bid + 2player + 1round

        # Device
        # NOTE: CPU is intentionally the default even when MPS/CUDA is
        # available.  This network is tiny (11 → 128×3 → n_actions, ~35k
        # params).  GPU transfer + synchronization overhead dwarfs the
        # actual computation for single-sample forward passes during the
        # tree walk (Phase A).  The batch training in Phase B is also
        # fast enough on CPU at this scale.
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # RNG
        self.rng = random.Random(seed)
        self.game = LiarsDice(num_dice=num_dice, num_faces=num_faces, seed=seed)

        # Neural network
        self.network = DeepCFRNetwork(
            feature_dim=self.feature_dim,
            n_actions=self.max_actions,
            hidden_size=hidden_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Reservoir buffer
        self.buffer = ReservoirBuffer(capacity=buffer_capacity)

        # State
        self._train_step_count = 0
        self._network_ever_trained = False
        self._trained = False
        self._seen_infosets: set[str] = set()
        self._num_infosets = 0

    @staticmethod
    def _normalize_probs(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.size == 0:
            return probs
        probs = np.maximum(probs, 0.0)
        total = float(probs.sum())
        if total <= 0.0:
            return np.full_like(probs, 1.0 / len(probs), dtype=np.float64)
        return probs / total

    # ── Training loop ────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> None:
        """Run the full neural Deep CFR training loop."""
        if verbose:
            print(f"Training Deep CFR ({self.iterations} iterations, "
                  f"{self.num_dice}d×{self.num_faces}f) "
                  f"on {self.device}...")

        log_interval = max(1, self.iterations // 20)

        iterator = range(self.iterations)
        if verbose and tqdm is not None:
            iterator = tqdm(
                iterator,
                total=self.iterations,
                desc=f"Deep CFR {self.num_dice}x{self.num_faces}",
                dynamic_ncols=True,
                mininterval=0.5,
            )

        for i in iterator:
            traverser = i % 2
            state = self.game.reset()
            self._walk(state, 1.0, 1.0, traverser)

            # Network update
            if i >= self.warmup_iters and (i + 1) % self.train_freq == 0:
                self._update_network()

            if verbose and (i + 1) % log_interval == 0:
                buf_sz = len(self.buffer)
                if tqdm is not None and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(buffer=buf_sz, updates=self._train_step_count)
                else:
                    print(f"  Iteration {i + 1}/{self.iterations}, "
                          f"buffer={buf_sz}")

        self._num_infosets = len(self._seen_infosets)
        self._trained = True
        if verbose:
            print(f"Deep CFR trained. {self._num_infosets} seen infosets, "
                  f"{self._train_step_count} network updates.")

    # ── Tree walk ────────────────────────────────────────────────────

    def _walk(
        self, state, p0_reach: float, p1_reach: float, traverser: int
    ) -> float:
        """Outcome-sampling tree walk with neural targets.

        All nodes sample a single action from the current network
        policy when available, otherwise from a uniform fallback.  At
        traverser nodes we bootstrap per-action values and derive a
        local regret-matching target on the fly.  No infoset-level
        strategy table is persisted.
        """
        if state.phase == "terminal":
            return state.rewards[traverser]

        player = state.current_player
        legal = self.game.get_legal_actions(state)
        infoset = encode_infoset(state, player)
        n = len(legal)
        self._seen_infosets.add(infoset)

        # -- Features & action mask ------------------------------------
        feat = encode_features(
            state, player,
            num_faces=self.num_faces,
            max_dice_count=self.num_dice,
            max_total_dice=self.max_total_dice,
        )
        mask_np = np.zeros(self.max_actions, dtype=np.float32)
        mask_np[:n] = 1.0

        # Network policy/value predictions for sampling and bootstrap.
        policy = np.full(n, 1.0 / n, dtype=np.float32)
        cfv_pred = np.zeros(n, dtype=np.float32)
        if self._network_ever_trained:
            with torch.no_grad():
                feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
                output = self.network(feat_t, mask_t)
                policy = output["strategy"].squeeze(0).cpu().numpy()[:n]
                cfv_pred = output["cfv"].squeeze(0).cpu().numpy()[:n]
            policy = self._normalize_probs(policy)

        # -- Sample ONE action (outcome-sampling) ----------------------
        idx = sample_index(policy, self.rng)

        action = legal[idx]
        next_state = self.game.step(state, action)

        # -- Recursive call --------------------------------------------
        cfv = self._walk(
            next_state, p0_reach, p1_reach, traverser
        )

        # -- Update (traverser only) -----------------------------------
        if player == traverser:
            # --- Bootstrapped neural targets ---
            combined_cfvs = cfv_pred.copy()
            combined_cfvs[idx] = cfv

            expected_val = float(np.dot(policy, combined_cfvs))
            advantages = combined_cfvs - expected_val
            policy_target = regret_matching_strategy(advantages.tolist(), n)

            strat_target = np.zeros(self.max_actions, dtype=np.float32)
            strat_target[:n] = np.asarray(policy_target, dtype=np.float32)
            cfv_target = np.zeros(self.max_actions, dtype=np.float32)
            cfv_target[:n] = combined_cfvs
            cfv_mask_np = np.zeros(self.max_actions, dtype=np.float32)
            cfv_mask_np[:n] = 1.0
            sample_weight = np.array(
                [opponent_reach(p0_reach, p1_reach, player)], dtype=np.float32
            )

            self.buffer.add(
                (feat, mask_np, strat_target, cfv_target, cfv_mask_np, sample_weight)
            )

        return cfv

    # ── Network update ───────────────────────────────────────────────

    def _update_network(self) -> None:
        """Sample a minibatch from the reservoir and update the network."""
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)

        # Unpack: (features, action_mask, strategy_target, cfv_target, cfv_mask, weight)
        features = np.stack([b[0] for b in batch])
        action_masks = np.stack([b[1] for b in batch])
        strategy_targets = np.stack([b[2] for b in batch])

        # CFV targets may be stored as 4th or 5th element depending on
        # whether we have the cfv_mask_np field (backward compat)
        cfv_targets = np.stack([b[3] for b in batch])
        if len(batch[0]) >= 5:
            cfv_masks = np.stack([b[4] for b in batch])
        else:
            # Fallback for old format: cfv has non-zero at sampled action
            cfv_masks = (cfv_targets != 0.0).astype(np.float32)
        if len(batch[0]) >= 6:
            sample_weights = np.stack([b[5] for b in batch]).reshape(-1)
        else:
            sample_weights = np.ones((len(batch),), dtype=np.float32)

        # Convert to torch tensors
        feat_t = torch.from_numpy(features).to(self.device)
        mask_t = torch.from_numpy(action_masks).to(self.device)
        strat_t = torch.from_numpy(strategy_targets).to(self.device)
        cfv_t = torch.from_numpy(cfv_targets).to(self.device)
        cfv_mask_t = torch.from_numpy(cfv_masks).to(self.device)
        weight_t = torch.from_numpy(sample_weights).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()

        output = self.network(feat_t, mask_t)
        pred_strat = output["strategy"]  # (B, max_actions) softmax over available
        pred_cfv = output["cfv"]         # (B, max_actions) raw values

        # Strategy loss: masked cross-entropy over ALL available actions
        log_pred = torch.log(pred_strat + 1e-10)
        strategy_loss = -(
            (strat_t * log_pred * mask_t).sum(dim=-1) * weight_t
        ).mean()

        # CFV loss: MSE over actions that have CFV targets (sampled actions)
        cfv_valid = cfv_mask_t * mask_t
        n_cfv = cfv_valid.sum(dim=-1).clamp(min=1)
        cfv_loss = (((pred_cfv - cfv_t) ** 2) * cfv_valid).sum(dim=-1) / n_cfv
        cfv_loss = (cfv_loss * weight_t).mean()

        loss = strategy_loss + cfv_loss
        loss.backward()
        self.optimizer.step()

        self.network.eval()
        self._train_step_count += 1
        self._network_ever_trained = True

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path_dir: str, metadata: dict | None = None) -> None:
        """Save network weights and config."""
        os.makedirs(path_dir, exist_ok=True)

        # Network weights
        torch.save(self.network.state_dict(), os.path.join(path_dir, "model.pt"))

        # Config
        config = {
            "num_dice": self.num_dice,
            "num_faces": self.num_faces,
            "iterations": self.iterations,
            "warmup_iters": self.warmup_iters,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer.capacity,
            "hidden_size": self.hidden_size,
            "feature_dim": self.feature_dim,
            "max_actions": self.max_actions,
            "device": str(self.device),
            "network_updates": self._train_step_count,
            "seen_infosets": self._num_infosets,
            "output_schema": {
                "policy": "average_policy",
                "values": "action_values",
                "state_value": "state_value",
                "regret": "regrets",
            },
        }
        if metadata:
            config["metadata"] = metadata
        with open(os.path.join(path_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path_dir: str) -> None:
        """Load network weights from disk."""
        # Load network
        model_path = os.path.join(path_dir, "model.pt")
        if os.path.exists(model_path):
            self.network.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.network.eval()
            self._network_ever_trained = True
        config_path = os.path.join(path_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    data = json.load(f)
                self._num_infosets = int(data.get("seen_infosets", 0))
            except Exception:
                pass
        self._trained = True

    def infoset_coverage(self) -> int:
        return self._num_infosets


# ═══════════════════════════════════════════════════════════════════════
#  5. Deep CFR Agent
# ═══════════════════════════════════════════════════════════════════════

class LiarsDiceDeepCFRMethod:
    """Deep CFR agent for Liar's Dice.

    Uses a trained DeepCFRNetwork to predict action strategies and
    action-level values at each decision point.  Falls back to uniform
    random if no network is loaded.

    Args:
        name: Display name.
        model_path: Path to saved ``model.pt`` (network weights).
        trainer: Optional trained trainer instance (alternative to model_path).
        num_dice: Initial dice per player (must match training).
        num_faces: Face values (must match training).
        seed: RNG seed.
        fallback: If True, fall back to uniform random when network is
                  unavailable (instead of raising).
    """

    _DEFAULT_DIR = "results/liars_dice/deep_cfr"

    def __init__(
        self,
        name: str = "DeepCFR",
        model_path: str = "",
        trainer: LiarsDiceDeepCFRTrainer | None = None,
        num_dice: int = 4,
        num_faces: int = 4,
        seed: int = 0,
        fallback: bool = True,
    ):
        self.name = name
        self.method_type = "deep_cfr"
        self._rng = random.Random(seed)
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.max_total_dice = 2 * num_dice
        self.max_actions = self.max_total_dice * num_faces + 1
        self.feature_dim = num_faces + 7
        self._fallback = fallback

        # CPU only — see LiarsDiceDeepCFRTraininer docstring for rationale.
        self.device = torch.device("cpu")

        # Load network
        self.network: DeepCFRNetwork | None = None
        if trainer is not None:
            # Use from an in-memory trainer
            self.network = trainer.network
            self.network.eval()
        elif model_path:
            if os.path.isfile(model_path):
                self._load_network(model_path)
            elif os.path.isdir(model_path):
                self._load_network(os.path.join(model_path, "model.pt"))
            elif os.path.isdir(os.path.dirname(model_path)):
                self._load_network(os.path.join(model_path, "model.pt"))
        else:
            # Try default directory
            default_pt = os.path.join(self._DEFAULT_DIR, "model.pt")
            if os.path.exists(default_pt):
                self._load_network(default_pt)

        if self.network is not None:
            self.network.eval()

    @staticmethod
    def _normalize_probs(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.size == 0:
            return probs
        probs = np.maximum(probs, 0.0)
        total = float(probs.sum())
        if total <= 0.0:
            return np.full_like(probs, 1.0 / len(probs), dtype=np.float64)
        return probs / total

    def _network_policy_and_values(
        self,
        observation: Observation,
        legal_actions: list[Action],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return network policy/value predictions for legal actions."""
        if self.network is None:
            return None, None

        n = len(legal_actions)
        if n == 0:
            return None, None

        feat = encode_features_from_obs(
            observation,
            num_faces=self.num_faces,
            max_dice_count=self.num_dice,
            max_total_dice=self.max_total_dice,
        )
        mask_np = np.zeros(self.max_actions, dtype=np.float32)
        mask_np[:n] = 1.0

        with torch.no_grad():
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
            self.network.eval()
            output = self.network(feat_t, mask_t)
            probs = output["strategy"].squeeze(0).cpu().numpy()[:n]
            cfv = output["cfv"].squeeze(0).cpu().numpy()[:n]

        return probs, cfv

    def predict_summary(
        self,
        observation: Observation,
        legal_actions: list[Action],
    ) -> dict[str, Any]:
        """Return structured policy/value/regret summary for ToM."""
        n = len(legal_actions)
        if n == 0:
            raise ValueError("No legal actions available.")

        action_keys = [encode_action(a) for a in legal_actions]
        net_probs: np.ndarray | None = None
        net_cfv: np.ndarray | None = None
        if self.network is not None:
            net_probs, net_cfv = self._network_policy_and_values(
                observation, legal_actions
            )

        policy_source = "uniform"
        decision_source = "uniform_fallback"
        avg_policy_vec: np.ndarray
        if net_probs is not None:
            avg_policy_vec = self._normalize_probs(net_probs)
            policy_source = "network_policy"
            decision_source = "network_policy"
        else:
            avg_policy_vec = np.full(n, 1.0 / n, dtype=np.float64)

        action_values_vec: np.ndarray | None = None
        value_source = "unavailable"
        if net_cfv is not None:
            action_values_vec = np.asarray(net_cfv, dtype=np.float64)
            value_source = "network_cfv"

        state_value: float | None = None
        regrets_vec: np.ndarray | None = None
        entropy: float | None = None
        if action_values_vec is not None:
            state_value = float(np.dot(avg_policy_vec, action_values_vec))
            regrets_vec = action_values_vec - state_value

        if avg_policy_vec.size:
            safe = np.clip(avg_policy_vec, 1e-12, 1.0)
            entropy = float(-np.sum(safe * np.log(safe)))

        summary = DeepCFRDecisionSummary(
            model=self.name,
            infoset=self._build_infoset_key(observation),
            decision_source=decision_source,
            policy_source=policy_source,
            value_source=value_source,
            action_keys=action_keys,
            average_policy={
                ak: float(p) for ak, p in zip(action_keys, avg_policy_vec)
            },
            action_values=(
                {ak: float(v) for ak, v in zip(action_keys, action_values_vec)}
                if action_values_vec is not None
                else None
            ),
            regrets=(
                {ak: float(v) for ak, v in zip(action_keys, regrets_vec)}
                if regrets_vec is not None
                else None
            ),
            state_value=state_value,
            entropy=entropy,
        )
        return summary.to_dict()

    def _load_network(self, model_path: str) -> None:
        """Load network weights from a .pt file."""
        net = DeepCFRNetwork(
            feature_dim=self.feature_dim,
            n_actions=self.max_actions,
        ).to(self.device)
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            net.load_state_dict(state)
            self.network = net
        except Exception:
            try:
                state = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
                net.load_state_dict(state, strict=False)
                self.network = net
            except Exception:
                pass

    # ── Act ──────────────────────────────────────────────────────────

    def _build_infoset_key(self, observation: Observation) -> str:
        """Build CFR infoset key from an Observation."""
        dice_str = "".join(str(d) for d in sorted(
            observation.private_info.get("my_dice", [])))
        bid = observation.public_info.get("last_bid")
        bid_str = f"{bid['quantity']},{bid['face']}" if bid else "None"
        my_count = observation.public_info.get("my_dice_count", self.num_dice)
        opp_count = observation.public_info.get("opponent_dice_count", self.num_dice)
        # Absolute count order (cf. encode_infoset: counts[0],counts[1])
        player = observation.player_id
        if player == 0:
            c0, c1 = my_count, opp_count
        else:
            c0, c1 = opp_count, my_count
        return f"dice={dice_str}|bid={bid_str}|counts={c0},{c1}|player={player}"

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel | None = None,
    ) -> tuple[Action, str, str]:
        """Select an action from the neural summary or uniform fallback."""
        n = len(legal_actions)
        if n == 0:
            raise ValueError("No legal actions available.")
        summary = self.predict_summary(observation, legal_actions)
        action_keys = summary["action_keys"]
        probs = [summary["average_policy"][k] + 1e-10 for k in action_keys]
        probs = [p / sum(probs) for p in probs]
        idx = sample_index(probs, self._rng)
        action = legal_actions[idx]
        summary["selected_action"] = action.model_dump()
        summary["selected_action_key"] = action_keys[idx]
        summary["selected_action_index"] = idx
        return action, json.dumps(summary, ensure_ascii=False), action.type

    # ── Reflect (no-op for CFR-based methods) ────────────────────────

    def reflect(
        self,
        trajectory,
        model: BaseModel | None = None,
        game_name: str = "",
    ) -> list:
        """Deep CFR does not learn from individual trajectories."""
        return []
