"""External-sampling MCCFR+ for Liar's Dice.

This rewrite keeps the implementation intentionally small:
  - one shared action ordering from ``methods.liars_dice_core``
  - one infoset encoding used by both training and inference
  - no fuzzy fallback, and no implicit training in the method constructor

The goal is to make the game-theoretic base trustworthy before layering
ToM on top of it.
"""

from __future__ import annotations

import itertools
import json
import ast
import os
import random
import time
from collections import Counter, defaultdict
from functools import lru_cache, wraps
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from base.interfaces import BaseModel
from base.types import Action, Observation
from games.liars_dice import LiarsDice
from base.liars_dice_core import (
    count_bid_matches,
    converged_strategy,
    infoset_key_from_state,
    opponent_reach,
    parse_bid_key,
    regret_matching_strategy,
    sample_index,
)


def time_log(label: str):
    """Log entry/exit timestamps for a function.

    Intended for coarse-grained tracing only. Recursive hot paths should
    either be wrapped conditionally or called through a depth-limited shim.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            print(f"[time_log] enter {label} t={started:.6f}", flush=True)
            try:
                return fn(*args, **kwargs)
            finally:
                ended = time.perf_counter()
                print(
                    f"[time_log] exit  {label} dt={ended - started:.6f}s "
                    f"t={ended:.6f}",
                    flush=True,
                )

        return wrapper

    return decorator


def encode_infoset(state, player: int) -> str:
    """Build the canonical infoset key from a game state."""
    return infoset_key_from_state(state, player)


InternalActionKey = tuple[int, ...]
InternalInfosetKey = tuple[Any, ...]


def _action_to_key(action: Action) -> InternalActionKey:
    if action.type == "challenge":
        return (0,)
    return (
        1,
        int(action.data.get("quantity", 0)),
        int(action.data.get("face", 0)),
    )


def _key_to_action(key: InternalActionKey, player_id: int) -> Action:
    if key[0] == 0:
        return Action(player_id=player_id, type="challenge")
    if key[0] != 1 or len(key) != 3:
        raise ValueError(f"Unknown internal action key: {key}")
    return Action(
        player_id=player_id,
        type="bid",
        data={"quantity": int(key[1]), "face": int(key[2])},
    )


def _infoset_key_from_state_internal(state, player: int) -> InternalInfosetKey:
    return (
        player,
        state.dice0 if player == 0 else state.dice1,
        state.count0,
        state.count1,
        state.last_bid,
    )


def _infoset_key_from_observation_internal(observation: Observation) -> InternalInfosetKey:
    player_id = observation.player_id
    my_dice = tuple(sorted(observation.private_info.get("my_dice", [])))
    my_count = int(observation.public_info.get("my_dice_count", len(my_dice)))
    opp_count = int(observation.public_info.get("opponent_dice_count", 0))
    last_bid = observation.public_info.get("last_bid")
    bid_tuple = None
    if isinstance(last_bid, dict):
        bid_tuple = (int(last_bid["quantity"]), int(last_bid["face"]))

    if player_id == 0:
        counts = (my_count, opp_count)
    else:
        counts = (opp_count, my_count)
    return (player_id, my_dice, counts[0], counts[1], bid_tuple)


def _parse_legacy_infoset_key(raw_key: str) -> InternalInfosetKey:
    """Parse the legacy string infoset format used by older saved strategies."""
    parts = {}
    for chunk in raw_key.split("|"):
        name, value = chunk.split("=", 1)
        parts[name] = value

    dice = tuple(int(ch) for ch in parts["dice"] if ch.isdigit())
    bid_raw = parts["bid"]
    if bid_raw == "None":
        bid_tuple = None
    else:
        quantity_str, face_str = bid_raw.split(",")
        bid_tuple = (int(quantity_str), int(face_str))

    count0_str, count1_str = parts["counts"].split(",")
    return (
        int(parts["player"]),
        dice,
        int(count0_str),
        int(count1_str),
        bid_tuple,
    )


def _load_infoset_key(raw_key: str) -> InternalInfosetKey:
    if raw_key.startswith("("):
        value = ast.literal_eval(raw_key)
        return tuple(value)
    return _parse_legacy_infoset_key(raw_key)


def _load_action_key(raw_key: str) -> InternalActionKey:
    if raw_key.startswith("("):
        value = ast.literal_eval(raw_key)
        return tuple(value)
    if raw_key == "challenge":
        return (0,)
    if raw_key.startswith("bid_"):
        quantity, face = parse_bid_key(raw_key)
        return (1, quantity, face)
    raise ValueError(f"Unknown saved action key: {raw_key}")


def encode_action(action: Action) -> str:
    """Encode an Action to a canonical string key."""
    key = _action_to_key(action)
    if key[0] == 0:
        return "challenge"
    return f"bid_{key[1]}_{key[2]}"


def decode_action(action_key: str, player: int, legal_actions: list[Action]) -> Action:
    """Decode an action key back to a legal Action.

    This only returns a legal action from the current state; it never creates
    an action that is not present in ``legal_actions``.
    """
    for action in legal_actions:
        if action_key == encode_action(action):
            return action
    raise ValueError(f"No legal action matching key: {action_key}")


@dataclass(slots=True)
class CFRState:
    """Compact internal state used by the training walk."""

    dice0: tuple[int, ...]
    dice1: tuple[int, ...]
    count0: int
    count1: int
    current_player: int
    last_bid: tuple[int, int] | None
    phase: str
    round_num: int
    rewards: tuple[float, float] | None = None


class LiarsDiceCFRTrainer:
    """External-sampling MCCFR+ trainer for Liar's Dice."""

    def __init__(
        self,
        num_dice: int = 4,
        num_faces: int = 4,
        iterations: int = 200_000,
        seed: int = 0,
        traverser_action_sample_size: int | None = None,
    ):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.game = LiarsDice(num_dice=num_dice, num_faces=num_faces, seed=seed)
        self.traverser_action_sample_size = (
            None
            if traverser_action_sample_size is None or traverser_action_sample_size <= 0
            else int(traverser_action_sample_size)
        )

        self.regret_sum: dict[InternalInfosetKey, dict[InternalActionKey, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.strategy_sum: dict[InternalInfosetKey, dict[InternalActionKey, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._legal_cache: dict[tuple[int, int, tuple[int, int] | None], tuple[InternalActionKey, ...]] = {}
        self.strategy: dict[InternalInfosetKey, dict[InternalActionKey, float]] = {}
        self._trained = False
        self._num_infosets = 0
        self._trace_enabled = False
        self._trace_depth = 0
        self._trace_walk_root_only = True

    def _reset_state(self) -> CFRState:
        dice0 = tuple(sorted(self.rng.randint(1, self.num_faces) for _ in range(self.num_dice)))
        dice1 = tuple(sorted(self.rng.randint(1, self.num_faces) for _ in range(self.num_dice)))
        return CFRState(
            dice0=dice0,
            dice1=dice1,
            count0=self.num_dice,
            count1=self.num_dice,
            current_player=0,
            last_bid=None,
            phase="bidding",
            round_num=1,
            rewards=None,
        )

    def _legal_keys(self, state: CFRState) -> tuple[InternalActionKey, ...]:
        cache_key = (state.count0, state.count1, state.last_bid)
        cached = self._legal_cache.get(cache_key)
        if cached is not None:
            return cached

        total_dice = state.count0 + state.count1
        if state.last_bid is None:
            keys: tuple[InternalActionKey, ...] = tuple(
                (1, quantity, face)
                for quantity in range(1, total_dice + 1)
                for face in range(1, self.num_faces + 1)
            )
        else:
            quantity, face = state.last_bid
            keys = ((0,),) + tuple(
                (1, quantity, next_face)
                for next_face in range(face + 1, self.num_faces + 1)
                if quantity <= total_dice
            ) + tuple(
                (1, next_quantity, next_face)
                for next_quantity in range(quantity + 1, total_dice + 1)
                for next_face in range(1, self.num_faces + 1)
            )
        self._legal_cache[cache_key] = keys
        return keys

    def _step(self, state: CFRState, action_key: InternalActionKey) -> CFRState:
        if state.phase == "terminal":
            return state

        player = state.current_player
        if action_key[0] == 0:
            assert state.last_bid is not None
            quantity, face = state.last_bid
            actual = count_bid_matches(state.dice0 + state.dice1, face)
            bid_true = actual >= quantity
            loser = player if bid_true else 1 - player
            count0, count1 = state.count0, state.count1
            if loser == 0:
                count0 -= 1
            else:
                count1 -= 1

            if count0 == 0 or count1 == 0:
                winner = 1 - loser
                return CFRState(
                    dice0=state.dice0,
                    dice1=state.dice1,
                    count0=count0,
                    count1=count1,
                    current_player=-1,
                    last_bid=None,
                    phase="terminal",
                    round_num=state.round_num,
                    rewards=(1.0, -1.0) if winner == 0 else (-1.0, 1.0),
                )

            next_dice0 = tuple(sorted(self.rng.randint(1, self.num_faces) for _ in range(count0)))
            next_dice1 = tuple(sorted(self.rng.randint(1, self.num_faces) for _ in range(count1)))
            return CFRState(
                dice0=next_dice0,
                dice1=next_dice1,
                count0=count0,
                count1=count1,
                current_player=loser,
                last_bid=None,
                phase="bidding",
                round_num=state.round_num + 1,
                rewards=None,
            )

        if action_key[0] != 1 or len(action_key) != 3:
            raise ValueError(f"Unknown action key: {action_key}")
        return CFRState(
            dice0=state.dice0,
            dice1=state.dice1,
            count0=state.count0,
            count1=state.count1,
            current_player=1 - player,
            last_bid=(int(action_key[1]), int(action_key[2])),
            phase="bidding",
            round_num=state.round_num,
            rewards=None,
        )

    @time_log("train")
    def train(self, verbose: bool = True, trace: bool = False,
              eval_every: int = 0) -> None:
        self._trace_enabled = trace
        if verbose:
            print(
                f"Training MCCFR+ ({self.iterations} iterations, "
                f"{self.num_dice}d×{self.num_faces}f)..."
            )

        started_at = time.perf_counter()
        with tqdm(total=self.iterations, unit="iter", desc="MCCFR+",
                  disable=not verbose) as pbar:
            for i in range(self.iterations):
                traverser = i % 2
                state = self._reset_state()
                self._walk(state, 1.0, 1.0, traverser)

                elapsed = time.perf_counter() - started_at
                it_per_sec = (i + 1) / max(elapsed, 1e-9)
                postfix = {
                    "infosets": len(self.strategy_sum),
                    "iter/s": f"{it_per_sec:.1f}",
                }

                if eval_every > 0 and (i + 1) % eval_every == 0:
                    exp = self.compute_exploitability()
                    postfix["exp"] = f"{exp:.6g}"

                pbar.set_postfix(postfix)
                pbar.update(1)

        self.strategy = {}
        for infoset, sums in self.strategy_sum.items():
            avg = converged_strategy(dict(sums))
            if avg:
                self.strategy[infoset] = avg

        self._num_infosets = len(self.strategy)
        self._trained = True
        if verbose:
            print(f"MCCFR+ trained. {self._num_infosets} information sets stored.")
            if eval_every > 0:
                final_exp = self.compute_exploitability()
                print(f"Final exploitability: {final_exp:.6g}")

    def _walk(self, state: CFRState, p0_reach: float, p1_reach: float, traverser: int) -> float:
        trace_here = self._trace_enabled and (
            not self._trace_walk_root_only or self._trace_depth == 0
        )
        if trace_here:
            started = time.perf_counter()
            print(
                f"[time_log] enter walk depth={self._trace_depth} "
                f"player={state.current_player} phase={state.phase} "
                f"last_bid={state.last_bid} t={started:.6f}",
                flush=True,
            )
            self._trace_depth += 1
        if state.phase == "terminal":
            assert state.rewards is not None
            result = state.rewards[traverser]
            if trace_here:
                self._trace_depth -= 1
                ended = time.perf_counter()
                print(
                    f"[time_log] exit  walk depth={self._trace_depth} "
                    f"dt={ended - started:.6f}s t={ended:.6f}",
                    flush=True,
                )
            return result

        player = state.current_player
        legal_keys = self._legal_keys(state)
        if not legal_keys:
            raise RuntimeError("No legal actions available for a non-terminal state.")

        infoset = _infoset_key_from_state_internal(state, player)

        regrets = self.regret_sum[infoset]
        if len(regrets) < len(legal_keys):
            for key in legal_keys:
                _ = regrets[key]

        strat = regret_matching_strategy([regrets[key] for key in legal_keys], len(legal_keys))

        opp_reach = opponent_reach(p0_reach, p1_reach, player)
        for idx, key in enumerate(legal_keys):
            self.strategy_sum[infoset][key] += opp_reach * strat[idx]

        if player == traverser:
            if (
                self.traverser_action_sample_size is not None
                and len(legal_keys) > self.traverser_action_sample_size
            ):
                ranked = sorted(
                    range(len(legal_keys)),
                    key=lambda idx: strat[idx],
                    reverse=True,
                )
                action_indices = ranked[: self.traverser_action_sample_size]
            else:
                action_indices = list(range(len(legal_keys)))

            sampled_keys = [legal_keys[idx] for idx in action_indices]
            sampled_values: list[float] = []
            for key in sampled_keys:
                next_state = self._step(state, key)
                sampled_values.append(
                    self._walk(next_state, p0_reach, p1_reach, traverser)
                )
            weight_sum = sum(strat[idx] for idx in action_indices) or 1.0
            node_value = sum(
                strat[idx] * value
                for idx, value in zip(action_indices, sampled_values)
            ) / weight_sum
            scale = len(legal_keys) / max(1, len(action_indices))
            for idx, key, value in zip(action_indices, sampled_keys, sampled_values):
                regrets[key] = max(
                    0.0,
                    regrets[key] + opp_reach * scale * (value - node_value),
                )
            if trace_here:
                self._trace_depth -= 1
                ended = time.perf_counter()
                print(
                    f"[time_log] exit  walk depth={self._trace_depth} "
                    f"dt={ended - started:.6f}s t={ended:.6f}",
                    flush=True,
                )
            return node_value

        sampled_idx = sample_index(strat, self.rng)
        sampled_key = legal_keys[sampled_idx]
        next_state = self._step(state, sampled_key)
        child_p0 = p0_reach * strat[sampled_idx] if player == 0 else p0_reach
        child_p1 = p1_reach * strat[sampled_idx] if player == 1 else p1_reach
        result = self._walk(next_state, child_p0, child_p1, traverser)
        if trace_here:
            self._trace_depth -= 1
            ended = time.perf_counter()
            print(
                f"[time_log] exit  walk depth={self._trace_depth} "
                f"dt={ended - started:.6f}s t={ended:.6f}",
                flush=True,
            )
        return result

    # ── Exploitability computation ──────────────────────────────────────

    @lru_cache(maxsize=None)
    def _dice_outcomes(self, num_dice: int) -> tuple:
        """Return tuple of ((sorted_dice_tuple, probability), ...) for num_dice dice."""
        counts: dict[tuple[int, ...], int] = Counter()
        for outcome in itertools.product(
            range(1, self.num_faces + 1), repeat=num_dice
        ):
            counts[tuple(sorted(outcome))] += 1
        total = self.num_faces ** num_dice
        return tuple((dice, cnt / total) for dice, cnt in counts.items())

    def compute_exploitability(self) -> float:
        """Exact exploitability via full-enumeration belief-state BR.

        Enumerates BR dice and ALL opponent dice outcomes (small set for 2×3
        or 1×4).  Opponent actions at opponent decision nodes are enumerated
        and grouped by resulting public state; within each group the belief
        is updated via Bayes' rule.

        This gives the correct imperfect-information best-response value.
        """
        saved_state = self.rng.getstate()
        try:
            snapshot = self._build_snapshot()
            br0 = self._br_player_value(0, snapshot)
            br1 = self._br_player_value(1, snapshot)
            return (br0 + br1) / 2.0
        finally:
            self.rng.setstate(saved_state)

    def _build_snapshot(self):
        snapshot = {}
        if self.strategy_sum:
            for infoset, sums in self.strategy_sum.items():
                avg = converged_strategy(dict(sums))
                if avg:
                    snapshot[infoset] = avg
        else:
            for infoset, avg in self.strategy.items():
                if avg:
                    snapshot[infoset] = dict(avg)
        return snapshot

    # ── Belief-state BR (cached enumeration) ─────────────────────────

    def _br_player_value(self, br_player: int, snapshot: dict) -> float:
        """BR value = Σ P(BR dice) × V(belief=prior, start_state)."""
        self._br_cache = {}
        total = 0.0
        for br_dice, br_prob in self._dice_outcomes(self.num_dice):
            belief = self._dice_outcomes(self.num_dice)  # [(opp_dice, prob), ...]
            val = self._br_belief(
                br_player, br_dice, belief, snapshot,
                self.num_dice, self.num_dice, 0, None, 1,
            )
            total += br_prob * val
        self._br_cache = {}
        return total

    def _legal_keys_for_pub(self, c0: int, c1: int,
                             last_bid: tuple[int, int] | None):
        """Legal action keys from public info (helpers cache internally)."""
        dummy = CFRState(
            dice0=(), dice1=(), count0=c0, count1=c1,
            current_player=0, last_bid=last_bid,
            phase="bidding", round_num=1, rewards=None,
        )
        return self._legal_keys(dummy)

    def _br_belief_cache_key(self, br_dice, belief, c0, c1,
                              current_player, last_bid, round_num):
        """Hashable cache key — belief weights aligned with _dice_outcomes order."""
        return (br_dice, tuple(w for _, w in belief),
                c0, c1, current_player, last_bid, round_num)

    def _br_belief(self, br_player, br_dice, belief, snapshot,
                   c0, c1, current_player, last_bid, round_num) -> float:
        """Belief-state BR — full enumeration with caching and DAG reuse."""
        # Terminal check
        if c0 == 0 or c1 == 0:
            winner = 1 if c0 == 0 else 0
            return 1.0 if winner == br_player else -1.0

        key = self._br_belief_cache_key(br_dice, belief, c0, c1,
                                         current_player, last_bid, round_num)
        cached = self._br_cache.get(key)
        if cached is not None:
            return cached

        legal_keys = self._legal_keys_for_pub(c0, c1, last_bid)

        if current_player == br_player:
            result = self._br_decision(
                br_player, br_dice, belief, snapshot,
                c0, c1, last_bid, round_num, legal_keys,
            )
        else:
            result = self._br_opponent_expect(
                br_player, br_dice, belief, snapshot,
                c0, c1, current_player, last_bid, round_num, legal_keys,
            )

        self._br_cache[key] = result
        return result

    # ── BR decision: average over belief, pick max action ─────────────

    def _br_decision(self, br_player, br_dice, belief, snapshot,
                     c0, c1, last_bid, round_num, legal_keys):
        best = -float('inf')
        for key in legal_keys:
            val = self._br_action_val(
                br_player, br_dice, belief, snapshot,
                c0, c1, last_bid, round_num, key,
            )
            if val > best:
                best = val
        return best

    def _br_action_val(self, br_player, br_dice, belief, snapshot,
                       c0, c1, last_bid, round_num, key):
        if key[0] == 0:  # challenge
            return self._br_challenge_val(
                br_player, br_dice, belief, snapshot,
                c0, c1, last_bid, round_num,
            )
        # bid
        return self._br_bid_val(
            br_player, br_dice, belief, snapshot,
            c0, c1, int(key[1]), int(key[2]), round_num,
        )

    def _br_challenge_val(self, br_player, br_dice, belief, snapshot,
                          c0, c1, last_bid, round_num):
        """Value of challenging opponent's bid — group by dice outcome."""
        assert last_bid is not None
        q, f = last_bid

        # Group belief by challenge outcome.
        # BR is the challenger: if bid_true the challenger (BR) loses,
        # otherwise the bidder (opponent) loses.
        groups: dict[tuple, list] = {}
        for opp_dice, w in belief:
            actual = count_bid_matches(br_dice + opp_dice, f)
            bid_true = actual >= q
            loser = br_player if bid_true else 1 - br_player
            new_c0 = c0 - (1 if loser == 0 else 0)
            new_c1 = c1 - (1 if loser == 1 else 0)
            groups.setdefault((new_c0, new_c1, loser), []).append((opp_dice, w))

        total = 0.0
        for (new_c0, new_c1, loser), comps in groups.items():
            group_w = sum(w for _, w in comps)
            if new_c0 == 0 or new_c1 == 0:
                winner = 1 - loser
                total += group_w * (1.0 if winner == br_player else -1.0)
            else:
                # New round: both players re-roll → re-enumerate BR + opp dice
                new_br_cnt = new_c0 if br_player == 0 else new_c1
                new_opp_cnt = new_c1 if br_player == 0 else new_c0
                new_opp_belief = self._dice_outcomes(new_opp_cnt)
                for new_br_dice, new_br_prob in self._dice_outcomes(new_br_cnt):
                    total += group_w * new_br_prob * self._br_belief(
                        br_player, new_br_dice, new_opp_belief, snapshot,
                        new_c0, new_c1, loser, None, round_num + 1,
                    )
        return total

    def _br_bid_val(self, br_player, br_dice, belief, snapshot,
                    c0, c1, q, f, round_num):
        """Value of bidding (q, f) — passes to opponent, belief unchanged."""
        return self._br_belief(
            br_player, br_dice, belief, snapshot,
            c0, c1, 1 - br_player, (q, f), round_num,
        )

    # ── Opponent decision: enumerate actions, group by outcome ────────

    def _br_opponent_expect(self, br_player, br_dice, belief, snapshot,
                            c0, c1, current_player, last_bid, round_num,
                            legal_keys):
        """Expected value at opponent decision node.

        Enumerates all actions for each belief component, groups by resulting
        public state, computes Bayesian belief update per group, and recurses.
        """
        groups: dict[tuple, list] = {}

        for opp_dice, w in belief:
            d0 = br_dice if br_player == 0 else opp_dice
            d1 = opp_dice if br_player == 0 else br_dice
            state = CFRState(
                dice0=d0, dice1=d1, count0=c0, count1=c1,
                current_player=current_player, last_bid=last_bid,
                phase="bidding", round_num=round_num, rewards=None,
            )
            ik = _infoset_key_from_state_internal(state, current_player)
            strat = snapshot.get(ik, {})

            probs = [strat.get(k, 0.0) for k in legal_keys]
            total_prob = sum(probs)

            if total_prob <= 0:
                uniform_p = 1.0 / len(legal_keys)
                for key in legal_keys:
                    next_state = self._step(state, key)
                    ok = self._br_outcome_key(next_state)
                    groups.setdefault(ok, []).append(
                        (opp_dice, w, uniform_p, next_state),
                    )
            else:
                for idx, key in enumerate(legal_keys):
                    prob = probs[idx]
                    if prob > 0:
                        next_state = self._step(state, key)
                        ok = self._br_outcome_key(next_state)
                        groups.setdefault(ok, []).append(
                            (opp_dice, w, prob, next_state),
                        )

        total_val = 0.0
        for ok, comps in groups.items():
            group_weight = sum(w * prob for _, w, prob, _ in comps)
            if group_weight <= 0:
                continue

            # Bayesian belief update for this group
            new_belief = [
                (opp_dice, w * prob / group_weight)
                for opp_dice, w, prob, _ in comps
            ]

            s0 = comps[0][3]
            if s0.phase == "terminal":
                val = s0.rewards[br_player]
            else:
                if s0.last_bid is None:
                    # Challenge continued → new round: re-enumerate dice
                    new_br_cnt = s0.count0 if br_player == 0 else s0.count1
                    new_opp_cnt = s0.count1 if br_player == 0 else s0.count0
                    new_belief = self._dice_outcomes(new_opp_cnt)
                    val = 0.0
                    for new_br_dice, new_br_prob in self._dice_outcomes(new_br_cnt):
                        val += new_br_prob * self._br_belief(
                            br_player, new_br_dice, new_belief, snapshot,
                            s0.count0, s0.count1, s0.current_player,
                            s0.last_bid, s0.round_num,
                        )
                else:
                    val = self._br_belief(
                        br_player, br_dice, new_belief, snapshot,
                        s0.count0, s0.count1, s0.current_player,
                        s0.last_bid, s0.round_num,
                    )

            total_val += group_weight * val

        return total_val

    @staticmethod
    def _br_outcome_key(state: CFRState) -> tuple:
        """Unique key for grouping opponent action outcomes."""
        if state.phase == "terminal":
            return ("terminal", state.count0, state.count1)
        return (state.count0, state.count1, state.current_player,
                state.last_bid, state.round_num)

    # ── Nash action-value computation (for ToM prompt) ────────────────

    def _nash_cache_key(
        self, my_dice, belief, c0, c1, current_player, last_bid, round_num,
    ):
        return (tuple(my_dice), tuple(w for _, w in belief),
                c0, c1, current_player, last_bid, round_num)

    def _nash_value(
        self,
        my_id: int,
        my_dice: tuple[int, ...],
        belief: list[tuple[tuple[int, ...], float]],
        snapshot: dict,
        c0: int, c1: int,
        current_player: int,
        last_bid: tuple[int, int] | None,
        round_num: int,
    ) -> float:
        """Expected value for *my_id* assuming both players follow the Nash strategy.

        At player nodes: average over legal actions using snapshot probabilities.
        At opponent nodes: group by action, Bayesian belief update, recurse.
        """
        if c0 == 0 or c1 == 0:
            winner = 1 if c0 == 0 else 0
            return 1.0 if winner == my_id else -1.0

        ck = self._nash_cache_key(my_dice, belief, c0, c1, current_player,
                                   last_bid, round_num)
        cached = self._nash_cache.get(ck)
        if cached is not None:
            return cached

        legal_keys = self._legal_keys_for_pub(c0, c1, last_bid)

        if current_player == my_id:
            result = self._nash_decision(
                my_id, my_dice, belief, snapshot,
                c0, c1, last_bid, round_num, legal_keys,
            )
        else:
            result = self._nash_opponent_expect(
                my_id, my_dice, belief, snapshot,
                c0, c1, current_player, last_bid, round_num, legal_keys,
            )
        self._nash_cache[ck] = result
        return result

    def _nash_decision(
        self, my_id, my_dice, belief, snapshot,
        c0, c1, last_bid, round_num, legal_keys,
    ) -> float:
        """At our decision node: average over legal actions per-belief-component."""
        total = 0.0
        for opp_dice, w in belief:
            my_dice_t = tuple(my_dice)
            d0 = my_dice_t if my_id == 0 else opp_dice
            d1 = opp_dice if my_id == 0 else my_dice_t
            state = CFRState(
                dice0=d0, dice1=d1, count0=c0, count1=c1,
                current_player=my_id, last_bid=last_bid,
                phase="bidding", round_num=round_num, rewards=None,
            )
            ik = _infoset_key_from_state_internal(state, my_id)
            strat = snapshot.get(ik, {})

            av = 0.0
            sp = 0.0
            for key in legal_keys:
                prob = strat.get(key, 0.0)
                if prob > 0:
                    av += prob * self._nash_action_value(
                        my_id, my_dice_t, belief, snapshot,
                        c0, c1, last_bid, round_num, key,
                    )
                    sp += prob

            if sp > 0:
                total += w * (av / sp)
            else:
                # Uniform fallback
                u_av = 0.0
                for key in legal_keys:
                    u_av += self._nash_action_value(
                        my_id, my_dice_t, belief, snapshot,
                        c0, c1, last_bid, round_num, key,
                    )
                total += w * (u_av / max(len(legal_keys), 1))
        return total

    def _nash_action_value(
        self, my_id, my_dice, belief, snapshot,
        c0, c1, last_bid, round_num, key,
    ) -> float:
        if key[0] == 0:  # challenge
            return self._nash_challenge_value(
                my_id, my_dice, belief, snapshot,
                c0, c1, last_bid, round_num,
            )
        # bid
        return self._nash_bid_value(
            my_id, my_dice, belief, snapshot,
            c0, c1, int(key[1]), int(key[2]), round_num,
        )

    def _nash_challenge_value(
        self, my_id, my_dice, belief, snapshot,
        c0, c1, last_bid, round_num,
    ) -> float:
        """Value of challenging — group belief by challenge outcome, then recurse."""
        assert last_bid is not None
        q, f = last_bid
        groups: dict[tuple, list] = {}
        for opp_dice, w in belief:
            actual = count_bid_matches(tuple(my_dice) + opp_dice, f)
            bid_true = actual >= q
            loser = my_id if bid_true else 1 - my_id
            new_c0 = c0 - (1 if loser == 0 else 0)
            new_c1 = c1 - (1 if loser == 1 else 0)
            groups.setdefault((new_c0, new_c1, loser), []).append((opp_dice, w))

        total = 0.0
        for (new_c0, new_c1, loser), comps in groups.items():
            group_w = sum(w for _, w in comps)
            if new_c0 == 0 or new_c1 == 0:
                winner = 1 - loser
                total += group_w * (1.0 if winner == my_id else -1.0)
            else:
                # New round: both re-roll
                new_my_cnt = new_c0 if my_id == 0 else new_c1
                new_opp_cnt = new_c1 if my_id == 0 else new_c0
                new_opp_belief = self._dice_outcomes(new_opp_cnt)
                for new_my_dice, new_my_prob in self._dice_outcomes(new_my_cnt):
                    total += group_w * new_my_prob * self._nash_value(
                        my_id, new_my_dice, new_opp_belief, snapshot,
                        new_c0, new_c1, loser, None, round_num + 1,
                    )
        return total

    def _nash_bid_value(
        self, my_id, my_dice, belief, snapshot,
        c0, c1, q, f, round_num,
    ) -> float:
        """Value of bidding (q, f) — passes to opponent, belief unchanged."""
        return self._nash_value(
            my_id, my_dice, belief, snapshot,
            c0, c1, 1 - my_id, (q, f), round_num,
        )

    def _nash_opponent_expect(
        self, my_id, my_dice, belief, snapshot,
        c0, c1, current_player, last_bid, round_num, legal_keys,
    ) -> float:
        """Expected value at opponent decision node under Nash play."""
        groups: dict[tuple, list] = {}
        for opp_dice, w in belief:
            d0 = tuple(my_dice) if my_id == 0 else opp_dice
            d1 = opp_dice if my_id == 0 else tuple(my_dice)
            state = CFRState(
                dice0=d0, dice1=d1, count0=c0, count1=c1,
                current_player=current_player, last_bid=last_bid,
                phase="bidding", round_num=round_num, rewards=None,
            )
            ik = _infoset_key_from_state_internal(state, current_player)
            strat = snapshot.get(ik, {})
            probs = [strat.get(k, 0.0) for k in legal_keys]
            total_prob = sum(probs)

            if total_prob <= 0:
                uniform_p = 1.0 / len(legal_keys)
                for key in legal_keys:
                    next_state = self._step(state, key)
                    ok = self._br_outcome_key(next_state)
                    groups.setdefault(ok, []).append((opp_dice, w, uniform_p, next_state))
            else:
                for idx, key in enumerate(legal_keys):
                    prob = probs[idx]
                    if prob > 0:
                        next_state = self._step(state, key)
                        ok = self._br_outcome_key(next_state)
                        groups.setdefault(ok, []).append((opp_dice, w, prob, next_state))

        total_val = 0.0
        for ok, comps in groups.items():
            group_weight = sum(w * prob for _, w, prob, _ in comps)
            if group_weight <= 0:
                continue
            new_belief = [
                (opp_dice, w * prob / group_weight)
                for opp_dice, w, prob, _ in comps
            ]
            s0 = comps[0][3]
            if s0.phase == "terminal":
                total_val += group_weight * s0.rewards[my_id]
            else:
                if s0.last_bid is None:
                    # Challenge continued → new round
                    new_my_cnt = s0.count0 if my_id == 0 else s0.count1
                    new_opp_cnt = s0.count1 if my_id == 0 else s0.count0
                    new_opp_belief = self._dice_outcomes(new_opp_cnt)
                    val = 0.0
                    for new_my_dice, new_my_prob in self._dice_outcomes(new_my_cnt):
                        val += new_my_prob * self._nash_value(
                            my_id, new_my_dice, new_opp_belief, snapshot,
                            s0.count0, s0.count1, s0.current_player,
                            s0.last_bid, s0.round_num,
                        )
                    total_val += group_weight * val
                else:
                    total_val += group_weight * self._nash_value(
                        my_id, tuple(my_dice), new_belief, snapshot,
                        s0.count0, s0.count1, s0.current_player,
                        s0.last_bid, s0.round_num,
                    )
        return total_val

    def action_values(
        self,
        observation: Observation,
        legal_actions: list[Action],
    ) -> list[tuple[str, float]]:
        """Compute Nash equilibrium EV for each legal action.

        Returns [(label, EV), ...] sorted by EV descending.
        EV is the expected per-round reward (+1 = win, -1 = lose, 0 = draw).
        """
        if not self.strategy:
            return []

        self._nash_cache: dict = {}
        snapshot = dict(self.strategy)
        my_id = observation.player_id
        my_dice = tuple(sorted(observation.private_info.get("my_dice", [])))
        my_count = int(observation.public_info.get("my_dice_count", len(my_dice)))
        opp_count = int(observation.public_info.get("opponent_dice_count", 0))
        last_bid = observation.public_info.get("last_bid")
        bid_tuple: tuple[int, int] | None = None
        if isinstance(last_bid, dict):
            bid_tuple = (int(last_bid["quantity"]), int(last_bid["face"]))

        if my_id == 0:
            c0, c1 = my_count, opp_count
        else:
            c0, c1 = opp_count, my_count

        belief = self._dice_outcomes(opp_count)

        results: list[tuple[str, float]] = []
        for action in legal_actions:
            key = _action_to_key(action)
            if key[0] == 0:  # challenge
                if bid_tuple is None:
                    ev = -1.0  # no bid to challenge
                else:
                    ev = self._nash_challenge_value(
                        my_id, my_dice, belief, snapshot,
                        c0, c1, bid_tuple, 1,
                    )
            else:
                q, f = int(key[1]), int(key[2])
                ev = self._nash_bid_value(
                    my_id, my_dice, belief, snapshot,
                    c0, c1, q, f, 1,
                )

            if action.type == "challenge":
                label = "challenge"
            else:
                label = f"bid {action.data.get('quantity')}x{action.data.get('face')}"
            results.append((label, round(ev, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save(self, path: str, metadata: dict | None = None) -> None:
        data = {
            "num_dice": self.num_dice,
            "num_faces": self.num_faces,
            "iterations": self.iterations,
            "traverser_action_sample_size": self.traverser_action_sample_size,
            "seed": self.rng.getstate(),
            "num_infosets": self._num_infosets,
            "strategy": {
                repr(k): {repr(ak): p for ak, p in v.items()}
                for k, v in self.strategy.items()
            },
        }
        if metadata:
            data["metadata"] = metadata

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.num_dice = int(data.get("num_dice", self.num_dice))
        self.num_faces = int(data.get("num_faces", self.num_faces))
        self.iterations = int(data.get("iterations", self.iterations))
        self.traverser_action_sample_size = data.get(
            "traverser_action_sample_size", self.traverser_action_sample_size
        )
        self.strategy = {
            _load_infoset_key(k): {
                _load_action_key(ak): float(p) for ak, p in v.items()
            }
            for k, v in data["strategy"].items()
        }
        self._num_infosets = len(self.strategy)
        self._trained = True

    def infoset_coverage(self) -> int:
        return len(self.strategy)


class LiarsDiceCFRMethod:
    """Load a trained CFR+ strategy and sample legal actions from it."""

    _DEFAULT_PATH = "results/liars_dice/cfr_model/cfr_4x4.json"

    def __init__(
        self,
        name: str = "CFR+",
        strategy_path: str = "",
        seed: int = 0,
    ):
        self.name = name
        self.method_type = "cfr"
        self._rng = random.Random(seed)

        path = strategy_path or self._DEFAULT_PATH
        self._trainer: LiarsDiceCFRTrainer | None = None
        self._strategy: dict[InternalInfosetKey, dict[InternalActionKey, float]] = {}

        if os.path.exists(path):
            self._trainer = LiarsDiceCFRTrainer()
            self._trainer.load(path)
            self._strategy = self._trainer.strategy
        else:
            print(f"CFR strategy not found at {path}; using uniform fallback.")

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel | None = None,
    ) -> tuple[Action, str, str]:
        info_key = _infoset_key_from_observation_internal(observation)
        strategy = self._strategy.get(info_key, {})

        action_keys = [_action_to_key(a) for a in legal_actions]
        probs = [strategy.get(key, 0.0) for key in action_keys]
        total = sum(probs)

        if total <= 0:
            idx = self._rng.randint(0, len(legal_actions) - 1)
            action = legal_actions[idx]
            return action, f"CFR+ uniform fallback: {info_key} -> selected={action}", action.type

        norm_probs = [p / total for p in probs]
        idx = sample_index(norm_probs, self._rng)
        action = legal_actions[idx]
        return action, f"CFR+ lookup: {info_key} -> selected={action}", action.type

    def reflect(self, trajectory, model=None, game_name=""):
        return []
