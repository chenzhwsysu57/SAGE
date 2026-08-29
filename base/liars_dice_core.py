"""Shared helpers for Liar's Dice game-theoretic implementations.

This module is the single source of truth for:
  - action encoding / decoding
  - legal action ordering
  - infoset key construction
  - basic CFR math helpers

Keeping these helpers in one place avoids the training/inference drift that
the previous Liar's Dice implementation accumulated.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from base.types import Action, Observation


def bid_key(quantity: int, face: int) -> str:
    return f"bid_{quantity}_{face}"


def action_key(action: Action) -> str:
    if action.type == "challenge":
        return "challenge"
    if action.type == "bid":
        quantity = int(action.data.get("quantity", 0))
        face = int(action.data.get("face", 0))
        return bid_key(quantity, face)
    raise ValueError(f"Unknown action type: {action.type}")


def action_from_key(action_key: str, player_id: int) -> Action:
    if action_key == "challenge":
        return Action(player_id=player_id, type="challenge")
    if not action_key.startswith("bid_"):
        raise ValueError(f"Unknown action key: {action_key}")
    _, quantity_str, face_str = action_key.split("_")
    return Action(
        player_id=player_id,
        type="bid",
        data={"quantity": int(quantity_str), "face": int(face_str)},
    )


def parse_bid_key(action_key: str) -> tuple[int, int]:
    if not action_key.startswith("bid_"):
        raise ValueError(f"Not a bid key: {action_key}")
    _, quantity_str, face_str = action_key.split("_")
    return int(quantity_str), int(face_str)


def legal_action_keys(
    *,
    total_dice: int,
    num_faces: int,
    last_bid: tuple[int, int] | None,
) -> tuple[str, ...]:
    """Return canonical legal action keys in deterministic order."""
    keys: list[str] = []
    if last_bid is None:
        for quantity in range(1, total_dice + 1):
            for face in range(1, num_faces + 1):
                keys.append(bid_key(quantity, face))
        return tuple(keys)

    quantity, face = last_bid
    keys.append("challenge")

    for next_face in range(face + 1, num_faces + 1):
        if quantity <= total_dice:
            keys.append(bid_key(quantity, next_face))

    for next_quantity in range(quantity + 1, total_dice + 1):
        for next_face in range(1, num_faces + 1):
            keys.append(bid_key(next_quantity, next_face))

    return tuple(keys)


def legal_actions_from_keys(keys: Iterable[str], player_id: int) -> list[Action]:
    return [action_from_key(key, player_id) for key in keys]


def match_legal_action(action: Action, legal_actions: Sequence[Action]) -> Action | None:
    try:
        target = action_key(action)
    except ValueError:
        return None
    for legal in legal_actions:
        if action_key(legal) == target:
            return legal
    return None


def infoset_key_from_parts(
    *,
    dice: Sequence[int],
    dice_counts: tuple[int, int],
    last_bid: tuple[int, int] | None,
    player_id: int,
) -> str:
    dice_str = "".join(str(d) for d in sorted(dice))
    bid_str = f"{last_bid[0]},{last_bid[1]}" if last_bid else "None"
    return (
        f"dice={dice_str}|bid={bid_str}|"
        f"counts={dice_counts[0]},{dice_counts[1]}|player={player_id}"
    )


def infoset_key_from_state(state: Any, player_id: int) -> str:
    return infoset_key_from_parts(
        dice=state.dice[player_id],
        dice_counts=(state.dice_counts[0], state.dice_counts[1]),
        last_bid=state.last_bid,
        player_id=player_id,
    )


def infoset_key_from_observation(
    observation: Observation,
    num_faces: int | None = None,
) -> str:
    """Build the CFR infoset key from an Observation.

    ``num_faces`` is accepted for API symmetry with other helpers but is not
    required for the current encoding.
    """
    player_id = observation.player_id
    my_dice = observation.private_info.get("my_dice", [])
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
    return infoset_key_from_parts(
        dice=my_dice,
        dice_counts=counts,
        last_bid=bid_tuple,
        player_id=player_id,
    )


def regret_matching_strategy(regrets: Sequence[float], n_actions: int) -> list[float]:
    positive = [max(float(r), 0.0) for r in regrets]
    total = sum(positive)
    if total > 0:
        return [r / total for r in positive]
    return [1.0 / n_actions] * n_actions


def converged_strategy(strategy_sum: dict[str, float]) -> dict[str, float]:
    total = sum(strategy_sum.values())
    if total > 0:
        return {key: value / total for key, value in strategy_sum.items()}
    return {}


def sample_index(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    last = len(probs) - 1
    for idx, prob in enumerate(probs):
        cumulative += prob
        if r <= cumulative:
            return idx
    return last


def opponent_reach(p0_reach: float, p1_reach: float, player: int) -> float:
    if player == 0:
        return p1_reach
    if player == 1:
        return p0_reach
    raise ValueError(f"Invalid player index: {player}")


def count_bid_matches(dice: Sequence[int], face: int) -> int:
    """Count how many dice satisfy a bid under wild-1 rules.

    Rules used in this repo:
      - If the challenged face is 1, only 1s count.
      - If the challenged face is > 1, both that face and wild 1s count.
    """
    exact = sum(1 for d in dice if d == face)
    if face == 1:
        return exact
    wild_ones = sum(1 for d in dice if d == 1)
    return exact + wild_ones
