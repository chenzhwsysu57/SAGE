"""Observable event bindings only: no strategy, inferred truth or rule changes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from base.types import Observation

GAMES = ("leduc", "liars_dice", "goofspiel")
TARGETS = {
    "leduc": "the opponent's private card in this hand",
    "liars_dice": "the opponent's dice in the round of the observed action",
    "goofspiel": "the cards available to the opponent at the start of the simultaneous bidding round",
}
BINDINGS = {
    "leduc": "Private cards remain fixed within a hand. A fold does not reveal the opponent's card.",
    "liars_dice": (
        "Dice belong to a particular round. Challenge reveals that round's dice. "
        "If the game continues, both players reroll their remaining dice for a new round; "
        "old revealed dice or old hypotheses are not observations of the new dice."
    ),
    "goofspiel": (
        "Both players start with cards 1 through 5. Completed-round bids are public and "
        "determine which cards have been used. The observation includes the prize sequence. "
        "Current bids are not revealed until both players have chosen. Model the held-card "
        "information as in EMO; do not substitute a model of personality or bidding intent."
    ),
}


def public_view(obs: Observation) -> dict[str, Any]:
    """Never pass private_info/state_text/legal actions of the current other actor."""
    return deepcopy({"observer": obs.player_id, "public_info": obs.public_info,
                     "history": obs.history})


def own_view(obs: Observation) -> dict[str, Any]:
    return {**public_view(obs), "private_info": deepcopy(obs.private_info)}


def scope(game: str, obs: Observation) -> int:
    return int(obs.public_info["round"]) if game != "leduc" else 1


def opponent_events(game: str, player: int, before: Observation,
                    after: Observation) -> list[dict[str, Any]]:
    """Extract only events newly visible to this player, never pending bids.

    Called after every engine transition, even if the acting player is the
    learner: P1's own bid reveals P0's bid in Goofspiel. Each binding expects
    exactly one game step, so no future public observation is used for an older
    unprocessed action.
    """
    if game not in GAMES or player not in (0, 1):
        raise ValueError("Unsupported game/player")
    if before.player_id != player or after.player_id != player:
        raise ValueError("Expected learner observations")
    start = len(before.history)
    if after.history[:start] != before.history or len(after.history) - start not in (0, 1):
        raise ValueError("Expected one consecutive engine transition")
    result = []
    for row in after.history[start:]:
        if game == "goofspiel":
            if row.get("action") != "round_result":
                raise ValueError("Unexpected Goofspiel public event")
            action = {"player": 1-player, "action": "bid",
                      "card": row[f"p{1-player}_card"], "round": row["round"]}
        elif row.get("player") == 1-player:
            action = deepcopy(row)
        else:
            continue
        result.append({"event_id": start + 1, "opponent": 1-player,
                       "scope": int(row.get("round", 1)), "action": action,
                       "before": public_view(before), "after": public_view(after)})
    return result