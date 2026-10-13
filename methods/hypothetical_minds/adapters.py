"""Only ego-observable evidence; no simulated hidden worlds or oracle opponent.

Targets: same-street Leduc response; same-round dice response; current simultaneous
Goofspiel bid. Chance/terminal boundaries are censored, not prediction failures.
This first adaptation is explicitly for the repository's default JJQQKK/2x3/n5.
"""

from copy import deepcopy

from base.liars_dice_core import legal_action_keys
from base.types import Action, Observation
from games.goofspiel import Goofspiel
from games.leduc import LeducHoldem
from games.liars_dice import LiarsDice

GAMES = ("leduc", "liars_dice", "goofspiel")
PRIVATE = {"leduc": ("card",), "liars_dice": ("my_dice",), "goofspiel": ("hand", "my_bids")}
PUBLIC = {
    "leduc": ("community_card", "pot", "street", "my_bet_total", "opponent_bet_total",
              "my_bet_this_round", "opponent_bet_this_round", "round_raises", "to_call"),
    "liars_dice": ("total_dice_remaining", "my_dice_count", "opponent_dice_count", "last_bid", "round"),
    "goofspiel": ("round", "prize", "prize_sequence", "scores", "outcome_history"),
}
HISTORY = {
    "leduc": ("player", "action", "amount", "street"),
    "liars_dice": ("player", "action", "quantity", "face", "round", "result", "bid_quantity",
                   "bid_face", "actual_count", "loser", "revealed_dice"),
    "goofspiel": ("round", "prize", "p0_card", "p1_card", "outcome", "player", "action"),
}


def make_game(name: str, seed: int):
    if name == "leduc":
        return LeducHoldem(seed=seed)
    if name == "liars_dice":
        return LiarsDice(num_dice=2, num_faces=3, seed=seed)
    if name == "goofspiel":
        return Goofspiel(n=5, seed=seed)
    raise ValueError("Unsupported game")


def visible_view(game: str, obs: Observation) -> dict:
    """Own private observations are legitimate; other players' observations are not."""
    if game not in GAMES or obs.player_id not in (0, 1):
        raise ValueError("Unsupported game/player")
    history = [{k: row[k] for k in HISTORY[game] if k in row} for row in obs.history]
    if game == "goofspiel" and any(
        row.get("action") != "round_result" or type(row.get("p0_card")) is not int
        or type(row.get("p1_card")) is not int for row in history
    ):
        raise ValueError("Goofspiel evidence must consist of fully revealed rounds")
    return deepcopy({"player_id": obs.player_id,
        "private_info": {k: obs.private_info[k] for k in PRIVATE[game] if k in obs.private_info},
        "public_info": {k: obs.public_info[k] for k in PUBLIC[game] if k in obs.public_info},
        "history": history})


def token(game: str, action: Action) -> str:
    if action.type == "bid":
        if game == "liars_dice":
            return f"bid_{action.data['quantity']}_{action.data['face']}"
        if game == "goofspiel":
            return f"bid_{action.amount}"
    return action.type


def resolve_action(game: str, value: str, legal: list[Action], player: int) -> Action:
    if type(value) is not str:
        raise ValueError("Action must be a canonical string")
    matches = [a for a in legal if a.player_id == player and token(game, a) == value]
    if len(matches) != 1:
        raise ValueError("Action must uniquely match the current legal set")
    return matches[0]


def scope(game: str, view: dict):
    if game == "leduc":
        return {"preflop": 0, "flop": 1}[view["public_info"]["street"]]
    return view["public_info"]["round"]


def row_scope(game: str, row: dict):
    return row["street"] if game == "leduc" else row["round"]


def opponent_label(game: str, row: dict, player: int) -> str | None:
    if game == "goofspiel":
        return f"bid_{row['p1_card' if player == 0 else 'p0_card']}"
    if row["player"] == player:
        return None
    if game == "liars_dice" and row["action"] == "bid":
        return f"bid_{row['quantity']}_{row['face']}"
    return row["action"]


def own_label(game: str, row: dict, player: int) -> str:
    if game == "goofspiel":
        return f"bid_{row['p0_card' if player == 0 else 'p1_card']}"
    if row["player"] != player:
        raise ValueError("Expected our committed action before opponent response")
    if game == "liars_dice" and row["action"] == "bid":
        return f"bid_{row['quantity']}_{row['face']}"
    return row["action"]


def response_options(game: str, obs: Observation, legal: list[Action]) -> dict[str, list[str]]:
    """Public-only legal response sets for ALL own choices, before selecting one.

    Goofspiel has a single '*' branch: its simultaneous prediction must not be
    conditioned on which unrevealed bid we intend to play. Empty options mean
    no same-scope response, so the model must return null for that branch.
    """
    view = visible_view(game, obs)
    if not legal or any(a.player_id != obs.player_id for a in legal):
        raise ValueError("A nonempty current-player legal set is required")
    if len({token(game, a) for a in legal}) != len(legal):
        raise ValueError("Ambiguous legal action tokens")
    if game == "goofspiel":
        played = {row['p1_card' if obs.player_id == 0 else 'p0_card'] for row in view["history"]}
        return {"*": [f"bid_{n}" for n in range(1, 6) if n not in played]}
    options = {}
    for a in legal:
        if game == "liars_dice":
            labels = [] if a.type == "challenge" else list(legal_action_keys(
                total_dice=view["public_info"]["total_dice_remaining"], num_faces=3,
                last_bid=(a.data["quantity"], a.data["face"])))
        elif a.type in ("fold", "call"):
            labels = []
        elif a.type == "check":
            closes = any(row["street"] == scope(game, view) and row["action"] == "check"
                         for row in view["history"])
            labels = [] if closes else ["check", "bet"]
        else:
            raises = view["public_info"]["round_raises"] + int(a.type == "raise")
            labels = ["fold", "call"] + (["raise"] if raises < 2 else [])
        options[token(game, a)] = labels
    return options