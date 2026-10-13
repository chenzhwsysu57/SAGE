"""Observation-only simulators for the repository's three two-player variants.

Hidden cards/dice use a uniform prior *ignoring action likelihood*. This is not
posterior inference, an equilibrium solver, or exact Strategist Appendix F.
Only sample_world accepts a root: it takes an Observation, never a live state.
World.state is an internal hypothetical engine state, not a heuristic input.

Leduc has two TOTAL raises per street: each player contributes at most
1 + 3*2 + 3*4 = 19 chips. Thus |net| <= 19, not 32; /32 is a conservative
requested normalization. Dice rewards are +/-1; Goofspiel score difference is
bounded by 1+2+3+4+5=15 (ties split, not carried over).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Any

from base.types import Action, Observation
from games.goofspiel import Goofspiel, GoofspielState
from games.leduc import Card, LEDUC_DECK, LeducHoldem, LeducState
from games.liars_dice import LiarsDice, LiarsDiceState

GAMES = ("leduc", "liars_dice", "goofspiel")
PAYOFF_SCALES = {"leduc": 32.0, "liars_dice": 1.0, "goofspiel": 15.0}
PAYOFF_BOUNDS = {"leduc": 19.0, "liars_dice": 1.0, "goofspiel": 15.0}
# Check/bet/raise/raise/call per Leduc street; dice rounds have at most
# (4*3+1)+(3*3+1)+(2*3+1)=30 events; Goofspiel has five round results.
MAX_HISTORY = {"leduc": 10, "liars_dice": 30, "goofspiel": 5}


def _seed(seed: int) -> None:
    if type(seed) is not int or not -(2**63) <= seed < 2**128:
        raise ValueError("seed must be an integer in [-2**63, 2**128)")


def make_game(game: str, seed: int):
    _seed(seed)
    if game == "leduc":
        return LeducHoldem(seed=seed)
    if game == "liars_dice":
        return LiarsDice(num_dice=2, num_faces=3, seed=seed)
    if game == "goofspiel":
        return Goofspiel(n=5, seed=seed)
    raise ValueError("Unsupported Strategist game")


def rules(game: str) -> str:
    text = make_game(game, 0).get_rules_text()
    if game == "leduc":
        text = text.replace("Max raises = 2 per player per round",
                            "Max raises = 2 total per round")
    if game == "goofspiel":
        text += "\nThe complete prize sequence is public in this variant."
    return text


def _json(value: Any) -> Any:
    """Detach and normalize at the safe boundary; reject, never truncate.

    Limits match the heuristic's JSON input budget. In particular, a malformed
    or oversized history must not silently become a different information set.
    """
    count = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal count
        count += 1
        if count > 4096 or depth > 32:
            raise ValueError("Observation exceeds safe input budget")
        if isinstance(item, Enum):
            return visit(item.value, depth + 1)
        if isinstance(item, Card):
            return str(item)
        if item is None or type(item) is bool:
            return item
        if type(item) in (int, float):
            if not -1e9 <= item <= 1e9 or not math.isfinite(item):
                raise ValueError("Nonfinite or oversized observation number")
            return item
        if type(item) is str:
            if len(item) > 512:
                raise ValueError("Oversized observation string")
            return item
        if type(item) not in (dict, list, tuple) or len(item) > 128:
            raise ValueError("Unsupported or oversized observation container")
        if type(item) is dict:
            result = {}
            for key, child in item.items():
                if type(key) not in (str, int):
                    raise ValueError("Observation keys must be strings or integers")
                key = visit(str(key), depth + 1)
                if key in result:
                    raise ValueError("Colliding normalized observation keys")
                result[key] = visit(child, depth + 1)
            return result
        return [visit(child, depth + 1) for child in item]

    return visit(value, 0)


def visible(obs: Observation) -> dict:
    """Structured information only: never read state_text or full_history_text."""
    if not isinstance(obs, Observation) or type(obs.player_id) is not int or obs.player_id not in (0, 1):
        raise ValueError("Expected a two-player Observation")
    return _json({"player_id": obs.player_id, "private_info": obs.private_info,
                  "public_info": obs.public_info, "history": obs.history})


def _history(game: str, history: list) -> None:
    if len(history) > MAX_HISTORY[game]:
        raise ValueError("History exceeds this game's maximum; not truncated")
    for event in history:
        if type(event) is not dict:
            raise ValueError("History events must be objects")
        if game == "leduc":
            keys = {"player", "action", "amount", "street"}
        elif game == "goofspiel":
            keys = {"round", "prize", "p0_card", "p1_card", "outcome", "player", "action", "text"}
        elif event.get("action") == "bid":
            keys = {"player", "action", "quantity", "face", "round"}
        else:
            keys = {"player", "action", "round", "result", "bid_quantity", "bid_face",
                    "actual_count", "loser", "revealed_dice"}
            revealed = event.get("revealed_dice")
            if (event.get("action") != "challenge" or type(revealed) is not dict
                    or set(revealed) != {"0", "1"}
                    or any(type(ds) is not list or not 1 <= len(ds) <= 2
                           or any(type(d) is not int or d not in (1, 2, 3) for d in ds)
                           for ds in revealed.values())):
                raise ValueError("Invalid public dice reveal")
        if set(event) != keys:
            raise ValueError("Unexpected history fields (possible hidden information)")
        if any(type(v) not in (str, int, float, bool, type(None))
               for k, v in event.items() if k != "revealed_dice"):
            raise ValueError("Unexpected nested history field")


@dataclass
class World:
    name: str
    game: Any
    state: Any
    hero: int
    pending: Action | None = None

    @property
    def terminal(self) -> bool:
        return self.game.is_terminal(self.state)

    @property
    def actor(self) -> int:
        if self.terminal:
            return -1
        if self.name == "goofspiel":
            return self.hero if self.pending is None else 1 - self.hero
        return self.game.get_current_player(self.state)

    def legal(self) -> list[Action]:
        if self.terminal:
            return []
        if self.name == "goofspiel":
            return [Action(player_id=self.actor, type="bid", amount=c)
                    for c in sorted(self.state.hands[self.actor])]
        return self.game.get_legal_actions(self.state)

    def view(self, player: int) -> dict:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("Invalid viewing seat")
        # Crucially, the engine state is still PRE-COMMIT for BOTH seats.
        # Neither a pending bid nor the card removed by it enters either view.
        return visible(self.game.get_observation(self.state, player))

    def features(self, player: int) -> dict:
        features = {"game": self.name, **self.view(player)}
        _history(self.name, features["history"])
        return _json(features)

    def advance(self, action: Action) -> World:
        if action not in self.legal():
            raise ValueError("Expected a canonical legal hypothetical action")
        child = deepcopy(self)  # includes independent future chance RNG state
        if self.name == "goofspiel":
            if self.pending is None:
                child.pending = action.model_copy(deep=True)
            else:
                bids = {self.hero: self.pending, 1 - self.hero: action}
                child.state = child.game.step(child.state, bids[0])
                child.state = child.game.step(child.state, bids[1])
                child.pending = None
        else:
            child.state = child.game.step(child.state, action)
        return child

    def terminal_values(self) -> tuple[float, float]:
        if not self.terminal:
            raise ValueError("Terminal values requested for a nonterminal world")
        if self.name == "leduc":
            net = [self.game.get_net_profit(self.state, p) for p in (0, 1)]
        else:
            rewards = self.game.get_rewards(self.state)
            net = [rewards[p] for p in (0, 1)]
        if (any(not math.isfinite(v) or abs(v) > PAYOFF_BOUNDS[self.name] for v in net)
                or abs(sum(net)) > 1e-8):
            raise ValueError("Engine payoff violates the fixed game's bound/zero sum")
        return (net[0] / PAYOFF_SCALES[self.name], net[1] / PAYOFF_SCALES[self.name])


def sample_world(game: str, obs: Observation, seed: int) -> World:
    """Independently sample hidden information and future chance from a root.

    Goofspiel P1 reconstruction discards the real engine's unknown P0 commitment:
    both simulated bids are chosen from their public pre-round hands and then
    resolved exactly once. The entire future prize sequence is already public.
    """
    if game not in GAMES:
        raise ValueError("Unsupported Strategist game")
    _seed(seed)
    data = visible(obs)
    _history(game, data["history"])
    hero, opp = obs.player_id, 1 - obs.player_id
    pub, private, history = data["public_info"], data["private_info"], data["history"]
    rng = random.Random(seed)
    # Domain-separated stream from the hidden draws; no reset() consumes it.
    engine = make_game(game, rng.getrandbits(128))
    try:
        if game == "leduc":
            cards = {str(c): c for c in LEDUC_DECK}
            own = cards[private["card"]]
            community = cards[pub["community_card"]] if pub["community_card"] else None
            if community == own:
                raise ValueError("Own card equals public board")
            remaining = [c for c in LEDUC_DECK if c != own and c != community]
            rng.shuffle(remaining)
            hands = {hero: own, opp: remaining.pop()}
            board = community if community is not None else remaining.pop()
            # Replay public actions, not a hidden-state dump. This also validates
            # amounts, actor, raise limits and phase progression against engine.
            state = LeducState(
                deck=[hands[0], hands[1], board, *remaining], private_cards=hands,
                community_card=None, pot=2, street=0, phase="preflop", acting_player=0,
                total_bets={0: 1, 1: 1}, round_bets={0: 0, 1: 0}, round_raises=0,
                history=[], rewards=None,
            )
            for event in history:
                state = engine.step(state, Action(player_id=event["player"],
                                                  type=event["action"], amount=event["amount"]))
        elif game == "liars_dice":
            counts = {hero: pub["my_dice_count"], opp: pub["opponent_dice_count"]}
            own = tuple(private["my_dice"])
            if (any(type(n) is not int or not 1 <= n <= 2 for n in counts.values())
                    or len(own) != counts[hero]
                    or any(type(d) is not int or d not in (1, 2, 3) for d in own)
                    or type(pub["round"]) is not int or pub["round"] != 5 - sum(counts.values())):
                raise ValueError("Invalid fixed 2x3 dice observation")
            last = pub["last_bid"]
            if last is not None and (type(last) is not dict or set(last) != {"quantity", "face"}
                    or type(last["quantity"]) is not int or not 1 <= last["quantity"] <= sum(counts.values())
                    or type(last["face"]) is not int or last["face"] not in (1, 2, 3)):
                raise ValueError("Invalid current bid")
            state = LiarsDiceState(
                dice={hero: own, opp: tuple(rng.randint(1, 3) for _ in range(counts[opp]))},
                dice_counts=counts, current_player=hero,
                last_bid=(last["quantity"], last["face"]) if last else None,
                phase="bidding", history=history, round_num=pub["round"], num_faces=3,
            )
        else:
            sequence = tuple(pub["prize_sequence"])
            if any(type(c) is not int for c in sequence) or sorted(sequence) != [1, 2, 3, 4, 5]:
                raise ValueError("Expected public n5 prize permutation")
            state = GoofspielState(
                n=5, round=1, prize=sequence[0], prize_sequence=sequence,
                hands={0: set(range(1, 6)), 1: set(range(1, 6))},
                scores={0: 0.0, 1: 0.0}, outcome_history=[], pending_bid=None,
                phase="bid_p0", current_player=0,
            )
            for event in history:
                state = engine.step(state, Action(player_id=0, type="bid", amount=event["p0_card"]))
                state = engine.step(state, Action(player_id=1, type="bid", amount=event["p1_card"]))
        world = World(game, engine, state, hero)
        if world.terminal or world.actor != hero or world.view(hero) != data:
            raise ValueError("Root reconstruction does not preserve a decision observation")
        expected = [f"bid_{a.amount}" if game == "goofspiel" else a.type for a in world.legal()]
        if sorted(obs.legal_actions) != sorted(expected):
            raise ValueError("Input legal actions disagree with canonical engine actions")
        world.features(hero)  # validate the complete safe heuristic boundary
        return world
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed observation for fixed game configuration") from exc