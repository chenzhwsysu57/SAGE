"""Observation-only, seeded hypothetical worlds; never accept a live game/state.

Leduc/dice use a uniform prior, NOT an inferred posterior. Search in sampled
worlds can suffer strategy fusion; this is an explicit imperfect-information
adaptation, not an equilibrium solver. Each branch owns its simulator RNG.
Goofspiel uses paired simultaneous bids, regardless of the engine's seat order.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

from base.types import Action, Observation
from games.goofspiel import Goofspiel, GoofspielState
from games.leduc import LEDUC_DECK, LeducHoldem, LeducState
from games.liars_dice import LiarsDice, LiarsDiceState

GAMES = ("leduc", "liars_dice", "goofspiel")


def make_game(name: str, seed: int):
    if name == "leduc":
        return LeducHoldem(seed=seed)
    if name == "liars_dice":
        return LiarsDice(num_dice=2, num_faces=3, seed=seed)
    if name == "goofspiel":
        return Goofspiel(n=5, seed=seed)
    raise ValueError("Unsupported ReTA game")


def visible(obs: Observation) -> dict:
    """Only structured player-visible inputs; no free-form/full-state fallback."""
    return deepcopy({"player_id": obs.player_id, "private_info": obs.private_info,
                     "public_info": obs.public_info, "history": obs.history})


def decision_seed(obs: Observation, seed: int) -> int:
    payload = json.dumps(visible(obs), sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(f"{seed}:{payload}".encode()).digest()[:8], "big")


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
        # Goofspiel state remains PRE-bid until both bids exist. The opponent's
        # view therefore cannot reveal our candidate through hand removal either.
        result = visible(self.game.get_observation(self.state, player))
        if self.pending is not None and player == self.hero:
            result["own_unrevealed_bid"] = self.pending.model_dump()
        return result

    def advance(self, action: Action) -> "World":
        if action not in self.legal():
            raise ValueError("Noncanonical hypothetical action")
        child = deepcopy(self)  # Also clones RNG, including future dice rerolls.
        if self.name == "goofspiel":
            if self.pending is None:
                child.pending = action
            else:
                bids = {self.hero: self.pending, 1 - self.hero: action}
                state = child.game.step(child.state, bids[0])
                child.state = child.game.step(state, bids[1])
                child.pending = None
        else:
            child.state = child.game.step(child.state, action)
        return child

    def terminal_net(self) -> float | None:
        return self.game.get_net_profit(self.state, self.hero) if self.terminal else None


def sample_world(name: str, obs: Observation, seed: int) -> World:
    """Fixed repository configurations: JJQQKK, dice 2x3 wild1, Goofspiel n5.

    Sampling is conditioned only on own private information and public history.
    Hidden hypotheses are independent of the real environment seed/RNG/state.
    """
    if obs.player_id not in (0, 1):
        raise ValueError("ReTA supports exactly two seats")
    hero, opp = obs.player_id, 1 - obs.player_id
    pub, private = obs.public_info, obs.private_info
    history = deepcopy(obs.history)
    rng = random.Random(seed)
    game = make_game(name, seed)
    if name == "leduc":
        cards = {str(c): c for c in LEDUC_DECK}
        own = cards[private["card"]]
        community = cards[pub["community_card"]] if pub["community_card"] else None
        remaining = [c for c in LEDUC_DECK if c != own and c != community]
        rng.shuffle(remaining)
        other = remaining.pop()
        board = community or remaining.pop()
        hands = {hero: own, opp: other}
        state = LeducState(
            deck=[hands[0], hands[1], board, *remaining], private_cards=hands,
            community_card=community, pot=pub["pot"],
            street=0 if pub["street"] == "preflop" else 1,
            phase=pub["street"], acting_player=hero,
            total_bets={hero: pub["my_bet_total"], opp: pub["opponent_bet_total"]},
            round_bets={hero: pub["my_bet_this_round"], opp: pub["opponent_bet_this_round"]},
            round_raises=pub["round_raises"], history=history, rewards=None,
        )
    elif name == "liars_dice":
        counts = {hero: pub["my_dice_count"], opp: pub["opponent_dice_count"]}
        own = tuple(private["my_dice"])
        if len(own) != counts[hero] or any(d not in (1, 2, 3) for d in own):
            raise ValueError("Invalid 2x3 own dice")
        if any(not 1 <= count <= 2 for count in counts.values()):
            raise ValueError("ReTA dice adapter requires 2x3 configuration")
        last = pub["last_bid"]
        state = LiarsDiceState(
            dice={hero: own, opp: tuple(rng.randint(1, 3) for _ in range(counts[opp]))},
            dice_counts=counts, current_player=hero,
            last_bid=(last["quantity"], last["face"]) if last else None,
            phase="bidding", history=history, round_num=pub["round"], num_faces=3,
        )
    else:
        # Reconstruct the simultaneous round's PRE-COMMIT information set for
        # BOTH seats. At real P1 time the unknown P0 bid has already been stored,
        # but each possible such bid was drawn from this pre-round hand. Search
        # evaluates candidate pairs from that set; it does NOT give P0 a second
        # action after its hidden bid or discard the hidden card twice. Resolution
        # below always makes exactly one canonical engine step for each seat.
        sequence = tuple(pub["prize_sequence"])
        if sorted(sequence) != [1, 2, 3, 4, 5]:
            raise ValueError("ReTA Goofspiel adapter requires n5")
        hands = {p: set(range(1, 6)) for p in (0, 1)}
        for event in history:
            hands[0].remove(event["p0_card"])
            hands[1].remove(event["p1_card"])
        if hands[hero] != set(private["hand"]):
            raise ValueError("Own hand disagrees with completed public rounds")
        scores = pub["scores"]
        state = GoofspielState(
            n=5, round=pub["round"], prize=pub["prize"], prize_sequence=sequence,
            hands=hands, scores={p: scores[p] if p in scores else scores[str(p)] for p in (0, 1)},
            outcome_history=list(pub["outcome_history"]), pending_bid=None,
            phase="bid_p0", current_player=0, history=history,
        )
    world = World(name, game, state, hero)
    if world.view(hero) != visible(obs):
        raise ValueError("Hypothetical reconstruction does not preserve observation")
    return world


def rules(name: str) -> str:
    text = make_game(name, 0).get_rules_text()
    if name == "leduc":
        # Correct the engine's legacy prose only inside this new adapter.
        text = text.replace("Max raises = 2 per player per round", "Max raises = 2 total per round")
    if name == "goofspiel":
        text += "\nThe complete prize sequence is public in this repository variant."
    return text