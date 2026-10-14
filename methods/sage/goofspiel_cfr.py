"""External-sampling MCCFR/MCCFR+ for II-Goofspiel.

No chance nodes — deterministic prize sequence, no dice.
Simultaneous bidding is sequentialised in the training walk
(P0 → P1, P1 does NOT see P0's bid).

Exploitability uses a round-level joint traversal that correctly
handles the imperfect-information constraint at P1's decision nodes.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import random
from tqdm import tqdm

from base.types import Action
from games.goofspiel import load_prize_pool


# ── helpers ─────────────────────────────────────────────────────────────

def hand_to_mask(hand: set[int]) -> int:
    m = 0
    for c in hand:
        m |= 1 << c
    return m


def mask_to_list(mask: int, n: int) -> list[int]:
    return [c for c in range(1, n + 1) if mask & (1 << c)]


def regret_matching_strategy(regrets: list[float]) -> list[float]:
    pos = [max(r, 0.0) for r in regrets]
    total = sum(pos)
    if total < 1e-12:
        return [1.0 / len(pos)] * len(pos)
    return [p / total for p in pos]


def sample_index(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


# ── action key ──────────────────────────────────────────────────────────

ActionKey = int  # card value (1..N)


# ── infoset key ─────────────────────────────────────────────────────────

# Infoset = (player, hand_mask, outcome_history, my_actions, opp_actions, prize_sequence)
# my_actions includes the player's own actions from all rounds (current included).
# opp_actions includes only COMPLETED-round opponent actions — the current-round
# opponent action is unknown under simultaneous bidding.
InfosetKey = tuple


def infoset_key_from_hand(player: int, hand_mask: int,
                           outcome_history: tuple[int, ...],
                           my_actions: tuple[int, ...],
                           opp_actions: tuple[int, ...] = (),
                           prize_sequence: tuple[int, ...] = ()) -> InfosetKey:
    return (player, hand_mask, outcome_history, my_actions, opp_actions, prize_sequence)


def infoset_key_from_state(state: "GoofspielCFRState", player: int) -> InfosetKey:
    """Construct infoset key with only completed-round opponent actions."""
    completed = len(state.outcome_history)
    return (player, state.hands[player], state.outcome_history,
            state.action_histories[player],
            state.action_histories[1 - player][:completed],
            state.prize_sequence)


# ── CFR state ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class GoofspielCFRState:
    n: int
    round: int
    prize_sequence: tuple[int, ...]            # random permutation of n..1
    hands: tuple[int, int]                     # bitmask per player
    action_histories: tuple[tuple[int, ...], tuple[int, ...]]  # cards played so far per player
    outcome_history: tuple[int, ...]           # +1 P0 win, -1 P1 win, 0 tie
    phase: str                                 # "bid_p0" | "bid_p1" | "terminal"
    current_player: int
    pending_bid: int | None
    scores: tuple[float, float]
    rewards: tuple[float, float] | None = None


# ── Trainer ─────────────────────────────────────────────────────────────

class GoofspielCFRTrainer:
    """CFR for II-Goofspiel (full CFR or external-sampling MCCFR)."""

    def __init__(
        self,
        n: int = 5,
        iterations: int = 50_000,
        seed: int = 0,
        cfr_plus: bool = True,
        full_opponent: bool = False,
        cfr_variant: str = "mccfr",
    ):
        self.n = n
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.cfr_plus = cfr_plus
        self.full_opponent = full_opponent
        self.cfr_variant = cfr_variant

        self._prize_pool = load_prize_pool(n)
        self.regret_sum: dict[InfosetKey, dict[ActionKey, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.strategy_sum: dict[InfosetKey, dict[ActionKey, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.strategy: dict[InfosetKey, dict[ActionKey, float]] = {}
        self._num_infosets = 0

    # ── internal state helpers ──────────────────────────────────────────

    def _initial_state(self) -> GoofspielCFRState:
        mask = hand_to_mask(set(range(1, self.n + 1)))
        prize_seq = self.rng.choice(self._prize_pool)
        return GoofspielCFRState(
            n=self.n,
            round=1,
            prize_sequence=prize_seq,
            hands=(mask, mask),
            action_histories=((), ()),
            outcome_history=(),
            phase="bid_p0",
            current_player=0,
            pending_bid=None,
            scores=(0.0, 0.0),
        )

    def _legal_keys(self, state: GoofspielCFRState) -> tuple[ActionKey, ...]:
        return tuple(mask_to_list(state.hands[state.current_player], self.n))

    def _step(self, state: GoofspielCFRState, key: ActionKey) -> GoofspielCFRState:
        """Deterministic transition for sequentialised training walk."""
        card = key
        player = state.current_player
        hand = state.hands[player]
        assert hand & (1 << card), f"card {card} not in player {player}'s hand"
        new_hand = hand & ~(1 << card)

        if state.phase == "bid_p0":
            return GoofspielCFRState(
                n=self.n,
                round=state.round,
                prize_sequence=state.prize_sequence,
                hands=(new_hand, state.hands[1]),
                action_histories=(state.action_histories[0] + (card,),
                                  state.action_histories[1]),
                outcome_history=state.outcome_history,
                phase="bid_p1",
                current_player=1,
                pending_bid=card,
                scores=state.scores,
            )

        # bid_p1 — resolve
        p0_card = state.pending_bid
        p1_card = card
        assert p0_card is not None
        prize = state.prize_sequence[state.round - 1]

        if p0_card > p1_card:
            outcome = 1
            s0 = state.scores[0] + prize
            s1 = state.scores[1]
        elif p0_card < p1_card:
            outcome = -1
            s0 = state.scores[0]
            s1 = state.scores[1] + prize
        else:
            outcome = 0
            s0 = state.scores[0] + prize / 2.0
            s1 = state.scores[1] + prize / 2.0

        new_outcomes = state.outcome_history + (outcome,)

        if state.round >= self.n:
            return GoofspielCFRState(
                n=self.n,
                round=state.round,
                prize_sequence=state.prize_sequence,
                hands=(state.hands[0], new_hand),
                action_histories=(state.action_histories[0],
                                  state.action_histories[1] + (card,)),
                outcome_history=new_outcomes,
                phase="terminal",
                current_player=-1,
                pending_bid=None,
                scores=(s0, s1),
                rewards=(s0 - s1, s1 - s0),
            )

        return GoofspielCFRState(
            n=self.n,
            round=state.round + 1,
            prize_sequence=state.prize_sequence,
            hands=(state.hands[0], new_hand),
            action_histories=(state.action_histories[0],
                              state.action_histories[1] + (card,)),
            outcome_history=new_outcomes,
            phase="bid_p0",
            current_player=0,
            pending_bid=None,
            scores=(s0, s1),
        )

    # ── joint round resolution (used by exploitability only) ────────────

    def _resolve_round(self, state: GoofspielCFRState,
                        c0: int, c1: int) -> GoofspielCFRState:
        """Resolve round from start-of-round state (bid_p0) with (c0, c1)."""
        assert state.phase == "bid_p0", f"expected bid_p0 got {state.phase}"
        new_h0 = state.hands[0] & ~(1 << c0)
        new_h1 = state.hands[1] & ~(1 << c1)
        prize = state.prize_sequence[state.round - 1]

        if c0 > c1:
            outcome = 1
            s0 = state.scores[0] + prize
            s1 = state.scores[1]
        elif c0 < c1:
            outcome = -1
            s0 = state.scores[0]
            s1 = state.scores[1] + prize
        else:
            outcome = 0
            s0 = state.scores[0] + prize / 2.0
            s1 = state.scores[1] + prize / 2.0

        new_outcomes = state.outcome_history + (outcome,)

        if state.round >= self.n:
            return GoofspielCFRState(
                n=state.n,
                round=state.round,
                prize_sequence=state.prize_sequence,
                hands=(new_h0, new_h1),
                action_histories=(state.action_histories[0] + (c0,),
                                  state.action_histories[1] + (c1,)),
                outcome_history=new_outcomes,
                phase="terminal",
                current_player=-1,
                pending_bid=None,
                scores=(s0, s1),
                rewards=(s0 - s1, s1 - s0),
            )

        return GoofspielCFRState(
            n=state.n,
            round=state.round + 1,
            prize_sequence=state.prize_sequence,
            hands=(new_h0, new_h1),
            action_histories=(state.action_histories[0] + (c0,),
                              state.action_histories[1] + (c1,)),
            outcome_history=new_outcomes,
            phase="bid_p0",
            current_player=0,
            pending_bid=None,
            scores=(s0, s1),
        )

    # ── MCCFR walk (simultaneous-move joint round) ──────────────────────

    def _walk(self, state: GoofspielCFRState,
              p0_reach: float, p1_reach: float,
              traverser: int) -> float:
        """External-sampling MCCFR matching OpenSpiel's algorithm.

        Uses the sequential game tree (P0→P1→P0→P1→…) via _step(), exactly
        matching OpenSpiel's turn_based_simultaneous_game structure.

        Regret updates and strategy accumulation are done WITHOUT explicit
        reach-probability weighting (matching OpenSpiel's SIMPLE averaging
        mode).  The reach-probability bias cancels out in regret matching
        and per-infoset strategy normalisation.

        Parameters p0_reach and p1_reach are accepted only for compatibility
        with existing callers; they are NOT used in this implementation.
        """
        if state.phase == "terminal":
            assert state.rewards is not None
            return state.rewards[traverser]

        cur_player = state.current_player
        ik = infoset_key_from_state(state, cur_player)
        cards = mask_to_list(state.hands[cur_player], self.n)

        r = self.regret_sum[ik]
        for c in cards:
            _ = r[c]
        s = regret_matching_strategy([r[c] for c in cards])

        if cur_player != traverser:
            # Opponent node: sample one action, return sampled value
            idx = sample_index(s, self.rng)
            ns = self._step(state, cards[idx])
            val = self._walk(ns, 0.0, 0.0, traverser)

            # SIMPLE strategy accumulation at opponent's info set
            for i, c in enumerate(cards):
                self.strategy_sum[ik][c] += s[i]

            return val

        # Traverser node: explore all actions, compute expected value
        child_vals: list[float] = []
        for c in cards:
            ns = self._step(state, c)
            child_vals.append(self._walk(ns, 0.0, 0.0, traverser))

        val = sum(s[i] * child_vals[i] for i in range(len(cards)))

        # Regret update (unweighted, matching OpenSpiel SIMPLE mode)
        for i, c in enumerate(cards):
            delta = child_vals[i] - val
            if self.cfr_plus:
                r[c] = max(0.0, r[c] + delta)
            else:
                r[c] += delta

        return val

    # ── full CFR walk (sequentialised, no sampling) ────────────────────

    def _walk_cfr(self, state: GoofspielCFRState,
                   p0_reach: float, p1_reach: float) -> tuple[float, float]:
        """Full CFR with sequentialised simultaneous-move handling.

        P0 acts first (traversing actions at I_0), then P1 acts without
        knowing P0's action (at I_1).  For each P0 action c0, the value
        is weighted by P1's strategy over all c1 at I_1.  This is
        equivalent to the joint formulation but uses the standard
        sequential update ordering that CFR guarantees convergence for.
        """
        if state.phase == "terminal":
            assert state.rewards is not None
            return state.rewards

        assert state.phase == "bid_p0"
        p0_cards = mask_to_list(state.hands[0], self.n)
        p1_cards = mask_to_list(state.hands[1], self.n)

        ik0 = infoset_key_from_hand(0, state.hands[0], state.outcome_history,
                                     state.action_histories[0], state.action_histories[1],
                                     state.prize_sequence)
        ik1 = infoset_key_from_hand(1, state.hands[1], state.outcome_history,
                                     state.action_histories[1], state.action_histories[0],
                                     state.prize_sequence)

        r0 = self.regret_sum[ik0]
        for c in p0_cards: _ = r0[c]

        s0 = regret_matching_strategy([r0[c] for c in p0_cards])

        # For each P0 action, compute exact EV against P1's strategy at I_1
        child_vals_act: list[float] = []
        for i0, c0 in enumerate(p0_cards):
            # P1's state (P0 has played c0 but P1 doesn't know it)
            p1_state = GoofspielCFRState(
                n=state.n,
                round=state.round,
                prize_sequence=state.prize_sequence,
                hands=(state.hands[0] & ~(1 << c0), state.hands[1]),
                action_histories=(state.action_histories[0] + (c0,),
                                  state.action_histories[1]),
                outcome_history=state.outcome_history,
                phase="bid_p1",
                current_player=1,
                pending_bid=c0,
                scores=state.scores,
            )
            ev_c0 = self._eval_p1(p1_state, ik1, p1_cards,
                                  p0_reach * s0[i0], p1_reach)
            child_vals_act.append(ev_c0)

        nv0 = sum(s0[i] * child_vals_act[i] for i in range(len(p0_cards)))

        # P0 regret update
        for i, c in enumerate(p0_cards):
            delta = p1_reach * (child_vals_act[i] - nv0)
            if self.cfr_plus:
                r0[c] = max(0.0, r0[c] + delta)
            else:
                r0[c] += delta

        # P0 strategy accumulation
        for idx, c in enumerate(p0_cards):
            self.strategy_sum[ik0][c] += p0_reach * s0[idx]

        return (nv0, -nv0)  # P1 value is negative of P0 (zero-sum)

    def _eval_p1(self, state: GoofspielCFRState,
                  ik1: InfosetKey, p1_cards: list[int],
                  p0_reach: float, p1_reach: float) -> float:
        """Enumerate P1 actions at info set I_1 (P1 doesn't see c0),
        resolve the round inline, recurse via sequential _walk_cfr."""
        r1 = self.regret_sum[ik1]
        for c in p1_cards: _ = r1[c]
        s1 = regret_matching_strategy([r1[c] for c in p1_cards])

        p0_card = state.pending_bid
        assert p0_card is not None
        prize = state.prize_sequence[state.round - 1]

        child_vals: list[float] = []
        for i1, c1 in enumerate(p1_cards):
            new_h0 = state.hands[0] & ~(1 << p0_card)
            new_h1 = state.hands[1] & ~(1 << c1)

            if p0_card > c1:
                outcome = 1
                s0_new = state.scores[0] + prize
                s1_new = state.scores[1]
            elif p0_card < c1:
                outcome = -1
                s0_new = state.scores[0]
                s1_new = state.scores[1] + prize
            else:
                outcome = 0
                s0_new = state.scores[0] + prize / 2.0
                s1_new = state.scores[1] + prize / 2.0

            new_outcomes = state.outcome_history + (outcome,)

            if state.round >= state.n:
                ns = GoofspielCFRState(
                    n=state.n, round=state.round,
                    prize_sequence=state.prize_sequence,
                    hands=(new_h0, new_h1),
                    action_histories=(state.action_histories[0],
                                      state.action_histories[1] + (c1,)),
                    outcome_history=new_outcomes,
                    phase="terminal", current_player=-1,
                    pending_bid=None,
                    scores=(s0_new, s1_new),
                    rewards=(s0_new - s1_new, s1_new - s0_new),
                )
            else:
                ns = GoofspielCFRState(
                    n=state.n, round=state.round + 1,
                    prize_sequence=state.prize_sequence,
                    hands=(new_h0, new_h1),
                    action_histories=(state.action_histories[0],
                                      state.action_histories[1] + (c1,)),
                    outcome_history=new_outcomes,
                    phase="bid_p0", current_player=0,
                    pending_bid=None,
                    scores=(s0_new, s1_new),
                )

            cp0 = p0_reach
            cp1 = p1_reach * s1[i1]
            v0, _ = self._walk_cfr(ns, cp0, cp1)
            child_vals.append(v0)

        nv1 = sum(s1[i] * child_vals[i] for i in range(len(p1_cards)))

        # P1 regret update — negated because child_vals[i]/nv1 are P0's EV,
        # and P1 minimizes P0's EV (zero-sum)
        for i, c in enumerate(p1_cards):
            delta = p0_reach * (nv1 - child_vals[i])
            if self.cfr_plus:
                r1[c] = max(0.0, r1[c] + delta)
            else:
                r1[c] += delta

        # P1 strategy accumulation
        for idx, c in enumerate(p1_cards):
            self.strategy_sum[ik1][c] += p1_reach * s1[idx]

        return nv1

    # ── train ───────────────────────────────────────────────────────────

    def train(self, eval_every: int = 1000) -> None:
        started = time.perf_counter()

        is_full = self.cfr_variant == "full"
        mode = "CFRfull" if is_full else ("MCCFR+" if self.cfr_plus else "MCCFR")
        with tqdm(total=self.iterations, unit="iter", desc=mode) as pbar:
            for i in range(self.iterations):
                state = self._initial_state()
                if is_full:
                    self._walk_cfr(state, 1.0, 1.0)
                else:
                    traverser = i % 2
                    self._walk(state, 1.0, 1.0, traverser)

                elapsed = time.perf_counter() - started
                it_per_sec = (i + 1) / max(elapsed, 1e-9)
                postfix: dict[str, Any] = {
                    "infosets": len(self.strategy_sum),
                    "iter/s": f"{it_per_sec:.1f}",
                }

                if eval_every > 0 and (i + 1) % eval_every == 0:
                    exp_avg = self.compute_exploitability()
                    exp_cur = self.compute_exploitability_current()
                    postfix["exp_avg"] = f"{exp_avg:.4f}"
                    postfix["exp_cur"] = f"{exp_cur:.4f}"
                    pbar.set_postfix(postfix)
                    pbar.update(eval_every if i + 1 >= eval_every else i + 1)
                    print(f"\nStep {i+1}: exp_avg={exp_avg:.4f}  exp_cur={exp_cur:.4f}")
                else:
                    pbar.set_postfix(postfix)
                    pbar.update(1)

        self.strategy = {}
        for infoset, sums in self.strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                self.strategy[infoset] = {k: v / total for k, v in sums.items()}
        self._num_infosets = len(self.strategy)

        final_exp_avg = self.compute_exploitability()
        final_exp_cur = self.compute_exploitability_current()
        print(f"\n{mode} done. {self._num_infosets} infosets, "
              f"final exp_avg={final_exp_avg:.4f}  exp_cur={final_exp_cur:.4f}")

    # ── exploitability (corrected II-aware joint round traversal) ───────

    def compute_exploitability(self) -> float:
        """II-correct exploitability via round-level joint BR traversal.

        At each round the BR player picks one action per info set, and the
        opponent's actions are weighted by the snapshot strategy.  This gives
        the correct imperfect-information best-response value.
        """
        snapshot = self._build_snapshot()
        br0 = self._br_player_value(0, snapshot)
        br1 = self._br_player_value(1, snapshot)
        return (br0 + br1) / 2.0

    def compute_exploitability_current(self) -> float:
        """Exploitability using current (regret-matching) strategy — no averaging bias."""
        snapshot = {}
        for infoset, regrets_dict in self.regret_sum.items():
            legal = sorted(regrets_dict.keys())
            strat_list = regret_matching_strategy(
                [max(regrets_dict.get(k, 0.0), 0.0) for k in legal]
            )
            snapshot[infoset] = {k: strat_list[i] for i, k in enumerate(legal)}

        br0 = self._br_player_value(0, snapshot)
        br1 = self._br_player_value(1, snapshot)
        return (br0 + br1) / 2.0

    def _build_snapshot(self) -> dict:
        snapshot = {}
        for infoset, sums in self.strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                snapshot[infoset] = {k: v / total for k, v in sums.items()}
        return snapshot

    def _br_player_value(self, br_player: int, snapshot: dict) -> float:
        """II-correct BR via belief-state enumeration.

        The BR player never sees the opponent's exact remaining hand.
        Instead, a belief distribution over possible opponent hands is
        maintained, conditioned on the BR player's info set (own hand +
        outcome history).  Opponent actions are weighted by both belief
        probability and the opponent's strategy at each possible info set.
        """

        n = self.n
        all_mask = hand_to_mask(set(range(1, n + 1)))
        initial_belief: dict[tuple[int, tuple[int, ...]], float] = {(all_mask, ()): 1.0}

        def traverse(state: GoofspielCFRState,
                     opp_belief: dict[tuple[int, tuple[int, ...]], float]) -> float:
            if state.phase == "terminal":
                assert state.rewards is not None
                return state.rewards[br_player]

            assert state.phase == "bid_p0", f"expected bid_p0 got {state.phase}"

            br_hand = state.hands[br_player]
            br_cards = mask_to_list(br_hand, n)

            best = -float("inf")
            for br_card in br_cards:
                ev = 0.0

                # Group by outcome so we can build a correct belief
                # for the next round (P0 observes the outcome but not c1).
                outcome_groups: dict[int,
                                     list[tuple[float, int, tuple[int, ...], int,
                                                GoofspielCFRState]]] = {
                    1: [], -1: [], 0: []
                }

                for (opp_hand, opp_action_history), belief_w in opp_belief.items():
                    if belief_w <= 0:
                        continue
                    opp_cards = mask_to_list(opp_hand, n)
                    ik_opp = (1 - br_player, opp_hand,
                              state.outcome_history, opp_action_history,
                              state.action_histories[br_player], state.prize_sequence)
                    strat_opp = snapshot.get(ik_opp, None)

                    if strat_opp is None:
                        u = 1.0 / len(opp_cards)
                        for opp_card in opp_cards:
                            if br_player == 0:
                                c0, c1 = br_card, opp_card
                            else:
                                c0, c1 = opp_card, br_card
                            ns = self._resolve_round(state, c0, c1)
                            outcome = (1 if c0 > c1
                                       else -1 if c0 < c1
                                       else 0)
                            outcome_groups[outcome].append(
                                (belief_w * u, opp_hand, opp_action_history,
                                 opp_card, ns))
                    else:
                        for opp_card in opp_cards:
                            prob = strat_opp.get(opp_card, 0.0)
                            if prob <= 0:
                                continue
                            if br_player == 0:
                                c0, c1 = br_card, opp_card
                            else:
                                c0, c1 = opp_card, br_card
                            ns = self._resolve_round(state, c0, c1)
                            outcome = (1 if c0 > c1
                                       else -1 if c0 < c1
                                       else 0)
                            outcome_groups[outcome].append(
                                (belief_w * prob, opp_hand, opp_action_history,
                                 opp_card, ns))

                for outcome, data in outcome_groups.items():
                    if not data:
                        continue

                    total_w = sum(d[0] for d in data)
                    if total_w <= 0:
                        continue

                    # Build the next-round belief by aggregating all
                    # (opp_hand, opp_card) → new_opp_hand weighted by
                    # belief_w * strategy_prob.
                    new_belief_raw: dict[tuple[int, tuple[int, ...]], float] = {}
                    for weight, opp_hand, opp_action_history, opp_card, _ in data:
                        new_hand = opp_hand & ~(1 << opp_card)
                        new_hist = opp_action_history + (opp_card,)
                        key = (new_hand, new_hist)
                        new_belief_raw[key] = (
                            new_belief_raw.get(key, 0.0) + weight
                        )

                    bsum = sum(new_belief_raw.values())
                    norm_belief = {h: v / bsum
                                   for h, v in new_belief_raw.items()}

                    # All entries in the same outcome group share the same
                    # norm_belief AND the same ns.hands[br_player] (since
                    # br_card & outcome are fixed).  traverse result is
                    # identical across entries — call it once.
                    _, _, _, _, ns = data[0]
                    sub_ev = traverse(ns, norm_belief)
                    ev += total_w * sub_ev

                if ev > best:
                    best = ev

            return best

        return traverse(self._initial_state(), initial_belief)
