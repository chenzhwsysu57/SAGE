"""CFR+ (Counterfactual Regret Minimization Plus) for Leduc Hold'em.

Computes a near-Nash equilibrium strategy through iterative self-play.
No LLM calls — purely algorithmic. Baseline for optimal-play comparison.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
from collections import defaultdict

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation

RANKS = ["J", "Q", "K"]
DECK = [(r, s) for r in RANKS for s in "HS"]  # 6 cards
ANTE = 1
PREFLOP_BET = 2
FLOP_BET = 4
MAX_RAISES = 2

# CFR action encoding: 'f'=fold, 'c'=check, 'k'=call, 'b'=bet, 'r'=raise
ACTION_TO_CODE = {"fold": "f", "check": "c", "call": "k", "bet": "b", "raise": "r"}
CODE_TO_ACTION = {v: k for k, v in ACTION_TO_CODE.items()}


class _StaticCFRTrainer:
    """Lightweight trainer-like wrapper for a loaded CFR strategy."""

    def __init__(self, strategy: dict):
        self.strategy = strategy
        self._trained = True


def load_or_train_cfr_strategy(path: str, iterations: int) -> tuple[object, bool]:
    """Load a saved CFR strategy if available, otherwise train and save one."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return _StaticCFRTrainer(data["strategy"]), False

    trainer = CFRPlusTrainer(iterations=iterations)
    trainer.train()
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "iterations": iterations,
            "num_infosets": len(trainer.strategy),
            "strategy": {k: dict(v) for k, v in trainer.strategy.items()},
        }, f)
    return trainer, True


def encode_history_for_cfr(history: list[dict], street: int) -> str:
    """Convert Leduc history dicts to the CFR trainer's flat history string."""
    cfr_hist = ""
    prev_street = 0
    for entry in history:
        entry_street = int(entry.get("street", 0))
        if entry_street > prev_street:
            cfr_hist += "d"
        cfr_hist += ACTION_TO_CODE.get(entry.get("action", ""), "?")
        prev_street = entry_street

    if street == 1 and "d" not in cfr_hist:
        cfr_hist += "d"
    return cfr_hist


def strategy_probs_for_legal_actions(
    strategy: dict,
    info: str,
    legal_actions: list[Action],
) -> tuple[dict[str, float], bool]:
    """Return normalized natural-action probabilities for legal actions."""
    probs = strategy.get(info, {})
    legal_codes = {ACTION_TO_CODE.get(a.type, ""): a.type for a in legal_actions}
    filtered_codes = {
        code: probs.get(code, 0.0)
        for code in legal_codes
        if code
    }
    total = sum(filtered_codes.values())
    used_fallback = total == 0
    if used_fallback:
        n = len(legal_actions)
        return {a.type: 1.0 / n for a in legal_actions}, True
    return {
        legal_codes[code]: prob / total
        for code, prob in filtered_codes.items()
    }, False


def sample_action_from_probs(
    legal_actions: list[Action],
    action_probs: dict[str, float],
    rng: random.Random,
) -> Action:
    r = rng.random()
    cumulative = 0.0
    chosen_type = legal_actions[0].type
    for action in legal_actions:
        cumulative += action_probs.get(action.type, 0.0)
        if r <= cumulative:
            chosen_type = action.type
            break

    for action in legal_actions:
        if action.type == chosen_type:
            return Action(
                player_id=action.player_id,
                type=action.type,
                amount=action.amount,
            )
    first = legal_actions[0]
    return Action(player_id=first.player_id, type=first.type, amount=first.amount)


def strategy_l1_diff(s1, s2):
    """L1 norm between two average strategy dictionaries (over shared infosets)."""
    keys = set(s1) | set(s2)
    total = 0.0
    n = 0
    for k in keys:
        acts = set(s1.get(k, {})) | set(s2.get(k, {}))
        for a in acts:
            total += abs(s1.get(k, {}).get(a, 0.0) - s2.get(k, {}).get(a, 0.0))
            n += 1
    return total / max(n, 1)


def card_str(c):
    return c[0] if c else "?"


def hand_strength(card, comm):
    """Pair = 100 + rank_idx, else = rank_idx. Higher is better."""
    if comm and card[0] == comm[0]:
        return 100 + RANKS.index(card[0])
    return RANKS.index(card[0])


def compare(c1, c2, comm):
    """Return 0 if P0 wins, 1 if P1 wins, -1 if tie."""
    s0 = hand_strength(c1, comm)
    s1 = hand_strength(c2, comm)
    if s0 > s1: return 0
    if s1 > s0: return 1
    return -1


def community_card(c1, c2):
    """Return the first unused card as community."""
    for c in DECK:
        if c != c1 and c != c2:
            return c
    return DECK[0]


def all_community_cards(c1, c2):
    """Return all possible community cards given two dealt hands."""
    return [c for c in DECK if c != c1 and c != c2]


def is_terminal(history, street):
    """Check if the game has ended (fold or showdown)."""
    if not history:
        return False
    if history[-1] == "f":
        return True
    # Get segment for current street
    seg = history.split("d")[street] if street < len(history.split("d")) else ""
    # Call on flop ends game (showdown)
    if street == 1 and seg and seg[-1] == "k":
        return True
    # Check-check on flop ends game
    if street == 1 and seg.count("c") >= 2 and len(seg) >= 2:
        return True
    return False


def next_round(history, street, c1, c2, known_comm=None):
    """Check if betting round ended. Returns (terminal, new_street, new_comm, new_history)."""
    seg = history.split("d")[street] if street < len(history.split("d")) else ""

    if not seg:
        return False, street, None, history

    if seg[-1] == "f":
        return True, street, None, history

    # Call ends the round
    if seg[-1] == "k":
        if street == 0:
            comm = known_comm if known_comm is not None else community_card(c1, c2)
            return False, 1, comm, history + "d"
        else:
            return True, street, None, history

    # Check-check ends the round
    if seg.count("c") >= 2 and len(seg) >= 2:
        if street == 0:
            comm = known_comm if known_comm is not None else community_card(c1, c2)
            return False, 1, comm, history + "d"
        else:
            return True, street, None, history

    return False, street, None, history


def get_legals(history, street):
    """Get legal actions as CFR codes for current state."""
    seg = history.split("d")[street] if street < len(history.split("d")) else ""

    if not seg:
        return ["c", "b"]

    last = seg[-1]
    if last == "f":
        return []  # terminal

    # Count bets+raises this segment
    bet_cnt = seg.count("b") + seg.count("r")

    if last == "c":
        return ["c", "b"]

    if last in ("b", "r"):
        acts = ["f", "k"]
        if bet_cnt < (MAX_RAISES + 1):  # max 2 raises means 3 total bets
            acts.append("r")
        return acts

    return []


def pot_from_history(history, street):
    """Compute pot at current point."""
    pot = ANTE * 2
    # Each 'b' or 'r' adds a bet
    # Each 'k' adds a call (matching the previous bet)
    # Need to track bet amounts per street
    seg0 = history.split("d")[0] if "d" not in history else history.split("d")[0]
    for ch in seg0:
        if ch in ("b", "r"):
            pot += PREFLOP_BET
        elif ch == "k":
            pot += PREFLOP_BET

    if "d" in history:
        seg1 = history.split("d")[1]
        for ch in seg1:
            if ch in ("b", "r"):
                pot += FLOP_BET
            elif ch == "k":
                pot += FLOP_BET

    return pot


class CFRPlusTrainer:
    """CFR+ algorithm for Leduc Hold'em."""

    def __init__(self, iterations: int = 100000, target_exp: float | None = None):
        self.iterations = iterations
        self.target_exp = target_exp
        self.regrets: dict = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: dict = defaultdict(lambda: defaultdict(float))
        self.strategy: dict = {}
        self._trained = False

    def train(self):
        """Run CFR+ iterations over all possible deals + community cards."""
        deals = list(itertools.permutations(DECK, 2))
        n_combos = len(deals) * 4
        log_interval = getattr(self, "log_interval", None) or max(1, self.iterations // 20)
        print(f"Training CFR+ ({self.iterations} iterations × {n_combos} deal-comm combos, logging every {log_interval})...")

        t0 = time.perf_counter()
        self._iterations_run = self.iterations
        from tqdm import tqdm
        for i in tqdm(range(self.iterations)):
            for c1, c2 in deals:
                for comm in all_community_cards(c1, c2):
                    self._walk(c1, c2, comm, "", 0, 0, 1.0, 1.0, ANTE, ANTE)

            if (i + 1) % log_interval == 0:
                el = time.perf_counter() - t0
                from runs.leduc.compute_exploitability import best_response_value
                strat_tmp = self._build_strategy_from_sum()
                br0 = best_response_value(strat_tmp, 0)
                br1 = best_response_value(strat_tmp, 1)
                exp = (br0 + br1) / 2.0
                print(f"  [{i+1:>6}/{self.iterations}]  {el:5.0f}s  "
                      f"exp={exp:.4f}  BR0={br0:+.4f}  BR1={br1:+.4f}  "
                      f"({self.iterations/(i+1)*el - el:5.0f}s rem)")
                self._last_exp = exp
                if self.target_exp is not None and exp <= self.target_exp:
                    self._iterations_run = i + 1
                    print(f"  -> Target exploitability {self.target_exp} reached at iteration {self._iterations_run}. Stopping early.")
                    break

            # --- Diagnostic: strategy snapshot at 1k / 2k / 5k (Task 1) ---
            diag_iters = {1000, 2000, 5000}
            if (i + 1) in diag_iters:
                snap = self._build_strategy_from_sum()
                if not hasattr(self, "_strategy_snapshots"):
                    self._strategy_snapshots = {}
                self._strategy_snapshots[i + 1] = snap
                print(f"  [Diagnostic] Saved strategy snapshot at iteration {i+1} "
                      f"(infosets: {len(snap)})")

                if len(self._strategy_snapshots) >= 2:
                    prev_it = sorted(self._strategy_snapshots.keys())[-2]
                    prev = self._strategy_snapshots[prev_it]
                    d = strategy_l1_diff(prev, snap)
                    print(f"  [Diagnostic] L1(avg_{prev_it}, avg_{i+1}) = {d:.8f}")

        # Build final average strategy
        self.strategy = self._build_strategy_from_sum()
        self._trained = True
        dt = time.perf_counter() - t0
        print(f"CFR+ trained. {len(self.strategy)} information sets. ({dt:.0f}s total)")

        # --- Diagnostic: L1 summary (Task 1) ---
        if hasattr(self, "_strategy_snapshots") and len(self._strategy_snapshots) >= 2:
            print("  [Diagnostic] Strategy L1 diffs between snapshots:")
            sorted_its = sorted(self._strategy_snapshots.keys())
            for i in range(len(sorted_its)):
                for j in range(i + 1, len(sorted_its)):
                    d = strategy_l1_diff(
                        self._strategy_snapshots[sorted_its[i]],
                        self._strategy_snapshots[sorted_its[j]],
                    )
                    print(f"    L1(avg_{sorted_its[i]}, avg_{sorted_its[j]}) = {d:.8f}")

    def _build_strategy_from_sum(self):
        strategy = {}
        for info, sums in self.strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                strategy[info] = {a: s / total for a, s in sums.items()}
            else:
                acts = list(sums.keys())
                k = len(acts)
                strategy[info] = {a: 1.0 / k for a in acts} if k else {}
        return strategy

    def _walk(self, c1, c2, comm, history, street, cur_player,
              p0_reach, p1_reach, p0_contrib, p1_contrib):
        """Recursive CFR tree walk. Returns counterfactual value for cur_player."""

        # Terminal check
        if is_terminal(history, street):
            if history[-1] == "f":
                # Opponent folded. From cur_player's perspective, net profit is
                # opponent's total contribution (we get our own back + theirs).
                return pot_from_history(history, street) - (
                    p0_contrib if cur_player == 0 else p1_contrib
                )
            # Showdown — compute net profit from cur_player's perspective
            my_contrib = p0_contrib if cur_player == 0 else p1_contrib
            w = compare(c1, c2, comm)
            pot = pot_from_history(history, street)
            if w == -1:  # tie
                return 0  # split, net 0
            if w == cur_player:
                return pot - my_contrib  # net win = opp's contribution
            return -my_contrib  # net loss = own contribution

        # Check round transition
        term, new_st, new_comm, new_hist = next_round(history, street, c1, c2, known_comm=comm)
        if term:
            return self._walk(c1, c2, new_comm, new_hist, new_st,
                             cur_player, p0_reach, p1_reach, p0_contrib, p1_contrib)

        if new_st > street and not term:
            # P0 always acts first on the flop in Leduc
            child = self._walk(c1, c2, new_comm, new_hist, new_st, 0,
                              p0_reach, p1_reach, p0_contrib, p1_contrib)
            # Round transition reset cur_player to 0; adjust perspective
            # back to the original cur_player for the calling convention
            if cur_player == 1:
                return -child
            return child

        # Get legal actions
        legals = get_legals(history, street)
        if not legals:
            return 0

        # Info set key — street=0 always uses "?" for community (not yet revealed)
        private_card = c1 if cur_player == 0 else c2
        comm_rank = card_str(comm) if street == 1 else "?"
        info = f"{private_card[0]}|{comm_rank}|{history}|{street}"

        # Regret matching
        regret_dict = self.regrets[info]
        pos_regrets = {a: max(regret_dict.get(a, 0), 0) for a in legals}
        total = sum(pos_regrets.values())
        if total > 0:
            sigma = {a: pos_regrets[a] / total for a in legals}
        else:
            sigma = {a: 1.0 / len(legals) for a in legals}

        # Compute action values
        vals = {}
        node_val = 0.0
        for a in legals:
            nh = history + a
            add = 0
            if a in ("b", "r", "k"):
                bet_size = PREFLOP_BET if street == 0 else FLOP_BET
                add = bet_size

            if cur_player == 0:
                child = self._walk(c1, c2, comm, nh, street, 1 - cur_player,
                                  p0_reach * sigma[a], p1_reach,
                                  p0_contrib + add, p1_contrib)
            else:
                child = self._walk(c1, c2, comm, nh, street, 1 - cur_player,
                                  p0_reach, p1_reach * sigma[a],
                                  p0_contrib, p1_contrib + add)

            # Zero-sum: opponent's value = -(current player's value)
            child = -child

            # Fold means losing what we already put in
            if a == "f":
                if cur_player == 0:
                    child = -p0_contrib
                else:
                    child = -p1_contrib

            vals[a] = child
            node_val += sigma[a] * child

        # Update regrets (CFR+)
        # Counterfactual regret = π_{-i}(h) × (action_value - node_value)
        cf_reach = p1_reach if cur_player == 0 else p0_reach
        for a in legals:
            inst_regret = cf_reach * (vals[a] - node_val)
            self.regrets[info][a] = max(self.regrets[info][a] + inst_regret, 0)

            # Strategy sum (weighted by reach probability)
            reach = p0_reach if cur_player == 0 else p1_reach
            self.strategy_sum[info][a] += reach * sigma[a]

        return node_val


class CFRMethod(BaseMethod):
    """CFR+ poker method — no LLM calls.

    Uses a pre-computed Nash equilibrium strategy via CFR+.
    Strategy is cached at module level — trains only once per process.

    Args:
        name: Display name.
        iterations: CFR+ training iterations (default 100k).
        seed: Random seed for action sampling.
    """

    # Module-level cache: (iterations) → trained trainer
    _cache: dict = {}
    _DEFAULT_PATH = "results/leduc/cfr_model/cfr_plus_21000.json"

    def __init__(
        self,
        name: str = "CFR+",
        iterations: int = 100000,
        seed: int = 0,
        strategy_path: str = "",
    ):
        self.name = name
        self.method_type = "cfr"
        self._iterations = iterations
        self._rng = random.Random(seed)
        self._strategy_path = strategy_path or self._DEFAULT_PATH

        cache_key = (iterations, self._strategy_path)
        if cache_key not in CFRMethod._cache:
            trainer, _ = load_or_train_cfr_strategy(self._strategy_path, iterations)
            CFRMethod._cache[cache_key] = trainer
        self._trainer = CFRMethod._cache[cache_key]

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel | None = None,
    ) -> tuple[Action, str, str]:
        card = observation.private_info.get("card", "??")
        rank = card[0] if len(card) > 0 else "?"
        comm_val = observation.public_info.get("community_card", "")
        comm_rank = comm_val[0] if comm_val else "?"
        street_str = observation.public_info.get("street", "preflop")
        street = 0 if street_str == "preflop" else 1

        hist_str = encode_history_for_cfr(observation.history, street)

        info = f"{rank}|{comm_rank}|{hist_str}|{street}"

        action_probs, used_fallback = strategy_probs_for_legal_actions(
            self._trainer.strategy, info, legal_actions,
        )
        action = sample_action_from_probs(legal_actions, action_probs, self._rng)
        fallback_note = " fallback=uniform" if used_fallback else ""
        return action, f"CFR+ lookup: {info}{fallback_note} -> selected={action}", action.type
