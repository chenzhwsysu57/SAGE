"""CFR query tools for Leduc Hold'em.

Given a game state, renders the saved Nash mixed strategy for prompt use.
"""

from __future__ import annotations

import json
import os

# ── Constants (mirror methods/cfr.py) ──────────────────────────────────────

RANKS = ["J", "Q", "K"]
SUITS = ["H", "S"]
DECK = [(r, s) for r in RANKS for s in SUITS]
MAX_RAISES = 2

# CFR action encoding
FOLD, CHECK, CALL, BET, RAISE = "f", "c", "k", "b", "r"
ACTION_MAP = {"fold": FOLD, "check": CHECK, "call": CALL, "bet": BET, "raise": RAISE}
REVERSE_MAP = {v: k for k, v in ACTION_MAP.items()}


def card_str(c):
    return c[0] if c else "?"


# ── Game tree helpers ─────────────────────────────────────────────────────

def _get_legals(history, street):
    seg = history.split("d")[street] if street < len(history.split("d")) else ""
    if not seg:
        return [CHECK, BET]
    last = seg[-1]
    if last == FOLD:
        return []
    bet_cnt = seg.count(BET) + seg.count(RAISE)
    if last == CHECK:
        return [CHECK, BET]
    if last in (BET, RAISE):
        acts = [FOLD, CALL]
        if bet_cnt < MAX_RAISES + 1:
            acts.append(RAISE)
        return acts
    return []


# ── CFR Strategy loader ───────────────────────────────────────────────────

class CFRTools:
    """Query tool for CFR Nash equilibrium strategy.

    Loads the pre-trained strategy from disk and supports:
      - mixed_strategy(): action frequencies from the saved CFR strategy
      - format_for_prompt(): human-readable string for LLM prompt injection
    """

    _DEFAULT_PATH = "results/leduc/cfr_model/cfr_plus_21000.json"

    def __init__(self, strategy_path: str = ""):
        path = strategy_path or self._DEFAULT_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CFR model not found at {path}. Run scripts/train_cfr.py first.")
        with open(path) as f:
            data = json.load(f)
        self._strategy = data["strategy"]
        self._num_infosets = data.get("num_infosets", len(self._strategy))

    # ── Public API ────────────────────────────────────────────────────

    def mixed_strategy(self, my_card: str, community_card: str,
                       history: list[dict], street: str) -> tuple[dict[str, float], bool]:
        """Return the CFR mixed strategy for the current public infoset.

        The returned action labels match prompt-facing Leduc actions. If the
        strategy table has no usable mass for this infoset, probabilities fall
        back to uniform over legal actions and the second return value is True.
        """
        cfr_hist, cfr_street = self._encode_history(history, street)
        legal_codes = _get_legals(cfr_hist, cfr_street)
        if not legal_codes:
            return {}, False

        my_card_tuple = self._parse_card(my_card)
        if my_card_tuple is None:
            return {}, False

        comm_tuple = self._parse_card(community_card) if community_card else None
        comm_rank = card_str(comm_tuple) if cfr_street == 1 else "?"
        info_key = f"{my_card_tuple[0]}|{comm_rank}|{cfr_hist}|{cfr_street}"

        raw_probs = self._strategy.get(info_key, {})
        filtered = {code: float(raw_probs.get(code, 0.0)) for code in legal_codes}
        total = sum(filtered.values())
        used_fallback = total <= 0.0
        if used_fallback:
            filtered = {code: 1.0 / len(legal_codes) for code in legal_codes}
        else:
            filtered = {code: prob / total for code, prob in filtered.items()}

        return {
            REVERSE_MAP.get(code, code): prob
            for code, prob in filtered.items()
        }, used_fallback

    def format_for_prompt(self, my_card: str, community_card: str,
                          history: list[dict], street: str,
                          my_id: int) -> str:
        """Render CFR mixed strategy as a prompt-ready string."""
        my_card_tuple = self._parse_card(my_card)
        comm_tuple = self._parse_card(community_card) if community_card else None
        if my_card_tuple is None:
            return "(no CFR data for this state)"
        chance_states = self._enumerate_chance_states(my_card_tuple, comm_tuple)
        if not chance_states:
            return "(no CFR data for this state)"

        comm_str = community_card[0] if community_card else "none"
        lines = [
            f"--- CFR Nash Equilibrium Analysis ---",
            f"Your hand: {my_card[0] if my_card else '?'}  |  "
            f"Board: {comm_str}  |  Street: {street}",
            "",
            "Opponent hand distribution (equiprobable among unseen cards):",
        ]

        # Show opponent hand probabilities
        probs = self._opp_hand_probs(chance_states)
        prob_parts = [f"  P(opp={r}): {p:.0%}" for r, p in probs.items()]
        lines.append(" ".join(prob_parts))
        lines.append("")

        mixed, used_fallback = self.mixed_strategy(my_card, community_card, history, street)
        if mixed:
            strategy_parts = [
                f"{action}: {prob:.1%}"
                for action, prob in mixed.items()
            ]
            suffix = " (uniform fallback: CFR infoset missing/zero mass)" if used_fallback else ""
            lines.append(f"Nash mixed strategy: {', '.join(strategy_parts)}{suffix}")
            lines.append("")

        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────

    def _parse_card(self, s: str):
        if not s or len(s) < 2:
            return None
        return (s[0], s[1])

    def _enumerate_chance_states(self, my_card, comm):
        """Return weighted (opponent_card, community_card, probability) states."""
        if my_card is None:
            return []
        if comm is not None:
            opp_hands = self._enumerate_opponent_hands(my_card, comm)
            prob = 1.0 / len(opp_hands) if opp_hands else 0.0
            return [(opp, comm, prob) for opp in opp_hands]

        states = []
        for opp in self._enumerate_opponent_hands(my_card, None):
            used = {my_card, opp}
            for future_comm in DECK:
                if future_comm in used:
                    continue
                states.append((opp, future_comm, 0.0))
        prob = 1.0 / len(states) if states else 0.0
        return [(opp, future_comm, prob) for opp, future_comm, _ in states]

    def _enumerate_opponent_hands(self, my_card, comm):
        used = {my_card}
        if comm:
            used.add(comm)
        # Also exclude any card with the same rank+suit as community if known
        return [c for c in DECK if c not in used]

    def _opp_hand_probs(self, chance_states):
        if not chance_states:
            return {}
        counts = {}
        for opp, _, prob in chance_states:
            counts[opp[0]] = counts.get(opp[0], 0.0) + prob
        return {
            rank: round(counts[rank], 4)
            for rank in RANKS
            if rank in counts
        }

    def _encode_history(self, history, street):
        """Convert game history list to CFR action string."""
        cfr_hist = ""
        prev_street = 0
        for entry in history:
            entry_street = entry.get("street", 0)
            if entry_street > prev_street:
                cfr_hist += "d"
            action = entry.get("action", "")
            cfr_hist += ACTION_MAP.get(action, "?")
            prev_street = entry_street

        cfr_street = 0 if street == "preflop" else 1
        if cfr_street == 1 and "d" not in cfr_hist:
            cfr_hist += "d"
        return cfr_hist, cfr_street
