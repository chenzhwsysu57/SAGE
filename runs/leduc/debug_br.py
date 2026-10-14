"""Debug BR computation by tracing a single deal in detail."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from methods.sage.cfr import CFRPlusTrainer, DECK, ANTE

# Train small model
trainer = CFRPlusTrainer(iterations=5000)
trainer.train()
strat = trainer.strategy

# Show only key preflop infosets
print("=== Key preflop strategy ===")
for k in sorted(strat.keys()):
    if "|0" not in k: continue
    v = strat[k]
    items = ", ".join(f"{a}: {p:.2%}" for a, p in sorted(v.items()))
    print(f"  {k}: {items}")
print(f"Total infosets: {len(strat)}")

# Now manually compute BR for a single deal with tracing
from runs.leduc.compute_exploitability import (
    _br_value, _terminal_value, _after_round_transition,
    _infoset_key, _who_folded,
)
from methods.sage.cfr import (
    is_terminal, next_round, get_legals, pot_from_history,
    PREFLOP_BET, FLOP_BET, compare, card_str, all_community_cards,
)

_call_depth = [0]

def _br_value_trace(c1, c2, comm, history, street, cur_player, br_player, br_bets, strategy, indent=0):
    """Traced version."""
    prefix = "  " * indent

    # Terminal
    if is_terminal(history, street):
        val = _terminal_value(history, street, br_player, c1, c2, comm, br_bets)
        is_br = "BR" if cur_player == br_player else "OP"
        print(f"{prefix}[TERM] P{cur_player}({is_br}) hist={history} st={street} br_bets={br_bets} -> {val:+.4f}")
        return val

    # Round transition
    r = _after_round_transition(c1, c2, comm, history, street, br_player, br_bets, strategy)
    if r is not None:
        is_br = "BR" if cur_player == br_player else "OP"
        print(f"{prefix}[RNDTRANS] P{cur_player}({is_br}) hist={history} st={street} br_bets={br_bets} -> {r:+.4f}")
        return r

    legals = get_legals(history, street)
    if not legals:
        return 0.0

    bet_amt = PREFLOP_BET if street == 0 else FLOP_BET

    if cur_player == br_player:
        best = -float("inf")
        best_a = None
        for a in legals:
            nh = history + a
            if a == "f":
                val = -float(br_bets)
            elif a in ("b", "r", "k"):
                val = _br_value_trace(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets + bet_amt, strategy, indent+1)
            else:
                val = _br_value_trace(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, indent+1)
            print(f"{prefix}[BR] P{cur_player}(BR) a={a} val={val:+.4f} best={best:+.4f}")
            if val > best:
                best = val
                best_a = a
        print(f"{prefix}[BR-CHOOSE] P{cur_player}(BR) hist={history} choose={best_a} val={best:+.4f}")
        return best
    else:
        info = _infoset_key(history, street, cur_player, c1, c2, comm)
        strat_probs = strategy.get(info, {})
        print(f"{prefix}[OP] P{cur_player}(OP) hist={history} info={info} strat={dict(strat_probs)}")

        total = 0.0
        accounted = 0.0
        for a in legals:
            prob = strat_probs.get(a, 0.0)
            if prob > 0:
                nh = history + a
                if a == "f":
                    val = float(pot_from_history(nh, street)) - float(br_bets)
                elif a in ("b", "r", "k"):
                    val = _br_value_trace(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, indent+1)
                else:
                    val = _br_value_trace(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, indent+1)
                total += prob * val
                accounted += prob
                print(f"{prefix}[OP] a={a} prob={prob:.3f} val={val:+.4f}")

        if accounted < 1.0 - 1e-12:
            remaining = 1.0 - accounted
            unvisited = [a for a in legals if strat_probs.get(a, 0.0) == 0.0]
            if unvisited:
                uniform = remaining / len(unvisited)
                for a in unvisited:
                    nh = history + a
                    if a == "f":
                        val = float(pot_from_history(nh, street)) - float(br_bets)
                    elif a in ("b", "r", "k"):
                        val = _br_value_trace(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, indent+1)
                    else:
                        val = _br_value_trace(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, indent+1)
                    total += uniform * val
                    print(f"{prefix}[OP-UNIF] a={a} prob={uniform:.3f} val={val:+.4f}")

        print(f"{prefix}[OP-EV] P{cur_player}(OP) hist={history} ev={total:+.4f}")
        return total

# Trace specific deals
print("\n\n=== Tracing deal: P0=JH, P1=JS, comm=K ===")
print("\n--- BR(P0) ---")
v0 = _br_value_trace(("J","H"), ("J","S"), ("K","H"), "", 0, 0, 0, ANTE, strat)
print(f"  BR(P0) = {v0:+.4f}")

print("\n\n=== Tracing deal: P0=JH, P1=QH, comm=K ===")
print("\n--- BR(P0) ---")
v0 = _br_value_trace(("J","H"), ("Q","H"), ("K","H"), "", 0, 0, 0, ANTE, strat)
print(f"  BR(P0) = {v0:+.4f}")
