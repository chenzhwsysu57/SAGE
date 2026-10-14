"""Verify BR consistency: BR(P0) should bound self-play EV."""
from __future__ import annotations
import sys, itertools
sys.path.insert(0, ".")

from methods.sage.cfr import (
    CFRPlusTrainer, ANTE, DECK, all_community_cards,
    is_terminal, next_round, get_legals, pot_from_history,
    PREFLOP_BET, FLOP_BET, compare, card_str,
)
from runs.leduc.compute_exploitability import _br_value

# Train
trainer = CFRPlusTrainer(iterations=2000)
trainer.train()
strat = trainer.strategy

def walk_with_strat(c1, c2, comm, history, street, cur_player, strategy, p0_contrib, p1_contrib):
    """Expected value for P0 given both follow strategy."""
    if is_terminal(history, street):
        if history[-1] == "f":
            return pot_from_history(history, street) - (p0_contrib if cur_player == 0 else p1_contrib)
        w = compare(c1, c2, comm)
        my = p0_contrib if cur_player == 0 else p1_contrib
        pot = pot_from_history(history, street)
        if w == -1: return 0
        if w == cur_player: return pot - my
        return -my
    term, ns, nc, nh = next_round(history, street, c1, c2, known_comm=comm)
    if term:
        return walk_with_strat(c1, c2, nc, nh, ns, cur_player, strategy, p0_contrib, p1_contrib)
    if ns > street and not term:
        v = walk_with_strat(c1, c2, nc, nh, ns, 0, strategy, p0_contrib, p1_contrib)
        if cur_player == 1: return -v
        return v
    legals = get_legals(history, street)
    if not legals: return 0
    pcard = c1 if cur_player == 0 else c2
    cr = card_str(comm) if street == 1 else "?"
    info = f"{pcard[0]}|{cr}|{history}|{street}"
    st = strategy.get(info, {})
    total = 0.0
    accounted = 0.0
    for a in legals:
        prob = st.get(a, 0.0)
        if prob > 0:
            nh2 = history + a
            add = PREFLOP_BET if street == 0 else FLOP_BET if a in ("b","r","k") else 0
            if cur_player == 0:
                val = walk_with_strat(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib)
            else:
                val = walk_with_strat(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add)
            val = -val
            total += prob * val
            accounted += prob
    if accounted < 1.0 - 1e-12:
        remaining = 1.0 - accounted
        unvisited = [a for a in legals if st.get(a, 0.0) == 0.0]
        if unvisited:
            uniform = remaining / len(unvisited)
            for a in unvisited:
                nh2 = history + a
                add = PREFLOP_BET if street == 0 else FLOP_BET if a in ("b","r","k") else 0
                if cur_player == 0:
                    val = walk_with_strat(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib)
                else:
                    val = walk_with_strat(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add)
                val = -val
                total += uniform * val
    return total

print("Comparing self-play EV vs BR(P0) for sample deals...")
total_ev = 0.0
total_br0 = 0.0
total_br1 = 0.0
n = 0
for c1, c2 in itertools.permutations(DECK, 2):
    for comm in all_community_cards(c1, c2):
        ev = walk_with_strat(c1, c2, comm, "", 0, 0, strat, ANTE, ANTE)
        br0 = _br_value(c1, c2, comm, "", 0, 0, 0, ANTE, strat)
        br1 = _br_value(c1, c2, comm, "", 0, 0, 1, ANTE, strat)
        total_ev += ev
        total_br0 += br0
        total_br1 += br1
        n += 1
        if br0 < ev - 1e-6:
            print(f"  BUG: BR(P0)={br0:.4f} < self-play={ev:.4f} for {c1} vs {c2} comm={comm}!")

avg_ev = total_ev / n
avg_br0 = total_br0 / n
avg_br1 = total_br1 / n

print(f"\nAverages over {n} deals:")
print(f"  Self-play EV(P0) = {avg_ev:.4f}")
print(f"  BR(P0) = {avg_br0:.4f}  (BR gain = {avg_br0-avg_ev:.4f})")
print(f"  BR(P1) = {avg_br1:.4f}  (BR gain = {avg_br1+avg_ev:.4f})")
print(f"  Exp = {(avg_br0+avg_br1)/2:.4f}")
print(f"  BR sum = {avg_br0+avg_br1:.4f} (≥0 in zero-sum)")
