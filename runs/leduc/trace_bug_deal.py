"""Trace the specific deal where BR(P0) < self-play EV."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from methods.cfr import CFRPlusTrainer, ANTE, DECK
from methods.cfr import (
    is_terminal, next_round, get_legals, pot_from_history,
    PREFLOP_BET, FLOP_BET, compare, card_str
)
from runs.leduc.compute_exploitability import _br_value, _terminal_value, _infoset_key, _after_round_transition, _who_folded

trainer = CFRPlusTrainer(iterations=2000)
trainer.train()
strat = trainer.strategy

c1, c2 = ("J","H"), ("Q","S")
comm = ("Q","H")

# Show strategy for all infosets reachable in this deal
print("=== Strategy at relevant infosets ===")
for k in sorted(strat.keys()):
    show = False
    if "J|" in k and "|0" in k: show = True  # preflop J
    if "Q|" in k and "|0" in k: show = True  # preflop Q
    if "|Q|" in k and "|1" in k and ("J|" in k or "Q|" in k): show = True  # flop J or Q with Q comm
    if show:
        items = ", ".join(f"{a}: {p:.4f}" for a,p in sorted(strat[k].items()))
        print(f"  {k}: {items}")

# Trace self-play for this deal
print(f"\n=== Self-play trace: P0={c1}, P1={c2}, comm={comm} ===")

contrib_trace = {"p0": ANTE, "p1": ANTE}

def walk_trace(c1, c2, comm, history, street, cur_player, strategy, p0_contrib, p1_contrib, indent=0):
    prefix = "  " * indent
    if is_terminal(history, street):
        if history[-1] == "f":
            val = pot_from_history(history, street) - (p0_contrib if cur_player == 0 else p1_contrib)
            print(f"{prefix}[TERM] fold, hist={history}, cur={cur_player}, p0c={p0_contrib}, p1c={p1_contrib}, val={val:.4f}")
            return val
        w = compare(c1, c2, comm)
        my = p0_contrib if cur_player == 0 else p1_contrib
        pot = pot_from_history(history, street)
        val = 0 if w == -1 else (pot - my if w == cur_player else -my)
        print(f"{prefix}[TERM] showdown, hist={history}, cur={cur_player}, w={w}, my={my}, pot={pot}, val={val:.4f}")
        return val
    term, ns, nc, nh = next_round(history, street, c1, c2, known_comm=comm)
    if term:
        return walk_trace(c1, c2, nc, nh, ns, cur_player, strategy, p0_contrib, p1_contrib, indent)
    if ns > street and not term:
        print(f"{prefix}[RND] round trans from {history} st={street}→{ns}")
        v = walk_trace(c1, c2, nc, nh, ns, 0, strategy, p0_contrib, p1_contrib, indent+1)
        if cur_player == 1: v = -v
        print(f"{prefix}[RND] -> {v:.4f}")
        return v
    legals = get_legals(history, street)
    if not legals: return 0
    pcard = c1 if cur_player == 0 else c2
    cr = card_str(comm) if street == 1 else "?"
    info = f"{pcard[0]}|{cr}|{history}|{street}"
    st = strategy.get(info, {})
    print(f"{prefix}[NODE] P{cur_player}({pcard[0]}) hist={history} st={street} info={info} strat={dict(st)}")
    total = 0.0
    for a in legals:
        prob = st.get(a, 0.0)
        if prob > 0:
            nh2 = history + a
            add = PREFLOP_BET if street == 0 else FLOP_BET if a in ("b","r","k") else 0
            if cur_player == 0:
                val = walk_trace(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib, indent+1)
            else:
                val = walk_trace(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add, indent+1)
            val = -val
            total += prob * val
            print(f"{prefix}  a={a} prob={prob:.4f} val={val:.4f}")
    if abs(total) > 0.1:
        print(f"{prefix}=> EV = {total:.4f}")
    return total

print()
ev = walk_trace(c1, c2, comm, "", 0, 0, strat, ANTE, ANTE)
print(f"\nSelf-play EV(P0) = {ev:.4f}")

# Now trace BR(P0) for this deal
print(f"\n=== BR(P0) trace ===")
br0 = _br_value(c1, c2, comm, "", 0, 0, 0, ANTE, strat)
print(f"BR(P0) = {br0:.4f}")
print(f"\n=== BR(P1) trace ===")
br1 = _br_value(c1, c2, comm, "", 0, 0, 1, ANTE, strat)
print(f"BR(P1) = {br1:.4f}")
