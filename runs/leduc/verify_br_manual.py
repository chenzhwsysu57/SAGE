"""Debug BR: compute BR for a minimal deterministic strategy and verify manually."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from methods.sage.cfr import (
    DECK, ANTE, PREFLOP_BET, FLOP_BET, compare, all_community_cards,
    is_terminal, next_round, get_legals, pot_from_history, card_str,
)
import itertools

# Build a strategy: always check-call at every infoset
# This is a very passive strategy.
# Then compute BR against it and compare with manual calculation.
check_call_strat = {}

# Cover all preflop infosets
for rank in "JQK":
    for hist in ["", "c", "b", "cb", "br", "cbr", "brr", "cbrr"]:
        for street in [0]:
            info = f"{rank}|?|{hist}|{street}"
            legals = get_legals(hist, street)
            if legals:
                d = {}
                for a in legals:
                    if a in ("f",):
                        d[a] = 0.0
                    elif a in ("c",):
                        d[a] = 1.0  # always check when possible
                    elif a == "k":
                        d[a] = 1.0  # always call when possible (no check option)
                    elif a in ("b", "r"):
                        d[a] = 0.0
                check_call_strat[info] = d

# Cover flop infosets (all community cards J, Q, K)
for rank in "JQK":
    for comm in "JQK":
        # Flop history starts with some preflop history + "d" + flop actions
        for preflop_hist in ["c", "b", "cb", "br", "cbr", "brr", "cbrr", ""]:
            for flop_hist in ["", "c", "b", "cb", "br", "cbr"]:
                full_hist = preflop_hist + "d" + flop_hist if flop_hist else ""
                if not full_hist: continue
                info = f"{rank}|{comm}|{full_hist}|1"
                legals = get_legals(full_hist, 1)
                if legals:
                    d = {}
                    for a in legals:
                        if a in ("f",):
                            d[a] = 0.0
                        elif a in ("c",):
                            d[a] = 1.0
                        elif a == "k":
                            d[a] = 1.0
                        elif a in ("b", "r"):
                            d[a] = 0.0
                    check_call_strat[info] = d

# Now also cover "ccd..." patterns (check-check preflop → flop)
rank_hist_pairs = []
for rank in "JQK":
    for hist_street in [(f, s) for f in ["", "c", "b", "cb", "br", "cbr", "brr", "cbrr"] for s in [0, 1]]:
        pass  # already covered above

# Special: "ccd" (check-check preflop)
info = "J|J|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "J|Q|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "J|K|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "Q|J|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "Q|Q|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "Q|K|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "K|J|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "K|Q|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}
info = "K|K|ccd|1"
check_call_strat[info] = {"b": 0.0, "c": 1.0}

# Also need "ccdc" etc
for r1 in "JQK":
    for rc in "JQK":
        for flop_suffix in ["c", "b", "cb", "br"]:
            info = f"{r1}|{rc}|ccd{flop_suffix}|1"
            legals = get_legals(f"ccd{flop_suffix}", 1)
            if legals:
                d = {}
                for a in legals:
                    d[a] = 1.0 if a in ("c","k") else 0.0
                check_call_strat[info] = d

print(f"Strategy built: {len(check_call_strat)} infosets")

# Now use the standard BR computation
from runs.leduc.compute_exploitability import best_response_value, _br_value, _terminal_value

# Compute BR for each deal
print(f"\n=== BR against check-call strategy ===")
header = f"{'Deal':>15} {'Comm':>4} {'BR(P0)':>8} {'BR(P1)':>8} {'Sum':>8}"
print(header)
print("-" * len(header))

total_br0 = 0.0
total_br1 = 0.0
n = 0
for c1, c2 in itertools.permutations(DECK, 2):
    for comm in all_community_cards(c1, c2):
        v0 = best_response_value(check_call_strat, 0) if False else None  # skip, compute per-deal
        # compute per-deal
        # Actually compute single values
        from methods.sage.cfr import is_terminal, next_round, get_legals, pot_from_history, PREFLOP_BET, FLOP_BET, compare, card_str
        from runs.leduc.compute_exploitability import _infoset_key, _after_round_transition, _who_folded

        def _br_value_single(c1, c2, comm, history, street, cur_player, br_player, br_bets, strategy, depth=0):
            if is_terminal(history, street):
                return _terminal_value(history, street, br_player, c1, c2, comm, br_bets)
            r = _after_round_transition(c1, c2, comm, history, street, br_player, br_bets, strategy)
            if r is not None:
                return r
            legals = get_legals(history, street)
            if not legals:
                return 0.0
            bet_amt = PREFLOP_BET if street == 0 else FLOP_BET
            if cur_player == br_player:
                best = -float("inf")
                for a in legals:
                    nh = history + a
                    if a == "f": val = -float(br_bets)
                    elif a in ("b", "r", "k"):
                        val = _br_value_single(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets + bet_amt, strategy, depth+1)
                    else:
                        val = _br_value_single(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, depth+1)
                    if val > best: best = val
                return best
            else:
                info = _infoset_key(history, street, cur_player, c1, c2, comm)
                sp = strategy.get(info, {})
                total = 0.0
                accounted = 0.0
                for a in legals:
                    prob = sp.get(a, 0.0)
                    if prob > 0:
                        nh = history + a
                        if a == "f": val = float(pot_from_history(nh, street)) - float(br_bets)
                        elif a in ("b", "r", "k"):
                            val = _br_value_single(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, depth+1)
                        else:
                            val = _br_value_single(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, depth+1)
                        total += prob * val
                        accounted += prob
                if accounted < 1.0 - 1e-12:
                    remaining = 1.0 - accounted
                    unvisited = [a for a in legals if sp.get(a, 0.0) == 0.0]
                    if unvisited:
                        uniform = remaining / len(unvisited)
                        for a in unvisited:
                            nh = history + a
                            if a == "f": val = float(pot_from_history(nh, street)) - float(br_bets)
                            elif a in ("b", "r", "k"):
                                val = _br_value_single(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, depth+1)
                            else:
                                val = _br_value_single(c1, c2, comm, nh, street, 1-cur_player, br_player, br_bets, strategy, depth+1)
                            total += uniform * val
                return total

        v0 = _br_value_single(c1, c2, comm, "", 0, 0, 0, ANTE, check_call_strat)
        v1 = _br_value_single(c1, c2, comm, "", 0, 0, 1, ANTE, check_call_strat)
        deal_label = f"{c1[0]}{c1[1]}-{c2[0]}{c2[1]}"
        if n < 5:  # show first 5 deals
            print(f"{deal_label:>15} {comm[0]:>4} {v0:>+8.4f} {v1:>+8.4f} {v0+v1:>+8.4f}")
        total_br0 += v0
        total_br1 += v1
        n += 1

avg_br0 = total_br0 / n
avg_br1 = total_br1 / n
print(f"\nAverage over {n} deals:")
print(f"  BR(P0) = {avg_br0:.4f}")
print(f"  BR(P1) = {avg_br1:.4f}")
print(f"  Exploit = {(avg_br0+avg_br1)/2:.4f}")
print(f"  Sum = {avg_br0+avg_br1:.4f}")

# Manual check: BR(P1) against always check-call P0
# With a check-call strategy, P0 always checks preflop. P1 (BR) can bet or check.
# If P1 bets: P0 always calls (k=1.0). Pot goes to 6, goes to flop.
# On flop, P1 can bet again and P0 will call.
# For P1 with K: P1 is ahead when P0 has J, behind when P0 has K (same), even when Q.
# The BR should maximize profit.
#
# Simple case: P0=JH, P1=KS. P0 checks (always). P1 (BR) can check or bet.
# If P1 checks: check-check, flop. All check-call. Net: depends on community.
# If P1 bets: P0 calls. br_bets = 1+2=3. Then flop.
#   On flop, P1 can bet again (since P0 always calls).
#   If P1 bets: P0 calls. br_bets = 3+4=7. Showdown.
#   The net depends on who wins.
#
# Let's compute P0=JH, P1=KS:
# If P1 checks preflop: check-check → flop. Each of 4 community cards:
#   If community K: P1 wins. BR's net for check: br_bets=1, pot=2+? hmm need to check.
print("\n\n=== Manual verification for one deal ===")
print("P0=JH, P1=KS (BR), check-call strategy:")
# P0 follows check-call at every node.
print("  Initial: br_bets=1 (ante)")

# With P1 as BR at root, P0 acts first. P0 checks (strategy: c=1.0).
# Now P1 (BR) acts. Legals: ["c", "b"].
# Check option: check-check → flop
#   On flop, P0 acts first.
print("  --- P1 (BR) checks ---")
print("    check-check → flop")
print("    On flop P0 checks first (always check)...")
print("    ...P1 (BR) can check or bet...")

# Actually, the point is to verify whether the BR computation gives sensible results.
# Let me just check what the BR actually computes for a specific deal.
c1, c2 = ("J","H"), ("K","S")
comm = ("Q","H")
v_p0_br = _br_value_single(c1, c2, comm, "", 0, 0, 0, ANTE, check_call_strat)
v_p1_br = _br_value_single(c1, c2, comm, "", 0, 0, 1, ANTE, check_call_strat)
print(f"\n  Deal: P0=JH, P1=KS, comm=QH")
print(f"  BR(P0) = {v_p0_br:.4f}")
print(f"  BR(P1) = {v_p1_br:.4f}")
print(f"  Sum = {v_p0_br+v_p1_br:.4f}")

# Now trace the BR computation for this deal with reduced verbosity
