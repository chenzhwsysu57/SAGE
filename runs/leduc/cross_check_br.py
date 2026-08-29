"""Cross-check BR computation: verify zero-sum and compare with self-play."""
from __future__ import annotations
import sys, itertools
sys.path.insert(0, ".")

from methods.cfr import CFRPlusTrainer, DECK, ANTE, all_community_cards
from runs.leduc.compute_exploitability import best_response_value, _br_value

trainer = CFRPlusTrainer(iterations=10_000)
trainer.train()
strat = trainer.strategy

# Test 1: For uniform random (1-iteration) strategy, check BR symmetry
trainer0 = CFRPlusTrainer(iterations=1)
trainer0.train()
strat0 = trainer0.strategy

br0_1 = best_response_value(strat0, 0)
br1_1 = best_response_value(strat0, 1)
print(f"=== 1 iter (≈uniform) ===")
print(f"BR(P0)={br0_1:+.4f}  BR(P1)={br1_1:+.4f}  sum={br0_1+br1_1:+.4f}")
print(f"If BR has bias b: sum = b0+b1 ≈ 2b (expect ~0 if unbiased)\n")

# Test 2: BR symmetry for 10k strategy
br0 = best_response_value(strat, 0)
br1 = best_response_value(strat, 1)
print(f"=== 10k iterations ===")
print(f"BR(P0)={br0:+.4f}  BR(P1)={br1:+.4f}  sum={br0+br1:+.4f}")

# Test 3: Compute BR for both players of a single deal, should be opposite
print(f"\n=== Per-deal BR check ===")
for c1, c2 in list(itertools.permutations(DECK, 2))[:3]:
    for comm in all_community_cards(c1, c2):
        v0 = _br_value(c1, c2, comm, "", 0, 0, 0, ANTE, strat)  # BR=P0
        v1 = _br_value(c1, c2, comm, "", 0, 0, 1, ANTE, strat)  # BR=P1
        print(f"  P0={c1[0]}{c1[1]}, P1={c2[0]}{c2[1]}, K={comm[0]}: BR(P0)={v0:+.4f} BR(P1)={v1:+.4f} sum={v0+v1:+.4f}")

# Test 4: Self-play expected value (both follow strategy, no BR)
print(f"\n=== Self-play EV (both follow strategy) ===")
def self_play_ev():
    total = 0.0
    n = 0
    for c1, c2 in itertools.permutations(DECK, 2):
        for comm in all_community_cards(c1, c2):
            # Walk the tree with BOTH players following the strategy
            val = _follow_strategy(c1, c2, comm, "", 0, 0, strat, ANTE, ANTE)
            total += val
            n += 1
    return total / n

def _follow_strategy(c1, c2, comm, history, street, cur_player, strategy, p0_contrib, p1_contrib):
    """Walk tree with both players following the strategy. Returns P0's net profit."""
    from methods.cfr import is_terminal, next_round, get_legals, pot_from_history, compare, PREFLOP_BET, FLOP_BET, card_str

    if is_terminal(history, street):
        if history[-1] == "f":
            # Opponent folded
            return pot_from_history(history, street) - (p0_contrib if cur_player == 0 else p1_contrib)
        w = compare(c1, c2, comm)
        my = p0_contrib if cur_player == 0 else p1_contrib
        pot = pot_from_history(history, street)
        if w == -1: return 0
        if w == cur_player: return pot - my
        return -my

    term, ns, nc, nh = next_round(history, street, c1, c2, known_comm=comm)
    if term:
        return _follow_strategy(c1, c2, nc, nh, ns, cur_player, strategy, p0_contrib, p1_contrib)
    if ns > street and not term:
        v = _follow_strategy(c1, c2, nc, nh, ns, 0, strategy, p0_contrib, p1_contrib)
        if cur_player == 1: return -v
        return v

    legals = get_legals(history, street)
    if not legals: return 0

    pcard = c1 if cur_player == 0 else c2
    cr = card_str(comm) if street == 1 else "?"
    info = f"{pcard[0]}|{cr}|{history}|{street}"
    strat = strategy.get(info, {})

    total = 0.0
    accounted = 0.0
    for a in legals:
        prob = strat.get(a, 0.0)
        if prob > 0:
            nh2 = history + a
            add = PREFLOP_BET if street == 0 else FLOP_BET if a in ("b","r","k") else 0
            if cur_player == 0:
                val = _follow_strategy(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib)
            else:
                val = _follow_strategy(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add)
            val = -val
            if a == "f":
                val = -p0_contrib if cur_player == 0 else -p1_contrib
            total += prob * val
            accounted += prob

    if accounted < 1.0 - 1e-12:
        remaining = 1.0 - accounted
        unvisited = [a for a in legals if strat.get(a, 0.0) == 0.0]
        if unvisited:
            uniform = remaining / len(unvisited)
            for a in unvisited:
                nh2 = history + a
                add = PREFLOP_BET if street == 0 else FLOP_BET if a in ("b","r","k") else 0
                if cur_player == 0:
                    val = _follow_strategy(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib)
                else:
                    val = _follow_strategy(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add)
                val = -val
                if a == "f":
                    val = -p0_contrib if cur_player == 0 else -p1_contrib
                total += uniform * val

    return total

ev = self_play_ev()
print(f"  Self-play P0 EV = {ev:+.4f}")
print(f"  BR(P0) = {br0:+.4f}  (P0 gains {br0-ev:+.4f} by deviating)")
print(f"  BR(P1) = {br1:+.4f}")
print(f"  In Nash: both BR values should approach 0")

# If self-play EV ≈ BR, the strategy is near-optimal for P0 against P1
if abs(ev - br0) < 0.5:
    print(f"  ✓ P0's strategy is near-optimal against P1")
else:
    print(f"  ✗ P0 can improve {br0-ev:.2f}/hand")
