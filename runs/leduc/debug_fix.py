"""Debug: dump strategy for all infosets and trace per-deal exploitability."""
from __future__ import annotations
import sys, itertools, json
sys.path.insert(0, ".")

from methods.sage.cfr import CFRPlusTrainer, RANKS, DECK, ANTE, PREFLOP_BET, FLOP_BET, compare, card_str, all_community_cards, is_terminal, next_round, get_legals, pot_from_history
from runs.leduc.compute_exploitability import best_response_value

trainer = CFRPlusTrainer(iterations=5_000)
trainer.train()

# Dump all strategies
print("=== Preflop strategies (sorted by card) ===")
preflop = {k: v for k, v in sorted(trainer.strategy.items()) if "|?|" in k and "|0" in k}
for k, v in preflop.items():
    print(f"  {k}: {v}")

print(f"\n=== Flop strategies ===")
flop = {k: v for k, v in sorted(trainer.strategy.items()) if "|1" in k}
for k, v in flop.items():
    print(f"  {k}: {v}")

# Check per-deal exploitability
print(f"\n=== Per-deal exploitability ===")
for c1, c2 in itertools.permutations(DECK, 2):
    v0 = best_response_value(trainer.strategy, br_player=0)
    break  # just check single deal

# Let me trace a specific deal manually
c1, c2 = ("J", "H"), ("Q", "S")
comm = ("K", "H")

print(f"\n=== Manual trace: P0={c1}, P1={c2}, comm={comm} ===")

def _infoset_key(history, street, cur_player, c1, c2, comm):
    private_card = c1 if cur_player == 0 else c2
    comm_rank = card_str(comm) if street == 1 else "?"
    return f"{private_card[0]}|{comm_rank}|{history}|{street}"

def _who_folded(history, street):
    seg = history.split("d")[street] if street < len(history.split("d")) else ""
    return (len(seg) - 1) % 2

def _terminal_value(history, street, br_player, c1, c2, comm, br_bets):
    if history[-1] == "f":
        folding_player = _who_folded(history, street)
        if folding_player == br_player:
            return -float(br_bets)
        else:
            return float(pot_from_history(history, street))
    w = compare(c1, c2, comm)
    pot = pot_from_history(history, street)
    if w == -1: return 0
    if w == br_player: return pot
    return -pot

def _after_round_transition(c1, c2, comm, history, street, br_player, br_bets, strategy):
    term, new_st, new_comm, new_hist = next_round(history, street, c1, c2, known_comm=comm)
    if not (new_st > street or term):
        return None
    if term and new_st == street:
        return _terminal_value(new_hist, new_st, br_player, c1, c2, new_comm, br_bets)
    if new_st > street:
        return _br_value(c1, c2, new_comm, new_hist, new_st, 0, br_player, br_bets, strategy)
    return None

def _br_value(c1, c2, comm, history, street, cur_player, br_player, br_bets, strategy):
    if is_terminal(history, street):
        return _terminal_value(history, street, br_player, c1, c2, comm, br_bets)
    r = _after_round_transition(c1, c2, comm, history, street, br_player, br_bets, strategy)
    if r is not None:
        return r
    legals = get_legals(history, street)
    if not legals:
        return 0
    bet_amt = PREFLOP_BET if street == 0 else FLOP_BET
    if cur_player == br_player:
        best = -float("inf")
        for a in legals:
            nh = history + a
            if a == "f":
                val = -float(br_bets)
            elif a in ("b", "r", "k"):
                val = _br_value(c1, c2, comm, nh, street, 1 - cur_player, br_player, br_bets + bet_amt, strategy)
            else:
                val = _br_value(c1, c2, comm, nh, street, 1 - cur_player, br_player, br_bets, strategy)
            if val > best:
                best = val
        return best
    else:
        info = _infoset_key(history, street, cur_player, c1, c2, comm)
        strat = strategy.get(info, {})
        total = 0.0
        accounted = 0.0
        for a in legals:
            prob = strat.get(a, 0.0)
            if prob > 0:
                nh = history + a
                if a == "f":
                    val = float(pot_from_history(nh, street))
                elif a in ("b", "r", "k"):
                    val = _br_value(c1, c2, comm, nh, street, 1 - cur_player, br_player, br_bets, strategy)
                else:
                    val = _br_value(c1, c2, comm, nh, street, 1 - cur_player, br_player, br_bets, strategy)
                total += prob * val
                accounted += prob
        if accounted < 1.0 - 1e-12:
            remaining = 1.0 - accounted
            unvisited = [a for a in legals if strat.get(a, 0.0) == 0.0]
            if unvisited:
                uniform = remaining / len(unvisited)
                for a in unvisited:
                    nh = history + a
                    if a == "f":
                        val = float(pot_from_history(nh, street))
                    elif a in ("b", "r", "k"):
                        val = _br_value(c1, c2, comm, nh, street, 1 - cur_player, br_player, br_bets, strategy)
                    else:
                        val = _br_value(c1, c2, comm, nh, street, 1 - cur_player, br_player, br_bets, strategy)
                    total += uniform * val
        return total

br_val = _br_value(c1, c2, comm, "", 0, 0, 0, ANTE, trainer.strategy)
print(f"  BR(P0) for this single deal = {br_val:.4f}")
