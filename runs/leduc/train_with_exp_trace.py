"""Train CFR+ and compute exploitability at intervals to show convergence curve."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")

from methods.sage.cfr import CFRPlusTrainer, ANTE, DECK
from runs.leduc.compute_exploitability import best_response_value
import itertools

# Use in-memory trainer for intermediate strategy extraction
trainer = CFRPlusTrainer(iterations=200_000)  # will do partial train

TOTAL_ITERS = 200_000
EVAL_INTERVAL = 10_000
deals = list(itertools.permutations(DECK, 2))

print(f"Training {TOTAL_ITERS} iterations, evaluating every {EVAL_INTERVAL}")
print(f"{'Iter':>7}  {'BR(P0)':>8}  {'BR(P1)':>8}  {'Exploit':>8}  {'SelfEV':>8}  {'SumBR':>8}  {'Time':>8}")
print("-" * 70)

t_start = time.perf_counter()

for i in range(TOTAL_ITERS):
    for c1, c2 in deals:
        for comm in itertools.permutations([c for c in DECK if c != c1 and c != c2], 1):
            comm_c = comm[0]
            trainer._walk(c1, c2, comm_c, "", 0, 0, 1.0, 1.0, ANTE, ANTE)

    if (i + 1) % EVAL_INTERVAL == 0:
        # Build strategy from current strategy_sum
        strategy = {}
        for info, sums in trainer.strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                strategy[info] = {a: s / total for a, s in sums.items()}
            else:
                acts = list(sums.keys())
                k = len(acts)
                strategy[info] = {a: 1.0 / k for a in acts} if k else {}

        t_eval = time.perf_counter()
        br0 = best_response_value(strategy, 0)
        br1 = best_response_value(strategy, 1)
        exp = (br0 + br1) / 2.0
        dt = time.perf_counter() - t_eval

        # Also compute self-play EV quickly (single walk through all deals)
        # Use the _walk with both players following the strategy
        # Actually just use the average deal value from _walk with uniform reach
        # Re-use the strategy for self-play
        import math
        from methods.sage.cfr import (
            is_terminal, next_round, get_legals, pot_from_history,
            PREFLOP_BET, FLOP_BET, compare, card_str, all_community_cards
        )

        def self_play_ev(strat):
            total = 0.0
            n = 0
            for c1, c2 in itertools.permutations(DECK, 2):
                for comm in all_community_cards(c1, c2):
                    val = _follow_strategy(c1, c2, comm, "", 0, 0, strat, ANTE, ANTE)
                    total += val
                    n += 1
            return total / n

        def _follow_strategy(c1, c2, comm, history, street, cur_player, strategy, p0_contrib, p1_contrib):
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
            st = strategy.get(info, {})

            total = 0.0
            accounted = 0.0
            for a in legals:
                prob = st.get(a, 0.0)
                if prob > 0:
                    nh2 = history + a
                    add = PREFLOP_BET if street == 0 else FLOP_BET if a in ("b","r","k") else 0
                    if cur_player == 0:
                        val = _follow_strategy(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib)
                    else:
                        val = _follow_strategy(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add)
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
                            val = _follow_strategy(c1, c2, comm, nh2, street, 1, strategy, p0_contrib+add, p1_contrib)
                        else:
                            val = _follow_strategy(c1, c2, comm, nh2, street, 0, strategy, p0_contrib, p1_contrib+add)
                        val = -val
                        total += uniform * val

            return total

        sev = self_play_ev(strategy)
        elapsed = time.perf_counter() - t_start
        print(f"{i+1:>7,}  {br0:>+8.4f}  {br1:>+8.4f}  {exp:>8.4f}  {sev:>+8.4f}  {br0+br1:>+8.4f}  {elapsed:>7.0f}s")

# Final build
for info, sums in trainer.strategy_sum.items():
    total = sum(sums.values())
    if total > 0:
        trainer.strategy[info] = {a: s / total for a, s in sums.items()}
    else:
        acts = list(sums.keys())
        k = len(acts)
        trainer.strategy[info] = {a: 1.0 / k for a in acts} if k else {}
trainer._trained = True

print(f"\nFinal strategy: {len(trainer.strategy)} infosets")
print(f"Total time: {time.perf_counter() - t_start:.0f}s")
