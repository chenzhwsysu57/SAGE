"""Joint round walk N=5 — 200K iterations with avg strategy inspection."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.sage.goofspiel_cfr import GoofspielCFRTrainer, hand_to_mask, regret_matching_strategy

t = GoofspielCFRTrainer(n=5, iterations=1, seed=42, cfr_plus=False)

mask_all = hand_to_mask(set(range(1, 6)))

for i in range(500):
    for _ in range(1000):
        traverser = t.rng.randint(0, 1)
        state = t._initial_state()
        t._walk(state, 1.0, 1.0, traverser)

    if (i + 1) % 10 == 0:
        # Build snapshot from average strategy
        snapshot = {}
        for infoset, sums in t.strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                snapshot[infoset] = {k: v / total for k, v in sums.items()}

        br0 = t._br_player_value(0, snapshot)
        br1 = t._br_player_value(1, snapshot)
        expl = (br0 + br1) / 2.0

        # Average strategy at round 1
        s0 = snapshot.get((0, mask_all, ()), {})
        s1 = snapshot.get((1, mask_all, ()), {})

        elapsed = (i+1)*1000
        s0_disp = {k: f"{v:.3f}" for k, v in sorted(s0.items())}
        s1_disp = {k: f"{v:.3f}" for k, v in sorted(s1.items())}
        print(f"{elapsed:6d}: BR_0={br0:.4f} BR_1={br1:.4f} expl={expl:.4f} "
              f"ratio={br1/max(br0,1e-10):.2f} | "
              f"P0={s0_disp} P1={s1_disp}")
