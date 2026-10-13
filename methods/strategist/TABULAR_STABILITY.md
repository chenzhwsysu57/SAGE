# Bounded value-stability continuation

This independently extends the [original local learner](TABULAR_TRAINING.md).
The MC update rule, exact visible-state keys, normalization, batch size, MCTS
budget and planner seeds are unchanged. No LLM, API, credentials, GTO training,
external PK, or Git writes. Parent runs are read-only, not resumed or overwritten.

## Predeclared default experiment

- Inherit each game's 320-game frozen table, including accumulated counts/sums.
- Add at most 2880 games (90 updates); cumulative cap 3200 games /100 updates.
- New environment seeds 1100320–1103199; init9100 and search9200/9201 inherited.
- Each batch uses a frozen table for 32 games, then applies cumulative first-visit
  terminal-return MC updates. This is not Bellman/TD bootstrap training.
- Every 10 additional updates (320 games), compare against the previous check's
  table. First reference is the 320-game parent table.
- Fixed panel = every key present in that parent table, equal weight. Its hash
  and size are saved before training. Newly discovered keys are learned but not
  added to the stop panel.
- Numerical pass: maximum absolute normalized P0 value change across the **whole
  fixed panel** is ≤0.1. It is not a mean or percentile threshold. Normalization
  makes 0.1 equivalent to Leduc3.2 chips, Dice0.1 expected reward, Goofspiel1.5 points.
- Coverage pass: at least90% of fixed panel keys each have ≥5 cumulative samples
  and ≥1 fresh sample since the previous check. Untouched keys cannot alone produce
  a passing interval. These are engineering safeguards, not accuracy confidence bounds.
- Both conditions must pass on3 consecutive checks. Any failed check resets the
  streak. Earliest possible stop is after960 new games (1280 total /40 total updates).
- Stop reasons: `value_stable` or `budget_exhausted_not_stable`. Always preserve
  all observations, including failures and non-monotonic changes. No automatic
  cap extension or post-result threshold revision.

The maximum is deliberately sensitive to rare keys, while this coverage gate is
strict for an exact full-history representation. Failure to pass can reflect
poor revisit coverage as well as genuine value movement. Report both separately.
Also report the first **numeric-only** passing check, explicitly not as convergence.
Even a full pass only describes the original panel, not all reachable information
sets, newly discovered states, accurate current-policy values, or Nash equilibrium.
Cumulative MC averages retain old policy data and have decreasing effective steps.

## Execution and artifacts

```bash
PYTHON_BIN=/Users/zww/miniconda3/envs/ToMResearch/bin/python bash scripts/strategist-tabular-continue.sh --from-run results/leduc/strategist_tabular_dev/20260912_193806_tabular_replay_dev_28b2d701 --threshold 0.1 --max-total-games 3200
```

Supply the corresponding parent directory for Dice or Goofspiel. Output defaults
to a new timestamped directory under `results/<game>/strategist_tabular_stability_dev/`.
`--dry-run` previews without writes. `--output-dir` changes the output parent under
results; `--stop-after` pauses at an additional-game count, not a cumulative count.

Resume the **continuation run**, not the parent, using `--resume` and matching
stability options. Original search/MC settings are inherited, not overridden.
Completed resume engine-replays saved trajectories and recomputes batch updates
and stop checks; it cannot add games after the stop. Missing prefix artifacts,
configuration changes, source changes, altered parent identities, or concurrent
writers are rejected. A not-yet-saved interrupted local game is recomputed.

Artifacts retain checksum envelopes and include:

- Config with fixed panel identity, budget, stop rule, command, source/provenance.
- Exact copied `metrics/initial_policy.json`, plus parent identity hashes.
- Incremental per-game trajectories and per-update full-table checkpoints.
- Stability check records (every10 updates) within the checkpoints and summary.
- Final frozen policy compatible with `FrozenTabularAgent`.
- Invocation receipts with duration, skipped/new counts, and failure type.

`additional_mcts_simulations` counts the sum of per-decision simulation budgets,
**not complete games**. Most MCTS paths terminate at heuristic leaves. The summary
also records actual simulated action transitions separately.

The continuation-only `Snapshot` caches the fingerprint of an immutable table;
this removes repeated full-table serialization without changing evaluator values
or gameplay. Exact trajectory-equivalence tests cover all three games. Original
learner files and historical source fingerprints remain unchanged.

## Validation

Pre-run regression:169 passed (6.95s), including0.1 equality/just-over boundary,
maximum-vs-mean distinction, no-visit false stability, cumulative/fresh visit
requirements, parent hashes, batch continuation, forced early-stop wiring,
interruption, completed zero-search resume, and distinct-process PYTHONHASHSEED.
Forced stop tests prove control flow, not empirical achievement of stability.

Protocol and execution evidence:
[validation config](../../results/leduc/strategist_tabular_stability_dev/20260912_195456_validation_dev/config.json).

## Observed result (2026-09-12)

All three runs stopped at the predeclared cap, with
`budget_exhausted_not_stable`. Each added2880 complete games /90 updates for
3200 total games /100 updates. No automatic budget extension.

| Game | Final max absolute value change | Final eligible panel coverage | First numeric-only pass | Entries | Continuation seconds |
|---|---:|---:|---|---:|---:|
| Leduc | 0.406250 | 35.14% | 2560 games /80 updates:0.061090 | 580 | 210.24 |
| Liar's Dice | 1.000000 | 3.52% | None at checked intervals | 11757 | 453.45 |
| Goofspiel | 0.247619 | 9.09% | None at checked intervals | 6904 | 706.23 |

Leduc also had0.075 at2880 games, before rebounding at3200. Therefore even
ignoring coverage, it did not sustain3 consecutive passing checks. No game met
the coverage condition in any check. The maximum threshold is sensitive to
rarely revisited keys; the means are much smaller, but changing from maximum
to mean afterward would be a different stopping rule and was not done.

Final-batch evaluator lookup hit rates improved from68.72/14.16/8.19% after320
games to80.27/34.31/32.83%. These are call-frequency-weighted search lookups;
they are NOT the equally weighted, fresh-visit-qualified panel coverage above.

| Game | Added MCTS simulation paths | Cumulative MCTS paths including initial320 games |
|---|---:|---:|
| Leduc | 419008 | 469120 |
| Liar's Dice | 659232 | 735296 |
| Goofspiel | 921600 | 1024000 |

The three continuations ran concurrently in about11.8 minutes wall time.
8640 new complete self-play games,62495 real legal actions,1999840 new MCTS
paths and4132101 simulated action transitions were recorded. There were zero
API requests, external PK games, or extra validation games.

Final frozen policies (not certified converged):

- [Leduc](../../results/leduc/strategist_tabular_stability_dev/20260912_200438_value_stability_dev_c5313d65/metrics/frozen_policy.json)
- [Liar's Dice](../../results/liars_dice/strategist_tabular_stability_dev/20260912_200438_value_stability_dev_f6d9c6d2/metrics/frozen_policy.json)
- [Goofspiel](../../results/goofspiel/strategist_tabular_stability_dev/20260912_200438_value_stability_dev_60f794fa/metrics/frozen_policy.json)
- [Audit summary and all27 checks](../../results/leduc/strategist_tabular_stability_dev/20260912_195456_validation_dev/summary.json)

Audit reconstructed every saved game, MC update and stability check in a fresh
process with search explicitly disabled. Completed resume skipped2880 games
per game, generated0;8925 scientific files and1008 parent files stayed identical.
The original320-game learner sources and parent directories were not modified.