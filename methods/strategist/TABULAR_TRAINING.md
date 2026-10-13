# Local tabular Monte Carlo + information-set MCTS

This is an independent, **no-LLM** path. Historical generated-code and NLP
Strategist workflows are unchanged. No model/provider, credentials, GTO policy,
API calls or Git writes are needed. Dirty Git status is provenance, not a gate.

## Trained policies (2026-09-12)

| Game | Training games | Batch updates | Learned information sets | Training time | Final batch lookup hit rate |
|---|---:|---:|---:|---:|---:|
| Leduc | 320 | 10 | 333 | 26.16 s | 68.72% |
| Liar's Dice 2×3 | 320 | 10 | 1535 | 48.34 s | 14.16% |
| Goofspiel n=5 | 320 | 10 | 1760 | 74.48 s | 8.19% |

Three processes ran concurrently: about 75 seconds of total training wall time.
Rates count all evaluator calls (real roots and hypothetical leaves), not all
reachable game states. Unseen states still receive seeded random initial values.
In particular, Dice and Goofspiel coverage remains sparse at this fixed budget.
No strength/GTO/convergence claim and no external evaluation games.

Audited policy files, each containing the table and frozen search configuration:

- [Leduc frozen policy](../../results/leduc/strategist_tabular_dev/20260912_193806_tabular_replay_dev_28b2d701/metrics/frozen_policy.json)
- [Dice frozen policy](../../results/liars_dice/strategist_tabular_dev/20260912_193806_tabular_replay_dev_88ad42a4/metrics/frozen_policy.json)
- [Goofspiel frozen policy](../../results/goofspiel/strategist_tabular_dev/20260912_193809_tabular_replay_dev_cc8db3c9/metrics/frozen_policy.json)
- [Validation summary](../../results/leduc/strategist_tabular_dev/20260912_192900_validation_dev/summary.json)

The audited directories are explicitly derived from the original completed
training directories. Initial cross-process replay exposed a floating-point
last-bit difference in a diagnostic reduction over an unordered set. Sorting
the diagnostic keys fixes it. The derived runs replay the exact same 960 saved
trajectories, with **zero new search/training games**. Every learned table entry
is identical to its original. All original files are retained unchanged; the
failed audit and provenance are in the validation logs and derived configs.

## Protocol

- One table per game, with both seats distinguished by `player_id`.
- Exact canonical key: game, own private information, public information, complete
  visible in-game history. No hidden opponent card/dice, unobserved pending bid,
  environment seed or free-text fields. No suit/dice/history abstraction.
- Lazy initialization: SHA256 of initialization seed + game + exact key maps to
  a value in ±0.05. Lookup does not insert an entry. No initial pseudo-count.
- Entry fields: `count`, `total`, `value=total/count`. Values are normalized P0
  terminal returns, independent of which seat's information set is the key.
  Evaluator returns `(value,-value)` in fixed P0/P1 order.
- Each real episode labels each distinct actually visited root once. Never label
  hypothetical MCTS branches with the real trajectory's outcome.
- Both seats search using the **same frozen batch table**. After 32 complete
  games, a new table is constructed from cumulative first-visit MC returns.
  Earlier samples remain in the average as the self-play search policy changes.
- Defaults: 10 rounds ×32 games, environment seeds 1100000–1100319, initialization
  seed 9100, planner seeds 9200/9201; 32 simulations/action, depth12, nodes128,
  uniform action priors, exploration1.4. No separate validation/PK episodes.
- Payoff scales: Leduc net chips /32, Dice ±1, Goofspiel score difference /15.
- The 10×32 episode budget borrows one Strategist paper experimental pattern;
  tabular MC replaces its LLM strategy evolution. It is **not a reproduction**
  of the paper's high-level strategy tree or idea queue.

## Entry points

From the repository root, using the selected Conda environment:

```bash
PYTHON_BIN=/Users/zww/miniconda3/envs/ToMResearch/bin/python bash scripts/strategist-tabular.sh
```

Defaults to three concurrent local workers, one per game. Smaller debug budgets,
`--game leduc|liars_dice|goofspiel|all`, `--workers 1|2|3`, `--dry-run`,
`--output-dir`, `--no-progress`, and search/seed parameters are supported.
`--output-dir` is a parent under `results/`; each invocation creates a new run.
The CLI deliberately limits this local protocol to ≤10 rounds and ≤32 games/round.

Completed or interrupted single-game runs can be resumed with matching settings:

```bash
bash scripts/strategist-tabular.sh --game leduc --resume results/leduc/strategist_tabular_dev/20260912_193806_tabular_replay_dev_28b2d701
```

Use `PYTHON_BIN` as above if the environment is not activated. Completed episodes
are engine-replayed and skipped, with no new search. An interrupted, not-yet-saved
local episode is deterministically recomputed. Missing episodes inside a completed
prefix, altered checksums, changed dependencies/configs and concurrent writers
are rejected. Completed checkpoints and policies are immutable; invocation logs
record skipped/new/remaining counts. `--stop-after` is an absolute game count;
partial batches do not update the table. Remove it to continue to completion.

## Frozen PK inference

Use `FrozenTabularAgent(policy_path).decide(observation)` from
[tabular_training.py](tabular_training.py), or `load_policy()` to get the table,
`SearchConfig`, and planner seed separately. The result is a `SearchDecision`;
its `action` is the canonical engine action and `trace` is serializable.

The agent does fresh information-set MCTS at each decision. Search visits and
backups are temporary; neither known entries nor unseen initial values change
during PK. The adapter has no outcome-based learning hook. Current games retain
uniform hidden priors and repository Goofspiel tie/public-prize semantics.

## Artifacts and validation

Run JSON files use `{"sha256": ..., "payload": ...}` envelopes. This detects
accidental corruption; it is not an adversarial signature. Table `entries` keys
are readable canonical visible-state JSON, not irreversible hashes.

- Configuration: phase, exact spec/seeds, source hashes, git provenance, command.
- Trajectories: every real observation/action, search trace, terminal targets.
- Round checkpoints: complete value table, source-episode hashes, lookup/update metrics.
- Frozen policy: table plus search config.
- Summary and per-invocation logs: completeness, timing, failures, resume counts.

Final regression: **143 passed (3.71s)**, including new-process recovery with
different `PYTHONHASHSEED`. Full engine replay: 960/960 games, 7143 legal actions,
30 batch updates, 999 scientific artifact hashes unchanged on completed resume.
Training, replay and validation subprocesses used a network/credential-read
guard. Source-only lint diagnostics and shell syntax checks passed. No Git
staging, commits, branch changes or historical paid-run resumes.