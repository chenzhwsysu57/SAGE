# Fixed-budget Strategist port

## Stage: fixed_budget_tree_dev (2026-09-12)

Status: implementation and full offline preflight completed; awaiting human review.
No API, real-key access, branch changes, staging or commits.
User asks whether the agreed paper-budget mode exists and requests implementation.
This stage adds independent files only; old1000/50 and seven-call protocols and
their historical results are NOT modified or relabeled.

## Attribution and budget

Reference: https://arxiv.org/html/2408.10635v3 section3.5 (GOPS feedback experiment).
The selected budget is10 rounds x32 complete feedback episodes =320/game.
The paper does NOT fully specify all the call/allocation choices below in that
sentence. We explicitly choose the following reproducible port, not exact official
code reproduction and not the separate40-strategy experiment:

- Two independently prompted initial functions; no bootstrap game episodes.
- Each round selects two DISTINCT archived parents, performs one joint reflection,
  generates two ideas and two children. Round1 reflection explicitly has no data.
- Global UCB idea queue; ideas may transfer across branches; source and target
  parents are both recorded. All archived strategies retain selection probability.
- Parent selection uses within-cohort midrank priorities, expansion UCB and age.
  Raw returns from different historical cohorts are not compared directly.
- Cohort = two parents + two children. Four-by-four directed matches including
  self x two fresh chance seeds =32 episodes/round, including ALL candidate scoring.
  Identical search budgets for both seats, equal opponent weights and both seats.
- Each policy has four return observations/opponent including self; paired games
  share seeds, NOT32 independent chance samples. No8-game promotion gate.
- Every candidate stays in the tree; no accept/reject champion chain.10rounds
  generate20 children,22functions total.52logical LLM calls, up to208physical
  attempts including at most3format corrections/call; no lifetime request cap.
- No extra development games, early stop on payoff, convergence rule, or automatic PK.
  Final selection uses ONLY four strategies' scores on the common final cohort.
  It is training-based selection, NOT independent validation or best-of-all22 claim.

## Method and game boundary

- LLM returns restricted Python function SOURCE evaluated by the existing AST
  interpreter (never exec). Branches, arithmetic, bounded loops and intermediate
  values can change; no fixed five-weight template. The accepted language is still
  narrower than Python; no claim of full paper expression space.
- Existing observation-keyed PUCT and game adapters unchanged. Uniform hidden
  priors, most-visited root selection and omitted AppendixF prior mixing remain
  known deviations. This stage matches the selected episode budget, NOT every
  MCTS detail or guaranteed optimal settings.
- Key feedback contains actor's actual observation including own private info,
  action, initial estimate, intermediate values, search estimate and terminal
  normalized reward. No real opponents' unrevealed data or hypothetical hidden
  state presented as observation. Top3per parent, not growing full histories.
- Leduc JJQQKK/net chips; dice2x3 wild1/net +/-1; Goofspiel n5/final score difference,
  public future prizes and split ties. Do not copy paper GOPS carry-over ties.
- Schema and recorded-state probes do not play full episodes; count them separately
  from320games. Runtime or transport failure stops, no random fallback/substitution.

## Deliverables and acceptance

- Independent CLI and shell with dry preview by default, explicit offline/live.
- Config before generation, incremental trajectories/schedules/policies/rounds,
  all semantic/physical IO, usage, final loadable code policy and per-game progress.
- Explicit safe-boundary pause/resume; verified checkpoints prevent missing
  successful calls from being resent. Completed replay zero requests/artifact changes.
- Failure/length/started/partial-game/source/schema changes require manual review.
- Live requires same-source/full-protocol offline evidence and approval.
  Per explicit user instruction, Git cleanliness is NOT an execution requirement:
  commit and dirty_worktree are recorded as provenance only. No manual-run flag,
  automatic staging, commits, branch changes or cleanup are needed.
- Validate320/32 accounting,2new policies each round,last candidate evaluation,
  multi-parent exploration,global idea reuse,same-cohort scoring,actor visibility,
  three real engines,format corrections,usage parity,replay and corruption guards.

## Entry points

- scripts/strategist-fixed-budget.sh forwards to runs/strategist_fixed_budget.py.
- Required --game leduc|liars_dice|goofspiel. No execution flag or --dry-run gives
  a plan preview without files, keys or API access. --offline executes fixtures.
- Defaults:10rounds,32episodes/round,train-seed1100000,planner-seed17,
  search simulations32/depth12/nodes128,format-retries3.
- --rounds1..9 exists only for small offline checks, explicitly not full budget.
- --output-dir stays in results/<game>/strategist_fixed_budget, relative to repo.
- --stop-after-games pauses before the next local game and, at round boundaries,
  before further model calls. --resume requires intact pause/completion receipts.
- --no-progress suppresses per-game and per-round progress. Prompts always saved.
- --live-api additionally requires --approved-live-api,--approval-by,
  --approved-real-measurement,--dev-evidence; --key-file may
  name the personal dotenv credential file. No unlimited-network-retry option.
- The three earlier scripts/strategist-{leduc,ld,goof}.sh retain their old series
  workflow. They DO NOT automatically switch protocols or load this policy.
- load_fixed_policy(path,game=...,evaluation_seeds=...) validates source/hash/game
  and rejects training-seed reuse; it returns Strategy and provenance metadata.
  External PK integration is not part of this training-only stage.

## Evidence (2026-09-12)

Root:results/api/strategist_fixed_budget_dev/20260912_182059_validation_dev/.

- Focused41 passed(34.11s); added4explicit coverage cases; complete Strategist
  suite498 passed(82.96s), no exclusions/failures. config.json and
  logs/regression_config.json contain exact commands; logs/focused.xml and
  logs/regression.xml retain test evidence. Editor diagnostics zero.
- New code tests cover raw-source correction success/exhaustion, runtime failure,
  length/HTTP500/interruption without retry, physical IO and usage via MockTransport,
  global idea reuse, arbitrary-CWD shell preview, source/config/artifact changes,
  missing-successful-call rejection, init/finalization failure and safe replay.
- logs/preflight.py launches the new shell from a non-repository CWD, both default
  arguments and full320 budget. Child sitecustomize denies socket connections;
  no real keys. Exact commands/source+protection hashes saved before execution in
  logs/preflight_config.json; report logs/preflight_summary.json.
- Leduc:results/leduc/strategist_fixed_budget/20260912_182406_fixed_budget_dev_a19ppnra/,
  320games,2186legal actions,15distinct parents selected,15cross-branch idea uses.
- Dice:results/liars_dice/strategist_fixed_budget/20260912_182406_fixed_budget_dev_3c6cuhvg/,
  320games,2050legal actions,13distinct parents selected,15cross-branch idea uses.
- Goofspiel:results/goofspiel/strategist_fixed_budget/20260912_182406_fixed_budget_dev_skdns80b/,
  320games,3200legal actions,15distinct parents selected,15cross-branch idea uses.
- Each has10rounds,22functions,20ideas,52scripted calls; no dev/test games.
  Total960games/7436legal actions. All3CLI completed replays skip320games and add
  zero calls; all1563core artifact hashes and original summaries remain unchanged.
  Old files/historical live artifacts and source hashes unchanged during preflight.
- Counts demonstrate mechanics only, NOT real strategy improvement or paper
  performance replication. Selected fixture policy IDs must not be presented as
  live-trained strategies. No paid training executed during this implementation.
- Read-only external subagent review was unavailable(authentication401); no
  independent review pass is claimed. Runtime tests and explicit manual review
  of budget/scoring/persistence paths are the available validation evidence.

## User-run correction (2026-09-12)

Removed the fixed-budget runner's global dirty-worktree rejection at the user's
explicit request. Reproducibility records still include commit, dirty status,
source hashes and exact configuration. All budget/secret/response/resume checks
remain. Earlier498-test/full320-game evidence predates this edit and is retained
unchanged; its source fingerprints cannot authorize a new-version live run.
Refresh offline evidence for this source version before using --dev-evidence.
The historical1000/50 runner is a separate protocol and was not changed here.

Validation after removing the Git blocker:45 affected tests passed(38.69s),
three actual-engine offline10x32 runs completed with52scripted calls/game,
same-source evidence validation passed. No new live/API/key access or Git writes.
Updated --dev-evidence directories:
- results/leduc/strategist_fixed_budget/20260912_184551_fixed_budget_dev_h2p0mm4w
- results/liars_dice/strategist_fixed_budget/20260912_184551_fixed_budget_dev_eibqzbme
- results/goofspiel/strategist_fixed_budget/20260912_184551_fixed_budget_dev_2tcdf6kl
Commands using the old182406 evidence must use these refreshed paths instead.
No --manual-run flag is required; dirty_worktree is recorded without rejection.