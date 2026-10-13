# Continuous repeated-opponent protocol

Status: game-frozen v2 offline implementation complete, human review pending
(2026-09-12). No new live/API run was started by this implementation stage.

## Why this entry exists

HM learns an opponent online by retaining natural-language hypotheses and their
prediction scores. Running several one-game processes resets that state and is
therefore NOT a valid test of cross-game adaptation. Historical independent
single-game runs remain valid only as connectivity/flow checks and must not be
retrospectively relabeled continuous.

New entry: `runs/hypothetical_minds_continuous_gto.py`.

## Series identity and state boundary

A series is one game type, one player seat (P0), one opponent policy/checkpoint,
and consecutive game seeds. The default is three games with seeds9100–9102.
The runner constructs exactly one `HypotheticalMindsMethod` and calls:

1. `begin_game(1)` → `prepare_game(...)` → play frozen policy → `finish_game(...)`
2. `begin_game(2)` → prepare from game-1 updates → play frozen policy → finish
3. `begin_game(3)` → prepare from games1–2 updates → play frozen policy → finish

`prepare_game()` runs before any public action and freezes the hypothesis IDs,
their start-of-game values, validation status and one high-level strategy. During
play, action calls execute that strategy and prediction calls cover only the
selected action's next public response. `observe()` defers scoring. At the game
boundary, `finish_game()` applies all prospective predictions in chronological
order exactly once. Thus game i evidence can affect game i+1 but never game i.

Retained across games:

- hypothesis IDs/text and Rescorla-Wagner values;
- correct/incorrect counts and all once-only prediction evaluations;
- currently validated hypothesis status (derived from retained values);
- public opponent evidence count and censored prediction records;
- terminal ego-visible summaries and payoff in `previous_games` memory.

Reset at each game boundary:

- engine state, cards/dice/hands and all current private information;
- current game public history;
- pending prediction (must already be resolved/censored);
- deferred prediction ledger (must already be applied and cleared);
- frozen game policy (copied into completed-game memory, then cleared);
- latest decision output cache.

Opponent strategy/checkpoint metadata must be identical across all games.
Each game recreates the fixed local GTO actor with its recorded seed; no hidden
GTO state or private observation is shared with HM. Large Goofspiel checkpoints
are loaded lazily one game at a time to avoid retaining three copies.

## Artifacts and auditability

One run directory contains:

- one immutable config with the complete seed schedule and **total series**
  request cap;
- `trajectories/P0/game_0001.json` etc. Each checkpoint stores
  `method_state_before` and `method_state` plus their digests and cumulative
  hypothesis/evaluation/memory counts;
- calls keyed by actual game index and environment step;
- incremental ego-visible steps, output-validation records and invocation logs;
- summary containing the final retained method state and per-game metrics.

For game `i>1`, `method_state_before` must equal the prior checkpoint's final
method state and contain exactly `i-1` previous-game memories. Every LLM prompt
in game i receives exactly `i-1` previous games. The hypothesis pool is not
reconstructed from payoff alone; cached replay reproduces the original model
outputs and every prediction update in order.

## Resume and failure semantics

`--resume` replays all completed games from the beginning using successful
cached calls, thereby rebuilding the one retained method object. A completed
series adds zero physical calls on resume. A partial next game then resumes at
its first missing call. Saved failures require explicit `--retry-failed`;
ambiguous `started` calls block automatic resume. Source/spec/opponent identity
and the total request cap cannot change on resume. No seed replacement or
outcome filtering is allowed.

Live support is gated by explicit live approval; it does not require Git state
or a repository-specific offline evidence artifact. The code does not authorize
a budget. Default total cap is450 attempts for the entire three-
game series, not450 per game. The campus wrapper remains one physical request;
the shared journal controls transient retries, and HM controls schema retries.

## Game-frozen v2 offline evidence

Validation evidence:
`results/api/hypothetical_minds_personal_dev/20260912_game_frozen_validation_dev/`.

- 156 HM tests passed in 7.14s; editor diagnostics were clear. Output failures
  receive initial+3 stage-only attempts; exhausted action output is explicitly
  recorded and selects a deterministic replay-compatible random legal action.
  Infrastructure/auth/budget/ambiguous failures still stop.
- Tests prove values/evaluations remain unchanged after every in-game transition,
  each game has one plan and at most one hypothesis-generation call, strategy and
  candidate IDs are constant within a game, deferred predictions apply only at
  `finish_game()`, hidden-information guards remain intact, and completed resume
  uses cached calls only.
- Three-game scripted runs against repository approximate GTO all completed:

| Game | Run | Calls | Predictions scored by game | Memory after | Resume new calls |
| --- | --- | ---: | --- | --- | ---: |
| Leduc | `20260912_134129_hm_continuous_gto_personal_offline_dev` | 27 | 2,4,6 | 1,2,3 | 0 |
| Dice | `20260912_134130_hm_continuous_gto_personal_offline_dev` | 30 | 2,4,6 | 1,2,3 | 0 |
| Goofspiel | `20260912_134131_hm_continuous_gto_personal_offline_dev` | 51 | 5,10,15 | 1,2,3 | 0 |

Every game had exactly one plan, at most one new hypothesis, one candidate-set
identity and one strategy identity. Increasing prediction counts across games
reflect the frozen candidate pool growing from one to two to three hypotheses,
not within-game adaptation. Fixture outputs and payoffs are not efficacy evidence.

## Historical v1 offline evidence

Validation run:
`results/liars_dice/hypothetical_minds_dev/20260911_213640_continuous_runner_validation_dev/`.

- 14 dedicated continuous tests passed, including same-object state chaining,
  clean independent-series initialization, one-game pause/resume equivalence,
  completed resume with zero new calls, partial game failure/recovery, opponent
  identity and budget gates, and live gate before key access.
- Expanded HM/shared regression: **232 passed (18.48s)**, editor errors zero.
- Real game engines + real local GTO checkpoints, three continuous scripted
  games each:

| Game | Run | Calls | Net by game | Hypotheses after | Evaluations after | Memory after |
| --- | --- | ---: | --- | --- | --- | --- |
| Leduc | `20260911_214332_hm_continuous_gto_offline_dev` | 55 | +7,-5,+1 | 3,5,6 | 3,10,21 | 1,2,3 |
| Dice | `20260911_214335_hm_continuous_gto_offline_dev` | 78 | -1,-1,-1 | 3,5,7 | 3,10,21 | 1,2,3 |
| Goofspiel | `20260911_214339_hm_continuous_gto_offline_dev` | 120 | +1,-7,0 | 5,10,15 | 15,45,75 | 1,2,3 |

All three complete resumes produced zero new physical calls and left prompt and
trajectory bytes unchanged. Fixture payoffs are not efficacy evidence.

Implementation changed only HM-owned `method.py` to clear last-decision cache at
a new game boundary, plus the new runner/test/document. It did not alter games,
other methods, shared APIs, Git branch/index, or historical live artifacts.

## Correct interpretation

- The previous three independent dice live games cannot be combined after the
  fact; their decisions were made from separate empty pools.
- This new protocol is the appropriate mechanism for a future repeated-opponent
  live test. Such a run must be new, explicitly approved, and reported separately.
- Three games are still a dev-debug sample. Retention being implemented and
  audited does not establish that the learned hypotheses improve reward.