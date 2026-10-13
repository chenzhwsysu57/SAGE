# Hypothetical Minds — independent three-game adaptation

## Ownership and stage

- Stage: `hypothetical_minds_three_games_dev`, offline implementation.
- Only newly created HM source, entrypoint, tests and HM-owned results are used.
- No edits to other methods, games, API wrappers, shared journal, registrations,
  AGENTS.md or PROGRESS_LOG.md. No Git branch/index/commit operations.
- This document is this stage's handoff supplement, not a replacement for the
  repository's authoritative rules. Validation status is recorded below.

## Source and attribution

Cross et al., **Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent
Tasks with Large Language Models**, arXiv:2407.07086v2, §3.2, Appendix B/D.3–D.5:
https://arxiv.org/html/2407.07086v2

Official implementation (MIT): https://github.com/locross93/Hypothetical-Minds
Reference revision: `e1788c051608099f4b89c63d19de889a3706e36b`.
Reference path: `llm_plan/agent/running_with_scissors_in_the_matrix__repeated/rws_hypothetical_minds.py`.

This is an independently written algorithm adaptation, not copied Melting Pot
navigation code or a reproduction of the paper's scores. No Melting Pot,
pathfinding, RL training, LLM weights, or external code download is required.

## Implemented algorithm (game-frozen v2)

1. Keep natural-language hypotheses about the opponent's **policy**, initially
   valued at zero. Preserve them across games in one opponent/seat series.
2. Before any action in each game, generate at most one new hypothesis when no
  hypothesis is validated and completed-game evidence has arrived. Select the
  highest-valued validated hypothesis, otherwise the latest, and freeze that
  hypothesis plus its candidate set for the whole game.
3. Generate exactly one game-level high-level strategy before play. It remains
  unchanged throughout the game. Current-game evidence cannot revise the
  hypothesis, candidate set, values, validation status, or strategy.
4. At each ego decision, one action call executes the frozen strategy against
  the current legal set. After selecting the action but before applying it to
  the engine, predict only its next public response branch under every frozen
  candidate hypothesis. Closing actions create no prediction call.
5. Goofspiel retains one unconditional `*` prediction: its simultaneous
  prediction prompt omits our selected, still-unrevealed bid.
6. Bind each prediction to a unique future public event. `observe()` records the
  result in a deferred ledger but does not mutate hypothesis values.
7. At `finish_game()`, replay deferred results in event order, give +c for an
  exact match and -c otherwise, and apply
  `V <- V + alpha * (reward - V)` exactly once per prediction. The updated pool
  affects only the next game. Validation remains strict `V > threshold`.

Defaults: alpha=.3, c=1, threshold=.7, top_k=5. These are hypothesis scores,
NOT a normalized posterior or calibrated confidence. From V=0, four consecutive
correct predictions produce .7599; one wrong prediction then yields .23193.
Game payoff is available in memory but **does not determine hypothesis value**.

## Game protocol, specified before tests

Supported configurations are the existing default two-player games only:

| Game | Prediction target | Ground truth / boundary |
| --- | --- | --- |
| Leduc | Opponent's next action after the selected own action | Next public response in the same betting street; calls/folds/closing checks have no such response |
| Liar's Dice 2x3, wild 1 | Exact `bid_quantity_face` or `challenge` after the selected own bid | Next public response in the same dice round; own challenge closes the scope |
| Goofspiel n=5 | One unconditional current-round opponent `bid_card` | Fully revealed round result only; never the stored P0 pending bid |

- Leduc fixed amounts are supplied by the real engine's legal action list;
  response labels use its two-total-raises-per-street cap. The existing rules
  text has a per-player wording discrepancy; no shared rules are modified here.
- Goofspiel opponent remaining cards are inferred only from the initial public
  deck 1..5 and revealed past bids. This inference must not inspect the opponent's
  mutated internal hand while a bid is pending. Both positions use one `*`
  prediction branch, independent of our current proposed bid.
- Own private card/dice/hand may be used. Other players' private observations,
  raw state objects, future board cards, opponent identities/styles, CFR advice
  and SAGE memory are never provided to HM.
- Dice history reveals belong to **past** rounds; they do not reveal the next
  rerolled dice. Leduc fold does not reveal the opponent's card. This adaptation
  scores public actions and does not add an oracle showdown-card channel.
- No response before chance/terminal boundaries means **censored**, not wrong.
  Unselected own-action branches are no longer requested; no counterfactual
  response is fabricated or scored.
- `observe()` must receive the ego observation after transitions. The runner
  does this for both actors and terminal states. `finish_game()` consumes the
  last event even when the method never receives another decision opportunity.
- Initial unpredicted opponent actions are retained as evidence but never
  retrospectively scored. Each game uses distinct event IDs; a scored forecast
  cannot be scored again at the next observation or during resume.

## Documented departures from the reference

- Symbolic observations replace pixel-to-text maps; one legal game action is the
  low-level execution unit instead of A* navigation and 3–6 movement subgoals.
  The high-level strategy and low-level action remain separate LLM calls.
- Behavior recognition and scoring use exact public action labels, not an LLM's
  inferred RWS inventory or an LLM judge. The paper's cooking variant also has
  environment-specific behavior extraction. Exact dice predictions are more
  difficult than binary/categorical predictions; the default threshold is NOT
  assumed calibrated for these games.
- Public forced responses (e.g. last Goofspiel card) can yield correct predictions
  without learning a useful policy. Prediction accuracy/validation alone is not
  evidence of strategic benefit. Random opponents can also keep good hypotheses
  below threshold. No threshold tuning based on held-out outcomes is performed.
- Original RWS code may evaluate stored forecasts that were not refreshed during
  validated-only prediction. This implementation binds each forecast to a unique
  future event and consumes only that forecast, preventing stale reuse.
- The paper updates HM after sparse task events inside a long episode. This
  adaptation deliberately uses a stricter completed-game barrier: hypothesis
  values and strategy update once per game. It is a cross-game HM variant, not
  an exact reproduction of the paper's event schedule.
- A game starts with `hypothesis(0/1) + plan`. Each ego decision then uses
  `action + frozen_candidates * selected_branch_prediction`; a closing action
  uses only `action`. This is substantially smaller than the former all-branch,
  full-HM-per-decision protocol.
- Natural-language reasoning with a final JSON object is accepted, including a
  complete JSON/Python-labelled Markdown fence containing actual JSON. Python
  literals, duplicate keys, aliases, NaN, trailing text, and incomplete fences
  are rejected. A schema error or explicit `finish_reason=length` allows up to
  3 additional stage-only corrections; raw invalid/partial replies remain recorded.
- If and only if all action-stage output attempts are exhausted, choose uniformly
  from the current engine-provided legal actions using a deterministic persisted
  hash seed. This is reported as fallback, never as a model action. Exhausted
  hypothesis/plan output creates a frozen degraded policy; exhausted prediction
  is censored. Transport, authentication, budget, ambiguous request and game-state
  failures remain fail-closed and cannot trigger random action fallback.
- No offline prompt-policy training/DFS/MCTS is added. Online adaptation occurs
  only within a declared opponent/seat series. Independent series start empty.

## Entry points and operation

### Bash launchers

- `scripts/hypothetical-minds-leduc.sh`
- `scripts/hypothetical-minds-ld.sh`
- `scripts/hypothetical-minds-goof.sh`

All default to dry-run. Shared options include `--opponent`, `--seed`, `--games`,
`--provider`, `--max-requests`/`--no-request-limit`, `--offline`, and gated
`--live-api --approved-live-api --approval-by`. Canonical panels:

- Leduc: `LAG`, `TAG`, `CallingStation`, `Nit`, `Maniac`, `GTO`.
- Liar's Dice: `Bluffer`, `Honest`, `AggressiveChallenger`,
  `ConservativeChallenger`, `Random`, `GTO`.
- Goofspiel: `Greedy`, `Conservative`, `Sacrificial`, `Random`, `GTO`, `Mirror`.

LLM styles reuse the repository's original benchmark archetype prompt renderers
and strict legal-action parser. Their calls are journaled under `prompts/P1` and
share the same physical personal backend log; exhausted opponent output never
uses HM's random-action fallback and stops the run. Each completed game
prints a `Result:` line with seed, net, cumulative calls and action fallbacks;
normal foreground operation also displays tqdm progress.

No Git repository metadata, clean-worktree condition, or prior offline evidence
directory is required to execute a live or real-measurement command. Every
invocation stores logical prompts/responses under `prompts/P0`, completed
game checkpoints under `trajectories/P0`, validation and fallback records under
`logs/`, and personal DeepSeek physical requests/full response bodies under
`logs/network`. Live dev runs are limited to at most three games; 30-game runs
require explicit `--run-phase real_measurement` and approval flags.

- Import `HMConfig`, `HypotheticalMindsMethod` from `methods.hypothetical_minds`.
- Method contract is the existing BaseMethod three-tuple:
  `act(observation, legal_actions, model) -> (Action, prompt, raw_response)`.
- Lifecycle: `begin_game(1)`, `prepare_game(initial_obs, model)`, observe/act,
  `finish_game(terminal_obs, net_profit)`, then repeat with the same opponent.
  Runners always prepare from an empty public history. Direct first `act()` can
  auto-prepare only when no game action has yet occurred.
- Dedicated CLI: `runs/hypothetical_minds_bench.py`, required `--game` one of
  leduc/liars_dice/goofspiel; optional `--games` (1..4 per seat, default 2),
  `--start-seed`, `--top-k`, `--alpha`, `--threshold`, `--output-retries`,
  `--max-requests` (default 2000 offline attempts), `--no-progress`.
- **CLI is deliberately offline only**, always ScriptedHMModel versus a local
  uniform-legal opponent. It has no live/API/key switches and rejects a real
  backend in `run_experiment()`. Direct method injection is provider-independent;
  future live integration needs its own approved plan and implementation.
- All runner artifacts stay under
  `results/<game>/hypothetical_minds_dev/<timestamp>_<slug>/`.
  `--output-dir` chooses a parent within that owned tree; a fresh run is created.
- `--stop-after N` pauses after N newly completed games; `--resume RUN` rebuilds
  both seats' memory by deterministic cached replay of completed and partial
  games. Completed games permit no new backend calls. Original spec, source
  hashes and total request budget must match; budgets cannot be raised on resume.
- Failed transport calls require explicit `--retry-failed --resume`; a persisted
  `started` call remains ambiguous and blocks resume even with that flag.
  Exhausted schema/length histories replay exactly and cannot gain new correction
  attempts. The same fallback action and audit record are reproduced on resume.
- Stored calls use the unchanged public journal's `decision` stage and unique
  `environment_step * 1000 + local_ordinal`; semantic stage lives in prompt JSON.
  HTTP/text success and parsed output validity are separate logs. No automatic
  physical network retries in this offline runner.

## Artifacts and review

- config.json: phase/spec/seeds/budget/source hashes/Git state/command.
- prompts/P0 or P1: every stage/correction raw prompt and response, finish reason,
  timing and provider usage if available (fixture has no real usage).
- logs/output_validation: valid/invalid/truncated/exhausted by semantic stage.
- logs/action_fallback: exact failed call identity, legal-set size, deterministic
  selection digest, selected legal action and failure category.
- logs/steps: incremental ego-view transitions and method diagnostics.
- trajectories: completed game checkpoints with final hypothesis pool, memory,
  once-only scoring events, censored predictions and content hashes.
- summary.json + metrics/hm_metrics.json: completion, failures, legal actions,
  hypothesis prediction counts, validation entries/exits, net and call counts.
- logs/invocations: each invocation's original run, resume/skip/failure/remaining.

State hashes detect accidental changes, not cryptographic authenticity against
an adversary who can rewrite both data and checksums. Checkpoint replay provides
an additional dynamics/memory check. No outcome-based exclusion or silent
fallback is allowed; every fallback is explicit in method data and logs.

## Validation status

### Mandatory prediction branches prompt fix (2026-09-11)

User requested a prompt/code fix for the live dice reply omitting
`predictions.challenge: null`. Only HM-owned method/protocol and this document
were modified; a new prediction-feedback test file was added. No live calls.

- Every initial prediction prompt now lists `required_branches`,
  `required_null_branches` and concrete `required_null_members`. It explicitly
  says null-valued branches are mandatory, must use literal JSON null, must not
  be omitted and do not select our final action. These are schema instructions,
  not hardcoded predicted actions or recommendations.
- Strict validation is unchanged: no missing-key completion, no alias/default
  insertion. Prediction errors produce safe diagnostics for missing fields,
  fields requiring null, invalid response values and extra-key count. Feedback
  uses trusted branch names/options; arbitrary model-generated extra keys are
  not echoed as instructions. The raw reply is still retained as untrusted data.
- Example correction: `Missing predictions.challenge: include "challenge": null.`
  The exact error is stored in both correction prompt and output-validation
  log. The whole original task/context remains present; only the failed stage
  is retried. The existing initial+3 limit, truncation and budget guards remain.
- **218 passed (17.48s)** across HM plus shared journal/truncation regressions,
  including22 new feedback tests. All four historical incomplete dice replies
  still fail strict validation. Synthetic first-invalid/second-valid replies
  verify targeted feedback and once-only cached replay, not real model success.
- Offline dice GTO seed9100 P0: one complete game, 19 scripted calls, 6 legal
  actions; new run `results/liars_dice/hypothetical_minds_dev/20260911_194434_hm_gto_offline_dev/`.
- Validation run: `results/liars_dice/hypothetical_minds_dev/20260911_194140_prediction_prompt_fix_dev/`.
  See `logs/test_command.json`, `logs/pytest.xml`, `summary.json`, and
  `logs/predict_prompt_0.json` / `predict_prompt_1.json` for first/correction
  prompt previews. All132 old live artifacts and protected source files have
  unchanged hashes. Branch and index unchanged, editor diagnostics clear.

This source change invalidates old run resume/evidence fingerprints. A new live
run would require new same-source offline evidence and explicit approval; the
historical live failure is neither overwritten nor reclassified as success.

### Initial implementation evidence (historical source version)

**Offline implementation complete; human review pending (2026-09-11).**
Initial evidence: 85 passed / 2 failed (10.13s). Failures were incomplete fence
acceptance and integer-vs-string hypothesis-ID sorting in checkpoint hashes.
Both corrected in HM-owned files; a JSON round-trip regression was added.

Final validation: **226 passed (16.55s)**, comprising 88 new HM tests and 138
existing public journal/retry/truncation regressions. All 11 new Python source
and test files have no editor diagnostics. Python: ToMResearch conda 3.11.15,
bytecode disabled, pytest cache disabled. Exact arguments and JUnit evidence:
`results/leduc/hypothetical_minds_dev/20260911_190039_implementation_validation_dev/logs/pytest_final_command.json`
and `logs/pytest_final.xml` in that validation run.

CLI smoke: **12/12 complete games**, two games per seat per game, P0 seeds
9100–9101 and P1 seeds 9102–9103. All 82 engine actions legal; 44 HM decisions,
286 offline fixture calls, 112 hypothesis predictions checked against 29 public
events. These prediction rows are not statistically independent samples.

| Game | Run under `results/<game>/hypothetical_minds_dev/` | Games | Offline calls | New calls on resume |
| --- | --- | --- | --- | --- |
| Goofspiel | `20260911_191252_hm_2g_dev` | 4 | 150 | 0 |
| Leduc | `20260911_191302_hm_2g_dev` | 4 | 56 | 0 |
| Liar's Dice | `20260911_191304_hm_2g_dev` | 4 | 80 | 0 |

The validation run's `logs/smoke_runs.json` records exact CLI/resume commands.
Its `logs/audit.py` independently checks all 112 prediction bindings were made
before their observed event and recomputes each score. `logs/audit.json` confirms
205 pre-existing source/config/handoff files have unchanged hashes, the branch
and index are unchanged, and HM source/tests plus smoke-run JSONs have no
credential-shaped matches. Real credentials were not read. Existing artifacts
were not opened for modification. Completed-game cached replay preserves raw
prompt/checkpoint bytes in the integration tests.

Synthetic Mirror-prediction tests (both seats) separately verify validation
reduces candidates from 1/2/3/4 to 1, includes the final round, and does not score
stale forecasts. Scripted-uniform smoke rewards are not an efficacy result.

No Git operations that change state, no real model calls, no policy efficacy
claims. Human acceptance and any live budget remain pending.