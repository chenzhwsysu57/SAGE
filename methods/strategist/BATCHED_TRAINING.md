# Batched training handoff

## Current stage

- Stage: three-game batched-self-play training_dev.
- Status: implementation and complete three-game offline validation finished;
  live training blocked pending human review and clean-worktree resolution.
  No new live calls, no key access, no staging/commit/branch change.
- Scope: new Strategist training module/CLI/wrapper/tests only. No branch, index,
  commit, shared method, existing launcher or historical artifact modifications.
- Requested schedule per game: 1000 train games,50/game batch,20 improvements.
- Separate development panel: two paired scenarios/update, candidate and parent
  each play both seats against the same archived opponent.8 games/update,160 total.
- Three-game totals:3000 train+480 development games,183 logical API calls.
  Each logical call permits initial+3 semantic corrections: at most732 physical
  attempts under the finite protocol, not a configurable lifetime request cap.
- Default search unchanged:32 simulations/depth12/nodes128. Official personal
  DeepSeek,temperature0.7,thinking disabled,max_tokens omitted,600/connect10,
  zero network retries. Local actions do not use an LLM.
- Fresh seeds:train1000000..1000499 (two seats/deal),dev2000000..2000039
  (candidate/parent,two seats/deal). Future PK must not reuse either range.
- Predeclared selection: strictly positive mean paired normalized development
  gain; ties keep parent. All candidates remain in the local opponent archive.
  Rejection is not a failed update. An update means one candidate-generation cycle.
- Feedback: latest completed50 games,mean reward,archive IDs,top3 visited public
  root discrepancies. No growing full-trajectory concatenation or future PK data.
- No external PK is included. Existing series launchers still start their own
  learning; they must NOT be used to evaluate this artifact without explicit loading.
- This extends the existing fixed NLP-weight evaluator, not full program synthesis
  or exact Strategist reproduction. More iterations do not guarantee better play.

## Gates and artifacts

Default CLI previews only. `--offline` uses bounded deterministic fixtures; it
does not represent LLM learning. Live long training is real_measurement and needs
explicit approval, same-source/full-protocol offline evidence and a clean worktree.
Do not auto-commit or relabel a long live experiment as dev to bypass this gate.

Every run is new under results/<game>/strategist_training/. Config precedes key
access. Trajectories and update receipts are incremental, prompts include every
semantic correction, and logs/network contains complete physical IO. Failed and
partial games are retained and block automatic resume. A safe pause can replay
completed games and successful calls. Completed replay changes no core evidence
and writes only a separate resume receipt. Hashes are consistency checks, not
authenticated tamper protection. Frozen policy has a dedicated strict loader.

## Acceptance criteria

- Real-engine small runs in all three games;legal actions,seat pairing,disjoint
  seeds,per-game progress,8 validation games/update and update after last batch.
- Exact1000/50 scheduler verification,61 logical calls;no API calls inside games.
- Safe pause/resume equivalence and completed replay zero requests;reject changed
  sources/config/trajectories,partial games,failed/ambiguous requests.
- Retain raw format errors and corrections;no fake fallback or retry replenishment.
- Source diagnostics,scoped regression,shell syntax,secret-shaped literal scan.
- Record failures and caveats before requesting human review/live execution.

## Validation evidence (2026-09-12)

- Evidence root:results/api/strategist_batched_training_dev/20260912_173119_validation_dev/.
- Initial focused suite28 passed(7.34s); read-only review then identified init
  failure receipts, completion ordering and usage-parity gaps. Fixed only new
  training files and added regression tests. No failed test output was discarded.
- Final complete Strategist regression453 passed(42.74s), no exclusions.
  Exact command in logs/full_regression_config.json; logs/regression.txt/xml.
- All three real engines with default32/12/128 and scripted replies completed
  1000 train+160 dev games and20 updates each;61 scripted calls/game.
  Offline driver logs/full_offline.py denies socket connections in every process.
  Source hashes and exact command recorded before runs in logs/full_offline_config.json.
- Leduc:results/leduc/strategist_training/20260912_174322_batched_training_dev_bkp2m1fd/,
  4231 legal actions;4 accepted candidate updates.
- Dice:results/liars_dice/strategist_training/20260912_174322_batched_training_dev_a88wqvxn/,
  5846 legal actions;2 accepted candidate updates.
- Goofspiel:results/goofspiel/strategist_training/20260912_174322_batched_training_dev_aicd8fr7/,
  11600 legal actions;8 accepted candidate updates.
- Total3480 legal complete local games,21677 legal actions. Each final policy
  exports/loads; completed replay skips1160 games and adds0 calls per game.
  All4161 core artifact hashes unchanged after replay; historical live artifacts
  and source hashes unchanged. Every log has1160 game progress+20 update lines.
- Editor diagnostics zero; syntax/shell checks pass, credential-shaped literals0,
  trailing whitespace0. Git index remains empty, branch feat/cross-game-baselines.
- Main caveat: these are deterministic scripted-mechanics results, NOT actual
  LLM learning or strategy effectiveness. Candidate acceptance counts do not
  support performance conclusions. Real API training has NOT started.
- Proposed live uses the same three protocols with personal DeepSeek;183 logical
  calls total and at most732 physical attempts including format corrections.
  No token-price forecast or guarantee; tokens depend on actual model replies.
- Current shared worktree is dirty with other people's work. Do not auto-clean,
  commit, change branch, create a snapshot, or relabel long training dev to launch.
  Human review and explicit clean-worktree arrangement are required next.
- User follow-up after evidence review: "不要改任何分支信息。直接进行".
  Live intent is explicit, but isolated branch/snapshot authorization is denied.
  No Git mutation or live request was performed. Clean-worktree hard gate remains
  unmet; stop here rather than override production guards or mislabel this run.