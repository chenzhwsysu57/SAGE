# Agent-Pro three-game adaptation

## Final-action reliability protocol (new runs only)

The runner defaults to `--action-failure-policy random_legal`. Direct
`AgentProMethod`/`ScenarioEvaluator` clients remain `strict` unless explicitly
enabled; use `--action-failure-policy strict` for historical fail-stop behavior.
This is a disclosed engineering protocol change, **not a model-generated action
or an exact stochastic reproduction of the paper**. Exact calibration reuse is
retained unchanged for successful calibration chains.

Existing decision format corrections remain initial + 0..3, unchanged. After
exhaustion, explicit `length` (even if the partial JSON appears valid), a known
empty output, or a typed `ModelRequestError`/HTTPX transport failure, choose
uniformly from the CURRENT legal actions. Length/transport/empty do not add
correction calls. A received invalid response with unconfirmed finish metadata
ends corrections and can fall back; this is NOT an unknown in-flight request.
No token cap or physical retry is added. Classified HTTP errors, including auth
errors, may fall back for that decision; a finite total budget still stops later
calls. Operators should inspect failure counts rather than treating completed
games as provider health or model quality evidence.

Sampling uses a fresh `random.Random` seeded by SHA256 of the protocol, explicit
scenario seed, game/seat, own decision index, original visible prompt hash,
sorted exact legal actions and retry count. Game/GTO/global RNGs are untouched.
The context uses no opponent hidden state; own private observation is allowed.
`source=random_fallback`, reason, selection index/probability and all attempts
are saved before the engine sees the action. Beliefs in fallback records are
null; the last genuine in-game beliefs are retained, not invented or revised.

`logs/model_failures/` contains Agent-Pro-only known-failure receipts bound to
the original failed journal record and usage. `logs/action_fallback/` contains
deterministically recomputed certificates with exact journal dependencies.
Original response/usage/failed status and all corrections remain intact.
Resume audits source/config, scope, prompt, legal set, attempts, predecessor
chain and selected action. It reuses a verified certificate, or reconstructs
from received text/verified failure receipts, with **zero reissued failed calls**.
Budget exhaustion, `started`/unmatched intent, `ResumeConflict`, source/config/hash
changes, unexpected programming exceptions and user interruption NEVER fall
back. A crash between a generic failed journal write and its typed receipt
remains blocked for manual review; unknown failures are not guessed. Direct
unjournaled clients have deterministic sampling but no durable request recovery.
Integrity hashes detect accidental/local tampering, not an attacker rewriting
every related artifact and its hashes coherently.

Any fallback-contaminated main/training episode is excluded from policy learning
(but its actual payoff remains reported). Contaminated verification is rejected
regardless of reward; contaminated dev scores never select a policy. Known
reflection output/transport failure stops the update and retains its input policy,
including reverting candidates accepted earlier in that update. No random policy
is created. Unexpected candidate-lineage/duplicate-policy/invariant exceptions
still stop. Summary `fallbacks` separates reasons/phases, failed requests by stage
and contaminated completed episodes; `phase_costs` keeps physical usage separate.
These reports are robustness evidence only; scores include the hybrid random
behavior and must not be reported as pure model-only Agent-Pro performance.

## Exact failed-episode calibration reuse

New runs use `agent_pro_exact_calibration_reuse_v1`: identical complete parent
PolicyState, failed episode (including game/seat), calibration prompt, split and
format-correction configuration share the first calibration call identity.
Each candidate still gets its OWN proposal and verification. Candidate counts,
DFS branches/depth/nodes, update frequency and verification retries are unchanged.
This applies to both online and explicit learning modes; it does not add dev
evaluation games or restore paper-scale search budgets.

`logs/reflection_contexts/*.json` pins each candidate's `calibration_key` and
`calibration_identity` to the first owner. The original response and any correction
chain remain in the existing prompt/validation journal; they are not duplicated as
physical requests. Reuse replays and validates that chain without billing, including
when restarting directly at a sibling candidate. Failed, truncated, exhausted or
uncertain owners cannot turn into a successful calibration or reset their retries.
Cross-candidate reuse requires a valid completion with `finish_reason=stop`.
Direct, nonjournal backends only reuse confirmed valid stop outputs in memory.

This is a disclosed sampling change: candidates now share one calibration rather
than independent calibration draws. It preserves the current algorithm's stages,
not exact stochastic equivalence to the paper. At a fixed search path it removes
duplicate calibration costs; the actual search path, responses and total token
cost can change. Historical run artifacts/protocols are never rewritten, and
source fingerprint checks prevent cross-version resume. The legacy summary field
`calibrations_and_proposals_completed` counts candidate completions, including
referenced calibrations; physical costs must be read from the call journal.

## Default: scored online play, NO prelearning

`--mode online` is now the default. Start from initial instructions (no imported
policy), score the first game immediately, and update only AFTER a completed
negative-net game. Accepted policies apply to the NEXT scored game. No train/dev
panel is built or played, no future scene is instantiated for selection, and the
last scored game does not launch a useless update. `--train-seeds`, `--dev-seeds`
and `--policy` are rejected in online mode. Use `--eval-games 3 --eval-seeds
9100,9101,9102` for three scored games alternating P0/P1/P0.

Between games, calibration/proposal and bounded policy DFS replay only the just
completed scene. Root payoff comes from the scored episode; every candidate must
strictly improve its parent's payoff on that same scene. This is retrospective
same-scene selection, NOT independent development validation or proof of general
improvement. Replays are EXTRA games and API calls, recorded separately; original
scores never change. Policies, not model weights, evolve. Beliefs still update
within every game. Search limits apply per update; total request budget (if set)
is shared by main play, reflection and verification, including corrections.

`metrics/online.json` records scored rows, policy-before/after and update status;
`metrics/online_updates/game_NNNN.json` records each retrospective search. Summary
costs distinguish online_main, online_reflection and online_verification. Resume
reconstructs the causal sequence through audited caches; snapshots alone never
authorize skipping requests. Source/config changes and uncertain/failed requests
remain blocked. There is no frozen-test phase in online mode.

Historical full/learn/evaluate modes below remain explicit for reproducing their
original protocols; their artifacts are NOT relabeled online. The full Leduc
live run at 20260911_193427 completed three frozen tests, not online adaptation.

## Full method versus decision-only evaluation

Use **`--mode full`** for the complete connected method: ordered training
exploration -> failed-episode belief calibration -> policy proposal -> same-scene
verification -> fixed-dev evaluation -> DFS descent/siblings/backtracking ->
freeze selected policy -> held-out test. No model weights are trained. This
invokes the existing paper-based components; it does not replace them with a
generic summarization method. `--mode learn` stops after freezing; `--mode
evaluate` alone NEVER learns. The historical 174646 nine-game live run used
evaluate on the INITIAL policy, so it validated decisions/corrections only.

Example OFFLINE command (ToMResearch environment):

```bash
python runs/agent_pro_bench.py --game leduc --opponent uniform --mode full --train-seeds 9100 --dev-seeds 9200 --eval-seeds 9300,9301,9302 --eval-games 3 --branches 2 --max-depth 2 --max-nodes 7 --verification-retries 1 --no-request-limit
```

For dice use game liars_dice/opponent GTO; Goofspiel uses game goofspiel/opponent
Mirror. B=2/depth=2 is a bounded development setting, not the paper's full B=8
measurement. Three test games DO NOT mean only three total games: training, dev
panels and candidate verification execute additional games. No-request-limit
now also supports full with explicit test count and finite node/depth bounds;
it cancels request count limits, not per-output correction or search bounds.
Full live learning requires explicit approval of its distinct learning workload;
prior approval for three decision-only games is not authorization for training.

Train/dev/test initial families are checked before constructing any backend.
Test episodes cannot reach the reflector or select the best policy; the selected
policy is frozen before test. On test failure the frozen artifact is preserved,
but final_test_performed stays false. Resume replays the deterministic search
using durable calls/episodes, never reissues an ambiguous test request, and
rejects changed source/configuration. Earlier failed runs must not be resumed
under this new source.

`metrics/components.json` and summary `components` report actual component
execution: training episode count, completed calibration/proposal pairs,
verification count/pass count, dev-scored policies, DFS nodes/backtracks,
accepted candidates, selected-policy change and frozen test games. No loss means
no reflection; no improved candidate means the initial policy may remain. These
are legitimate outcomes, not proof that every branch was exercised live.
Summary `phase_costs` separates cumulative learning and frozen-test requests and
reported provider usage (including decision corrections). Unknown usage is
explicit, never estimated. Invocation `new_requests` excludes successful replay.

Current reliability scope: decision JSON/action corrections are implemented and
live-tested. Calibration/proposal now have their own bounded JSON/schema/text
corrections, tested OFFLINE only; full workflow live reliability remains unvalidated.
`--reflection-parse-retries` defaults to 3 additional calls PER stage (0 disables),
independent of `--decision-parse-retries` and `--verification-retries`. Calibration
uses correctness/consistency/rationality/reasons (6000 characters each); proposal
uses the five policy strings (12000 each AND 12000 total). No decision action
schema is reused. Original first-call prompts are unchanged. Each correction
preserves the full original prompt, JSON-encodes the previous output as untrusted
data, and repeats the stage's complete return schema and length instructions last.
The model must regenerate the whole object; no local repair, clipping or guessing.

Only local JSON/schema/text failures after a nonempty confirmed `stop` can trigger
correction. Direct providers' finish metadata is cleared before each learning call;
missing/unknown/stale metadata never authorizes retry. Empty/truncated responses,
transport/authentication/budget errors and ambiguous started requests stop normally.
Duplicate but valid policies are semantic rejection, NOT format correction or a
reason to relax payoff verification. A successful calibration is cached while
proposal is corrected; the parent and evidence remain unchanged.

Actual semantic stages are `calibration_retry_1..3` and `proposal_retry_1..3`.
Each consumes the existing global budget (also when the outer cap is null) and
keeps the original reflection identity/learning phase. Raw replies remain in
separate journal files. Learning validity is separately recorded under
`logs/reflection_output_validation/`; immutable context pins under
`logs/reflection_contexts/` bind original calibration prompt, previous candidates,
source protocol and count. `logs/agent_pro_reflection_protocol.json` pins all
Agent-Pro Python sources. The ledger audits prior invalid/stop, stage order,
exact correction prompt/schema and calibration-to-proposal linkage on restart,
before any new request. Successful and exhausted chains replay without new calls;
exhaustion cannot reset, and changed source/config/context cannot resume.
These consistency hashes are not external cryptographic signatures against an
attacker rewriting an entire run. Untrusted-data framing reduces instruction
confusion but cannot guarantee model resistance to prompt injection.

### Learning-output correction offline evidence (2026-09-11)

1077 passed, 0 failed/errors/skipped, 77.97s; 129 new reflection-retry tests.
Evidence: results/leduc/agent_pro_dev/20260911_184503_learning_retry_dev/.
Config records the exact ToMResearch command before execution; logs/pytest.xml
and pytest.txt retain the first 1065-pass run, and logs/pytest_final.xml and
pytest_final.txt contain the final run after stale finish-metadata hardening.
Both use that run's pytest_tmp as basetemp. Existing optimizer tests also create
their own timestamped, configured results/leduc/agent_pro_dev fixture roots.

Final source_id:
cf680755ae70dfe6978904fb6bb1295f8f9a5522d0392a75fa869be1a9e2fb9a

Three real-engine full-workflow fixtures inject malformed first calibration AND
proposal replies on every reflection identity. Leduc: 6 pairs, 83 total fixture
requests; dice: 4 pairs, 54 requests; Goofspiel: 4 pairs, 56 requests. Each pair
has one correction per stage. All freeze before held-out tests and resume with
zero new calls; learning/test costs, source/lineage and immutable artifacts are
verified. Summary lists exact nested directories, seeds/seats and costs.
No external API calls, real credential reads, shared-source edits or Git writes.
The 33 shared/protected files hashed before/after final validation were unchanged.
This is not evidence of real-model correction success or learned-policy quality.

### Full-workflow offline evidence (2026-09-11)

948 passed, 0 failed/skipped, 44.69s. Evidence:
results/leduc/agent_pro_dev/20260911_183407_full_workflow_dev/.
New full-workflow tests cover all three real game engines, split protection,
cost separation, component reporting and uncertain test-call interruption.
First fixture attempt used uniform opponents in every game: dice/Goofspiel had
no training loss and correctly skipped reflection. The fixture was corrected to
the declared GTO/Mirror opponents; no learning gate was weakened.

Standalone offline full runs (train9100/dev9200/test9300–9302):
- Leduc: results/leduc/agent_pro_dev/20260911_183808_agent_pro_full_dev/.
   6 calibration/proposal pairs, 6 verifications, 7 policy nodes, 3 dev-scored
   policies, 3 test games. Selected policy changed; 68 learning + 7 test calls.
- Dice: results/liars_dice/agent_pro_dev/20260911_183811_agent_pro_full_dev/.
   4 calibration/proposal pairs, 4 verifications, 5 nodes, 3 test games.
   Candidates rejected by verification; 42 learning + 11 test calls.
- Goofspiel: results/goofspiel/agent_pro_dev/20260911_183813_agent_pro_full_dev/.
   4 calibration/proposal pairs, 4 verifications, 5 nodes, 3 test games.
   Candidates rejected by verification; 43 learning + 15 test calls.

All three resumed with zero new calls; 244 prompt/trajectory/metric file hashes
unchanged. These are SCRIPTED fixtures, not real model learning/effectiveness.
No new live requests or Git writes in this stage; work remains on the current
feat/cross-game-baselines worktree and other methods are unchanged.

## Protocol v1 (declared before results)

Reference: Zhang et al., ACL 2024, https://aclanthology.org/2024.acl-long.292/,
sections 3.1–3.3, A.4 and E.4. Official reference implementation:
https://github.com/ZJU-OmniAI/Agent-Pro/tree/8f410d751ade72ce3e67b97fb8b2c05182679bd8.
The inspected official runtime provides dynamic beliefs and summaries, not a
complete verification/DFS driver. This is an independent paper-based adaptation,
not a claim of author-code or original numerical-result reproduction.

Scope: Leduc JJQQKK/equal antes; Liar's Dice 2x3/wild1/multi-round;
Goofspiel n5/split ties/visible prize sequence. No SAGE/CFR advice, statistical
archetype labels, Suspicion pipeline or ReTA action search in method prompts.

### Algorithm

1. One decision call jointly returns self-belief, world-belief and canonical action.
   Beliefs carry within a game only; a frozen policy carries across games.
2. Only negative-net training episodes trigger calibration (correctness,
   consistency, rationality, reasons), then a separate call proposes a full policy
   revision containing goals/strategy/demonstrations and rule/opponent modeling.
3. Verification re-executes the same initial scenario with fresh beliefs, restored
   environment/opponent RNG, and reactive opponents. Strict net improvement is
   required. `verification_retries=3` means initial proposal plus at most three new
   proposals; these are NOT network retries. Exhausted format corrections or
   truncated/failed calls stop the run.
4. Verified candidates are scored on a fixed development panel, shared by parent
   and child. Strict mean-net improvement permits descent; DFS explores siblings
   and backtracks, retaining the globally best evaluated policy. The paper uses
   B=8; offline defaults B=2/depth=2 are flow tests only. All bounds are explicit.
5. Final evaluation freezes the policy. Development/test feedback is never sent to
   the reflector; final test scenarios are disjoint from training/development.

Training scenario order is deterministic. A node uses the first negative-net
training episode under its policy; exhaustion ends that branch. This bounded
selection differs from the paper's random challenge-hand exploration. API errors
are not game losses. Root/tied payoffs never imply a meaningful policy improvement.

### Information / randomness

The method receives only structured own observations and legal actions, never raw
engine state, opponent identity, scenario seed/ID or opponent model diagnostics.
Reflection adds only rule-visible reveal: Leduc showdown (not fold), historical
dice challenge reveals, completed Goofspiel rounds. Missing evidence remains
unknown; losing does not prove a belief was wrong. Author-code all-hand revelation
is deliberately not copied. Learned demonstrations are training experience, not
an oracle cache; no scenario-key lookup is used during play.

Leduc/dice panels cross method seat with initial private allocation swap (four
conditions); Goofspiel uses two seats, not duplicate identical hand swaps. Dice
future rolls consume a restored RNG stream; different losses/dice counts can
consume it differently. This reduces but cannot eliminate chance or LLM sampling
variance. Game-state fingerprints, not just seeds, detect split overlap, including
equivalent allocations and repeated Goofspiel prize sequences.

### Metrics / failure / budget

Primary: mean terminal net profit, separately per game. Leduc uses payout minus
bets, dice retains ±1, Goofspiel uses score difference. In two-player zero-sum net
units the paper's own-minus-opponent metric is twice net profit. Report both;
do not merge games or treat permutations as independent statistical samples.
Verification is training-only; dev selection scores are not final-test evidence.
Learning and frozen evaluation usage are reported separately.

No silent legal-action fallback, semantic repair, policy clipping or provider
retry. Overlong/invalid learning output has bounded full-object format correction;
all discarded replies remain in durable evidence, and exhaustion stops the run.
Single global attempt budget covers every node and stage, including failures.
Started/ambiguous or failed calls are never automatically retried. Resume replays
successful calls and deterministic control flow, restoring DFS without re-billing;
source/config/input fingerprints must match. The CLI defaults to offline. Live
requires `--live-api --approved-live-api --approval-by NAME --max-requests N`
or the explicitly approved `--no-request-limit` mode with `--mode evaluate
--eval-games N`. Flags alone are not authorization. No-limit mode cancels the
total request cap, not the game-count limit or failure checks.
The provider is the existing campus wrapper, with no wrapper or ledger automatic
retry. A `--key-file` path is runtime-only, not copied to artifacts. The default
temperature is 1.0 (paper setting), max_tokens is omitted (server limit remains),
read/write/pool timeout is 600 seconds and connect timeout is 10 seconds. These
timeouts are per phase, not an overall deadline. No real_measurement CLI exists.

### Development stages

P0 protocol/isolation; P1 decisions; P2 reflection; P3 scenario verification;
P4 DFS; P5 CLI, recovery and offline evidence. All artifacts belong under
results/<game>/agent_pro_dev/<timestamp_slug>/ (including CLI test artifacts).
Formal training, API calls and real_measurement require separate human approval.

### Implementation and entry point

- `policy.py`: immutable content-addressed policies and strict text/JSON schemas.
- `method.py`: one belief/action call per decision, fresh beliefs per episode.
- `reflection.py`: training-only calibration then full candidate proposal.
- `scenarios.py`: paired initial allocations, isolated RNG, visible feedback,
   checksummed completed episodes with local replay and journal audit.
- `optimizer.py`: bounded DFS, strict verification/dev gates, backtracking and
   atomic diagnostic snapshots. Resume replays control flow instead of trusting
   a mutable search snapshot.
- `calls.py`: global persistent semantic call identities and immutable budget.
- `scripted.py`: deterministic fixtures, NOT competitive learned policies.
- `runs/agent_pro_bench.py`: independent `--mode learn|evaluate` entry point.

Example offline learning: `python runs/agent_pro_bench.py --game leduc --mode
learn --train-seeds 9100 --dev-seeds 9200 --branches 2 --max-depth 2 --max-nodes 7
--verification-retries 1 --max-requests 500`. The actual command is a single line.
Use the ToMResearch interpreter. For dice add `--game liars_dice --opponent GTO`
(instead of the Leduc game argument); for Goofspiel use `--game goofspiel
--opponent Mirror`. Uniform is the local default; GTO means the stored CFR
approximation, not an exact-equilibrium claim.

Learning saves `metrics/frozen_policy.json`, including train/dev family exclusions.
Evaluate with `--mode evaluate --policy PATH --eval-seeds 9300,9301`; an omitted
policy uses the initial w/o-learning policy. An input frozen artifact must retain
its checksum and split provenance; a bare prompt-policy file is not accepted.
Learn does not automatically evaluate test data. Importing a policy preserves
its prior split exclusions; overlapping new scenes are refused, not reselected.
Families treat Leduc suit-relabelings as equivalent and ignore undealt unused
cards; Goofspiel uses the prize sequence rather than seed identity.

`--output-dir` is an artifact parent underneath repository `results/`; a new
timestamp directory is allocated every time. For `--resume RUN_DIR`, repeat all
semantic flags, including the original attempt budget and imported artifact.
Paths, source changes, scene changes, model settings and budgets are audited.
Resume cannot retry failed/ambiguous requests or grant extra budget. Completed
calls/episodes retain their artifacts. Process crash before an episode commit
can replay successful calls; failures require manual review and a separately
approved new run. Development seeds/panels cannot silently be changed to select
better results. Small game state spaces limit independent scene families.

### Output guide

- `config.json`: phase, complete specification, source hash, read-only Git
   provenance, generation settings, approval and budget; never the key.
- `logs/agent_pro_call_ids.json`: semantic stage, policy/node/scenario/step ->
   physical call file mapping. All stages share one ledger.
- `prompts/P0/gNNNN_s0000_decision.json`: **journal storage namespace**, NOT
   necessarily actual game player P0 or decision stage. Look up the mapping for
   actual `decision/calibration/proposal` and semantic identity. Each file has
   full prompt/response, status, finish_reason and reported usage.
- `trajectories/<identity hash>.json`: own belief/action records, visible
   feedback, and full-state diagnostic trajectory. Never feed the full artifact
   directly into an LLM; only the sanitized reflector inputs are admissible.
- `metrics/search.json`: all candidates, gate outcomes, dev scores and DFS events.
- `metrics/frozen_policy.json`: selected policy with split provenance.
- `metrics/evaluation.json`: frozen-policy per-condition net returns.
- `summary.json` and `logs/invocations/`: total/cached/new calls, completed/skipped
   episodes and failure class. Search episode count is outcome-dependent, so
   `remaining=null` during learn; evaluate reports the fixed remaining game count.

### Development evidence (initial local implementation)

2026-09-10, ToMResearch Python 3.11.15: 446 Agent-Pro tests, plus 138 shared
journal/retry/truncation regression tests: **584 passed (12.63s)**.
Evidence: `results/leduc/agent_pro_dev/20260910_221510_final_validation_dev/`.
Includes three-game 12-episode decision smoke, full learn/frozen evaluation,
MockTransport live-path tests (zero external requests), failure gates, split
isolation, policy-tree backtracking and resume with zero new model requests.

Standalone scripted learning runs (train 9100, dev 9200, B=2/depth=2/node cap=7,
one verification retry):
- Leduc/uniform: `results/leduc/agent_pro_dev/20260910_221606_agent_pro_learn_dev/`.
- Dice/GTO: `results/liars_dice/agent_pro_dev/20260910_221607_agent_pro_learn_dev/`.
- Goofspiel/Mirror: `results/goofspiel/agent_pro_dev/20260910_221608_agent_pro_learn_dev/`.

The standalone fixtures exercise one accepted Leduc child, rejected validation
candidates and backtracking. Dice/Goofspiel candidates do not improve the
verification result; retaining the initial policy is the correct outcome, not a
reason to relax the gate. Fixture scores are not scientific performance evidence.

All current development is local on `feat/cross-game-baselines`. No branch/index
or commit operations are part of this implementation. Shared handoff documents
and other methods were deliberately not edited.

### Model response protocol v2 — 2026-09-11

The first real three-game dev run stopped at each first response: all three
responses used complete Markdown JSON fences, and Leduc returned `"check"`
instead of the required action object. All had finish_reason=stop, not length;
0/9 complete games, three actual API calls. Original failed artifacts are kept
unchanged under the 20260911_123422/123434/123450 evaluate runs. Transport success
in the journal does not mean schema validation or game execution succeeded.

The fix keeps the algorithm and API settings unchanged:

- Prompts for decision, calibration and proposal now include explicit output
   schemas and require bare JSON. Decision action is an object whose enum is the
   complete current legal-action list, with explicit player/type/amount/data
   instructions. This is prompt-level guidance, not provider-enforced JSON mode.
- Model-only parsing accepts bare JSON or one entire `json`/unlabelled triple-
   backtick code block, with no surrounding prose. Disk config, policy and journal
   parsing remains strict. Duplicate fields, NaN/numeric overflow, trailing text,
   multiple blocks, broken JSON, Python literals and incomplete fences are rejected.
- Full action objects must still exactly match a legal object. No missing fields,
   incorrect amounts, extra fields or bool/float coercion are repaired.
- Explicit strings are normalized only when they identify exactly one currently
   legal action: Leduc fold/check/call/bet/raise, dice challenge, or Goofspiel bid_N
   naming an exact legal card. Leduc amounts come from the unique engine action;
   bare dice/Goofspiel `bid` is rejected even with only one available bid. Dice
   quantity and face are never inferred from the belief prose.
- Raw responses and finish_reason/usage are preserved unchanged. Parsing occurs
   only after the truncation check, and belief/policy state updates occur only
   after successful full validation. No added model calls or automatic retries.
- The run specification records `response_protocol=agent_pro_json_envelope_v2`.
   Existing source/config hash gates forbid resuming old runs under this protocol;
   no prior live artifact is retroactively relabelled as a completed game.

Final regression: **713 passed, 0 failed/skipped (16.42s)**, including 90 new
protocol unit tests and 32 integration tests. Three actual historical replies
now parse to check / bid(quantity=2,face=3) / bid(card=2) offline. Their 18 old
artifact files have identical hashes before and after. Fenced/alias fixtures
complete three games per engine (nine total), with 6/11/15 fixture calls and
zero new calls on resume. Calibration/proposal fenced responses and MockTransport
truncation/error paths are covered as well.

Evidence: `results/leduc/agent_pro_dev/20260911_125150_response_protocol_final_dev/`.
The three-game fixtures use seeds 9100–9102, positions P0/P1/P0, no allocation
swaps, and local uniform opponents. No real credentials were read and no new
live request was sent for this fix. Live revalidation remains pending: the
protocol fix does not establish reasoning correctness, service reliability or
Agent-Pro effectiveness, and is not permission to retry old ambiguous requests.

### Terminal return format v3 — 2026-09-11

The v2 live retry made six API calls but completed no games. Three responses
omitted action: Leduc 130219 request 1, dice 130241 request 3, Goofspiel 130418
request 2. All ended with stop. The previous sorted JSON prompt placed output
instructions in the middle (before policy/previous_belief/task), not last. This
ordering is confirmed; it is not proof of the cause of model omissions.

At the user's request, decisions now end literally with:

```text
return format:
{"self_belief": "your belief here",
"world_belief": "your belief here",
"action": "valid action"}
```

Nothing follows this template. The request is a structured JSON context followed
by output instructions and this exact template, not a single sorted JSON object.
The requested action is a STRING from valid_actions: Leduc check/bet/call/raise/
fold as legal, dice bid_Q_F (quantity/face) or challenge, Goofspiel bid_N (card).
Labels map one-to-one onto the existing complete legal objects. Incomplete or
ambiguous bid labels remain invalid; full old action objects remain accepted
only when exact. Missing action is still an error, even when the belief prose
mentions an action or only one action is legal. No fallback action is inferred.

The decision output_schema now agrees with the string-valued terminal template.
The run response_protocol is agent_pro_terminal_action_format_v3. The structured
rules, observation, policy and previous beliefs are unchanged. Reflection and
proposal keep their separate schemas and never receive the decision action
template. Their model JSON parsing and all truncation protections are unchanged.
Offline fixtures use read_prompt_context only to inspect requests; model output
and disk JSON parsing have NOT been made permissive. Resume across this prompt
change is refused by existing source/config checks.

Validation: 778 tests passed, 0 failed/skipped (17.23s), including 65 new terminal
format tests, nine scripted string-action games, nine fenced-compatible games,
and zero-new-request resumes. All three missing-action historical replies remain
rejected without belief mutation or extra model calls. Six historical runs (39
files) remained unchanged. Evidence and readable old/new prompts are under
results/leduc/agent_pro_dev/20260911_132523_terminal_format_tests_dev/.
New prompt previews are prompts/<game>/new_prompt.txt; they are not live outputs.
No new API calls, secret reads, Git writes, or changes to other methods occurred.

### Explicit no-total-request-limit mode — 2026-09-11

User requested no maximum request count while the v3 batch was in flight. That
batch was allowed to finish under its original recorded configuration before
changing source (otherwise the source audit would fail). Its original 60-request
limits were never reached and were not rewritten: Leduc completed 3/3 (-1,+9,+3,
7 requests); dice completed 3/3 (-1,+1,-1,13 requests); Goofspiel completed 1/3
(+1,8 requests), then a malformed JSON reply stopped the second game. This was
not BudgetExceeded. All 28 calls returned stop; one was invalid model JSON.
These small initial-policy runs do not establish learned-policy effectiveness.

Now `--no-request-limit` is mutually exclusive with `--max-requests`; config,
approval, invocation and semantic mapping persist max_requests=null. The new
spec explicitly declares request_budget_mode=unbounded_user_approved. The flag
requires evaluate and an explicit positive exact --eval-games schedule, so it
does not silently authorize unlimited policy training/search. Finite defaults
are unchanged for other callers. Future user-requested uncapped tests should use
this flag, not a made-up large integer.

There is no total request threshold in the outer ledger. It grants exactly the
next audited attempt to the unchanged shared journal under an exclusive lock,
with automatic retries disabled. The grant renews for every new request; it is
a one-call reservation, not a lifetime cap. Stats and raw responses are still
fully recorded. The existing integer identity schema boundary remains a storage
constraint, not a research request budget. Changing finite/unbounded mode on
resume is refused, and failed/ambiguous requests are never reissued automatically.

Offline evidence: results/leduc/agent_pro_dev/20260911_140043_unbounded_dev/:
821 passed (19.18s), including >60 real ledger fixture calls, global counts,
no-limit exact-three-game evaluate/resume, immutable mode and error handling.
Initial indentation and invalid test basetemp-name errors were fixed without
relaxing test gates; intermediate logs remain. No additional API calls, shared
source edits, branch/ref/index writes or commits during this change.