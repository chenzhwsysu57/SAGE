# Independent bounded Strategist learning workflow

New entry point: `runs/strategist_live_bench.py`. Implementation:
`methods/strategist/live_workflow.py`; tests: `tests/test_strategist_live_workflow.py`.
The previous runner and retry bridge are unchanged. No ScriptedModel-derived
live shim or process-global monkeypatch is used. No existing handoff/Git files
are changed by this work.

## Protocol fixed before results

Each game has its own new `results/<game>/strategist_dev/<timestamp>_strategist_{offline|live}_dev`
directory (collision suffix if needed), its own 64-attempt ceiling, and no
automatic new batch. Phase is **dev_debug**, never real_measurement. Three games
have an aggregate maximum of 192 physical attempts. Seven logical generations
(initial source; twice reflection → idea → code) normally consume seven calls;
at most three extra format corrections per logical call, hence at most 28 with
this seven-stage protocol. The configured global ceiling can be lowered, never
raised above 64. Network retries are zero at both journal and campus wrapper.

- Train seed **9100**: genuine local-engine population self-play, common cohort,
  both seat assignments, equal opponent weighting including self. Cohorts have
  sizes 1, 2, 3, producing 2+6+12 round-robin games. Child/parent evaluation uses
  an identical frozen cohort and both seats: 4+8 additional games. **32 train games**.
- Reuse `Strategy`, `IdeaQueue`, public-only `key_feedback`, `StrategistSearch`
  and the restricted `ValueHeuristic` interpreter. Idea value is paired child
  minus parent score, not population-score subtraction across changed cohorts.
- Dev seed **9200**: each of the three policies plays both seats against the
  initial policy, **6 dev games**. Highest mean normalized return selects the
  frozen policy; ties choose smallest strategy ID. Dev scores are never included
  in reflection/idea/code prompts or training parent selection.
- Only after writing the frozen artifact, play exactly **3 final scored games**:
  `(9300,P0), (9301,P1), (9302,P0)`. No post-test updates or baseline games.
- Final opponents: Leduc existing CFR+21000 GTO, dice existing 2x3 CFR GTO,
  Goofspiel local Mirror. Reuse `methods.agent_pro.opponents` (which delegates
  dice/Mirror to `methods.baseline_opponents`); never train a table or substitute
  uniform. Model paths and SHA256 are in config/evaluation. Missing information
  sets or invalid distributions stop explicitly.
- All 41 local games use real engines. LLM calls only generate value functions
  and high-level feedback; local observation-keyed PUCT chooses actions, default
  simulations32/depth12/nodes128. No unbounded sweep or hidden-state prompt.

## Model boundary and code language

`run(..., model=backend, model_spec=ModelSpec(...))` accepts independent providers
with `generate(str)`, writable `last_usage`, and `last_finish_reason`; an optional
`redact(str)` sanitizes responses. Caller must disable backend retries, truthfully
declare provenance, and include nonsecret generation changes in
`cache_fingerprint`. Callable name/source and whitelisted settings enter the
resume fingerprint. A `ScriptedModel` cannot be declared live. Offline custom
HTTP fixtures declare `mock_transport`; their usage is diagnostic, not real API
usage. A caller-provided provider is responsible for truthful configuration;
the code cannot infer whether an arbitrary external implementation actually
uses a network or has hidden retry logic.

Default CLI is the existing deterministic scripted fixture. Explicit
`--live-api --approved-live-api --approval-by ...` selects a lazy campus adapter:
existing HKUSTGZ wrapper, Flash, temperature1, max_tokens=None (omitted),
timeout600/connect10, network0. No client/key access until the first uncached
generation, after config and approval checks. Completed resume reads no key.
HTTPX phase timeouts are not a hard wall-clock deadline.

Every original logical prompt carries real rules, exact allowed AST node names,
builtin positional arities, limitations and independent P0/P1 input fixtures
from non-experiment seeds0/1. No real episode's other hidden hand is supplied.
Keys are `game/player_id/private_info/public_info/history`; nested dictionary
keys are strings. Guard optional keys/nulls. No attributes or `.get/.items`,
imports, comprehensions, unpacking assignments, slices, mutation, recursion,
keyword calls, exceptions or exec. Output is one unannotated
`def evaluate_state(state)` returning `((v0,v1), explanation_dict)`, finite,
normalized, zero sum, fixed seat coordinates. Leduc scale32, dice1, Goofspiel15.

`CorrectingCalls` validates source using the existing AST interpreter and both
initial-seat fixtures. Corrections retain the entire original contract and raw
previous output; transport/length/budget failures never enter format correction.
Initial fixture validation is not exhaustive. Any later runtime error saves the
partial trajectory and failure context and stops, without changing policy in-game.

## Artifacts and recovery

Root and metrics JSON use the existing `{payload, sha256}` integrity envelope;
physical request journal JSON uses its existing v3 schema (not this envelope).

- `config.json`: written before calls; dev phase, safe command, signature,
  source hashes, model spec, opponent path/hash, schedule, Git read-only metadata,
  approval and budget, predeclared primary metric/exclusions/caveats.
- `summary.json`: success/pause/failure/interruption, last phase/item/seed/step,
  error type, completed phase counts, total/new attempts, usage/provenance,
  phase costs, final-game results only when complete.
- `prompts/ledger/prompts/P0/*.json`: every physical raw prompt/response,
  status, finish_reason, latency, usage; logical ID is `game_index`, correction
  ordinal is `step`. P0 here is the model-journal namespace, not game seat.
- `prompts/ledger/logs/output_validation/*.json`: semantic validity separate
  from transport success. `metrics/requests` within ledger: logical prompts.
- `metrics/policies`, `lineage.json`, `rounds`, `dev`, `frozen_policy.json`,
  `evaluation.json`, `phase_costs.json`; `trajectories/*.json`: incrementally
  persisted legal actions, actor-visible observations, local search traces,
  specified-opponent diagnostics and final raw/normalized rewards.
- `metrics/completion_manifest.json`: hashes of all completed JSON evidence.

`--resume` requires the same scientific config, source hashes, schema, backend
class/source/settings and opponent model hash. All known-success calls are
cached; paused local games replay deterministically. Completed games replay
legal engine transitions and opponent actions without search or LLM calls.
Completed runs verify their evidence manifest and return with zero calls and
zero artifact changes (original summary remains unchanged). Failed runs and
any started/failed request are rejected, not repaired or resent. Partial local
games after a safe interruption may restart with the same seed; model ambiguity
still blocks restart. Checksums are consistency checks, not authentication.

## Parent-agent live handoff (NOT executed here)

Before launching, review the exact-source offline validation and all failures,
confirm the user's live approval covers these three bounded runs, and verify
there is no active or queued duplicate dispatcher. Do not infer real_measurement
approval; do not increase budget or reopen a failed run. No branch/staging/commit
is needed or authorized. Each game may be launched independently; within a game
learning precedes frozen evaluation.

Run from repository root using the configured ToMResearch environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/zww/miniconda3/envs/ToMResearch/bin/python -B -u runs/strategist_live_bench.py --game leduc --live-api --approved-live-api --approval-by zww --key-file .ustgz.key --max-requests 64 --format-retries 3 --simulations 32 --max-depth 12 --max-nodes 128
PYTHONDONTWRITEBYTECODE=1 /Users/zww/miniconda3/envs/ToMResearch/bin/python -B -u runs/strategist_live_bench.py --game liars_dice --live-api --approved-live-api --approval-by zww --key-file .ustgz.key --max-requests 64 --format-retries 3 --simulations 32 --max-depth 12 --max-nodes 128
PYTHONDONTWRITEBYTECODE=1 /Users/zww/miniconda3/envs/ToMResearch/bin/python -B -u runs/strategist_live_bench.py --game goofspiel --live-api --approved-live-api --approval-by zww --key-file .ustgz.key --max-requests 64 --format-retries 3 --simulations 32 --max-depth 12 --max-nodes 128
```

Omit all three live approval options and the credential path for offline runs.
`--output-dir` must remain inside that game's strategist_dev tree, with no
symlink parents. Every fresh call allocates a new directory; history is not
overwritten. Seeds and update count are deliberately not configurable here.

## Limits

Scripted runs verify mechanics, not real LLM learning or efficacy. The restricted
language may require corrections and a real heuristic may fail on a later
state. Uniform hidden priors, determinization bias and tiny repeated cohorts
limit generalization. Disjoint seeds do not imply state-family independence.
Three final games cannot estimate effectiveness. Mirror is a deterministic
rule, not a Goofspiel equilibrium opponent. Provider-reported missing usage
remains unknown; there is no invented billing estimate.