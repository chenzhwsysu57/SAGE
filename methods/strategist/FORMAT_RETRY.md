# Strategist output correction — additive implementation

## Scope / status

2026-09-11: implemented and offline validated. This change adds only
`format_retry.py`, `runs/strategist_retry_bench.py`,
`tests/test_strategist_format_retry.py`, and this note. Existing Strategist,
shared API/game code, AGENTS.md, PROGRESS_LOG.md and Git branches/index are not
edited. No live API was invoked or enabled in the new CLI.

## How to opt in

Use the new `runs/strategist_retry_bench.py` entry instead of the original CLI.
It supports the original workflow arguments plus:

- `--format-retries 3`: at most three **extra** calls; `0` disables corrections.
- `--retry-ledger PATH`: a new dedicated directory under
  `results/<game>/strategist_dev/`, outside the workflow run directory. Use a
  timestamped name such as `YYYYMMDD_HHMMSS_format_retry_dev`.
- `--max-requests N`: the physical ledger caps the sum of initial calls and
  corrections (and physical network retries when used through the generic API).
- On resume, pass the same `--retry-ledger`, `--format-retries`, scientific
  arguments and the original workflow `--resume` path. Existing ledgers cannot
  be reused for a new workflow without explicit resume. Missing physical evidence,
  changed retry settings/configuration/backend/source and ambiguous calls fail
  closed. Added Python files also change the old runner's implementation hash:
  do not try to resume historical pre-change runs across that fingerprint.

This CLI deliberately still uses ScriptedModel. A clean scripted run normally
does not trigger correction. Tests inject malformed outputs to exercise recovery.
It does not provide a live API switch or imply real LLM learning was tested.
`CorrectingCalls` accepts a provider-independent backend for later integration;
the caller must identify its model/configuration, disable backend-internal
retries, and obtain separate approval before any live calls.

## Semantics

- Seed/code stages: parse and evaluate through the existing bounded AST
  interpreter on both player views of a fixed development fixture. Invalid
  syntax/restricted constructs, shape, range and zero-sum return failures are
  correctable. No parser relaxation, exec, fallback policy or payoff selection.
- Reflection/idea stages: nonempty bounded text, limits 12000/4000 characters.
- Correction prompts include the original request, previous response as data,
  safe error category and explicit return contract. They repeat only the failing
  generation stage, not self-play, reflection or ideas already completed.
- API failures, length truncation, authorization errors, request budget failures,
  persistence errors and started/uncertain calls do not enter format correction.
  The shared journal classifies empty/whitespace/non-string backend output as a
  failed call before semantic validation; that also stops without correction.
- Network retries default to zero. The generic API can independently select
  0..3 using the existing shared journal retry policy. Authentication and length
  remain nonretryable, even then.
- The inner journal records HTTP/transport success separately from per-attempt
  semantic `valid` results. Exhaustion retains every invalid reply and stops.
  Replay uses cached attempts and cannot obtain a fresh correction allowance.
- Validation is preflight, NOT proof the function works on every reachable
  state. A later search/runtime failure stops the workflow; it does not trigger
  policy mutation mid-match.

## Evidence and accounting

The outer workflow summary's `calls.requests` counts logical generations, NOT
physical retries. Authoritative physical requests/usage are the **retry ledger**
`prompts/P0/*.json` (including each record's `attempts`). Semantic validation is
in its `logs/output_validation/*.json`. Never add outer and inner counts or
usage together. The original raw invalid replies live in the inner journal;
the outer workflow sees only validated output or a failure.

First regression: 358 passed / 20.96s. After adding ledger-cache consistency,
CLI resume, partial-loop replay and combined network/format budget cases:
**364 passed / 21.70s**, five Strategist test files. Final evidence:
`results/leduc/strategist_dev/20260911_201209_format_retry_validation_dev/logs/pytest_final.xml`.
Three game full-workflow fixtures each use train9100/dev9200/test9300, both
seats, simulations2/depth2/nodes8. Each injects three invalid first code replies:
7 logical calls / 10 scripted backend attempts, two completed improvements,
frozen evaluation; resume adds zero backend attempts. These are software tests,
not game-performance or actual model-correction-success evidence.

Python: ToMResearch conda 3.11.15. `PYTHONDONTWRITEBYTECODE=1`,
`GIT_OPTIONAL_LOCKS=0`, `pytest -q -p no:cacheprovider`; basetemp and JUnit output
inside the new validation directory. No Git staging/commit/checkout; no live API.