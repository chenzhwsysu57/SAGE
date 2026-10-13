# Personal DeepSeek continuous GTO dev run — 2026-09-12

Status: completed with partial failures; human review pending. No automatic
resume, replacement samples, or additional API calls were made.

This is historical **v1** evidence. It used full HM inference and all-branch
prediction at every ego decision. The current game-frozen v2 protocol does not
reinterpret or resume these runs; new live use requires fresh same-source v2
offline evidence and separate explicit approval.

## Approved protocol

Three game series launched concurrently at 01:10:14. Within each series, one
P0 `HypotheticalMindsMethod` instance runs seeds9100,9101,9102 sequentially and
retains hypotheses, values, evaluations and previous-game memory. Series are
isolated across Leduc, Liar's Dice and Goofspiel. Opponents are the repository's
local pretrained approximate GTO policies; no GTO fallback is permitted.

Provider: personal official DeepSeek via `.deepseek.key`, official host
`api.deepseek.com`, model `deepseek-v4-flash`, temperature0.7, thinking disabled,
max_tokens omitted, direct connection with trust_env false, timeout600/connect10.
No total request-count cap was configured by explicit user request; network and
schema correction loops remain bounded at initial+3 attempts each. Server output
limits still apply. Failures are retained and stop only the affected game series.

Each physical request is stored at `logs/network/request_NNNNNN.json` with exact
request JSON and complete raw response body. The complete body includes usage and
would include reasoning content when returned. With thinking disabled, this batch
returned no nonempty `reasoning_content`; nothing was stripped. Key values are
redacted before persistence. Logical prompt/response and output-validation logs
are separate, preserving the distinction between HTTP/text and schema success.

## Preflight

Evidence run:
`results/api/hypothetical_minds_personal_dev/20260912_010257_personal_continuous_preflight_dev/`.

- 306 offline tests passed (25.95s), including personal wrapper MockTransport,
  host/auth/body/thinking settings, complete response/usage persistence, secret
  redaction, explicit unbounded journal, continuous memory and recovery.
- Same-source personal-target scripted continuous series all completed:
  Leduc `20260912_010616`, dice `20260912_010617`, Goofspiel `20260912_010619`,
  under each game's `results/<game>/hypothetical_minds_dev/` directory.
- Each had memory counts1,2,3 and game2/game3 prompts contained1/2 prior games.
- Credential shape and official host were validated without printing key value.

## Live results

All runs are under `results/<game>/hypothetical_minds_dev/` with run ID
`20260912_011014_hm_continuous_gto_personal_live_dev`.

| Game | Complete | Per-game HM net | Physical requests | Reported tokens | Stop reason |
| --- | ---: | --- | ---: | ---: | --- |
| Leduc | 3/3 | +3,+1,+1 | 27 | 44,298 | completed |
| Liar's Dice | 0/3 | none | 20 | 50,626 | explicit output length at game1 step7 predict |
| Goofspiel | 0/3 | none | 18 | 27,746 | plan schema exhausted at game1 step5 |

Total:3/9 target games completed,65 physical requests,122,670 reported tokens.
All65 physical response bodies were preserved and JSON-parseable; all65 contain
usage. Leduc and Goofspiel had only HTTP-success records. Dice had19 success and
one failed explicit length completion; partial response length25,363 characters
was retained. No request remained in `started` state.

## Cross-game history verification

Leduc provides actual live evidence of accumulation:

| Game | Memory before→after | Hypotheses before→after | Evaluations before→after | Net |
| --- | --- | --- | --- | ---: |
| 1 | 0→1 | 0→2 | 0→3 | +3 |
| 2 | 1→2 | 2→3 | 3→6 | +1 |
| 3 | 2→3 | 3→4 | 6→10 | +1 |

Every Leduc model prompt in games1/2/3 had exactly0/1/2 `previous_games` items.
Checkpoint2's method-state-before equals checkpoint1's final pool; checkpoint3
similarly equals checkpoint2. This verifies that history was supplied and state
was retained. It does not prove that the model used history causally or that the
positive rewards resulted from it.

Dice and Goofspiel stopped before game1 completed, so they provide no live
cross-game history-use evidence. Their prompts correctly had zero previous games.

## Failures and corrections

- Leduc had one invalid game1 plan response without a complete final JSON object;
  the first correction succeeded.
- Dice had one invalid hypothesis response corrected successfully. Later, game1
  step7 predict returned explicit `finish_reason=length`; per protocol it was not
  parsed or format-retried, and the series stopped with no terminal payoff.
- Goofspiel game1 step3 plan initially lacked a complete final JSON and corrected
  successfully. At step5, the initial plan plus all three corrections returned
  both `strategy` and extra `predictions` keys. The plan stage requires exactly
  `strategy`; after correction exhaustion the series stopped. API delivery was
  successful for all four replies, but schema validation failed.

No partial result is counted as a zero payoff. No local repair, field deletion,
parser relaxation, seed substitution, or run restart was applied.

## Evidence and limitations

- Read-only audit: preflight `logs/live_audit.json`.
- Input/output: each live run's `prompts/P0/*.json`.
- Physical transport: each live run's `logs/network/*.json`.
- Schema status: `logs/output_validation/P0/*.json`.
- Complete game checkpoints: `trajectories/P0/*.json`.

No credential-shaped strings were found in run JSON. Git branch/index remained
unchanged during preflight; concurrent agents' files were left untouched. This
is dev-debug flow evidence, not a measurement of HM effectiveness. The positive
three-game Leduc sample is too small for conclusions, and the other two methods
still require reviewed output-robustness changes before another live run.