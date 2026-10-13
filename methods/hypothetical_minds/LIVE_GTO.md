# HM single-game GTO live probe — 2026-09-11

Historical v1 evidence: this predates the game-frozen and output-fallback
protocol. It is retained unchanged and must not be resumed or relabeled as v2.

User approved one Leduc, one Liar's Dice and one Goofspiel game versus GTO.
New-file-only extension; original HM offline source and entrypoint remain unchanged.
No Git branch/index/commit operations. Other agents' concurrent work is not modified.

## Entry and protocol

Dedicated new entry: `runs/hypothetical_minds_gto_live.py`.
Default remains scripted/offline. Live needs `--game`, `--live-api`,
and `--approved-live-api`. No Git state or prior evidence artifact is required.
Approval flags do not replace a human approval. One P0 game only, seed9100,
opponent RNG seed1009103.

- Campus DeepSeek-V4-Flash, temperature .1, omit max_tokens (server default,
  NOT unlimited), HTTPX read/write/pool600/connect10 (not total deadline).
- Maximum150 actual attempts per game, including ALL retries; total450 across
  this approved three-game batch. No auto-addition of budget or games.
- Existing HM initial+3 stage-only schema corrections, unchanged prompts/parser.
- This new entry explicitly enables public journal initial+3 transient retries,
  5/10/20-second backoff or Retry-After, whichever is longer. Wrapper sends once.
  Auth errors, explicit truncation, exhausted schema/transport retries and budget
  exhaustion stop the run. Read-timeout retries can duplicate upstream billing;
  all attempts are counted conservatively, unknown usage is not estimated.
- One-game probe does NOT support resume. Failed/in-flight calls remain for
  review and are not automatically replayed. New runs never overwrite old ones.
- Local GTO loaders are reused read-only from existing opponent adapters; no
  ReTA prompts/search or SAGE components are used. Missing strategy rows stop,
  rather than silently falling back to uniform. No training or model file writes.

GTO checkpoints: Leduc CFR+21000 (528 infosets), dice CFR2x3 (348 rows),
Goofspiel continued n5 MCCFR+ (1,449,491 infosets). These are pretrained
approximations, not certified exact equilibria. Model hashes are recorded in
each config's opponent metadata.

## Offline evidence

Preflight: `results/leduc/hypothetical_minds_dev/20260911_192100_live_preflight_dev/`.
172 tests passed (15.59s), including 16 new harness/MockTransport tests. An
initial test incorrectly expected a full 5-second sleep despite elapsed time;
fixed its clock fixture, not production backoff. No editor errors.

Same-source seed9100 P0 vs real local GTO scripted runs:

- Leduc `20260911_192632_hm_gto_offline_dev`: 1 complete game, 19 fixture calls.
- Dice `20260911_192633_hm_gto_offline_dev`: 1 complete game, 19 fixture calls.
- Goofspiel `20260911_192634_hm_gto_offline_dev`: 1 complete game, 30 fixture calls.

All under the matching game's `results/<game>/hypothetical_minds_dev/` tree.
Exact pytest arguments and results in preflight logs. Phase=dev_debug, never
real_measurement. Primary checks: completion, legal actions, prospective
predictions, corrections, finish reason, usage; payoff is descriptive only.
No outcome-based exclusions, no guarantee that the live game follows fixture
call counts, no efficacy/convergence claim from one game.

## Live run identities

- Leduc: `results/leduc/hypothetical_minds_dev/20260911_192744_hm_gto_live_dev/`.
- Dice: `results/liars_dice/hypothetical_minds_dev/20260911_192931_hm_gto_live_dev/`.
- Goofspiel: `results/goofspiel/hypothetical_minds_dev/20260911_193014_hm_gto_live_dev/`.

Input/output: each run's `prompts/P0/*.json`, fields `prompt` and `response`.
The prompt is a serialized JSON string containing semantic `stage`. Storage
stage is always decision; journal step encodes environment_step*1000+ordinal.
`logs/output_validation/P0/` separates HTTP/text success from schema validity.
`logs/steps/` is incremental and survives failures; `metrics/progress.json`
stores the last method snapshot; completed runs also have a trajectory.

Leduc completed, +3 chips, 13 normal-stop replies. Dice stopped at environment
step3, predict stage, 9 normal-stop replies total: all four attempts for the new
hypothesis prediction omit required `challenge: null`; the other three bid
branches are present. The schema requires every own-action branch including
null for an own challenge closing the round. This is schema exhaustion, not
HTTP failure, truncation, or a failed final-action parse. Two environment
actions occurred; no terminal payoff exists. No local null insertion, parser
relaxation, prompt edit, extra game, or retry-budget reset was performed.

## Final results and audit

All three requested attempts ended; **2/3 target games completed**. No extra
batch, recovery, replacement sample or budget increase was started.

| Game | Completed | HM payoff | API attempts | Reported tokens |
| --- | --- | --- | --- | --- |
| Leduc | 1/1 | +3 chips | 13 | 17,589 |
| Liar's Dice | 0/1 | unavailable, partial game | 9 | 10,237 |
| Goofspiel | 1/1 | 0 point difference | 30 | 31,702 |

Total52/450 attempts, 59,528 reported tokens. All52 replies were normal stop;
no HTTP/transport failures, explicit length completions, network retries, or
ambiguous calls. Four dice replies failed schema validation despite successful
API delivery. The three extra correction calls ran as implemented but returned
the same incomplete JSON each time; this does not establish why the provider
repeated it. Current correction feedback restates the schema generically rather
than explicitly naming the missing key; an improved error-specific correction
would be a separate implementation/review step, not a reason to relabel this run.

Read-only audit `logs/audit_live.py` / `logs/live_audit.json` in preflight:
all52 exact prompts/responses replayed, all17 performed engine actions legal,
all9 HM decisions and GTO draws matched, final snapshots and 19 scored
hypothesis predictions reproduced, and the dice failure reproduced exactly.
Correct predictions: Leduc1/3, dice0/1, Goofspiel7/15 (multiple hypotheses per
event, NOT independent samples). No hypothesis crossed the .7 validation
threshold in this tiny live batch. Longest successful requests were23.47s,
11.30s and47.03s respectively; maximum completion tokens1444,582,3488.
All132 original run files retained their hashes during audit; credential-shaped
scan of run JSON had no hits, no real credentials were read by the audit.

HM and shared dependency fingerprints still match the offline evidence. Branch
and index stayed unchanged. Concurrent EMO and AGENTS.md changes were observed
by the protection scan and left untouched; this session edited none of them.
Human review and any follow-up live budget remain pending.