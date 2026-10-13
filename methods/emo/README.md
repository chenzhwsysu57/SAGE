# EMO — paper-method adaptation (paper-history fidelity corrected; review pending)

## Manual benchmark CLI — 2026-09-12

Current stage: implemented and validated explicit `--run-phase real_measurement`
for the user's manually executed three-game/six-opponent/30-per-seat benchmark.
This supersedes the earlier statement that real_measurement is not exposed.
No benchmark is launched by this development stage. No Git writes or shared-file edits.
Live benchmark requires `--live-api --approved-live-api --approved-real-measurement
--approval-by NAME`; an explicitly human-run `--manual-run` permits a dirty worktree
under the repository's manual-execution exception, recording approval, dirty state,
source hashes and exact command. Otherwise a dirty real-measurement run is refused.
Default dev/dry-run behavior remains unchanged. EMO algorithms and retry budgets
are unchanged. Output uses `results/<game>/emo_benchmark/<run_id>/`, not emo_dev.
For `--games 30 --position both --seed 9100`, P0 seeds are 9100–9129 and P1 seeds
9130–9159. Each invocation has its own total physical request cap; default 12000
for this size, including opponent calls and retries. This is not a token ceiling.
All scheduled outcomes are retained, failed runs are not automatically replaced,
and primary mean/total payoff is reported per opponent/seat. Prior GTO-only live
evidence does not prove live six-style behavior. Do not resume old runs across
this source change.

Validation: **177 passed (232.11s)**, including all 18 shell commands with an echo
executable (no backend launch), a complete 60-game MockTransport run with P0
9100–9129/P1 9130–9159, completed resume without new physical calls, stage/output
directory consistency and explicit manual dirty-worktree override. Evidence:
`results/leduc/emo_dev/20260912_185536_manual_benchmark_cli_dev/` (config and JUnit).
No real credentials read, no live API or actual benchmark started, no Git writes.

Authority: Yu, Zhang & Lu, *LLM-Based Explicit Models of Opponents for Multi-Agent Games*, NAACL 2025, pp. 892–911, https://aclanthology.org/2025.naacl-long.41.pdf . Algorithm 1–2, equations (1)–(6), appendices A/D are the implementation specification. No confirmed official source repository was located.

## Scope / handoff — 2026-09-11

- User approved implementation of the paper method only. **New files only; do not modify shared AGENTS.md / PROGRESS_LOG.md, existing methods, rules, API code, branches or Git index. No commits.** This EMO-owned document records this stage instead.
- Stage `emo_paper_port_dev`: **paper-history fidelity correction complete offline, awaiting human review**. Independent workflow verified in all three games, including both seats, event timing, hidden-information isolation, exact iteration counts, parsing, budget and resume. Not an effectiveness experiment.
- The initial implementation request did not itself approve live API or real measurement. Later user messages separately approved the historical one-game and parallel nine-game dev runs documented below. The CLI remains scripted by default; no real-measurement entry is opened.

## Fixed algorithm (no optional algorithm variants)

For each newly **observed opponent action**, perform INIT, two SELF_FEEDBACK / SELF_REFINE pairs, then two GLOBAL_FEEDBACK / GLOBAL_REFINE pairs. Atomic models generate initial/local/global-refined hypotheses; a distinct central context generates global feedback. Before each own action, the central context summarizes all current opponent hypotheses and three CoT calls analyze the situation, compare legal choices, and select an action. All roles may use the same backend, but their contexts are separate. This follows Algorithm 2 (CoT + EMO); it does not reproduce every optional reasoning backbone in the paper.

Appendix A explicitly specifies two iterations at each level; use two pairs, not three from Algorithm 1's inconsistent inclusive `0..S` pseudocode. The initial prediction sees history **before** the current opponent action; feedback sees that actual action. No future opponent action is used. Full update phases also run for an observed terminal opponent action; there is no subsequent terminal decision or summary.

No CFR recommendation, style classifier, Bayesian update, search, probability scoring, learned policy, additional self-reflection, candidate pruning, cross-game/episode memory, or ablation condition is added. A format-only engineering retry may ask the same failed structured-output stage to restate its answer, but it does not rerun, replace or add an EMO reasoning stage. Engine legality checks are an execution boundary, not a validator replacing the LLM.

## Necessary environment bindings (not original benchmark replication)

The original environments are five-player WITU and Avalon with hidden roles/cards. Here:

- Leduc: target the opponent's private card. The agent alone sees its own private card; atomic/central models receive public information/history (equations 1–5). Decision CoT receives the agent's private observation. Fold does not reveal the opponent card.
- Liar's Dice: target the opponent's dice in the observed action's round. Public challenge reveals belong to that round, not to the freshly rerolled round. Hypotheses and their round tags remain in history, but expired hypotheses are not presented as current ones. No new LLM stage is added for rerolls.
- Goofspiel: retain the private-card target, **do not invent intent/style modeling**. Starting cards and previously revealed bids determine the available cards at each simultaneous round's start. Pending opponent bids remain hidden. EMO may be redundant in this environment; that is a transfer limitation to measure, not something to fix by adding a new method. Updates occur only when both bids are revealed, for either controlled seat. Historical card hypotheses are tagged with their round; future CoT receives their scope, not a claim that a used card is still held.

Hypotheses retain the Appendix-D list/None output convention (up to three textual candidate descriptions). No artificial hidden-role slot is introduced where all player roles are known. Candidate content is not corrected by code. Game-specific tasks replace WITU/Avalon role/word language with actual game objects; all rules come unchanged from the engine. CoT's WITU-only description/voting tasks necessarily become situation analysis / legal-action comparison / action selection; this is disclosed, not claimed to be identical original prompts.

Each episode starts with empty agent, atomic and central histories as Algorithm 2 specifies. There is one atomic opponent model in these two-player environments, so the port cannot demonstrate the original multi-opponent coordination advantage.

### Exact history semantics and fidelity correction

Algorithm 1 line 15 defines each atomic history update as `{p_tilde_j,t, q_hat_j,t, a_j,t}`. Algorithm 2 line 15 defines agent history as `{p_i,t, q_i,t, a_dot,t}`. The transient self/global feedback and CoT states `s_k` are inputs to their immediately following refinement/reasoning call; the paper does **not** merge those full texts into persistent history.

Protocol `emo_alg1_alg2_compact_history_v2` now enforces those tuples:

- atomic history: current public-information snapshot, final globally refined private-information prediction, observed opponent action;
- agent history: current public-information snapshot, own private information, observed or chosen action;
- no feedback response, refinement transcript, summary, CoT reasoning, or nested cumulative action history is persisted inside another history entry;
- `cot_2` receives only `cot_1` as `s_{k-1}` and `cot_3` receives only `cot_2`, as Algorithm 2 specifies;
- raw calls remain in artifact logs for auditing but are never recursively injected into later prompts.

Earlier protocol `emo_alg1_alg2_appA_two_iterations_v1` incorrectly stored full feedback traces and full prior CoT reasoning in persistent history. That caused substantial non-paper token amplification. All v1 live runs below are retained as historical dev artifacts, but **must not be used as a faithful EMO performance or token-cost result**. Source/protocol fingerprints prevent resuming them under v2.

## Bash command-line launcher

### Six-style integration stage — 2026-09-12

Complete offline, awaiting human review: connected all six existing benchmark opponents per game through EMO-owned
adapters and the existing Bash entry. Reuse original archetype prompts (five Leduc,
four dice and three Goofspiel LLM styles), local Random/Mirror and existing GTO
adapters; never substitute rule bots for LLM styles. Acceptance: all 18 selections
in both seats offline, strict legal actions, separately attributable opponent IO
under the shared physical-attempt budget, replay without new calls, Bash arguments
and per-game stdout. No EMO algorithm change, live calls, shared-file edits or Git
writes. Existing live/real-measurement gates remain unchanged in this stage.

| Game | Supported benchmark opponents |
| --- | --- |
| Leduc | LAG, TAG, CallingStation, Nit, Maniac, GTO |
| Liar's Dice | Bluffer, Honest, AggressiveChallenger, ConservativeChallenger, Random, GTO |
| Goofspiel | Greedy, Conservative, Sacrificial, Random, GTO, Mirror |

All games additionally retain `uniform`. The first five Leduc, first four dice,
and first three Goofspiel opponents render the original benchmark style prompt
and call the selected provider. Only their prompt renderers are reused, not their
permissive parsers or fallback logic. One complete JSON envelope/fenced envelope
must select a unique current legal action. Up to `--format-retries` extra format
corrections are separately logged; exhaustion, explicit length and ambiguous calls
remain blocked on replay. These engineering calls do not change EMO stages.
Random/Mirror remain original local policies and GTO remains the existing local
approximation. Leduc now also rejects missing/invalid/zero-mass policy rows before
calling the original sampler, rather than accepting its historical uniform fallback.

Example preview (no execution):
`bash scripts/emo.sh --game leduc --opponent LAG --seed 9100 --games 30 --position both`.
For zero-network flow validation, append `--offline` and use a small game count.
For approved small live pilots, use `--live-api --approved-live-api --approval-by NAME`.
This integration does **not** remove the live three-games-per-position gate or
start the 1080-game experiment; no real-measurement CLI is implemented here.

IO attribution: `prompts/P<controlled-seat>/gNNNN_sNNNN_decision.json` contains EMO
paper stages/corrections, while `..._pattern.json` contains opponent calls in a
separate numbered stream. Here the folder denotes the experiment's **controlled
seat**, not necessarily the speaker. `logs/emo_opponent_calls/` records both
`controlled_player` and the actual opponent `actor`, engine step and journal path;
`logs/emo_opponent_validation/` records strict output validity and canonical action.
Both streams share a single physical-attempt budget and, for personal DeepSeek,
the same numbered `logs/network/` full-body IO. Trajectories preserve both sides'
raw prompts/responses for audit only; opponent private prompts are never injected
into EMO. Per-game results add opponent decision/correction counts separately from
the unchanged `9 * opponent_events + 4 * method_actions` EMO logical count.

Final evidence: `results/leduc/emo_dev/20260912_183144_six_styles_dev/final_summary.json`;
**328 passed (76.43s), zero external API calls**. New coverage includes all 18 styles
in both seats (36 complete games, seeds 9100/9101), completed replay with zero new
calls, three four-game partial-replay comparisons, strict JSON/dice parameters,
budget/length/ambiguity/correction exhaustion, opponent artifact tampering, hidden
Goofspiel pending bids, and personal DeepSeek full physical IO using MockTransport.
All 18 Bash plans accept `--games 30 --position both` in preview; wrong-game styles
and long live remain rejected. Three actual Bash offline runs (LAG/Bluffer/Greedy,
one game per seat) completed 6/6, saved 56/84/140 prompt records (including 4/6/10
opponent calls), and each emitted exactly two `Game completed` lines after saving.
Final source fingerprints remained unchanged throughout validation. First focused
run had 60 passes/one test-fixture failure (dataclass `model_copy` misuse), fixed
with `deepcopy`; that original XML and the intermediate 324-pass run are retained.
No original game/archetype/API/shared journal/other-agent files were edited, no
EMO core algorithm change, no real credential read, no staging/commit/branch change.
Mock success demonstrates plumbing, not actual model style fidelity or performance.

Use `bash scripts/emo.sh --help`. The launcher is an EMO-owned, foreground wrapper around `runs/emo_bench.py`; it does not reimplement the method. It supports:

- `--game leduc|liars_dice|goofspiel`;
- `--opponent` with the game's six benchmark styles above, plus `uniform`;
- `--seed`, `--games`, `--position P0|P1|both`;
- `--provider deepseek|hkustgz`, runtime-only `--key-file`, and optional `--cfr-model`;
- `--max-requests`, `--format-retries`, `--max-retries`, timeout and output-parent controls;
- mutually exclusive default dry-run, `--offline`, or explicitly approved `--live-api` modes.

Default dry-run prints the exact runner command and creates no files, reads no key, and calls no API. Live development requires both `--live-api --approved-live-api --approval-by NAME` and refuses more than three games per position; this cannot bypass real-measurement review. `--games N --position both` means `N` games for each seat, not `N` total.

Execution stays in the foreground, so runner `tqdm` updates after each completed game and displays seed/position/output. It prints `Game completed` after trajectory/metrics/summary are saved, as well as its allocated run directory and final status. It saves config/summary, per-game trajectories, both sides' logical prompt/response journals and output validation; official personal DeepSeek also saves complete physical request JSON, raw response body, `reasoning_content`, and usage under `logs/network/`, with the key redacted.

Validation on 2026-09-12: Bash syntax/help/default dry-run; invalid game/opponent, long-live and misplaced-approval gates; arbitrary repository-relative paths; and a real zero-network Goofspiel Mirror offline run with `--seed 7 --games 2 --position both`. The offline run completed 4/4 games, emitted visible per-game progress, saved 260 prompt records and four trajectories at `results/goofspiel/emo_dev/launcher_offline_validation/20260912_133129_emo_paper_dev/`. No key or API was used. Script-specific editor diagnostics and `git diff --check` passed.

## Execution / reproducibility

Files owned by this stage: this README, `methods/emo/{__init__,adapters,core,calls,opponents,scripted}.py`, `runs/emo_bench.py`, `tests/test_emo_method.py`, `tests/test_emo_runner.py`. No shared registry is modified. Import directly from `methods.emo`.

The entry supports `--game leduc|liars_dice|goofspiel`, `--games` (per seat), `--position`, `--start-seed`, `--opponent` (six benchmark styles per game plus uniform), `--max-requests`, `--format-retries`, `--max-retries`, `--output-dir`, `--resume`, `--stop-after`, and `--no-progress`. GTO means each game's existing local CFR/MCCFR approximation and is not exact-equilibrium certification; Mirror is the existing local Goofspiel rule. Neither is EMO's decision mechanism.

Algorithm settings are fixed: no reduced iterations, alternative EMO conditions or search. Format correction defaults to three extra attempts and can be set to 0–3; transient transport retry likewise defaults to three extra attempts. These are labeled engineering reliability calls, not paper stages. Separately approved live development needs `--live-api --approved-live-api --approval-by <approver> --max-requests <approved-cap>` and a runtime credential. Actual live backend is the existing campus wrapper, configurable by `--model`, temperature 1.0; its top_p is not explicitly sent. This is **not** a claim to reproduce GPT-4-0613 or the original API defaults. No real_measurement mode is exposed before dev review.

All generated output goes to new `results/<game>/emo_dev/<timestamp>_<slug>/` directories. Existing source/artifacts are not overwritten. Default seed schedule is P0: 9100,9101; P1: 9102,9103 for two games/seat, with new EMO memory each episode. Opponent RNG resets to game seed + 1000003. Config records exact specification, source hashes, provider settings, Git state and request cap. Progress identifies game/seat/seed/output.

Each logical call uses the unchanged journal transport stage `decision` and a monotonic per-episode call index. Its true EMO stage is inside the prompt JSON and `logs/emo_calls`; these sidecars refer to existing journal paths and are cross-checked during resume. Outputs are in `prompts/P0` or `prompts/P1`. `logs/emo_validation` distinguishes **provider text success** from **output shape/legality success**. When a candidate list/None or final legal action fails parsing, the original response is retained and a new `format_correction` call asks only for the required envelope; previous EMO stages are not repeated. Each correction has its own prompt/response, retry index and request charge. After the configured maximum, the game stops; a resume replays the same exhausted replies rather than silently extending retries. Transient network/429/5xx failures use the existing bounded transport retry history. Truncation, authentication/non-transient failure, budget exhaustion and ambiguous in-flight responses still block. Request caps cannot be raised by resume.

Completed episodes are immutable and verified against their call-file hashes before skipping; interrupted episodes reconstruct from seed and cached responses. `logs/episode_progress` records diagnostic actions (including pre-reveal actions) and is never supplied to a model or treated as authoritative recovery state. Only learner observations/public events feed EMO. Raw errors are not printed, credentials are redacted by the unchanged wrapper/journal, and all actual attempts count toward the cap. HTTPX timeouts are per phase, not a total wall-clock limit.

## Validation evidence — 2026-09-11

**Current v2 fidelity evidence:** `results/leduc/emo_dev/20260911_214246_paper_history_fidelity_dev/`; **247 passed** (35.68s), zero API calls. Tests explicitly reject persistent/nested `history`, `feedback`, `response`, `reasoning`, `summary`, or `interactions` fields, and verify the one-step CoT chain. This supersedes v1 as the implementation accepted for future testing.

The evidence and runs below predate the v2 history correction and are retained for provenance only.

- Environment: ToMResearch conda, Python 3.11.15, `sys.dont_write_bytecode=True`, `GIT_OPTIONAL_LOCKS=0`.
- Validation directory: `results/leduc/emo_dev/20260911_190237_paper_port_validation_dev/`.
- First EMO-only run: **71 passed** (12.35s), `logs/pytest.txt` and `logs/pytest.xml`.
- Final regression after strengthening semantic sidecar audit: **235 passed** (14.33s). Includes the 71 new EMO tests plus unchanged journal, truncation, API retry and campus wrapper tests. Exact pytest arguments/executable in `logs/regression_command.json`; output in `logs/regression.txt` and `logs/regression.xml`. All pytest temporary artifacts stayed under that validation run.
- Format/transport retry regression after user-requested reliability change: **244 passed** (15.19s), `logs/retry_regression.txt` and `logs/retry_regression.xml`. Added coverage for correction success at init/local-refine/global-refine/action stages, correction exhaustion, retry count zero, shared request cap, transient success after retry and non-transient no-retry behavior. No live API was used.
- Editor diagnostics: eight Python files, no errors. AST parse: eight files passed. Source/credential-shape audit: `logs/final_audit.json`.
- Offline CLI commands passed to `main`: `--game <game> --opponent <opponent> --games 2 --start-seed 9100 --max-requests 2000 --no-progress`; then the identical arguments with `--resume <run>` for each game. Exact arrays/logs in `logs/smoke.json` and per-game smoke/resume logs. Socket connections were blocked during smoke and **zero attempts** occurred.

| Game | Local opponent | Run | Complete games | Scripted logical calls | New calls on resume |
| --- | --- | --- | ---: | ---: | ---: |
| Leduc | uniform | `results/leduc/emo_dev/20260911_191326_emo_paper_dev/` | 4 | 112 | 0 |
| Liar's Dice | GTO approximation | `results/liars_dice/emo_dev/20260911_191327_emo_paper_dev/` | 4 | 128 | 0 |
| Goofspiel | Mirror | `results/goofspiel/emo_dev/20260911_191328_emo_paper_dev/` | 4 | 260 | 0 |

- Independent read-only reconstruction: **12 games, 80 legal actions, 500 stage calls**; every own decision has summary + three CoT calls, every visible opponent event has exactly nine modeling calls. Goofspiel has 65 logical calls/game, including its final revealed opponent action. Prompt stage, sidecar engine step, observation and terminal payoff match engine replay. No provider tokens are fabricated for scripted output.
- Privacy tests cover hidden Leduc/dice substitutions, P1 indistinguishability under different Goofspiel pending bids, both-seat reveal timing, fold non-reveal, old dice/new round scope and role-separated model inputs. Tests also preserve wrong-but-shaped hypotheses (no local pruning), enforce all-or-nothing atomic update, and cover failed/invalid/truncated/ambiguous outputs, cap/source/sidecar tampering, partial replay and complete skip.
- Protection: **all 248 pre-existing files recorded at start have unchanged SHA256** (including AGENTS/PROGRESS and shared method/API/runner code). Branch remains `feat/cross-game-baselines`, HEAD remains `07f087c14651d7a7f13c2dd5c2211b696120fc67`, staged file list unchanged/empty. No checkout, staging or commit. Only EMO-owned new files and new result directories were written.
- The initial validation above used no live model inference or performance measurement. Scripted `None` hypotheses and legal-action fixtures validate wiring, not reasoning quality or strategy effectiveness. The separately approved one-game live check below does not change that limitation into performance evidence.

## One-game Leduc live dev check — 2026-09-11

> **Historical v1 only — not faithful-performance evidence.** This run used recursive internal-text history removed by v2.

- User approved exactly one game. EMO played P0 against the repository's existing local CFR+ 21,000-iteration approximation (`results/leduc/cfr_model/cfr_plus_21000.json`, SHA256 recorded in config), seed 9100. This opponent is labeled GTO for the benchmark but is **not an exact-equilibrium certification**. P1 was not run.
- Preflight after adding the EMO-local Leduc opponent adapter and single-seat schedule: **246 passed** (16.41s), including all EMO/public journal/truncation/retry/wrapper tests. Evidence: `results/leduc/emo_dev/20260911_190237_paper_port_validation_dev/logs/leduc_gto_preflight.txt` and XML.
- Live command parameters: game leduc, opponent GTO, games 1, position P0, start seed 9100, max requests 100, format retries 3, transient retries 3, temperature 1, max tokens omitted, read/write/pool timeout 600s and connect timeout 10s. Phase `dev_debug`; campus DeepSeek-V4-Flash. Run: `results/leduc/emo_dev/20260911_192953_emo_paper_dev/`.
- Result: completed 1/1; EMO P0 net profit **+3 chips**. Cards were P0 J♠ / P1 Q♠, community J♥. Legal action sequence: P0 check, CFR P1 bet 2, P0 call 2, P0 bet 4, CFR P1 fold. Both CFR actions reproduce under the recorded model/seed.
- Calls: 31/100 successful, 0 transport failures, 0 ambiguous, all `finish_reason=stop`. Input 137,655 / output 17,104 / total 154,759 provider-reported tokens. Latency median 2.553s, max 38.905s, summed request latency 350.327s.
- Format correction worked as intended: first opponent-model `init` returned explanatory prose followed by `None`, which failed the strict candidate envelope; one separately logged correction returned `["J", "Q", "K"]`, and the same game continued. Thus 30 paper-stage logical calls plus one engineering correction call. No preceding EMO stage was rerun.
- Read-only audit replayed all five actions through the engine, checked legality/payoff, reproduced both CFR actions, matched all 31 prompt/sidecar/validation identities and source hashes, and found no credential-shaped strings. Evidence: `results/leduc/emo_dev/20260911_192953_emo_paper_dev/logs/live_audit.json`.
- This one selected dev game is only a connectivity/flow example. Its +3 result is not a performance estimate, and no sample was discarded or chosen based on payoff. No additional game, resume, or API call was made.

## Parallel three-game × three-game live dev check — 2026-09-11

> **Historical v1 only — not faithful-performance or faithful token-cost evidence.** Its 2.92M tokens include recursive internal feedback/CoT text that Algorithm 1/2 does not put into history.

- User approved three P0 games per environment versus repository-local approximate GTO/CFR opponents, seeds 9100–9102, with Leduc/Liar's Dice/Goofspiel starting concurrently. No P1 games or payoff-based replacement samples were run. Each game process had a 500-attempt cap, three format corrections and three transient retries.
- Preflight after adding strict no-fallback Goofspiel n5 MCCFR+ support: **247 passed** (35.11s), `results/leduc/emo_dev/20260911_190237_paper_port_validation_dev/logs/three_gto_parallel_preflight.txt` and XML.
- Batch launcher started all three processes at 19:41:34 +08:00; all exited code 0 at 20:34:29 +08:00. Wall time: approximately **52 minutes 56 seconds**. Launcher evidence: `results/leduc/emo_dev/20260911_194112_parallel_3x3_live_dev/logs/batch_result.json`.
- Runs: `results/leduc/emo_dev/20260911_194134_emo_paper_dev/`, `results/liars_dice/emo_dev/20260911_194134_emo_paper_dev/`, and `results/goofspiel/emo_dev/20260911_194134_emo_paper_dev/`.
- Descriptive P0 net results: Leduc `[-1,+1,+1]` (sum +1, mean +0.333); Liar's Dice `[+1,-1,-1]` (sum -1, mean -0.333); Goofspiel `[-6,+7,-2]` (sum -1, mean -0.333). These nine dev games are not an effectiveness estimate.
- Calls: Leduc 45 attempts; dice 146; Goofspiel 196. Total **387 physical attempts**, 386 successful text responses and one Goofspiel read-timeout attempt that recovered through the bounded retry on the same call key. No ambiguous request or truncation. Two Leduc `init` replies needed one format correction each; both corrections succeeded. Dice and Goofspiel needed no format correction.
- Provider-reported usage: 2,743,454 prompt + 178,600 completion = **2,922,054 total tokens**. Longest successful calls were 304.348s Leduc, 288.287s dice and 285.577s Goofspiel; HTTPX values remain phase-inactivity timeouts rather than a total request deadline.
- Read-only audit reconstructed all 61 engine actions, matched terminal payoffs, reproduced all 28 local GTO actions under recorded checkpoints/seeds, checked call/sidecar/output-validation identity, source hashes and credential shapes. No live process remained afterward. Evidence: `results/leduc/emo_dev/20260911_194112_parallel_3x3_live_dev/logs/final_audit.json`.
- Branch remained `feat/cross-game-baselines`, HEAD `07f087c14651d7a7f13c2dd5c2211b696120fc67`, and the Git index remained empty. No checkout, staging or commit.
