# NLP/structured Strategist adaptation

This additive implementation replaces model-authored Python with a natural-language strategy plus five bounded weights and one bias. The model output is strict JSON (`strategist.nlp_policy.v1`). Free-form strategy prose is persisted for interpretation but never executed. Repository-owned `compile_policy()` maps validated coefficients into a fixed evaluator template so the existing observation-keyed PUCT interface can be reused.

The learning schedule remains Strategist-like: initial policy, then two cycles of training-feedback reflection, idea generation, policy revision, population self-play, paired parent/child evaluation, dev-only selection, freezing, and three held-out test games. There are seven logical model calls. During all train/dev/test games the selected policy is fixed; no LLM call or policy update occurs inside a game. Test results never affect selection.

Files are additive only: `nlp_strategy.py`, `nlp_backend.py`, `runs/strategist_nlp_live_bench.py`, and `tests/test_strategist_nlp.py`. The prior free-code implementation and historical failed live runs are preserved.

Safety and scope:

- exact schema; duplicate/missing/extra keys rejected;
- each coefficient finite and in `[-2,2]`;
- no code/expression field exists;
- NLP text is base64-encoded only in a source comment for stable policy serialization;
- executable template is entirely repository-owned and checked by the bounded `ValueHeuristic` interpreter;
- fixed local executor and MCTS are used during games;
- format correction remains available for invalid JSON/schema output;
- this is a **template-based NLP/structured Strategist adaptation**, not the paper's GOPS free-function protocol or an exact Appendix-F reproduction.

Offline evidence: `results/leduc/strategist_dev/20260912_011313_nlp_strategy_validation_dev/`. Final test suite: 148 passed in 12.03s. Three full real-engine offline workflows also completed: Leduc `20260912_011509_strategist_offline_dev`, Liar's Dice `20260912_011511_strategist_offline_dev`, Goofspiel `20260912_011512_strategist_offline_dev`; each has 32 train, 6 dev, 3 frozen test games and seven scripted calls, with zero API calls. Scripted results validate mechanics only, not efficacy.

Live use requires separate approval and the same gates as the existing live workflow:

`python -B -u runs/strategist_nlp_live_bench.py --game <game> --live-api --approved-live-api --approval-by <name> --key-file <path>`

Do not reuse or resume the old free-code failed runs under this changed representation.
