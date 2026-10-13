# Strategist independent three-game port — development record

## Scope and ownership

Started 2026-09-11. User approved new-file-only offline development; no branch changes, staging, commits, existing-file edits or live API calls. Existing `value.py` and `tests/test_strategist_value.py` are read-only dependencies, not claimed as files created in this turn. Shared AGENTS.md and PROGRESS_LOG.md are deliberately not edited under the user's narrower scope.

## Algorithm and source

Independent implementation based on Strategist (ICLR 2025), arXiv:2408.10635v3 sections 2.2–2.4, appendices D/F/H. Reference source: https://github.com/jonathanmli/Avalon-LLM/tree/2439c6b50df7b45bae4cfcaa67e1d973e4dd8fa5/strategist . No upstream implementation is copied; publication of the reference code does not establish a redistribution license.

Goal: executable value heuristics, observation-keyed MCTS, a strategy genealogy and softmax-UCB idea library, population self-play feedback, development selection, frozen held-out evaluation. Not parameter training, ReTA minimax, or per-action LLM prompting.

## Acceptance plan

1. Read interfaces and preserve existing source files.
2. Add observation-only adapters and bounded search for JJQQKK Leduc, 2x3 wild-one dice, n5 split-tie public-sequence Goofspiel.
3. Add strategy evolution and a separate offline runner with request/game/search caps and resumable artifacts under results/<game>/strategist_dev/.
4. Validate all three games, both seats, hidden-information invariance, exact rewards, budgets, generated-function failures, lineage/idea updates, held-out split and replay. Scripted results validate software only, not learned-policy effectiveness.

## Status

Implementation in progress. No live API or real measurement approved or executed. A restricted Python AST interpreter is used rather than unrestricted exec; this limits the heuristic language and is an explicit deviation from the paper.
