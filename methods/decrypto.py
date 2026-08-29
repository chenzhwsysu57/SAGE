"""DecryptoMethod with sectioned prompts (aligned with Leduc format).

Prompt sections: GAME RULES / YOUR ROLE / GAME STATE / YOUR TASK.
Persona system: default baseline, anti_convergence, tom, theme_loyalist.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
from tools.decrypto.interceptor_style import (
    build_interceptor_model,
    infer_interceptor_style_llm,
)

if TYPE_CHECKING:
    from modeling.decrypto.semantic_belief import DecryptoBeliefState
    from modeling.decrypto.locked_theme import LockedThemeState

# ═══════════════════════════════════════════════════════════════════════════
# GAME RULES (static, shared by all personas)
# ═══════════════════════════════════════════════════════════════════════════

_GAME_RULES_TEMPLATE = """#### GAME RULES ####

You are playing Decrypto, a 3-player code guessing game:
- Team 1: Encoder + Decoder (share 4 secret keywords numbered 1-4).
- Team 2: Interceptor (does NOT know the keywords).

Each turn:
1. Encoder receives a 3-digit secret code (digits 1-4, non-repeating, e.g. 3-1-2).
2. Encoder gives 3 hints (a, b, c) — one per code digit — referencing the keyword.
3. Decoder sees hints + keywords → guesses the 3-digit code.
4. Interceptor sees hints only → guesses the 3-digit code.

Scoring:
- Decoder guess wrong → Team 1 gets 1 Miscommunication token.
- Interceptor guess right → Team 2 gets 1 Interception token.
- {{MISCOMMUNICATION_THRESHOLD}} Miscommunications OR {{INTERCEPTION_THRESHOLD}} Interceptions → Team 2 (Interceptor) wins.
- Survive {{MAX_TURNS}} turns → Team 1 (Encoder + Decoder) wins.

Hints: 1-2 words. Codes: unique, never repeat in a game.

Example keywords: {1: "airplane", 2: "guitar", 3: "hat", 4: "plant"}, code 3-1-2.
Encoder needs hints for {3: hat, 1: airplane, 2: guitar} → {"hints": ["gentleman", "wing", "music"]}.

Ignore the example keywords above — new keywords are sampled every game.
Respond with step-by-step reasoning, then final answer after "ANSWER:"."""

# ═══════════════════════════════════════════════════════════════════════════
# ROLE PROMPTS (per persona)
# ═══════════════════════════════════════════════════════════════════════════

# --- Encoder ---

_ENCODER_DEFAULT = """#### YOUR ROLE — Encoder ####

You are the Encoder. Provide your hints as:
  ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}

The Decoder knows the 4 keywords. The Interceptor does NOT.
Hints: 1-2 words. Order matches the code order.

#### ENCODING STRATEGY ####

Choose straightforward clues that make the Decoder's mapping as natural as
possible. A good hint has a clear common-knowledge connection to the intended
keyword and is less natural for the other three keywords.

Do not use the exact secret keyword as a hint. Keep hints short and distinct.
Prioritize Bob's correctness over cleverness."""

_ENCODER_ANTI = """#### YOUR ROLE — Encoder ####

You are the Encoder. Provide your hints as:
  ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}

The Decoder knows the 4 keywords. The Interceptor does NOT.
Hints: 1-2 words. Order matches the code order.

#### ANTI-CONVERGENCE STRATEGY ####

The Interceptor sees all past hints grouped by keyword number and tries to
infer a theme for each keyword. Prevent easy convergence while keeping Bob's
mapping clear.

Before giving a hint for a keyword, look at the previous hints for it. If they
already suggest an obvious theme, choose a new clue from a different meaning,
function, context, or association of the same keyword.

The Decoder knows the keyword and all its meanings. As long as the connection
exists in common knowledge, the Decoder can find it.

Before finalizing: for each hint, Bob's most natural match must be the intended
keyword, not a different one."""

_ENCODER_TOM = """#### YOUR ROLE — ToM Encoder ####

You are the Encoder. Provide your hints as:
  ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}

The Decoder knows the 4 keywords. The Interceptor does NOT.
Hints: 1-2 words. Order matches the code order.

#### THEORY OF MIND STRATEGY ####

Use the INTERCEPTOR MODEL section, if present, to infer Eve's current style
from her past guesses. Adapt your hints to exploit that style:
- theme_loyalist: rotate away from first-hint themes.
- literal: prefer indirect, functional, or contextual clues.
- adaptive: keep each keyword's hints semantically diverse over time.

This is not an oracle. If evidence is sparse or uncertain, use robust baseline
clues and gather more evidence.

Hard constraint: Bob's most natural match must still be the intended keyword."""

_ENCODER_TL = """#### YOUR ROLE — Encoder ####

You are the Encoder. Provide your hints as:
  ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}

#### OPPONENT ANALYSIS ####

The Interceptor locks ONE theme per keyword based on the FIRST hint she sees,
then NEVER changes it. This makes her predictable.

#### REASONING PROCESS — Follow every turn ####

STEP 1 — Infer Eve's locked themes:
  Review the Hint History. The FIRST hint for each keyword defines what Eve locked.
    K1: first hint "___" → Eve locked: ___
    K2: first hint "___" → Eve locked: ___
    K3: first hint "___" → Eve locked: ___
    K4: first hint "___" → Eve locked: ___
  (No past hints → Eve has no theme yet.)

STEP 2 — For each keyword in your code, pick a hint that VIOLATES Eve's lock:
  Eve has locked a specific theme for this keyword. Choose a hint from a
  DIFFERENT semantic angle — one that does NOT fit her locked theme.
    Example: Eve locked WATER on K1. Your keyword is water.
    Bad hint: "stream" (fits WATER → confirms Eve's lock).
    Good hint: "life" (water is essential for life → Eve sees "life",
    doesn't think WATER → confused). The Decoder knows water=life → correct.

STEP 3 — Decoder check:
  For each hint: "Does the Decoder have at least ONE clear path from this
  hint to the keyword?" If not, pick a different hint.
  The Decoder matches hints to keywords normally. You don't need cross-theme
  tricks — just make sure each hint connects to the right keyword while
  violating Eve's locked theme.

STEP 4 — Output:
  ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}"""

# --- Decoder ---

_DECODER_DEFAULT = """#### YOUR ROLE — Decoder ####

You are the Decoder. You know the 4 keywords. Provide your guess as:
  ANSWER: {"guess": "X-Y-Z"}

Match each hint to one keyword. 3 hints → 3 different keywords (no repeats).
Digits follow hint order (a, b, c).

#### DECODING STRATEGY ####

When a hint doesn't immediately fit any keyword, broaden your search:
a keyword may have several distinct meanings, and the Encoder may be
using one you haven't considered yet.

Consecutive hints for the same keyword may feel completely unrelated
to each other. That is intentional — the Encoder may draw from
different facets of the word each turn to keep the Interceptor
guessing. Your advantage is that you know the actual keyword, so you
can recognize connections that an outsider would miss.

Process: for each hint (a, b, c), consider all the meanings of each
keyword. Find the best match. If two hints point to the same keyword,
reassign the weaker one."""

_DECODER_TL = """#### YOUR ROLE — Decoder ####

You are the Decoder. You know the 4 keywords. Provide your guess as:
  ANSWER: {"guess": "X-Y-Z"}

#### REASONING PROCESS — Follow every turn ####

The Encoder tracks what themes Eve has locked and deliberately VIOLATES them
by choosing hints from unexpected angles. A hint that seems surprising or
unrelated to a keyword's usual meaning may still be correct — the Encoder
is avoiding Eve's expectations, not avoiding the keyword.

STEP 1 — Match each hint to the best keyword:
  Hint a "___": best match is K_ because ___
  Hint b "___": best match is K_ because ___
  Hint c "___": best match is K_ because ___

STEP 2 — Verify all 3 keywords differ. If two hints map to the same keyword,
  reassign the weaker match to the next-best keyword.

STEP 3 — Output:
  ANSWER: {"guess": "X-Y-Z"}"""

# --- Interceptor ---

_INTERCEPTOR_DEFAULT = """#### YOUR ROLE — Interceptor ####

You are the Interceptor. You do NOT know the 4 keywords. Provide your guess as:
  ANSWER: {"guess": "X-Y-Z"}

Use the hint history to infer themes per keyword. Match each current hint to the
keyword whose theme fits best. Digits do not repeat (1-4)."""

_INTERCEPTOR_LITERAL = """#### YOUR ROLE — Interceptor ####

You are the Interceptor. You work from the visible hints and candidate words in
three sequential word steps. Provide your guess as:
  ANSWER: {"guess": "X-Y-Z"}

#### YOUR APPROACH ####

You are a Literal player. The three current hints are processed one by one:
STEP 1 — For the first hint, compare it to the remaining candidate words and
select the one with the most direct surface-level match.
STEP 2 — The chosen word identifies a keyword group. That group is removed
from the remaining pool before the next step.
STEP 3 — Repeat the same direct-match process for the second hint using only
the remaining candidate words.
STEP 4 — Repeat the same direct-match process for the third hint using only
the remaining candidate words.
STEP 5 — Output exactly:
  ANSWER: {"guess": "X-Y-Z"}

Stay close to the surface of each hint and keep the interpretation local and
concrete. Use three distinct digits from 1 to 4."""

_INTERCEPTOR_ADAPTIVE = """#### YOUR ROLE — Interceptor ####

You are the Interceptor. You do NOT know the 4 keywords. Provide your guess as:
  ANSWER: {"guess": "X-Y-Z"}

#### YOUR APPROACH ####

You are an Adaptive player. Follow this fixed process:
STEP 1 — Read the current three hints and all completed hint history.
STEP 2 — For each keyword, build the best current theme from all evidence so far.
STEP 3 — If newer hints conflict with earlier impressions, revise the theme
instead of locking the first impression forever.
STEP 4 — Weight recent completed rounds slightly more than very old rounds.
STEP 5 — Match each current hint to the keyword whose updated theme fits best.
STEP 6 — Output exactly:
  ANSWER: {"guess": "X-Y-Z"}

Do not freeze on first impressions. Digits do not repeat (1-4)."""

_INTERCEPTOR_TL = """#### YOUR ROLE — Interceptor ####

You are the Interceptor. You do NOT know the 4 keywords. Provide your guess as:
  ANSWER: {"guess": "X-Y-Z"}

#### YOUR APPROACH ####

You are a Theme-Locking player. You are given a set of locked themes —
one per keyword — in the YOUR LOCKED THEMES section. These locked themes
were formed from the very first hint for each keyword and are now permanent.

Your job is to match each new hint to the locked theme it best fits.
Do not question or update the locked themes. They are fixed facts for you.

#### REASONING PROCESS ####

STEP 1 — Match each hint to a locked theme:
    Hint a "___": fits locked theme K_ ("...") because ___
    Hint b "___": fits locked theme K_ ("...") because ___
    Hint c "___": fits locked theme K_ ("...") because ___

STEP 2 — Resolve:
  - Two hints fit the same locked theme → assign the weaker fit elsewhere.
  - Hint fits no locked theme → assign to a keyword with no locked theme.
  - All 3 keywords must differ (1-4, non-repeating).

STEP 3 — Output:
  ANSWER: {"guess": "X-Y-Z"}"""

# ═══════════════════════════════════════════════════════════════════════════
# PERSONA LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

PERSONA_SP = {
    "theme_loyalist": {
        0: _ENCODER_TL,
        1: _DECODER_TL,
        2: _INTERCEPTOR_TL,
    },
    "anti_convergence": {
        0: _ENCODER_ANTI,
        1: _DECODER_DEFAULT,
        2: _INTERCEPTOR_DEFAULT,
    },
    "tom": {
        0: _ENCODER_TOM,
        1: _DECODER_DEFAULT,
        2: _INTERCEPTOR_DEFAULT,
    },
    "literal": {
        0: _ENCODER_DEFAULT,
        1: _DECODER_DEFAULT,
        2: _INTERCEPTOR_LITERAL,
    },
    "adaptive": {
        0: _ENCODER_DEFAULT,
        1: _DECODER_DEFAULT,
        2: _INTERCEPTOR_ADAPTIVE,
    },
    "default": {
        0: _ENCODER_DEFAULT,
        1: _DECODER_DEFAULT,
        2: _INTERCEPTOR_DEFAULT,
    },
}


def _get_system_prompt(pid: int, persona: str | None, obs: Observation) -> str:
    key = persona if persona and persona in PERSONA_SP else "default"
    pub = obs.public_info
    game_rules = (
        _GAME_RULES_TEMPLATE
        .replace("{{MAX_TURNS}}", str(pub.get("max_turns", 12)))
        .replace(
            "{{MISCOMMUNICATION_THRESHOLD}}",
            str(pub.get("miscommunication_threshold", 3)),
        )
        .replace(
            "{{INTERCEPTION_THRESHOLD}}",
            str(pub.get("interception_threshold", 3)),
        )
    )
    return game_rules + "\n\n" + PERSONA_SP[key][pid]


# ═══════════════════════════════════════════════════════════════════════════
# GAME STATE builder (dynamic — replaces old common info + turn state)
# ═══════════════════════════════════════════════════════════════════════════

def _build_game_state(obs: Observation) -> str:
    """Build #### GAME STATE #### section: score, keywords, hints, histories."""
    pub = obs.public_info
    priv = obs.private_info
    pid = obs.player_id
    turn = pub["turn"]
    mis = pub["miscommunications"]
    inter = pub["interceptions"]
    max_turns = pub.get("max_turns", 12)
    mis_thresh = pub.get("miscommunication_threshold", 3)
    int_thresh = pub.get("interception_threshold", 3)

    lines = [
        "#### GAME STATE ####",
        f"Turn {turn + 1}/{max_turns}.  Miscommunications: {mis}/{mis_thresh}.  Interceptions: {inter}/{int_thresh}.",
    ]

    # Keywords (Alice and Bob only)
    if pid in (0, 1):
        keywords = priv.get("keywords", [])
        if keywords:
            lines.append("")
            lines.append("Secret keywords:")
            for i, kw in enumerate(keywords):
                lines.append(f"  {i + 1}: {kw}")

    # Current code (Alice only)
    if pid == 0:
        code = priv.get("current_code", [])
        if code:
            code_str = "-".join(str(d) for d in code)
            lines.append(f"")
            lines.append(f"Your assigned code: {code_str}")
            kw_indices = [int(d) - 1 for d in code]
            kw_list = ", ".join(
                f"{code[i]}={keywords[kw_indices[i]]}" for i in range(3)
            )
            lines.append(f"Corresponding keywords: {kw_list}")

    # Current hints (Bob and Eve only)
    ch = pub.get("current_hints")
    if ch and pid != 0:
        lines.append("")
        lines.append("Encoder's hints this turn:")
        lines.append(f"  a: {ch[0]}")
        lines.append(f"  b: {ch[1]}")
        lines.append(f"  c: {ch[2]}")

    # Hint history (all players)
    hh = pub.get("hint_history", {})
    if turn > 0:
        lines.append("")
        lines.append("Hint History (per keyword):")
        for pos in ("1", "2", "3", "4"):
            hints = hh.get(pos, [])
            hint_str = ", ".join(hints) if hints else "(none)"
            lines.append(f"  Keyword {pos}: {hint_str}")

    # Code history (Encoder/Decoder only)
    if pid in (0, 1):
        ch_list = pub.get("code_history", [])
        if ch_list:
            codes_str = ", ".join("-".join(str(d) for d in c) for c in ch_list)
            lines.append(f"Past codes: {codes_str}")

    # Guess history (Encoder sees both, for assessing Eve's progress)
    if pid == 0:
        gh = pub.get("guess_history", [])
        if gh:
            lines.append("")
            lines.append("Past guesses (per round):")
            for ri, rec in enumerate(gh):
                code_str = "-".join(str(d) for d in rec["code"])
                bob_str = "-".join(str(d) for d in rec["bob"]) if rec["bob"] else "?"
                eve_str = "-".join(str(d) for d in rec["eve"]) if rec["eve"] else "?"
                bob_ok = "✓" if rec["bob"] == rec["code"] else "✗"
                eve_ok = "✓" if rec["eve"] == rec["code"] else "✗"
                lines.append(
                    f"  R{ri + 1}: code={code_str}, Bob={bob_str}{bob_ok}, Eve={eve_str}{eve_ok}"
                )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# YOUR TASK builders (dynamic — role-specific turn instructions)
# ═══════════════════════════════════════════════════════════════════════════

def _build_encoder_task(obs: Observation) -> str:
    pub = obs.public_info
    priv = obs.private_info
    keywords = priv["keywords"]
    code = priv.get("current_code", [])
    code_str = "-".join(str(d) for d in code)
    turn = pub["turn"]

    lines = [
        "#### YOUR TASK ####",
        f"Provide 3 hints for the code {code_str}.",
        "One or two words each, referencing the meaning of each keyword.",
        "",
        f"The keywords for this code are:",
    ]
    for i, d in enumerate(code):
        kw = keywords[int(d) - 1]
        lines.append(f"  digit {d}: {kw}")

    lines.extend([
        "",
        "Think step-by-step about each hint before giving your final answer.",
        'Output: ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}',
    ])
    return "\n".join(lines)


def _build_decoder_task(obs: Observation) -> str:
    hints = obs.public_info.get("current_hints", [])
    if len(hints) < 3:
        hint_text = "Waiting for hints..."
        hint_tmpl = "a: ?, b: ?, c: ?"
    else:
        hint_text = f"Deduce the 3-digit code from the Encoder's hints:"
        hint_tmpl = f"  a: {hints[0]},  b: {hints[1]},  c: {hints[2]}"

    lines = [
        "#### YOUR TASK ####",
        hint_text,
        hint_tmpl if hint_tmpl != "a: ?, b: ?, c: ?" else "",
        "",
        "Match each hint to one of the 4 keywords above.",
        "Digits follow hint order (a, b, c). No repeating digits.",
        "",
        "Apply concise, step-by-step thinking.",
        'Output: ANSWER: {"guess": "X-Y-Z"}',
    ]
    return "\n".join(line for line in lines if line)


def _build_interceptor_task(obs: Observation) -> str:
    hints = obs.public_info.get("current_hints", [])
    if len(hints) < 3:
        hint_text = "Waiting for hints..."
        hint_tmpl = "a: ?, b: ?, c: ?"
    else:
        hint_text = f"Guess the 3-digit code from the Encoder's hints:"
        hint_tmpl = f"  a: {hints[0]},  b: {hints[1]},  c: {hints[2]}"

    lines = [
        "#### YOUR TASK ####",
        hint_text,
        hint_tmpl,
        "",
        "You do NOT know the keywords. Use the Hint History above.",
        "Digits do not repeat (1-4).",
        "",
        "Reply with very concise, step-by-step thinking.",
        'Output: ANSWER: {"guess": "X-Y-Z"}',
    ]
    return "\n".join(lines)


def _build_literal_interceptor_task(obs: Observation) -> str:
    hints = obs.public_info.get("current_hints", [])
    if len(hints) < 3:
        hint_text = "Waiting for hints..."
        hint_tmpl = "word 1: ?, word 2: ?, word 3: ?"
    else:
        hint_text = "Resolve the 3 current hints one by one:"
        hint_tmpl = (
            f"word 1: {hints[0]}\n"
            f"word 2: {hints[1]}\n"
            f"word 3: {hints[2]}"
        )

    lines = [
        "#### YOUR TASK ####",
        hint_text,
        hint_tmpl,
        "",
        "For each word, compare the current hint to the remaining candidate words.",
        "Choose the most direct surface-level match and commit it before moving on.",
        "After a word is chosen, remove that word's keyword group from the pool.",
        "",
        "Use concise step-by-step thinking.",
        'Output: ANSWER: {"guess": "X-Y-Z"}',
    ]
    return "\n".join(lines)


TASK_BUILDERS = {
    0: _build_encoder_task,
    1: _build_decoder_task,
    2: _build_interceptor_task,
}


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_decrypto_response(raw: str, player_id: int) -> dict | None:
    text = raw.strip()
    m = re.search(r'ANSWER:\s*(\{.*\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_literal_scores(raw: str, candidates: list[str]) -> dict[str, float] | None:
    text = raw.strip()
    m = re.search(r'ANSWER:\s*(\{.*\})', text, re.DOTALL)
    if m:
        payload = m.group(1)
    else:
        m = re.search(r'\{[^{}]*\}', text)
        if not m:
            return None
        payload = m.group(0)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    scores: dict[str, float] = {}
    raw_scores = data.get("scores")
    if isinstance(raw_scores, dict):
        score_source = raw_scores
    elif isinstance(data, dict):
        score_source = {
            key: value
            for key, value in data.items()
            if key not in {"best_word", "best_digit", "guess"}
        }
    else:
        score_source = {}

    for word in candidates:
        value = (
            score_source.get(word)
            if isinstance(score_source, dict)
            else None
        )
        if value is None:
            value = score_source.get(str(word).lower())
        if value is None:
            value = score_source.get(str(word).upper())
        if value is None:
            continue
        try:
            scores[word] = float(value)
        except (TypeError, ValueError):
            continue
    if scores:
        return scores

    best_word = data.get("best_word", data.get("guess"))
    if best_word is not None:
        word = str(best_word)
        if word in candidates:
            return {word: 1.0}
        lowered = word.lower()
        if lowered in candidates:
            return {lowered: 1.0}
    return None


def _literal_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in {"the", "a", "an", "of", "to", "and"}
    }


def _literal_candidate_pool(
    remaining_groups: dict[int, list[str]],
) -> tuple[list[str], dict[str, list[int]]]:
    """Flatten remaining digit groups into concrete word candidates."""
    candidates: list[str] = []
    word_to_digits: dict[str, list[int]] = {}
    seen: set[str] = set()

    for digit in sorted(remaining_groups):
        for word in remaining_groups[digit]:
            token = str(word)
            word_to_digits.setdefault(token, [])
            if digit not in word_to_digits[token]:
                word_to_digits[token].append(digit)
            if token not in seen:
                seen.add(token)
                candidates.append(token)

    return candidates, word_to_digits


def _select_digit_for_word(
    word: str,
    word_to_digits: dict[str, list[int]],
    remaining_groups: dict[int, list[str]],
) -> int | None:
    digits = word_to_digits.get(word, [])
    if not digits:
        return None
    ranked: list[tuple[int, int, int]] = []
    for digit in digits:
        observed = remaining_groups.get(digit, [])
        ranked.append((len(observed), -digit, digit))
    ranked.sort(reverse=True)
    return ranked[0][2]


def _build_literal_guess(
    observation: Observation,
    model: BaseModel,
) -> tuple[str, list[dict[str, Any]]]:
    """Greedy literal interceptor policy.

    The policy processes the three hints one by one. For each hint it asks the
    model to score the remaining candidate words by direct surface similarity,
    then removes the selected word's keyword group from the candidate pool.
    """
    public = observation.public_info
    current_hints = [str(h) for h in public.get("current_hints", [])][:3]
    hint_history = public.get("hint_history", {}) or {}

    remaining_groups: dict[int, list[str]] = {
        digit: [str(h) for h in hint_history.get(str(digit), [])]
        for digit in (1, 2, 3, 4)
    }

    if not any(remaining_groups.values()):
        trace = [
            {
                "word_index": 0,
                "hint": "(bootstrap)",
                "candidate_words": [],
                "selected_word": None,
                "selected_digit": None,
                "score": 0.0,
                "evidence": "(no history)",
                "prompt": "",
                "response": "",
                "remaining_after": [1, 2, 3, 4],
                "bootstrap": True,
            }
        ]
        return "1-2-3", trace

    chosen: list[int] = []
    trace: list[dict[str, Any]] = []

    for idx, hint in enumerate(current_hints):
        candidate_words, word_to_digits = _literal_candidate_pool(remaining_groups)
        word_order = {word: pos for pos, word in enumerate(candidate_words)}
        best_digit: int | None = None
        best_word: str | None = None
        best_score = -1.0
        best_history_len = -1
        best_evidence = ""

        scoring_prompt = _build_literal_scoring_prompt(hint, candidate_words)
        raw_scoring = model.generate(scoring_prompt)
        score_map = _parse_literal_scores(raw_scoring, candidate_words)

        if score_map:
            for word in candidate_words:
                if word not in score_map:
                    continue
                score = score_map[word]
                digits = word_to_digits.get(word, [])
                selected_digit = _select_digit_for_word(
                    word, word_to_digits, remaining_groups
                )
                observed = (
                    remaining_groups.get(selected_digit, [])
                    if selected_digit is not None
                    else []
                )
                evidence = ", ".join(observed[-3:]) if observed else "(no history)"
                candidate = (score, len(observed), -word_order[word])
                incumbent = (
                    best_score,
                    best_history_len,
                    -(word_order[best_word] if best_word else 10_000),
                )
                if (
                    best_word is None
                    or candidate > incumbent
                    or (candidate == incumbent and evidence < best_evidence)
                ):
                    best_word = word
                    best_digit = selected_digit
                    best_score = score
                    best_history_len = len(observed)
                    best_evidence = evidence
        else:
            for word in candidate_words:
                selected_digit = _select_digit_for_word(
                    word, word_to_digits, remaining_groups
                )
                observed = (
                    remaining_groups.get(selected_digit, [])
                    if selected_digit is not None
                    else []
                )
                score, evidence = _best_literal_fallback_score(hint, [word])
                candidate = (score, len(observed), -word_order[word])
                incumbent = (
                    best_score,
                    best_history_len,
                    -(word_order[best_word] if best_word else 10_000),
                )
                if (
                    best_word is None
                    or candidate > incumbent
                    or (candidate == incumbent and evidence < best_evidence)
                ):
                    best_word = word
                    best_digit = selected_digit
                    best_score = score
                    best_history_len = len(observed)
                    best_evidence = evidence

        if best_word is None or best_digit is None:
            best_word = candidate_words[0] if candidate_words else ""
            best_digit = _select_digit_for_word(best_word, word_to_digits, remaining_groups)
        if best_digit is None:
            best_digit = next(iter(remaining_groups))
            best_score = 0.0
            best_evidence = "(fallback)"
            best_word = next(
                (word for word in candidate_words
                 if best_digit in word_to_digits.get(word, [])),
                best_word,
            )

        del remaining_groups[best_digit]
        chosen.append(best_digit)
        trace.append(
            {
                "word_index": idx + 1,
                "hint": hint,
                "candidate_words": candidate_words,
                "selected_word": best_word,
                "selected_digit": best_digit,
                "score": best_score,
                "evidence": best_evidence,
                "prompt": scoring_prompt,
                "response": raw_scoring,
                "remaining_after": list(remaining_groups.keys()),
            }
        )

    guess = "-".join(str(d) for d in chosen)
    return guess, trace


def _best_literal_fallback_score(hint: str, observed_hints: list[str]) -> tuple[float, str]:
    if not observed_hints:
        return 0.0, "(no history)"

    best_score = 0.0
    best_evidence = ""
    for prev in observed_hints:
        score = _local_literal_similarity(hint, prev)
        if score > best_score or (score == best_score and prev < best_evidence):
            best_score = score
            best_evidence = prev
    if best_score <= 0.0:
        return 0.0, "(no surface match)"
    return best_score, best_evidence


def _local_literal_similarity(a: str, b: str) -> float:
    a_tokens = _literal_tokens(a)
    b_tokens = _literal_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0

    overlap = len(a_tokens & b_tokens)
    if overlap:
        return 3.0 + (overlap - 1) * 0.5

    score = 0.0
    for left in a_tokens:
        for right in b_tokens:
            if len(left) >= 4 and len(right) >= 4:
                if left.startswith(right[:4]) or right.startswith(left[:4]):
                    score = max(score, 1.5)
            if left in right or right in left:
                score = max(score, 1.0)
    return score


def _build_literal_scoring_prompt(
    hint: str,
    candidate_words: list[str],
) -> str:
    lines = [
        "#### LITERAL SCORING ####",
        f"Current hint: {hint}",
        "",
        "Score how similar each remaining keyword matches this hint.",
        "Return JSON only with a scores object keyed by the candidate words and a best_word.",
        "",
        "Candidate words:",
    ]
    for word in candidate_words:
        lines.append(f"- {word}")
    lines.append("")
    lines.append(
        'Return: {"scores": {"word": 0-1, "...": 0-1}, "best_word": "word"}'
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Method class
# ═══════════════════════════════════════════════════════════════════════════

class DecryptoMethod(BaseMethod):
    """Decrypto method with sectioned prompt (GAME RULES / YOUR ROLE / GAME STATE / YOUR TASK).

    Parameters:
        persona: "default", "anti_convergence", "tom", "literal", or
            "theme_loyalist".
        belief_state: optional DecryptoBeliefState for semantic keyword inference.
    """

    def __init__(
        self,
        name: str = "Decrypto",
        lang: str = "en",
        belief_state: DecryptoBeliefState | None = None,
        locked_theme_state: LockedThemeState | None = None,
        persona: str | None = None,
    ):
        self.name = name
        self.method_type = "decrypto"
        self.lang = lang
        self.belief_state = belief_state
        self.locked_theme_state = locked_theme_state
        self.persona = persona

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple[Action, str, str]:
        method_data = None
        tom_estimate = None
        if self.locked_theme_state is not None and observation.player_id == 2:
            events = self.locked_theme_state.update_from_observation(
                observation, model
            )
            method_data = {
                "locked_theme_state": self.locked_theme_state.to_dict(),
            }
            if events:
                method_data["locked_theme_events"] = events
        if self.persona == "tom" and observation.player_id == 0:
            tom_estimate, style_prompt, style_raw = infer_interceptor_style_llm(
                observation,
                model,
            )
            method_data = method_data or {}
            method_data["interceptor_style_model"] = tom_estimate.to_dict()
            if style_prompt:
                method_data["interceptor_style_prompt"] = style_prompt
            if style_raw:
                method_data["interceptor_style_raw_response"] = style_raw
        prompt = self._render_prompt(
            observation,
            legal_actions,
            interceptor_style=tom_estimate,
        )
        if self.persona == "literal" and observation.player_id == 2:
            guess, trace = _build_literal_guess(observation, model)
            action = Action(player_id=2, type="guess", data={"code": guess})
            raw_response = self._format_literal_response(trace, guess)
            method_data = method_data or {}
            method_data["literal_policy"] = trace
            method_data["literal_final_response"] = raw_response
            return action, prompt, raw_response, method_data
        raw_response = model.generate(prompt)
        parsed = _parse_decrypto_response(raw_response, observation.player_id)
        action = self._build_action(observation.player_id, parsed)
        if method_data is not None:
            return action, prompt, raw_response, method_data
        return action, prompt, raw_response

    def _render_prompt(
        self,
        obs: Observation,
        legal_actions: list[Action],
        interceptor_style=None,
    ) -> str:
        pid = obs.player_id

        sections = [
            _get_system_prompt(pid, self.persona, obs),  # GAME RULES + YOUR ROLE
        ]

        # Locked-theme injection (TL Eve — right after role, before game state)
        if self.locked_theme_state is not None and pid == 2:
            text = self.locked_theme_state.to_prompt()
            if text:
                sections.append(text)

        # Encoder-side ToM style model injection.
        if self.persona == "tom" and pid == 0:
            estimate = interceptor_style or build_interceptor_model(obs)
            sections.append(estimate.to_prompt())

        # Belief state injection (Eve — semantic keyword inference)
        if self.belief_state is not None and pid == 2:
            self.belief_state.update_from_observation(obs)
            text = self.belief_state.to_prompt(pid)
            if text:
                sections.append(text)

        sections.append(_build_game_state(obs))     # GAME STATE
        if self.persona == "literal" and pid == 2:
            sections.append(_build_literal_interceptor_task(obs))
        else:
            sections.append(TASK_BUILDERS[pid](obs))    # YOUR TASK

        return "\n\n".join(sections)

    @staticmethod
    def _build_action(player_id: int, parsed: dict | None) -> Action:
        if parsed is None:
            return Action(player_id=player_id, type="?", data={})
        if player_id == 0:
            hints = parsed.get("hints", [])
            return Action(player_id=0, type="hint", data={"hints": hints})
        else:
            guess = parsed.get("guess", "")
            return Action(player_id=player_id, type="guess", data={"code": guess})

    @staticmethod
    def _format_literal_response(trace: list[dict[str, Any]], guess: str) -> str:
        lines = []
        for step in trace:
            if step.get("bootstrap"):
                lines.append(
                    "BOOTSTRAP — no completed history yet; defaulting to code 1-2-3"
                )
                continue
            lines.append(
                f"STEP {step['word_index']} — hint: {step['hint']} -> "
                f"word {step['selected_word']} / digit {step['selected_digit']} "
                f"(history match: {step['evidence']}, score={step['score']:.1f})"
            )
            remaining = ",".join(str(d) for d in step["remaining_after"]) or "(none)"
            lines.append(f"  Remaining pool after commit: {remaining}")
        lines.append(f"ANSWER: {{\"guess\": \"{guess}\"}}")
        return "\n".join(lines)
