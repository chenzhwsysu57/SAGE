"""Statistical Interceptor style model for Decrypto.

This module is intentionally lightweight: it uses only information visible to
the Encoder after completed rounds (hint history, code history, Eve guesses).
It does not call an LLM or embeddings.

Primary entrypoint:
    build_interceptor_model(history) -> InterceptorStyleEstimate
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from typing import Any

STYLES = ("theme_loyalist", "literal", "adaptive")


@dataclass
class InterceptorStyleEstimate:
    posterior: dict[str, float]
    features: dict[str, Any]
    rounds_observed: int
    exploit: str

    @property
    def top_style(self) -> str:
        return max(self.posterior, key=self.posterior.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "posterior": dict(self.posterior),
            "features": dict(self.features),
            "rounds_observed": self.rounds_observed,
            "top_style": self.top_style,
            "exploit": self.exploit,
        }

    def to_prompt(self) -> str:
        lines = ["#### INTERCEPTOR MODEL ####"]
        if self.rounds_observed == 0:
            lines.append("No completed Eve guesses yet. Use robust baseline clues.")
        else:
            lines.append(f"Completed rounds observed: {self.rounds_observed}")
            lines.append("Estimated Eve style:")
            for style, prob in sorted(
                self.posterior.items(), key=lambda item: -item[1]
            ):
                lines.append(f"- {style}: {prob:.2f}")
        lines.append("")
        lines.append(f"Exploit guidance: {self.exploit}")
        lines.append(
            "Hard constraint: Bob must still have the most natural match for every hint."
        )
        return "\n".join(lines)


@dataclass
class InterceptorStyleModel:
    """Compatibility wrapper around ``build_interceptor_model``."""

    temperature: float = 1.0
    history: list[dict[str, Any]] = field(default_factory=list)
    latest: InterceptorStyleEstimate | None = None

    def update(self, history: Any) -> InterceptorStyleEstimate:
        estimate, records = build_interceptor_model(
            history, temperature=self.temperature, return_records=True
        )
        self.history = records
        self.latest = estimate
        return estimate

    def update_from_observation(self, obs: Any) -> InterceptorStyleEstimate:
        return self.update(obs)

    def to_dict(self) -> dict[str, Any]:
        estimate = self.latest or build_interceptor_model(
            [], temperature=self.temperature
        )
        return {
            "temperature": self.temperature,
            "history": list(self.history),
            "latest": estimate.to_dict(),
        }


def build_interceptor_model(
    history: Any,
    temperature: float = 1.0,
    *,
    return_records: bool = False,
) -> InterceptorStyleEstimate | tuple[InterceptorStyleEstimate, list[dict[str, Any]]]:
    """Build an interceptor style posterior from historical completed rounds.

    Args:
        history: one of:
            - list[round_record] where each record already has code/hints/bob/eve
            - trajectory-like dict with ``steps`` or ``state_after``
            - observation-like object with ``public_info``
            - raw public_info dict containing ``code_history`` / ``hint_history`` /
              ``guess_history``
        temperature: softmax temperature for the posterior.
        return_records: if True, return ``(estimate, normalized_records)``.
    """
    records = normalize_interceptor_history(history)
    estimate = estimate_interceptor_style(records, temperature)
    if return_records:
        return estimate, records
    return estimate


def normalize_interceptor_history(history: Any) -> list[dict[str, Any]]:
    """Normalize various history shapes into round records.

    This is the only place that knows how to peel data out of observations,
    trajectories, or already materialized per-round records.
    """
    if history is None:
        return []

    if isinstance(history, list):
        if not history:
            return []
        if all(isinstance(item, dict) and "code" in item for item in history):
            return [_normalize_round_record(item, idx + 1) for idx, item in enumerate(history)]
        # Allow a raw trajectory.steps-like list of step dicts.
        if all(isinstance(item, dict) and "state_after" in item for item in history):
            state_after = history[-1].get("state_after", {})
            return reconstruct_round_records(state_after)
        return []

    if hasattr(history, "public_info"):
        records = reconstruct_round_records(history.public_info)
        if records:
            return records
        if hasattr(history, "history"):
            return reconstruct_round_records_from_actions(history.history)
        return []

    if hasattr(history, "model_dump") and not isinstance(history, dict):
        data = history.model_dump()
        if "public_info" in data:
            records = reconstruct_round_records(data["public_info"])
            if records:
                return records
            if "history" in data:
                return reconstruct_round_records_from_actions(data["history"])
            return records
        return _history_from_trajectory_dict(data)

    if isinstance(history, dict):
        if "public_info" in history:
            records = reconstruct_round_records(history["public_info"])
            if records:
                return records
            if "history" in history:
                return reconstruct_round_records_from_actions(history["history"])
            return records
        if "steps" in history:
            return _history_from_trajectory_dict(history)
        if "history" in history:
            return reconstruct_round_records_from_actions(history["history"])
        if "code_history" in history or "guess_history" in history or "hint_history" in history:
            return reconstruct_round_records(history)

    return []


def reconstruct_round_records(public_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct per-round hints from code_history and grouped hint_history."""
    code_history = public_info.get("code_history", [])
    guess_history = public_info.get("guess_history", [])
    hint_history = public_info.get("hint_history", {})
    consumed = {str(pos): 0 for pos in (1, 2, 3, 4)}
    records = []

    for idx, code in enumerate(code_history):
        if idx >= len(guess_history):
            break
        hints = []
        for digit in code:
            key = str(digit)
            digit_hints = hint_history.get(key, [])
            offset = consumed.get(key, 0)
            hint = digit_hints[offset] if offset < len(digit_hints) else ""
            hints.append(hint)
            consumed[key] = offset + 1

        guess = guess_history[idx]
        records.append({
            "round": idx + 1,
            "code": [int(d) for d in code],
            "hints": hints,
            "bob": guess.get("bob"),
            "eve": guess.get("eve"),
        })

    return records


def reconstruct_round_records_from_actions(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct completed round records from a structured action history.

    This accepts the encoder-visible ``Observation.history`` format used in the
    Decrypto trajectories:
        - encoder hint action
        - decoder guess action
        - interceptor guess action
    grouped by turn.
    """
    if not history:
        return []

    by_turn: dict[int, dict[str, Any]] = {}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        turn = entry.get("turn")
        if not isinstance(turn, int):
            continue
        bucket = by_turn.setdefault(turn, {"hints": [], "bob": None, "eve": None, "code": None})
        player = entry.get("player")
        action = entry.get("action")
        data = entry.get("data") or {}
        if action == "hint" and player == 0:
            hints = data.get("hints")
            if isinstance(hints, list):
                bucket["hints"] = [str(h) for h in hints]
            code = data.get("code")
            if isinstance(code, list):
                bucket["code"] = [int(d) for d in code]
        elif action == "guess" and player == 1:
            guess = data.get("guess") or data.get("code")
            bucket["bob"] = _parse_guess(guess)
            code = data.get("code")
            if isinstance(code, list):
                bucket["code"] = [int(d) for d in code]
        elif action == "guess" and player == 2:
            guess = data.get("guess") or data.get("code")
            bucket["eve"] = _parse_guess(guess)
            code = data.get("code")
            if isinstance(code, list):
                bucket["code"] = [int(d) for d in code]

    records: list[dict[str, Any]] = []
    for turn in sorted(by_turn):
        row = by_turn[turn]
        if not row["hints"] or row["bob"] is None or row["eve"] is None or row["code"] is None:
            continue
        records.append(
            {
                "round": turn,
                "code": row["code"],
                "hints": row["hints"],
                "bob": row["bob"],
                "eve": row["eve"],
            }
        )
    return records


def _history_from_trajectory_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    steps = data.get("steps", [])
    if not steps:
        return []
    last = steps[-1]
    if isinstance(last, dict) and "state_after" in last:
        return reconstruct_round_records(last.get("state_after", {}))
    return []


def _normalize_round_record(item: dict[str, Any], default_round: int) -> dict[str, Any]:
    code = item.get("code", [])
    hints = item.get("hints", [])
    return {
        "round": int(item.get("round", default_round)),
        "code": [int(d) for d in code],
        "hints": list(hints),
        "bob": item.get("bob"),
        "eve": item.get("eve"),
    }


def _parse_guess(raw_guess: Any) -> list[int]:
    if raw_guess is None:
        return []
    if isinstance(raw_guess, list):
        return [int(x) for x in raw_guess]
    if isinstance(raw_guess, str):
        parts = re.findall(r"\d+", raw_guess)
        return [int(p) for p in parts]
    return []


# ── Per-round centroids (learned from 90-game corpus) ──────────────────
# Each entry: n_rounds -> {style -> {feature: mean}}
# Features: stability, entropy, switch_rate, hit_rate, acc_improvement
_CENTROIDS: dict[int, dict[str, dict[str, float]]] = {
    2:  {"theme_loyalist": {"stability": 0.619, "entropy": 0.761, "switch_rate": 0.761, "hit_rate": 0.503, "acc_improvement": 0.483},
         "literal":        {"stability": 0.617, "entropy": 0.767, "switch_rate": 0.767, "hit_rate": 0.439, "acc_improvement": 0.356},
         "adaptive":       {"stability": 0.619, "entropy": 0.761, "switch_rate": 0.761, "hit_rate": 0.519, "acc_improvement": 0.517}},
    3:  {"theme_loyalist": {"stability": 0.639, "entropy": 0.825, "switch_rate": 0.640, "hit_rate": 0.531, "acc_improvement": 0.456},
         "literal":        {"stability": 0.598, "entropy": 0.902, "switch_rate": 0.718, "hit_rate": 0.474, "acc_improvement": 0.467},
         "adaptive":       {"stability": 0.684, "entropy": 0.734, "switch_rate": 0.567, "hit_rate": 0.615, "acc_improvement": 0.557}},
    4:  {"theme_loyalist": {"stability": 0.662, "entropy": 0.867, "switch_rate": 0.550, "hit_rate": 0.586, "acc_improvement": 0.387},
         "literal":        {"stability": 0.592, "entropy": 1.025, "switch_rate": 0.650, "hit_rate": 0.514, "acc_improvement": 0.420},
         "adaptive":       {"stability": 0.692, "entropy": 0.810, "switch_rate": 0.488, "hit_rate": 0.653, "acc_improvement": 0.439}},
    5:  {"theme_loyalist": {"stability": 0.661, "entropy": 0.927, "switch_rate": 0.497, "hit_rate": 0.608, "acc_improvement": 0.315},
         "literal":        {"stability": 0.588, "entropy": 1.106, "switch_rate": 0.630, "hit_rate": 0.538, "acc_improvement": 0.304},
         "adaptive":       {"stability": 0.715, "entropy": 0.806, "switch_rate": 0.427, "hit_rate": 0.695, "acc_improvement": 0.321}},
    6:  {"theme_loyalist": {"stability": 0.674, "entropy": 0.961, "switch_rate": 0.474, "hit_rate": 0.646, "acc_improvement": 0.297},
         "literal":        {"stability": 0.580, "entropy": 1.192, "switch_rate": 0.612, "hit_rate": 0.555, "acc_improvement": 0.248},
         "adaptive":       {"stability": 0.740, "entropy": 0.801, "switch_rate": 0.383, "hit_rate": 0.733, "acc_improvement": 0.310}},
    7:  {"theme_loyalist": {"stability": 0.676, "entropy": 0.969, "switch_rate": 0.448, "hit_rate": 0.669, "acc_improvement": 0.277},
         "literal":        {"stability": 0.592, "entropy": 1.202, "switch_rate": 0.578, "hit_rate": 0.570, "acc_improvement": 0.226},
         "adaptive":       {"stability": 0.759, "entropy": 0.778, "switch_rate": 0.345, "hit_rate": 0.762, "acc_improvement": 0.302}},
    8:  {"theme_loyalist": {"stability": 0.701, "entropy": 0.953, "switch_rate": 0.433, "hit_rate": 0.693, "acc_improvement": 0.256},
         "literal":        {"stability": 0.595, "entropy": 1.231, "switch_rate": 0.580, "hit_rate": 0.571, "acc_improvement": 0.156},
         "adaptive":       {"stability": 0.771, "entropy": 0.777, "switch_rate": 0.326, "hit_rate": 0.773, "acc_improvement": 0.296}},
    9:  {"theme_loyalist": {"stability": 0.711, "entropy": 0.944, "switch_rate": 0.416, "hit_rate": 0.709, "acc_improvement": 0.272},
         "literal":        {"stability": 0.610, "entropy": 1.239, "switch_rate": 0.561, "hit_rate": 0.590, "acc_improvement": 0.163},
         "adaptive":       {"stability": 0.781, "entropy": 0.759, "switch_rate": 0.311, "hit_rate": 0.785, "acc_improvement": 0.312}},
    10: {"theme_loyalist": {"stability": 0.721, "entropy": 0.937, "switch_rate": 0.403, "hit_rate": 0.720, "acc_improvement": 0.263},
         "literal":        {"stability": 0.612, "entropy": 1.261, "switch_rate": 0.554, "hit_rate": 0.597, "acc_improvement": 0.158},
         "adaptive":       {"stability": 0.788, "entropy": 0.750, "switch_rate": 0.309, "hit_rate": 0.790, "acc_improvement": 0.323}},
    11: {"theme_loyalist": {"stability": 0.724, "entropy": 0.939, "switch_rate": 0.399, "hit_rate": 0.722, "acc_improvement": 0.253},
         "literal":        {"stability": 0.618, "entropy": 1.264, "switch_rate": 0.549, "hit_rate": 0.611, "acc_improvement": 0.164},
         "adaptive":       {"stability": 0.791, "entropy": 0.746, "switch_rate": 0.305, "hit_rate": 0.793, "acc_improvement": 0.318}},
}

_FEAT_NAMES = ("stability", "entropy", "switch_rate", "hit_rate", "acc_improvement")


def _keyword_chains(records: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Extract per-keyword assignment chains from completed round records."""
    chains: dict[int, list[int]] = {k: [] for k in (1, 2, 3, 4)}
    for rec in records:
        code = rec["code"]
        eve = rec.get("eve") or []
        for pos, kw_num in enumerate(code):
            if pos < len(eve):
                chains[kw_num].append(int(eve[pos]))
    return chains


def _chain_features(chains: dict[int, list[int]]) -> dict[str, float]:
    """Compute the five key features from per-keyword tracking chains."""
    stab_vals: list[float] = []
    ent_vals: list[float] = []
    total_sw = 0
    total_tr = 0
    n_correct = 0
    n_total = 0
    first_c = first_t = last_c = last_t = 0

    for kw in (1, 2, 3, 4):
        chain = chains.get(kw, [])
        n = len(chain)
        if n < 2:
            continue

        counts: dict[int, int] = {}
        for v in chain:
            counts[v] = counts.get(v, 0) + 1
        most = max(counts.values())
        stab_vals.append(most / n)

        dist = [c / n for c in counts.values()]
        ent_val = -sum(p * math.log2(p) for p in dist if p > 0)
        ent_vals.append(ent_val)

        for i in range(1, n):
            total_tr += 1
            if chain[i] != chain[i - 1]:
                total_sw += 1

        mid = n // 2
        for i, v in enumerate(chain):
            correct = 1 if v == kw else 0
            n_total += 1
            n_correct += correct
            if i < mid:
                first_t += 1
                first_c += correct
            else:
                last_t += 1
                last_c += correct

    if not stab_vals or total_tr == 0:
        return {"stability": 0.5, "entropy": 1.0, "switch_rate": 0.5,
                "hit_rate": 0.5, "acc_improvement": 0.0}

    first_acc = first_c / first_t if first_t > 0 else 0.5
    last_acc = last_c / last_t if last_t > 0 else 0.5

    return {
        "stability": sum(stab_vals) / len(stab_vals),
        "entropy": sum(ent_vals) / len(ent_vals),
        "switch_rate": total_sw / total_tr,
        "hit_rate": n_correct / n_total,
        "acc_improvement": last_acc - first_acc,
    }


def estimate_interceptor_style(
    records: list[dict[str, Any]],
    temperature: float = 1.0,
) -> InterceptorStyleEstimate:
    n = len(records)
    if n < 2:
        posterior = {s: 1.0 / 3 for s in STYLES}
        exploit = _exploit_text("theme_loyalist", n)
        return InterceptorStyleEstimate(
            posterior=posterior,
            features={"rounds_observed": n},
            rounds_observed=n,
            exploit=exploit,
        )

    chains = _keyword_chains(records)
    features = _chain_features(chains)
    features["rounds_observed"] = n

    # Look up centroids for this round count (clamp to 3..11)
    centroid_key = max(3, min(n, 11))
    centroids = _CENTROIDS[centroid_key]

    # Compute negative squared-distance scores
    scores: dict[str, float] = {}
    for style in STYLES:
        dist_sq = sum(
            (features.get(fn, 0.5) - centroids[style][fn]) ** 2
            for fn in _FEAT_NAMES
        )
        scores[style] = -dist_sq

    posterior = _softmax(scores, temperature)
    top = max(posterior, key=posterior.get)
    exploit = _exploit_text(top, n)
    return InterceptorStyleEstimate(
        posterior=posterior,
        features=features,
        rounds_observed=n,
        exploit=exploit,
    )


def _build_keyword_table(records: list[dict[str, Any]]) -> list[str]:
    """Build a keyword-tracking table: for each keyword (1-4), show hints
    and Eve's guessed-position across rounds."""
    kw_hints: dict[int, list[tuple[int, str]]] = {k: [] for k in (1, 2, 3, 4)}
    kw_eve: dict[int, list[tuple[int, int]]] = {k: [] for k in (1, 2, 3, 4)}
    for rec in records:
        rnd = int(rec.get("round", 0))
        code = rec.get("code", [])
        hints = rec.get("hints", [])
        eve = rec.get("eve", [])
        for pos, (actual_kw, hint) in enumerate(zip(code, hints)):
            if isinstance(actual_kw, int):
                kw_hints[actual_kw].append((rnd, hint))
        for pos, guessed_kw in enumerate(eve):
            if pos < len(code):
                actual_kw = code[pos]
                if isinstance(actual_kw, int) and isinstance(guessed_kw, int):
                    kw_eve[actual_kw].append((rnd, guessed_kw))

    lines = [
        "",
        "  Keyword tracking (Hint history + Eve's guess per keyword):",
        f"  {'Kw':>3} | {'Hints (round:hint)':<40} | Eve assigns kw# across rounds",
        f"  {'-'*3}-+-{'-'*40}-+-{'-'*40}",
    ]
    for kw in (1, 2, 3, 4):
        hints_str = ", ".join(f"r{r}:{h}" for r, h in kw_hints[kw])
        eve_str = ", ".join(f"r{r}:→{g}" for r, g in kw_eve[kw])
        # Consistency check
        assignments = [g for _, g in kw_eve[kw]]
        if len(assignments) >= 2 and len(set(assignments)) == 1:
            tag = "  ← always kw" + str(assignments[0])
        elif len(hints_str) > 0:
            tag = "  (varied)"
        else:
            tag = ""
        lines.append(f"  {kw:>3} | {hints_str:<40} | {eve_str:<40}{tag}")

    return lines


def build_interceptor_style_prompt(records: list[dict[str, Any]]) -> str:
    """Render an LLM prompt that classifies Eve's style from visible history."""
    lines = [
        "#### INTERCEPTOR STYLE INFERENCE ####",
        "",
        "You are an encoder-side observer in Decrypto.",
        "Analyze the visible completed-round history in a structured way.",
        "Use only the completed-round history shown here.",
        "",
        "Step 1 — Build a round-by-round table of four signals (0-3):",
        "- anchoring: does Eve keep reusing an early pattern across rounds?",
        "- revision: does Eve change earlier guesses when new evidence arrives?",
        "- locality: is each guess driven by surface similarity to that round's hints?",
        "- stability: does Eve stay with a consistent assignment pattern?",
        "",
        "Step 2 — Keyword tracking (KEY diagnostic below).",
        "Each code position maps to a keyword (1-4). The tracking table shows,",
        "for each keyword, the sequence of hints it received across rounds and",
        "what position Eve assigned to it each time.",
        "",
        "Use the tracking table LIKE THIS:",
        "- Check each keyword column: does Eve assign it to the same position",
        "  across rounds, or does it jump around?",
        "- A keyword that maps to the same position for 3+ consecutive rounds",
        "  suggests a LOCKED mapping (theme_loyalist).",
        "- If assignments IMPROVE over time (early wrong, later correct),",
        "  the Eve learns from evidence (adaptive).",
        "- If assignments ping-pong with no pattern or improvement,",
        "  Eve does fresh surface matching each round (literal).",
    ]
    lines.extend(_build_keyword_table(records))
    lines.extend([
        "",
        "Step 3 — Compare signals to these style families:",
        "- theme_loyalist: Keyword tracking shows CONSISTENT keyword mapping.",
        "  Eve may start with a wrong mapping (round 1-2 bootstrap), but once",
        "  set she does NOT change it — even when wrong. High anchoring + stability;",
        "  low revision. The keyword column shows the same position repeated.",
        "- literal: Fresh surface matching each round. Keyword tracking shows",
        "  NO consistency: each round's assignment driven by whichever current",
        "  hint has the most obvious concrete match. High locality; low anchoring,",
        "  revision, and stability.",
        "- adaptive: Starts uncertain, improves over rounds. Keyword tracking",
        "  shows early varied assignments that gradually converge to correct",
        "  positions. High revision; medium anchoring early, low late;",
        "  increasing stability.",
        "",
        "Step 4 — Return JSON only with this schema:",
        '{'
        '"round_signals": ['
        '{"round": 1, "anchoring": 0, "revision": 0, "locality": 0, "stability": 0, "notes": "..."}, '
        '{"round": 2, "anchoring": 0, "revision": 0, "locality": 0, "stability": 0, "notes": "..."}'
        '], '
        '"signals": {'
        '"anchoring": {"score": 0, "evidence": ["..."]}, '
        '"revision": {"score": 0, "evidence": ["..."]}, '
        '"locality": {"score": 0, "evidence": ["..."]}, '
        '"stability": {"score": 0, "evidence": ["..."]}'
        '}, '
        '"posterior": {"theme_loyalist": 0.0, "literal": 0.0, "adaptive": 0.0}, '
        '"top_style": "theme_loyalist|literal|adaptive", '
        '"confidence": 0.0, '
        '"evidence": ["..."], '
        '"exploit": "..."'
        '}',
        "",
        "Completed rounds:",
    ])
    if not records:
        lines.append("  (none)")
    else:
        for rec in records:
            round_num = rec.get("round", "?")
            code = "-".join(str(d) for d in rec.get("code", [])) or "(none)"
            hints = rec.get("hints", [])
            hints_text = ", ".join(str(h) for h in hints) if hints else "(none)"
            bob = _format_guess(rec.get("bob"))
            eve = _format_guess(rec.get("eve"))
            lines.extend(
                [
                    f"  Round {round_num}:",
                    f"    code: {code}",
                    f"    hints: {hints_text}",
                    f"    Bob guess: {bob}",
                    f"    Eve guess: {eve}",
                ]
            )
    return "\n".join(lines)


def infer_interceptor_style_llm(
    history: Any,
    model: Any,
    temperature: float = 1.0,
) -> tuple[InterceptorStyleEstimate, str, str]:
    """Infer Eve style with an LLM, falling back to the heuristic estimator.

    Returns:
        estimate, prompt, raw_response
    """
    records = normalize_interceptor_history(history)
    if len(records) < 2:
        estimate = estimate_interceptor_style(records, temperature)
        return estimate, "", ""

    prompt = build_interceptor_style_prompt(records)
    raw_response = model.generate(prompt)
    parsed = _parse_interceptor_style_response(raw_response, records, temperature)
    if parsed is None:
        parsed = estimate_interceptor_style(records, temperature)
    return parsed, prompt, raw_response


def _parse_interceptor_style_response(
    raw_response: str,
    records: list[dict[str, Any]],
    temperature: float,
) -> InterceptorStyleEstimate | None:
    json_str = _extract_json(raw_response)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    posterior_data = data.get("posterior") or data.get("scores")
    posterior = _normalize_posterior(posterior_data)
    if posterior is None:
        return None

    top_style = str(data.get("top_style") or max(posterior, key=posterior.get))
    if top_style not in STYLES:
        top_style = max(posterior, key=posterior.get)
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = posterior[top_style]
    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    exploit = data.get("exploit")
    if not isinstance(exploit, str) or not exploit.strip():
        exploit = _exploit_text(top_style, len(records))

    chains = _keyword_chains(records)
    features = _chain_features(chains)
    features["rounds_observed"] = len(records)
    return InterceptorStyleEstimate(
        posterior=posterior,
        features={
            **features,
            "llm_confidence": float(confidence),
            "llm_evidence": [str(item) for item in evidence],
        },
        rounds_observed=len(records),
        exploit=exploit,
    )


def _normalize_posterior(data: Any) -> dict[str, float] | None:
    if not isinstance(data, dict):
        return None
    values = {style: float(data.get(style, 0.0) or 0.0) for style in STYLES}
    total = sum(max(v, 0.0) for v in values.values())
    if total <= 0:
        return None
    return {style: max(values[style], 0.0) / total for style in STYLES}


def _extract_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _format_guess(guess: Any) -> str:
    if isinstance(guess, list):
        return "-".join(str(d) for d in guess)
    if guess is None:
        return "(none)"
    return str(guess)


def _exploit_text(style: str, rounds_observed: int) -> str:
    if rounds_observed < 2:
        return (
            "Evidence is sparse. Use clear baseline clues and avoid creating "
            "an obvious repeated theme."
        )
    if style == "theme_loyalist":
        return (
            "Eve appears to overweight first-hint themes. Rotate to a different "
            "meaning or context for repeated keywords while preserving Bob's "
            "clearest match."
        )
    if style == "literal":
        return (
            "Eve appears to rely on surface clue matching. Prefer indirect, "
            "functional, or contextual clues that Bob can map to the keyword."
        )
    return (
        "Eve appears adaptive. Keep each keyword's hints semantically diverse "
        "over time and avoid converging on one cluster."
    )


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    temp = max(temperature, 1e-6)
    scaled = {k: v / temp for k, v in scores.items()}
    max_score = max(scaled.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scaled.items()}
    total = sum(exp_scores.values())
    return {k: exp_scores[k] / total for k in STYLES}
