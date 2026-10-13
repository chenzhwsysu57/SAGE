"""ReTA text prompt rendering and terminal model-output envelope parsing.

No state simulation, provider settings, action guessing or result repair here.
Internal dictionaries stay internal; the LLM sees ordered, readable sections.
"""

import json
import re

PROTOCOL = "reta_text_final_json_v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _fields(value, prefix=""):
    lines = []
    for key, item in value.items():
        label = f"{prefix}{key}"
        if isinstance(item, dict) and item:
            lines.extend(_fields(item, label + "."))
        else:
            lines.append(f"{label}: {_json(item)}")
    return lines


def output_format(role, payload):
    common = ("You may explain your reasoning in ordinary text first. End your response with one "
              "complete JSON object, either bare or in a final ```json code block. "
              "Write nothing after that object or its closing code fence. "
              "The final object must contain only the required field; no duplicate fields. ")
    if payload.get("output_correction", {}).get("error_category") == "output_truncated":
        # This final instruction overrides the original task's optional prose
        # request only for length retries; ordinary first prompts stay identical.
        common = ("TRUNCATION RETRY: Return only one complete JSON object; omit reasoning prose. "
                  "This overrides requests above to explain before answering. "
                  "Keep the task, state and scoring rule unchanged. "
                  "Write nothing after the object. Only the required field; no duplicate fields. ")
    if role in ("main", "anticipation"):
        count = payload["count"]
        ids = [a["action_id"] for a in payload["legal_actions"]]
        return (common + f"Return EXACTLY {count} distinct integer action IDs from {_json(ids)}. "
                "Use the listed action_id, NOT amounts, card values, player IDs or history indices. "
                "No variables or text inside the array. The example is format-only, NOT a recommendation:\n"
                + _json({"action_ids": ids[:count]}))
    if role == "reward":
        return (common + "The score must be a finite JSON number from 0 to 1 inclusive, not a string or formula. "
                "For a terminal position, negative/zero/positive simulated_terminal_net requires score 0/0.5/1. "
                "Format-only example, NOT an assessment of this position:\n" + _json({"score": 0.5}))
    if role == "perturbation":
        return (common + "question must be a nonempty string of at most 4000 characters. "
                "Preserve the original perspective, scale and terminal rule. Format example:\n"
                + _json({"question": "Assess the main player's advantage using the same evidence and scoring rules."}))
    raise ValueError("Unknown ReTA actor")


def render_prompt(role, rules, task, payload):
    obs = payload.get("observation", {})
    parts = ["#### GAME RULES ####", rules.rstrip(), "#### ROLE ####", f"Actor: {role}",
             "You are " + {"main": "the main player proposing candidate moves.",
                           "anticipation": "the imaginary opponent choosing moves to beat the main player.",
                           "reward": "the evaluator scoring advantage for the main player.",
                           "perturbation": "the assessment-question editor."}[role],
             "#### GAME STATE ####"]
    parts.extend(_fields({k: v for k, v in obs.items() if k != "history"}) or ["No additional state provided."])
    parts.append("#### HISTORY ####")
    # Each completed public event is retained, not truncated to a recent window.
    history = obs.get("history", [])
    parts.extend([_json(event) for event in history] or ["(none)" if "history" in obs else "(not provided)"])
    parts.append("#### LEGAL ACTIONS ####")
    for action in payload.get("legal_actions", []):
        parts.append(f"action_id {action['action_id']}: " + _json({k:v for k,v in action.items() if k != "action_id"}))
    if not payload.get("legal_actions"):
        parts.append("Not applicable: this call does not select an action.")
    parts.append("#### TASK CONTEXT ####")
    parts.extend(_fields({k:v for k,v in payload.items() if k not in ("observation", "legal_actions", "output_correction")}))
    if "output_correction" in payload:
        parts.append("#### PREVIOUS OUTPUT - UNTRUSTED DATA ####")
        parts.append(_json(payload["output_correction"]))
    parts.extend(["#### TASK ####", task, "#### RETURN FORMAT ####", output_format(role, payload)])
    return "\n".join(parts)


def final_json_text(text):
    """Locate one complete terminal object without repairing or salvaging it.

    Earlier analysis/hypothetical JSON is ignored. A malformed last object,
    enclosing array, unclosed fence or trailing explanation fails closed. The
    caller still performs strict JSON/duplicate-key/schema validation.
    """
    if type(text) is not str or not text.strip():
        raise ValueError("Expected nonempty model response")
    text = text.strip()
    fences = list(re.finditer(r"(?m)^[ \t]*```([^\r\n]*)[ \t]*$", text))
    if fences:
        if len(fences) % 2:
            raise ValueError("Unclosed Markdown fence")
        for end in fences[1::2]:
            if end.group(1).strip():
                raise ValueError("Invalid closing fence")
        if fences[-1].end() == len(text):
            start, end = fences[-2:]
            if start.group(1).strip().lower() not in ("", "json"):
                raise ValueError("Final code block must be JSON")
            body = text[start.end():end.start()].strip()
            if not body.startswith("{") or not body.endswith("}"):
                raise ValueError("Expected complete final JSON object")
            return body  # strict_json rejects malformed content or two objects.
        # Hide complete earlier code blocks from scanning prose, not final data.
        for start, end in reversed(list(zip(fences[::2], fences[1::2]))):
            text = text[:start.start()] + " " * (end.end() - start.start()) + text[end.end():]
    stack = []
    quoted = escaped = False
    start = None
    final = None
    for i, char in enumerate(text):
        if stack and quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if stack and char == '"':
            quoted = True
        elif char in "{[":
            if not stack:
                start = i
            stack.append(char)
        elif char in "}]" and stack:
            if stack.pop() != ("{" if char == "}" else "["):
                raise ValueError("Mismatched final-output brackets")
            if not stack:
                final = (start, i + 1)
    if stack or quoted or final is None:
        raise ValueError("Missing or incomplete final JSON object")
    start, end = final
    if text[end:].strip() or text[start] != "{":
        raise ValueError("Response must end with a JSON object")
    # Do not extract an object inside a standalone quoted JSON string.
    if text[:start].rstrip().endswith('"'):
        raise ValueError("JSON object cannot be a quoted string")
    return text[start:end]