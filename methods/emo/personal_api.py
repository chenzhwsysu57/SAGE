"""EMO official-DeepSeek transport with physical/logical audit linkage."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from apis.deepseek_journal_wrapper import DeepSeekJournalModel
from base.baseline_calls import ResumeConflict


class PersonalEMOModel(DeepSeekJournalModel):
    """Keep full physical IO while linking it to the current EMO call identity."""

    def bind_physical_context(self, entry: dict) -> None:
        self.emo_call_context = deepcopy(entry)

    def _save(self, path, value):
        super()._save(path, {**value, "emo_call": deepcopy(getattr(self, "emo_call_context", None))})

    def audit_physical(self, run_dir: str | Path) -> dict:
        run_dir = Path(run_dir).resolve()
        files = sorted(self.network_dir.glob("request_*.json"))
        logical_attempts = {}
        for path in (run_dir / "prompts").rglob("*.json"):
            row = json.loads(path.read_text())
            for attempt in [*row["attempts"], row]:
                logical_attempts[attempt["attempt_number"]] = (path, row, attempt)
        if len(files) != len(logical_attempts):
            raise ResumeConflict("Personal physical/logical attempt count mismatch")
        success = failure = reasoning = 0
        for number, path in enumerate(files, 1):
            physical = json.loads(path.read_text())
            logical_path, logical, attempt = logical_attempts.get(number, (None, None, None))
            if (path.is_symlink() or path.name != f"request_{number:06d}.json"
                    or physical.get("request_number") != number
                    or physical.get("url") != self.ENDPOINT
                    or physical.get("request_json") != self._request_payload(logical["prompt"])
                    or physical.get("emo_call", {}).get("journal_path") != str(logical_path.relative_to(run_dir))):
                raise ResumeConflict("Personal physical request differs from EMO journal")
            body_text = physical.get("response_body")
            if physical["status"] == "success":
                try:
                    body = json.loads(body_text)
                    choice = body["choices"][0]
                    message = choice["message"]
                    if (physical["http_status"] != 200 or attempt["status"] != "success"
                            or message["content"] != logical["response"]
                            or choice.get("finish_reason") != logical["finish_reason"]):
                        raise ValueError()
                    reasoning += int(bool(message.get("reasoning_content")))
                    success += 1
                except (ValueError, KeyError, TypeError, IndexError, AttributeError):
                    raise ResumeConflict("Personal raw response differs from EMO journal") from None
            else:
                if attempt["status"] not in ("failed", "started"):
                    raise ResumeConflict("Personal failure differs from EMO journal")
                failure += 1
        return {"physical_requests": len(files), "successes": success,
                "failures_or_ambiguous": failure,
                "responses_with_reasoning_content": reasoning}