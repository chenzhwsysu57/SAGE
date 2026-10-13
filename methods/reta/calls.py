"""Reuse the unchanged durable journal with collision-free recursive call IDs.

Storage stage is always `decision`. Semantic actor is explicit in the JSON
prompt. Encoded journal step = environment_step * 100000 + local call ordinal.
This preserves the public journal schema/budget/truncation/replay protections.
"""

import hashlib
import json

from base.baseline_calls import ModelCallJournal, ResumeConflict


class ReTACallJournal(ModelCallJournal):
    def __init__(self, model, run_dir, max_requests, retry_failed=False, max_retries=3):
        # ReTA-only opt-in: None means no request-count budget. The unchanged
        # base journal uses this attribute ONLY in two >= comparisons. Infinity
        # disables those comparisons without a disguised large finite limit;
        # it is never serialized. All public configuration records use null.
        super().__init__(model, run_dir, 0 if max_requests is None else max_requests,
                         retry_failed=retry_failed, max_retries=max_retries)
        if max_requests is None:
            self.max_requests = float("inf")

    def record_output_validation(self, prompt, response, validation):
        """Separate immutable method outcome; never relabel HTTP success as failure.

        Only the method decides to retry/accept. This hook persists the outcome
        for successful-text and exact failed/length calls, before any next
        call. Stable records survive deterministic replay with no rewrites.
        """
        with self._locked():
            if self._context is None:
                raise ResumeConflict("Output validation requires a current call")
            journal_path = self._path(self._context, "decision")
            row = self._read_records().get(journal_path)
            status, category = validation.get("status"), validation.get("error_category")
            actor_context = getattr(self, "_response_actor_context", None)
            truncated = status == "truncated" and category == "output_truncated"
            normal = ((status == "valid" and category is None) or
                      (status == "invalid" and category in {
                          "invalid_json", "invalid_schema", "terminal_score_mismatch", "invalid_action"}))
            response_hash = hashlib.sha256(response.encode("utf-8", errors="surrogatepass")).hexdigest()
            if (row is None or row["context"] != self._context
                    or self._context != getattr(self, "_response_context", None)
                    or row["prompt_hash"] != hashlib.sha256(prompt.encode("utf-8", errors="surrogatepass")).hexdigest()
                    or row["response"] != self._redact(response)
                    or not actor_context or validation.get("actor") != actor_context["actor"]
                    or ("node" in validation and validation["node"] != actor_context["node"])
                    or ("response_sha256" in validation and validation["response_sha256"] != response_hash)
                    or not ((normal and row["status"] == "success" and row.get("finish_reason") != "length")
                            or (truncated and row["status"] == "failed" and row.get("finish_reason") == "length"
                                and actor_context["actor"] in {"main", "anticipation", "reward", "perturbation"}
                                and validation.get("node") == actor_context["node"]
                                and validation.get("response_sha256") == response_hash))):
                raise ResumeConflict("Output validation does not match cached response")
            path = self.run_dir / "logs/output_validation" / f"P{self._context['player_id']}" / journal_path.name
            record = {"schema_version": 1, "context": dict(self._context),
                      "journal_path": str(journal_path.relative_to(self.run_dir)),
                      "attempt_number": row["attempt_number"], "validation": validation}
            self._safe_path(path)
            if path.exists():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise ResumeConflict("Unreadable output validation record") from None
                if previous != record:
                    raise ResumeConflict("Output validation changed on replay")
            else:
                super()._write(path, record)  # Atomic persistence; outside prompts/ journal namespace.
            return str(path.relative_to(self.run_dir))

    def set_actor_context(self, actor, node):
        self.actor_context = {"actor": actor, "node": node}

    def _write(self, path, record):
        super()._write(path, record)
        # Called once for EVERY durable physical attempt, including retries.
        # Successful replay never emits a new started record or network audit.
        bind = getattr(self.model, "set_request_context", None)
        if record["status"] == "started" and callable(bind):
            bind(context=record["context"], attempt_number=record["attempt_number"],
                 attempt_in_call=len(record["attempts"]) + 1,
                 journal_path=str(path.relative_to(self.run_dir)),
                 actor_context=getattr(self, "actor_context", None))

    def set_decision(self, game: str, player_id: int, game_index: int, step: int):
        if type(step) is not int or not 1 <= step <= 20000:
            raise ValueError("Environment step outside ReTA journal range")
        self.decision = {"game": game, "player_id": player_id, "game_index": game_index}
        self.environment_step = step
        self.ordinal = 0
        self.actor_context = None

    def generate(self, prompt: str) -> str:
        if not hasattr(self, "decision") or self.ordinal >= 99999:
            raise ValueError("Missing decision context or call identity exhausted")
        with self._locked():
            self.ordinal += 1
            self.set_context(**self.decision, step=self.environment_step * 100000 + self.ordinal)
            actor_context = dict(getattr(self, "actor_context", None) or {})
            self._response_actor_context = None
            self._response_context = None
            self.last_response_usage = {}
            previous = self.pass_length_output
            self.pass_length_output = actor_context.get("actor") in {
                "main", "anticipation", "reward", "perturbation"}
            try:
                # Dynamic dispatch preserves the runner's read-only replay and
                # evidence collection AFTER super().generate_stage returns.
                response = self.generate_stage("decision", prompt)
                row = self._read_records()[self._path(self._context, "decision")]
                # Trace usage is historical evidence, NOT newly billed usage.
                # Base last_usage deliberately remains empty for cached calls.
                self.last_response_usage = dict(row["usage"])
                self._response_actor_context = actor_context
                self._response_context = dict(self._context)
                return response
            finally:
                self.pass_length_output = previous