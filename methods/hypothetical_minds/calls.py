"""Reuse public journal unchanged; HM stage is recorded in the JSON prompt.

Store all calls as 'decision' with step = environment_step * 1000 + ordinal.
Schema corrections have their own call identities; all attempts share the cap.
"""

import hashlib

from base.baseline_calls import ModelCallJournal, ResumeConflict
from .storage import immutable_json


class HMCallJournal(ModelCallJournal):
    def __init__(self, model, run_dir, max_requests, *, retry_failed=False):
        # Explicit None means no request-count cap. Use infinity only in the
        # inherited comparisons; configs/logs serialize None, never infinity.
        super().__init__(model, run_dir, 0 if max_requests is None else max_requests,
                         retry_failed=retry_failed, max_retries=0,
                         pass_length_output=True)
        if max_requests is None:
            self.max_requests = float("inf")
        self.decision_context = None
        self.ordinal = 0
        self.replay_only = False

    def set_decision(self, game, player_id, game_index, step, *, replay_only=False):
        if type(step) is not int or not 0 <= step <= 200:
            raise ValueError("HM decision step outside offline guard")
        self.decision_context = dict(game=game, player_id=player_id, game_index=game_index)
        self.environment_step = step
        self.ordinal = 0
        self.replay_only = replay_only

    def generate(self, prompt):
        if self.decision_context is None or self.ordinal >= 999:
            raise ValueError("Missing or exhausted HM call context")
        self.ordinal += 1
        self.set_context(**self.decision_context, step=self.environment_step * 1000 + self.ordinal)
        if self.replay_only:
            with self._locked():
                row = self._read_records().get(self._path(self._context, "decision"))
                if (row is None or (row["status"] != "success" and not (
                        row["status"] == "failed" and row.get("finish_reason") == "length"))):
                    raise ResumeConflict("Completed game replay cannot issue new calls")
        return self.generate_stage("decision", prompt)

    def record_validation(self, prompt, raw, validation):
        with self._locked():
            path = self._path(self._context, "decision")
            row = self._read_records().get(path)
            if (row is None or (row["status"] != "success" and not (
                    row["status"] == "failed" and row.get("finish_reason") == "length"))
                    or row["response"] != self._redact(raw)
                    or row["prompt_hash"] != hashlib.sha256(prompt.encode("utf-8", errors="surrogatepass")).hexdigest()):
                raise ResumeConflict("HM validation must correspond to the exact journal call")
            target = self.run_dir / "logs/output_validation" / path.relative_to(self.run_dir / "prompts")
            self._safe_path(target)
            immutable_json(target, {"journal_path": str(path.relative_to(self.run_dir)),
                "context": row["context"], "attempt_number": row["attempt_number"], **validation})

    def record_hm_fallback(self, fallback):
        with self._locked():
            if self._context is None or type(fallback) is not dict:
                raise ValueError("HM fallback requires an active call context")
            path = self._path(self._context, "decision")
            row = self._read_records().get(path)
            if row is None or row["status"] not in {"success", "failed"}:
                raise ResumeConflict("HM fallback must follow completed output attempts")
            target = (self.run_dir / "logs/action_fallback" /
                      path.relative_to(self.run_dir / "prompts"))
            self._safe_path(target)
            immutable_json(target, {"schema_version": 1,
                "journal_path": str(path.relative_to(self.run_dir)),
                "context": row["context"], "attempt_number": row["attempt_number"],
                **fallback})