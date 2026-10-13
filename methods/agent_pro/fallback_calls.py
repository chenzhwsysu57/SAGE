"""Durable Agent-Pro failure receipts and random-action certificates.

The shared journal remains unchanged and fail-closed. Only locally classified
known failures receive a receipt AFTER the journal is durable. A crash before
that receipt leaves the failed/started request blocked, never reissued.
"""

import hashlib
from pathlib import Path

from base.baseline_calls import ResumeConflict
from base.model_output import ERROR_CATEGORIES
from .fallback import (PROTOCOL, KnownModelFailure, check_policy, failed_attempt,
                       fallback_payload)
from .policy import canonical, digest, strict_json
from .retry import RETRY_STAGES, attempt_record, correction_prompt


class FallbackLedgerMixin:
    failure_policy = "strict"

    def _failure_protocol(self, policy):
        return {"protocol": PROTOCOL, "policy": check_policy(policy), "sources": {
            name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
            for name in ("fallback", "fallback_calls", "calls", "method", "retry")}}

    def configure_action_failure(self, policy):
        with self._locked():
            self._read_and_audit()
            expected = self._failure_protocol(policy)
            path = self._logs / "agent_pro_failure_protocol.json"
            if path.exists():
                if strict_json(path.read_text()) != expected:
                    raise ResumeConflict("Action failure protocol/configuration changed")
            elif policy != "strict":
                self._write(expected, path)
            self.failure_policy = policy
            self._proxy.failure_policy = policy

    def _receipt_path(self, index):
        return self._logs / "model_failures" / f"g{index:04d}.json"

    def _receipt(self, entry, record, reason, response):
        return {"protocol": PROTOCOL, "entry": entry,
                "journal_sha256": digest(record),
                "attempt": failed_attempt(entry["stage"], record["prompt"], response,
                    reason, record["finish_reason"], record["usage"])}

    def _save_known_failure(self, entry, reason, response):
        record = strict_json(self._record_path(entry["game_index"]).read_text())
        if record["status"] != "failed":
            raise ResumeConflict("Known failure requires a completed failed journal")
        if record["finish_reason"] == "length":
            reason, response = "length", record["response"]
        elif reason != "empty_response":
            response = None  # Arbitrary transport exception bodies are not model output.
        value = self._receipt(entry, record, reason, response)
        self._write(value, self._receipt_path(entry["game_index"]))
        return KnownModelFailure(value["attempt"])

    def _certificate_path(self, identity):
        return self._logs / "action_fallback" / (digest(identity) + ".json")

    def _fallback_dependencies(self, calls, identity, payload, receipts):
        scope = [e for e in calls if e["identity"] == identity]
        attempts = payload["decision_attempts"]
        if len(scope) != len(attempts) or not attempts:
            raise ValueError("Fallback scope/evidence mismatch")
        dependencies = []
        original = None
        for i, (entry, attempt) in enumerate(zip(scope, attempts)):
            stage = "decision" if i == 0 else RETRY_STAGES[i - 1]
            if entry["stage"] != stage or attempt["stage"] != stage:
                raise ValueError("Fallback attempt order changed")
            record = strict_json(self._record_path(entry["game_index"]).read_text())
            if i == 0:
                original = record["prompt"]
            else:
                previous = attempts[i - 1]
                if (previous["validation"] != "invalid" or previous["finish_reason"] != "stop"
                        or attempt["prompt"] != correction_prompt(original, previous["response"], previous["error_category"], i)):
                    raise ValueError("Fallback correction predecessor changed")
            if record["status"] == "success":
                expected = attempt_record(stage, record["prompt"], record["response"],
                    payload["decision_parse_retries"], record["finish_reason"])
                validation = strict_json(self.validation_path(entry["game_index"]).read_text())
                if validation["attempt"] != expected or expected["validation"] != "invalid":
                    raise ValueError("Fallback cannot replace a valid model decision")
            elif entry["game_index"] in receipts and i == len(attempts) - 1:
                expected = receipts[entry["game_index"]]["attempt"]
            else:
                raise ValueError("Fallback requires a verified known failure")
            if canonical(attempt) != canonical(expected):
                raise ValueError("Fallback evidence changed")
            dependencies.append({"entry": entry, "journal_sha256": digest(record)})
        last = attempts[-1]
        reason = (last["error_category"] if last["validation"] == "failed" else
                  "parse_exhausted" if len(attempts) == payload["decision_parse_retries"] + 1 else
                  "unconfirmed_output")
        if last["validation"] != "failed" and reason == "unconfirmed_output" and last["finish_reason"] == "stop":
            raise ValueError("Corrections not exhausted")
        context = payload["fallback_context"]
        from .scripted import read_prompt_context
        visible = read_prompt_context(original)
        if (context["original_prompt_sha256"] != hashlib.sha256(original.encode("utf-8", errors="surrogatepass")).hexdigest()
                or context["game"] != visible["game"] or context["player_id"] != visible["observation"]["player_id"]
                or context["legal_actions"] != sorted(visible["legal_actions"], key=canonical)
                or context["parse_retries"] != payload["decision_parse_retries"]
                or context["protocol"] != PROTOCOL):
            raise ValueError("Fallback visible context changed")
        if canonical(payload) != canonical(fallback_payload(context, attempts, reason, payload["policy_id"])):
            raise ValueError("Random action/provenance changed")
        return dependencies

    def _audit_fallbacks(self, calls):
        protocol = self._logs / "agent_pro_failure_protocol.json"
        directories = [self._logs / name for name in ("model_failures", "action_fallback")]
        if protocol.is_symlink() or any(p.is_symlink() for p in directories):
            raise ValueError("Symlink failure artifact")
        if protocol.exists():
            value = strict_json(protocol.read_text())
            if value != self._failure_protocol(value["policy"]):
                raise ValueError("Failure protocol changed")
        elif any(p.exists() for p in directories):
            raise ValueError("Missing failure protocol")
        entries = {e["game_index"]: e for e in calls}
        receipts = {}
        for path in directories[0].glob("*"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Invalid failure receipt path")
            value = strict_json(path.read_text())
            entry = entries[value["entry"]["game_index"]]
            record = strict_json(self._record_path(entry["game_index"]).read_text())
            a = value["attempt"]
            if (record["status"] != "failed" or a["error_category"] not in {*ERROR_CATEGORIES, "length", "empty_response"}
                    or (a["error_category"] == "length" and (record["finish_reason"] != "length" or a["response"] != record["response"]))
                    or (a["error_category"] in ERROR_CATEGORIES and
                        (a["error_category"] != record["error_category"] or a["response"] is not None))
                    or (a["error_category"] == "empty_response" and (record["response"] is not None or a["response"] not in (None, "") and a["response"].strip()))
                    or value != self._receipt(entry, record, a["error_category"], a["response"])
                    or path != self._receipt_path(entry["game_index"])):
                raise ValueError("Failure receipt/journal mismatch")
            receipts[entry["game_index"]] = value
        for path in directories[1].glob("*"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Invalid fallback path")
            value = strict_json(path.read_text())
            expected = {"protocol": PROTOCOL, "identity": value["identity"], "payload": value["payload"],
                        "dependencies": self._fallback_dependencies(calls, value["identity"], value["payload"], receipts)}
            if canonical(value) != canonical(expected) or path != self._certificate_path(value["identity"]):
                raise ValueError("Fallback certificate changed")
        return receipts

    def restore_action_fallback(self, context, policy_id):
        with self._locked():
            self._read_and_audit()
            path = self._certificate_path(self._identity)
            if not path.exists():
                return None
            value = strict_json(path.read_text())["payload"]
            if value["fallback_context"] != context or value["policy_id"] != policy_id:
                raise ResumeConflict("Fallback context/policy changed")
            self._journal.last_usage = {}
            return value

    def record_action_fallback(self, payload):
        with self._locked():
            calls = self._read_and_audit()[0]["calls"]
            receipts = self._audit_fallbacks(calls)
            try:
                dependencies = self._fallback_dependencies(calls, self._identity, payload, receipts)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                raise ResumeConflict("Cannot verify random fallback evidence") from None
            value = {"protocol": PROTOCOL, "identity": self._identity,
                     "payload": payload, "dependencies": dependencies}
            path = self._certificate_path(self._identity)
            if path.exists():
                if strict_json(path.read_text()) != value:
                    raise ResumeConflict("Fallback already exists with different evidence")
            else:
                self._write(value, path)