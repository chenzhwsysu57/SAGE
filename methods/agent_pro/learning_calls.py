"""Agent-Pro-only learning audit extension to its globally locked call ledger.

No shared journal changes. Pins are durable before the first calibration call;
missing HTTP validation may be reconstructed by replay, never by a new request.
"""

import hashlib
from pathlib import Path

from base.baseline_calls import ResumeConflict
from .policy import canonical, digest, strict_json
from .reflection_retry import (LEARNING_PROTOCOL, LEARNING_RETRY_STAGES, LEARNING_STAGES,
                               attempt_record, correction_prompt, parse_reflection,
                               read_context, stage_parts)
from .retry import retry_count


class LearningLedgerMixin:
    @property
    def _learning_protocol_path(self):
        return self._logs / "agent_pro_reflection_protocol.json"

    @property
    def _learning_context_dir(self):
        return self._logs / "reflection_contexts"

    def reflection_validation_path(self, index):
        return self._logs / "reflection_output_validation" / f"g{index:04d}.json"

    @staticmethod
    def _learning_protocol(count):
        return {"schema_version": 1, "protocol": LEARNING_PROTOCOL,
                "parse_retries": retry_count(count), "sources": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob("*.py"))}}

    @staticmethod
    def _learning_read(path):
        if path.is_symlink():
            raise ValueError("Symlink learning artifact")
        return strict_json(path.read_text(encoding="utf-8"))

    def configure_reflection_retries(self, count):
        with self._locked():
            self._read_and_audit()
            expected = self._learning_protocol(count)
            path = self._learning_protocol_path
            if path.exists():
                if canonical(self._learning_read(path)) != canonical(expected):
                    raise ResumeConflict("Reflection retry configuration/source changed")
            else:
                self._write(expected, path)

    def bind_reflection_context(self, original, previous):
        """Pin all inputs, including rejected candidates, under the SAME identity."""
        return self._bind_reflection_context(original, previous)

    def bind_reusable_reflection_context(self, original, previous, calibration_key):
        """Pin a candidate to the first exact calibration owner, before any call.

        Failed or incomplete owners remain owners: a new candidate cannot reset
        retries or reissue an ambiguous physical request after a restart.
        """
        return self._bind_reflection_context(original, previous, calibration_key)

    def _bind_reflection_context(self, original, previous, calibration_key=None):
        with self._locked():
            self._read_and_audit()
            if self._identity is None or not self._learning_protocol_path.exists():
                raise ResumeConflict("Reflection needs a configured bound identity")
            read_context(original, "calibration")
            expected = {"schema_version": 1, "identity": self._identity,
                        "calibration_prompt_hash": digest(original),
                        "previous_hash": digest(previous),
                        "context_hash": digest([original, previous]),
                        "protocol_hash": digest(self._learning_read(self._learning_protocol_path))}
            if calibration_key is not None:
                if (type(calibration_key) is not str or len(calibration_key) != 64
                        or any(c not in "0123456789abcdef" for c in calibration_key)):
                    raise ResumeConflict("Invalid calibration reuse key")
                owners = [self._learning_read(p) for p in self._learning_context_dir.glob("*.json")]
                owners = [p for p in owners if p.get("calibration_key") == calibration_key]
                owner = owners[0]["calibration_identity"] if owners else self._identity
                if any(p["calibration_prompt_hash"] != digest(original) for p in owners):
                    raise ResumeConflict("Calibration reuse context changed")
                expected.update(calibration_key=calibration_key, calibration_identity=owner)
            path = self._learning_context_dir / (digest(self._identity) + ".json")
            if path.exists():
                if canonical(self._learning_read(path)) != canonical(expected):
                    raise ResumeConflict("Reflection original context changed")
            else:
                self._write(expected, path)
            return expected.get("calibration_identity", self._identity)

    def _learning_payload(self, entry, record, protocol, pin):
        if record["status"] != "success":
            raise ValueError("Only successful learning text has validation")
        return {"schema_version": 1, "game_index": entry["game_index"],
                "identity": entry["identity"], "prompt_hash": entry["prompt_hash"],
                "context_hash": pin["context_hash"], "protocol_hash": digest(protocol),
                "journal_sha256": digest(record),
                "attempt": attempt_record(entry["stage"], record["prompt"], record["response"],
                                          protocol["parse_retries"], record["finish_reason"])}

    def record_reflection_validation(self, attempt):
        with self._locked():
            data, _ = self._read_and_audit()
            try:
                protocol = self._learning_read(self._learning_protocol_path)
                pin = self._learning_read(self._learning_context_dir / (digest(self._identity) + ".json"))
                entry = next(e for e in data["calls"] if e["identity"] == self._identity
                             and e["stage"] == attempt["stage"])
                record = self._learning_read(self._record_path(entry["game_index"]))
                payload = self._learning_payload(entry, record, protocol, pin)
                if canonical(payload["attempt"]) != canonical(attempt):
                    raise ValueError("Learning validation mismatch")
                path = self.reflection_validation_path(entry["game_index"])
                if path.exists():
                    if canonical(self._learning_read(path)) != canonical(payload):
                        raise ValueError("Learning validation changed")
                else:
                    self._write(payload, path)
            except (OSError, ValueError, TypeError, KeyError, StopIteration):
                raise ResumeConflict("Learning validation requires the pinned protocol and context") from None

    def _check_learning_call(self, calls, stage, prompt, identity, before=None):
        """Check exact predecessor/order/context before allocation AND on restart."""
        scope = [e for e in calls if e["identity"] == identity
                 and (before is None or e["game_index"] < before)]
        pin_path = self._learning_context_dir / (digest(identity) + ".json")
        # Raw ledger clients (not reflectors) retain their existing stage API.
        if not pin_path.exists() and stage in LEARNING_STAGES:
            if self._learning_protocol_path.exists():
                raise ResumeConflict("Learning call missing pinned context")
            return
        try:
            pin = self._learning_read(pin_path)
            context, original = read_context(prompt, stage)
            base, index = stage_parts(stage)
            stages = {e["stage"]: e for e in scope}
            if base == "calibration":
                if (digest(original) != pin["calibration_prompt_hash"]
                        or pin.get("calibration_identity", identity) != identity):
                    raise ValueError("Changed calibration context")
            else:
                owner = pin.get("calibration_identity", identity)
                cal_scope = [e for e in calls if e["identity"] == owner
                             and (before is None or e["game_index"] < before)]
                first = next(e for e in cal_scope if e["stage"] == "calibration")
                cal_record = self._learning_read(self._record_path(first["game_index"]))
                cal_context, cal_original = read_context(cal_record["prompt"], "calibration")
                # Proposal must consume exactly one successful, validated final calibration.
                cal_entries = [e for e in cal_scope if e["stage"] in (*LEARNING_STAGES, *LEARNING_RETRY_STAGES)
                               and stage_parts(e["stage"])[0] == "calibration"]
                last = cal_entries[-1]
                validation = self._learning_read(self.reflection_validation_path(last["game_index"]))["attempt"]
                if (validation["validation"] != "valid"
                    or owner != identity and validation["finish_reason"] != "stop"):
                    raise ValueError("Proposal without valid calibration")
                reflection = parse_reflection(validation["response"], "calibration", context["game"])
                keys = ("game", "policy", "records", "feedback")
                if (any(context[k] != cal_context[k] for k in keys)
                        or context["reflection"] != reflection
                        or digest(context["previous_candidates"]) != pin["previous_hash"]
                        or digest([cal_original, context["previous_candidates"]]) != pin["context_hash"]):
                    raise ValueError("Proposal context/calibration changed")
            if index:
                prior_stage = base if index == 1 else f"{base}_retry_{index - 1}"
                prior = stages[prior_stage]
                validation = self._learning_read(self.reflection_validation_path(prior["game_index"]))["attempt"]
                first_record = self._learning_read(self._record_path(stages[base]["game_index"]))
                if (validation["validation"] != "invalid" or validation["finish_reason"] != "stop"
                        or index > validation["parse_retries"]
                        or prompt != correction_prompt(first_record["prompt"], validation["response"],
                                                       validation["error_category"], index, base)):
                    raise ValueError("Invalid learning retry predecessor")
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
            raise ResumeConflict("Learning call requires the same pinned context and validated predecessor") from None

    def _audit_learning(self, calls):
        protocol_path = self._learning_protocol_path
        validation_dir = self._logs / "reflection_output_validation"
        for path in (protocol_path, self._learning_context_dir, validation_dir):
            if path.is_symlink():
                raise ValueError("Symlink learning artifact")
        pins = {}
        protocol = self._learning_read(protocol_path) if protocol_path.exists() else None
        if protocol is not None and canonical(protocol) != canonical(self._learning_protocol(protocol["parse_retries"])):
            raise ValueError("Learning protocol/source changed")
        for path in self._learning_context_dir.glob("*"):
            pin = self._learning_read(path)
            fields = {"schema_version", "identity", "calibration_prompt_hash", "previous_hash", "context_hash", "protocol_hash"}
            if (set(pin) not in (fields, fields | {"calibration_key", "calibration_identity"})
                    or type(pin["schema_version"]) is not int or pin["schema_version"] != 1
                    or path.name != digest(self._scope(pin["identity"])) + ".json"
                    or protocol is None or pin["protocol_hash"] != digest(protocol)):
                raise ValueError("Changed learning context pin")
            pins[digest(pin["identity"])] = pin
        owners = {}
        for pin in pins.values():
            if "calibration_key" not in pin:
                continue
            key = pin["calibration_key"]
            owner_id = self._scope(pin["calibration_identity"])
            owner = pins[digest(owner_id)]
            if (owner.get("calibration_identity") != owner_id
                    or owner.get("calibration_key") != key
                    or owner["calibration_prompt_hash"] != pin["calibration_prompt_hash"]
                    or owner["identity"].get("policy_id") != pin["identity"].get("policy_id")
                    or owner["identity"].get("source_id") != pin["identity"].get("source_id")
                    or key in owners and owners[key] != owner_id):
                raise ValueError("Invalid calibration owner reference")
            owners[key] = owner_id
        entries = {e["game_index"]: e for e in calls}
        for path in validation_dir.glob("*"):
            value = self._learning_read(path)
            entry = entries[value["game_index"]]
            record = self._learning_read(self._record_path(entry["game_index"]))
            expected = self._learning_payload(entry, record, protocol, pins[digest(entry["identity"])])
            if path != self.reflection_validation_path(entry["game_index"]) or canonical(value) != canonical(expected):
                raise ValueError("Learning validation changed")
        for entry in calls:
            if entry["stage"] in (*LEARNING_STAGES, *LEARNING_RETRY_STAGES):
                record = self._learning_read(self._record_path(entry["game_index"]))
                self._check_learning_call(calls, entry["stage"], record["prompt"], entry["identity"], entry["game_index"])