"""Global, fail-closed Agent-Pro call identities over ModelCallJournal.

One run directory belongs exclusively to one ledger. All policies, scenarios and
search branches share its immutable attempt budget. Semantic identities are JSON
values, with identifier-like strings (no paths, whitespace or credential fields).
The mapping is fsynced before the journal can invoke a backend. An interruption
between those two writes requires manual review, never an automatic retry.

Cooperating ledgers take a nonblocking POSIX lock on the logs directory throughout
mapping validation and the model call. Backend retries must be disabled by the
caller. No network configuration or credentials are read by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Iterator

from base.baseline_calls import (
    BudgetExceeded, CallFailed, ModelCallJournal, ResumeConflict,
)
from base.interfaces import BaseModel
from .learning_calls import LearningLedgerMixin
from .fallback_calls import FallbackLedgerMixin
from .fallback import KnownModelFailure, failure_reason
from base.model_output import ModelRequestError
from .reflection_retry import LEARNING_RETRY_STAGES, LEARNING_STAGES
from .retry import RETRY_PROTOCOL, RETRY_STAGES, attempt_record, correction_prompt, retry_count


STAGES = frozenset({"decision", "calibration", "proposal", "opponent", *RETRY_STAGES, *LEARNING_RETRY_STAGES})
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|"
    r"refresh[_-]?token|credential)", re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"\bsk-[A-Za-z0-9_-]{8,}\b|"
    r"\bBearer\s+[A-Za-z0-9._~+/-]+=*|"
    r'''(?i:(?:api[_-]?key|password|secret|access[_-]?token)["']?\s*[:=]\s*["']?)[^\s,"';}]+''',
    re.IGNORECASE,
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"))


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


class _StageProxy(BaseModel):
    """Translate the journal's decision namespace, not provider metadata.

    Metadata setters deliberately reach the backend: otherwise the journal could
    accidentally attribute stale usage or finish reasons to a subsequent call.
    """

    def __init__(self, backend: BaseModel) -> None:
        self.backend = backend
        self.stage = "decision"
        self.failure_policy = "strict"
        self.known_failure = None
        self.unexpected_failure = None

    @property
    def last_usage(self) -> Any:
        return getattr(self.backend, "last_usage", {})

    @last_usage.setter
    def last_usage(self, value: Any) -> None:
        self.backend.last_usage = value  # type: ignore[attr-defined]

    @property
    def last_finish_reason(self) -> Any:
        return getattr(self.backend, "last_finish_reason", None)

    @last_finish_reason.setter
    def last_finish_reason(self, value: Any) -> None:
        self.backend.last_finish_reason = value  # type: ignore[attr-defined]

    def redact(self, text: str) -> str:
        try:
            hook = getattr(self.backend, "redact", None)
            safe = hook(text) if callable(hook) else text
            if type(safe) is str:
                return _CREDENTIAL.sub("<redacted>", safe)
        except Exception:
            pass
        raise CallFailed("Model text could not be safely logged") from None

    def generate(self, prompt: str) -> str:
        method = getattr(self.backend, "generate_stage", None)
        try:
            raw = method(self.stage, prompt) if callable(method) else self.backend.generate(prompt)
        except Exception as exc:
            if self.failure_policy != "strict":
                reason = failure_reason(exc)
                if reason:
                    self.known_failure = (reason, getattr(exc, "response", None))
                    if reason == "transport_error" and not isinstance(exc, ModelRequestError):
                        raise ModelRequestError("transport_error") from None
                else:
                    self.unexpected_failure = exc
            raise
        if self.failure_policy != "strict" and (type(raw) is not str or not raw.strip()):
            self.known_failure = ("empty_response", raw if type(raw) is str else None)
        return raw

    def generate_stage(self, stage: str, prompt: str) -> str:
        if stage != "decision":
            raise ValueError("Unexpected internal journal stage")
        return self.generate(prompt)


class AgentProCallLedger(FallbackLedgerMixin, LearningLedgerMixin, BaseModel):
    """Bind a semantic scope, then issue decision/calibration/proposal calls.

    ``last_usage`` and ``last_finish_reason`` report journal semantics (in
    particular, replay has empty usage). Setters also reach the backend. Provider
    exceptions and explicit length completions are handled by the shared journal.
    """

    STAGES = STAGES

    def __init__(self, backend: BaseModel, run_dir: Path, max_requests: int | None) -> None:
        if max_requests is not None and (type(max_requests) is not int or max_requests < 0):
            raise ValueError("max_requests must be a nonnegative integer or None")
        self._max_requests = max_requests
        self._proxy = _StageProxy(backend)
        self._identity: dict[str, Any] | None = None
        self._mutex = threading.Lock()
        try:
            self.run_dir = Path(run_dir).resolve()
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._logs = self.run_dir / "logs"
            if self._logs.is_symlink():
                raise ValueError("Symlink")
            self._logs.mkdir(exist_ok=True)
        except (OSError, ValueError, RuntimeError):
            raise ResumeConflict("Cannot open Agent-Pro ledger directory") from None
        self._map_path = self._logs / "agent_pro_call_ids.json"
        self._protocol_path = self._logs / "agent_pro_decision_protocol.json"
        self._validation_dir = self._logs / "output_validation"
        with self._locked():
            # One semantic mapping entry represents exactly one request attempt.
            self._journal = ModelCallJournal(self._proxy, self.run_dir,
                                             0 if max_requests is None else max_requests,
                                             max_retries=0)
            if not self._map_path.exists() and not self._map_path.is_symlink():
                if self._journal.stats()["requests"]:
                    raise ResumeConflict("Journal exists without a semantic mapping")
                self._write({"schema_version": 1, "max_requests": max_requests, "calls": []})
            self._read_and_audit()

    @property
    def max_requests(self) -> int | None:
        return self._max_requests

    @property
    def last_usage(self) -> dict[str, int]:
        return self._journal.last_usage

    @last_usage.setter
    def last_usage(self, value: dict[str, int]) -> None:
        self._proxy.last_usage = value
        self._journal.last_usage = value

    @property
    def last_finish_reason(self) -> str | None:
        return self._journal.last_finish_reason

    @last_finish_reason.setter
    def last_finish_reason(self, value: str | None) -> None:
        self._proxy.last_finish_reason = value
        self._journal.last_finish_reason = value

    def redact(self, text: str) -> str:
        return self._proxy.redact(text)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if not self._mutex.acquire(blocking=False):
            raise ResumeConflict("Agent-Pro ledger is busy")
        fd = None
        try:
            try:
                fd = os.open(self._logs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ResumeConflict("Agent-Pro ledger is busy or cannot be locked") from None
            yield
        finally:
            if fd is not None:
                os.close(fd)
            self._mutex.release()

    def _scope(self, identity: Any) -> dict[str, Any]:
        def check(value: Any, depth: int = 0) -> None:
            if depth > 16:
                raise ValueError("Semantic identity is too deeply nested")
            if type(value) is dict:
                for key, item in value.items():
                    if (type(key) is not str
                            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", key)
                            or _SECRET_KEY.search(key) or self.redact(key) != key):
                        raise ValueError("Unsafe semantic identity field")
                    check(item, depth + 1)
            elif type(value) is list:
                for item in value:
                    check(item, depth + 1)
            elif type(value) is str:
                if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}", value)
                        or ".." in value or self.redact(value) != value):
                    raise ValueError("Unsafe semantic identity string")
            elif value is None or type(value) in (bool, int):
                pass
            elif type(value) is float and math.isfinite(value):
                pass
            else:
                raise ValueError("Semantic identity must contain finite JSON values")

        if type(identity) is not dict or not identity:
            raise ValueError("Semantic identity must be a nonempty JSON object")
        check(identity)
        serialized = _json(identity)
        if len(serialized) > 16384:
            raise ValueError("Semantic identity is too large")
        return json.loads(serialized)

    def bind(self, identity: dict[str, Any]) -> None:
        """Replace scope with a defensive copy; invalid input unbinds old scope."""
        with self._locked():
            self._identity = None
            self._identity = self._scope(identity)

    @staticmethod
    def _context(index: int, identity: dict[str, Any], stage: str) -> dict[str, Any]:
        # Bind the semantic scope to the journal as well, so a valid-looking edit
        # to an identity/stage in the mapping cannot silently relabel a response.
        digest = hashlib.sha256(_json([identity, stage]).encode("ascii")).hexdigest()
        return {"game": "ap_" + digest[:60], "player_id": 0, "game_index": index, "step": 0}

    def _record_path(self, index: int) -> Path:
        return self.run_dir / "prompts" / "P0" / f"g{index:04d}_s0000_decision.json"

    def validation_path(self, index: int) -> Path:
        return self._validation_dir / f"g{index:04d}_decision.json"

    @staticmethod
    def _protocol(parse_retries):
        return {"schema_version": 1, "protocol": RETRY_PROTOCOL,
                "parse_retries": retry_count(parse_retries), "sources": {
                    name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
                    for name in ("method", "calls", "retry", "policy", "scripted")}}

    def configure_decision_retries(self, parse_retries):
        """Pin the protocol before the first call, also for standalone methods."""
        with self._locked():
            self._read_and_audit()
            expected = self._protocol(parse_retries)
            if self._protocol_path.exists():
                saved = json.loads(self._protocol_path.read_text(), object_pairs_hook=_unique)
                if saved != expected:
                    raise ResumeConflict("Agent-Pro decision retry configuration/source changed")
            else:
                self._write(expected, self._protocol_path)

    def _audit_validations(self, calls):
        if self._protocol_path.is_symlink() or self._validation_dir.is_symlink():
            raise ValueError("Symlink validation path")
        protocol = None
        if self._protocol_path.exists():
            protocol = json.loads(self._protocol_path.read_text(), object_pairs_hook=_unique)
            if _json(protocol) != _json(self._protocol(protocol["parse_retries"])):
                raise ValueError("Decision protocol changed")
        entries = {entry["game_index"]: entry for entry in calls}
        validations = {}
        for path in self._validation_dir.glob("*"):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise ValueError("Unexpected validation artifact")
            value = json.loads(path.read_text(), object_pairs_hook=_unique)
            index = value["game_index"]
            entry = entries[index]
            record = json.loads(self._record_path(index).read_text(), object_pairs_hook=_unique)
            expected = self._validation_payload(entry, record, protocol["parse_retries"])
            if path != self.validation_path(index) or _json(value) != _json(expected):
                raise ValueError("Decision validation changed")
            validations[index] = value["attempt"]
        return validations

    @staticmethod
    def _validation_payload(entry, record, parse_retries):
        if record["status"] != "success" or entry["stage"] not in {"decision", *RETRY_STAGES}:
            raise ValueError("Only successful decision text has output validation")
        attempt = attempt_record(entry["stage"], record["prompt"], record["response"],
                                 parse_retries, record["finish_reason"])
        return {"schema_version": 1, "game_index": entry["game_index"], "identity": entry["identity"],
                "prompt_hash": entry["prompt_hash"], "attempt": attempt,
                "journal_sha256": hashlib.sha256(_json(record).encode()).hexdigest()}

    def record_output_validation(self, attempt):
        """Output validity is separate from HTTP success; never rewrite the journal."""
        with self._locked():
            data, _ = self._read_and_audit()
            # Validate against the pinned run protocol BEFORE writing evidence.
            # Trusting the caller's count here can persist a validation that the
            # very next audit rejects, making a recoverable HTTP success unusable.
            try:
                protocol = json.loads(self._protocol_path.read_text(), object_pairs_hook=_unique)
                parse_retries = retry_count(attempt["parse_retries"])
                if parse_retries != protocol["parse_retries"]:
                    raise ValueError("Decision retry count mismatch")
            except (OSError, ValueError, KeyError, TypeError):
                raise ResumeConflict("Decision validation requires the pinned retry protocol") from None
            entry = next((e for e in data["calls"] if e["identity"] == self._identity
                          and e["stage"] == attempt["stage"]), None)
            if entry is None:
                raise ResumeConflict("Missing decision validation identity")
            record = json.loads(self._record_path(entry["game_index"]).read_text(), object_pairs_hook=_unique)
            payload = self._validation_payload(entry, record, protocol["parse_retries"])
            if _json(payload["attempt"]) != _json(attempt):
                raise ResumeConflict("Decision validation disagrees with journal")
            path = self.validation_path(entry["game_index"])
            if path.exists():
                if _json(json.loads(path.read_text(), object_pairs_hook=_unique)) != _json(payload):
                    raise ResumeConflict("Decision validation changed")
            else:
                self._write(payload, path)

    def _check_retry_predecessor(self, calls, stage, prompt, identity=None):
        index = RETRY_STAGES.index(stage) + 1
        identity = self._identity if identity is None else identity
        scope = [e for e in calls if e["identity"] == identity]
        stages = {e["stage"]: e for e in scope}
        prior_stage = "decision" if index == 1 else RETRY_STAGES[index - 2]
        try:
            first = stages["decision"]
            prior = stages[prior_stage]
            validation = json.loads(self.validation_path(prior["game_index"]).read_text(), object_pairs_hook=_unique)["attempt"]
            original = json.loads(self._record_path(first["game_index"]).read_text(), object_pairs_hook=_unique)["prompt"]
            if (validation["validation"] != "invalid" or validation["finish_reason"] != "stop"
                    or index > validation["parse_retries"]
                    or prompt != correction_prompt(original, validation["response"], validation["error_category"], index)):
                raise ValueError("Invalid retry predecessor")
        except (OSError, ValueError, KeyError, TypeError):
            raise ResumeConflict("Decision retry requires the same validated invalid predecessor") from None

    def _read_and_audit(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            if self._map_path.is_symlink():
                raise ValueError("Symlink")
            data = json.loads(self._map_path.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique)
            if (type(data) is not dict
                    or set(data) != {"schema_version", "max_requests", "calls"}
                    or type(data["schema_version"]) is not int or data["schema_version"] != 1
                    or (data["max_requests"] is not None and type(data["max_requests"]) is not int)
                    or data["max_requests"] != self.max_requests
                    or type(data["calls"]) is not list):
                raise ValueError("Invalid mapping or changed budget")
            stats = self._journal.stats()  # Validate journal schemas and symlinks first.
            if (stats["requests"] != len(data["calls"])
                    or self.max_requests is not None and stats["requests"] > self.max_requests):
                raise ValueError("Unmatched semantic mapping or journal attempts")
            seen = set()
            for index, entry in enumerate(data["calls"], 1):
                if (type(entry) is not dict
                        or set(entry) != {"game_index", "identity", "stage", "journal_context", "prompt_hash"}
                        or type(entry["game_index"]) is not int
                        or entry["game_index"] != index or index > 2**31 - 1
                        or type(entry["stage"]) is not str or entry["stage"] not in STAGES
                        or _json(entry["journal_context"]) != _json(
                            self._context(index, entry["identity"], entry["stage"]))
                        or type(entry["prompt_hash"]) is not str
                        or not re.fullmatch(r"[0-9a-f]{64}", entry["prompt_hash"])):
                    raise ValueError("Invalid semantic call")
                key = (_json(self._scope(entry["identity"])), entry["stage"])
                if key in seen:
                    raise ValueError("Duplicate semantic call")
                seen.add(key)
                record = json.loads(self._record_path(index).read_text(encoding="utf-8"),
                                    object_pairs_hook=_unique)
                if (record["prompt_hash"] != entry["prompt_hash"]
                        or _json(record["context"]) != _json(entry["journal_context"])
                        or record["stage"] != "decision" or record["attempts"]
                        or record["attempt_number"] != index):
                    raise ValueError("Semantic identity and journal mismatch")
            self._audit_validations(data["calls"])
            self._audit_learning(data["calls"])
            self._audit_fallbacks(data["calls"])
            # Audit every saved correction, even before bind() and when the
            # final episode cache does not exist. Missing predecessor validation
            # must never silently authorize a new request after resumption.
            for entry in data["calls"]:
                if entry["stage"] in RETRY_STAGES:
                    record = json.loads(self._record_path(entry["game_index"]).read_text(),
                                        object_pairs_hook=_unique)
                    self._check_retry_predecessor(data["calls"], entry["stage"],
                                                  record["prompt"], entry["identity"])
            return data, stats
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, CallFailed):
            raise ResumeConflict("Invalid Agent-Pro mapping, budget or journal; manual review required") from None

    def _write(self, data: dict[str, Any], path: Path | None = None) -> None:
        path = self._map_path if path is None else path
        temporary = None
        try:
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("Symlink")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".agent-pro-", suffix=".tmp", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(_json(data))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, ValueError, TypeError):
            raise ResumeConflict("Cannot persist Agent-Pro call identity") from None
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def generate(self, prompt: str) -> str:
        return self.generate_stage("decision", prompt)

    def set_opponent_fixture(self, game, observation, legal):
        hook = getattr(self._proxy.backend, "set_opponent_fixture", None)
        if callable(hook):
            hook(game, observation, legal)

    def generate_stage(self, stage: str, prompt: str) -> str:
        with self._locked():
            self._journal.last_usage = {}
            self._journal.last_finish_reason = None
            self._proxy.stage = "decision"
            try:
                if type(stage) is not str or stage not in STAGES:
                    raise ValueError("Unknown Agent-Pro stage")
                if type(prompt) is not str:
                    raise ValueError("prompt must be text")
                if self._identity is None:
                    raise ValueError("Bind a semantic identity before generating")
                data, stats = self._read_and_audit()
                if stage in RETRY_STAGES:
                    self._check_retry_predecessor(data["calls"], stage, prompt)
                if stage in (*LEARNING_STAGES, *LEARNING_RETRY_STAGES):
                    self._check_learning_call(data["calls"], stage, prompt, self._identity)
                scope = _json(self._identity)
                prompt_hash = hashlib.sha256(prompt.encode("utf-8", errors="surrogatepass")).hexdigest()
                entry = next((item for item in data["calls"]
                              if _json(item["identity"]) == scope and item["stage"] == stage), None)
                if entry is not None:
                    if entry["prompt_hash"] != prompt_hash:
                        raise ResumeConflict("Semantic scope/stage was used with a different prompt")
                    receipt = self._receipt_path(entry["game_index"])
                    if self.failure_policy != "strict" and receipt.exists():
                        # Only audited locally-classified receipts may replay a
                        # failure. No physical request or usage is added.
                        raise KnownModelFailure(json.loads(receipt.read_text())["attempt"])
                else:
                    if self.max_requests is not None and stats["requests"] >= self.max_requests:
                        raise BudgetExceeded("Global Agent-Pro request budget exhausted")
                    index = len(data["calls"]) + 1
                    if index > 2**31 - 1:
                        raise ResumeConflict("Agent-Pro call identity space exhausted")
                    entry = {"game_index": index, "identity": self._identity,
                             "stage": stage,
                             "journal_context": self._context(index, self._identity, stage),
                             "prompt_hash": prompt_hash}
                    data["calls"].append(entry)
                    self._write(data)  # Durable semantic intent, before journal/backend.
                self._proxy.stage = stage
                self._journal.set_context(**entry["journal_context"])
                physical_context = getattr(self._proxy.backend, "bind_physical_context", None)
                if callable(physical_context):
                    physical_context(entry)
                if self.max_requests is None:
                    # No lifetime request cap. Under the exclusive Agent-Pro lock,
                    # grant exactly the next attempt to the unchanged shared
                    # journal. This is a one-call reservation, not a large fake
                    # total budget; it grows from actual audited usage. Cached
                    # identities still replay, errors/started are never retried.
                    self._journal.max_requests = stats["requests"] + 1
                self._proxy.known_failure = self._proxy.unexpected_failure = None
                try:
                    return self._journal.generate_stage("decision", prompt)
                except CallFailed:
                    if self.failure_policy != "strict":
                        if self._proxy.unexpected_failure is not None:
                            raise self._proxy.unexpected_failure
                        record = json.loads(self._record_path(entry["game_index"]).read_text())
                        known = self._proxy.known_failure
                        if record["finish_reason"] == "length" and record["status"] == "failed":
                            known = ("length", record["response"])
                        if known is not None:
                            reason, raw = known
                            raw = self.redact(raw) if type(raw) is str else None
                            raise self._save_known_failure(entry, reason, raw) from None
                    raise
            finally:
                self._proxy.stage = "decision"

    def stats(self) -> dict[str, Any]:
        with self._locked():
            return self._read_and_audit()[1]