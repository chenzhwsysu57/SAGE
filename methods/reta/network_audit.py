"""ReTA-only HTTP application-layer audit. No packet capture or extra requests.

Subclass the campus wrapper without changing its payload, transport or retry
behavior. HTTPX hooks capture the prepared request and full decoded response,
including non-2xx bodies, before the wrapper raises. Credentials/cookies are
redacted. No length clipping, no raw binary/auth headers or unsafe tracebacks.
The shared journal remains the ONLY retry/budget owner.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote

from apis.hkustgz_wrapper import HKUSTGZModel
from base.baseline_calls import ResumeConflict
from base.model_output import ModelRequestError, OutputTruncated

SENSITIVE = re.compile(r"authorization|cookie|api.?key|secret|password|credential|access.?token|refresh.?token", re.I)
SAFE_HEADERS = {"host", "content-type", "content-length", "content-encoding", "accept", "accept-encoding",
                "connection", "user-agent", "date", "server", "retry-after", "request-id", "x-request-id",
                "x-correlation-id", "traceparent", "cf-ray", "transfer-encoding"}


def atomic_audit(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".audit-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


class AuditedCampusModel(HKUSTGZModel):
    def __init__(self, run_dir, **kwargs):
        super().__init__(**kwargs)
        self.run_dir = Path(run_dir).resolve()
        self.proxy = kwargs.get("proxy", "socks5h://127.0.0.1:1080")
        self._audit_context = None
        self._audit = None
        self._audit_path = None
        self._sensitive_values = set()
        self.last_network_record = None
        self._client.event_hooks["request"].append(self._request_hook)
        self._client.event_hooks["response"].append(self._response_hook)

    def redact(self, text):
        text = super().redact(text)
        # Also remove escaped/URL-encoded credentials and header values echoed by
        # the server. Never persist a copied Set-Cookie in body or trace metadata.
        values = {self._key, *getattr(self, "_sensitive_values", set())}
        for value in sorted(values, key=len, reverse=True):
            if value:
                for variant in (value, json.dumps(value)[1:-1], quote(value, safe="")):
                    text = text.replace(variant, "<redacted>")
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "<redacted>", text)

    def _clean(self, value):
        if isinstance(value, dict):
            return {self.redact(str(k)): "<redacted>" if SENSITIVE.search(str(k)) else self._clean(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self._clean(v) for v in value]
        return self.redact(value) if isinstance(value, str) else value

    def _headers(self, headers):
        # Preserve header names/duplicates, hide unknown values conservatively.
        for name, value in headers.multi_items():
            if SENSITIVE.search(name):
                self._sensitive_values.add(value)
                if "cookie" in name.lower():
                    for part in value.split(";"):
                        if "=" in part:
                            self._sensitive_values.add(part.split("=", 1)[1].strip())
        return [[name, self.redact(value) if name.lower() in SAFE_HEADERS
                 or name.lower().startswith("x-ratelimit-") else "<redacted>"]
                for name, value in headers.multi_items()]

    def _body(self, raw, encoding="utf-8"):
        text = raw.decode(encoding or "utf-8", errors="replace")
        try:
            value = self._clean(json.loads(text))
            text = json.dumps(value, ensure_ascii=False)
        except (ValueError, TypeError):
            value = None
            text = self.redact(text)
            # Plain error pages occasionally echo named credentials.
            text = re.sub(r"(?i)((?:authorization|api[_-]?key|password|secret|cookie)\s*[:=]\s*)[^\r\n<]+",
                          r"\1<redacted>", text)
        return {"body_text": text, "body_json": value, "body_bytes_received": len(raw),
                "body_clipped": False, "representation": "decoded and redacted; not byte-identical wire data",
                "redacted_body_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def set_request_context(self, *, context, attempt_number, attempt_in_call, journal_path, actor_context=None):
        self._audit_context = {"context": deepcopy(context), "attempt_number": attempt_number,
                               "attempt_in_call": attempt_in_call, "journal_path": journal_path,
                               "actor_context": deepcopy(actor_context)}

    def _save(self):
        atomic_audit(self._audit_path, self._audit)

    def _request_hook(self, request):
        if self._audit is None or self._audit["request"] is not None:
            raise ResumeConflict("HTTP request outside an accounted audit attempt")
        headers = self._headers(request.headers)
        self._audit["request"] = {"method": request.method, "url": self.redact(str(request.url)),
            "headers": headers, **self._body(request.content),
            "timeouts": dict(request.extensions.get("timeout", {})),
            "transport_handoff": "about_to_send; remote_receipt_not_proven"}
        self._save()  # If persistence fails, request hook raises BEFORE send.

    def _response_hook(self, response):
        headers = self._headers(response.headers)
        self._audit["response"] = {"http_status": response.status_code, "headers": headers,
            "http_version": response.http_version, "body_read_complete": False}
        self._save()  # Retain status/headers even if reading the body times out.
        response.read()
        self._audit["response"].update(self._body(response.content, response.encoding))
        self._audit["response"]["body_read_complete"] = True
        self._save()

    def generate(self, prompt):
        if self._audit_context is None:
            raise ResumeConflict("Network audit requires durable journal context")
        ctx = self._audit_context
        number = ctx["attempt_number"]
        self._audit_path = self.run_dir / "logs/network" / f"request_{number:04d}.json"
        for parent in (self.run_dir / "logs", self.run_dir / "logs/network", self._audit_path):
            if parent.is_symlink():
                raise ResumeConflict("Audit paths cannot be symlinks")
        if self._audit_path.exists():
            raise ResumeConflict("Audit attempt already exists; never overwrite or resend")
        try:
            message = json.loads(prompt)
            actor = message.get("actor")
        except (ValueError, AttributeError):
            actor = None
        # Text prompts need no parsing for audit: the method binds trusted role
        # metadata before the durable journal attempt. Legacy JSON is fallback.
        actor = (ctx.get("actor_context") or {}).get("actor", actor)
        started = time.perf_counter()
        self._audit = {"schema_version": 1, "phase": "dev_debug", "live_api": True, **ctx,
            "environment_step": ctx["context"]["step"] // 100000,
            "call_ordinal": ctx["context"]["step"] % 100000, "actor": actor,
            "prompt": self.redact(prompt), "started_at": datetime.now(timezone.utc).isoformat(),
            "state": "started", "request": None, "response": None, "error": None,
            "proxy": self.redact(self.proxy), "trust_env": False, "follow_redirects": False,
            "tls_verify": True, "audit_scope": "HTTP application data, not SSH/TLS packet trace"}
        self._save()
        self.last_network_record = str(self._audit_path.relative_to(self.run_dir))
        try:
            result = super().generate(prompt)
            self._audit["state"] = "text_returned_not_schema_validated"
            return result
        except BaseException as exc:
            self._audit["state"] = "interrupted" if not isinstance(exc, Exception) else "failed"
            self._audit["error"] = {"type": type(exc).__name__,
                "category": exc.category if isinstance(exc, ModelRequestError) else
                            "output_truncated" if isinstance(exc, OutputTruncated) else "client_or_interrupt",
                "http_status": exc.http_status if isinstance(exc, ModelRequestError) else None}
            raise
        finally:
            self._audit.update(latency_seconds=time.perf_counter() - started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                finish_reason=self.last_finish_reason, usage=self.last_usage)
            self._save()
            self._audit = None