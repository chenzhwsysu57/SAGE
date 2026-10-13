"""Run-scoped official DeepSeek backend for NLP Strategist.

Credentials are parsed, not sourced. Every physical request and complete response
body is persisted by the existing audited transport. The provider performs one
POST per call, with no hidden retry and no proxy environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from apis.deepseek_journal_wrapper import DeepSeekJournalModel
from base.baseline_calls import ResumeConflict
from methods.strategist.nlp_backend import NLPPolicyBackend
from methods.strategist.nlp_strategy import compile_policy, parse_policy


@dataclass(frozen=True)
class PersonalModelSpec:
    kind: str = "live"
    name: str = "deepseek-v4-flash"
    host: str = "api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 600.0
    connect_timeout: float = 10.0
    network_retries: int = 0

    def validate(self, live: bool) -> None:
        if not live or self.kind != "live":
            raise ValueError("Personal DeepSeek is live-only")
        if (self.name != "deepseek-v4-flash" or self.host != "api.deepseek.com"
                or self.temperature != 0.7 or self.max_tokens is not None
                or self.timeout != 600 or self.connect_timeout != 10
                or self.network_retries != 0):
            raise ValueError("Unexpected personal DeepSeek model settings")


class PersonalStrategistBackend:
    """Lazy audited transport whose network directory belongs to the new run."""

    cache_fingerprint = "strategist-personal-deepseek-v4-flash-v1"
    model = "deepseek-v4-flash"
    model_name = "deepseek-v4-flash"
    temperature = 0.7
    max_tokens = None
    max_retries = 0

    def __init__(self, key_file: str | Path, *, transport=None, network_subdir="logs/network"):
        self.key_file = Path(key_file).resolve()
        self.transport = transport
        self.network_subdir = network_subdir
        self.backend = None
        self.run_directory = None
        self.last_usage = {}
        self.last_finish_reason = None

    def bind_run_directory(self, directory: str | Path) -> None:
        directory = Path(directory).resolve()
        if self.run_directory is not None and self.run_directory != directory:
            raise ResumeConflict("Personal backend cannot move between run directories")
        self.run_directory = directory
        if self.backend is None:
            self.backend = DeepSeekJournalModel(
                key_file=self.key_file,
                network_dir=directory / self.network_subdir,
                model=self.model,
                temperature=self.temperature,
                timeout=600,
                connect_timeout=10,
                max_tokens=None,
                transport=self.transport,
            )

    def generate(self, prompt: str) -> str:
        if self.backend is None:
            raise RuntimeError("Bind personal backend to a run directory before generating")
        try:
            return self.backend.generate(prompt)
        finally:
            self.last_usage = self.backend.last_usage
            self.last_finish_reason = self.backend.last_finish_reason

    def redact(self, text: str) -> str:
        return self.backend.redact(text) if self.backend is not None else text

    def audit_physical(self, run_directory: str | Path) -> None:
        if self.backend is None:
            raise ResumeConflict("Personal backend was not initialized")
        run_directory = Path(run_directory).resolve()
        if run_directory != self.run_directory:
            raise ResumeConflict("Personal audit run directory mismatch")
        logical_files = sorted((run_directory / "prompts/ledger/prompts").rglob("*.json"))
        physical_files = sorted((run_directory / "logs/network").glob("request_*.json"))
        if len(logical_files) != len(physical_files):
            raise ResumeConflict("Personal physical/logical request count mismatch")
        for index, (logical_path, physical_path) in enumerate(zip(logical_files, physical_files), 1):
            logical = json.loads(logical_path.read_text(encoding="utf-8"))
            physical = json.loads(physical_path.read_text(encoding="utf-8"))
            outbound = NLPPolicyBackend.outbound_prompt(logical["prompt"])
            if (physical_path.is_symlink() or physical.get("request_number") != index
                    or physical.get("url") != self.backend.ENDPOINT
                    or physical.get("request_json") != self.backend._request_payload(outbound)):
                raise ResumeConflict("Personal physical request differs from semantic journal")
            if logical["status"] == "success":
                try:
                    body = json.loads(physical["response_body"])
                    choice = body["choices"][0]
                    request = json.loads(logical["prompt"])
                    raw_content = choice["message"]["content"]
                    semantic_content = raw_content
                    if request["true_stage"] in ("seed_code", "code"):
                        try:
                            semantic_content = compile_policy(parse_policy(raw_content, request["game"]))
                        except ValueError:
                            # Invalid text is deliberately retained verbatim so
                            # the correction ledger can validate and repair it.
                            semantic_content = raw_content
                    checks = {
                        "physical_success": physical["status"] == "success",
                        "http_200": physical["http_status"] == 200,
                        "semantic_response": semantic_content == logical["response"],
                        "finish_reason": choice.get("finish_reason") == logical["finish_reason"],
                    }
                    if not all(checks.values()):
                        failed = ",".join(name for name, passed in checks.items() if not passed)
                        raise ResumeConflict(f"Personal response audit failed at request {index}: {failed}")
                except ResumeConflict:
                    raise
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    stage = "unknown"
                    try:
                        stage = json.loads(logical["prompt"]).get("true_stage", "unknown")
                    except (ValueError, KeyError, TypeError):
                        pass
                    raise ResumeConflict(
                        f"Personal response audit could not parse request {index} stage {stage}: {type(exc).__name__}"
                    ) from None
            elif physical["status"] not in ("failed", "started"):
                raise ResumeConflict("Personal failed request status mismatch")

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
