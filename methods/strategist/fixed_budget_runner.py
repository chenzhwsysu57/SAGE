"""Durable fixed-budget runner; raw code transport, no NLP-policy translation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

from base.baseline_calls import ModelCallJournal, ResumeConflict, _usage
from methods.strategist.batched_training import evidence_hashes
from methods.strategist.evolution import ScriptedModel, Strategy, digest, integer
from methods.strategist.fixed_budget import (
    CAVEATS, ROOT, SCHEMA, FixedBudgetFixture, FixedBudgetSpec, FixedBudgetWorkflow, source_hashes,
)
from methods.strategist.format_retry import validate_response
from methods.strategist.code_output import FixedCodeHeuristic
from methods.strategist.live_workflow import ModelSpec, Paused, _safe_directory, phase_costs
from methods.strategist.personal_backend import PersonalModelSpec, PersonalStrategistBackend
from methods.strategist.runner import atomic, immutable, read_json, run_lock, sealed, unseal, _git


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_directory(path, game):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = _safe_directory(path, ROOT / "results" / game / "strategist_fixed_budget")
    if path.exists() and any(p.is_symlink() for p in path.rglob("*")):
        raise ResumeConflict("Symlink in fixed-budget artifacts")
    return path


def audit_raw_physical(directory):
    """Read-only exact raw-code IO/status/usage audit, no key or API construction."""
    logical = sorted((directory / "prompts/ledger/prompts").rglob("*.json"))
    physical = sorted((directory / "logs/network").glob("request_*.json"))
    if len(logical) != len(physical):
        raise ResumeConflict("Raw physical/semantic request count mismatch")
    for index, (left, right) in enumerate(zip(logical, physical), 1):
        row, network = read_json(left), read_json(right)
        expected = {"model": "deepseek-v4-flash", "temperature": 0.7,
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "thinking": {"type": "disabled"}}
        if (network["request_number"] != index or network["request_json"] != expected
                or network["url"] not in ("https://api.deepseek.com/chat/completions",
                                           "https://api.deepseek.com/v1/chat/completions")):
            raise ResumeConflict("Raw physical request mismatch")
        body = json.loads(network["response_body"])
        if (row["status"] != "success" or network["status"] != "success" or network["http_status"] != 200
                or body["choices"][0]["message"]["content"] != row["response"]
                or body["choices"][0].get("finish_reason") != row["finish_reason"]
                or _usage(body.get("usage")) != row["usage"]):
            raise ResumeConflict("Raw physical response/status/usage mismatch")


class CodePersonalBackend(PersonalStrategistBackend):
    cache_fingerprint = "strategist-fixed-budget-personal-raw-code-v1"

    def audit_physical(self, directory):
        audit_raw_physical(directory)


class ReplayOnly:
    """Completed replays must never generate even if cached records are missing."""
    last_usage = {}
    last_finish_reason = None

    def generate(self, prompt):
        raise ResumeConflict("Completed replay attempted a new model call")


def load_fixed_policy(path, *, game, evaluation_seeds):
    data = unseal(read_json(Path(path)))
    if data["schema"] != SCHEMA or data["game"] != game:
        raise ResumeConflict("Frozen policy schema/game mismatch")
    if not evaluation_seeds:
        raise ValueError("Declare PK seeds explicitly")
    for seed in evaluation_seeds:
        integer(seed, 0, 2**31-1, "PK seed")
    if set(evaluation_seeds) & set(data["train_seeds"]):
        raise ResumeConflict("PK seed overlaps learning data")
    heuristic = FixedCodeHeuristic(data["strategy"]["source"])
    strategy = Strategy(data["strategy"]["id"], heuristic.source, heuristic.sha256)
    if strategy.sha256 != data["strategy"]["sha256"]:
        raise ResumeConflict("Frozen strategy hash mismatch")
    return strategy, data


def validate_evidence(spec, path):
    directory = checked_directory(path, spec.game)
    config = unseal(read_json(directory / "config.json"))
    summary = unseal(read_json(directory / "summary.json"))
    if (config["phase"] != "dev_debug" or config["live_api"]
            or config["signature"]["protocol"] != spec.protocol()
            or config["signature"]["sources"] != source_hashes()
            or summary["status"] != "completed" or summary["rounds_completed"] != 10
            or summary["completed_by_phase"] != {"train": 320, "dev": 0}
            or summary["logical_calls"] != 52):
        raise ResumeConflict("Need reviewed same-source full10x32 offline evidence")
    if unseal(read_json(directory / "metrics/completion_manifest.json")) != evidence_hashes(directory):
        raise ResumeConflict("Offline evidence changed")
    rounds = sorted((directory / "metrics/rounds").glob("*.json"))
    games = sorted((directory / "trajectories").glob("*.json"))
    if len(rounds) != 10 or len(games) != 320:
        raise ResumeConflict("Offline evidence lacks real complete games")
    for index, path in enumerate(rounds, 1):
        data = unseal(read_json(path))
        if data["round"] != index or data["completed_games"] != index*32:
            raise ResumeConflict("Incorrect offline round accounting")
    for path in games:
        data = unseal(read_json(path))
        if data["status"] != "completed" or data["inputs"]["phase"] != "train" or not data["steps"]:
            raise ResumeConflict("Offline evidence has partial/mock-schedule games")
    return str(directory)


def check_resume(directory, signature, model):
    config = unseal(read_json(directory / "config.json"))
    if config["signature"] != signature:
        raise ResumeConflict("Source/model/config changed; cannot cross-version resume")
    previous = unseal(read_json(directory / "summary.json"))
    if previous["status"] not in ("completed", "paused"):
        raise ResumeConflict("Failed/interrupted run requires review; never resend")
    # Snapshot protects ALL existing receipts even for partial safe-boundary pauses.
    manifest = (directory / "metrics/completion_manifest.json" if previous["status"] == "completed"
                else directory / "logs/pause_manifest.json")
    if unseal(read_json(manifest)) != evidence_hashes(directory):
        raise ResumeConflict("Persisted run evidence missing or changed")
    ledger = ModelCallJournal(model, directory / "prompts/ledger", 0, max_retries=0)
    if any(r["status"] != "success" or r["attempts"] for r in ledger._read_records().values()):
        raise ResumeConflict("Failed/started requests block resume")
    for path in (directory / "trajectories").glob("*.json"):
        if unseal(read_json(path))["status"] != "completed":
            raise ResumeConflict("Partial local game requires review")
    return previous


def run_fixed_budget(spec=FixedBudgetSpec(), *, output_dir=None, resume=None, progress=True,
                     stop_after_games=None, key_file=None, approved_real_measurement=False,
                     dev_evidence=None, command=None, model=None, model_spec=None):
    """No expensive execution on import; no PK; safe pauses support explicit resume.

    Model injection is for scripted/MockTransport tests; caller attests provenance.
    Production live uses a fixed audited personal raw-code backend, no injection.
    """
    spec.validate()
    if stop_after_games is not None:
        integer(stop_after_games, 1, spec.train_games, "stop_after_games")
    if not spec.live_api and (key_file is not None or approved_real_measurement or dev_evidence):
        raise ValueError("Offline accepts neither real keys nor live approval")
    if spec.live_api and (model is not None or model_spec is not None):
        raise ValueError("Live uses only audited personal backend")
    # Completed live replay is read-only and needs no new API approval/key/worktree changes.
    replay_completed = False
    if resume:
        directory = checked_directory(resume, spec.game)
        replay_completed = unseal(read_json(directory / "summary.json"))["status"] == "completed"
    evidence = None
    if spec.live_api and not replay_completed:
        if not approved_real_measurement or not dev_evidence:
            raise ValueError("Live needs explicit real_measurement approval and dev evidence")
        evidence = validate_evidence(spec, dev_evidence)
        # Git state is provenance, not a prerequisite for user-run experiments.
        # Record it below without blocking unrelated parallel development.
    if model is None:
        model = CodePersonalBackend(key_file or ROOT / ".deepseek.key") if spec.live_api else FixedBudgetFixture()
        model_spec = PersonalModelSpec() if spec.live_api else ModelSpec(name=FixedBudgetFixture.model_name)
    if model_spec is None:
        raise ValueError("Injected backend requires explicit provenance")
    model_spec.validate(spec.live_api)
    if (model_spec.kind == "scripted" and not isinstance(model, ScriptedModel)
            or not callable(getattr(model, "generate", None)) or getattr(model, "max_retries", 0) != 0):
        raise ValueError("Backend provenance/retry mismatch")
    sources = source_hashes()
    signature = {"schema": SCHEMA, "protocol": spec.protocol(), "sources": sources,
                 "model": asdict(model_spec), "live_api": spec.live_api, "python": sys.version,
                 "backend": type(model).__module__+"."+type(model).__qualname__,
                 "backend_source": digest(inspect.getsource(type(model))),
                 "backend_fingerprint": getattr(model, "cache_fingerprint", None)}
    previous = check_resume(directory, signature, model) if resume else None
    if not resume:
        base = checked_directory(output_dir or ROOT / "results" / spec.game / "strategist_fixed_budget", spec.game)
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = Path(mkdtemp(prefix=f"{stamp}_fixed_budget_{'live' if spec.live_api else 'dev'}_", dir=base))
        for folder in ("prompts", "metrics", "trajectories", "logs", "figures"):
            (directory / folder).mkdir()
        immutable(directory / "config.json", {"schema": SCHEMA,
            "phase": "real_measurement" if spec.live_api else "dev_debug", "live_api": spec.live_api,
            "dry_run": not spec.live_api, "signature": signature, "plan": spec.plan(),
            "created_at": now(), "command": command or [], "git_commit": _git(["rev-parse", "HEAD"]),
            "dirty_worktree": bool(_git(["status", "--porcelain"])),
            "approval": {"by": spec.approval_by, "time": now(), "explicit": approved_real_measurement,
                         "dev_evidence": evidence, "max_requests": None},
            "primary_metrics": ["legal completed episodes", "same-cohort mean normalized return",
                                "paired child-parent gain", "actual provider usage"],
            "exclusions": "none; no payoff-based early stop, resampling or retry of ambiguous calls",
            "caveats": CAVEATS})
    print(f"Run directory: {directory}", flush=True)
    try:
        with run_lock(directory):
            if resume:
                previous = check_resume(directory, signature, model)
            ledger = ModelCallJournal(model, directory / "prompts/ledger", 0, max_retries=0)
            start_requests = ledger.stats()["requests"]
            workflow, frozen = None, None
            status, error, context = "completed", None, {"phase": "bind_or_init"}
            try:
                execution_model = ReplayOnly() if replay_completed else model
                if not replay_completed:
                    bind = getattr(model, "bind_run_directory", None)
                    if callable(bind):
                        bind(directory)
                workflow = FixedBudgetWorkflow(spec, directory, signature, execution_model,
                    stop_after_games=stop_after_games, progress=progress)
                frozen = workflow.execute()
                if sources != source_hashes():
                    raise ResumeConflict("Source changed during execution")
                if spec.live_api or isinstance(model, CodePersonalBackend):
                    audit_raw_physical(directory)
            except Paused:
                status = "paused"
            except KeyboardInterrupt:
                status, error = "interrupted", "KeyboardInterrupt"
            except Exception as exc:
                status, error = "failed", type(exc).__name__
            journal = workflow.calls.journal if workflow else ledger
            summary = {"schema": SCHEMA, "status": status, "error_type": error,
                "context": workflow.context if workflow else context,
                "completed_by_phase": dict(workflow.counts) if workflow else {"train": 0, "dev": 0},
                "rounds_completed": len(workflow.rounds) if workflow else 0,
                "strategies_generated": len(workflow.strategies) if workflow else 0,
                "logical_calls": workflow.call_id if workflow else 0, "calls": journal.stats(),
                "phase_costs": phase_costs(journal, spec.live_api),
                "new_requests": journal.stats()["requests"]-start_requests,
                "new_games": workflow.new_games if workflow else 0,
                "skipped_games": workflow.skipped if workflow else 0,
                "remaining_games": spec.train_games-(workflow.counts["train"] if workflow else 0),
                "validation_games": 0, "external_pk_games": 0,
                "frozen_policy": "metrics/frozen_policy.json" if frozen else None,
                "ready_for_pk": status == "completed" and spec.live_api, "finished_at": now()}
            if replay_completed:
                if status != "completed" or summary["new_requests"] or summary["new_games"]:
                    raise ResumeConflict("Completed replay attempted additional work")
                if unseal(read_json(directory / "metrics/completion_manifest.json")) != evidence_hashes(directory):
                    raise ResumeConflict("Replay changed evidence")
            else:
                try:
                    if status == "completed":
                        immutable(directory / "metrics/completion_manifest.json", evidence_hashes(directory))
                    elif status == "paused":
                        atomic(directory / "logs/pause_manifest.json", sealed(evidence_hashes(directory)))
                except Exception as exc:
                    summary.update(status="failed", error_type=type(exc).__name__, ready_for_pk=False,
                                   context={"phase": "completion_or_pause_manifest"})
                atomic(directory / "summary.json", sealed(summary))
            if resume:
                name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                immutable(directory / "logs/resume" / f"{name}.json", {
                    "original_run_directory": str(directory), "resumed_at": now(),
                    "completed": summary["completed_by_phase"]["train"], "skipped": summary["skipped_games"],
                    "failed": int(summary["status"] in ("failed", "interrupted")),
                    "remaining": summary["remaining_games"], "new_requests": summary["new_requests"]})
            return directory, summary
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()