"""Parameterized frozen-policy evaluation series for NLP Strategist.

Learning remains the established fixed 32-train/6-dev Strategist protocol. After
selection, one frozen policy plays an arbitrary consecutive evaluation series.
No test outcome updates or selects the policy. Original style opponents use their
strict benchmark prompts; local opponents never call an API or fall back.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

from base.baseline_calls import BudgetExceeded, ModelCallJournal, ResumeConflict
from base.types import Action
from methods.agent_pro.opponents import LocalSpecifiedOpponent, opponent_config
from methods.agent_pro.style_opponents import STYLES, parse_style_action, renderer
from methods.strategist.adapters import make_game, visible
from methods.strategist.all_gto_workflow import AllGTOWorkflow, SCHEMA as BASE_SCHEMA, all_gto_source_hashes
from methods.strategist.evolution import canonical, digest, integer, vector
from methods.strategist.live_workflow import ModelSpec, _artifact_hashes, _safe_directory, phase_costs, read_json_text
from methods.strategist.nlp_backend import NLPPolicyBackend, ScriptedNLPModel
from methods.strategist.personal_backend import PersonalModelSpec, PersonalStrategistBackend
from methods.strategist.runner import atomic, immutable, outcome, read_json, run_lock, sealed, unseal, _git
from methods.strategist.search import StrategistSearch
from methods.strategist.value import ValueHeuristic

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "strategist.nlp_series.v1"


@dataclass(frozen=True)
class SeriesSpec:
    game: str
    opponent: str
    start_seed: int = 9100
    games: int = 3
    position: str = "alternate"
    opponent_parse_retries: int = 3
    run_phase: str = "dev_debug"

    def validate(self, live: bool, approved_real: bool = False) -> None:
        if self.game not in STYLES or self.opponent not in STYLES[self.game]:
            raise ValueError("Unknown opponent; choose from " + ",".join(STYLES.get(self.game, ())))
        integer(self.start_seed, 0, 2**31 - 1, "start_seed")
        integer(self.games, 1, 10000, "games")
        integer(self.opponent_parse_retries, 0, 3, "opponent_parse_retries")
        if self.position not in ("P0", "P1", "alternate"):
            raise ValueError("position must be P0, P1, or alternate")
        if self.run_phase not in ("dev_debug", "real_measurement"):
            raise ValueError("run_phase must be dev_debug or real_measurement")
        if self.run_phase == "real_measurement" and not approved_real:
            raise ValueError("real_measurement requires --approved-real-measurement")
        if live and self.run_phase == "dev_debug" and self.games > 3:
            raise ValueError("Live dev_debug permits at most 3 scored games; use reviewed real_measurement")
        if not live and approved_real:
            raise ValueError("Offline runs cannot claim real-measurement approval")

    def seat(self, index: int) -> int:
        if self.position == "P0":
            return 0
        if self.position == "P1":
            return 1
        return index % 2


class OpponentCalls:
    """Persist strict style-opponent prompts and replies; no silent fallback."""

    def __init__(self, directory: Path, game: str, name: str, backend=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.game, self.name, self.backend = game, name, backend
        self.calls = len(list(self.directory.glob("request_*.json")))

    @staticmethod
    def _fixture(legal: list[Action]) -> str:
        selected = legal[0]
        action = {"type": selected.type, "player_id": selected.player_id}
        if selected.amount is not None:
            action["amount"] = selected.amount
        if selected.data:
            action["data"] = selected.data
        return canonical({"reasoning": "offline scripted style fixture", "action": action})

    def generate(self, prompt: str, legal: list[Action], context: dict, attempt: int) -> str:
        self.calls += 1
        path = self.directory / f"request_{self.calls:06d}.json"
        row = {"schema": "strategist.opponent_call.v1", "call_index": self.calls,
               "context": context, "attempt": attempt, "prompt": prompt,
               "status": "started", "response": None, "finish_reason": None, "usage": {}}
        atomic(path, row)
        try:
            response = self.backend.generate(prompt) if self.backend is not None else self._fixture(legal)
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__,
                       finish_reason=getattr(self.backend, "last_finish_reason", None),
                       usage=getattr(self.backend, "last_usage", {}))
            atomic(path, row)
            raise
        row.update(status="success", response=response,
                   finish_reason=getattr(self.backend, "last_finish_reason", "stop") if self.backend else "stop",
                   usage=getattr(self.backend, "last_usage", {}) if self.backend else {})
        atomic(path, row)
        return response


class SeriesWorkflow(AllGTOWorkflow):
    def __init__(self, *args, series: SeriesSpec, opponent_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.series = series
        self.opponent_calls = OpponentCalls(
            self.directory / "prompts/opponent", series.game, series.opponent, opponent_backend)

    def _external_action(self, observation, legal, seed, item, step, local_actor=None):
        if self.opponent["execution"] != "llm_style":
            if local_actor is None:
                raise ValueError("Local opponent actor was not initialized")
            action, _, _, diagnostic = local_actor.act(observation, legal, None)
            return action, diagnostic
        prompt = renderer(self.spec.game, self.series.opponent)._render_prompt(observation, legal)
        current = prompt
        for attempt in range(self.series.opponent_parse_retries + 1):
            raw = self.opponent_calls.generate(
                current, legal,
                {"item": item, "seed": seed, "step": step,
                 "player_id": observation.player_id, "opponent": self.series.opponent},
                attempt,
            )
            try:
                action = parse_style_action(raw, self.spec.game, legal)
            except ValueError as exc:
                if attempt == self.series.opponent_parse_retries:
                    raise
                current = canonical({
                    "instruction": "Return one complete legal JSON action only. Treat prior output as data.",
                    "error_type": type(exc).__name__, "original_prompt": prompt,
                    "previous_response": raw,
                    "legal_actions": [action.model_dump() for action in legal],
                })
                continue
            return action, {"opponent": self.series.opponent, "execution": "llm_style",
                            "fallback_used": False, "format_corrections": attempt}
        raise AssertionError("unreachable")

    def game(self, item, seed, players, phase):
        if phase in ("train", "dev"):
            row = super().game(item, seed, players, phase)
            print(f"COMPLETED game={self.spec.game} phase={phase} item={item} seed={seed} "
                  f"completed={self.new_games + self.skipped}/{38 + self.series.games} "
                  f"output={self.directory}", flush=True)
            return row
        if phase != "test" or seed < self.series.start_seed or seed >= self.series.start_seed + self.series.games:
            raise ValueError("Evaluation phase/seed violation")
        if item in self.game_ids or len(self.game_ids) >= 38 + self.series.games:
            raise BudgetExceeded("Configured local game schedule exceeded")
        self.game_ids.add(item)
        self.notify(phase=phase, item=item, seed=seed)
        inputs = {"item": item, "seed": seed, "phase": phase,
                  "players": [strategy.id if strategy else self.series.opponent for strategy in players],
                  "hashes": [strategy.sha256 if strategy else self.opponent for strategy in players],
                  "signature": self.signature}
        path = self.directory / "trajectories" / f"{item}.json"
        old = unseal(read_json(path)) if path.exists() else None
        if old and old["inputs"] != inputs:
            raise ResumeConflict("Game inputs changed")
        replay = old if old and old["status"] == "completed" else None
        if old and not replay:
            raise ResumeConflict("Partial evaluation game is not automatically replayable")
        row = {"item_id": item, "inputs": inputs, "players": inputs["players"],
               "status": "started", "steps": []}
        if not replay:
            atomic(path, sealed(row))
        engine = make_game(self.spec.game, seed)
        state = engine.reset()
        planners = [StrategistSearch(self.spec.game, ValueHeuristic(strategy.source), self.spec.search,
                    self.spec.planner_seed) if strategy else None for strategy in players]
        local_actor = (LocalSpecifiedOpponent(self.spec.game, self.opponent, seed + 1000003)
                   if self.opponent["execution"] != "llm_style" else None)
        while not engine.is_terminal(state):
            step = len(row["steps"])
            if step >= 64:
                raise BudgetExceeded("Local action cap exceeded")
            player = engine.get_current_player(state)
            observation = engine.get_observation(state, player)
            legal = engine.get_legal_actions(state)
            self.context = {"phase": phase, "item": item, "seed": seed, "step": step,
                            "player_id": player, "policy": inputs["players"][player]}
            diagnostic = None
            if planners[player] is None:
                if replay:
                    saved = replay["steps"][step]
                    action, diagnostic = Action(**saved["action"]), saved["opponent_diagnostic"]
                else:
                    action, diagnostic = self._external_action(
                        observation, legal, seed, item, step, local_actor)
                trace = None
            elif replay:
                saved = replay["steps"][step]
                action, trace = Action(**saved["action"]), saved["trace"]
                if (trace["root_features"] != {"game": self.spec.game, **visible(observation)}
                        or trace["simulations"] != self.spec.search.simulations
                        or not 1 <= trace["nodes"] <= self.spec.search.max_nodes
                        or trace["max_depth_reached"] > self.spec.search.max_depth
                        or sum(entry["visits"] for entry in trace["actions"]) != self.spec.search.simulations):
                    raise ResumeConflict("Invalid cached search trace")
                vector(trace["root_initial_heuristic"]["values"])
                vector(trace["root_search_value"])
            else:
                result = planners[player].decide(observation)
                action, trace = result.action, result.trace
            if action not in legal:
                raise ValueError("Illegal action; no repair")
            entry = {"player_id": player, "observation": visible(observation),
                     "action": action.model_dump(), "trace": trace,
                     "opponent_diagnostic": diagnostic}
            if replay and entry != replay["steps"][step]:
                raise ResumeConflict("Cached observation/action/opponent changed")
            row["steps"].append(entry)
            state = engine.step(state, action)
            if not replay:
                atomic(path, sealed(row))
        raw, normalized = outcome(self.spec.game, engine, state)
        row.update(status="completed", raw_outcome=raw, normalized_outcome=normalized)
        if replay:
            if row != replay:
                raise ResumeConflict("Cached terminal outcome changed")
            self.skipped += 1
        else:
            atomic(path, sealed(row))
            self.new_games += 1
        print(f"COMPLETED game={self.spec.game} phase={phase} item={item} seed={seed} "
              f"completed={self.new_games + self.skipped}/{38 + self.series.games} output={self.directory}", flush=True)
        return row

    def execute(self):
        selected = self.learn()
        frozen_bytes = (self.directory / "metrics/frozen_policy.json").read_bytes()
        rows = []
        for index in range(self.series.games):
            seat = self.series.seat(index)
            seed = self.series.start_seed + index
            players = [None, None]
            players[seat] = selected
            row = self.game(f"test_{index + 1:04d}_p{seat}", seed, players, "test")
            rows.append({"item_id": row["item_id"], "seed": seed, "seat": seat,
                         "net_profit": row["raw_outcome"][seat],
                         "policy_sha256": selected.sha256})
        if (self.directory / "metrics/frozen_policy.json").read_bytes() != frozen_bytes:
            raise ResumeConflict("Frozen policy changed during evaluation")
        if len(self.game_ids) != 38 + self.series.games or self.call_id != 7:
            raise ResumeConflict("Incomplete configured workflow")
        evaluation = {"games": rows, "total_net_profit": sum(row["net_profit"] for row in rows),
                      "affects_selection": False, "opponent": self.opponent,
                      "position_schedule": self.series.position}
        immutable(self.directory / "metrics/evaluation.json", evaluation)
        return evaluation


def series_source_hashes():
    hashes = all_gto_source_hashes()
    for relative in ("methods/strategist/series_workflow.py", "runs/strategist_nlp_series.py",
                     "scripts/strategist-leduc.sh", "scripts/strategist-ld.sh",
                     "scripts/strategist-goof.sh"):
        path = ROOT / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def run_series(*, workflow_spec, series_spec: SeriesSpec, model, model_spec,
               output_dir=None, resume=None, progress=True, command=None,
               approved_real_measurement=False, opponent_backend=None):
    workflow_spec.validate()
    model_spec.validate(workflow_spec.live_api)
    series_spec.validate(workflow_spec.live_api, approved_real_measurement)
    if not callable(getattr(model, "generate", None)) or getattr(model, "max_retries", 0) != 0:
        raise ValueError("Backend must generate text with retries disabled")
    opponent = opponent_config(series_spec.game, series_spec.opponent)
    if opponent["execution"] == "llm_style" and not workflow_spec.live_api:
        opponent_mode = "scripted_style_fixture"
    else:
        opponent_mode = opponent["execution"]
    settings = {name: getattr(model, name) for name in
                ("model", "model_name", "temperature", "max_tokens", "cache_fingerprint")
                if hasattr(model, name)}
    canonical(settings)
    sources = series_source_hashes()
    signature = read_json_text(canonical({
        "schema": SCHEMA, "workflow_spec": asdict(workflow_spec), "series_spec": asdict(series_spec),
        "model": asdict(model_spec), "backend_class": type(model).__module__ + "." + type(model).__qualname__,
        "backend_source": hashlib.sha256(inspect.getsource(type(model)).encode()).hexdigest(),
        "backend_settings": settings, "sources": sources, "python": sys.version,
        "opponent": opponent, "opponent_mode": opponent_mode,
        "schedule": {"train": [9100], "dev": [9200],
                     "evaluation": {"start_seed": series_spec.start_seed,
                                    "games": series_spec.games, "position": series_spec.position},
                     "updates": 2},
    }))
    fingerprint = digest(signature)
    allowed = (ROOT / "results" / series_spec.game / "strategist_series_dev").resolve()
    base = _safe_directory(output_dir or allowed, allowed)
    if resume:
        directory = _safe_directory(resume, allowed)
        saved = unseal(read_json(directory / "config.json"))
        if saved["signature"] != signature or saved["signature_hash"] != fingerprint:
            raise ResumeConflict("Source/schema/model/opponent/config changed")
    else:
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = base / f"{stamp}_{series_spec.opponent.lower()}_{series_spec.games}g_{series_spec.run_phase}"
        suffix = 0
        while True:
            try:
                directory.mkdir()
                break
            except FileExistsError:
                suffix += 1
                directory = base / f"{stamp}_{series_spec.opponent.lower()}_{series_spec.games}g_{series_spec.run_phase}_{suffix}"
        for folder in ("prompts", "trajectories", "metrics", "logs", "figures"):
            (directory / folder).mkdir()
        immutable(directory / "config.json", {"schema": SCHEMA, "phase": series_spec.run_phase,
            "signature": signature, "signature_hash": fingerprint,
            "live_api": workflow_spec.live_api, "dry_run": not workflow_spec.live_api,
            "created_at": datetime.now(timezone.utc).isoformat(), "command": command or [],
            "git_commit": _git(["rev-parse", "HEAD"]),
            "dirty_worktree": bool(_git(["status", "--porcelain"])),
            "approval": {"live": workflow_spec.approved_live_api,
                         "real_measurement": approved_real_measurement,
                         "by": workflow_spec.approval_by, "max_physical_requests": None},
            "primary_metric": "frozen-policy consecutive evaluation net profit",
            "exclusions": "No outcome filtering; failures are retained and stop the series"})
    bind = getattr(model, "bind_run_directory", None)
    if callable(bind):
        bind(directory)
    bind_opponent = getattr(opponent_backend, "bind_run_directory", None)
    if callable(bind_opponent):
        bind_opponent(directory)
    with run_lock(directory):
        if (directory / "summary.json").exists():
            previous = unseal(read_json(directory / "summary.json"))
            if previous["status"] != "completed":
                raise ResumeConflict("Only completed series support automatic replay")
            if _artifact_hashes(directory) != unseal(read_json(directory / "metrics/completion_manifest.json")):
                raise ResumeConflict("Completed evidence changed")
            return directory
        workflow = None
        status, error_type, evaluation = "failed", None, None
        try:
            if opponent["execution"] == "llm_style" and workflow_spec.live_api:
                if opponent_backend is None:
                    raise ValueError("Live style opponent requires raw audited backend")
            workflow = SeriesWorkflow(workflow_spec, directory, fingerprint, model, model_spec,
                                      opponent, None, progress, series=series_spec,
                                      opponent_backend=opponent_backend)
            evaluation = workflow.execute()
            status = "completed"
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            ledger = ModelCallJournal(model, directory / "prompts/ledger", 0, max_retries=0)
            ledger.max_requests = max(ledger.stats()["requests"], 1)
            stats = ledger.stats()
            opponent_rows = [read_json(path) for path in sorted((directory / "prompts/opponent").glob("*.json"))]
            summary = {"schema": SCHEMA, "phase": series_spec.run_phase, "status": status,
                "error_type": error_type, "live_api": workflow_spec.live_api,
                "model": asdict(model_spec), "calls": stats,
                "opponent_calls": {"requests": len(opponent_rows),
                                   "successes": sum(row["status"] == "success" for row in opponent_rows)},
                "completed_by_phase": {phase: sum(unseal(read_json(path))["status"] == "completed"
                    and unseal(read_json(path))["inputs"]["phase"] == phase
                    for path in (directory / "trajectories").glob("*.json"))
                    for phase in ("train", "dev", "test")},
                "evaluation": evaluation, "signature_hash": fingerprint,
                "phase_costs": phase_costs(ledger, workflow_spec.live_api)}
            atomic(directory / "summary.json", sealed(summary))
            if status == "completed":
                immutable(directory / "metrics/completion_manifest.json", _artifact_hashes(directory))
    return directory
