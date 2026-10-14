"""Independent bounded Strategist workflow with an injected text model.

No inheritance from the offline runner/model bridge, exec, global patching,
weight training, or LLM decisions inside games. Only high-level heuristic
generation uses the single physical call ledger. Existing algorithms are reused.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

from base.baseline_calls import BudgetExceeded, ModelCallJournal, ResumeConflict
from base.types import Action
from methods.agent_pro.opponents import LocalSpecifiedOpponent, opponent_config
from methods.strategist.adapters import GAMES, PAYOFF_SCALES, make_game, rules, visible
from methods.strategist.evolution import Idea, IdeaQueue, ScriptedModel, Strategy, canonical, digest, integer, key_feedback, vector
from methods.strategist.format_retry import CorrectingCalls
from methods.strategist.runner import atomic, immutable, outcome, read_json, run_lock, sealed, unseal, _git
from methods.strategist.search import SearchConfig, StrategistSearch
from methods.strategist.value import ValueHeuristic, _ALLOWED, _ARITY

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "strategist.live_workflow.v1"
OPPONENTS = {"leduc": "GTO", "liars_dice": "GTO", "goofspiel": "Mirror"}
CAVEATS = [
    "Small dev_debug workflow, not real_measurement or evidence of effectiveness.",
    "Independent paper-inspired adaptation; uniform-prior information-set PUCT is not exact Appendix F.",
    "Train seed 9100 and dev seed 9200 are reused; disjoint seeds do not guarantee disjoint state families.",
    "Self-play uses a common cohort including self, both seats, equal opponent weights.",
    "Dev chooses among three heuristics against the initial heuristic; selection is biased at this sample size.",
    "Test consists of three frozen-policy games only, not an initial-baseline comparison.",
    "GTO tables are existing CFR approximations, not certified exact equilibria.",
    "Schema fixtures cover initial states only; subsequent heuristic runtime failure stops without repair.",
    "HTTPX 600s phase timeout is not a total deadline; ambiguous calls are never resent.",
    "Checksums detect accidental corruption, not coordinated malicious artifact rewriting.",
]


@dataclass(frozen=True)
class LiveSpec:
    game: str = "leduc"
    max_requests: int = 64
    format_retries: int = 3
    search: SearchConfig = field(default_factory=lambda: SearchConfig(32, 12, 128))
    planner_seed: int = 17
    live_api: bool = False
    approved_live_api: bool = False
    approval_by: str = ""

    def validate(self):
        if self.game not in GAMES:
            raise ValueError("Unsupported game")
        integer(self.max_requests, 0, 64, "max_requests")
        integer(self.format_retries, 0, 3, "format_retries")
        integer(self.planner_seed, 0, 2**31-1, "planner_seed")
        if not isinstance(self.search, SearchConfig):
            raise ValueError("Expected SearchConfig")
        self.search.__post_init__()
        if self.search.simulations > 32 or self.search.max_depth > 12 or self.search.max_nodes > 128:
            raise ValueError("Bounded search cap is 32/12/128")
        if type(self.live_api) is not bool or type(self.approved_live_api) is not bool:
            raise ValueError("Boolean live flags required")
        if self.live_api and (not self.approved_live_api or not self.approval_by.strip()):
            raise ValueError("Live requires explicit approval and approver")
        if not self.live_api and (self.approved_live_api or self.approval_by):
            raise ValueError("Do not mark an offline run as live-approved")


@dataclass(frozen=True)
class ModelSpec:
    """Caller attests provenance; arbitrary providers must disable internal retries."""
    kind: str = "scripted"
    name: str = "strategist-scripted-mock-v1"
    host: str = "none"
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 600.0
    connect_timeout: float = 10.0
    network_retries: int = 0

    @classmethod
    def campus(cls):
        return cls("live", "DeepSeek-V4-Flash", "aigc-api.hkust-gz.edu.cn", 1.0)

    def validate(self, live):
        if self.kind not in ("scripted", "mock_transport", "live") or (self.kind == "live") != live:
            raise ValueError("Model provenance does not match live flag")
        if (self.network_retries != 0 or type(self.network_retries) is not int
                or self.max_tokens is not None or self.timeout != 600 or self.connect_timeout != 10):
            raise ValueError("No network retries; fixed timeout600/connect10 and max_tokens=None")
        if live and self.temperature != 1:
            raise ValueError("Live temperature must be 1")
        # Descriptor contains no credentials or arbitrary model object dumps.
        import re
        if any(type(v) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", v)
               or v.startswith("sk-") for v in (self.name, self.host)):
            raise ValueError("Unsafe model name/host")


class CampusBackend:
    """Lazy construction: config/approval/resume checks precede even key access."""
    def __init__(self, key_file=None):
        self.key_file, self.backend = key_file, None
        self.last_usage, self.last_finish_reason = {}, None

    def generate(self, prompt):
        if self.backend is None:
            from apis.hkustgz_wrapper import HKUSTGZModel
            self.backend = HKUSTGZModel(key_file=self.key_file, max_tokens=None,
                                       temperature=1, timeout=600, connect_timeout=10)
        try:
            return self.backend.generate(prompt)
        finally:
            self.last_usage = self.backend.last_usage
            self.last_finish_reason = self.backend.last_finish_reason

    def redact(self, text):
        return self.backend.redact(text) if self.backend is not None else text

    def close(self):
        if self.backend is not None:
            self.backend.close()


def source_hashes():
    paths = set()
    for folder in ("base", "games", "methods/strategist", "tools/leduc"):
        paths.update((ROOT / folder).rglob("*.py"))
    for name in ("runs/strategist_live_bench.py", "apis/hkustgz_wrapper.py", "methods/sage/cfr.py",
                 "methods/baseline_opponents.py", "methods/sage/liars_dice_cfr.py",
                 "methods/agent_pro/opponents.py", "methods/agent_pro/policy.py",
                 "modeling/goofspiel/archetypes.py", "methods/baseline/simple_prompt.py",
                 "games/prize_pool_n5.txt"):
        paths.add(ROOT / name)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def language_contract(game):
    examples = []
    # Independent, non-experiment fixtures; never concatenate both hands of a real episode.
    for player in (0, 1):
        engine = make_game(game, player)
        examples.append({"game": game, **visible(engine.get_observation(engine.reset(), player))})
    return {
        "rules": rules(game), "input_examples_independent_fixtures": examples,
        "input_schema": "Exact keys game/player_id/private_info/public_info/history. Dict keys are strings; own private data only. Cards are strings, not objects. Optional fields may be null; history can grow and dice counts shrink. Guard missing keys with 'in', not .get().",
        "ast_nodes": sorted(t.__name__ for t in _ALLOWED), "builtin_positional_arities": _ARITY,
        "language": "One unannotated def evaluate_state(state). Assign only local names. if/elif/else, bounded for over list/tuple/range; no while, comprehensions, attributes, methods (.get/.items), imports, slices, mutation, unpacking assignments, nested functions, keyword calls, exceptions, pow, recursion or exec. Numeric + - * / // %, comparisons, and/or/not, membership, conditionals, subscript, list/tuple/string-keyed dict literals. min/max/sum take numeric list/tuple; no generators. Arithmetic excludes bool. Return ((v0,v1), explanation_dict), finite [-1,1], zero sum tolerance1e-8, always fixed P0/P1 order even when player_id=1. Source <=12000 chars, AST<=2000/depth60, runtime fuel10000, containers<=128, string<=512. No fallback on runtime failure.",
        "target": f"Expected terminal net return divided by {PAYOFF_SCALES[game]}; not a win probability. Do not request or infer access to hidden engine fields.",
        "minimal_valid_function": 'def evaluate_state(state):\n    p = state["player_id"]\n    v = 0.0\n    return (v, -v), {"perspective": p}\n',
    }


class Paused(RuntimeError):
    """Safe pause between local games; successful calls replay from cache."""


class LiveWorkflow:
    def __init__(self, spec, directory, signature, model, model_spec, opponent, stop_after_games=None, progress=True):
        self.spec, self.directory, self.signature = spec, directory, signature
        self.model_spec, self.opponent = model_spec, opponent
        self.stop_after_games, self.progress = stop_after_games, progress
        self.strategies, self.rounds, self.improvements = [], [], []
        self.queue = IdeaQueue()
        self.game_ids = set()
        self.call_id = self.new_games = self.skipped = 0
        self.context = {"phase": "setup"}
        self.calls = CorrectingCalls(model, directory / "prompts/ledger", game=spec.game,
                                     max_requests=spec.max_requests, format_retries=spec.format_retries,
                                     network_retries=0, identity={"signature": signature, "model": asdict(model_spec)})

    def notify(self, **context):
        self.context = context
        if self.progress:
            print(f"game={self.spec.game} {canonical(context)} output={self.directory}", flush=True)

    def call(self, stage, payload):
        self.call_id += 1
        self.notify(phase="learn", stage=stage, logical_id=self.call_id)
        request = {"schema": SCHEMA, "true_stage": stage, "game": self.spec.game,
                   "contract": language_contract(self.spec.game), **payload}
        return self.calls.generate(self.call_id, stage, canonical(request))

    def snapshot(self):
        atomic(self.directory / "metrics/lineage.json", sealed({
            "strategies": [asdict(s) for s in self.strategies], "rounds": self.rounds,
            "paired_improvements": self.improvements, "ideas": self.queue.snapshot()}))

    def generate(self, identity, source, parent=None, idea=None):
        strategy = Strategy.create(identity, source, parent, idea)
        self.strategies.append(strategy)
        immutable(self.directory / "metrics/policies" / f"{identity}.json", asdict(strategy))
        self.snapshot()
        return strategy

    def game(self, item, seed, players, phase):
        if phase not in ("train", "dev", "test") or seed not in {"train": (9100,), "dev": (9200,), "test": (9300,9301,9302)}[phase]:
            raise ValueError("Phase/seed violation")
        if item in self.game_ids or len(self.game_ids) >= 41:
            raise BudgetExceeded("Fixed local game schedule exceeded")
        self.game_ids.add(item)
        self.notify(phase=phase, item=item, seed=seed)
        inputs = {"item": item, "seed": seed, "phase": phase,
                  "players": [s.id if s else OPPONENTS[self.spec.game] for s in players],
                  "hashes": [s.sha256 if s else self.opponent for s in players], "signature": self.signature}
        path = self.directory / "trajectories" / f"{item}.json"
        old = unseal(read_json(path)) if path.exists() else None
        if old and old["inputs"] != inputs:
            raise ResumeConflict("Game inputs changed")
        replay = old if old and old["status"] == "completed" else None
        if not replay and self.stop_after_games is not None and self.new_games >= self.stop_after_games:
            raise Paused("Local game boundary pause")
        row = {"item_id": item, "inputs": inputs, "players": inputs["players"], "status": "started", "steps": []}
        if not replay:
            atomic(path, sealed(row))
        engine = make_game(self.spec.game, seed)
        state = engine.reset()
        planners = [StrategistSearch(self.spec.game, ValueHeuristic(s.source), self.spec.search,
                                    self.spec.planner_seed) if s else None for s in players]
        opponent = LocalSpecifiedOpponent(self.spec.game, self.opponent, seed+1000003) if None in players else None
        while not engine.is_terminal(state):
            step = len(row["steps"])
            if step >= 64:
                raise BudgetExceeded("Local action cap exceeded")
            player = engine.get_current_player(state)
            obs = engine.get_observation(state, player)
            legal = engine.get_legal_actions(state)
            self.context = {"phase": phase, "item": item, "seed": seed, "step": step,
                            "player_id": player, "policy": inputs["players"][player]}
            diagnostic = None
            if planners[player] is None:
                action, _, _, diagnostic = opponent.act(obs, legal, None)
                trace = None
            elif replay:
                saved = replay["steps"][step]
                action, trace = Action(**saved["action"]), saved["trace"]
                if (trace["root_features"] != {"game": self.spec.game, **visible(obs)}
                        or trace["simulations"] != self.spec.search.simulations
                        or not 1 <= trace["nodes"] <= self.spec.search.max_nodes
                        or trace["max_depth_reached"] > self.spec.search.max_depth
                        or sum(x["visits"] for x in trace["actions"]) != self.spec.search.simulations):
                    raise ResumeConflict("Invalid cached search trace")
                vector(trace["root_initial_heuristic"]["values"])
                vector(trace["root_search_value"])
            else:
                result = planners[player].decide(obs)
                action, trace = result.action, result.trace
            if action not in legal:
                raise ValueError("Illegal action; no repair")
            entry = {"player_id": player, "observation": visible(obs), "action": action.model_dump(),
                     "trace": trace, "opponent_diagnostic": diagnostic}
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
        return row

    def versus(self, policy, cohort, seed, label, phase):
        rows, scores = [], []
        for j, opponent in enumerate(cohort):
            for seat in (0,1):
                players = [opponent, opponent]
                players[seat] = policy
                row = self.game(f"{label}_o{j}_p{seat}", seed, players, phase)
                rows.append(row)
                scores.append(row["normalized_outcome"][seat])
        return sum(scores)/len(scores), rows

    def round_robin(self, iteration):
        cohort = list(self.strategies)
        returns = {s.id: {o.id: [] for o in cohort} for s in cohort}
        rows = []
        for i, left in enumerate(cohort):
            for right in cohort[i:]:
                for swap in (0,1):
                    players = [left,right] if swap == 0 else [right,left]
                    row = self.game(f"round{iteration}_{left.id}_{right.id}_swap{swap}", 9100, players, "train")
                    rows.append(row)
                    for seat in (0,1):
                        returns[players[seat].id][players[1-seat].id].append(row["normalized_outcome"][seat])
        for s in cohort:
            s.score = sum(sum(v)/len(v) for v in returns[s.id].values())/len(cohort)
            s.score_round = iteration
        record = {"round": iteration, "cohort": [s.id for s in cohort], "seed": 9100,
                  "both_seats": True, "scores": {s.id: s.score for s in cohort},
                  "game_ids": [r["item_id"] for r in rows]}
        self.rounds.append(record)
        immutable(self.directory / "metrics/rounds" / f"{iteration}.json", record)
        self.snapshot()
        return rows

    def develop(self, candidate, initial):
        score, rows = self.versus(candidate, [initial], 9200, f"dev_{candidate.id}", "dev")
        candidate.dev_score = score
        immutable(self.directory / "metrics/dev" / f"{candidate.id}.json",
                  {"score": score, "game_ids": [r["item_id"] for r in rows], "seed": 9200})

    def learn(self):
        source = self.call("seed_code", {"instruction": "Generate the initial game-specific value heuristic using the contract and guarded access to actual input keys. Return function source only."})
        initial = self.generate("s0", source)
        training = self.round_robin(0)
        self.develop(initial, initial)
        for iteration in (1,2):
            pool = sorted(self.strategies, key=lambda s: (-s.score, s.id))
            parent = pool[(iteration-1) % len(pool)]
            feedback = key_feedback(training, parent.id)
            # Explicit projection: NO dev_score, evaluation metrics, or future states.
            reflection = self.call("reflection", {"iteration": iteration,
                "parent": {"id": parent.id, "source": parent.source, "train_score": parent.score},
                "feedback": feedback, "instruction": "Reflect on visited training root value discrepancies and limitations. Return plain text <=12000 characters."})
            idea_text = self.call("idea", {"iteration": iteration, "parent_id": parent.id,
                "reflection": reflection, "instruction": "Propose one implementable value heuristic improvement, plain text <=4000 characters."})
            self.queue.add(Idea(f"i{iteration}", parent.id, idea_text))
            chosen = self.queue.select(self.spec.planner_seed+iteration, {s.id for s in pool})
            parent = next(s for s in pool if s.id == chosen.parent_id)
            source = self.call("code", {"iteration": iteration, "parent_source": parent.source,
                "idea": asdict(chosen), "instruction": "Implement the selected improvement in the restricted value function. Return function source only."})
            candidate = self.generate(f"s{iteration}", source, parent.id, chosen.id)
            child_score, child_rows = self.versus(candidate, pool, 9100, f"paired{iteration}_child", "train")
            parent_score, parent_rows = self.versus(parent, pool, 9100, f"paired{iteration}_parent", "train")
            self.queue.observe(chosen.id, child_score, parent_score)
            self.improvements.append({"iteration": iteration, "parent_id": parent.id, "child_id": candidate.id,
                "idea_id": chosen.id, "cohort": [s.id for s in pool], "seed": 9100, "both_seats": True,
                "child_score": child_score, "parent_score": parent_score, "gain": child_score-parent_score,
                "child_games": [r["item_id"] for r in child_rows], "parent_games": [r["item_id"] for r in parent_rows]})
            training = self.round_robin(iteration)
            self.develop(candidate, initial)
            self.snapshot()
        selected = sorted(self.strategies, key=lambda s: (-s.dev_score, s.id))[0]
        frozen = {"schema": SCHEMA, "strategy": asdict(selected), "selection": "dev_only_ties_smallest_id",
                  "train_seeds": [9100], "dev_seeds": [9200], "model": asdict(self.model_spec),
                  "search": asdict(self.spec.search), "signature": self.signature}
        immutable(self.directory / "metrics/frozen_policy.json", frozen)
        return selected

    def execute(self):
        selected = self.learn()
        frozen_bytes = (self.directory / "metrics/frozen_policy.json").read_bytes()
        rows = []
        for i, seat in enumerate((0,1,0)):
            players = [None,None]
            players[seat] = selected
            row = self.game(f"test_{i+1}_p{seat}", 9300+i, players, "test")
            rows.append({"item_id": row["item_id"], "seed": 9300+i, "seat": seat,
                         "net_profit": row["raw_outcome"][seat], "policy_sha256": selected.sha256})
        if (self.directory / "metrics/frozen_policy.json").read_bytes() != frozen_bytes:
            raise ResumeConflict("Frozen policy changed during test")
        if len(self.game_ids) != 41 or self.call_id != 7:
            raise ResumeConflict("Incomplete fixed workflow")
        actual = {p.stem for p in (self.directory / "trajectories").glob("*.json")}
        if actual != self.game_ids:
            raise ResumeConflict("Unexpected trajectory artifacts")
        records = self.calls.journal._read_records().values()
        if {r["context"]["game_index"] for r in records} != set(range(1,8)):
            raise ResumeConflict("Unexpected logical call schedule")
        evaluation = {"games": rows, "total_net_profit": sum(r["net_profit"] for r in rows),
                      "affects_selection": False, "opponent": self.opponent}
        immutable(self.directory / "metrics/evaluation.json", evaluation)
        return evaluation


def phase_costs(journal, live):
    phases = {}
    for row in journal._read_records().values():
        stage = json.loads(row["prompt"])["true_stage"]
        if stage not in ("seed_code", "reflection", "idea", "code"):
            raise ResumeConflict("Unexpected model phase")
        cost = phases.setdefault(stage, {"physical_attempts": 0, "usage": {}, "unknown_usage_attempts": 0})
        for attempt in [*row["attempts"], row]:
            cost["physical_attempts"] += 1
            if not attempt["usage"]:
                cost["unknown_usage_attempts"] += 1
            for key, value in attempt["usage"].items():
                cost["usage"][key] = cost["usage"].get(key,0)+value
    return {"live_api": live, "usage_is_provider_reported": live, "phases": phases}


def _safe_directory(path, allowed):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(allowed):
        raise ValueError("Paths must stay within results/<game>/strategist_dev without symlinks")
    return path


def run(spec=LiveSpec(), *, model=None, model_spec=None, output_dir=None, resume=None,
        key_file=None, stop_after_games=None, progress=True, command=None):
    """Fixed 41 local games: 32 train, 6 dev, 3 frozen test; 7 logical calls.

    Models expose generate(str), last_usage and last_finish_reason. Caller owns
    injected models and must truthfully declare model_spec; no subclass shim.
    Failed runs are fail-closed. Paused/interrupted known-success runs can replay.
    """
    spec.validate()
    model_spec = model_spec or (ModelSpec.campus() if spec.live_api else ModelSpec())
    model_spec.validate(spec.live_api)
    if key_file is not None and not spec.live_api:
        raise ValueError("Offline never accepts a credential path")
    if stop_after_games is not None:
        integer(stop_after_games, 1, 41, "stop_after_games")
    owned = model is None
    if model is None:
        if model_spec not in (ModelSpec(), ModelSpec.campus()):
            raise ValueError("Nondefault model descriptor requires injected backend")
        model = CampusBackend(key_file) if spec.live_api else ScriptedModel()
    if not callable(getattr(model, "generate", None)):
        raise ValueError("Model must implement generate")
    if isinstance(model, ScriptedModel) and model_spec.kind != "scripted":
        raise ValueError("ScriptedModel cannot masquerade as live/transport")
    if model_spec.kind == "scripted" and not isinstance(model, ScriptedModel):
        raise ValueError("Custom offline backend must declare mock_transport provenance")
    if getattr(model, "max_retries", 0) != 0:
        raise ValueError("Backend retries must be disabled")
    settings = {}
    for name in ("model", "model_name", "temperature", "max_tokens", "cache_fingerprint"):
        if hasattr(model, name):
            settings[name] = getattr(model, name)
    canonical(settings)  # reject non-JSON/nonfinite backend settings, never inspect keys
    for name in ("temperature", "max_tokens"):
        if name in settings and settings[name] != getattr(model_spec, name):
            raise ValueError("Declared model settings differ from backend")
    opponent = opponent_config(spec.game, OPPONENTS[spec.game])
    sources = source_hashes()
    signature = {"schema": SCHEMA, "spec": asdict(spec), "model": asdict(model_spec),
                 "backend_class": type(model).__module__+"."+type(model).__qualname__,
                 "backend_source": hashlib.sha256(inspect.getsource(type(model)).encode()).hexdigest(),
                 "backend_settings": settings,
                 "sources": sources, "python": sys.version, "opponent": opponent,
                 "schedule": {"train": [9100], "dev": [9200], "test": [[9300,0],[9301,1],[9302,0]], "updates": 2}}
    signature = read_json_text(canonical(signature))
    fingerprint = digest(signature)
    allowed = (ROOT / "results" / spec.game / "strategist_dev").resolve()
    base = _safe_directory(output_dir or allowed, allowed)
    if resume:
        directory = _safe_directory(resume, allowed)
        if not directory.is_dir() or any(p.is_symlink() for p in directory.rglob("*")):
            raise ResumeConflict("Invalid resume directory")
        saved = unseal(read_json(directory / "config.json"))
        if saved["signature"] != signature or saved["signature_hash"] != fingerprint:
            raise ResumeConflict("Source/schema/model/opponent/config changed")
    else:
        base.mkdir(parents=True, exist_ok=True)
        from tempfile import mkdtemp
        label = "live" if spec.live_api else "offline"
        # Microseconds avoid collisions without overwriting any historical path.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = base / f"{stamp}_strategist_{label}_dev"
        try:
            directory.mkdir()
        except FileExistsError:
            directory = Path(mkdtemp(prefix=f"{stamp}_strategist_{label}_dev_", dir=base))
        for folder in ("prompts", "trajectories", "metrics", "logs", "figures"):
            (directory / folder).mkdir()
        immutable(directory / "config.json", {"schema": SCHEMA, "phase": "dev_debug",
            "signature": signature, "signature_hash": fingerprint, "live_api": spec.live_api,
            "scripted": model_spec.kind == "scripted", "dry_run": not spec.live_api,
            "created_at": datetime.now(timezone.utc).isoformat(), "command": command or [],
            "git_commit": _git(["rev-parse", "HEAD"]), "dirty_worktree": bool(_git(["status", "--porcelain"])),
            "approval": {"by": spec.approval_by, "explicit": spec.approved_live_api,
                         "max_physical_requests": spec.max_requests},
            "primary_metric": "three frozen-test games raw net profit; never used for selection",
            "exclusions": "No outcome filtering; any runtime/transport/truncation failure stops and is retained",
            "caveats": CAVEATS})
    with run_lock(directory):
        workflow, evaluation, start_requests = None, None, 0
        status, error_type = "failed", None
        try:
            # Audit journal before any replay, including calls outside the reached prefix.
            ledger = ModelCallJournal(model, directory / "prompts/ledger", spec.max_requests, max_retries=0)
            records = ledger._read_records()
            start_requests = ledger.stats()["requests"]
            if start_requests > spec.max_requests or any(r["status"] != "success" or r["attempts"] for r in records.values()):
                raise ResumeConflict("Failed/started calls require manual review, never resend")
            if (directory / "summary.json").exists():
                previous = unseal(read_json(directory / "summary.json"))
                if previous["status"] == "failed":
                    raise ResumeConflict("Failed workflow is not automatically repairable/resumable")
                if previous["status"] == "completed":
                    manifest = unseal(read_json(directory / "metrics/completion_manifest.json"))
                    if _artifact_hashes(directory) != manifest:
                        raise ResumeConflict("Completed evidence changed or disappeared")
                    if progress:
                        print(f"completed replay: zero new requests; output={directory}", flush=True)
                    return directory
            workflow = LiveWorkflow(spec, directory, fingerprint, model, model_spec, opponent, stop_after_games, progress)
            evaluation = workflow.execute()
            if source_hashes() != sources or opponent_config(spec.game, OPPONENTS[spec.game]) != opponent:
                raise ResumeConflict("Source or opponent changed during execution")
            status = "completed"
        except Paused:
            status = "paused"
        except KeyboardInterrupt:
            status, error_type = "interrupted", "KeyboardInterrupt"
            raise
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            if owned and isinstance(model, CampusBackend):
                model.close()
            # Never overwrite completed/failed evidence during a rejected resume.
            preserve = resume and (directory / "summary.json").exists() and unseal(read_json(directory / "summary.json"))["status"] in ("completed", "failed")
            if not preserve:
                journal = ModelCallJournal(model, directory / "prompts/ledger", spec.max_requests, max_retries=0)
                stats = journal.stats()
                costs = phase_costs(journal, spec.live_api)
                atomic(directory / "metrics/phase_costs.json", sealed(costs))
                rows = [unseal(read_json(p)) for p in (directory / "trajectories").glob("*.json")]
                summary = {"schema": SCHEMA, "phase": "dev_debug", "status": status, "error_type": error_type,
                    "last_context": workflow.context if workflow else {"phase": "resume_preflight"},
                    "live_api": spec.live_api, "model": asdict(model_spec), "calls": stats,
                    "new_physical_requests": stats["requests"]-start_requests,
                    "actual_api_requests": stats["requests"] if spec.live_api else 0,
                    "actual_api_usage": stats["usage_totals"] if spec.live_api else None,
                    "completed_by_phase": {phase: sum(r["status"] == "completed" and r["inputs"]["phase"] == phase for r in rows) for phase in ("train","dev","test")},
                    "new_games": workflow.new_games if workflow else 0, "skipped_games": workflow.skipped if workflow else 0,
                    "updates": len(workflow.improvements) if workflow else 0, "evaluation": evaluation,
                    "phase_costs": costs, "signature_hash": fingerprint, "caveats": CAVEATS}
                atomic(directory / "summary.json", sealed(summary))
                if status == "completed":
                    immutable(directory / "metrics/completion_manifest.json", _artifact_hashes(directory))
    return directory


def read_json_text(text):
    from methods.strategist.runner import strict_json
    return strict_json(text)


def _artifact_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*.json")) if p.name != "completion_manifest.json"}