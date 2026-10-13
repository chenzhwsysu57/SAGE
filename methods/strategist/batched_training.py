"""Training-only Strategist adaptation: local batches, paid high-level updates.

No external opponent or PK is scheduled. Archive opponents are frozen local
heuristics. This is NOT unrestricted program synthesis or exact paper code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import random
from pathlib import Path
import sys

from base.baseline_calls import ModelCallJournal, ResumeConflict, _usage
from base.types import Action
from methods.strategist.adapters import GAMES, make_game, visible
from methods.strategist.all_gto_workflow import UncappedCorrectingCalls
from methods.strategist.evolution import Idea, IdeaQueue, Strategy, canonical, digest, integer, key_feedback, vector
from methods.strategist.live_workflow import LiveWorkflow, ModelSpec, Paused, _safe_directory, phase_costs
from methods.strategist.nlp_backend import NLPPolicyBackend, ScriptedNLPModel
from methods.strategist.nlp_strategy import compile_policy, extract_policy, parse_policy
from methods.strategist.runner import atomic, immutable, outcome, read_json, run_lock, sealed, unseal, _git
from methods.strategist.search import SearchConfig, StrategistSearch
from methods.strategist.value import ValueHeuristic

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "strategist.batched_training.v1"
CAVEATS = [
    "Independent Strategist-inspired bounded NLP-weight adaptation, not exact paper reproduction.",
    "NLP prose is recorded; only five weights and bias affect the fixed local evaluator.",
    "Local search uses uniform hidden-state priors; no LLM action calls or external opponent training.",
    "Two seats share each deal seed: paired games are not independent observations.",
    "Fresh validation panels guide selection and are development data, NOT held-out test evidence.",
    "Eight validation games/update are noisy; strict positive paired mean gain is not a significance test.",
    "No outcome filtering or automatic resampling; errors stop and preserve evidence.",
    "No lifetime request ceiling; 1+3*updates logical calls, initial plus <=3 corrections per call.",
    "No network retry or random action fallback; interrupted in-flight requests are never resent.",
    "Completed replay validates recorded legal trajectories, not every internal MCTS simulation.",
]


@dataclass(frozen=True)
class TrainingSpec:
    game: str = "leduc"
    train_games: int = 1000
    update_every: int = 50
    validation_pairs: int = 2
    train_seed: int = 1000000
    validation_seed: int = 2000000
    planner_seed: int = 17
    format_retries: int = 3
    search: SearchConfig = field(default_factory=lambda: SearchConfig(32, 12, 128))
    live_api: bool = False
    approved_live_api: bool = False
    approval_by: str = ""

    @property
    def updates(self):
        return self.train_games // self.update_every

    @property
    def validation_games(self):
        return self.updates * self.validation_pairs * 4

    def validate(self):
        if self.game not in GAMES:
            raise ValueError("Unsupported game")
        for name, low, high in (("train_games", 2, 1000), ("update_every", 2, 1000),
                                ("validation_pairs", 1, 5), ("format_retries", 0, 3),
                                ("train_seed", 0, 2**31-10000),
                                ("validation_seed", 0, 2**31-10000),
                                ("planner_seed", 0, 2**31-1)):
            integer(getattr(self, name), low, high, name)
        if self.update_every % 2 or self.train_games % self.update_every or self.updates > 20:
            raise ValueError("Even seat-paired batches must divide training games; at most20 updates")
        if set(self.train_seeds()) & set(self.validation_seeds()):
            raise ValueError("Training and validation seeds overlap")
        if not isinstance(self.search, SearchConfig):
            raise ValueError("Expected SearchConfig")
        self.search.__post_init__()
        if self.search.simulations > 32 or self.search.max_depth > 12 or self.search.max_nodes > 128:
            raise ValueError("Search maximum32/12/128")
        if type(self.live_api) is not bool or type(self.approved_live_api) is not bool:
            raise ValueError("Boolean live flags required")
        if self.live_api and (not self.approved_live_api or not self.approval_by.strip()):
            raise ValueError("Explicit live approval required")
        if not self.live_api and (self.approved_live_api or self.approval_by):
            raise ValueError("Offline must not claim live approval")

    def train_seeds(self):
        return range(self.train_seed, self.train_seed + self.train_games // 2)

    def validation_seeds(self):
        return range(self.validation_seed, self.validation_seed + self.updates * self.validation_pairs)

    def protocol(self):
        data = asdict(self)
        for key in ("live_api", "approved_live_api", "approval_by"):
            data.pop(key)
        return data

    def plan(self):
        return {"train_games": self.train_games, "updates": self.updates,
                "update_every": self.update_every, "validation_games": self.validation_games,
                "total_local_games": self.train_games + self.validation_games,
                "logical_calls": 1 + 3*self.updates,
                "maximum_physical_attempts_from_finite_protocol": (1+3*self.updates)*(1+self.format_retries),
                "max_requests": None, "external_pk_games": 0,
                "train_seeds": list(self.train_seeds()), "validation_seeds": list(self.validation_seeds())}


def source_hashes():
    paths = set()
    for folder in ("base", "games", "methods/strategist"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.update((ROOT / "games").glob("*.txt"))
    for name in ("runs/strategist_train.py", "scripts/strategist-train.sh",
                 "apis/deepseek_journal_wrapper.py", "apis/hkustgz_wrapper.py"):
        paths.add(ROOT / name)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def evidence_hashes(directory):
    """Exclude mutable progress/resume receipts, but include physical IO and final policy."""
    paths = [directory / "config.json"]
    for folder in ("trajectories", "prompts", "metrics", "logs/network"):
        paths.extend((directory / folder).rglob("*.json"))
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.name != "completion_manifest.json"}


class TrainingCalls(UncappedCorrectingCalls):
    def __init__(self, model, directory, spec, signature):
        self.directory, self.game, self.format_retries = directory, spec.game, spec.format_retries
        directory.mkdir(parents=True, exist_ok=True)
        immutable(directory / "config.json", {"schema": SCHEMA,
            "phase": "real_measurement" if spec.live_api else "dev_debug",
            "max_requests": None, "network_retries": 0, "format_retries": spec.format_retries,
            "identity": signature})
        self.journal = ModelCallJournal(model, directory, 0, max_retries=0)


class TrainingFixture(ScriptedNLPModel):
    """The old fixture exceeds the weight contract after iteration16; bounded here."""
    cache_fingerprint = "strategist-batched-training-fixture-v1"

    def generate(self, prompt):
        import json
        request = json.loads(prompt)
        request["iteration"] = request.get("iteration", 0) % 3
        return super().generate(canonical(request))


class TrainingWorkflow(LiveWorkflow):
    """Reuse validated stage calls and immutable policies, not the fixed41 schedule."""
    heuristic_type = ValueHeuristic

    def __init__(self, spec, directory, signature, model, model_spec, stop_after_games=None, progress=True):
        self.spec, self.directory, self.signature = spec, directory, signature
        self.model_spec = model_spec
        self.stop_after_games, self.progress = stop_after_games, progress
        self.strategies, self.rounds, self.improvements = [], [], []
        self.queue, self.game_ids = IdeaQueue(), set()
        self.call_id = self.new_games = self.skipped = 0
        self.counts = {"train": 0, "dev": 0}
        self.context = {"phase": "setup"}
        self.calls = TrainingCalls(model, directory / "prompts/ledger", spec, signature)

    def snapshot(self):
        # Immutable per-update receipts replace mutable lineage snapshots on replay.
        pass

    def call(self, stage, payload):
        if source_hashes() != self.signature["sources"]:
            raise ResumeConflict("Source changed before a strategy call")
        return super().call(stage, payload)

    def game(self, item, seed, players, phase):
        permitted = self.spec.train_seeds() if phase == "train" else self.spec.validation_seeds()
        if phase not in self.counts or seed not in permitted or any(s is None for s in players):
            raise ValueError("Only scheduled local train/dev games permitted")
        cap = self.spec.train_games if phase == "train" else self.spec.validation_games
        if item in self.game_ids or self.counts[phase] >= cap:
            raise ResumeConflict("Unexpected game schedule")
        self.game_ids.add(item)
        inputs = {"item": item, "seed": seed, "phase": phase,
                  "players": [s.id for s in players], "hashes": [s.sha256 for s in players],
                  "signature": digest(self.signature)}
        path = self.directory / "trajectories" / f"{item}.json"
        old = unseal(read_json(path)) if path.exists() else None
        if old and old["inputs"] != inputs:
            raise ResumeConflict("Saved game inputs changed")
        if old and old["status"] != "completed":
            raise ResumeConflict("Partial game requires manual review, not overwritten")
        if not old and self.stop_after_games is not None and self.new_games >= self.stop_after_games:
            raise Paused("Safe local game boundary")
        self.context = {"phase": phase, "item": item, "seed": seed}
        row = {"item_id": item, "inputs": inputs, "players": inputs["players"], "status": "started", "steps": []}
        if not old:
            atomic(path, sealed(row))
        engine = make_game(self.spec.game, seed)
        state = engine.reset()
        planners = [StrategistSearch(self.spec.game, self.heuristic_type(s.source), self.spec.search,
                                    self.spec.planner_seed) for s in players]
        while not engine.is_terminal(state):
            step = len(row["steps"])
            if step >= 64:
                raise ValueError("Game exceeded64 actions; no fallback")
            player = engine.get_current_player(state)
            obs = engine.get_observation(state, player)
            if old:
                saved = old["steps"][step]
                action, trace = Action(**saved["action"]), saved["trace"]
                if (trace["root_features"] != {"game": self.spec.game, **visible(obs)}
                        or trace["simulations"] != self.spec.search.simulations
                        or not 1 <= trace["nodes"] <= self.spec.search.max_nodes
                        or trace["max_depth_reached"] > self.spec.search.max_depth
                        or sum(a["visits"] for a in trace["actions"]) != self.spec.search.simulations):
                    raise ResumeConflict("Cached search trace mismatch")
                vector(trace["root_initial_heuristic"]["values"])
                vector(trace["root_search_value"])
            else:
                result = planners[player].decide(obs)
                action, trace = result.action, result.trace
            if action not in engine.get_legal_actions(state):
                raise ResumeConflict("Illegal action")
            row["steps"].append({"player_id": player, "observation": visible(obs),
                                 "action": action.model_dump(), "trace": trace})
            state = engine.step(state, action)
        raw, normalized = outcome(self.spec.game, engine, state)
        row.update(status="completed", raw_outcome=raw, normalized_outcome=normalized)
        if old:
            if old != row:
                raise ResumeConflict("Trajectory replay mismatch")
            self.skipped += 1
        else:
            atomic(path, sealed(row))
            self.new_games += 1
        self.counts[phase] += 1
        if self.progress:
            print(f"Game completed game={self.spec.game} phase={phase} count={self.counts[phase]}/{cap} "
                  f"seed={seed} players={inputs['players']} net={raw} replay={bool(old)} run={self.directory}", flush=True)
        return row

    def execute(self):
        initial = self.generate("s000", self.call("seed_code", {"instruction": "Generate initial policy."}))
        champion = initial
        for iteration in range(1, self.spec.updates + 1):
            pool = list(self.strategies)
            rng = random.Random(self.spec.planner_seed + iteration)
            rng.shuffle(pool)
            rows, rewards = [], []
            for pair in range(self.spec.update_every // 2):
                opponent = pool[pair % len(pool)]
                seed = self.spec.train_seed + (iteration-1)*(self.spec.update_every//2) + pair
                for seat in (0, 1):
                    players = [opponent, opponent]
                    players[seat] = champion
                    row = self.game(f"train_{iteration:03d}_{pair:03d}_p{seat}", seed, players, "train")
                    rows.append(row)
                    rewards.append(row["normalized_outcome"][seat])
            training_score = sum(rewards)/len(rewards)
            reflection = self.call("reflection", {"iteration": iteration,
                "parent": {"id": champion.id, "source": champion.source, "train_score": training_score},
                "batch": {"games": len(rows), "completed_train_games": self.counts["train"],
                          "opponents": [s.id for s in pool]},
                "feedback": key_feedback(rows, champion.id, limit=3),
                "instruction": "Reflect on this completed training batch only; <=12000 characters."})
            idea_text = self.call("idea", {"iteration": iteration, "parent_id": champion.id,
                "reflection": reflection, "instruction": "Propose one bounded policy improvement; <=4000 characters."})
            self.queue.add(Idea(f"i{iteration:03d}", champion.id, idea_text))
            chosen = self.queue.select(self.spec.planner_seed + iteration, {champion.id})
            candidate = self.generate(f"s{iteration:03d}", self.call("code", {"iteration": iteration,
                "parent_source": champion.source, "idea": asdict(chosen),
                "instruction": "Implement the selected policy improvement."}), champion.id, chosen.id)
            scores = {"parent": [], "candidate": []}
            validation_ids = []
            # A fresh panel per update; candidate/parent use identical deals, opponents, seats.
            for pair in range(self.spec.validation_pairs):
                opponent = pool[pair % len(pool)]
                seed = self.spec.validation_seed + (iteration-1)*self.spec.validation_pairs + pair
                for label, policy in (("parent", champion), ("candidate", candidate)):
                    for seat in (0, 1):
                        players = [opponent, opponent]
                        players[seat] = policy
                        item = f"dev_{iteration:03d}_{pair:03d}_{label}_p{seat}"
                        row = self.game(item, seed, players, "dev")
                        validation_ids.append(item)
                        scores[label].append(row["normalized_outcome"][seat])
            means = {k: sum(v)/len(v) for k, v in scores.items()}
            accepted = means["candidate"] > means["parent"]
            self.queue.observe(chosen.id, means["candidate"], means["parent"])
            record = {"iteration": iteration, "train_completed": self.counts["train"],
                      "parent": champion.id, "candidate": candidate.id, "accepted": accepted,
                      "next_policy": candidate.id if accepted else champion.id,
                      "train_mean_normalized": training_score, "validation_means": means,
                      "paired_deltas": [c-p for c, p in zip(scores["candidate"], scores["parent"])],
                      "validation_ids": validation_ids, "training_ids": [r["item_id"] for r in rows],
                      "logical_calls": self.call_id, "queue": self.queue.snapshot()}
            immutable(self.directory / "metrics/updates" / f"{iteration:03d}.json", record)
            self.improvements.append(record)
            champion = candidate if accepted else champion
            if self.progress:
                print(f"Update completed game={self.spec.game} update={iteration}/{self.spec.updates} "
                      f"accepted={accepted} next={champion.id} run={self.directory}", flush=True)
        if self.counts != {"train": self.spec.train_games, "dev": self.spec.validation_games} or self.call_id != 1+3*self.spec.updates:
            raise ResumeConflict("Incomplete training protocol")
        policy = {"schema": SCHEMA, "game": self.spec.game, "strategy": asdict(champion),
                  "nlp_policy": extract_policy(champion.source), "protocol": self.spec.protocol(),
                  "signature": digest(self.signature), "train_seeds": list(self.spec.train_seeds()),
                  "validation_seeds": list(self.spec.validation_seeds()), "external_pk_games": 0,
                  "provenance": "personal_api" if self.spec.live_api else "scripted_fixture",
                  "selection": "sequential strictly-positive paired validation gain; ties retain parent"}
        immutable(self.directory / "metrics/frozen_policy.json", policy)
        return policy


def load_training_policy(path, game, evaluation_seeds):
    """Future PK consumers must explicitly load this artifact, not rerun learning."""
    data = unseal(read_json(Path(path)))
    if data["schema"] != SCHEMA or data["game"] != game:
        raise ResumeConflict("Policy schema/game mismatch")
    policy = parse_policy(canonical(data["nlp_policy"]), game)
    strategy = Strategy.create(data["strategy"]["id"], compile_policy(policy))
    if strategy.sha256 != data["strategy"]["sha256"] or strategy.source != data["strategy"]["source"]:
        raise ResumeConflict("Frozen NLP/source/hash mismatch")
    if set(evaluation_seeds) & (set(data["train_seeds"]) | set(data["validation_seeds"])):
        raise ResumeConflict("PK seeds overlap training or validation")
    return strategy


def validate_dev_evidence(spec, evidence):
    directory = Path(evidence).absolute()
    allowed = ROOT / "results" / spec.game / "strategist_training"
    _safe_directory(directory, allowed)
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ResumeConflict("Symlink evidence")
    config = unseal(read_json(directory / "config.json"))
    summary = unseal(read_json(directory / "summary.json"))
    if (config["phase"] != "dev_debug" or config["live_api"]
            or config["signature"]["protocol"] != spec.protocol()
            or config["signature"]["sources"] != source_hashes()
            or summary["status"] != "completed"
            or summary["completed_by_phase"] != {"train": spec.train_games, "dev": spec.validation_games}
            or summary["updates_completed"] != spec.updates):
        raise ResumeConflict("Need completed same-source same-protocol offline training evidence")
    if unseal(read_json(directory / "metrics/completion_manifest.json")) != evidence_hashes(directory):
        raise ResumeConflict("Offline evidence changed")
    phases = {"train": 0, "dev": 0}
    for path in (directory / "trajectories").glob("*.json"):
        row = unseal(read_json(path))
        if row["status"] != "completed" or row["inputs"]["phase"] not in phases:
            raise ResumeConflict("Incomplete offline trajectory")
        phases[row["inputs"]["phase"]] += 1
    if phases != summary["completed_by_phase"] or len(list((directory / "metrics/updates").glob("*.json"))) != spec.updates:
        raise ResumeConflict("Offline evidence lacks actual games or updates")
    return str(directory)


def audit_usage(directory):
    """Supplement the existing physical payload/body audit with cost parity."""
    import json
    logical = sorted((directory / "prompts/ledger/prompts").rglob("*.json"))
    physical = sorted((directory / "logs/network").glob("request_*.json"))
    if len(logical) != len(physical):
        raise ResumeConflict("Physical/semantic accounting count mismatch")
    for left, right in zip(logical, physical):
        row, network = read_json(left), read_json(right)
        body = json.loads(network["response_body"])
        if _usage(body.get("usage")) != row["usage"]:
            raise ResumeConflict("Physical/semantic usage mismatch")


def run_training(spec, *, output_dir=None, resume=None, stop_after_games=None, progress=True,
                 key_file=None, approved_real_measurement=False, dev_evidence=None, command=None):
    """Replay only explicit safe pauses/completions; preserve failed or partial runs."""
    spec.validate()
    if stop_after_games is not None:
        integer(stop_after_games, 1, spec.train_games+spec.validation_games, "stop_after_games")
    if not spec.live_api and (key_file is not None or approved_real_measurement or dev_evidence is not None):
        raise ValueError("Offline must not read keys or claim live approval")
    evidence = None
    if spec.live_api:
        if not approved_real_measurement or not dev_evidence:
            raise ValueError("Long training requires reviewed dev evidence and real_measurement approval")
        evidence = validate_dev_evidence(spec, dev_evidence)
        if _git(["status", "--porcelain"]) != "":
            raise ResumeConflict("Real measurement requires clean worktree; no automatic staging/commit")
    from methods.strategist.personal_backend import PersonalModelSpec, PersonalStrategistBackend
    raw = PersonalStrategistBackend(key_file or ROOT / ".deepseek.key") if spec.live_api else TrainingFixture()
    model = NLPPolicyBackend(raw)
    model_spec = PersonalModelSpec() if spec.live_api else ModelSpec(name="strategist-batched-fixture-v1")
    model_spec.validate(spec.live_api)
    sources = source_hashes()
    signature = {"schema": SCHEMA, "protocol": spec.protocol(), "sources": sources,
                 "model": asdict(model_spec), "live_api": spec.live_api, "python": sys.version,
                 "backend": raw.cache_fingerprint}
    allowed = ROOT / "results" / spec.game / "strategist_training"
    base = _safe_directory(output_dir or allowed, allowed)
    if resume:
        directory = _safe_directory(resume, allowed)
        if any(p.is_symlink() for p in directory.rglob("*")):
            raise ResumeConflict("Symlink in resumed evidence")
        if unseal(read_json(directory / "config.json"))["signature"] != signature:
            raise ResumeConflict("Source/config/model changed; cannot resume")
    else:
        from tempfile import mkdtemp
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = Path(mkdtemp(prefix=f"{stamp}_batched_training_{'live' if spec.live_api else 'dev'}_", dir=base))
        immutable(directory / "config.json", {"schema": SCHEMA,
            "phase": "real_measurement" if spec.live_api else "dev_debug",
            "live_api": spec.live_api, "dry_run": not spec.live_api, "signature": signature,
            "plan": spec.plan(), "command": command or [], "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]), "dirty_worktree": bool(_git(["status", "--porcelain"])),
            "approval": {"by": spec.approval_by, "approved": approved_real_measurement,
                         "dev_evidence": evidence, "time": datetime.now(timezone.utc).isoformat()},
            "primary_metrics": ["completed legal train games", "completed updates", "paired normalized validation gain", "provider usage"],
            "exclusions": "none; errors stop without automatic replacement", "caveats": CAVEATS})
    print(f"Training run: {directory}", flush=True)
    try:
        with run_lock(directory):
            previous = unseal(read_json(directory / "summary.json")) if (directory / "summary.json").exists() else None
            # All rejection checks precede bind/key access and do not replace existing summaries.
            if previous and previous["status"] not in ("paused", "completed"):
                raise ResumeConflict("Failed/interrupted runs need manual review, no automatic resume")
            if previous and previous["status"] == "completed":
                if unseal(read_json(directory / "metrics/completion_manifest.json")) != evidence_hashes(directory):
                    raise ResumeConflict("Completed evidence changed")
            ledger = ModelCallJournal(model, directory / "prompts/ledger", 0, max_retries=0)
            if any(r["status"] != "success" or r["attempts"] for r in ledger._read_records().values()):
                raise ResumeConflict("Failed/ambiguous calls cannot be resent")
            for path in (directory / "trajectories").glob("*.json"):
                if unseal(read_json(path))["status"] != "completed":
                    raise ResumeConflict("Partial local game requires manual review")
            start_requests = ledger.stats()["requests"]
            workflow = None
            status, error, frozen = "completed", None, None
            try:
                model.bind_run_directory(directory)
                workflow = TrainingWorkflow(spec, directory, signature, model, model_spec, stop_after_games, progress)
                frozen = workflow.execute()
                if sources != source_hashes():
                    raise ResumeConflict("Source changed during training")
                if spec.live_api:
                    raw.audit_physical(directory)
                    audit_usage(directory)
            except Paused:
                status = "paused"
            except KeyboardInterrupt:
                status, error = "interrupted", "KeyboardInterrupt"
            except Exception as exc:
                status, error = "failed", type(exc).__name__
            journal = workflow.calls.journal if workflow else ledger
            summary = {"schema": SCHEMA, "status": status, "error_type": error,
                "context": workflow.context if workflow else {"phase": "bind_or_init"},
                "completed_by_phase": workflow.counts if workflow else {"train": 0, "dev": 0},
                "updates_completed": len(workflow.improvements) if workflow else 0,
                "logical_calls": workflow.call_id if workflow else 0, "calls": journal.stats(),
                "new_requests": journal.stats()["requests"]-start_requests,
                "new_games": workflow.new_games if workflow else 0, "skipped_games": workflow.skipped if workflow else 0,
                "phase_costs": phase_costs(journal, spec.live_api),
                "external_pk_games": 0, "frozen_policy": "metrics/frozen_policy.json" if frozen else None,
                "ready_for_pk": status == "completed" and spec.live_api,
                "finished_at": datetime.now(timezone.utc).isoformat()}
            if previous and previous["status"] == "completed":
                if status != "completed" or summary["new_requests"] or summary["new_games"]:
                    raise ResumeConflict("Completed replay must make zero new requests/games")
                if unseal(read_json(directory / "metrics/completion_manifest.json")) != evidence_hashes(directory):
                    raise ResumeConflict("Completed replay changed evidence")
            else:
                if status == "completed":
                    try:
                        immutable(directory / "metrics/completion_manifest.json", evidence_hashes(directory))
                    except Exception as exc:
                        summary.update(status="failed", error_type=type(exc).__name__, ready_for_pk=False,
                                       context={"phase": "completion_manifest"})
                        status = "failed"
                atomic(directory / "summary.json", sealed(summary))
            if resume:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                immutable(directory / "logs/resume" / f"{stamp}.json", {"original_directory": str(directory),
                    "time": stamp, "completed": sum(summary["completed_by_phase"].values()), "skipped": summary["skipped_games"],
                    "failed": int(status == "failed"),
                    "remaining": spec.train_games+spec.validation_games-sum(summary["completed_by_phase"].values()),
                    "new_requests": summary["new_requests"]})
            return directory, summary
    finally:
        model.close()