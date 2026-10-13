"""Bounded offline Strategist experiment orchestration (independent reimplementation).

No live backend is available. Train uses evolving population self-play, dev uses
a frozen initial opponent, and test compares frozen selected/initial policies.
All low-level decisions are local PUCT. Resume replays the deterministic high-
level computation, caches calls and validates/skips completed games. An offline
started game may restart from its seed: games contain no model calls. Ambiguous
model calls NEVER restart. Checksums detect accidental corruption, not an attacker
able to rewrite both an artifact and all its integrity metadata.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile

from base.baseline_calls import BudgetExceeded, ModelCallJournal, ResumeConflict
from base.types import Action
from methods.strategist.adapters import GAMES, PAYOFF_BOUNDS, PAYOFF_SCALES, make_game, rules, visible
from methods.strategist.evolution import (
    Idea, IdeaQueue, ScriptedModel, Strategy, canonical, digest, integer,
    key_feedback, seed_list, vector,
)
from methods.strategist.search import SearchConfig, StrategistSearch
from methods.strategist.value import ValueHeuristic


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "strategist.offline.v1"
CAVEATS = [
    "Independent paper-inspired adaptation; not copied upstream or exact Appendix F reproduction.",
    "Scripted mock outputs test the workflow; they are not real LLM learning or efficacy evidence.",
    "PUCT uses observation-only uniform hidden priors, not CFR, posterior inference or equilibrium certification.",
    "Values are fixed P0/P1 normalized zero-sum returns: Leduc /32, dice /1, Goofspiel /15.",
    "Public-only key feedback omits private inputs and free-form explanations that could disclose them.",
    "Tiny reused training seeds and population selection can overfit; dev selection also has selection bias.",
    "Completed games are immutable; offline started games restart safely, ambiguous model calls are rejected.",
    "Checksums provide consistency, not authenticated protection against coordinated malicious rewriting.",
    "Live API deliberately unavailable until separate evidence review and approval.",
]


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path: Path):
    return strict_json(path.read_text(encoding="utf-8"))


def sealed(payload: dict) -> dict:
    return {"payload": payload, "sha256": digest(payload)}


def unseal(record: object) -> dict:
    if (type(record) is not dict or set(record) != {"payload", "sha256"}
            or type(record["payload"]) is not dict or record["sha256"] != digest(record["payload"])):
        raise ResumeConflict("Artifact integrity mismatch")
    return record["payload"]


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".strategist-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def immutable(path: Path, value: dict) -> None:
    if path.exists():
        if unseal(read_json(path)) != value:
            raise ResumeConflict("Immutable artifact changed")
    else:
        atomic(path, sealed(value))


def implementation_hash() -> str:
    paths = [ROOT / "runs/strategist_bench.py"]
    for folder in ("methods/strategist", "base", "games"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "games").glob("prize_pool*.txt"))
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(set(paths))})


@dataclass(frozen=True)
class RunSpec:
    game: str = "leduc"
    mode: str = "full"
    train_seeds: tuple[int, ...] = (9100,)
    dev_seeds: tuple[int, ...] = (9200,)
    eval_seeds: tuple[int, ...] = (9300,)
    implementations: int = 2
    pool_size: int = 3
    max_requests: int = 16
    max_games: int = 128
    search: SearchConfig = field(default_factory=lambda: SearchConfig(4, 3, 16))
    planner_seed: int = 17
    evaluation_opponent: str = "initial"

    def validate(self) -> None:
        if self.game not in GAMES or self.mode not in ("full", "learn", "evaluate"):
            raise ValueError("Unsupported game or mode")
        groups = [seed_list(getattr(self, name), name) for name in ("train_seeds", "dev_seeds", "eval_seeds")]
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
            raise ValueError("train/dev/test seeds must be disjoint")
        for name, low, high in (("implementations", 2, 4), ("pool_size", 2, 4),
                                ("max_requests", 0, 64), ("max_games", 1, 512),
                                ("planner_seed", 0, 2**31-1)):
            integer(getattr(self, name), low, high, name)
        if not isinstance(self.search, SearchConfig):
            raise ValueError("Expected SearchConfig")
        self.search.__post_init__()
        # These runner caps deliberately tighter than the general search API.
        if self.search.simulations > 32 or self.search.max_depth > 12 or self.search.max_nodes > 128:
            raise ValueError("Offline runner search cap: simulations32/depth12/nodes128")
        if self.evaluation_opponent not in ("initial", "uniform"):
            raise ValueError("Only initial or local uniform evaluation opponent supported")
        if self.planned_games() > self.max_games:
            raise ValueError("Declared workflow exceeds max_games; reduce schedule or raise bounded cap")

    def planned_games(self) -> int:
        if self.mode == "evaluate":
            return 4 * len(self.eval_seeds)
        t, d = len(self.train_seeds), len(self.dev_seeds)
        count = 2*t + 2*d
        for i in range(1, self.implementations + 1):
            count += 4*min(self.pool_size, i)*t + (i+1)*(i+2)*t + 2*d
        return count + (4*len(self.eval_seeds) if self.mode == "full" else 0)

    def payload(self) -> dict:
        return strict_json(canonical(asdict(self)))


POLICY_FIELDS = {"schema", "game", "source", "sha256", "initial", "train_seeds", "dev_seeds",
                 "search_config", "planner_seed", "provenance", "caveats", "selection"}


def load_policy(path: Path, game: str, eval_seeds: tuple[int, ...]) -> dict:
    """Strict frozen schema/hash validation BEFORE allocating an output run."""
    policy = unseal(read_json(path))
    if set(policy) != POLICY_FIELDS or policy["schema"] != SCHEMA or policy["game"] != game:
        raise ValueError("Policy schema/game mismatch")
    seeds = [seed_list(policy[k], k) for k in ("train_seeds", "dev_seeds")]
    evaluation = seed_list(eval_seeds, "eval_seeds")
    if set(seeds[0]) & set(seeds[1]) or set(evaluation) & (set(seeds[0]) | set(seeds[1])):
        raise ValueError("Evaluation seeds overlap policy training/development")
    if type(policy["initial"]) is not dict or set(policy["initial"]) != {"source", "sha256"}:
        raise ValueError("Invalid initial policy schema")
    for row in (policy, policy["initial"]):
        heuristic = ValueHeuristic(row["source"])
        if heuristic.source != row["source"] or heuristic.sha256 != row["sha256"]:
            raise ValueError("Policy source/hash mismatch")
        engine = make_game(game, 0)
        heuristic.evaluate({"game": game, **visible(engine.get_observation(engine.reset(), 0))})
    if policy["provenance"] != ScriptedModel.provenance:
        raise ValueError("This runner accepts only explicitly scripted/mock policies")
    if (type(policy["caveats"]) is not list or not policy["caveats"]
            or any(type(c) is not str for c in policy["caveats"])):
        raise ValueError("Policy caveats missing")
    selection = policy["selection"]
    if (type(selection) is not dict or set(selection) != {"criterion", "strategy_id", "dev_score"}
            or selection["criterion"] != "dev_only" or type(selection["strategy_id"]) is not str
            or type(selection["dev_score"]) not in (int, float)
            or not math.isfinite(selection["dev_score"]) or abs(selection["dev_score"]) > 1):
        raise ValueError("Invalid policy selection provenance")
    if type(policy["search_config"]) is not dict or set(policy["search_config"]) != set(asdict(SearchConfig())):
        raise ValueError("Invalid policy search schema")
    SearchConfig(**policy["search_config"])
    integer(policy["planner_seed"], 0, 2**31-1, "policy planner seed")
    return policy


@contextmanager
def run_lock(directory: Path):
    """Separate nonblocking whole-run lock; journal's directory lock is nested."""
    fd = os.open(directory / ".run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ResumeConflict("Run already owned by another process") from None
        yield
    finally:
        os.close(fd)


class InterruptedRun(RuntimeError):
    """An intentional offline game-boundary stop, safely resumable."""


def outcome(game: str, engine, state) -> tuple[list[float], list[float]]:
    raw = ([engine.get_net_profit(state, p) for p in (0, 1)] if game == "leduc"
           else [engine.get_rewards(state)[p] for p in (0, 1)])
    if any(not math.isfinite(v) or abs(v) > PAYOFF_BOUNDS[game] for v in raw):
        raise ValueError("Invalid engine outcome")
    return raw, vector([v / PAYOFF_SCALES[game] for v in raw])


class Workflow:
    def __init__(self, spec: RunSpec, directory: Path, config_hash: str,
                 model: ScriptedModel, policy: dict | None, stop_after_games: int | None):
        self.spec, self.directory, self.config_hash = spec, directory, config_hash
        self.model, self.policy, self.stop_after_games = model, policy, stop_after_games
        self.journal = ModelCallJournal(model, directory, spec.max_requests, max_retries=0)
        # Inspect EVERY journal item, including ones not reached by replay.
        records = self.journal._read_records()
        for path, row in records.items():
            if row["status"] == "started":
                raise ResumeConflict("Ambiguous model call; review required")
            if (row["stage"] != "decision" or row["context"]["game"] != spec.game
                    or row["context"]["player_id"] != 0 or row["context"]["game_index"] != 0
                    or row["attempts"] or row["prompt_hash"] != hashlib.sha256(row["prompt"].encode()).hexdigest()):
                raise ResumeConflict("Unexpected journal identity or prompt integrity")
            request = strict_json(row["prompt"])
            step = row["context"]["step"]
            expected_stage = "seed_code" if step == 1 else ("reflection", "idea", "code")[(step-2) % 3]
            if (not 1 <= step <= 1 + 3*spec.implementations or request.get("true_stage") != expected_stage
                    or request.get("config_hash") != config_hash or spec.mode == "evaluate"):
                raise ResumeConflict("Unknown or altered journal entry")
            side = directory / "metrics/calls" / f"{step:04d}.json"
            if side.exists():
                meta = unseal(read_json(side))
                if meta != self.call_metadata(step, expected_stage, row["prompt"], row["response"]):
                    raise ResumeConflict("Model call side metadata mismatch")
        if len(records) > spec.max_requests:
            raise ResumeConflict("Persisted calls exceed request cap")
        self.call_step = 0
        self.strategies: list[Strategy] = []
        self.queue = IdeaQueue()
        self.game_ids: set[str] = set()
        self.rounds: list[dict] = []
        self.improvements: list[dict] = []
        self.rejections: list[dict] = []
        self.skipped = self.new_games = self.restarted = self.evolutions = 0

    def call_metadata(self, step, stage, prompt, response):
        return {"step": step, "true_stage": stage, "journal_stage": "decision",
                "config_hash": self.config_hash, "prompt_sha256": digest(prompt),
                "response_sha256": digest(response), "provenance": ScriptedModel.provenance}

    def call(self, stage: str, payload: dict) -> str:
        self.call_step += 1
        request = {"schema": SCHEMA, "true_stage": stage, "config_hash": self.config_hash,
                   "game": self.spec.game, "mock_only": True, **payload}
        prompt = canonical(request)
        self.journal.set_context(game=self.spec.game, player_id=0, game_index=0, step=self.call_step)
        response = self.journal.generate_stage("decision", prompt)
        immutable(self.directory / "metrics/calls" / f"{self.call_step:04d}.json",
                  self.call_metadata(self.call_step, stage, prompt, response))
        if len(response) > 12000:
            raise ValueError("Generated response too long; original preserved in journal")
        return response

    def snapshot(self) -> None:
        atomic(self.directory / "metrics/evolution.json", sealed({
            "schema": SCHEMA, "strategies": [asdict(s) for s in self.strategies],
            "idea_queue": self.queue.snapshot(), "rounds": self.rounds,
            "improvements": self.improvements, "rejections": self.rejections,
            "evolutions": self.evolutions, "config_hash": self.config_hash}))

    def generated(self, identity: str, source: str, parent=None, idea=None) -> Strategy:
        try:
            strategy = Strategy.create(identity, source, parent, idea)
            engine = make_game(self.spec.game, 0)
            ValueHeuristic(strategy.source).evaluate(
                {"game": self.spec.game, **visible(engine.get_observation(engine.reset(), 0))})
        except Exception as exc:
            self.rejections.append({"strategy_id": identity, "parent_id": parent, "idea_id": idea,
                                    "source": source, "status": "failed", "error_type": type(exc).__name__})
            self.snapshot()
            raise
        self.strategies.append(strategy)
        self.snapshot()
        return strategy

    def validate_game(self, row: dict, inputs: dict) -> None:
        if (set(row) != {"schema", "status", "inputs", "item_id", "players", "steps", "raw_outcome", "normalized_outcome"}
                or row["schema"] != SCHEMA or row["status"] != "completed" or row["inputs"] != inputs
                or row["item_id"] != inputs["item_id"] or row["players"] != inputs["players"]
                or type(row["steps"]) is not list or not 1 <= len(row["steps"]) <= 64):
            raise ResumeConflict("Completed game identity/schema mismatch")
        engine = make_game(self.spec.game, inputs["seed"])
        state = engine.reset()
        for step in row["steps"]:
            if engine.is_terminal(state) or set(step) != {"player_id", "observation", "action", "trace"}:
                raise ResumeConflict("Invalid saved game step")
            player = engine.get_current_player(state)
            obs = engine.get_observation(state, player)
            action = Action(**step["action"])
            if (step["player_id"] != player or step["observation"] != visible(obs)
                    or action not in engine.get_legal_actions(state)):
                raise ResumeConflict("Saved observation/action mismatch")
            trace = step["trace"]
            if inputs["hashes"][player] is None:
                if trace is not None:
                    raise ResumeConflict("Uniform opponent cannot have a search trace")
            else:
                cfg = self.spec.search
                if (type(trace) is not dict or trace["root_features"] != {"game": self.spec.game, **visible(obs)}
                        or trace["simulations"] != cfg.simulations
                        or not 1 <= trace["nodes"] <= cfg.max_nodes
                        or not 1 <= trace["transitions"] <= cfg.simulations*cfg.max_depth
                        or trace["max_depth_reached"] > cfg.max_depth
                        or sum(a["visits"] for a in trace["actions"]) != cfg.simulations):
                    raise ResumeConflict("Saved search trace mismatch")
                vector(trace["root_initial_heuristic"]["values"])
                vector(trace["root_search_value"])
            state = engine.step(state, action)
        if not engine.is_terminal(state):
            raise ResumeConflict("Completed game is not terminal")
        raw, normalized = outcome(self.spec.game, engine, state)
        if raw != row["raw_outcome"] or normalized != row["normalized_outcome"]:
            raise ResumeConflict("Saved engine outcome mismatch")

    def game(self, item_id: str, seed: int, players: list[Strategy | None], phase: str) -> dict:
        if item_id in self.game_ids:
            raise ValueError("Duplicate schedule identity")
        self.game_ids.add(item_id)
        if len(self.game_ids) > self.spec.max_games:
            raise BudgetExceeded("Local game cap exhausted")
        inputs = {"item_id": item_id, "seed": seed, "phase": phase,
                  "players": [p.id if p else "uniform" for p in players],
                  "hashes": [p.sha256 if p else None for p in players],
                  "search": asdict(self.spec.search), "config_hash": self.config_hash,
                  "planner_seed": self.spec.planner_seed}
        path = self.directory / "trajectories" / f"{item_id}.json"
        old = unseal(read_json(path)) if path.exists() else None
        if old is not None:
            if old.get("inputs") != inputs:
                raise ResumeConflict("Game input fingerprint changed")
            if old.get("status") == "completed":
                self.validate_game(old, inputs)
                self.skipped += 1
                return old
            if old.get("status") != "started":
                raise ResumeConflict("Unknown game state")
        if self.stop_after_games is not None and self.new_games >= self.stop_after_games:
            raise InterruptedRun("Intentional offline game-boundary pause")
        if old is not None:
            self.restarted += 1
        row = {"schema": SCHEMA, "status": "started", "inputs": inputs, "item_id": item_id,
               "players": inputs["players"], "steps": []}
        atomic(path, sealed(row))
        engine = make_game(self.spec.game, seed)
        state = engine.reset()
        # SAME search/planner settings for every policy, baseline and opponent;
        # planner randomness never gets the environment seed or policy identity.
        planners = [StrategistSearch(self.spec.game, ValueHeuristic(p.source), self.spec.search,
                                    self.spec.planner_seed) if p else None for p in players]
        uniform_rng = random.Random(int(digest([self.spec.planner_seed, seed, "uniform"]), 16))
        while not engine.is_terminal(state):
            if len(row["steps"]) >= 64:
                raise BudgetExceeded("Local transition cap exhausted")
            player = engine.get_current_player(state)
            obs = engine.get_observation(state, player)
            if planners[player] is None:
                action, trace = uniform_rng.choice(engine.get_legal_actions(state)), None
            else:
                decision = planners[player].decide(obs)
                action, trace = decision.action, decision.trace
            if action not in engine.get_legal_actions(state):
                raise ValueError("Planner returned illegal real action")
            row["steps"].append({"player_id": player, "observation": visible(obs),
                                 "action": action.model_dump(), "trace": trace})
            # Goofspiel engine receives exactly P0 then P1; P1 planner receives
            # only its sanitized observation, never P0's pending commitment.
            state = engine.step(state, action)
            atomic(path, sealed(row))
        raw, normalized = outcome(self.spec.game, engine, state)
        row.update(status="completed", raw_outcome=raw, normalized_outcome=normalized)
        self.validate_game(row, inputs)
        atomic(path, sealed(row))
        self.new_games += 1
        return row

    def versus(self, policy: Strategy, opponents: list[Strategy | None], seeds: tuple[int, ...],
               label: str, phase: str) -> tuple[float, list[dict]]:
        rows, scores = [], []
        for j, opponent in enumerate(opponents):
            for seed in seeds:
                for seat in (0, 1):
                    players = [opponent, opponent]
                    players[seat] = policy
                    row = self.game(f"{label}_o{j}_s{seed}_p{seat}", seed, players, phase)
                    rows.append(row)
                    scores.append(row["normalized_outcome"][seat])
        return sum(scores)/len(scores), rows

    def round_robin(self, number: int) -> list[dict]:
        cohort = list(self.strategies)
        returns = {s.id: {o.id: [] for o in cohort} for s in cohort}
        rows = []
        for i, left in enumerate(cohort):
            for right in cohort[i:]:
                for seed in self.spec.train_seeds:
                    for swap in (0, 1):
                        players = [left, right] if not swap else [right, left]
                        row = self.game(f"round{number}_{left.id}_{right.id}_s{seed}_swap{swap}",
                                        seed, players, "train")
                        rows.append(row)
                        for seat in (0, 1):
                            returns[players[seat].id][players[1-seat].id].append(row["normalized_outcome"][seat])
        # Equal opponent weighting (including self, with exact seat symmetry).
        for strategy in cohort:
            strategy.score = sum(sum(v)/len(v) for v in returns[strategy.id].values())/len(cohort)
            strategy.score_round = number
        record = {"round": number, "cohort": [s.id for s in cohort],
                  "seeds": list(self.spec.train_seeds), "both_seats": True,
                  "scores": {s.id: s.score for s in cohort}, "game_ids": [r["item_id"] for r in rows]}
        self.rounds.append(record)
        immutable(self.directory / "metrics/rounds" / f"{number:04d}.json", record)
        self.snapshot()
        return rows

    def develop(self, strategy: Strategy, initial: Strategy) -> None:
        score, rows = self.versus(strategy, [initial], self.spec.dev_seeds, f"dev_{strategy.id}", "dev")
        strategy.dev_score = score
        immutable(self.directory / "metrics/dev" / f"{strategy.id}.json",
                  {"strategy_id": strategy.id, "score": score, "opponents": [initial.sha256],
                   "seeds": list(self.spec.dev_seeds), "search": asdict(self.spec.search),
                   "game_ids": [r["item_id"] for r in rows]})
        self.snapshot()

    def learn(self) -> tuple[Strategy, Strategy]:
        language = ("Return only restricted Python def evaluate_state(state), returning ((v0,v1), explanation_dict). "
                    "Values normalized [-1,1], zero-sum, fixed player P0/P1 order. No exec/imports/attributes. "
                    "state has game, player_id, private_info, public_info, history. Only own private information. "
                    "No external opponent, no CFR. This is an offline mock workflow.")
        source = self.call("seed_code", {"instruction": language, "rules": rules(self.spec.game)})
        initial = self.generated("s0", source)
        training_rows = self.round_robin(0)
        self.develop(initial, initial)
        for iteration in range(1, self.spec.implementations + 1):
            pool = sorted(self.strategies, key=lambda s: (-s.score, s.id))[:self.spec.pool_size]
            parent = pool[(iteration-1) % len(pool)]
            feedback = key_feedback(training_rows, parent.id)
            reflection = self.call("reflection", {"iteration": iteration, "parent": asdict(parent),
                                                   "feedback": feedback, "instruction": "Reflect on visited discrepancies, not test outcomes."})
            idea_text = self.call("idea", {"iteration": iteration, "parent_id": parent.id,
                                           "reflection": reflection, "instruction": "Propose one value-function implementation idea."})
            idea = Idea(f"i{iteration}", parent.id, idea_text)
            self.queue.add(idea)
            chosen = self.queue.select(self.spec.planner_seed + iteration, {s.id for s in pool})
            selected_parent = next(s for s in pool if s.id == chosen.parent_id)
            self.snapshot()
            source = self.call("code", {"iteration": iteration, "parent_source": selected_parent.source,
                                       "idea": asdict(chosen), "instruction": language})
            candidate = self.generated(f"s{iteration}", source, selected_parent.id, chosen.id)
            # Freeze BEFORE any new score is observed. Parent and child are
            # scored against exactly the same cohort/seeds/both seats/budget.
            child_score, _ = self.versus(candidate, pool, self.spec.train_seeds, f"paired{iteration}_child", "train")
            parent_score, _ = self.versus(selected_parent, pool, self.spec.train_seeds, f"paired{iteration}_parent", "train")
            self.queue.observe(chosen.id, child_score, parent_score)
            self.improvements.append({"iteration": iteration, "child_id": candidate.id,
                                      "parent_id": selected_parent.id, "idea_id": chosen.id,
                                      "cohort": [s.id for s in pool], "cohort_hashes": [s.sha256 for s in pool],
                                      "seeds": list(self.spec.train_seeds), "child_score": child_score,
                                      "parent_score": parent_score, "improvement": child_score-parent_score})
            training_rows = self.round_robin(iteration)
            self.develop(candidate, initial)
            self.evolutions = iteration
            self.snapshot()
            immutable(self.directory / "metrics/evolutions" / f"{iteration:04d}.json",
                      {"strategies": [asdict(s) for s in self.strategies], "idea_queue": self.queue.snapshot(),
                       "improvement": self.improvements[-1], "feedback": feedback})
        # Test is not called until AFTER this frozen artifact has been written.
        selected = sorted(self.strategies, key=lambda s: (-s.dev_score, s.id))[0]
        policy = {"schema": SCHEMA, "game": self.spec.game, "source": selected.source, "sha256": selected.sha256,
                  "initial": {"source": initial.source, "sha256": initial.sha256},
                  "train_seeds": list(self.spec.train_seeds), "dev_seeds": list(self.spec.dev_seeds),
                  "search_config": asdict(self.spec.search), "planner_seed": self.spec.planner_seed,
                  "provenance": ScriptedModel.provenance, "caveats": CAVEATS,
                  "selection": {"criterion": "dev_only", "strategy_id": selected.id, "dev_score": selected.dev_score}}
        immutable(self.directory / "metrics/frozen_policy.json", policy)
        return selected, initial

    def evaluate(self, selected: Strategy, initial: Strategy) -> dict:
        opponent = initial if self.spec.evaluation_opponent == "initial" else None
        comparison = {}
        for label, policy in (("selected", selected), ("initial_baseline", initial)):
            score, rows = self.versus(policy, [opponent], self.spec.eval_seeds, f"test_{label}", "test")
            comparison[label] = {"score": score, "source_sha256": policy.sha256,
                                 "game_ids": [r["item_id"] for r in rows], "games": len(rows)}
        result = {"schema": SCHEMA, "comparison": comparison, "seeds": list(self.spec.eval_seeds),
                  "search": asdict(self.spec.search), "planner_seed": self.spec.planner_seed,
                  "opponent": self.spec.evaluation_opponent,
                  "opponent_sha256": initial.sha256 if opponent else None, "both_seats": True,
                  "affects_selection": False, "provenance": ScriptedModel.provenance}
        immutable(self.directory / "metrics/evaluation.json", result)
        return result

    def execute(self) -> dict | None:
        if self.spec.mode == "evaluate":
            policy = self.policy
            selected = Strategy.create("frozen", policy["source"])
            initial = Strategy.create("initial", policy["initial"]["source"])
            self.strategies = [selected, initial]
            immutable(self.directory / "metrics/frozen_policy.json", policy)
        else:
            selected, initial = self.learn()
        result = self.evaluate(selected, initial) if self.spec.mode != "learn" else None
        # No extraneous completed games/calls may silently hide outside schedule.
        actual_ids = {p.stem for p in (self.directory / "trajectories").glob("*.json")}
        if actual_ids != self.game_ids or len(self.game_ids) != self.spec.planned_games():
            raise ResumeConflict("Unexpected/missing game schedule items")
        if len(self.journal._read_records()) != self.call_step:
            raise ResumeConflict("Unexpected journal items after replay")
        return result


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _preflight(directory: Path) -> dict:
    # Verify every persisted JSON artifact, including incremental snapshots.
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ResumeConflict("Symlink in run artifacts")
        if path.is_file() and path.suffix == ".json" and "prompts" not in path.relative_to(directory).parts:
            unseal(read_json(path))
    return unseal(read_json(directory / "config.json"))


def run(spec: RunSpec, *, output_dir: str | Path | None = None, resume: str | Path | None = None,
        policy_file: str | Path | None = None, model: ScriptedModel | None = None,
        stop_after_games: int | None = None, command: list[str] | None = None) -> Path:
    """Execute or resume a tiny offline workflow; failures persist then propagate.

    ``stop_after_games`` limits NEW games in this invocation (excluded from the
    immutable scientific config). Inject only ScriptedModel subclasses for tests.
    No custom backend/live API is accepted. Evaluate requires explicit policy and
    exactly its frozen search budget/planner seed; CLI adopts those by default.
    """
    spec.validate()
    if stop_after_games is not None:
        integer(stop_after_games, 1, spec.max_games, "stop_after_games")
    model = model if model is not None else ScriptedModel()
    if not isinstance(model, ScriptedModel) or model.provenance != ScriptedModel.provenance:
        raise ValueError("Only explicitly scripted offline backend is supported")
    policy = None
    if spec.mode == "evaluate":
        if policy_file is None:
            raise ValueError("evaluate requires an explicit policy file")
        policy = load_policy(Path(policy_file).resolve(), spec.game, spec.eval_seeds)
        if policy["search_config"] != asdict(spec.search) or policy["planner_seed"] != spec.planner_seed:
            raise ValueError("Evaluation must use the frozen policy's search budget/seed")
    elif policy_file is not None:
        raise ValueError("policy input is only valid in evaluate mode")
    allowed = (ROOT / "results" / spec.game / "strategist_dev").resolve()
    base = Path(output_dir).resolve() if output_dir is not None else allowed
    if not base.is_relative_to(allowed):
        raise ValueError("output-dir must be under results/<game>/strategist_dev")
    signature = {"schema": SCHEMA, "spec": spec.payload(), "implementation_hash": implementation_hash(),
                 "model": model.model_name, "backend_hash": digest(inspect.getsource(type(model))),
                 "provenance": model.provenance, "policy_hash": digest(policy) if policy else None}
    signature_hash = digest(signature)
    # No artifact writes before all argument/policy/source/config validation.
    if resume is not None:
        directory = Path(resume).resolve()
        if not directory.is_relative_to(allowed) or not directory.is_dir():
            raise ValueError("Resume must be an existing strategist_dev run")
        saved = _preflight(directory)
        if saved.get("signature") != signature or saved.get("signature_hash") != signature_hash:
            raise ResumeConflict("Run source/config/backend/policy changed")
    else:
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in range(100):
            directory = base / (f"{stamp}_strategist_{spec.mode}_dev" + (f"_{suffix}" if suffix else ""))
            try:
                directory.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("No free run directory")
    with run_lock(directory):
        if resume is not None:
            saved = _preflight(directory)
            if saved["signature"] != signature:
                raise ResumeConflict("Config changed during lock acquisition")
        else:
            for folder in ("prompts", "trajectories", "metrics", "logs"):
                (directory / folder).mkdir()
            saved = {"schema": SCHEMA, "phase": "dev_debug", "live_api": False, "dry_run": True,
                     "signature": signature, "signature_hash": signature_hash,
                     "command": command or [], "created_at": datetime.now(timezone.utc).isoformat(),
                     "git_commit": _git(["rev-parse", "HEAD"]),
                     "dirty_worktree": bool(_git(["status", "--porcelain", "--untracked-files=normal"])),
                     "model": model.model_name, "scripted": True, "caveats": CAVEATS,
                     "predeclared_metrics": {"selection": "highest mean dev normalized return, ties smallest strategy id",
                                             "idea": "child minus parent, frozen same training cohort",
                                             "key_feedback": "largest squared fixed-P0/P1 root value discrepancy",
                                             "test": "selected vs initial under identical seeds/opponent/search budget"}}
            atomic(directory / "config.json", sealed(saved))
        workflow = None
        start_calls = model.calls
        status, error_type, evaluation = "running", None, None
        try:
            workflow = Workflow(spec, directory, signature_hash, model, policy, stop_after_games)
            evaluation = workflow.execute()
            status = "completed"
        except (InterruptedRun, KeyboardInterrupt) as exc:
            status, error_type = "interrupted", type(exc).__name__
            if isinstance(exc, KeyboardInterrupt):
                raise
        except BaseException as exc:
            status, error_type = "failed", type(exc).__name__
            raise
        finally:
            # Do not mask the original exception with success or fallback.
            if workflow is not None:
                workflow.snapshot()
            completed = sum(unseal(read_json(p)).get("status") == "completed"
                            for p in (directory / "trajectories").glob("*.json"))
            calls = workflow.journal.stats() if workflow is not None else None
            summary = {"schema": SCHEMA, "status": status, "error_type": error_type,
                       "phase": "dev_debug", "live_api": False, "scripted": True,
                       "completed": completed, "remaining": max(0, spec.planned_games()-completed),
                       "planned_games": spec.planned_games(), "max_games": spec.max_games,
                       "skipped": workflow.skipped if workflow else 0,
                       "new_games": workflow.new_games if workflow else 0,
                       "restarted_offline_games": workflow.restarted if workflow else 0,
                       "evolutions": workflow.evolutions if workflow else 0,
                       "calls": calls, "max_requests": spec.max_requests,
                       "new_scripted_calls": model.calls-start_calls, "api_requests": 0,
                       "actual_api_usage": None, "estimated_tokens": None,
                       "evaluation": evaluation, "config_hash": signature_hash,
                       "resumed": resume is not None, "updated_at": datetime.now(timezone.utc).isoformat(),
                       "caveats": CAVEATS}
            atomic(directory / "summary.json", sealed(summary))
    return directory