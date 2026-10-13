"""No-provider local MCTS self-play, batch MC updates and replayable artifacts.

Completed episodes are reused after engine-only replay (no repeated search).
An interrupted unsaved episode is deterministically recomputed: local CPU only.
Completed policies/checkpoints/trajectories are never overwritten.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

from tqdm import tqdm

from base.types import Action
from methods.strategist.adapters import GAMES, PAYOFF_SCALES, World, _seed, make_game, visible
from methods.strategist.search import SearchConfig, StrategistSearch
from methods.strategist.tabular_value import FrozenValueTable, canonical, digest, state_key

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TabularSpec:
    rounds: int = 10
    games_per_round: int = 32
    start_seed: int = 1100000
    init_seed: int = 9100
    search_seed: int = 9200
    initial_scale: float = 0.05
    search: SearchConfig = field(default_factory=lambda: SearchConfig(32, 12, 128))

    def validate(self):
        for name, upper in (("rounds", 10), ("games_per_round", 32)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be in [1,{upper}] for this local budget")
        for seed in (self.start_seed, self.start_seed + self.games - 1, self.init_seed,
                     self.search_seed, self.search_seed + 1):
            _seed(seed)
        self.search.__post_init__()
        for game in GAMES:
            FrozenValueTable(game, self.init_seed, self.initial_scale)

    @property
    def games(self):
        return self.rounds * self.games_per_round


def source_hashes() -> dict:
    paths = ["methods/strategist/tabular_value.py", "methods/strategist/tabular_training.py",
             "methods/strategist/search.py", "methods/strategist/adapters.py",
             "methods/strategist/value.py", "runs/strategist_tabular.py",
             "base/types.py", "base/interfaces.py", "base/liars_dice_core.py",
             "games/leduc.py", "games/liars_dice.py", "games/goofspiel.py",
             "games/prize_pool_n5.txt"]
    import hashlib
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def git_provenance() -> dict:
    def command(*args):
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = command("status", "--porcelain", "--untracked-files=normal")
    return {"commit": command("rev-parse", "HEAD"),
            "dirty_worktree": bool(status) if status is not None else None}


def seal(payload: dict) -> dict:
    return {"sha256": digest(payload), "payload": payload}


def read_artifact(path: Path) -> dict:
    def unique(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("Duplicate JSON key")
            data[key] = value
        return data
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    if set(data) != {"sha256", "payload"} or data["sha256"] != digest(data["payload"]):
        raise ValueError(f"Artifact checksum mismatch: {path.name}")
    return data["payload"]


def write_artifact(path: Path, payload: dict, *, replace: bool = False):
    """Atomic and durable file publication; scientific records are immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(seal(payload)) + "\n"
    if path.exists() and not replace:
        if canonical(read_artifact(path)) != canonical(payload):
            raise ValueError(f"Existing artifact conflicts: {path.name}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic no-clobber publication
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def lock_run(path: Path):
    with (path / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This local run is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class CountedValue:
    """Per-game diagnostics live outside the immutable learned parameters."""
    def __init__(self, table: FrozenValueTable):
        self.table, self.hits, self.misses = table, 0, 0

    def evaluate(self, features):
        values, explanation = self.table.evaluate(features)
        if explanation["count"]:
            self.hits += 1
        else:
            self.misses += 1
        return values, explanation


def episode_context(game: str, spec: TabularSpec, index: int, fingerprint: str) -> dict:
    return {"game": game, "index": index, "round": index // spec.games_per_round + 1,
            "seed": spec.start_seed + index, "table_before": fingerprint}


def play_episode(game: str, table: FrozenValueTable, spec: TabularSpec, index: int) -> dict:
    context = episode_context(game, spec, index, table.fingerprint)
    engine = make_game(game, context["seed"])
    state = engine.reset()
    counted = CountedValue(table)
    planners = [StrategistSearch(game, counted, spec.search, spec.search_seed + p) for p in (0, 1)]
    steps = []
    while not engine.is_terminal(state):
        if len(steps) >= 64:
            raise ValueError("Game exceeded conservative finite horizon")
        actor = engine.get_current_player(state)
        obs = engine.get_observation(state, actor)
        decision = planners[actor].decide(obs)
        if decision.action not in engine.get_legal_actions(state):
            raise ValueError("MCTS selected an illegal real action")
        features = {"game": game, **visible(obs)}
        if features != decision.trace["root_features"]:
            raise ValueError("Root features differ from actual visible observation")
        steps.append({"features": features, "action": decision.action.model_dump(),
                      "search": decision.trace})
        state = engine.step(state, decision.action)
    values = list(World(game, engine, state, 0).terminal_values())
    if table.fingerprint != context["table_before"]:
        raise ValueError("Persistent value table changed during self-play")
    return {**context, "status": "completed", "steps": steps,
            "normalized_outcome": values, "outcome": [v * PAYOFF_SCALES[game] for v in values],
            "lookups": {"hits": counted.hits, "misses": counted.misses}}


def audit_episode(row: dict, game: str, spec: TabularSpec, index: int,
                  table: FrozenValueTable):
    """Reconstruct only the real game, not MCTS; rejects incomplete/illegal rows."""
    expected = episode_context(game, spec, index, table.fingerprint)
    if any(row.get(k) != v for k, v in expected.items()) or row.get("status") != "completed":
        raise ValueError("Episode identity/configuration conflict")
    engine = make_game(game, expected["seed"])
    state = engine.reset()
    hits = misses = 0
    if not 1 <= len(row["steps"]) <= 64:
        raise ValueError("Invalid episode length")
    for step in row["steps"]:
        if engine.is_terminal(state):
            raise ValueError("Actions after terminal")
        actor = engine.get_current_player(state)
        features = {"game": game, **visible(engine.get_observation(state, actor))}
        action = Action(**step["action"])
        trace = step["search"]
        if (features != step["features"] or features != trace["root_features"]
                or action not in engine.get_legal_actions(state)
                or trace["simulations"] != spec.search.simulations
                or sum(a["visits"] for a in trace["actions"]) != spec.search.simulations):
            raise ValueError("Saved observation/action/search mismatch")
        values, explanation = table.evaluate(features)
        if trace["root_initial_heuristic"] != {"values": list(values), "explanation": explanation}:
            raise ValueError("Saved root value differs from batch table")
        hits += int(explanation["count"] > 0)
        misses += int(explanation["count"] == 0)
        state = engine.step(state, action)
    values = list(World(game, engine, state, 0).terminal_values())
    if row["normalized_outcome"] != values or row["outcome"] != [v * PAYOFF_SCALES[game] for v in values]:
        raise ValueError("Saved terminal return mismatch")
    if row["lookups"]["hits"] < hits or row["lookups"]["misses"] < misses:
        raise ValueError("Invalid lookup statistics")


def round_metrics(rows: list[dict], before: FrozenValueTable, after: FrozenValueTable) -> dict:
    # Set iteration is salted per process. Stable summation order is required
    # for byte-identical checkpoint regeneration during a fresh-process resume.
    keys = sorted({state_key(step["features"]) for row in rows for step in row["steps"]})
    changed = [abs(after.entries[k].value - (before.entries[k].value if k in before.entries
                                            else before.initial_value(k))) for k in keys]
    hits = sum(r["lookups"]["hits"] for r in rows)
    misses = sum(r["lookups"]["misses"] for r in rows)
    root_hits = sum(state_key(s["features"]) in before.entries for r in rows for s in r["steps"])
    actions = sum(len(r["steps"]) for r in rows)
    # These are training observations, NOT independent generalization scores.
    return {"round": rows[0]["round"], "games": len(rows), "actions": actions,
            "table_before": before.fingerprint, "table_after": after.fingerprint,
            "entries": len(after.entries), "new_entries": len(after.entries) - len(before.entries),
            "sample_count": sum(e.count for e in after.entries.values()),
            "mean_abs_updated_value_change": sum(changed) / len(changed),
            "root_hits": root_hits, "root_lookups": actions,
            "lookups": {"hits": hits, "misses": misses, "hit_rate": hits / (hits + misses)},
            "p0_net": sum(r["outcome"][0] for r in rows),
            "p0_wins": sum(r["outcome"][0] > 0 for r in rows),
            "ties": sum(r["outcome"][0] == 0 for r in rows)}


def train(game: str, spec: TabularSpec = TabularSpec(), *, output_dir: Path | None = None,
          resume: Path | None = None, stop_after: int | None = None,
          progress: bool = True) -> Path:
    spec.validate()
    if game not in GAMES:
        raise ValueError("Unsupported game")
    if stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after <= spec.games):
        raise ValueError("stop_after must be an absolute completed-game count")
    root = (ROOT / "results").resolve()
    parent = Path(output_dir or root / game / "strategist_tabular_dev").resolve()
    path = Path(resume).resolve() if resume else parent / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + "tabular_dev_" + uuid.uuid4().hex[:8])
    if not path.is_relative_to(root) or path == root:
        raise ValueError("All local training artifacts must stay under results/")
    hashes = source_hashes()
    if resume:
        config = read_artifact(path / "config.json")
        if config["spec"] != asdict(spec) or config["game"] != game or config["source_hashes"] != hashes:
            raise ValueError("Resume source/configuration conflict (Git dirtiness is irrelevant)")
    else:
        path.mkdir(parents=True, exist_ok=False)
        config = {"schema": "strategist.tabular_run.v1", "phase": "dev_debug",
                  "method": "tabular_mc_information_set_mcts", "game": game,
                  "spec": asdict(spec), "created_at": now(), "command": sys.argv,
                  "python": platform.python_version(), "git": git_provenance(),
                  "source_hashes": hashes, "live_api": False, "dry_run": False,
                  "model": None, "base_url_host": None, "api_requests": 0,
                  "purpose": "Train independent local value tables under a fixed episode budget",
                  "opponent": "same_batch_frozen_self_play", "baseline": None,
                  "primary_metrics": ["completed_legal_games", "updates", "entries", "lookup_hit_rate"],
                  "exclusion_rule": "No exclusions; any illegal action or invariant failure stops",
                  "budget": {"training_games": spec.games, "validation_games": 0, "pk_games": 0},
                  "caveats": ["10x32 borrows an episode budget, not the paper's LLM strategy tree",
                              "Exact-key sparse table; unseen keys retain seeded random initialization",
                              "Cumulative targets mix changing self-play policies; not GTO or convergence proof",
                              "MCTS uniform hidden priors and repository-specific rules retained"]}
        write_artifact(path / "config.json", config)
    with lock_run(path):
        return _train_locked(path, config, spec, stop_after, progress, bool(resume))


def _train_locked(path, config, spec, stop_after, progress, resuming):
    started = time.monotonic()
    invocation = uuid.uuid4().hex
    game = config["game"]
    table = FrozenValueTable(game, spec.init_seed, spec.initial_scale)
    metrics, all_rows = [], []
    completed = skipped = new_games = 0
    status = "running"
    log = {"started_at": now(), "resume": resuming, "run_directory": str(path)}
    write_artifact(path / "logs" / f"invocation_{invocation}.json", log)
    bar = tqdm(total=spec.games, disable=not progress, desc=f"{game} tabular/MCTS self-play dev_debug")
    try:
        planned = {f"g{i:04d}.json" for i in range(spec.games)}
        saved = {p.name for p in (path / "trajectories").glob("*.json")}
        if saved - planned:
            raise ValueError("Unexpected trajectory outside planned schedule")
        if saved != {f"g{i:04d}.json" for i in range(len(saved))}:
            raise ValueError("Missing trajectory inside completed prefix; refuse regeneration")
        for checkpoint in (path / "metrics").glob("round_*.json"):
            round_number = read_artifact(checkpoint)["round"]
            if not 1 <= round_number <= spec.rounds or round_number * spec.games_per_round > len(saved):
                raise ValueError("Checkpoint depends on missing episodes")
        policy_path = path / "metrics" / "frozen_policy.json"
        if policy_path.exists() and len(saved) != spec.games:
            raise ValueError("Frozen policy depends on missing episodes")
        for batch in range(spec.rounds):
            rows = []
            for j in range(spec.games_per_round):
                index = batch * spec.games_per_round + j
                episode_path = path / "trajectories" / f"g{index:04d}.json"
                if episode_path.exists():
                    row = read_artifact(episode_path)
                    audit_episode(row, game, spec, index, table)
                    skipped += 1
                else:
                    if stop_after is not None and completed >= stop_after:
                        status = "paused"
                        break
                    row = play_episode(game, table, spec, index)
                    audit_episode(row, game, spec, index, table)
                    write_artifact(episode_path, row)
                    new_games += 1
                rows.append(row)
                all_rows.append(row)
                completed += 1
                bar.set_postfix(round=batch + 1, seed=row["seed"], entries=len(table.entries), run=path.name)
                bar.update(1)
            if len(rows) != spec.games_per_round:
                break
            updated = table.updated(rows)
            metric = round_metrics(rows, table, updated)
            checkpoint = {"round": batch + 1, "table": updated.payload(), "metrics": metric,
                          "episodes": [digest(r) for r in rows]}
            write_artifact(path / "metrics" / f"round_{batch+1:02d}.json", checkpoint)
            metrics.append(metric)
            table = updated
            if progress:
                bar.write(f"Update completed: {game} round={batch+1}/{spec.rounds} "
                          f"games={completed}/{spec.games} entries={len(table.entries)} "
                          f"lookup_hit_rate={metric['lookups']['hit_rate']:.3f}")
        if completed == spec.games:
            policy = {"schema": "strategist.frozen_tabular_policy.v1", "frozen": True,
                      "game": game, "table": table.payload(), "search": asdict(spec.search),
                      "search_seed": spec.search_seed, "training_games": completed,
                      "updates": len(metrics), "source_hashes": config["source_hashes"]}
            write_artifact(path / "metrics" / "frozen_policy.json", policy)
            status = "completed"
        if source_hashes() != config["source_hashes"]:
            raise ValueError("Training dependencies changed during execution")
        summary = {"status": status, "game": game, "phase": "dev_debug", "api_requests": 0,
                   "completed": completed, "planned": spec.games, "updates": len(metrics),
                   "entries": len(table.entries), "table_sha256": table.fingerprint,
                   "actions": sum(len(r["steps"]) for r in all_rows), "rounds": metrics,
                   "validation_games": 0, "external_pk_games": 0,
                   "policy": "metrics/frozen_policy.json" if status == "completed" else None}
        target = path / "summary.json"
        if target.exists() and read_artifact(target).get("status") == "completed":
            write_artifact(target, summary)
        else:
            write_artifact(target, summary, replace=True)
    except BaseException as exc:
        status = "failed"
        log["error_type"] = type(exc).__name__
        target = path / "summary.json"
        # Keep a completed result intact when a subsequent audit fails.
        if not target.exists() or read_artifact(target).get("status") != "completed":
            write_artifact(target, {"status": "failed", "game": game, "phase": "dev_debug",
                                   "completed": completed, "planned": spec.games,
                                   "updates": len(metrics), "api_requests": 0,
                                   "error_type": type(exc).__name__}, replace=True)
        raise
    finally:
        bar.close()
        log.update({"finished_at": now(), "status": status, "completed": completed,
                    "skipped": skipped, "new_games": new_games, "remaining": spec.games - completed,
                    "seconds": time.monotonic() - started})
        write_artifact(path / "logs" / f"invocation_{invocation}.json", log, replace=True)
    return path


def load_policy(path: Path) -> tuple[FrozenValueTable, SearchConfig, int]:
    payload = read_artifact(Path(path))
    if payload["schema"] != "strategist.frozen_tabular_policy.v1" or payload["frozen"] is not True:
        raise ValueError("Expected frozen tabular policy")
    table = FrozenValueTable.from_payload(payload["table"])
    if table.game != payload["game"]:
        raise ValueError("Policy game mismatch")
    search = SearchConfig(**payload["search"])
    _seed(payload["search_seed"])
    return table, search, payload["search_seed"]


class FrozenTabularAgent:
    """PK inference adapter: fresh search per action, no learning API."""
    def __init__(self, policy_path: Path):
        self.table, self.search, self.seed = load_policy(policy_path)

    def decide(self, observation):
        return StrategistSearch(self.table.game, self.table, self.search,
                                self.seed + observation.player_id).decide(observation)