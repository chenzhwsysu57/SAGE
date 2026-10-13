"""Bounded continuation with a predeclared *value stability* stopping rule.

This does not certify Bellman residuals, accurate values or Nash equilibrium.
The fixed panel is ALL information sets in the initial table, equally weighted.
Every checkpoint compares to the previous check, not to the previous batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
import hashlib
import math
from pathlib import Path
import sys
import time
import uuid

from tqdm import tqdm

from methods.strategist import tabular_training as core
from methods.strategist.search import SearchConfig
from methods.strategist.tabular_value import FrozenValueTable, canonical, digest


@dataclass(frozen=True)
class StabilitySpec:
    max_total_games: int = 3200
    threshold: float = 0.1
    check_every_updates: int = 10
    patience: int = 3
    min_total_visits: int = 5
    min_fresh_visits: int = 1
    min_panel_coverage: float = 0.9

    def validate(self):
        for name, upper in (("max_total_games", 3200), ("check_every_updates", 100),
                            ("patience", 100), ("min_total_visits", 3200), ("min_fresh_visits", 3200)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1,{upper}]")
        for name, upper in (("threshold", 2), ("min_panel_coverage", 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f"Invalid {name}")


class Snapshot(FrozenValueTable):
    """Cache a fingerprint of an immutable batch table, not evaluator outputs.

    The old implementation serializes the entire growing table several times
    per episode. Caching avoids quadratic logging overhead without changing any
    entry, lookup, arithmetic, random seed, MC target or MCTS action.
    """
    @cached_property
    def fingerprint(self):
        return super().fingerprint


def implementation_hashes():
    paths = ["methods/strategist/tabular_stability.py", "runs/strategist_tabular_continue.py",
             "scripts/strategist-tabular-continue.sh"]
    return {**core.source_hashes(), **{
        name: hashlib.sha256((core.ROOT / name).read_bytes()).hexdigest() for name in paths}}


def check_values(panel: tuple[str, ...], previous: FrozenValueTable, current: FrozenValueTable,
                 spec: StabilitySpec, previous_streak: int, total_games: int, total_updates: int):
    spec.validate()
    if not panel or len(panel) != len(set(panel)) or previous.game != current.game:
        raise ValueError("Invalid fixed stability panel")
    changes, fresh_changes, eligible = [], [], []
    for key in sorted(panel):
        old, new = previous.entries[key], current.entries[key]
        fresh = new.count - old.count
        if fresh < 0:
            raise ValueError("Visit counts went backwards")
        delta = abs(new.value - old.value)
        changes.append(delta)
        if fresh:
            fresh_changes.append(delta)
        if new.count >= spec.min_total_visits and fresh >= spec.min_fresh_visits:
            eligible.append(key)
    maximum = max(changes)
    coverage = len(eligible) / len(panel)
    value_passed = maximum <= spec.threshold
    coverage_passed = coverage >= spec.min_panel_coverage
    passed = value_passed and coverage_passed
    streak = previous_streak + 1 if passed else 0
    return {
        "total_games": total_games, "total_updates": total_updates,
        "previous_table": previous.fingerprint, "current_table": current.fingerprint,
        "panel_sha256": digest(sorted(panel)), "panel_size": len(panel),
        "max_abs_value_change": maximum, "mean_abs_value_change": math.fsum(changes) / len(changes),
        "fresh_panel_keys": len(fresh_changes), "eligible_panel_keys": len(eligible),
        "eligible_coverage": coverage, "threshold": spec.threshold,
        "value_passed": value_passed, "coverage_passed": coverage_passed,
        "passed": passed, "consecutive_passes": streak,
        "stable": streak >= spec.patience,
        "reason": "passed" if passed else "value_and_coverage" if not value_passed and not coverage_passed
                  else "value_change" if not value_passed else "insufficient_fresh_coverage",
        "scope": "Fixed initial-table panel only, not all reachable states or newly discovered keys",
    }


def verify_parent(parent: Path):
    """Read-only provenance/terminal-checkpoint validation, never resumes parent."""
    config = core.read_artifact(parent / "config.json")
    summary = core.read_artifact(parent / "summary.json")
    policy = core.read_artifact(parent / "metrics/frozen_policy.json")
    if (config["schema"] != "strategist.tabular_run.v1" or summary["status"] != "completed"
            or config["source_hashes"] != core.source_hashes()
            or policy["source_hashes"] != config["source_hashes"]):
        raise ValueError("Require a complete compatible original tabular run")
    source = core.TabularSpec(**{**config["spec"], "search": SearchConfig(**config["spec"]["search"])})
    source.validate()
    table = Snapshot.from_payload(policy["table"])
    if (policy["schema"] != "strategist.frozen_tabular_policy.v1" or policy["frozen"] is not True
            or table.game != config["game"] or summary["game"] != table.game
            or policy["game"] != table.game or summary["completed"] != source.games
            or summary["updates"] != source.rounds or policy["training_games"] != source.games
            or policy["updates"] != source.rounds or policy["search"] != asdict(source.search)
            or policy["search_seed"] != source.search_seed or not table.entries
            or summary["table_sha256"] != table.fingerprint):
        raise ValueError("Parent metadata/table mismatch")
    checkpoint = core.read_artifact(parent / "metrics" / f"round_{source.rounds:02d}.json")
    if checkpoint["table"] != policy["table"]:
        raise ValueError("Parent final checkpoint/table mismatch")
    names = {p.name for p in (parent / "trajectories").glob("*.json")}
    if names != {f"g{i:04d}.json" for i in range(source.games)}:
        raise ValueError("Parent trajectories missing")
    return config, summary, policy, source, table


def continue_training(parent: Path | None, spec: StabilitySpec = StabilitySpec(), *,
                      output_dir: Path | None = None, resume: Path | None = None,
                      stop_after: int | None = None, progress: bool = True) -> Path:
    spec.validate()
    hashes = implementation_hashes()
    if resume:
        path = Path(resume).resolve()
        config = core.read_artifact(path / "config.json")
        if config["stability"] != asdict(spec) or config["source_hashes"] != hashes:
            raise ValueError("Continuation source/configuration conflict")
        if parent is not None and str(Path(parent).resolve()) != config["parent_run"]:
            raise ValueError("Conflicting parent run")
        parent = Path(config["parent_run"])
    elif parent is None:
        raise ValueError("Provide a completed parent run")
    parent = Path(parent).resolve()
    parent_config, parent_summary, policy, source, initial = verify_parent(parent)
    new_games = spec.max_total_games - source.games
    if new_games <= 0 or new_games % source.games_per_round:
        raise ValueError("Maximum total must exceed parent games by whole batches")
    if stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after <= new_games):
        raise ValueError("stop_after counts additional completed games")
    # Only play_episode/audit_episode consume this spec; no old runner limit is
    # bypassed. The extension has its own explicit bounded protocol above.
    play_spec = core.TabularSpec(rounds=new_games // source.games_per_round,
                                games_per_round=source.games_per_round,
                                start_seed=source.start_seed + source.games,
                                init_seed=source.init_seed, search_seed=source.search_seed,
                                initial_scale=source.initial_scale, search=source.search)
    core._seed(play_spec.start_seed + new_games - 1)
    panel = tuple(sorted(initial.entries))
    parent_ids = {"config": digest(parent_config), "summary": digest(parent_summary), "policy": digest(policy)}
    if resume:
        if config["parent_ids"] != parent_ids or config["play_spec"] != asdict(play_spec):
            raise ValueError("Parent changed since continuation started")
        if core.read_artifact(path / "metrics/initial_policy.json") != policy:
            raise ValueError("Stored initial table conflict")
    else:
        base = Path(output_dir or core.ROOT / "results" / initial.game / "strategist_tabular_stability_dev").resolve()
        path = base / (datetime.now().strftime("%Y%m%d_%H%M%S_") + "value_stability_dev_" + uuid.uuid4().hex[:8])
        config = {"schema": "strategist.tabular_continuation.v1", "phase": "dev_debug",
                  "created_at": core.now(), "command": sys.argv, "game": initial.game,
                  "method": "cumulative_first_visit_mc_information_set_mcts",
                  "source_hashes": hashes, "git": core.git_provenance(), "parent_run": str(parent),
                  "parent_ids": parent_ids, "initial_games": source.games, "initial_updates": source.rounds,
                  "play_spec": asdict(play_spec), "stability": asdict(spec),
                  "panel": {"selection": "all_initial_table_keys_equal_weight", "size": len(panel),
                            "sha256": digest(list(panel))},
                  "live_api": False, "dry_run": False, "model": None, "base_url_host": None,
                  "api_requests": 0, "validation_games": 0, "pk_games": 0,
                  "purpose": "Observe whether a predeclared 0.1 value-stability rule is met within a local cap",
                  "baseline": "frozen 320-game source table", "opponent": "same_batch_frozen_self_play",
                  "primary_metrics": ["stop_reason", "total_games", "total_updates", "max_abs_value_change",
                                      "eligible_coverage", "mcts_simulations"],
                  "approval": "User: 好那你帮我做一下。初步定在阈值差距0.1看看要多少局模拟和更新？",
                  "exclusions": "None; any failure stops and preserves evidence",
                  "caveats": ["Engineering stability, not Bellman residual or correctness certification",
                              "Threshold is on normalized P0 values; no policy/equilibrium claim",
                              "No new-state generalization; fixed original panel excludes newly discovered keys",
                              "Cumulative means shrink step sizes and mix changing self-play policies",
                              "This larger local budget no longer matches the original 10x32 pattern"]}
    if not path.is_relative_to((core.ROOT / "results").resolve()):
        raise ValueError("Continuation artifacts must stay below results/")
    if not resume:
        path.mkdir(parents=True, exist_ok=False)
        core.write_artifact(path / "config.json", config)
        core.write_artifact(path / "metrics/initial_policy.json", policy)
    with core.lock_run(path):
        return _run(path, config, play_spec, initial, spec, panel, stop_after, progress, bool(resume))


def _run(path, config, play_spec, initial, spec, panel, stop_after, progress, resuming):
    started, invocation = time.monotonic(), uuid.uuid4().hex
    table = reference = initial
    metrics, checks = [], []
    completed = skipped = generated = actions = simulations = transitions = streak = 0
    status, reason = "running", None
    log = {"started_at": core.now(), "resume": resuming, "run_directory": str(path)}
    core.write_artifact(path / "logs" / f"invocation_{invocation}.json", log)
    bar = tqdm(total=play_spec.games, disable=not progress,
               desc=f"{initial.game} MC+MCTS self-play stability dev_debug")
    try:
        saved = {p.name for p in (path / "trajectories").glob("*.json")}
        if len(saved) > play_spec.games or saved != {f"g{i:04d}.json" for i in range(len(saved))}:
            raise ValueError("Missing or out-of-budget saved episode")
        final_path = path / "metrics/frozen_policy.json"
        if final_path.exists():
            final = core.read_artifact(final_path)
            if final["additional_games"] != len(saved):
                raise ValueError("Saved policy depends on missing episodes")
        for cp in (path / "metrics").glob("round_*.json"):
            r = core.read_artifact(cp)["round"]
            if not 1 <= r <= play_spec.rounds or r * play_spec.games_per_round > len(saved):
                raise ValueError("Checkpoint depends on missing episodes")
        for batch in range(play_spec.rounds):
            rows = []
            for j in range(play_spec.games_per_round):
                index = batch * play_spec.games_per_round + j
                target = path / "trajectories" / f"g{index:04d}.json"
                if target.exists():
                    row = core.read_artifact(target)
                    skipped += 1
                else:
                    if stop_after is not None and completed >= stop_after:
                        status, reason = "paused", "user_pause"
                        break
                    row = core.play_episode(initial.game, table, play_spec, index)
                    generated += 1
                core.audit_episode(row, initial.game, play_spec, index, table)
                core.write_artifact(target, row)
                rows.append(row)
                completed += 1
                actions += len(row["steps"])
                simulations += sum(s["search"]["simulations"] for s in row["steps"])
                transitions += sum(s["search"]["transitions"] for s in row["steps"])
                bar.update(1)
                if progress:
                    bar.set_postfix(batch=batch+1, seed=str(row["seed"]), run=path.name)
            if len(rows) != play_spec.games_per_round:
                break
            updated = Snapshot.from_payload(table.updated(rows).payload())
            metric = core.round_metrics(rows, table, updated)
            table = updated
            metrics.append(metric)
            check = None
            if (batch + 1) % spec.check_every_updates == 0:
                check = check_values(panel, reference, table, spec, streak,
                                     config["initial_games"] + completed,
                                     config["initial_updates"] + batch + 1)
                checks.append(check)
                streak = check["consecutive_passes"]
                reference = table
            core.write_artifact(path / "metrics" / f"round_{batch+1:03d}.json",
                                {"round": batch+1, "table": table.payload(), "metrics": metric,
                                 "stability_check": check, "episodes": [digest(r) for r in rows]})
            if progress:
                message = (f"Update completed: {initial.game} total_games={config['initial_games']+completed} "
                           f"total_updates={config['initial_updates']+batch+1} entries={len(table.entries)}")
                if check:
                    message += (f" max_delta={check['max_abs_value_change']:.6f} "
                                f"fresh_qualified_coverage={check['eligible_coverage']:.3%} streak={streak}/{spec.patience}")
                bar.write(message)
            if check and check["stable"]:
                if completed < len(saved):
                    raise ValueError("Saved games after the predeclared stopping point")
                status, reason = "completed", "value_stable"
                break
        if completed == play_spec.games and reason != "value_stable":
            status, reason = "completed", "budget_exhausted_not_stable"
        if implementation_hashes() != config["source_hashes"]:
            raise ValueError("Source changed during continuation")
        if status == "completed":
            policy = {"schema": "strategist.frozen_tabular_policy.v1", "frozen": True, "game": initial.game,
                      "table": table.payload(), "search": asdict(play_spec.search), "search_seed": play_spec.search_seed,
                      "training_games": config["initial_games"] + completed,
                      "updates": config["initial_updates"] + len(metrics), "additional_games": completed,
                      "source_hashes": config["source_hashes"], "parent_policy_sha256": config["parent_ids"]["policy"],
                      "stop_reason": reason, "stability": asdict(spec)}
            core.write_artifact(final_path, policy)
        summary = {"status": status, "stop_reason": reason, "game": initial.game, "phase": "dev_debug",
                   "initial_games": config["initial_games"], "additional_games": completed,
                   "total_games": config["initial_games"] + completed, "additional_updates": len(metrics),
                   "total_updates": config["initial_updates"] + len(metrics), "entries": len(table.entries),
                   "table_sha256": table.fingerprint, "additional_actions": actions,
                   "additional_mcts_simulations": simulations, "additional_simulated_transitions": transitions,
                   "checks": checks, "rounds": metrics, "api_requests": 0, "validation_games": 0, "pk_games": 0,
                   "policy": "metrics/frozen_policy.json" if status == "completed" else None}
        target = path / "summary.json"
        terminal = target.exists() and core.read_artifact(target).get("status") == "completed"
        core.write_artifact(target, summary, replace=not terminal)
    except BaseException as exc:
        status = "failed"
        log["error_type"] = type(exc).__name__
        target = path / "summary.json"
        if not target.exists() or core.read_artifact(target).get("status") != "completed":
            core.write_artifact(target, {"status": status, "additional_games": completed,
                                        "error_type": type(exc).__name__, "phase": "dev_debug"}, replace=True)
        raise
    finally:
        bar.close()
        log.update({"finished_at": core.now(), "status": status, "seconds": time.monotonic() - started,
                    "completed_additional": completed, "skipped": skipped, "new_games": generated,
                    "remaining_to_cap": play_spec.games - completed})
        core.write_artifact(path / "logs" / f"invocation_{invocation}.json", log, replace=True)
    return path