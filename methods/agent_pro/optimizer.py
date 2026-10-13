"""Bounded DFS over prompt policies, not actions (Agent-Pro adaptation).

A parent fixes its first losing training scenario for all sibling proposals.
Children restart the ordered training scan; repeated sources are permitted, but
depth and *all proposal attempts* are bounded. Verification retries regenerate
policies, never retry failed model calls. Every verified child sees the identical
ordered dev panel; only strict parent improvement permits descent/global-best
selection. The paper's zero-sum delta is twice mean net profit.

search.json is an atomic diagnostic snapshot, NEVER a recovery cache. optimize
always replays from root with deterministic identities; the evaluator and model
ledger own validated episode/call replay. The CLI must pin run configuration and
source versions and supply a single shared, budgeted ledger. This module neither
configures a provider nor reads credentials. Serial, single-writer use only.

progress receives a defensive copy of each persisted event (``event`` names such
as node_created, episode_completed, accepted, rejected, backtrack, error). It can
count node_created or episode_completed for a CLI progress bar. Diagnostic logs
contain policy text and episode IDs/net/record counts, never episode trajectories.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile

from .policy import PolicyState, canonical, digest
from .scenarios import Scenario, validate_splits
from .fallback import contaminated, learning_failure


class _KnownReflectionStop(RuntimeError):
    """Internal stop: preserve initial policy, not an accepted random policy."""


@dataclass(frozen=True)
class SearchConfig:
    branches: int = 2
    max_depth: int = 2
    max_nodes: int = 7
    verification_retries: int = 3

    def __post_init__(self):
        # Explicit safety ceilings, not recommended experimental settings.
        for name, low, high in (("branches", 1, 64), ("max_depth", 0, 32),
                                ("max_nodes", 1, 10000), ("verification_retries", 0, 32)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}], not bool")

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(value) -> float:
    if type(value) not in (int, float):
        raise ValueError("Expected finite numeric net profit")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("Expected finite numeric net profit") from None
    if not math.isfinite(result):
        raise ValueError("Expected finite numeric net profit")
    return result


class DFSPolicyOptimizer:
    def __init__(self, evaluator, reflector, config: SearchConfig, run_dir: Path, progress=None):
        if not isinstance(config, SearchConfig):
            raise ValueError("Expected SearchConfig")
        if not callable(getattr(evaluator, "evaluate", None)) or not callable(getattr(reflector, "propose", None)):
            raise ValueError("Expected evaluator.evaluate and reflector.propose")
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable")
        self.evaluator = evaluator
        self.reflector = reflector
        self.config = config
        self.run_dir = Path(run_dir).absolute()
        self.progress = progress

    def _save(self):
        path = self.run_dir / "metrics" / "search.json"
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Symlink search snapshot path")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".search-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(canonical(self._state))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _emit(self, kind, *, notify=True, **data):
        event = {"event_index": len(self._state["events"]), "event": kind, **data}
        self._state["events"].append(event)
        self._save()
        if notify and self.progress is not None:
            self.progress(deepcopy(event))

    def _identity(self, node, purpose, item_index=0, **extra):
        return {"search_id": self._state["search_id"], "node_id": node["node_id"],
                "purpose": purpose, "item_index": item_index, **extra}

    def _new_node(self, parent, depth, branch_index=None, proposal_index=None, policy=None):
        if len(self._state["nodes"]) >= self.config.max_nodes:
            raise RuntimeError("Internal node budget violation")
        node = {"node_id": f"n{len(self._state['nodes']) + 1:04d}",
                "parent_node_id": None if parent is None else parent["node_id"],
                "depth": depth, "branch_index": branch_index, "proposal_index": proposal_index,
                "policy": None if policy is None else policy.to_dict(), "status": "created",
                "mean_net": None, "delta": None, "dev": [], "exploration": [],
                "verification": [], "source": None, "stop_reason": None}
        self._state["nodes"].append(node)
        self._active = {"node_id": node["node_id"], "purpose": "reflection" if parent else "dev"}
        self._emit("node_created", node_id=node["node_id"], parent_node_id=node["parent_node_id"])
        return node

    def _evaluate(self, policy, scenario, node, purpose, item_index):
        identity = self._identity(node, purpose, item_index)
        self._active = identity
        self._emit("evaluation_started", **identity, scenario_id=scenario.scenario_id)
        episode = self.evaluator.evaluate(policy, scenario, deepcopy(identity))
        expected = {"game": policy.game, "policy_id": policy.policy_id,
                    "scenario_id": scenario.scenario_id, "split": scenario.split,
                    "player_id": scenario.player_id}
        if (type(episode) is not dict or any(episode.get(k) != v for k, v in expected.items())
                or type(episode.get("player_id")) is not int
                or type(episode.get("records")) is not list):
            raise ValueError("Evaluator returned inconsistent episode metadata")
        net = _finite(episode.get("net_profit"))
        summary = {**expected, "item_index": item_index, "net_profit": net,
                   "records_count": len(episode["records"])}
        if contaminated(episode):
            summary["contaminated"] = True
        node[purpose].append(summary)
        self._emit("episode_completed", **identity, episode=summary)
        return episode, net

    def _score(self, policy, node):
        values = [self._evaluate(policy, scenario, node, "dev", i)[1]
                  for i, scenario in enumerate(self._dev)]
        if any(e.get("contaminated") for e in node["dev"]):
            self._emit("dev_contaminated", node_id=node["node_id"])
            return None
        # Divide first to avoid overflowing the sum of otherwise finite scores.
        score = _finite(math.fsum(v / len(values) for v in values))
        node.update(mean_net=score, delta=_finite(2 * score))
        self._emit("scored", node_id=node["node_id"], mean_net=score, delta=node["delta"])
        return score

    def _best(self, policy, node):
        if self._state["best_score"] is None or node["mean_net"] > self._state["best_score"]:
            self._state.update(best_policy=policy.to_dict(), best_score=node["mean_net"],
                               best={"node_id": node["node_id"], "policy": policy.to_dict(),
                                     "score": node["mean_net"], "delta": node["delta"]})
            self._emit("best_updated", node_id=node["node_id"], mean_net=node["mean_net"])

    def _at_budget(self, node):
        if len(self._state["nodes"]) < self.config.max_nodes:
            return False
        self._budget_hit = True
        node["stop_reason"] = "node_limit"
        self._emit("branch_terminated", node_id=node["node_id"], reason="node_limit")
        return True

    def _leave(self, node, reason):
        node["stop_reason"] = reason
        self._state["stack"].pop()
        self._emit("backtrack", node_id=node["node_id"],
                   parent_node_id=node["parent_node_id"], reason=reason)

    def _walk(self, policy, node):
        frame = {"node_id": node["node_id"], "branch_index": None,
                 "proposal_index": None, "source_id": None}
        self._state["stack"].append(frame)
        self._emit("enter", node_id=node["node_id"])
        if node["depth"] >= self.config.max_depth:
            self._leave(node, "depth_limit")
            return
        if self._at_budget(node):
            self._leave(node, "node_limit")
            return
        source = None
        for item_index, scenario in enumerate(self._train):
            episode, net = self._evaluate(policy, scenario, node, "exploration", item_index)
            if contaminated(episode):
                self._emit("training_contaminated", node_id=node["node_id"], item_index=item_index)
                continue
            if net < 0:
                source = (item_index, scenario, episode, net)
                node["source"] = deepcopy(node["exploration"][-1])
                frame["source_id"] = scenario.scenario_id
                self._emit("source_selected", node_id=node["node_id"], source=node["source"])
                break
            self._emit("training_skipped", node_id=node["node_id"], item_index=item_index, net_profit=net)
        if source is None:
            self._leave(node, "no_training_failure")
            return
        item_index, scenario, episode, source_net = source
        previous = []  # All earlier proposals of THIS parent, including siblings.
        for branch_index in range(self.config.branches):
            frame["branch_index"] = branch_index
            for proposal_index in range(self.config.verification_retries + 1):
                frame["proposal_index"] = proposal_index
                if self._at_budget(node):
                    self._leave(node, "node_limit")
                    return
                child = self._new_node(node, node["depth"] + 1, branch_index, proposal_index)
                child["source"] = deepcopy(node["source"])
                child["status"] = "proposing"
                identity = self._identity(child, "reflection", item_index,
                                          branch_index=branch_index, proposal_index=proposal_index)
                self._active = identity
                self._emit("proposal_started", **identity)
                try:
                    candidate = self.reflector.propose(policy, deepcopy(episode), deepcopy(identity), deepcopy(previous))
                except Exception as exc:
                    if (getattr(self.evaluator, "action_failure_policy", "strict") == "strict"
                            or not learning_failure(exc)):
                        raise
                    child["status"] = "reflection_failed"
                    self._emit("reflection_failed", node_id=child["node_id"], error_type=type(exc).__name__)
                    raise _KnownReflectionStop() from None
                if not isinstance(candidate, PolicyState):
                    raise ValueError("Reflector must return PolicyState")
                child["policy"] = candidate.to_dict()
                if (candidate.game != policy.game or candidate.parent_id != policy.policy_id
                        or candidate.source_id != scenario.scenario_id
                        or candidate.instructions() == policy.instructions()
                        or any(candidate.instructions() == p.instructions() for p in
                               (PolicyState.from_dict(old) for old in previous))):
                    raise ValueError("Invalid candidate lineage or duplicate instructions")
                previous.append(candidate.to_dict())
                child["status"] = "verifying"
                self._emit("proposed", node_id=child["node_id"], policy_id=candidate.policy_id)
                verified, verified_net = self._evaluate(candidate, scenario, child, "verification", item_index)
                if contaminated(verified):
                    child.update(status="rejected", stop_reason="verification_contaminated")
                    self._emit("rejected", node_id=child["node_id"], reason=child["stop_reason"])
                    continue
                if verified_net <= source_net:
                    child.update(status="rejected", stop_reason="verification_not_improved")
                    self._emit("rejected", node_id=child["node_id"], reason=child["stop_reason"],
                               source_net=source_net, candidate_net=verified_net)
                    if proposal_index == self.config.verification_retries:
                        self._emit("branch_exhausted", node_id=node["node_id"], branch_index=branch_index)
                    continue
                child["status"] = "verified"
                self._emit("verified", node_id=child["node_id"], source_net=source_net, candidate_net=verified_net)
                score = self._score(candidate, child)
                if score is None or score <= node["mean_net"]:
                    child.update(status="rejected", stop_reason="dev_contaminated" if score is None else "dev_not_improved")
                    self._emit("rejected", node_id=child["node_id"], reason=child["stop_reason"],
                               parent_score=node["mean_net"], candidate_score=score)
                else:
                    child["status"] = "accepted"
                    self._emit("accepted", node_id=child["node_id"], parent_score=node["mean_net"], candidate_score=score)
                    self._best(candidate, child)
                    self._walk(candidate, child)  # Finish the subtree BEFORE next sibling.
                break  # Dev rejection does not regenerate within this branch.
        self._leave(node, "siblings_exhausted")

    def optimize(self, initial: PolicyState, train: list[Scenario], dev: list[Scenario]) -> dict:
        """Replay a complete bounded search; propagate errors with a safe snapshot.

        Result/snapshot includes full best_policy, float best_score/root_score,
        nodes (including failed proposal slots), events, active stack and config.
        termination is node_limit, depth_limit, no_training_failure or
        search_exhausted; failures persist termination=error and re-raise.
        """
        if not isinstance(initial, PolicyState):
            raise ValueError("Expected initial PolicyState")
        if type(train) is not list or type(dev) is not list or not train or not dev:
            raise ValueError("Nonempty training and development lists are required")
        if any(not isinstance(s, Scenario) or s.game != initial.game for s in [*train, *dev]):
            raise ValueError("All scenarios must match the policy game")
        validate_splits(train, dev)
        self._train, self._dev = tuple(train), tuple(dev)
        panels = {name: [s.to_dict() for s in panel] for name, panel in (("train", train), ("dev", dev))}
        inputs = {"initial": initial.to_dict(), "panels": panels, "config": self.config.to_dict()}
        self._state = {"schema_version": 1, "phase": "dev_debug", "config": self.config.to_dict(),
                       "search_id": digest(inputs), "panels": panels, "best": None,
                       "best_policy": None, "best_score": None, "root_score": None, "root_delta": None,
                       "nodes": [], "events": [], "stack": [], "termination": "running"}
        self._budget_hit = False
        self._active = {"purpose": "dev"}
        try:
            self._emit("search_started")
            root = self._new_node(None, 0, policy=initial)
            self._state["root_score"] = self._score(initial, root)
            if self._state["root_score"] is None:
                self._state.update(best_policy=initial.to_dict(), termination="contaminated_root_dev")
                root["status"] = "contaminated"
                self._emit("search_completed", termination="contaminated_root_dev")
                return deepcopy(self._state)
            self._state["root_delta"] = root["delta"]
            root["status"] = "accepted"
            self._best(initial, root)
            self._walk(initial, root)
            reason = root["stop_reason"]
            self._state["termination"] = ("node_limit" if self._budget_hit else reason if reason in
                                          ("depth_limit", "no_training_failure") else "search_exhausted")
            self._emit("search_completed", termination=self._state["termination"])
            return deepcopy(self._state)
        except _KnownReflectionStop:
            self._state.update(best_policy=initial.to_dict(), best_score=self._state["root_score"],
                               best=None, termination="skipped_model_failure", stack=[])
            self._emit("search_completed", termination="skipped_model_failure")
            return deepcopy(self._state)
        except BaseException as exc:
            self._state["termination"] = "error"
            error = {"exception_type": type(exc).__name__, "operation": deepcopy(self._active)}
            self._state["error"] = error
            for node in self._state["nodes"]:
                if node["node_id"] == self._active.get("node_id"):
                    node.update(status="error", error=deepcopy(error))
            # Never serialize exception messages, traceback, or arbitrary model objects.
            # If storage itself fails, preserve the original exception and last durable snapshot.
            try:
                self._emit("error", notify=False, **error)
            except (OSError, ValueError):
                pass
            raise