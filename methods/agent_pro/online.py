"""Causal scored play with retrospective, bounded policy DFS between games.

No warm-up, dev panel or frozen test phase. Only the just-completed scene is
available to each update; the root payoff is reused, not played in advance.
Same-scene selection is an adaptation, NOT independent validation or evidence
of generalization. Diagnostic snapshots are never trusted as recovery state:
control flow is reconstructed through the existing audited episode/call caches.
"""

from copy import deepcopy
from pathlib import Path

from .optimizer import SearchConfig, _finite
from .policy import PolicyState
from .scenarios import Scenario, _atomic
from .fallback import contaminated, learning_failure
from .reflection import DuplicatePolicyInstructions

PROTOCOL = "agent_pro_online_completed_scene_v1"


def persist(path, value):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink online artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic(path, value)


class OnlinePolicyUpdater:
    """One completed scored episode -> bounded DFS -> next-game policy."""

    def __init__(self, evaluator, reflector, config, run_dir, progress=None):
        if not isinstance(config, SearchConfig):
            raise ValueError("Expected SearchConfig")
        self.evaluator, self.reflector, self.config = evaluator, reflector, config
        self.run_dir, self.progress = Path(run_dir), progress

    def update(self, policy, scenario, episode, game_index):
        expected = {"game": policy.game, "policy_id": policy.policy_id,
                    "scenario_id": scenario.scenario_id, "split": "online",
                    "player_id": scenario.player_id}
        if (scenario.split != "online" or any(episode.get(k) != v for k, v in expected.items())
                or not episode.get("records") or not episode.get("feedback")):
            raise ValueError("Online update requires a matching completed episode")
        net = _finite(episode["net_profit"])
        if contaminated(episode):
            raise ValueError("Contaminated source cannot drive online policy learning")
        if net >= 0:
            raise ValueError("Online update requires a loss")
        state = {"protocol": PROTOCOL, "after_game": game_index + 1,
                 "selection": "retrospective_same_scene_not_independent_dev",
                 "source_scenario_id": scenario.scenario_id, "root_policy": policy.to_dict(),
                 "root_net": net, "best_policy": policy.to_dict(), "best_net": net,
                 "nodes": [], "events": [], "status": "running"}
        path = self.run_dir / "metrics/online_updates" / f"game_{game_index + 1:04d}.json"

        def emit(event, **data):
            state["events"].append({"event": event, **data})
            persist(path, state)
            if self.progress:
                self.progress({"event": event, "game": game_index + 1, **data})

        root = {"node_id": "n0001", "parent_node_id": None, "depth": 0,
                "policy": policy.to_dict(), "net_profit": net, "status": "root"}
        state["nodes"].append(root)
        emit("update_started")

        def walk(parent_policy, parent_episode, parent):
            if (parent["depth"] >= self.config.max_depth or parent["net_profit"] >= 0
                    or len(state["nodes"]) >= self.config.max_nodes):
                return
            previous = []
            for branch in range(self.config.branches):
                for attempt in range(self.config.verification_retries + 1):
                    if len(state["nodes"]) >= self.config.max_nodes:
                        return
                    node = {"node_id": f"n{len(state['nodes']) + 1:04d}",
                            "parent_node_id": parent["node_id"], "depth": parent["depth"] + 1,
                            "status": "proposing", "policy": None, "net_profit": None}
                    state["nodes"].append(node)
                    scope = {"protocol": PROTOCOL, "item_index": game_index,
                             "node_id": node["node_id"], "branch_index": branch,
                             "proposal_index": attempt, "purpose": "online_reflection"}
                    emit("proposal_started", node_id=node["node_id"])
                    try:
                        candidate = self.reflector.propose(parent_policy, deepcopy(parent_episode), scope, deepcopy(previous))
                    except DuplicatePolicyInstructions:
                        node["status"] = "rejected"
                        emit("rejected", node_id=node["node_id"], reason="duplicate_policy")
                        # A duplicate is a valid model response that yielded no
                        # new search node. Keep the DFS alive and try the next
                        # branch/verification slot.
                        continue
                    except Exception as exc:
                        if (getattr(self.evaluator, "action_failure_policy", "strict") == "strict"
                                or not learning_failure(exc)):
                            raise
                        # Abort the entire update, including earlier accepted
                        # candidates. Never synthesize a policy after failure.
                        state.update(status="skipped_model_failure", best_policy=policy.to_dict(), best_net=net)
                        node["status"] = "reflection_failed"
                        emit("reflection_failed", node_id=node["node_id"], error_type=type(exc).__name__)
                        return
                    if (not isinstance(candidate, PolicyState) or candidate.game != policy.game
                            or candidate.parent_id != parent_policy.policy_id
                            or candidate.source_id != scenario.scenario_id
                            or candidate.instructions() == parent_policy.instructions()
                            or any(candidate.instructions() == PolicyState.from_dict(p).instructions() for p in previous)):
                        raise ValueError("Invalid online candidate lineage or duplicate policy")
                    previous.append(candidate.to_dict())
                    node.update(policy=candidate.to_dict(), status="verifying")
                    emit("proposed", node_id=node["node_id"])
                    verified = self.evaluator.evaluate(candidate, scenario,
                        {**scope, "purpose": "online_verification"})
                    if any(verified.get(k) != v for k, v in {**expected, "policy_id": candidate.policy_id}.items()):
                        raise ValueError("Verification changed online scene or player")
                    value = _finite(verified["net_profit"])
                    node["net_profit"] = value
                    if contaminated(verified):
                        node["status"] = "rejected"
                        emit("rejected", node_id=node["node_id"], reason="verification_contaminated")
                        continue
                    if value <= parent["net_profit"]:
                        node["status"] = "rejected"
                        emit("rejected", node_id=node["node_id"], reason="verification_not_improved")
                        continue
                    node["status"] = "accepted"
                    if value > state["best_net"]:
                        state.update(best_policy=candidate.to_dict(), best_net=value)
                    emit("accepted", node_id=node["node_id"])
                    walk(candidate, verified, node)
                    if state["status"] == "skipped_model_failure":
                        return
                    break
            emit("backtrack", node_id=parent["node_id"])

        try:
            walk(policy, episode, root)
            if state["status"] != "skipped_model_failure":
                state["status"] = "completed"
            emit("update_completed")
        except BaseException as exc:
            state.update(status="failed", error_type=type(exc).__name__)
            emit("update_failed")
            raise
        return PolicyState.from_dict(state["best_policy"]), deepcopy(state)


def play_online(initial, game, seed_list, evaluator, updater, run_dir, on_progress=None, position=None):
    """Instantiate each future scenario only after the preceding update finishes."""
    if position is not None and (type(position) is not int or position not in (0, 1)):
        raise ValueError("Expected fixed position 0/1 or None")
    policy, rows, updates = initial, [], []
    path = Path(run_dir) / "metrics/online.json"

    def report(status):
        nets = [r["net_profit"] for r in rows]
        value = {"protocol": PROTOCOL, "status": status, "rows": rows, "updates": updates,
                 "current_policy": policy.to_dict(), "games": len(rows),
                 "remaining": len(seed_list) - len(rows), "frozen": False,
                 "prelearning_games": 0, "total_net_profit": sum(nets),
                 "mean_net_profit": sum(nets) / len(nets) if nets else None,
                 "wins": sum(n > 0 for n in nets), "draws": sum(n == 0 for n in nets),
                 "losses": sum(n < 0 for n in nets)}
        persist(path, value)
        if on_progress:
            on_progress(deepcopy(value))
        return value

    report("running")
    try:
        for index, seed in enumerate(seed_list):
            scenario = Scenario(game, seed, index % 2 if position is None else position, False, "online")
            episode = evaluator.evaluate(policy, scenario,
                {"protocol": PROTOCOL, "purpose": "online_main", "item_index": index})
            row = {"game": index + 1, "scenario": scenario.to_dict(),
                   "policy_before": policy.to_dict(), "net_profit": episode["net_profit"],
                   "method_decisions": len(episode["records"]), "update_status": "pending"}
            if contaminated(episode):
                row["action_fallback_count"] = episode.get("action_fallback_count", 0)
                row["opponent_fallback_count"] = sum(
                    s.get("opponent_metadata", {}).get("source") == "random_fallback"
                    for s in episode.get("trajectory", {}).get("steps", []))
            rows.append(row)
            report("updating")  # Original score is durable BEFORE any extra call.
            if contaminated(episode):
                row["update_status"] = "skipped_contaminated_episode"
            elif index == len(seed_list) - 1:
                row["update_status"] = "skipped_final_game"
            elif episode["net_profit"] >= 0:
                row["update_status"] = "skipped_nonloss"
            else:
                policy, update = updater.update(policy, scenario, episode, index)
                updates.append({"after_game": index + 1, "nodes": len(update["nodes"]),
                    "proposals": sum(e["event"] == "proposed" for e in update["events"]),
                    "accepted": sum(e["event"] == "accepted" for e in update["events"]),
                    "policy_changed": policy.policy_id != row["policy_before"]["policy_id"]})
                row["update_status"] = update.get("status", "completed")
            row["policy_after_id"] = policy.policy_id
            report("running")
        return report("completed")
    except BaseException:
        report("failed")
        raise
