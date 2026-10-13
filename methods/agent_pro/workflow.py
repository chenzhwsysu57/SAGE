"""Observable full-workflow coverage and disjoint learning/test call accounting.

Reports actual execution, not merely installed capabilities. A losing training
episode or improved candidate may not exist; do not force them to pass a gate.
"""

from collections import Counter
import json


def component_report(search, initial, selected, test_games=0):
    nodes = search["nodes"] if search else []
    events = Counter(e["event"] for e in search["events"]) if search else Counter()
    return {
        "learning_executed": search is not None,
        "training_episodes": sum(len(n["exploration"]) for n in nodes),
        "calibrations_and_proposals_completed": events["proposed"],
        "same_scenario_verifications": sum(len(n["verification"]) for n in nodes),
        "verification_passed": events["verified"],
        "dev_scored_policies": events["scored"],
        "dfs_nodes": len(nodes), "dfs_backtracks": events["backtrack"],
        "accepted_candidate_policies": sum(n["parent_node_id"] is not None and n["status"] == "accepted" for n in nodes),
        "selected_policy_changed": selected.instructions() != initial.instructions(),
        "frozen_test_games": test_games,
        "note": "Untriggered reflection or rejected policies are valid algorithm outcomes, not evidence those live components passed.",
    }


def call_costs(run_dir):
    """Count each journal request once; corrections inherit their original phase.

    A frozen evaluation identity never counts as learning. Cached requests count
    in original totals, not as new API calls. Unknown provider usage stays unknown.
    """
    mapping = run_dir / "logs/agent_pro_call_ids.json"
    totals = {k: {"requests": 0, "usage_totals": {}, "usage_unknown_requests": 0}
              for k in ("learning", "frozen_evaluation")}
    if not mapping.exists():
        return totals
    entries = json.loads(mapping.read_text())["calls"]
    if any(e["identity"].get("purpose", "").startswith("online_") for e in entries):
        totals = {k: {"requests": 0, "usage_totals": {}, "usage_unknown_requests": 0}
                  for k in ("online_main", "online_reflection", "online_verification")}
    if any(e["identity"].get("actor") == "opponent" for e in entries):
        totals.update({k: {"requests": 0, "usage_totals": {}, "usage_unknown_requests": 0}
                       for k in ("opponent_main", "opponent_verification")})
    for entry in entries:
        purpose = entry["identity"].get("purpose")
        phase = purpose if purpose in totals else "frozen_evaluation" if purpose == "frozen_evaluation" else "learning"
        if entry["identity"].get("actor") == "opponent":
            phase = "opponent_main" if purpose == "online_main" else "opponent_verification"
        record = json.loads((run_dir / "prompts/P0" /
                            f"g{entry['game_index']:04d}_s0000_decision.json").read_text())
        target = totals[phase]
        target["requests"] += 1
        usage = record.get("usage") or {}
        if "total_tokens" not in usage:
            target["usage_unknown_requests"] += 1
        for key, value in usage.items():
            target["usage_totals"][key] = target["usage_totals"].get(key, 0) + value
    return totals