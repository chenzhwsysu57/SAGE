"""Tests for the parametric Leduc OOD opponent (theta tables + controllers)."""

from __future__ import annotations

import json
import math

from base.types import Action, Observation
from modeling.leduc.parametric_style import (
    CounterModelingController,
    ParametricDriftController,
    ParametricStyleAgent,
    estimate_theta_from_steps,
    estimate_theta_tables,
    farthest_style_map,
    farthest_two_style_map,
    interpolate_theta,
    make_theta,
    signature_feature_vector,
)
from modeling.leduc.statistical_profile import (
    StatisticalProfileMethod,
    _build_profile,
    _profile_likelihood,
)


def _obs(strength_card: str, facing_bet: bool) -> Observation:
    return Observation(
        player_id=0,
        private_info={"card": strength_card},
        public_info={},
        history=[],
        legal_actions=["fold", "call", "raise"] if facing_bet else ["check", "bet"],
    )


def _legal(facing_bet: bool) -> list[Action]:
    if facing_bet:
        return [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
            Action(player_id=0, type="raise", amount=6),
        ]
    return [
        Action(player_id=0, type="check"),
        Action(player_id=0, type="bet", amount=2),
    ]


def _steps_from_counts(counts: dict[str, dict[str, dict[str, int]]]) -> list[dict]:
    card_by_strength = {"weak": "J", "medium": "Q", "strong": "K"}
    steps: list[dict] = []
    for strength, contexts in counts.items():
        card = card_by_strength[strength]
        for action, n in contexts.items():
            for _ in range(n):
                steps.append({
                    "player_id": 0,
                    "action": f"P0 {action}s",
                    "state_before": {"private_cards": {"0": card}},
                })
    return steps


AGGRESSIVE_COUNTS = {
    "weak": {"check": 2, "bet": 8, "call": 3, "fold": 1, "raise": 6},
    "medium": {"check": 4, "bet": 6, "call": 4, "fold": 1, "raise": 5},
    "strong": {"check": 1, "bet": 9, "call": 2, "fold": 1, "raise": 7},
}

TIGHT_COUNTS = {
    "weak": {"check": 9, "bet": 1, "call": 1, "fold": 9, "raise": 0},
    "medium": {"check": 8, "bet": 2, "call": 2, "fold": 7, "raise": 1},
    "strong": {"check": 2, "bet": 8, "call": 4, "fold": 2, "raise": 4},
}


def test_make_theta_shape_and_normalization():
    theta = make_theta()
    for strength in ("weak", "medium", "strong"):
        assert abs(sum(theta[strength]["no_bet"].values()) - 1.0) < 1e-9
        assert abs(sum(theta[strength]["facing_bet"].values()) - 1.0) < 1e-9


def test_interpolate_theta_endpoints_and_monotonicity():
    a = make_theta()
    b = make_theta()
    b["strong"]["no_bet"]["bet"] = 1.0
    b["strong"]["no_bet"]["check"] = 0.0

    assert interpolate_theta(a, b, 0.0)["strong"]["no_bet"]["bet"] == a["strong"]["no_bet"]["bet"]
    assert interpolate_theta(a, b, 1.0)["strong"]["no_bet"]["bet"] == 1.0
    mid = interpolate_theta(a, b, 0.5)
    assert abs(mid["strong"]["no_bet"]["bet"] - 0.75) < 1e-9
    # monotone in lambda
    vals = [
        interpolate_theta(a, b, lam)["strong"]["no_bet"]["bet"]
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert vals == sorted(vals)


def test_parametric_agent_acts_legally_and_tracks_distribution():
    theta = make_theta()
    theta["strong"]["no_bet"] = {"check": 0.1, "bet": 0.9}
    agent = ParametricStyleAgent(theta=theta, name="OOD", seed=7)
    agent.theta_label = "lambda=0.5"
    bets = 0
    for _ in range(400):
        action, prompt, response, md = agent.act(_obs("K", facing_bet=False), _legal(False), None)
        assert action.type in ("check", "bet")
        assert action.amount == 2 if action.type == "bet" else True
        assert prompt == "" and response == ""
        assert md["theta_label"] == "lambda=0.5"
        bets += action.type == "bet"
    assert 0.75 <= bets / 400 <= 0.98

    # facing-bet context
    for _ in range(50):
        action, _, _, _ = agent.act(_obs("J", facing_bet=True), _legal(True), None)
        assert action.type in ("fold", "call", "raise")
        assert action.amount in (0, 2, 6)


def test_drift_controller_lambda_schedule():
    ctrl = ParametricDriftController(
        make_theta(), make_theta(),
        drift_ramp=30, drift_pre_stable=10, drift_post_stable=10,
    )
    # 10 stable games at A, 30-game ramp, 10 stable games at B (50 total)
    for g in range(1, 11):
        assert ctrl.lambda_for_game(g) == 0.0
    assert abs(ctrl.lambda_for_game(11) - 1 / 30) < 1e-9
    assert abs(ctrl.lambda_for_game(25) - 0.5) < 1e-9
    assert ctrl.lambda_for_game(40) == 1.0
    assert ctrl.lambda_for_game(50) == 1.0  # post-stable window

    # no stable windows -> ramp from game 1
    ctrl = ParametricDriftController(make_theta(), make_theta(), drift_ramp=40)
    assert abs(ctrl.lambda_for_game(1) - 1 / 40) < 1e-9
    assert ctrl.lambda_for_game(40) == 1.0

    late = ParametricDriftController(
        make_theta(), make_theta(), drift_ramp=10, drift_pre_stable=20,
    )
    assert late.lambda_for_game(20) == 0.0
    assert abs(late.lambda_for_game(25) - 0.5) < 1e-9
    assert late.lambda_for_game(30) == 1.0


def test_farthest_style_map_matches_brute_force():
    sigs = {
        "A": {"no_bet_check": 0.9, "no_bet_bet": 0.1, "fb_fold": 0.1,
              "fb_call": 0.8, "fb_raise": 0.1, "selectivity": 0.0, "entropy": 1.0},
        "B": {"no_bet_check": 0.1, "no_bet_bet": 0.9, "fb_fold": 0.8,
              "fb_call": 0.1, "fb_raise": 0.1, "selectivity": 0.5, "entropy": 0.5},
        "C": {"no_bet_check": 0.5, "no_bet_bet": 0.5, "fb_fold": 0.3,
              "fb_call": 0.3, "fb_raise": 0.4, "selectivity": -0.2, "entropy": 0.9},
    }
    m = farthest_style_map(sigs)
    for style in sigs:
        expected = max(
            (t for t in sigs if t != style),
            key=lambda t: math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(
                    signature_feature_vector(sigs[style]),
                    signature_feature_vector(sigs[t]),
                )
            )),
        )
        assert m[style] == expected
        assert m[style] != style


def test_farthest_two_style_map_orders_by_distance():
    sigs = {
        "A": {"no_bet_check": 0.9, "no_bet_bet": 0.1, "fb_fold": 0.1,
              "fb_call": 0.8, "fb_raise": 0.1, "selectivity": 0.0, "entropy": 1.0},
        "B": {"no_bet_check": 0.1, "no_bet_bet": 0.9, "fb_fold": 0.8,
              "fb_call": 0.1, "fb_raise": 0.1, "selectivity": 0.5, "entropy": 0.5},
        "C": {"no_bet_check": 0.5, "no_bet_bet": 0.5, "fb_fold": 0.3,
              "fb_call": 0.3, "fb_raise": 0.4, "selectivity": -0.2, "entropy": 0.9},
    }
    two = farthest_two_style_map(sigs)
    for style in sigs:
        ordered = sorted(
            (t for t in sigs if t != style),
            key=lambda t: -math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(
                    signature_feature_vector(sigs[style]),
                    signature_feature_vector(sigs[t]),
                )
            )),
        )
        assert two[style] == ordered[:2]
        assert two[style][0] == farthest_style_map(sigs)[style]


def test_counter_modeling_controller_continuous_switching():
    theta_by_style = {s: make_theta() for s in ("A", "B", "C")}
    sigs = {
        "A": {"no_bet_check": 0.9, "no_bet_bet": 0.1, "fb_fold": 0.1,
              "fb_call": 0.8, "fb_raise": 0.1, "selectivity": 0.0, "entropy": 1.0},
        "B": {"no_bet_check": 0.1, "no_bet_bet": 0.9, "fb_fold": 0.8,
              "fb_call": 0.1, "fb_raise": 0.1, "selectivity": 0.5, "entropy": 0.5},
        "C": {"no_bet_check": 0.5, "no_bet_bet": 0.5, "fb_fold": 0.3,
              "fb_call": 0.3, "fb_raise": 0.4, "selectivity": -0.2, "entropy": 0.9},
    }
    ctrl = CounterModelingController(
        theta_by_style, sigs, threshold=0.5, min_consecutive=2,
        second_prob=0.0, seed=1,
    )
    ctrl.set_start_style("A")

    belief = {"A": 0.9, "B": 0.05, "C": 0.05}
    ctrl.observe_belief(belief, 1)
    assert ctrl.current_style == "A"  # not yet 2 consecutive
    ctrl.observe_belief(belief, 2)
    target1 = ctrl.current_style
    assert target1 == farthest_style_map(sigs)["A"]
    assert ctrl._switched_at_game == 3  # theta applies from the next game

    # continuous: method re-models the new style -> cheater switches again
    ctrl.observe_belief({target1: 0.95}, 3)
    assert ctrl.current_style == target1  # window restarted, need 2 again
    ctrl.observe_belief({target1: 0.95}, 4)
    target2 = ctrl.current_style
    assert target2 != target1
    assert target2 in farthest_two_style_map(sigs)[target1]
    assert ctrl._switched_at_game == 5


def test_counter_modeling_controller_second_prob():
    theta_by_style = {s: make_theta() for s in ("A", "B", "C")}
    sigs = {
        "A": {"no_bet_check": 0.9, "no_bet_bet": 0.1, "fb_fold": 0.1,
              "fb_call": 0.8, "fb_raise": 0.1, "selectivity": 0.0, "entropy": 1.0},
        "B": {"no_bet_check": 0.1, "no_bet_bet": 0.9, "fb_fold": 0.8,
              "fb_call": 0.1, "fb_raise": 0.1, "selectivity": 0.5, "entropy": 0.5},
        "C": {"no_bet_check": 0.5, "no_bet_bet": 0.5, "fb_fold": 0.3,
              "fb_call": 0.3, "fb_raise": 0.4, "selectivity": -0.2, "entropy": 0.9},
    }
    ctrl = CounterModelingController(
        theta_by_style, sigs, threshold=0.5, min_consecutive=1,
        second_prob=1.0, seed=0,
    )
    ctrl.set_start_style("A")
    ctrl.observe_belief({"A": 0.9}, 1)
    assert ctrl.current_style in farthest_two_style_map(sigs)["A"]
    assert ctrl.current_style == farthest_two_style_map(sigs)["A"][1] or \
        ctrl.current_style == farthest_style_map(sigs)["A"]  # never self


def test_counter_modeling_controller_low_confidence_resets():
    ctrl = CounterModelingController(
        {s: make_theta() for s in ("A", "B")},
        {"A": {"no_bet_check": 0.9, "no_bet_bet": 0.1, "fb_fold": 0.1,
               "fb_call": 0.8, "fb_raise": 0.1, "selectivity": 0.0, "entropy": 1.0},
         "B": {"no_bet_check": 0.1, "no_bet_bet": 0.9, "fb_fold": 0.8,
               "fb_call": 0.1, "fb_raise": 0.1, "selectivity": 0.5, "entropy": 0.5}},
        threshold=0.5, min_consecutive=3,
    )
    ctrl.set_start_style("A")
    ctrl.observe_belief({"A": 0.9}, 1)
    ctrl.observe_belief({"A": 0.9}, 2)
    ctrl.observe_belief({"A": 0.4, "B": 0.35, "C": 0.25}, 3)  # argmax conf < threshold
    ctrl.observe_belief({"A": 0.9}, 4)
    ctrl.observe_belief({"A": 0.9}, 5)
    assert ctrl.current_style == "A"  # counter reset, need 3 consecutive again
    ctrl.observe_belief({"A": 0.9}, 6)
    assert ctrl.current_style != "A"


def test_estimate_theta_from_steps_counts_and_smoothing():
    steps = []
    for _ in range(9):
        steps.append({"player_id": 0, "action": "P0 bets 2",
                      "state_before": {"private_cards": {"0": "K"}}})
    for _ in range(1):
        steps.append({"player_id": 0, "action": "P0 checks",
                      "state_before": {"private_cards": {"0": "K"}}})
    theta = estimate_theta_from_steps(steps, 0, smoothing=0.0)
    assert abs(theta["strong"]["no_bet"]["bet"] - 0.9) < 1e-9
    assert abs(theta["strong"]["no_bet"]["check"] - 0.1) < 1e-9
    # unknown strength (missing card) is skipped, not counted
    steps.append({"player_id": 0, "action": "P0 bets 2",
                  "state_before": {"private_cards": {"0": ""}}})
    theta2 = estimate_theta_from_steps(steps, 0, smoothing=0.0)
    assert abs(theta2["strong"]["no_bet"]["bet"] - 0.9) < 1e-9


def test_theta_round_trip_classification_separates_styles():
    steps_a = _steps_from_counts(AGGRESSIVE_COUNTS)
    steps_b = _steps_from_counts(TIGHT_COUNTS)

    theta_a = estimate_theta_from_steps(steps_a, 0, smoothing=0.0)
    theta_b = estimate_theta_from_steps(steps_b, 0, smoothing=0.0)
    assert theta_a["weak"]["facing_bet"]["raise"] > theta_b["weak"]["facing_bet"]["raise"]
    assert theta_a["weak"]["facing_bet"]["fold"] < theta_b["weak"]["facing_bet"]["fold"]

    def profile_from_steps(steps: list[dict]):
        actions = []
        strengths = []
        for s in steps:
            raw = s["action"].split()[1]
            actions.append(raw.rstrip("s") if raw.endswith("s") else raw)
            card = s["state_before"]["private_cards"]["0"]
            strengths.append(
                "weak" if card == "J" else "medium" if card == "Q" else "strong"
            )
        return _build_profile(actions, strengths)

    sigs = StatisticalProfileMethod.estimate_signatures({
        "A": profile_from_steps(steps_a),
        "B": profile_from_steps(steps_b),
    })
    for steps, expected in ((steps_a, "A"), (steps_b, "B")):
        likelihood = _profile_likelihood(profile_from_steps(steps), sigs)
        assert max(likelihood, key=lambda k: likelihood[k]) == expected


def test_counter_theta_after_switch_uses_farthest_style():
    theta_a = make_theta()
    theta_b = make_theta()
    theta_c = make_theta()
    theta_b["strong"]["no_bet"]["bet"] = 1.0
    theta_b["strong"]["no_bet"]["check"] = 0.0
    sigs = {
        "A": {"no_bet_check": 0.9, "no_bet_bet": 0.1, "fb_fold": 0.1,
              "fb_call": 0.8, "fb_raise": 0.1, "selectivity": 0.0, "entropy": 1.0},
        "B": {"no_bet_check": 0.1, "no_bet_bet": 0.9, "fb_fold": 0.8,
              "fb_call": 0.1, "fb_raise": 0.1, "selectivity": 0.5, "entropy": 0.5},
        "C": {"no_bet_check": 0.5, "no_bet_bet": 0.5, "fb_fold": 0.3,
              "fb_call": 0.3, "fb_raise": 0.4, "selectivity": -0.2, "entropy": 0.9},
    }
    ctrl = CounterModelingController(
        {"A": theta_a, "B": theta_b, "C": theta_c},
        sigs,
        threshold=0.5, min_consecutive=1, second_prob=0.0,
    )
    ctrl.set_start_style("A")
    ctrl.observe_belief({"A": 0.9}, 1)
    target = ctrl.current_style
    assert target == farthest_two_style_map(sigs)["A"][0]
    theta_used = ctrl.theta_for_game(2)
    assert theta_used is ctrl.theta_by_style[target]


def test_estimate_theta_tables_ingests_p0_and_p1_batches(tmp_path):
    """Regression: labeled P1 batches must count toward the archetype's theta
    using the batch's own player id (mirrors build_empirical_signature_bundle)."""
    train_path = tmp_path / "train.json"
    traj_p0 = {
        "steps": [
            {"player_id": 0, "action": "P0 checks",
             "private_cards": {"0": "JS", "1": "QS"}},
            {"player_id": 0, "action": "P0 checks",
             "private_cards": {"0": "JS", "1": "QS"}},
        ]
    }
    traj_p1 = {
        "steps": [
            {"player_id": 1, "action": "P1 raises 4",
             "private_cards": {"0": "KS", "1": "JH"}},
            {"player_id": 1, "action": "P1 raises 4",
             "private_cards": {"0": "KS", "1": "JH"}},
        ]
    }
    train_path.write_text(json.dumps({
        "batches": [
            {"label": "StyleA_vs_X_P0", "sample_trajectories": [traj_p0]},
            {"label": "ToM_vs_StyleA_P0", "sample_trajectories": [traj_p1]},
        ]
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "entries": {"StyleA": {"train_json": str(train_path), "test_json": None}}
    }))
    thetas, meta = estimate_theta_tables(str(manifest_path), smoothing=0.0)
    theta = thetas["StyleA"]
    assert abs(theta["weak"]["no_bet"]["check"] - 1.0) < 1e-9  # from P0 batch
    assert theta["weak"]["facing_bet"]["raise"] > 0.9          # from P1 batch
    assert meta["archetypes"]["StyleA"]["games"] == 2
