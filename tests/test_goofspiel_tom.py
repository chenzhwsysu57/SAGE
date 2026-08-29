import json

import pytest

from base.case_memory import CASE_MEMORY_ACTIVE_IDS_KEY, CASE_MEMORY_BANK_KEY
from base.memory import BaseMemory
from base.types import Action, Observation, Trajectory, TrajectoryStep
from methods.goofspiel_tom import GoofspielToMMethod


class DummyModel:
    def generate(self, prompt: str) -> str:
        return "{}"


@pytest.fixture
def bayes_model_path(tmp_path):
    personas = ["Greedy", "Conservative", "Sacrificial", "Random", "GTO", "Mirror"]
    models = {
        persona: {
            str(prize): {str(bid): 0.2 for bid in range(1, 6)}
            for prize in range(1, 6)
        }
        for persona in personas
    }
    path = tmp_path / "bayes_model.json"
    path.write_text(json.dumps({"personas": personas, "N": 5, "models": models}))
    return str(path)


def _make_method(bayes_model_path):
    return GoofspielToMMethod(
        memory=BaseMemory(), use_cfr=False, bayes_model_path=bayes_model_path
    )


def _obs(player_id: int, round_no: int) -> Observation:
    return Observation(
        player_id=player_id,
        public_info={"round": round_no},
        history=[],
        rules_text="rules",
        state_text=f"Round {round_no}",
    )


def test_goofspiel_tom_parses_numeric_predicted_bid(bayes_model_path) -> None:
    method = _make_method(bayes_model_path)

    assert method._infer_predicted_action_label("4") == "4"
    assert method._infer_predicted_action_label("bid 4") == "4"
    assert method._infer_predicted_action_label("Bid 5.") == "5"


def test_goofspiel_recent_belief_lines_are_strict_game_records(bayes_model_path) -> None:
    method = _make_method(bayes_model_path)
    method._belief_history[1] = [
        {
            "game": i + 1,
            "rounds": i + 2,
            "beliefs": {
                "Greedy": 0.2 + i * 0.01,
                "GTO": 0.5,
                "Mirror": 0.3 - i * 0.01,
            },
        }
        for i in range(6)
    ]

    lines = method._recent_belief_lines(1)
    text = "\n".join(lines)

    assert len(lines) == 5
    assert lines[0].startswith("game 6:")
    assert lines[-1].startswith("game 2:")
    assert "game 1:" not in text
    assert "(6 rounds)" not in text
    assert "(2 rounds)" not in text
    assert "Most possible" not in text


def test_goofspiel_opponent_model_section_omits_persona_estimate_summary(bayes_model_path) -> None:
    method = _make_method(bayes_model_path)
    method._all_records[1] = [{"round": i + 1, "bid": i + 1} for i in range(4)]
    method._posterior_cache[1] = {
        "Greedy": 0.306,
        "GTO": 0.229,
        "Random": 0.126,
        "Conservative": 0.118,
        "Sacrificial": 0.117,
        "Mirror": 0.104,
    }
    method._belief_history[1] = [{
        "game": 4,
        "beliefs": method._posterior_cache[1],
    }]

    text = "\n".join(method._opponent_model_section(1))

    assert "Recent belief history (newest first):" in text
    assert "game 4: Greedy 30.6%, GTO 22.9%" in text
    assert "Opponent persona estimate" not in text
    assert "observed bids" not in text
    assert "← most likely" not in text


def test_goofspiel_self_evolve_reflect_matches_same_round_opponent_bid(bayes_model_path) -> None:
    memory = BaseMemory()
    method = GoofspielToMMethod(
        memory=memory, use_cfr=False, bayes_model_path=bayes_model_path
    )
    p0_step = TrajectoryStep(
        step=1,
        player_id=0,
        observation=_obs(0, 3),
        rendered_prompt="",
        raw_llm_response="",
        parsed_action=Action(player_id=0, type="bid", amount=5),
        method_data={"self_evolve": {"memory_records": 0}},
    )
    p1_step = TrajectoryStep(
        step=2,
        player_id=1,
        observation=_obs(1, 3),
        rendered_prompt="",
        raw_llm_response="",
        parsed_action=Action(player_id=1, type="bid", amount=4),
    )
    trajectory = Trajectory(
        game_name="Goofspiel",
        players=["ToM", "Mirror"],
        steps=[p0_step, p1_step],
    )

    method.reflect(trajectory, DummyModel(), "2")

    records = memory.get("goofspiel_self_evolve_records", [])
    assert records == []


def test_goofspiel_self_evolve_reflect_marks_wrong_bid_failure(bayes_model_path) -> None:
    memory = BaseMemory()
    method = GoofspielToMMethod(
        memory=memory, use_cfr=False, bayes_model_path=bayes_model_path
    )
    p0_step = TrajectoryStep(
        step=1,
        player_id=0,
        observation=_obs(0, 2),
        rendered_prompt="",
        raw_llm_response="",
        parsed_action=Action(player_id=0, type="bid", amount=5),
        method_data={"self_evolve": {"memory_records": 0}},
    )
    p1_step = TrajectoryStep(
        step=2,
        player_id=1,
        observation=_obs(1, 2),
        rendered_prompt="",
        raw_llm_response="",
        parsed_action=Action(player_id=1, type="bid", amount=4),
    )
    trajectory = Trajectory(
        game_name="Goofspiel",
        players=["ToM", "Mirror"],
        steps=[p0_step, p1_step],
    )

    method.reflect(trajectory, DummyModel(), "2")

    records = memory.get("goofspiel_self_evolve_records", [])
    assert records == []
    assert method._self_evolve_memory_text(0) == ""

    memory.set(CASE_MEMORY_BANK_KEY, [
        {
            "memory_id": "001_0_step003",
            "game": "goofspiel",
            "position": "P0",
            "memory_status": "memory",
            "suggested_action": "bid_4",
            "opponent_action_consistency": "consistent",
            "quality_flags": [],
            "memory": "In round 2 with prize 2, bidding 4 may work better because it blocks Mirror.",
            "source_case": {"similarity": 0.94},
        },
        {
            "memory_id": "002_0_step003",
            "game": "goofspiel",
            "position": "P0",
            "memory_status": "memory",
            "suggested_action": "bid_4",
            "opponent_action_consistency": "consistent",
            "quality_flags": [],
            "memory": "In a similar prize-2 spot, bidding 4 may work better because it protects later high cards.",
            "source_case": {"similarity": 0.87},
        },
        {
            "memory_id": "003_0_step003",
            "game": "goofspiel",
            "position": "P0",
            "memory_status": "memory",
            "suggested_action": "bid_4",
            "opponent_action_consistency": "consistent",
            "quality_flags": [],
            "memory": "In a nearby scoring sequence, bidding 4 may work better because it maintains control.",
            "source_case": {"similarity": 0.79},
        },
    ])
    memory.set(CASE_MEMORY_ACTIVE_IDS_KEY, [
        "001_0_step003",
        "002_0_step003",
        "003_0_step003",
    ])
    memory_text = method._self_evolve_memory_text(0)
    assert "#### SELF EVOLVE MEMORY ####" in memory_text
    assert "complete battle case" in memory_text
    assert "suggested action" not in memory_text
    assert "In round 2 with prize 2" in memory_text
    assert "case 3:" in memory_text
    assert "trajectory before predict:" not in memory_text
    assert "result: failed" not in memory_text
    assert "observed:" not in memory_text
    assert "note:" not in memory_text

    class PromptModel:
        def generate(self, prompt: str) -> str:
            return '{"reasoning": "memory-guided action", "action": {"type": "bid", "amount": 4}}'

    _, third_prompt, _, _ = method.act(
        _obs(0, 2),
        [Action(player_id=0, type="bid", amount=4)],
        PromptModel(),
    )
    assert third_prompt.index("#### SELF EVOLVE MEMORY ####") < third_prompt.index("Reason in this order:")
    assert third_prompt.index("#### SELF EVOLVE MEMORY ####") < third_prompt.index("Legal actions:")
