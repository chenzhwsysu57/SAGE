from __future__ import annotations

from base.memory import BaseMemory
from base.types import Action, Observation, Trajectory, TrajectoryStep
from analysis.memory_case_reflection_batch import (
    _build_triggered_requests,
    extract_memory,
)
from base.case_memory import CASE_MEMORY_ACTIVE_IDS_KEY, CASE_MEMORY_BANK_KEY
from base.case_memory_runtime import (
    _select_archive_case,
    append_step_case_to_archive,
    append_trajectory_cases_to_archive,
    refresh_case_memory_bank_for_step,
    refresh_case_memory_bank_from_trajectory,
)


def _case(similarity: float, game_id: int, step: int) -> dict:
    return {
        "public_prefix": {"similarity": similarity},
        "game_id": game_id,
        "step": step,
        "legal_actions": ["fold", "call", "raise"],
        "future_history": [{"action": "call"}],
        "opponent_estimate": {"beliefs": {"bluffer": 0.7}},
        "actioned": "call",
        "outcome": {"reward": 1},
    }


def test_extract_memory_structured_reasoning_template() -> None:
    response = """Situation: flop, my hand K, board Q, opponent bet 4 after a preflop raise, pot 10
Opponent behavior in source case: opponent bet and re-raised, consistent with the LAG estimate
Possible opponent patterns in similar spots: re-raise with value; may fold to pressure
If replayed, what to reason about: if opponent re-raises again on a Q-high board, their range is value-heavy
Transferability caveats: history length and opponent style may differ in other cases
"""
    result = extract_memory(response)
    assert result["status"] == "memory"
    assert result["memory"].startswith("Situation: flop, my hand K")
    assert "If replayed, what to reason about" in result["memory"]


def _leduc_feature(hand_rank: str, *, board_rank: str = "Q") -> dict:
    return {
        "game": "leduc",
        "history": [],
        "legal": ["check", "bet"],
        "stage": (1, "flop"),
        "hand_rank": hand_rank,
        "board_rank": board_rank,
        "pair": False,
        "to_call": 0,
        "raises": 0,
        "pot": 6,
    }


def _archive_case(game_id: int, step: int, feature: dict) -> dict:
    return {
        "_archive_key": ("leduc", 0, game_id, step),
        "game": "leduc",
        "position": 0,
        "condition": "tom_cfr",
        "context": {
            "source_game": "leduc",
            "source_condition": "tom_cfr",
            "source_position": "P0",
            "source_trajectory_dir": None,
            "current_game_id": game_id,
            "current_step": step,
        },
        "feature": feature,
        "case": {
            "public_prefix": {
                "similarity": 1.0,
                "stage": [1, "flop"],
                "board": "Q",
                "to_call": 0,
                "raises": 0,
            },
            "opponent_estimate": {
                "source": "opponent_belief_module.opp_1",
                "n_actions": 2,
                "action_counts": {"check": 1, "call": 1},
                "action_distribution": {"check": 0.5, "call": 0.5},
            },
            "game_id": game_id,
            "step": step,
            "history": [],
            "legal_actions": ["check", "bet"],
            "actioned": {"type": "check", "amount": 0},
            "future_history": [
                {"action": {"actor": "opponent", "value": {"type": "check", "amount": 0}}}
            ],
            "outcome": {"phase": "terminal", "reward": 0},
        },
    }


def _leduc_state_before() -> dict:
    return {
        "private_cards": {"0": "K", "1": "?"},
        "round_bets": {"0": 2, "1": 2},
        "street": 1,
        "phase": "flop",
        "pot": 6,
        "round_raises": 0,
        "community_card": "Q",
    }


def _gradient_archive() -> list[dict]:
    # Current flop feature has hand K; K/K/Q past features give top-1=1.0,
    # top-3=0.879 and a 0.121 gradient, so the top-3 trigger passes.
    return [
        _archive_case(1, 3, _leduc_feature("K")),
        _archive_case(2, 3, _leduc_feature("K")),
        _archive_case(3, 3, _leduc_feature("Q")),
    ]


def test_triggered_requests_keep_gradient_cases_only() -> None:
    sim_cases = {
        "series": [
            {
                "game": "leduc",
                "condition": "full",
                "position": "P0",
                "trajectory_dir": "results/leduc/full/P0",
                "case_sets": [
                    {
                        "current_game_id": 2,
                        "current_step": 4,
                        "cases": [
                            _case(0.95, 1, 3),
                            _case(0.88, 1, 1),
                            _case(0.81, 1, 2),
                        ],
                    },
                    {
                        "current_game_id": 3,
                        "current_step": 2,
                        "cases": [
                            _case(1.0, 1, 3),
                            _case(1.0, 1, 1),
                            _case(1.0, 1, 2),
                        ],
                    },
                ],
            }
        ]
    }

    requests = _build_triggered_requests(
        sim_cases,
        games={"leduc"},
        require_opponent_estimate=True,
    )

    assert len(requests) == 1
    assert requests[0]["case"]["game_id"] == 1
    assert requests[0]["case"]["step"] == 3


def test_refresh_helper_accumulates_all_current_cases_into_archive() -> None:
    memory = BaseMemory()
    archive: list[dict] = []
    trajectory = Trajectory(
        game_name="Leduc",
        players=["P0", "P1"],
        steps=[
            TrajectoryStep(
                step=1,
                player_id=0,
                observation=Observation(
                    player_id=0,
                    legal_actions=["check", "bet"],
                    rules_text="rules",
                    state_text="state",
                ),
                rendered_prompt="",
                raw_llm_response="",
                parsed_action=Action(player_id=0, type="check"),
                state_before={
                    "private_cards": {"0": "Q", "1": "J"},
                    "round_bets": {"0": 0, "1": 0},
                    "street": 0,
                    "phase": "preflop",
                    "pot": 2,
                    "round_raises": 0,
                },
                state_after={
                    "private_cards": {"0": "Q", "1": "J"},
                    "round_bets": {"0": 0, "1": 0},
                    "street": 0,
                    "phase": "preflop",
                    "pot": 2,
                    "round_raises": 0,
                },
            )
        ],
    )

    result = refresh_case_memory_bank_from_trajectory(
        game="leduc",
        trajectory=trajectory,
        position=0,
        current_game_id=1,
        memory=memory,
        archive=archive,
        live_api=False,
    )

    assert result["selected"] == 0
    assert len(archive) == 1
    assert memory.get("case_based_self_evolve_refresh_keys", []) == []


def test_step_archive_entry_is_finalized_in_place() -> None:
    archive: list[dict] = []
    state_before = {
        "private_cards": {"0": "Q", "1": "J"},
        "round_bets": {"0": 0, "1": 0},
        "street": 0,
        "phase": "preflop",
        "pot": 2,
        "round_raises": 0,
        "history": [],
    }
    append_step_case_to_archive(
        game="leduc",
        state_before=state_before,
        legal_actions=["check", "bet"],
        position=0,
        current_game_id=1,
        current_step=1,
        archive=archive,
        condition="tom_cfr",
    )

    assert len(archive) == 1
    assert "case" not in archive[0]

    trajectory = Trajectory(
        game_name="Leduc",
        players=["P0", "P1"],
        steps=[
            TrajectoryStep(
                step=1,
                player_id=0,
                observation=Observation(
                    player_id=0,
                    legal_actions=["check", "bet"],
                    rules_text="rules",
                    state_text="state",
                ),
                rendered_prompt="",
                raw_llm_response="",
                parsed_action=Action(player_id=0, type="check"),
                state_before=state_before,
                state_after={**state_before, "phase": "flop"},
                reward=0,
            )
        ],
    )
    append_trajectory_cases_to_archive(
        game="leduc",
        trajectory=trajectory,
        position=0,
        current_game_id=1,
        archive=archive,
        condition="tom_cfr",
    )

    assert len(archive) == 1
    assert "case" in archive[0]
    scores, selected_case = _select_archive_case(
        game="leduc",
        current_feature=archive[0]["feature"],
        archive=archive,
        position=0,
    )
    assert scores and scores[0] == 1.0
    assert selected_case is not None
    assert selected_case["case"]["game_id"] == 1


def test_step_refresh_skips_step1() -> None:
    memory = BaseMemory()
    result = refresh_case_memory_bank_for_step(
        game="leduc",
        state_before=_leduc_state_before(),
        legal_actions=[{"type": "check", "amount": 0}, {"type": "bet", "amount": 4}],
        position=0,
        current_game_id=4,
        current_step=1,
        memory=memory,
        archive=_gradient_archive(),
        live_api=False,
    )
    assert result["triggered"] == 0
    assert result["generated"] == 0
    assert memory.get(CASE_MEMORY_ACTIVE_IDS_KEY) == []
    assert memory.get(CASE_MEMORY_BANK_KEY, []) == []


def test_step_refresh_generates_for_matched_cases_only() -> None:
    memory = BaseMemory()
    # Game 1 already has a stored memory; games 2 and 3 do not.
    memory.set(CASE_MEMORY_BANK_KEY, [
        {
            "memory_id": "001_0_step003",
            "game": "leduc",
            "position": "P0",
            "memory_status": "memory",
            "memory": "In a flop with board Q, betting may work better because it extracts value.",
        },
    ])
    result = refresh_case_memory_bank_for_step(
        game="leduc",
        state_before=_leduc_state_before(),
        legal_actions=[{"type": "check", "amount": 0}, {"type": "bet", "amount": 4}],
        position=0,
        current_game_id=4,
        current_step=3,
        memory=memory,
        archive=_gradient_archive(),
        live_api=False,
    )
    assert result["triggered"] == 1
    assert result["selected"] == 3
    assert result["generated"] == 2
    assert result["accepted"] == 0  # dry-run memories are not stored
    # Only the pre-existing memory is renderable; failed/dry-run ones are not active.
    assert memory.get(CASE_MEMORY_ACTIVE_IDS_KEY) == ["001_0_step003"]
    assert [item["memory_id"] for item in memory.get(CASE_MEMORY_BANK_KEY, [])] == [
        "001_0_step003"
    ]


def test_step_refresh_does_not_duplicate_existing_memories() -> None:
    memory = BaseMemory()
    memory.set(CASE_MEMORY_BANK_KEY, [
        {
            "memory_id": "001_0_step003",
            "game": "leduc",
            "position": "P0",
            "memory_status": "memory",
            "memory": "In a flop with board Q, betting may work better because it extracts value.",
        },
        {
            "memory_id": "002_0_step003",
            "game": "leduc",
            "position": "P0",
            "memory_status": "memory",
            "memory": "In a flop with board Q, checking may work better because it controls risk.",
        },
        {
            "memory_id": "003_0_step003",
            "game": "leduc",
            "position": "P0",
            "memory_status": "memory",
            "memory": "In a flop with board Q, betting may work better because it pressures the caller.",
        },
    ])
    first = refresh_case_memory_bank_for_step(
        game="leduc",
        state_before=_leduc_state_before(),
        legal_actions=[{"type": "check", "amount": 0}, {"type": "bet", "amount": 4}],
        position=0,
        current_game_id=4,
        current_step=3,
        memory=memory,
        archive=_gradient_archive(),
        live_api=False,
    )
    second = refresh_case_memory_bank_for_step(
        game="leduc",
        state_before=_leduc_state_before(),
        legal_actions=[{"type": "check", "amount": 0}, {"type": "bet", "amount": 4}],
        position=0,
        current_game_id=5,
        current_step=3,
        memory=memory,
        archive=_gradient_archive(),
        live_api=False,
    )
    assert first["generated"] == 0
    assert first["active_ids"] == ["001_0_step003", "002_0_step003", "003_0_step003"]
    assert second["generated"] == 0
    assert len(memory.get(CASE_MEMORY_BANK_KEY, [])) == 3
