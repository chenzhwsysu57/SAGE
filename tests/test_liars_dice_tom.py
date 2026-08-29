from __future__ import annotations

import pytest

from base.case_memory import CASE_MEMORY_ACTIVE_IDS_KEY, CASE_MEMORY_BANK_KEY
from base.memory import BaseMemory
from base.types import Action, Observation, Trajectory, TrajectoryStep
from methods.liars_dice_tom import LiarsDiceToMMethod


class MockModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _obs() -> Observation:
    return Observation(
        player_id=0,
        private_info={"my_dice": [3, 3, 1, 2]},
        public_info={
            "total_dice_remaining": 8,
            "my_dice_count": 4,
            "opponent_dice_count": 4,
            "last_bid": {"quantity": 3, "face": 3},
            "round": 1,
        },
        history=[
            {"player": 1, "action": "bid", "quantity": 3, "face": 3, "round": 1},
        ],
        legal_actions=["bid", "challenge"],
        rules_text="Liar's Dice rules",
        state_text="Your dice (4 left): 3, 3, 1, 2\nCurrent bid: 3 of face 3",
    )


def _actions() -> list[Action]:
    return [
        Action(player_id=0, type="challenge"),
        Action(player_id=0, type="bid", data={"quantity": 4, "face": 3}),
    ]


def test_liars_dice_tom_single_stage_prompt_and_memory() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory())
    obs = _obs()
    model = MockModel([
        '{"reasoning": "CFR baseline then opponent may challenge", '
        '"action": {"type": "bid", "quantity": 4, "face": 3}}',
    ])

    action, prompt, raw, method_data = method.act(obs, _actions(), model)

    assert action.type == "bid"
    assert len(model.prompts) == 1
    assert "#### KEY NODE DECISION ####" not in model.prompts[0]
    assert "predicted_opponent_action" not in prompt
    assert "Action EV" not in prompt
    assert "equilibrium EV" not in prompt
    assert method_data["self_evolve"]["memory_records"] == 0

    trajectory = Trajectory(
        game_name="LiarsDice",
        players=["LD-ToM"],
        steps=[
            TrajectoryStep(
                step=1,
                player_id=0,
                observation=obs,
                rendered_prompt=prompt,
                raw_llm_response=raw,
                parsed_action=action,
                method_data=method_data,
            )
        ],
    )
    method.reflect(trajectory, model, "LiarsDice")

    records = method.memory.get("liars_dice_self_evolve_records", [])
    assert records == []


def test_liars_dice_tom_non_key_hides_prediction_field() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory())
    model = MockModel([
        '{"reasoning": "routine baseline action", '
        '"action": {"type": "bid", "quantity": 4, "face": 3}}',
    ])

    _, prompt, _, method_data = method.act(_obs(), _actions(), model)

    assert method_data["self_evolve"]["memory_records"] == 0
    assert "predicted_opponent_action" not in prompt
    assert "SELF EVOLVE MEMORY" not in prompt


def test_liars_dice_tom_key_node_replays_self_evolve_memory_on_later_prompt() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory())
    obs = _obs()
    first_model = MockModel([
        '{"reasoning": "CFR baseline then opponent may challenge", '
        '"action": {"type": "bid", "quantity": 4, "face": 3}}',
    ])

    action, prompt, raw, method_data = method.act(obs, _actions(), first_model)
    trajectory = Trajectory(
        game_name="LiarsDice",
        players=["LD-ToM"],
        steps=[
            TrajectoryStep(
                step=1,
                player_id=0,
                observation=obs,
                rendered_prompt=prompt,
                raw_llm_response=raw,
                parsed_action=action,
                method_data=method_data,
            ),
            TrajectoryStep(
                step=2,
                player_id=1,
                observation=_obs(),
                rendered_prompt="",
                raw_llm_response="",
                parsed_action=Action(player_id=1, type="bid", data={"quantity": 5, "face": 3}),
            ),
        ],
    )
    method.reflect(trajectory, first_model, "9")

    second_model = MockModel([
        '{"reasoning": "memory-guided action", '
        '"action": {"type": "bid", "quantity": 4, "face": 3}}',
    ])
    _, second_prompt, _, second_method_data = method.act(obs, _actions(), second_model)

    assert "SELF EVOLVE MEMORY" not in second_prompt
    method.memory.set(CASE_MEMORY_BANK_KEY, [
        {
            "memory_id": "001_0_step003",
            "game": "liars_dice",
            "position": "P0",
            "memory_status": "memory",
            "suggested_action": "bid",
            "opponent_action_consistency": "consistent",
            "quality_flags": [],
            "memory": "In round 1 with strong dice support, bidding may work better because it keeps initiative.",
            "source_case": {"similarity": 0.95},
        },
        {
            "memory_id": "002_0_step003",
            "game": "liars_dice",
            "position": "P0",
            "memory_status": "memory",
            "suggested_action": "bid",
            "opponent_action_consistency": "consistent",
            "quality_flags": [],
            "memory": "In a similar opening bid spot, bidding may work better because it sustains pressure.",
            "source_case": {"similarity": 0.87},
        },
        {
            "memory_id": "003_0_step003",
            "game": "liars_dice",
            "position": "P0",
            "memory_status": "memory",
            "suggested_action": "bid",
            "opponent_action_consistency": "consistent",
            "quality_flags": [],
            "memory": "In a nearby bluff-heavy round, bidding may work better because it keeps tempo.",
            "source_case": {"similarity": 0.79},
        },
    ])
    method.memory.set(CASE_MEMORY_ACTIVE_IDS_KEY, [
        "001_0_step003",
        "002_0_step003",
        "003_0_step003",
    ])
    third_model = MockModel([
        '{"reasoning": "case-memory action", '
        '"action": {"type": "bid", "quantity": 4, "face": 3}}',
    ])
    _, third_prompt, _, second_method_data = method.act(obs, _actions(), third_model)

    assert "SELF EVOLVE MEMORY" in third_prompt
    assert "complete battle case" in third_prompt
    assert "suggested action" not in third_prompt
    assert "In round 1 with strong dice support" in third_prompt
    assert "case 3:" in third_prompt
    assert "trajectory before predict:" not in third_prompt
    assert "result: failed" not in third_prompt
    assert "predicted_opponent_action: challenge" not in second_prompt
    assert second_method_data["self_evolve"]["memory_records"] == 3
    assert third_prompt.index("#### SELF EVOLVE MEMORY ####") < third_prompt.index("Reason in this order:")
    assert third_prompt.index("#### SELF EVOLVE MEMORY ####") < third_prompt.index("Legal actions:")


def test_liars_dice_opponent_model_shows_recent_five_belief_snapshots() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory())
    method._all_records = [
        {"game": 0, "type": "bid"},
        {"game": 1, "type": "challenge"},
    ]
    for i in range(6):
        method._belief_history.append({
            "game": i + 1,
            "actions": i + 2,
            "beliefs": {
                "Bluffer": 0.1 + i * 0.01,
                "Honest": 0.3,
                "GTO": 0.6 - i * 0.01,
            },
        })

    prompt = method._render_prompt(
        _obs(),
        _actions(),
        persona="GTO",
        confidence=0.55,
        beliefs={"GTO": 0.55, "Honest": 0.25, "Bluffer": 0.2},
        cfr_strategy=[],
    )

    assert "Recent persona probabilities:" in prompt
    assert "game 6:" in prompt
    assert "game 2:" in prompt
    assert "game 1:" not in prompt
    assert prompt.index("game 6:") < prompt.index("game 5:")
    assert "(6 actions)" not in prompt
    assert "(2 actions)" not in prompt
    assert "Most possible" not in prompt
    assert "Classification mode: q_window_ema" in prompt
    assert "Q-window: last 10 games" in prompt
    assert "EMA alpha: 0.85" in prompt
    assert "Persona style reference" in prompt
    assert "Habitual liar" in prompt


def test_liars_dice_q_window_ema_updates_posterior() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory(), q_alpha=0.8)
    method._bank = {"personas": {"Bluffer": {}, "Honest": {}}}
    method._posterior_cache = {"Bluffer": 0.25, "Honest": 0.75}

    posterior = method._update_persona_posterior({"Bluffer": 0.75, "Honest": 0.25})

    assert posterior == pytest.approx({"Bluffer": 0.35, "Honest": 0.65})


def test_liars_dice_q_window_selects_recent_games() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory(), q_window=2)
    method._all_records = [
        {"game": 0, "type": "bid", "qty": 1, "face": 1},
        {"game": 1, "type": "bid", "qty": 2, "face": 1},
        {"game": 2, "type": "challenge"},
    ]

    records = method._records_for_window()

    assert [r["game"] for r in records] == [1, 2]


def test_liars_dice_belief_history_keeps_one_row_per_game() -> None:
    method = LiarsDiceToMMethod(name="LD-ToM", memory=BaseMemory())
    method._current_game = 14
    method._all_records = [
        {"game": 14, "type": "bid"},
        {"game": 14, "type": "challenge"},
    ]

    method._record_belief_snapshot({"GTO": 0.6, "Random": 0.4})
    method._all_records.append({"game": 14, "type": "bid"})
    method._record_belief_snapshot({"GTO": 0.7, "Random": 0.3})

    assert len(method._belief_history) == 1
    assert method._belief_history[0]["game"] == 15
    assert method._belief_history[0]["actions"] == 3
    assert method._belief_history[0]["beliefs"] == {"GTO": 0.7, "Random": 0.3}
