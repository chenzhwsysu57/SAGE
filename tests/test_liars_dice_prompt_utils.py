from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from runs.liars_dice.prompt_utils import save_step_artifacts


def test_save_step_artifacts_writes_decision_sidecar(tmp_path: Path) -> None:
    step = SimpleNamespace(
        step=3,
        player_id=1,
        rendered_prompt="prompt text",
        raw_llm_response='{"action":"bid"}',
        parsed_action=SimpleNamespace(
            player_id=1,
            type="bid",
            amount=0,
            data={"quantity": 5, "face": 2},
        ),
        action_valid=True,
        retry_count=1,
        rejection_errors=["illegal bid"],
        method_data=None,
    )

    save_step_artifacts(tmp_path, "step003_P1_Test", step, "TestMethod")

    decision_path = tmp_path / "step003_P1_Test_decision.json"
    prompt_path = tmp_path / "step003_P1_Test_prompt.txt"
    response_path = tmp_path / "step003_P1_Test_response.txt"

    assert prompt_path.read_text() == "prompt text"
    assert response_path.read_text() == '{"action":"bid"}'

    data = json.loads(decision_path.read_text())
    assert data["step"] == 3
    assert data["player_id"] == 1
    assert data["method_name"] == "TestMethod"
    assert data["selected_action"]["type"] == "bid"
    assert data["selected_action"]["quantity"] == 5
    assert data["selected_action"]["face"] == 2


def test_save_step_artifacts_does_not_write_keynode_prompt_response(tmp_path: Path) -> None:
    step = SimpleNamespace(
        step=1,
        player_id=0,
        rendered_prompt="main prompt",
        raw_llm_response='{"action":{"type":"bid"}}',
        parsed_action=None,
        action_valid=True,
        retry_count=0,
        rejection_errors=[],
        method_data={"self_evolve": {"memory_records": 0}},
    )

    save_step_artifacts(tmp_path, "step001_P0_Test", step, "TestMethod")

    assert not (tmp_path / "step001_P0_Test_keynode_prompt.txt").exists()
    assert not (tmp_path / "step001_P0_Test_keynode_response.txt").exists()
