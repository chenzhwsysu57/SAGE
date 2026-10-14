from __future__ import annotations

import json
from types import SimpleNamespace

from base.interfaces import BaseModel
from base.types import Action, Observation
from methods.baseline.strategic_reasoning_goofspiel import SearchConfig, StrategicGoofspielMethod
from runs.goofspiel.strategic_reasoning_bench import (
    build_metrics,
    save_planning_texts,
    save_step_texts,
    summarize_token_usage,
)


class FixedModel(BaseModel):
    def __init__(
        self,
        response: str = '{"selected_bid":2,"value_estimate":0.5,"terminal":false,"recommendation":"continue","action":{"type":"bid","amount":2}}',
    ) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def obs(player_id: int = 0) -> Observation:
    return Observation(
        player_id=player_id,
        private_info={"hand": [1, 2, 3], "my_bids": []},
        public_info={
            "round": 1,
            "prize": 2,
            "prize_sequence": [2, 3, 1],
            "scores": {0: 0.0, 1: 0.0},
            "outcome_history": [],
        },
        history=[],
        legal_actions=["bid_1", "bid_2", "bid_3"],
        rules_text="Goofspiel rules",
        state_text="Round 1/3 | Prize = 2\nYour hand: [1, 2, 3]",
    )


def actions(player_id: int = 0) -> list[Action]:
    return [Action(player_id=player_id, type="bid", amount=i) for i in [1, 2, 3]]


def test_strategic_modes_render_expected_sections_and_choose_legal_action() -> None:
    expected = {
        "vanilla": "#### YOUR MOVE ####",
        "llm_eq": "#### EQUILIBRIUM STRATEGY ####",
        "llm_tot": "#### TREE OF THOUGHT DELIBERATION ####",
        "llm_pomcts": "#### LLM PARTIALLY OBSERVED MCTS PLANNING TRACE ####",
    }
    for mode, marker in expected.items():
        method = StrategicGoofspielMethod(
            name=mode,
            mode=mode,
            tot_branches=3,
            search_config=SearchConfig(width=1, depth=3, sims=3, seed=7),
        )
        model = FixedModel()
        action, prompt, raw, data = method.act(obs(), actions(), model)
        assert action.type == "bid"
        assert action.amount == 2
        assert marker in prompt
        assert raw == model.response
        assert data["mode"] == mode
        if mode == "llm_pomcts":
            assert len(model.prompts) == 10
            assert len(data["pomcts"]["simulations"]) == 3
            assert len(data["pomcts"]["planning_records"]) == 18
            assert data["final_selector"] == "mean_simulated_payoff"
        else:
            assert len(model.prompts) == 1


def test_tot_prompt_uses_requested_branch_count() -> None:
    method = StrategicGoofspielMethod(name="tot", mode="llm_tot", tot_branches=5)
    prompt = method._render_prompt(obs(), actions(), {})
    assert "Generate exactly 5 distinct candidate thoughts" in prompt


def test_pomcts_prompt_records_llm_planning_trace_for_p1_hidden_bid() -> None:
    method = StrategicGoofspielMethod(
        name="pomcts",
        mode="llm_pomcts",
        search_config=SearchConfig(width=1, depth=3, sims=3, seed=3),
    )
    model = FixedModel()
    p1_obs = obs(player_id=1)
    action, prompt, _, data = method.act(p1_obs, actions(player_id=1), model)
    assert action.amount == 2
    assert "width=1, depth=3, effective_depth=3, sims=3" in prompt
    assert "Recommended bid by mean simulated payoff:" in prompt
    assert "Action value table:" in prompt
    assert "Simulation 3: root_bid=2" in prompt
    assert "mean_payoff" in prompt
    assert "Highest estimated bid:" not in prompt
    assert model.prompts[0].count("#### LLM MCTS SIMULATED DECISION ####") == 1
    assert "Legal actions:" in model.prompts[0]
    assert "SIMULATED TRAJECTORY SO FAR" in model.prompts[1]
    assert len(data["pomcts"]["planning_records"]) == 15
    assert any(r["phase"] == "opponent_cfr" for r in data["pomcts"]["planning_records"])


def test_token_usage_summary_reads_prompt_and_response_files(tmp_path) -> None:
    out = tmp_path / "prompts" / "Greedy" / "vanilla" / "P0"
    out.mkdir(parents=True)
    (out / "g001_step01_P0_prompt.txt").write_text("abcd" * 5)
    (out / "g001_step01_P0_response.txt").write_text("efgh" * 2)

    usage = summarize_token_usage(tmp_path)
    assert usage["by_condition"]["vanilla"]["prompt_chars"] == 20
    assert usage["by_condition"]["vanilla"]["response_chars"] == 8
    assert usage["by_condition"]["vanilla"]["approx_tokens"] == 7

    metrics = build_metrics([
        {
            "condition": "vanilla",
            "persona": "Greedy",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "winrate": 50.0,
            "total_chips": 4.0,
            "avg_chips": 2.0,
        }
    ], usage)
    point = metrics["by_condition"]["vanilla"]
    assert point["approx_tokens"] == 7
    assert point["avg_chips"] == 2.0
    assert json.loads(json.dumps(metrics))


def test_save_step_texts_keeps_only_strategic_method_player(tmp_path) -> None:
    planning = {
        "pomcts": {
            "planning_records": [
                {
                    "phase": "path_step",
                    "sim": 1,
                    "depth": 2,
                    "selected_bid": 3,
                    "prompt": "planning prompt",
                    "response": "planning response",
                }
            ]
        }
    }
    trajectory = SimpleNamespace(steps=[
        SimpleNamespace(
            step=1,
            player_id=0,
            rendered_prompt="opponent prompt should not count",
            raw_llm_response="opponent response should not count",
            method_data=None,
        ),
        SimpleNamespace(
            step=2,
            player_id=1,
            rendered_prompt="strategic prompt",
            raw_llm_response="strategic response",
            method_data=planning,
        ),
    ])

    save_step_texts(
        tmp_path,
        "Greedy",
        "llm_pomcts",
        "P1",
        1,
        trajectory,
        method_position=1,
    )

    files = sorted((tmp_path / "prompts").rglob("*.txt"))
    assert sorted(p.name for p in files) == sorted([
        "g001_step02_P1_sim01_depth02_path_step_bid3_prompt.txt",
        "g001_step02_P1_sim01_depth02_path_step_bid3_response.txt",
        "g001_step02_P1_prompt.txt",
        "g001_step02_P1_response.txt",
    ])
    usage = summarize_token_usage(tmp_path)
    assert usage["by_condition"]["llm_pomcts"]["prompt_chars"] == len("strategic prompt") + len("planning prompt")
    assert usage["by_condition"]["llm_pomcts"]["response_chars"] == len("strategic response") + len("planning response")


def test_save_planning_texts_noops_for_non_pomcts_data(tmp_path) -> None:
    save_planning_texts(tmp_path, "g001_step01_P0", {"mode": "vanilla"})
    assert not list(tmp_path.rglob("*.txt"))
