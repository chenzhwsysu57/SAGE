"""Tests for trajectory logging and runner."""

import json
import tempfile
from pathlib import Path

from base.types import Action, Observation, Trajectory, TrajectoryStep
from games.leduc import LeducHoldem
from rollout import RolloutRunner, save_trajectory, load_trajectory
from methods.baseline.simple_prompt import SimplePromptMethod


class DummyModel:
    """Deterministic model for testing the runner."""
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["check"]
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


class SequencedModel:
    """Fake LLM that returns scripted responses and records prompts."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        idx = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[idx]


class DummyMethod:
    """Simple method that always picks the first legal action."""
    def __init__(self, name: str = "Dummy"):
        self.name = name
        self.method_type = "dummy"

    def act(self, observation, legal_actions, model):
        action = legal_actions[0]
        prompt = "dummy prompt"
        raw = model.generate(prompt)
        return action, prompt, raw


class TestTrajectorySerialization:
    def test_trajectory_step_serialization(self):
        step = TrajectoryStep(
            step=1,
            player_id=0,
            observation=Observation(
                player_id=0,
                private_info={"card": "KS"},
                public_info={"pot": 2},
                history=[],
                legal_actions=["check", "bet"],
            ),
            rendered_prompt="test prompt",
            raw_llm_response="check",
            parsed_action=Action(player_id=0, type="check"),
            action_valid=True,
            retry_count=1,
            rejection_errors=["Illegal action"],
            retry_attempts=[{"raw_response": "raise", "error": "Illegal action"}],
            state_before={"phase": "preflop"},
            state_after={"phase": "preflop"},
            reward=0.0,
        )
        d = step.model_dump()
        assert d["step"] == 1
        assert d["parsed_action"]["type"] == "check"
        assert d["retry_count"] == 1
        assert d["rejection_errors"] == ["Illegal action"]
        assert d["retry_attempts"][0]["raw_response"] == "raise"

    def test_trajectory_roundtrip(self):
        trajectory = Trajectory(
            game_name="TestGame",
            players=["A", "B"],
            steps=[
                TrajectoryStep(
                    step=1,
                    player_id=0,
                    observation=Observation(
                        player_id=0,
                        private_info={},
                        public_info={},
                        history=[],
                        legal_actions=["act"],
                    ),
                    rendered_prompt="p",
                    raw_llm_response="r",
                    parsed_action=Action(player_id=0, type="act"),
                    action_valid=True,
                    state_before={},
                    state_after={},
                    reward=0.0,
                )
            ],
            payout={0: 10, 1: 0},
            winner=0,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_trajectory(trajectory, f.name)
            path = f.name

        loaded = load_trajectory(path)
        assert loaded.game_name == "TestGame"
        assert loaded.payout == {0: 10, 1: 0}
        assert loaded.winner == 0
        assert len(loaded.steps) == 1

        Path(path).unlink()


class TestRolloutRunner:
    def test_runner_completes_game(self):
        game = LeducHoldem(seed=42)
        # Use check-only strategy to ensure showdown
        model = DummyModel(responses=["check"] * 20)
        methods = {
            0: DummyMethod(name="P1"),
            1: DummyMethod(name="P2"),
        }
        runner = RolloutRunner(game=game, methods=methods, model=model)
        trajectory = runner.run()

        assert len(trajectory.steps) >= 4  # at least 4 checks
        assert trajectory.game_name == "LeducHoldem"
        # Final rewards should have one winner
        total = sum(trajectory.payout.values())
        assert total > 0

    def test_runner_fold_ends_early(self):
        game = LeducHoldem(seed=42)
        # First player bets, second folds - but DummyMethod always picks first action.
        # Need a method that folds when possible.
        class FoldMethod:
            name = "Folder"
            method_type = "folder"
            def act(self, observation, legal_actions, model):
                # Pick fold if available
                for a in legal_actions:
                    if a.type == "fold":
                        return a, "prompt", "response"
                return legal_actions[-1], "prompt", "response"

        class BetMethod:
            name = "Better"
            method_type = "better"
            def act(self, observation, legal_actions, model):
                for a in legal_actions:
                    if a.type == "bet":
                        return a, "prompt", "response"
                return legal_actions[-1], "prompt", "response"

        model = DummyModel()
        game = LeducHoldem(seed=42)
        methods = {
            0: BetMethod(),
            1: FoldMethod(),
        }
        runner = RolloutRunner(game=game, methods=methods, model=model)
        trajectory = runner.run()

        assert game.is_terminal(game.reset()) is False  # sanity
        assert len(trajectory.steps) == 2  # bet, fold
        assert trajectory.steps[-1].parsed_action.type == "fold"

    def test_runner_saves_and_loads(self):
        game = LeducHoldem(seed=42)
        model = DummyModel(responses=["check"] * 20)
        methods = {
            0: DummyMethod(name="P1"),
            1: DummyMethod(name="P2"),
        }
        runner = RolloutRunner(game=game, methods=methods, model=model)
        trajectory = runner.run()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_trajectory(trajectory, f.name)
            path = f.name

        loaded = load_trajectory(path)
        assert len(loaded.steps) == len(trajectory.steps)
        assert loaded.payout == trajectory.payout

        Path(path).unlink()

    def test_runner_retries_after_invalid_llm_action(self):
        game = LeducHoldem(seed=42)
        model = SequencedModel(["raise", "check", "check", "check", "check", "check"])
        methods = {
            0: SimplePromptMethod(name="P1"),
            1: SimplePromptMethod(name="P2"),
        }
        runner = RolloutRunner(game=game, methods=methods, model=model)
        trajectory = runner.run()

        first_step = trajectory.steps[0]
        assert first_step.action_valid is False
        assert first_step.retry_count == 1
        assert first_step.rejection_errors
        assert first_step.retry_attempts[0]["raw_response"] == "raise"
        assert first_step.parsed_action.type == "check"
        assert "YOUR LAST ACTION WAS REJECTED" in model.prompts[1]

    def test_runner_raises_after_repeated_invalid_actions(self):
        game = LeducHoldem(seed=42)
        model = SequencedModel(["raise", "raise", "raise"])
        methods = {
            0: SimplePromptMethod(name="P1"),
            1: SimplePromptMethod(name="P2"),
        }
        runner = RolloutRunner(game=game, methods=methods, model=model)

        try:
            runner.run()
        except ValueError as e:
            assert "Illegal action" in str(e)
        else:
            raise AssertionError("Expected runner to fail after repeated invalid actions")
