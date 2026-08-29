"""Tests for Decrypto game logic."""

import pytest
from base.types import Action, Observation
from games.decrypto import Decrypto, DecryptoState, _parse_guess_code
from methods.decrypto import DecryptoMethod
from modeling.decrypto import LockedThemeState
from runs.decrypto.metrics import aggregate_metrics, summarize_trajectory
from tools.decrypto.interceptor_style import (
    build_interceptor_model,
    estimate_interceptor_style,
    infer_interceptor_style_llm,
    normalize_interceptor_history,
    reconstruct_round_records,
)


def _hint(hints):
    """Helper: Alice's hint action."""
    return Action(player_id=0, type="hint",
                  data={"hints": list(hints)})


def _guess(player, code):
    """Helper: Bob or Eve's guess action.  code can be list or "X-Y-Z" str."""
    return Action(player_id=player, type="guess",
                  data={"code": code})


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class TestDecryptoGame:
    def test_reset_initial_state(self):
        game = Decrypto(seed=42)
        state = game.reset()

        assert state.phase == "alice_hint"
        assert state.acting_player == 0
        assert state.turn == 0
        assert state.miscommunications == 0
        assert state.interceptions == 0
        assert len(state.keywords) == 4
        assert state.current_code is not None  # code is pre-assigned
        assert len(state.current_code) == 3
        assert len(state.code_history) == 0
        assert not game.is_terminal(state)

    def test_num_players(self):
        game = Decrypto()
        assert game.num_players() == 3

    def test_deterministic_reset(self):
        g1 = Decrypto(seed=42)
        g2 = Decrypto(seed=42)
        s1 = g1.reset()
        s2 = g2.reset()
        assert s1.keywords == s2.keywords
        assert s1.current_code == s2.current_code  # same seed → same code

    def test_alice_hint_goes_to_bob(self):
        game = Decrypto(seed=42)
        state = game.reset()
        assigned_code = state.current_code

        state = game.step(state, _hint(("spring", "garden", "bloom")))

        assert state.phase == "bob_guess"
        assert state.acting_player == 1
        assert state.current_code == assigned_code  # unchanged
        assert state.current_hints == ("spring", "garden", "bloom")

    def test_bob_guess_goes_to_eve(self):
        game = Decrypto(seed=42)
        state = game.reset()
        code = state.current_code
        state = game.step(state, _hint(("a", "b", "c")))
        state = game.step(state, _guess(1, list(code)))

        assert state.phase == "eve_guess"
        assert state.acting_player == 2
        assert state.bob_guess == code

    def test_full_turn_bob_right_eve_wrong(self):
        game = Decrypto(seed=42, max_turns=8)
        state = game.reset()
        code = state.current_code

        state = game.step(state, _hint(("x", "y", "z")))
        state = game.step(state, _guess(1, list(code)))  # Bob right
        state = game.step(state, _guess(2, (4, 3, 1)))  # Eve wrong

        assert state.turn == 1
        assert state.miscommunications == 0
        assert state.interceptions == 0
        assert state.phase in ("alice_hint", "terminal")

    def test_eve_intercepts(self):
        game = Decrypto(seed=42, max_turns=8)
        state = game.reset()
        code = state.current_code

        state = game.step(state, _hint(("a", "b", "c")))
        state = game.step(state, _guess(1, (4, 3, 1)))  # Bob wrong
        state = game.step(state, _guess(2, list(code)))  # Eve right

        assert state.interceptions == 1
        assert state.miscommunications == 1

    def test_eve_wins_after_3_interceptions(self):
        game = Decrypto(seed=42, max_turns=12)
        state = game.reset()

        for _ in range(3):
            code = state.current_code
            state = game.step(state, _hint(("x", "y", "z")))
            state = game.step(state, _guess(1, (1, 1, 1)))  # Bob wrong
            state = game.step(state, _guess(2, list(code)))  # Eve right

        assert game.is_terminal(state)
        assert game.get_rewards(state)[2] > 0

    def test_alice_bob_win_after_max_turns(self):
        game = Decrypto(seed=42, max_turns=2)
        state = game.reset()

        for _ in range(2):
            code = state.current_code
            state = game.step(state, _hint(("a", "b", "c")))
            state = game.step(state, _guess(1, list(code)))  # Bob right
            state = game.step(state, _guess(2, (4, 3, 1)))  # Eve wrong

        assert game.is_terminal(state)
        rewards = game.get_rewards(state)
        assert rewards[0] > 0
        assert rewards[1] > 0

    def test_guess_string_format(self):
        """Guess code accepts "X-Y-Z" string format."""
        game = Decrypto(seed=42)
        state = game.reset()
        code = state.current_code

        state = game.step(state, _hint(("a", "b", "c")))
        code_str = "-".join(str(d) for d in code)
        state = game.step(state, _guess(1, code_str))

        assert state.bob_guess == code

    def test_guess_list_format(self):
        """Guess code accepts [X, Y, Z] list format (backward compat)."""
        game = Decrypto(seed=42)
        state = game.reset()
        code = state.current_code

        state = game.step(state, _hint(("a", "b", "c")))
        state = game.step(state, _guess(1, list(code)))

        assert state.bob_guess == code

    def test_get_observation_alice(self):
        game = Decrypto(seed=42)
        state = game.reset()
        obs = game.get_observation(state, 0)

        assert obs.player_id == 0
        assert "keywords" in obs.private_info
        assert "current_code" in obs.private_info  # Alice sees assigned code
        assert obs.legal_actions == ["hint"]

    def test_get_observation_bob(self):
        game = Decrypto(seed=42)
        state = game.reset()
        state = game.step(state, _hint(("a", "b", "c")))
        obs = game.get_observation(state, 1)

        assert obs.player_id == 1
        assert "keywords" in obs.private_info
        assert obs.legal_actions == ["guess"]
        assert "current_hints" in obs.public_info

    def test_get_observation_eve(self):
        game = Decrypto(seed=42)
        state = game.reset()
        state = game.step(state, _hint(("a", "b", "c")))
        state = game.step(state, _guess(1, list(state.current_code)))
        obs = game.get_observation(state, 2)

        assert obs.player_id == 2
        assert "keywords" not in obs.private_info
        assert obs.legal_actions == ["guess"]
        assert "current_hints" in obs.public_info

    def test_illegal_action_rejected(self):
        game = Decrypto(seed=42)
        state = game.reset()
        with pytest.raises(ValueError, match="Illegal action"):
            game.step(state, _guess(0, (1, 2, 3)))

    def test_wrong_player_rejected(self):
        game = Decrypto(seed=42)
        state = game.reset()
        with pytest.raises(ValueError, match="Wrong player"):
            game.step(state, _hint(("a", "b", "c")).model_copy(
                update={"player_id": 1}))

    def test_hint_count_validation(self):
        """Alice must provide exactly 3 hints."""
        game = Decrypto(seed=42)
        state = game.reset()
        with pytest.raises(ValueError, match="Expected 3 hints"):
            game.step(state, Action(player_id=0, type="hint",
                                  data={"hints": ["x", "y"]}))

    def test_hint_history_tracks_per_keyword(self):
        game = Decrypto(seed=42)
        state = game.reset()
        code = state.current_code
        hints = ("h1", "h2", "h3")

        state = game.step(state, _hint(hints))
        state = game.step(state, _guess(1, list(code)))
        state = game.step(state, _guess(2, (4, 3, 1)))

        if not game.is_terminal(state):
            # hint_history maps digit → list of hints
            for i, digit in enumerate(code):
                assert state.hint_history[digit] == [hints[i]]

    def test_state_serialization(self):
        game = Decrypto(seed=42)
        state = game.reset()
        d = game.state_to_dict(state)

        assert d["phase"] == "alice_hint"
        assert len(d["keywords"]) == 4
        assert d["current_code"] is not None
        assert len(d["current_code"]) == 3

    def test_state_clone(self):
        game = Decrypto(seed=42)
        state = game.reset()
        cloned = state.clone()

        assert cloned.keywords == state.keywords
        cloned.hint_history[1].append("test")
        assert len(state.hint_history[1]) == 0

    def test_legal_actions_empty_at_terminal(self):
        game = Decrypto(seed=42, max_turns=2)
        state = game.reset()

        for _ in range(2):
            code = state.current_code
            state = game.step(state, _hint(("a", "b", "c")))
            state = game.step(state, _guess(1, list(code)))
            state = game.step(state, _guess(2, (4, 3, 1)))

        assert game.is_terminal(state)
        assert game.get_legal_actions(state) == []

    def test_eve_wins_after_3_bob_mistakes(self):
        game = Decrypto(seed=42, max_turns=12)
        state = game.reset()

        for _ in range(3):
            state = game.step(state, _hint(("x", "y", "z")))
            state = game.step(state, _guess(1, (4, 3, 2)))  # Bob wrong
            state = game.step(state, _guess(2, (1, 2, 3)))  # Eve also wrong

        assert game.is_terminal(state)
        assert game.get_rewards(state)[2] > 0

    def test_rules_text(self):
        game = Decrypto()
        assert "Decrypto Rules" in game.get_rules_text()

    def test_rules_text_uses_configured_thresholds(self):
        game = Decrypto(
            max_turns=7,
            interception_threshold=5,
            miscommunication_threshold=4,
        )
        text = game.get_rules_text()

        assert "Bob guesses wrong 4 times" in text
        assert "Eve guesses right 5 times" in text
        assert "Survive 7 turns" in text

    def test_state_to_text_alice(self):
        game = Decrypto(seed=42)
        state = game.reset()
        text = game.state_to_text(state, 0)

        assert "Keywords:" in text
        assert "Current code:" in text  # Alice sees the assigned code
        assert state.keywords[0] in text

    def test_state_to_text_uses_configured_thresholds(self):
        game = Decrypto(
            seed=42,
            interception_threshold=5,
            miscommunication_threshold=4,
        )
        state = game.reset()
        text = game.state_to_text(state, 0)

        assert "Miscommunications (Bob wrong): 0/4" in text
        assert "Interceptions (Eve right): 0/5" in text

    def test_state_to_text_bob(self):
        game = Decrypto(seed=42)
        state = game.reset()
        code = state.current_code
        state = game.step(state, _hint(("h1", "h2", "h3")))
        text = game.state_to_text(state, 1)

        assert "Keywords:" in text
        assert "h1" in text
        assert "h2" in text

    def test_state_to_text_eve(self):
        game = Decrypto(seed=42)
        state = game.reset()
        text = game.state_to_text(state, 2)

        assert "Keywords:" not in text  # Eve doesn't see keywords

    def test_method_prompt_uses_configured_thresholds(self):
        game = Decrypto(
            seed=42,
            max_turns=7,
            interception_threshold=5,
            miscommunication_threshold=4,
        )
        state = game.reset()
        obs = game.get_observation(state, 0)
        method = DecryptoMethod()

        prompt = method._render_prompt(obs, game.get_legal_actions(state))

        assert "4 Miscommunications OR 5 Interceptions" in prompt
        assert "Survive 7 turns" in prompt

    def test_default_encoder_prompt_is_clean_baseline(self):
        game = Decrypto(seed=42)
        state = game.reset()
        obs = game.get_observation(state, 0)
        method = DecryptoMethod(persona="default")

        prompt = method._render_prompt(obs, game.get_legal_actions(state))

        assert "ANTI-CONVERGENCE" not in prompt
        assert "INTERCEPTOR MODEL" not in prompt
        assert "different facet" not in prompt
        assert "disrupting" not in prompt
        assert "Prioritize Bob's correctness" in prompt

    def test_anti_convergence_encoder_prompt_contains_old_strong_strategy(self):
        game = Decrypto(seed=42)
        state = game.reset()
        obs = game.get_observation(state, 0)
        method = DecryptoMethod(persona="anti_convergence")

        prompt = method._render_prompt(obs, game.get_legal_actions(state))

        assert "ANTI-CONVERGENCE STRATEGY" in prompt
        assert "different meaning" in prompt
        assert "Prevent easy convergence" in prompt

    def test_literal_interceptor_prompt_is_surface_matching(self):
        game = Decrypto(seed=42)
        state = game.reset()
        state = game.step(state, _hint(("soil", "cube", "airplane")))
        state = game.step(state, _guess(1, list(state.current_code)))
        obs = game.get_observation(state, 2)
        method = DecryptoMethod(persona="literal")

        prompt = method._render_prompt(obs, game.get_legal_actions(state))

        assert "Literal player" in prompt
        assert "three sequential word steps" in prompt
        assert "STEP 1" in prompt
        assert "STEP 2" in prompt
        assert "STEP 3" in prompt
        assert "surface-level" in prompt
        assert 'ANSWER: {"guess": "X-Y-Z"}' in prompt
        assert "Stay close to the surface of each hint" in prompt
        assert "Use three distinct digits from 1 to 4." in prompt
        assert "Do not" not in prompt

    def test_literal_interceptor_act_is_greedy_and_does_not_call_model(self):
        class ScriptedScoreModel:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt):
                self.prompts.append(prompt)
                if "Current hint: horn" in prompt:
                    return (
                        'ANSWER: {"scores": {"horn": 9, "ground": 2, '
                        '"brick": 1, "snake": 0}, "best_word": "horn"}'
                    )
                if "Current hint: ground" in prompt:
                    return (
                        'ANSWER: {"scores": {"ground": 9, "brick": 2, '
                        '"snake": 1}, "best_word": "ground"}'
                    )
                if "Current hint: brick" in prompt:
                    return (
                        'ANSWER: {"scores": {"brick": 9, "snake": 2}, '
                        '"best_word": "brick"}'
                    )
                raise AssertionError(f"unexpected literal scoring prompt: {prompt}")

        obs = Observation(
            player_id=2,
            public_info={
                "turn": 2,
                "max_turns": 12,
                "miscommunications": 0,
                "interceptions": 0,
                "miscommunication_threshold": 3,
                "interception_threshold": 3,
                "hint_history": {
                    "1": ["metal", "trumpet", "horn"],
                    "2": ["soil", "ground"],
                    "3": ["cube", "brick"],
                    "4": ["cobra", "snake"],
                },
                "current_hints": ["horn", "ground", "brick"],
                "code_history": [[1, 2, 3]],
                "guess_history": [
                    {"code": [1, 2, 3], "bob": [1, 2, 3], "eve": [1, 2, 3]}
                ],
            },
            legal_actions=["guess"],
        )
        method = DecryptoMethod(persona="literal")
        model = ScriptedScoreModel()

        action, prompt, raw_response, method_data = method.act(
            obs, [Action(player_id=2, type="guess")], model
        )

        assert action.type == "guess"
        assert action.data["code"] == "1-2-3"
        assert "Remaining pool after commit" in raw_response
        assert 'ANSWER: {"guess": "1-2-3"}' in raw_response
        assert method_data["literal_policy"][0]["selected_word"] == "horn"
        assert method_data["literal_policy"][0]["selected_digit"] == 1
        assert method_data["literal_policy"][1]["selected_word"] == "ground"
        assert method_data["literal_policy"][1]["selected_digit"] == 2
        assert method_data["literal_policy"][2]["selected_word"] == "brick"
        assert method_data["literal_policy"][2]["selected_digit"] == 3
        assert "Candidate words:" in method_data["literal_policy"][0]["prompt"]
        assert "metal" in method_data["literal_policy"][0]["prompt"]
        assert "Keyword 1" not in method_data["literal_policy"][0]["prompt"]
        assert "horn" in method_data["literal_policy"][0]["prompt"]
        assert "metal" not in method_data["literal_policy"][1]["prompt"]
        assert "trumpet" not in method_data["literal_policy"][1]["prompt"]
        assert method_data["literal_policy"][0]["response"].startswith("ANSWER:")
        assert len(model.prompts) == 3
        assert "Literal player" in prompt

    def test_literal_interceptor_bootstraps_to_123_without_history(self):
        class NoCallModel:
            def generate(self, prompt):
                raise AssertionError(f"bootstrap should not call model: {prompt}")

        obs = Observation(
            player_id=2,
            public_info={
                "turn": 1,
                "max_turns": 12,
                "miscommunications": 0,
                "interceptions": 0,
                "miscommunication_threshold": 3,
                "interception_threshold": 3,
                "hint_history": {
                    "1": [],
                    "2": [],
                    "3": [],
                    "4": [],
                },
                "current_hints": ["month", "golf", "sheet"],
                "code_history": [],
                "guess_history": [],
            },
            legal_actions=["guess"],
        )
        method = DecryptoMethod(persona="literal")

        action, prompt, raw_response, method_data = method.act(
            obs, [Action(player_id=2, type="guess")], NoCallModel()
        )

        assert action.data["code"] == "1-2-3"
        assert "BOOTSTRAP" in raw_response
        assert method_data["literal_policy"][0]["bootstrap"] is True
        assert method_data["literal_policy"][0]["word_index"] == 0
        assert "Literal player" in prompt

    def test_adaptive_interceptor_prompt_is_history_updating(self):
        game = Decrypto(seed=42)
        state = game.reset()
        state = game.step(state, _hint(("soil", "cube", "airplane")))
        state = game.step(state, _guess(1, list(state.current_code)))
        obs = game.get_observation(state, 2)
        method = DecryptoMethod(persona="adaptive")

        prompt = method._render_prompt(obs, game.get_legal_actions(state))

        assert "Adaptive player" in prompt
        assert "all completed hint history" in prompt
        assert "revise the theme" in prompt
        assert "Weight recent completed rounds slightly more" in prompt
        assert 'ANSWER: {"guess": "X-Y-Z"}' in prompt
        assert "Do not freeze on first impressions." in prompt


class TestDecryptoMetrics:
    def test_summarize_trajectory_counts_process_metrics(self):
        trajectory = {
            "payout": {"2": 1.0},
            "steps": [
                {
                    "state_after": {
                        "guess_history": [
                            {
                                "code": [1, 2, 3],
                                "bob": [1, 2, 3],
                                "eve": [1, 3, 2],
                            },
                            {
                                "code": [2, 3, 4],
                                "bob": [2, 4, 3],
                                "eve": [2, 3, 4],
                            },
                            {
                                "code": [3, 4, 1],
                                "bob": [3, 4, 2],
                                "eve": [3, 4, 1],
                            },
                        ]
                    }
                }
            ],
        }

        metrics = summarize_trajectory(
            trajectory,
            max_turns=12,
            interception_threshold=2,
            miscommunication_threshold=3,
        )

        assert metrics["rounds"] == 3
        assert metrics["eve_interceptions"] == 2
        assert metrics["bob_miscommunications"] == 2
        assert metrics["eve_first_hit_round"] == 2
        assert metrics["end_cause"] == "interception"

    def test_aggregate_metrics_builds_round_and_stage_rates(self):
        game_metrics = [
            {
                "rounds": 3,
                "eve_interceptions": 2,
                "bob_miscommunications": 1,
                "eve_first_hit_round": 2,
                "eve_won": True,
                "ab_win": False,
                "end_cause": "interception",
                "per_round": [
                    {"round": 1, "eve_hit": False, "bob_wrong": False},
                    {"round": 2, "eve_hit": True, "bob_wrong": True},
                    {"round": 3, "eve_hit": True, "bob_wrong": False},
                ],
            }
        ]

        aggregate = aggregate_metrics(game_metrics, max_turns=12)

        assert aggregate["total_rounds"] == 3
        assert aggregate["eve_int_rate"] == pytest.approx(2 / 3)
        assert aggregate["bob_mis_rate"] == pytest.approx(1 / 3)
        assert aggregate["eve_first_hit_round_avg"] == 2
        assert aggregate["end_causes"] == {"interception": 1}
        assert aggregate["per_round"][1]["eve_int_rate"] == 1.0
        assert aggregate["stages"]["early"]["attempts"] == 3


class TestInterceptorStyleModel:
    def test_reconstruct_round_records_from_grouped_hint_history(self):
        public_info = {
            "code_history": [[1, 2, 3], [1, 2, 4]],
            "hint_history": {
                "1": ["water", "river"],
                "2": ["tree", "forest"],
                "3": ["sun"],
                "4": ["moon"],
            },
            "guess_history": [
                {"code": [1, 2, 3], "bob": [1, 2, 3], "eve": [1, 2, 4]},
                {"code": [1, 2, 4], "bob": [1, 2, 4], "eve": [1, 2, 4]},
            ],
        }

        records = reconstruct_round_records(public_info)

        assert records[0]["hints"] == ["water", "tree", "sun"]
        assert records[1]["hints"] == ["river", "forest", "moon"]

    def test_estimate_interceptor_style_returns_valid_structure(self):
        records = [
            {
                "round": 1,
                "code": [1, 2, 3],
                "hints": ["water", "tree", "sun"],
                "bob": [1, 2, 3],
                "eve": [1, 2, 3],
            },
            {
                "round": 2,
                "code": [1, 2, 4],
                "hints": ["waterfall", "treetop", "moon"],
                "bob": [1, 2, 4],
                "eve": [1, 2, 3],
            },
        ]

        estimate = estimate_interceptor_style(records)

        assert estimate.rounds_observed == 2
        assert set(estimate.posterior.keys()) == {"theme_loyalist", "literal", "adaptive"}
        assert abs(sum(estimate.posterior.values()) - 1.0) < 1e-6
        assert "stability" in estimate.features
        assert "entropy" in estimate.features
        assert "switch_rate" in estimate.features
        assert "hit_rate" in estimate.features
        assert "acc_improvement" in estimate.features
        assert isinstance(estimate.exploit, str) and len(estimate.exploit) > 0

    def test_estimate_interceptor_style_tracks_switching_and_trend(self):
        records = [
            {
                "round": 1,
                "code": [1, 2, 3],
                "hints": ["water", "tree", "sun"],
                "bob": [1, 2, 3],
                "eve": [4, 4, 4],
            },
            {
                "round": 2,
                "code": [1, 2, 4],
                "hints": ["waterfall", "treetop", "moon"],
                "bob": [1, 2, 4],
                "eve": [1, 2, 3],
            },
            {
                "round": 3,
                "code": [1, 2, 4],
                "hints": ["water", "tree", "moon"],
                "bob": [1, 2, 4],
                "eve": [1, 2, 4],
            },
        ]

        estimate = estimate_interceptor_style(records)

        assert estimate.features["switch_rate"] > 0.0
        assert estimate.features["acc_improvement"] > 0.0
        assert estimate.top_style == "adaptive"

    def test_tom_encoder_injects_style_model_and_method_data(self):
        obs = Observation(
            player_id=0,
            private_info={
                "keywords": ["water", "tree", "sun", "moon"],
                "current_code": [1, 2, 4],
            },
            public_info={
                "turn": 2,
                "max_turns": 12,
                "miscommunications": 0,
                "interceptions": 1,
                "miscommunication_threshold": 3,
                "interception_threshold": 5,
                "code_history": [[1, 2, 3], [1, 2, 4]],
                "hint_history": {
                    "1": ["water", "waterfall"],
                    "2": ["tree", "treetop"],
                    "3": ["sun"],
                    "4": ["moon"],
                },
                "guess_history": [
                    {"code": [1, 2, 3], "bob": [1, 2, 3], "eve": [1, 2, 3]},
                    {"code": [1, 2, 4], "bob": [1, 2, 4], "eve": [1, 2, 3]},
                ],
            },
            legal_actions=["hint"],
        )
        model = ScriptedModel([
            '{"posterior": {"theme_loyalist": 0.70, "literal": 0.10, "adaptive": 0.20}, '
            '"top_style": "theme_loyalist", "confidence": 0.70, '
            '"evidence": ["first hints anchor themes"], '
            '"exploit": "rotate away from first-hint themes"}',
            'ANSWER: {"hints": ["river", "forest", "night"]}',
        ])
        method = DecryptoMethod(persona="tom")

        result = method.act(obs, [Action(player_id=0, type="hint")], model)

        assert len(result) == 4
        action, prompt, _, method_data = result
        assert action.data["hints"] == ["river", "forest", "night"]
        assert len(model.prompts) == 2
        assert "INTERCEPTOR STYLE INFERENCE" in model.prompts[0]
        assert "Completed rounds:" in model.prompts[0]
        assert "Round 1:" in model.prompts[0]
        assert "#### INTERCEPTOR MODEL ####" in prompt
        assert "theme_loyalist" in prompt
        assert "Hard constraint: Bob" in prompt
        style_data = method_data["interceptor_style_model"]
        assert style_data["top_style"] == "theme_loyalist"
        assert "interceptor_style_prompt" in method_data
        assert "interceptor_style_raw_response" in method_data

    def test_infer_interceptor_style_llm_parses_json_response(self):
        obs = Observation(
            player_id=0,
            private_info={
                "keywords": ["water", "tree", "sun", "moon"],
                "current_code": [1, 2, 3],
            },
            public_info={
                "turn": 2,
                "max_turns": 12,
                "miscommunications": 0,
                "interceptions": 0,
                "miscommunication_threshold": 3,
                "interception_threshold": 5,
                "code_history": [[1, 2, 3], [1, 2, 4]],
                "hint_history": {
                    "1": ["water", "waterfall"],
                    "2": ["tree", "treetop"],
                    "3": ["sun"],
                    "4": ["moon"],
                },
                "guess_history": [
                    {"code": [1, 2, 3], "bob": [1, 2, 3], "eve": [1, 2, 3]},
                    {"code": [1, 2, 4], "bob": [1, 2, 4], "eve": [1, 2, 3]},
                ],
            },
            legal_actions=["hint"],
        )

        model = ScriptedModel([
            '{"posterior": {"theme_loyalist": 0.60, "literal": 0.25, '
            '"adaptive": 0.15}, "top_style": "theme_loyalist", '
            '"confidence": 0.60, "evidence": ["first hints stabilize"], '
            '"exploit": "rotate away from first-hint themes"}',
        ])

        estimate, prompt, raw = infer_interceptor_style_llm(obs, model)

        assert "INTERCEPTOR STYLE INFERENCE" in prompt
        assert "Step 1 — Build a round-by-round table" in prompt
        assert "round_signals" in prompt
        assert '"signals": {' in prompt
        assert "anchoring" in prompt
        assert "revision" in prompt
        assert raw.startswith('{"posterior"')
        assert estimate.top_style == "theme_loyalist"
        assert estimate.posterior["theme_loyalist"] == pytest.approx(0.60)
        assert estimate.features["llm_confidence"] == pytest.approx(0.60)
        assert estimate.features["llm_evidence"] == ["first hints stabilize"]
        assert "first-hint themes" in estimate.exploit
        assert "Round 1:" in prompt

    def test_build_interceptor_model_accepts_trajectory_dict(self):
        trajectory = {
            "steps": [
                {
                    "state_after": {
                        "code_history": [[1, 2, 3], [1, 2, 4]],
                        "hint_history": {
                            "1": ["water", "waterfall"],
                            "2": ["tree", "treetop"],
                            "3": ["sun"],
                            "4": ["moon"],
                        },
                        "guess_history": [
                            {"code": [1, 2, 3], "bob": [1, 2, 3], "eve": [1, 2, 3]},
                            {"code": [1, 2, 4], "bob": [1, 2, 4], "eve": [1, 2, 3]},
                        ],
                    }
                }
            ]
        }

        estimate = build_interceptor_model(trajectory)

        assert estimate.rounds_observed == 2
        assert set(estimate.posterior.keys()) == {"theme_loyalist", "literal", "adaptive"}
        assert abs(sum(estimate.posterior.values()) - 1.0) < 1e-6
        assert "stability" in estimate.features

    def test_normalize_interceptor_history_accepts_records(self):
        records = [
            {
                "round": 1,
                "code": [1, 2, 3],
                "hints": ["water", "tree", "sun"],
                "bob": [1, 2, 3],
                "eve": [1, 2, 3],
            }
        ]

        normalized = normalize_interceptor_history(records)

        assert normalized[0]["code"] == [1, 2, 3]
        assert normalized[0]["hints"] == ["water", "tree", "sun"]


class TestLockedThemeAudit:
    def test_locked_theme_events_are_returned_in_method_data(self):
        obs = Observation(
            player_id=2,
            public_info={
                "turn": 1,
                "max_turns": 12,
                "miscommunications": 0,
                "interceptions": 0,
                "miscommunication_threshold": 3,
                "interception_threshold": 5,
                "hint_history": {
                    "1": ["can"],
                    "2": [],
                    "3": [],
                    "4": [],
                },
                "current_hints": ["metal", "road", "sky"],
            },
            legal_actions=["guess"],
        )
        model = ScriptedModel([
            "preserved food containers",
            'ANSWER: {"guess": "1-2-3"}',
        ])
        method = DecryptoMethod(
            locked_theme_state=LockedThemeState(),
            persona="theme_loyalist",
        )

        result = method.act(obs, [Action(player_id=2, type="guess")], model)

        assert len(result) == 4
        action, prompt, raw_response, method_data = result
        assert action.data == {"code": "1-2-3"}
        assert raw_response == 'ANSWER: {"guess": "1-2-3"}'
        assert "preserved food containers" in prompt
        assert method_data["locked_theme_state"]["themes"] == {
            1: "preserved food containers",
        }
        event = method_data["locked_theme_events"][0]
        assert event["keyword"] == 1
        assert event["first_hint"] == "can"
        assert event["theme"] == "preserved food containers"
        assert event["raw_response"] == "preserved food containers"
        assert 'A player gave the hint "can"' in event["prompt"]


class TestParseGuessCode:
    def test_string_format(self):
        action = Action(player_id=1, type="guess", data={"code": "1-3-4"})
        assert _parse_guess_code(action) == [1, 3, 4]

    def test_list_format(self):
        action = Action(player_id=1, type="guess", data={"code": [1, 3, 4]})
        assert _parse_guess_code(action) == [1, 3, 4]

    def test_list_strings(self):
        action = Action(player_id=1, type="guess", data={"code": ["1", "3", "4"]})
        assert _parse_guess_code(action) == [1, 3, 4]

    def test_empty(self):
        action = Action(player_id=1, type="guess", data={})
        assert _parse_guess_code(action) == []
