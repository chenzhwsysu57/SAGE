"""Tests for SimplePromptMethod."""

from base.types import Action, Observation, Trajectory, TrajectoryStep
from base.case_memory import CASE_MEMORY_ACTIVE_IDS_KEY, CASE_MEMORY_BANK_KEY
from methods.helpers import build_example, parse_action_response
from methods.leduc_tom import LeducToMMethod
from methods.simple_prompt import SimplePlaceboPromptMethod, SimplePromptMethod
from methods.suspicion_prompt import SuspicionPromptMethod
from methods.belief import OpponentProfile
from methods.tom import ToMMethod
from modeling.base import BaseModelingMethod
from modeling.liars_dice.archetypes import ArchetypeAgent


class MockModel:
    """Mock model that returns a fixed response."""

    def __init__(self, response: str | list[str] = "call"):
        self.response = response
        self.last_prompt = ""
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


class NullModelingMethod(BaseModelingMethod):
    def update(self, opp_id, trajectory, model=None) -> None:
        return None

    def statement(self, opp_id: int, **kwargs) -> str:
        return ""


class TestSimplePromptMethod:
    def test_build_example_uses_placeholders(self):
        legal_actions = [
            Action(player_id=0, type="challenge"),
            Action(player_id=0, type="bid", data={"quantity": 5, "face": 2}),
        ]

        example = build_example(legal_actions)

        assert example["reasoning"] == "step by step analysis of the current situation"
        assert example["action"]["type"] == "<action_type>"
        assert example["action"]["quantity"] == "<quantity:int>"
        assert example["action"]["face"] == "<face:int>"

    def test_render_prompt_basic(self):
        method = SimplePromptMethod(name="TestAgent")
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        prompt = method._render_prompt(obs, legal_actions)
        assert "KS" in prompt
        assert "Fake rules" in prompt
        assert "check" in prompt
        assert "bet" in prompt
        assert "If any persona guidance is provided above" in prompt
        assert "Respond with ONLY your chosen action word." not in prompt

    def test_render_prompt_with_history(self):
        method = SimplePromptMethod(name="TestAgent")
        obs = Observation(
            player_id=1,
            private_info={"card": "QH"},
            public_info={},
            history=[{"player": 0, "action": "bet", "amount": 2, "street": 0}],
            legal_actions=["fold", "call", "raise"],
            rules_text="Fake rules",
            state_text="Your card: QH\nHistory:\n  P0 bets 2",
        )
        legal_actions = [
            Action(player_id=1, type="fold"),
            Action(player_id=1, type="call", amount=2),
            Action(player_id=1, type="raise", amount=4),
        ]

        prompt = method._render_prompt(obs, legal_actions)
        assert "QH" in prompt
        assert "p0 bets 2" in prompt.lower()

    def test_render_prompt_with_role_guidance_uses_your_role_section(self):
        method = SimplePromptMethod(name="TestAgent")
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
            role_guidance="Play aggressively and never check.",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        prompt = method._render_prompt(obs, legal_actions)
        assert "#### YOUR ROLE ####" in prompt
        assert "Play aggressively and never check." in prompt
        assert "#### YOUR MOVE ####" in prompt
        assert prompt.index("#### YOUR ROLE ####") < prompt.index("#### YOUR MOVE ####")

    def test_simple_placebo_prompt_adds_neutral_context(self):
        method = SimplePlaceboPromptMethod(
            name="PlaceboAgent",
            style_labels=["Greedy", "Conservative", "Random"],
            target_chars=4200,
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[{"player": 1, "action": "check"}],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        prompt = method._render_prompt(obs, legal_actions)

        assert "#### PLACEBO STRATEGIC REFERENCE ####" in prompt
        assert "#### PLACEBO OPPONENT BELIEF ####" in prompt
        assert "#### PLACEBO RECENT WINDOW ####" in prompt
        assert "#### PLACEBO MEMORY ####" in prompt
        assert "#### PLACEBO DECISION SCAFFOLD ####" in prompt
        assert "#### PLACEBO LENGTH CONTROL ####" in prompt
        assert "Greedy: uniform placeholder prior" in prompt
        assert "Conservative: uniform placeholder prior" in prompt
        assert "Random: uniform placeholder prior" in prompt
        assert "does not provide a calibrated Nash, CFR, EV, or action-value estimate" in prompt
        assert "Generic play checklist" in prompt
        assert "General style reminders" in prompt
        assert "Recent-window evidence is intentionally withheld" in prompt
        assert "do not treat it as evidence" in prompt
        assert "no posterior probability has been estimated" in prompt
        assert len(prompt) >= 4200
        assert "Legal actions:" in prompt
        assert '"action"' in prompt

    def test_simple_placebo_act_uses_existing_action_parser(self):
        method = SimplePlaceboPromptMethod()
        model = MockModel(response='{"action": "bet"}')
        obs = Observation(
            player_id=0,
            private_info={},
            public_info={},
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Pot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        action, prompt, raw = method.act(obs, legal_actions, model)

        assert action.type == "bet"
        assert action.amount == 2
        assert method.method_type == "simple_placebo"
        assert "PLACEBO" in prompt
        assert raw == '{"action": "bet"}'

    def test_parse_response_simple_word(self):
        method = SimplePromptMethod()
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]
        action = parse_action_response("call", 0, legal_actions)
        assert action.type == "call"
        assert action.amount == 2

    def test_parse_response_json(self):
        method = SimplePromptMethod()
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]
        action = parse_action_response('{"action": "call"}', 0, legal_actions)
        assert action.type == "call"

    def test_parse_response_json_with_reasoning_wrapper(self):
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]
        raw = 'I should continue.\n{"action": "CALL", "reasoning": "pot odds"}'
        action = parse_action_response(raw, 0, legal_actions)
        assert action.type == "call"
        assert action.amount == 2
        assert action.data == {"reasoning": "pot odds"}

    def test_parse_response_accepts_common_json_action_keys(self):
        legal_actions = [
            Action(player_id=1, type="check"),
            Action(player_id=1, type="bet", amount=2),
        ]
        for raw in ('{"move": "bet"}', '{"choice": "bet"}', '{"type": "bet"}'):
            action = parse_action_response(raw, 1, legal_actions)
            assert action.type == "bet"
            assert action.amount == 2
            assert action.data == {}

    def test_parse_response_falls_back_after_malformed_json(self):
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]
        action = parse_action_response('{"action": "call"  I will call.', 0, legal_actions)
        assert action.type == "call"

    def test_parse_response_whitespace(self):
        method = SimplePromptMethod()
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]
        action = parse_action_response("  call  ", 0, legal_actions)
        assert action.type == "call"

    def test_parse_response_rejects_non_legal_action(self):
        method = SimplePromptMethod()
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]
        # "raise" is not legal — dynamic regex only matches legal actions
        action = parse_action_response("raise", 0, legal_actions)
        assert action.type == ""  # no match → game will reject loudly

    def test_parse_response_extracts_word(self):
        method = SimplePromptMethod()
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]
        action = parse_action_response("I choose to call.", 0, legal_actions)
        assert action.type == "call"

    def test_act_returns_tuple(self):
        method = SimplePromptMethod(name="TestAgent")
        model = MockModel(response="call")
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={
                "community_card": None,
                "pot": 4,
                "street": "preflop",
                "my_bet_total": 1,
                "opponent_bet_total": 3,
                "my_bet_this_round": 0,
                "opponent_bet_this_round": 2,
                "round_raises": 1,
                "to_call": 2,
            },
            history=[{"player": 0, "action": "bet", "amount": 2, "street": 0}],
            legal_actions=["fold", "call"],
        )
        legal_actions = [
            Action(player_id=0, type="fold"),
            Action(player_id=0, type="call", amount=2),
        ]

        action, prompt, raw = method.act(obs, legal_actions, model)
        assert action.type == "call"
        assert len(prompt) > 0
        assert raw == "call"

    def test_archetype_prompt_inserts_separate_rule_section(self):
        method = ArchetypeAgent("Bluffer")
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[],
            legal_actions=["challenge", "bid"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="challenge"),
            Action(player_id=0, type="bid", data={"quantity": 5, "face": 2}),
        ]

        prompt = method._render_prompt(obs, legal_actions)
        assert "#### YOUR ROLE ####" in prompt
        assert "--- Your Playing Style ---" not in prompt
        assert "bold, slippery, and confident" in prompt
        assert prompt.index("#### YOUR ROLE ####") < prompt.index("#### YOUR MOVE ####")


class TestToMMethod:
    def test_prompt_uses_placeholder_example(self):
        method = ToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[],
            legal_actions=["challenge", "bid"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="challenge"),
            Action(player_id=0, type="bid", data={"quantity": 5, "face": 2}),
        ]

        prompt = method._render_prompt(obs, legal_actions, [])

        assert '"quantity": "<quantity:int>"' in prompt
        assert '"face": "<face:int>"' in prompt
        assert '"<action_type>"' in prompt

    def test_initial_prompt_includes_empty_opponent_models_panel(self):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        prompt = method._render_prompt(obs, legal_actions, [])

        assert "OPPONENT MODELS" in prompt
        assert "Opponent P1: no actions observed yet" in prompt
        assert "no opponent-specific exploit is justified" not in prompt
        assert "Reason in this order" in prompt
        assert "opponent model" in prompt
        assert "GTO EV" not in prompt
        assert "equilibrium EV" not in prompt

    def test_prompt_with_history_includes_observed_opponent_status(self):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={},
            history=[{"player": 1, "action": "call", "amount": 2, "street": 0}],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 4 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        action, prompt, raw, _ = method.act(
            obs,
            legal_actions,
            MockModel([
                '{"action": {"type": "check"}}',
            ]),
        )

        assert action.type == "check"
        assert raw == '{"action": {"type": "check"}}'
        assert "OPPONENT MODELS" in prompt
        assert "Opponent P1: 1 actions observed" not in prompt

    def test_leduc_prompt_omits_key_node_fields_inside_your_move(self):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={
                "street": "preflop",
                "to_call": 0,
                "round_raises": 0,
            },
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        prompt = method._render_prompt(obs, legal_actions, [])

        assert "GTO BASELINE" not in prompt
        assert "CFR Nash" not in prompt
        assert "#### KEY NODE DECISION ####" not in prompt
        assert "#### SELF EVOLVE ####" not in prompt
        assert "#### YOUR MOVE ####" in prompt
        assert "is_key_node" not in prompt
        assert "predicted_opponent_action" not in prompt
        assert prompt.index("#### YOUR MOVE ####") < prompt.index("Legal actions:")

    def test_leduc_cfr_prompt_reasons_from_cfr_before_opponent_model(self):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=True,
            modeling_method=NullModelingMethod(),
        )
        method._cfr_renderer = lambda observation, legal_actions: "CFR says mix check and bet."
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={
                "street": "preflop",
                "to_call": 0,
                "round_raises": 0,
            },
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        prompt = method._render_prompt(
            obs,
            legal_actions,
            [],
        )

        assert "Start from the CFR/GTO strategy guidance above" in prompt
        assert "Then adjust from that baseline using the opponent model" in prompt
        assert prompt.index("Start from the CFR/GTO strategy guidance above") < prompt.index(
            "Then adjust from that baseline using the opponent model"
        )
        assert "GTO EV" not in prompt
        assert "equilibrium EV" not in prompt
        assert "is_key_node" not in prompt

    def test_leduc_self_evolve_reflect_records_memory(self):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={
                "street": "preflop",
                "to_call": 0,
                "round_raises": 0,
            },
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]
        model = MockModel([
            '{"reasoning": "value is thin and action changes initiative", '
            '"action": {"type": "check"}}',
        ])

        action, prompt, raw, method_data = method.act(obs, legal_actions, model)

        assert action.type == "check"
        assert "SELF EVOLVE" not in prompt
        assert method_data["self_evolve"]["memory_records"] == 0
        assert model.calls == 1

        opp_obs = Observation(
            player_id=1,
            private_info={"card": "QH"},
            public_info={
                "street": "preflop",
                "to_call": 0,
                "round_raises": 0,
            },
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Opponent to act\nPot: 2 chips",
        )
        trajectory = Trajectory(
            game_name="LeducHoldem",
            players=["ToM-test"],
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
                    observation=opp_obs,
                    rendered_prompt="",
                    raw_llm_response="",
                    parsed_action=Action(player_id=1, type="bet", amount=2),
                ),
            ],
        )

        method.reflect(trajectory, model, "7")

        records = method.memory.get("leduc_self_evolve_records", [])
        assert records == []
        assert method._self_evolve_memory_text(0) == ""

        method.memory.set(CASE_MEMORY_BANK_KEY, [
            {
                "memory_id": "001_0_step003",
                "game": "leduc",
                "position": "P0",
                "memory_status": "memory",
                "suggested_action": "bet",
                "opponent_action_consistency": "consistent",
                "quality_flags": [],
                "memory": "In preflop with a strong King, betting may work better because it keeps initiative.",
                "source_case": {"similarity": 0.96},
            },
            {
                "memory_id": "002_0_step003",
                "game": "leduc",
                "position": "P0",
                "memory_status": "memory",
                "suggested_action": "bet",
                "opponent_action_consistency": "consistent",
                "quality_flags": [],
                "memory": "In a similar preflop spot, betting may work better because it keeps pressure on the opponent.",
                "source_case": {"similarity": 0.89},
            },
            {
                "memory_id": "003_0_step003",
                "game": "leduc",
                "position": "P0",
                "memory_status": "memory",
                "suggested_action": "bet",
                "opponent_action_consistency": "consistent",
                "quality_flags": [],
                "memory": "In a nearby preflop sequence, betting may work better because it protects initiative.",
                "source_case": {"similarity": 0.82},
            },
        ])
        method.memory.set(CASE_MEMORY_ACTIVE_IDS_KEY, [
            "001_0_step003",
            "002_0_step003",
            "003_0_step003",
        ])
        memory_text = method._self_evolve_memory_text(0)
        assert "#### SELF EVOLVE MEMORY ####" in memory_text
        assert "complete battle case" in memory_text
        assert "suggested action" not in memory_text
        assert "In preflop with a strong King" in memory_text
        assert "case 3:" in memory_text
        assert "trajectory before predict:" not in memory_text
        assert "result: failed" not in memory_text

        memory_model = MockModel([
            '{"reasoning": "memory-guided action", '
            '"action": {"type": "check"}}',
        ])
        _, memory_prompt, _, _ = method.act(obs, legal_actions, memory_model)
        assert memory_prompt.index("#### SELF EVOLVE MEMORY ####") < memory_prompt.index("Reason in this order:")
        assert memory_prompt.index("#### SELF EVOLVE MEMORY ####") < memory_prompt.index("Legal actions:")

    def test_leduc_self_evolve_renders_only_active_ids_without_cleanliness_filter(self):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
        )
        # Bank content alone must not render: rendering is driven by active ids.
        method.memory.set(CASE_MEMORY_BANK_KEY, [
            {
                "memory_id": "001_0_step003",
                "game": "leduc",
                "position": "P0",
                "memory_status": "memory",
                "quality_flags": ["contains_probability_or_confidence"],
                "memory": "In preflop with a strong King, betting may work better with 70% confidence.",
                "source_case": {"similarity": 0.96},
            },
            {
                "memory_id": "002_0_step003",
                "game": "leduc",
                "position": "P0",
                "memory_status": "memory",
                "quality_flags": ["suggested_action_same_as_actioned"],
                "memory": "In a similar preflop spot, betting may work better because it keeps pressure.",
                "source_case": {"similarity": 0.89},
            },
        ])
        assert method._self_evolve_memory_text(0) == ""

        # Once the refresh marks the same cases as active, flagged memories render as-is.
        method.memory.set(CASE_MEMORY_ACTIVE_IDS_KEY, [
            "001_0_step003",
            "002_0_step003",
        ])
        memory_text = method._self_evolve_memory_text(0)
        assert "#### SELF EVOLVE MEMORY ####" in memory_text
        assert "with 70% confidence" in memory_text
        assert "case 2:" in memory_text

    def test_leduc_md_carries_style_posterior_into_opponent_estimate(self):
        class StylePosteriorModeling:
            def statement(self, opp_id, **kwargs):
                return ""

            def style_posterior(self, opp_id):
                return {
                    "LAG": 0.54,
                    "Maniac": 0.457,
                    "TAG": 0.001,
                    "GTO": 0.0,
                    "Nit": 0.0,
                    "CallingStation": 0.0,
                }

        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=StylePosteriorModeling(),
        )
        profile = OpponentProfile()
        profile.record("raise")
        profile.record("raise")
        profile.record("call")
        method.memory.set("profile_1", profile.to_dict())

        obs = Observation(
            player_id=0,
            history=[],
            legal_actions=["check", "bet"],
            rules_text="rules",
            state_text="state",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]
        _, _, _, md = method.act(
            obs,
            legal_actions,
            MockModel(['{"reasoning": "x", "action": {"type": "check"}}']),
        )
        assert md["beliefs"]["LAG"] == 0.54
        assert md["persona"] == "LAG"
        assert md["confidence"] == 0.54
        assert md["n_actions"] == 3

    def test_verbose_mode_prints_full_prompt_without_truncation(self, capsys):
        method = LeducToMMethod(
            name="ToM-test",
            use_cfr=False,
            modeling_method=NullModelingMethod(),
            verbose=True,
        )
        obs = Observation(
            player_id=0,
            private_info={"card": "KS"},
            public_info={
                "street": "preflop",
                "to_call": 0,
                "round_raises": 0,
            },
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules\nLine 2\nLine 3",
            state_text="Your card: KS\nPot: 2 chips",
        )
        legal_actions = [
            Action(player_id=0, type="check"),
            Action(player_id=0, type="bet", amount=2),
        ]

        method.act(
            obs,
            legal_actions,
            MockModel([
                '{"action": {"type": "check"}}',
            ]),
        )
        out = capsys.readouterr().out

        assert "... (" not in out
        assert "Fake rules" in out
        assert "Line 3" in out


class TestSuspicionPromptMethod:
    """Tests for SuspicionPromptMethod, the Suspicion-Agent port."""

    def make_obs(self, **kwargs):
        defaults = dict(
            player_id=0,
            private_info={"card": "KS"},
            public_info={"street": "preflop", "to_call": 0, "round_raises": 0},
            history=[],
            legal_actions=["check", "bet"],
            rules_text="Fake rules",
            state_text="Your card: KS\nPot: 2 chips",
        )
        defaults.update(kwargs)
        return Observation(**defaults)

    def make_legal(self, types: list[str] | None = None):
        types = types or ["check", "bet"]
        return [Action(player_id=0, type=t, amount=2 if t == "bet" else 0) for t in types]

    # ── Prompt rendering tests ────────────────────────────────────────

    def test_pattern_prompt_contains_expected_sections(self):
        method = SuspicionPromptMethod(name="Suspect")
        method._long_memory = ["1th Game Start!\nPrevious game summary."]
        prompt = method._render_pattern_prompt(
            self.make_obs(), "Opponent",
        )
        assert "game pattern" in prompt.lower()
        assert "strategy improvement" in prompt.lower()
        assert "tree-structure" in prompt
        assert "Suspect" in prompt
        assert "Opponent" in prompt
        assert "Fake rules" in prompt

    def test_belief_prompt_contains_pattern_and_obs(self):
        method = SuspicionPromptMethod(name="Suspect")
        method._pattern = "Opponent tight with J, aggressive with K"
        prompt = method._render_belief_prompt(
            self.make_obs(), "Opponent", "P0 bet, P1 called",
        )
        assert "Suspect" in prompt
        assert "Opponent" in prompt
        assert "Your card: KS" in prompt
        assert "tight with J" in prompt
        assert "P0 bet, P1 called" in prompt

    def test_planning_prompt_contains_json_instruction(self):
        method = SuspicionPromptMethod(name="Suspect")
        method._pattern = "Pattern analysis..."
        method._belief = "Belief analysis..."
        prompt = method._render_planning_prompt(
            self.make_obs(), self.make_legal(), "Opponent",
        )
        assert "Expected Chips Gain" in prompt
        assert "Plan Selection" in prompt
        assert '"action": "<fold|check|call|bet|raise>"' in prompt
        assert "check" in prompt
        assert "bet" in prompt

    # ── Full pipeline tests ───────────────────────────────────────────

    def test_first_act_runs_pipeline_returns_valid_action(self):
        method = SuspicionPromptMethod(name="Suspect")
        obs = self.make_obs()
        legal = self.make_legal()
        model = MockModel([
            "Opponent pattern: aggressive with K, passive with J",
            "Belief: Opponent likely holds Q (50%) or K (50%)",
            '{"action": "check"}',
        ])
        action, prompt, raw, md = method.act(obs, legal, model)
        assert action.type == "check"
        assert model.calls == 3
        assert "suspicion" in md
        suspicion = md["suspicion"]
        assert suspicion["pattern_prompt"]
        assert suspicion["pattern_response"] == (
            "Opponent pattern: aggressive with K, passive with J"
        )
        assert suspicion["belief_prompt"]
        assert suspicion["planning_prompt"]

    def test_second_act_skips_pattern(self):
        method = SuspicionPromptMethod(name="Suspect")
        obs = self.make_obs()
        legal = self.make_legal()
        # First act: pattern + belief + planning
        model1 = MockModel([
            "Pattern: Opponent plays GTO style",
            "Belief: Even distribution across J/Q/K",
            '{"action": "check"}',
        ])
        method.act(obs, legal, model1)
        assert not method._needs_refresh

        # Second act: only belief + planning
        model2 = MockModel([
            "Belief: Opponent likely has K given raise",
            '{"action": "call"}',
        ])
        action2, _, _, md2 = method.act(obs, legal, model2)
        assert action2.type == "call"
        assert model2.calls == 2
        assert md2["suspicion"]["pattern_prompt"] == ""

    def test_reflect_populates_long_memory(self):
        method = SuspicionPromptMethod(name="Suspect")
        obs = self.make_obs()
        traj = Trajectory(
            game_name="LeducHoldem",
            players=["Suspect"],
            steps=[
                TrajectoryStep(
                    step=1, player_id=0, observation=obs,
                    rendered_prompt="", raw_llm_response="",
                    parsed_action=Action(player_id=0, type="check"),
                ),
            ],
        )
        model = MockModel("Game summary: P0 had K and checked, won pot")
        method.reflect(traj, model, "LeducHoldem")
        assert len(method._long_memory) == 1
        assert "Game" in method._long_memory[0]
        assert method._needs_refresh is True

    def test_long_memory_window_trimmed_to_20(self):
        method = SuspicionPromptMethod(name="Suspect")
        method._long_memory = [f"Game {i} summary" for i in range(25)]
        obs = self.make_obs()
        traj = Trajectory(
            game_name="LeducHoldem",
            players=["Suspect"],
            steps=[
                TrajectoryStep(
                    step=1, player_id=0, observation=obs,
                    rendered_prompt="", raw_llm_response="",
                    parsed_action=Action(player_id=0, type="check"),
                ),
            ],
        )
        model = MockModel("New game summary")
        method.reflect(traj, model, "LeducHoldem")
        assert len(method._long_memory) <= 20
        # Should still contain the latest summary
        assert "New game summary" in method._long_memory[-1]
