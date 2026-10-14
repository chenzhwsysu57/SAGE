import json
import random

from base.types import Action, Observation
import methods.sage.cfr as cfr_mod
from methods.sage.cfr import (
    CFRMethod,
    encode_history_for_cfr,
    strategy_probs_for_legal_actions,
)
from modeling.leduc.archetypes import CFRGTOAgent
from tools.leduc.cfr_tools import CFRTools


class FakeTrainer:
    def __init__(self, strategy):
        self.strategy = strategy


def _facing_bet_observation() -> Observation:
    return Observation(
        player_id=1,
        private_info={"card": "KH"},
        public_info={"community_card": None, "street": "preflop"},
        history=[{"player": 0, "action": "bet", "amount": 2, "street": 0}],
        legal_actions=["fold", "call", "raise"],
        rules_text="",
        state_text="",
    )


def _facing_bet_actions() -> list[Action]:
    return [
        Action(player_id=1, type="fold"),
        Action(player_id=1, type="call", amount=2),
        Action(player_id=1, type="raise", amount=4),
    ]


def test_encode_history_for_cfr_includes_street_delimiter():
    history = [
        {"player": 0, "action": "check", "street": 0},
        {"player": 1, "action": "check", "street": 0},
        {"player": 0, "action": "bet", "street": 1},
    ]

    assert encode_history_for_cfr(history, street=1) == "ccdb"


def test_strategy_probs_maps_cfr_codes_to_legal_action_types():
    strategy = {"K|?|b|0": {"f": 0.1, "k": 0.7, "r": 0.2}}

    probs, used_fallback = strategy_probs_for_legal_actions(
        strategy,
        "K|?|b|0",
        _facing_bet_actions(),
    )

    assert used_fallback is False
    assert probs == {"fold": 0.1, "call": 0.7, "raise": 0.2}


def test_cfr_method_uses_encoded_strategy_actions():
    method = CFRMethod.__new__(CFRMethod)
    method.name = "CFR-test"
    method.method_type = "cfr"
    method._rng = random.Random(0)
    method._trainer = FakeTrainer({"K|?|b|0": {"f": 0.0, "k": 1.0, "r": 0.0}})

    action, prompt, raw = method.act(
        _facing_bet_observation(),
        _facing_bet_actions(),
    )

    assert action.type == "call"
    assert action.amount == 2
    assert "fallback=uniform" not in prompt
    assert raw == "call"


def test_cfr_gto_agent_uses_encoded_strategy_actions():
    agent = CFRGTOAgent.__new__(CFRGTOAgent)
    agent.name = "GTO-test"
    agent.method_type = "cfr"
    agent._rng = random.Random(0)
    agent._strategy = {"K|?|b|0": {"f": 0.0, "k": 0.0, "r": 1.0}}

    action, prompt, raw = agent.act(
        _facing_bet_observation(),
        _facing_bet_actions(),
    )

    assert action.type == "raise"
    assert action.amount == 4
    assert "fallback=uniform" not in prompt
    assert raw == "raise"


def test_cfr_tools_preflop_chance_states_enumerate_future_board():
    tools = CFRTools.__new__(CFRTools)
    states = tools._enumerate_chance_states(("K", "H"), None)

    assert len(states) == 20
    probs = tools._opp_hand_probs(states)
    assert probs == {"J": 0.4, "Q": 0.4, "K": 0.2}
    assert all(comm != ("K", "H") and comm != opp for opp, comm, _ in states)


def test_cfr_tools_mixed_strategy_uses_current_infoset():
    tools = CFRTools.__new__(CFRTools)
    tools._strategy = {"Q|?|c|0": {"c": 0.25, "b": 0.75}}
    history = [{"player": 0, "action": "check", "street": 0}]

    mixed, used_fallback = tools.mixed_strategy("QS", "", history, "preflop")

    assert used_fallback is False
    assert mixed == {"check": 0.25, "bet": 0.75}


def test_cfr_tools_prompt_includes_nash_mixed_strategy():
    tools = CFRTools.__new__(CFRTools)
    tools._strategy = {
        "Q|?|c|0": {"c": 0.25, "b": 0.75},
        "J|?|cb|0": {"f": 1.0},
        "Q|?|cb|0": {"f": 1.0},
        "K|?|cb|0": {"f": 1.0},
        "J|?|cc|0": {"c": 1.0},
        "Q|?|cc|0": {"c": 1.0},
        "K|?|cc|0": {"c": 1.0},
    }
    history = [{"player": 0, "action": "check", "street": 0}]

    prompt = tools.format_for_prompt("QS", "", history, "preflop", my_id=1)

    assert "Nash mixed strategy: check: 25.0%, bet: 75.0%" in prompt
    assert "Action EVs" not in prompt
    assert "Nash equilibrium recommendation" not in prompt


def test_cfr_method_loads_existing_model_without_training(tmp_path, monkeypatch):
    model_path = tmp_path / "cfr_plus.json"
    model_path.write_text(json.dumps({"strategy": {"K|?|b|0": {"k": 1.0}}}))

    def fail_train(self):  # pragma: no cover - defensive
        raise AssertionError("train() should not be called when model exists")

    monkeypatch.setattr(cfr_mod.CFRPlusTrainer, "train", fail_train)
    CFRMethod._cache.clear()

    method = CFRMethod(name="CFR-test", strategy_path=str(model_path))
    assert method._trainer.strategy == {"K|?|b|0": {"k": 1.0}}


def test_cfr_method_trains_and_saves_when_model_missing(tmp_path, monkeypatch):
    model_path = tmp_path / "nested" / "cfr_plus.json"
    calls = {"train": 0}

    class DummyTrainer:
        def __init__(self, iterations: int):
            self.iterations = iterations
            self.strategy = {"K|?|b|0": {"k": 1.0}}

        def train(self):
            calls["train"] += 1

    monkeypatch.setattr(cfr_mod, "CFRPlusTrainer", DummyTrainer)
    CFRMethod._cache.clear()

    method = CFRMethod(name="CFR-test", strategy_path=str(model_path))
    assert calls["train"] == 1
    assert model_path.exists()
    saved = json.loads(model_path.read_text())
    assert saved["strategy"] == {"K|?|b|0": {"k": 1.0}}
    assert method._trainer.strategy == {"K|?|b|0": {"k": 1.0}}
