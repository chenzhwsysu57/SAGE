from base.ablation import AblationConfig
from runs.additive import ADDITIVE_CONDITIONS, FirstLegalMockModel, parse_csv


def test_additive_conditions_map_to_deletion_flags() -> None:
    llm = ADDITIVE_CONDITIONS["llm"].to_ablation()
    assert llm.no_nash is True
    assert llm.no_belief is True

    nash = ADDITIVE_CONDITIONS["llm_nash"].to_ablation()
    assert nash.no_nash is False
    assert nash.no_belief is True

    om = ADDITIVE_CONDITIONS["llm_om"].to_ablation()
    assert om.no_nash is True
    assert om.no_belief is False

    nash_om = ADDITIVE_CONDITIONS["llm_nash_om"].to_ablation()
    assert nash_om.no_nash is False
    assert nash_om.no_belief is False

    full = ADDITIVE_CONDITIONS["full"].to_ablation()
    assert full.no_nash is False
    assert full.no_belief is False
    assert not hasattr(full, "no_failed_trajectory")
    assert not hasattr(full, "no_qwindow")


def test_failed_trajectory_is_no_longer_a_valid_ablation_flag() -> None:
    assert "failed_trajectory" not in AblationConfig._FLAG_MAP
    try:
        AblationConfig.from_comma_separated("failed_trajectory")
    except ValueError:
        pass
    else:
        raise AssertionError("failed_trajectory should no longer be a valid ablation flag")


def test_qwindow_is_no_longer_a_valid_ablation_flag() -> None:
    assert "qwindow" not in AblationConfig._FLAG_MAP
    try:
        AblationConfig.from_comma_separated("qwindow")
    except ValueError:
        pass
    else:
        raise AssertionError("qwindow should no longer be a valid ablation flag")


def test_additive_condition_metadata_records_enabled_components() -> None:
    metadata = ADDITIVE_CONDITIONS["llm_nash_om"].as_dict()
    assert metadata["use_nash"] is True
    assert metadata["use_opponent_model"] is True
    assert metadata["use_qwindow"] is True
    assert metadata["use_memory"] is False


def test_parse_csv_rejects_unknown_values() -> None:
    try:
        parse_csv("llm,bogus", valid=set(ADDITIVE_CONDITIONS), label="condition")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("parse_csv accepted an unknown condition")


def test_first_legal_mock_model_returns_parseable_json() -> None:
    model = FirstLegalMockModel()
    response = model.generate(
        "Legal actions:\n"
        "  bid (amount=3)\n"
        "  challenge\n"
    )
    assert '"type": "bid"' in response
    assert '"amount": 3' in response
