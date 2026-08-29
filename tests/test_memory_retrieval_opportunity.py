from __future__ import annotations

from analysis.memory_retrieval_opportunity import observable_similarity


def _leduc_feature(*, history: tuple = (), hand: str = "K", board: str = "Q", stage=(0, "preflop"), to_call=0, raises=0, pot=0, pair=False):
    return {
        "history": history,
        "legal": ("check", "bet"),
        "stage": stage,
        "hand_rank": hand,
        "board_rank": board,
        "pair": pair,
        "to_call": to_call,
        "raises": raises,
        "pot": pot,
    }


def test_leduc_similarity_is_one_for_identical_features():
    current = _leduc_feature(
        history=(("me", "bet", 0, 2), ("opp", "call", 0, 2)),
        hand="Q",
        board="K",
        stage=(1, "flop"),
        to_call=2,
        raises=1,
        pot=6,
        pair=True,
    )
    past = _leduc_feature(
        history=(("me", "bet", 0, 2), ("opp", "call", 0, 2)),
        hand="Q",
        board="K",
        stage=(1, "flop"),
        to_call=2,
        raises=1,
        pot=6,
        pair=True,
    )

    assert observable_similarity("leduc", current, past) == 1.0


def test_leduc_similarity_becomes_continuous_when_only_partially_matched():
    current = _leduc_feature(
        history=(("me", "bet", 0, 2), ("opp", "call", 0, 2), ("me", "check", 1, 0)),
        hand="Q",
        board="K",
        stage=(1, "flop"),
        to_call=2,
        raises=1,
        pot=6,
        pair=True,
    )
    past = _leduc_feature(
        history=(("me", "bet", 0, 2), ("opp", "raise", 0, 4)),
        hand="Q",
        board="J",
        stage=(1, "flop"),
        to_call=1,
        raises=0,
        pot=4,
        pair=False,
    )

    score = observable_similarity("leduc", current, past)

    assert 0.0 < score < 1.0
