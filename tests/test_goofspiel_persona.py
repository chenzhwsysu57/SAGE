"""Tests for Goofspiel persona agents (LLM-based with rule-based fallback)."""

from __future__ import annotations

import pytest

from games.goofspiel import Goofspiel
from modeling.goofspiel import (
    GoofspielArchetypeAgent,
    GoofspielPersonaAgent,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def make_observation(game: Goofspiel, state, player_id: int):
    return game.get_observation(state, player_id)


def set_prize(state, prize_seq: list[int], round: int):
    """Override prize fields for deterministic testing."""
    state.prize_sequence = tuple(prize_seq)
    state.round = round
    state.prize = prize_seq[round - 1]


# ── Greedy: prize-driven binary (top-third → max, else → min) ────────────

def test_greedy_fights_for_top_tier_prize():
    """Greedy bids max when prize is in top third of remaining."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [6, 1, 5, 2, 4, 3], 1)  # round 1, prize=6, top third ✓
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Greedy")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 6, f"Greedy should bid 6 for top-tier prize, got {action.amount}"


def test_greedy_folds_for_low_tier_prize():
    """Greedy bids min when prize is NOT in top third of remaining."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [1, 6, 5, 2, 4, 3], 1)  # round 1, prize=1, bottom tier ✓
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Greedy")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 1, f"Greedy should bid 1 for low-tier prize, got {action.amount}"


def test_greedy_considers_remaining_not_absolute():
    """Greedy judges top-third relative to REMAINING prizes, not all prizes."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [3, 6, 5, 2, 4, 1], 1)  # prize=3, remaining=[3,6,5,2,4,1]
    # remaining top third: sorted([6,5,4,3,2,1], reverse)[:2] = [6,5]
    # prize=3 → NOT top third → fold
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Greedy")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 1, f"Greedy should fold with prize=3 when 6,5 remain, got {action.amount}"


def test_greedy_fights_in_late_game():
    """Greedy still fights for top-tier prizes late in the game."""
    game = Goofspiel(n=6)
    state = game.reset()
    state.hands[0] = {1, 2, 3, 4}
    set_prize(state, [2, 1, 4, 3, 6, 5], 5)  # round 5, prize=6, remaining=[6,5] → top third ✓
    state.outcome_history = [-1, 1, -1, 1]
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Greedy")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 4, f"Greedy should bid 4 (max remaining) for top-tier prize, got {action.amount}"


# ── Conservative: discount-seeker (target = max(1, prize-1)) ────────────

def test_conservative_targets_discount():
    """Conservative targets prize-1."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [5, 1, 2, 3, 4, 6], 1)  # prize=5, target=4
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Conservative")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 4, f"Conservative should target 4 (prize-1=4), got {action.amount}"


def test_conservative_clamps_to_1():
    """Conservative never targets below 1."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [1, 2, 3, 4, 5, 6], 1)  # prize=1, target=max(1,0)=1
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Conservative")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 1, f"Conservative should bid 1 when prize=1, got {action.amount}"


def test_conservative_shifts_with_partial_hand():
    """Conservative picks closest to prize-1 when exact card is gone."""
    game = Goofspiel(n=6)
    state = game.reset()
    state.hands[0] = {1, 2, 3, 5, 6}
    set_prize(state, [5, 1, 2, 3, 4, 6], 1)  # prize=5, target=4, 4 not in hand
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Conservative")
    action, _, _ = agent.act(obs, actions, model=None)
    # closest to 4 from {1,2,3,5,6} is 3 or 5 (tie → min)
    assert action.amount in (3, 5), f"Conservative should pick closest to 4, got {action.amount}"


# ── Sacrificial: largest prize → prize-2, others → prize ──────────

def test_sacrificial_bids_below_on_largest_prize():
    """Sacrificial bids prize-2 when prize is the largest remaining."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [6, 1, 2, 3, 4, 5], 1)  # round 1, prize=6 (largest remaining)
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Sacrificial")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 4, f"Sacrificial should bid 4 (prize-2) when prize is largest, got {action.amount}"


def test_sacrificial_bids_prize_on_non_largest():
    """Sacrificial bids prize value when prize is NOT the largest remaining."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [1, 2, 6, 5, 4, 3], 2)  # round 2, prize=2 (largest remaining is 6)
    state.outcome_history = [-1]
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Sacrificial")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 2, f"Sacrificial should bid 2 (prize value), got {action.amount}"


def test_sacrificial_bids_prize_midgame():
    """Sacrificial bids prize value on non-largest prizes in midgame."""
    game = Goofspiel(n=6)
    state = game.reset()
    state.hands[0] = {1, 2, 3}
    set_prize(state, [2, 1, 4, 3, 5, 6], 4)  # round 4, prize=4 (largest remaining is 6)
    state.outcome_history = [1, -1, 1]
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Sacrificial")
    action, _, _ = agent.act(obs, actions, model=None)
    # target=4, closest from {1,2,3} is 3
    assert action.amount == 3, f"Sacrificial should bid 3 (closest to prize=4), got {action.amount}"


# ── Random: uniform ────────────────────────────────────────────────────

def test_random_uses_all_cards():
    """Random picks from legal actions only (fallback)."""
    game = Goofspiel(n=6)
    state = game.reset()
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    seen = set()
    for seed in range(100):
        agent = GoofspielArchetypeAgent("Random", seed=seed)
        action, _, _ = agent.act(obs, actions, model=None)
        seen.add(action.amount)
    assert seen == {1, 2, 3, 4, 5, 6}, f"Random should cover all cards, got {seen}"


def test_random_only_legal():
    """Random never bids a card they don't have (fallback)."""
    game = Goofspiel(n=6)
    state = game.reset()
    state.hands[0] = {2, 4}
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    for seed in range(50):
        agent = GoofspielArchetypeAgent("Random", seed=seed)
        action, _, _ = agent.act(obs, actions, model=None)
        assert action.amount in {2, 4}, \
            f"Random should only bid legal cards, got {action.amount}"


# ── Mirror: bid = prize ──────────────────────────────────────

def test_mirror_bids_prize_value():
    """Mirror bids exactly the prize value when available."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [4, 1, 2, 3, 5, 6], 1)  # prize=4, hand has {1..6}
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Mirror")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 4, f"Mirror should bid 4 (prize value), got {action.amount}"


def test_mirror_bids_prize_value_midgame():
    """Mirror bids closest available when exact prize card is gone."""
    game = Goofspiel(n=6)
    state = game.reset()
    set_prize(state, [1, 5, 2, 4, 3, 6], 3)  # round 3, prize=2
    state.hands[0] = {1, 3, 4, 5}  # card 2 already used
    state.outcome_history = [1, -1]
    obs = make_observation(game, state, 0)
    actions = game.legal_actions(state)
    agent = GoofspielArchetypeAgent("Mirror")
    action, _, _ = agent.act(obs, actions, model=None)
    assert action.amount == 1, f"Mirror should bid 1 (closest to prize=2), got {action.amount}"


# ── Test GoofspielPersonaAgent (BaseMethod wrapper) ─────────────────────

def test_persona_agent_interface():
    """GoofspielPersonaAgent conforms to BaseMethod interface."""
    agent = GoofspielPersonaAgent("Greedy", seed=0)
    game = Goofspiel(n=6)
    state = game.reset()
    _, bid_tracker, _ = agent.act(*_quick_obs(game, state), model=None)
    assert bid_tracker == "persona_fallback"


def test_persona_agent_reflect_returns_list():
    """GoofspielPersonaAgent.reflect() returns empty list."""
    agent = GoofspielPersonaAgent("Greedy", seed=0)
    assert agent.reflect(None, None, "") == []


def _quick_obs(game, state):
    obs = game.get_observation(state, 0)
    return obs, game.legal_actions(state)
