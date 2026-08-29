"""Fast benchmark: CFR vs Mirror and CFR vs CFR (10k games each).

Usage:
    python runs/goofspiel/bench_cfr_mirror.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
from tqdm import tqdm

import pickle

from games.goofspiel import Goofspiel
from methods.goofspiel_cfr_method import GoofspielCFRMethod
from modeling.goofspiel.archetypes import GoofspielArchetypeAgent

# Load strategy once, share read-only across CFR instances
_CFR_STRATEGY = None

def _get_cfr_strategy(path):
    global _CFR_STRATEGY
    if _CFR_STRATEGY is not None:
        return _CFR_STRATEGY
    with open(path, "rb") as f:
        data = pickle.load(f)
    strategy_sum = data.get("strategy_sum", data)
    strategy = {}
    for infoset, sums in strategy_sum.items():
        total = sum(sums.values())
        if total > 0:
            strategy[infoset] = {k: v / total for k, v in sums.items()}
    _CFR_STRATEGY = strategy
    return strategy


def make_cfr(path=None, seed=0):
    """Create a CFR method with shared strategy dict (no reload)."""
    if path is None:
        path = MODEL_PATH
    m = GoofspielCFRMethod.__new__(GoofspielCFRMethod)
    m.name = "CFR"
    m.method_type = "cfr"
    m._rng = random.Random(seed)
    m._fallback_uniform = True
    m._strategy = _get_cfr_strategy(path)
    m._n_infosets = len(m._strategy)
    m._n = 5
    return m

N = 5
GAMES = 10000
MODEL_PATH = "results/goofspiel/models/goofspiel_n5_mccfr_cfrplus_seed42.pkl"
SEED = 42


def run_one_game(game, p0_method, p1_method, seed):
    """Play one game, return chips for P0 and P1."""
    state = game.reset(seed=seed)
    p0_method.name = "P0"
    p1_method.name = "P1"

    while not game.is_terminal(state):
        player = state.current_player
        obs = game.get_observation(state, player)
        actions = game.legal_actions(state)
        method = p0_method if player == 0 else p1_method
        action, _, _ = method.act(obs, actions, model=None)
        state = game.step(state, action)

    return state.scores[0] - state.scores[1]


def run_experiment(label, p0_method, p1_method, n_games):
    wins = 0
    losses = 0
    draws = 0
    total_chips = 0.0

    for i in tqdm(range(n_games), desc=label):
        chips = run_one_game(Goofspiel(n=N), p0_method, p1_method, seed=SEED + i)
        total_chips += chips
        if chips > 0:
            wins += 1
        elif chips < 0:
            losses += 1
        else:
            draws += 1

    wr = wins / n_games * 100
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"  Games:      {n_games}")
    print(f"  Win rate:   {wr:.2f}%  ({wins}W / {losses}L / {draws}D)")
    print(f"  Net chips:  {total_chips:+.1f}")
    print(f"  Avg chips:  {total_chips/n_games:+.4f}/game")
    return total_chips


if __name__ == "__main__":
    import time
    start = time.perf_counter()

    cfr = make_cfr()
    mirror = GoofspielArchetypeAgent("Mirror", seed=0)

    # CFR vs Mirror (P0=CFR, P1=Mirror)
    run_experiment("CFR (P0) vs Mirror (P1)", cfr, mirror, GAMES)

    # Mirror vs CFR (P0=Mirror, P1=CFR)
    run_experiment("Mirror (P0) vs CFR (P1)", mirror, make_cfr(), GAMES)

    # CFR vs CFR (self-play)
    run_experiment("CFR (P0) vs CFR (P1) — self-play",
                   make_cfr(),
                   make_cfr(),
                   GAMES)

    elapsed = time.perf_counter() - start
    print(f"\nTotal time: {elapsed:.0f}s")
