"""Persona classifier module for ToM method integration.

Usage:
    from modeling.goofspiel.classifier import PersonaClassifier, PersonaModelingMethod

    # Load pre-computed signatures, then classify:
    classifier = PersonaClassifier()
    classifier.record_game(trajectory, my_player_id)
    posterior = classifier.posterior()  # {"Greedy": 0.8, "Conservative": 0.1, ...}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow import from repo root
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ── Observable features (same as classifier analysis) ─────────────────────

OBSERVABLE_FEATURES = [
    "opp_win_rate", "tie_rate",
    "opp_win_rate_high_prize", "opp_win_rate_mid_prize", "opp_win_rate_low_prize",
    "opp_win_rate_early", "opp_win_rate_late", "opp_win_shift",
    "opp_win_rate_trailing", "opp_win_rate_leading", "opp_responsiveness",
    "avg_prize_when_opp_won", "avg_prize_when_opp_lost",
    "opp_final_score_mean", "opp_final_score_std",
    "inferred_lower_bound_mean", "inferred_upper_bound_mean",
    "exact_card_knowledge_rate",
    "my_bid_when_opp_won_mean", "my_bid_when_i_won_mean",
]

# Default signature path (auto-detect newest)
_DEFAULT_SIG_DIR = "results/goofspiel/persona_signature"


def _latest_signature_path() -> str:
    """Find the newest _signature.json file."""
    sig_dir = _repo_root / _DEFAULT_SIG_DIR
    candidates = sorted(sig_dir.glob("*_signature.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No signature file found in {sig_dir}. "
            "Run 'python analysis/goofspiel/persona_signature.py' first."
        )
    return str(candidates[0])


def _load_reference_signatures(
    path: str | None = None,
) -> tuple[dict[str, dict], list[str], np.ndarray, np.ndarray]:
    """Load reference signatures and pre-compute normalization.

    Returns:
        ref_signatures: {persona: {feature: value}}
        personas: list of persona names
        ref_matrix: (n_personas, n_features) array
        feature_names: list of feature names
    """
    with open(path or _latest_signature_path()) as f:
        data = json.load(f)

    raw_features = data["features"]
    all_personas = sorted(raw_features.keys())

    # Filter to only observable features that exist in ALL personas
    ref_signatures = {}
    for pn in all_personas:
        ref_signatures[pn] = {
            fn: raw_features[pn].get(fn, 0.0)
            for fn in OBSERVABLE_FEATURES
            if fn in raw_features[pn]
        }

    # Determine actual available features
    feature_names = [fn for fn in OBSERVABLE_FEATURES
                     if all(fn in ref_signatures[p] for p in all_personas)]

    personas = sorted(ref_signatures.keys())
    ref_matrix = np.array([
        [ref_signatures[p][fn] for fn in feature_names]
        for p in personas
    ])

    return ref_signatures, personas, ref_matrix, feature_names


# ── Signature computation from round data ─────────────────────────────────

def _sig_from_rounds(all_rounds: list[dict]) -> dict:
    """Compute observable feature vector from list of round dicts.

    Each round dict must have:
        round: int (1-based)
        prize: int
        my_bid: int
        outcome: int (+1=win, -1=loss, 0=tie)
    """
    if not all_rounds:
        return {fn: 0.0 for fn in OBSERVABLE_FEATURES}

    outcomes = np.array([r["outcome"] for r in all_rounds])
    prizes = np.array([r["prize"] for r in all_rounds])
    rounds_arr = np.array([r["round"] for r in all_rounds])
    my_bids = np.array([r["my_bid"] for r in all_rounds])

    opp_won = outcomes == -1  # opponent (persona) won
    opp_lost = outcomes == 1   # I won → opponent lost
    tied = outcomes == 0

    features: dict = {fn: 0.0 for fn in OBSERVABLE_FEATURES}
    n = len(all_rounds)

    # Win/loss/tie rates
    features["opp_win_rate"] = round(float(np.mean(opp_won)), 4) if n else 0.0
    features["tie_rate"] = round(float(np.mean(tied)), 4) if n else 0.0

    # By prize level
    high = prizes >= 5
    mid = (prizes >= 3) & (prizes <= 4)
    low = prizes <= 2
    for name, mask in [("high_prize", high), ("mid_prize", mid), ("low_prize", low)]:
        if mask.sum() > 0:
            features[f"opp_win_rate_{name}"] = round(float(np.mean(opp_won[mask])), 4)

    # Early vs Late
    early = rounds_arr <= 3
    late = rounds_arr >= 4
    if early.sum() > 0:
        features["opp_win_rate_early"] = round(float(np.mean(opp_won[early])), 4)
    if late.sum() > 0:
        features["opp_win_rate_late"] = round(float(np.mean(opp_won[late])), 4)
    if early.sum() > 0 and late.sum() > 0:
        features["opp_win_shift"] = round(
            float(np.mean(opp_won[late]) - np.mean(opp_won[early])), 4
        )

    # Score-state conditional
    # Compute before-round scores
    score_per_before = []
    running = 0.0
    for r in all_rounds:
        score_per_before.append(running)
        if r["outcome"] == -1:
            running += r["prize"]
        elif r["outcome"] == 0:
            running += r["prize"] / 2

    score_cfr_before = []
    running = 0.0
    for r in all_rounds:
        score_cfr_before.append(running)
        if r["outcome"] == 1:
            running += r["prize"]
        elif r["outcome"] == 0:
            running += r["prize"] / 2

    score_diff_before = np.array(score_per_before) - np.array(score_cfr_before)
    trailing = score_diff_before < -1
    leading = score_diff_before > 1
    close = (score_diff_before >= -1) & (score_diff_before <= 1)

    for name, mask in [("trailing", trailing), ("leading", leading), ("close", close)]:
        if mask.sum() > 0:
            features[f"opp_win_rate_{name}"] = round(float(np.mean(opp_won[mask])), 4)

    if trailing.sum() > 0 and leading.sum() > 0:
        wr_trail = np.mean(opp_won[trailing])
        wr_lead = np.mean(opp_won[leading])
        features["opp_responsiveness"] = round(float(abs(wr_trail - wr_lead)), 4)

    # Final score
    if all_rounds:
        features["opp_final_score_mean"] = round(float(all_rounds[-1].get("score_opponent", 0)), 2)
        features["opp_final_score_std"] = 0.0  # single game, no std

    # Average prize won/lost
    prize_when_won = prizes[opp_won]
    prize_when_lost = prizes[opp_lost]
    features["avg_prize_when_opp_won"] = round(float(np.mean(prize_when_won)), 2) if len(prize_when_won) > 0 else 0.0
    features["avg_prize_when_opp_lost"] = round(float(np.mean(prize_when_lost)), 2) if len(prize_when_lost) > 0 else 0.0

    # Inferred bounds
    opp_won_mask = opp_won
    if opp_won_mask.sum() > 0:
        lb = my_bids[opp_won_mask] + 1
        features["inferred_lower_bound_mean"] = round(float(np.mean(lb)), 2)
    opp_lost_mask = opp_lost
    if opp_lost_mask.sum() > 0:
        ub = my_bids[opp_lost_mask] - 1
        features["inferred_upper_bound_mean"] = round(float(np.mean(ub)), 2)

    features["exact_card_knowledge_rate"] = features["tie_rate"]

    if opp_won_mask.sum() > 0:
        features["my_bid_when_opp_won_mean"] = round(float(np.mean(my_bids[opp_won_mask])), 2)
    if opp_lost_mask.sum() > 0:
        features["my_bid_when_i_won_mean"] = round(float(np.mean(my_bids[opp_lost_mask])), 2)

    return features


def _sig_from_multi_game_rounds(all_games_rounds: list[list[dict]]) -> dict:
    """Compute cumulative signature from multiple games of round data."""
    flat = [r for game in all_games_rounds for r in game]
    sig = _sig_from_rounds(flat)

    # Compute std for final scores across games
    final_scores = [game[-1].get("score_opponent", 0) if game else 0
                    for game in all_games_rounds if game]
    if len(final_scores) > 1:
        sig["opp_final_score_mean"] = round(float(np.mean(final_scores)), 2)
        sig["opp_final_score_std"] = round(float(np.std(final_scores)), 2)

    return sig


# ── PersonaClassifier ────────────────────────────────────────────────────

class PersonaClassifier:
    """Cumulative persona classifier.

    Maintains per-round data across games. After each game, call
    record_game() to feed the trajectory. Then query posterior() to
    get the current belief distribution.
    """

    def __init__(self, signature_path: str | None = None):
        self.ref_sigs, self.personas, self.ref_mat, self.fnames = \
            _load_reference_signatures(signature_path)

        # Pre-compute normalization stats from reference
        self._fmin = self.ref_mat.min(axis=0)
        self._fmax = self.ref_mat.max(axis=0)
        self._frange = self._fmax - self._fmin
        self._frange[self._frange < 1e-10] = 1.0

        # Cumulative game data: list of lists of round dicts
        self._games: list[list[dict]] = []

    @property
    def n_games(self) -> int:
        return len(self._games)

    @property
    def n_rounds(self) -> int:
        return sum(len(g) for g in self._games)

    def record_game(self, trajectory, my_player_id: int) -> None:
        """Extract observable round data from a completed trajectory."""
        game_rounds = _extract_rounds(trajectory, my_player_id)
        if game_rounds:
            self._games.append(game_rounds)

    def posterior(self, temperature: float = 0.5) -> dict[str, float]:
        """Compute P(persona | all observed games).

        Returns {persona: probability} sorted by probability descending.
        """
        if not self._games:
            return {p: 1.0 / len(self.personas) for p in self.personas}

        sig = _sig_from_multi_game_rounds(self._games)
        vec = np.array([sig.get(fn, 0.0) for fn in self.fnames], dtype=float)
        vec_n = (vec - self._fmin) / self._frange
        ref_n = (self.ref_mat - self._fmin) / self._frange

        dists = np.array([np.linalg.norm(vec_n - ref_n[i])
                          for i in range(len(self.personas))])
        neg_d = -dists / temperature
        exp_v = np.exp(neg_d - neg_d.max())
        probs = exp_v / exp_v.sum()

        result = dict(zip(self.personas, [round(float(p), 4) for p in probs]))
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def top_k(self, k: int = 3, temperature: float = 0.5) -> list[tuple[str, float]]:
        """Return top-k (persona, probability) pairs."""
        return list(self.posterior(temperature).items())[:k]

    def reset(self) -> None:
        self._games.clear()


def _extract_rounds(trajectory, my_player_id: int) -> list[dict]:
    """Extract ToM-observable per-round data from trajectory.

    Returns list of {round, prize, my_bid, outcome, score_opponent}.
    outcome: +1 = I won, -1 = opponent won, 0 = tie.
    """
    # Get my steps sorted by round
    my_steps = sorted(
        [s for s in trajectory.steps if s.player_id == my_player_id],
        key=lambda s: s.step,
    )

    if not my_steps:
        return []

    rounds = []
    for s in my_steps:
        pub = s.observation.public_info
        if not pub:
            continue
        rnd_num = pub.get("round", 0)
        prize = pub.get("prize", 0)
        my_bid = s.parsed_action.amount if s.parsed_action else 0
        oh = pub.get("outcome_history", [])

        # outcome_history from P0's perspective
        # At this point, outcomes of rounds before this one are known
        # If I'm P0 at round r: oh has r-1 entries
        # If I'm P1 at round r: oh has r entries (P1 acts after P0 in same round,
        #   so outcome of round r is already known due to sequential resolution)

        # Determine which round this step is for
        # The step's round is in pub["round"]

        rounds.append({
            "round": rnd_num,
            "prize": prize,
            "my_bid": my_bid,
            "outcome": 0,  # placeholder, fixed below
            "_oh": oh,
            "_step": s.step,
            "_pid": s.player_id,
        })

    if not rounds:
        return []

    # Determine outcomes from outcome_history
    # outcome_history is from P0 perspective
    # We need to map it to my_player_id's perspective

    # Sort by step to process in order
    rounds.sort(key=lambda x: x["_step"])

    # For P0 (acts first):
    #   My step at round r sees oh with r-1 entries (outcomes before this round)
    #
    # For P1 (acts second):
    #   My step at round r sees oh with r entries (outcomes before this round,
    #   including the current round's outcome if the opponent has already acted)
    #
    # The key insight: after both players have acted in round r, the outcome
    # goes into outcome_history. When P0 enters round r+1, they see outcome of r.
    # When P1 enters round r+1, they also see outcome of r.

    # Actually, let me use a simpler approach: compute outcomes from
    # comparing scores between consecutive steps.

    # Get ALL steps sorted
    all_steps = sorted(trajectory.steps, key=lambda s: s.step)

    # Build a map: step_idx -> scores after that step
    # We can determine scores from the next step's observation
    scores_after = {}
    for i, step in enumerate(all_steps):
        pub = step.observation.public_info
        if pub and "scores" in pub:
            scores_after[step.step] = pub["scores"]

    # For my P0 rounds: my round r is at step 2r-2 (0-indexed)
    # The outcome is visible in the next P0 step's scores
    # My score before round r = scores at my step r
    # My score after round r = scores at my step r+1 (next round)
    # Score delta = my score change due to this round

    # For the final round, use the last observation or final payout

    # Actually, let me use the outcome_history. Given the sequential nature:
    # P0 acts first in round r (step 2r-2, 0-indexed)
    # P1 acts second in round r (step 2r-1)
    # After step 2r-1 (P1's step for round r), outcome becomes known
    #
    # So for P0's step for round r+1 (step 2r), outcome_history has r entries
    #   (outcomes of rounds 1 through r)
    # For P1's step for round r+1 (step 2r+1), outcome_history also has r entries
    #
    # But wait — the Goofspiel II variant is:
    #   P0 bids (P0's bid is publicly revealed)
    #   P1 bids (knowing P0's bid)
    #   Result is resolved
    #
    # So:
    # - P0's step at round r: oh before = r-1 entries
    # - P1's step at round r: oh before = r-1 entries (outcome of r-1)
    # - After P1's step: outcome resolved, oh = r entries
    #
    # So the outcome of round r is available at step 2r (P0 of round r+1)
    #   as oh[r-1] (0-indexed)

    # OK, using the final trajectory, after all steps are done:
    # outcome_history at the LAST step should have all 6 entries
    # (or none if something is strange)

    # Get the FINAL outcome_history from state_after of the last step.
    # observation.public_info only has outcomes of rounds completed BEFORE
    # the current step, so it misses the last round's outcome.
    last_oh = []
    last_step = all_steps[-1]
    if last_step.state_after and "outcome_history" in last_step.state_after:
        last_oh = last_step.state_after["outcome_history"]
    else:
        # Fallback: try from observation
        for step in reversed(all_steps):
            pub = step.observation.public_info
            if pub and pub.get("outcome_history"):
                last_oh = pub["outcome_history"]
                break

    # Also try to get outcome_history from my last step
    my_last_oh = rounds[-1].get("_oh", [])

    # Use the longer one
    oh = last_oh if len(last_oh) >= len(my_last_oh) else my_last_oh

    # Build a map: round_number -> outcome from P0's perspective
    rnd_to_outcome_p0 = {}
    for i, o in enumerate(oh):
        rnd_to_outcome_p0[i + 1] = o  # oh[0] = round 1, oh[1] = round 2, etc.

    # If some rounds are missing from oh (shouldn't happen with 6 rounds),
    # fill them from score changes
    if len(rnd_to_outcome_p0) < len(rounds):
        # Compute from score changes
        for i, rnd in enumerate(rounds):
            if rnd["round"] not in rnd_to_outcome_p0:
                # Try from the next round's scores
                pass

    # Apply outcomes to my round data
    for rnd in rounds:
        rn = rnd["round"]
        if rn in rnd_to_outcome_p0:
            outcome_p0 = rnd_to_outcome_p0[rn]
            if my_player_id == 0:
                rnd["outcome"] = outcome_p0
            else:
                rnd["outcome"] = -outcome_p0
        else:
            rnd["outcome"] = 0  # default tie

    # Add opponent's final score for this round
    for rnd in rounds:
        rn = rnd["round"]
        # When I'm P0 at round r and step 2r-2, the scores are from before round r
        # The opponent's score before = scores[1]
        # But we want the opponent's score AFTER the round
        # We can compute it: opponent score before + prize if they won, etc.
        # But this is complex. For now, use the score AFTER from the next available step.

        # Simpler: compute from the outcome directly
        # opponent score after round r = opponent score before r + change
        pass

    # Compute opponent's running score
    opp_scores = []
    running = 0.0
    for rnd in sorted(rounds, key=lambda x: x["round"]):
        opp_scores.append(running)
        if rnd["outcome"] == -1:  # opponent won
            running += rnd["prize"]
        elif rnd["outcome"] == 0:  # tie
            running += rnd["prize"] / 2
        # loss: no change

    for i, rnd in enumerate(rounds):
        rnd["score_opponent"] = opp_scores[i]

    # Clean up internal fields
    result = []
    for rnd in sorted(rounds, key=lambda x: x["round"]):
        result.append({
            "round": rnd["round"],
            "prize": rnd["prize"],
            "my_bid": rnd["my_bid"],
            "outcome": rnd["outcome"],
            "score_opponent": rnd["score_opponent"],
        })

    return result


# ── PersonaModelingMethod (plugs into ToMMethod) ─────────────────────────

class PersonaModelingMethod:
    """BaseModelingMethod-compatible wrapper around PersonaClassifier.

    Plugs into ToMMethod as modeling_method.

    Usage:
        classifier = PersonaModelingMethod()
        tom = ToMMethod(name="ToM", modeling_method=classifier)
    """

    def __init__(self, signature_path: str | None = None):
        self.classifier = PersonaClassifier(signature_path)
        self._opponent_id: int | None = None

    def update(self, opp_id: int, trajectory, model=None) -> None:
        """Process a completed game. Called by ToMMethod.reflect()."""
        self._opponent_id = opp_id
        # The trajectory contains both players' steps.
        # The ToM player's ID is NOT my_id — it's the player that ISN'T the opponent.
        # from_opponent = trajectory has steps with player_id
        # My player ID = whoever is NOT the opponent
        all_ids = {s.player_id for s in trajectory.steps if s.player_id >= 0}
        my_ids = all_ids - {opp_id}
        my_id = next(iter(my_ids)) if my_ids else 0
        self.classifier.record_game(trajectory, my_id)

    def statement(self, opp_id: int, **kwargs) -> str:
        """Human-readable posterior for LLM prompt injection."""
        post = self.classifier.posterior(temperature=0.5)
        top = list(post.items())[:3]

        n = self.classifier.n_games
        if n == 0:
            return ""

        lines = [
            f"Persona classification after {n} game(s):",
        ]
        for p, prob in top:
            pct = prob * 100
            lines.append(f"  - {p}: {pct:.0f}%")

        hint = self.exploit_hint(opp_id)
        if hint:
            lines.append(f"Advice: {hint}")
        else:
            lines.append(
                "Implication: adapt play to exploit the most likely persona type."
            )
        return "\n".join(lines)

    def exploit_hint(self, opp_id: int) -> str:
        """Exploitation advice based on top persona."""
        top = self.classifier.top_k(1, temperature=0.5)
        if not top:
            return ""
        persona, prob = top[0]
        if prob < 0.4:
            return "Low confidence — play balanced GTO."
        hints = {
            "Greedy": "Opponent likely overbids. Bid just above their expected card.",
            "Conservative": "Opponent likely underbids. You can win with low bids.",
            "Sacrificial": "Opponent sacrifices round 1, then overbids. Win round 1 cheaply, then mirror.",
            "ScoreAware": "Opponent reacts to score. Build lead early to make them conservative.",
            "Random": "No exploitable pattern — play GTO.",
            "GTO": "Opponent plays Nash equilibrium — play GTO.",
        }
        return hints.get(persona, "Play GTO.")

    def reset(self, opp_id: int | None = None) -> None:
        self.classifier.reset()


# ── Entry point: print-convergence script ─────────────────────────────────

def main():
    """Run a simulation showing classifier convergence for each persona."""
    from games.goofspiel import Goofspiel
    from rollout import RolloutRunner
    from methods.sage.goofspiel_cfr_method import GoofspielCFRMethod
    from modeling.goofspiel import GoofspielArchetypeAgent

    print("=" * 80)
    print("  PERSONA CLASSIFIER — Convergence Simulation")
    print("  (CFR method vs each persona, 30 games each)")
    print("=" * 80)

    classifier = PersonaClassifier()

    for persona in ["Greedy", "Conservative", "Sacrificial",
                    "ScoreAware", "Random", "GTO"]:
        print(f"\n{'─' * 80}")
        print(f"  True persona: {persona}")
        print(f"{'─' * 80}")

        classifier.reset()

        for seed in range(30):
            game = Goofspiel(n=6)
            methods = {
                0: GoofspielCFRMethod(model_path=None, name="CFR", seed=seed),
                1: GoofspielArchetypeAgent(persona, name=persona, seed=seed + 1000),
            }
            runner = RolloutRunner(game, methods, model=None, verbose=False)
            trajectory = runner.run()

            # Classify from P0's perspective (CFR = ToM player)
            classifier.record_game(trajectory, my_player_id=0)

            post = classifier.posterior(temperature=0.5)
            top = list(post.items())
            rank = next(i + 1 for i, (p, _) in enumerate(top) if p == persona)
            p_correct = post[persona]

            if (seed + 1) in [1, 2, 3, 5, 10, 15, 20, 30]:
                top3 = ", ".join(f"{p}={prob:.2f}" for p, prob in top[:3])
                print(f"  Game {seed+1:>2}: P(correct)={p_correct:.3f}  rank={rank}  [{top3}]")

    print(f"\nDone. Classifier used {len(classifier.fnames)} observable features.")


if __name__ == "__main__":
    main()
