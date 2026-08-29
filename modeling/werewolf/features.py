"""Feature extraction from Werewolf trajectories for opponent modeling.

Per-player behavioral signals, computed per-game then aggregated across games.
Designed to test whether roles leave distinct statistical signatures.

Features:
  ── Voting ──
  vote_targets       — list of who the player voted for (per round)
  h_vote             — Shannon entropy of voting targets (0 = always same person)

  ── Discussion ──
  mention_targets    — list of P<N> mentioned in discussion (per round)
  h_speak            — Shannon entropy of mention targets (0 = always same person)

  ── Consistency ──
  c_accuse_vote      — fraction of rounds where at least one mention matches vote

  ── Accuracy ──
  vote_accuracy      — fraction of votes that targeted a werewolf
  mention_accuracy   — fraction of discussion mentions that reference a werewolf

  ── Social ──
  majority_alignment — fraction of votes that matched the round's majority

  ── Volume ──
  speak_count        — number of discussion messages
  total_words        — total words spoken
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from base.types import Trajectory


@dataclass
class PlayerFeatures:
    """Per-player behavioral features from a single game."""
    player_id: int
    role: str
    side: str

    # ── Raw targets (per-round, for accumulation across games) ──
    vote_targets: list[int] = field(default_factory=list)
    mention_targets: list[int] = field(default_factory=list)  # all P<N> refs

    # ── Consistency ──
    matched_rounds: int = 0       # rounds where mention ∩ vote ≠ ∅
    total_rounds: int = 0         # rounds where both mention and vote exist

    # ── Accuracy ──
    votes_on_wolf: int = 0
    mentions_on_wolf: int = 0

    # ── Social ──
    voted_with_majority: int = 0
    votes_cast: int = 0

    # ── Volume ──
    speak_count: int = 0
    total_words: int = 0

    # ── Derived properties ──

    @property
    def h_vote(self) -> float:
        """Shannon entropy of voting targets. 0 = always same person."""
        return _entropy(self.vote_targets)

    @property
    def h_speak(self) -> float:
        """Shannon entropy of discussion mention targets."""
        return _entropy(self.mention_targets)

    @property
    def c_accuse_vote(self) -> float:
        """Fraction of rounds where at least one mention matches the vote."""
        if self.total_rounds == 0:
            return 0.0
        return self.matched_rounds / self.total_rounds

    @property
    def vote_accuracy(self) -> float:
        """Fraction of votes that targeted a werewolf."""
        if self.votes_cast == 0:
            return 0.0
        return self.votes_on_wolf / self.votes_cast

    @property
    def mention_accuracy(self) -> float:
        """Fraction of discussion mentions that referenced a werewolf."""
        total = len(self.mention_targets)
        if total == 0:
            return 0.0
        return self.mentions_on_wolf / total

    @property
    def majority_alignment(self) -> float:
        """Fraction of votes that matched the round's majority."""
        if self.votes_cast == 0:
            return 0.0
        return self.voted_with_majority / self.votes_cast

    @property
    def avg_words_per_speak(self) -> float:
        if self.speak_count == 0:
            return 0.0
        return self.total_words / self.speak_count


def _entropy(values: list[int]) -> float:
    """Shannon entropy on a list of discrete values."""
    if not values:
        return 0.0
    counter = Counter(values)
    total = len(values)
    h = -sum(c / total * math.log2(c / total) for c in counter.values())
    return h + 0.0  # avoid -0.0


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_player_refs(text: str, num_players: int) -> list[int]:
    """Return all P<N> references in text as ints."""
    return [int(m.group(1)) for m in re.finditer(r'\bP(\d+)\b', text)
            if 0 <= int(m.group(1)) < num_players]


def _normalize_target(target) -> int | None:
    """Accept 'P2', '2', or 2."""
    if target is None:
        return None
    if isinstance(target, int):
        return target
    s = str(target).strip().upper()
    if s.startswith("P"):
        try:
            return int(s[1:])
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


# ── Extraction ───────────────────────────────────────────────────────────

def extract_player_features(
    trajectory: Trajectory,
    roles: dict[int, str],
    role_specs: dict[str, dict],
) -> dict[int, PlayerFeatures]:
    """Extract per-player behavioral features from a single game trajectory."""
    # JSON serialization converts int keys to strings; normalize to int
    if roles and all(isinstance(k, str) for k in roles):
        roles = {int(k): v for k, v in roles.items()}

    num_players = len(roles)
    werewolves = {pid for pid, r in roles.items() if r == "werewolf"}

    feats = {
        pid: PlayerFeatures(
            player_id=pid,
            role=roles[pid],
            side=role_specs.get(roles[pid], {}).get("side", "unknown"),
        )
        for pid in roles
    }

    # Track last mention set per player per round (reset each round)
    round_mentions: dict[int, set[int]] = {}

    for step in trajectory.steps:
        pid = step.player_id
        action = step.parsed_action
        if action is None:
            continue
        feat = feats[pid]
        rd = step.state_after.get("round") or step.state_before.get("round")

        if action.type == "speak":
            msg = str(action.data.get("message", ""))
            feat.speak_count += 1
            feat.total_words += len(msg.split())
            refs = _extract_player_refs(msg, num_players)
            for r in refs:
                feat.mention_targets.append(r)
                if r in werewolves:
                    feat.mentions_on_wolf += 1
            if refs:
                round_mentions[pid] = set(refs)

        elif action.type == "vote":
            target = _normalize_target(action.data.get("target"))
            if target is None:
                continue
            feat.vote_targets.append(target)
            feat.votes_cast += 1
            if target in werewolves:
                feat.votes_on_wolf += 1

            # Accusation-vote consistency: any mention in this round matches vote
            if pid in round_mentions:
                feat.total_rounds += 1
                if target in round_mentions[pid]:
                    feat.matched_rounds += 1

    # Majority alignment
    _compute_majority_alignment(trajectory, feats)

    return feats


def _compute_majority_alignment(
    trajectory: Trajectory,
    feats: dict[int, PlayerFeatures],
) -> None:
    """Compute per-round majority and check each voter's alignment."""
    round_votes: dict[int, dict[int, int]] = {}

    for step in trajectory.steps:
        pid = step.player_id
        action = step.parsed_action
        if action is None or action.type != "vote":
            continue
        rd = step.state_after.get("round") or step.state_before.get("round")
        if rd is None:
            continue
        target = _normalize_target(action.data.get("target"))
        if target is None:
            continue
        if rd not in round_votes:
            round_votes[rd] = {}
        round_votes[rd][pid] = target

    for rd, votes in round_votes.items():
        if len(votes) < 2:
            continue
        tally = Counter(votes.values())
        max_count = max(tally.values())
        leaders = {t for t, c in tally.items() if c == max_count}
        if len(leaders) == 1:
            majority_target = leaders.pop()
            for voter in votes:
                if voter in feats and votes[voter] == majority_target:
                    feats[voter].voted_with_majority += 1


# ── Aggregation ──────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "h_vote", "h_speak", "c_accuse_vote",
    "vote_accuracy", "mention_accuracy",
    "majority_alignment", "speak_count", "total_words",
]


def aggregate_features(
    per_game: list[dict[int, PlayerFeatures]],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate features across games, grouped by role."""
    agg: dict[str, dict[str, list[float]]] = {}
    for game_feats in per_game:
        for pid, feat in game_feats.items():
            role = feat.role
            if role not in agg:
                agg[role] = {k: [] for k in FEATURE_NAMES}
            for name in FEATURE_NAMES:
                agg[role][name].append(getattr(feat, name))
    return agg


def format_aggregate_summary(agg: dict[str, dict[str, list[float]]]) -> str:
    """Pretty-print aggregate feature stats per role."""
    lines = []
    for role in sorted(agg):
        lines.append(f"\n── {role} ──")
        for feat_name in FEATURE_NAMES:
            values = agg[role].get(feat_name, [])
            if not values:
                continue
            mean_v = sum(values) / len(values)
            stdev_v = math.sqrt(
                sum((x - mean_v) ** 2 for x in values) / len(values)
            ) if len(values) > 1 else 0.0
            lines.append(
                f"  {feat_name:22s}: μ={mean_v:.3f}  σ={stdev_v:.3f}  n={len(values)}"
            )
    return "\n".join(lines)
