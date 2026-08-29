"""Werewolf opponent behavioral profile via cumulative feature tracking.

Mirrors the Leduc StatisticalProfileMethod architecture:
  - Online:  profile.update(opp_id, trajectory) after each game
  - Offline: profile.load_from_json(path) to replay saved trajectories
  - ToM:     profile.statement(opp_id) / profile.exploit_hint(opp_id)

Accumulates raw targets across games, recomputes derived features on demand.
This means h_vote and h_speak stabilise as more data arrives — no per-game
entropy collapse from only 1-2 votes per game.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field

from base.memory import BaseMemory
from base.types import Trajectory
from modeling.base import BaseModelingMethod
from modeling.werewolf.features import (
    PlayerFeatures,
    extract_player_features,
    _normalize_target,
    _entropy,
)


# ── Accumulated profile ──────────────────────────────────────────────────

@dataclass
class WerewolfProfile:
    """Cumulative behavioral profile for one opponent across games."""

    opp_id: int

    vote_targets: list[int] = field(default_factory=list)
    mention_targets: list[int] = field(default_factory=list)
    mentions_on_wolf: int = 0
    votes_on_wolf: int = 0
    matched_rounds: int = 0
    total_rounds: int = 0
    voted_with_majority: int = 0
    votes_cast: int = 0
    speak_count: int = 0
    total_words: int = 0
    game_count: int = 0

    def ingest(self, feat: PlayerFeatures) -> None:
        """Absorb one game's features into the cumulative profile."""
        self.vote_targets.extend(feat.vote_targets)
        self.mention_targets.extend(feat.mention_targets)
        self.mentions_on_wolf += feat.mentions_on_wolf
        self.votes_on_wolf += feat.votes_on_wolf
        self.matched_rounds += feat.matched_rounds
        self.total_rounds += feat.total_rounds
        self.voted_with_majority += feat.voted_with_majority
        self.votes_cast += feat.votes_cast
        self.speak_count += feat.speak_count
        self.total_words += feat.total_words
        self.game_count += 1

    # ── Derived (recomputed from accumulated raw data) ─────────────────

    @property
    def h_vote(self) -> float:
        return _entropy(self.vote_targets)

    @property
    def h_speak(self) -> float:
        return _entropy(self.mention_targets)

    @property
    def c_accuse_vote(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return self.matched_rounds / self.total_rounds

    @property
    def vote_accuracy(self) -> float:
        if self.votes_cast == 0:
            return 0.0
        return self.votes_on_wolf / self.votes_cast

    @property
    def mention_accuracy(self) -> float:
        total = len(self.mention_targets)
        if total == 0:
            return 0.0
        return self.mentions_on_wolf / total

    @property
    def majority_alignment(self) -> float:
        if self.votes_cast == 0:
            return 0.0
        return self.voted_with_majority / self.votes_cast

    @property
    def n_votes(self) -> int:
        return len(self.vote_targets)

    @property
    def n_mentions(self) -> int:
        return len(self.mention_targets)

    def summary(self) -> str:
        lines = [
            f"WerewolfProfile — P{self.opp_id} ({self.game_count} games, "
            f"{self.n_votes} votes, {self.n_mentions} mentions)",
        ]
        pairs = [
            ("h_vote", self.h_vote), ("h_speak", self.h_speak),
            ("c_accuse_vote", self.c_accuse_vote),
            ("vote_accuracy", self.vote_accuracy),
            ("mention_accuracy", self.mention_accuracy),
            ("majority_alignment", self.majority_alignment),
            ("speak_count", self.speak_count),
            ("total_words", self.total_words),
        ]
        for name, val in pairs:
            lines.append(f"  {name:22s}: {val:.3f}")
        return "\n".join(lines)


# ── Modeling method ──────────────────────────────────────────────────────

class WerewolfProfileMethod(BaseModelingMethod):
    """Cumulative behavioral opponent modeling for Werewolf.

    Accumulates per-game features for each opponent across games.
    Derived features (h_vote, h_speak, etc.) are recomputed from
    the full accumulated history — no single-game entropy collapse.

    Online use (via ToMMethod):
        profile.update(opp_id, trajectory)

    Offline use:
        profile.load_from_json("path/to/pilot_summary.json")

    Parameters:
        memory: persistence backend for cross-session state.
    """

    def __init__(self, memory: BaseMemory | None = None):
        self.memory = memory or BaseMemory()
        self._profiles: dict[int, WerewolfProfile] = {}

    # ── Persistence keys ──────────────────────────────────────────────

    def _profile_key(self, opp_id: int) -> str:
        return f"ww_profile_{opp_id}"

    # ── BaseModelingMethod interface ──────────────────────────────────

    def update(
        self,
        opp_id: int,
        trajectory: Trajectory,
        model=None,
    ) -> None:
        """Process a completed game. Accumulates features for the given opponent."""
        roles = trajectory.static_state.get("roles", {})
        if not roles:
            return

        from games.werewolf import ROLE_SPECS

        feats = extract_player_features(trajectory, roles, ROLE_SPECS)
        if opp_id not in feats:
            return

        profile = self._profiles.get(opp_id)
        if profile is None:
            profile = WerewolfProfile(opp_id=opp_id)
            self._profiles[opp_id] = profile

        profile.ingest(feats[opp_id])
        self._save(opp_id, profile)

    def statement(self, opp_id: int) -> str:
        """Human-readable profile summary for LLM prompt injection."""
        profile = self._load(opp_id)
        if profile is None or profile.game_count == 0:
            return ""
        return profile.summary()

    def exploit_hint(self, opp_id: int) -> str:
        """Return exploitation advice based on behavioral profile."""
        profile = self._load(opp_id)
        if profile is None or profile.game_count == 0:
            return ""

        hints = []

        # Vote accuracy = 0 → likely werewolf (never votes for wolf)
        if profile.vote_accuracy == 0.0 and profile.n_votes >= 2:
            hints.append(
                f"P{opp_id} has NEVER voted for a werewolf in {profile.n_votes} votes. "
                "High probability of being a werewolf."
            )

        # High h_speak + high h_vote → unfocused, possibly confused villager
        if profile.h_speak > 1.5 and profile.h_vote > 1.0:
            hints.append(
                f"P{opp_id} shows unfocused behavior (h_speak={profile.h_speak:.1f}, "
                "h_vote={profile.h_vote:.1f}) — likely a villager with no special information."
            )

        # Low c_accuse_vote → hiding something
        if profile.c_accuse_vote < 0.5 and profile.total_rounds >= 2:
            hints.append(
                f"P{opp_id} is inconsistent between speech and vote "
                f"(C={profile.c_accuse_vote:.2f}) — may be hiding their true intent."
            )

        if not hints:
            return f"P{opp_id}: insufficient behavioral data for exploitation hints."

        return " ".join(hints)

    def reset(self, opp_id: int) -> None:
        """Clear accumulated data for this opponent."""
        self._profiles.pop(opp_id, None)
        self.memory.set(self._profile_key(opp_id), None)

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self, opp_id: int, profile: WerewolfProfile) -> None:
        self.memory.set(self._profile_key(opp_id), {
            "opp_id": profile.opp_id,
            "vote_targets": profile.vote_targets,
            "mention_targets": profile.mention_targets,
            "mentions_on_wolf": profile.mentions_on_wolf,
            "votes_on_wolf": profile.votes_on_wolf,
            "matched_rounds": profile.matched_rounds,
            "total_rounds": profile.total_rounds,
            "voted_with_majority": profile.voted_with_majority,
            "votes_cast": profile.votes_cast,
            "speak_count": profile.speak_count,
            "total_words": profile.total_words,
            "game_count": profile.game_count,
        })

    def _load(self, opp_id: int) -> WerewolfProfile | None:
        if opp_id in self._profiles:
            return self._profiles[opp_id]
        data = self.memory.get(self._profile_key(opp_id))
        if data is None:
            return None
        profile = WerewolfProfile(opp_id=data["opp_id"])
        profile.vote_targets = data["vote_targets"]
        profile.mention_targets = data["mention_targets"]
        profile.mentions_on_wolf = data["mentions_on_wolf"]
        profile.votes_on_wolf = data["votes_on_wolf"]
        profile.matched_rounds = data["matched_rounds"]
        profile.total_rounds = data["total_rounds"]
        profile.voted_with_majority = data["voted_with_majority"]
        profile.votes_cast = data["votes_cast"]
        profile.speak_count = data["speak_count"]
        profile.total_words = data["total_words"]
        profile.game_count = data["game_count"]
        self._profiles[opp_id] = profile
        return profile

    # ── Offline analysis ──────────────────────────────────────────────

    def load_from_json(self, path: str, opp_id: int | None = None) -> None:
        """Replay trajectories from a pilot summary or trajectory file to build
        profiles offline.

        If opp_id is given, only track that opponent. Otherwise track all.
        """
        with open(path) as f:
            data = json.load(f)

        # Support both RunLog format and trajectory format
        trajectories: list[Trajectory] = []
        if "games" in data:
            # RunLog format
            for g in data["games"]:
                traj = Trajectory.model_validate({
                    "game_name": "Werewolf",
                    "steps": g["steps"],
                    "payout": g["payout"],
                    "winner": g["winner"],
                    "static_state": g.get("static_state", {}),
                })
                trajectories.append(traj)
        elif "steps" in data:
            # Single Trajectory format
            trajectories.append(Trajectory.model_validate(data))
        elif isinstance(data, list):
            for item in data:
                if "steps" in item:
                    traj = Trajectory.model_validate({
                        "game_name": "Werewolf",
                        "steps": item["steps"],
                        "payout": item.get("payout", {}),
                        "winner": item.get("winner"),
                        "static_state": item.get("static_state", {}),
                    })
                    trajectories.append(traj)

        for traj in trajectories:
            roles = traj.static_state.get("roles", {})
            if not roles:
                continue

            from games.werewolf import ROLE_SPECS
            feats = extract_player_features(traj, roles, ROLE_SPECS)

            for pid in feats:
                if opp_id is not None and pid != opp_id:
                    continue
                profile = self._profiles.get(pid)
                if profile is None:
                    profile = WerewolfProfile(opp_id=pid)
                    self._profiles[pid] = profile
                profile.ingest(feats[pid])
                self._save(pid, profile)

    def get_profile(self, opp_id: int) -> WerewolfProfile | None:
        """Return the cumulative profile for an opponent, or None."""
        return self._load(opp_id)

    def all_profiles(self) -> dict[int, WerewolfProfile]:
        """Return all tracked profiles."""
        return dict(self._profiles)
