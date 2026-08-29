"""Werewolf (狼人杀) game implementation.

Multi-player hidden-role game with night/day phases.
Roles: werewolf, villager, seer, witch, hunter.

Night: werewolves kill → seer checks → witch saves/poisons → resolve deaths
Day: all alive players vote to eliminate → resolve vote → check win
"""

from __future__ import annotations

import random
from typing import Any

from base.interfaces import BaseGame
from base.types import Action, Observation

# --- Role definitions ---

ROLE_SPECS = {
    "werewolf": {"side": "evil", "night_action": "kill", "team_knows": True},
    "villager": {"side": "good", "night_action": None, "team_knows": False},
    "seer": {"side": "good", "night_action": "check", "team_knows": False},
    "witch": {"side": "good", "night_action": "save_and_poison", "team_knows": False},
    "hunter": {"side": "good", "night_action": None, "team_knows": False},
}

DEFAULT_ROLES = {"werewolf": 2, "villager": 2, "seer": 1}


# --- State ---

class WerewolfState:
    def __init__(
        self,
        *,
        roles: dict[int, str],
        num_players: int,
        alive: list[int],
        phase: str,
        acting_player: int,
        round: int,
        night_kill_target: int | None,
        night_seer_target: int | None,
        seer_result: str | None,
        seer_checked: list[int],
        seer_results: dict[int, str],  # permanent record: player_id → role
        witch_save_used: bool,
        witch_poison_used: bool,
        witch_saved: bool,
        witch_poison_target: int | None,
        night_deaths: list[int],
        hunter_pending: int | None,
        hunter_shot_target: int | None,
        hunter_shot_context: str | None,
        day_votes: dict[int, int],
        day_vote_results: list[dict[str, Any]],
        day_discussion_messages: list[dict[str, Any]],
        day_discussion_spoken: list[int],
        last_words_pending: int | None,
        last_words_messages: list[dict[str, Any]],
        eliminated: list[dict[str, Any]],
        events: list[str],
        last_phase: str | None,
        rewards: dict[int, float] | None,
        discussion_start: int = 0,
        clockwise: bool = True,
    ):
        self.roles = roles
        self.num_players = num_players
        self.alive = alive
        self.phase = phase
        self.acting_player = acting_player
        self.round = round
        self.discussion_start = discussion_start
        self.clockwise = clockwise
        self.night_kill_target = night_kill_target
        self.night_seer_target = night_seer_target
        self.seer_result = seer_result
        self.seer_checked = seer_checked
        self.seer_results = seer_results
        self.witch_save_used = witch_save_used
        self.witch_poison_used = witch_poison_used
        self.witch_saved = witch_saved
        self.witch_poison_target = witch_poison_target
        self.night_deaths = night_deaths
        self.hunter_pending = hunter_pending
        self.hunter_shot_target = hunter_shot_target
        self.hunter_shot_context = hunter_shot_context
        self.day_votes = day_votes
        self.day_vote_results = day_vote_results
        self.day_discussion_messages = day_discussion_messages
        self.day_discussion_spoken = day_discussion_spoken
        self.last_words_pending = last_words_pending
        self.last_words_messages = last_words_messages
        self.eliminated = eliminated
        self.events = events
        self.last_phase = last_phase
        self.rewards = rewards

    def clone(self) -> WerewolfState:
        return WerewolfState(
            roles=dict(self.roles),
            num_players=self.num_players,
            alive=list(self.alive),
            phase=self.phase,
            acting_player=self.acting_player,
            round=self.round,
            night_kill_target=self.night_kill_target,
            night_seer_target=self.night_seer_target,
            seer_result=self.seer_result,
            seer_checked=list(self.seer_checked),
            seer_results=dict(self.seer_results),
            witch_save_used=self.witch_save_used,
            witch_poison_used=self.witch_poison_used,
            witch_saved=self.witch_saved,
            witch_poison_target=self.witch_poison_target,
            night_deaths=list(self.night_deaths),
            hunter_pending=self.hunter_pending,
            hunter_shot_target=self.hunter_shot_target,
            hunter_shot_context=self.hunter_shot_context,
            day_votes=dict(self.day_votes),
            day_vote_results=list(self.day_vote_results),
            day_discussion_messages=list(self.day_discussion_messages),
            day_discussion_spoken=list(self.day_discussion_spoken),
            last_words_pending=self.last_words_pending,
            last_words_messages=list(self.last_words_messages),
            eliminated=list(self.eliminated),
            events=list(self.events),
            last_phase=self.last_phase,
            rewards=dict(self.rewards) if self.rewards else None,
            discussion_start=self.discussion_start,
            clockwise=self.clockwise,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": dict(self.roles),
            "num_players": self.num_players,
            "alive": list(self.alive),
            "phase": self.phase,
            "acting_player": self.acting_player,
            "round": self.round,
            "night_kill_target": self.night_kill_target,
            "night_seer_target": self.night_seer_target,
            "seer_result": self.seer_result,
            "seer_checked": list(self.seer_checked),
            "seer_results": dict(self.seer_results),
            "witch_save_used": self.witch_save_used,
            "witch_poison_used": self.witch_poison_used,
            "witch_saved": self.witch_saved,
            "witch_poison_target": self.witch_poison_target,
            "night_deaths": list(self.night_deaths),
            "hunter_pending": self.hunter_pending,
            "hunter_shot_target": self.hunter_shot_target,
            "hunter_shot_context": self.hunter_shot_context,
            "day_votes": dict(self.day_votes),
            "day_vote_results": list(self.day_vote_results),
            "day_discussion_messages": list(self.day_discussion_messages),
            "day_discussion_spoken": list(self.day_discussion_spoken),
            "last_words_pending": self.last_words_pending,
            "last_words_messages": list(self.last_words_messages),
            "eliminated": list(self.eliminated),
            "events": list(self.events),
            "last_phase": self.last_phase,
            "rewards": dict(self.rewards) if self.rewards else None,
            "discussion_start": self.discussion_start,
            "clockwise": self.clockwise,
        }


# --- Game ---

class Werewolf(BaseGame[WerewolfState]):
    """Configurable Werewolf game.

    Args:
        seed: Random seed for role assignment.
        roles: Dict mapping role name to count, e.g. {"werewolf": 2, "villager": 2, "seer": 1}.
    """

    def __init__(
        self,
        seed: int | None = None,
        roles: dict[str, int] | None = None,
    ):
        self.rng = random.Random(seed)
        self._role_config = dict(roles) if roles else dict(DEFAULT_ROLES)

        if "werewolf" not in self._role_config or self._role_config["werewolf"] < 1:
            raise ValueError("Must have at least 1 werewolf")

        total = sum(self._role_config.values())
        if total < 4:
            raise ValueError(f"Need at least 4 players, got {total}")

        for role_name in self._role_config:
            if role_name not in ROLE_SPECS:
                raise ValueError(f"Unknown role: {role_name}")

        self._players = total

    def num_players(self) -> int:
        return self._players

    def get_static_state(self, state: WerewolfState) -> dict[str, Any]:
        return {"roles": dict(state.roles), "num_players": state.num_players}

    def get_rules_text(self) -> str:
        return self._build_rules_text()

    def _build_rules_text(self) -> str:
        role_descriptions = {
            "werewolf": "- Werewolf: Evil. Kill one player each night. Knows other werewolves.",
            "villager": "- Villager: Good. No special power. Votes during day.",
            "seer": "- Seer: Good. Checks one player's role each night.",
            "witch": "- Witch: Good. Has one save potion and one poison potion (one use each).",
            "hunter": "- Hunter: Good. When eliminated, can shoot one player.",
        }
        enabled_roles = [r for r in role_descriptions if r in self._role_config]

        phases = ["1. Night: Werewolves choose kill target."]
        if "seer" in self._role_config:
            phases.append("2. Night: Seer investigates one player.")
        if "witch" in self._role_config:
            phases.append("3. Night: Witch may save the kill target and/or poison someone.")
        phases.extend([
            "4. Day: Deaths announced. All alive players discuss and share information.",
            "5. Day: All alive players vote to eliminate one player.",
        ])

        lines = [
            "Werewolf Rules:",
            f"- {self._players} players. Roles: " +
            ", ".join(f"{r}×{c}" for r, c in self._role_config.items()) + ".",
            "",
            "Roles:",
        ] + [role_descriptions[r] for r in enabled_roles] + [
            "",
            "Phases per round:",
        ] + phases + [
            "",
            "Win conditions:",
            "- All werewolves dead → Good team wins.",
            "- Werewolves >= good players alive → Evil team wins.",
            "- Day vote ties → no elimination.",
        ]
        return "\n".join(lines)

    def _build_full_history_text(self, state: WerewolfState) -> str:
        """Build a structured timeline of all game events, round by round."""
        sections = []
        for r in range(1, state.round + 1):
            # ── Night ──
            sections.append(f"======Night {r}======")
            night_killed = [e for e in state.eliminated
                           if e["round"] == r and e["phase"] == "night"]
            if night_killed:
                for e in night_killed:
                    sections.append(
                        f"P{e['player']} ({state.roles[e['player']]}) was killed by werewolves")
            else:
                sections.append("Nobody died tonight")

            # ── Day Discussion ──
            sections.append(f"======Day {r} Discussion======")
            for m in state.day_discussion_messages:
                if m.get("round") != r or m.get("player", -1) < 0:
                    continue
                msg_text = str(m.get("message", ""))
                sections.append(f"P{m['player']}: \"{msg_text}\"")

            # ── Vote Result ──
            for vr in state.day_vote_results:
                if vr["round"] != r:
                    continue
                sections.append(f"======Day {r} Vote Result======")
                votes_str = ", ".join(
                    f"P{v}→P{t}" for v, t in sorted(vr["votes"].items()))
                tally_str = ", ".join(
                    f"P{t}:{c}" for t, c in sorted(vr["tally"].items()))
                if vr["eliminated"] is not None:
                    sections.append(
                        f"Votes: {votes_str} | Tally: {tally_str} → "
                        f"P{vr['eliminated']} ({state.roles[vr['eliminated']]}) eliminated")
                else:
                    sections.append(
                        f"Votes: {votes_str} | Tally: {tally_str} → tie, no elimination")

            # ── Last Words ──
            for lw in state.last_words_messages:
                if lw.get("round") != r:
                    continue
                msg = lw.get("message", "")
                if msg:
                    sections.append(f"======Day {r} Last Words======")
                    sections.append(f"P{lw['player']}: \"{msg}\"")

        return "\n".join(sections)

    def state_to_text(self, state: WerewolfState, player_id: int) -> str:
        role = state.roles[player_id]
        side = ROLE_SPECS[role]["side"]
        lines = [
            f"Round {state.round} — {self._phase_label(state.phase)}",
            f"You are P{player_id}. Your role: {role} ({side})",
        ]

        if role == "werewolf":
            other_wolves = [p for p in state.alive
                            if state.roles[p] == "werewolf" and p != player_id]
            if other_wolves:
                lines.append(f"Fellow werewolves: {other_wolves}")
            if state.night_kill_target is not None:
                lines.append(f"Current kill target: P{state.night_kill_target}")

        if role == "seer" and state.seer_results:
            results_str = ", ".join(
                f"P{p} → {r}" for p, r in sorted(state.seer_results.items())
            )
            lines.append(f"Investigation results: {results_str}")

        if role == "witch":
            lines.append(
                "Save potion: " +
                ("ALREADY USED" if state.witch_save_used else "available"))
            lines.append(
                "Poison potion: " +
                ("ALREADY USED" if state.witch_poison_used else "available"))

        if state.phase in ("night_witch_save", "night_witch_poison") and role == "witch":
            if state.night_kill_target is not None:
                lines.append(f"Tonight's kill target: P{state.night_kill_target}")

        alive_str = ", ".join(
            f"P{p}(you)" if p == player_id else f"P{p}"
            for p in state.alive
        )
        lines.append(f"Alive: [{alive_str}]")

        # Show valid targets for target-picking phases
        target_phases = {
            "night_werewolf": [t for t in state.alive if state.roles[t] != "werewolf"],
            "night_seer": [t for t in state.alive if t != player_id and t not in state.seer_checked],
            "night_witch_poison": [t for t in state.alive if t != player_id],
            "hunter_shot": [t for t in state.alive if t != player_id],
            "day_vote": [t for t in state.alive if t != player_id],
        }
        if state.phase in target_phases:
            valid = target_phases[state.phase]
            if valid:
                lines.append("Valid targets: " + ", ".join(f"P{t}" for t in valid))

        # Last night's deaths
        if state.night_deaths:
            dead_names = ", ".join(f"P{p}" for p in state.night_deaths)
            lines.append(f"Died last night: {dead_names}")
        elif state.eliminated:
            lines.append("Nobody died last night")

        # Full elimination history
        if state.eliminated:
            eliminated_str = ", ".join(
                f"P{e['player']}({state.roles[e['player']]}, {e['phase']} round {e['round']})"
                for e in state.eliminated
            )
            lines.append(f"Eliminated so far: {eliminated_str}")

        if state.phase == "last_words" and state.last_words_pending is not None:
            lines.append(
                f"P{state.last_words_pending} was eliminated and may say last words."
            )

        if state.phase == "day_discussion":
            waiting = [p for p in self._speaking_order(state)
                       if p not in state.day_discussion_spoken]
            if waiting:
                lines.append(f"\nWaiting to speak (in order): {waiting}")

        if state.day_vote_results:
            latest_vote = state.day_vote_results[-1]
            tally = ", ".join(
                f"P{p}:{c}" for p, c in sorted(latest_vote.get("tally", {}).items())
            )
            lines.append("")
            lines.append("Public vote results:")
            lines.append(f"  Tally: {tally}")
            eliminated = latest_vote.get("eliminated")
            if eliminated is None:
                lines.append("  Result: tie, no elimination")
            else:
                lines.append(f"  Result: P{eliminated} eliminated")

        if state.last_words_messages:
            lines.append("")
            lines.append("Last words:")
            for msg in state.last_words_messages:
                text = msg.get("message", "")
                if text:
                    lines.append(f"  P{msg['player']}: {text}")

        if state.phase == "day_vote":
            unvoted = [p for p in state.alive if p not in state.day_votes]
            if unvoted:
                lines.append(f"Still waiting for votes from: {unvoted}")

        return "\n".join(lines)

    @staticmethod
    def _role_guidance(state, player_id: int, role: str) -> str:
        """Role-specific playbook: practical scenarios, not abstract tradeoffs."""
        if role == "werewolf":
            return (
                "You and your fellow werewolves know each other's identities. The "
                "villagers know nothing. Your win condition is simple: survive long "
                "enough that werewolves equal or outnumber the good team.\n\n"
                "The most powerful tool the good team has is the seer, who can verify "
                "one player's identity each night. If the seer checks a werewolf and "
                "reveals that information before a day vote, that werewolf is almost "
                "certainly dead. Your primary threat is an unchecked seer. Your primary "
                "opportunity is that villagers cannot tell the difference between a real "
                "seer and a fake one.\n\n"
                "Blending in means acting like a villager: expressing uncertainty, asking "
                "questions, voting with the majority. This is safe and often works if the "
                "seer never checks you. But a passive wolf contributes nothing to misdirect "
                "the village, and if the seer does check you, you have no defense.\n\n"
                "Fake-claiming seer is the most powerful wolf strategy. Pick a target "
                "(not your teammate), claim you investigated them, and announce the result. "
                "If you claim a villager is a wolf, you can get them eliminated. If you "
                "claim another wolf is a villager, you build their cover. Either way, you "
                "create confusion about who the real seer is. The risk: the real seer will "
                "know you are lying and may counter-claim, forcing the village to choose "
                "between you. If you lose that confrontation, you die and waste a wolf.\n\n"
                "If the real seer accuses you or your teammate, counter-accuse immediately. "
                "Claim THEY are the fake seer, or claim they are a wolf pretending. Give a "
                "specific reason: \"If they were the real seer, why didn't they reveal "
                "earlier?\" or \"They waited until now because they're a wolf setting up a "
                "frame.\" A vague defense looks guilty; a specific counter-accusation looks "
                "legitimate.\n\n"
                "Coordinate with your wolf teammate in discussion. Do NOT openly defend "
                "each other — that makes the link obvious. Instead, one wolf should push "
                "suspicion toward a villager while the other plays neutral. If your teammate "
                "is under heavy suspicion, you may need to vote against them to maintain "
                "your cover. Sacrificing one wolf to protect the other is a valid trade: "
                "with one wolf hidden deep, the game remains winnable.\n\n"
                "Your night kill choice is a strategic signal. Killing the most talkative "
                "player leaves quieter ones who are easier to manipulate. Killing the seer "
                "(if you know who they are) removes the village's only source of truth. "
                "But do not kill the person you accused in discussion — that makes you "
                "obviously guilty."
            )
        if role == "villager":
            return (
                "You have no special information. You do not know who the wolves are, "
                "who the seer checked, or what the witch saw. Your only weapon is your "
                "vote and your reasoning. A village that votes randomly will lose, because "
                "the wolves, who know each other, will coordinate their votes while the "
                "good team splits its numbers.\n\n"
                "Your job is not to find the wolves single-handedly. It is to push for clarity so "
                "that inconsistencies become visible. Pay attention to who suspects whom and "
                "whether their reasoning holds up. Track voting patterns — if "
                "two players consistently vote together, one may be a wolf protecting the "
                "other.\n\n"
                "Be skeptical of early role claims. A werewolf has every incentive to "
                "fake-claim seer or witch to gain trust. A real seer usually stays hidden "
                "until they have useful information. A real witch usually stays quiet to "
                "avoid becoming a target. If someone claims a powerful role on day 1 with "
                "no pressure, question them: why reveal now? What do you gain?\n\n"
                "If two players claim the same role, one of them is lying — and since "
                "wolves are the only ones who need to lie about their role, one is almost "
                "certainly a wolf. Push both to give specific details. The fake one will "
                "struggle to provide consistent answers.\n\n"
                "If a claimed seer survives multiple nights without being killed by wolves, "
                "they are almost certainly a wolf. Real seers are the wolves' top priority "
                "target. A \"seer\" who is still alive on day 3 is suspicious.\n\n"
                "When you vote, decide strategically what to share and what to hold back — "
                "your words shape the village's judgment as much as your vote."
            )
        if role == "seer":
            return (
                "You can check one player's true identity each night. This information "
                "is the village's most valuable asset — and it dies with you if you "
                "stay silent too long.\n\n"
                "The core dilemma: reveal your findings to guide the vote, or stay "
                "hidden to gather more information? The answer depends on what you found. "
                "If you checked a werewolf, revealing before the day vote can save the "
                "village — one guaranteed wolf elimination is worth exposing yourself. "
                "If you have only checked villagers so far, stay hidden and check again "
                "tonight. An early reveal with no wolf found wastes your life for nothing.\n\n"
                "If you checked a player who later died, that information is safe to "
                "share — it confirms their role without exposing you to any additional "
                "risk, since they are already dead. The village benefits from knowing "
                "\"P3 was confirmed a villager\" without you needing to reveal yourself "
                "as the source.\n\n"
                "If someone else claims to be the seer and you know they are lying (because "
                "you are the real seer, or you checked them and they are not the seer), "
                "counter-claim immediately. A fake seer left unchallenged will mislead "
                "the village for the entire game. State your role, reveal your check "
                "results with the specific player name, and explain why the other claim "
                "is false. The village WILL be confused by two seer claims — give them "
                "enough concrete information to identify the real one.\n\n"
                "Once you reveal, expect to die that night. The witch may save you once, "
                "but not twice. Use your remaining time in discussion to decide what "
                "information matters most to pass on. Your final contribution is leaving "
                "the village with a clear information trail.\n\n"
                "Choose investigation targets strategically. Checking the same player "
                "twice wastes a night. Check players whose behavior seems inconsistent — "
                "if they are a wolf, you have a kill. If they are a villager, you have "
                "a trusted ally who can carry your information after you die."
            )
        if role == "witch":
            return (
                "You hold two one-use potions. The SAVE potion prevents the night kill "
                "target from dying. The POISON potion kills one player of your choice. "
                "Both are single-use: once spent, they are gone forever. Each night, you "
                "see which player the werewolves chose to kill, though you do not know "
                "which specific player is a werewolf.\n\n"
                "The save potion is most valuable when used on the seer or on yourself. "
                "If the seer has revealed themselves, saving them guarantees one more "
                "night of investigations — that can be the difference between winning "
                "and losing. If you are targeted, saving yourself keeps both potions in "
                "play. Saving a random villager is almost always a waste: you spend your "
                "most powerful tool for minimal information gain.\n\n"
                "The poison potion is dangerous. If you poison a villager, you have "
                "effectively done the wolves' work for them and wasted your only offensive "
                "tool. Only use poison when you are confident — someone who survived "
                "multiple rounds while claiming a powerful role, or someone who was "
                "accused by a now-dead confirmed seer. When in doubt, hold the poison.\n\n"
                "Do not reveal your role early. A known witch becomes the wolves' "
                "second-favorite target after the seer. Vote with your best judgment and "
                "save your potions for critical moments. "
                "The exception: if the seer is dead and you have accumulated information "
                "from watching the night kills, you may become the village's most informed "
                "player. At that point, revealing what you know can be worth the risk.\n\n"
                "If the wolves' kill target DID die (you chose not to save), you know "
                "that player was NOT a werewolf. This is valuable deductive information "
                "that only you possess. Use it to evaluate who might be lying about "
                "their suspicions."
            )
        if role == "hunter":
            return (
                "When you die — by night kill or day vote — you may shoot one player "
                "before leaving the game. This ability fires exactly once.\n\n"
                "If you reveal your role while alive: wolves will avoid killing you at "
                "night because your death lets you shoot. This gives you survival power "
                "but also means you will only use your ability if voted out — and if the "
                "village wastes a day voting for you, the wolves gain a free round. "
                "If you stay hidden: wolves may kill you by accident, and you will get "
                "your revenge shot. But a hidden hunter who has not been paying close "
                "attention will shoot blind.\n\n"
                "When you shoot, you have one chance. Shooting the wrong person kills "
                "a teammate and wastes your only contribution. Not shooting at all also "
                "wastes your contribution. Watch carefully throughout the game so that "
                "when your moment comes, you have a target in mind. If you are voted out "
                "by a unanimous vote, the person who led the accusation against you is "
                "worth examining — a wolf may have orchestrated your elimination."
            )
        return ""


    @staticmethod
    def _phase_label(phase: str) -> str:
        labels = {
            "night_werewolf": "Night — Werewolves choose target",
            "night_seer": "Night — Seer investigates",
            "night_witch_save": "Night — Witch: use SAVE potion or pass?",
            "night_witch_poison": "Night — Witch: use POISON potion or pass?",
            "hunter_shot": "Hunter's revenge shot",
            "last_words": "Day — Last words",
            "day_discussion": "Day — Discussion",
            "day_vote": "Day — Vote to eliminate",
            "terminal": "Game over",
        }
        return labels.get(phase, phase)

    # --- Reset ---

    def reset(self) -> WerewolfState:
        all_roles: list[str] = []
        for role_name, count in self._role_config.items():
            all_roles.extend([role_name] * count)
        self.rng.shuffle(all_roles)
        roles = {i: all_roles[i] for i in range(len(all_roles))}

        state = WerewolfState(
            roles=roles,
            num_players=len(all_roles),
            alive=sorted(roles.keys()),
            phase="night_werewolf",
            acting_player=0,
            round=1,
            discussion_start=self.rng.randint(0, self._players - 1),
            night_kill_target=None,
            night_seer_target=None,
            seer_result=None,
            seer_checked=[],
            seer_results={},
            witch_save_used=False,
            witch_poison_used=False,
            witch_saved=False,
            witch_poison_target=None,
            night_deaths=[],
            hunter_pending=None,
            hunter_shot_target=None,
            hunter_shot_context=None,
            day_votes={},
            day_vote_results=[],
            day_discussion_messages=[],
            day_discussion_spoken=[],
            last_words_pending=None,
            last_words_messages=[],
            eliminated=[],
            events=[],
            last_phase=None,
            rewards=None,
        )
        self._advance_phase(state)
        return state

    # --- Core interface ---

    def get_current_player(self, state: WerewolfState) -> int:
        return state.acting_player

    def is_terminal(self, state: WerewolfState) -> bool:
        return state.phase == "terminal"

    def get_rewards(self, state: WerewolfState) -> dict[int, float]:
        if state.rewards is None:
            return {p: 0.0 for p in state.roles}
        return dict(state.rewards)

    def state_to_dict(self, state: WerewolfState) -> dict[str, Any]:
        return state.to_dict()

    # --- Legal actions ---

    def get_legal_actions(self, state: WerewolfState) -> list[Action]:
        if self.is_terminal(state):
            return []

        p = state.acting_player

        if state.phase == "night_werewolf":
            targets = [t for t in state.alive if state.roles[t] != "werewolf"]
            actions: list[Action] = []
            if targets:
                actions.append(Action(player_id=p, type="kill",
                              data={"target": "<your choice>"}))
            if state.night_kill_target is not None:
                actions.append(Action(player_id=p, type="pass"))
            return actions

        if state.phase == "night_seer":
            targets = [t for t in state.alive
                       if t != p and t not in state.seer_checked]
            if targets:
                return [Action(player_id=p, type="check",
                               data={"target": "<your choice>"})]
            return [Action(player_id=p, type="pass")]

        if state.phase == "night_witch_save":
            actions = [Action(player_id=p, type="pass")]
            if not state.witch_save_used and state.night_kill_target is not None:
                actions.append(Action(player_id=p, type="save",
                    data={"target": state.night_kill_target}))
            return actions

        if state.phase == "night_witch_poison":
            targets = [t for t in state.alive if t != p]
            actions = [Action(player_id=p, type="pass")]
            if not state.witch_poison_used and targets:
                actions.append(Action(player_id=p, type="poison",
                              data={"target": "<your choice>"}))
            return actions

        if state.phase == "hunter_shot":
            targets = [t for t in state.alive if t != p]
            if targets:
                return [Action(player_id=p, type="shoot",
                               data={"target": "<your choice>"})]
            return [Action(player_id=p, type="pass")]

        if state.phase == "day_discussion":
            return [
                Action(player_id=p, type="speak",
                       data={"message": "<your message>"}),
                Action(player_id=p, type="pass"),
            ]

        if state.phase == "last_words":
            return [
                Action(player_id=p, type="speak",
                       data={"message": "<your last words>"}),
                Action(player_id=p, type="pass"),
            ]

        if state.phase == "day_vote":
            targets = [t for t in state.alive if t != p]
            if targets:
                return [Action(player_id=p, type="vote",
                               data={"target": "<your choice>"})]
            return []

        return []

    # --- Step ---

    def step(self, state: WerewolfState, action: Action) -> WerewolfState:
        legal_actions = self.get_legal_actions(state)
        legal_types = {a.type for a in legal_actions}

        if action.type not in legal_types:
            raise ValueError(
                f"Illegal action '{action.type}' for P{action.player_id}. "
                f"Legal: {list(legal_types)}"
            )
        if action.player_id != state.acting_player:
            raise ValueError(
                f"Wrong player P{action.player_id}. Current: P{state.acting_player}"
            )

        state = state.clone()
        state.last_phase = state.phase

        if state.phase == "night_werewolf":
            if action.type == "kill":
                target = self._normalize_target(action.data.get("target"))
                valid_non_wolf = [t for t in state.alive if state.roles[t] != "werewolf"]
                if target is None or target not in state.alive:
                    detail = f" P{target} is DEAD." if isinstance(target, int) and target not in state.alive else ""
                    raise ValueError(
                        f"Invalid kill target: {target}.{detail}"
                        f" Valid targets (alive, non-werewolf): {valid_non_wolf}")
                if state.roles[target] == "werewolf":
                    raise ValueError(
                        f"Cannot kill fellow werewolf P{target}")
                state.night_kill_target = target
                state.events.append(
                    f"🐺 P{action.player_id}({state.roles[action.player_id]}) chose to kill P{target}")

        elif state.phase == "night_seer":
            if action.type == "check":
                target = self._normalize_target(action.data.get("target"))
                valid_check = [t for t in state.alive if t != action.player_id and t not in state.seer_checked]
                if target is None or target not in state.alive:
                    detail = f" P{target} is DEAD." if isinstance(target, int) and target not in state.alive else ""
                    raise ValueError(
                        f"Invalid check target: {target}.{detail}"
                        f" Valid targets (alive, not yet checked): {valid_check}")
                if target == action.player_id:
                    raise ValueError("Cannot check yourself")
                if target in state.seer_checked:
                    raise ValueError(f"Already checked P{target}")
                state.night_seer_target = target
                state.seer_result = state.roles[target]
                state.seer_checked.append(target)
                state.seer_results[target] = state.roles[target]
                state.events.append(
                    f"🔮 P{action.player_id}({state.roles[action.player_id]}) checked P{target} → {state.roles[target]}")

        elif state.phase == "night_witch_save":
            if action.type == "save":
                if state.witch_save_used:
                    raise ValueError("Save potion already used")
                state.witch_save_used = True
                state.witch_saved = True
                state.events.append(f"🧙 P{action.player_id}({state.roles[action.player_id]}) used SAVE potion")

        elif state.phase == "night_witch_poison":
            if action.type == "poison":
                if state.witch_poison_used:
                    raise ValueError("Poison potion already used")
                target = self._normalize_target(action.data.get("target"))
                valid_poison = [t for t in state.alive if t != action.player_id]
                if target is None or target not in state.alive:
                    detail = f" P{target} is DEAD." if isinstance(target, int) and target not in state.alive else ""
                    raise ValueError(
                        f"Invalid poison target: {target}.{detail}"
                        f" Valid targets (alive): {valid_poison}")
                if target == action.player_id:
                    raise ValueError("Cannot poison yourself")
                state.witch_poison_used = True
                state.witch_poison_target = target
                state.events.append(
                    f"🧙 P{action.player_id}({state.roles[action.player_id]}) used POISON on P{target}")

        elif state.phase == "hunter_shot":
            if action.type == "shoot":
                target = self._normalize_target(action.data.get("target"))
                valid_shoot = [t for t in state.alive if t != action.player_id]
                if target is None or target not in state.alive:
                    detail = f" P{target} is DEAD." if isinstance(target, int) and target not in state.alive else ""
                    raise ValueError(
                        f"Invalid shoot target: {target}.{detail}"
                        f" Valid targets (alive): {valid_shoot}")
                if target == action.player_id:
                    raise ValueError("Cannot shoot yourself")
                state.hunter_shot_target = target
                state.events.append(
                    f"🏹 P{action.player_id}({state.roles[action.player_id]}) shoots P{target}")

        elif state.phase == "day_discussion":
            state.day_discussion_spoken.append(action.player_id)
            if action.type == "speak":
                msg = action.data.get("message", "")
                state.day_discussion_messages.append({
                    "player": action.player_id, "message": str(msg),
                    "round": state.round,
                })
                if msg:
                    state.events.append(
                        f'💬 P{action.player_id}: "{msg}"')
                else:
                    state.events.append(
                        f"💬 P{action.player_id}: (said nothing)")

        elif state.phase == "last_words":
            if action.type == "speak":
                msg = action.data.get("message", "")
                state.last_words_messages.append({
                    "player": action.player_id, "message": str(msg),
                    "round": state.round,
                })
                if msg:
                    state.events.append(
                        f'💬 P{action.player_id} last words: "{msg}"')
                else:
                    state.events.append(
                        f"💬 P{action.player_id} gave no last words")
            else:
                state.last_words_messages.append({
                    "player": action.player_id, "message": "",
                    "round": state.round,
                })
                state.events.append(
                    f"💬 P{action.player_id} gave no last words")

        elif state.phase == "day_vote":
            if action.type == "vote":
                target = self._normalize_target(action.data.get("target"))
                if target is None or target not in state.alive:
                    valid = [t for t in state.alive if t != action.player_id]
                    detail = ""
                    if isinstance(target, int) and target not in state.alive:
                        detail = f" P{target} is DEAD and cannot be voted for."
                    elif target == action.player_id:
                        detail = " You cannot vote for yourself."
                    raise ValueError(
                        f"Invalid vote target: {target}.{detail}"
                        f" Alive players you can vote for: {valid}")
                if target == action.player_id:
                    raise ValueError(
                        f"Cannot vote for yourself (P{action.player_id})")
                state.day_votes[action.player_id] = target
                state.events.append(
                    f"🗳️ P{action.player_id} voted to eliminate P{target}")

        self._advance_phase(state)
        return state

    @staticmethod
    def _normalize_target(target) -> int | None:
        """Accept 'P2', '2', or 2 as a valid target."""
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

    # --- Phase machine ---

    def _advance_phase(self, state: WerewolfState) -> None:
        while state.phase != "terminal":
            if state.phase == "night_werewolf":
                wolves = self._alive_werewolves(state)
                if not wolves:
                    state.phase = "night_seer"
                    continue
                # Find the next werewolf to act
                # If we just entered this phase (last_phase != night_werewolf),
                # start from the first wolf. Otherwise find the next wolf.
                if state.last_phase != "night_werewolf":
                    # Fresh entry to werewolf phase
                    state.acting_player = wolves[0]
                    return
                # A wolf just acted — find the next one
                current_idx = wolves.index(state.acting_player) if state.acting_player in wolves else -1
                if current_idx >= 0 and current_idx + 1 < len(wolves):
                    state.acting_player = wolves[current_idx + 1]
                    return
                # All wolves acted — fallback: ensure kill target set
                if state.night_kill_target is None:
                    # Default: kill first non-werewolf alive
                    non_wolves = [p for p in state.alive if state.roles[p] != "werewolf"]
                    if non_wolves:
                        state.night_kill_target = non_wolves[0]
                state.phase = "night_seer"
                continue

            elif state.phase == "night_seer":
                # Only enter seer phase once
                if state.last_phase != "night_seer":
                    seer = self._get_alive_role(state, "seer")
                    if seer is not None:
                        state.acting_player = seer
                        return
                state.phase = "night_witch_save"
                continue

            elif state.phase == "night_witch_save":
                if state.last_phase != "night_witch_save":
                    witch = self._get_alive_role(state, "witch")
                    if witch is not None and not state.witch_save_used:
                        state.acting_player = witch
                        return
                state.phase = "night_witch_poison"
                continue

            elif state.phase == "night_witch_poison":
                if state.last_phase != "night_witch_poison":
                    witch = self._get_alive_role(state, "witch")
                    if witch is not None and not state.witch_poison_used:
                        state.acting_player = witch
                        return
                self._resolve_night(state)
                if state.phase == "terminal":
                    return
                if state.hunter_pending is not None:
                    state.phase = "hunter_shot"
                    state.acting_player = state.hunter_pending
                    state.hunter_shot_context = "night"
                    return
                state.phase = "day_discussion"
                continue

            elif state.phase == "hunter_shot":
                target = state.hunter_shot_target
                if target is not None and target in state.alive:
                    state.alive.remove(target)
                    state.eliminated.append({
                        "player": target, "phase": "hunter_shot",
                        "round": state.round,
                        "shot_by": state.hunter_pending,
                    })
                state.hunter_pending = None
                state.hunter_shot_target = None
                winner = self._check_winner(state)
                if winner:
                    state.rewards = self._compute_rewards(state, winner)
                    state.phase = "terminal"
                    state.acting_player = -1
                    return
                ctx = state.hunter_shot_context
                state.hunter_shot_context = None
                if ctx == "night":
                    state.phase = "day_discussion"
                    continue
                elif ctx == "day":
                    self._start_new_round(state)
                    continue
                state.phase = "day_discussion"
                state.events.append(f"💬 Round {state.round} — Discussion begins")
                continue

            elif state.phase == "last_words":
                eliminated = state.last_words_pending
                state.last_words_pending = None

                if eliminated is not None and state.roles[eliminated] == "hunter":
                    state.hunter_pending = eliminated
                    state.phase = "hunter_shot"
                    state.acting_player = eliminated
                    state.hunter_shot_context = "day"
                    return

                winner = self._check_winner(state)
                if winner:
                    state.rewards = self._compute_rewards(state, winner)
                    state.phase = "terminal"
                    state.acting_player = -1
                    state.events.append(
                        f"🏆 Game over: {'Evil' if winner == 'evil' else 'Good'} team wins!")
                    return

                self._start_new_round(state)
                continue

            elif state.phase == "day_discussion":
                # Set up discussion order based on eliminations (once per round)
                if state.last_phase != "day_discussion":
                    self._setup_discussion_order(state)
                for p in self._speaking_order(state):
                    if p not in state.day_discussion_spoken:
                        state.acting_player = p
                        return
                # All alive players have spoken → transition to vote
                state.phase = "day_vote"
                state.events.append("🗳️ Discussion over — Time to vote")
                continue

            elif state.phase == "day_vote":
                for p in sorted(state.alive):
                    if p not in state.day_votes:
                        state.acting_player = p
                        return
                self._resolve_day_vote(state)
                if state.phase == "terminal":
                    return
                if state.phase == "last_words":
                    return
                self._start_new_round(state)
                continue

            else:
                raise RuntimeError(f"Unknown phase: {state.phase}")

    # --- Night resolution ---

    def _resolve_night(self, state: WerewolfState) -> None:
        killed: set[int] = set()

        if (state.night_kill_target is not None
                and state.night_kill_target in state.alive
                and not state.witch_saved):
            killed.add(state.night_kill_target)

        if (state.witch_poison_target is not None
                and state.witch_poison_target in state.alive):
            killed.add(state.witch_poison_target)

        state.night_deaths = sorted(killed)

        if killed:
            dead_names = ", ".join(f"P{p}" for p in sorted(killed))
            state.events.append(f"☀️ {dead_names} died last night")
        else:
            state.events.append("☀️ Nobody died tonight")

        # Check for hunter before removing from alive
        hunter_dead = None
        for p in killed:
            if state.roles[p] == "hunter":
                hunter_dead = p
                break

        for p in killed:
            if p in state.alive:
                state.alive.remove(p)
                state.eliminated.append({
                    "player": p, "phase": "night", "round": state.round,
                })

        # Reset night state
        state.witch_saved = False

        if hunter_dead is not None:
            state.hunter_pending = hunter_dead
            return  # Win check after hunter shot

        winner = self._check_winner(state)
        if winner:
            state.rewards = self._compute_rewards(state, winner)
            state.phase = "terminal"
            state.acting_player = -1
            state.events.append(
                f"🏆 Game over: {'Evil' if winner == 'evil' else 'Good'} team wins!")

    # --- Day resolution ---

    def _resolve_day_vote(self, state: WerewolfState) -> None:
        if not state.day_votes:
            return

        tally: dict[int, int] = {}
        for target in state.day_votes.values():
            tally[target] = tally.get(target, 0) + 1

        max_votes = max(tally.values())
        leaders = sorted(
            [p for p, v in tally.items() if v == max_votes],
            key=lambda x: tally[x], reverse=True,
        )

        # Announce tally
        tally_str = ", ".join(
            f"P{p}:{v}" for p, v in sorted(tally.items(), key=lambda x: -x[1]))
        state.events.append(f"🗳️ Vote tally: {tally_str}")

        eliminated = None
        if len(leaders) == 1:
            eliminated = leaders[0]
            state.events.append(
                f"🗳️ P{eliminated} eliminated ({max_votes} votes)")
            if eliminated in state.alive:
                state.alive.remove(eliminated)
                state.eliminated.append({
                    "player": eliminated, "phase": "day_vote",
                    "round": state.round,
                })

        state.day_vote_results.append({
            "round": state.round,
            "votes": dict(state.day_votes),
            "tally": dict(sorted(tally.items())),
            "eliminated": eliminated,
            "tie": eliminated is None,
        })

        if eliminated is not None:
            state.last_words_pending = eliminated
            state.phase = "last_words"
            state.acting_player = eliminated
            return

        winner = self._check_winner(state)
        if winner:
            state.rewards = self._compute_rewards(state, winner)
            state.phase = "terminal"
            state.acting_player = -1
            state.events.append(
                f"🏆 Game over: {'Evil' if winner == 'evil' else 'Good'} team wins!")

    # --- New round ---

    def _start_new_round(self, state: WerewolfState) -> None:
        state.round += 1
        state.night_kill_target = None
        state.night_seer_target = None
        state.seer_result = None
        state.witch_saved = False
        state.witch_poison_target = None
        state.day_votes = {}
        state.day_discussion_spoken = []
        state.last_phase = None
        state.phase = "night_werewolf"
        # NOTE: day_discussion_messages persists across rounds for full history
        # NOTE: discussion_start is set dynamically in _setup_discussion_order
        #       when the day_discussion phase is first entered each round.

    # --- Helpers ---

    def _speaking_order(self, state: WerewolfState) -> list[int]:
        """Return day_discussion order: start from discussion_start, then
        clockwise (ascending) or counter-clockwise (descending) depending
        on state.clockwise."""
        total = state.num_players
        start = state.discussion_start % total
        if state.clockwise:
            order = [(start + i) % total for i in range(total)]
        else:
            order = [(start - i) % total for i in range(total)]
        return [p for p in order if p in state.alive]

    def _setup_discussion_order(self, state: WerewolfState) -> None:
        """Set discussion_start based on the most recent elimination.
        Real 狼人杀 rules: start from the dead player's neighbor,
        alternate direction each round."""
        if not state.eliminated:
            return  # Day 1 — keep the random initial start

        # Find the most recent elimination(s)
        last_round = max(e["round"] for e in state.eliminated)
        recent = [e for e in state.eliminated if e["round"] == last_round]
        # For double deaths, use the smaller-numbered player
        last_dead = sorted(e["player"] for e in recent)[0]

        # Start from the dead player's neighbor
        if state.clockwise:
            state.discussion_start = (last_dead + 1) % state.num_players
        else:
            state.discussion_start = (last_dead - 1 + state.num_players) % state.num_players

        # Toggle direction for next round
        state.clockwise = not state.clockwise

    def _alive_werewolves(self, state: WerewolfState) -> list[int]:
        return sorted(
            p for p in state.alive if state.roles[p] == "werewolf"
        )

    def _get_alive_role(self, state: WerewolfState, role: str) -> int | None:
        for p in sorted(state.alive):
            if state.roles[p] == role:
                return p
        return None

    def _check_winner(self, state: WerewolfState) -> str | None:
        wolves = sum(1 for p in state.alive if state.roles[p] == "werewolf")
        good = len(state.alive) - wolves
        if wolves == 0:
            return "good"
        if wolves >= good:
            return "evil"
        return None

    def _compute_rewards(
        self, state: WerewolfState, winner: str
    ) -> dict[int, float]:
        rewards: dict[int, float] = {}
        for p in state.roles:
            side = ROLE_SPECS[state.roles[p]]["side"]
            if (winner == "good" and side == "good") or \
               (winner == "evil" and side == "evil"):
                rewards[p] = 1.0
            else:
                rewards[p] = -1.0
        return rewards

    # --- Observation ---

    def get_observation(self, state: WerewolfState, player_id: int) -> Observation:
        legal_actions = self.get_legal_actions(state)
        role = state.roles[player_id]

        return Observation(
            player_id=player_id,
            private_info=self._private_info(state, player_id),
            public_info=self._public_info(state, player_id),
            history=self._build_history(state, player_id),
            legal_actions=[a.type for a in legal_actions],
            rules_text=self.get_rules_text(),
            state_text=self.state_to_text(state, player_id),
            full_history_text=self._build_full_history_text(state),
            role_guidance=self._role_guidance(state, player_id, role),
        )

    def _build_history(
        self, state: WerewolfState, player_id: int,
    ) -> list[dict[str, Any]]:
        """Build structured history from events and game state.

        Key events visible to all: night deaths, discussion messages,
        vote results, elimination records. Private killer identity
        excluded for non-wolves.
        """
        history: list[dict[str, Any]] = []
        killed_by_night = set()
        eliminated_by_vote = set()

        for e in state.eliminated:
            player = e["player"]
            if e["phase"] == "night":
                killed_by_night.add(player)
                history.append({
                    "round": e["round"],
                    "phase": "night_result",
                    "action": "death",
                    "player": player,
                    "role_revealed": state.roles[player],
                })
            elif e["phase"] == "day_vote":
                eliminated_by_vote.add(player)
                history.append({
                    "round": e["round"],
                    "phase": "day_result",
                    "action": "eliminated_by_vote",
                    "player": player,
                    "role_revealed": state.roles[player],
                })
            elif e["phase"] == "hunter_shot":
                history.append({
                    "round": e["round"],
                    "phase": "hunter_shot",
                    "action": "shot",
                    "player": player,
                    "shot_by": e.get("shot_by"),
                    "role_revealed": state.roles[player],
                })

        # Discussion messages (visible to all)
        for msg in state.day_discussion_messages:
            history.append({
                "player": msg["player"],
                "round": msg.get("round"),
                "phase": "day_discussion",
                "action": "speak",
                "message": msg["message"],
            })

        # Resolved votes are public; in-progress votes are intentionally hidden
        # so the sequential runner behaves like simultaneous voting.
        for result in state.day_vote_results:
            # Also include a tally summary entry (player=-1 so history_to_text renders cleanly)
            history.append({
                "round": result["round"],
                "player": -1,
                "phase": "day_vote_result",
                "action": "vote_result",
                "tally": dict(result["tally"]),
                "eliminated": result["eliminated"],
                "tie": result["tie"],
            })
            for voter, target in sorted(result["votes"].items()):
                history.append({
                    "round": result["round"],
                    "player": voter,
                    "phase": "day_vote_result",
                    "action": "vote",
                    "target": target,
                })

        for msg in state.last_words_messages:
            history.append({
                "round": msg["round"],
                "player": msg["player"],
                "phase": "last_words",
                "action": "speak",
                "message": msg["message"],
            })

        return history

    def _private_info(self, state: WerewolfState, player_id: int) -> dict[str, Any]:
        role = state.roles[player_id]
        info: dict[str, Any] = {"role": role}

        if role == "werewolf":
            other_wolves = [
                p for p in state.alive
                if state.roles[p] == "werewolf" and p != player_id
            ]
            info["other_werewolves"] = other_wolves
            if state.night_kill_target is not None:
                info["night_kill_target"] = state.night_kill_target

        if role == "seer":
            info["seer_results"] = dict(state.seer_results)

        if role == "witch":
            info["save_available"] = not state.witch_save_used
            info["poison_available"] = not state.witch_poison_used

        return info

    def _public_info(self, state: WerewolfState, player_id: int) -> dict[str, Any]:
        info: dict[str, Any] = {
            "round": state.round,
            "phase": state.phase,
            "alive": list(state.alive),
            "eliminated": list(state.eliminated),
        }

        if state.night_deaths:
            info["night_deaths"] = list(state.night_deaths)

        if state.phase == "day_discussion":
            info["discussion"] = list(state.day_discussion_messages)

        if state.day_vote_results:
            info["day_vote_results"] = list(state.day_vote_results)

        if state.last_words_messages:
            info["last_words"] = list(state.last_words_messages)

        return info
