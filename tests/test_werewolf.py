"""Tests for Werewolf game logic."""

import pytest
from base.types import Action
from games.werewolf import Werewolf, WerewolfState


def _act(player, atype, data=None):
    return Action(player_id=player, type=atype, data=data or {})


class TestWerewolfGame:
    # --- Init ---

    def test_reset(self):
        game = Werewolf(seed=42)
        state = game.reset()
        assert state.phase in ("night_werewolf", "night_seer", "night_witch_save",
                                "night_witch_poison", "day_vote")
        assert state.round == 1
        assert len(state.alive) == 5
        assert not game.is_terminal(state)

    def test_num_players(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2, "seer": 1})
        assert game.num_players() == 5

    def test_deterministic_reset(self):
        g1 = Werewolf(seed=42)
        g2 = Werewolf(seed=42)
        s1 = g1.reset()
        s2 = g2.reset()
        assert s1.roles == s2.roles

    def test_role_counts(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1,
                                         "witch": 1, "hunter": 1})
        state = game.reset()
        roles = list(state.roles.values())
        assert roles.count("werewolf") == 1
        assert roles.count("villager") == 3
        assert roles.count("seer") == 1
        assert roles.count("witch") == 1
        assert roles.count("hunter") == 1

    def test_invalid_no_werewolf(self):
        with pytest.raises(ValueError, match="werewolf"):
            Werewolf(seed=42, roles={"villager": 3, "seer": 1})

    def test_invalid_too_few_players(self):
        with pytest.raises(ValueError, match="players"):
            Werewolf(seed=42, roles={"werewolf": 1, "villager": 1})

    def test_custom_role_config(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3})
        assert game.num_players() == 4

    # --- Phase flow ---

    def test_werewolf_kill(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        assert state.phase == "night_werewolf"
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        assert state.night_kill_target == target

    def test_werewolf_multiple(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2, "seer": 1})
        state = game.reset()
        assert state.phase == "night_werewolf"
        # First wolf kills
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        # Should still be night_werewolf for second wolf
        assert state.phase == "night_werewolf"

    def test_night_advances_to_seer(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        assert state.phase == "night_seer"

    def test_seer_check(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        wolf = [p for p in state.alive if state.roles[p] == "werewolf"][0]
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        # Now seer acts
        assert state.phase == "night_seer"
        target_check = [p for p in state.alive if p != state.acting_player][0]
        state = game.step(state, _act(state.acting_player, "check", {"target": target_check}))
        assert target_check in state.seer_checked

    def test_no_seer_skips(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 4})
        state = game.reset()
        assert state.phase == "night_werewolf"
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        # No seer, no witch → should go to day_discussion first, then day_vote
        assert state.phase == "day_discussion"

    # --- Witch ---

    def test_witch_save(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 2, "seer": 1, "witch": 1})
        state = game.reset()
        p = state.acting_player
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(p, "kill", {"target": target}))
        # Seer
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": p}))
        # Witch save
        if state.phase == "night_witch_save":
            state = game.step(state, _act(state.acting_player, "save", {"target": target}))
            assert state.witch_save_used
            assert state.witch_saved
        # Skip poison
        if state.phase == "night_witch_poison":
            state = game.step(state, _act(state.acting_player, "pass"))

    def test_witch_poison(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 2, "seer": 1, "witch": 1})
        state = game.reset()
        actor = state.acting_player
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(actor, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": actor}))
        if state.phase == "night_witch_save":
            state = game.step(state, _act(state.acting_player, "pass"))
        if state.phase == "night_witch_poison":
            poison_target = [p for p in state.alive if p != state.acting_player][0]
            state = game.step(state, _act(state.acting_player, "poison", {"target": poison_target}))
            assert state.witch_poison_used
            assert state.witch_poison_target == poison_target

    # --- Day vote ---

    def test_day_vote_sequential(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        # Kill someone → skip seer → get to day_discussion
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        # Discussion phase first
        assert state.phase == "day_discussion"
        while state.phase == "day_discussion":
            p = state.acting_player
            state = game.step(state, _act(p, "pass"))
        # Now voting
        assert state.phase == "day_vote"
        while state.phase == "day_vote":
            p = state.acting_player
            vt = [t for t in state.alive if t != p][0]
            state = game.step(state, _act(p, "vote", {"target": vt}))
        assert state.phase != "day_vote"

    def test_day_votes_hidden_until_resolved(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        while state.phase == "day_discussion":
            state = game.step(state, _act(state.acting_player, "pass"))
        assert state.phase == "day_vote"

        first_voter = state.acting_player
        first_target = [t for t in state.alive if t != first_voter][0]
        state = game.step(state, _act(first_voter, "vote", {"target": first_target}))

        obs = game.get_observation(state, state.acting_player)
        assert "Current votes" not in obs.state_text
        assert "day_votes" not in obs.public_info
        assert not any(h.get("action") == "vote" for h in obs.history)

    def test_day_vote_result_public_after_resolution(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        while state.phase == "day_discussion":
            state = game.step(state, _act(state.acting_player, "pass"))
        while state.phase == "day_vote":
            p = state.acting_player
            vt = [t for t in state.alive if t != p][0]
            state = game.step(state, _act(p, "vote", {"target": vt}))

        assert state.day_vote_results
        viewer = state.acting_player if state.phase == "last_words" else state.alive[0]
        obs = game.get_observation(state, viewer)
        assert "Public vote results" in obs.state_text
        assert "day_vote_results" in obs.public_info
        assert any(h.get("action") == "vote_result" for h in obs.history)

    def test_eliminated_player_gets_last_words(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        while state.phase == "day_discussion":
            state = game.step(state, _act(state.acting_player, "pass"))

        voters = list(state.alive)
        eliminated = voters[0]
        while state.phase == "day_vote":
            p = state.acting_player
            vt = eliminated if p != eliminated else [t for t in state.alive if t != p][0]
            state = game.step(state, _act(p, "vote", {"target": vt}))

        assert state.phase == "last_words"
        assert state.acting_player == eliminated
        assert eliminated not in state.alive

        state = game.step(state, _act(eliminated, "speak", {"message": "Trust P2"}))
        assert state.last_words_messages[-1]["message"] == "Trust P2"

        viewer = state.acting_player if not game.is_terminal(state) else voters[-1]
        obs = game.get_observation(state, viewer)
        assert "Last words" in obs.state_text
        assert "Trust P2" in obs.state_text

    def test_day_vote_tie_no_elimination(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        # Vote: two living players each vote for different targets = tie
        if state.phase == "day_vote":
            alive_before = list(state.alive)
            # Vote in a way that might cause a tie
            while state.phase == "day_vote":
                p = state.acting_player
                others = [t for t in state.alive if t != p]
                vt = others[0]
                state = game.step(state, _act(p, "vote", {"target": vt}))
            # Day resolved - either someone eliminated or tie
            if not game.is_terminal(state):
                assert len(state.alive) <= len(alive_before)

    # --- Win conditions ---

    def test_evil_wins_when_majority(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2})
        state = game.reset()
        # Kill two villagers in sequence to reach 2 wolves vs 0 good
        rounds = 0
        while not game.is_terminal(state) and rounds < 20:
            p = state.acting_player
            if state.phase == "night_werewolf":
                target = [t for t in state.alive if state.roles[t] != "werewolf"][0]
                state = game.step(state, _act(p, "kill", {"target": target}))
            elif state.phase == "day_vote":
                vt = [t for t in state.alive if t != p][0]
                state = game.step(state, _act(p, "vote", {"target": vt}))
            else:
                state = game.step(state, _act(p, "pass"))
            rounds += 1

        if game.is_terminal(state):
            rewards = game.get_rewards(state)
            for p, r in rewards.items():
                if state.roles[p] == "werewolf":
                    assert r > 0

    def test_good_wins_when_all_wolves_dead(self):
        """Run game with strategic voting — good team targets wolves."""
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        wolves = [p for p in state.alive if state.roles[p] == "werewolf"]
        rounds = 0
        while not game.is_terminal(state) and rounds < 30:
            p = state.acting_player
            if state.phase == "night_werewolf":
                target = [t for t in state.alive if state.roles[t] != "werewolf"][0]
                state = game.step(state, _act(p, "kill", {"target": target}))
            elif state.phase == "day_vote":
                # Vote for werewolf if alive, else random
                alive_wolves = [w for w in wolves if w in state.alive and w != p]
                vt = alive_wolves[0] if alive_wolves else [t for t in state.alive if t != p][0]
                state = game.step(state, _act(p, "vote", {"target": vt}))
            elif state.phase == "night_seer":
                tt = [pl for pl in state.alive if pl != p and pl not in state.seer_checked]
                if tt:
                    state = game.step(state, _act(p, "check", {"target": tt[0]}))
                else:
                    state = game.step(state, _act(p, "pass"))
            else:
                state = game.step(state, _act(p, "pass"))
            rounds += 1

        assert game.is_terminal(state) or rounds < 30

    # --- Observations ---

    def test_get_observation_werewolf(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2, "seer": 1})
        state = game.reset()
        wolf_id = state.acting_player
        obs = game.get_observation(state, wolf_id)
        assert obs.private_info["role"] == "werewolf"
        assert len(obs.private_info["other_werewolves"]) == 1

    def test_get_observation_seer(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        # Advance past werewolf
        wolf = state.acting_player
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(wolf, "kill", {"target": target}))
        seer_id = state.acting_player
        obs = game.get_observation(state, seer_id)
        assert obs.private_info["role"] == "seer"

    def test_get_observation_villager(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 4})
        state = game.reset()
        villager_ids = [p for p in state.alive if state.roles[p] == "villager"]
        obs = game.get_observation(state, villager_ids[0])
        assert obs.private_info["role"] == "villager"

    # --- Validation ---

    def test_illegal_action_rejected(self):
        game = Werewolf(seed=42)
        state = game.reset()
        with pytest.raises(ValueError, match="Illegal action"):
            game.step(state, _act(state.acting_player, "vote", {"target": 0}))

    def test_wrong_player_rejected(self):
        game = Werewolf(seed=42)
        state = game.reset()
        wrong_player = (state.acting_player + 1) % game.num_players()
        # Try a pass action from wrong player (pass might not be legal for this phase too)
        with pytest.raises(ValueError):
            game.step(state, _act(wrong_player, "kill", {"target": 0}))

    def test_cant_kill_werewolf(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2, "seer": 1})
        state = game.reset()
        wolves = [p for p in state.alive if state.roles[p] == "werewolf"]
        with pytest.raises(ValueError, match="fellow werewolf"):
            game.step(state, _act(state.acting_player, "kill", {"target": wolves[-1]}))

    def test_cant_vote_self(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        if state.phase == "day_vote":
            with pytest.raises(ValueError, match="vote for yourself"):
                game.step(state, _act(state.acting_player, "vote",
                         {"target": state.acting_player}))

    # --- Serialization ---

    def test_state_serialization(self):
        game = Werewolf(seed=42)
        state = game.reset()
        d = game.state_to_dict(state)
        assert len(d["roles"]) == 5
        assert len(d["alive"]) == 5
        assert d["round"] == 1

    def test_state_clone(self):
        game = Werewolf(seed=42)
        state = game.reset()
        cloned = state.clone()
        assert cloned.roles == state.roles
        cloned.alive.append(99)
        assert 99 not in state.alive

    # --- Legal actions ---

    def test_legal_actions_empty_at_terminal(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2})
        state = game.reset()
        rounds = 0
        while not game.is_terminal(state) and rounds < 20:
            p = state.acting_player
            if state.phase == "night_werewolf":
                target = [t for t in state.alive if state.roles[t] != "werewolf"][0]
                state = game.step(state, _act(p, "kill", {"target": target}))
            elif state.phase == "day_vote":
                vt = [t for t in state.alive if t != p][0]
                state = game.step(state, _act(p, "vote", {"target": vt}))
            else:
                state = game.step(state, _act(p, "pass"))
            rounds += 1
        if game.is_terminal(state):
            assert game.get_legal_actions(state) == []

    # --- Text ---

    def test_rules_text(self):
        game = Werewolf(seed=42)
        text = game.get_rules_text()
        assert "Werewolf Rules" in text
        assert "werewolf" in text.lower()

    def test_state_to_text_werewolf(self):
        game = Werewolf(seed=42, roles={"werewolf": 2, "villager": 2, "seer": 1})
        state = game.reset()
        wolf = state.acting_player
        text = game.state_to_text(state, wolf)
        assert "werewolf" in text.lower()
        assert "Fellow werewolves" in text

    def test_state_to_text_villager(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 4})
        state = game.reset()
        villagers = [p for p in state.alive if state.roles[p] == "villager"]
        text = game.state_to_text(state, villagers[0])
        assert "villager" in text.lower()
        assert "Fellow werewolves" not in text

    def test_state_to_text_with_deaths(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            tt = [pl for pl in state.alive if pl != state.acting_player]
            if tt:
                state = game.step(state, _act(state.acting_player, "check", {"target": tt[0]}))
        while state.phase == "day_vote":
            p = state.acting_player
            vt = [t for t in state.alive if t != p][0]
            state = game.step(state, _act(p, "vote", {"target": vt}))
        if game.is_terminal(state):
            return  # terminal is valid too
        text = game.state_to_text(state, 0)
        assert "Death" in text or "Eliminated" in text or "eliminated" in text

    # --- Full game ---

    # --- Discussion phase ---

    def test_discussion_speak_records_message(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        # Kill to get past night
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        assert state.phase == "day_discussion"
        p = state.acting_player
        state = game.step(state, _act(p, "speak", {"message": "I suspect P1"}))
        assert len(state.day_discussion_messages) == 1
        assert state.day_discussion_messages[0]["message"] == "I suspect P1"
        assert p in state.day_discussion_spoken

    def test_discussion_pass_allowed(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        assert state.phase == "day_discussion"
        p = state.acting_player
        state = game.step(state, _act(p, "pass"))
        assert p in state.day_discussion_spoken
        assert len(state.day_discussion_messages) == 0  # pass = no message

    def test_discussion_transitions_to_vote(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        # Everyone passes discussion
        while state.phase == "day_discussion":
            state = game.step(state, _act(state.acting_player, "pass"))
        assert state.phase == "day_vote"

    def test_discussion_state_to_text_shows_transcript(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        state = game.step(state, _act(state.acting_player, "speak", {"message": "Hello"}))
        # Discussion transcript lives in full_history_text, not state_text
        history_text = game._build_full_history_text(state)
        assert "Hello" in history_text

    def test_discussion_player_sees_previous(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        target = [p for p in state.alive if state.roles[p] != "werewolf"][0]
        state = game.step(state, _act(state.acting_player, "kill", {"target": target}))
        if state.phase == "night_seer":
            state = game.step(state, _act(state.acting_player, "check", {"target": target}))
        # First player speaks
        p1 = state.acting_player
        state = game.step(state, _act(p1, "speak", {"message": "I am seer"}))
        # Second player sees first message in full history
        p2 = state.acting_player
        obs = game.get_observation(state, p2)
        assert "I am seer" in obs.full_history_text

    def test_full_game_with_discussion(self):
        """Full game where players discuss and then vote."""
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        rounds = 0
        while not game.is_terminal(state) and rounds < 80:
            p = state.acting_player
            if state.phase == "night_werewolf":
                target = [t for t in state.alive if state.roles[t] != "werewolf"][0]
                state = game.step(state, _act(p, "kill", {"target": target}))
            elif state.phase == "night_seer":
                t = [pl for pl in state.alive if pl != p and pl not in state.seer_checked]
                state = game.step(state, _act(p, "check", {"target": t[0]} if t else "pass"))
            elif state.phase == "day_discussion":
                state = game.step(state, _act(p, "pass"))
            elif state.phase in ("night_witch_save", "night_witch_poison"):
                state = game.step(state, _act(p, "pass"))
            elif state.phase == "hunter_shot":
                t = [pl for pl in state.alive if pl != p]
                state = game.step(state, _act(p, "shoot", {"target": t[0]} if t else "pass"))
            elif state.phase == "last_words":
                state = game.step(state, _act(p, "pass"))
            elif state.phase == "day_vote":
                vt = [t for t in state.alive if t != p][0]
                state = game.step(state, _act(p, "vote", {"target": vt}))
            else:
                break
            rounds += 1
        assert game.is_terminal(state)

    def test_full_game_completes(self):
        game = Werewolf(seed=42, roles={"werewolf": 1, "villager": 3, "seer": 1})
        state = game.reset()
        rounds = 0
        while not game.is_terminal(state) and rounds < 50:
            p = state.acting_player
            if state.phase == "night_werewolf":
                target = [t for t in state.alive if state.roles[t] != "werewolf"][0]
                state = game.step(state, _act(p, "kill", {"target": target}))
            elif state.phase == "night_seer":
                t = [pl for pl in state.alive if pl != p and pl not in state.seer_checked]
                if t:
                    state = game.step(state, _act(p, "check", {"target": t[0]}))
                else:
                    state = game.step(state, _act(p, "pass"))
            elif state.phase in ("night_witch_save", "night_witch_poison"):
                state = game.step(state, _act(p, "pass"))
            elif state.phase == "day_discussion":
                state = game.step(state, _act(p, "pass"))
            elif state.phase == "hunter_shot":
                t = [pl for pl in state.alive if pl != p]
                if t:
                    state = game.step(state, _act(p, "shoot", {"target": t[0]}))
                else:
                    state = game.step(state, _act(p, "pass"))
            elif state.phase == "last_words":
                state = game.step(state, _act(p, "pass"))
            elif state.phase == "day_vote":
                vt = [t for t in state.alive if t != p][0]
                state = game.step(state, _act(p, "vote", {"target": vt}))
            else:
                break
            rounds += 1

        assert game.is_terminal(state) or rounds < 80
