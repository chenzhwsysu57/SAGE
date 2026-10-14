"""Paired, isolated episodes for Agent-Pro; no training, API setup or action repair.

Full states below are audit artifacts, never decision/reflection inputs. Families
ignore future RNG streams deliberately: identical initial deals cannot cross
splits merely by using a different seed. Completed caches are checksummed AND
re-executed locally against current rules/prompts/opponent before being returned.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import random
import re
import sys
import tempfile

from base.baseline_calls import ResumeConflict
from games.goofspiel import Goofspiel
from games.leduc import LeducHoldem
from games.liars_dice import LiarsDice
from .opponents import LocalSpecifiedOpponent, opponent_config
from .calls import AgentProCallLedger
from .method import AgentProMethod, visible_view
from .policy import GAMES, PolicyState, canonical, digest, strict_json
from .retry import retry_count
from .fallback import check_policy, fallback_payload

ROOT = Path(__file__).resolve().parents[2]
OPPONENT_SEED_OFFSET = 1000003
GAME_CONFIG = {"leduc": "JJQQKK_equal_antes", "liars_dice": "2x3_wild1",
               "goofspiel": "n5_split_ties_visible_prize_sequence"}


def _snapshot(game, state):
    data = deepcopy(game.state_to_dict(state))
    # The existing Goofspiel serializer omits pending bid and history.
    data["history"] = deepcopy(state.history)
    if isinstance(game, Goofspiel):
        data["pending_bid"] = state.pending_bid
    return strict_json(canonical(data))


@dataclass(frozen=True)
class Scenario:
    game: str
    seed: int
    player_id: int = 0
    swap_private: bool = False
    split: str = "train"

    def __post_init__(self):
        if type(self.game) is not str or self.game not in GAMES:
            raise ValueError("Unsupported scenario game")
        if type(self.seed) is not int or type(self.player_id) is not int or self.player_id not in (0, 1):
            raise ValueError("Scenario seed/seat must be integers, not bool")
        if type(self.swap_private) is not bool:
            raise ValueError("swap_private must be bool")
        if self.game == "goofspiel" and self.swap_private:
            raise ValueError("Goofspiel has no private allocation to swap")
        if type(self.split) is not str or self.split not in ("train", "dev", "test", "online"):
            raise ValueError("Expected train/dev/test/online split")

    def instantiate(self):
        """Return fresh (engine, initial state), with RNG *after* reset preserved."""
        if self.game == "leduc":
            game = LeducHoldem(seed=self.seed)
        elif self.game == "liars_dice":
            game = LiarsDice(num_dice=2, num_faces=3, seed=self.seed)
        else:
            game = Goofspiel(n=5, seed=self.seed)
        state = game.reset()
        if self.swap_private:
            if self.game == "leduc":
                state.deck[0], state.deck[1] = state.deck[1], state.deck[0]
                state.private_cards[0], state.private_cards[1] = state.private_cards[1], state.private_cards[0]
            else:
                state.dice[0], state.dice[1] = state.dice[1], state.dice[0]
        return game, state

    @property
    def scenario_id(self) -> str:
        game, state = self.instantiate()
        return digest({"version": 1, "game": self.game, "config": GAME_CONFIG[self.game],
                       "initial_state": _snapshot(game, state), "player_id": self.player_id,
                       "seed": self.seed, "rng_after_reset": game.rng.getstate() if hasattr(game, "rng") else None,
                       "opponent_seed": self.seed + OPPONENT_SEED_OFFSET,
                       "rng_protocol": "python_random_post_reset_v1"})

    @property
    def family_id(self) -> str:
        _, state = self.instantiate()
        if self.game == "goofspiel":
            initial = list(state.prize_sequence)
        elif self.game == "leduc":
            # Undealt deck[3:] and suits cannot affect Leduc payoffs. Rank-equivalent
            # deals must stay in the same split, not only byte-identical cards.
            initial = {"private_cards": sorted(c.rank for c in state.private_cards.values()),
                       "community_card": state.deck[2].rank}
        else:
            initial = sorted(sorted(dice) for dice in state.dice.values())
        return digest({"game": self.game, "config": GAME_CONFIG[self.game], "initial": initial})

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self), "scenario_id": self.scenario_id,
                "family_id": self.family_id}

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        fields = {"game", "seed", "player_id", "swap_private", "split"}
        if (type(data) is not dict or set(data) != fields | {"schema_version", "scenario_id", "family_id"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1):
            raise ValueError("Invalid scenario schema")
        result = cls(**{k: data[k] for k in fields})
        if result.to_dict() != data:
            raise ValueError("Scenario fingerprint mismatch")
        return result


def make_panel(game, seeds, split) -> list[Scenario]:
    # Validate even an empty panel's game/split.
    Scenario(game, 0, split=split)
    return [Scenario(game, seed, seat, swap, split) for seed in seeds
            for seat in (0, 1) for swap in ((False,) if game == "goofspiel" else (False, True))]


def validate_splits(train, dev, test=()) -> None:
    seen = set()
    for split, panel in (("train", train), ("dev", dev), ("test", test)):
        families = set()
        for scenario in panel:
            if not isinstance(scenario, Scenario) or scenario.split != split:
                raise ValueError("Scenario has incorrect split")
            families.add(scenario.family_id)
        if seen & families:
            raise ValueError("Initial-state family overlaps across splits")
        seen.update(families)


def _implementation_id():
    paths = ["methods/agent_pro/" + f + ".py" for f in ("scenarios", "method", "policy", "calls", "retry", "scripted", "opponents", "fallback", "fallback_calls")]
    paths += ["games/leduc.py", "games/liars_dice.py", "games/goofspiel.py", "games/prize_pool_n5.txt",
              "base/interfaces.py", "base/types.py", "base/liars_dice_core.py", "base/model_output.py",
              "base/baseline_calls.py", "methods/baseline_opponents.py", "methods/sage/liars_dice_cfr.py",
              "modeling/goofspiel/archetypes.py", "methods/baseline/simple_prompt.py",
              "methods/sage/cfr.py", "tools/leduc/cfr_tools.py"]
    paths += ["methods/agent_pro/style_opponents.py", "modeling/leduc/archetypes.py",
              "modeling/liars_dice/archetypes.py", "methods/helpers.py", "methods/sage/goofspiel_cfr.py"]
    return digest({"python": sys.version, "sources": {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}})


def _model_id(model):
    # Read only explicit non-secret generation settings, never __dict__/keys or
    # transport configuration. Custom backends can declare a JSON-valued
    # cache_fingerprint for settings not represented by this narrow whitelist.
    backend = model._proxy.backend if isinstance(model, AgentProCallLedger) else model
    settings = {}
    for name in ("model", "model_name", "temperature", "top_p", "max_tokens", "cache_fingerprint"):
        value = getattr(backend, name, None)
        if value is None or type(value) in (str, int, float, bool, dict, list):
            settings[name] = value
        elif name == "cache_fingerprint":
            raise ValueError("cache_fingerprint must be a finite JSON value, not a callable")
    return digest({"wrapper": type(model).__module__ + "." + type(model).__qualname__,
                   "backend": type(backend).__module__ + "." + type(backend).__qualname__,
                   "generation_settings": settings})


def _identity(identity):
    def check(value):
        if type(value) is dict:
            for key, item in value.items():
                if (type(key) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", key)
                        or re.search(r"key|secret|password|authorization|token|credential", key, re.I)):
                    raise ValueError("Unsafe episode identity")
                check(item)
        elif type(value) is list:
            for item in value:
                check(item)
        elif type(value) is str:
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}", value) or ".." in value or value.startswith("sk-"):
                raise ValueError("Unsafe episode identity")
        elif value is not None and type(value) not in (int, float, bool):
            raise ValueError("Episode identity must be JSON")
    if type(identity) is not dict or not identity:
        raise ValueError("Expected nonempty episode identity")
    encoded = canonical(identity)
    if len(encoded) > 16384:
        raise ValueError("Episode identity is too large")
    check(identity)
    return strict_json(encoded)


def _atomic(path, value):
    if path.is_symlink():
        raise ResumeConflict("Symlink episode path")
    fd, temporary = tempfile.mkstemp(prefix=".episode-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ScenarioEvaluator:
    """Serial evaluator; caller supplies a journal-bound model for crash recovery.

    A new identity executes anew. Same identity returns a validated full episode;
    without a complete episode, re-enter the model journal normally. Exceptions
    propagate, are never recorded as losses, and do not increment completed.
    """

    def __init__(self, model, run_dir: Path, opponent: str = "uniform", cfr_model=None,
                 decision_parse_retries: int = 3, *, action_failure_policy="strict"):
        from .style_opponents import STYLES
        if opponent not in {"uniform", *[s for v in STYLES.values() for s in v]}:
            raise ValueError("Unsupported local opponent")
        if opponent != "GTO" and cfr_model is not None:
            raise ValueError("Only GTO accepts a CFR model")
        self.model = model
        self.action_failure_policy = check_policy(action_failure_policy)
        self.decision_parse_retries = retry_count(decision_parse_retries)
        self.run_dir = Path(run_dir).absolute()
        self.opponent = opponent
        self.cfr_model = cfr_model
        self.completed = 0
        self.skipped = 0
        self._directory = self.run_dir / "trajectories"
        if any(p.is_symlink() for p in (self._directory, *self._directory.parents)):
            raise ResumeConflict("Symlink run directory")
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def cache_hits(self):
        return self.skipped

    def _journal_dependencies(self, policy, scenario, identity, episode):
        """Pin only this episode's journal entries; unrelated appends remain valid.

        The ledger has no public export API. Read its documented on-disk mapping
        after stats() validates it; never alter the mapping or invoke a backend.
        """
        if not isinstance(self.model, AgentProCallLedger):
            return []
        self.model.stats()
        mapping = strict_json((self.model.run_dir / "logs" / "agent_pro_call_ids.json").read_text(encoding="utf-8"))
        entries = {(canonical(entry["identity"]), entry["stage"]): entry for entry in mapping["calls"]}
        dependencies = []
        for step in episode["trajectory"]["steps"]:
            if step["player_id"] != scenario.player_id:
                if not step.get("opponent_metadata"):
                    continue
                scope = {**identity, "policy_id": policy.policy_id, "scenario_id": scenario.scenario_id,
                         "player_id": step["player_id"], "step": step["step"], "actor": "opponent"}
                entry = entries.get((canonical(scope), "opponent"))
                if entry is None:
                    raise ResumeConflict("Missing opponent journal dependency")
                path = self.model._record_path(entry["game_index"])
                if path.is_symlink():
                    raise ResumeConflict("Symlink opponent journal")
                content = path.read_bytes();record = strict_json(content.decode())
                if record["status"] != "success" or record["prompt"] != step["prompt"] or record["response"] != step["response"]:
                    raise ResumeConflict("Opponent episode and journal disagree")
                dependencies.append({"entry": entry, "sha256": hashlib.sha256(content).hexdigest()})
                continue
            scope = {**identity, "policy_id": policy.policy_id, "scenario_id": scenario.scenario_id,
                     "player_id": scenario.player_id, "step": step["step"]}
            if step["decision_metadata"].get("source") == "random_fallback":
                path = self.model._certificate_path(scope)
                content = path.read_bytes()
                certificate = strict_json(content.decode())
                if canonical(certificate["payload"]) != canonical(step["decision_metadata"]):
                    raise ResumeConflict("Episode fallback certificate changed")
                dependencies.append({"fallback_identity": scope,
                                     "fallback_sha256": hashlib.sha256(content).hexdigest()})
            for attempt in step["decision_metadata"]["decision_attempts"]:
                entry = entries.get((canonical(scope), attempt["stage"]))
                if entry is None:
                    raise ResumeConflict("Completed episode is missing its journal identity")
                path = self.model._record_path(entry["game_index"])
                validation_path = self.model.validation_path(entry["game_index"])
                if path.is_symlink() or validation_path.is_symlink():
                    raise ResumeConflict("Symlink episode journal/validation")
                record_bytes = path.read_bytes()
                record = strict_json(record_bytes.decode("utf-8"))
                if attempt["validation"] == "failed":
                    receipt_path = self.model._receipt_path(entry["game_index"])
                    content = receipt_path.read_bytes()
                    if (record["status"] != "failed" or
                            canonical(strict_json(content.decode())["attempt"]) != canonical(attempt)):
                        raise ResumeConflict("Episode failure receipt changed")
                    dependencies.append({"entry": entry, "sha256": hashlib.sha256(record_bytes).hexdigest(),
                                         "failure_sha256": hashlib.sha256(content).hexdigest()})
                    continue
                validation_bytes = validation_path.read_bytes()
                if (entry["prompt_hash"] != hashlib.sha256(attempt["prompt"].encode("utf-8", errors="surrogatepass")).hexdigest()
                        or record["response"] != attempt["response"] or record["status"] != "success"
                        or canonical(strict_json(validation_bytes.decode())["attempt"]) != canonical(attempt)):
                    raise ResumeConflict("Completed episode and model journal disagree")
                dependencies.append({"entry": entry, "sha256": hashlib.sha256(record_bytes).hexdigest(),
                                     "validation_sha256": hashlib.sha256(validation_bytes).hexdigest()})
        return dependencies

    def _play(self, policy, scenario, identity, config, replay=None):
        game, state = scenario.instantiate()
        initial = _snapshot(game, state)
        agent = AgentProMethod(policy, scenario.player_id, parse_retries=self.decision_parse_retries,
                       action_failure_policy=self.action_failure_policy, fallback_seed=scenario.seed)
        seed = scenario.seed + OPPONENT_SEED_OFFSET
        opponent = None if self.opponent == "uniform" else LocalSpecifiedOpponent(scenario.game, config, seed)
        rng = random.Random(seed)
        steps = []
        while not game.is_terminal(state):
            step = len(steps) + 1
            if step > 256:
                raise ValueError("Episode exceeded finite-game safety limit")
            player = game.get_current_player(state)
            observation = game.get_observation(state, player)
            legal = game.get_legal_actions(state)
            before = _snapshot(game, state)
            prompt = response = ""
            metadata = {}
            opponent_metadata = None
            if player == scenario.player_id:
                if replay is None:
                    bind = getattr(self.model, "bind", None)
                    if callable(bind):
                        bind({**identity, "policy_id": policy.policy_id, "scenario_id": scenario.scenario_id,
                              "player_id": player, "step": step})
                    backend = self.model
                else:
                    saved = replay[step - 1]
                    class Replay:
                        def __init__(self):
                            self.index = 0

                        def restore_action_fallback(self, context, policy_id):
                            md = saved["decision_metadata"]
                            if md.get("source") != "random_fallback":
                                return None
                            expected = fallback_payload(context, md["decision_attempts"], md["fallback_reason"], policy_id)
                            if canonical(expected) != canonical(md):
                                raise ResumeConflict("Cached fallback does not reproduce")
                            self.index = len(md["decision_attempts"])
                            return expected

                        def generate_stage(self, stage, text):
                            attempt = saved["decision_metadata"]["decision_attempts"][self.index]
                            if text != attempt["prompt"] or stage != attempt["stage"]:
                                raise ResumeConflict("Cached decision prompt/stage changed")
                            # Preserve metadata-less legacy clients on the first
                            # attempt, but never carry a prior stop into a later
                            # completion whose finish metadata is unknown.
                            if attempt["finish_reason"] is not None or hasattr(self, "last_finish_reason"):
                                self.last_finish_reason = attempt["finish_reason"]
                            self.index += 1
                            return attempt["response"]
                    backend = Replay()
                action, prompt, response, metadata = agent.act(observation, legal, backend)
                if replay is not None and backend.index != len(saved["decision_metadata"]["decision_attempts"]):
                    raise ResumeConflict("Unconsumed cached decision attempts")
            else:
                if config.get("execution") == "llm_style":
                    if replay is None:
                        bind = getattr(self.model, "bind", None)
                        if callable(bind):
                            bind({**identity, "policy_id": policy.policy_id, "scenario_id": scenario.scenario_id,
                                  "player_id": player, "step": step, "actor": "opponent"})
                        backend = self.model
                        fixture_context = getattr(backend, "set_opponent_fixture", None)
                        if callable(fixture_context):
                            fixture_context(scenario.game, observation, legal)
                    else:
                        saved = replay[step - 1]
                        class OpponentReplay:
                            def generate_stage(self, stage, text):
                                if stage != "opponent" or text != saved["prompt"]:
                                    raise ResumeConflict("Opponent replay prompt changed")
                                return saved["response"]
                        backend = OpponentReplay()
                    action, prompt, response, opponent_metadata = opponent.act(observation, legal, backend)
                else:
                    action = rng.choice(legal) if opponent is None else opponent.act(observation, legal, None)[0]
            if not any(canonical(action.model_dump()) == canonical(a.model_dump()) for a in legal):
                raise ValueError("Illegal episode action; no automatic repair")
            state = game.step(state, action)
            steps.append({"step": step, "player_id": player, "action": action.model_dump(),
                          "state_before": before, "state_after": _snapshot(game, state),
                          "prompt": prompt, "response": response, "decision_metadata": metadata})
            if opponent_metadata is not None:
                steps[-1]["opponent_metadata"] = opponent_metadata
        terminal = game.get_observation(state, scenario.player_id)
        net = game.get_net_profit(state, scenario.player_id)
        feedback = {"game": scenario.game, "player_id": scenario.player_id, "protocol": "rules_visible_v1",
                    "rules_text": game.get_rules_text(), "net_profit": net,
                    "terminal_observation": visible_view(scenario.game, terminal),
                    "public_history": deepcopy(terminal.history), "public_reveal": {}}
        if scenario.game == "leduc" and state.history[-1]["action"] != "fold":
            feedback["public_reveal"] = {"showdown_cards": {str(p): str(c) for p, c in state.private_cards.items()}}
        episode = {"game": scenario.game, "split": scenario.split, "policy_id": policy.policy_id,
                "scenario_id": scenario.scenario_id, "family_id": scenario.family_id,
                "player_id": scenario.player_id, "net_profit": net, "records": agent.records,
                "feedback": feedback, "trajectory": {"initial_state": initial, "steps": steps,
                "final_state": _snapshot(game, state), "net_profit": {str(p): game.get_net_profit(state, p) for p in (0, 1)}}}
        count = sum(r.get("source") == "random_fallback" for r in agent.records)
        if count:
            episode.update(action_fallback_count=count, learning_eligible=False)
        return episode

    def evaluate(self, policy: PolicyState, scenario: Scenario, identity: dict) -> dict:
        if policy.game != scenario.game:
            raise ValueError("Policy/scenario game mismatch")
        identity = _identity(identity)
        for key, value in (("policy_id", policy.policy_id), ("scenario_id", scenario.scenario_id),
                           ("player_id", scenario.player_id)):
            if key in identity and canonical(identity[key]) != canonical(value):
                raise ValueError("Conflicting episode identity")
        if "step" in identity:
            raise ValueError("Step is owned by the evaluator")
        config = ({"name": "uniform", "execution": "local_uniform", "live_api": False}
                  if self.opponent == "uniform" else opponent_config(scenario.game, self.opponent, self.cfr_model))
        context = {"identity": identity, "scenario": scenario.to_dict(), "policy": policy.to_dict(),
                   "opponent": config, "implementation_id": _implementation_id(),
                   "decision_parse_retries": self.decision_parse_retries,
                   "action_failure_policy": self.action_failure_policy,
                   "model_id": _model_id(self.model)}
        path = self._directory / (digest({"identity": identity, "scenario_id": scenario.scenario_id,
                                         "policy_id": policy.policy_id}) + ".json")
        if path.is_symlink():
            raise ResumeConflict("Symlink episode cache")
        if path.exists():
            try:
                saved = strict_json(path.read_text(encoding="utf-8"))
                if (set(saved) != {"schema_version", "context", "episode", "checksum", "journal_dependencies"}
                        or type(saved["schema_version"]) is not int or saved["schema_version"] != 1
                        or canonical(saved["context"]) != canonical(context)
                        or saved["checksum"] != digest({"context": context, "episode": saved["episode"]})):
                    raise ValueError("Episode checksum/configuration mismatch")
                # A ledger audits its existing journal even when no calls are needed.
                stats = getattr(self.model, "stats", None)
                if callable(stats):
                    stats()
                if canonical(saved["journal_dependencies"]) != canonical(
                        self._journal_dependencies(policy, scenario, identity, saved["episode"])):
                    raise ValueError("Episode journal changed since completion")
                rebuilt = self._play(policy, scenario, identity, config, saved["episode"]["trajectory"]["steps"])
                if canonical(rebuilt) != canonical(saved["episode"]):
                    raise ValueError("Episode does not reproduce under current engine")
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                raise ResumeConflict("Invalid completed episode; manual review required") from exc
            self.skipped += 1
            return deepcopy(saved["episode"])
        episode = strict_json(canonical(self._play(policy, scenario, identity, config)))
        if context["implementation_id"] != _implementation_id() or context["model_id"] != _model_id(self.model):
            raise ResumeConflict("Implementation or model configuration changed during evaluation")
        if self.opponent != "uniform" and config != opponent_config(scenario.game, self.opponent, self.cfr_model):
            raise ResumeConflict("Opponent model changed during evaluation")
        dependencies = self._journal_dependencies(policy, scenario, identity, episode)
        _atomic(path, {"schema_version": 1, "context": context, "episode": episode,
                       "journal_dependencies": dependencies,
                       "checksum": digest({"context": context, "episode": episode})})
        self.completed += 1
        return deepcopy(episode)