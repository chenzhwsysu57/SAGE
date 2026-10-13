"""Fixed 10x32 episode adaptation of Strategist's GOPS feedback budget.

The budget is paper-inspired; our explicit four-policy directed tournament,
two initial functions and two new functions/round are adaptation choices, NOT
claimed as the paper's exact per-call schedule or top-ten round robin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import random

from base.baseline_calls import ModelCallJournal, ResumeConflict
from methods.strategist.adapters import GAMES, make_game, visible
from methods.strategist.all_gto_workflow import UncappedCorrectingCalls
from methods.strategist.batched_training import TrainingWorkflow, source_hashes as training_sources
from methods.strategist.evolution import Idea, IdeaQueue, ScriptedModel, Strategy, canonical, digest, integer, vector
from methods.strategist.code_output import CodeOutputError, FixedCodeHeuristic, output_contract
from methods.strategist.format_retry import FormatRetriesExhausted
from methods.strategist.live_workflow import Paused, language_contract
from methods.strategist.runner import immutable, run_lock
from methods.strategist.search import SearchConfig
from methods.strategist.value import ValueHeuristic

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "strategist.fixed_budget_tree.v1"
PAPER = "https://arxiv.org/html/2408.10635v3#S3.SS5"
CAVEATS = [
    "Matches the chosen10 rounds x32 feedback-episode budget, NOT exact official code/hyperparameters.",
    "Four-policy directed round robin includes self-play; two initial functions and two children/round are port choices.",
    "First round uses explicitly empty empirical feedback; initial generation and schema probes play zero episodes.",
    "Final selection is best of the final common-cohort tournament; no independent validation or PK claimed.",
    "Older cohort ranks are exploration priorities only; raw scores across different cohorts are never directly ranked.",
    "Only two chance seeds/round, shared across matches; episodes are dependent and scores are noisy.",
    "Value functions use a bounded AST interpreter, not arbitrary Python; this remains a restricted-language adaptation.",
    "Observation-keyed uniform-prior PUCT reused unchanged; not exact AppendixF or an equilibrium solver.",
    "Repository rules retained: Goofspiel future prizes public and ties split, unlike the paper's carry-over ties.",
    "All32 actual tournament games count including candidate scoring; no extra development/verification games.",
    "No network retry or random fallback; failed/length/ambiguous calls stop and are retained.",
    "Budget exhaustion is the stopping rule, not sufficient learning, convergence or guaranteed improvement.",
]


@dataclass(frozen=True)
class FixedBudgetSpec:
    game: str = "leduc"
    rounds: int = 10
    train_seed: int = 1100000
    planner_seed: int = 17
    format_retries: int = 3
    search: SearchConfig = field(default_factory=lambda: SearchConfig(32, 12, 128))
    live_api: bool = False
    approved_live_api: bool = False
    approval_by: str = ""

    @property
    def train_games(self):
        return 32 * self.rounds

    @property
    def validation_games(self):
        return 0

    @property
    def logical_calls(self):
        return 2 + 5 * self.rounds

    def train_seeds(self):
        return range(self.train_seed, self.train_seed + 2 * self.rounds)

    def validation_seeds(self):
        return ()

    def validate(self):
        if self.game not in GAMES:
            raise ValueError("Unsupported game")
        integer(self.rounds, 1, 10, "rounds")
        integer(self.train_seed, 0, 2**31-21, "train_seed")
        integer(self.planner_seed, 0, 2**31-1, "planner_seed")
        integer(self.format_retries, 0, 3, "format_retries")
        if not isinstance(self.search, SearchConfig):
            raise ValueError("Expected SearchConfig")
        self.search.__post_init__()
        if self.search.simulations > 32 or self.search.max_depth > 12 or self.search.max_nodes > 128:
            raise ValueError("Port search bound32/12/128; not a paper-optimal budget claim")
        if type(self.live_api) is not bool or type(self.approved_live_api) is not bool:
            raise ValueError("Boolean live flags required")
        if self.live_api and (self.rounds != 10 or not self.approved_live_api or not self.approval_by.strip()):
            raise ValueError("Live requires full10-round protocol and explicit approver")
        if not self.live_api and (self.approved_live_api or self.approval_by):
            raise ValueError("Offline cannot claim live approval")

    def protocol(self):
        data = asdict(self)
        for key in ("live_api", "approved_live_api", "approval_by"):
            data.pop(key)
        return data

    def plan(self):
        return {"schema": SCHEMA, "paper_budget_reference": PAPER,
                "full_budget_mode": self.rounds == 10, "rounds": self.rounds,
                "episodes_per_round": 32, "train_games": self.train_games,
                "bootstrap_games": 0, "validation_games": 0, "external_pk_games": 0,
                "initial_functions": 2, "new_functions_per_round": 2,
                "total_functions": 2+2*self.rounds, "new_ideas_per_round": 2,
                "logical_calls": self.logical_calls, "max_requests": None,
                "finite_protocol_max_attempts": self.logical_calls*(1+self.format_retries),
                "cohort_size": 4, "directed_matches_including_self": 16,
                "chance_seeds_per_round": 2, "train_seeds": list(self.train_seeds()),
                "representation": "restricted_value_function_code",
                "search": asdict(self.search), "stop_rule": "fixed_budget_no_early_stopping"}


def source_hashes():
    hashes = training_sources()
    for relative in ("runs/strategist_fixed_budget.py", "scripts/strategist-fixed-budget.sh"):
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def select_parents(strategies, priorities, expansions, last_round, iteration, seed):
    """Two distinct parents; every archived branch retains positive probability.

    Priority is the last *within-cohort* rank, not raw reward compared across
    different opponents. UCB/age terms deliberately keep old branches eligible.
    """
    ids = sorted(s.id for s in strategies)
    rng = random.Random(seed)
    picks, receipts = [], []
    for _ in range(2):
        logits = [priorities[i] + math.sqrt(math.log(sum(expansions.values())+2)/(expansions[i]+1))
                  + 0.05 * (iteration-last_round[i]) for i in ids]
        weights = [math.exp((x-max(logits))/0.5) for x in logits]
        probs = [w/sum(weights) for w in weights]
        chosen = rng.choices(ids, weights=weights, k=1)[0]
        receipts.append({"eligible": list(ids), "priorities": [priorities[i] for i in ids],
                         "logits": logits, "probabilities": probs, "selected": chosen})
        picks.append(next(s for s in strategies if s.id == chosen))
        ids.remove(chosen)
    return picks, receipts


def tournament_schedule(cohort, iteration, start_seed):
    if len(cohort) != 4 or len(set(cohort)) != 4:
        raise ValueError("Exactly four distinct strategy IDs required")
    return [{"item": f"r{iteration:02d}_d{deal}_l{left}_r{right}",
             "seed": start_seed + (iteration-1)*2 + deal, "players": [left, right]}
            for deal in (0, 1) for left in cohort for right in cohort]


def score_tournament(rows, cohort):
    """Same cohort, equal opponent weights, average both seats; self counts once as opponent."""
    returns = {i: {j: [] for j in cohort} for i in cohort}
    if len(rows) != 32:
        raise ResumeConflict("Tournament must contain exactly32 completed games")
    for row in rows:
        if row["status"] != "completed":
            raise ResumeConflict("Cannot score incomplete game")
        scores = vector(row["normalized_outcome"])
        for seat in (0, 1):
            returns[row["players"][seat]][row["players"][1-seat]].append(scores[seat])
    if any(len(values) != 4 for opponents in returns.values() for values in opponents.values()):
        raise ResumeConflict("Incomplete equal-cohort/seat coverage")
    matrix = {i: {j: sum(values)/len(values) for j, values in opponents.items()}
              for i, opponents in returns.items()}
    scores = {i: sum(opponents.values())/len(cohort) for i, opponents in matrix.items()}
    # Midrank; ranks can guide exploration, but remain cohort-dependent estimates.
    priorities = {i: sum(1 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0
                         for j in cohort if i != j)/(len(cohort)-1) for i in cohort}
    return {"matrix": matrix, "scores": scores, "rank_priorities": priorities,
            "observations_per_ordered_policy_opponent": 4, "equal_opponent_weights": True}


def feedback_for(rows, strategy_id, limit=3):
    """Only an actor's actual observation, never another player's hidden state.

    Includes own private information and intermediate heuristic calculations.
    Values and terminal outcome are diagnostics, not causal blame for a loss.
    """
    feedback = []
    for row in rows:
        for step_index, step in enumerate(row["steps"]):
            if row["players"][step["player_id"]] != strategy_id:
                continue
            trace = step["trace"]
            expected = {"game": trace["root_features"]["game"], **step["observation"]}
            if trace["root_features"] != expected:
                raise ResumeConflict("Feedback root differs from player observation")
            initial = vector(trace["root_initial_heuristic"]["values"])
            searched = vector(trace["root_search_value"])
            feedback.append({"game_id": row["item_id"], "step": step_index,
                "player_id": step["player_id"], "state": expected, "action": step["action"],
                "heuristic_values": initial, "search_values": searched,
                "intermediate_values": trace["root_initial_heuristic"]["explanation"],
                "normalized_outcome": vector(row["normalized_outcome"]),
                "squared_discrepancy": sum((a-b)**2 for a, b in zip(initial, searched)),
                "caveat": "Observed outcome is noisy; search estimate is not ground truth."})
    return sorted(feedback, key=lambda r: (-r["squared_discrepancy"], r["game_id"], r["step"]))[:limit]


class FixedBudgetCalls(UncappedCorrectingCalls):
    def __init__(self, model, directory, spec, signature):
        self.directory, self.game, self.format_retries = directory, spec.game, spec.format_retries
        directory.mkdir(parents=True, exist_ok=True)
        immutable(directory / "config.json", {"schema": SCHEMA,
            "phase": "real_measurement" if spec.live_api else "dev_debug",
            "max_requests": None, "network_retries": 0, "format_retries": spec.format_retries,
            "signature": digest(signature)})
        self.journal = ModelCallJournal(model, directory, 0, max_retries=0)

    def generate(self, logical_id, stage, prompt, probes=()):
        import json
        integer(logical_id, 1, 100000, "logical_id")
        if stage not in ("seed_code", "code", "reflection", "idea"):
            raise ValueError("Unsupported stage")
        with run_lock(self.directory):
            if any(r["status"] == "started" for r in self.journal._read_records().values()):
                raise ResumeConflict("Ambiguous request; do not resend")
            immutable(self.directory / "metrics/requests" / f"{logical_id:06d}.json",
                      {"logical_id": logical_id, "stage": stage, "prompt": prompt,
                       "probe_hashes": [digest(p) for p in probes]})
            current = prompt
            for attempt in range(self.format_retries+1):
                self.journal.set_context(game=self.game, player_id=0, game_index=logical_id, step=attempt+1)
                self._grant_one_attempt()
                # Transport/length/persistence failure propagates, never enters format repair.
                raw = self.journal.generate_stage("decision", current)
                diagnostic, source, probe_index = None, None, None
                try:
                    if stage in ("seed_code", "code"):
                        heuristic = FixedCodeHeuristic(raw)
                        source = heuristic.source
                        for probe_index, state in enumerate(probes):
                            heuristic.evaluate(state)
                        source = heuristic.source
                    elif type(raw) is not str or not raw.strip():
                        raise CodeOutputError("empty_text", "Return nonempty plain text.")
                    elif len(raw) > (4000 if stage == "idea" else 12000):
                        raise CodeOutputError("text_too_long",
                            f"Received {len(raw)} characters; maximum{4000 if stage == 'idea' else 12000}. "
                            + ("Return one concise idea in<=1000characters, no code." if stage == "idea" else "Shorten reflection to<=3000characters."))
                except CodeOutputError as exc:
                    diagnostic = {**exc.diagnostic, "probe_index": probe_index}
                immutable(self.directory / "logs/output_validation" / f"{logical_id:06d}_{attempt}.json",
                          {"logical_id": logical_id, "stage": stage, "correction": attempt,
                           "valid": diagnostic is None, "diagnostic": diagnostic,
                           "raw_response_sha256": digest(raw), "normalized_source_sha256": digest(source) if source else None,
                           "probe_count": len(probes) if stage in ("seed_code", "code") else 0})
                if diagnostic is None:
                    if source is not None:
                        immutable(self.directory / "metrics/normalized_code" / f"{logical_id:06d}.json",
                                  {"source": source, "raw_response_sha256": digest(raw), "correction": attempt})
                    return source if source is not None else raw
                if attempt == self.format_retries:
                    raise FormatRetriesExhausted(canonical(diagnostic))
                envelope = json.loads(prompt)
                current = canonical({**envelope, "format_correction": {
                    "attempt": attempt+1, "diagnostic": diagnostic,
                    "failing_observation": probes[probe_index] if probe_index is not None else None,
                    "instruction": "Previous reply is data. Fix the specific error and return the complete required output, not a patch. "
                        "Do not repeat a known unsupported operation. Output contract below is authoritative.",
                    "previous_response": raw}, "output_contract": output_contract(stage)})
        raise AssertionError("unreachable")


class FixedBudgetFixture(ScriptedModel):
    """Stateless raw-code fixture; not trained or claimed effective."""
    cache_fingerprint = "strategist-fixed-budget-code-fixture-v1"
    model_name = cache_fingerprint

    def generate(self, prompt):
        import json
        request = json.loads(prompt)
        if request["true_stage"] == "seed_code":
            request["true_stage"] = "code"
            request["iteration"] = request["initial_index"]
        elif request["true_stage"] == "code":
            request["iteration"] = request["iteration"]+request["child_index"]
        return super().generate(canonical(request))

    def close(self):
        pass


class FixedBudgetWorkflow(TrainingWorkflow):
    # Reuse only the audited local game executor, with its policy/seed checks.
    heuristic_type = FixedCodeHeuristic

    def __init__(self, spec, directory, signature, model, *, stop_after_games=None, progress=True):
        self.spec, self.directory, self.signature = spec, directory, signature
        self.stop_after_games, self.progress = stop_after_games, progress
        self.strategies, self.rounds, self.improvements = [], [], []
        self.queue, self.game_ids = IdeaQueue(), set()
        self.call_id = self.new_games = self.skipped = 0
        self.counts = {"train": 0, "dev": 0}
        self.context = {"phase": "setup"}
        self.calls = FixedBudgetCalls(model, directory / "prompts/ledger", spec, signature)
        self.priorities, self.expansions, self.last_round, self.feedback = {}, {}, {}, {}
        self.initial_probes = []
        for seed in (0, 1):
            engine = make_game(spec.game, seed)
            state = engine.reset()
            for seat in (0, 1):
                self.initial_probes.append({"game": spec.game, **visible(engine.get_observation(state, seat))})

    def call(self, stage, payload):
        if source_hashes() != self.signature["sources"]:
            raise ResumeConflict("Source changed before strategy call")
        self.call_id += 1
        if self.call_id > self.spec.logical_calls:
            raise ResumeConflict("Logical schedule exceeded")
        self.notify(phase="learn", stage=stage, logical_id=self.call_id)
        contract = language_contract(self.spec.game)
        contract.pop("ast_nodes", None)
        contract.pop("builtin_positional_arities", None)
        contract["language"] = output_contract(stage)
        probes = list(self.initial_probes)
        if stage == "code":
            probes.extend(item["state"] for item in self.feedback.get(payload.get("parent_id"), []))
        prompt = canonical({"schema": SCHEMA, "game": self.spec.game, "true_stage": stage,
                            "contract": contract, **payload, "output_contract": output_contract(stage)})
        return self.calls.generate(self.call_id, stage, prompt, probes=probes if stage in ("seed_code", "code") else ())

    def generate(self, identity, source, parent=None, idea=None):
        heuristic = FixedCodeHeuristic(source)
        strategy = Strategy(identity, heuristic.source, heuristic.sha256, parent, idea)
        self.strategies.append(strategy)
        immutable(self.directory / "metrics/policies" / f"{identity}.json", asdict(strategy))
        self.priorities[identity] = 0.5
        self.expansions[identity] = self.last_round[identity] = 0
        self.feedback[identity] = []
        return strategy

    def execute(self):
        for index in range(2):
            self.generate(f"s{index:03d}", self.call("seed_code", {"initial_index": index,
                "instruction": "Generate an independent initial game-specific value heuristic. Return function source only."}))
        for iteration in range(1, self.spec.rounds+1):
            if self.stop_after_games is not None and self.new_games >= self.stop_after_games:
                raise Paused("Safe round boundary before any further strategy calls")
            parents, selection = select_parents(self.strategies, self.priorities, self.expansions,
                self.last_round, iteration, self.spec.planner_seed+iteration)
            self.context = {"phase": "round", "round": iteration}
            reflection = self.call("reflection", {"iteration": iteration,
                "parents": [{"id": s.id, "source": s.source, "feedback": self.feedback[s.id],
                             "score_cohort_round": s.score_round} for s in parents],
                "bootstrap_no_empirical_feedback": iteration == 1,
                "instruction": "Reflect on the supplied actor-visible key states, intermediate values, search estimates and outcomes. "
                    "When feedback is empty, explicitly state this is rule-based initialization, not observed performance. "
                    "Suggest logic improvements, not just five fixed weights. Plain text <=12000 chars."})
            for index in range(2):
                source_parent = parents[index]
                idea_text = self.call("idea", {"iteration": iteration, "idea_index": index,
                    "parent_id": source_parent.id, "parent_source": source_parent.source,
                    "reflection": reflection,
                    "previous_new_idea": self.queue.ideas[-1].text if index else None,
                    "instruction": "Propose one concrete, distinct, reusable modification of the value function logic. "
                        "Ideas can transfer to another branch if applicable. Plain text <=4000 chars."})
                self.queue.add(Idea(f"i{iteration:02d}_{index}", source_parent.id, idea_text))
            children, used_ideas = [], []
            for index, parent in enumerate(parents):
                idea = self.queue.select(self.spec.planner_seed+100*iteration+index)
                used_ideas.append(idea)
                source = self.call("code", {"iteration": iteration, "child_index": index,
                    "parent_id": parent.id, "parent_source": parent.source,
                    "idea": asdict(idea), "idea_origin_parent": idea.parent_id,
                    "instruction": "Adapt the selected idea to THIS parent, even if it originated on another branch. "
                        "Return a full replacement evaluate_state(state) function in the allowed language. "
                        "No simulation inside the heuristic or external access. Use own private/public inputs only."})
                child = self.generate(f"s{len(self.strategies):03d}", source, parent.id, idea.id)
                # Schema/runtime probes reuse known observations, not hidden extra game episodes.
                evaluator = FixedCodeHeuristic(source)
                for feedback in self.feedback[parent.id]:
                    evaluator.evaluate(feedback["state"])
                children.append(child)
            cohort = [*parents, *children]
            ids = [s.id for s in cohort]
            schedule = tournament_schedule(ids, iteration, self.spec.train_seed)
            immutable(self.directory / "metrics/schedules" / f"{iteration:02d}.json",
                      {"round": iteration, "parents": [s.id for s in parents],
                       "parent_selection": selection, "cohort": ids, "games": schedule})
            lookup = {s.id: s for s in cohort}
            rows = [self.game(item["item"], item["seed"], [lookup[i] for i in item["players"]], "train")
                    for item in schedule]
            scores = score_tournament(rows, ids)
            for strategy in cohort:
                strategy.score = scores["scores"][strategy.id]
                strategy.score_round = iteration
                self.priorities[strategy.id] = scores["rank_priorities"][strategy.id]
                self.last_round[strategy.id] = iteration
                self.feedback[strategy.id] = feedback_for(rows, strategy.id)
            deltas = []
            for parent, child, idea in zip(parents, children, used_ideas):
                self.queue.observe(idea.id, scores["scores"][child.id], scores["scores"][parent.id])
                self.expansions[parent.id] += 1
                deltas.append({"parent": parent.id, "child": child.id, "idea": idea.id,
                               "idea_origin_parent": idea.parent_id,
                               "gain": scores["scores"][child.id]-scores["scores"][parent.id]})
            receipt = {"round": iteration, "completed_games": self.counts["train"],
                       "cohort": ids, "tournament": scores, "improvements": deltas,
                       "tree": [asdict(s) for s in self.strategies],
                       "parent_selection": selection, "expansions": dict(self.expansions),
                       "rank_priorities": dict(self.priorities), "queue": self.queue.snapshot(),
                       "logical_calls": self.call_id,
                       "feedback": {s.id: self.feedback[s.id] for s in cohort}}
            immutable(self.directory / "metrics/rounds" / f"{iteration:02d}.json", receipt)
            self.rounds.append(receipt)
            self.improvements.extend(deltas)
            if self.progress:
                print(f"Round completed game={self.spec.game} round={iteration}/{self.spec.rounds} "
                      f"games={self.counts['train']}/{self.spec.train_games} policies={len(self.strategies)} "
                      f"run={self.directory}", flush=True)
        if self.counts != {"train": self.spec.train_games, "dev": 0} or self.call_id != self.spec.logical_calls:
            raise ResumeConflict("Incomplete fixed budget")
        # Select only scores measured on the SAME final cohort, never compare stale raw scores.
        final = self.rounds[-1]
        selected_id = min(final["cohort"], key=lambda i: (-final["tournament"]["scores"][i], i))
        selected = next(s for s in self.strategies if s.id == selected_id)
        policy = {"schema": SCHEMA, "game": self.spec.game, "strategy": asdict(selected),
                  "search": asdict(self.spec.search), "planner_seed": self.spec.planner_seed,
                  "train_seeds": list(self.spec.train_seeds()), "signature": digest(self.signature),
                  "provenance": "personal_api" if self.spec.live_api else "scripted_fixture",
                  "selection": "best_final_common_cohort_mean; ties smallest_id; no independent test",
                  "cohort": final["cohort"], "score_round": self.spec.rounds,
                  "completed_train_games": self.counts["train"], "external_pk_games": 0}
        immutable(self.directory / "metrics/frozen_policy.json", policy)
        if len(self.game_ids) != self.spec.train_games:
            raise ResumeConflict("Game identity mismatch")
        records = self.calls.journal._read_records()
        if {r["context"]["game_index"] for r in records.values()} != set(range(1, self.spec.logical_calls+1)):
            raise ResumeConflict("Unexpected logical journal entries")
        return policy