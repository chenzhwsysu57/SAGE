"""Additive component experiments for LLM game agents.

This entrypoint keeps the older deletion-style ``--ablate`` experiments
unchanged. It expresses each condition as components added to an LLM baseline:

    llm              = LLM only
    llm_nash         = LLM + Nash/CFR strategy
    llm_om           = LLM + opponent modeling
    llm_nash_om      = LLM + Nash/CFR strategy + opponent modeling
    full             = LLM + Nash/CFR strategy + opponent modeling + memory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apis.openai_wrapper import OpenAIModel
from base.ablation import AblationConfig
from methods.liars_dice_cfr import LiarsDiceCFRMethod
from methods.liars_dice_deep_cfr import LiarsDiceDeepCFRMethod
from methods.liars_dice_tom import LiarsDiceToMMethod
from runs.artifacts import git_commit, make_run_dir, serializable_args, write_json
import runs.goofspiel.goofspiel_bench as goofspiel_bench
import runs.leduc.leduc_tom_ablation as leduc_ablation
import runs.liars_dice.ld_tom_ablation as ld_ablation


@dataclass(frozen=True)
class AdditiveCondition:
    key: str
    label: str
    use_nash: bool
    use_opponent_model: bool
    use_memory: bool

    def to_ablation(self) -> AblationConfig:
        return AblationConfig(
            no_nash=not self.use_nash,
            no_belief=not self.use_opponent_model,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "use_nash": self.use_nash,
            "use_opponent_model": self.use_opponent_model,
            "use_qwindow": self.use_opponent_model,
            "use_memory": self.use_memory,
            "ablation_flags": {
                "no_nash": not self.use_nash,
                "no_belief": not self.use_opponent_model,
            },
        }


ADDITIVE_CONDITIONS: dict[str, AdditiveCondition] = {
    "llm": AdditiveCondition(
        "llm", "LLM", use_nash=False, use_opponent_model=False, use_memory=False,
    ),
    "llm_nash": AdditiveCondition(
        "llm_nash", "LLM+Nash", use_nash=True, use_opponent_model=False, use_memory=False,
    ),
    "llm_om": AdditiveCondition(
        "llm_om", "LLM+OM", use_nash=False, use_opponent_model=True, use_memory=False,
    ),
    "llm_nash_om": AdditiveCondition(
        "llm_nash_om", "LLM+Nash+OM", use_nash=True, use_opponent_model=True, use_memory=False,
    ),
    "full": AdditiveCondition(
        "full", "Full", use_nash=True, use_opponent_model=True, use_memory=True,
    ),
}


class FirstLegalMockModel:
    """No-live debug model that returns a parseable first legal action."""

    def generate(self, prompt: str) -> str:
        return json.dumps({
            "reasoning": "mock first legal action for additive dev_debug",
            "action": self._first_legal_action(prompt),
        })

    def _first_legal_action(self, prompt: str) -> dict[str, Any]:
        section = prompt.split("Legal actions:", 1)[-1]
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            action_type = line.split()[0].lower()
            if action_type not in {"fold", "check", "call", "bet", "raise", "bid", "challenge"}:
                continue
            action: dict[str, Any] = {"type": action_type}
            amount_match = re.search(r"amount=(\d+)", line)
            if amount_match:
                action["amount"] = int(amount_match.group(1))
            if "provide:" in line:
                data_text = line.split("provide:", 1)[1].strip()
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError:
                    data = {}
                if isinstance(data, dict):
                    action.update(data)
            return action
        return {"type": "check"}


DEFAULT_CONDITIONS = ["llm", "llm_nash", "llm_om", "llm_nash_om"]
DEFAULT_LEDUC_ARCHETYPES = ["Maniac", "LAG", "GTO", "CallingStation", "Nit", "TAG"]
DEFAULT_LIARS_ARCHETYPES = [
    "Bluffer",
    "Honest",
    "AggressiveChallenger",
    "ConservativeChallenger",
    "Random",
    "GTO",
]
DEFAULT_GOOFSPIEL_PERSONAS = ["Greedy", "Conservative", "Sacrificial", "Random", "GTO", "Mirror"]


def parse_csv(raw: str, *, valid: set[str] | None = None, label: str = "value") -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if valid is not None:
        invalid = [item for item in values if item not in valid]
        if invalid:
            raise ValueError(f"Unknown {label}(s): {invalid}. Valid: {sorted(valid)}")
    return values


def configure_liars_dice_conditions(args: argparse.Namespace) -> None:
    ld_ablation.CONDITIONS["cfr"] = (
        "CFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceCFRMethod(
            name=name,
            strategy_path=args.cfr_model_path,
        ),
    )
    ld_ablation.CONDITIONS["deep_cfr"] = (
        "DeepCFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceDeepCFRMethod(
            name=name,
            model_path=args.deep_cfr_model_path,
        ),
    )
    ld_ablation.CONDITIONS["tom_cfr"] = (
        "ToM+CFR",
        lambda name, memory, modeling_method, verbose, **kw: LiarsDiceToMMethod(
            name=name,
            memory=memory,
            verbose=verbose,
            lang="en",
            strategy_path=args.cfr_model_path,
            signature_bank_path=args.signature_bank_path,
            ablation=kw.get("ablation"),
        ),
    )


def is_dirty_worktree() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return True
    return bool(out.strip())


def run_leduc(args: argparse.Namespace, model: OpenAIModel, run_dir: Path,
              conditions: list[AdditiveCondition]) -> list[dict[str, Any]]:
    archetypes = parse_csv(
        args.leduc_archetypes,
        valid=set(leduc_ablation.ARCHETYPE_NAMES),
        label="Leduc archetype",
    )
    signatures = None
    signature_metadata = None
    if args.leduc_modeling_method == "statistical":
        signatures, signature_metadata = leduc_ablation.build_empirical_signature_bundle(
            args.leduc_signature_manifest,
            max_games_per_archetype=args.leduc_signature_games,
        )

    results = []
    for archetype in archetypes:
        for condition in conditions:
            condition_dir = run_dir / "leduc" / condition.key / archetype
            condition_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            stats = leduc_ablation.run_condition(
                model,
                "tom_cfr",
                archetype,
                args.games,
                args.start_seed,
                verbose=args.verbose,
                save_prompts=args.save_prompts,
                run_dir=condition_dir,
                modeling_method=args.leduc_modeling_method,
                decay=args.leduc_decay,
                signatures=signatures,
                signature_metadata=signature_metadata,
                stat_mode=args.leduc_stat_mode,
                q_window=args.leduc_q_window,
                q_alpha=args.leduc_q_alpha,
                q_tau=args.leduc_q_tau,
                ablation=condition.to_ablation(),
                refresh_memory_bank=condition.use_memory,
            )
            stats.update({
                "game": "leduc",
                "additive_condition": condition.as_dict(),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            results.append(stats)
            print_result("leduc", condition, archetype, stats)
    return results


def run_liars_dice(args: argparse.Namespace, model: OpenAIModel, run_dir: Path,
                   conditions: list[AdditiveCondition]) -> list[dict[str, Any]]:
    configure_liars_dice_conditions(args)
    archetypes = parse_csv(
        args.liars_archetypes,
        valid=set(ld_ablation.ARCHETYPE_NAMES),
        label="Liar's Dice archetype",
    )
    results = []
    for archetype in archetypes:
        for condition in conditions:
            condition_dir = run_dir / "liars_dice" / condition.key / archetype
            condition_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            stats = ld_ablation.run_condition(
                model,
                "tom_cfr",
                archetype,
                args.games,
                args.start_seed,
                args.num_dice,
                args.num_faces,
                args.cfr_model_path,
                verbose=args.verbose,
                save_prompts=args.save_prompts,
                run_dir=condition_dir,
                ablation=condition.to_ablation(),
                refresh_memory_bank=condition.use_memory,
            )
            stats.update({
                "game": "liars_dice",
                "additive_condition": condition.as_dict(),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            results.append(stats)
            print_result("liars_dice", condition, archetype, stats)
    return results


def run_goofspiel(args: argparse.Namespace, model: OpenAIModel, run_dir: Path,
                  conditions: list[AdditiveCondition]) -> list[dict[str, Any]]:
    personas = parse_csv(
        args.goofspiel_personas,
        valid=set(goofspiel_bench.PERSONA_NAMES),
        label="Goofspiel persona",
    )
    results = []
    for persona in personas:
        for condition in conditions:
            condition_dir = run_dir / "goofspiel" / condition.key / persona
            condition_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            stats = goofspiel_bench.run_condition(
                model,
                "tom_bayes",
                persona,
                args.games,
                args.start_seed,
                verbose=args.verbose,
                dry_run=False,
                save_prompts=args.save_prompts,
                run_dir=condition_dir,
                cfr_model=args.goofspiel_cfr_model,
                n=args.goofspiel_n,
                ablation=condition.to_ablation(),
                refresh_memory_bank=condition.use_memory,
            )
            stats.update({
                "game": "goofspiel",
                "additive_condition": condition.as_dict(),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            results.append(stats)
            print_result("goofspiel", condition, persona, stats)
    return results


def print_result(game: str, condition: AdditiveCondition, opponent: str,
                 stats: dict[str, Any]) -> None:
    print(
        f"{game} {condition.label} vs {opponent}: "
        f"wr={stats['winrate']:.0f}% chips={stats['total_chips']:+.0f} "
        f"avg={stats['avg_chips']:+.1f}/game "
        f"W{stats['wins']}/L{stats['losses']}/D{stats['draws']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run additive LLM component experiments.",
    )
    parser.add_argument("--game", choices=["leduc", "liars_dice", "goofspiel", "all"],
                        default="all")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                        help=f"Comma-separated additive conditions. Valid: {list(ADDITIVE_CONDITIONS)}")
    parser.add_argument("--run-phase", default="dev_debug",
                        choices=["dev_debug", "real_measurement"],
                        help="Experiment phase recorded in config; real_measurement requires approval.")
    parser.add_argument("--approved-real-measurement", action="store_true",
                        help="Required when --run-phase real_measurement is used.")
    parser.add_argument("--games", type=int, default=2,
                        help="Games per position per condition.")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--mock-model", action="store_true",
                        help="Use a no-live first-legal-action model for dev_debug only.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save-prompts", action="store_false", dest="save_prompts",
                        default=True)
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument("--leduc-archetypes", default=",".join(DEFAULT_LEDUC_ARCHETYPES))
    parser.add_argument("--leduc-modeling-method", default="statistical",
                        choices=["llm", "statistical"])
    parser.add_argument("--leduc-signature-manifest",
                        default="runs/leduc/manifests/classify_all6_legacy.json")
    parser.add_argument("--leduc-signature-games", type=int, default=0)
    parser.add_argument("--leduc-stat-mode", default="q_window_log",
                        choices=["q_window_ema", "q_window_log", "cumulative"])
    parser.add_argument("--leduc-q-window", type=int, default=10)
    parser.add_argument("--leduc-q-alpha", type=float, default=0.85)
    parser.add_argument("--leduc-q-tau", type=float, default=1.5)
    parser.add_argument("--leduc-decay", type=float, default=0.85)

    parser.add_argument("--liars-archetypes", default=",".join(DEFAULT_LIARS_ARCHETYPES))
    parser.add_argument("--num-dice", type=int, default=2)
    parser.add_argument("--num-faces", type=int, default=3)
    parser.add_argument("--cfr-model-path",
                        default="results/liars_dice/cfr/cfr_2x3_evalevery/model")
    parser.add_argument("--signature-bank-path",
                        default="results/liars_dice/persona_signature_bank/20260624_190658_signature_bank_wild1/metrics/signature_bank.json")
    parser.add_argument("--deep-cfr-model-path",
                        default="results/liars_dice/deep_cfr/20260621_174108_deep_cfr_4x4_10m/model")

    parser.add_argument("--goofspiel-personas", default=",".join(DEFAULT_GOOFSPIEL_PERSONAS))
    parser.add_argument("--goofspiel-n", type=int, default=5)
    parser.add_argument("--goofspiel-cfr-model", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    condition_keys = parse_csv(
        args.conditions,
        valid=set(ADDITIVE_CONDITIONS),
        label="additive condition",
    )
    if args.run_phase == "real_measurement" and not args.approved_real_measurement:
        parser.error("--run-phase real_measurement requires --approved-real-measurement")
    conditions = [ADDITIVE_CONDITIONS[key] for key in condition_keys]
    games = ["leduc", "liars_dice", "goofspiel"] if args.game == "all" else [args.game]

    slug = f"{args.game}_{args.games}g_{'_'.join(condition_keys)}"
    output_dir = args.output_dir or f"results/{'all_games' if args.game == 'all' else args.game}/additive_components"
    run_dir = make_run_dir(output_dir, slug)
    config = {
        "experiment": "additive_components",
        "phase": args.run_phase,
        "ablation_mode": "additive",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "args": serializable_args(args),
        "git_commit": git_commit(),
        "dirty_worktree": is_dirty_worktree(),
        "live_api": not args.mock_model,
        "mock_model": args.mock_model,
        "conditions": [condition.as_dict() for condition in conditions],
    }
    write_json(run_dir / "config.json", config)
    print(f"Run dir: {run_dir}")
    print(f"Games: {', '.join(games)}")
    print(f"Conditions: {', '.join(condition.key for condition in conditions)}")
    print(f"Phase: {args.run_phase}")

    if args.mock_model:
        model = FirstLegalMockModel()
    else:
        model = OpenAIModel(
            model=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    all_results: list[dict[str, Any]] = []
    if "leduc" in games:
        all_results.extend(run_leduc(args, model, run_dir, conditions))
    if "liars_dice" in games:
        all_results.extend(run_liars_dice(args, model, run_dir, conditions))
    if "goofspiel" in games:
        all_results.extend(run_goofspiel(args, model, run_dir, conditions))

    summary = {
        "experiment": "additive_components",
        "phase": args.run_phase,
        "ablation_mode": "additive",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "games_per_position": args.games,
        "start_seed": args.start_seed,
        "conditions": [condition.as_dict() for condition in conditions],
        "live_api": not args.mock_model,
        "mock_model": args.mock_model,
        "results": all_results,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Results saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
