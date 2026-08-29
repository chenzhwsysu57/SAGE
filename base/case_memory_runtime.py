"""Runtime helpers for on-demand case-based memory refresh."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from analysis.memory_bank_builder import (
    infer_action_consistency,
    infer_suggested_action,
    quality_flags as memory_quality_flags,
)
from analysis.memory_case_reflection_batch import extract_memory
from analysis.memory_case_reflection_debug import build_prompt
from analysis.memory_retrieval_opportunity import (
    action_summary,
    extract_features,
    history_with_states,
    observable_similarity,
    opponent_estimate_summary,
    outcome_summary,
    public_prefix_value,
)
from base.case_memory import (
    CASE_MEMORY_ACTIVE_IDS_KEY,
    CASE_MEMORY_BANK_KEY,
    CASE_MEMORY_MIN_STEP,
    CASE_MEMORY_MIN_TOP1,
    CASE_MEMORY_MIN_TOP3,
    CASE_MEMORY_MIN_TOP_GAP,
    CASE_MEMORY_REFRESH_KEYS,
    CASE_MEMORY_REQUIRED_TOP_K,
)
from base.memory import BaseMemory
from base.types import Trajectory
from runs.artifacts import write_json


def _memory_get(memory: BaseMemory | None, key: str, default: Any = None) -> Any:
    if memory is None:
        return default
    getter = getattr(memory, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _memory_set(memory: BaseMemory | None, key: str, value: Any) -> None:
    if memory is None:
        return
    setter = getattr(memory, "set", None)
    if callable(setter):
        setter(key, value)


def _case_labels(legal_actions: list[Any]) -> set[str]:
    labels: set[str] = set()
    for action in legal_actions:
        if isinstance(action, str):
            labels.add(action.lower())
            continue
        if isinstance(action, dict):
            action_type = str(action.get("type") or action.get("action") or "").lower()
            if action_type:
                labels.add(action_type)
            amount = action.get("amount")
            if amount is not None:
                labels.add(f"bid_{int(amount)}")
                labels.add(f"bid {int(amount)}")
            data = action.get("data")
            if isinstance(data, dict):
                quantity = data.get("quantity")
                face = data.get("face")
                if quantity is not None and face is not None:
                    labels.add(f"bid {quantity}x{face}")
                    labels.add(f"bid_{quantity}x{face}")
    return labels


def _top3_gradient_ok(similarities: list[float]) -> bool:
    if len(similarities) < CASE_MEMORY_REQUIRED_TOP_K:
        return False
    top1 = float(similarities[0])
    top3 = float(similarities[CASE_MEMORY_REQUIRED_TOP_K - 1])
    if top3 >= 0.999999:
        return False
    if top1 < CASE_MEMORY_MIN_TOP1:
        return False
    if top3 < CASE_MEMORY_MIN_TOP3:
        return False
    if top1 - top3 < CASE_MEMORY_MIN_TOP_GAP:
        return False
    return True


def _refresh_key(item: dict[str, Any]) -> tuple[Any, ...]:
    context = item.get("context") or {}
    case = item.get("case") or {}
    return (
        context.get("source_game"),
        context.get("source_position"),
        context.get("source_condition"),
        context.get("current_game_id"),
        context.get("current_step"),
        case.get("game_id"),
        case.get("step"),
    )


def _archive_key(
    *,
    game: str,
    position: int,
    current_game_id: int,
    current_step: int,
) -> tuple[Any, ...]:
    return (game, position, current_game_id, current_step)


def _case_step(past: dict[str, Any]) -> int | None:
    context = past.get("context") or {}
    case = past.get("case") or {}
    step = context.get("current_step")
    if step is None:
        step = case.get("step")
    return int(step) if step is not None else None


def _case_memory_id(*, game: str, position: int, past: dict[str, Any]) -> str:
    """Stable memory id for a case: the game/step of the case itself."""
    context = past.get("context") or {}
    case = past.get("case") or {}
    game_id = context.get("current_game_id")
    if game_id is None:
        game_id = case.get("game_id")
    step = _case_step(past) or 0
    return f"{int(game_id):03d}_{position}_step{int(step):03d}"


def _match_top_cases(
    *,
    game: str,
    position: int,
    current_feature: dict[str, Any],
    archive: list[dict[str, Any]],
    min_step: int = CASE_MEMORY_MIN_STEP,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Return top-3 similar completed cases (step >= min_step) and their scores.

    This single result drives both the generation chain and the render chain,
    so the two chains are identical. An empty result means the decision point
    does not qualify: too few candidates or the similarity gradient is unmet.
    """
    candidates = [
        past
        for past in archive
        if isinstance(past, dict)
        and past.get("game") == game
        and past.get("position") == position
        and past.get("feature")
        and past.get("case")
        and (_case_step(past) or 0) >= min_step
    ]
    if len(candidates) < CASE_MEMORY_REQUIRED_TOP_K:
        return [], []
    scored = sorted(
        (
            (observable_similarity(game, current_feature, past["feature"]), past)
            for past in candidates
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    scores = [score for score, _ in scored]
    if not _top3_gradient_ok(scores):
        return [], []
    return (
        scores[:CASE_MEMORY_REQUIRED_TOP_K],
        [past for _, past in scored[:CASE_MEMORY_REQUIRED_TOP_K]],
    )


def _trajectory_steps_dict(trajectory: Trajectory) -> list[dict[str, Any]]:
    return [step.model_dump() for step in trajectory.steps]


def _step_feature_step_dict(
    *,
    player_id: int,
    state_before: dict[str, Any],
    legal_actions: list[Any],
) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "state_before": state_before,
        "observation": {
            "legal_actions": [
                action.model_dump() if hasattr(action, "model_dump") else action
                for action in legal_actions
            ]
        },
    }


def append_trajectory_cases_to_archive(
    *,
    game: str,
    trajectory: Trajectory,
    position: int,
    current_game_id: int,
    archive: list[dict[str, Any]],
    condition: str | None = None,
    trajectory_dir: str | None = None,
) -> int:
    """Finalize completed case snapshots from a finished trajectory into archive."""
    steps = _trajectory_steps_dict(trajectory)
    appended = 0
    for index, step in enumerate(steps):
        if int(step.get("player_id", -1)) != position:
            continue
        current_case = build_current_case(
            game=game,
            trajectory=trajectory,
            step_index=index,
            position=position,
            current_game_id=current_game_id,
            condition=condition,
            trajectory_dir=trajectory_dir,
        )
        key = _archive_key(
            game=game,
            position=position,
            current_game_id=current_game_id,
            current_step=int(step.get("step") or 0),
        )
        updated = False
        for item in archive:
            if not isinstance(item, dict):
                continue
            if item.get("_archive_key") != key:
                continue
            item.update(current_case)
            updated = True
            break
        if not updated:
            current_case["_archive_key"] = key
            archive.append(current_case)
        appended += 1
    return appended


def append_step_case_to_archive(
    *,
    game: str,
    state_before: dict[str, Any],
    legal_actions: list[Any],
    position: int,
    current_game_id: int,
    current_step: int,
    archive: list[dict[str, Any]],
    condition: str | None = None,
) -> int:
    """Append a lightweight case snapshot for the just-finished decision point."""
    current_feature = extract_features(
        game,
        _step_feature_step_dict(
            player_id=position,
            state_before=state_before,
            legal_actions=legal_actions,
        ),
        position,
    )
    key = _archive_key(
        game=game,
        position=position,
        current_game_id=current_game_id,
        current_step=current_step,
    )
    for item in archive:
        if not isinstance(item, dict):
            continue
        if item.get("_archive_key") != key:
            continue
        item["feature"] = current_feature
        item["context"] = {
            "source_game": game,
            "source_condition": condition,
            "source_position": f"P{position}",
            "source_trajectory_dir": None,
            "current_game_id": current_game_id,
            "current_step": current_step,
        }
        return 0

    archive.append(
        {
            "_archive_key": key,
            "game": game,
            "position": position,
            "condition": condition,
            "context": {
                "source_game": game,
                "source_condition": condition,
                "source_position": f"P{position}",
                "source_trajectory_dir": None,
                "current_game_id": current_game_id,
                "current_step": current_step,
            },
            "feature": current_feature,
        }
    )
    return 1


def _select_archive_case(
    *,
    game: str,
    current_feature: dict[str, Any],
    archive: list[dict[str, Any]],
    position: int,
) -> tuple[list[float], dict[str, Any] | None]:
    scores = [
        observable_similarity(game, current_feature, past["feature"])
        for past in archive
        if past.get("game") == game and past.get("position") == position and past.get("feature")
    ]
    scores.sort(reverse=True)
    selected_case = None
    if scores and archive:
        ranked = [
            past
            for past in archive
            if past.get("game") == game
            and past.get("position") == position
            and past.get("feature")
            and past.get("case")
        ]
        ranked.sort(
            key=lambda past: observable_similarity(game, current_feature, past["feature"]),
            reverse=True,
        )
        selected_case = ranked[0] if ranked else None
    return scores, selected_case


def build_current_case(
    *,
    game: str,
    trajectory: Trajectory,
    step_index: int,
    position: int,
    current_game_id: int,
    condition: str | None = None,
    trajectory_dir: str | None = None,
) -> dict[str, Any]:
    steps = _trajectory_steps_dict(trajectory)
    step = steps[step_index]
    current_feature = extract_features(game, step, position)
    case = {
        "public_prefix": {
            "similarity": 1.0,
            **public_prefix_value(game, current_feature),
        },
        "opponent_estimate": opponent_estimate_summary(step),
        "game_id": current_game_id,
        "step": step.get("step"),
        "history": history_with_states(game, steps, position, before_step=int(step.get("step") or 0)),
        "legal_actions": list(step.get("observation", {}).get("legal_actions") or []),
        "actioned": action_summary(step.get("parsed_action")),
        "future_history": history_with_states(game, steps, position, after_step=int(step.get("step") or 0)),
        "outcome": outcome_summary(game, steps, position),
    }
    return {
        "context": {
            "source_game": game,
            "source_condition": condition,
            "source_position": f"P{position}",
            "source_trajectory_dir": trajectory_dir,
            "current_game_id": current_game_id,
            "current_step": step.get("step"),
        },
        "case": case,
        "feature": current_feature,
    }


def refresh_case_memory_bank_from_trajectory(
    *,
    game: str,
    trajectory: Trajectory,
    position: int,
    current_game_id: int,
    memory: BaseMemory | None,
    archive: list[dict[str, Any]],
    run_dir: Path | None = None,
    condition: str | None = None,
    trajectory_dir: str | None = None,
    live_api: bool = False,
    model_name: str = "deepseek-v4-flash",
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 1,
    parallel: bool = False,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Generate and install self-evolve memories for cases that clear the trigger."""
    if memory is None:
        return {"selected": 0, "accepted": 0, "triggered": 0}

    refreshed_keys = {
        tuple(item) if isinstance(item, (list, tuple)) else (item,)
        for item in (_memory_get(memory, CASE_MEMORY_REFRESH_KEYS, []) or [])
    }

    steps = _trajectory_steps_dict(trajectory)
    trajectory_cases: list[dict[str, Any]] = []
    current_cases: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if int(step.get("player_id", -1)) != position:
            continue
        if int(step.get("step") or 0) < CASE_MEMORY_MIN_STEP:
            continue
        current_case = build_current_case(
            game=game,
            trajectory=trajectory,
            step_index=index,
            position=position,
            current_game_id=current_game_id,
            condition=condition,
            trajectory_dir=trajectory_dir,
        )
        trajectory_cases.append(current_case)
        scores = [
            observable_similarity(game, current_case["feature"], past["feature"])
            for past in archive
            if past.get("game") == game
            and past.get("position") == position
            and (_case_step(past) or 0) >= CASE_MEMORY_MIN_STEP
        ]
        scores.sort(reverse=True)
        if not _top3_gradient_ok(scores):
            continue
        refresh_key = _refresh_key(current_case)
        if refresh_key in refreshed_keys:
            continue
        current_case["retrieval_scores"] = scores[:CASE_MEMORY_REQUIRED_TOP_K]
        current_case["case"]["public_prefix"]["similarity"] = scores[0]
        current_case["refresh_key"] = refresh_key
        current_cases.append(current_case)

    if not current_cases:
        archive.extend(
            {
                "game": game,
                "position": position,
                "feature": build_current_case(
                    game=game,
                    trajectory=trajectory,
                    step_index=index,
                    position=position,
                    current_game_id=current_game_id,
                    condition=condition,
                    trajectory_dir=trajectory_dir,
                )["feature"],
            }
            for index, step in enumerate(steps)
            if int(step.get("player_id", -1)) == position
        )
        return {"selected": 0, "accepted": 0, "triggered": 0}

    def _generate(item: dict[str, Any], idx: int) -> dict[str, Any]:
        prompt = build_prompt(item)
        item_id = (
            f"{current_game_id:03d}_{position}_step{int(item['case'].get('step') or 0):03d}"
        )
        if run_dir is not None:
            item_dir = run_dir / "memory_refresh" / f"{game}" / f"P{position}" / item_id
            item_dir.mkdir(parents=True, exist_ok=True)
            write_json(item_dir / "selected_case.json", item)
            (item_dir / "prompt.txt").write_text(prompt)
        response = ""
        memory_result = {"status": "dry_run", "memory": None}
        if live_api:
            from apis.openai_wrapper import OpenAIModel

            model = OpenAIModel(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            response = model.generate(prompt, max_retries=max_retries)
            memory_result = extract_memory(response)
            if run_dir is not None:
                (item_dir / "response.txt").write_text(response)
        if run_dir is not None:
            write_json(item_dir / "memory.json", memory_result)
        legal_actions = item["case"].get("legal_actions") or []
        memory_text = str(memory_result.get("memory") or "").strip() or None
        suggested_action = infer_suggested_action(memory_text or "", legal_actions) if memory_text else None
        bank_item = {
            "memory_id": item_id,
            "game": game,
            "position": f"P{position}",
            "source_condition": condition,
            "public_prefix": item["case"].get("public_prefix"),
            "opponent_estimate": item["case"].get("opponent_estimate"),
            "opponent_action_consistency": infer_action_consistency(memory_text or ""),
            "suggested_action": suggested_action,
            "actioned": item["case"].get("actioned"),
            "legal_actions": legal_actions,
            "memory_status": memory_result.get("status"),
            "memory": memory_text,
            "quality_flags": memory_quality_flags(
                memory_text or "",
                suggested_action,
                item["case"].get("actioned"),
                legal_actions,
            ) if memory_text else ["missing_memory_text"],
            "source_case": {
                "batch_item_id": item_id,
                "current_game_id": current_game_id,
                "current_step": item["case"].get("step"),
                "case_game_id": current_game_id,
                "case_step": item["case"].get("step"),
                "similarity": item.get("retrieval_scores", [0.0])[0] if item.get("retrieval_scores") else 0.0,
            },
        }
        return bank_item

    generated: list[dict[str, Any]] = []
    if parallel and live_api and len(current_cases) > 1:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(_generate, item, idx): idx
                for idx, item in enumerate(current_cases, start=1)
            }
            for future in as_completed(futures):
                generated.append(future.result())
        generated.sort(key=lambda item: item["memory_id"])
    else:
        for idx, item in enumerate(current_cases, start=1):
            generated.append(_generate(item, idx))

    bank = list(_memory_get(memory, CASE_MEMORY_BANK_KEY, []))
    if not isinstance(bank, list):
        bank = []
    accepted = [item for item in generated if item.get("memory_status") == "memory" and item.get("memory")]
    bank.extend(accepted)
    _memory_set(memory, CASE_MEMORY_BANK_KEY, bank)
    refreshed_keys.update(tuple(item.get("refresh_key") or ()) for item in current_cases)
    _memory_set(memory, CASE_MEMORY_REFRESH_KEYS, [list(item) for item in sorted(refreshed_keys)])
    archive.extend(
        {
            "game": game,
            "position": position,
            "feature": item["feature"],
        }
        for item in trajectory_cases
    )
    return {
        "selected": len(current_cases),
        "accepted": len(accepted),
        "triggered": len(current_cases),
        "refreshed_keys": len(refreshed_keys),
    }


def refresh_case_memory_bank_for_step(
    *,
    game: str,
    state_before: dict[str, Any],
    legal_actions: list[Any],
    position: int,
    current_game_id: int,
    current_step: int,
    memory: BaseMemory | None,
    archive: list[dict[str, Any]],
    run_dir: Path | None = None,
    condition: str | None = None,
    live_api: bool = False,
    model_name: str = "deepseek-v4-flash",
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 1,
    parallel: bool = False,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Unified case-memory chain for one decision point (step >= 3 only).

    When the current decision point has top-3 similar completed cases in the
    archive, generate a memory for each of those cases that does not already
    have one, then record the matched case ids in memory so the prompt renders
    exactly those memories. Step-1 decision points never participate in the
    similarity search.
    """
    if memory is None:
        return {"selected": 0, "accepted": 0, "triggered": 0}

    active_ids: list[str] = []
    matched: list[dict[str, Any]] = []
    generated = 0
    accepted = 0

    if int(current_step or 0) >= CASE_MEMORY_MIN_STEP:
        current_feature = extract_features(
            game,
            _step_feature_step_dict(
                player_id=position,
                state_before=state_before,
                legal_actions=legal_actions,
            ),
            position,
        )
        scores, matched = _match_top_cases(
            game=game,
            position=position,
            current_feature=current_feature,
            archive=archive,
            min_step=CASE_MEMORY_MIN_STEP,
        )
        if matched:
            bank = list(_memory_get(memory, CASE_MEMORY_BANK_KEY, []))
            if not isinstance(bank, list):
                bank = []
            existing_ids = {
                item.get("memory_id")
                for item in bank
                if isinstance(item, dict) and item.get("memory_id")
            }
            for rank, past in enumerate(matched):
                case_id = _case_memory_id(game=game, position=position, past=past)
                if case_id in existing_ids:
                    active_ids.append(case_id)
                    continue
                generated += 1
                prompt = build_prompt(past)
                item_dir = None
                if run_dir is not None:
                    item_dir = run_dir / "memory_refresh" / f"{game}" / f"P{position}" / case_id
                    item_dir.mkdir(parents=True, exist_ok=True)
                    write_json(item_dir / "selected_case.json", past)
                    (item_dir / "prompt.txt").write_text(prompt)
                response = ""
                memory_result = {"status": "dry_run", "memory": None}
                if live_api:
                    from apis.openai_wrapper import OpenAIModel

                    model = OpenAIModel(
                        model=model_name,
                        temperature=temperature,
                        api_key=api_key,
                        base_url=base_url,
                        timeout=timeout,
                    )
                    response = model.generate(prompt, max_retries=max_retries)
                    memory_result = extract_memory(response)
                    if item_dir is not None:
                        (item_dir / "response.txt").write_text(response)
                if item_dir is not None:
                    write_json(item_dir / "memory.json", memory_result)

                case = past.get("case") or {}
                legal = case.get("legal_actions") or []
                memory_text = str(memory_result.get("memory") or "").strip() or None
                suggested_action = infer_suggested_action(memory_text or "", legal) if memory_text else None
                bank_item = {
                    "memory_id": case_id,
                    "game": game,
                    "position": f"P{position}",
                    "source_condition": condition,
                    "public_prefix": case.get("public_prefix"),
                    "opponent_estimate": case.get("opponent_estimate"),
                    "opponent_action_consistency": infer_action_consistency(memory_text or ""),
                    "suggested_action": suggested_action,
                    "actioned": case.get("actioned"),
                    "legal_actions": legal,
                    "memory_status": memory_result.get("status"),
                    "memory": memory_text,
                    "quality_flags": memory_quality_flags(
                        memory_text or "",
                        suggested_action,
                        case.get("actioned"),
                        legal,
                    ) if memory_text else ["missing_memory_text"],
                    "source_case": {
                        "batch_item_id": case_id,
                        "current_game_id": (past.get("context") or {}).get("current_game_id"),
                        "current_step": _case_step(past),
                        "case_game_id": case.get("game_id"),
                        "case_step": case.get("step"),
                        "similarity": scores[rank] if rank < len(scores) else 0.0,
                    },
                }
                if bank_item.get("memory_status") == "memory" and bank_item.get("memory"):
                    bank.append(bank_item)
                    accepted += 1
                    active_ids.append(case_id)
            _memory_set(memory, CASE_MEMORY_BANK_KEY, bank)

    _memory_set(memory, CASE_MEMORY_ACTIVE_IDS_KEY, active_ids)
    return {
        "selected": len(matched),
        "generated": generated,
        "accepted": accepted,
        "triggered": 1 if matched else 0,
        "active_ids": active_ids,
    }
