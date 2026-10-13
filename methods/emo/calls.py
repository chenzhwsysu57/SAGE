"""EMO identities over the unchanged public journal; no global stage mutation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from base.baseline_calls import ModelCallJournal, ResumeConflict
from methods.emo.core import (DECISION_STAGES, UPDATE_STAGES, OutputProtocolError,
                              parse_action, parse_hypothesis)
from base.types import Action


def atomic_json(path: Path, value: Any) -> None:
    """Write only artifacts owned by the enclosing EMO run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".emo-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def immutable_json(path: Path, value: Any) -> None:
    if path.is_symlink():
        raise ResumeConflict("EMO artifact cannot be a symlink")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ResumeConflict("EMO artifact differs on replay")
    else:
        atomic_json(path, value)


class EMOCalls:
    """One monotonically numbered logical-call stream per game and seat.

    The shared journal's `step` is a logical call number, NOT an engine step.
    Existing `decision` is the transport stage; semantic EMO stage/engine step
    and observation event are recorded in the prompt and independent sidecar.
    No stages, defaults or behavior of another method are changed.
    """

    def __init__(self, journal: ModelCallJournal, game: str, player: int, game_index: int,
                 format_retries: int = 3):
        if type(format_retries) is not int or not 0 <= format_retries <= 3:
            raise ValueError("format_retries must be an integer in [0,3]")
        self.journal = journal
        self.game = game
        self.player = player
        self.game_index = game_index
        self.format_retries = format_retries
        self.sequence = 0
        self.logical_calls = 0
        self.format_retry_calls = 0
        self.engine_step = 0
        self.last_stage: str | None = None
        self.last_finish_reason: str | None = None

    @staticmethod
    def _correction_prompt(stage: str, prompt: str, response: str, attempt: int) -> str:
        value = json.loads(prompt)
        if stage == "cot_3":
            instruction = ("The previous response was not exactly one complete JSON object copied from "
                           "legal_actions. Return only one complete legal action object, with identical "
                           "field names, values and types; no Markdown or explanation.")
        else:
            instruction = ("The previous response was not a list of one to three nonempty textual candidate "
                           "descriptions or None. Return only that required list/None; no Markdown or explanation.")
        value["format_correction"] = {"attempt": attempt, "invalid_response": response,
                                      "instruction": instruction,
                                      "scope": "format_only_do_not_reconsider_the_inference_or_action"}
        value["task"] = instruction
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def _physical_call(self, stage: str, prompt: str, format_retry_index: int) -> str:
        if stage not in UPDATE_STAGES + DECISION_STAGES or json.loads(prompt)["stage"] != stage:
            raise ValueError("Invalid EMO call identity")
        self.sequence += 1
        if format_retry_index:
            self.format_retry_calls += 1
        self.last_stage = stage
        self.last_finish_reason = None
        self.journal.set_context(game=self.game, player_id=self.player,
                                 game_index=self.game_index, step=self.sequence)
        name = f"g{self.game_index:04d}_s{self.sequence:04d}_decision.json"
        relative = f"prompts/P{self.player}/{name}"
        identity = {"game": self.game, "player": self.player, "game_index": self.game_index,
                    "engine_step": self.engine_step, "call_index": self.sequence,
                "emo_stage": stage, "format_retry_index": format_retry_index,
                "call_kind": "format_correction" if format_retry_index else "paper_stage",
                "journal_path": relative}
        immutable_json(self.journal.run_dir / "logs" / "emo_calls" / f"P{self.player}" / name, identity)
        bind = getattr(self.journal.model, "bind_physical_context", None)
        if callable(bind):
            bind(identity)
        response = self.journal.generate(prompt)
        self.last_finish_reason = self.journal.last_finish_reason
        valid = False
        try:
            if stage == "init" or "refine" in stage:
                parse_hypothesis(response)
            elif stage == "cot_3":
                legal = [Action(**a) for a in json.loads(prompt)["context"]["legal_actions"]]
                parse_action(response, legal)
            valid = True
        finally:
            immutable_json(self.journal.run_dir / "logs" / "emo_validation" / f"P{self.player}" / name,
                           {**identity, "output_valid": valid,
                            "checked": "hypothesis_list" if stage == "init" or "refine" in stage
                            else "legal_action" if stage == "cot_3" else "nonempty_text"})
        return response

    def call(self, stage: str, prompt: str) -> str:
        """Run one paper stage, with separately identified format-only corrections."""
        self.logical_calls += 1
        current_prompt = prompt
        last_error: OutputProtocolError | None = None
        for attempt in range(self.format_retries + 1):
            try:
                return self._physical_call(stage, current_prompt, attempt)
            except OutputProtocolError as exc:
                last_error = exc
                if attempt >= self.format_retries or not (stage == "init" or "refine" in stage or stage == "cot_3"):
                    raise
                # The journal returns redacted text. Preserve it as data in a new,
                # independently budgeted call without rerunning any earlier stage.
                source = (self.journal.run_dir / "prompts" / f"P{self.player}" /
                          f"g{self.game_index:04d}_s{self.sequence:04d}_decision.json")
                response = json.loads(source.read_text(encoding="utf-8"))["response"]
                current_prompt = self._correction_prompt(stage, prompt, response, attempt + 1)
        raise last_error or OutputProtocolError("Format correction failed")