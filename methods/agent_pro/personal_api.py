"""Agent-Pro personal transport: full physical IO linked to semantic calls.

Reuse the shared single-POST adapter without changing other methods. No retry
here: one ledger intent corresponds to one physical request. Resume validates
physical logs against the logical journal before permitting any new call.
"""
from copy import deepcopy
import json
from pathlib import Path

from apis.deepseek_journal_wrapper import DeepSeekJournalModel
from base.baseline_calls import ResumeConflict


class PersonalAgentProModel(DeepSeekJournalModel):
    def bind_physical_context(self, entry):
        self.call_context = deepcopy(entry)

    def _save(self, path, value):
        value = {**value, "agent_pro_call": deepcopy(getattr(self, "call_context", None))}
        super()._save(path, value)

    def audit_physical(self, run_dir):
        run_dir = Path(run_dir)
        mapping_path = run_dir / "logs/agent_pro_call_ids.json"
        entries = json.loads(mapping_path.read_text())["calls"] if mapping_path.exists() else []
        files = sorted(self.network_dir.glob("request_*.json"))
        if len(files) != len(entries):
            raise ResumeConflict("Personal physical/logical request count mismatch")
        for index, (entry, path) in enumerate(zip(entries, files), 1):
            if path.is_symlink() or path.name != f"request_{index:06d}.json":
                raise ResumeConflict("Invalid physical request path")
            physical = json.loads(path.read_text())
            logical = json.loads((run_dir / "prompts/P0" / f"g{index:04d}_s0000_decision.json").read_text())
            if (physical.get("agent_pro_call") != entry or physical.get("request_number") != index
                    or physical.get("request_json") != self._request_payload(logical["prompt"])
                    or physical.get("url") != self.ENDPOINT):
                raise ResumeConflict("Personal physical request differs from semantic journal")
            if logical["status"] == "success":
                try:
                    body = json.loads(physical["response_body"])
                    choice = body["choices"][0]
                    if (physical["status"] != "success" or physical["http_status"] != 200
                            or choice["message"]["content"] != logical["response"]
                            or choice.get("finish_reason") != logical["finish_reason"]):
                        raise ValueError()
                except (ValueError, KeyError, TypeError, IndexError):
                    raise ResumeConflict("Personal raw response differs from semantic journal") from None