from __future__ import annotations

import json
import re

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation

# ANSI color codes
DIM = "\033[2m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
BOLD = "\033[1m"
RESET = "\033[0m"


class HumanMethod(BaseMethod):
    """Interactive human player via CLI.

    Shows game state, legal actions, and retries until a valid action is given.
    Color-coded output: yellow for personal info, dim for guidance,
    green for actions, magenta for system events.
    """

    def __init__(self, name: str = "Human"):
        self.name = name
        self.method_type = "human"

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel | None = None,
    ) -> tuple[Action, str, str]:
        legal_by_type = {a.type: a for a in legal_actions}

        print()
        print(f"{BOLD}{'='*50}{RESET}")
        print(f"{BOLD}  {self.name}'s turn (P{observation.player_id}){RESET}")
        print(f"{BOLD}{'='*50}{RESET}")
        print()

        # Colorize state text
        for line in observation.state_text.split("\n"):
            print(self._colorize(line))
        print()

        print(f"  {GREEN}Legal actions (type one of these):{RESET}")
        for a in legal_actions:
            print(f"    {GREEN}{self._format_input_example(a)}{RESET}")
        print()

        while True:
            raw_input = input(f"  [{self.name}] Action: ").strip()

            action_type = ""
            for t in legal_by_type:
                if t in raw_input.lower():
                    action_type = t
                    break

            if not action_type:
                examples = ", ".join(
                    self._format_input_example(a) for a in legal_actions
                )
                print(f"  {YELLOW}Unknown action. Try: {examples}{RESET}")
                continue

            template = legal_by_type[action_type]
            action_data: dict = {}

            if template.data:
                remaining = raw_input[len(action_type):].strip()
                if remaining:
                    parts = self._split_input(remaining)
                    action_data = self._parse_action_data(template.data, parts)
                else:
                    example = self._format_input_example(template)
                    extra = input(f"  [{self.name}] Example: {example}: ").strip()
                    if not extra:
                        print(f"  {YELLOW}Data required. Example: {example}{RESET}")
                        continue
                    parts = self._split_input(extra)
                    action_data = self._parse_action_data(template.data, parts)

            action = Action(
                player_id=observation.player_id,
                type=action_type,
                amount=template.amount,
                data=action_data,
            )
            return action, "", raw_input

    @staticmethod
    def _colorize(line: str) -> str:
        """Apply color based on content category. First match wins."""
        s = line.strip()

        # Phase header
        if s.startswith("Round ") or s.startswith("---"):
            return f"{CYAN}{line}{RESET}"
        # Player identity
        if s.startswith("You are P"):
            return f"{BOLD}{YELLOW}{line}{RESET}"
        # Personal / secret info
        if any(kw in line for kw in (
            "Your role:", "Fellow werewolves", "Current kill target",
            "Save potion:", "Poison potion:", "Past investigations:",
            "Tonight's kill target", "seer_checked", "Last check:",
        )):
            return f"{YELLOW}{line}{RESET}"
        # Strategy guidance
        if any(kw in line for kw in (
            "Your goal:", "You are the only", "You have no special",
            "You and your", "You hold two", "You die only once",
            "The village can only",
        )):
            return f"{DIM}{line}{RESET}"
        # System events
        if any(kw in line for kw in (
            "Died last night:", "No deaths", "Eliminated so far:",
            "Miscommunications", "Interceptions",
        )):
            return f"{MAGENTA}{line}{RESET}"
        # Discussion messages — lines starting with Pn:
        if re.match(r"\s*P\d+:", line):
            return f"{WHITE}{line}{RESET}"
        return line

    @staticmethod
    def _split_input(text: str) -> list[str]:
        """Split input by spaces or commas: '1,2,3' -> ['1','2','3']"""
        return [t.strip() for t in re.split(r"[,\s]+", text) if t.strip()]

    @staticmethod
    def _format_input_example(action: Action) -> str:
        """Format a concrete input example for an action."""
        if not action.data:
            return f"'{action.type}'"
        data_parts = []
        for k, v in action.data.items():
            if isinstance(v, list):
                data_parts.append(", ".join(str(x) for x in v))
            elif isinstance(v, int):
                data_parts.append(str(v))
            else:
                data_parts.append(str(v))
        return f"'{action.type} {' '.join(data_parts)}'"

    @staticmethod
    def _parse_action_data(template: dict, parts: list) -> dict:
        """Map template keys to input values in order."""
        result: dict = {}
        idx = 0
        for key, val in template.items():
            if isinstance(val, list):
                n = len(val)
                result[key] = parts[idx:idx + n]
                idx += n
            elif isinstance(val, int):
                if idx < len(parts):
                    p = parts[idx].strip().upper()
                    result[key] = int(p[1:]) if p.startswith("P") else int(p)
                else:
                    result[key] = val
                idx += 1
            else:
                result[key] = " ".join(parts[idx:]) if idx < len(parts) else val
                idx = len(parts)
        return result
