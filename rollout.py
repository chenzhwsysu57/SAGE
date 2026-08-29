from __future__ import annotations

import json
from typing import Any, Callable

from base.interfaces import BaseGame, BaseMethod, BaseModel
from base.types import GameResult, RunLog, Trajectory, TrajectoryStep


class RolloutRunner:
    """Runs one complete game rollout with full trajectory logging.

    Generic: no game-specific knowledge.
    Verbose output uses game.get_state_summary() for game-specific details.
    """

    def __init__(
        self,
        game: BaseGame,
        methods: dict[int, BaseMethod],
        model: BaseModel,
        verbose: bool = False,
        before_act_hook: Callable[[dict[str, Any]], None] | None = None,
        after_step_hook: Callable[[dict[str, Any]], None] | None = None,
        after_game_hook: Callable[[Trajectory], None] | None = None,
    ):
        self.game = game
        self.methods = methods
        self.model = model
        self.verbose = verbose
        self.before_act_hook = before_act_hook
        self.after_step_hook = after_step_hook
        self.after_game_hook = after_game_hook

    def _v(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")

    def run(self) -> Trajectory:
        state = self.game.reset()
        steps: list[TrajectoryStep] = []
        step_num = 0
        rules_text = ""
        static_state = self.game.get_static_state(state)

        if self.verbose:
            players = ", ".join(
                f"P{p} ({m.name})" for p, m in sorted(self.methods.items())
            )
            print(f"\n  ── New game [{players}] ──")
            summary = self.game.get_state_summary(state)
            if summary:
                print(f"    {summary}")

        while not self.game.is_terminal(state):
            step_num += 1
            player = self.game.get_current_player(state)
            method = self.methods[player]
            observation = self.game.get_observation(state, player)
            if not rules_text:
                rules_text = observation.rules_text
            legal_actions = self.game.get_legal_actions(state)

            state_before_raw = self.game.state_to_dict(state)
            state_before = {k: v for k, v in state_before_raw.items()
                           if k not in static_state}

            self._v(
                f"Step {step_num}: P{player} ({method.name}) to act — "
                f"{self.game.get_state_summary(state)}"
            )

            if self.before_act_hook is not None:
                self.before_act_hook(
                    {
                        "game": self.game,
                        "state": state,
                        "state_before_raw": state_before_raw,
                        "state_before": state_before,
                        "observation": observation,
                        "legal_actions": legal_actions,
                        "step_num": step_num,
                        "player": player,
                        "method": method,
                        "static_state": static_state,
                    }
                )

            result = method.act(observation, legal_actions, self.model)
            action, prompt, raw_response = result[0], result[1], result[2]
            method_data = result[3] if len(result) > 3 else None

            # Apply action, retry on validation error
            action_valid = True
            retries = 0
            rejection_errors: list[str] = []
            retry_attempts: list[dict] = []
            while True:
                try:
                    state = self.game.step(state, action)
                    break
                except ValueError as e:
                    retries += 1
                    err_msg = str(e)
                    rejection_errors.append(err_msg)
                    retry_attempts.append({
                        "attempt": retries,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parsed_action": action.model_dump() if action else None,
                        "error": err_msg,
                    })
                    print(f"    ⚠ {err_msg}")
                    if retries >= 3:
                        raise
                    retry_obs = observation.model_copy()
                    retry_obs.state_text = (
                        f"⚠ YOUR LAST ACTION WAS REJECTED: {err_msg}\n\n"
                        + observation.state_text
                    )
                    retry_result = method.act(retry_obs, legal_actions, self.model)
                    action, prompt, raw_response = retry_result[0], retry_result[1], retry_result[2]
                    method_data = retry_result[3] if len(retry_result) > 3 else method_data
                    action_valid = False

            self._v(f"  → {action}")

            state_after_raw = self.game.state_to_dict(state)
            state_after = {k: v for k, v in state_after_raw.items()
                          if k not in static_state}

            reward = self.game.get_rewards(state).get(player, 0.0)

            # Strip rules_text and full_history_text from stored observation
            # (they're in Trajectory.rules_text and are large)
            stored_obs = observation.model_copy()
            stored_obs.rules_text = ""
            stored_obs.full_history_text = ""
            stored_obs.role_guidance = ""

            steps.append(
                TrajectoryStep(
                    step=step_num,
                    player_id=player,
                    observation=stored_obs,
                    rendered_prompt=prompt,
                    raw_llm_response=raw_response,
                    parsed_action=action,
                    action_valid=action_valid,
                    retry_count=retries,
                    rejection_errors=rejection_errors,
                    retry_attempts=retry_attempts,
                    state_before=state_before,
                    state_after=state_after,
                    reward=reward,
                    method_data=method_data,
                )
            )

            if self.after_step_hook is not None:
                self.after_step_hook(
                    {
                        "game": self.game,
                        "state": state,
                        "state_before_raw": state_before_raw,
                        "state_before": state_before,
                        "state_after_raw": state_after_raw,
                        "state_after": state_after,
                        "observation": observation,
                        "legal_actions": legal_actions,
                        "step_num": step_num,
                        "player": player,
                        "method": method,
                        "trajectory_step": steps[-1],
                        "static_state": static_state,
                    }
                )

        payout = self.game.get_rewards(state)
        net_profit = {p: self.game.get_net_profit(state, p) for p in payout}
        max_r = max(payout.values()) if payout else None
        winners = [p for p, r in payout.items() if r == max_r]
        winner = winners[0] if len(winners) == 1 else None

        if self.verbose:
            print(f"  ── Game Over ──")
            summary = self.game.get_state_summary(state)
            if summary:
                print(f"    {summary}")
            for p_id in sorted(payout):
                chips = payout.get(p_id, 0)
                sign = "+" if chips > 0 else ""
                print(f"    P{p_id} ({self.methods[p_id].name}): {sign}{chips}")
            print()

        trajectory = Trajectory(
            game_name=self.game.__class__.__name__,
            players=[m.name for m in self.methods.values()],
            player_methods={p: m.method_type for p, m in self.methods.items()},
            rules_text=rules_text,
            static_state=static_state,
            steps=steps,
            payout={k: v for k, v in payout.items()},
            net_profit=net_profit,
            winner=winner,
        )
        if self.after_game_hook is not None:
            self.after_game_hook(trajectory)
        return trajectory


def save_trajectory(trajectory: Trajectory, path: str) -> None:
    with open(path, "w") as f:
        json.dump(trajectory.model_dump(), f, indent=2, default=str, ensure_ascii=False)


def load_trajectory(path: str) -> Trajectory:
    with open(path) as f:
        data = json.load(f)
    return Trajectory.model_validate(data)


def save_run_log(run_log: RunLog, path: str) -> None:
    with open(path, "w") as f:
        json.dump(run_log.model_dump(), f, indent=2, default=str, ensure_ascii=False)


def load_run_log(path: str) -> RunLog:
    with open(path) as f:
        data = json.load(f)
    return RunLog.model_validate(data)
