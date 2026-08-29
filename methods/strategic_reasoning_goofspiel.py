from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation
from games.goofspiel import Goofspiel, GoofspielState
from methods.goofspiel_cfr_method import GoofspielCFRMethod
from methods.helpers import build_example, history_to_text, parse_action_response, render_action_descs


@dataclass(frozen=True)
class SearchConfig:
    width: int = 1
    depth: int = 5
    sims: int = 3
    seed: int = 0


class StrategicGoofspielMethod(BaseMethod):
    """LLM strategic reasoning variants for Goofspiel.

    The method keeps all strategic variants in one implementation so the
    benchmark can compare prompt scaffolds while preserving the same parser and
    game loop.
    """

    _VALID_MODES = {"vanilla", "llm_eq", "llm_tot", "llm_pomcts"}

    def __init__(
        self,
        name: str,
        mode: str,
        *,
        cfr_model_path: str | None = None,
        tot_branches: int = 3,
        search_config: SearchConfig | None = None,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown strategic reasoning mode: {mode}")
        self.name = name
        self.method_type = f"strategic_{mode}"
        self.mode = mode
        self.tot_branches = max(1, int(tot_branches))
        self.search_config = search_config or SearchConfig()
        self._cfr = GoofspielCFRMethod(model_path=cfr_model_path) if mode == "llm_eq" else None
        self._search_cfr = GoofspielCFRMethod(
            model_path=cfr_model_path,
            name="POMCTS-CFR-Opponent",
            seed=self.search_config.seed + 7919,
        ) if mode == "llm_pomcts" else None

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple[Action, str, str, dict[str, Any]]:
        method_data: dict[str, Any] = {"mode": self.mode}
        if self.mode == "llm_pomcts":
            method_data["pomcts"] = self._llm_pomcts_plan(observation, legal_actions, model)
        prompt = self._render_prompt(observation, legal_actions, method_data)
        raw_response = model.generate(prompt)
        if self.mode == "llm_pomcts":
            action = self._recommended_action(observation.player_id, legal_actions, method_data)
            method_data["final_selector"] = "mean_simulated_payoff"
            method_data["final_llm_response_used_for"] = "explanation_only"
        else:
            action = parse_action_response(raw_response, observation.player_id, legal_actions)
        return action, prompt, raw_response, method_data

    def _render_prompt(
        self,
        obs: Observation,
        legal_actions: list[Action],
        method_data: dict[str, Any] | None = None,
    ) -> str:
        parts = [
            "#### GAME RULES ####",
            obs.rules_text,
            "",
            "#### GAME STATE ####",
            obs.state_text,
        ]

        hist = history_to_text(obs.history, max_entries=10)
        if hist:
            parts.extend(["", hist])
        if obs.full_history_text:
            parts.extend(["", "#### COMPLETED ROUND HISTORY ####", obs.full_history_text])

        if self.mode == "llm_eq":
            eq_text = self._equilibrium_text(obs, legal_actions)
            parts.extend(["", "#### EQUILIBRIUM STRATEGY ####", eq_text or "No exact equilibrium row is available for this information set."])
        elif self.mode == "llm_tot":
            parts.extend(["", "#### TREE OF THOUGHT DELIBERATION ####", self._tot_instruction()])
        elif self.mode == "llm_pomcts":
            plan = method_data.get("pomcts", {}) if method_data else {}
            parts.extend(["", "#### LLM PARTIALLY OBSERVED MCTS PLANNING TRACE ####", self._format_llm_planning_summary(plan)])

        parts.extend([
            "",
            "#### YOUR MOVE ####",
            self._move_instruction(),
            "Legal actions:",
            "\n".join(render_action_descs(legal_actions)),
            "",
            f"Output as JSON: {json.dumps(build_example(legal_actions))}",
        ])
        return "\n".join(parts)

    def _equilibrium_text(self, obs: Observation, legal_actions: list[Action]) -> str:
        if self._cfr is None:
            return ""
        return self._cfr.format_for_prompt(obs, legal_actions)

    def _tot_instruction(self) -> str:
        return (
            f"Generate exactly {self.tot_branches} distinct candidate thoughts before the final action. "
            "Each thought must name one legal bid, describe the payoff upside, the risk if the opponent counters it, "
            "and when that bid would be dominated. Compare the thoughts, then choose one legal action."
        )

    def _move_instruction(self) -> str:
        if self.mode == "vanilla":
            return "Reason from the visible rules, score, remaining prizes, hand, and completed rounds. Choose one legal bid."
        if self.mode == "llm_eq":
            return "Start from the equilibrium bid distribution, then decide whether the current score and history justify following or deviating from it."
        if self.mode == "llm_tot":
            return "Use the Tree of Thought deliberation format above internally, then return the final legal bid in JSON."
        return "Use the simulated MCTS rollout table. Choose the recommended legal bid exactly unless it is unavailable."

    def _recommended_action(
        self,
        player_id: int,
        legal_actions: list[Action],
        method_data: dict[str, Any],
    ) -> Action:
        legal_by_amount = {a.amount: a for a in legal_actions}
        pomcts = method_data.get("pomcts", {})
        recommended = pomcts.get("recommended_bid") if isinstance(pomcts, dict) else None
        if isinstance(recommended, int) and recommended in legal_by_amount:
            return legal_by_amount[recommended]
        return legal_actions[0] if legal_actions else Action(player_id=player_id, type="bid", amount=0)

    def _llm_pomcts_plan(
        self,
        obs: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> dict[str, Any]:
        cfg = self.search_config
        game = Goofspiel(n=len(obs.public_info.get("prize_sequence", [])) or 5)
        records: list[dict[str, Any]] = []
        simulations: list[dict[str, Any]] = []
        root_returns: dict[int, list[float]] = {}
        effective_depth = self._effective_depth(obs)
        for sim in range(1, max(1, cfg.sims) + 1):
            state = self._reconstruct_state(obs, game)
            trajectory: list[dict[str, Any]] = []
            root_bid: int | None = None
            for depth in range(1, effective_depth + 1):
                if game.is_terminal(state):
                    break
                step_records = self._advance_simulated_round(
                    game,
                    state,
                    obs.player_id,
                    model,
                    sim=sim,
                    depth=depth,
                    root_bid=root_bid,
                    trajectory=trajectory,
                )
                if step_records["root_bid"] is not None:
                    root_bid = step_records["root_bid"]
                state = step_records["state"]
                records.extend(step_records["records"])
                trajectory.extend(step_records["trajectory"])
            payoff = game.get_net_profit(state, obs.player_id) if game.is_terminal(state) else self._score_diff(state, obs.player_id)
            if root_bid is not None:
                root_returns.setdefault(root_bid, []).append(float(payoff))
            simulations.append({
                "sim": sim,
                "root_bid": root_bid,
                "terminal": game.is_terminal(state),
                "payoff": float(payoff),
                "final_state": game.state_to_dict(state),
                "trajectory": trajectory,
            })

        action_values = {
            bid: {
                "count": len(values),
                "mean_payoff": sum(values) / len(values),
                "payoffs": values,
            }
            for bid, values in sorted(root_returns.items())
            if values
        }
        recommended_bid = None
        if action_values:
            recommended_bid = max(action_values.items(), key=lambda kv: (kv[1]["mean_payoff"], kv[1]["count"], -kv[0]))[0]

        return {
            "config": {
                "width": cfg.width,
                "depth": cfg.depth,
                "effective_depth": effective_depth,
                "sims": cfg.sims,
                "opponent_policy": "cfr",
            },
            "recommended_bid": recommended_bid,
            "action_values": action_values,
            "simulations": simulations,
            "planning_records": records,
        }

    def _advance_simulated_round(
        self,
        game: Goofspiel,
        state: GoofspielState,
        method_player: int,
        model: BaseModel,
        *,
        sim: int,
        depth: int,
        root_bid: int | None,
        trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        round_events: list[dict[str, Any]] = []
        s = state
        while not game.is_terminal(s):
            current = game.get_current_player(s)
            legal = game.get_legal_actions(s)
            obs = game.get_observation(s, current)
            if current == method_player:
                prompt = self._render_pomcts_decision_prompt(
                    obs,
                    legal,
                    sim=sim,
                    depth=depth,
                    root_bid=root_bid,
                    trajectory=trajectory + round_events,
                )
                response = model.generate(prompt)
                action = self._safe_parse_sim_action(response, current, legal)
                if root_bid is None:
                    root_bid = action.amount
                records.append({
                    "phase": "path_step",
                    "sim": sim,
                    "depth": depth,
                    "player": current,
                    "root_bid": root_bid,
                    "selected_bid": action.amount,
                    "prompt": prompt,
                    "response": response,
                    "state_before": game.state_to_dict(s),
                })
                actor = "method_llm"
            else:
                action, cfr_prompt, cfr_response = self._cfr_action(obs, legal)
                records.append({
                    "phase": "opponent_cfr",
                    "sim": sim,
                    "depth": depth,
                    "player": current,
                    "root_bid": root_bid,
                    "selected_bid": action.amount,
                    "prompt": cfr_prompt,
                    "response": cfr_response,
                    "state_before": game.state_to_dict(s),
                })
                actor = "opponent_cfr"
            s_next = game.step(s, action)
            event = {
                "depth": depth,
                "player": current,
                "actor": actor,
                "action": action.model_dump(),
                "state_before": game.state_to_dict(s),
                "state_after": game.state_to_dict(s_next),
            }
            round_events.append(event)
            records[-1]["state_after"] = game.state_to_dict(s_next)
            s = s_next
            if s.phase == "bid_p0" or game.is_terminal(s):
                break
        return {
            "state": s,
            "root_bid": root_bid,
            "records": records,
            "trajectory": round_events,
        }

    def _safe_parse_sim_action(self, response: str, player_id: int, legal_actions: list[Action]) -> Action:
        legal_by_amount = {a.amount: a for a in legal_actions}
        try:
            action = parse_action_response(response, player_id, legal_actions)
            if action.amount in legal_by_amount:
                return legal_by_amount[action.amount]
        except Exception:
            pass
        return legal_actions[0] if legal_actions else Action(player_id=player_id, type="bid", amount=0)

    def _cfr_action(self, obs: Observation, legal_actions: list[Action]) -> tuple[Action, str, str]:
        if self._search_cfr is None:
            return legal_actions[0], "cfr_unavailable", f"Fallback first legal action: {legal_actions[0]}"
        return self._search_cfr.act(obs, legal_actions, None)

    def _reconstruct_state(self, obs: Observation, game: Goofspiel) -> GoofspielState:
        n = len(obs.public_info.get("prize_sequence", [])) or game.n
        prize_sequence = tuple(int(x) for x in obs.public_info.get("prize_sequence", range(1, n + 1)))
        rnd = int(obs.public_info.get("round", 1))
        pid = obs.player_id
        my_hand = set(int(x) for x in obs.private_info.get("hand", []))
        used = {0: set(), 1: set()}
        history = []
        for entry in obs.history:
            if entry.get("p0_card") is None:
                continue
            p0 = int(entry["p0_card"])
            p1 = int(entry["p1_card"])
            used[0].add(p0)
            used[1].add(p1)
            history.append(dict(entry))
        hands = {
            pid: set(my_hand),
            1 - pid: set(range(1, n + 1)) - used[1 - pid],
        }
        # The observation's hand is authoritative for the acting player.
        all_cards = set(range(1, n + 1))
        hands[0] = {c for c in hands.get(0, all_cards) if c in all_cards}
        hands[1] = {c for c in hands.get(1, all_cards) if c in all_cards}
        scores_raw = obs.public_info.get("scores", {0: 0.0, 1: 0.0})
        scores = {
            0: float(scores_raw.get(0, scores_raw.get("0", 0.0))),
            1: float(scores_raw.get(1, scores_raw.get("1", 0.0))),
        }
        phase = "bid_p0" if pid == 0 else "bid_p1"
        pending_bid = None
        current_player = pid
        state = GoofspielState(
            n=n,
            round=rnd,
            prize=int(obs.public_info.get("prize", prize_sequence[rnd - 1])),
            prize_sequence=prize_sequence,
            hands=hands,
            scores=scores,
            outcome_history=list(obs.public_info.get("outcome_history", [])),
            pending_bid=pending_bid,
            phase=phase,
            current_player=current_player,
            history=history,
        )
        if pid == 1:
            # P1 acts after a hidden P0 bid. Approximate that hidden bid with the CFR opponent.
            pre_state = state.clone()
            pre_state.phase = "bid_p0"
            pre_state.current_player = 0
            pre_state.pending_bid = None
            p0_obs = game.get_observation(pre_state, 0)
            p0_legal = game.get_legal_actions(pre_state)
            p0_action, _, _ = self._cfr_action(p0_obs, p0_legal)
            state = game.step(pre_state, p0_action)
        return state

    def _score_diff(self, state: GoofspielState, player_id: int) -> float:
        return float(state.scores[player_id] - state.scores[1 - player_id])

    def _effective_depth(self, obs: Observation) -> int:
        cfg_depth = max(1, int(self.search_config.depth))
        prize_sequence = obs.public_info.get("prize_sequence", [])
        current_round = int(obs.public_info.get("round", 1))
        n = len(prize_sequence) if prize_sequence else cfg_depth
        remaining_rounds = max(1, n - current_round + 1)
        return min(cfg_depth, remaining_rounds)

    def _render_pomcts_decision_prompt(
        self,
        obs: Observation,
        legal_actions: list[Action],
        *,
        sim: int,
        depth: int,
        root_bid: int | None,
        trajectory: list[dict[str, Any]],
    ) -> str:
        cfg = self.search_config
        lines = [
            "#### GAME RULES ####",
            obs.rules_text,
            "",
            "#### GAME STATE ####",
            obs.state_text,
        ]
        hist = history_to_text(obs.history, max_entries=10)
        if hist:
            lines.extend(["", hist])
        if trajectory:
            lines.extend(["", "#### SIMULATED TRAJECTORY SO FAR ####", _format_sim_trajectory(trajectory)])
        lines.extend([
            "",
            "#### LLM MCTS SIMULATED DECISION ####",
            f"Search config: width={cfg.width}, depth={cfg.depth}, sims={cfg.sims}, opponent_policy=CFR",
            f"Simulation index: {sim}",
            f"Rollout depth: {depth}",
            f"Root bid already selected for this path: {root_bid if root_bid is not None else 'not selected yet'}",
            "This is a simulated environment state produced by applying prior simulated actions to the real game rules.",
            "Choose exactly one legal bid for your side in this simulated state. The opponent response will be supplied by the CFR game-theory model after your action.",
            "Use the simulation index to diversify plausible root choices and continuations across simulations when more than one legal bid is strategically reasonable.",
            "Legal actions:",
            "\n".join(render_action_descs(legal_actions)),
            "",
            f"Output as JSON: {json.dumps(build_example(legal_actions))}",
        ])
        return "\n".join(lines)

    def _parse_selected_bid(self, response: str, legal_actions: list[Action]) -> int | None:
        legal = sorted(a.amount for a in legal_actions)
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("selected_bid", "root_bid", "bid"):
                bid = parsed.get(key)
                if isinstance(bid, int) and bid in legal:
                    return bid
            action = parsed.get("action")
            if isinstance(action, dict):
                bid = action.get("amount")
                if isinstance(bid, int) and bid in legal:
                    return bid
        for bid in legal:
            if str(bid) in response:
                return bid
        return legal[0] if legal else None

    def _format_llm_planning_summary(self, plan: dict[str, Any]) -> str:
        cfg = plan.get("config", {})
        simulations = plan.get("simulations", [])
        action_values = plan.get("action_values", {})
        recommended = plan.get("recommended_bid")
        lines = [
            f"Search config: width={cfg.get('width')}, depth={cfg.get('depth')}, effective_depth={cfg.get('effective_depth')}, sims={cfg.get('sims')}, opponent_policy={cfg.get('opponent_policy')}",
            f"Recommended bid by mean simulated payoff: {recommended}",
            "Action value table:",
        ]
        for bid, row in action_values.items():
            lines.append(f"  bid {bid}: count={row.get('count')}, mean_payoff={row.get('mean_payoff'):+.2f}, payoffs={row.get('payoffs')}")
        for sim in simulations:
            lines.append("")
            lines.append(
                f"Simulation {sim.get('sim')}: root_bid={sim.get('root_bid')}, "
                f"terminal={sim.get('terminal')}, payoff={sim.get('payoff'):+.2f}"
            )
            for event in sim.get("trajectory", []):
                action = event.get("action", {})
                after = event.get("state_after", {})
                lines.append(
                    f"  depth={event.get('depth')} {event.get('actor')} P{event.get('player')} "
                    f"bid {action.get('amount')} -> round={after.get('round')} "
                    f"phase={after.get('phase')} scores={after.get('scores')}"
                )
        return "\n".join(lines)


def _compact_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated in final decision prompt; full text saved separately]"


def _format_sim_trajectory(trajectory: list[dict[str, Any]]) -> str:
    lines = []
    for event in trajectory[-12:]:
        before = event.get("state_before", {})
        after = event.get("state_after", {})
        action = event.get("action", {})
        lines.append(
            f"depth={event.get('depth')} {event.get('actor')} P{event.get('player')} "
            f"bid {action.get('amount')} | before round={before.get('round')} "
            f"phase={before.get('phase')} scores={before.get('scores')} -> "
            f"after round={after.get('round')} phase={after.get('phase')} scores={after.get('scores')}"
        )
    return "\n".join(lines)
