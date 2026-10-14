"""Suspicion-Agent prompt strategy port.

Reference:
  "Suspicion Agent: Playing Imperfect Information Games
   with Theory of Mind Aware GPT-4" (Guo et al., 2023)

3 LLM calls per act():
  1. Pattern (once per game) — opponent behavioral analysis from long memory
  2. Belief (per round) — hand-card distribution + second-order inference
  3. Planning+Action (per round) — EV-based plan selection with JSON output
"""

from __future__ import annotations

from base.interfaces import BaseMethod, BaseModel
from base.types import Action, Observation, Trajectory
from methods.helpers import (
    history_to_text, parse_action_response,
)


class SuspicionPromptMethod(BaseMethod):
    """Suspicion-Agent's second-order ToM prompt pipeline for Leduc.

    Self-contained: no external modeling_method or memory needed.
    Tracks its own short-memory (per game) and long-memory (cross-game).
    """

    def __init__(self, name: str = "SuspicionPrompt", lang: str = "en"):
        self.name = name
        self.method_type = "suspicion"
        self.lang = lang
        # Per-game
        self._short_memory: list[str] = []
        # Cross-game (populated by reflect(), consumed by pattern prompt)
        self._long_memory: list[str] = []
        self._needs_refresh = True
        # Cached LLM outputs for downstream prompts
        self._pattern = ""
        self._belief = ""
        # Debugging (stored for prompt-save in method_data)
        self._dbg_pattern_prompt = ""
        self._dbg_belief_prompt = ""

    # ── Public interface ──────────────────────────────────────────────

    def act(
        self,
        observation: Observation,
        legal_actions: list[Action],
        model: BaseModel,
    ) -> tuple[Action, str, str, dict]:
        opponent = self._opponent_name()

        # --- Step 1: Pattern (once per game) ---
        pattern_prompt = ""
        if self._needs_refresh:
            self._short_memory = []
            pattern_prompt = self._render_pattern_prompt(observation, opponent)
            self._dbg_pattern_prompt = pattern_prompt
            self._pattern = model.generate(pattern_prompt)
            self._needs_refresh = False

        # --- Step 2: Short-memory summary ---
        short_summary = self._render_short_summary(observation)

        # --- Step 3: Belief ---
        belief_prompt = self._render_belief_prompt(
            observation, opponent, short_summary,
        )
        self._dbg_belief_prompt = belief_prompt
        self._belief = model.generate(belief_prompt)

        # --- Step 4: Planning + Action ---
        planning_prompt = self._render_planning_prompt(
            observation, legal_actions, opponent,
        )
        raw_response = model.generate(planning_prompt)
        action = parse_action_response(
            raw_response, observation.player_id, legal_actions,
        )

        # --- Step 5: Store method data ---
        method_data = {
            "suspicion": {
                "pattern_prompt": pattern_prompt,
                "pattern_response": self._pattern,
                "belief_prompt": belief_prompt,
                "belief_response": self._belief,
                "planning_prompt": planning_prompt,
                "planning_response": raw_response,
            },
        }

        return action, planning_prompt, raw_response, method_data

    def reflect(
        self, trajectory: Trajectory, model: BaseModel, game_name: str = "",
    ) -> None:
        """Summarize the finished game and append to long memory."""
        game_history = self._trajectory_to_text(trajectory)
        summary_prompt = self._render_summary_prompt(game_history)
        summary = model.generate(summary_prompt)
        game_num = len(self._long_memory) + 1
        self._long_memory.append(f"{game_num}th Game Start!\n{summary}")
        # Keep window of 20
        if len(self._long_memory) > 20:
            self._long_memory = self._long_memory[-20:]
        self._needs_refresh = True

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _opponent_name() -> str:
        return "Opponent"

    def _render_short_summary(self, observation: Observation) -> str:
        return history_to_text(observation.history, max_entries=15)

    @staticmethod
    def _trajectory_to_text(trajectory: Trajectory) -> str:
        lines = []
        for step in trajectory.steps:
            obs = step.observation
            action = step.parsed_action
            player = step.player_id
            card = obs.private_info.get("card", "?")
            street = obs.public_info.get("street", "?")
            lines.append(
                f"P{player} (card={card}, street={street}): {action}"
            )
        return "\n".join(lines)

    # ── Pattern prompt ────────────────────────────────────────────────

    def _render_pattern_prompt(
        self, observation: Observation, opponent: str,
    ) -> str:
        long_memory_str = (
            "\n\n".join(self._long_memory[-20:])
            if self._long_memory
            else "No previous game memory available yet - this is the first game."
        )
        return (
            f"You are the objective player behind a NPC character called {self.name}, "
            f"and you are playing Leduc Hold'em with {opponent}.\n"
            f" The game rule is: {observation.rules_text}\n"
            f" Your previous game memory including observations, actions "
            f"and conversations with {opponent} is: {long_memory_str}\n"
            f"  {opponent}'s game pattern: Understanding all given information "
            f"and your understanding about the Leduc Hold'em, please infer and "
            f"estimate as many as possible reasonable {opponent}'s game behaviour "
            f"pattern/preferences for each card he holds and each round with "
            f"probability (normalize to number 100% in total for each pattern item) "
            f"and please also infer advantages of his card, and analysis how the "
            f"{opponent}'s behaviour pattern/preferences are influenced by my actions "
            f"when he holds different cards step by step. Output as a tree-structure\n"
            f"Output: When {opponent} holds card1 and the combination of public card "
            f"(if release): if {opponent} is the first to act, he would like to do "
            f"action1 (probabilities), action2 (probabilities) ... continue .. "
            f"If {opponent} sees the action1/action2/action3 of the opponent or not, "
            f"he would like to do action1 (probabilities), action2 (probabilities) "
            f"... continue ... (normalize to number 100% in total), if {opponent} "
            f"sees the action2 of the opponent or not, ... continue ..(more patterns "
            f"with different actions).. in the 1st round, ; If {opponent} sees the "
            f"action1 of the opponent or not, he would like to do action1 (probabilities), "
            f"action2 (probabilities) ... continue... (normalize to number 100% in "
            f"total), ... continue ..(more patterns)..In the 2nd round,;\n"
            f"When {opponent} holds card2 and combination of public card (if release): "
            f"if {opponent} is the first to act, he would like to do action1 "
            f"(probabilities), action2 (probabilities) ... continue .. If {opponent} "
            f"sees the action1 of the opponent or not, he would like to do action1 "
            f"(probabilities), action2 (probabilities) .. continue ... (normalize to "
            f"number 100% in total)...in the 1st round,; .. continue ..(more patterns "
            f"with different actions).in the 2nd round ..\n"
            f" (more patterns with different cards).. continue..\n"
            f"  {opponent}'s guess on my game pattern: Understanding all given "
            f"information, please infer several reasonable believes about my game "
            f"pattern/preference when holding different cards from the perspective "
            f"of {opponent} (please consider the advantages of the card, actions "
            f"and the match with the public card (if release)) for every round of "
            f"the game in detail as a tree-structure output step by step\n"
            f"Output: In the 1st round, When name holds card1 with public card "
            f"(if release), he would like to do (probabilities), action2 "
            f"(probabilities) (normalize to number 100% in total) o ... continue "
            f".. and then do action ...;\n"
            f"When name holds card2 with public card (if release), ...\n"
            f" .. continue..\n"
            f" Strategy Improvement: Understanding the above information, think "
            f"about what strategies I can adopt to exploit the game pattern of "
            f"{opponent} and {opponent}'s guess on my game pattern for winning "
            f"{opponent} in the whole game step by step. (Note that you cannot "
            f"observe the cards of the opponent during the game, but you can "
            f"observe his actions). Output as a tree-structure:\n"
            f"When I hold card and the public card (if release), and see the "
            f"action of the opponent, I would like to do action1; ..."
        )

    # ── Belief prompt ─────────────────────────────────────────────────

    def _render_belief_prompt(
        self,
        observation: Observation,
        opponent: str,
        short_summary: str,
    ) -> str:
        return (
            f"You are the player behind a NPC character called {self.name}, "
            f"and you are playing the board game Leduc Hold'em with {opponent}. \n"
            f" The game rule is: {observation.rules_text}\n"
            f" Your estimated judgement about the behaviour pattern of {opponent} "
            f"and improved strategy is: {self._pattern}\n"
            f" Your observation now is: {observation.state_text}\n"
            f" Your current game progress summarization including actions and "
            f"conversations with {opponent} is: {short_summary}\n"
            f" Understanding the game rule, the cards you have, your observation, "
            f"progress summarization in the current game, the estimated behaviour "
            f"pattern of {opponent}, the potential guess pattern of {opponent} on "
            f"you, and your knowledge about the Leduc Hold'em, can you do following "
            f"things? "
            f" Analysis on my Cards: Understanding all given information and your "
            f"knowledge about the Leduc Hold'em, please analysis what is your best "
            f"combination and advantages of your cards in the current round step "
            f"by step."
            f" Belief on {opponent}'s cards: Understanding all given information, "
            f"please infer the probabilities about the cards of {opponent} "
            f"(normalize to number 100% in total) objectively step by step."
            f"Output: {opponent} saw my history actions (or not) and then did "
            f"action1 (probability) in the 1st round , ... continue..... Before "
            f"this round, {opponent} say my history actions (or not) and did "
            f"action1 (probability), because {opponent}'s behaviour pattern and "
            f"the match with the public card (if release), {opponent} tends to "
            f"have card1 (probability), card2 (probability) ..continue.. (normalize "
            f"to number 100% in total)."
            f" Analysis on {opponent}'s Cards: Understanding all given information "
            f"and your knowledge about the Leduc Hold'em, please analysis what is "
            f"{opponent}'s best combination and advantages of {opponent}'s cards "
            f"in the current round step by step."
            f" Potential {opponent}'s current believes about your cards: "
            f"Understanding all given information and your knowledge about the "
            f"Leduc Hold'em, If you were {opponent} (he can only observe my actions "
            f"but cannot see my cards), please infer the {opponent}'s believes about "
            f"your cards with probability (normalize to number 100% in total) step "
            f"by step. Output: {self.name} did action1 (probability) (after I did "
            f"action or not) in the 1st round, , ... continue... {self.name} did "
            f"action1 (probability) (after I did action or not) in the current "
            f"round,, from the perspective of {opponent}, {self.name} tends to "
            f"have card1 (probability), card2 (probability) ... (normalize to "
            f"number 100% in total) ."
        )

    # ── Planning + Action prompt ──────────────────────────────────────

    def _render_planning_prompt(
        self,
        observation: Observation,
        legal_actions: list[Action],
        opponent: str,
    ) -> str:
        valid_action_list = str([a.type for a in legal_actions])
        return (
            f"You are the objective player behind a NPC character called "
            f"{self.name}, and you are playing the board game Leduc Hold'em "
            f"with {opponent}.\n"
            f" The game rule is: {observation.rules_text}\n"
            f"{self._pattern}\n"
            f" Your observation about the game status now is: "
            f"{observation.state_text}\n"
            f"{self._belief}\n"
            f" Understanding all given information, can you do following things:"
            f" Make Reasonable Plans: Please plan several strategies according "
            f"to actions {valid_action_list} you can play now to win the finally "
            f"whole Leduc Hold'em games step by step. Note that you can say "
            f"something or keep silent to confuse your opponent. "
            f" Potential {opponent}'s actions (if release) and Estimate "
            f"Winning/Lose/Draw Rate for Each Plan: From the perspective of "
            f"{opponent}, please infer what the action {opponent} with probability "
            f"(normalize to number 100% in total) would do when {opponent} holds "
            f"different cards and then calculate the winning/lose/draw rates when "
            f"{opponent} holds different cards step by step. At last, please "
            f"calculate the overall winning/lose/draw rates for each plan step "
            f"by step considering {opponent}'s behaviour pattern. Output in a "
            f"tree-structure:\n"
            f"Output: Plan 1: If I execute plan1.\n"
            f"The winning/lose/draw rates when {opponent} holds card1: Based on "
            f"{opponent}'s behaviour pattern, In the xx round, because {opponent} "
            f"holds card1 (probability) and the combination with current public "
            f"card (if release) (based on my belief on {opponent}), and if he sees "
            f"my action, {opponent} will do action1 (probability) (I actually hold "
            f"card and the public card (if reveal) is, he holds card1 and the public "
            f"card (if reveal), considering Single Game Win/Draw/Lose Rule, please "
            f"infer I will win/draw/lose step by step), action2 (probability) "
            f"(considering Single Game Win/Draw/Lose Rule, please infer I will "
            f"win/draw/lose step by step),.. (normalize to number 100% in total);\n"
            f"   Overall (winning rate for his card1) is (probability = his card "
            f"probability * win action probability), (lose rate for his card2) is "
            f"(probability= his card probability * lose action probability), (draw "
            f"rate for his card2) is (probability = his card probability * draw "
            f"action probability)\n"
            f"The winning/lose/draw rates when {opponent} holds card2: Based on "
            f"{opponent}'s behaviour pattern, In the xx round, because {opponent} "
            f"holds card2 (probability) and the combination with current public "
            f"card (if release) (based on my belief on {opponent}), and if he sees "
            f"my action, he will do action1 (probability) (I actually hold card and "
            f"the public card (if reveal) is, he holds card1 and the public card "
            f"(if reveal), considering Single Game Win/Draw/Lose Rule, please infer "
            f"I will win/draw/lose step by step).. action2 (probability) (normalize "
            f"to number 100% in total) (considering Single Game Win/Draw/Lose Rule, "
            f"please infer I will win/draw/lose step by step),.. ;..... continue "
            f"....\n"
            f" Overall (winning rate for his card2) is (probability = his card "
            f"probability * win action probability), (lose rate for his card2) is "
            f"(probability= his card probability * lose action probability), (draw "
            f"rate for his card2) is (probability = his card probability * draw "
            f"action probability)\n"
            f"...\n"
            f"Plan1 overall {self.name}'s Winning/Lose/Draw rates: the Winning "
            f"rate (probability) for plan 1 is (winning rate for his card1) + "
            f"(winning rate for his card2) + ..; Lose rate (probability) for "
            f"plan 1: (lose rate for his card1) + (lose rate for his card2) + ..; "
            f"Draw Rate (probability) for plan 1: (draw rate for his card1) + "
            f"(draw rate for his card2) + ...; (normalize to number 100% in "
            f"total) for plan1\n"
            f"Plan 2: If I execute plan2, The winning/lose/draw rates when "
            f"{opponent} holds card1: Based on {opponent}'s behaviour pattern, "
            f"In the xx round, if {opponent} holds card1 (probability) and the "
            f"combination with current public card (if release), .. (format is "
            f"similar with before) ... continue ..\n"
            f"Plan 3: .. Coninue ...\n"
            f" The number of payoffs for each plan: Understanding your current "
            f"observation, each new plans, please infer the number of wininng/lose "
            f"payoffs for each plan step by step, Output: Plan1: After the action, "
            f"All chips in the pot: If win, the winning payoff would be (Calculated "
            f"by Winning Payoff Rules step by step): After the action, All chips "
            f"in the pot: If lose, the lose payoff would be: (Calculated by Lose "
            f"Payoff Rules step by step). Plan2: After the action, All chips in "
            f"the pot: If win, the winning chips would be (Calculated by Winning "
            f"Payoff Rules step by step): After the action, All chips in the pot: "
            f"If lose, the lose chips would be: (Calculated by Lose Payoff Rules "
            f"step by step). If the number of my chips in pots have no change, "
            f"please directly output them.\n"
            f" Estimate Expected Chips Gain for Each Plan: Understanding all the "
            f"information and Estimate Winning/Lose/Draw Rate for Each Plan, please "
            f"estimate the overall average Expected Chips Gain for each "
            f"plan/strategy in the current game by calculating winning rate * "
            f"(Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff "
            f"Rule in the game rule) step by step"
            f" Plan Selection: Please output the rank of estimated expected chips "
            f"gains for every plan objectively step by step, and select the "
            f"plan/strategy with the highest estimated expected chips gain "
            f"considering both the strategy improvement.\n"
            f"\n"
            f"After completing the above analysis, output your selected action "
            f"in JSON format:\n"
            f'{{"action": "<fold|check|call|bet|raise>"}}'
        )

    # ── Summarization prompt (reflect) ────────────────────────────────

    @staticmethod
    def _render_summary_prompt(game_history: str) -> str:
        return (
            "You are a player in Leduc Hold'em. Please summarize the following "
            "round-by-round game history in a few sentences. Include the cards held, "
            "actions taken, and final result.\n\n"
            f"Game history:\n{game_history}\n\n"
            "Summary:"
        )
