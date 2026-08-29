"""Chinese-language version of Leduc Hold'em.

Same game logic as LeducHoldem, but all user-facing text in Chinese.
Action types (fold/check/call/bet/raise) remain English internally for
game logic; Chinese labels are used for display and prompts.
"""

from __future__ import annotations

from typing import Any

from games.leduc import LeducHoldem, LeducState, RANK_ORDER
from base.types import Action, Observation

# Chinese labels for poker terms
CN_ACTIONS = {
    "fold": "弃牌", "check": "过牌", "call": "跟注",
    "bet": "下注", "raise": "加注",
}

CN_PHASE = {"preflop": "翻牌前", "flop": "翻牌后"}

CN_SUIT = {"S": "黑桃", "H": "红桃"}


def _card_cn(card_str: str) -> str:
    """Convert 'KS' → 'K黑桃'."""
    if not card_str or len(card_str) < 2:
        return str(card_str)
    rank = card_str[0]
    suit = card_str[1]
    return f"{rank}{CN_SUIT.get(suit, suit)}"


CN_RULES = """Leduc Hold'em 扑克规则（简化版）：
- 2名玩家，每人先下注1筹码（底注）。
- 牌堆：J黑桃 J红桃 Q黑桃 Q红桃 K黑桃 K红桃（共6张牌）。
- 每人发1张私有牌。
- 第一轮下注（翻牌前）：下注额为2，最多2次加注。
- 然后翻1张公共牌。
- 第二轮下注（翻牌后）：下注额为4，最多2次加注。
- 摊牌：如果私有牌与公共牌相同点数，则为"一对"，否则为"高牌"。
  对子 > 高牌；大对子 > 小对子；大高牌 > 小高牌。
- J < Q < K。
- 注意："下注"是每轮第一次主动投入筹码（不算加注），只有"加注"计入加注次数限制。"""


class LeducHoldemCN(LeducHoldem):
    """Chinese-language Leduc Hold'em.

    Identical game logic to LeducHoldem. Only text output differs.
    """

    def get_rules_text(self) -> str:
        return CN_RULES

    def state_to_text(self, state: LeducState, player_id: int) -> str:
        p = player_id
        opponent = 1 - p
        card_str = str(state.private_cards[p]) if state.private_cards[p] else "?"
        comm_str = _card_cn(str(state.community_card)) if state.community_card else "尚未翻牌"

        parts = [
            f"你的手牌: {_card_cn(card_str)}",
            f"公共牌: {comm_str}",
            f"当前阶段: {CN_PHASE.get('preflop' if state.street == 0 else 'flop')}",
            f"底池: {state.pot} 筹码",
            f"你已下注: {state.total_bets[p]}",
            f"对手已下注: {state.total_bets[opponent]}",
            f"需要跟注: {state.round_bets[opponent] - state.round_bets[p]} 筹码",
        ]

        if state.history:
            parts.append("历史动作:")
            for h in state.history:
                cn_action = CN_ACTIONS.get(h["action"], h["action"])
                amount_str = f" {h['amount']}" if h.get("amount") else ""
                parts.append(f"  P{h['player']} {cn_action}{amount_str}")

        return "\n".join(parts)

    def get_observation(self, state: LeducState, player_id: int) -> Observation:
        p = player_id
        opponent = 1 - p
        legal_actions = self.get_legal_actions(state)
        card = state.private_cards[p]

        public_info: dict[str, Any] = {
            "community_card": str(state.community_card) if state.community_card else None,
            "pot": state.pot,
            "street": "preflop" if state.street == 0 else "flop",
            "my_bet_total": state.total_bets[p],
            "opponent_bet_total": state.total_bets[opponent],
            "my_bet_this_round": state.round_bets[p],
            "opponent_bet_this_round": state.round_bets[opponent],
            "round_raises": state.round_raises,
            "to_call": state.round_bets[opponent] - state.round_bets[p],
        }

        # Append Chinese action labels to state_text for display
        cn_actions_str = " / ".join(
            f"{a}={CN_ACTIONS.get(a, a)}" for a in [act.type for act in legal_actions]
        )
        state_text = self.state_to_text(state, p)
        state_text += f"\n可选动作: {cn_actions_str}"

        return Observation(
            player_id=p,
            private_info={"card": str(card)} if card else {},
            public_info=public_info,
            history=state.history,
            legal_actions=[a.type for a in legal_actions],
            rules_text=self.get_rules_text(),
            state_text=state_text,
        )
