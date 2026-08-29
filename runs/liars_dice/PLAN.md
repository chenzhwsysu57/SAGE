# Liar's Dice 2p 开发计划

## 研究目标

将 Leduc 验证的 **ToM+CFR > 纯 CFR** 范式跨领域复制到 Liar's Dice，验证：
1. ToM 对手建模在 bluff/质疑游戏中的有效性
2. 有偏剥削在全新游戏类型上的泛化性
3. 简化版（CFR 精确解）→ 完整版（Deep CFR）的扩展路径

---

## Phase 0：预验证（可选，1-2天）

目标：确认简化版保留完整 bluff 结构，且对手风格可分离。

- 用 LLM ArchetypeAgent（Bluffer/Honest/Aggressive/Conservative）跑 30 局
- 分析行为指标：bluff 频率、质疑阈值、叫牌起点
- 输出风格可分性报告

**验收标准**：风格间 Cohen's d > 1.0（效果量大）

---

## Phase 1：游戏引擎（3-5天）

### 涉及文件

| 文件 | 内容 |
|------|------|
| `games/liars_dice.py` | 游戏状态机 + 规则文本 + action 空间 |
| `tests/test_liars_dice.py` | 单元测试 |

### 游戏状态

```python
class LiarsDiceState:
    player_dice: dict[int, list[int]]  # 每个玩家的骰子面值
    dice_counts: dict[int, int]        # 每个玩家剩余骰子数
    current_player: int
    last_bid: tuple[int, int] | None   # (quantity, face)，None = 还没开始叫
    phase: str                         # "bidding" | "terminal"
    history: list[dict]
    rewards: dict[int, float] | None
```

### 规则参数

- 简化版：**3 dice × 3 faces**（面值 1-3），1 点不是万能点（简化 CFR 树）
- 完整版：**5 dice × 6 faces**（面值 1-6），1 点万能
- Action 类型：`bid(quantity, face)` / `challenge`
- 质疑失败 → 叫牌方失 1 骰；质疑成功 → 质疑方失 1 骰
- 骰子归零 → 出局，对方获胜

### 验收标准

- 100 局随机 vs 随机对局可正确跑完
- 每一步的 legal actions 正确
- 质疑结果正确
- `pytest tests/test_liars_dice.py -q` 通过

---

## Phase 2：CFR 基线（3-5天）

### 涉及文件

| 文件 | 内容 |
|------|------|
| `games/liars_dice_cfr.py` | CFR+ 训练器（简化版专用） |
| `modeling/liars_dice/cfr_agent.py` | CFR 策略 agent（查表） |
| `tools/liars_dice/` | CFR 工具函数 |
| `tests/test_liars_dice_cfr.py` | CFR 测试 |

### 简化版（3×3）CFR

- InfoSet 大小约 10^3-10^4
- 迭代次数 100k-500k
- 约 1 分钟完成训练（CPU）

### 验收标准

- CFR 策略收敛（exploitability < 0.05 chips/hand）
- CFR agent 随机种子可复现
- `pytest tests/test_liars_dice_cfr.py -q` 通过

---

## Phase 3：对手类型与 ArchetypeAgent（2-3天）

### 涉及文件

| 文件 | 内容 |
|------|------|
| `modeling/liars_dice/archetypes.py` | 风格定义 + ArchetypeAgent |

### 对手风格设计

| 风格 | Bluff 频率 | 质疑阈值 | 叫牌风格 |
|------|-----------|---------|---------|
| **Bluffer** | 高（~60%） | 宽松（怀疑很多） | 持续提高数量，实际持有远低于叫牌 |
| **Honest** | 低（~10%） | 严格 | 按实际骰子叫，很少夸大 |
| **AggressiveChallenger** | 中 | 非常严格（经常质疑） | 正常叫牌但频繁质疑 |
| **ConservativeChallenger** | 中 | 非常宽松（几乎不质疑） | 正常叫牌，很少质疑 |
| **Random** | 随机 | 随机 | 随机均匀选择 |
| **GTO** | CFR 策略 | — | CFR 纳什均衡 |

### 验收标准

- 每个 ArchetypeAgent 的行为特征可独立验证
- Bluffer 的 bluff 频率显著 > Honest（统计检验）

---

## Phase 4：对手建模适配（2-3天）

### 涉及文件

| 文件 | 内容 |
|------|------|
| `modeling/liars_dice/__init__.py` | package init |
| `modeling/liars_dice/llm_belief.py` | LLM 分类 + belief tracking（参考 Leduc 版） |
| `modeling/liars_dice/statistical_profile.py` | 统计 profile（参考 Leduc 版） |
| `modeling/liars_dice/signatures.py` | 风格特征向量 |
| `runs/liars_dice/manifests/` | 历史数据 manifest |

### 改动量

Leduc 的 `modeling/leduc/llm_belief.py` 和 `statistical_profile.py` > 80% 逻辑可复用，只需更换：
- Action types（check/call/bet/raise/fold → bid/challenge）
- 风格分类维度
- Hand strength → dice distribution

### 验收标准

- LLMBeliefMethod 能对 30 局 bluff 型对手收敛到正确分类
- 统计 profile 能在 10 局内识别极端的 bluff/从不质疑风格

---

## Phase 5：实验脚本（2-3天）

### 涉及文件

| 文件 | 内容 |
|------|------|
| `runs/liars_dice/ld_tom_ablation.py` | ToM+CFR vs CFR vs Simple 对比实验 |

### 实验设计

- 8 个条件：`tom_cfr`, `tom_nocfr`, `simple`, `cfr`
- 6 个对手类型（除 GTO 外 5 个 + GTO 自身）
- 参数：seed 42-101（同 Leduc 协议），30 局/位置
- 指标：净筹码、胜率、质疑成功率
- 添加 Liar's Dice 特有指标：bluff 检测率、叫牌偏差

### 验收标准

- 所有条件命生成有效结果
- `config.json` 完整记录命令行参数、git hash、时间戳

---

## Phase 6：完整版 Deep CFR（可选，扩展 7-10天）

### 如果简化版通过，扩展到 5d6

- 用 Deep CFR / Neural CFR + PyTorch
- MLP 3-4 层，~100k 参数
- 单 GPU 训练 4-6 小时
- 结果缓存到 `results/liars_dice/deep_cfr/`

---

## 总时间线

| Phase | 内容 | 预计时间 | 依赖 |
|-------|------|---------|------|
| 0 | 预验证 | 1-2 天 | 无 |
| 1 | 游戏引擎 | 3-5 天 | 无 |
| 2 | CFR 基线 | 3-5 天 | Phase 1 |
| 3 | ArchetypeAgent | 2-3 天 | Phase 1 |
| 4 | 对手建模 | 2-3 天 | Phase 1, 3 |
| 5 | 实验脚本 | 2-3 天 | Phase 2, 3, 4 |
| 6 | Deep CFR（可选） | 7-10 天 | Phase 2 |

**核心链路（6-10 天）**：1 → 2 → 3 → 4 → 5

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 简化版 bluff 结构不够丰富 | 低 | Phase 0 预验证先跑 |
| 风格可分性不够 | 中 | 设计极端 Archetype（如从不质疑 vs 总是质疑） |
| CFR 训练收敛慢 | 低 | 用 3×3 极小版调试，验证后再扩 |
| Deep CFR 训练不稳定 | 中 | Phase 6 为可选，不影响核心结果 |
