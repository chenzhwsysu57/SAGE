# ReTA：三游戏独立适配（dev_debug）

> **2026-09-12 截断处理更新**：新增 `ReTAConfig.truncation_retries=3`。
> 每个ReTA任务收到明确length后，用新调用ID重试同一状态/任务；不带回长截断正文，末尾要求仅完整JSON。
> 首次及额外3次均截断（累计4次length）后，停止当前搜索，从真实根节点全部合法动作均匀随机选一项。
> 随机源与全局RNG隔离，种子由可见决策上下文及专用namespace确定，重放动作一致。
> `method_data.reta.fallback`记录原因、原actor/node、随机种子、合法集合与索引；`selected_value=null`、
> `root=[]`、`search_discarded=true`，不伪称minimax或模型选择。三个runner摘要单列fallback次数及受影响局。
> schema和length预算分别计数、不互相重置，默认混合序列最多7条方法回复；schema耗尽仍停止。
> 非length错误、鉴权、网络异常、预算耗尽、started和持久化错误不触发兜底；真实LLM对手仍为length停止。
> 原始length日志维持failed/length并保存正文及usage；恢复重放已收到文本，不重复计费或重置次数。
> 这是用户批准的工程协议变化，不是论文新增算法；旧run不得跨源码恢复，不改写旧结果。本次只离线验证。

> **当前协议（2026-09-11）**：`reta_text_final_json_v1`，输入为规则→角色→状态→历史→
> 合法动作→任务上下文/纠正反馈→任务→返回格式的分段文本。允许普通文本分析后接末尾JSON
> 或最终JSON代码块。末尾内容仍严格验证，坏末尾不回退到早先的示例或假设JSON。
> 假想终局评分必须与已知模拟净收益符号一致，否则进入有限方法重试，不静默改分。
> `--no-request-limit` 无请求数量上限行为保留；本次未调用live。以下早期JSON输入/裸JSON
> 限制为历史记录，当前实现与证据见文末最新阶段。

论文：[Duan et al., ReTA: Recursively Thinking Ahead to Improve the Strategic
Reasoning of Large Language Models, NAACL 2024](https://aclanthology.org/2024.naacl-long.123/)，
重点为 §5.2、§5.3、Figure 7。根据论文独立编写，没有复制作者实现。

## 实现与论文的对应

| 论文机制 | 实现 |
|---|---|
| Main Actor 提出 n 个候选 | `ReTAMethod` / `_Search.propose`，一次结构化调用列出至多 n 个不同合法动作 ID |
| Anticipation Actor 假想敌 | 从假想行动者自己的观察提出候选，以击败主玩家为目标 |
| 深度 k 递归 pseudo-gaming | `World.advance` 使用未修改的真实游戏规则推进；k 按单人动作层计数，遇 terminal 提前停止 |
| Reward Actor 的优势评分 | 叶节点由模型输出有限 `[0,1]` 数值；0=败、0.5=中性/平、1=胜 |
| minimax 回传 | 主玩家节点 max，对手节点 min；使用实际行动者，不机械假设永远轮流（骰子新局/扑克换街可能连续行动） |
| §5.3 majority vote | `reward_samples` 次数字评分按论文的数值情况取均值，不对连续分数做离散投票 |
| §5.3 perturbation | 模型改写评估问题，原问题+改写问题各重复评分，最终取均值；状态事实不交给改写器修改 |

最终直接选取 root 回传分数最大的候选；平分按模型原始候选顺序，不额外调一次模型改变结论。
不是 ToT、MCTS、CFR、SAGE 或检索方法；没有训练、跨局记忆、访问次数或 UCB。

### 明确的工程选择与复现边界

- 论文实验包含 Tic-Tac-Toe、Connect-4、Texas Hold'em；这里适配 Leduc、Liar's Dice、Goofspiel，不声称重现原论文实验结果。
- 同一 prompt 批量给出 n 个独立合法候选，代替 n 次独立生成；严格去重，n 不大于合法动作数。排序可能影响平分选步。
- 使用论文正文/Figure 7 的普通数值 minimax；不额外乘不可得的模型动作概率，不宣称实现式(2)的概率加权版本。
- 相同角色/观察/合法集合的候选提议在一次真实决策内复用；叶点评分不复用，以保留重复评分语义。
- 语义扰动不能由 JSON schema 保证语义等价，仍需人工查看；离线 fixture 会原样返回问题，仅验证接口。
- terminal 仍调用 Reward Actor，并提供**假想**净收益，要求按符号给 0/0.5/1。真实统计保留净收益幅度；当前搜索目标是优势/胜负，不是严格最大化预期筹码。

## 三游戏与信息边界

**Leduc**：JJQQKK、每人 ante=1、每街合计最多两次 raise。只读取己方牌、已公开公共牌和公共下注历史；从剩余物理牌无放回均匀采样对手牌及未揭公共牌。不会偷看真实 deck。ReTA 局部规则文本纠正旧规则字符串的“每玩家两次 raise”，不改游戏或其他 prompt。

**Liar's Dice**：每人初始 2 个骰子、3 面、1 wild。当前对手骰子使用独立均匀先验；历史 challenge reveal 与新轮骰子分离。假想 challenge 可揭骰、减少数量、重新掷骰并递归下一轮；每分支复制其 RNG，不影响真实游戏及兄弟分支。

对两种不完全信息游戏，所有样本共享 root 候选，先在每个样本内做 minimax，再平均各候选分数。先验不根据对手行为更新；有限样本/策略融合（strategy fusion）/后续假想私人信息均可能产生偏差，不是 belief-state 最优解或均衡。

**Goofspiel**：n=5、平分 prize、完整 prize sequence 公开。将两座位都还原为本轮 **pre-commit 信息集**，从公开的已完成轮次重建剩余牌。即使真实 P1 时 P0 已提交隐藏出价，也不读该出价；改为评估可能的 `(本方候选, 对手候选)` 对。

- 当前候选只存于 `World.pending`，在对手 prompt 中不显示，也不提前从对手能推断的本方手牌集合删除。
- 对手提议跨本方候选复用；不能因看见本方候选而改变提议。
- 每个候选对都严格按 engine 的 P0→P1 各执行一次来结算，两张牌各删除一次。这不是让真实已出价的 P0 再出第二张牌。
- 对候选对做**纯策略保守 maximin**，不是同时行动博弈的混合策略均衡；候选剪枝不能保证涵盖真正最坏回应。
- depth=2 可完成一对出价；奇数深度可在己方未揭出价后截断，此时叶评分只能估计尚未结算的局面。

方法只接受 `Observation` 和 canonical legal actions，不接受真实 game/state。真实轨迹的全知状态仅供人工调试，禁止回灌模型。

## 使用与预算

独立入口：`runs/reta_bench.py`；直接方法接口：`ReTAMethod(game, player_id, ReTAConfig(...)).act(observation, legal_actions, model)`。

默认：depth=2、width=2、hidden_samples=2、reward_samples=1、perturbations=0、simulation_seed=7；这只是低成本开发配置，不是论文最佳配置。

入口接受 `--game leduc|liars_dice|goofspiel`、`--games`（每座位1–4局，默认2）、`--start-seed`（默认9100）、上述搜索选项、`--max-requests`（默认1000）、`--max-calls-per-decision`（默认500）、`--output-dir`、`--resume`、`--stop-after`、`--retry-failed`。两座位种子为 `start_seed + seat * games + index - 1`。

CLI **只有 ScriptedReTAModel**，无 live/密钥/real_measurement 参数。自定义 `BaseModel` 接口保留供将来获批后接入 provider；本阶段不批准任何 live 请求。Scripted 模型仅测试协议和推进，不能用于方法优劣比较。

树复杂度随 width^depth 增长。每叶评分调用数为 `(1 + perturbations) * reward_samples`，另加 `perturbations` 次改写调用；提前 terminal、合法动作不足、提议缓存会减少调用。总预算/单步预算任一到限即停止，不返回部分树选步，不 silent fallback。

## Artifact 与恢复

默认 `results/<game>/reta_dev/<timestamp>_reta_<games>g_dev/`：

- config：phase、准确配置/种子、git/dirty、源码 hash、模型及预算。
- prompts/P0 或 P1：逐调用 prompt/response、usage、finish_reason、失败或 started 状态。
- trajectories/P0 或 P1：逐局合法动作、可见观察、完整搜索树/分数/模型调用、全知调试状态。
- metrics、summary、logs/invocations：进度、失败、完整局数、每局净收益及每次恢复参数。

不修改公共 journal。存储 stage 使用其既有 `decision`，语义角色在 prompt JSON 的 `actor` 中。journal 的 `step = environment_step * 100000 + call_ordinal`；ordinal 从1开始、每真实决策重置，tree 的 call/node 字段提供递归节点映射。假想节点 ID 不发给模型，避免间接提示候选路径。

恢复严格核对源码与配置：完成局 checksum 检查后跳过；中断局以相同种子重演，已成功调用直接读缓存。明确失败只有 `--retry-failed` 可重试并保留历史；started 不确定调用永不自动重发。文本成功但 JSON 不合法会保留原文并在恢复时再次拒绝，不能用 retry-failed 偷换成合法答案。usage 缺失不估计费用。

## 验收

测试：`tests/test_reta.py`、`tests/test_reta_runner.py`。覆盖非贪心 minimax、深层 main 回归、终局停止、数值schema、非法动作、三游戏双位置、真实隐藏变量不变性、同时出价逐对 engine 等价、骰子 RNG、重复评分/扰动、预算、截断、错误保留与恢复。

本阶段有效性证据只限离线协议/程序行为。需要人工审核实际树和 prompt 后，另行申请小预算 live 验证；不自动开始训练或实测。

## 2026-09-10：独立 GTO 单局 live 入口与结果

以上 offline CLI 描述适用于 `runs/reta_bench.py`。新增的 ReTA 专属入口
`runs/reta_leduc_live.py --game leduc|liars_dice|goofspiel` 每次只运行一个游戏、P0、seed9100、一局。
默认 scripted；校园调用须同时 `--live-api --approved-live-api`。不动分支、不提交、不修改其他方法或共享交接文件。

用户本轮批准三游戏各1局对 GTO，并明确“不传上限”：请求 **不包含 max_tokens**，
由服务端决定输出限制（不是无限）；HTTPX read/write/pool60秒、connect10秒，非总墙钟时限。
每游戏最多60次实际尝试，合计180；在公共journal显式 `max_retries=0`，禁止重试或格式修复。
模型明确返回length时仍保存原文并停止。新配置不恢复旧run。

GTO均为既有本地预训练CFR近似策略，不是精确均衡声明：Leduc21000迭代模型、骰子2x3模型、
Goofspiel n5 continued MCCFR+平均策略。模型hash记录在config；无训练、无对手API调用、缺策略行拒绝fallback。
Goofspiel限制pickle可加载类，保留strategy_sum并仅归一化当前行，使用现有信息集编码，隐藏pending bid不进输入。

验证：241 tests passed（6.03s），MockTransport证明不发送上限且截断/脱敏/零重试保护保留；
新配置三游戏各1局scripted成功（46/48/27调用），各resume零新增调用。
证据：`results/leduc/reta_dev/20260910_215247_three_game_no_cap_prelive_dev/`。

Live依次执行并已结束（未重试/恢复/追加预算）：

| 游戏 | run（均在results/<game>/reta_dev/） | 请求数 | 结果 |
|---|---|---:|---|
| Leduc | 20260910_215346_reta_vs_gto_1g_live_dev | 1 | 首次main HTTP500，停止 |
| Liar's Dice | 20260910_215347_reta_vs_gto_1g_live_dev | 3 | main/anticipation合法；reward HTTP500，停止 |
| Goofspiel | 20260910_215354_reta_vs_gto_1g_live_dev | 2 | main合法；anticipation动作数组中是中文说明，JSONDecodeError停止 |

三局均0完成、0真实动作，GTO尚未执行动作；不能声称live对战通过。共6/180次尝试，4次文本、
2次HTTP500；所有文本finish_reason=stop，无明确length。已知usage输入3071、输出49、总3120；
2个500的usage未知，不估算账单。500不能仅据此归因于不传上限，坏JSON也不是截断。

旧Leduc尝试 `20260910_212851_reta_vs_gto_1g_live_dev` 实际发出1次请求，返回1个候选而非2个，
其失败及原始artifact保留不覆盖；新prompt明确EXACTLY数量，严格parser不放宽。
本批6次不包含该旧尝试。各新run `prompts/P0/*.json` 保存prompt/response，
`logs/reta_validation.json`为审计；汇总在验证目录 `metrics/live_audit.json`。
下一步须审核HTTP错误与输出schema后另批重测；本次失败后未再改算法或调用API。人工验收pending。

## 2026-09-10 22:08：完整网络审计与3次重试复验（最新）

本节取代上一轮入口的“零重试”设置；旧run与旧说明作为历史证据保留。用户要求重跑、
每次请求完整上下文/网络记录、在ReTA内部实现3次重试。仅修改ReTA自有源码与测试/README，
未修改共享API/journal、其他方法或共享交接，未操作分支/暂存区/提交。

### 当前行为

- `runs/reta_leduc_live.py` 启用共享 `ModelCallJournal(max_retries=3)`，不另写第二套重试循环。
	首次+最多3次额外尝试，每次均计入每游戏60、三游戏180预算。仅暂时网络错误/429/5xx重试，
	5/10/20秒退避并遵守Retry-After。鉴权/明确length/普通非法JSON不自动重试。
- `methods/reta/network_audit.py` 的 `AuditedCampusModel` 是ReTA专属wrapper子类；
	只给原HTTPX client装request/response hooks，不改变请求payload、代理、TLS校验、模型输出或游戏决策。
	request hook发送前落盘准备好的请求，response hook在原wrapper抛错前读出正文并落盘。
- `ReTACallJournal` 在每个durable started尝试后绑定global attempt、logical call、actor及node。
	重试同一call产生不同HTTP文件；缓存重放不发请求、不覆盖日志。不允许绕过账本直接调用审计模型。
- `logs/network/request_NNNN.json`：每一次物理尝试的完整prompt、搜索node、URL/method、
	脱敏headers、实际request.body_json/body_text、timeout、proxy、response状态码/headers/
	完整decoded body、finish_reason、usage、时间与安全错误分类；无64KB截断。
- `logs/decisions/step_NNNN.json`：真实决策时本方observation、rules、canonical actions、配置；
	`prompts/P0/`保留共享journal成功/失败历史；`summary.last_request`定位导致停止的最后请求。
- 这是HTTP应用层记录，**不是**TCP/SSH/TLS抓包。正文完整但解码/JSON规范化/脱敏后不保证字节原样；
	Authorization/Cookie/密钥及未知header值隐藏，安全header保留。失败前已有响应头但body读取中断时明确标记。
	读超时重试可能产生服务端重复计费；缺失usage不估计。

### 验证与本次结果

277项测试通过（5.51s），包括70KB错误正文保留、request ID、凭据/Cookie回显脱敏、
首次加3次重试、Retry-After、累计预算、截断/鉴权不重试、传输失败、审计失败不发送、
started不重发、本地JSON错误定位和完整局resume无新增。
证据 `results/leduc/reta_dev/20260910_220639_network_audit_retry_prelive_dev/`；三游戏同配置scripted各1局成功，46/48/27调用。

新live（均在 `results/<game>/reta_dev/`，seed9100/P0/各1局、无max_tokens）：

| 游戏 | run | 物理请求 | 最终停止 |
|---|---|---:|---|
| Leduc | 20260910_220801_reta_vs_gto_1g_live_dev | 9 | #8 HTTP500，等待约5秒后#9同payload返回200，但action_ids含字母O而非数字0，JSONDecodeError |
| 骰子 | 20260910_220815_reta_vs_gto_1g_live_dev | 8 | #8 HTTP200，action_ids使用history_index_of_challenge/history_index_of_bid_4_1占位符，JSONDecodeError |
| Goofspiel | 20260910_220828_reta_vs_gto_1g_live_dev | 2 | #2 HTTP200，action_ids含π，JSONDecodeError |

全部停在首个真实决策的假想树内：0完整局、0真实动作，无GTO对战效果结论。
合计19/180实际请求，18 HTTP200、1 HTTP500；已知usage输入14746、输出198、总14944；
500 usage未知。仅发生1次自动额外重试；三次内容解析失败后无再次发送，原prompt/schema未放宽。
本批500正文为网关返回的 `detail: upstream error ... litellm.InternalServerError ...
OpenAIException - Internal Server Error ... Received Model Group=DeepSeek-V4-Flash ...
Available Model Group Fallbacks=None`。只能说网关报告上游异常，不据此断言具体根因。

19条已准备HTTP请求的model/messages均为合法JSON、不含max_tokens，message.content逐字匹配journal.prompt；
每物理尝试与journal一一对应。Leduc #8/#9请求体相同，第二次成功返回HTTP200，不能将第一次500解释为必然无效的JSON语法。
响应未包含请求追踪header ID（完整响应JSON中的模型completion ID如存在仍保留）。
46既有源码与27旧artifact hash未变。审计在各run `logs/audit_review.json`，汇总在验证run `metrics/live_review.json`。

### 手动重试材料（不自动执行）

每条请求另导出 `logs/replay/request_NNNN.body.json`，内容等于HTTP记录 `request.body_json`，
可直接作为POST body。不要把整个network记录当请求体。URL/method/proxy在对应network记录，
认证需运行时读取校园密钥，Authorization按原值传递，不额外加Bearer。
header中的 `<redacted>` 是占位，不可直接当真实凭据发送。

优先查看Leduc #8（500原文）与#9（重试后的非法内容）、骰子#8、Goofspiel#2。
手动重发会另计费，请把新响应保存在新诊断目录，不覆盖原网络/journal文件。
当前入口resume不会自动重发被成功缓存的坏JSON；它只重放原文然后再次报错，因此不要用resume期待修复答案。
本轮已停止，不再自动调用API，等待用户自行复验/审核。

## 2026-09-11：数字动作ID与合法JSON示例修正（仅离线）

用户同意修正输出提示词。本轮仅修改ReTA `core.py` 的候选/评分提示词、增加
`tests/test_reta_output_protocol.py` 及本说明；不改变其他方法、共享API、重试/预算设置、
分支、暂存区或提交，不新增live请求，不覆盖历史run。

- `input.legal_actions` 中每个动作新增展示字段 `action_id`，按原canonical动作顺序从0编号；
	不写入实际Action对象，不改变动作排序/参数。task明确只能返回这些整数ID，不能返回筹码数、
	出价牌值、玩家编号或历史索引。
- 删除 `{"action_ids":[zero-based indices into legal_actions]}` 伪JSON模板。
	例子由json.dumps生成，长度匹配min(width,合法动作数)：双候选为 `{"action_ids": [0, 1]}`，
	单候选为 `{"action_ids": [0]}`。明确例子只表示格式，不是行动推荐，也不是默认fallback。
- 同时删除reward的 `{"score": a number in [0,1]}` 伪JSON，改为明确数值范围加
	`{"score": 0.5}` 合法示例，并标注不是当前局面评分；仍接受任意有限[0,1]数值。
- 保持原JSON prompt外层结构、user消息、role信息边界与缓存规则。strict_json/candidate_ids/
	reward_score、ReTAMethod和配置的AST不变；不纠正字母O、π、中文、变量、不填补候选、不允许重复ID。
- 示例可能影响LLM选择，格式声明不能完全排除锚定偏差；没有声称修复已通过live或HTTP500原因已确定。

验证：ToMResearch Python3.11.15，32新增、309相关测试通过（5.56s），覆盖三游戏双位置、
width1/2/16、唯一剩余动作、动作ID不等于出价、实际坏回复拒绝与不同于示例的合法回复。
修改前后6个scripted局面相同选步、树值、采样seed和调用数；受保护源码及旧live artifact hash未变。
证据 `results/leduc/reta_dev/20260911_121405_numeric_id_protocol_dev/`，含精确命令、日志和prompt示例。

新配置三游戏P0 seed9100各1局scripted对GTO成功（分别位于results/<game>/reta_dev/）：
Leduc `20260911_121537_reta_vs_gto_1g_offline_dev` 46调用；
骰子 `20260911_121539_reta_vs_gto_1g_offline_dev` 48调用；
Goofspiel `20260911_121544_reta_vs_gto_1g_offline_dev` 27调用。
全部resume跳过已完成局、零新增调用。源码hash变化后旧run不得恢复；后续需新run另批live复验。

## 2026-09-11：方法内部非法输出重试（最新，离线验收）

用户要求“非法返回不进行重试吗？这个重试逻辑应该让方法内部自己实现”。本节取代此前
“普通非法JSON立即终止”的行为；旧run记录仍保留，不能按新行为重解释旧结果。
本轮没有新增live调用，没有改分支/暂存/提交，也没有修改其他方法、共享API/journal或共享交接文件。

### 两层重试职责

1. **网络层**：沿用公共journal最多3次额外暂时错误重试，相同prompt/同一逻辑call key，
	 每个实际尝试独立编号与HTTP审计；鉴权/明确截断等不自动重试。
2. **方法层**：`_Search.ask_validated()` 接收解析函数。只有模型返回文本后的JSON解析/schema
	 校验失败会触发纠正请求。main、anticipation、reward、perturbation均覆盖。
	 `ReTAConfig.output_retries=3` 表示首次+最多3次额外生成，允许0禁用（离线入口有 `--output-retries`）。
	 CLI不在外层重复整个决策，API wrapper不判断动作是否合法，实际游戏引擎异常不会触发本重试。

每个纠正请求保留原actor/task/input，在 `input.output_correction` 中加入上次失败回复、
安全错误类别、当前尝试序号与纠正要求。上次回复明确作为无效数据，而不是新指令。
不会拼入其他角色/世界的私有上下文；Goofspiel同信息集成功的假想对手提议仍跨本方候选复用。
首个prompt与原先相同，正常成功路径不增加调用。

方法重试是**重新请求模型生成合法结果**，不是用代码把O/π/中文/占位符替换成数字。
不足/重复/越界ID、错误类型、额外字段、坏score、坏perturbation都重试；只把最后通过校验的候选或评分用于树。
reward采样的均值只统计有效评分，不把失败尝试算作新票或0分。
重试耗尽抛 `OutputValidationFailed` 并停止，不返回部分树结果或默认动作。

明确length、BudgetExceeded、网络/存储/不确定请求异常不被当作格式错误捕获；网络错误耗尽后不会
被方法层再扩展出新轮网络重试。公共journal无法交付文本的空响应仍是CallFailed，不在本层猜测内容。

### 预算与记录

- 每个纠正请求使用新call ordinal、新journal key；不得覆盖此前的HTTP200/非法文本记录。
	每个纠正请求可能触发已有网络重试。两层都共享同一个全局实际请求预算：live入口仍最多60/游戏。
	单次actor评价理论上最多4个输出尝试×4个网络尝试=16个实际请求，但会提前受总预算限制。
	单决策 `max_calls_per_decision` 另限制方法调用数（包括纠正请求）；实际HTTP重试由journal总预算计数。
- call trace新增 `validation`：valid/invalid、invalid_json/invalid_schema、output_attempt、
	max_output_attempts、first_call_ordinal、response_sha256、actor/node。
- `logs/output_validation/P0|P1/gNNNN_sNNNN_decision.json` 持久化每个文本调用的内容校验结果，
	关联journal路径及物理attempt_number。网络文件继续在logs/network，原prompt/response在prompts。
	HTTP200文本success并不等于格式通过：必须同时查看validation.status。
- 校验记录在下次请求前写盘，写失败就停止；恢复时同记录只比较、不覆盖。成功的纠正请求直接缓存重放，
	历史非法文本也按原顺序重放并再次校验，确定性地走向下一条缓存，不重复计费。
	若4次均非法，resume也仍在第4次终止，不重置额度。若中断在纠正请求发送前，可恢复未完成项；
	started不确定调用仍禁止自动重发。源码/配置变化继续禁止旧run跨版本恢复。
- `summary.last_request.output_validation_path` 指向最终错误的校验文件；原response不写入异常消息。

### 验证

新增 `tests/test_reta_output_retry.py` 59项；相关11文件 **368 passed（7.51s）**。
覆盖4种actor在第4次成功、持续失败与禁用、实际历史坏输出、校验类型、预算、
两层重试编号区分/共享计数、纠正请求网络失败不扩大次数、明确截断不重试、存储失败与篡改拒绝、
半途恢复与耗尽后恢复无新增、Goofspiel信息集不变、HTTP MockTransport脱敏、完整局checkpoint。
原strict_json/candidate_ids/reward_score/backup函数AST未变，6个正常scripted观察的选步/树/调用数未变。

验证run：`results/leduc/reta_dev/20260911_123238_method_output_retry_dev/`（config含精确命令、summary、测试日志）。
三游戏正常scripted单局对GTO完成，46/48/27调用；分别在
`20260911_123608`、`20260911_123609`、`20260911_123614_reta_vs_gto_1g_offline_dev`。
每游戏另注入一条真实历史非法anticipation回复，下一次用scripted合法回复模拟恢复：
Leduc `20260911_123619_output_retry_injected_dev` 47调用、骰子同时间戳49调用、
Goofspiel `20260911_123620_output_retry_injected_dev` 28调用，均完成且resume零新增。
所有路径位于results/<game>/reta_dev/，细节在验证run metrics/smoke.json。
受保护共享源码与旧live artifact hash未变。

限制：这证明重试控制流、预算、恢复正确，**不证明真实LLM能被纠正或整局成功**。
纠正反馈改变生成分布，是显式记录的稳健性工程设置，不声称原论文固定协议。
新策略待人工验收、另批live复验；没有启动每游戏3局或批量实测。

## 2026-09-11：文本分段输入、分析＋末尾JSON、终局语义校验（最新）

用户要求参照已有vanilla/Suspicion输出呈现，减少无意义的格式失败。本次不再将整个内部dict
序列化为prompt；新增 `methods/reta/protocol.py`，按以下顺序构造真实发给模型的user文本：

1. GAME RULES：真实换行的规则文本（不是转义后的一整个JSON字符串）。
2. ROLE：main/anticipation/reward/perturbation职责。
3. GAME STATE：玩家ID与其可见私有/公共字段，逐行显示，不读取真实隐藏状态。
4. HISTORY：完整已发生的可见事件，逐条显示，不裁成最近N步。
5. LEGAL ACTIONS：每行显式action_id与具体动作参数；评分/改写调用不选择动作时注明不适用。
6. TASK CONTEXT：候选数量、模拟终局净收益等本次所需元数据；可选纠正反馈位于原任务之前。
7. TASK：本次分析/候选/评分任务。
8. RETURN FORMAT：放最后，允许先分析再以裸JSON或最终JSON代码块结束，附合法示例。

仍使用标准API外层model/messages；变动的是messages[0].content。内部数据、journal/网络日志保持结构化。
搜索节点只在审计元数据中，不发给模型。Goofspiel跨本方候选复用同一盲对手信息集的逻辑不变。
scripted模型与测试通过专用 `decode_fixture_prompt` 解读文本，**live不附加隐藏JSON副本**；
审计actor从已绑定的method metadata读取，不再依赖prompt必须可json.loads。

### 输出解析和重试

- `strict_json`仍是严格JSON对象/重复键检查；模型回复先经 `final_json_text` 提取完整末尾对象。
	支持前置自然语言和已闭合的早期示例JSON/代码块，不把这些误当最终动作。
- 末尾JSON内部仍必须合法，候选只能是合法范围内的互异整数ID，数量必须匹配。
	嵌套对象/数组不能被提取当结果，重复字段/坏最终对象/未闭合代码块/尾部追加解释均拒绝。
	不从文本抓动作词，不替换O/π/中文/变量，不回退到较早的合法JSON。
- 同一方法任务的格式重试仍最多3次额外请求；反馈保留原坏回复为无效数据，允许下一次先分析，
	结果只取新生成且通过校验的末尾对象。网络重试/不确定started/length保护不变。
- 新增 `validated_reward`：对**已终止的假想世界**，负/零/正净收益要求0/0.5/1。
	合法JSON但与该符号不符的分数记为terminal_score_mismatch，进入原有限重试；耗尽则停止，
	不悄悄替换成引擎分数。非终局仍为LLM有限[0,1]估计；不是读取真实隐藏状态或改真实payoff。
- 允许推理文本可能增加token；所有完整响应/纠正/网络重试仍计入原账本usage。并不声称JSON合法
	就代表推理正确。修改协议需新run；历史run的prompt/response/结果不覆盖，不跨源码恢复。

### 验证与状态

新测试49项，全ReTA和相关公共回归 **434 passed（11.88s）**。首轮431通过/3断言失败为
Goofspiel整数键JSON规范化及嵌入失败回复中的标题字符串误匹配；仅修正测试比较，未放宽生产校验。
包含带分析/代码块的候选与评分、坏末尾不误取旧JSON、字段/类型/范围拒绝、文字完整性、
终局语义失败纠正/耗尽、三游戏双位置narrative fixture、HTTP MockTransport真实文本请求、
完整响应计数/审计角色、无输出上限及旧恢复保护。

新九局scripted P0对GTO（seeds9100–9102，--no-request-limit）全部完成，恢复均零新增请求：
Leduc46/33/26，骰子48/46/44，Goofspiel27/27/27。目录及精确命令保存在
`results/leduc/reta_dev/20260911_134705_text_final_json_protocol_dev/metrics/smoke.json` 和 config.json。
此前不限请求九局252个已返回文本只读复查：212个格式有效、40个格式无效，在新末尾解析下结论一致，
没有靠宽松解析把厨房词/变量伪装成有效动作。受保护公共源码和全部旧live artifacts哈希未变。

审核新prompt：同一旧QH失败局面的新输入在验证run
`prompts/leduc_previous_failure_new_prompt.txt`；三游戏main/anticipation/reward各有独立文本示例。
示例response为scripted，不是live证据。本阶段**零新增live API**，不改分支/暂存区/其他方法/共享交接。

下一步：用户确认后先三游戏各1局新协议live，沿用本地GTO与记录要求、不设请求次数上限且不传max_tokens，
保留有限搜索/重试/步数保护。报告完整局token、最终payoff、失败原因与所有重试成本；失败局单列。
当前只完成离线验证，不保证服务端或模型永远无错，不自动扩采或反复跑到成功。