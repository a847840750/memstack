# Changelog

本文件收录 MemStack 各版本的实质改动。评测数字的完整口径见
[IMPROVEMENTS.md](IMPROVEMENTS.md) 与 [ARCHITECTURE.md](ARCHITECTURE.md)。

## [0.4.1] - 2026-11-01

### 修复：抽取事实把原始证据挤出上下文（评测实测的最大缺陷）

LongMemEval oracle 上 `MemStack`（规则抽取 + 图）比 `MemStack-ep`（只存原文）低 12.6pp，
All-Unit 73.8% vs 96.8%：高分事实条目吃满预算，原文 episode（日期、原话、精确数值）
反而进不了上下文。三处改动：

- `episode_reserve_ratio`（默认 0.30）+ `episode_reserve_items`（默认 3）：预算打包前
  先按比例给 episode 留额度，再用剩余预算填事实；`_fit_budget` 改为返回实际入选的 hits，
  修正此前 `items = hits[:used]` 与实际顺序不一致的问题。
- 时序查询（`when / how long / before / after …`）不再降权 episode：`_intent_weights`
  给 episode 1.45 倍权重，`slot_query` 的 ×0.65 惩罚对时序/聚合查询关闭。
- 聚合查询（`how many / list all / across …`）自动放宽 `top_k`（`aggregation_top_k_bonus`，默认 4）。

**实测（LongMemEval 200 题分层子集，800 token，同 prompt 同 reader/judge）**：
memstack 37.0%→**43.5%**、All-Unit 79.0%→**84.5%**、answer-in-context 42.5%→47.0%、
F1 31.9%→38.2；与 `memstack-ep` 的差距从 7.0pp 收敛到 0.9pp。
LongMemEval 500 题全量：32.4%→**43.4%**（All-Unit 73.8%→85.6%）。

### 新增：写入时相对时间归一化

- `normalize_relative_time()`（`extraction.py`）：把 `yesterday / last Saturday /
  3 years ago / next month …` 换算成绝对日期，原文措辞保留在括号内
  （`yesterday (8 June 2023)`），`relative_time_normalization` 默认开启。
- episode 行同时输出 ISO 与人类可读日期：`from=2023-06-09|on=9 June 2023`。

**实测（LoCoMo 300 题子集，600 token）**：时序题 53.3%→**58.3%**，
answer-in-context 14.8%→**19.2%**，F1 26.2%→**29.7%**。
**实测（LoCoMo 全量 1986 题）**：总体 49.6%→**54.3%**（时序 7.5%→**47.4%**，
F1 31.1→37.4，All-Unit 73.2%→74.1%）；检索层 Ans Token 33.2%→**43.1%**、
All-Unit 74.3%→**75.6%**。

### 新增：读取侧协议（`qa_eval.py`）

原 prompt 只说「只用记忆作答」，导致两类系统性失败：时序题 55% 直接拒答 / 79 题
复述 "Yesterday / Last week"；开放域题 84% 拒答。改为显式要求「把记忆行里的相对时间
换算成绝对日期」+ 推荐类问题按用户偏好作答；新增 `--grounding adaptive`
（仅对 open-domain 允许世界知识，对抗题仍强制拒答）。

**实测（同一 300 题）**：总体 38.3%→**46.7%**（strict）/ **52.7%**（adaptive）；
时序 7.5%→58.3%；开放域 9.4%→**35.0%**（adaptive）。

### 修复：评测吞吐

- `OpenAICompatClient.complete(..., max_tokens=)`：此前 reader/judge 没有输出上限，
  小模型偶发长生成让单题从 200ms 变 31s，整轮评测被拖慢一个数量级。

## [0.4.0] - 2026-10-29

### 新增：证据链检索（evidence-chain retrieval）

- 新增 `memstack/evidence_chain.py`：query 结构感知的分解 → 逐跳检索 → 分名额合并。
  - 四类触发形态：枚举聚合（`what hobbies / how many / what did X VERB?`）、图谱桥接
    （`headquarters / whose / employed by / lead of`，优先于枚举防误判）、时间锚点、
    行为动词尾。
  - 分名额合并：保留首轮 `top-(k − n_sub × slots)` 证据，每子查询只补未见过的新证据
    （枚举 4 个、其他 2 个名额），补证分数 ×0.9 公平竞争——根治旧词袋补证"挤掉首轮
    高分证据"的问题（v2 教训）。
  - 双路径扩证落地：首跳低分（熟悉度不足）时走 PPR 近邻扩证，而非盲查整库。
- `public_eval.py` 新增 `MemStack-mr`（`evidence_chain=False` + 旧多轮补证）同库对照，
  用于隔离证据链的增量。

**实测（LoCoMo 300 分层，800-token）**：multi-hop TurnToken 48.9%→51.6%、
AnsToken 22.2%→26.4%、All-Unit 27.1%→28.8%；总体 TurnHit 54.5%→57.2%；
LongMemEval oracle 500 中性不退化；合成基准 33 题 100% 不退化。

### 新增：性能与可插拔性

- **证据链共享缓存**：同一 scope/时间窗下，链内各跳复用候选集、实体节点与 PPR 图分数，
  子查询延迟 53.5ms→16.9ms（约 -69%）。
- **cache-aware 打包**（`status_aware_packing`，默认开）：预算不足时
  active 记忆优先进入上下文、历史值放尾部；预算充足时行为不变。
- **`LLMQueryDecomposer`**：LLM 化分解器（结构化输出 JSON 校验、markdown fence 容忍、
  异常/非法输出自动回退规则版），`MemoryStack(query_decomposer=...)` 注入；
  默认不调 LLM，离线零依赖路径不变。

### 评测与基础设施

- 全新全量 LoCoMo 评测（1986 题，10 种方法，结果见 `results/locomo_full_comprehensive.json`）。
- 54 个单元测试（新增证据链分解/合并/防挤占、LLM 分解回退、status 打包等 22 个）。
- CI 增加 LoCoMo 40 题公开评测 smoke（离线数据）。
- `pyproject.toml`：版本 0.4.0，补充 classifiers、project.urls、keywords。

## [0.3.0] - 2026-10-28

### 新增

- **多 Agent 协作层**：`MultiAgentHub`（私有/团队共享记忆、open/closed 团队、显式授权
  跨 agent 读、事件总线），Scope 表达归属，不复制存储。
- **零 SDK MCP server**（stdio）：`python -m memstack.mcp_server`，工具面与
  `as_tools()` 一致。
- **QA 层评测**：`qa_eval.py`，reader/judge 三种后端（本地 HF / OpenAI 兼容端点 /
  规则，其中规则零依赖），断点续跑，对抗题确定性拒答检测；可选外部库对照基线
  （`--methods extlib`，需安装 `mem0ai`）。
- **LLM 开箱工厂**：`build_stack` / `build_llm_client` / `LLMExtractor` /
  `LLMSummarizer`，环境变量无感接入任意 OpenAI 兼容端点。

### 优化

- 存储层：float32 BLOB 向量、revision 版本缓存、MMR numpy 化（4314 条 1550ms→520ms）。

## [0.2.0] - 2026-10-27

### 新增

- 多轮补证（multi-round retrieval）：多跳/聚合问题第二轮检索补足跨会话证据。
- 原始证据回填（evidence expansion）：top-3 事实回取 provenance 原文附入上下文。
- 公开数据集评测 `public_eval.py`：NDCG@k、每方法延迟、bootstrap 95% CI、LoCoMo 分层
  抽样；Fact+Hybrid / Temporal-Graph / Fact+Dense / Episode+Prior 四个同预算机制基线；
  所有基线统一 800-token 裁剪保证公平。

## [0.1.0] - 2026-10-26

### 初始版本

- 六层记忆流水线：接入层（Python / 工具 / HTTP）→ 七类记忆 → 写入管线（分段 / 规则抽取 /
  ADD-MERGE-SUPERSEDE-NOOP / 实体链接）→ SQLite 存储 → 五通道混合检索（dense + BM25 +
  entity + prior + PPR、熵路由、双时间、启发式重排、MMR、token 预算）→ 离线巩固
  （去重 / 冲突消解 / 会话摘要 / 衰减 / 画像块）。
- 五个 Protocol（Embedder / Extractor / Summarizer / Reranker / Clock / MemoryStore），
  零第三方依赖默认路径。
- 合成基准 `benchmark.py`（33 题：当前/历史地点、偏好、过敏、项目栈、流程、多跳总部等），
  覆盖陈旧回答率与 token 成本。
- 中文规则抽取基础支持；遗忘（软/硬 + 级联）与审计。