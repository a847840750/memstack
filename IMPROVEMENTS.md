# MemStack 改进说明与综合评测

> 本文件记录：当前项目还缺什么、这次改了什么、改进前后效果、与公开 SOTA 记忆系统的分数对照，以及仍然不能解决的问题。
>
> 配套代码：[public_eval.py](public_eval.py) · 原始结果：[results/](results) · 架构：[ARCHITECTURE.md](ARCHITECTURE.md)

---

## 0. 30 秒版

**改了什么**

1. **多轮补证（multi-round retrieval）**：多跳/聚合类问题在第一轮后，用已命中记忆的实体/槽位做第二轮检索，补足跨会话证据。
2. **原始证据回填（evidence expansion）**：对 top-3 事实，从 provenance episode 中找回与 query 最相关的原始轮次，附在上下文里，减少“事实对但原文被压缩掉”的问题。
3. **评测扩展**：新增 `NDCG@k`、每方法延迟、bootstrap 95% CI；新增 4 个 SOTA 风格本地基线（Mem0-like / Zep-like / A-MEM-like / MemoryOS-like），并与 Dense、BM25、Full context、MemStack、MemStack-ep 共 9 种方法同预算对比。
4. **公平性修正**：所有检索基线都按同一 800-token 预算裁剪，避免“多塞 token 赢”；LoCoMo 改为分层抽样，覆盖 5 类题型和 10 个 conversation。
5. **证据链检索（v4，本轮）**：把 v2 的“词袋扩词补证”升级为 query 结构感知的证据链——枚举聚合 / 图谱桥接 / 时间锚点 / 行为动词尾四类分解，每个子查询独立检索、分名额合并；首跳低分时走 RF-Mem 式 PPR 近邻扩证（双路径）。评测中新增 `MemStack-mr`（旧补证）做同库对照。

**效果**

- LoCoMo 200 题改进前后：TurnHit 56.3%→56.9%，TurnToken 69.3%→69.9%，AnsToken 22.9%→23.4%；代价是上下文 591.8→632.8 token。**提升有限**，因为主要瓶颈不是“证据回填”，而是多跳召回。
- LongMemEval oracle 500 题：MemStack-ep / MemoryOS-like 的 **All-Unit Recall 达到 95.0% / 95.6%**，同预算下 Dense / BM25 / Full context 只有 **35.2%**；MemStack 完整版为 **79.2%**。
- LoCoMo 300 题：MemStack **Unit R@k 83.8%、All-Unit 69.0%、TurnToken 69.6%**，同预算 BM25 57.6%/46.8%/69.3%，Dense 23.6%/17.5%/50.4%，Full context 2.4%/1.7%/35.7%。
- **QA 层已补上（v3）**：`qa_eval.py` 支持本地 HF / OpenAI 兼容端点 / 规则 reader 三种后端。LoCoMo 50 题分层子集上，MemStack 总体准确率 **46.0%**、可答题 **47.5%**、token-F1 **28.5%**；官方 mem0 包（infer=False）为 **22.0% / 20.0% / 13.7%**。
- **证据链（v4，本轮）**：LoCoMo 300 题上，MemStack 相对 `MemStack-mr`（v3 词袋补证）：总体 TurnHit 54.5%→**57.2%**、All-Unit 68.0%→**69.0%**、TurnToken 68.0%→**69.6%**、AnsToken 20.3%→**21.4%**；multi-hop 分题型 All-Unit 27.1%→**28.8%**、TurnToken 48.9%→**51.6%**、AnsToken 22.2%→**26.4%**。LongMemEval oracle 500 上中性不退化。
- **全量 LoCoMo 1986 题（v4.1）**：MemStack All-Unit **74.3%** [72.4%, 76.3%] vs MemStack-mr 71.5%（CI 区间外，统计可信）；multi-hop 分题型 All-Unit 18.5%→**24.2%**（+5.7pp）；超越 Mem0-like（72.4%）与 Zep-like（72.8%），仅次于 MemStack-ep（79.2%）；链内共享缓存把证据链的延迟增量压到 +14ms/查询。
- **没有解决**：multi-hop 的 All-Unit 仍低（全量 1986 题口径 24.2%、n=281；300 题分层口径 28.8%）：枚举题的 gold 证据散布在 7–19 个会话，800-token 预算内结构上不可能全部召回；TurnToken 仍低于"整段会话"基线；规则抽取器在开放闲聊上噪声明显；QA 层的 hf 小模型本次无法加载，50 题 hf 数字仍为 v3 口径。

---

## 1. 改进前的主要短板

结合 v1 的 LoCoMo 400 / LongMemEval oracle 500 结果：

| 短板 | 证据 | 影响 |
|---|---|---|
| **多跳覆盖不足** | LoCoMo multi-hop All-Unit 仅 22.8%（v1）；LongMemEval multi-session All-Unit 66.2% | 跨会话问题只取到部分证据，无法回答“总和/先后/关联” |
| **证据逐字丢失** | 同预算下 MemStack TurnToken 55.7%–56.4%，低于整段会话 69%–70% | 事实对了，但原始轮次被 episode 压缩截断 |
| **没有 SOTA 对照** | 只有 Dense/BM25/Full 基线 | 无法判断 MemStack 相对 Mem0/Zep 类设计的取舍 |
| **评测指标单薄** | 只有 Recall/MRR/token | 缺排序质量（NDCG）、延迟、置信区间 |
| **基线预算不公平** | v1 Dense/BM25 取 8 个完整会话，约 6000 token | 对比被“多塞 token”污染 |
| **规则抽取器噪声** | LoCoMo 开放闲聊出现 `who.supports = you` 类事实 | 完整 MemStack 不如 episode-only |
| **没有 QA 能力** | 只有检索层指标 | 不能回答“最终答对率” |

---

## 2. 这次实现的改进

### 2.1 多轮补证（multi-round）

位置：`memstack/core.py::MemoryStack.recall`。

```text
第一轮：正常混合检索 → top-k
  ↓ 若 query 含多跳/聚合/顺序线索（before/after/how many/both/total/...）
从 top-5 命中中抽取实体与槽位值（最多 8 个）
  ↓
第二轮：query + 扩展词 → 重新检索
  ↓
合并：保留第一轮 top-(k-2) 证据，用第二轮补足剩余 2 个名额
```

设计要点：

- **只补 2 个名额**：避免第二轮把第一轮的高分证据挤掉（第一版实现会挤掉，导致 AllUnit 反而下降；已修正）。
- **只在多跳/聚合线索触发**：普通单跳查询不增加延迟。
- **共享同一个 store**：不重复写入，只改变读取 query。

效果：LoCoMo multi-hop 的 TurnToken 47.6%→49.5%，All-Unit 未显著提升；
说明**瓶颈在“第二轮扩展词不够精准”**，不是“轮数不够”。这是下一步要改的地方。

### 2.2 原始证据回填（evidence expansion）

位置：`memstack/retrieval.py::_source_snippet`。

- 对 top-3 的 fact/preference/procedure，沿着 `source_ids` 找到 provenance episode；
- 在 episode 的消息中选与 query + 事实文本重叠最高的一条原始轮次；
- 追加 `| source="..."`，最多 320 字符；
- 若原始轮次与已展示 evidence 的前 12 个 token 相同，则不重复。

效果：LoCoMo 200 题 TurnHit +0.6pp、TurnToken +0.6pp、AnsToken +0.2pp，token +41；
**收益小但稳定**。在真实 Agent 场景中，证据原文对审计/引用更有价值。

### 2.3 评测指标扩展

`public_eval.py` 新增：

- `NDCG@k`：二值相关性下的排序质量（互补 MRR）；
- `avg_latency_ms`：每个方法的平均检索延迟；
- `ci95`：对 Unit Recall、All-Unit、Turn Token、Ans Token 做 1000 次 bootstrap 的 95% 置信区间；
- 分层抽样：LoCoMo 按 5 类题型轮询抽样，避免只取前几个 conversation；
- 所有 Dense/BM25 基线按预算裁剪（v1 会返回 6000 token，已修正）。

### 2.4 SOTA 风格本地基线

同一份写入结果，只替换读取配置，避免“不同写入管线”混淆：

| 方法 | 近似对象 | 配置 |
|---|---|---|
| **Mem0-like** | Mem0 的抽取 + 混合检索 | 抽取事实；dense/BM25/entity；无时间/图/路由/重排 |
| **Zep-like** | Zep/Graphiti 的时序图 + 混合检索 | 时间 + 图 + 混合检索；无路由 |
| **A-MEM-like** | A-MEM 的笔记 + 向量 | 事实/笔记；纯 dense |
| **MemoryOS-like** | MemoryOS 的分层 + 热度/时效 | 仅 episode；dense + 热度/时效先验 |
| **MemStack-ep** | 只存原始 episode | 不做事实抽取；混合检索 |
| **MemStack** | 完整系统 | 多轮 + 证据回填 + 时间 + 图 + 路由 + 重排 |

这些是**机制近似**，不是运行 Mem0/Zep/A-MEM/MemoryOS 原仓库；因此只能做同数据/同预算的相对比较。

### 2.5 证据链检索 evidence chain（v4，本轮）

位置：`memstack/evidence_chain.py` + `core.py::_evidence_chain_recall`；开关 `MemoryConfig.evidence_chain`。

**为什么做**：v2 多轮补证把首轮 top-5 命中的实体词拼回 query 再查一轮，实测证明"扩展词漂移"——拼进去的实体词指向已命中会话的同款词汇，分散在长尾会话的证据依然拿不到（LoCoMo multi-hop All-Unit 停在 27.1%）。分解 LoCoMo multi-hop 题型后发现：它们大多是**枚举聚合**（"What hobbies does Evan pursue?"，gold 证据散布在 7–19 个会话），需要的不是扩词，而是"主体 + 槽位"的枚举检索。

**怎么做**（编号步骤）：

```text
1. needs_chain(query)：四种触发形态之一才继续：
   枚举聚合（how many / what hobbies / what did X VERB?）
   图谱桥接（headquarters / whose / employed by / lead of —— 优先于枚举，防止
             "What company is the team lead of ...?" 的 team 被误判成枚举）
   行为动词尾（What did Caroline research? → 尾部动词=槽位词）
   时间锚点（after/before + 事件）
2. 分解：锚定主体（query 专名优先，其次首轮命中高频实体）
   + 槽位同义词扩展（hobby→hobbies/pursue/enjoy/activities；tournament→games/won...），
   最多 2 个子查询（枚举 1 个 / 桥接 2 个）。
3. 逐跳检索：每个子查询独立 search（共享 store，touch=False）。
4. RF-Mem 双路径：首跳最高分 < 0.30（熟悉度不足）→ 从 top-3 种子做 PPR，
   取图中与 query 共享实体/词面的近邻记忆 4 条（不走全局盲查）。
5. 分名额合并：保留首轮 top-(k − n_sub × slots)；每子查询只补 slots 个
   未见过（id 级）的新证据（枚举 4 个、其他 2 个）；补证分数 ×0.9 再总排序。
```

**验收规则**：新证据 = id 未在首轮出现；重复不占名额。首轮高分证据因保序名额机制不受挤（单测 `test_keeps_first_round_evidence`）。

**实测（同库、同预算，与 `MemStack-mr` 对照）**：

| 数据集 | 指标 | MemStack-mr（v3） | MemStack（v4） |
|---|---:|---:|
| LoCoMo 300 | multi-hop All-Unit | 27.1% | **28.8%**（+1.7pp） |
| | multi-hop TurnToken / AnsToken | 48.9% / 22.2% | **51.6% / 26.4%**（+2.7 / +4.2pp） |
| | 总体 TurnHit / All-Unit | 54.5% / 68.0% | **57.2% / 69.0%** |
| | 每查询延迟 | 76.5ms | 100.7ms（+24ms） |
| LongMemEval 500 | All-Unit / AnsToken | 79.2% / 46.3% | 79.2% / 45.8%（中性，波动内） |
| 合成 benchmark 33 题 | 准确率 | 100% | 100%（含 multihop，不退化） |

**读数**：

1. v4 在 LoCoMo 全面优于 v3，且不破坏其他数据集；提升来自"面向槽位的枚举检索"把长尾会话顶进上下文（TurnToken/AnsToken 显著上升 = 逐字证据变多）。
2. multi-hop **All-Unit 的天花板**：枚举题 gold 证据 7–19 个会话、800-token 预算，按定义（全部证据都要召回）结构性不可达；All-Unit 的 +1.7pp 是有限的，但 Turn/AnsToken 的大幅提升才是可直接被 reader 消费的收益。
3. 延迟 +24ms/查询 是二次检索的固定成本；对 hf/llm QA（~1900ms/题）可忽略，对纯检索层敏感场景可 `evidence_chain=False` 关闭。

**QA 层对照（本轮）**：hf 小模型本次因环境无法加载（`OSError: Can't load the model for 'Qwen/Qwen3.5-0.8B'`），v3 的 50 题 hf 表数字**未重跑**（口径仍是 v3）。替代方案跑了 `extractive+f1`（50 题分层、600 token）：检索覆盖 memstack 89.6% vs memstack-mr 87.5%（Unit R@k，+2.1pp 与检索层一致），但规则 reader 在开放答案上 Accuracy 全 0、F1 4.5% 无区分度——结论：**QA 层增益验证待本地模型恢复后重跑**。

### 2.5.1 v4.1 增量：共享缓存 / status-aware 打包 / LLM 分解（本轮）

1. **链内共享缓存**：`HybridRetriever.search(..., _shared=...)` 让证据链各跳复用候选集、实体节点与 PPR 图分数（key 含数据 revision，写入即失效）。实测子查询延迟 53.5ms→16.9ms（约 -69%）。
2. **status-aware 打包**（`status_aware_packing`，默认开）：预算裁剪时 active 优先进上下文、superseded/archived 放尾部；预算充足时行为不变。对应 TokenPilot 的"生命周期驱逐"。单测：`test_active_before_superseded_when_over_budget`。
3. **`LLMQueryDecomposer`**：LLM 化分解（JSON 结构化输出 + fence 容忍 + 字段校验 + 异常回退规则版）。默认不启用保持零依赖；`build_stack(llm_decompose=True)` 可开箱启用。单测覆盖正常解析、坏 JSON 回退、调用异常回退、MemoryStack 集成（trace 记录 llm / rule-fallback）。

### 2.5.2 全量 LoCoMo 1986 题实测（v4.1）

数据：LoCoMo 全部 10 个 conversation、**1986 题**（adversarial 446 / single-hop 841 / temporal 320 / multi-hop 281 / open-domain 89）；800-token；`top_k=8`；episode 3 轮。
结果文件：[results/locomo_full_comprehensive.json](results/locomo_full_comprehensive.json)。

| 方法 | Unit R@k | All-Unit | TurnHit | Turn Token | Ans Token | NDCG | Tokens | Lat ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MemStack**（v4.1） | **84.3%** | **74.3%** | **59.6%** | **70.0%** | **33.2%** | 0.605 | 599 | 126 |
| **MemStack-mr**（v3） | 82.0% | 71.5% | 56.5% | 67.9% | 31.8% | 0.590 | 583 | 112 |
| **MemStack-ep** | **88.3%** | **79.2%** | **66.5%** | **77.5%** | **36.7%** | **0.701** | 722 | 66 |
| Mem0-like | 82.0% | 72.4% | 56.3% | 69.7% | 32.8% | 0.623 | 632 | 85 |
| Zep-like | 83.1% | 72.8% | 57.9% | 69.3% | 32.2% | 0.609 | 598 | 104 |
| A-MEM-like | 66.2% | 56.1% | 35.4% | 55.2% | 24.6% | 0.428 | 591 | 88 |
| MemoryOS-like | 76.8% | 67.0% | 49.9% | 67.2% | 30.6% | 0.544 | 698 | 78 |
| BM25 only | 64.0% | 56.0% | 60.2% | 74.0% | 33.2% | 0.640 | 650 | 3 |
| Dense only | 23.3% | 19.6% | 22.4% | 49.4% | 18.8% | 0.232 | 633 | 2 |
| Full context | 2.9% | 2.4% | 2.9% | 35.0% | 9.3% | 0.029 | 585 | 0 |

**关键读数（大样本 1986 题）：**

1. **v4.1 相对 v3 的增益在放大**：全量下 All-Unit +2.8pp（71.5%→74.3%）、TurnHit +3.1pp、TurnToken +2.1pp、Unit R@k +2.3pp。bootstrap 1000 次 95% CI：MemStack All-Unit 74.3% [72.4%, 76.3%]，v3 的 71.5% 落在区间外——**证据链的领先在 1986 题上是统计可信的**（300 题子集上仅有趋势）。
2. **multi-hop 分题型（n=281）**：All-Unit 18.5%→**24.2%**（+5.7pp，相对 +31%）、TurnToken +3.7pp、AnsToken +3.4pp——与 300 题子集的方向一致但幅度更大。
3. **MemStack 超过两个 SOTA 风格基线**（Mem0-like 72.4%、Zep-like 72.8%），仅次于只存原始 episode 的 MemStack-ep（79.2%）；MemStack-ep 仍是开放对话下的覆盖上限，印证"事实抽取在开放闲聊里会损覆盖"的既有结论。
4. **延迟**：v4.1 的链内共享缓存把绝对延迟压到 +14ms/查询（相对 v3 的 112ms），此前 300 题未优化版本为 +24ms。
5. 效度口径不变：检索层指标、规则抽取器、LoCoMo 标注错误（约 6.4%）与被审计的 judge 问题照旧适用；数字不可与 LLM-as-judge 的公开分数直接并列。

---

## 3. 改进前后对比（LoCoMo 200）

| 指标 | v1 | v2（多轮 + 证据回填） | 变化 |
|---|---:|---:|---:|
| Unit R@k | 85.3% | 85.3% | 0.0 |
| All-Unit | 71.1% | 71.1% | 0.0 |
| Turn Strict | 56.3% | 56.9% | +0.6pp |
| Turn Token | 69.3% | 69.9% | +0.6pp |
| Ans Token | 22.9% | 23.4% | +0.5pp |
| Tokens | 591.8 | 632.8 | +41.0 |

结论：**证据回填是有效的但边际收益小**；多轮补证在 LoCoMo 上没有解决 All-Unit，说明需要更好的 query decomposition / 证据链扩展。

---

## 4. LoCoMo 300 综合对比

数据：LoCoMo 10 conversations，分层 300 题（每类约 60 题；297 题有证据）；800-token 预算；`top_k=8`；episode 3 轮。
结果文件：[results/locomo_300_comprehensive.json](results/locomo_300_comprehensive.json)。

| 方法 | Unit R@k | All-Unit | Turn Token | Ans Token | NDCG@k | Tokens | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MemStack**（v4 证据链） | **83.8%** | **69.0%** | **69.6%** | **21.4%** | **0.601** | 626.9 | 100.7 |
| **MemStack-mr**（v3 词袋补证） | 82.8% | 68.0% | 68.0% | 20.3% | 0.597 | 620.4 | 76.5 |
| **MemStack-ep** | **86.5%** | **72.4%** | **74.1%** | **24.3%** | **0.651** | 715.5 | 44.8 |
| Mem0-like | 82.2% | 68.0% | 67.4% | 21.2% | 0.603 | 648.7 | 55.9 |
| Zep-like | 83.5% | 68.0% | 68.6% | 20.8% | 0.607 | 632.7 | 71.8 |
| A-MEM-like | 65.7% | 51.9% | 53.1% | 15.1% | 0.412 | 613.1 | 55.8 |
| MemoryOS-like | 75.1% | 62.0% | 65.1% | 20.3% | 0.521 | 702.6 | 50.8 |
| Dense only | 23.6% | 17.5% | 50.4% | 13.6% | 0.236 | 644.1 | 1.2 |
| BM25 only | 57.6% | 46.8% | 69.3% | 20.6% | 0.576 | 667.6 | 2.0 |
| Full context | 2.4% | 1.7% | 35.7% | 5.8% | 0.024 | 522.8 | 0.3 |

**读数：**

- **v4 证据链全线优于 v3 词袋补证**：TurnHit 54.5%→57.2%（+2.7pp）、TurnToken 68.0%→69.6%、AnsToken 20.3%→21.4%、NDCG 0.597→0.601；multi-hop 分题型 TurnToken +2.7pp、AnsToken +4.2pp、All-Unit +1.7pp（见 §2.5）。代价 +24ms/查询。
- **MemStack 与 Mem0-like / Zep-like 同档且略高**：Unit R 82%–84%，All-Unit 68%–69%，TurnToken 67%–70%。
- **MemStack-ep 的覆盖最高**（86.5% / 72.4%），且 TurnToken 74.1%；说明在开放对话里，**原始 episode 比规则事实更可靠**。
- **A-MEM-like 明显弱**（65.7% / 51.9% / 53.1%）：纯 dense + 事实，缺少 BM25/实体/多轮。
- **MemoryOS-like 居中**（75.1% / 62.0% / 65.1%）：只取 episode + 热度，覆盖不如 MemStack-ep，但成本更低。
- **Dense / Full context 崩塌**：同预算下只能保留最近片段，早期证据几乎取不到；Full context 的 Unit R 仅 2.4%。
- **多跳仍是最大短板**：MemStack multi-hop Unit R 79.7%，All-Unit **28.8%**（v3 为 27.1%）；TurnToken 51.6%（v3 为 48.9%），见 §2.5 的天花板分析。

**95% bootstrap 置信区间（MemStack v4）：**

- Unit R：83.8% [79.8%, 87.9%]
- All-Unit：69.0% [63.3%, 74.4%]
- TurnToken：69.6% [65.8%, 73.1%]
- AnsToken：21.4% [18.0%, 25.1%]

区间宽度约 5–8pp，说明 300 题上的差异要大于这个量级才值得解读；TurnToken 的 +1.6pp（68.0%→69.6%）本身在区间宽度之内，但 multi-hop 分题型的 +2.7pp 与"全线同向"的趋势使结论可信。

---

## 5. LongMemEval oracle 500 综合对比

数据：LongMemEval oracle（只含证据会话，不含 distractor）；500 题；800-token；`top_k=8`；episode 2 轮。
结果文件：[results/longmemeval_comprehensive.json](results/longmemeval_comprehensive.json)。

| 方法 | Unit R@k | All-Unit | Turn Token | Ans Token | NDCG | Tokens | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MemStack**（v4 证据链） | 100% | 79.2% | 58.0% | 45.8% | 1.000 | 764.7 | 23.9 |
| **MemStack-mr**（v3 词袋补证） | 100% | 79.2% | 57.6% | 46.3% | 1.000 | 766.4 | 23.8 |
| **MemStack-ep** | 100% | **96.6%** | 56.3% | 52.3% | 1.000 | **409.5** | 14.5 |
| Mem0-like | 100% | 84.6% | 59.4% | 49.1% | 1.000 | 721.6 | 16.8 |
| Zep-like | 100% | 80.0% | 57.6% | 46.9% | 1.000 | 766.9 | 71.7 |
| A-MEM-like | 100% | 85.2% | 54.9% | 45.0% | 1.000 | 707.8 | 16.7 |
| **MemoryOS-like** | 100% | **95.6%** | 55.3% | 50.8% | 1.000 | **401.8** | 12.7 |
| Dense only | 100% | 35.2% | 69.2% | 50.5% | 1.000 | 791.9 | 0.9 |
| BM25 only | 100% | 35.2% | 69.4% | 52.2% | 1.000 | 792.2 | 1.4 |
| Full context | 100% | 35.2% | 70.0% | 55.5% | 1.000 | 793.2 | 0.6 |

**读数：**

- oracle 中所有 session 都是证据，Unit R 对所有方法都是 100%，**All-Unit 才是关键**。
- 同 800-token 预算，Dense/BM25/Full 只能装 1–2 个会话，All-Unit 35.2%；MemStack-ep / MemoryOS-like 用 400 token 覆盖 **96.6% / 95.6%** 证据会话。
- MemStack 完整版的 All-Unit 79.2%，低于 MemStack-ep/MemoryOS-like：**事实抽取 + 路由/重排在这个数据集上反而压缩了会话覆盖**。
- **v4 证据链在 LME 上中性**：All-Unit 79.2% 持平、TurnToken 58.0% vs 57.6%、AnsToken 45.8% vs 46.3%（±0.5pp 波动内）——证据链只取多两个名额，不挤占原有证据，也没有拖累。
- 代价：整段会话基线的 TurnToken 69%–70%，MemStack-ep 56.3%；覆盖与逐字证据是两种能力。
- MemStack/most variants 的 NDCG=1.000 是因为所有 session 都是证据，排序无区分度；不要把这个 1.000 当成强指标。

---

## 6. LLM reader / judge 与官方 mem0 基线（v3）

> **v4 增补（本轮）**：证据链改变了检索层返回的上下文，理论上本节数字需要重跑。本轮本地 hf 小模型因环境无法加载（`OSError: Can't load the model for 'Qwen/Qwen3.5-0.8B'`），**下表数字仍是 v3 口径**。临时以 `extractive+f1`（`python qa_eval.py --limit 50 --budget 600 --methods memstack,memstack-mr,bm25 --reader extractive --judge f1`）做了同口径对照：检索覆盖 memstack 89.6% vs memstack-mr 87.5%，但规则 reader 在开放答案上 Accuracy 全 0、无法区分（`results/locomo_qa_v4.json`）。待模型可用后应重跑 hf 口径并替换下表。

前面 4、5 节只评**检索层**：证据会话有没有被取回。但公开榜单（Mem0 66.88、Zep 75.14）都是 **reader 生成答案 + judge 判分**之后的分数，指标不同就不能并列。本节补上这一层。

### 6.1 新增 `qa_eval.py`

| 组件 | 可选后端 | 说明 |
|---|---|---|
| reader | `extractive` / `hf` / `llm` | 规则抽取下界 / 本地 transformers（默认 Qwen3.5-0.8B）/ 任意 OpenAI 兼容端点 |
| judge | `f1` / `hf` / `llm` | token-F1 + EM / 本地 LLM 判分 / 在线 LLM 判分 |
| 指标 | 总体准确率、可答题准确率、拒答准确率、F1、EM、QA 延迟 | 可答题 = LoCoMo category 1–4；对抗类单独统计 |
| 工程 | 每 5 题 checkpoint、`--resume` 续跑 | 长任务被中断后不用从头再跑 |

两个判分细节：

1. **对抗题（category 5）不用 LLM judge**。它没有 gold answer，官方口径是"模型是否拒绝作答"；而 0.8B judge 会把 `No information available` 判成 INCORRECT。这里改成确定性拒答检测（`no information / not mentioned / cannot be determined / ...`）。
2. **可答题用 LLM judge + token-F1 双记**。LLM judge 给"语义等价"判断，token-F1 给与 A-MEM / SimpleMem 公开数字同口径的数值。

### 6.2 官方 mem0 仓库基线（不是机制近似）

```bash
pip install mem0ai    # 2.2.1
```

做法：`Memory.from_config` + qdrant 本地库 + `add(..., infer=False)` 写原始消息 + `search(...)` 取回。
为了不引入额外模型下载，把 mem0 的 embedder 换成 MemStack 的零依赖 hashing 向量（通过 `embedding_model` 注入）。

**必须知道的三个限制**：

1. **没有 LLM API key**，所以没有跑 mem0 的 LLM 事实抽取，只存原始消息；
2. `fastembed` 的 BM25 模型下载失败（网络），所以是**纯 dense**，不是 mem0 v2 的混合检索；
3. 因此这一行是**官方代码的检索下限**，不能代表 mem0 论文里的 66.88。

### 6.3 实测：LoCoMo 50 题分层子集

设置：LoCoMo 10 个 conversation、按 5 类题型分层抽 50 题（每类 10 题）、800→**600 token** 预算、`top_k=8`、reader 与 judge 都是 **Qwen3.5-0.8B-Instruct（CPU，离线）**。

| 方法 | Unit R@k | All-Unit | Turn Token | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MemStack** | 87.5% | 77.1% | 68.3% | 567 | **46.0%** | **47.5%** | **40.0%** | **28.5%** | **15.0%** | 1866 |
| MemStack-ep | 87.5% | **79.2%** | **71.7%** | 585 | 42.0% | 45.0% | 30.0% | 27.4% | 12.5% | 1861 |
| BM25 only | 52.1% | 43.8% | 64.6% | 585 | 32.0% | 35.0% | 20.0% | 25.8% | 12.5% | 1917 |
| **Mem0（官方包）** | **91.7%** | **83.3%** | 55.4% | 577 | 22.0% | 20.0% | 30.0% | 13.7% | 5.0% | 1971 |

95% bootstrap CI（2000 次，按题重采样）：

- MemStack 总体准确率 46.0% [32.0%, 60.0%]，F1 28.5% [18.0%, 39.5%]
- Mem0 总体准确率 22.0% [10.0%, 34.0%]，F1 13.7% [6.2%, 22.5%]

分题型准确率：

| 方法 | multi-hop | temporal | open-domain | single-hop | adversarial |
|---|---:|---:|---:|---:|---:|
| MemStack | 40% | 60% | 30% | 60% | 40% |
| MemStack-ep | 50% | 40% | 30% | 60% | 30% |
| BM25 only | 30% | 10% | 40% | 60% | 20% |
| Mem0（官方包） | 20% | 0% | 30% | 30% | 30% |

### 6.4 怎么读这组数

1. **MemStack 46.0% vs 官方 mem0 22.0%**：差距主要不在覆盖（mem0 的 Unit R 反而更高，91.7% vs 87.5%），而在**上下文结构**。mem0 返回的是一袋原始消息，0.8B reader 很难从"Caroline: ..."的碎片里拼出答案；MemStack 返回带槽位、时间、来源的条目，可读性更好。
2. **覆盖 ≠ 可答**：这正好说明为什么必须加 QA 层。只看 All-Unit，会得出"mem0 更好"的相反结论。
3. **与公开分数不可比**：Mem0 66.88 / Zep 75.14 用 GPT-4o 做 reader+judge、全量 1540 题、约 1.8k token；这里是 0.8B 小模型、50 题、600 token。我们的 28.5% F1 也不是 A-Mem 32.58 / SimpleMem 43.24 的同条件对比（它们用 GPT-4.1-mini）。
4. **对抗题全部偏低**（20%–40%）：说明小 reader 倾向于"硬答"而不是拒答，这是 reader 能力问题，不是记忆层问题。
5. **50 题太少**：准确率的 95% CI 宽约 ±14pp，两个方法差 8pp 以内不应下结论。

### 6.5 换成大模型重跑

```bash
# 任意 OpenAI 兼容端点（OpenAI / DeepSeek / vLLM / Ollama /v1）
export MEMSTACK_LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python qa_eval.py --limit 300 --budget 800 --reader llm --judge llm \
  --out results/locomo_qa_llm.json
```

只有在同 reader、同 budget、同题集的条件下，我们的数字才可以和 Mem0/Zep 的公开分数放在同一张表里讨论。

---

## 7. 与公开 SOTA 记忆系统的分数对照

> **必须先读**：下表是各系统在论文/博客中**自己报告**的分数，指标、backbone、reader、judge、数据版本都不同；不能与我们的检索层指标直接比较。LoCoMo 本身还存在约 6.4% 标注错误、LLM judge 会接受部分错误答案等已知问题。

| 系统 | Benchmark | 指标 | 公开分数 | 口径 / 备注 | 来源 |
|---|---|---|---:|---|---|
| **Mem0**（ECAI 2025 论文） | LoCoMo | LLM-as-Judge J | **66.88** | GPT-4o reader+judge；对抗类不进 J | arXiv:2504.19413 |
| **Zep**（2025 官方复跑） | LoCoMo | J | **75.14** | 指出 Mem0 论文里 Zep 配置有误后复跑 | Zep 博客 |
| **OpenAI Memory** | LoCoMo | J | **52.90** | 同 Mem0 论文对照表 | arXiv:2504.19413 |
| **Full-context 上限** | LoCoMo | J | **72.90** | ~26k token/对话 | arXiv:2504.19413 |
| **Mem0**（2026 厂商博客） | LoCoMo / LongMemEval | 自报 | **92.5 / 94.4** | 新算法；~6,956 / 6,787 token per query | mem0.ai blog |
| **SimpleMem**（ICML 2026） | LoCoMo-10 | token-F1 | **43.24** | GPT-4.1-mini reader；~550 token/query | OpenReview / README |
| **Mem0**（同表复跑） | LoCoMo-10 | token-F1 | **34.20** | GPT-4.1-mini | SimpleMem 表 |
| **A-Mem** | LoCoMo-10 | token-F1 | **32.58** | GPT-4.1-mini | SimpleMem 表 |
| **LightMem** | LoCoMo-10 | token-F1 | **24.63** | GPT-4.1-mini | SimpleMem 表 |
| **MemoryOS** | LoCoMo | F1 / BLEU-1 | **相对 +49.11% / +46.18%** | 是相对提升，不是绝对分 | MemoryOS 论文/仓库 |
| **MemGPT** | DMR | Accuracy | **93.4** | 不同 benchmark | MemGPT 论文 |
| **Zep** | DMR | Accuracy | **94.8** | 不同 benchmark | Zep 论文 |

**怎么用这张表：**

1. 不要拿检索层的 `Unit R@k` / `Turn Token` 去和 66.88 / 75.14 / 92.5 比——那些是 reader+judge 之后的 QA 分数。
2. 真正可比的是**同数据、同预算、同 reader** 的对照：我们已提供 4 个机制近似基线（Mem0-like / Zep-like / A-MEM-like / MemoryOS-like）与**官方 mem0 包**；但后者没有 LLM 抽取，只是 mem0 的检索下限。
3. 公开分数的方差极大：同一 Zep 在 2025 年复跑是 75.14，2026 厂商博客口径变成 90+；Mem0 从 66.88 到 92.5。差异主要来自 reader/judge/数据版本/token 预算，而不是架构一年内提升 25 分。
4. 如果要在同一张榜上比较，必须：固定 backbone、固定 reader/judge、跑原系统或官方脚本、报告 95% CI 和 token/latency。
5. **对抗类（LoCoMo category 5）没有 gold answer**，Mem0 的 J-score 口径直接把它排除；我们的 `qa_eval.py` 单独统计"拒答准确率"，并把可答题准确率与总体准确率分开报。

---

## 8. 推荐配置（按场景）

| 场景 | 推荐 | 理由 |
|---|---|---|
| 结构化 Agent 记忆（用户偏好、任务状态） | **MemStack 完整** | 事实 + 时间 + 冲突消解 + 溯源 |
| 开放对话/长文证据检索 | **MemStack-ep** | 覆盖最高（LoCoMo 86.5/72.4，LME 96.6），TurnToken 更高 |
| 多会话覆盖优先、预算极紧 | **MemoryOS-like**（episode + 热度） | LME All-Unit 95.6%，402 token |
| 需要逐字证据/引用 | **BM25 + 整段会话 / Full context** | TurnToken 69%–70%，但 All-Unit 仅 35% |
| 时序/知识更新 | MemStack / Zep-like | 双时间 + supersede；LoCoMo temporal Unit R 90% |
| 多跳 | **MemStack（v4 证据链）** | LoCoMo multi-hop TurnToken 51.6%、AnsToken 26.4%（vs v3 48.9%/22.2%）；All-Unit 28.8% 受 token 预算天花板限制 |

---

## 9. 还没有解决的问题

1. **多跳 All-Unit 依旧低（28.8%）**：v4 证据链把 TurnToken/AnsToken 各提升 +2.7/+4.2pp，但 All-Unit 受"gold 证据 7–19 个会话 vs 800-token 预算"的天花板限制；真正的解法是加大证据预算或 reader 侧迭代式查证，而不是继续堆检索轮次。
2. **规则抽取器噪声**：开放闲聊里会抽到无意义事实；`LLMExtractor` 是接口预留，未在公开评测中启用（无 API key）。
3. **QA 层还没跑到"可下结论"的规模**：只有 50 题、600 token、0.8B reader+judge，准确率 CI 宽约 ±14pp；且本轮 hf 模型不可用，QA 数字仍是 v3 口径；要真正对标 Mem0/Zep，需要全量 1540 题 + GPT-4o 级 reader。
4. **官方 mem0 只跑了检索下限**：`infer=False`、纯 dense、hashing 向量，没有 LLM 事实抽取和 BM25；这不是完整 mem0 管线。
5. **LongMemEval 只跑 oracle**：没有 distractor；`longmemeval_s` 约 277MB / 25k sessions，CPU 写入成本高。
6. **延迟**：v4 证据链使 LoCoMo 查询延迟 76.5→100.7ms（约 +24ms/查询）；Dense 仅 ~1.2ms。BM25 倒排已有缓存，PPR 邻接表有版本缓存，但证据链的逐子查询检索仍是一次额外扫描。
7. **SOTA 对照**：mem0 已跑官方包但被上面第 4 条限制；Zep / A-MEM / MemoryOS 仍只有机制近似或公开分数，没有跑原仓库。
8. **评测口径风险**：LoCoMo 子集、token 估算器、bootstrap 仅覆盖本题集；需要独立复跑。
9. **分解器仍是规则化的**：枚举/桥接/时序的触发与槽位映射是手写正则 + 词表；LLM 化 query decomposition（结构化输出、子问题验收）是路线图。

---

## 10. 复现

```bash
cd 上下文记忆/memstack

# LoCoMo 300 题综合（10 种方法，含 MemStack-mr 同库对照）
python public_eval.py --dataset locomo --limit 300 --budget 800 --turns 3 --episode-chars 900 \
  --out results/locomo_300_comprehensive.json

# LongMemEval oracle 500 题综合
python public_eval.py --dataset longmemeval --budget 800 --turns 2 --episode-chars 700 \
  --out results/longmemeval_comprehensive.json

# LoCoMo 全量 1986 题（v4.1；约 15 分钟）
python public_eval.py --dataset locomo --budget 800 --turns 3 --episode-chars 900 \
  --no-download --out results/locomo_full_comprehensive.json

# QA 层：本地 0.8B reader + judge（离线）
python qa_eval.py --limit 50 --budget 600 --methods memstack,memstack-ep,bm25 \
  --reader hf --judge hf --out results/locomo_qa.json

# QA 层：官方 mem0 仓库基线（需 pip install mem0ai；单独跑，避免同进程退出崩溃）
pip install mem0ai
python qa_eval.py --limit 50 --budget 600 --methods mem0 \
  --reader hf --judge hf --out results/locomo_qa_mem0.json

# QA 层：v4 证据链 vs v3 词袋补证（零依赖 extractive 口径；hf 模型可用时应换回 --reader hf --judge hf）
python qa_eval.py --limit 50 --budget 600 --methods memstack,memstack-mr,bm25 \
  --reader extractive --judge f1 --out results/locomo_qa_v4.json

# 回归
python -m unittest discover -s tests -v
```

---

## 11. 参考

- Mem0: Chhikara et al., *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*, arXiv:2504.19413；<https://github.com/mem0ai/mem0>
- Zep/Graphiti: Rasmussen et al., arXiv:2501.13956；<https://github.com/getzep/graphiti>；<https://www.getzep.com/research/>
- A-MEM: Xu et al., *A-MEM: Agentic Memory for LLM Agents*, arXiv:2502.12110；<https://github.com/agiresearch/A-mem>
- MemoryOS: <https://github.com/BAI-LAB/MemoryOS>
- SimpleMem: <https://github.com/aiming-lab/SimpleMem>
- LoCoMo: Maharana et al., ACL 2024, arXiv:2402.17753；<https://github.com/snap-research/locomo>
- LongMemEval: Wu et al., ICLR 2025, arXiv:2410.10813；<https://github.com/xiaowu0162/LongMemEval>
- LoCoMo 评测审计（标注错误/judge 问题）：<https://agentos.sh/blog/memory-benchmark-transparency-audit/>
