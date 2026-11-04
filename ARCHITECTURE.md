# MemStack 系统架构

> 目标：把 Agent 记忆中的写入、检索、更新、遗忘与评测，收敛成一个**可移植、可运行、可解释、可评测**的记忆层。
>
> 本文回答两件事：**当前系统长什么样**，以及**为什么这么构建、放弃了什么**。

---

## 0. 30 秒版

MemStack 是一条六层记忆流水线：

```text
Agent / HTTP / 工具调用
        ↓
┌──────────────────────────────────────────────────────────────┐
│ 1 接入层    Scope（tenant/user/agent/session）+ 5 个工具       │
│ 2 记忆类型  episode / fact / preference / procedure / summary  │
│            / entity / reflection                              │
│ 3 写入管线  归一化 → 分段 → 抽取 → ADD/MERGE/SUPERSEDE/NOOP    │
│            → 实体链接 + 溯源边 + 相似边                        │
│ 4 存储索引  SQLite：memories / edges / blocks / audit          │
│ 5 检索组装  dense + BM25 + entity + prior + PPR               │
│            → 熵路由 → 时间策略 → 重排 → MMR → token 预算       │
│ 6 离线巩固  去重 / 冲突消解 / 会话摘要 / 衰减归档 / 画像块      │
└──────────────────────────────────────────────────────────────┘
```

一句话总结：**写入时保留原文并抽取知识，检索时多信号融合并按预算现拼上下文，更新时追加历史而不是覆盖，遗忘时级联隐藏证据。**

---

## 1. 设计目标与边界

### 1.1 要解决的问题

Agent 的记忆需求可以拆成五个失败模式：

| 失败模式 | 典型症状 | MemStack 的对应机制 |
|---|---|---|
| **记不住** | 换会话就忘记用户偏好、项目约束 | 长期存储 + 用户画像块 |
| **找不到** | 纯向量检索漏掉专名、数字、时间、跨会话关联 | dense + BM25 + entity + PPR |
| **记错** | 用户搬家/换工作后仍回答旧值 | 双时间 + `superseded` + 历史查询 |
| **塞不下** | 把整段历史塞进上下文，token 爆炸、中间遗忘 | 多粒度 + 重排 + MMR + token 预算 |
| **删不掉** | 删除事实后原文仍能被检索到 | 软删除 + 级联隐藏 episode/摘要 + 审计 |

### 1.2 设计目标

1. **可移植**：不绑定任何 Agent 框架、向量数据库或模型供应商；核心只依赖 Python 标准库。
2. **可解释**：每个注入项都带 id、kind、状态、时间、来源、分数。
3. **可评测**：内置合成数据、基线、消融、陈旧回答率与 token 成本。
4. **可渐进增强**：默认规则抽取 + hashing 向量即可跑；生产可替换 LLM、BGE、Postgres、Neo4j。
5. **可治理**：作用域隔离、软删除、级联遗忘、审计日志。

### 1.3 非目标（当前版本不做）

- 不做 Agent 运行时（不替代任何编排框架）。
- 不做分布式图数据库 / 向量数据库；单机 SQLite 是默认实现，不是上限。
- 不做端到端可微记忆训练（可学习更新策略是路线图，不是现状）。
- 不做多模态、代码仓库状态、团队级 ACL、加密存储。
- 不声称对标任何厂商公布分数；合成基准只验证机制趋势。

---

## 2. 设计取舍速览

MemStack 不追求“又一个记忆 API”，而是把记忆层拆成可以单独开关、替换、评测的机制集合。几个关键取舍如下。

### 2.1 机制取舍

| 取舍点 | 选择 | 放弃的方案 | 原因 |
|---|---|---|---|
| 存储形态 | 可嵌入库 + HTTP / MCP | 托管服务 | 便于嵌入现有 Agent，也便于做受控实验 |
| 记忆单元 | episode + fact / preference / procedure / summary | 只存摘要；只存向量块 | 证据与知识职责不同，单层必然顾此失彼 |
| 更新语义 | 追加式 + 双时间 + 状态机 | 直接覆盖文本 | 历史查询、审计、冲突解释都需要旧值 |
| 检索 | 五通道融合 + 图扩展 | 纯向量 | 向量对专名、数字、时间不敏感 |
| 上下文 | 预算内现拼 + 来源标注 | 全量历史塞入 | token 成本与可解释性 |
| 依赖 | Python 标准库 + SQLite | 图库 / 向量库 / Agent 框架 | 降低接入门槛，规模上来再替换 |
| 抽取 | 规则抽取默认，LLM 可插拔 | 默认调 LLM | 离线可跑、确定性、零 API 成本 |

### 2.2 与外部系统的关系

MemStack 面向通用 Agent 记忆需求设计，不依赖任何特定 Agent 框架、模型供应商或向量数据库。需要替换实现时，按 `memstack/interfaces.py` 的 Protocol 注入即可，不需要修改核心代码，也不需要安装额外依赖。

---

## 3. 总体架构

### 3.1 六层视图

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 接入层                                                              │
│  Python API  ·  OpenAI tools  ·  HTTP gateway  ·  MCP server(stdio)  │
│  MemoryMiddleware  ·  MultiAgentHub（多 Agent 协作层，v0.3）          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Scope(tenant,user,agent,session)
┌───────────────────────────────▼─────────────────────────────────────┐
│ 门面层 MemoryStack                                                  │
│  remember / recall / search / forget / consolidate / blocks / export │
└───────────────┬─────────────────────────────────────┬───────────────┘
                │                                     │
┌───────────────▼──────────────────┐   ┌──────────────▼───────────────┐
│ 写入管线                         │   │ 检索管线                      │
│ TopicSegmenter                   │   │ HybridRetriever               │
│ Extractor (rule / LLM)           │   │ 5 channels + entropy routing  │
│ ConflictResolver (ADD/MERGE/...) │   │ temporal policy + rerank      │
│ GraphLinker (entity/provenance)  │   │ MMR + budget + confidence     │
└───────────────┬──────────────────┘   └──────────────┬───────────────┘
                │                                     │
┌───────────────▼─────────────────────────────────────▼───────────────┐
│ 存储与索引层                                                        │
│ SQLiteStore: memories / edges / blocks / audit / meta               │
│ 逻辑索引: 向量（JSON）、BM25（查询时构建）、实体图、时间窗、状态机    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ 离线巩固层 Consolidator                                             │
│ dedupe / resolve_conflicts / summarize_sessions / decay / profile    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 一次写入的生命周期

```text
messages
  │ normalize（str / dict / list）
  ▼
TopicSegmenter（最大轮数 + 相邻语义相似度切段）
  │ 每个 segment → episode（原文 + 哈希去重）
  ▼
Extractor（规则 / LLM）
  │ fact / preference / procedure / count / allergy ...
  ▼
_ensure_slot_keys → (subject_key, predicate_key, object_key, polarity)
  │
  ├─ 同槽位同值        → MERGE（合并来源、提高置信度）
  ├─ 同槽位不同值      → SUPERSEDE（旧值 valid_to=新值 valid_from）
  ├─ 语义高度重复      → MERGE
  └─ 其他              → ADD
  ▼
GraphLinker：mentions→entity、provenance→episode、similar→近邻、supersedes→旧值
  ▼
audit + stats
```

### 3.3 一次读取的生命周期

```text
query + scope
  │
  ├─ 时间意图判断：current / history / 明确年份
  ├─ 查询实体识别 + 同义词扩展（where→location/moved；work→occupation/employer）
  ▼
候选集（按 scope + status + 时间窗过滤，最多 1200 条）
  │
  ├─ dense（hashing / 可替换）
  ├─ BM25（轻量词干化）
  ├─ entity overlap
  ├─ prior（importance × confidence × recency × access）
  ▼
融合 → 熵路由（kind 级软权重）→ 意图谓词 boost → 历史 boost
  ▼
PPR 图扩展（seed top-5；多跳查询加权）
  ▼
重排（query coverage + entity + slot + predicate + status）
  ▼
MMR 去重 → top-K → token 预算打包
  ▼
ContextBundle：text + items + confidence + coverage + gaps + conflicts + trace
```

---

## 4. 数据模型

### 4.1 记忆类型

| kind | 认知类比 | 作用 | 是否直接进上下文 | 典型字段 |
|---|---|---|---|---|
| `episode` | 情景记忆 | 原始交互窗口，可验证证据 | 是（压缩后） | messages、turns、topic |
| `fact` | 语义记忆 | 原子事实命题 | 是 | subject/predicate/object、entities |
| `preference` | 语义记忆子类 | 用户偏好，需时效/冲突 | 是 | topic、polarity |
| `procedure` | 程序记忆 | 可复用流程 | 是 | intent、steps |
| `summary` | 导航层 | 会话/主题摘要 | 是 | session_id、episodes |
| `entity` | 索引节点 | 实体桥接，用于图扩展 | **否** | canonical |
| `reflection` | 高层结论 | 离线归纳 | 是 | 预留 |

为什么把 `preference` 从 `fact` 里单列：偏好的更新频率高、需要 `topic` 维度、需要正负极性；如果和普通事实混在一起，`I prefer X` 与 `I live in Y` 会共享冲突逻辑，容易误取代。

### 4.2 统一 `MemoryItem`

```python
MemoryItem(
    id, kind, text, scope,
    data,            # 结构化槽位、entities、steps、number...
    status,          # active | superseded | archived | deleted
    created_at, updated_at,
    valid_from, valid_to,   # 事件时间（事实在现实中何时为真）
    last_access, access_count,
    importance, confidence,
    embedding,
    source_ids,      # 溯源：这条知识来自哪些 episode
    meta,            # content_hash、aliases、expires_at...
)
```

关键点：

- **`text` 保留自然语言原话**，便于证据引用；**`data` 存结构化槽位**，便于冲突判断与精确检索。
- **`source_ids` 是溯源链**：fact → episode；summary → episodes。删除/审计/证据回取都靠它。
- **`scope` 是四元组**：`tenant_id / user_id / agent_id / session_id`。前三者决定隔离，`session_id` 只用于会话内分组，跨会话检索时留空。

### 4.3 双时间与生命周期

```text
事实 A：user.location = Lisbon
  valid_from = 2026-04-28   ← 事件时间
  created_at = 2026-04-28   ← 写入时间
  status = active

事实 B：user.location = Porto
  valid_from = 2026-09-28（“last month”解析）
  created_at = 2026-10-28（写入时间）
  status = active

发生 SUPERSEDE 后：
  A.status = superseded
  A.valid_to = 2026-09-28
  B.status = active
  edge: B --supersedes--> A
```

- **当前查询**只取 `active`，旧值不参与排序。
- **历史查询**（`before / used to / in 2024`）取 `active + superseded + archived`，并给非 active 项加权。
- **删除**默认是 `deleted`（软删除），物理删除需要 `hard=True`。

为什么不用“直接 UPDATE 文本”：

1. 历史问题（“我以前住哪”）无法回答；
2. 冲突消解没有审计轨迹；
3. 多跳/摘要可能仍引用旧证据，覆盖会导致不一致；
4. 历史版本需要保留，但检索时只让当前值参与排序，二者不能混为一谈。

### 4.4 图边

| 边 | 方向 | 用途 |
|---|---|---|
| `provenance` | 派生记忆 → episode | 证据回取、级联遗忘、引用 |
| `mentions` | 记忆 → entity | 实体桥接、多跳 |
| `similar` | 记忆 ↔ 记忆 | 近邻扩散、相关证据 |
| `supersedes` | 新事实 → 旧事实 | 时间链、审计 |
| `derived_from` | summary → episode | 摘要溯源 |
| `next` | episode → episode | 预留时序链 |

图不是独立数据库，而是 `memories + edges` 上的逻辑视图；PPR 在查询时构建邻接表。这样存储层可以整体替换（SQLite → Postgres → Neo4j），图算法不变。

### 4.5 SQLite 表

```sql
memories(id, kind, text, tenant_id, user_id, agent_id, session_id,
         data, status, created_at, updated_at, valid_from, valid_to,
         last_access, access_count, importance, confidence,
         embedding, source_ids, meta, content_hash)

edges(src, dst, type, weight, created_at, meta,
      PRIMARY KEY(src, dst, type))

blocks(tenant_id, user_id, agent_id, label, value, "limit", updated_at,
       PRIMARY KEY(tenant_id, user_id, agent_id, label))

audit(id, ts, op, scope, detail)
```

索引：`(tenant_id,user_id,agent_id,session_id)`、`(kind,status)`、`content_hash`、`edges(src)`、`edges(dst)`。

---

## 5. 写入管线

### 5.1 归一化与分段

输入支持：

```python
stack.remember("I live in Lisbon.", scope)
stack.remember({"role": "user", "content": "..."}, scope)
stack.remember([{"role": "user", "content": "..."}, ...], scope)
```

分段策略（`TopicSegmenter`）：

1. 相邻消息向量相似度低于 `topic_split_threshold`，且当前段长度 ≥ 2，则在成对边界切分；
2. 当前段达到 `max_turns_per_episode` 强制切分；
3. 过短尾段并回上一段，避免碎片。

为什么不用固定 token 窗口：固定窗口会切断语义单元，主题分段与 turn/session 粒度都在解决这个问题。为什么还保留最大轮数：防止长会话形成超大 episode，影响预算与压缩。

### 5.2 抽取

默认 `RuleExtractor`（离线、确定性）：

| 类型 | 规则示例 | 输出槽位 |
|---|---|---|
| 身份 | `my name is X` / `我叫X` | `user.name` |
| 地点 | `I live in X` / `moved to X` / `我住在X` | `user.location` |
| 历史地点 | `I used to live in X` | `user.previous_location` |
| 职业 | `I work as a X` / `my new role is X` | `user.occupation` |
| 雇主 | `I work at X` / `work as ... at X` | `user.employer` |
| 过敏 | `allergic to X` / `我对X过敏` | `user.allergy`（多值） |
| 偏好 | `I like/love/prefer X` / `我喜欢X` | `user.preference:<topic>` |
| 计数 | `completed N courses on X` | `user.completed_count:<unit>:<platform>`（多值） |
| 项目 | `Project Atlas uses X` | `project:atlas.stack` |
| 组织 | `X is headquartered in Y` | `x.headquarters` |
| 流程 | `first ... then ... finally ...` | `user.procedure:<intent>` + `steps` |

生产环境实现 `Extractor` 协议即可换成 LLM：

```python
from memstack import LLMExtractor, OpenAICompatClient

client = OpenAICompatClient(model="gpt-4o-mini")
stack = MemoryStack(extractor=LLMExtractor(client))
```

`LLMExtractor` 失败会自动回退到规则抽取器，避免“模型抖动导致整条写入失败”。

### 5.3 冲突消解：ADD / MERGE / SUPERSEDE / NOOP

核心伪代码（`core.py::_write_extracted`）：

```python
candidates = store.find(scope, kind, status=ACTIVE)
best = argmax_similarity(candidates, new)
slot_match = first(c for c in candidates
                   if c.subject_key == new.subject_key
                   and c.predicate_key == new.predicate_key)

if slot_match and same_object(slot_match, new):
    MERGE
elif slot_match and not multi_valued and object_differs:
    SUPERSEDE
elif best.similarity >= duplicate_threshold:
    MERGE
else:
    ADD
```

四条规则的白话解释：

1. **同槽位同值**：用户重复说“我花生过敏” → 合并来源，不新增条目；
2. **同槽位不同值**：用户说“我搬到 Porto” → 旧地点 `superseded`，新地点 `active`；
3. **没有槽位但语义高度重复**：两句措辞不同但意思一样 → 合并，保留一个主文本 + aliases；
4. **其他**：新增。

多值例外：`allergy`、`diet_restriction` 标记 `multi=True`，不同对象不互相取代，只做同值合并；`completed_count` 把平台/单位编进 predicate（`completed_count:courses:coursera`），因此同一平台的次数更新会走 SUPERSEDE，不同平台则自然并存。

### 5.4 乱序写入与“变更”语义

双时间带来一个容易忽略的问题：**新写入的 fact 可能携带更早的事件时间**。

```text
先写：I live in Lisbon        （event_time = 2026-04-28）
后写：I moved to Porto last month（event_time = 2026-09-28）
```

这很正常。但如果顺序反过来：

```text
先写：I moved to Porto last month（event_time = 2026-09-28）
后写：I live in Lisbon            （event_time = 2026-04-28）
```

后者其实是更早的历史，不应该把 Porto 取代掉。因此：

- 若 `new.valid_from < old.valid_from` 且新句不是“变更语义”（moved/now/new）→ 新记忆直接标为 `superseded`，旧记忆保持 active；
- 若是“变更语义”（`is_change=True`）→ 以新写入为准，旧记忆失效，避免“用户刚纠正但时间解析偏差”导致纠正失败。

这是启发式，不是完整的双时间推理；局限见 §12.2。

### 5.5 溯源与实体链接

写入后 `MemoryGraph.link()`：

1. 每个实体 → 创建/复用一个 `entity` 节点（id 由 scope + 实体名哈希决定）；
2. 记忆 → entity 建 `mentions` 边；
3. 派生记忆 → `source_ids` 建 `provenance` 边；
4. 与同 scope 的 active 记忆做近邻，超过 `similar_threshold` 建 `similar` 边（最多 `max_similar_edges` 条）；
5. `SUPERSEDE` 时建 `supersedes` 边。

实体节点不直接进上下文，只用于 PPR 桥接。这样“Project Atlas”既能连接 `stack` 事实，也能连接 `lead` 事实，支持“由某人领导的项目用什么数据库”这类两跳问题。

---

## 6. 检索与上下文组装

### 6.1 候选集与时间策略

```python
statuses = [ACTIVE]
if use_temporal and history_mode:
    statuses = [ACTIVE, SUPERSEDED, ARCHIVED]

candidates = store.find(scope, kinds, statuses, since, until, limit=1200)
```

`history_mode` 的判断：

- 显式 `include_history` 优先；
- 出现 `now / currently / 现在` → current；
- 出现 `used to / before / previously / in 2024 / 以前` → history；
- 否则 current。

如果查询里出现明确日期，会构造一年窗口（`valid_from <= until AND (valid_to IS NULL OR valid_to >= since)`），支持“2024 年我在哪”这类时间点查询。

### 6.2 五通道融合

| 通道 | 计算 | 解决什么 |
|---|---|---|
| `dense` | query 向量与记忆向量的 cosine | 语义相似 |
| `lexical` | BM25（k1=1.2, b=0.75） | 专名、数字、精确词 |
| `entity` | query 实体 ∩ 记忆实体 | 人名/项目/公司桥接 |
| `prior` | `importance × confidence × recency × log(access)` | 重要性、时效、热度 |
| `graph` | PPR 分数 | 多跳、溯源、相关证据 |

默认权重：

```python
{"dense": 0.46, "lexical": 0.26, "entity": 0.08, "prior": 0.12, "graph": 0.08}
```

为什么不是纯向量：`Where does Nora live?` 与 `I moved to Porto last month.` 在纯向量下可能不如 `My name is Nora.` 相似；BM25 的词干化（live/moved）和谓词意图 boost 才能把地点事实顶上来。这也是混合检索在长对话记忆里普遍优于纯向量的原因。

### 6.3 多粒度熵路由

对每个 `kind` 分组，把组内融合分数做 softmax，计算熵：

```text
H_g = -Σ p_i log p_i
confidence_g = 1 - H_g / log(n_g)
weight_g = exp(confidence_g) × intent_prior_g
```

- 熵低（该粒度明确指向少数记忆）→ 权重高；
- 熵高（该粒度模糊匹配很多记忆）→ 权重低；
- 再乘查询意图先验：`how to`→procedure，`prefer`→preference，`how many`→count，`summarize`→summary。

**与硬路由的差异**：硬路由直接把 `w_g ∝ 1/H_g` 当作主信号；本实现把路由做成**温和先验**：

```python
base *= 0.85 + 0.15 * normalized_kind_weight
```

原因：在候选数少、粒度异构的小规模场景里，熵估计不稳定（一个 kind 只有 2 条也会得到“高置信”），硬路由会把正确粒度压掉。合成基准中，关闭路由准确率不变，说明当前权重是“不伤害”的保守设置；在更大规模数据上可通过 `weights` / `kind_prior_strength` 调高。

### 6.4 图扩展（PPR）

1. 取融合分数 top-`seed_k`（默认 5）作为种子；
2. 在 `allowed_ids`（同 scope 的全部节点）上构建无向加权邻接表；
3. 边权 = 类型权重 × 边权：
   - provenance 1.0、derived_from 0.9、mentions 0.55、similar 0.5、supersedes 0.3、next 0.4；
4. 带重启随机游走（damping=0.85, 20 次迭代）；
5. 归一化后按 `graph` 权重加到候选分数；
6. 多跳查询（`headquarters / company / whose / works for`）把图权重 ×2.5。

为什么 PPR 而不是简单邻居扩散：PPR 能同时考虑多条路径和节点度数，避免高 degree 实体节点淹没结果；这是图记忆系统的常用做法。

### 6.5 重排

启发式 rerank（无 cross-encoder 时）：

```text
rerank = 0.30 × query coverage
       + 0.10 × entity overlap
       + 0.05 × status bonus
       + 0.05 × kind prior
       + 0.15 × slot bonus
       + 0.35 × predicate match
final = 0.70 × fused + 0.30 × rerank
```

谓词匹配是最强信号：`Where...live` → location；`work/job` → occupation/employer；`allergic` → allergy；`how many` → completed_count；`database/stack` → stack；`headquarters/company` → headquarters。

### 6.6 MMR 去重

```text
value(i) = λ × relevance(i) − (1−λ) × max_sim(i, selected)
```

默认 `λ=0.75`。作用：

- 防止同一事实的多个近邻/多个 episode 占满预算；
- 保留“旧值 + 新值”这类必要的时间对照（它们在语义上相近但时间状态不同，分数差异足够大时仍会被选中）。

### 6.7 预算打包

`ContextBundle` 的组装顺序：

```text
## Core memory
### user_profile
- user.location = Porto
...

## Retrieved memory
- [FACT|score=0.98|active|from=...|src=ep-...] slot=user.location | value=Porto | evidence="..."
- [EPISODE|score=0.61|active|...] ...（只保留与 query 最相关的 1–2 句）
...
```

- 先放常驻 block，再逐条加记忆，超过 `token_budget` 就截断/停止；
- episode 会按 query 词覆盖度选句压缩，而不是整段塞入；
- 每条都带 `src`，agent 可以引用来源；
- 返回 `token_estimate`，便于上层做成本核算。

### 6.8 置信度、缺口与“继续检索”

```python
confidence = 0.40 × top_score
           + 0.20 × score_margin
           + 0.40 × query_coverage
           − 0.15 × conflicts
```

- `coverage`：query 内容词在上下文中的覆盖率；
- `margin`：top1 与 top2 的差距；
- `conflicts`：同一单值槽位出现多个 active 值（正常写入不会出现，导入/外部数据可能出现）；
- `gaps`：没有命中、覆盖不足、query 实体缺少证据；
- `should_continue = confidence < 0.42 and (gaps or no hits)`。

这就是“熟悉度不足就继续回忆、出现冲突就继续检索”的双信号设计。注意：`should_continue` 只是建议，不强制多轮；Agent 可以据此调用 `memory_search` 继续查证。

### 6.9 证据链检索（evidence chain，v4）

v2 的多轮补证（multi_round）只会把首轮命中里的实体词拼回 query 再查一遍，实测证明瓶颈在"扩展词不精准"：LoCoMo multi-hop 的 All-Unit 停在 27.1%。v4 换成**query 结构感知的证据链**：把复杂查询分解为结构化子查询，每个子查询独立检索、独立名额，合并时保留首轮证据、只补缺口。

代码：`core.py::_evidence_chain_recall` + `memstack/evidence_chain.py`。

```text
第一轮：常规混合检索 → top-k
  ↓ needs_chain(query)？四种触发形态：
  ├─ 枚举聚合（What hobbies does Evan pursue? / how many ... / what did X VERB?）
  ├─ 图谱桥接（headquarters / whose / employed by / lead of ...）
  ├─ 行为动词尾（What did Caroline research? → 尾部动词即槽位词）
  └─ 时间锚点（after/before + 事件）
分解：锚定主体（query 专名优先，其次首轮命中高频实体）
      + 槽位同义词扩展（hobby→hobbies/pursue/enjoy/activities ...），最多 2 个子查询
  ↓ 每个子查询独立检索（共享 store，touch=False）
  ↓ 双路径扩证：首跳最高分 < evidence_chain_min_top_score（熟悉度不足）→
     PPR 近邻扩证（similar/provenance/mentions 边上的补证，而非盲查整库）
合并：保留首轮 top-(k − n_sub × slots)，每个子查询只补 slots 个全新证据
      （枚举类 4 个名额、其他 2 个；补证分数 ×0.9 与首轮尾部公平竞争后取 top-k）
```

设计要点：

- **只补缺口名额**：子查询召回的证据若已在首轮（id 相同）算重复、不占名额；不存在 v2"补证挤掉首轮高分证据"的问题（单测覆盖 `test_keeps_first_round_evidence`）。
- **二次检索面向槽位而非词面**：枚举轮用"主体 + 槽位同义词"检索，把分散在多个会话的长尾取值顶上来；v2 的实体扩词会漂移到已命中会话的同款词汇。
- **触发保守**：普通单跳查询（`Where does Nora live now?`）不触发，延迟不受影响。
- **兼容开关**：`evidence_chain=False` 回退到 v2 多轮补证（`multi_round`）；`public_eval.py` 同时保留两种配置（`MemStack` / `MemStack-mr`）做同库对照。

实测（LoCoMo 300 题分层，800-token，`MemStack-mr` 为同库对照）：

| 指标 | MemStack-mr（v3 词袋补证） | MemStack（v4 证据链） |
|---|---:|---:|
| multi-hop All-Unit | 27.1% | **28.8%**（+1.7pp） |
| multi-hop TurnToken | 48.9% | **51.6%**（+2.7pp） |
| multi-hop AnsToken | 22.2% | **26.4%**（+4.2pp） |
| 总体 TurnHit | 54.5% | **57.2%**（+2.7pp） |
| 总体 All-Unit | 68.0% | **69.0%**（+1.0pp） |
| 每查询延迟 | 76.5ms | 100.7ms（+24ms） |

LongMemEval oracle 500 上中性不退化。multi-hop All-Unit 仍是天花板问题——枚举题的 gold 证据散布在 7–19 个会话，800-token 预算内很难全部召回；但 TurnToken +2.7pp / AnsToken +4.2pp 说明逐字证据确实更多进入了上下文。

**遗留的多轮补证与证据回填（v2，作为兼容路径）**：`evidence_chain=False` 时启用词袋扩词（保留 top-(k-2)、补 2 个名额）；证据回填对 top-3 fact/preference/procedure 回取 provenance episode 附加 `source="..."`（最多 320 字符），LoCoMo 200 题 TurnHit +0.6pp、token +41。

**v4.1 增量（本轮）**：

1. **链内共享缓存**：`HybridRetriever.search(..., _shared=...)` 支持同 scope/状态/时间窗下复用候选集、实体节点与 PPR 图分数（种子在链内各跳高度重叠）。子查询延迟 53.5ms→16.9ms（约 -69%）；LoCoMo 100 题实测 MemStack 与 MemStack-mr 的延迟差距从 +24ms 收窄到 +9ms。缓存 key 含数据 revision，任何写入自动失效。
2. **status-aware 打包**：`status_aware_packing=True`（默认）时，预算不足的裁剪路径中 active 记忆优先进入上下文、superseded/archived 历史值放尾部；预算充足时顺序与 hits 完全一致。即“生命周期驱逐”——旧值不挤占当前值。
3. **`LLMQueryDecomposer`**：证据链分解器可插拔。任意 OpenAI 兼容客户端（仅需 `complete(prompt, system)`）返回 JSON 数组即被采纳；输出校验（结构、reason 枚举、text 长度、markdown fence 容忍）失败或调用异常时自动回退规则版 `decompose_subqueries`，检索链路不断。默认不启用（零依赖）；`trace["evidence_chain"]["decomposer"]` 记录实际路径（llm / rule）。

### 6.10 多 Agent 协作层（MultiAgentHub，v0.3）

记忆本体仍全部存于 memories 表，用 Scope 表达归属，不复制存储：

- **私有记忆**：`Scope(tenant, user_id="agents", agent_id=<agent>)`；
- **团队共享**：`Scope(tenant, user_id="team:<team_id>", agent_id="shared")`；
- 元数据（agent 档案 / 团队 / 成员 / 授权 / 事件）放独立 hub 表，默认与主库同文件。

权限模型：

| 能力 | 规则 |
|---|---|
| 团队读 | open 团队人人可读；closed 团队仅成员 |
| 团队写 | 仅成员（事件发布同） |
| 跨 agent 读私有 | 需显式 `grant(owner, principal, "read")` |
| 事件总线 | `post_event` → `pull_events(team, since_id)` 增量拉取，可选同步写为共享记忆 |

`recall_as` 把私有 + 各团队共享 + 被授权他人记忆合并为带 `[agent-private]/[team:x]/[granted:y]` 来源段落的单一 ContextBundle，与现有检索/巩固/遗忘管线完全兼容。

---

## 7. 离线巩固（sleep-time）

`stack.consolidate(scope)` 执行：

### 7.1 去重

按 `(kind, subject_key, predicate_key, object_key, polarity)` 分组，同组保留最新/最高置信度的一条，其余标记 `superseded`，合并 `source_ids` 与 aliases。

### 7.2 冲突消解

对 `fact/preference` 中**单值**且同槽位多对象的分组，保留 `valid_from` 最新的一条，其余 `superseded`。这主要修复：

- 外部导入的数据；
- LLM 抽取漏判冲突；
- 手工写入/迁移。

### 7.3 会话摘要

每个 session 的 episode 数 ≥ `summary_min_episodes` 时，生成一条 `summary`：

- `data.session_id`、`data.episodes`；
- `source_ids` = 所有 episode；
- 边 `derived_from` → episode；
- 摘要由 `Summarizer` 生成：默认抽取式（数字/实体/偏好句优先），可换 `LLMSummarizer`。

摘要用于“跨会话概览 / 导航”，不替代原文证据。

### 7.4 衰减与归档

```text
decayed = importance × 0.5^(age / decay_half_life)
```

- 支持 `meta.expires_at` 显式 TTL；
- `episode/summary/reflection` 超过 `archive_after_days`、`access_count == 0`、`decayed < 0.15` → `archived`；
- 稳定事实（如过敏）不会仅因时间久远被归档；只归档低价值、无访问的叙事记忆。

### 7.5 画像块

从 active 的 fact/preference 中选核心槽位（name/location/occupation/employer/allergy/diet/preference），生成 `user_profile` block（字符上限 `profile_max_chars`）。效果是**每次请求都可以直接看到**，不需要检索。

---

## 8. 可移植性设计

### 8.1 五个 Protocol

`memstack/interfaces.py`：

| Protocol | 默认实现 | 生产替换 |
|---|---|---|
| `Embedder` | `HashingEmbedder`（512 维，确定性） | BGE / OpenAI / 自研 |
| `Extractor` | `RuleExtractor` | `LLMExtractor` / 领域模型 |
| `Summarizer` | `RuleSummarizer` | `LLMSummarizer` |
| `Reranker` | 启发式（内置） | cross-encoder / ColBERT |
| `Clock` | `SystemClock` | 测试用模拟时钟 |
| `MemoryStore` | `SQLiteStore` | Postgres / Redis / 向量库 / 图库 |

核心代码只依赖这些协议，不 import 任何第三方框架。

### 8.2 工具接口（OpenAI function-calling）

`stack.as_tools()` 返回 5 个工具：

| 工具 | 作用 |
|---|---|
| `memory_search` | 检索相关记忆，返回带来源的上下文 |
| `memory_add` | 写入对话/事件/文档 |
| `memory_forget` | 按 id 或查询遗忘（软/硬） |
| `memory_profile` | 读取画像块与库统计 |
| `memory_consolidate` | 触发离线巩固 |

`dispatch_tool` 可直接接 OpenAI / Anthropic / 自研 function-calling 循环；`MemoryMiddleware` 提供 `before(query)` 注入上下文、`after(messages)` 写回交互的最小封装。

### 8.3 HTTP 网关

`python -m memstack.server` 启动标准库 `ThreadingHTTPServer`，端点见 README。任何语言/框架都可以通过 HTTP 接入；生产可加反向代理、鉴权、限流。

### 8.4 作用域与隔离

```python
Scope(tenant_id="acme", user_id="nora", agent_id="assistant", session_id="s3")
```

- 检索默认要求 tenant/user/agent 完全匹配；
- `session_id` 只用于写入分组与摘要，跨会话检索时留空；
- 团队共享记忆可通过相同 `tenant_id + agent_id`、不同 `user_id` 实现“共享 Agent 记忆”，但当前没有 ACL/可见性分级（路线图）。

### 8.5 替换后端时不需要改什么

- 换 Embedder：写入时重新 embed 旧数据（提供 `export/import` 或迁移脚本）；
- 换 Store：实现 `MemoryStore` 协议（save/get/find/update_status/edges/blocks/audit/export 等约 18 个方法）；
- 换 Extractor：实现 `extract(text, event_time, context)`；
- 换 Reranker：实现 `rerank(query, candidates)`。

---

## 9. 为什么这么构建（决策表）

| # | 决策 | 备选方案 | 为什么选它（技术依据） |
|---:|---|---|---|
| 1 | 分层：episode / fact / preference / procedure / summary / entity | 只存对话摘要；只存向量块 | 不同记忆的生命周期、冲突规则、检索方式不同，单层必然顾此失彼 |
| 2 | 原文 episode 永久保留，派生知识另存 | 只存摘要（省空间）；只存原文（省抽取） | 摘要是导航，原文是证据；只存摘要会丢数字/时间/细节，只存原文检索噪声大 |
| 3 | 追加式 + 双时间，不物理覆盖 | UPDATE 文本；只留最新 | 历史查询、审计、冲突解释都需要旧值；覆盖会不可逆丢信息 |
| 4 | 混合检索（dense+BM25+entity+prior+graph） | 纯向量；纯 BM25 | 向量对专名/数字/时间弱，BM25 对改写弱，实体对多跳弱 |
| 5 | 熵路由做“温和先验”而非硬路由 | 硬选一个粒度；固定权重 | 小规模下熵估计不稳定；硬路由会误杀正确粒度 |
| 6 | PPR 图扩展，实体节点只做桥接 | 邻居扩散；社区摘要 | PPR 考虑多路径与度数；实体桥接适合“项目→负责人→公司”两跳 |
| 7 | 时间策略自动切换 current/history | 永远只查 active；永远带历史 | 大多数问题问当前，历史问题需要旧值；自动判断减少调用方负担 |
| 8 | 离线巩固而不是在线全量维护 | 每次写入都去重/摘要 | 在线写入要快；去重/摘要/冲突消解是 O(N) 级操作，放睡眠期更合理 |
| 9 | 常驻 `user_profile` block | 全部靠检索 | 身份/偏好是高频、稳定、低歧义信息，常驻比每次检索更稳 |
| 10 | 级联遗忘（事实→证据→摘要） | 只删事实行 | 只删事实会让原文 episode 继续泄漏；隐私要求“删得干净” |
| 11 | 规则抽取默认 + LLM 可插拔 | 默认调 LLM | 离线可跑、可测试、零 API 成本；生产可无缝换 LLM |
| 12 | 纯标准库 + SQLite | 直接上 Postgres/Neo4j/Chroma | 降低接入门槛；协议留好，规模上来再换 |
| 13 | 返回 `ContextBundle` 而非字符串 | 只返回拼接文本 | 上层需要置信度、来源、缺口、是否继续检索；可观测性是一等需求 |
| 14 | 工具 + HTTP + Middleware 三种接入 | 只提供 Python SDK | 不同 Agent 框架接入方式不同；HTTP 支持跨语言 |
| 15 | 复杂查询走证据链（分解→逐跳→分名额合并） | 词袋扩词补一轮；单次 top-K | 扩词会漂移到已命中会话；链式子查询面向槽位、只补缺口名额，首轮证据不被挤掉 |

---

## 10. 适用边界与选型

MemStack 的定位是**可拆开研究的参考架构**：每个机制都能单独关闭、替换、评测；默认零依赖，便于嵌入任何 Agent 或做受控实验。

| 场景 | 是否适合 | 说明 |
|---|---|---|
| 自托管 Agent、需要审计与时间线 | **适合** | 双时间、状态机、溯源边、审计日志都是一等公民 |
| 需要把记忆层嵌进现有 Agent | **适合** | Python API / 工具调用 / HTTP / MCP 四种接入 |
| 需要替换存储、Embedder 或抽取器 | **适合** | 核心只依赖 Protocol，替换不改核心代码 |
| 受控实验、消融、检索层评测 | **适合** | 每个机制有独立开关，内置合成基准与公开数据评测 |
| 千万级向量、低延迟 ANN 检索 | 不适合 | 默认 SQLite + 暴力扫描；需替换 `MemoryStore` 并接 ANN 索引 |
| 需要托管服务、多租户计费 | 不适合 | 本项目是库 / 自托管服务，不含控制平面 |
| 多模态（图像、音频）、代码仓库状态 | 暂不支持 | 可扩展 `MemoryItem.data`，当前版本未实现 |
| 企业级 ACL、加密存储、合规删除证明 | 部分支持 | 有多 Agent 团队与 grant，但没有字段级权限与加密 |

---

## 11. 评测

### 11.1 设置

- 数据：3 个用户 × 4 个会话 ×（1 条事实消息 + 4 条 filler 对），共 60 个 episode；
- 问题：每用户 11 题，共 33 题：当前地点、历史地点、职业、雇主、偏好、过敏、项目栈、课程总数、流程、书籍、多跳总部；
- 预算：500 token；
- reader：规则化模拟（从 `slot=` 行或原始文本中抽取答案），不是真实 LLM；
- 运行：`python benchmark.py`，固定随机种子。

### 11.2 结果

| 方法 | 准确率 | token/题 | 证据命中 | 陈旧回答 | 时延 ms |
|---|---:|---:|---:|---:|---:|
| No memory | 0.0% | 0.0 | 0.0% | 0.0% | 0.0 |
| Full context（不限预算） | 100.0% | 544.0 | 90.9% | 0.0% | 0.0 |
| Full context（500 token） | 81.8% | 515.3 | 78.8% | 6.1% | 0.1 |
| Naive vector Top-4 | 69.7% | 251.7 | 81.8% | 21.2% | 3.3 |
| **MemStack（完整）** | **100.0%** | **474.9** | **100.0%** | **0.0%** | 65.0 |
| w/o temporal | 90.9% | 475.0 | 100.0% | 9.1% | 79.8 |
| w/o graph | 100.0% | 482.0 | 100.0% | 0.0% | 74.5 |
| w/o routing | 100.0% | 475.1 | 100.0% | 0.0% | 77.2 |
| w/o rerank | 100.0% | 460.2 | 100.0% | 0.0% | 82.0 |
| w/o consolidation | 100.0% | 402.8 | 100.0% | 0.0% | 72.2 |
| episodes only | 60.6% | 483.7 | 100.0% | 27.3% | 49.8 |

### 11.3 怎么读这张表

1. **同预算下，结构化的价值**：截断全上下文 81.8%、朴素向量 69.7%，MemStack 100%。
2. **陈旧事实是纯检索的硬伤**：朴素向量 21.2% 的当前问题答成了旧值；episodes-only 27.3%；MemStack 0%。
3. **成本**：MemStack 达到不限预算全上下文的准确率，token 少约 13%（544→475）。v2 的证据回填/多轮补证带来约 +96 token，仍低于全上下文；真实长历史下全上下文线性增长，MemStack 受 `token_budget` 约束。
4. **时间感知是刚需**：关闭后历史地点直接错（90.9%），陈旧回答 9.1%。
5. **图/路由/重排在本集合上饱和**：候选池小、reader 简单，三者没有改变最终准确率；它们的作用更多体现在证据覆盖、来源质量和更大规模数据上的鲁棒性。这是本评测的局限，不应解读为“机制无效”。
6. **巩固的代价**：`w/o consolidation` 少 72 token 但准确率不变；因为本集合没有“跨会话总结/冷启动画像”类问题。常驻画像是一个明确的成本换稳定性选择，可按场景关闭或缩小。

### 11.4 效度威胁

- **合成数据**：模板化表达，规则抽取器天然占优；换成真实对话，抽取错误率会上升。
- **规则 reader**：它按 `slot=` 和固定正则作答，不理解语义；真实 LLM 可能从更少上下文答对，也可能被旧值干扰。
- **小规模**：3 用户、60 episode，图/路由/重排的差异被饱和。
- **无真实 LLM**：没有 token 计费、延迟、模型波动、prompt 注入。
- **单一语言**：中文只做了基础规则与哈希向量验证，没有中文 benchmark。
- **不可与厂商公布分数比较**：这里的数字只验证机制趋势，不是产品级 benchmark。

### 11.5 公开数据集评测

完整协议、分题型结果与效度威胁见 [IMPROVEMENTS.md](IMPROVEMENTS.md)；
v1 结果见 [PUBLIC_EVAL.md](PUBLIC_EVAL.md)。

| 数据集 | 方法 | Unit R@k | All-Unit | Turn Token | Ans Token | Tokens |
|---|---|---:|---:|---:|---:|---:|
| **LoCoMo**（全量 1986 题） | **MemStack**（v4.1） | **84.3%** | **74.3%** | **70.0%** | **33.2%** | **599** |
| | MemStack-mr（v3 词袋补证） | 82.0% | 71.5% | 67.9% | 31.8% | 583 |
| | MemStack-ep | **88.3%** | **79.2%** | **77.5%** | **36.7%** | 722 |
| | Fact+Hybrid | 82.0% | 72.4% | 69.7% | 32.8% | 632 |
| | Temporal-Graph | 83.1% | 72.8% | 69.3% | 32.2% | 598 |
| | BM25 only | 64.0% | 56.0% | 74.0% | 33.2% | 650 |
| **LoCoMo**（300 题分层） | **MemStack** | **83.8%** | **69.0%** | **69.6%** | **21.4%** | **627** |
| | MemStack-mr（v3 词袋补证） | 82.8% | 68.0% | 68.0% | 20.3% | 620 |
| | MemStack-ep | **86.5%** | **72.4%** | **74.1%** | **24.3%** | 716 |
| **LongMemEval oracle**（500 题） | **MemStack** | 100% | **79.2%** | 58.0% | 45.8% | 765 |
| | MemStack-mr | 100% | 79.2% | 57.6% | 46.3% | 766 |
| | MemStack-ep | 100% | **96.6%** | 56.3% | 51.9% | **410** |
| | Episode+Prior | 100% | **95.6%** | 55.3% | 50.8% | **402** |
| | Dense / BM25 / Full | 100% | 35.2% | 69–70% | 50–55% | 792 |

> 表中为 v4 口径；v4.1（episode 保底 + 链内缓存）的最新数字见 `results/*_v041.json` 与 [README.md](README.md)。

三点结论：

1. **证据链（v4）相对词袋补证（v3）全面提升，且大样本下更显著**：全量 1986 题 All-Unit 71.5%→74.3%（+2.8pp，bootstrap CI [72.4%, 76.3%]，v3 在区间外）、TurnHit +3.1pp、Unit R@k +2.3pp；multi-hop 分题型（n=281）All-Unit 18.5%→24.2%（相对 +31%）。LongMemEval 上中性不退化。链内共享缓存后延迟增量收窄到 +14ms/查询。
2. **覆盖 vs 逐字**：MemStack 把更多短 episode 打包，Turn Token 55%–58%，低于整段会话基线的 69%–70%；LoCoMo 上 MemStack-ep 最高（74.1%）。没有单一最优粒度。
3. **多跳仍弱但有天花板因素**：LoCoMo multi-hop 的 All-Unit 24.2%（全量口径）——枚举类题的 gold 证据散布在 7–19 个会话，800-token 预算内结构上不可能全部召回；真正可提升的是 TurnToken/AnsToken，这个证据链已经改善。

口径：这是**检索层评测**，不是 LLM-as-judge QA；LoCoMo 为 300 题分层子集（297 题有证据），LongMemEval 为 oracle 版本（只含证据会话，Unit R 对所有方法均为 100%）。

**QA 层**：[qa_eval.py](qa_eval.py) 在检索结果之上加了 reader + judge。
LoCoMo 50 题分层、600 token、Qwen3.5-0.8B 本地 reader/judge 下：MemStack 总体准确率 46.0%、可答题 47.5%、F1 28.5%。
覆盖高不等于可答率高——两类指标必须分开测，完整口径见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

---

## 12. 压力测试与失败模式

### 12.1 陈旧事实（已覆盖）

- 机制：`SUPERSEDE` + `valid_to` + 默认 current 过滤；
- 证据：合成基准 stale=0%，no_temporal=9.1%，naive=21.2%；
- 单测：`test_temporal_supersede`、`test_preference_update`。

### 12.2 乱序写入 / 事件时间歧义（部分覆盖）

- 机制：`is_change` 语义 + `valid_from` 比较；
- 风险：`last month` 这类相对时间解析依赖写入时间；跨时区、模糊日期、用户口误都会造成窗口偏差；
- 改进：引入显式 `event_time` 字段、消息时间戳、时间解析 LLM、人工校正接口。

### 12.3 抽取错误传播

- 机制：规则抽取可预测但覆盖有限；LLM 抽取覆盖广但会幻觉；
- 风险：错误事实进入图后被多跳放大；
- 缓解：`source_ids` 溯源、`confidence`、LLM 抽取回退、离线冲突消解；
- 未做：事实级人工审核、可纠错的训练信号。

### 12.4 “自信地错”与停止准则

- MemStack 不直接做多轮问答，但 `confidence` 可能因上下文覆盖高而偏高；
- 如果检索到的是错误但一致的事实，`should_continue` 不会触发；
- 这与“假共识”同源：**一致 ≠ 正确**；
- 路线：停止条件应同时检查“一致 + 证据覆盖 + 反方证据”，而不是只看置信度。

### 12.5 图膨胀与 PPR 成本

- 每个实体节点 + 每条 mentions 边都会增加图规模；高 degree 实体（如常用城市）会放大 PPR；
- 当前 PPR 是 O(E) / 查询，没有社区划分、增量索引、并行化；
- 缓解：`max_similar_edges=0` 可关闭相似边；限制 `seed_k`；生产换图数据库/近似 PPR。

### 12.6 隐私与遗忘

- `forget` 默认 `cascade=True`：事实删除时，若其 source episode 不再被其他 active 记忆引用，则 episode 一并 `deleted`；引用它的 summary 也删除；
- 画像块会刷新，避免残留；
- 未做：加密、字段级脱敏、跨用户 ACL、审计导出、合规删除证明。

### 12.7 多 Agent / 团队记忆

- 当前 scope 支持同 tenant/agent 下多 user，但没有角色、可见性、共享审批、冲突投票；
- 超边 / 角色感知检索是路线图；
- 多 Agent 同时写同一 scope 时，SQLite 写锁 + 单进程假设；生产需要事务/并发控制。

### 12.8 上下文预算与 KV cache

- 当前按 token 估算裁剪，不感知 prompt 前缀稳定性；
- 注意：改变 prompt 前缀会导致 KV cache miss，省 token 可能反而更贵；
- 路线：固定 system/block 前缀，把动态记忆放在尾部；支持 cache-aware 排序。

---

## 13. 局限与路线图

### 13.1 当前局限

| 局限 | 影响 | 现状 |
|---|---|---|
| 默认规则抽取 | 真实对话覆盖有限 | 可换 `LLMExtractor` |
| SQLite 暴力扫描 | 10 万级记忆后变慢 | v0.3 已缓解：float32 BLOB、revision 缓存、MMR numpy 化（4314 条 1550ms→520ms）；仍无 ANN 索引，可换 `MemoryStore` / 向量索引 |
| 图是普通图 | 多 Agent 协作事件表达弱 | 超图路线图 |
| 更新策略是规则 | 无法从反馈学习 | 可学习更新策略路线图 |
| 无双路径检索 | 复杂查询一次 top-K 可能不足 | 双路径检索路线图 |
| 多跳覆盖不足 | LoCoMo multi-hop All-Unit 27.1% | 已有多轮补证；需要 query decomposition |
| 无并行上下文分支 | 长程 Agent 的上下文路由未覆盖 | 并行上下文路由路线图 |
| 无多模态 | 图像/表格/代码状态不可记 | 扩展 MemoryItem.data |
| 无团队 ACL | 企业共享记忆受限 | **v0.3 已提供**：`MultiAgentHub` 团队 open/closed + grant，见 §6.10 |
| 无 LLM-as-judge QA | 只能报告检索层指标 | 已跑 LoCoMo 300 / LongMemEval oracle 500；`qa_eval.py` 已接 LLM reader/judge（v3） |

### 13.2 路线图（按优先级）

1. **LLM 抽取 + 结构化输出校验**：JSON schema、重试、回退、置信度校准。
2. **向量索引适配**：`MemoryStore` 之外再加 `VectorIndex` 协议，支持 HNSW/FAISS/pgvector。v0.3 已做内存级优化（revision 缓存 + numpy 批量）但仍是暴力扫描。
3. **cache-aware 上下文组装**：固定前缀、动态尾部、生命周期驱逐。**v4.1 已落一部分**：blocks 前缀稳定 + `status_aware_packing` 预算裁剪时 active 优先/历史值尾部；完整 cache-aware 排序（按 KV 重用率安排动态条目）仍待做。
4. **可学习更新策略**：用对比反馈归纳 guideline，转成过程奖励。
5. **双路径检索**：熟悉度低时聚类 + α-mix 迭代扩证。**v4 已提供首版**：首跳低分时走 PPR 近邻扩证（`evidence_chain_neighbor`），仍未做聚类层。
6. **并行上下文路由**：多分支 lookahead + η/ρ 评估。
7. **超图/角色感知**：多 Agent 协作事件、ACL、共享记忆。
8. **真实 benchmark 扩展**：全量 LoCoMo 1986 题 + LongMemEval-s（带 distractor）；**LLM reader/judge 已由 `qa_eval.py` 提供（本地 HF / OpenAI 兼容端点 / 规则 reader 三种后端，v3）**，下一步是同 backbone、同预算复跑全部方法。
9. **隐私工程**：加密、字段级脱敏、删除证明、审计导出。
10. **Query decomposition / 证据链**：把多跳问题拆成子问题，逐跳检索并验证证据覆盖。**v4 已提供规则化首版**（`memstack/evidence_chain.py` + §6.9）：枚举/桥接/时间锚点/行为动词尾四类分解，LoCoMo multi-hop All-Unit 27.1%→28.8%、TurnToken +2.7pp、AnsToken +4.2pp。下一步：LLM 化分解、按子问题验收证据覆盖、更大证据预算下的 All-Unit。

---

## 14. 接入示例

### 14.1 Python

```python
from memstack import MemoryStack, Scope, SQLiteStore

stack = MemoryStack(store=SQLiteStore("./memory.db"))
scope = Scope(tenant_id="acme", user_id="nora", agent_id="assistant")

stack.remember(messages, scope, event_time=...)
bundle = stack.recall("Where does the user live now?", scope, token_budget=800)
stack.forget(query="旧地址", scope=scope)
stack.consolidate(scope)
```

### 14.2 OpenAI function-calling

```python
tools = stack.as_tools()
# 把 tools 传给模型；模型返回 tool_call 后：
result = stack.dispatch_tool(tool_call.function.name, tool_call.function.arguments)
```

### 14.3 HTTP

```bash
python -m memstack.server --db ./memstack.db --port 8765 --token secret
```

```bash
curl -X POST http://127.0.0.1:8765/v1/search \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{"query":"Where does Nora live now?","scope":{"user_id":"nora"}}'
```

### 14.4 Agent 中间件

```python
from memstack import MemoryMiddleware

middleware = MemoryMiddleware(stack, scope)
system_prompt, bundle = middleware.before(user_query)
# 调模型时把 system_prompt 作为 system message
reply = llm(system_prompt, user_query)
middleware.after([
    {"role": "user", "content": user_query},
    {"role": "assistant", "content": reply},
])
```

### 14.5 替换 LLM / Embedder

```python
from memstack import LLMExtractor, LLMSummarizer, OpenAICompatClient

client = OpenAICompatClient(model="gpt-4o-mini")
stack = MemoryStack(
    extractor=LLMExtractor(client),
    summarizer=LLMSummarizer(client),
)
```

Embedder 只需实现：

```python
class MyEmbedder:
    dim = 1024
    def embed(self, texts): ...
```

---

## 附录 A：核心 API

| API | 说明 |
|---|---|
| `remember(messages, scope, event_time=None, metadata=None, force=False)` | 写入一段交互，返回决策列表 |
| `recall(query, scope, top_k=None, token_budget=None, include_history=None, kinds=None)` | 返回 `ContextBundle` |
| `search(query, scope, top_k=None, kinds=None, include_history=None)` | 返回 `RecallHit` 列表 |
| `forget(memory_ids=None, query=None, scope=None, hard=False, cascade=True)` | 软/硬删除 + 级联 |
| `consolidate(scope)` | 离线巩固 |
| `set_block(label, value, scope, limit)` / `get_block` | 常驻上下文块 |
| `stats(scope)` | 类型/状态/边统计 |
| `as_tools()` / `dispatch_tool(name, args)` | 工具接入 |
| `export(scope)` / `import_data(payload)` | 迁移/备份 |

## 附录 B：关键配置

| 参数 | 默认 | 作用 |
|---|---|---|
| `embedding_dim` | 512 | hashing 向量维度 |
| `duplicate_threshold` | 0.90 | 语义重复合并阈值 |
| `similar_threshold` | 0.70 | 相似边阈值 |
| `max_similar_edges` | 3 | 每节点相似边上限（0=关闭） |
| `history_policy` | append_only | 追加式 / 覆盖标记 |
| `max_turns_per_episode` | 6 | episode 最大轮数 |
| `top_k` | 8 | 返回条数 |
| `token_budget` | 800 | 上下文 token 预算 |
| `seed_k` | 5 | PPR 种子数 |
| `use_graph` | True | 图扩展 |
| `use_routing` | True | 熵路由 |
| `use_temporal` | True | 时间策略 |
| `use_rerank` | True | 启发式重排 |
| `weights` | dense .46 / lexical .26 / entity .08 / prior .12 / graph .08 | 通道权重 |
| `evidence_chain` | True | 证据链检索（v4）：复杂查询分解→逐跳检索→分名额合并 |
| `evidence_chain_max_subs` | 2 | 最多分解出的子查询数 |
| `evidence_chain_slots` / `evidence_chain_enum_slots` | 2 / 4 | 每子查询补证名额（一般 / 枚举聚合类） |
| `evidence_chain_neighbor` / `evidence_chain_min_top_score` | True / 0.30 | 双路径扩证：首跳低于阈值走图近邻 |
| `status_aware_packing` | True | 生命周期感知打包：预算裁剪时 active 优先、历史值放尾部 |
| `multi_round` | True | v2 词袋补证（`evidence_chain=False` 时的兼容路径） |
| `recency_half_life_days` | 90 | 时间衰减半衰期 |
| `continue_confidence` | 0.42 | 继续检索阈值 |
| `mmr_lambda` | 0.75 | 多样性强度 |
| `summary_min_episodes` | 3 | 摘要触发阈值 |
| `archive_after_days` | 730 | 归档天数 |
| `profile_max_chars` | 1200 | 画像块上限 |

## 附录 C：术语

| 术语 | 含义 |
|---|---|
| Episode | 一次交互窗口的原文，作为可验证证据 |
| Fact / Proposition | 原子事实命题 |
| Preference | 带 topic/polarity 的用户偏好 |
| Procedure / Prescription | 可复用流程（intent + steps） |
| Provenance | 派生知识到原始证据的溯源边 |
| Bi-temporal | 事件时间（valid_from/valid_to）+ 写入时间（created_at） |
| Supersede | 新事实取代旧事实，旧事实保留为历史 |
| PPR | Personalized PageRank，带重启的图随机游走 |
| Entropy routing | 按粒度分数分布的熵分配权重 |
| MMR | Maximal Marginal Relevance，相关性 + 多样性 |
| ContextBundle | 带上下文、来源、置信度、缺口、继续信号的返回包 |
| Sleep-time consolidation | 离线去重、摘要、冲突消解、衰减 |
| Cascade forget | 删除事实时连带隐藏只服务于它的证据与摘要 |
