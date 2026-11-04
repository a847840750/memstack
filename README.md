# MemStack

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-0.4.1-orange)](CHANGELOG.md)
[![CI](https://img.shields.io/badge/CI-ubuntu%20%7C%20windows%20%7C%203.10--3.12-blueviolet)](.github/workflows/ci.yml)

**面向 LLM Agent 的可移植分层记忆栈。** 纯 Python 标准库核心，零第三方依赖；可作为库直接嵌入现有 Agent，也可以作为独立 HTTP 记忆服务或 MCP server 运行。

[快速开始](#快速开始) · [核心概念](#核心概念) · [使用示例](#使用示例) · [配置](#配置) · [评测](#评测) · [English](README_EN.md)

---

## 为什么用 MemStack

Agent 的记忆问题通常不是“存不下来”，而是下面五种失败模式：

| 失败模式 | 典型症状 | MemStack 的对应机制 |
|---|---|---|
| **记不住** | 换会话就忘记用户偏好、项目约束 | 长期存储 + 用户画像块 |
| **找不到** | 纯向量检索漏掉专名、数字、时间、跨会话关联 | dense + BM25 + entity + prior + PPR 五通道融合 |
| **记错** | 用户搬家/换工作后仍回答旧值 | 双时间 + `superseded` + 历史查询 |
| **塞不下** | 把整段历史塞进上下文，token 爆炸 | 多粒度路由 + 重排 + MMR + token 预算 |
| **删不掉** | 删除事实后原文仍能被检索到 | 软删除 + 级联隐藏 episode/摘要 + 审计 |

## 特性

- **写入**：原始 episode + fact / preference / procedure / summary / entity 索引，自动去重、合并、冲突取代
- **时间**：双时间（事件时间 + 写入时间）、`active / superseded / archived / deleted` 生命周期；写入时把 `yesterday / last week / N years ago` 归一化为绝对日期（原文保留）
- **检索**：dense + BM25 + 实体 + 先验 + 图扩展（PPR），多粒度熵路由，时间策略，重排，MMR，token 预算；**原文 episode 预算保底**，抽取事实不会挤掉证据
- **证据链**：多跳 / 枚举聚合类查询自动分解为子查询逐跳检索、分名额合并，只补缺口不挤占首轮证据；可选 LLM 化分解（失败自动回退规则版）
- **巩固**：离线去重、冲突消解、会话摘要、衰减归档、用户画像块
- **多 Agent**：agent 注册、团队（open / closed）共享记忆、跨 agent 授权（grant）、事件总线
- **接入**：Python API / 工厂（`build_stack`）/ OpenAI function-calling 工具 / HTTP 网关 / **MCP server（零 SDK，stdio）** / Agent 中间件
- **可移植**：`Embedder / Extractor / Store / Reranker / Clock` 全部是 Protocol，默认实现可整体替换
- **LLM 可选**：设置 `MEMSTACK_LLM_MODEL` + `OPENAI_API_KEY` 即自动切换 LLM 抽取 / 摘要 / 查询分解，兼容任意 OpenAI 风格端点

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)；API 成本预估：`python cost_estimate.py`。

---

## 快速开始

要求：Python 3.10+，**核心无需安装任何依赖**。

```bash
# 可选：本地安装为可导入包
pip install -e .

# 可选：外部库对照基线（只影响 qa_eval.py --methods extlib）
pip install mem0ai
```

```python
from memstack import MemoryStack, Scope

stack = MemoryStack()  # 默认 SQLite :memory:；也可传 SQLiteStore("memory.db")

user = Scope(user_id="nora", agent_id="assistant")
session = Scope(user_id="nora", agent_id="assistant", session_id="s1")

stack.remember(
    [
        {"role": "user", "content": "I live in Lisbon. I work as a data analyst at Northwind Labs."},
        {"role": "assistant", "content": "Noted."},
    ],
    session,
    event_time=1767225600,  # 可选；不传则用当前时间
)

bundle = stack.recall("Where does the user live now?", user, token_budget=600)
print(bundle.text)
print(bundle.confidence, bundle.gaps, bundle.should_continue)
```

检索结果不是一段纯文本，而是带来源、状态、时间、分数的 `ContextBundle`：

```text
## Retrieved memory
- [FACT|score=1.00|active|from=2026-01-01|src=ep-...] slot=user.location | value=Lisbon | evidence="I live in Lisbon."
```

---

## 核心概念

| 概念 | 说明 |
|---|---|
| `Scope` | 四元组 `tenant_id / user_id / agent_id / session_id`；前三者决定隔离，`session_id` 只用于会话内分组 |
| `MemoryKind` | `episode / fact / preference / procedure / summary / entity / reflection` |
| `MemoryStatus` | `active / superseded / archived / deleted`；默认只检索 active，历史查询才带旧值 |
| `ContextBundle` | 检索返回包：`text + items + confidence + coverage + gaps + conflicts + trace` |
| `RememberResult` | 写入返回包：每条新记忆的 `ADD / MERGE / SUPERSEDE / NOOP` 决策与来源 |

`episode` 是证据原文，`fact / preference / procedure` 是抽取出的知识，`summary` 是导航层，`entity` 只用于图桥接、不直接进上下文。

---

## 使用示例

### 多 Agent 协作（团队共享记忆 + 事件总线）

```python
from memstack import build_hub

hub = build_hub(db="team.db")           # 设置了 LLM 环境变量时自动启用 LLM 抽取
hub.register_agent("planner", role="planner")
hub.register_agent("executor", role="executor")
hub.create_team("trip", members=["planner", "executor"], policy="closed")

hub.remember_as("planner", [{"role": "user", "content": "预算上限 5000 元。"}],
                team_id="trip", share=True)
bundle = hub.recall_as("executor", "这次旅行的预算是多少？", teams=["trip"])
print(bundle.text)   # 私有 + 团队共享的合并上下文，带来源段落

hub.post_event("trip", "planner", "plan_ready", "行程已确认，等待付款")
events = hub.pull_events("executor", "trip")   # 增量拉取事件流
```

closed 团队非成员不可读；跨 agent 读他人私有记忆需显式 `hub.grant("executor", "planner", "read")`。
完整示例：`python examples/multiagent_orchestration.py`。

### LLM 开箱即用（可选）

```bash
export MEMSTACK_LLM_MODEL=gpt-4o-mini    # 或 deepseek-chat / vLLM / Ollama 任意 OpenAI 兼容端点
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://.../v1    # 非官方端点时设置
python -c "from memstack import build_stack; s = build_stack(db='m.db'); \
           s.remember('用户说搬家到了柏林'); print(s.recall('用户住在哪').text)"
```

不设置环境变量时系统退化为零依赖规则抽取，完全不联网。

### 模型基座（可选，评测/推理用）

内置基座免去手写端点，直接按名字选用（定义见 `memstack/backends.py`）：

```bash
# 本地 Ollama 上的 Qwen3.5-9B（OpenAI 兼容 /v1）
python qa_eval.py --reader llm --judge llm --model-base qwen35-9b-local
# 或走环境变量
export MEMSTACK_LLM_BASE=qwen35-9b-local
```

端点与密钥可用 `OPENAI_BASE_URL` / `MEMSTACK_API_KEY` / `MEMSTACK_REASONING_EFFORT` 覆盖，不写死在代码里。Qwen3.5-9B 默认开启思考会让 `content` 为空，该基座默认 `reasoning_effort="none"`。

LLM 化查询分解（可选，用于多跳/枚举类问题的证据链；输出非法 JSON 或调用失败自动回退规则版，检索链路不断）：

```python
from memstack import MemoryStack, LLMQueryDecomposer, OpenAICompatClient

client = OpenAICompatClient(model="gpt-4o-mini")
stack = MemoryStack(query_decomposer=LLMQueryDecomposer(client))
bundle = stack.recall("What hobbies does Evan pursue?", scope)
print(bundle.trace["evidence_chain"]["decomposer"])  # "llm" 或 "rule-fallback"
```

### OpenAI function-calling 与 Agent 中间件

```python
tools = stack.as_tools()          # 5 个工具：search / add / forget / profile / consolidate
result = stack.dispatch_tool(tool_call.function.name, tool_call.function.arguments)

from memstack import MemoryMiddleware

middleware = MemoryMiddleware(stack, scope)
system_prompt, bundle = middleware.before(user_query)
reply = llm(system_prompt, user_query)
middleware.after([
    {"role": "user", "content": user_query},
    {"role": "assistant", "content": reply},
])
```

### HTTP 网关

```bash
python -m memstack.server --db ./memstack.db --port 8765 --token secret
```

```text
GET  /health
POST /v1/remember      {messages, scope, event_time?, metadata?}
POST /v1/search        {query, scope?, top_k?, token_budget?, include_history?}
POST /v1/forget        {ids?|query?, scope?, hard?}
POST /v1/consolidate   {scope?}
GET  /v1/profile?user_id=..
```

### MCP server（零 SDK）

```bash
python -m memstack.mcp_server --hub --db mem.db   # stdio JSON-RPC，可被 Cursor/Claude Desktop 直接挂载
python -m memstack.mcp_server list-tools          # 查看 12 个工具（记忆 + 多 agent）
python examples/mcp_client.py                     # 用标准库驱动 server 的演示客户端
```

客户端 `mcp.json` 配置：

```json
{"mcpServers": {"memstack": {"command": "python", "args": ["-m", "memstack.mcp_server", "--hub", "--db", "mem.db"]}}}
```

### 自定义后端

```python
class MyEmbedder:
    dim = 1024
    def embed(self, texts):
        ...

stack = MemoryStack(embedder=MyEmbedder())
```

五个可替换点：`Embedder`（BGE / OpenAI / 自研）、`Extractor`（`LLMExtractor` / 领域模型）、`Summarizer`、`Reranker`、`MemoryStore`（Postgres / Redis / 向量库）。核心代码只依赖 Protocol，不 import 第三方框架。

---

## 配置

常用开关（完整列表见 [ARCHITECTURE.md](ARCHITECTURE.md) 附录 B）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `top_k` | 8 | 返回条数 |
| `token_budget` | 800 | 上下文 token 预算 |
| `episode_reserve_ratio` | 0.30 | 原文 episode 的预算保底比例 |
| `use_temporal` / `use_graph` / `use_routing` / `use_rerank` | True | 时间策略 / 图扩展 / 熵路由 / 重排 |
| `evidence_chain` | True | 证据链检索：分解 → 逐跳检索 → 分名额合并 |
| `status_aware_packing` | True | 预算裁剪时 active 优先、历史值放尾部 |
| `mmr_lambda` | 0.75 | MMR 多样性强度 |
| `archive_after_days` | 730 | 无访问记忆的归档天数 |

---

## 目录

```text
memstack/
├── memstack/
│   ├── core.py            # 门面：remember / recall / forget / consolidate / blocks
│   ├── multiagent.py      # 多 Agent：注册、团队共享、授权(grant)、事件总线
│   ├── mcp_server.py      # 零 SDK 的 MCP server（stdio JSON-RPC）
│   ├── factory.py         # build_stack / build_hub：LLM 环境变量驱动的开箱即用工厂
│   ├── types.py           # MemoryItem / Scope / ContextBundle / 枚举
│   ├── config.py          # 全部可调参数
│   ├── interfaces.py      # 可移植性 Protocol
│   ├── embeddings.py      # 零依赖 hashing 向量 + 批量余弦（numpy 可选加速）
│   ├── extraction.py      # 规则抽取器 / LLM 抽取器 / 摘要器
│   ├── storage.py         # SQLite 存储（memories / edges / blocks / audit）
│   ├── graph.py           # 实体链接、溯源边、Personalized PageRank
│   ├── retrieval.py       # 混合召回、熵路由、时间策略、重排、预算组装
│   ├── evidence_chain.py  # 证据链：查询分解、逐跳检索、分名额合并
│   ├── consolidation.py   # 睡眠期去重、冲突消解、摘要、衰减、画像
│   ├── tools.py           # OpenAI 工具 schema、dispatch、Agent 中间件
│   ├── llm.py             # 零 SDK 的 OpenAI 兼容客户端
│   ├── backends.py        # 内置模型基座（OpenAI 兼容端点）
│   └── server.py          # 标准库 HTTP 网关
├── examples/              # 多 agent 编排 / 框架集成 / MCP 客户端
├── tests/                 # 56 个 unittest
├── data/                  # LoCoMo / LongMemEval 本地数据
├── results/               # 公开评测原始结果（JSON / Markdown）
├── demo.py                # 端到端演示
├── benchmark.py           # 合成基准（同预算对比 + 消融）
├── public_eval.py         # LoCoMo / LongMemEval 检索层评测
├── qa_eval.py             # LoCoMo / LongMemEval QA 层评测：reader + judge
├── cost_estimate.py       # OpenAI 兼容 API 评测成本估算
├── ARCHITECTURE.md        # 系统架构与设计取舍
├── IMPROVEMENTS.md        # 改进说明与综合评测
├── PUBLIC_EVAL.md         # v1 公开数据集结果
├── CHANGELOG.md
└── LICENSE                # Apache-2.0
```

---

## 运行

```bash
python demo.py
python benchmark.py
python -m unittest discover -s tests -v

# HTTP 服务
python -m memstack.server --db ./memstack.db --port 8765

# 公开数据集：检索层（首次运行自动下载数据）
python public_eval.py --dataset locomo --limit 300 --turns 3 --episode-chars 900 \
  --out results/locomo_300_comprehensive.json
python public_eval.py --dataset locomo --turns 3 --episode-chars 900 --no-download \
  --out results/locomo_full_comprehensive.json    # 全量 1986 题（约 15 分钟）
python public_eval.py --dataset longmemeval --turns 2 --episode-chars 700 \
  --out results/longmemeval_comprehensive.json

# 公开数据集：QA 层（本地小模型当 reader + judge，离线）
python qa_eval.py --limit 50 --budget 600 --reader hf --judge hf \
  --out results/locomo_qa.json

# QA 层：外部库对照基线（可选，需 pip install mem0ai）
python qa_eval.py --limit 50 --budget 600 --methods extlib \
  --reader hf --judge hf --out results/locomo_qa_extlib.json

# QA 层：换成任意 OpenAI 兼容端点
MEMSTACK_LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
  python qa_eval.py --limit 200 --reader llm --judge llm
```

---

## 评测

### 合成基准（机制级）

同一 500-token 上下文预算，3 个用户 × 4 个会话 × 11 个问题（共 33 题），规则化 reader。
数值只用于验证机制趋势，不可与生产 benchmark 直接比较。

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

结论：

1. 同预算下，MemStack 100% vs 截断全上下文 81.8% vs 朴素向量 69.7%；
2. 陈旧回答 0%，朴素向量 21.2%，只存原始 episode 27.3%；
3. 达到不限预算全上下文的准确率，同时少用约 13% token（544→475）；历史越长，差距越大；
4. 图扩展 / 粒度路由 / 重排在当前合成集上饱和（准确率无差异），差异主要体现在证据覆盖与成本；时间感知是唯一显著改变准确率的消融。

### 公开数据集（检索层）

同 800-token 预算，`top_k=8`；LoCoMo 全量 1986 题，LongMemEval 为 oracle 版本。
详细协议、分题型结果与效度威胁见 [IMPROVEMENTS.md](IMPROVEMENTS.md)，v1 结果见 [PUBLIC_EVAL.md](PUBLIC_EVAL.md)。

| 数据集 | 方法 | Unit R@k | All-Unit | Turn Token | Ans Token | NDCG | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| **LoCoMo**（全量 1986 题） | **MemStack**（v4.1 + episode 保底） | **85.4%** | **75.6%** | **70.7%** | **43.1%** | **0.636** | **636** |
| | MemStack-mr（v3 词袋补证） | 83.6% | 73.6% | 69.3% | 41.8% | 0.623 | 622 |
| | MemStack-ep | **89.0%** | **79.8%** | **77.4%** | **46.3%** | **0.712** | 746 |
| | Fact+Hybrid | 83.0% | 73.4% | 69.7% | 42.0% | 0.637 | 667 |
| | Temporal-Graph | 84.3% | 74.3% | 69.8% | 42.1% | 0.639 | 633 |
| | Fact+Dense | 67.5% | 57.3% | 55.8% | 33.1% | 0.444 | 626 |
| **LoCoMo**（300 题分层） | **MemStack**（v4 证据链） | **83.8%** | **70.4%** | 69.3% | 37.3% | **0.626** | **648** |
| | MemStack-mr（v3 词袋补证） | 81.8% | 69.0% | 66.9% | 35.5% | 0.616 | 636 |
| | MemStack-ep | **86.9%** | **72.1%** | **72.8%** | **39.7%** | **0.653** | 740 |
| | BM25 only | 57.6% | 46.8% | 69.3% | 20.6% | 0.576 | 668 |
| | Dense only | 23.6% | 17.5% | 50.4% | 13.6% | 0.236 | 644 |
| **LongMemEval oracle**（500 题） | **MemStack** | 100% | **87.6%** | 64.5% | 52.5% | 1.000 | 759 |
| | MemStack-mr | 100% | 86.4% | 64.5% | 52.4% | 1.000 | 761 |
| | MemStack-ep | 100% | **95.8%** | 57.2% | **52.7%** | 1.000 | **450** |
| | Episode+Prior | 100% | **96.8%** | 58.0% | 53.7% | 1.000 | **506** |
| | Dense / BM25 / Full | 100% | 35.2% | 72–75% | 50–56% | 1.000 | 792 |

- 同预算下，MemStack / MemStack-ep / Episode+Prior 的证据会话覆盖显著高于纯会话检索；
- **0.4.1 的 episode 预算保底**把 LongMemEval 的 All-Unit 从 79.2% 提到 **87.6%**、Ans Token 45.8%→**52.5%**：抽取事实不再把原文证据挤出上下文；
- **v4 证据链**（query 分解 → 逐跳检索 → 分名额合并）相对 v3 词袋补证：LoCoMo 全量 All-Unit 73.6%→**75.6%**、Ans Token 41.8%→**43.1%**；机制细节见 [IMPROVEMENTS.md](IMPROVEMENTS.md)；
- LongMemEval oracle 中所有 session 都是证据，所以 Unit R 对所有方法都是 100%，有区分度的是 **All-Unit**；
- LoCoMo 多跳 All-Unit 仍有天花板（全量 26% 左右）：枚举题 gold 证据散布在 7–19 个会话，600–800-token 预算内无法全召回。

> 口径：这是**检索层评测**，不是 LLM-as-judge QA。LoCoMo 同时报告全量 1986 题与 300 题分层子集，LongMemEval 为 oracle 版本；LongMemEval 为控制 CPU 成本关闭实体图，因此其 MemStack 数字是保守下界。
> 原始文件：`results/locomo_full_comprehensive_v041.json` · `results/locomo_300_comprehensive_v041.json` · `results/longmemeval_comprehensive_v041.json`。

### QA 层（LLM reader + judge）

`qa_eval.py` 在检索结果之上做生成与判分。下表为**全量**结果，reader 与 judge 均为本地 **Qwen3.5-9B**（Ollama `qwen35-9b-local`）：

**LoCoMo 全量 1986 题（600 token 预算，2726s）**

| 方法 | Unit R@k | All-Unit | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MemStack** | 84.2% | 74.1% | **54.3%** | **47.2%** | 78.7% | **37.4%** | **20.3%** |
| MemStack-mr | 82.7% | 72.5% | 52.5% | 45.8% | 75.6% | 36.5% | 19.8% |

**外部库对照基线（50 题分层，600 token，同一 0.8B reader/judge）**

| 方法 | Unit R@k | All-Unit | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| External-Lib（原始消息，无 LLM 抽取） | 91.7% | 83.3% | 22.0% | 20.0% | 30.0% | 13.7% | 5.0% |

**LongMemEval oracle 全量 500 题（800 token 预算，1334s）**

| 方法 | Unit R@k | All-Unit | 总体准确率 | F1 | EM |
|---|---:|---:|---:|---:|---:|
| MemStack | 100% | 85.6% | 43.4% | 32.9% | 24.4% |
| **MemStack-ep**（只存原始 episode） | 100% | **96.4%** | **46.8%** | **36.3%** | **25.0%** |

- 分题型（MemStack）：时序 47.4%（0.4.1 前是 7.5%）、单跳 54.7%、多跳 36.2%、开放域 13.5%、对抗拒答 78.7%；`--grounding adaptive` 只对开放域放开世界知识，300 题子集上开放域 8.3%→35.0%；
- LongMemEval 上 `MemStack` 与 `MemStack-ep` 的差距从 12.6pp 收敛到 3.4pp（episode 保底生效）；multi-session / preference 仍是弱项；
- 本次数字是 9B 本地模型自判，reader 与 judge 的能力边界会直接影响分数；换任意 OpenAI 兼容端点即可复跑同口径数字：
  `MEMSTACK_LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... python qa_eval.py --dataset locomo --reader llm --judge llm`；
- **外部库基线是检索下限**：`infer=False`、纯 dense、hashing 向量，没有 LLM 事实抽取和 BM25；它只用于同题集、同预算的相对对照，不代表该系统的完整能力。

---

## 设计原则

1. **原文与知识都保留**：episode 做证据，fact / preference / procedure 做路由；不要把唯一副本压成摘要。
2. **追加式而不是覆盖式**：冲突用 `superseded + valid_to` 表达，历史查询仍可回答。
3. **检索是多信号的**：纯向量对专名、数字、时间不敏感；dense + BM25 + entity + prior + graph 融合。
4. **粒度由查询决定**：不是所有问题都该看 turn、fact 或 summary；用熵路由 + 意图先验做软选择。
5. **上下文是预算问题**：只注入 top-K、去重（MMR）、带来源、可解释，而不是把整库塞进去。
6. **复杂查询走证据链**：多跳 / 枚举聚合类问题分解成子查询逐跳检索、分名额合并，只补缺口、不挤占首轮证据。
7. **更新与遗忘是一等公民**：软删除、级联隐藏证据、审计、离线巩固。
8. **框架无关**：只依赖 Protocol；Python / 工具调用 / HTTP / MCP 都能接。

---

## 局限

- 默认抽取器是规则化的，适合演示与受控基准；生产应换成 `LLMExtractor`（接口已留好）。
- 证据链的查询分解器同样是规则化的（枚举/桥接/时序正则），LLM 化分解在路线图中。
- SQLite + 暴力向量扫描面向单机、中小规模；大规模请替换 `MemoryStore` / 向量索引。
- 图扩展是 O(E) 的 PPR，没有社区划分、增量图索引或分布式图存储。
- 多模态、代码仓库状态、字段级 ACL 与加密存储尚未实现；多 Agent 团队与 grant 已提供。
- 合成基准的 reader 是规则化模拟，不能替代真实 LLM 端到端评测。
- `qa_eval.py` 的全量公开 QA 已跑（LoCoMo 1986 题、LongMemEval 500 题），但 reader/judge 仍是 9B 本地模型、预算 600–800 token；对外报分需在相同 backbone、相同预算下复跑。
- 默认 `--grounding strict`（只用召回记忆作答），开放域题因此只有 13.5%；`--grounding adaptive` 可放开这一类，但两种口径要分开报。
- 规则抽取器对 multi-session 聚合与隐式偏好（LongMemEval preference 6.7%）仍弱；这两类更适合 LLM 抽取 + 聚合类检索。

更完整的取舍、压力测试与路线图见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## License

Apache-2.0，见 [LICENSE](LICENSE)。
