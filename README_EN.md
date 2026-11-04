# MemStack

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-0.4.1-orange)](CHANGELOG.md)
[![CI](https://img.shields.io/badge/CI-ubuntu%20%7C%20windows%20%7C%203.10--3.12-blueviolet)](.github/workflows/ci.yml)

**A portable layered memory stack for LLM agents.** The core is pure Python standard library with zero third-party dependencies. Embed it in an existing agent, run it as a standalone HTTP memory service, or mount it as an MCP server.

[Quick start](#quick-start) · [Core concepts](#core-concepts) · [Usage](#usage) · [Configuration](#configuration) · [Evaluation](#evaluation) · [中文](README.md)

---

## Why MemStack

Agent memory problems are rarely about storage capacity. They show up as five failure modes:

| Failure mode | Symptom | MemStack mechanism |
|---|---|---|
| **Can't remember** | User preferences and project constraints vanish across sessions | Long-term storage + user profile block |
| **Can't find** | Pure vector search misses names, numbers, dates, and cross-session links | Five-channel fusion: dense + BM25 + entity + prior + PPR |
| **Gets it wrong** | Keeps answering with an old city or employer after the user moves on | Bi-temporal state + `superseded` + history queries |
| **Doesn't fit** | Dumping all history into context explodes token cost | Granularity routing + rerank + MMR + token budget |
| **Can't delete** | The underlying episode is still retrievable after a fact is deleted | Soft delete + cascade hiding of episodes/summaries + audit |

## Features

- **Write**: raw episodes plus facts / preferences / procedures / summaries / entity index, with dedupe, merge, and conflict supersession
- **Time**: bi-temporal (event time + write time), `active / superseded / archived / deleted` lifecycle; relative expressions such as `yesterday / last week / N years ago` are normalized to absolute dates on write (original wording preserved)
- **Retrieve**: dense + BM25 + entity + prior + graph expansion (PPR), entropy-based granularity routing, temporal policy, rerank, MMR, token budget; **a reserved budget for raw episodes** so extracted facts cannot crowd out evidence
- **Evidence chain**: multi-hop and enumeration queries are decomposed into sub-queries, retrieved hop by hop, and merged with per-sub-query slots — gaps are filled without displacing first-round evidence; optional LLM decomposition with automatic rule fallback
- **Consolidate**: offline dedupe, conflict resolution, session summaries, decay-based archiving, user profile blocks
- **Multi-agent**: agent registry, team spaces (open / closed), cross-agent grants, event bus
- **Interfaces**: Python API, `build_stack` factory, OpenAI function-calling tools, HTTP gateway, **zero-SDK MCP server (stdio)**, agent middleware
- **Portable**: `Embedder / Extractor / Store / Reranker / Clock` are all Protocols with replaceable implementations
- **LLM optional**: set `MEMSTACK_LLM_MODEL` + `OPENAI_API_KEY` to enable LLM extraction / summarization / query decomposition against any OpenAI-compatible endpoint

Architecture details: [ARCHITECTURE.md](ARCHITECTURE.md) (Chinese). API cost estimates: `python cost_estimate.py`.

---

## Quick start

Requires Python 3.10+. **The core has zero dependencies.**

```bash
# Optional: install as an importable package
pip install -e .

# Optional: external-library comparison baseline (only used by qa_eval.py --methods extlib)
pip install mem0ai
```

```python
from memstack import MemoryStack, Scope

stack = MemoryStack()  # SQLite :memory: by default; pass SQLiteStore("memory.db") for persistence

user = Scope(user_id="nora", agent_id="assistant")
session = Scope(user_id="nora", agent_id="assistant", session_id="s1")

stack.remember(
    [
        {"role": "user", "content": "I live in Lisbon. I work as a data analyst at Northwind Labs."},
        {"role": "assistant", "content": "Noted."},
    ],
    session,
    event_time=1767225600,  # optional; defaults to now
)

bundle = stack.recall("Where does the user live now?", user, token_budget=600)
print(bundle.text)
print(bundle.confidence, bundle.gaps, bundle.should_continue)
```

The return value is not plain text. It is a `ContextBundle` with sources, status, timestamps, and scores:

```text
## Retrieved memory
- [FACT|score=1.00|active|from=2026-01-01|src=ep-...] slot=user.location | value=Lisbon | evidence="I live in Lisbon."
```

---

## Core concepts

| Concept | Description |
|---|---|
| `Scope` | Tuple of `tenant_id / user_id / agent_id / session_id`; the first three define isolation, `session_id` only groups messages within a session |
| `MemoryKind` | `episode / fact / preference / procedure / summary / entity / reflection` |
| `MemoryStatus` | `active / superseded / archived / deleted`; only active items are retrieved by default, history queries opt in to old values |
| `ContextBundle` | Retrieval result: `text + items + confidence + coverage + gaps + conflicts + trace` |
| `RememberResult` | Write result: per-item `ADD / MERGE / SUPERSEDE / NOOP` decisions and provenance |

`episode` is verbatim evidence. `fact / preference / procedure` are extracted knowledge. `summary` is a navigation layer. `entity` only bridges the graph and never enters context directly.

---

## Usage

### Multi-agent teams (shared memory + event bus)

```python
from memstack import build_hub

hub = build_hub(db="team.db")           # LLM env vars auto-enable LLM extraction
hub.register_agent("planner", role="planner")
hub.register_agent("executor", role="executor")
hub.create_team("trip", members=["planner", "executor"], policy="closed")

hub.remember_as("planner", [{"role": "user", "content": "Budget cap is 5000."}],
                team_id="trip", share=True)
bundle = hub.recall_as("executor", "what is the trip budget?", teams=["trip"])
print(bundle.text)   # merged private + team context with source sections

hub.post_event("trip", "planner", "plan_ready", "Itinerary confirmed.")
events = hub.pull_events("executor", "trip")   # incremental event stream
```

Closed teams are member-only. Reading another agent's private memory requires an explicit
`hub.grant("executor", "planner", "read")`. Full example: `python examples/multiagent_orchestration.py`.

### Optional LLM integration

```bash
export MEMSTACK_LLM_MODEL=gpt-4o-mini    # or deepseek-chat / vLLM / Ollama, any OpenAI-compatible endpoint
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://.../v1    # set for non-default endpoints
python -c "from memstack import build_stack; s = build_stack(db='m.db'); \
           s.remember('The user moved to Berlin'); print(s.recall('where does the user live').text)"
```

Without these variables the system falls back to the zero-dependency rule extractor and never touches the network.

### Model backends (optional, for evaluation/inference)

Built-in backends remove the need to hand-write endpoints (see `memstack/backends.py`):

```bash
# Local Qwen3.5-9B served by Ollama (OpenAI-compatible /v1)
python qa_eval.py --reader llm --judge llm --model-base qwen35-9b-local
# Or via environment variable
export MEMSTACK_LLM_BASE=qwen35-9b-local
```

Endpoint and keys can be overridden with `OPENAI_BASE_URL` / `MEMSTACK_API_KEY` / `MEMSTACK_REASONING_EFFORT`; nothing is hard-coded. Qwen3.5-9B produces empty `content` when thinking is enabled by default, so this backend sets `reasoning_effort="none"`.

LLM query decomposition (optional, used by the evidence chain for multi-hop/enumeration questions; invalid JSON or call failures fall back to the rule version without breaking retrieval):

```python
from memstack import MemoryStack, LLMQueryDecomposer, OpenAICompatClient

client = OpenAICompatClient(model="gpt-4o-mini")
stack = MemoryStack(query_decomposer=LLMQueryDecomposer(client))
bundle = stack.recall("What hobbies does Evan pursue?", scope)
print(bundle.trace["evidence_chain"]["decomposer"])  # "llm" or "rule-fallback"
```

### OpenAI function calling and agent middleware

```python
tools = stack.as_tools()          # 5 tools: search / add / forget / profile / consolidate
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

### HTTP gateway

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

### MCP server (zero SDK)

```bash
python -m memstack.mcp_server --hub --db mem.db   # stdio JSON-RPC; mount in Cursor / Claude Desktop
python -m memstack.mcp_server list-tools          # list the 12 tools (memory + multi-agent)
python examples/mcp_client.py                     # stdlib demo client
```

Client `mcp.json`:

```json
{"mcpServers": {"memstack": {"command": "python", "args": ["-m", "memstack.mcp_server", "--hub", "--db", "mem.db"]}}}
```

### Custom backends

```python
class MyEmbedder:
    dim = 1024
    def embed(self, texts):
        ...

stack = MemoryStack(embedder=MyEmbedder())
```

Five replaceable points: `Embedder` (BGE / OpenAI / in-house), `Extractor` (`LLMExtractor` / domain model), `Summarizer`, `Reranker`, and `MemoryStore` (Postgres / Redis / vector store). The core depends only on Protocols and imports no third-party framework.

---

## Configuration

Common switches (full list in [ARCHITECTURE.md](ARCHITECTURE.md), Appendix B):

| Parameter | Default | Purpose |
|---|---|---|
| `top_k` | 8 | Number of returned items |
| `token_budget` | 800 | Context token budget |
| `episode_reserve_ratio` | 0.30 | Reserved budget share for raw episodes |
| `use_temporal` / `use_graph` / `use_routing` / `use_rerank` | True | Temporal policy / graph expansion / entropy routing / rerank |
| `evidence_chain` | True | Evidence chain: decompose → hop-by-hop retrieval → slot-based merge |
| `status_aware_packing` | True | Prefer active items and push historical values to the tail when over budget |
| `mmr_lambda` | 0.75 | MMR diversity strength |
| `archive_after_days` | 730 | Archive threshold for unaccessed memories |

---

## Layout

```text
memstack/
├── memstack/
│   ├── core.py            # facade: remember / recall / forget / consolidate / blocks
│   ├── multiagent.py      # agent registry, team spaces, grants, event bus
│   ├── mcp_server.py      # zero-dependency MCP server (stdio JSON-RPC)
│   ├── factory.py         # build_stack / build_hub (env-driven, optional LLM)
│   ├── types.py           # MemoryItem / Scope / ContextBundle / enums
│   ├── config.py          # all tunables
│   ├── interfaces.py      # portability Protocols
│   ├── embeddings.py      # zero-dep hashing vectors + batched cosine (numpy optional)
│   ├── extraction.py      # rule / LLM extractors, summarizers
│   ├── storage.py         # SQLite store (memories / edges / blocks / audit)
│   ├── graph.py           # entity linking, provenance, PPR
│   ├── retrieval.py       # hybrid recall, entropy routing, temporal policy, budget packing
│   ├── evidence_chain.py  # query decomposition, hop-by-hop retrieval, slot-based merge
│   ├── consolidation.py   # offline dedupe, summaries, decay, profile blocks
│   ├── tools.py           # OpenAI tool schemas, dispatch, agent middleware
│   ├── llm.py             # zero-SDK OpenAI-compatible client
│   ├── backends.py        # built-in OpenAI-compatible model backends
│   └── server.py          # stdlib HTTP gateway
├── examples/              # multi-agent orchestration, framework integration, MCP client
├── tests/                 # 56 unittests
├── data/                  # local LoCoMo / LongMemEval data
├── results/               # raw public evaluation results (JSON / Markdown)
├── demo.py                # end-to-end demo
├── benchmark.py           # synthetic benchmark (same-budget comparison + ablations)
├── public_eval.py         # LoCoMo / LongMemEval retrieval-layer evaluation
├── qa_eval.py             # LoCoMo / LongMemEval QA-layer evaluation: reader + judge
├── cost_estimate.py       # API cost estimation
├── ARCHITECTURE.md        # architecture and design trade-offs (Chinese)
├── IMPROVEMENTS.md        # improvements and comprehensive evaluation (Chinese)
├── PUBLIC_EVAL.md         # v1 public dataset results (Chinese)
├── CHANGELOG.md
└── LICENSE                # Apache-2.0
```

---

## Running

```bash
python demo.py
python benchmark.py
python -m unittest discover -s tests -v

# HTTP service
python -m memstack.server --db ./memstack.db --port 8765

# Public datasets: retrieval layer (data auto-downloads on first run)
python public_eval.py --dataset locomo --limit 300 --turns 3 --episode-chars 900 \
  --out results/locomo_300_comprehensive.json
python public_eval.py --dataset locomo --turns 3 --episode-chars 900 --no-download \
  --out results/locomo_full_comprehensive.json    # full 1986 questions (~15 min)
python public_eval.py --dataset longmemeval --turns 2 --episode-chars 700 \
  --out results/longmemeval_comprehensive.json

# Public datasets: QA layer (local small model as reader + judge, offline)
python qa_eval.py --limit 50 --budget 600 --reader hf --judge hf \
  --out results/locomo_qa.json

# QA layer: external-library comparison baseline (optional, requires pip install mem0ai)
python qa_eval.py --limit 50 --budget 600 --methods extlib \
  --reader hf --judge hf --out results/locomo_qa_extlib.json

# QA layer: switch to any OpenAI-compatible endpoint
MEMSTACK_LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
  python qa_eval.py --limit 200 --reader llm --judge llm
```

---

## Evaluation

### Synthetic benchmark (mechanism-level)

One 500-token context budget, 3 users × 4 sessions × 11 questions (33 total), rule-based reader.
Numbers validate mechanism trends only and are not comparable to production benchmarks.

| Method | Accuracy | Tokens/query | Evidence hit | Stale answers | Latency ms |
|---|---:|---:|---:|---:|---:|
| No memory | 0.0% | 0.0 | 0.0% | 0.0% | 0.0 |
| Full context (unbounded) | 100.0% | 544.0 | 90.9% | 0.0% | 0.0 |
| Full context (500 tokens) | 81.8% | 515.3 | 78.8% | 6.1% | 0.1 |
| Naive vector Top-4 | 69.7% | 251.7 | 81.8% | 21.2% | 3.3 |
| **MemStack (full)** | **100.0%** | **474.9** | **100.0%** | **0.0%** | 65.0 |
| w/o temporal | 90.9% | 475.0 | 100.0% | 9.1% | 79.8 |
| w/o graph | 100.0% | 482.0 | 100.0% | 0.0% | 74.5 |
| w/o routing | 100.0% | 475.1 | 100.0% | 0.0% | 77.2 |
| w/o rerank | 100.0% | 460.2 | 100.0% | 0.0% | 82.0 |
| w/o consolidation | 100.0% | 402.8 | 100.0% | 0.0% | 72.2 |
| episodes only | 60.6% | 483.7 | 100.0% | 27.3% | 49.8 |

Takeaways:

1. At the same budget, MemStack reaches 100% vs 81.8% for truncated full context and 69.7% for naive vectors;
2. Stale answers drop to 0% (naive vectors 21.2%, episodes-only 27.3%);
3. It matches unbounded full-context accuracy while using ~13% fewer tokens (544→475); the gap grows with longer history;
4. Graph expansion / granularity routing / rerank saturate on this small synthetic set; they mainly affect evidence coverage and cost. Temporal awareness is the only ablation that materially changes accuracy.

### Public datasets (retrieval layer)

Same 800-token budget, `top_k=8`; LoCoMo full 1986 questions, LongMemEval oracle version.
Full protocol, per-category results, and validity threats: [IMPROVEMENTS.md](IMPROVEMENTS.md); v1 results: [PUBLIC_EVAL.md](PUBLIC_EVAL.md).

| Dataset | Method | Unit R@k | All-Unit | Turn Token | Ans Token | NDCG | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| **LoCoMo** (full 1986) | **MemStack** (v4.1 + episode reserve) | **85.4%** | **75.6%** | **70.7%** | **43.1%** | **0.636** | **636** |
| | MemStack-mr (v3 bag-of-words expansion) | 83.6% | 73.6% | 69.3% | 41.8% | 0.623 | 622 |
| | MemStack-ep | **89.0%** | **79.8%** | **77.4%** | **46.3%** | **0.712** | 746 |
| | Fact+Hybrid | 83.0% | 73.4% | 69.7% | 42.0% | 0.637 | 667 |
| | Temporal-Graph | 84.3% | 74.3% | 69.8% | 42.1% | 0.639 | 633 |
| | Fact+Dense | 67.5% | 57.3% | 55.8% | 33.1% | 0.444 | 626 |
| **LoCoMo** (300 stratified) | **MemStack** (v4 evidence chain) | **83.8%** | **70.4%** | 69.3% | 37.3% | **0.626** | **648** |
| | MemStack-mr (v3 bag-of-words expansion) | 81.8% | 69.0% | 66.9% | 35.5% | 0.616 | 636 |
| | MemStack-ep | **86.9%** | **72.1%** | **72.8%** | **39.7%** | **0.653** | 740 |
| | BM25 only | 57.6% | 46.8% | 69.3% | 20.6% | 0.576 | 668 |
| | Dense only | 23.6% | 17.5% | 50.4% | 13.6% | 0.236 | 644 |
| **LongMemEval oracle** (500) | **MemStack** | 100% | **87.6%** | 64.5% | 52.5% | 1.000 | 759 |
| | MemStack-mr | 100% | 86.4% | 64.5% | 52.4% | 1.000 | 761 |
| | MemStack-ep | 100% | **95.8%** | 57.2% | **52.7%** | 1.000 | **450** |
| | Episode+Prior | 100% | **96.8%** | 58.0% | 53.7% | 1.000 | **506** |
| | Dense / BM25 / Full | 100% | 35.2% | 72–75% | 50–56% | 1.000 | 792 |

- At the same budget, MemStack / MemStack-ep / Episode+Prior cover significantly more evidence sessions than session-only retrieval;
- **The 0.4.1 episode budget reserve** lifts LongMemEval All-Unit from 79.2% to **87.6%** and Ans Token from 45.8% to **52.5%**: extracted facts no longer push verbatim evidence out of context;
- **The v4 evidence chain** (query decomposition → hop-by-hop retrieval → slot-based merge) vs v3 bag-of-words expansion: LoCoMo full All-Unit 73.6%→**75.6%**, Ans Token 41.8%→**43.1%**; mechanism details in [IMPROVEMENTS.md](IMPROVEMENTS.md);
- In LongMemEval oracle every session is evidence, so Unit R is 100% for all methods; **All-Unit** is the discriminating metric;
- LoCoMo multi-hop All-Unit still has a ceiling (~26% on the full set): gold evidence for enumeration questions is spread across 7–19 sessions and cannot all fit in a 600–800-token budget.

> Scope: this is **retrieval-layer evaluation**, not LLM-as-judge QA. LoCoMo is reported both on the full 1986 questions and on a 300-question stratified subset; LongMemEval uses the oracle version. Entity graph is disabled for LongMemEval to control CPU cost, so its MemStack numbers are a conservative lower bound.
> Raw files: `results/locomo_full_comprehensive_v041.json` · `results/locomo_300_comprehensive_v041.json` · `results/longmemeval_comprehensive_v041.json`.

### QA layer (LLM reader + judge)

`qa_eval.py` adds answer generation and judging on top of retrieved context. The tables below are **full-set** results with a local **Qwen3.5-9B** as both reader and judge (Ollama `qwen35-9b-local`).

**LoCoMo full 1986 questions (600-token budget, 2726s)**

| Method | Unit R@k | All-Unit | Overall accuracy | Answerable accuracy | Refusal accuracy | F1 | EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MemStack** | 84.2% | 74.1% | **54.3%** | **47.2%** | 78.7% | **37.4%** | **20.3%** |
| MemStack-mr | 82.7% | 72.5% | 52.5% | 45.8% | 75.6% | 36.5% | 19.8% |

**External-library baseline (50 stratified questions, 600 tokens, same 0.8B reader/judge)**

| Method | Unit R@k | All-Unit | Overall accuracy | Answerable accuracy | Refusal accuracy | F1 | EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| External-Lib (raw messages, no LLM extraction) | 91.7% | 83.3% | 22.0% | 20.0% | 30.0% | 13.7% | 5.0% |

**LongMemEval oracle full 500 questions (800-token budget, 1334s)**

| Method | Unit R@k | All-Unit | Overall accuracy | F1 | EM |
|---|---:|---:|---:|---:|---:|
| MemStack | 100% | 85.6% | 43.4% | 32.9% | 24.4% |
| **MemStack-ep** (raw episodes only) | 100% | **96.4%** | **46.8%** | **36.3%** | **25.0%** |

- Per category (MemStack): temporal 47.4% (7.5% before 0.4.1), single-hop 54.7%, multi-hop 36.2%, open-domain 13.5%, adversarial refusal 78.7%. `--grounding adaptive` allows world knowledge only for open-domain questions; on the 300-question subset open-domain goes 8.3%→35.0%;
- On LongMemEval the gap between `MemStack` and `MemStack-ep` narrows from 12.6pp to 3.4pp thanks to the episode reserve; multi-session and preference questions remain weak;
- These numbers are self-judged by a 9B local model, so reader/judge capability directly bounds the score. Re-run with any OpenAI-compatible endpoint for an equivalent protocol:
  `MEMSTACK_LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... python qa_eval.py --dataset locomo --reader llm --judge llm`;
- **The external-library baseline is a retrieval lower bound**: `infer=False`, dense-only, hashing vectors, with no LLM extraction or BM25. It is only meant for relative comparison under the same questions and budget, not as a measure of that system's full capability.

---

## Design principles

1. **Keep both evidence and knowledge**: episodes are evidence, facts/preferences/procedures route retrieval; never compress the only copy.
2. **Append, don't overwrite**: conflicts become `superseded + valid_to`; history stays answerable.
3. **Retrieval is multi-signal**: pure vectors are weak on names, numbers, and dates; fuse dense + BM25 + entity + prior + graph.
4. **Granularity follows the query**: not every question should read turns, facts, or summaries; use entropy routing plus intent priors as soft weights.
5. **Context is a budget problem**: inject top-K only, dedupe with MMR, keep sources, stay explainable.
6. **Complex queries use an evidence chain**: decompose multi-hop and enumeration questions into sub-queries, retrieve hop by hop, and merge with per-sub-query slots that fill gaps without displacing first-round evidence.
7. **Update and forget are first-class**: soft delete, cascade hiding, audit log, offline consolidation.
8. **Framework-neutral**: Protocols only; Python / tool calling / HTTP / MCP all work.

---

## Limitations

- The default extractor is rule-based, suitable for demos and controlled benchmarks; use `LLMExtractor` in production (interface ready).
- The evidence-chain decomposer is also rule-based (enumeration / bridging / temporal regex); LLM decomposition is on the roadmap.
- SQLite + brute-force vector scan targets single-node, small-to-medium scale; swap `MemoryStore` or the vector index at scale.
- PPR graph expansion is O(E); there is no community detection, incremental graph index, or distributed graph store.
- Multimodal data, codebase state, field-level ACL, and encrypted storage are not implemented; multi-agent teams and grants are available.
- The synthetic benchmark uses a rule-based reader and cannot replace end-to-end LLM evaluation.
- The full public QA runs (LoCoMo 1986, LongMemEval 500) use a 9B local reader/judge and a 600–800-token budget; publish numbers only after re-running with the same backbone and budget.
- `--grounding strict` (answer only from retrieved memory) is the default, which is why open-domain scores 13.5%; `--grounding adaptive` relaxes this but the two protocols must be reported separately.
- The rule extractor is weak on multi-session aggregation and implicit preferences (LongMemEval preference 6.7%); those need LLM extraction plus aggregation-aware retrieval.

For fuller trade-offs, stress tests, and roadmap, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
