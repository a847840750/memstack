# MemStack

**A portable layered memory stack for LLM agents.** Pure Python standard library core — zero third-party dependencies — embeddable in any agent, or runnable as an HTTP memory service or MCP server.

- Write: raw episodes + facts / preferences / procedures / summaries / entity index, with dedupe, merge, and conflict supersession
- Time: bi-temporal (event time + write time), `active / superseded / archived / deleted` lifecycle
- Retrieve: dense + BM25 + entity + prior + Personalized PageRank, entropy-based granularity routing, temporal policy, rerank, MMR, token budget
- Consolidate: offline dedupe, conflict resolution, session summaries, decay archiving, user profile blocks
- Multi-agent: agent registry, team spaces (open/closed), cross-agent grants, event bus
- Interfaces: Python API, `build_stack()` factory, OpenAI function-calling tools, HTTP gateway, **zero-SDK MCP server (stdio)**, agent middleware
- Portable: `Embedder / Extractor / Store / Reranker / Clock` are all Protocols with replaceable implementations
- LLM optional: set `MEMSTACK_LLM_MODEL` + `OPENAI_API_KEY` to switch on LLM extraction/summarization against any OpenAI-compatible endpoint

For architecture details see [ARCHITECTURE.md](ARCHITECTURE.md) (Chinese); API cost estimates: `python cost_estimate.py`.

---

## Quick start

Requires Python 3.10+. The core has **zero dependencies**; the listed extras are optional.

```bash
pip install -e .            # optional local install
pip install numpy           # optional: accelerates dense ranking
pip install mem0ai          # optional: qa_eval.py official-mem0 baseline only
```

```python
from memstack import build_stack, Scope

stack = build_stack(db="memory.db")   # LLM env vars auto-enable LLM extraction

user = Scope(user_id="nora", agent_id="assistant")
stack.remember(
    [{"role": "user", "content": "I live in Lisbon and work at Northwind Labs."}],
    user,
)
bundle = stack.recall("Where does the user live now?", user, token_budget=600)
print(bundle.text)          # "- [FACT|score=1.00|active|src=ep-...] slot=user.location | value=Lisbon"
print(bundle.confidence, bundle.gaps, bundle.should_continue)
```

### Multi-agent teams

```python
from memstack import build_hub

hub = build_hub(db="team.db")
hub.register_agent("planner", role="planner")
hub.register_agent("executor", role="executor")
hub.create_team("trip", members=["planner", "executor"], policy="closed")

hub.remember_as("planner", [{"role": "user", "content": "Budget cap is 5000."}],
                team_id="trip", share=True)
bundle = hub.recall_as("executor", "what is the trip budget", teams=["trip"])
print(bundle.text)  # merged private + team context with source sections

hub.post_event("trip", "planner", "plan_ready", "Itinerary confirmed.")
print(hub.pull_events("executor", "trip"))
```

- closed teams: members only; open teams: anyone can read, members write.
- Cross-agent access to private memory requires an explicit grant:
  `hub.grant("executor", "planner", "read")`.

### MCP server (zero SDK)

```bash
python -m memstack.mcp_server --hub --db mem.db   # stdio JSON-RPC; mount in Cursor/Claude Desktop
python -m memstack.mcp_server list-tools          # list the 12 tools
python examples/mcp_client.py                     # stdlib-driven demo client
```

Client config (`mcp.json`):

```json
{"mcpServers": {"memstack": {"command": "python", "args": ["-m", "memstack.mcp_server", "--hub", "--db", "mem.db"]}}}
```

### Optional real LLM extraction

```bash
export MEMSTACK_LLM_MODEL=deepseek-chat        # any OpenAI-compatible model
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.deepseek.com/v1
```

Without these variables the system degrades gracefully to the zero-dependency rule extractor — no network, deterministic, fully offline.

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
│   ├── consolidation.py   # offline dedupe, summaries, decay, profile blocks
│   ├── tools.py           # OpenAI tool schemas, dispatch, agent middleware
│   ├── llm.py             # zero-SDK OpenAI-compatible client
│   └── server.py          # stdlib HTTP gateway
├── examples/              # multi-agent orchestration, framework integration, MCP client
├── demo.py / benchmark.py / public_eval.py / qa_eval.py / cost_estimate.py
├── tests/                 # 26 unittests
└── README.md / ARCHITECTURE.md / PUBLIC_EVAL.md / IMPROVEMENTS.md
```

---

## Evaluations (measured)

Retrieval-layer results on public benchmarks, and a QA-layer harness (`qa_eval.py`) with
local-HF / OpenAI-compatible / extractive readers and f1 / LLM judges, including a run of the
**official `mem0ai` package** as a real-repo baseline. Full protocol, caveats, and the public
SOTA score table live in [IMPROVEMENTS.md](IMPROVEMENTS.md) (Chinese).

| Benchmark | Method | Unit R@k | All-Unit | QA Accuracy (0.8B reader) |
|---|---:|---:|---:|
| LoCoMo (300 stratified) | MemStack | 82.5% | 67.7% | — |
| LongMemEval oracle (500) | MemStack-ep | 100% | 95.0% | — |
| LoCoMo QA (50 stratified) | MemStack | 87.5% | 77.1% | 46.0% |
| LoCoMo QA (50 stratified) | Official mem0 (infer=False) | 91.7% | 83.3% | 22.0% |

All numbers are retrieval/QA-layer measures under an 800-token budget; they are **not**
comparable to vendor-reported LoCoMo J / F1 scores (different readers, judges, and protocols).

---

## Design principles (one-liners)

1. Keep both evidence and knowledge: episodes are the evidence, facts route; never compress the only copy.
2. Append, don't overwrite: conflicts become `superseded + valid_to`; history stays answerable.
3. Retrieval is multi-signal: dense + BM25 + entity + prior + graph.
4. Granularity follows the query: entropy routing + intent priors, not one size fits all.
5. Context is a budget problem: top-K, MMR-deduplicated, sourced, inspectable.
6. Update and forget are first-class: soft delete, cascade hiding, audit log.
7. Framework-neutral and dependency-free: Protocols everywhere; Python / tools / HTTP / MCP all work.

## Limitations

- Default extractor is rule-based (demo/controlled benchmarks); production should enable LLM extraction via env vars.
- SQLite + brute-force vector scan targets single-node small/medium scale; swap the store/embedder for scale.
- PPR graph expansion is O(E); no community detection or incremental graph index.
- No multimodal, codebase state, or encrypted storage yet.
- Public QA results use a 0.8B local reader on a 50-question subset — widen the sample and use a stronger reader before publishing numbers.

## License

Apache-2.0. See [LICENSE](LICENSE).