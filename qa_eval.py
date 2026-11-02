"""LoCoMo QA 层评测：LLM reader + LLM judge，以及官方 mem0 仓库基线。

与 public_eval.py 的分工：
  - public_eval.py：只评**检索层**（证据会话是否召回、token 成本、延迟），不需要 LLM。
  - qa_eval.py    ：在检索结果之上加**reader 生成答案**和**judge 判分**，测最终答对率。

三种 reader / judge 后端，按可得资源选择：
  1. `hf`        ：本地 transformers 模型（默认 Qwen/Qwen3.5-0.8B，离线、无需 API key）
  2. `llm`       ：任意 OpenAI 兼容端点（`MEMSTACK_LLM_MODEL` + `OPENAI_API_KEY`）
  3. `extractive`/`f1`：纯规则，零依赖，作为**下界**而不是上限

用法：
    # 离线全流程：本地小模型当 reader 和 judge
    python qa_eval.py --limit 40 --budget 600 --reader hf --judge hf

    # 有 API key 时换成真正的大模型
    MEMSTACK_LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
        python qa_eval.py --limit 200 --reader llm --judge llm

    # 只算 token-F1 / EM（可与 A-MEM、SimpleMem 的 F1 口径对照，但仍受 reader 限制）
    python qa_eval.py --limit 300 --reader extractive --judge f1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memstack import HashingEmbedder, MemoryConfig, MemoryStack  # noqa: E402
from memstack.backends import build_backend_client, resolve_backend  # noqa: E402
from memstack.embeddings import estimate_tokens, tokenize  # noqa: E402
from memstack.llm import OpenAICompatClient  # noqa: E402

from public_eval import (  # noqa: E402
    DATA_DIR,
    LOCOMO_URL,
    LONGMEMEVAL_URL,
    RESULTS_DIR,
    EvalItem,
    MethodOutput,
    NullExtractor,
    UnitIndex,
    aggregate,
    bm25_output,
    build_lme_item,
    build_locomo_items,
    download_if_needed,
    full_output,
    ingest_lme,
    ingest_locomo,
    load_lme,
    load_locomo,
    memstack_output,
    score_output,
)

ARTICLES = {"a", "an", "the"}
PUNCT = set(string.punctuation + "，。！？、；：“”‘’（）【】《》…—·")


# ======================================================================================
# 文本与判分工具
# ======================================================================================
def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in PUNCT else " " for ch in text)
    tokens = [t for t in text.split() if t and t not in ARTICLES]
    return " ".join(tokens)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = defaultdict(int)
    for token in pred_tokens:
        common[token] += 1
    overlap = 0
    for token in gold_tokens:
        if common[token] > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


REFUSAL_PATTERNS = (
    "no information", "not mentioned", "not provided", "not specified", "unknown",
    "cannot be determined", "can't be determined", "insufficient information",
    "no relevant", "not in the memory", "unable to answer", "i don't know",
)


def is_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [p.strip(" -*") for p in parts if p and p.strip(" -*")]


# ======================================================================================
# Reader
# ======================================================================================
class ExtractiveReader:
    """零依赖 reader：从上下文里挑与问题词重叠最高的句子。只作为 oracle 下界。"""

    name = "extractive"

    def answer(self, question: str, context: str, category: str) -> str:
        q_tokens = {t for t in tokenize(question) if len(t) > 1}
        sentences = split_sentences(context)
        if not sentences or not q_tokens:
            return "No information available"
        best, best_score = "", 0.0
        for sentence in sentences:
            tokens = set(tokenize(sentence))
            if not tokens:
                continue
            overlap = len(q_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / (len(tokens) ** 0.5)
            if score > best_score:
                best, best_score = sentence, score
        if not best:
            return "No information available"
        return best[:240]


def reader_system_prompt(category: str, grounding: str = "strict") -> str:
    """读取侧协议：严格接地（默认）或对开放域问题允许世界知识。

    两个要点来自实测失败模式：
      1. 时序题（LoCoMo temporal 7.5%）模型直接复述 "yesterday / last week"，
         必须显式要求换算成绝对日期（记忆行里已带 from=/on=）。
      2. 开放域题（9.4%）清一色拒答，是"只用记忆"协议的必然结果。
    """
    system = (
        "You answer questions using the retrieved memory. "
        "Every memory line carries metadata: from=YYYY-MM-DD, and episode lines also carry on=<date>. "
        "If the question asks when something happened, convert the relative time expressions in the "
        "memory (yesterday, last week, N years ago) into an absolute date using that memory line's date, "
        "and answer with the absolute date. "
        "If the question asks for a recommendation, answer what the user would prefer according to the memory. "
        "Reply with a short phrase (a few words), no explanation. "
    )
    if grounding == "adaptive" and category == "open-domain":
        return system + (
            "This question asks about general world knowledge: use the retrieved memory when it is relevant, "
            "otherwise answer from your own knowledge."
        )
    return system + "If the memory does not contain the answer, reply exactly: No information available"


class HFReader:
    """本地 transformers reader：离线、无需 API key，速度取决于模型大小。"""

    def __init__(self, model_name: str, max_new_tokens: int = 24, grounding: str = "strict") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.grounding = grounding
        self.name = f"hf:{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def complete(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip()

    def answer(self, question: str, context: str, category: str) -> str:
        system = reader_system_prompt(category, self.grounding)
        user = f"Retrieved memory:\n{context}\n\nQuestion: {question}\nAnswer:"
        reply = self.complete(system, user)
        return reply.split("\n")[0].strip() or "No information available"


class APIReader:
    """OpenAI 兼容端点 reader。"""

    def __init__(
        self, model: str, client: OpenAICompatClient | None = None, grounding: str = "strict",
        max_new_tokens: int = 48,
    ) -> None:
        self.client = client if client is not None else OpenAICompatClient(model=model)
        self.grounding = grounding
        self.max_new_tokens = max_new_tokens
        self.name = f"llm:{model}"

    def complete(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        return self.client.complete(
            user, system=system, max_tokens=max_new_tokens or self.max_new_tokens,
        ).strip()

    def answer(self, question: str, context: str, category: str) -> str:
        system = reader_system_prompt(category, self.grounding)
        user = f"Retrieved memory:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.client.complete(user, system=system, max_tokens=self.max_new_tokens).strip().split("\n")[0]


# ======================================================================================
# Judge
# ======================================================================================
class F1Judge:
    """确定性判分：token-F1 + Exact Match，适合和公开 F1 口径对照。"""

    name = "f1"

    def score(self, question: str, gold: str, pred: str, adversarial: bool) -> dict[str, Any]:
        if adversarial:
            return {"correct": 1.0 if is_refusal(pred) else 0.0, "f1": 0.0, "em": 0.0}
        return {
            "correct": 1.0 if token_f1(pred, gold) >= 0.5 else 0.0,
            "f1": token_f1(pred, gold),
            "em": exact_match(pred, gold),
        }


class LLMJudge:
    """LLM-as-judge：二元判定预测是否与标准答案语义等价；对抗题判是否拒答。"""

    def __init__(self, backend: Any, name: str) -> None:
        self.backend = backend
        self.name = f"judge:{name}"

    def score(self, question: str, gold: str, pred: str, adversarial: bool) -> dict[str, Any]:
        if adversarial:
            # 对抗题没有 gold answer，官方口径是「模型是否拒绝作答」；
            # 小模型当 judge 时对「No information available」也常误判，
            # 因此这里用确定性拒答检测，不再调用 LLM。
            return {"correct": 1.0 if is_refusal(pred) else 0.0, "f1": 0.0, "em": 0.0}
        system = (
            "You are a reasonable evaluator of question answering. "
            "Reply CORRECT if the prediction conveys the same core fact as the gold answer. "
            "Accept paraphrases, singular/plural differences, extra detail, and partial answers "
            "that include the gold fact. "
            "Reply INCORRECT if the prediction contradicts the gold answer, names a different "
            "entity, or does not contain the gold fact. "
            "Reply with exactly CORRECT or INCORRECT."
        )
        user = (
            f"Question: {question}\nGold answer: {gold}\n"
            f"Prediction: {pred}\nVerdict:"
        )
        verdict = self.backend.complete(system, user, max_new_tokens=6).upper()
        correct = 1.0 if "CORRECT" in verdict and "INCORRECT" not in verdict else 0.0
        return {"correct": correct, "f1": token_f1(pred, gold), "em": exact_match(pred, gold)}


# ======================================================================================
# 官方 mem0 仓库基线
# ======================================================================================
class HashEmbedderAdapter:
    """把 MemStack 的零依赖 hashing 向量接进 mem0 的 EmbeddingBase 接口。"""

    def __init__(self, embedder: HashingEmbedder) -> None:
        self.embedder = embedder
        self.config = SimpleNamespace(embedding_dims=embedder.dim)

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self.embedder.embed([text])[0]

    def embed_batch(self, texts: Sequence[str], memory_action: str = "add") -> list[list[float]]:
        return self.embedder.embed(list(texts))


class Mem0Backend:
    """运行 mem0ai 官方包（infer=False 原始消息模式），而不是机制近似。

    限制（重要）：
      - 没有 LLM API key，因此不做 mem0 的 LLM 事实抽取，只存原始消息；
      - 向量后端换成零依赖 hashing 向量，避免下载 sentence-transformers 模型；
      - 检索是 mem0 自己的 dense（装了 fastembed 时会自动叠加 BM25）。
    因此这一行是「官方代码的检索下限」，不是 mem0 论文里的完整管线。
    """

    name = "Mem0(官方, 无LLM抽取)"

    def __init__(self, samples: Sequence[dict[str, Any]], budget: int, top_k: int = 40) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        from mem0 import Memory  # 延迟导入：没装 mem0ai 时给出明确提示

        self.budget = budget
        self.top_k = top_k
        self.tmpdir = tempfile.mkdtemp(prefix="mem0_eval_")
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "locomo",
                    "path": os.path.join(self.tmpdir, "qdrant"),
                    "embedding_model_dims": 512,
                    "on_disk": False,
                },
            },
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "api_key": "sk-dummy"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", "api_key": "sk-dummy"}},
            "history_db_path": os.path.join(self.tmpdir, "history.db"),
        }
        self.memory = Memory.from_config(config)
        self.memory.embedding_model = HashEmbedderAdapter(HashingEmbedder())
        self._ingest(samples)

    def _ingest(self, samples: Sequence[dict[str, Any]]) -> None:
        for sample in samples:
            user_id = f"locomo:{sample['sample_id']}"
            conversation = sample["conversation"]
            keys = sorted(
                [k for k in conversation if re.fullmatch(r"session_\d+", k)],
                key=lambda key: int(key.split("_")[1]),
            )
            for key in keys:
                messages = [
                    {"role": "user", "content": f"{t.get('speaker', '')}: {t.get('text', '')}"}
                    for t in conversation.get(key) or []
                ]
                if messages:
                    self.memory.add(
                        messages, user_id=user_id, infer=False, metadata={"session_id": key},
                    )

    def output(self, item: EvalItem) -> tuple[MethodOutput, float]:
        started = time.perf_counter()
        result = self.memory.search(
            item.question, filters={"user_id": item.scope.user_id}, top_k=self.top_k,
        )
        rows = result.get("results", result) if isinstance(result, dict) else result
        chosen: list[str] = []
        units: list[str] = []
        total = 0
        for row in rows or []:
            text = row.get("memory") or ""
            cost = estimate_tokens(text)
            if chosen and total + cost > self.budget:
                break
            chosen.append(text)
            total += cost
            session_id = (row.get("metadata") or {}).get("session_id")
            if session_id and session_id not in units:
                units.append(session_id)
        latency_ms = (time.perf_counter() - started) * 1000
        context = "\n".join(f"- {text}" for text in chosen)
        return MethodOutput(context=context, units=units, tokens=total), latency_ms

    def close(self) -> None:
        """显式关闭 qdrant 本地客户端；否则进程退出时 __del__ 可能触发原生崩溃。"""
        try:
            client = getattr(self.memory.vector_store, "client", None)
            if client is not None:
                client.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响评测结果
            pass


# ======================================================================================
# 主流程
# ======================================================================================
def build_reader(
    kind: str, model: str, max_new_tokens: int, client: Any | None = None, grounding: str = "strict",
) -> Any:
    if kind == "extractive":
        return ExtractiveReader()
    if kind == "hf":
        return HFReader(model, max_new_tokens=max_new_tokens, grounding=grounding)
    if kind == "llm":
        return APIReader(
            model, client=client, grounding=grounding, max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"未知 reader: {kind}")


def build_judge(kind: str, model: str, reader: Any, client: Any | None = None) -> Any:
    if kind == "f1":
        return F1Judge()
    if kind == "hf":
        # 复用同一个本地模型，避免两份权重；max_new_tokens 由每次调用显式传入
        backend = reader if isinstance(reader, HFReader) else HFReader(model, max_new_tokens=6)
        return LLMJudge(backend, getattr(reader, "name", model))
    if kind == "llm":
        return LLMJudge(APIReader(model, client=client), model)
    raise ValueError(f"未知 judge: {kind}")


def run(args: argparse.Namespace, backend: Any | None = None) -> dict[str, Any]:
    base_time = 1_672_531_200.0  # 2023-01-01
    # 本次优化引入的三个可对照开关，留空则用 MemoryConfig 默认值
    overrides: dict[str, Any] = {}
    if getattr(args, "episode_reserve", None) is not None:
        overrides["episode_reserve_ratio"] = args.episode_reserve
    if getattr(args, "agg_bonus", None) is not None:
        overrides["aggregation_top_k_bonus"] = args.agg_bonus
    if getattr(args, "no_relative_time", False):
        overrides["relative_time_normalization"] = False
    config = MemoryConfig(
        max_turns_per_episode=args.turns,
        max_similar_edges=0,
        conflict_candidate_limit=120,
        enable_entity_linking=True,
        use_graph=True,
        top_k=args.top_k,
        token_budget=args.budget,
        episode_char_limit=args.episode_chars,
        **overrides,
    )
    episode_config = MemoryConfig(
        max_turns_per_episode=args.turns,
        max_similar_edges=0,
        conflict_candidate_limit=60,
        enable_entity_linking=False,
        use_graph=False,
        top_k=args.top_k,
        token_budget=args.budget,
        episode_char_limit=args.episode_chars,
        relative_time_normalization=overrides.get("relative_time_normalization", True),
    )
    stack = MemoryStack(config=config)
    stack_ep = MemoryStack(config=episode_config, extractor=NullExtractor())
    # v4 证据链 vs v3 词袋补证的 QA 层对照：共享同一记忆库，只换读取配置
    stack_mr = MemoryStack(
        config=replace(config, evidence_chain=False, multi_round=True),
        store=stack.store,
        embedder=stack.embedder,
        extractor=stack.extractor,
        summarizer=stack.summarizer,
        clock=stack.clock,
    )
    started = time.time()
    episode_maps: dict[str, dict[str, str]] = {}
    episode_maps_ep: dict[str, dict[str, str]] = {}
    samples: list[dict[str, Any]] = []
    if args.dataset == "longmemeval":
        path = DATA_DIR / "longmemeval_oracle.json"
        if not args.no_download:
            path = download_if_needed(path, LONGMEMEVAL_URL)
        raw_items = load_lme(path, args.limit)
        print(f"[longmemeval] questions={len(raw_items)}")
        items: list[EvalItem] = []
        for raw in raw_items:
            qid = raw["question_id"]
            episode_maps[qid] = ingest_lme(stack, raw, base_time)
            episode_maps_ep[qid] = ingest_lme(stack_ep, raw, base_time)
            items.append(build_lme_item(raw))
        print(f"[longmemeval] memstack ingest {time.time() - started:.1f}s")
    else:
        path = DATA_DIR / "locomo10.json"
        if not args.no_download:
            path = download_if_needed(path, LOCOMO_URL)
        samples = load_locomo(path, args.conversations)
        print(f"[locomo] conversations={len(samples)}")
        for sample in samples:
            episode_maps[sample["sample_id"]] = ingest_locomo(stack, sample, base_time)
            episode_maps_ep[sample["sample_id"]] = ingest_locomo(stack_ep, sample, base_time)
        print(f"[locomo] memstack ingest {time.time() - started:.1f}s")
        items = build_locomo_items(samples, episode_maps, args.limit)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.dataset != "locomo":
        methods = [m for m in methods if m != "mem0"]
    mem0_backend = None
    if "mem0" in methods:
        t0 = time.time()
        try:
            mem0_backend = Mem0Backend(samples, args.budget, top_k=args.top_k * 5)
            print(f"[mem0] ingest {time.time() - t0:.1f}s（官方包，infer=False）")
        except Exception as exc:  # noqa: BLE001 - 没装 mem0ai 时降级
            print(f"[mem0] 跳过：{type(exc).__name__}: {exc}")
            methods = [m for m in methods if m != "mem0"]

    print(f"[{args.dataset}] questions={len(items)}  budget={args.budget}  methods={methods}")

    embedder = HashingEmbedder()
    index_cache: dict[int, UnitIndex] = {}

    def get_index(item: EvalItem) -> UnitIndex:
        key = id(item.units)
        if key not in index_cache:
            index_cache[key] = UnitIndex(item.units, embedder)
        return index_cache[key]

    client = build_backend_client(backend) if backend is not None else None
    reader = build_reader(
        args.reader, args.model_name, args.max_new_tokens, client,
        grounding=getattr(args, "grounding", "strict"),
    )
    judge = build_judge(args.judge, args.model_name, reader, client)
    print(f"[setup] reader={reader.name}  judge={judge.name}")
    if backend is not None:
        print(f"[setup] model-base={backend.name}  base_url={backend.base_url}")

    rows: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    done: dict[str, set[str]] = {m: set() for m in methods}
    if args.resume and Path(args.out).exists():
        previous = json.loads(Path(args.out).read_text(encoding="utf-8"))
        for method, data in (previous.get("methods") or {}).items():
            if method not in rows:
                continue
            for sample in data.get("samples") or []:
                rows[method].append(sample)
                done[method].add(sample["qid"])
        print(f"[resume] 已恢复 {sum(len(v) for v in done.values())} 条记录")

    def checkpoint() -> None:
        if not args.save_every:
            return
        partial = build_result(args, methods, rows, reader.name, judge.name, len(items), started)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    for idx, item in enumerate(items):
        sample_id = item.qid.split(":", 1)[0]
        adversarial = item.category == "adversarial"
        outputs: dict[str, tuple[MethodOutput, float]] = {}

        if "memstack" in methods and item.qid not in done["memstack"]:
            t0 = time.perf_counter()
            out = memstack_output(stack, item, episode_maps[sample_id], args.budget)
            outputs["memstack"] = (out, (time.perf_counter() - t0) * 1000)
        if "memstack-mr" in methods and item.qid not in done["memstack-mr"]:
            t0 = time.perf_counter()
            out = memstack_output(stack_mr, item, episode_maps[sample_id], args.budget)
            outputs["memstack-mr"] = (out, (time.perf_counter() - t0) * 1000)
        if "memstack-ep" in methods and item.qid not in done["memstack-ep"]:
            t0 = time.perf_counter()
            out = memstack_output(stack_ep, item, episode_maps_ep[sample_id], args.budget)
            outputs["memstack-ep"] = (out, (time.perf_counter() - t0) * 1000)
        if "mem0" in methods and mem0_backend is not None and item.qid not in done["mem0"]:
            outputs["mem0"] = mem0_backend.output(item)
        if "bm25" in methods and item.qid not in done["bm25"]:
            index = get_index(item)
            t0 = time.perf_counter()
            selected = bm25_output(item, index, args.top_k, args.budget)
            outputs["bm25"] = (selected, (time.perf_counter() - t0) * 1000)
        if "full" in methods and item.qid not in done["full"]:
            t0 = time.perf_counter()
            outputs["full"] = (full_output(item, args.budget), (time.perf_counter() - t0) * 1000)

        for method, (output, latency_ms) in outputs.items():
            record = score_output(item, output, latency_ms)
            prediction = reader.answer(item.question, output.context, item.category)
            qa_started = time.perf_counter()
            verdict = judge.score(item.question, item.answer, prediction, adversarial)
            qa_latency = (time.perf_counter() - qa_started) * 1000
            record.update(
                {
                    "qid": item.qid,
                    "category": item.category,
                    "question": item.question,
                    "gold": item.answer,
                    "prediction": prediction,
                    "correct": verdict["correct"],
                    "f1": verdict["f1"],
                    "em": verdict["em"],
                    "qa_latency_ms": qa_latency,
                }
            )
            rows[method].append(record)
            done[method].add(item.qid)
        if (idx + 1) % 5 == 0:
            print(f"  qa {idx + 1}/{len(items)}", flush=True)
            checkpoint()

    stack.close()
    stack_ep.close()
    if mem0_backend is not None:
        mem0_backend.close()

    result = build_result(args, methods, rows, reader.name, judge.name, len(items), started)
    print_summary(result["methods"], reader.name, judge.name, len(items), args.dataset)
    result["elapsed_s"] = round(time.time() - started, 1)
    return result


def build_result(
    args: argparse.Namespace,
    methods: Sequence[str],
    rows: dict[str, list[dict[str, Any]]],
    reader_name: str,
    judge_name: str,
    total_questions: int,
    started: float,
) -> dict[str, Any]:
    """把逐题记录汇总成结果字典；checkpoint 与最终输出共用同一份逻辑。"""
    summary: dict[str, Any] = {}
    for method in methods:
        method_rows = rows[method]
        answerable = [r for r in method_rows if r["category"] != "adversarial"]
        adversarial = [r for r in method_rows if r["category"] == "adversarial"]
        total = len(method_rows)
        summary[method] = {
            "retrieval": aggregate(method_rows),
            "qa": {
                "n": total,
                "accuracy": sum(r["correct"] for r in method_rows) / max(total, 1),
                "answerable_accuracy": (
                    sum(r["correct"] for r in answerable) / len(answerable) if answerable else 0.0
                ),
                "refusal_accuracy": (
                    sum(r["correct"] for r in adversarial) / len(adversarial) if adversarial else 0.0
                ),
                "refusal_n": len(adversarial),
                "f1": sum(r["f1"] for r in answerable) / len(answerable) if answerable else 0.0,
                "em": sum(r["em"] for r in answerable) / len(answerable) if answerable else 0.0,
                "avg_qa_latency_ms": (
                    sum(r["qa_latency_ms"] for r in method_rows) / max(total, 1)
                ),
                "by_category": {
                    category: {
                        "n": len(group),
                        "accuracy": sum(r["correct"] for r in group) / len(group),
                        "f1": sum(r["f1"] for r in group) / len(group),
                    }
                    for category, group in sorted(_group_by(method_rows, "category").items())
                },
            },
            "samples": method_rows,
        }
    return {
        "dataset": f"{args.dataset}_qa",
        "questions": total_questions,
        "completed": max((len(v) for v in rows.values()), default=0),
        "budget": args.budget,
        "reader": reader_name,
        "judge": judge_name,
        "methods": summary,
        "elapsed_s": round(time.time() - started, 1),
    }


def _group_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)


def print_summary(
    summary: dict[str, Any], reader: str, judge: str, n: int, dataset: str = "locomo",
) -> None:
    label = "LongMemEval" if dataset == "longmemeval" else "LoCoMo"
    print(f"\n{'=' * 118}")
    print(f"{label} QA  reader={reader}  judge={judge}  n={n}")
    print("=" * 118)
    print(
        f"{'Method':<26}{'UnitR@k':>9}{'AllUnit':>9}{'Tokens':>8}"
        f"{'Accuracy':>10}{'AnsAcc':>9}{'Refusal':>9}{'F1':>8}{'EM':>7}{'LatMs':>9}"
    )
    print("-" * 118)
    for method, data in summary.items():
        retrieval = data["retrieval"]
        qa = data["qa"]
        print(
            f"{method:<26}{retrieval['unit_recall_any'] * 100:>8.1f}%"
            f"{retrieval['unit_recall_all'] * 100:>8.1f}%{retrieval['avg_tokens']:>8.1f}"
            f"{qa['accuracy'] * 100:>9.1f}%{qa['answerable_accuracy'] * 100:>8.1f}%"
            f"{qa['refusal_accuracy'] * 100:>8.1f}%{qa['f1'] * 100:>7.1f}%"
            f"{qa['em'] * 100:>6.1f}%{qa['avg_qa_latency_ms']:>9.0f}"
        )


def to_markdown(result: dict[str, Any]) -> str:
    label = "LongMemEval" if result.get("dataset") == "longmemeval_qa" else "LoCoMo"
    lines = [
        f"# {label} QA 层评测（LLM reader + judge）",
        "",
        f"- 问题数：{result['questions']}；上下文预算：{result['budget']} token",
        f"- Reader：`{result['reader']}`；Judge：`{result['judge']}`",
        f"- 模型基座：`{(result.get('model_base') or {}).get('name', '未启用（环境变量端点）')}`",
        f"- 耗时：{result['elapsed_s']}s",
        "",
        "| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, data in result["methods"].items():
        retrieval, qa = data["retrieval"], data["qa"]
        refusal = (
            f"{qa['refusal_accuracy'] * 100:.1f}%" if qa.get("refusal_n") else "—"
        )
        lines.append(
            f"| {method} | {retrieval['unit_recall_any'] * 100:.1f}% | "
            f"{retrieval['unit_recall_all'] * 100:.1f}% | {retrieval['avg_tokens']:.1f} | "
            f"{qa['accuracy'] * 100:.1f}% | {qa['answerable_accuracy'] * 100:.1f}% | "
            f"{refusal} | {qa['f1'] * 100:.1f}% | "
            f"{qa['em'] * 100:.1f}% | {qa['avg_qa_latency_ms']:.0f} |"
        )
    lines.append("")
    lines.append("## 分题型准确率")
    lines.append("")
    for method, data in result["methods"].items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| 类别 | n | 准确率 | F1 |")
        lines.append("|---|---:|---:|---:|")
        for category, row in data["qa"]["by_category"].items():
            lines.append(
                f"| {category} | {row['n']} | {row['accuracy'] * 100:.1f}% | "
                f"{row['f1'] * 100:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo / LongMemEval QA evaluation with LLM reader/judge")
    parser.add_argument(
        "--dataset", choices=["locomo", "longmemeval"], default="locomo",
        help="评测数据集：locomo（1986 QA）或 longmemeval（oracle 500 QA）",
    )
    parser.add_argument("--limit", type=int, default=40, help="问题数（分层抽样），0=全部")
    parser.add_argument("--conversations", type=int, default=0)
    parser.add_argument("--budget", type=int, default=600, help="上下文 token 预算")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--episode-chars", type=int, default=900)
    parser.add_argument(
        "--methods", default="memstack,memstack-mr,memstack-ep,mem0,bm25",
        help="逗号分隔：memstack,memstack-mr,memstack-ep,mem0,bm25,full（full=预算内最近历史）",
    )
    parser.add_argument("--reader", choices=["hf", "llm", "extractive"], default="extractive")
    parser.add_argument("--judge", choices=["hf", "llm", "f1"], default="f1")
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--judge-model", default="")
    parser.add_argument(
        "--model-base",
        default="",
        help="可选模型基座（内置：qwen35-9b-local；留空读 MEMSTACK_LLM_BASE）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--grounding", choices=["strict", "adaptive"], default="strict",
        help="strict=只用召回记忆；adaptive=开放域题允许世界知识（对抗题仍强制拒答）",
    )
    parser.add_argument(
        "--episode-reserve", type=float, default=None,
        help="原文 episode 的预算保底比例，默认取 MemoryConfig（0.30）；0=关闭",
    )
    parser.add_argument(
        "--agg-bonus", type=int, default=None,
        help="聚合类问题额外放宽的 top_k，默认取 MemoryConfig（4）；0=关闭",
    )
    parser.add_argument(
        "--no-relative-time", action="store_true",
        help="关闭写入时的相对时间归一化（用于 A/B 对照）",
    )
    parser.add_argument("--save-every", type=int, default=5, help="每 N 题落一次 checkpoint，0=关闭")
    parser.add_argument("--resume", action="store_true", help="从已有 --out 结果续跑")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out", default="", help="结果路径；留空按数据集命名")
    args = parser.parse_args()
    if not args.out:
        args.out = str(
            RESULTS_DIR / ("locomo_qa.json" if args.dataset == "locomo" else "longmemeval_qa.json")
        )

    backend = resolve_backend(args.model_base or None)
    args.model_name = backend.model if backend is not None else (args.judge_model or args.hf_model)
    if args.reader == "llm" or args.judge == "llm":
        has_env = bool(os.getenv("MEMSTACK_LLM_MODEL")) and bool(os.getenv("OPENAI_API_KEY"))
        if backend is None and not has_env:
            parser.error(
                "--reader/--judge 使用 llm 时需要 --model-base（如 qwen35-9b-local），"
                "或设置 MEMSTACK_LLM_MODEL 与 OPENAI_API_KEY"
            )

    result = run(args, backend)
    if backend is not None:
        result["model_base"] = {
            "name": backend.name,
            "model": backend.model,
            "base_url": backend.base_url,
            "reasoning_effort": backend.reasoning_effort,
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.out).with_suffix(".md")
    markdown_path.write_text(to_markdown(result), encoding="utf-8")
    print(f"\n[saved] {args.out}\n[saved] {markdown_path}")
    sys.stdout.flush()
    # Windows 上 qdrant-local + torch 在同进程退出时可能触发 0xC0000005；
    # 结果已落盘，这里直接退出避免把成功运行误报成失败。
    os._exit(0)


if __name__ == "__main__":
    main()
