"""公开数据集评测：LoCoMo + LongMemEval (oracle)。

评测目标是衡量**记忆检索层**，而不是端到端 QA 分数：
  - Unit Recall@k：证据所在会话是否被取回
  - All-Unit Recall：所有证据会话是否都取回
  - Evidence-Turn Hit：证据原句是否出现在上下文中
  - Answer-in-Context：标准答案是否出现在上下文中
  - MRR / token 成本

对比方法：
  - MemStack：完整写入 + 混合检索 + 时间/图/重排
  - Dense only：原始会话窗口 + hashing 向量
  - BM25 only：原始会话窗口 + 词法检索
  - Full context：同预算下截断的最近历史

用法：
    python public_eval.py --dataset locomo --limit 100
    python public_eval.py --dataset longmemeval --limit 100
    python public_eval.py --dataset both
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memstack import HashingEmbedder, MemoryConfig, MemoryKind, MemoryStack, Scope  # noqa: E402
from memstack.embeddings import cosine, estimate_tokens, tokenize  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

LOCOMO_URL = "https://modelscope.cn/datasets/evalscope/locomo/resolve/master/locomo10.json"
LONGMEMEVAL_URL = "https://hf-mirror.com/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle"

LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


# ======================================================================================
# 通用工具
# ======================================================================================
def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    text = re.sub(r"[^\w\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def download_if_needed(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}\n        -> {path}")
    import urllib.request

    started = time.time()
    urllib.request.urlretrieve(url, path)
    print(f"[download] {path.stat().st_size / 1e6:.1f} MB in {time.time() - started:.1f}s")
    return path


def parse_locomo_date(value: str, fallback: float) -> float:
    if not value:
        return fallback
    match = re.search(
        r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})",
        value, re.I,
    )
    if not match:
        return fallback
    hour, minute = int(match.group(1)), int(match.group(2))
    if match.group(3).lower() == "pm" and hour != 12:
        hour += 12
    if match.group(3).lower() == "am" and hour == 12:
        hour = 0
    month = MONTHS.get(match.group(5).lower())
    if not month:
        return fallback
    try:
        dt = datetime(int(match.group(6)), month, int(match.group(4)), hour, minute)
        return float(calendar.timegm(dt.timetuple()))
    except ValueError:
        return fallback


def parse_lme_date(value: str, fallback: float) -> float:
    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value or "")
    if not match:
        return fallback
    try:
        dt = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return float(calendar.timegm(dt.timetuple()))
    except ValueError:
        return fallback


def bm25(query: str, docs: Sequence[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    q_tokens = [t for t in tokenize(query) if len(t) > 1]
    doc_tokens = [tokenize(text) for text in docs]
    n = len(doc_tokens)
    if not q_tokens or n == 0:
        return [0.0] * n
    avgdl = sum(len(d) for d in doc_tokens) / n
    df: dict[str, int] = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] += 1
    scores = []
    for tokens in doc_tokens:
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        score = 0.0
        for token in set(q_tokens):
            if token not in tf:
                continue
            idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
            denom = tf[token] + k1 * (1 - b + b * len(tokens) / max(avgdl, 1e-6))
            score += idf * tf[token] * (k1 + 1) / max(denom, 1e-6)
        scores.append(score)
    return scores


def truncate_units(units: Sequence["Unit"], budget: int) -> list["Unit"]:
    """Full-context 基线：在预算内保留最近的历史单元。"""
    chosen: list[Unit] = []
    total = 0
    for unit in reversed(units):
        cost = estimate_tokens(unit.text)
        if total + cost > budget:
            if not chosen and unit.text:
                cut = max(160, int(len(unit.text) * budget / max(cost, 1)))
                chosen.append(Unit(id=unit.id, session_id=unit.session_id, text=unit.text[:cut]))
            break
        chosen.append(unit)
        total += cost
    chosen.reverse()
    return chosen


# ======================================================================================
# 数据结构
# ======================================================================================
@dataclass
class Unit:
    id: str
    session_id: str
    text: str
    turns: list[dict[str, Any]] = field(default_factory=list)


class UnitIndex:
    """会话级检索索引：向量与 BM25 统计只构建一次，供同一 conversation 的所有问题复用。"""

    def __init__(self, units: Sequence[Unit], embedder: HashingEmbedder) -> None:
        self.units = list(units)
        self.embedder = embedder
        self.vectors = embedder.embed([u.text for u in self.units]) if self.units else []
        self.doc_tokens = [tokenize(u.text) for u in self.units]
        self.avgdl = (
            sum(len(tokens) for tokens in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens else 0.0
        )
        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1

    def dense(self, query: str, top_k: int) -> list[Unit]:
        if not self.units:
            return []
        query_vec = self.embedder.embed([query])[0]
        scored = [
            (cosine(query_vec, vec), unit)
            for vec, unit in zip(self.vectors, self.units)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [unit for _, unit in scored[:top_k]]

    def bm25(self, query: str, top_k: int) -> list[Unit]:
        if not self.units:
            return []
        q_tokens = [t for t in tokenize(query) if len(t) > 1]
        n = len(self.units)
        scores = []
        for tokens in self.doc_tokens:
            tf: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            score = 0.0
            for token in set(q_tokens):
                if token not in tf:
                    continue
                idf = math.log(1 + (n - self.df[token] + 0.5) / (self.df[token] + 0.5))
                denom = tf[token] + 1.2 * (1 - 0.75 + 0.75 * len(tokens) / max(self.avgdl, 1e-6))
                score += idf * tf[token] * 2.2 / max(denom, 1e-6)
            scores.append(score)
        ranked = sorted(zip(scores, self.units), key=lambda pair: pair[0], reverse=True)
        return [unit for _, unit in ranked[:top_k]]


def fit_ranked_units(units: Sequence[Unit], budget: int) -> list[Unit]:
    """把 top-k 会话裁剪到 token 预算内，保证与 MemStack 同预算比较。"""
    chosen: list[Unit] = []
    total = 0
    for unit in units:
        cost = estimate_tokens(unit.text)
        if total + cost > budget:
            if not chosen and unit.text:
                cut = max(160, int(len(unit.text) * budget / max(cost, 1)))
                chosen.append(Unit(id=unit.id, session_id=unit.session_id, text=unit.text[:cut]))
            break
        chosen.append(unit)
        total += cost
    return chosen


@dataclass
class EvalItem:
    qid: str
    question: str
    answer: str
    category: str
    units: list[Unit]
    gold_units: set[str]
    gold_turns: list[str]
    scope: Scope
    has_evidence: bool = True


@dataclass
class MethodOutput:
    context: str
    units: list[str]
    tokens: int


# ======================================================================================
# LoCoMo
# ======================================================================================
def load_locomo(path: Path, conversations: int = 0) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:conversations] if conversations else data


def locomo_units(sample: dict[str, Any]) -> tuple[list[Unit], dict[str, str], dict[str, str]]:
    conversation = sample["conversation"]
    keys = sorted(
        [key for key in conversation if re.fullmatch(r"session_\d+", key)],
        key=lambda key: int(key.split("_")[1]),
    )
    units: list[Unit] = []
    dia_to_session: dict[str, str] = {}
    dia_to_text: dict[str, str] = {}
    for key in keys:
        turns = conversation.get(key) or []
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")
            dia = turn.get("dia_id")
            if dia:
                dia_to_session[dia] = key
                dia_to_text[dia] = text
        units.append(Unit(id=key, session_id=key, text="\n".join(lines), turns=turns))
    return units, dia_to_session, dia_to_text


def ingest_locomo(stack: MemoryStack, sample: dict[str, Any], base_time: float) -> dict[str, str]:
    """写入一个 LoCoMo conversation，返回 episode_id -> session_id 映射。"""
    conversation = sample["conversation"]
    sample_id = sample["sample_id"]
    keys = sorted(
        [key for key in conversation if re.fullmatch(r"session_\d+", key)],
        key=lambda key: int(key.split("_")[1]),
    )
    for idx, key in enumerate(keys):
        turns = conversation.get(key) or []
        event_time = parse_locomo_date(
            conversation.get(f"{key}_date_time", ""), base_time + idx * 86400 * 7,
        )
        messages = []
        for turn_idx, turn in enumerate(turns):
            messages.append(
                {
                    "role": "user",
                    "content": f"{turn.get('speaker', '')}: {turn.get('text', '')}",
                    "dia_id": turn.get("dia_id"),
                    "session_id": key,
                    "turn_index": turn_idx,
                }
            )
        stack.remember(
            messages,
            Scope(user_id=f"locomo:{sample_id}", agent_id="locomo", session_id=key),
            event_time=event_time,
        )
    scope = Scope(user_id=f"locomo:{sample_id}", agent_id="locomo")
    mapping: dict[str, str] = {}
    for episode in stack.store.find(scope, kinds=[MemoryKind.EPISODE], limit=5000):
        session_id = episode.scope.session_id
        if not session_id:
            for message in episode.data.get("messages", []):
                session_id = message.get("session_id")
                if session_id:
                    break
        if session_id:
            mapping[episode.id] = session_id
    return mapping


def build_locomo_items(
    samples: Sequence[dict[str, Any]],
    episode_maps: dict[str, dict[str, str]],
    limit: int = 0,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        units, dia_to_session, dia_to_text = locomo_units(sample)
        unit_by_id = {unit.id: unit for unit in units}
        for qa in sample.get("qa", []):
            evidence = qa.get("evidence") or []
            if isinstance(evidence, str):
                evidence = re.findall(r"D\d+:\d+", evidence)
            evidence = [str(e) for e in evidence]
            gold_units = {dia_to_session[d] for d in evidence if d in dia_to_session}
            gold_turns = [dia_to_text[d] for d in evidence if d in dia_to_text]
            category = LOCOMO_CATEGORIES.get(qa.get("category"), f"category-{qa.get('category')}")
            items.append(
                EvalItem(
                    qid=f"{sample_id}:{len(items)}",
                    question=qa.get("question", ""),
                    answer=str(qa.get("answer", "")),
                    category=category,
                    units=units,
                    gold_units=gold_units,
                    gold_turns=gold_turns,
                    scope=Scope(user_id=f"locomo:{sample_id}", agent_id="locomo"),
                    has_evidence=bool(gold_units),
                )
            )
    if limit and len(items) > limit:
        items = stratified_sample(items, limit, "category")
    return items


# ======================================================================================
# LongMemEval
# ======================================================================================
def stratified_sample(items: Sequence[Any], limit: int, key: str) -> list[Any]:
    """按类别轮询抽样，保证子集覆盖所有题型。"""
    if not limit or len(items) <= limit:
        return list(items)
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        groups[str(getattr(item, key))].append(item)
    ordered = [groups[k] for k in sorted(groups)]
    sampled: list[Any] = []
    cursor = 0
    while len(sampled) < limit:
        added = False
        for group in ordered:
            if cursor < len(group):
                sampled.append(group[cursor])
                added = True
                if len(sampled) >= limit:
                    break
        if not added:
            break
        cursor += 1
    return sampled


def load_lme(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit and len(data) > limit:
        # dict 没有属性，按 question_type 分组轮询
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in data:
            groups[str(item.get("question_type", "unknown"))].append(item)
        ordered = [groups[k] for k in sorted(groups)]
        sampled: list[dict[str, Any]] = []
        cursor = 0
        while len(sampled) < limit:
            added = False
            for group in ordered:
                if cursor < len(group):
                    sampled.append(group[cursor])
                    added = True
                    if len(sampled) >= limit:
                        break
            if not added:
                break
            cursor += 1
        return sampled
    return data


def ingest_lme(stack: MemoryStack, item: dict[str, Any], base_time: float) -> dict[str, str]:
    """写入一个 LongMemEval haystack，返回 episode_id -> session_id 映射。"""
    qid = item["question_id"]
    session_ids = item.get("haystack_session_ids") or []
    sessions = item.get("haystack_sessions") or []
    dates = item.get("haystack_dates") or []
    for idx, session in enumerate(sessions):
        session_id = str(session_ids[idx]) if idx < len(session_ids) else f"s{idx}"
        date_text = dates[idx] if idx < len(dates) else ""
        event_time = parse_lme_date(date_text, base_time + idx * 86400)
        messages = []
        for turn_idx, turn in enumerate(session):
            messages.append(
                {
                    "role": str(turn.get("role", "user")),
                    "content": str(turn.get("content", "")),
                    "session_id": session_id,
                    "turn_index": turn_idx,
                    "has_answer": bool(turn.get("has_answer", False)),
                }
            )
        stack.remember(
            messages,
            Scope(user_id=f"lme:{qid}", agent_id="lme", session_id=session_id),
            event_time=event_time,
        )
    scope = Scope(user_id=f"lme:{qid}", agent_id="lme")
    mapping: dict[str, str] = {}
    for episode in stack.store.find(scope, kinds=[MemoryKind.EPISODE], limit=5000):
        session_id = episode.scope.session_id
        if not session_id:
            for message in episode.data.get("messages", []):
                session_id = message.get("session_id")
                if session_id:
                    break
        if session_id:
            mapping[episode.id] = session_id
    return mapping


def lme_units(item: dict[str, Any]) -> list[Unit]:
    session_ids = item.get("haystack_session_ids") or []
    sessions = item.get("haystack_sessions") or []
    units: list[Unit] = []
    for idx, session in enumerate(sessions):
        session_id = str(session_ids[idx]) if idx < len(session_ids) else f"s{idx}"
        turns = [
            {"role": str(t.get("role", "user")), "content": str(t.get("content", "")),
             "has_answer": bool(t.get("has_answer", False))}
            for t in session
        ]
        text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        units.append(Unit(id=session_id, session_id=session_id, text=text, turns=turns))
    return units


def build_lme_item(item: dict[str, Any]) -> EvalItem:
    qid = item["question_id"]
    units = lme_units(item)
    gold_units = {str(s) for s in (item.get("answer_session_ids") or [])}
    gold_turns = []
    for unit in units:
        for turn in unit.turns:
            if turn.get("has_answer"):
                gold_turns.append(turn["content"])
    if not gold_turns and gold_units:
        for unit in units:
            if unit.id in gold_units:
                gold_turns.extend(t["content"] for t in unit.turns)
    qtype = item.get("question_type") or "unknown"
    return EvalItem(
        qid=qid,
        question=item.get("question", ""),
        answer=str(item.get("answer", "")),
        category=qtype,
        units=units,
        gold_units=gold_units,
        gold_turns=gold_turns,
        scope=Scope(user_id=f"lme:{qid}", agent_id="lme"),
        has_evidence=bool(gold_units or gold_turns),
    )


# ======================================================================================
# 方法与指标
# ======================================================================================
class NullExtractor:
    def extract(self, text: str, event_time: float, context: dict[str, Any] | None = None) -> list[Any]:
        return []


def variant_stack(base: MemoryStack, **overrides: Any) -> MemoryStack:
    """复用同一份写入结果，只替换检索配置，做同库可控对照。"""
    config = replace(base.config, **overrides)
    config.validate()
    return MemoryStack(
        config=config,
        store=base.store,
        embedder=base.embedder,
        extractor=base.extractor,
        summarizer=base.summarizer,
        clock=base.clock,
    )


def build_variant_stacks(base: MemoryStack) -> dict[str, tuple[MemoryStack, list[MemoryKind] | None]]:
    """机制基线：共享同一记忆库，只改变读取策略，避免重复写入带来的差异。"""
    return {
        # MemStack-mr：v2/v3 的多轮词袋补证（关闭 v4 证据链），用于隔离证据链的增量
        "MemStack-mr": (
            variant_stack(base, evidence_chain=False, multi_round=True),
            None,
        ),
        # Fact+Hybrid：抽取事实 + 语义/词法/实体混合，不做时间/图/路由/重排
        "Fact+Hybrid": (
            variant_stack(
                base,
                use_temporal=False, use_graph=False, use_routing=False, use_rerank=False,
                multi_round=False, evidence_expansion=False, evidence_chain=False,
                weights={"dense": 0.55, "lexical": 0.30, "entity": 0.15, "prior": 0.0, "graph": 0.0},
            ),
            None,
        ),
        # Temporal-Graph：时间 + 图 + 混合检索
        "Temporal-Graph": (
            variant_stack(
                base,
                use_temporal=True, use_graph=True, use_routing=False, use_rerank=True,
                multi_round=False, evidence_expansion=True, evidence_chain=False,
            ),
            None,
        ),
        # Fact+Dense：事实笔记 + 纯向量
        "Fact+Dense": (
            variant_stack(
                base,
                use_temporal=False, use_graph=False, use_routing=False, use_rerank=False,
                multi_round=False, evidence_expansion=False, evidence_chain=False,
                weights={"dense": 1.0, "lexical": 0.0, "entity": 0.0, "prior": 0.0, "graph": 0.0},
            ),
            None,
        ),
        # Episode+Prior：只取 episode + 热度/时效先验
        "Episode+Prior": (
            variant_stack(
                base,
                use_temporal=False, use_graph=False, use_routing=False, use_rerank=False,
                multi_round=False, evidence_expansion=False, evidence_chain=False,
                weights={"dense": 0.55, "lexical": 0.15, "entity": 0.0, "prior": 0.30, "graph": 0.0},
            ),
            [MemoryKind.EPISODE],
        ),
    }


def memstack_output(
    stack: MemoryStack,
    item: EvalItem,
    episode_map: dict[str, str],
    budget: int,
    kinds: Sequence[MemoryKind] | None = None,
) -> MethodOutput:
    bundle = stack.recall(item.question, item.scope, token_budget=budget, kinds=kinds)
    selected_units: list[str] = []
    seen: set[str] = set()
    for hit in bundle.items:
        episode_ids = [hit.memory.id] if hit.memory.kind == MemoryKind.EPISODE else hit.memory.source_ids
        for episode_id in episode_ids:
            unit_id = episode_map.get(episode_id)
            if unit_id and unit_id not in seen:
                seen.add(unit_id)
                selected_units.append(unit_id)
    return MethodOutput(context=bundle.text, units=selected_units, tokens=bundle.token_estimate)


def dense_output(item: EvalItem, index: UnitIndex, top_k: int, budget: int) -> MethodOutput:
    chosen = fit_ranked_units(index.dense(item.question, top_k), budget)
    context = "\n\n".join(u.text for u in chosen)
    return MethodOutput(context=context, units=[u.id for u in chosen], tokens=estimate_tokens(context))


def bm25_output(item: EvalItem, index: UnitIndex, top_k: int, budget: int) -> MethodOutput:
    chosen = fit_ranked_units(index.bm25(item.question, top_k), budget)
    context = "\n\n".join(u.text for u in chosen)
    return MethodOutput(context=context, units=[u.id for u in chosen], tokens=estimate_tokens(context))


def full_output(item: EvalItem, budget: int) -> MethodOutput:
    chosen = truncate_units(item.units, budget)
    context = "\n\n".join(u.text for u in chosen)
    return MethodOutput(context=context, units=[u.id for u in chosen], tokens=estimate_tokens(context))


def token_recall(gold: str, context: str) -> float:
    gold_tokens = set(tokenize(gold, keep_stopwords=False))
    if not gold_tokens:
        return 0.0
    context_tokens = set(tokenize(context, keep_stopwords=False))
    return len(gold_tokens & context_tokens) / len(gold_tokens)


def score_output(item: EvalItem, output: MethodOutput, latency_ms: float = 0.0) -> dict[str, Any]:
    unit_set = set(output.units)
    gold_units = item.gold_units
    unit_any = bool(gold_units & unit_set) if gold_units else False
    unit_all = gold_units.issubset(unit_set) if gold_units else False
    rank = 0
    for idx, unit_id in enumerate(output.units, start=1):
        if unit_id in gold_units:
            rank = idx
            break
    mrr = 1.0 / rank if rank else 0.0
    relevance = [1.0 if unit_id in gold_units else 0.0 for unit_id in output.units]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal_hits = min(len(gold_units), len(output.units))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    context_norm = norm(output.context)
    turn_hit = any(norm(turn) and norm(turn) in context_norm for turn in item.gold_turns)
    answer_hit = bool(item.answer) and norm(item.answer) in context_norm
    turn_recalls = [token_recall(turn, output.context) for turn in item.gold_turns]
    answer_recall = token_recall(item.answer, output.context) if item.answer else 0.0
    return {
        "unit_any": unit_any,
        "unit_all": unit_all,
        "mrr": mrr,
        "ndcg": ndcg,
        "turn_hit": turn_hit,
        "turn_token_recall": sum(turn_recalls) / len(turn_recalls) if turn_recalls else 0.0,
        "answer_hit": answer_hit,
        "answer_token_recall": answer_recall,
        "tokens": output.tokens,
        "latency_ms": latency_ms,
        "has_evidence": item.has_evidence,
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def aggregate(records: Sequence[dict[str, Any]], key: str | None = None) -> dict[str, Any]:
    rows = [r for r in records if r["has_evidence"]] if key is None else records
    if not rows:
        return {"n": 0}
    metrics = {
        "unit_recall_any": [float(r["unit_any"]) for r in rows],
        "unit_recall_all": [float(r["unit_all"]) for r in rows],
        "turn_token_recall": [r["turn_token_recall"] for r in rows],
        "answer_token_recall": [r["answer_token_recall"] for r in rows],
    }
    result = {
        "n": len(rows),
        "unit_recall_any": sum(metrics["unit_recall_any"]) / len(rows),
        "unit_recall_all": sum(metrics["unit_recall_all"]) / len(rows),
        "mrr": sum(r["mrr"] for r in rows) / len(rows),
        "ndcg": sum(r["ndcg"] for r in rows) / len(rows),
        "turn_hit": sum(r["turn_hit"] for r in rows) / len(rows),
        "turn_token_recall": sum(metrics["turn_token_recall"]) / len(rows),
        "answer_in_context": sum(r["answer_hit"] for r in rows) / len(rows),
        "answer_token_recall": sum(metrics["answer_token_recall"]) / len(rows),
        "avg_tokens": sum(r["tokens"] for r in rows) / len(rows),
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
        "ci95": {
            name: list(bootstrap_ci(values))
            for name, values in metrics.items()
        },
    }
    return result


def print_table(dataset: str, method_rows: dict[str, list[dict[str, Any]]]) -> None:
    print(f"\n{'=' * 140}")
    print(f"Dataset: {dataset}")
    print(f"{'=' * 140}")
    print(
        f"{'Method':<18}{'UnitR@k':>9}{'AllUnit':>9}{'TurnHit':>9}{'TurnTok':>9}"
        f"{'AnsInCtx':>9}{'AnsTok':>8}{'NDCG':>8}{'MRR':>8}{'Tokens':>8}{'LatMs':>8}{'N':>6}"
    )
    print("-" * 140)
    for method, records in method_rows.items():
        agg = aggregate(records)
        print(
            f"{method:<18}{agg['unit_recall_any'] * 100:>8.1f}%"
            f"{agg['unit_recall_all'] * 100:>8.1f}%{agg['turn_hit'] * 100:>8.1f}%"
            f"{agg['turn_token_recall'] * 100:>8.1f}%{agg['answer_in_context'] * 100:>8.1f}%"
            f"{agg['answer_token_recall'] * 100:>7.1f}%{agg['ndcg']:>8.3f}{agg['mrr']:>8.3f}"
            f"{agg['avg_tokens']:>8.1f}{agg['avg_latency_ms']:>8.1f}{agg['n']:>6}"
        )


def category_table(records: Sequence[dict[str, Any]], label: str) -> None:
    print(f"\n{label}（MemStack）")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
    for category, rows in sorted(groups.items()):
        agg = aggregate(rows)
        print(
            f"  {category:<22} n={agg['n']:<4} unitR={agg['unit_recall_any'] * 100:5.1f}% "
            f"turnTok={agg['turn_token_recall'] * 100:5.1f}% ansTok={agg['answer_token_recall'] * 100:5.1f}%"
        )


# ======================================================================================
# 主流程
# ======================================================================================
def run_locomo(args: argparse.Namespace) -> dict[str, Any]:
    path = download_if_needed(DATA_DIR / "locomo10.json", LOCOMO_URL) if not args.no_download else DATA_DIR / "locomo10.json"
    samples = load_locomo(path, args.conversations)
    print(f"[locomo] conversations={len(samples)}")
    base_time = calendar.timegm(datetime(2023, 1, 1).timetuple())
    config = MemoryConfig(
        max_turns_per_episode=args.turns,
        max_similar_edges=0,
        conflict_candidate_limit=120,
        enable_entity_linking=True,
        use_graph=True,
        top_k=args.top_k,
        token_budget=args.budget,
        episode_char_limit=args.episode_chars,
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
    )
    stack = MemoryStack(config=config)
    stack_ep = MemoryStack(config=episode_config, extractor=NullExtractor())
    started = time.time()
    episode_maps: dict[str, dict[str, str]] = {}
    episode_maps_ep: dict[str, dict[str, str]] = {}
    for idx, sample in enumerate(samples):
        episode_maps[sample["sample_id"]] = ingest_locomo(stack, sample, base_time)
        episode_maps_ep[sample["sample_id"]] = ingest_locomo(stack_ep, sample, base_time)
        print(f"  ingested {idx + 1}/{len(samples)} {sample['sample_id']}", end="\r")
    print(f"\n[locomo] ingest {time.time() - started:.1f}s")
    items = build_locomo_items(samples, episode_maps, args.limit)
    print(f"[locomo] questions={len(items)}  eval budget={args.budget} top_k={args.top_k}")

    embedder = HashingEmbedder()
    index_cache: dict[int, UnitIndex] = {}

    def get_index(item: EvalItem) -> UnitIndex:
        key = id(item.units)
        if key not in index_cache:
            index_cache[key] = UnitIndex(item.units, embedder)
        return index_cache[key]

    methods: dict[str, list[dict[str, Any]]] = {
        "MemStack": [], "MemStack-mr": [], "MemStack-ep": [], "Fact+Hybrid": [],
        "Temporal-Graph": [], "Fact+Dense": [], "Episode+Prior": [], "Dense only": [],
        "BM25 only": [], "Full context": [],
    }
    variants = build_variant_stacks(stack)
    eval_started = time.time()
    for idx, item in enumerate(items):
        sample_id = item.qid.split(":", 1)[0]
        episode_map = episode_maps[sample_id]
        episode_map_ep = episode_maps_ep[sample_id]
        index = get_index(item)
        outputs: dict[str, tuple[MethodOutput, float]] = {}

        def timed(fn: Any) -> tuple[MethodOutput, float]:
            start = time.perf_counter()
            output = fn()
            return output, (time.perf_counter() - start) * 1000

        outputs["MemStack"] = timed(lambda: memstack_output(stack, item, episode_map, args.budget))
        outputs["MemStack-ep"] = timed(
            lambda: memstack_output(stack_ep, item, episode_map_ep, args.budget)
        )
        for name, (variant, kinds) in variants.items():
            outputs[name] = timed(
                lambda variant=variant, kinds=kinds: memstack_output(
                    variant, item, episode_map, args.budget, kinds,
                )
            )
        outputs["Dense only"] = timed(lambda: dense_output(item, index, args.top_k, args.budget))
        outputs["BM25 only"] = timed(lambda: bm25_output(item, index, args.top_k, args.budget))
        outputs["Full context"] = timed(lambda: full_output(item, args.budget))
        for method, (output, latency_ms) in outputs.items():
            record = score_output(item, output, latency_ms)
            record.update({"qid": item.qid, "category": item.category})
            methods[method].append(record)
        if (idx + 1) % 100 == 0:
            print(f"  evaluated {idx + 1}/{len(items)}", end="\r")
    print(f"\n[locomo] eval {time.time() - eval_started:.1f}s")
    stack.close()
    stack_ep.close()

    print_table(f"LoCoMo ({len(samples)} conversations, {len(items)} questions)", methods)
    category_table(methods["MemStack"], "LoCoMo 分题型")
    return {
        "dataset": "locomo",
        "conversations": len(samples),
        "questions": len(items),
        "budget": args.budget,
        "top_k": args.top_k,
        "elapsed_s": round(time.time() - started, 1),
        "methods": {
            method: {"overall": aggregate(records), "by_category": {
                cat: aggregate([r for r in records if r["category"] == cat])
                for cat in sorted({r["category"] for r in records})
            }}
            for method, records in methods.items()
        },
    }


def run_lme(args: argparse.Namespace) -> dict[str, Any]:
    path = download_if_needed(DATA_DIR / "longmemeval_oracle.json", LONGMEMEVAL_URL) if not args.no_download else DATA_DIR / "longmemeval_oracle.json"
    raw_items = load_lme(path, args.limit)
    print(f"[longmemeval] questions={len(raw_items)}")
    base_time = calendar.timegm(datetime(2023, 1, 1).timetuple())
    config = MemoryConfig(
        max_turns_per_episode=args.turns,
        max_similar_edges=0,
        conflict_candidate_limit=60,
        enable_entity_linking=False,
        use_graph=False,
        top_k=args.top_k,
        token_budget=args.budget,
        episode_char_limit=args.episode_chars,
    )
    episode_config = MemoryConfig(
        max_turns_per_episode=args.turns,
        max_similar_edges=0,
        conflict_candidate_limit=30,
        enable_entity_linking=False,
        use_graph=False,
        top_k=args.top_k,
        token_budget=args.budget,
        episode_char_limit=args.episode_chars,
    )
    stack = MemoryStack(config=config)
    stack_ep = MemoryStack(config=episode_config, extractor=NullExtractor())
    started = time.time()
    items: list[EvalItem] = []
    episode_maps: dict[str, dict[str, str]] = {}
    episode_maps_ep: dict[str, dict[str, str]] = {}
    for idx, raw in enumerate(raw_items):
        episode_maps[raw["question_id"]] = ingest_lme(stack, raw, base_time)
        episode_maps_ep[raw["question_id"]] = ingest_lme(stack_ep, raw, base_time)
        items.append(build_lme_item(raw))
        if (idx + 1) % 50 == 0:
            print(f"  ingested {idx + 1}/{len(raw_items)}", end="\r")
    print(f"\n[longmemeval] ingest {time.time() - started:.1f}s")

    embedder = HashingEmbedder()
    index_cache: dict[int, UnitIndex] = {}

    def get_index(item: EvalItem) -> UnitIndex:
        key = id(item.units)
        if key not in index_cache:
            index_cache[key] = UnitIndex(item.units, embedder)
        return index_cache[key]

    methods: dict[str, list[dict[str, Any]]] = {
        "MemStack": [], "MemStack-mr": [], "MemStack-ep": [], "Fact+Hybrid": [],
        "Temporal-Graph": [], "Fact+Dense": [], "Episode+Prior": [], "Dense only": [],
        "BM25 only": [], "Full context": [],
    }
    variants = build_variant_stacks(stack)
    eval_started = time.time()
    for idx, item in enumerate(items):
        episode_map = episode_maps[item.qid]
        episode_map_ep = episode_maps_ep[item.qid]
        index = get_index(item)
        outputs: dict[str, tuple[MethodOutput, float]] = {}

        def timed(fn: Any) -> tuple[MethodOutput, float]:
            start = time.perf_counter()
            output = fn()
            return output, (time.perf_counter() - start) * 1000

        outputs["MemStack"] = timed(lambda: memstack_output(stack, item, episode_map, args.budget))
        outputs["MemStack-ep"] = timed(
            lambda: memstack_output(stack_ep, item, episode_map_ep, args.budget)
        )
        for name, (variant, kinds) in variants.items():
            outputs[name] = timed(
                lambda variant=variant, kinds=kinds: memstack_output(
                    variant, item, episode_map, args.budget, kinds,
                )
            )
        outputs["Dense only"] = timed(lambda: dense_output(item, index, args.top_k, args.budget))
        outputs["BM25 only"] = timed(lambda: bm25_output(item, index, args.top_k, args.budget))
        outputs["Full context"] = timed(lambda: full_output(item, args.budget))
        for method, (output, latency_ms) in outputs.items():
            record = score_output(item, output, latency_ms)
            record.update({"qid": item.qid, "category": item.category})
            methods[method].append(record)
        if (idx + 1) % 100 == 0:
            print(f"  evaluated {idx + 1}/{len(items)}", end="\r")
    print(f"\n[longmemeval] eval {time.time() - eval_started:.1f}s")
    stack.close()
    stack_ep.close()

    print_table(f"LongMemEval oracle ({len(items)} questions)", methods)
    category_table(methods["MemStack"], "LongMemEval 分题型")
    return {
        "dataset": "longmemeval_oracle",
        "questions": len(items),
        "budget": args.budget,
        "top_k": args.top_k,
        "elapsed_s": round(time.time() - started, 1),
        "methods": {
            method: {"overall": aggregate(records), "by_category": {
                cat: aggregate([r for r in records if r["category"] == cat])
                for cat in sorted({r["category"] for r in records})
            }}
            for method, records in methods.items()
        },
    }


def to_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# MemStack 公开数据集评测结果", ""]
    lines.append("> 评测对象是记忆检索层；不是 LLM-as-judge QA 分数。")
    lines.append("")
    for result in results:
        lines.append(f"## {result['dataset']}")
        lines.append("")
        lines.append(
            f"- 问题数：{result['questions']}；预算：{result['budget']} token；"
            f"top_k：{result['top_k']}；耗时：{result['elapsed_s']}s"
        )
        lines.append("")
        lines.append("| 方法 | Unit R@k | All-Unit | Turn Token | Ans Token | NDCG@k | MRR | Tokens | Latency ms |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method, data in result["methods"].items():
            overall = data["overall"]
            lines.append(
                f"| {method} | {overall['unit_recall_any'] * 100:.1f}% | "
                f"{overall['unit_recall_all'] * 100:.1f}% | {overall['turn_token_recall'] * 100:.1f}% | "
                f"{overall['answer_token_recall'] * 100:.1f}% | {overall['ndcg']:.3f} | "
                f"{overall['mrr']:.3f} | {overall['avg_tokens']:.1f} | "
                f"{overall['avg_latency_ms']:.1f} |"
            )
        lines.append("")
        lines.append("### MemStack 分题型")
        lines.append("")
        lines.append("| 类别 | n | Unit R@k | Turn Token | Ans Token | NDCG@k |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for category, data in result["methods"]["MemStack"]["by_category"].items():
            lines.append(
                f"| {category} | {data['n']} | {data['unit_recall_any'] * 100:.1f}% | "
                f"{data['turn_token_recall'] * 100:.1f}% | {data['answer_token_recall'] * 100:.1f}% | "
                f"{data['ndcg']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MemStack public dataset evaluation")
    parser.add_argument("--dataset", choices=["locomo", "longmemeval", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="限制问题数，0=全部")
    parser.add_argument("--conversations", type=int, default=0, help="LoCoMo conversation 数，0=全部")
    parser.add_argument("--budget", type=int, default=800, help="上下文 token 预算")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--turns", type=int, default=6, help="每个 episode 最大轮数")
    parser.add_argument("--episode-chars", type=int, default=800, help="episode 进上下文的最大字符数")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR / "public_eval.json"))
    args = parser.parse_args()

    results = []
    if args.dataset in {"locomo", "both"}:
        results.append(run_locomo(args))
    if args.dataset in {"longmemeval", "both"}:
        results.append(run_lme(args))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.out).with_suffix(".md")
    markdown_path.write_text(to_markdown(results), encoding="utf-8")
    print(f"\n[saved] {args.out}\n[saved] {markdown_path}")


if __name__ == "__main__":
    main()
