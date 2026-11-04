"""检索与上下文组装：混合召回 → 多粒度路由 → 图扩展 → 时间策略 → 预算打包。

设计要点：
  - 多信号融合：dense + BM25 + entity + prior + graph，覆盖语义、词法、实体、热度与多跳；
  - 粒度路由：按 kind 的相似度分布熵计算软权重，避免硬路由误杀正确粒度；
  - 图扩展：沿 provenance / mentions 边做带重启随机游走，补充多跳证据；
  - 置信度不足时返回继续检索信号，而不是假装已经答完；
  - 按 token 预算注入，只放与查询相关的片段，不把整库塞进上下文。
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Any, Sequence

from .config import MemoryConfig
from .embeddings import (
    STOPWORDS,
    content_tokens,
    cosine,
    cosine_batch,
    entropy,
    estimate_tokens,
    normalize_scores,
    softmax,
    tokenize,
)
from .extraction import (
    STOP_ENTITIES,
    extract_entities,
    human_date,
    parse_date,
    predicate_synonyms,
    split_sentences,
)
from .graph import MemoryGraph
from .interfaces import Clock, Embedder, MemoryStore, Reranker
from .types import (
    ContextBundle,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    RecallHit,
    RecallQuery,
    iso,
)

HISTORY_CUES = re.compile(
    r"\b(used to|previously|before|earlier|formerly|last (?:year|month|week)|"
    r"in (?:19|20)\d{2}|history|historical)\b|过去|以前|之前|曾经|原来",
    re.I,
)
CURRENT_CUES = re.compile(
    r"\b(now|currently|current|latest|today|at present|these days)\b|现在|目前|当前|如今",
    re.I,
)
SLOT_CUES = re.compile(
    r"\b(where|what|who|whom|when|which|how many|how much|does|is|are|did)\b|哪里|哪儿|什么|谁|多少|几个",
    re.I,
)
MULTIHOP_CUES = re.compile(
    r"\b(headquarter|headquarters|company|whose|works for|employer of|the .* of the .* that)\b|总部|公司",
    re.I,
)
PROCEDURE_CUES = re.compile(r"\b(how (?:do|can|should) i|how to|steps?|procedure|process)\b|怎么|如何|步骤|流程", re.I)
PREFERENCE_CUES = re.compile(r"\b(prefer|preference|favorite|favourite|like|dislike)\b|偏好|喜欢|讨厌", re.I)
SUMMARY_CUES = re.compile(r"\b(summarize|summary|overview|across|overall|in total)\b|总结|总体|一共|总共", re.I)
# 时序问题（when / how long / how many days / before / after…）：答案往往只存在于
# 原始 episode 的日期里，抽取事实会把它们挤掉，因此这类查询不应降权 episode。
TEMPORAL_CUES = re.compile(
    r"\b(when|what date|what day|how long|how many (?:days|weeks|months|years)|"
    r"before|after|first|last time|between|earlier|later)\b|什么时候|何时|多久|几天|之前|之后",
    re.I,
)
# 聚合问题（how many / list all / across …）：答案分散在多个 episode，需要更宽的候选集。
AGGREGATION_CUES = re.compile(
    r"\b(how many|how much|list|all of|all the|each|every|across|in total|altogether|"
    r"what are the)\b|多少|几个|所有|全部|一共|总共",
    re.I,
)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    return normalize_scores(scores)


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        graph: MemoryGraph,
        config: MemoryConfig,
        clock: Clock,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.graph = graph
        self.config = config
        self.clock = clock
        self.reranker = reranker
        # BM25 候选集索引缓存：keyed by (scope, kinds, statuses, 时间窗, 数据版本号)。
        # 同 scope 连续多次查询只构建一次倒排；任何写操作 bump revision 自动失效。
        self._bm25_cache: dict[tuple[Any, ...], tuple[list[list[str]], dict[str, int], float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_version(self) -> int | None:
        version_fn = getattr(self.store, "revision", None)
        return version_fn() if version_fn is not None else None

    # ----------------------------------------------------------------------------------
    def search(
        self,
        query: RecallQuery,
        touch: bool = False,
        _shared: dict[Any, Any] | None = None,
    ) -> list[RecallHit]:
        """混合检索。

        `_shared`：证据链内部复用缓存（`core._evidence_chain_recall` 传入）。同一
        scope/状态/时间窗下，候选集、实体节点与 PPR 图分数在链内各跳之间保持不变，
        二次查询时直接复用，避免重复 SQLite 扫描与 O(E) 图游走。
        """
        started = time.perf_counter()
        now = self.clock.now()
        history_mode = self._history_mode(query)
        since, until = self._time_window(query, history_mode, now)

        statuses = [MemoryStatus.ACTIVE]
        if self.config.use_temporal and history_mode:
            statuses = [MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED, MemoryStatus.ARCHIVED]

        kinds = query.kinds or [
            MemoryKind.EPISODE, MemoryKind.FACT, MemoryKind.PREFERENCE,
            MemoryKind.PROCEDURE, MemoryKind.SUMMARY, MemoryKind.REFLECTION,
        ]
        share_key: Any = None
        if _shared is not None:
            version_fn = getattr(self.store, "revision", None)
            share_key = (
                query.scope.tenant_id, query.scope.user_id, query.scope.agent_id,
                tuple(sorted(k.value for k in kinds)),
                tuple(sorted(s.value for s in statuses)),
                since, until,
                version_fn() if version_fn is not None else 0,
            )
            cached = _shared.get(share_key)
        else:
            cached = None
        if cached is not None:
            candidates, entity_nodes, shared_graph = cached
        else:
            candidates = self.store.find(
                query.scope, kinds=kinds, statuses=statuses,
                since=since, until=until, limit=1200,
            )
            candidates = [c for c in candidates if c.kind != MemoryKind.ENTITY]
            entity_nodes = self.store.find(
                query.scope, kinds=[MemoryKind.ENTITY], statuses=[MemoryStatus.ACTIVE], limit=2000,
            )
            shared_graph: dict[str, float] | None = None
        if not candidates:
            return []
        q_entities = self._query_entities(query.text, candidates, entity_nodes)
        user_mentioned = any(
            str(c.data.get("object", "")).lower() in query.text.lower()
            for c in candidates
            if str(c.data.get("predicate_key", "")) == "name" and c.data.get("object")
        )

        expanded_query = self._expand_query(query.text)
        qvec = self.embedder.embed([expanded_query])[0]
        dense_raw: dict[str, float] = {}
        missing = [c for c in candidates if c.embedding is None]
        if missing:
            for c, vec in zip(missing, self.embedder.embed([c.text for c in missing])):
                c.embedding = vec
        for item, score in zip(candidates, cosine_batch(qvec, [c.embedding for c in candidates])):
            dense_raw[item.id] = score
        lexical_raw, rerank_sets = self._bm25_cached(
            expanded_query, candidates, query.scope, kinds, statuses, since, until,
        )
        entity_raw = self._entity_scores(q_entities, candidates, user_mentioned)
        prior_raw = self._priors(candidates, now, history_mode)

        dense = _minmax(dense_raw)
        lexical = _minmax(lexical_raw)
        entity = _minmax(entity_raw)
        prior = _minmax(prior_raw)

        weights = self.config.weights
        base: dict[str, float] = {}
        for item in candidates:
            base[item.id] = (
                weights["dense"] * dense.get(item.id, 0.0)
                + weights["lexical"] * lexical.get(item.id, 0.0)
                + weights["entity"] * entity.get(item.id, 0.0)
                + weights["prior"] * prior.get(item.id, 0.0)
            )

        intent = self._intent_weights(query.text)
        slot_query = (
            bool(SLOT_CUES.search(query.text))
            and not SUMMARY_CUES.search(query.text)
            and not TEMPORAL_CUES.search(query.text)
            and not AGGREGATION_CUES.search(query.text)
        )
        if slot_query:
            for item in candidates:
                if item.kind == MemoryKind.EPISODE:
                    base[item.id] *= 0.65
        kind_weights: dict[str, float] = {}
        if self.config.use_routing:
            kind_weights = self._route_by_entropy(candidates, base, intent)
            for item in candidates:
                weight = kind_weights.get(item.kind.value, 1.0)
                base[item.id] *= 0.85 + 0.15 * weight
        # 查询意图直接作用于槽位谓词，避免“人名事实”压过真正被问的槽位
        for item in candidates:
            if self._predicate_match(query.text, item):
                base[item.id] += 0.20
        if history_mode:
            for item in candidates:
                if item.status != MemoryStatus.ACTIVE:
                    base[item.id] += 0.18

        graph_scores: dict[str, float] = {}
        if self.config.use_graph and candidates:
            if shared_graph:
                # 链内复用：PPR 只依赖种子与图结构，种子在证据链各跳间高度重叠，
                # 复用首跳的图分数可省去 O(E) 游走（PPR 是子查询检索的最大单项开销）。
                graph_scores = shared_graph
            else:
                ranked = sorted(base.items(), key=lambda kv: kv[1], reverse=True)
                seeds = {mid: score for mid, score in ranked[: self.config.seed_k] if score > 0}
                if seeds:
                    graph_scores = self.graph.ppr(seeds, query.scope)
                    shared_graph = graph_scores
            if graph_scores:
                graph_weight = weights["graph"] * (2.5 if MULTIHOP_CUES.search(query.text) else 1.0)
                for item in candidates:
                    base[item.id] += graph_weight * graph_scores.get(item.id, 0.0)

        rerank = self._rerank(query.text, candidates, q_entities, history_mode, intent, rerank_sets)
        fused: dict[str, float] = {}
        for item in candidates:
            if self.config.use_rerank and self.reranker is None:
                fused[item.id] = 0.70 * base[item.id] + 0.30 * rerank.get(item.id, 0.0)
            else:
                fused[item.id] = base[item.id]
        if self.reranker is not None:
            ordered = sorted(candidates, key=lambda i: fused.get(i.id, 0.0), reverse=True)
            extra = self.reranker.rerank(query.text, ordered)
            for item, score in zip(ordered, extra):
                fused[item.id] = 0.6 * fused.get(item.id, 0.0) + 0.4 * float(score)

        fused = _minmax(fused)
        selected = self._select_mmr(candidates, fused, qvec, self._effective_top_k(query))
        hits: list[RecallHit] = []
        for rank, item in enumerate(selected):
            if fused[item.id] < query.min_score and rank > 0:
                continue
            hits.append(
                RecallHit(
                    memory=item,
                    score=fused[item.id],
                    channel_scores={
                        "dense": dense.get(item.id, 0.0),
                        "lexical": lexical.get(item.id, 0.0),
                        "entity": entity.get(item.id, 0.0),
                        "prior": prior.get(item.id, 0.0),
                        "graph": graph_scores.get(item.id, 0.0),
                    },
                    reason=self._reason(item, history_mode, q_entities),
                )
            )
        if touch and hits:
            self.store.bump_access([h.memory.id for h in hits], now)

        self._last_trace = {
            "candidates": len(candidates),
            "query_entities": q_entities,
            "history_mode": history_mode,
            "time_window": (since, until),
            "kind_weights": {k: round(v, 4) for k, v in kind_weights.items()},
            "channel_weights": weights,
            "graph_seeds": self.config.seed_k if graph_scores else 0,
            "cache": {"bm25_hits": self._cache_hits, "bm25_misses": self._cache_misses},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if _shared is not None and share_key is not None and share_key not in _shared:
            _shared[share_key] = (candidates, entity_nodes, shared_graph)
        return hits

    # ----------------------------------------------------------------------------------
    def build_context(self, query: RecallQuery, hits: Sequence[RecallHit]) -> ContextBundle:
        now = self.clock.now()
        blocks = self.store.list_blocks(query.scope)
        lines: list[str] = []
        if blocks:
            lines.append("## Core memory")
            for block in blocks:
                lines.append(f"### {block.label}")
                lines.append(block.value)
        if hits:
            lines.append("## Retrieved memory")
            for rank, hit in enumerate(hits):
                lines.append(
                    self._format_item(
                        hit.memory, query.text, hit.score,
                        include_evidence=(
                            self.config.evidence_expansion and rank < self.config.evidence_top_hits
                        ),
                    )
                )
        else:
            lines.append("## Retrieved memory")
            lines.append("(no relevant memory found)")

        text = "\n".join(lines)
        text, selected = self._fit_budget(text, hits, query, blocks)
        coverage = self._coverage(query.text, text)
        conflicts = self._conflicts(hits, self._history_mode(query))
        gaps = self._gaps(query.text, hits, coverage)
        confidence = self._confidence(hits, coverage, conflicts)
        should_continue = confidence < self.config.continue_confidence and bool(gaps or not hits)

        trace = dict(getattr(self, "_last_trace", {}))
        trace.update(
            {
                "blocks": [b.label for b in blocks],
                "selected_ids": [h.memory.id for h in selected],
                "token_budget": query.token_budget,
            }
        )
        return ContextBundle(
            text=text,
            items=list(selected),
            query=query.text,
            token_estimate=estimate_tokens(text),
            confidence=confidence,
            coverage=coverage,
            gaps=gaps,
            conflicts=conflicts,
            should_continue=should_continue,
            trace=trace,
        )

    # ----------------------------------------------------------------------------------
    # 各通道
    # ----------------------------------------------------------------------------------
    def _bm25_index(
        self,
        items: Sequence[MemoryItem],
    ) -> tuple[list[list[str]], dict[str, int], float, dict[str, set[str]]]:
        """对候选集合构建 BM25 统计（doc tokens / df / avgdl）与 rerank token 集。与 query 无关。"""
        docs: list[list[str]] = []
        rerank_sets: dict[str, set[str]] = {}
        for item in items:
            doc = tokenize(self._search_text(item))
            docs.append(doc)
            if item.id not in rerank_sets:
                # 与 _rerank 原先逐次调用 content_tokens 的语义完全一致，只是提前建好缓存。
                rerank_text = item.text + " " + str(item.data.get("object_key", ""))
                rerank_sets[item.id] = set(content_tokens(rerank_text))
        n = len(docs)
        avgdl = sum(len(doc) for doc in docs) / max(n, 1)
        df: dict[str, int] = defaultdict(int)
        for doc in docs:
            for token in set(doc):
                df[token] += 1
        return docs, df, avgdl, rerank_sets

    def _bm25_cached(
        self,
        query: str,
        items: Sequence[MemoryItem],
        scope: Any,
        kinds: Sequence[MemoryKind],
        statuses: Sequence[MemoryStatus],
        since: float | None,
        until: float | None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> tuple[dict[str, float], dict[str, set[str]]]:
        """带版本号缓存的 BM25：同 scope 同候选集连续查询只构建一次倒排。返回 (打分, rerank token 集)。"""
        q_tokens = [t for t in tokenize(query) if len(t) > 1]
        if not q_tokens or not items:
            return {item.id: 0.0 for item in items}, {}
        version = self._cache_version()
        cache_key: tuple[Any, ...] | None = None
        if version is not None:
            cache_key = (
                scope.tenant_id, scope.user_id, scope.agent_id,
                tuple(k.value for k in kinds), tuple(s.value for s in statuses),
                since, until, version,
            )
        if cache_key is not None and cache_key in self._bm25_cache:
            docs, df, avgdl, rerank_sets = self._bm25_cache[cache_key]
            self._cache_hits += 1
        else:
            docs, df, avgdl, rerank_sets = self._bm25_index(items)
            if cache_key is not None:
                if len(self._bm25_cache) >= 16:  # 写密集时只保留最近数据版本
                    oldest = next(iter(self._bm25_cache))
                    self._bm25_cache.pop(oldest)
                self._bm25_cache[cache_key] = (docs, df, avgdl, rerank_sets)
                self._cache_misses += 1
        n = len(docs)
        scores: dict[str, float] = {}
        for item, doc in zip(items, docs):
            tf: dict[str, int] = defaultdict(int)
            for token in doc:
                tf[token] += 1
            score = 0.0
            for token in set(q_tokens):
                if token not in tf:
                    continue
                idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
                denom = tf[token] + k1 * (1 - b + b * len(doc) / max(avgdl, 1e-6))
                score += idf * tf[token] * (k1 + 1) / max(denom, 1e-6)
            scores[item.id] = score
        return scores, rerank_sets

    def _query_entities(
        self,
        query: str,
        items: Sequence[MemoryItem],
        entity_nodes: Sequence[MemoryItem],
    ) -> list[str]:
        q_lower = query.lower()
        found: list[str] = []
        seen: set[str] = set()
        for item in list(items) + list(entity_nodes):
            names = item.entities if item.kind != MemoryKind.ENTITY else [item.text]
            for name in names:
                key = name.lower()
                min_len = 2 if re.search(r"[\u3400-\u9fff]", name) else 3
                if key in seen or len(name) < min_len:
                    continue
                if key in STOP_ENTITIES:
                    continue
                if key in q_lower:
                    seen.add(key)
                    found.append(name)
        for name in extract_entities(query):
            key = name.lower()
            if key not in seen and key not in STOP_ENTITIES and len(name) >= 2:
                seen.add(key)
                found.append(name)
        return found[:12]

    def _entity_scores(
        self,
        q_entities: Sequence[str],
        items: Sequence[MemoryItem],
        user_mentioned: bool = False,
    ) -> dict[str, float]:
        if not q_entities:
            return {item.id: 0.0 for item in items}
        q_set = {e.lower() for e in q_entities}
        scores: dict[str, float] = {}
        for item in items:
            item_set = {e.lower() for e in item.entities}
            overlap = len(q_set & item_set)
            score = overlap / max(len(q_set), 1)
            if user_mentioned and item.data.get("subject_key") == "user":
                score = max(score, 0.7)
            scores[item.id] = score
        return scores

    def _priors(self, items: Sequence[MemoryItem], now: float, history_mode: bool) -> dict[str, float]:
        half_life = self.config.recency_half_life_days * 86400
        scores: dict[str, float] = {}
        for item in items:
            anchor = item.valid_to if (history_mode and item.valid_to) else max(item.valid_from, item.created_at)
            age = max(0.0, now - anchor)
            recency = 0.5 ** (age / max(half_life, 1.0))
            recency = max(recency, 0.05)
            frequency = 1.0 + 0.12 * math.log1p(item.access_count)
            scores[item.id] = item.importance * item.confidence * recency * frequency
        return scores

    def _route_by_entropy(
        self,
        items: Sequence[MemoryItem],
        base: dict[str, float],
        intent: dict[str, float],
    ) -> dict[str, float]:
        groups: dict[str, list[MemoryItem]] = defaultdict(list)
        for item in items:
            groups[item.kind.value].append(item)
        raw_weights: dict[str, float] = {}
        for kind, group in groups.items():
            scores = [max(0.0, base.get(item.id, 0.0)) for item in group]
            if len(scores) == 1:
                confidence = 1.0
            else:
                probs = softmax(scores, temperature=0.15)
                h = entropy(probs) / math.log(len(scores) + 1e-9)
                confidence = max(0.0, 1.0 - h)
            raw_weights[kind] = math.exp(1.0 * confidence) * intent.get(kind, 1.0)
        total = sum(raw_weights.values()) or 1.0
        normalized = {kind: value / total for kind, value in raw_weights.items()}
        max_weight = max(normalized.values()) if normalized else 1.0
        return {kind: value / max_weight for kind, value in normalized.items()}

    def _rerank(
        self,
        query: str,
        items: Sequence[MemoryItem],
        q_entities: Sequence[str],
        history_mode: bool,
        intent: dict[str, float],
        cached_tokens: dict[str, set[str]] | None = None,
    ) -> dict[str, float]:
        q_tokens = set(content_tokens(query))
        q_set = {e.lower() for e in q_entities}
        scores: dict[str, float] = {}
        for item in items:
            if cached_tokens is not None and item.id in cached_tokens:
                item_tokens = cached_tokens[item.id]
            else:
                item_tokens = set(content_tokens(item.text + " " + str(item.data.get("object_key", ""))))
            coverage = len(q_tokens & item_tokens) / max(len(q_tokens), 1)
            entity_bonus = 1.0 if q_set & {e.lower() for e in item.entities} else 0.0
            if history_mode:
                status_bonus = 1.0 if item.status != MemoryStatus.ACTIVE else 0.5
            else:
                status_bonus = 1.0 if item.status == MemoryStatus.ACTIVE else 0.2
            kind_bonus = min(1.0, intent.get(item.kind.value, 1.0) - 0.6)
            slot_bonus = 1.0 if item.slot else 0.0
            predicate_bonus = 1.0 if self._predicate_match(query, item) else 0.0
            scores[item.id] = (
                0.30 * coverage + 0.10 * entity_bonus + 0.05 * status_bonus
                + 0.05 * kind_bonus + 0.15 * slot_bonus + 0.35 * predicate_bonus
            )
        return _minmax(scores)

    # ----------------------------------------------------------------------------------
    # 选择与打包
    # ----------------------------------------------------------------------------------
    def _select_mmr(
        self,
        items: Sequence[MemoryItem],
        scores: dict[str, float],
        qvec: Sequence[float],
        top_k: int,
    ) -> list[MemoryItem]:
        """MMR 贪心选择。numpy 可用时把 pairwise 余弦整体向量化（数学等价，结果不变）。"""
        remaining = sorted(items, key=lambda i: scores.get(i.id, 0.0), reverse=True)
        if len(remaining) <= top_k or not remaining:
            return list(remaining)
        try:
            import numpy as np
        except ImportError:
            return self._select_mmr_loop(remaining, scores, qvec, top_k)

        matrix = np.asarray([list(item.embedding) for item in remaining], dtype=np.float64)
        relevance = np.asarray([scores.get(i.id, 0.0) for i in remaining])
        # 与纯 Python 版保持一致：cosine() 语义就是裸内积（默认 embedder 输出已归一化），
        # 这里不做额外归一化，保证两种路径数值完全一致。
        q = np.asarray(list(qvec), dtype=np.float64)
        sim_to_q = matrix @ q

        selected: list[MemoryItem] = []
        selected_vecs: np.ndarray | None = None
        while len(selected) < top_k and len(matrix) > 0:
            if selected_vecs is None:
                sim = sim_to_q
            else:
                sim = np.max(matrix @ selected_vecs.T, axis=1)
            value = self.config.mmr_lambda * relevance - (1 - self.config.mmr_lambda) * sim
            idx = int(np.argmax(value))
            chosen = remaining[idx]
            selected.append(chosen)
            chosen_vec = matrix[idx : idx + 1]
            selected_vecs = chosen_vec if selected_vecs is None else np.vstack([selected_vecs, chosen_vec])
            mask = np.ones(len(matrix), dtype=bool)
            mask[idx] = False
            matrix = matrix[mask]
            relevance = relevance[mask]
            sim_to_q = sim_to_q[mask]
            remaining = [item for j, item in enumerate(remaining) if j != idx]
        return selected

    def _select_mmr_loop(
        self,
        remaining: list[MemoryItem],
        scores: dict[str, float],
        qvec: Sequence[float],
        top_k: int,
    ) -> list[MemoryItem]:
        """numpy 不可用时的纯 Python MMR（原实现）。"""
        selected: list[MemoryItem] = []
        while remaining and len(selected) < top_k:
            best = None
            best_value = -1e9
            for item in remaining:
                relevance = scores.get(item.id, 0.0)
                if selected:
                    sim = max(cosine(item.embedding, s.embedding) for s in selected)
                else:
                    sim = cosine(item.embedding, qvec)
                value = self.config.mmr_lambda * relevance - (1 - self.config.mmr_lambda) * sim
                if value > best_value:
                    best_value = value
                    best = item
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
        return selected

    def _format_item(
        self,
        item: MemoryItem,
        query: str,
        score: float,
        include_evidence: bool = False,
    ) -> str:
        src = ",".join(item.source_ids[:2]) or "-"
        stamp = iso(item.valid_from)
        head = f"- [{item.kind.value.upper()}|score={score:.2f}|{item.status.value}|from={stamp}"
        if item.valid_to:
            head += f"|to={iso(item.valid_to)}"
        if item.kind == MemoryKind.EPISODE:
            # 同时给出 ISO 与人类可读日期：读取侧不必再自己把 2023-05-08 换算成 "8 May 2023"
            head += f"|on={human_date(item.valid_from)}"
        head += f"|src={src}]"
        if item.slot:
            subject, predicate, obj = item.slot
            subject_key = item.data.get("subject_key") or subject.lower()
            predicate_key = item.data.get("predicate_key") or predicate.lower().replace(" ", "_")
            polarity = item.data.get("polarity")
            value = f"{polarity + ':' if polarity == 'negative' else ''}{obj}"
            line = f"{head} slot={subject_key}.{predicate_key} | value={value}"
            if item.kind == MemoryKind.PROCEDURE and item.data.get("steps"):
                steps = " -> ".join(str(s) for s in item.data["steps"])
                line += f" | steps={steps}"
            line += f' | evidence="{self._clip(item.text, 220)}"'
            if include_evidence:
                snippet = self._source_snippet(item, query)
                if snippet:
                    line += f' | source="{snippet}"'
            return line
        if item.kind == MemoryKind.EPISODE:
            if len(item.text) <= self.config.episode_char_limit:
                return f"{head} {item.text}"
            return f"{head} {self._compress_episode(item.text, query, self.config.episode_char_limit)}"
        if item.kind == MemoryKind.SUMMARY:
            return f"{head} {self._clip(item.text, 400)}"
        return f"{head} {self._clip(item.text, 320)}"

    def _source_snippet(self, item: MemoryItem, query: str) -> str:
        """从 provenance episode 中回填与 query 最相关的一条原始轮次。"""
        if not item.source_ids:
            return ""
        query_tokens = set(content_tokens(query))
        item_tokens = set(content_tokens(item.text))
        best = ""
        best_score = 0.0
        for source_id in item.source_ids[:2]:
            episode = self.store.get(source_id)
            if episode is None:
                continue
            for message in episode.data.get("messages", []):
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                content_tokens_set = set(content_tokens(content))
                score = len(query_tokens & content_tokens_set) + 0.5 * len(item_tokens & content_tokens_set)
                if score > best_score:
                    best_score = score
                    best = content
        if not best:
            return ""
        # 如果原始轮次与已展示的 evidence 基本相同，就不重复占预算
        if content_tokens(best)[:12] == content_tokens(item.text)[:12]:
            return ""
        return self._clip(best, self.config.evidence_snippet_chars)

    def _fit_budget(
        self,
        text: str,
        hits: Sequence[RecallHit],
        query: RecallQuery,
        blocks: Sequence[Any],
    ) -> tuple[str, list[RecallHit]]:
        if estimate_tokens(text) <= query.token_budget:
            return text, list(hits)
        # 先压缩 block，再逐条裁剪；至少保留第一条记忆
        header_parts = []
        if blocks:
            header_parts.append("## Core memory")
            header_parts.extend(f"### {b.label}\n{b.value}" for b in blocks)
        header_parts.append("## Retrieved memory")
        header = "\n".join(header_parts)
        remaining = query.token_budget - estimate_tokens(header) - 8
        if remaining < 40:
            header = (
                "## Core memory\n" + blocks[0].value[:200]
                if blocks else "## Retrieved memory"
            )
            remaining = max(40, query.token_budget - estimate_tokens(header))
        lines = [header] if header else []
        # 生命周期感知打包：预算不足时"当前值（active）优先进入上下文，
        # 历史值（superseded/archived）放尾部"，避免旧事实挤掉当前事实；前缀仍保持
        # blocks → 记忆条目的稳定结构。预算充足时顺序与 hits 完全一致，不影响现有指标。
        if (
            self.config.status_aware_packing
            and any(hit.memory.status != MemoryStatus.ACTIVE for hit in hits)
        ):
            ordered = [h for h in hits if h.memory.status == MemoryStatus.ACTIVE] + [
                h for h in hits if h.memory.status != MemoryStatus.ACTIVE
            ]
        else:
            ordered = list(hits)

        selected: list[RecallHit] = []
        selected_ids: set[str] = set()

        # 证据保底：抽取事实通常分更高，会把原文 episode 整段挤出上下文，
        # 而日期、原话、精确数值只在原文里。先按比例给 episode 留额度，再用剩余预算填事实。
        if self.config.episode_reserve_ratio > 0 and self.config.episode_reserve_items > 0:
            reserve = int(remaining * self.config.episode_reserve_ratio)
            episodes = [h for h in ordered if h.memory.kind == MemoryKind.EPISODE]
            for hit in episodes[: self.config.episode_reserve_items]:
                if reserve < 40:
                    break
                line = self._format_item(hit.memory, query.text, hit.score)
                if estimate_tokens(line) > reserve:
                    line = line[: max(60, len(line) * reserve // max(estimate_tokens(line), 1))] + " ..."
                lines.append(line)
                spent = estimate_tokens(line)
                remaining -= spent
                reserve -= spent
                selected.append(hit)
                selected_ids.add(hit.memory.id)

        for idx, hit in enumerate(ordered):
            if remaining <= 20:
                break
            if hit.memory.id in selected_ids:
                continue
            line = self._format_item(
                hit.memory, query.text, hit.score,
                include_evidence=(
                    self.config.evidence_expansion and idx < self.config.evidence_top_hits
                ),
            )
            if estimate_tokens(line) > remaining:
                line = line[: max(80, len(line) * remaining // max(estimate_tokens(line), 1))]
                line += " ..."
            lines.append(line)
            remaining -= estimate_tokens(line)
            selected.append(hit)
            selected_ids.add(hit.memory.id)
        return "\n".join(lines), selected

    def _compress_episode(self, text: str, query: str, max_chars: int = 320) -> str:
        if len(text) <= max_chars:
            return text
        q_tokens = set(content_tokens(query))
        sentences = split_sentences(text)
        scored = []
        for idx, sent in enumerate(sentences):
            overlap = len(q_tokens & set(content_tokens(sent)))
            bonus = 0.3 if re.search(r"\d", sent) else 0.0
            scored.append((overlap + bonus, idx, sent))
        scored.sort(reverse=True)
        chosen = sorted(scored[:2], key=lambda row: row[1])
        result = " ".join(row[2] for row in chosen)
        return self._clip(result, max_chars) + (" ..." if len(result) > max_chars else "")

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."

    # ----------------------------------------------------------------------------------
    # 策略与诊断
    # ----------------------------------------------------------------------------------
    def _history_mode(self, query: RecallQuery) -> bool:
        if query.include_history is not None:
            return query.include_history
        if not self.config.use_temporal:
            return False
        text = query.text
        if CURRENT_CUES.search(text):
            return False
        return bool(HISTORY_CUES.search(text))

    def _time_window(self, query: RecallQuery, history_mode: bool, now: float) -> tuple[float | None, float | None]:
        if query.time_range:
            return query.time_range
        if not (self.config.use_temporal and history_mode):
            return None, None
        stamp = parse_date(query.text, now)
        if stamp is None:
            return None, None
        # 查询里出现明确日期时，取该日期所在的一年窗口
        return stamp, stamp + 366 * 86400

    def _intent_weights(self, query: str) -> dict[str, float]:
        weights: dict[str, float] = defaultdict(lambda: 1.0)
        if PROCEDURE_CUES.search(query):
            weights[MemoryKind.PROCEDURE.value] = 1.6
        if PREFERENCE_CUES.search(query):
            weights[MemoryKind.PREFERENCE.value] = 1.5
        if SUMMARY_CUES.search(query):
            weights[MemoryKind.SUMMARY.value] = 1.5
        if not (PROCEDURE_CUES.search(query) or PREFERENCE_CUES.search(query)):
            weights[MemoryKind.FACT.value] = 1.6 if SLOT_CUES.search(query) else 1.3
        if TEMPORAL_CUES.search(query):
            # 时序答案（"7 May 2023"、"the week before ..."）几乎只出现在原文里
            weights[MemoryKind.EPISODE.value] = 1.45
        elif SLOT_CUES.search(query) and not SUMMARY_CUES.search(query) and not PROCEDURE_CUES.search(query):
            weights[MemoryKind.EPISODE.value] = 0.75
        return dict(weights)

    def _effective_top_k(self, query: RecallQuery) -> int:
        """聚合类查询自动放宽候选集：答案分散在多个 episode，固定 top_k 覆盖不全。"""
        if AGGREGATION_CUES.search(query.text):
            return query.top_k + max(0, self.config.aggregation_top_k_bonus)
        return query.top_k

    @staticmethod
    def _expand_query(query: str) -> str:
        checks = [
            (r"\b(where|live|lives|living|based|city|move|moved)\b|住|城市",
             "location live lives based city moved move address"),
            (r"\b(work|works|job|role|title|employer|company)\b|工作|职业|公司",
             "work job occupation employer role company"),
            (r"\ballerg", "allergy allergic"),
            (r"\b(prefer|preference|favorite|favourite|like|dislike|cuisine|food)\b|喜欢|偏好|口味",
             "preference favorite like love food cuisine"),
            (r"\b(how many|how much|total|completed|count)\b|多少|几个|总共|一共",
             "completed count total number"),
            (r"\b(stack|database|built with|technology|uses?)\b|数据库|技术栈",
             "stack database uses technology"),
            (r"\b(how do|how to|steps|process|procedure)\b|怎么|如何|步骤|流程",
             "procedure steps process how to"),
            (r"\b(headquarter|headquarters|company)\b|总部|公司",
             "headquarters headquartered company city"),
        ]
        extras = [expansion for pattern, expansion in checks if re.search(pattern, query, re.I)]
        return query + (" " + " ".join(extras) if extras else "")

    @staticmethod
    def _search_text(item: MemoryItem) -> str:
        predicate = str(item.data.get("predicate_key", ""))
        parts = [
            item.text,
            str(item.data.get("subject_key", "")),
            predicate,
            predicate_synonyms(predicate),
            str(item.data.get("object_key", "")),
            str(item.data.get("polarity", "")),
        ]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _predicate_match(query: str, item: MemoryItem) -> bool:
        predicate = str(item.data.get("predicate_key") or "")
        if not predicate:
            return False
        checks = [
            (r"\b(where|live|lives|living|based|city|move|moved)\b|住|城市",
             ("location", "previous_location")),
            (r"\b(work|works|job|role|title|employer|company)\b|工作|职业|公司",
             ("occupation", "employer")),
            (r"\ballerg", ("allergy",)),
            (r"\b(prefer|preference|favorite|favourite|like|dislike|cuisine|food)\b|喜欢|偏好|口味",
             ("preference", "favorite")),
            (r"\b(how many|how much|total|completed|count)\b|多少|几个|总共|一共",
             ("completed_count", "member_count")),
            (r"\b(stack|database|built with|technology|uses?)\b|数据库|技术栈",
             ("stack", "uses", "supports")),
            (r"\b(name|who is)\b|名字|是谁", ("name",)),
            (r"\b(how do|how to|steps|process|procedure)\b|怎么|如何|步骤|流程", ("procedure",)),
            (r"\b(headquarter|headquarters|company)\b|总部|公司", ("headquarters",)),
        ]
        for pattern, prefixes in checks:
            if re.search(pattern, query, re.I) and any(predicate.startswith(p) for p in prefixes):
                return True
        return False

    @staticmethod
    def _reason(item: MemoryItem, history_mode: bool, q_entities: Sequence[str]) -> str:
        bits = [item.kind.value]
        if history_mode and item.status != MemoryStatus.ACTIVE:
            bits.append("history")
        if q_entities and {e.lower() for e in q_entities} & {e.lower() for e in item.entities}:
            bits.append("entity-match")
        if item.source_ids:
            bits.append("provenance")
        return "+".join(bits)

    @staticmethod
    def _coverage(query: str, text: str) -> float:
        q_tokens = set(content_tokens(query))
        if not q_tokens:
            return 0.0
        return len(q_tokens & set(content_tokens(text))) / len(q_tokens)

    def _conflicts(self, hits: Sequence[RecallHit], history_mode: bool) -> list[str]:
        if history_mode:
            return []
        slots: dict[tuple[str, str], set[str]] = defaultdict(set)
        for hit in hits:
            item = hit.memory
            if item.kind not in {MemoryKind.FACT, MemoryKind.PREFERENCE}:
                continue
            if item.status != MemoryStatus.ACTIVE or item.data.get("multi"):
                continue
            if item.slot:
                subject, predicate, obj = item.slot
                slots[(item.data.get("subject_key") or subject, item.data.get("predicate_key") or predicate)].add(
                    item.data.get("object_key") or obj.lower()
                )
        return [f"{subject}.{predicate} 存在多个当前值" for (subject, predicate), values in slots.items() if len(values) > 1]

    @staticmethod
    def _gaps(query: str, hits: Sequence[RecallHit], coverage: float) -> list[str]:
        gaps: list[str] = []
        if not hits:
            gaps.append("没有命中任何记忆")
        if coverage < 0.35 and hits:
            gaps.append("上下文对查询词覆盖不足")
        q_entities = [
            entity for entity in extract_entities(query)
            if entity.lower() not in STOP_ENTITIES
        ]
        hit_entities = {e.lower() for hit in hits for e in hit.memory.entities}
        for entity in q_entities:
            if entity.lower() not in hit_entities:
                gaps.append(f"缺少实体相关证据: {entity}")
        return gaps[:5]

    @staticmethod
    def _confidence(hits: Sequence[RecallHit], coverage: float, conflicts: Sequence[str]) -> float:
        if not hits:
            return 0.0
        top = hits[0].score
        second = hits[1].score if len(hits) > 1 else 0.0
        margin = (top - second) / max(top, 1e-6)
        value = 0.40 * top + 0.20 * margin + 0.40 * coverage
        if conflicts:
            value -= 0.15
        return max(0.0, min(1.0, value))
