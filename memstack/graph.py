"""记忆图：实体索引、溯源边、相似边与 Personalized PageRank 扩展。

图不是独立数据库，而是建立在 memories / edges 两张表上的逻辑视图：
  - 记忆 --mentions--> 实体
  - 派生记忆 --provenance--> 原始 episode
  - 摘要 --derived_from--> 来源
  - 新事实 --supersedes--> 旧事实
  - 近邻 --similar--> 近邻
这样存储层可以整体替换，而图算法不变。
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

from .config import MemoryConfig
from .embeddings import cosine
from .interfaces import Embedder, MemoryStore
from .types import EdgeType, MemoryItem, MemoryKind, MemoryStatus, Scope, new_id

EDGE_WEIGHTS = {
    EdgeType.PROVENANCE.value: 1.0,
    EdgeType.DERIVED_FROM.value: 0.9,
    EdgeType.MENTIONS.value: 0.55,
    EdgeType.SIMILAR.value: 0.5,
    EdgeType.SUPERSEDES.value: 0.3,
    EdgeType.NEXT.value: 0.4,
}


class MemoryGraph:
    def __init__(self, store: MemoryStore, embedder: Embedder, config: MemoryConfig) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self._adj_cache: dict[tuple[Any, ...], dict[str, dict[str, float]]] = {}

    # ----------------------------------------------------------------------------------
    def entity_id(self, scope: Scope, name: str) -> str:
        raw = f"{scope.tenant_id}|{scope.user_id}|{scope.agent_id}|{name.strip().lower()}"
        digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
        return f"ent-{digest[:12]}"

    def ensure_entity(self, scope: Scope, name: str, now: float) -> str | None:
        name = name.strip()
        if not name or len(name) < 2 or len(name) > 60:
            return None
        if name.lower() in {"user", "i", "me", "my", "it", "this", "that"}:
            return None
        eid = self.entity_id(scope, name)
        existing = self.store.get(eid)
        if existing:
            return eid
        item = MemoryItem(
            id=eid,
            kind=MemoryKind.ENTITY,
            text=name,
            scope=scope,
            data={"canonical": name.lower(), "entities": [name]},
            created_at=now,
            updated_at=now,
            valid_from=now,
            importance=0.3,
            confidence=0.9,
            embedding=self.embedder.embed([name])[0],
            meta={"content_hash": f"entity:{name.lower()}"},
        )
        self.store.save(item)
        return eid

    # ----------------------------------------------------------------------------------
    def link(self, item: MemoryItem, now: float) -> None:
        """写入后建立实体、溯源与近邻边。"""
        if self.config.enable_entity_linking:
            for entity in item.entities:
                eid = self.ensure_entity(item.scope, entity, now)
                if eid:
                    self.store.put_edge(
                        item.id, eid, EdgeType.MENTIONS.value, weight=1.0, created_at=now,
                    )

        for source_id in item.source_ids:
            if source_id and source_id != item.id:
                self.store.put_edge(
                    item.id, source_id, EdgeType.PROVENANCE.value, weight=1.0, created_at=now,
                )

        if item.kind == MemoryKind.ENTITY or item.embedding is None:
            return
        if self.config.max_similar_edges <= 0:
            return

        candidates = self.store.find(
            item.scope,
            kinds=[k.value for k in (
                MemoryKind.FACT, MemoryKind.PREFERENCE, MemoryKind.PROCEDURE,
                MemoryKind.SUMMARY, MemoryKind.EPISODE, MemoryKind.REFLECTION,
            )],
            statuses=[MemoryStatus.ACTIVE],
            limit=400,
        )
        scored: list[tuple[float, MemoryItem]] = []
        for other in candidates:
            if other.id == item.id or other.embedding is None:
                continue
            sim = cosine(item.embedding, other.embedding)
            if sim >= self.config.similar_threshold:
                scored.append((sim, other))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for sim, other in scored[: self.config.max_similar_edges]:
            self.store.put_edge(
                item.id, other.id, EdgeType.SIMILAR.value,
                weight=round(sim, 4), created_at=now,
            )

    # ----------------------------------------------------------------------------------
    def ppr(
        self,
        seed_scores: dict[str, float],
        scope: Scope,
        allowed_ids: Iterable[str] | None = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[str, float]:
        """带重启的随机游走。种子是检索高置信候选，边权重按类型缩放。"""
        if not seed_scores:
            return {}
        allowed = set(allowed_ids) if allowed_ids is not None else self.store.ids_in_scope(scope)
        adjacency: dict[str, dict[str, float]]
        version_fn = getattr(self.store, "revision", None)
        cache_key: tuple[Any, ...] | None = None
        if version_fn is not None:
            cache_key = (
                scope.tenant_id, scope.user_id, scope.agent_id,
                version_fn(), len(allowed), hash(frozenset(allowed)),
            )
        if cache_key is not None and cache_key in self._adj_cache:
            adjacency = self._adj_cache[cache_key]
        else:
            edges = self.store.edges()
            adjacency = {}
            for edge in edges:
                src, dst = edge["src"], edge["dst"]
                if src not in allowed or dst not in allowed:
                    continue
                weight = EDGE_WEIGHTS.get(edge["type"], 0.3) * float(edge.get("weight") or 1.0)
                if weight <= 0:
                    continue
                adjacency.setdefault(src, {})[dst] = adjacency.setdefault(src, {}).get(dst, 0.0) + weight
                adjacency.setdefault(dst, {})[src] = adjacency.setdefault(dst, {}).get(src, 0.0) + weight
            if cache_key is not None:
                if len(self._adj_cache) >= 4:  # 只保留最近几个数据版本，避免写密集时膨胀
                    oldest = next(iter(self._adj_cache))
                    self._adj_cache.pop(oldest)
                self._adj_cache[cache_key] = adjacency

        nodes = set(adjacency) | {n for n in seed_scores if n in allowed}
        if not nodes:
            return {}
        total_seed = sum(seed_scores.values()) or 1.0
        seed = {n: seed_scores.get(n, 0.0) / total_seed for n in nodes}
        # 数组化幂迭代：与字典版数学等价，少一半以上 dict 开销
        node_list = list(nodes)
        node_idx = {node: i for i, node in enumerate(node_list)}
        size = len(node_list)
        seed_vec = [seed.get(node, 0.0) for node in node_list]
        out_weights = [0.0] * size
        adj_idx: list[list[tuple[int, float]]] = [[] for _ in range(size)]
        for src, neighbors in adjacency.items():
            i = node_idx[src]
            weighted = [(node_idx[dst], w) for dst, w in neighbors.items() if dst in node_idx]
            adj_idx[i] = weighted
            out_weights[i] = sum(w for _, w in weighted)
        rank = list(seed_vec)
        for _ in range(iterations):
            nxt = [(1.0 - damping) * s for s in seed_vec]
            dangling = 0.0
            for i, neighbors in enumerate(adj_idx):
                out_weight = out_weights[i]
                if out_weight <= 0:
                    dangling += rank[i]
                    continue
                share = damping * rank[i] / out_weight
                for j, weight in neighbors:
                    nxt[j] += share * weight
            if dangling:
                for i in range(size):
                    nxt[i] += damping * dangling * seed_vec[i]
            rank = nxt
        max_score = max(rank) if rank else 0.0
        if max_score <= 0:
            return {node_list[i]: s for i, s in enumerate(rank)}
        return {node_list[i]: s / max_score for i, s in enumerate(rank)}
