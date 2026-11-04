"""MemStack 门面：写入、检索、遗忘、巩固、可观测性。

对外只暴露少量稳定 API：
    stack.remember(...)    写入一段交互
    stack.recall(...)      取回可直接注入 agent 的上下文包
    stack.search(...)      取回带分数的记忆列表
    stack.forget(...)      软/硬删除
    stack.consolidate(...) 离线整合（睡眠期）
    stack.set_block(...)   常驻上下文块
    stack.as_tools()       OpenAI function-calling 工具描述
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Sequence

from .config import MemoryConfig
from .embeddings import HashingEmbedder, cosine
from .evidence_chain import (
    decompose_subqueries,
    merge_chain_hits,
    needs_chain,
    neighbor_top_scores,
    should_neighbor_expand,
)
from .extraction import (
    RuleExtractor,
    RuleSummarizer,
    extract_entities,
    normalize_relative_time,
    predicate_synonyms,
)
from .graph import MemoryGraph
from .interfaces import Clock, Embedder, Extractor, MemoryStore, Reranker, Summarizer
from .retrieval import HybridRetriever, MULTIHOP_CUES
from .storage import SQLiteStore
from .types import (
    ContextBundle,
    EdgeType,
    ExtractedMemory,
    MemoryBlock,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    RecallHit,
    RecallQuery,
    RememberDecision,
    RememberResult,
    Scope,
    WriteAction,
    new_id,
)


class SystemClock:
    def now(self) -> float:
        return time.time()


class TopicSegmenter:
    """按最大轮数 + 相邻语义相似度切段；保证不会把 pair 切成半句。"""

    def __init__(self, embedder: Embedder, config: MemoryConfig) -> None:
        self.embedder = embedder
        self.config = config

    def segment(self, messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not messages:
            return []
        texts = [str(m.get("content", "")) for m in messages]
        vectors = self.embedder.embed(texts)
        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [messages[0]]
        for idx in range(1, len(messages)):
            similarity = cosine(vectors[idx - 1], vectors[idx])
            too_long = len(current) >= self.config.max_turns_per_episode
            topic_shift = (
                len(current) >= max(2, self.config.min_turns_per_episode * 2)
                and similarity < self.config.topic_split_threshold
                and len(current) % 2 == 0
            )
            if too_long or topic_shift:
                segments.append(current)
                current = []
            current.append(messages[idx])
        if current:
            # 过短的尾段并入上一段，避免产生大量碎片
            if segments and len(current) < self.config.min_turns_per_episode:
                segments[-1].extend(current)
            else:
                segments.append(current)
        return segments


class MemoryStack:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
        summarizer: Summarizer | None = None,
        clock: Clock | None = None,
        reranker: Reranker | None = None,
        query_decomposer: Any | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.config.validate()
        self.clock = clock or SystemClock()
        self.store = store or SQLiteStore(":memory:")
        self.embedder = embedder or HashingEmbedder(self.config.embedding_dim)
        self.extractor = extractor or RuleExtractor()
        self.summarizer = summarizer or RuleSummarizer()
        self.query_decomposer = query_decomposer  # 可选：LLM 化证据链分解（失败回退规则版）
        self.segmenter = TopicSegmenter(self.embedder, self.config)
        self.graph = MemoryGraph(self.store, self.embedder, self.config)
        self.retriever = HybridRetriever(
            self.store, self.embedder, self.graph, self.config, self.clock, reranker,
        )

    # ==================================================================================
    # 写入
    # ==================================================================================
    def remember(
        self,
        messages: str | dict[str, Any] | Sequence[dict[str, Any]],
        scope: Scope | None = None,
        event_time: float | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> RememberResult:
        scope = scope or Scope()
        now = self.clock.now()
        normalized = self._normalize_messages(messages)
        if not normalized:
            return RememberResult()
        event_time = self._coerce_time(event_time) or self._message_time(normalized) or now
        result = RememberResult()

        for segment in self.segmenter.segment(normalized):
            rendered = self._render(segment)
            # 相对时间归一化：写入时把 "yesterday / last week / 3 years ago" 换算成绝对日期，
            # 检索侧才能直接回答 "When ..."；原始措辞保留在括号内，不丢语义。
            if self.config.relative_time_normalization:
                display = self._render(
                    [
                        {
                            **message,
                            "content": normalize_relative_time(
                                str(message.get("content", "")),
                                self._coerce_time(message.get("event_time")) or event_time,
                            ),
                        }
                        for message in segment
                    ]
                )
            else:
                display = rendered
            content_hash = self._hash(
                f"{scope.tenant_id}|{scope.user_id}|{scope.agent_id}|{scope.session_id}|{rendered}"
            )
            existing_episode = self.store.find_by_hash(scope, content_hash, MemoryKind.EPISODE.value)
            if existing_episode and not force:
                result.decisions.append(
                    RememberDecision(WriteAction.NOOP, existing_episode.id, "episode", rendered[:80],
                                     reason="完全相同的 episode 已存在")
                )
                result.episode_ids.append(existing_episode.id)
                continue

            episode = MemoryItem(
                id=new_id("ep"),
                kind=MemoryKind.EPISODE,
                text=display,
                scope=scope,
                data={
                    "messages": segment,
                    "raw_text": rendered,
                    "turns": len(segment),
                    "topic": self._topic_hint(segment),
                    "content_hash": content_hash,
                    "entities": self._message_entities(segment),
                },
                created_at=now,
                updated_at=now,
                valid_from=event_time,
                importance=0.4,
                confidence=0.9,
                embedding=self.embedder.embed([display])[0],
                meta={"content_hash": content_hash, **dict(metadata or {})},
            )
            self.store.save(episode)
            self.graph.link(episode, now)
            self.store.audit("remember.episode", scope, {"id": episode.id, "turns": len(segment)})
            result.episode_ids.append(episode.id)
            result.decisions.append(
                RememberDecision(WriteAction.ADD, episode.id, "episode", rendered[:80])
            )

            extracted = self.extractor.extract(
                "\n".join(str(m.get("content", "")) for m in segment),
                event_time,
                {"scope": scope.to_dict()},
            )
            result.extracted += len(extracted)
            for item in extracted:
                decision = self._write_extracted(item, scope, episode.id, event_time, now)
                result.decisions.append(decision)
                if decision.action != WriteAction.NOOP:
                    result.memory_ids.append(decision.memory_id)

        return result

    def _write_extracted(
        self,
        extracted: ExtractedMemory,
        scope: Scope,
        episode_id: str,
        event_time: float,
        now: float,
    ) -> RememberDecision:
        self._ensure_slot_keys(extracted)
        valid_from = extracted.event_time or event_time
        slot_text = self._slot_text(extracted)
        embedding = self.embedder.embed([f"{extracted.text} {slot_text}"])[0]

        candidates = self.store.find(
            scope,
            kinds=[extracted.kind.value],
            statuses=[MemoryStatus.ACTIVE],
            limit=self.config.conflict_candidate_limit,
        )
        best_sim = 0.0
        best: MemoryItem | None = None
        slot_match: MemoryItem | None = None
        for candidate in candidates:
            sim = cosine(embedding, candidate.embedding) if candidate.embedding else 0.0
            if sim > best_sim:
                best_sim, best = sim, candidate
            if (
                candidate.data.get("subject_key") == extracted.data.get("subject_key")
                and candidate.data.get("predicate_key") == extracted.data.get("predicate_key")
                and extracted.data.get("predicate_key")
            ):
                slot_match = candidate
                break

        action = WriteAction.ADD
        target: MemoryItem | None = None
        reason = "新记忆"
        if slot_match is None and best is not None and best_sim >= self.config.duplicate_threshold:
            slot_match = best

        if slot_match is not None:
            same_object = (
                slot_match.data.get("object_key") == extracted.data.get("object_key")
                and slot_match.data.get("polarity") == extracted.data.get("polarity")
            )
            multi = bool(slot_match.data.get("multi") or extracted.data.get("multi"))
            if same_object:
                action, target, reason = WriteAction.MERGE, slot_match, "同一事实重复出现"
            elif not multi and self._is_conflicting(slot_match, extracted):
                action, target, reason = WriteAction.SUPERSEDE, slot_match, "同一槽位出现新值"
            elif best_sim >= self.config.duplicate_threshold:
                action, target, reason = WriteAction.MERGE, slot_match, "语义重复"

        if action == WriteAction.MERGE and target is not None:
            target.source_ids = sorted(set(target.source_ids + [episode_id]))
            target.confidence = max(target.confidence, extracted.confidence)
            target.importance = max(target.importance, extracted.importance)
            target.updated_at = now
            aliases = list(target.meta.get("aliases") or [])
            if extracted.text != target.text and extracted.text not in aliases:
                aliases.append(extracted.text)
            target.meta["aliases"] = aliases[-5:]
            self.store.save(target)
            self.store.put_edge(target.id, episode_id, EdgeType.PROVENANCE.value, 1.0, created_at=now)
            self.store.audit("remember.merge", scope, {"id": target.id, "episode": episode_id})
            return RememberDecision(action, target.id, extracted.kind.value, extracted.text,
                                    target_id=target.id, reason=reason)

        item = MemoryItem(
            id=new_id(self._prefix(extracted.kind)),
            kind=extracted.kind,
            text=extracted.text,
            scope=scope,
            data=dict(extracted.data),
            created_at=now,
            updated_at=now,
            valid_from=valid_from,
            importance=extracted.importance,
            confidence=extracted.confidence,
            embedding=embedding,
            source_ids=[episode_id],
            meta={
                "content_hash": self._hash(
                    f"{extracted.kind.value}|{extracted.data.get('subject_key')}|"
                    f"{extracted.data.get('predicate_key')}|{extracted.data.get('object_key')}|"
                    f"{extracted.data.get('polarity')}"
                )
            },
        )
        self.store.save(item)
        self.store.put_edge(item.id, episode_id, EdgeType.PROVENANCE.value, 1.0, created_at=now)

        if action == WriteAction.SUPERSEDE and target is not None:
            self._supersede(target, item, scope, now)
            self.store.audit("remember.supersede", scope, {
                "new": item.id, "old": target.id, "slot": item.data.get("predicate_key"),
            })
        else:
            self.store.audit("remember.add", scope, {
                "id": item.id, "kind": item.kind.value,
                "slot": item.data.get("predicate_key"),
            })

        self.graph.link(item, now)
        return RememberDecision(action, item.id, extracted.kind.value, extracted.text,
                                target_id=target.id if target else None, reason=reason)

    def _supersede(self, old: MemoryItem, new: MemoryItem, scope: Scope, now: float) -> None:
        if new.valid_from < old.valid_from and not new.data.get("is_change"):
            # 乱序写入且不是“变更”语义：新记忆其实是更早的历史，直接标记为已失效
            new.status = MemoryStatus.SUPERSEDED
            new.valid_to = old.valid_from
            self.store.save(new)
            self.store.put_edge(old.id, new.id, EdgeType.SUPERSEDES.value, 1.0, created_at=now)
        else:
            old.status = MemoryStatus.SUPERSEDED
            old.valid_to = new.valid_from if new.valid_from >= old.valid_from else now
            old.updated_at = now
            if self.config.history_policy == "overwrite":
                old.meta["overwritten_by"] = new.id
            self.store.save(old)
            self.store.put_edge(new.id, old.id, EdgeType.SUPERSEDES.value, 1.0, created_at=now)

    # ==================================================================================
    # 检索
    # ==================================================================================
    def search(
        self,
        query: str,
        scope: Scope | None = None,
        top_k: int | None = None,
        kinds: Sequence[MemoryKind] | None = None,
        include_history: bool | None = None,
    ) -> list[RecallHit]:
        recall_query = RecallQuery(
            text=query,
            scope=scope or Scope(),
            top_k=top_k or self.config.top_k,
            kinds=list(kinds) if kinds else None,
            include_history=include_history,
        )
        return self.retriever.search(recall_query, touch=True)

    def recall(
        self,
        query: str,
        scope: Scope | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
        include_history: bool | None = None,
        kinds: Sequence[MemoryKind] | None = None,
    ) -> ContextBundle:
        recall_query = RecallQuery(
            text=query,
            scope=scope or Scope(),
            top_k=top_k or self.config.top_k,
            token_budget=token_budget or self.config.token_budget,
            kinds=list(kinds) if kinds else None,
            include_history=include_history,
        )
        shared: dict[Any, Any] = {}
        hits = self.retriever.search(recall_query, touch=True, _shared=shared)
        if self.config.evidence_chain and hits:
            hits = self._evidence_chain_recall(recall_query, hits, shared)
        elif self.config.multi_round and self._should_second_round(query, hits):
            terms = self._expansion_terms(query, hits)
            if terms:
                extra_query = replace(
                    recall_query,
                    text=f"{recall_query.text} {' '.join(terms)}",
                )
                extra_hits = self.retriever.search(extra_query, touch=False, _shared=shared)
                hits = self._merge_hits(hits, extra_hits, recall_query.top_k)
                self.retriever._last_trace["multi_round"] = {
                    "terms": terms,
                    "extra_hits": len(extra_hits),
                }
        return self.retriever.build_context(recall_query, hits)

    # ----------------------------------------------------------------------------------
    # 证据链检索（v4）：分解 → 逐跳检索 → 分名额合并；辅以双路径近邻扩证
    # ----------------------------------------------------------------------------------
    def _evidence_chain_recall(
        self,
        recall_query: RecallQuery,
        hits: Sequence[RecallHit],
        shared: dict[Any, Any],
    ) -> list[RecallHit]:
        cfg = self.config
        trace: dict[str, Any] = {}
        if not needs_chain(recall_query.text):
            return list(hits)
        if self.query_decomposer is not None:
            # LLM 分解（可插拔）；内部失败/校验不过时回退规则版
            subs = self.query_decomposer.decompose(
                recall_query.text, list(hits), cfg.evidence_chain_max_subs,
            )
            if self.query_decomposer.last_fallback:
                trace["decomposer"] = "rule-fallback"
            else:
                trace["decomposer"] = "llm"
        else:
            subs = decompose_subqueries(recall_query.text, hits, cfg.evidence_chain_max_subs)
        sub_results: list[list[RecallHit]] = []
        for sub in subs:
            if sub.text.strip().lower() == recall_query.text.strip().lower():
                continue  # 分解失败退化成原查询，跳过避免无意义重查
            sub_query = replace(recall_query, text=sub.text)
            sub_hits = self.retriever.search(sub_query, touch=False, _shared=shared)
            sub_results.append(sub_hits)
            trace.setdefault("subqueries", []).append(
                {"text": sub.text, "reason": sub.reason, "hits": len(sub_hits)}
            )
        # 双路径：首跳最高分过低（熟悉度不足）→ 走图近邻扩证的"回忆"路径
        if cfg.evidence_chain_neighbor and sub_results and should_neighbor_expand(
            max((h.score for h in hits), default=0.0), cfg.evidence_chain_min_top_score,
        ):
            neighbor_hits = self._nearby_expand(recall_query, hits)
            if neighbor_hits:
                sub_results.append(neighbor_hits)
                trace["neighbor"] = {"hits": len(neighbor_hits)}
        if not sub_results:
            return list(hits)
        enum = any(sub.reason == "enumeration" for sub in subs)
        slots = cfg.evidence_chain_enum_slots if enum else cfg.evidence_chain_slots
        merged = merge_chain_hits(hits, sub_results, recall_query.top_k, slots)
        if hasattr(self.retriever, "_last_trace"):
            self.retriever._last_trace["evidence_chain"] = {
                "decomposer": trace.get("decomposer", "rule"),
                "subqueries": trace.get("subqueries", []),
                "neighbor_hits": trace.get("neighbor", {}).get("hits", 0),
                "slots_per_sub": slots,
                "merged": len(merged),
            }
        return merged

    def _nearby_expand(self, recall_query: RecallQuery, hits: Sequence[RecallHit]) -> list[RecallHit]:
        """回忆路径：PPR 从首跳种子扩散，取与 query 共享实体/词面的近邻记忆。"""
        seeds = neighbor_top_scores(hits, k=3)
        ppr = self.graph.ppr(seeds, recall_query.scope)
        if not ppr:
            return []
        query_terms = set(extract_entities(recall_query.text))
        query_terms |= {t for t in recall_query.text.lower().split() if len(t) > 3}
        scored: list[tuple[float, MemoryItem]] = []
        for mid, score in ppr.items():
            if mid in seeds or score <= 0:
                continue
            item = self.store.get(mid)
            if item is None or item.kind == MemoryKind.ENTITY:
                continue
            item_entities = {e.lower() for e in item.entities}
            text_lower = item.text.lower()
            overlap = query_terms & item_entities
            word_overlap = any(t in text_lower for t in query_terms if len(t) > 4)
            if not overlap and not word_overlap:
                continue
            bonus = 0.3 if overlap else 0.15
            scored.append((score * 0.7 + bonus, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RecallHit(memory=item, score=round(s, 4), reason="neighbor-expand")
            for s, item in scored[:4]
        ]

    @staticmethod
    def _should_second_round(query: str, hits: Sequence[RecallHit]) -> bool:
        if not hits:
            return False
        if MULTIHOP_CUES.search(query):
            return True
        # 聚合/顺序类问题也常需要跨会话补证
        return bool(re.search(
            r"\b(how many|how much|in total|both|across|after|before|first|last|all)\b"
            r"|总共|一共|多少|两个|所有|之前|之后",
            query, re.I,
        ))

    @staticmethod
    def _expansion_terms(query: str, hits: Sequence[RecallHit]) -> list[str]:
        query_lower = query.lower()
        terms: list[str] = []
        seen: set[str] = set()
        for hit in hits[:5]:
            item = hit.memory
            candidates = list(item.entities)
            for key in ("object", "subject"):
                value = item.data.get(key)
                if value:
                    candidates.append(str(value))
            for candidate in candidates:
                candidate = candidate.strip()
                if not (2 <= len(candidate) <= 40):
                    continue
                key = candidate.lower()
                if key in seen or key in query_lower:
                    continue
                if key in {"user", "i", "me", "my", "yes", "no"}:
                    continue
                seen.add(key)
                terms.append(candidate)
                if len(terms) >= 8:
                    return terms
        return terms

    @staticmethod
    def _merge_hits(
        first: Sequence[RecallHit],
        second: Sequence[RecallHit],
        top_k: int,
    ) -> list[RecallHit]:
        """保留第一轮高分证据，只用第二轮补足剩余名额，避免补证把原有证据挤掉。"""
        first_sorted = sorted(first, key=lambda hit: hit.score, reverse=True)
        keep_slots = max(1, top_k - 2)
        merged: dict[str, RecallHit] = {
            hit.memory.id: hit for hit in first_sorted[:keep_slots]
        }
        for hit in sorted(second, key=lambda h: h.score, reverse=True):
            if len(merged) >= top_k:
                break
            if hit.memory.id in merged:
                continue
            hit.score *= 0.9
            merged[hit.memory.id] = hit
        return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:top_k]

    # ==================================================================================
    # 遗忘与巩固
    # ==================================================================================
    def forget(
        self,
        memory_ids: Sequence[str] | None = None,
        query: str | None = None,
        scope: Scope | None = None,
        hard: bool = False,
        cascade: bool = True,
    ) -> list[str]:
        # 兼容 forget(ids, scope) 这种常见调用：第二个位置参数若是 Scope，则视为作用域
        if isinstance(query, Scope) and scope is None:
            scope = query
            query = None
        if scope is None and memory_ids:
            first = self.store.get(memory_ids[0])
            if first:
                scope = first.scope
        scope = scope or Scope()
        targets: list[str] = []
        if memory_ids:
            for mid in memory_ids:
                item = self.store.get(mid)
                if item and item.scope.tenant_id == scope.tenant_id \
                        and item.scope.user_id == scope.user_id \
                        and item.scope.agent_id == scope.agent_id:
                    targets.append(mid)
        elif query:
            targets = [hit.memory.id for hit in self.search(query, scope, top_k=self.config.top_k)]
        if cascade:
            self._cascade_forget(scope, targets, hard)
        for mid in targets:
            if hard:
                self.store.delete(mid)
            else:
                self.store.update_status(mid, MemoryStatus.DELETED)
            self.store.audit("forget.hard" if hard else "forget.soft", scope, {"id": mid})
        if targets:
            self._refresh_profile(scope)
        return targets

    def _cascade_forget(self, scope: Scope, target_ids: Sequence[str], hard: bool) -> None:
        """删除事实时，连带隐藏只服务于该事实的原始证据与摘要，避免“删了事实却还能从原文读到”。"""
        targets = set(target_ids)
        if not targets:
            return
        all_items = self.store.find(scope, limit=5000)
        active_like = {
            MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value, MemoryStatus.ARCHIVED.value,
        }
        episodes: set[str] = set()
        target_episodes: set[str] = set()
        for item in all_items:
            if item.id in targets:
                if item.kind == MemoryKind.EPISODE:
                    target_episodes.add(item.id)
                else:
                    episodes.update(item.source_ids)
        # 显式删除 episode 时，连带删除只依赖这些 episode 的派生事实
        derived_ids: set[str] = set()
        if target_episodes:
            derived = [
                item for item in all_items
                if item.kind != MemoryKind.EPISODE
                and item.status.value in active_like
                and item.source_ids
                and set(item.source_ids) <= target_episodes
            ]
            derived_ids = {item.id for item in derived}
            for item in derived:
                if hard:
                    self.store.delete(item.id)
                else:
                    self.store.update_status(item.id, MemoryStatus.DELETED)
                self.store.audit("forget.cascade.derived", scope, {"id": item.id})
        episodes.update(target_episodes)
        if not episodes:
            return
        still_used = {
            source
            for item in all_items
            if item.id not in targets
            and item.id not in derived_ids
            and item.status.value in active_like
            and item.kind != MemoryKind.EPISODE
            for source in item.source_ids
        }
        removable = episodes - still_used
        summaries = [
            item for item in all_items
            if item.kind == MemoryKind.SUMMARY and item.status == MemoryStatus.ACTIVE
            and set(item.source_ids) & removable
        ]
        for episode_id in removable:
            if hard:
                self.store.delete(episode_id)
            else:
                self.store.update_status(episode_id, MemoryStatus.DELETED)
            self.store.audit("forget.cascade.episode", scope, {"id": episode_id})
        for summary in summaries:
            if hard:
                self.store.delete(summary.id)
            else:
                self.store.update_status(summary.id, MemoryStatus.DELETED)
            self.store.audit("forget.cascade.summary", scope, {"id": summary.id})

    def _refresh_profile(self, scope: Scope) -> None:
        from .consolidation import Consolidator

        consolidator = Consolidator(
            self.store, self.embedder, self.summarizer, self.graph, self.config, self.clock,
        )
        consolidator.refresh_profile(scope)

    def consolidate(self, scope: Scope | None = None) -> dict[str, Any]:
        from .consolidation import Consolidator

        consolidator = Consolidator(
            self.store, self.embedder, self.summarizer, self.graph, self.config, self.clock,
        )
        return consolidator.run(scope or Scope())

    # ==================================================================================
    # 常驻上下文块
    # ==================================================================================
    def set_block(self, label: str, value: str, scope: Scope | None = None, limit: int = 2000) -> MemoryBlock:
        return self.store.set_block(scope or Scope(), label, value, limit)

    def get_block(self, label: str, scope: Scope | None = None) -> MemoryBlock | None:
        return self.store.get_block(scope or Scope(), label)

    # ==================================================================================
    # 观测 / 工具 / 导入导出
    # ==================================================================================
    def stats(self, scope: Scope | None = None) -> dict[str, Any]:
        scope = scope or Scope()
        data = self.store.stats(scope)
        data["config"] = {
            "embedding_dim": self.config.embedding_dim,
            "history_policy": self.config.history_policy,
            "graph": self.config.use_graph,
            "routing": self.config.use_routing,
            "temporal": self.config.use_temporal,
        }
        return data

    def as_tools(self) -> list[dict[str, Any]]:
        from .tools import openai_tool_specs

        return openai_tool_specs()

    def dispatch_tool(self, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
        from .tools import dispatch_tool

        return dispatch_tool(self, name, arguments)

    def export(self, scope: Scope | None = None) -> dict[str, Any]:
        return self.store.export_scope(scope or Scope())

    def import_data(self, payload: dict[str, Any]) -> int:
        return self.store.import_data(payload)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "MemoryStack":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ==================================================================================
    # 内部工具
    # ==================================================================================
    @staticmethod
    def _normalize_messages(messages: str | dict[str, Any] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, dict):
            return [messages]
        out: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, str):
                out.append({"role": "user", "content": message})
            elif isinstance(message, dict):
                # 保留 dia_id / session_id / has_answer 等外部评测元数据
                out.append({
                    **message,
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                })
        return [m for m in out if m.get("content")]

    @staticmethod
    def _render(messages: Sequence[dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            role = str(message.get("role", "user")).capitalize()
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _message_time(messages: Sequence[dict[str, Any]]) -> float | None:
        stamps = []
        for message in messages:
            stamp = message.get("timestamp")
            if isinstance(stamp, (int, float)):
                stamps.append(float(stamp))
            elif isinstance(stamp, str):
                try:
                    stamps.append(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    pass
        return max(stamps) if stamps else None

    @staticmethod
    def _coerce_time(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    @staticmethod
    def _message_entities(messages: Sequence[dict[str, Any]]) -> list[str]:
        from .extraction import extract_entities

        entities: list[str] = []
        for message in messages:
            entities.extend(extract_entities(str(message.get("content", ""))))
        seen: set[str] = set()
        out: list[str] = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                out.append(entity)
        return out

    @staticmethod
    def _topic_hint(messages: Sequence[dict[str, Any]]) -> str:
        text = " ".join(str(m.get("content", "")) for m in messages)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:80]

    @staticmethod
    def _slot_text(extracted: ExtractedMemory) -> str:
        return " ".join(str(x) for x in (
            extracted.data.get("subject_key", ""),
            extracted.data.get("predicate_key", ""),
            predicate_synonyms(str(extracted.data.get("predicate_key", ""))),
            extracted.data.get("object_key", ""),
            extracted.data.get("polarity", ""),
        ) if x)

    @staticmethod
    def _ensure_slot_keys(extracted: ExtractedMemory) -> None:
        if not extracted.data.get("subject_key") and extracted.subject:
            subject = extracted.subject.strip().lower()
            if subject in {"user", "i", "me"}:
                extracted.data["subject_key"] = "user"
            else:
                key = re.sub(r"[^\w\u3400-\u9fff]+", "-", subject).strip("-")
                extracted.data["subject_key"] = key
        if not extracted.data.get("predicate_key") and extracted.predicate:
            extracted.data["predicate_key"] = re.sub(
                r"[^\w\u3400-\u9fff:.\-]+", "_", extracted.predicate.strip().lower()
            ).strip("_")
        if extracted.object is not None and not extracted.data.get("object_key"):
            extracted.data["object_key"] = re.sub(
                r"^(?:a|an|the)\s+", "", str(extracted.object).strip().lower()
            )

    @staticmethod
    def _is_conflicting(old: MemoryItem, new: ExtractedMemory) -> bool:
        old_key = old.data.get("object_key")
        new_key = new.data.get("object_key")
        if old_key is None or new_key is None or old_key == new_key:
            return bool(old_key == new_key and old.data.get("polarity") != new.data.get("polarity"))
        return True

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()

    @staticmethod
    def _prefix(kind: MemoryKind) -> str:
        return {
            MemoryKind.FACT: "fact",
            MemoryKind.PREFERENCE: "pref",
            MemoryKind.PROCEDURE: "proc",
            MemoryKind.SUMMARY: "sum",
            MemoryKind.REFLECTION: "refl",
            MemoryKind.EPISODE: "ep",
            MemoryKind.ENTITY: "ent",
        }.get(kind, "mem")
