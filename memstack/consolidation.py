"""睡眠期巩固：去重、冲突消解、会话摘要、衰减归档、画像块刷新。

对应论文机制：
  - LightMem：在线只做 soft insert，离线做整合
  - Graphiti/MAGE：冲突不物理删除，保留历史与生命周期
  - MemoryOS：按访问热度与时间衰减
  - MemGAS/LightMem：生成会话级摘要作为导航层
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Sequence

from .config import MemoryConfig
from .graph import MemoryGraph
from .interfaces import Clock, Embedder, MemoryStore, Summarizer
from .types import (
    EdgeType,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    Scope,
    new_id,
)


class Consolidator:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        summarizer: Summarizer,
        graph: MemoryGraph,
        config: MemoryConfig,
        clock: Clock,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.summarizer = summarizer
        self.graph = graph
        self.config = config
        self.clock = clock

    def run(self, scope: Scope) -> dict[str, Any]:
        started = time.perf_counter()
        now = self.clock.now()
        items = self.store.find(scope, limit=5000)
        report = {
            "deduped": self._dedupe(items, scope, now),
            "conflicts_resolved": self._resolve_conflicts(items, scope, now),
            "summaries_created": self._summarize_sessions(items, scope, now),
            "archived": self._decay(items, scope, now),
            "profile_chars": 0,
            "elapsed_ms": 0.0,
        }
        report["profile_chars"] = self._refresh_profile(scope, now)
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        audit = getattr(self.store, "audit", None)
        if audit:
            audit("consolidate", scope, report)
        return report

    def refresh_profile(self, scope: Scope) -> int:
        return self._refresh_profile(scope, self.clock.now())

    # ----------------------------------------------------------------------------------
    def _dedupe(self, items: Sequence[MemoryItem], scope: Scope, now: float) -> int:
        buckets: dict[tuple[str, str, str, str, str], list[MemoryItem]] = defaultdict(list)
        for item in items:
            if item.status != MemoryStatus.ACTIVE or item.kind == MemoryKind.EPISODE:
                continue
            slot = item.slot
            if not slot:
                continue
            subject = item.data.get("subject_key") or slot[0]
            predicate = item.data.get("predicate_key") or slot[1]
            obj = item.data.get("object_key") or slot[2].lower()
            polarity = str(item.data.get("polarity", ""))
            buckets[(item.kind.value, subject, predicate, obj, polarity)].append(item)

        merged = 0
        for group in buckets.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda i: (i.valid_from, i.confidence, i.created_at), reverse=True)
            keeper = group[0]
            for other in group[1:]:
                keeper.source_ids = sorted(set(keeper.source_ids + other.source_ids))
                keeper.confidence = max(keeper.confidence, other.confidence)
                keeper.importance = max(keeper.importance, other.importance)
                aliases = list(keeper.meta.get("aliases") or [])
                if other.text not in aliases and other.text != keeper.text:
                    aliases.append(other.text)
                keeper.meta["aliases"] = aliases[-5:]
                other.status = MemoryStatus.SUPERSEDED
                other.valid_to = keeper.valid_from
                other.meta["deduped_into"] = keeper.id
                self.store.save(other)
                self.store.put_edge(keeper.id, other.id, EdgeType.SUPERSEDES.value, 0.8, created_at=now)
                merged += 1
            self.store.save(keeper)
        return merged

    def _resolve_conflicts(self, items: Sequence[MemoryItem], scope: Scope, now: float) -> int:
        groups: dict[tuple[str, str], list[MemoryItem]] = defaultdict(list)
        for item in items:
            if item.status != MemoryStatus.ACTIVE or item.kind not in {
                MemoryKind.FACT, MemoryKind.PREFERENCE,
            }:
                continue
            if item.data.get("multi"):
                continue
            slot = item.slot
            if not slot:
                continue
            subject = item.data.get("subject_key") or slot[0]
            predicate = item.data.get("predicate_key") or slot[1]
            groups[(subject, predicate)].append(item)

        resolved = 0
        for group in groups.values():
            if len(group) < 2:
                continue
            objects = {(g.data.get("object_key"), g.data.get("polarity")) for g in group}
            if len(objects) < 2:
                continue
            group.sort(key=lambda i: (i.valid_from, i.created_at), reverse=True)
            keeper = group[0]
            for other in group[1:]:
                other.status = MemoryStatus.SUPERSEDED
                other.valid_to = keeper.valid_from
                other.meta["conflict_resolved_into"] = keeper.id
                self.store.save(other)
                self.store.put_edge(keeper.id, other.id, EdgeType.SUPERSEDES.value, 1.0, created_at=now)
                resolved += 1
        return resolved

    def _summarize_sessions(self, items: Sequence[MemoryItem], scope: Scope, now: float) -> int:
        episodes: dict[str, list[MemoryItem]] = defaultdict(list)
        existing: set[str] = set()
        for item in items:
            if item.kind == MemoryKind.EPISODE and item.status == MemoryStatus.ACTIVE:
                episodes[item.scope.session_id or "default"].append(item)
            elif item.kind == MemoryKind.SUMMARY and item.status == MemoryStatus.ACTIVE:
                session = item.data.get("session_id")
                if session:
                    existing.add(str(session))

        created = 0
        for session, group in episodes.items():
            if session in existing or len(group) < self.config.summary_min_episodes:
                continue
            group.sort(key=lambda i: i.valid_from)
            text = self.summarizer.summarize([g.text for g in group], max_chars=600)
            if not text:
                continue
            summary = MemoryItem(
                id=new_id("sum"),
                kind=MemoryKind.SUMMARY,
                text=text,
                scope=group[0].scope,
                data={
                    "session_id": session,
                    "episodes": [g.id for g in group],
                    "entities": sorted({e for g in group for e in g.entities}),
                },
                created_at=now,
                updated_at=now,
                valid_from=group[0].valid_from,
                importance=0.6,
                confidence=0.8,
                embedding=self.embedder.embed([text])[0],
                source_ids=[g.id for g in group],
                meta={"content_hash": f"summary:{session}"},
            )
            self.store.save(summary)
            for episode in group:
                self.store.put_edge(summary.id, episode.id, EdgeType.DERIVED_FROM.value, 1.0, created_at=now)
            self.graph.link(summary, now)
            created += 1
        return created

    def _decay(self, items: Sequence[MemoryItem], scope: Scope, now: float) -> int:
        archived = 0
        half_life = self.config.decay_half_life_days * 86400
        for item in items:
            if item.status != MemoryStatus.ACTIVE:
                continue
            expires_at = item.meta.get("expires_at")
            if expires_at and float(expires_at) < now:
                self.store.update_status(item.id, MemoryStatus.ARCHIVED, valid_to=now)
                archived += 1
                continue
            age = max(0.0, now - max(item.valid_from, item.created_at))
            if age < self.config.archive_after_days * 86400:
                continue
            decayed = item.importance * (0.5 ** (age / max(half_life, 1.0)))
            if item.kind in {MemoryKind.EPISODE, MemoryKind.SUMMARY, MemoryKind.REFLECTION} \
                    and item.access_count == 0 and decayed < 0.15:
                self.store.update_status(item.id, MemoryStatus.ARCHIVED, valid_to=now)
                archived += 1
        return archived

    def _refresh_profile(self, scope: Scope, now: float) -> int:
        items = self.store.find(scope, limit=5000)
        core_predicates = {
            "name": 10, "location": 9, "occupation": 9, "employer": 8,
            "allergy": 10, "diet": 8, "diet_restriction": 7,
        }
        rows: list[tuple[float, str]] = []
        for item in items:
            if item.status != MemoryStatus.ACTIVE or item.kind not in {
                MemoryKind.FACT, MemoryKind.PREFERENCE,
            }:
                continue
            slot = item.slot
            if not slot:
                continue
            predicate = item.data.get("predicate_key") or slot[1]
            base_predicate = predicate.split(":")[0]
            priority = core_predicates.get(base_predicate, 3 if item.kind == MemoryKind.PREFERENCE else 1)
            if priority <= 1 and not predicate.startswith(("preference:", "favorite:", "project:")):
                continue
            value = str(item.data.get("object") or slot[2])
            polarity = item.data.get("polarity")
            if polarity == "negative":
                value = f"dislikes {value}"
            subject = "user" if (item.data.get("subject_key") or slot[0]) == "user" else (
                item.data.get("subject_key") or slot[0]
            )
            line = f"- {subject}.{predicate} = {value}"
            rows.append((priority + item.importance, line))
        rows.sort(key=lambda pair: pair[0], reverse=True)
        lines: list[str] = []
        total = 0
        for _, line in rows:
            if line in lines:
                continue
            if total + len(line) > self.config.profile_max_chars:
                break
            lines.append(line)
            total += len(line)
        value = "\n".join(lines)
        self.store.set_block(scope, "user_profile", value, self.config.profile_max_chars)
        return len(value)
