"""系统配置。所有可调参数集中在这里，避免散落在实现中。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    # ---------------- 写入 ----------------
    embedding_dim: int = 512
    duplicate_threshold: float = 0.90       # 高于此值视为同一条记忆
    similar_threshold: float = 0.70         # 建立 SIMILAR 边的阈值
    max_similar_edges: int = 3
    conflict_candidate_limit: int = 400
    enable_entity_linking: bool = True
    history_policy: str = "append_only"     # append_only | overwrite
    max_turns_per_episode: int = 6
    min_turns_per_episode: int = 1
    topic_split_threshold: float = 0.20     # 相邻轮次相似度低于此值则切段
    # 写入时把 "yesterday / last week / 3 years ago" 归一化为绝对日期，
    # 让时序问题的上下文里存在可直接比对的日期（原始措辞保留在括号内）。
    relative_time_normalization: bool = True

    # ---------------- 检索 ----------------
    top_k: int = 8
    token_budget: int = 800
    episode_char_limit: int = 700
    seed_k: int = 5
    min_score: float = 0.02
    use_graph: bool = True
    use_routing: bool = True
    use_temporal: bool = True
    use_rerank: bool = True
    multi_round: bool = True
    multi_round_max_entities: int = 8
    # ---------------- 证据链检索（v4，优先于 multi_round 词袋扩词） ----------------
    evidence_chain: bool = True            # 复杂查询走"分解 → 逐跳检索 → 分名额合并"
    evidence_chain_max_subs: int = 2       # 最多分解出的子查询数
    evidence_chain_slots: int = 2          # 一般子查询的补证名额（每子查询）
    evidence_chain_enum_slots: int = 4     # 枚举聚合类补证名额（答案分散度更高）
    evidence_chain_neighbor: bool = True   # RF-Mem 双路径：首跳低分时走图近邻扩证
    evidence_chain_min_top_score: float = 0.30  # 低于此值视为"熟悉度不足"
    status_aware_packing: bool = True  # TokenPilot 落点：预算不足时 active 优先，历史值放尾部
    # 证据保底：预算被高分的抽取事实吃掉时，仍为原始 episode 留出固定比例，
    # 否则原文证据（精确数值/日期/原话）会被挤出上下文。
    episode_reserve_ratio: float = 0.30
    episode_reserve_items: int = 3
    # 聚合类查询（how many / list all / across ...）自动放宽 top_k，
    # 因为答案分散在多个 episode，8 条候选通常不够覆盖。
    aggregation_top_k_bonus: int = 4
    evidence_expansion: bool = True
    evidence_top_hits: int = 3
    evidence_snippet_chars: int = 320
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "dense": 0.46,
            "lexical": 0.26,
            "entity": 0.08,
            "prior": 0.12,
            "graph": 0.08,
        }
    )
    kind_prior_strength: float = 0.25
    recency_half_life_days: float = 90.0
    continue_confidence: float = 0.42
    mmr_lambda: float = 0.75

    # ---------------- 离线巩固 ----------------
    dedupe_threshold: float = 0.93
    summary_min_episodes: int = 3
    archive_after_days: float = 730.0
    decay_half_life_days: float = 180.0
    profile_max_chars: int = 1200

    def validate(self) -> None:
        if self.history_policy not in {"append_only", "overwrite"}:
            raise ValueError("history_policy 必须是 append_only 或 overwrite")
        if not 0 < self.duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold 必须在 (0, 1]")
        if self.token_budget < 50:
            raise ValueError("token_budget 过小")
        if not 0 <= self.episode_reserve_ratio <= 1:
            raise ValueError("episode_reserve_ratio 必须在 [0, 1]")
        if self.episode_reserve_items < 0:
            raise ValueError("episode_reserve_items 不能为负")
        missing = {"dense", "lexical", "entity", "prior", "graph"} - set(self.weights)
        if missing:
            raise ValueError(f"weights 缺少通道: {missing}")
