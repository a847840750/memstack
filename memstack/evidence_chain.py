"""证据链检索：query 结构感知分解 → 逐跳检索 → 子问题分名额合并（v4）。

设计要点：
  - 多跳问题沿溯源图逐跳补证，而不是一次性 top-K；
  - 证据不足/冲突时继续多轮检索，但每轮只补缺口名额；
  - 首跳得分过低（熟悉度不足）时走"回忆"路径（近邻扩证），而非盲目重查整库；
  - 枚举聚合类查询的答案分散在多个会话，需要"主体 + 槽位"的枚举检索，
    而不是把命中词袋拼回 query（v2 多轮补证的瓶颈）

与 v2 多轮补证（multi_round 词袋扩词）的区别：
  v2 把 top-5 命中里的实体词全部拼回 query → 漂移到已命中会话的同款词汇，
  长尾会话（其他取值）依然拿不到；且扩词没有"缺口"概念。
  v4 把查询分解为结构化的子查询（枚举槽位 / 图桥接 / 时间锚点 / 近邻扩证），
  每个子查询独立检索、独立名额，合并时保留首轮高分证据，只补缺口。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .extraction import extract_entities

if TYPE_CHECKING:  # 仅类型检查，避免运行时循环依赖
    from .config import MemoryConfig
    from .types import RecallHit

# --------------------------------------------------------------------------------------
# 触发模式
# --------------------------------------------------------------------------------------
_QUANT_CUES = re.compile(
    r"\b(how many|how much|how often)\b|多少|几个|几次|一共",
    re.I,
)
# 枚举型：what/which + 复数槽位名词 + 行为动词（LoCoMo multi-hop 的主力题型）
_ENUM_CUES = re.compile(
    r"\b(what|which|list|name|enumerate)\b[^.?]*(?:\s+(?:kind|kinds|types|type)s?\s+(?:of\s+)?)?"
    r"(books?|hobbies?|activit\w*|recipes?|recommend\w*|things?|songs?|movies?|films?|games?|"
    r"foods?|meals?|places?|events?|authors?|ways?|ideas?|interests?|subjects?|articles?|"
    r"shows?|series|tournaments?|reasons?|qualities?|goals?|plans?|restaurants?|cities?|"
    r"countries?|teams?|lessons?|classes?|courses?|languages?|sports?|reads?|watched|"
    r"read|designed|considered|traveled|visited|participated|researched|learned|changed|"
    r"started|tried|shared|received|recommended|made|cooked|baked|joined|attended)"
    r"\b|什么|哪些|什么(?:样的)?",
    re.I,
)
# 图谱桥接型：实体 A → 关系 → 实体 B。强模式优先于枚举判断
# （"What company is the team lead of Project Atlas employed by?" 带 "team" 一词，
#   若不优先会被误判成枚举，子查询就变成了团队列表，而不是 bridge 补证）
_BRIDGE_STRONG = re.compile(
    r"\b(employed by|works for|headquarters?|headquartered|whose|lead(?:er)? of|"
    r"owner of|creator of|founder of|made by|built by)\b|总部",
    re.I,
)
_BRIDGE_WEAK = re.compile(r"\b(company|companies|organization|隶属)\b", re.I)
# "What ... did/does/... VERB?" 构型：取助动词之后的最后一个实义词作为槽位词。
# 用两段式解析而不是单条正则：单条正则里 lazy 字符类与相邻 \s+ 组合存在回溯陷阱。
_WHAT_CUES = re.compile(r"^\s*what\b", re.I)
_AUX_CUES = re.compile(r"\b(did|does|do|has|have|had)\b\s+([\w\s'\-]*?)\s*[?.!]*$", re.I)
_BAD_TAIL = {
    "for", "with", "about", "from", "into", "after", "before", "during", "when",
    "where", "while", "that", "this", "them", "there", "then", "also", "just",
    "recently", "lately", "together", "instead", "what", "who", "how",
    "now", "today", "anymore", "else", "yet", "ever", "instead", "back",
}
# 时间锚点型：after / before / earlier / later + 事件
_TIMELINE_CUES = re.compile(
    r"\b(after|before|earlier|later|previously)\b.*\b(\w+)\b|在[\u4e00-\u9fff]{0,8}(?:之前|之后)",
    re.I,
)

# 槽位词干 → 同义词扩展（枚举轮第二跳的检索词，词袋拼入，hash-embedding + BM25 都能吃到）
SLOT_EXPANSIONS: dict[str, str] = {
    "hobby": "hobbies hobby pursue enjoy activities interests likes",
    "activity": "activities activity did does enjoy hobbies",
    "recipe": "recipes recipe cooked baked made cook make",
    "book": "books book read reading author authors novels",
    "game": "games game played play tournament tournaments won participated",
    "tournament": "tournaments tournament games won participated play",
    "movie": "movies movie watched watch films film",
    "song": "songs song listened music listen",
    "food": "foods food eat ate meals meal restaurants restaurant",
    "place": "places place visited travel traveled cities city countries",
    "recommend": "recommendations recommended recommend suggests suggested",
    "team": "teams team joined plays played",
    "course": "courses course classes class completed took",
    "reason": "reasons reason because motivated",
    "goal": "goals goal plans plan decided",
}

_STOP_ANCHORS = {"user", "i", "me", "my", "you", "we", "they", "what", "who", "which"}


@dataclass
class SubQuery:
    """证据链的一个跳：独立检索的查询文本 + 其语义角色。"""

    text: str
    reason: str  # enumeration | bridge | timeline | neighbor

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "reason": self.reason}


# --------------------------------------------------------------------------------------
# 触发与分解
# --------------------------------------------------------------------------------------
def needs_chain(query: str) -> bool:
    """是否需要证据链检索（第一跳之后仍可能有分散在其他会话的证据）。"""
    if _QUANT_CUES.search(query):
        return True
    if _ENUM_CUES.search(query):
        return True
    if _BRIDGE_STRONG.search(query) or _BRIDGE_WEAK.search(query):
        return True
    if _verb_tail(query):
        return True
    return bool(_TIMELINE_CUES.search(query))


def _anchor_candidates(query: str, hits: Sequence["RecallHit"]) -> list[str]:
    """问题聚焦的主体：优先 query 中出现的专名，其次首轮命中里最频繁的非用户实体。"""
    found: list[str] = []
    for entity in extract_entities(query):
        if entity.lower() not in _STOP_ANCHORS:
            found.append(entity)
    counter: Counter[str] = Counter()
    for hit in hits[:5]:
        item = hit.memory
        for key in ("subject", "object"):
            value = item.data.get(key)
            if isinstance(value, str) and len(value) >= 2:
                counter[value.lower()] += 1
    for value, _count in counter.most_common():
        if value in _STOP_ANCHORS:
            continue
        if any(value == f.lower() or f.lower() in value for f in found):
            continue
        if len(value) <= 40:
            found.append(value)
        if len(found) >= 2:
            break
    return found


def _slot_keywords(query: str) -> str:
    """从 query 提取槽位词：同义词扩展表命中则展开，同时保留原问句中的实义词。"""
    words = query.lower()
    words = re.sub(r"[^a-z\u4e00-\u9fff0-9\s]", " ", words)
    tokens = [w for w in words.split() if len(w) > 2]
    expansions: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for stem, expansion in SLOT_EXPANSIONS.items():
            if token.startswith(stem) or stem.startswith(token):
                if stem not in seen:
                    seen.add(stem)
                    expansions.append(expansion)
    kept = [t for t in tokens if t not in {
        "what", "which", "when", "where", "who", "whom", "how", "many", "much",
        "did", "does", "do", "has", "have", "had", "was", "were", "is", "are",
        "the", "and", "for", "with", "your", "you", "kind", "kinds", "type", "types",
    }]
    parts: list[str] = []
    for group in (expansions, kept):
        for item in group:
            if item.lower() not in parts:
                parts.append(item)
            if len(" ".join(parts).split()) >= 10:
                break
    return " ".join(parts)


def _verb_tail(query: str) -> str:
    """'What did Caroline research?' 取尾部行为动词作为槽位词；助动词/介词/副词结尾返回空。"""
    if not _WHAT_CUES.search(query):
        return ""
    m = _AUX_CUES.search(query)
    if not m:
        return ""
    tail = m.group(2).strip()
    words = re.findall(r"[a-z']{3,}", tail.lower())
    if not words:
        return ""
    verb = words[-1]
    if verb in _BAD_TAIL:
        return ""
    return verb


def _dedupe_words(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for word in text.split():
        key = word.lower()
        if key not in seen:
            seen.add(key)
            out.append(word)
    return " ".join(out)


def decompose_subqueries(
    query: str,
    hits: Sequence["RecallHit"],
    max_subqueries: int = 2,
) -> list[SubQuery]:
    """把复杂查询分解为结构化子查询（最多 max_subqueries 个）。"""
    subs: list[SubQuery] = []

    # 1) 图谱桥接（强模式优先）：实体定位 + 关系词补证
    if _BRIDGE_STRONG.search(query):
        anchors = _anchor_candidates(query, hits)
        for anchor in anchors[:2]:
            text = _dedupe_words(f"{anchor} {_slot_keywords(query)}")
            subs.append(SubQuery(text=text, reason="bridge"))
            if len(subs) >= max_subqueries:
                break
        return subs

    # 2) 枚举聚合：主体 + 槽位同义词（召回分散在多会话的长尾取值）
    enum_title = bool(_QUANT_CUES.search(query) or _ENUM_CUES.search(query))
    verb_tail = _verb_tail(query)
    if enum_title or verb_tail:
        anchors = _anchor_candidates(query, hits)
        slot = _slot_keywords(query)
        if verb_tail and verb_tail not in " ".join(anchors):
            slot = f"{slot} {verb_tail}".strip()
        parts = []
        if anchors:
            parts.append(anchors[0])
        if slot:
            parts.append(slot)
        if parts:
            text = _dedupe_words(" ".join(parts))
            subs.append(SubQuery(text=text, reason="enumeration"))

    # 3) 时间锚点：先定位事件时间，再召回前后证据
    if not subs and _TIMELINE_CUES.search(query):
        anchor = _anchor_candidates(query, hits)
        text = "when " + (anchor[0] + " " if anchor else "") + "date time"
        subs.append(SubQuery(text=text, reason="timeline"))

    return subs[:max_subqueries]


# --------------------------------------------------------------------------------------
# LLM 化分解（可插拔）：结构化输出校验 + 失败回退规则版
# --------------------------------------------------------------------------------------
_DECOMPOSE_SYSTEM = (
    "You decompose a complex memory-retrieval query into 1-2 focused sub-queries. "
    "Answer with a JSON array only, like "
    '[{"text": "Evan hobbies pursue enjoy", "reason": "enumeration"}] . '
    '"reason" must be one of: enumeration (enumerate/aggregate answers scattered '
    "across sessions), bridge (multi-hop entity relation), timeline (before/after events)."
)
REASON_SET = {"enumeration", "bridge", "timeline"}


def parse_subqueries_json(raw: str) -> list[SubQuery]:
    """把 LLM 输出解析为 SubQuery 列表；任何结构问题返回空列表（触发回退）。

    容忍 markdown code fence 与前后杂讯，但每一项必须满足：
    - text 为非空字符串（长度 1..200）
    - reason 在 {enumeration, bridge, timeline} 内
    """
    if not raw:
        return []
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    else:
        match = re.search(r"\[.*\]", raw, re.S)
        raw = match.group(0) if match else raw
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    subs: list[SubQuery] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        reason = entry.get("reason")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 200:
            continue
        if reason not in REASON_SET:
            continue
        subs.append(SubQuery(text=text.strip(), reason=reason))
    return subs


class LLMQueryDecomposer:
    """LLM 化 query 分解器：兼容任意 OpenAI Chat Completions 客户端
    （仅需 `complete(prompt, system=None)` 方法，如 OpenAICompatClient / CallableLLM）。

    失败策略：LLM 调用异常、输出非法 JSON、字段不合法时，全部回退到规则版
    `decompose_subqueries`，保证显存/网络抖动不会中断检索链路。
    """

    def __init__(self, client: Any, fallback: bool = True, verbose: bool = False) -> None:
        self.client = client
        self.fallback = fallback
        self.verbose = verbose
        self.calls = 0
        self.fallbacks = 0
        self.last_fallback = False

    def decompose(
        self,
        query: str,
        hits: Sequence["RecallHit"],
        max_subqueries: int = 2,
    ) -> list[SubQuery]:
        rule_subs = decompose_subqueries(query, hits, max_subqueries)
        if not needs_chain(query):
            return rule_subs
        prompt = (
            f"Query: {query}\n"
            f"First-round top hits: " + "; ".join(
                h.memory.text[:80] for h in hits[:3]
            ) + "\n"
            f"Return at most {max_subqueries} sub-queries as a JSON array."
        )
        self.calls += 1
        self.last_fallback = False
        try:
            raw = self.client.complete(prompt, system=_DECOMPOSE_SYSTEM)
            subs = parse_subqueries_json(raw)[:max_subqueries]
        except Exception as exc:  # noqa: BLE001 - LLM 输出不可控，必须兜底
            subs = []
            if self.verbose:
                print(f"[LLMQueryDecomposer] 调用失败回退规则版: {type(exc).__name__}: {exc}")
        if not subs:
            self.fallbacks += 1
            self.last_fallback = True
            return rule_subs if self.fallback else []
        return subs


# --------------------------------------------------------------------------------------
# 近邻扩证（双路径中的"回忆"路径）
# --------------------------------------------------------------------------------------
def should_neighbor_expand(top_score: float, min_top_score: float) -> bool:
    """首跳最高分过低 = 熟悉度不足：走图近邻扩证，而不是盲目重查整库。"""
    return top_score > 0.0 and top_score < min_top_score


def neighbor_top_scores(hits: Sequence["RecallHit"], k: int = 3) -> dict[str, float]:
    return {h.memory.id: h.score for h in hits[:k] if h.score > 0}


# --------------------------------------------------------------------------------------
# 合并与验收
# --------------------------------------------------------------------------------------
def merge_chain_hits(
    first: Sequence["RecallHit"],
    sub_results: Sequence[Sequence["RecallHit"]],
    top_k: int,
    slots_per_sub: int,
) -> list["RecallHit"]:
    """分名额合并证据链。

    规则（v2 教训的直接修正）：
    - 首轮前 (top_k - slots_per_sub × 子查询数) 名证据保序保留，不被挤掉；
    - 每个子查询最多补 slots_per_sub 个全新证据（id 未出现的才算新证据）；
    - 补证分数 ×0.9 与首轮尾部公平竞争，按分数排序后取 top_k。
    """
    first_sorted = sorted(first, key=lambda hit: hit.score, reverse=True)
    reserve = slots_per_sub * len(sub_results)
    keep_first = max(1, top_k - reserve)
    merged: list["RecallHit"] = list(first_sorted[:keep_first])
    used: set[str] = {hit.memory.id for hit in merged}
    for sub_hits in sub_results:
        taken = 0
        for hit in sorted(sub_hits, key=lambda h: h.score, reverse=True):
            if taken >= slots_per_sub:
                break
            if hit.memory.id in used:
                continue
            hit.score *= 0.9  # 补证证据弱化，避免"后手"压过首轮答案证据
            merged.append(hit)
            used.add(hit.memory.id)
            taken += 1
    return sorted(merged, key=lambda hit: hit.score, reverse=True)[:top_k]