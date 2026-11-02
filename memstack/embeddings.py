"""零依赖的确定性向量化与文本工具。

默认使用 hashing trick：无需下载模型、无网络、跨进程稳定，适合作为可移植默认层。
生产环境只需实现 Embedder Protocol 换成 BGE/OpenAI 等模型，其他层无需改动。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_'\-]*", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'\-]*|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on",
    "at", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "my", "your", "our", "their", "do", "does", "did", "have",
    "has", "had", "will", "would", "can", "could", "should", "may", "might", "not",
    "no", "yes", "so", "than", "too", "very", "just", "about", "into", "over", "after",
    "before", "now", "also", "there", "here", "what", "which", "who", "whom", "when",
    "where", "why", "how", "all", "any", "both", "each", "more", "most", "other",
    "some", "such", "only", "own", "same", "up", "down", "out", "off", "again",
}


def _stem(token: str) -> str:
    """轻量英文词干化：让 headquarters/headquartered、works/working 能相互匹配。"""
    if not token.isascii() or len(token) <= 4 or not token.isalpha():
        return token
    for suffix in ("ingly", "edly", "ing", "ies", "ied", "ers", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if suffix in {"ies", "ied"}:
                return base + "y"
            if suffix in {"ing", "ed"} and base[-1] in "vtklzsc":
                return base + "e"
            return base
    return token


def tokenize(text: str, keep_stopwords: bool = True) -> list[str]:
    """英文按词、中文按单字；返回小写、轻量词干化后的 token。"""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def content_tokens(text: str) -> list[str]:
    return tokenize(text, keep_stopwords=False)


def estimate_tokens(text: str) -> int:
    """保守的 token 估算：英文约 1.3 token/词，中文约 1 字 1 token。"""
    if not text:
        return 0
    words = _WORD_RE.findall(text)
    cjk = len(_CJK_RE.findall(text))
    return int(len(words) * 1.3 + cjk + 0.999)


def _hash(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashingEmbedder:
    """确定性 hashing 向量：词 unigram + 词 bigram + 中文字 bigram。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        weighted: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
        for a, b in zip(tokens, tokens[1:]):
            weighted.append((f"{a}_{b}", 0.5))
        # 中文双字片段，缓解单字歧义
        cjk = _CJK_RE.findall(text)
        for a, b in zip(cjk, cjk[1:]):
            weighted.append((f"zh:{a}{b}", 0.8))
        for token, weight in weighted:
            idx, sign = _hash(token, self.dim)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def cosine_batch(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    """批量余弦相似度。numpy 可用时向量化计算（数量级加速），否则回退纯 Python。"""
    if not matrix:
        return []
    try:
        import numpy as np
    except ImportError:  # numpy 是可选加速依赖
        return [max(0.0, cosine(query, vec)) for vec in matrix]
    q = np.asarray(list(query), dtype=np.float64)
    m = np.asarray([list(vec) for vec in matrix], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return (m @ q).clip(min=0.0).tolist()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    t = max(temperature, 1e-6)
    mx = max(scores)
    exp = [math.exp((s - mx) / t) for s in scores]
    total = sum(exp) or 1.0
    return [e / total for e in exp]


def entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
