"""MemStack：面向 Agent 的可移植分层记忆栈。"""

from .backends import BUILTIN_BACKENDS, ModelBackend, build_backend_client, resolve_backend
from .config import MemoryConfig
from .core import MemoryStack, SystemClock, TopicSegmenter
from .embeddings import HashingEmbedder, cosine, cosine_batch, estimate_tokens
from .evidence_chain import (
    LLMQueryDecomposer,
    SubQuery,
    decompose_subqueries,
    needs_chain,
    parse_subqueries_json,
)
from .extraction import LLMExtractor, LLMSummarizer, RuleExtractor, RuleSummarizer
from .factory import build_hub, build_llm_client, build_stack
from .interfaces import Clock, Embedder, Extractor, MemoryStore, Reranker, Summarizer
from .llm import CallableLLM, OpenAICompatClient
from .multiagent import AccessDenied, MultiAgentHub
from .storage import SQLiteStore
from .tools import MemoryMiddleware, dispatch_tool, openai_tool_specs
from .types import (
    ContextBundle,
    EdgeType,
    MemoryBlock,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    RecallHit,
    RecallQuery,
    RememberResult,
    Scope,
    WriteAction,
)

__version__ = "0.4.0"

__all__ = [
    "MemoryStack",
    "MemoryConfig",
    "SystemClock",
    "TopicSegmenter",
    "HashingEmbedder",
    "SQLiteStore",
    "RuleExtractor",
    "RuleSummarizer",
    "LLMExtractor",
    "LLMSummarizer",
    "LLMQueryDecomposer",
    "SubQuery",
    "decompose_subqueries",
    "needs_chain",
    "parse_subqueries_json",
    "OpenAICompatClient",
    "ModelBackend",
    "BUILTIN_BACKENDS",
    "resolve_backend",
    "build_backend_client",
    "CallableLLM",
    "MemoryMiddleware",
    "dispatch_tool",
    "openai_tool_specs",
    "MultiAgentHub",
    "AccessDenied",
    "build_stack",
    "build_hub",
    "build_llm_client",
    "Scope",
    "MemoryItem",
    "MemoryKind",
    "MemoryStatus",
    "EdgeType",
    "WriteAction",
    "RecallQuery",
    "RecallHit",
    "ContextBundle",
    "RememberResult",
    "MemoryBlock",
    "Embedder",
    "Extractor",
    "Summarizer",
    "MemoryStore",
    "Reranker",
    "Clock",
    "cosine",
    "cosine_batch",
    "estimate_tokens",
]
