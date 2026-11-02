"""工厂函数：从环境变量开箱即用地构造完整系统（含可选的真实 LLM 抽取/摘要/查询分解）。

不设置任何环境变量时，构造的是零依赖默认系统（规则抽取 + hashing 向量 + 规则分解）；
设置 LLM 后自动切换成 LLM 抽取、摘要与查询分解：

    # 任意 OpenAI 兼容端点（OpenAI / DeepSeek / vLLM / Ollama /v1 均可）
    export MEMSTACK_LLM_MODEL=gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    python -c "from memstack import build_stack; s = build_stack(db='mem.db')"

    # 多 agent 系统
    export MEMSTACK_HUB=1
"""

from __future__ import annotations

import os
from typing import Any

from .backends import build_backend_client, resolve_backend
from .config import MemoryConfig
from .core import MemoryStack
from .evidence_chain import LLMQueryDecomposer
from .extraction import LLMExtractor, LLMSummarizer, RuleExtractor, RuleSummarizer
from .llm import OpenAICompatClient
from .multiagent import MultiAgentHub
from .storage import SQLiteStore


def build_llm_client() -> OpenAICompatClient | None:
    """按环境变量构造 OpenAI 兼容客户端；未配置返回 None。

    优先识别模型基座（``MEMSTACK_LLM_BASE``，如 ``qwen35-9b-local``），
    否则走通用的 ``MEMSTACK_LLM_MODEL`` + API key 组合。
    """
    backend = resolve_backend()
    if backend is not None:
        return build_backend_client(backend)
    model = os.getenv("MEMSTACK_LLM_MODEL") or os.getenv("OPENAI_MODEL")
    api_key = os.getenv("MEMSTACK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not model or not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MEMSTACK_BASE_URL")
    return OpenAICompatClient(model=model, api_key=api_key, base_url=base_url)


def build_stack(
    db: str = ":memory:",
    config: MemoryConfig | None = None,
    llm_client: Any | None = None,
    llm_decompose: bool = False,
    **config_overrides: Any,
) -> MemoryStack:
    """构造 MemoryStack；有 LLM 客户端时自动启用 LLM 抽取/摘要，失败回退规则实现。

    `llm_decompose=True` 时额外启用 LLM 化查询分解（证据链子查询）；分解失败自动回退
    规则版。默认关闭，保持复杂查询的零 LLM 成本。
    """
    config = config or MemoryConfig()
    if config_overrides:
        from dataclasses import replace
        config = replace(config, **config_overrides)
    client = llm_client if llm_client is not None else build_llm_client()
    store = SQLiteStore(db) if db != ":memory:" else None
    stack = MemoryStack(config=config, store=store)
    if client is not None:
        try:
            stack.extractor = LLMExtractor(client, fallback=RuleExtractor())
            stack.summarizer = LLMSummarizer(client, fallback=RuleSummarizer())
            if llm_decompose:
                stack.query_decomposer = LLMQueryDecomposer(client)
        except Exception:  # noqa: BLE001 - LLM 不可用时回退规则实现
            pass
    return stack


def build_hub(
    db: str = ":memory:",
    tenant_id: str | None = None,
    config: MemoryConfig | None = None,
) -> MultiAgentHub:
    """构造带多 agent 支持的完整系统。"""
    stack = build_stack(db=db, config=config)
    return MultiAgentHub(stack, tenant_id=tenant_id or os.getenv("MEMSTACK_TENANT", "default"))