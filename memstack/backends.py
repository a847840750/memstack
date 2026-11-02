"""可选模型基座（LLM backend）配置。

模型基座把「评测 / 推理时调用的 LLM 服务」抽成可选项：默认不启用（零依赖规则版），
需要时用 ``MEMSTACK_LLM_BASE`` 或命令行 ``--model-base`` 选择内置 preset。

个人端点与密钥用环境变量覆盖（``OPENAI_BASE_URL`` / ``MEMSTACK_API_KEY`` 等），
不写死在代码里。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ModelBackend:
    """一个模型基座的连接参数。"""

    name: str
    model: str
    base_url: str
    api_key: str = ""
    reasoning_effort: str | None = None
    timeout: float = 120.0
    extra_payload: dict[str, Any] = field(default_factory=dict)


# 内置基座：本地 Ollama 上的 Qwen3.5-9B（OpenAI 兼容 /v1 端点）
BUILTIN_BACKENDS: dict[str, ModelBackend] = {
    "qwen35-9b-local": ModelBackend(
        name="qwen35-9b-local",
        model="qwen35-9b-local",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        # Qwen3.5-9B 默认开启思考会让 content 为空，这里默认关掉
        reasoning_effort="none",
        timeout=180.0,
    ),
}

ENV_BASE = "MEMSTACK_LLM_BASE"


def resolve_backend(name: str | None = None) -> ModelBackend | None:
    """按名称解析基座；名称缺省时读 ``MEMSTACK_LLM_BASE``。未配置返回 None。"""
    key = (name or os.getenv(ENV_BASE) or "").strip()
    if not key:
        return None
    if key not in BUILTIN_BACKENDS:
        available = ", ".join(sorted(BUILTIN_BACKENDS))
        raise ValueError(f"未知模型基座 {key!r}，可选：{available}")
    backend = BUILTIN_BACKENDS[key]
    model = os.getenv("MEMSTACK_LLM_MODEL") or backend.model
    base_url = (
        os.getenv("MEMSTACK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or backend.base_url
    ).rstrip("/")
    api_key = os.getenv("MEMSTACK_API_KEY") or os.getenv("OPENAI_API_KEY") or backend.api_key
    effort = os.getenv("MEMSTACK_REASONING_EFFORT")
    if effort is None:
        effort = backend.reasoning_effort
    return replace(
        backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=effort or None,
    )


def build_backend_client(backend: ModelBackend) -> Any:
    """按基座配置构造 OpenAI 兼容客户端。"""
    from .llm import OpenAICompatClient

    return OpenAICompatClient(
        model=backend.model,
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=backend.timeout,
        reasoning_effort=backend.reasoning_effort,
        extra_payload=backend.extra_payload,
    )
