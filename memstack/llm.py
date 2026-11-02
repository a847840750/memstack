"""可选的真实 LLM 适配层：零 SDK 依赖，仅标准库 urllib。

默认系统不调用 LLM；只有显式注入 LLMExtractor / LLMSummarizer 时才会用到。
任何兼容 OpenAI Chat Completions 的服务都可接入（OpenAI、DeepSeek、Qwen、vLLM、
Ollama 的 /v1 端点等）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        extra_payload: dict[str, Any] | None = None,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        # 部分推理模型（如 Qwen3.5-9B）默认开启思考会让 content 为空
        self.reasoning_effort = reasoning_effort
        self.extra_payload = dict(extra_payload or {})

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        # 评测/判分这类短答案场景必须封顶：否则小模型偶尔会跑出长文本，
        # 把整个评测的吞吐拖到单题数十秒（实测 200ms → 31s）。
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.extra_payload:
            payload.update(self.extra_payload)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        # 长跑评测里本地端点会偶发 500（模型重载/显存抖动），重试即可续上，
        # 否则一轮几十分钟的评测会在单次抖动处整轮失败。
        attempts = max(1, int(self.max_retries))
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                if attempt == attempts - 1:
                    raise RuntimeError(f"LLM 调用失败: HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
                if attempt == attempts - 1:
                    raise RuntimeError(f"LLM 调用失败: {exc}") from exc
            time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("LLM 调用失败: 重试耗尽")


class CallableLLM:
    """把任意函数包装成 LLM 客户端，便于接入已有 agent 的模型调用。"""

    def __init__(self, fn: Callable[[str, str | None], str]) -> None:
        self.fn = fn

    def complete(self, prompt: str, system: str | None = None) -> str:
        return self.fn(prompt, system)


class SequenceLLM:
    """测试用：按顺序返回预设回复。"""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        if not self.replies:
            return ""
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def build_default_router() -> Any:
    """辅助函数：从环境变量创建客户端；未配置则返回 None。"""
    model = os.getenv("MEMSTACK_LLM_MODEL")
    if not model:
        return None
    return OpenAICompatClient(model=model)
