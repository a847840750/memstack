"""与 LangGraph / CrewAI / AutoGen 的集成示例（内置最小 stub，未安装框架也能跑通逻辑）。

真实的接入方式与本示例完全一致——这些框架都在"节点/任务函数"里调你注入的服务：
  - LangGraph：在节点函数里调用 recall() 拿记忆、remember() 存记忆；
  - CrewAI：  用 @tool 装饰器把 MemoryStack 方法包成 crew 工具；
  - AutoGen： 把 recall/remember 作为 Agent 的工具函数传给 register_for_llm。

运行：python examples/framework_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memstack import MemoryStack, Scope, build_stack


# --------------------------------------------------------------------------------------
# 1. LangGraph 风格：每个节点先召回记忆、再写入本轮结果
# --------------------------------------------------------------------------------------
def langgraph_example(stack: MemoryStack) -> None:
    user = Scope(user_id="nora", agent_id="assistant")

    def recall_node(state: dict[str, Any]) -> dict[str, Any]:
        # 在真实 LangGraph 里这就是 Graph.add_node("recall", recall_node)
        bundle = stack.recall(state["input"], user, token_budget=500)
        return {"context": bundle.text, "confidence": bundle.confidence}

    def act_node(state: dict[str, Any]) -> dict[str, Any]:
        # 模拟 agent 的回复；真实场景把 context 交给 LLM 生成
        reply = f"(基于记忆回答) {state['context'][:80]}"
        stack.remember(
            [{"role": "user", "content": state["input"]}, {"role": "assistant", "content": reply}],
            user,
        )
        return {"reply": reply}

    # 两个节点的串行执行 = 迷你 graph
    state = {"input": "我下周要去里斯本出差"}
    state.update(recall_node(state))
    state.update(act_node(state))
    stack.remember([{"role": "user", "content": "我喜欢靠窗的座位"}], user)
    print("[langgraph] 第二轮召回:", stack.recall("用户喜欢什么座位", user).text[:120])


# --------------------------------------------------------------------------------------
# 2. CrewAI 风格：把记忆方法包成 @tool
# --------------------------------------------------------------------------------------
def crewai_example(stack: MemoryStack) -> None:
    user = Scope(user_id="nora", agent_id="assistant")

    def tool(func: Callable[..., Any]) -> Callable[..., Any]:
        func.tool = True  # 真实 CrewAI 的 @tool 会生成 schema；这里只是标记
        return func

    @tool
    def memory_recall(query: str) -> str:
        """Recall long-term memory relevant to the query."""
        return stack.recall(query, user, token_budget=400).text

    @tool
    def memory_remember(fact: str) -> str:
        """Persist a fact to long-term memory."""
        stack.remember([{"role": "user", "content": fact}], user)
        return "saved"

    stack.remember([{"role": "user", "content": "Nora 的过敏史: 青霉素"}], user)
    print("[crewai] 工具调用:", memory_recall("过敏史"))
    print("[crewai] 工具可被发现:", bool(getattr(memory_recall, "tool", False)))


# --------------------------------------------------------------------------------------
# 3. AutoGen 风格：函数直接注册给 LLM（function-calling）
# --------------------------------------------------------------------------------------
def autogen_example(stack: MemoryStack) -> None:
    user = Scope(user_id="nora", agent_id="assistant")

    def memory_recall(query: str) -> str:
        """回忆与 query 相关的长期记忆。"""
        return stack.recall(query, user, token_budget=400).text

    def memory_remember(fact: str) -> str:
        """把事实写入长期记忆。"""
        stack.remember([{"role": "user", "content": fact}], user)
        return "saved"

    # AutoGen 里：agent.register_for_llm()(memory_recall)
    registered = [memory_recall, memory_remember]
    print("[autogen] 可注册工具数:", len(registered))


def main() -> None:
    stack = build_stack()  # 设置了 MEMSTACK_LLM_MODEL+API key 时自动启用 LLM 抽取
    langgraph_example(stack)
    crewai_example(stack)
    autogen_example(stack)


if __name__ == "__main__":
    main()