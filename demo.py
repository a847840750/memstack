"""MemStack 端到端演示：写入 → 更新 → 检索 → 巩固 → 工具调用。

运行：
    python demo.py
"""

from __future__ import annotations

import calendar
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memstack import MemoryStack, Scope  # noqa: E402


def ts(year: int, month: int, day: int) -> float:
    return float(calendar.timegm(datetime(year, month, day).timetuple()))


def show(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    stack = MemoryStack()
    ingest = Scope(user_id="nora", agent_id="assistant", session_id="s1")
    query_scope = Scope(user_id="nora", agent_id="assistant")

    show("1. 写入会话 1（2026-04-28）")
    result = stack.remember(
        [
            {"role": "user", "content": "My name is Nora. I live in Lisbon. I work as a data analyst at Northwind Labs."},
            {"role": "assistant", "content": "Nice to meet you, Nora. How can I help?"},
            {"role": "user", "content": "I love Peruvian food. I am allergic to peanuts. I completed 3 courses on Coursera."},
            {"role": "assistant", "content": "Noted. Congratulations on the courses."},
        ],
        ingest,
        event_time=ts(2026, 4, 28),
    )
    for decision in result.decisions:
        print(f"  [{decision.action.value:9}] {decision.kind:10} {decision.text[:72]}")

    show("2. 写入会话 2（2026-10-28）：搬家、换工作、偏好变化")
    s2 = Scope(user_id="nora", agent_id="assistant", session_id="s2")
    result = stack.remember(
        [
            {"role": "user", "content": "I moved to Porto last month. I now work as a product manager at Blue Harbor."},
            {"role": "assistant", "content": "Congratulations on the move and the new role."},
            {"role": "user", "content": "I now prefer Brazilian food. Project Atlas now uses ClickHouse."},
            {"role": "assistant", "content": "I have updated your preferences and project notes."},
        ],
        s2,
        event_time=ts(2026, 10, 28),
    )
    for decision in result.decisions:
        print(f"  [{decision.action.value:9}] {decision.kind:10} {decision.text[:72]}  ({decision.reason})")

    show("3. 写入会话 3（2026-11-05）：流程、重复过敏信息、课程累加")
    s3 = Scope(user_id="nora", agent_id="assistant", session_id="s3")
    result = stack.remember(
        [
            {"role": "user", "content": "I am allergic to peanuts. I completed 2 courses on edX."},
            {"role": "assistant", "content": "I will keep that in mind."},
            {
                "role": "user",
                "content": (
                    "To submit an expense report, first open the finance portal, "
                    "then fill in the report form, then attach receipts, and finally submit it for approval."
                ),
            },
            {"role": "assistant", "content": "Understood."},
        ],
        s3,
        event_time=ts(2026, 11, 5),
    )
    for decision in result.decisions:
        print(f"  [{decision.action.value:9}] {decision.kind:10} {decision.text[:72]}")

    show("4. 当前状态（时间感知 + 冲突消解）")
    for question in [
        "Where does Nora live now?",
        "Where did Nora live before?",
        "What does Nora do for work?",
        "What food does Nora prefer now?",
        "What database does Project Atlas use now?",
        "How many online courses has Nora completed in total?",
        "How should Nora submit an expense report?",
    ]:
        bundle = stack.recall(question, query_scope, token_budget=620)
        print(f"\nQ: {question}")
        print(bundle.text)
        print(
            f"   confidence={bundle.confidence:.2f} coverage={bundle.coverage:.2f} "
            f"tokens≈{bundle.token_estimate} history={bundle.trace.get('history_mode')}"
        )
        if bundle.gaps:
            print(f"   gaps={bundle.gaps}")

    show("5. 离线巩固（睡眠期）")
    report = stack.consolidate(query_scope)
    print("  report:", report)
    profile = stack.get_block("user_profile", query_scope)
    print("\n  user_profile block:")
    print("   " + (profile.value.replace("\n", "\n   ") if profile else "(empty)"))

    show("6. 记忆库统计")
    print("  ", stack.stats(query_scope))

    show("7. 作为 Agent 工具调用（OpenAI function-calling 格式）")
    tools = stack.as_tools()
    print("  tools:", [tool["function"]["name"] for tool in tools])
    response = stack.dispatch_tool(
        "memory_search",
        {"query": "Where does Nora live now?", "scope": query_scope.to_dict(), "token_budget": 400},
    )
    print("  memory_search ->")
    print("   ", response["result"]["context"].splitlines()[0:4])
    response = stack.dispatch_tool(
        "memory_add",
        {
            "messages": [{"role": "user", "content": "My favorite book is The Left Hand of Darkness."}],
            "scope": query_scope.to_dict(),
        },
    )
    print("  memory_add -> actions:", [d["action"] for d in response["result"]["decisions"]])

    stack.close()


if __name__ == "__main__":
    main()
