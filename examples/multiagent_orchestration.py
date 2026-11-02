"""多 Agent 协作示例：planner / executor / auditor 三个 agent 通过团队共享记忆与事件总线协作。

运行：python examples/multiagent_orchestration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memstack import build_hub


def main() -> None:
    hub = build_hub()  # 内存库；生产改为 build_hub(db="team.db")

    # 1. 注册 agent 并成立 closed 团队（非成员不可见）
    for name, role in [("planner", "planner"), ("executor", "executor"), ("auditor", "auditor")]:
        hub.register_agent(name, role=role)
    hub.create_team("dashcam", members=["planner", "executor", "auditor"], policy="closed")

    # 2. planner 写入共享约束
    hub.remember_as(
        "planner",
        [{"role": "user", "content": "预算上限 5000 元, 交付时间 6 月 1 日。"}],
        team_id="dashcam", share=True,
    )

    # 3. executor 干活，并把产出广播为事件（同步写共享记忆）
    hub.remember_as(
        "executor",
        [{"role": "user", "content": "已完成行车记录仪外壳选型: ABS 塑料, 单价 12 元。"}],
        team_id="dashcam", share=True,
    )
    event_id = hub.post_event(
        "dashcam", "executor", "milestone",
        "外壳选型完成，等待 planner 确认", as_memory=True,
    )

    # 4. auditor 增量拉取事件，检查约束一致性
    events = hub.pull_events("auditor", "dashcam", since_id=0)
    context = hub.recall_as("auditor", "预算与选型成本", teams=["dashcam"])
    print("=== auditor 拉取到的事件 ===")
    for event in events:
        print(f"  [{event['type']}] {event['sender']}: {event['text']}")
    print("\n=== auditor 的合并上下文（私有 + 团队共享）===")
    print(context.text[:600])

    # 5. 越权演示：已注册但未入队的 agent 无法读 closed 团队
    hub.register_agent("intruder")
    try:
        hub.recall_as("intruder", "预算", teams=["dashcam"])
    except PermissionError as exc:
        print(f"\n=== 越权被拒: {exc} ===")

    print("\n=== hub 统计 ===")
    print(hub.stats())


if __name__ == "__main__":
    main()