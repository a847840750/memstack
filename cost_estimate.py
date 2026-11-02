"""OpenAI 兼容 API 评测成本估算。

按 MemStack 现有评测管线的真实调用结构逐项估算 token 与费用：

  - 写入抽取（LLMExtractor）：每个对话分段（≤ N 轮）调用一次 LLM 抽取结构化事实
  - 巩固（可选）：consolidate 时的会话摘要
  - reader：每个 QA 调用一次 LLM 生成答案（上下文受 token_budget 限制）
  - judge：每个 QA 调用一次 LLM 判 CORRECT/INCORRECT（对抗题用确定性规则，不花 LLM）

价格默认值取自 2026-10 各厂商公开定价页，运行 `--help` 查看如何覆盖。
本脚本只做算术，不调用任何 API。

用法：
    python cost_estimate.py                 # 默认方案对比表
    python cost_estimate.py --price deepseek 0.27 1.10   # 自定义单价($/1M)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Price:
    name: str
    per_1m_in: float   # 美元 / 1M input token
    per_1m_out: float  # 美元 / 1M output token
    note: str = ""


# 2026-10 公开定价（美元/百万 token），使用前请以官网为准
DEFAULT_PRICES = [
    Price("DeepSeek-chat", 0.27, 1.10, "deepseek.com 官方价；人民币 ≈ $×7.2"),
    Price("GPT-4.1-mini", 0.40, 1.60, "SimpleMem 论文同 backbone"),
    Price("GPT-4o-mini", 0.15, 0.60, "便宜但抽取质量弱，不建议大规模 judge"),
    Price("GPT-4o", 2.50, 10.00, "高配；Mem0/Zep 论文口径，成本高一个量级"),
]

USD_TO_CNY = 7.2


@dataclass
class Task:
    name: str
    ingest_calls: int      # 写入抽取调用次数（每分段一次）
    ingest_in: int
    ingest_out: int
    reader_calls: int
    reader_in: int
    reader_out: int
    judge_calls: int
    judge_in: int
    judge_out: int
    note: str = ""


def build_tasks(turns_per_segment: int = 3) -> list[Task]:
    """按数据集统计量构造任务。轮数与分段数来自公开数据集实际规模。"""
    locomo_turns = 5882                       # locomo10.json 实测总轮数
    locomo_segments = locomo_turns // turns_per_segment
    locomo_qa = 1986                          # 全量 QA
    locomo_qa_50 = 50                         # 分层 50 题小实验
    locomo_judge_50 = 50
    # 抽取调用：出入 token 按每段 ~450 词对话 + prompt 估算
    return [
        Task(
            name="LoCoMo 全量（1986 题）",
            ingest_calls=locomo_segments, ingest_in=500, ingest_out=250,
            reader_calls=locomo_qa, reader_in=900, reader_out=30,
            judge_calls=locomo_qa, judge_in=250, judge_out=5,
            note="对抗题已按规则判分但仍计入保守上界",
        ),
        Task(
            name="LoCoMo 50 题小实验（写入全部 10 会话）",
            ingest_calls=locomo_segments, ingest_in=500, ingest_out=250,
            reader_calls=locomo_qa_50, reader_in=900, reader_out=30,
            judge_calls=locomo_judge_50, judge_in=250, judge_out=5,
            note="写入成本与全量相同；只有 reader/judge 按 50 题计",
        ),
        Task(
            name="LongMemEval oracle 500（写入 + 作答）",
            ingest_calls=7500, ingest_in=600, ingest_out=250,
            reader_calls=500, reader_in=900, reader_out=30,
            judge_calls=500, judge_in=250, judge_out=5,
            note="oracle 数据平均每问题约 15 个证据会话、每会话约 3 轮",
        ),
        Task(
            name="LongMemEval-s 带 distractor（不推荐直接跑）",
            ingest_calls=25000, ingest_in=600, ingest_out=250,
            reader_calls=500, reader_in=900, reader_out=30,
            judge_calls=500, judge_in=250, judge_out=5,
            note="约 25k 个会话全部要抽取；建议先做检索层评测、按需小批量 QA",
        ),
    ]


def task_cost(task: Task, price: Price, overhead: float = 1.15) -> dict[str, float]:
    """返回该任务在该价格下的 token 量与费用（含 overhead 重试/长尾）。"""
    total_in = (
        task.ingest_calls * task.ingest_in
        + task.reader_calls * task.reader_in
        + task.judge_calls * task.judge_in
    )
    total_out = (
        task.ingest_calls * task.ingest_out
        + task.reader_calls * task.reader_out
        + task.judge_calls * task.judge_out
    )
    usd = (total_in / 1e6 * price.per_1m_in + total_out / 1e6 * price.per_1m_out) * overhead
    return {
        "in_tokens": total_in,
        "out_tokens": total_out,
        "usd": usd,
        "cny": usd * USD_TO_CNY,
    }


def format_tokens(value: float) -> str:
    return f"{value / 1e6:.2f}M"


def render(tasks: list[Task], prices: list[Price], overhead: float = 1.15) -> str:
    lines: list[str] = []
    for price in prices:
        lines.append(f"\n=== {price.name}（${price.per_1m_in}/1M in, ${price.per_1m_out}/1M out）{price.note} ===")
        lines.append(
            f"{'任务':<40}{'输入':>10}{'输出':>10}{'美元':>9}{'人民币':>9}"
        )
        total_usd = 0.0
        for task in tasks:
            cost = task_cost(task, price, overhead)
            total_usd += cost["usd"]
            lines.append(
                f"{task.name:<40}{format_tokens(cost['in_tokens']):>10}"
                f"{format_tokens(cost['out_tokens']):>10}"
                f"{cost['usd']:>8.2f}$ {cost['cny']:>7.1f}¥"
            )
        lines.append(f"{'合计（4 项全跑）':<40}{'':>10}{'':>10}{total_usd:>8.2f}$ {total_usd * USD_TO_CNY:>7.1f}¥")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MemStack OpenAI 兼容 API 评测成本估算")
    parser.add_argument("--turns-per-segment", type=int, default=3,
                        help="每个抽取分段的最大轮数（与 --turns 一致）")
    parser.add_argument("--overhead", type=float, default=1.15,
                        help="重试/长尾/缓存未命中的额外系数，默认 1.15")
    parser.add_argument("--price", nargs=3, action="append", metavar=("名字", "in$/1M", "out$/1M"),
                        help="追加自定义价格方案")
    parser.add_argument("--rates", type=float, default=USD_TO_CNY, help="美元兑人民币汇率")
    args = parser.parse_args()

    tasks = build_tasks(args.turns_per_segment)
    prices = list(DEFAULT_PRICES)
    if args.price:
        for name, pin, pout in args.price:
            prices.append(Price(name, float(pin), float(pout)))

    header = (
        f"假设：每分段 ≤{args.turns_per_segment} 轮一次抽取调用；"
        f"reader 输入 = 上下文预算 800 + prompt ≈ 900 token；"
        f"judge 输入 ≈ 250 token；overhead ×{args.overhead}；汇率 {args.rates}。\n"
        f"注意：这是估算上界，实际费用只低不高（缓存命中、加长分段都会省钱）。"
    )
    print(header)
    print(render(tasks, prices, args.overhead))


if __name__ == "__main__":
    main()