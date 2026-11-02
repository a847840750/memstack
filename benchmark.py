"""MemStack 合成基准：在相同 token 预算下对比不同记忆策略。

这不是论文原表复现，而是机制级压力测试：
  - 事实更新（搬家/换工作/偏好变化/项目栈变化/书目变化）
  - 多会话聚合（课程总数）
  - 历史查询（以前住哪）
  - 多跳（公司总部）
  - 流程记忆（报销步骤）
  - 同预算下的成本-准确率权衡

运行：
    python benchmark.py
"""

from __future__ import annotations

import calendar
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memstack import HashingEmbedder, MemoryConfig, MemoryStack, Scope  # noqa: E402
from memstack.embeddings import cosine, estimate_tokens  # noqa: E402

TOKEN_BUDGET = 500
SEED = 20261028

USERS = [
    {
        "name": "Nora", "city_a": "Lisbon", "city_b": "Porto",
        "job1": "data analyst", "job2": "product manager",
        "emp1": "Northwind Labs", "emp2": "Blue Harbor", "hq": "Oslo",
        "cuisine1": "Peruvian", "cuisine2": "Brazilian", "allergen": "peanuts",
        "n1": 3, "n2": 2, "book1": "The Overstory", "book2": "The Left Hand of Darkness",
        "db1": "PostgreSQL", "db2": "ClickHouse", "lead": "Maya Chen",
    },
    {
        "name": "Elias", "city_a": "Osaka", "city_b": "Kyoto",
        "job1": "site engineer", "job2": "project lead",
        "emp1": "Cedar Works", "emp2": "Orbit Media", "hq": "Berlin",
        "cuisine1": "Japanese", "cuisine2": "Korean", "allergen": "shellfish",
        "n1": 4, "n2": 1, "book1": "The Wind-Up Bird Chronicle", "book2": "Kafka on the Shore",
        "db1": "MySQL", "db2": "DuckDB", "lead": "Tomas Novak",
    },
    {
        "name": "Priya", "city_a": "Nairobi", "city_b": "Mombasa",
        "job1": "research assistant", "job2": "data scientist",
        "emp1": "Lumen Health", "emp2": "Vertex Systems", "hq": "Toronto",
        "cuisine1": "Ethiopian", "cuisine2": "Lebanese", "allergen": "sesame",
        "n1": 2, "n2": 3, "book1": "The Remains of the Day", "book2": "Never Let Me Go",
        "db1": "SQLite", "db2": "Snowflake", "lead": "Aisha Bello",
    },
]

FILLERS = [
    "The weather has been rainy all week.",
    "I spent the weekend reorganizing the apartment.",
    "The commute took longer than usual this morning.",
    "We watched a documentary about deep sea creatures.",
    "The team had a long planning meeting on Tuesday.",
    "I need to buy groceries and pick up a parcel.",
    "Traffic was heavy on the bridge again.",
    "The coffee machine on our floor broke yesterday.",
    "I finished a puzzle and started another one.",
    "The bookstore downtown was crowded on Saturday.",
    "I finally cleaned out the closet in the hallway.",
    "The neighbors are painting their balcony this week.",
    "I missed the bus and walked to the office.",
    "We tried a new recipe with roasted vegetables.",
    "The library extended its opening hours.",
    "I sorted through old photos and letters.",
    "The office printer jammed three times today.",
    "A friend recommended a podcast about urban planning.",
]

ASSISTANT_REPLIES = [
    "Thanks for sharing that.",
    "That sounds like a busy day.",
    "Noted. Let me know if you need anything.",
    "Interesting. Anything else on your mind?",
    "I see. I will keep that in context.",
]

PROCEDURE_KEYWORDS = ["finance portal", "report form", "receipts", "approval"]


@dataclass
class Session:
    user: str
    index: int
    event_time: float
    scope: Scope
    messages: list[dict[str, str]]


@dataclass
class Question:
    user: str
    qtype: str
    text: str
    gold: str
    stale: str | None = None


def ts(year: int, month: int, day: int) -> float:
    return float(calendar.timegm(datetime(year, month, day).timetuple()))


def build_dataset() -> tuple[list[Session], list[Question]]:
    rng = random.Random(SEED)
    sessions: list[Session] = []
    questions: list[Question] = []
    dates = [(2025, 1, 15), (2025, 3, 1), (2025, 5, 1), (2025, 7, 1)]

    for user in USERS:
        name = user["name"]
        fact_texts = [
            (
                f"My name is {name}. I live in {user['city_a']}. I work as a {user['job1']} "
                f"at {user['emp1']}. I love {user['cuisine1']} food. I am allergic to {user['allergen']}. "
                f"I completed {user['n1']} courses on Coursera."
            ),
            (
                f"Project Atlas uses {user['db1']}. I prefer morning meetings. "
                f"My favorite book is {user['book1']}. {user['emp2']} is headquartered in {user['hq']}."
            ),
            (
                f"I moved to {user['city_b']} last month. I now work as a {user['job2']} "
                f"at {user['emp2']}. I now prefer {user['cuisine2']} food."
            ),
            (
                f"I completed {user['n2']} courses on edX. Project Atlas now uses {user['db2']}. "
                f"My favorite book is {user['book2']}. Project Atlas is led by {user['lead']}. "
                "To submit an expense report, first open the finance portal, then fill in the report form, "
                "then attach receipts, and finally submit it for approval."
            ),
        ]
        for idx, (year, month, day) in enumerate(dates):
            picks = rng.sample(FILLERS, 4)
            messages = [
                {"role": "user", "content": fact_texts[idx]},
                {"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)},
                {"role": "user", "content": picks[0]},
                {"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)},
                {"role": "user", "content": picks[1]},
                {"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)},
                {"role": "user", "content": picks[2]},
                {"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)},
                {"role": "user", "content": picks[3]},
                {"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)},
            ]
            sessions.append(
                Session(
                    user=name, index=idx, event_time=ts(year, month, day),
                    scope=Scope(user_id=name, agent_id="assistant", session_id=f"s{idx + 1}"),
                    messages=messages,
                )
            )

        questions.extend(
            [
                Question(name, "current_location", f"Where does {name} live now?", user["city_b"], user["city_a"]),
                Question(name, "history_location", f"Where did {name} live before?", user["city_a"], user["city_b"]),
                Question(name, "current_job", f"What does {name} do for work?", user["job2"], user["job1"]),
                Question(name, "employer", f"Where does {name} work now?", user["emp2"], user["emp1"]),
                Question(name, "preference", f"What food does {name} prefer now?", user["cuisine2"], user["cuisine1"]),
                Question(name, "allergy", f"What is {name} allergic to?", user["allergen"]),
                Question(name, "stack", "What database does Project Atlas use now?", user["db2"], user["db1"]),
                Question(name, "count", f"How many online courses has {name} completed in total?", str(user["n1"] + user["n2"]), str(user["n1"])),
                Question(name, "procedure", f"How should {name} submit an expense report?", "procedure"),
                Question(name, "book", f"What is {name}'s favorite book now?", user["book2"], user["book1"]),
                Question(name, "multihop", f"Which city is the headquarters of the company {name} works for now?", user["hq"]),
            ]
        )
    return sessions, questions


# --------------------------------------------------------------------------------------
# 规则化 reader：模拟一个只读上下文、不做额外推理的小模型
# --------------------------------------------------------------------------------------
def norm(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def norm_subject(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "-", value.lower()).strip("-")


def parse_slots(context: str) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    for line in context.splitlines():
        match = re.search(r"slot=([^\s|]+)\s*\|\s*value=([^|]+)", line)
        if match:
            status = "active"
            status_match = re.search(r"\|(active|superseded|archived)\|", line)
            if status_match:
                status = status_match.group(1)
            slots.append({"slot": match.group(1), "value": match.group(2).strip(), "status": status})
            continue
        profile = re.match(r"^- ([\w:.\-]+) = (.+)$", line.strip())
        if profile:
            slots.append({"slot": profile.group(1), "value": profile.group(2).strip(), "status": "active"})
    return slots


def raw_locations(text: str) -> list[str]:
    values = re.findall(
        r"(?:I live in|I moved to|I now live in|I'm based in)\s+([A-Z][\w\- ]+)", text
    )
    cleaned = []
    for value in values:
        value = re.sub(r"\s+(?:last|this|next)\s+\w+$", "", value.strip())
        if value:
            cleaned.append(value)
    return cleaned


def raw_jobs(text: str) -> list[str]:
    values = re.findall(
        r"(?:I work as (?:a|an)|I now work as (?:a|an)|my new role is)\s+([^.,;\n]+)", text, re.I
    )
    return [re.split(r"\s+at\s+", value.strip())[0] for value in values]


def raw_employers(text: str) -> list[str]:
    values = re.findall(
        r"(?:work as [^.,;\n]+?\s+at|work at|work for)\s+([A-Z][\w\- ]+)", text
    )
    return [value.strip() for value in values]


def raw_foods(text: str) -> list[str]:
    return [value.strip() for value in re.findall(
        r"(?:I love|I now prefer|I prefer)\s+([^.,;\n]+?)\s+food", text, re.I
    )]


def raw_allergies(text: str) -> list[str]:
    return [value.strip() for value in re.findall(r"allergic to\s+([^.,;\n]+)", text, re.I)]


def raw_stacks(text: str) -> list[str]:
    return [value.strip() for value in re.findall(
        r"Project Atlas\s+(?:now\s+)?uses\s+([^.,;\n]+)", text, re.I
    )]


def raw_counts(text: str) -> int:
    return sum(int(value) for value in re.findall(r"completed\s+(\d+)\s+courses", text, re.I))


def raw_books(text: str) -> list[str]:
    return [value.strip() for value in re.findall(
        r"favorite book is\s+([^.,;\n]+)", text, re.I
    )]


def raw_hq(text: str) -> str:
    employer_values = raw_employers(text)
    employer = employer_values[-1] if employer_values else ""
    for match in re.finditer(r"([A-Z][\w\- ]+)\s+is headquartered in\s+([^.,;\n]+)", text):
        company, city = match.group(1).strip(), match.group(2).strip()
        if employer and (norm(employer) in norm(company) or norm(company) in norm(employer)):
            return city
    return ""


def value_of(slots: list[dict[str, str]], suffixes: Sequence[str], status: str | None = "active") -> str:
    for slot in slots:
        if status and slot["status"] != status:
            continue
        if any(slot["slot"].endswith(suffix) for suffix in suffixes):
            return slot["value"]
    return ""


def answer_question(question: Question, context: str) -> str:
    slots = parse_slots(context)
    if question.qtype == "current_location":
        value = value_of(slots, [".location"])
        if value:
            return value
        values = raw_locations(context)
        return values[-1] if values else ""
    if question.qtype == "history_location":
        value = value_of(slots, [".location"], status=None)
        superseded = [s["value"] for s in slots if s["slot"].endswith(".location") and s["status"] != "active"]
        if superseded:
            return superseded[0]
        values = raw_locations(context)
        return values[0] if values else ""
    if question.qtype == "current_job":
        value = value_of(slots, [".occupation", ".employer"])
        if value:
            return value
        values = raw_jobs(context)
        return values[-1] if values else ""
    if question.qtype == "employer":
        value = value_of(slots, [".employer"])
        if value:
            return value
        values = raw_employers(context)
        return values[-1] if values else ""
    if question.qtype == "preference":
        value = value_of(slots, [".preference:food", ".favorite:food"])
        if value:
            return value
        values = raw_foods(context)
        return values[-1] if values else ""
    if question.qtype == "book":
        value = value_of(slots, [".favorite:book"])
        if value:
            return value
        values = raw_books(context)
        return values[-1] if values else ""
    if question.qtype == "allergy":
        values = [s["value"] for s in slots if s["slot"].endswith(".allergy") and s["status"] == "active"]
        if values:
            return ",".join(values)
        values = raw_allergies(context)
        return values[0] if values else ""
    if question.qtype == "stack":
        value = value_of(slots, [".stack"])
        if value:
            return value
        values = raw_stacks(context)
        return values[-1] if values else ""
    if question.qtype == "count":
        values = [
            int(s["value"]) for s in slots
            if ".completed_count:" in s["slot"] and s["status"] == "active"
            and re.fullmatch(r"\d+", s["value"].strip())
        ]
        if values:
            return str(sum(values))
        return str(raw_counts(context))
    if question.qtype == "procedure":
        lower = context.lower()
        return "yes" if all(keyword in lower for keyword in PROCEDURE_KEYWORDS) else "no"
    if question.qtype == "multihop":
        employer = value_of(slots, [".employer"])
        if employer:
            key = norm_subject(employer)
            for slot in slots:
                if slot["slot"].lower().startswith(key + ".headquarters"):
                    return slot["value"]
        return raw_hq(context)
    return ""


def is_correct(question: Question, answer: str) -> bool:
    if question.qtype == "procedure":
        return answer == "yes"
    if question.qtype == "count":
        return answer.strip() == question.gold
    gold = norm(question.gold)
    got = norm(answer)
    return bool(gold and got and (gold in got or got in gold))


def is_stale(question: Question, answer: str) -> bool:
    if not question.stale:
        return False
    gold = norm(question.stale)
    got = norm(answer)
    return bool(gold and got and (gold in got or got in gold))


def evidence_hit(question: Question, context: str) -> bool:
    if question.qtype == "procedure":
        lower = context.lower()
        return all(keyword in lower for keyword in PROCEDURE_KEYWORDS)
    return norm(question.gold) in norm(context)


# --------------------------------------------------------------------------------------
# Baseline 上下文构造
# --------------------------------------------------------------------------------------
def user_messages(sessions: Sequence[Session], user: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for session in sessions:
        if session.user == user:
            messages.extend(session.messages)
    return messages


def full_context(messages: Sequence[dict[str, str]], budget: int | None) -> str:
    if budget is None:
        selected = list(messages)
    else:
        selected = []
        total = 0
        for message in reversed(messages):
            cost = estimate_tokens(message["content"])
            if total + cost > budget:
                break
            selected.append(message)
            total += cost
        selected.reverse()
    return "\n".join(f"{m['role']}: {m['content']}" for m in selected)


def naive_vector_context(
    query: str,
    messages: Sequence[dict[str, str]],
    embedder: HashingEmbedder,
    top_k: int = 4,
    window: int = 4,
) -> str:
    chunks = [
        messages[i:i + window]
        for i in range(0, len(messages), max(1, window // 2))
    ]
    if not chunks:
        return ""
    texts = ["\n".join(m["content"] for m in chunk) for chunk in chunks]
    query_vec = embedder.embed([query])[0]
    scored = [
        (cosine(query_vec, embedder.embed([text])[0]), text)
        for text in texts
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n".join(text for _, text in scored[:top_k])


# --------------------------------------------------------------------------------------
# 方法执行
# --------------------------------------------------------------------------------------
@dataclass
class Row:
    name: str
    accuracy: float
    tokens: float
    evidence: float
    stale: float
    latency_ms: float
    by_type: dict[str, float]
    note: str = ""


class NullExtractor:
    def extract(self, text: str, event_time: float, context: dict[str, Any] | None = None) -> list[Any]:
        return []


def base_config(**overrides: Any) -> MemoryConfig:
    config = MemoryConfig(
        max_turns_per_episode=2,
        top_k=8,
        token_budget=TOKEN_BUDGET,
        max_similar_edges=0,
        summary_min_episodes=3,
        duplicate_threshold=0.92,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def evaluate_contexts(
    name: str,
    questions: Sequence[Question],
    context_fn: Any,
    note: str = "",
) -> Row:
    correct = 0
    tokens = []
    evidence = 0
    stale = 0
    latencies = []
    by_type: dict[str, list[bool]] = {}
    for question in questions:
        started = time.perf_counter()
        context = context_fn(question)
        latencies.append((time.perf_counter() - started) * 1000)
        answer = answer_question(question, context)
        ok = is_correct(question, answer)
        correct += int(ok)
        by_type.setdefault(question.qtype, []).append(ok)
        tokens.append(estimate_tokens(context))
        evidence += int(evidence_hit(question, context))
        stale += int(is_stale(question, answer))
    total = max(len(questions), 1)
    return Row(
        name=name,
        accuracy=correct / total,
        tokens=statistics.mean(tokens) if tokens else 0.0,
        evidence=evidence / total,
        stale=stale / total,
        latency_ms=statistics.mean(latencies) if latencies else 0.0,
        by_type={k: sum(v) / len(v) for k, v in by_type.items()},
        note=note,
    )


def run_stack(
    name: str,
    sessions: Sequence[Session],
    questions: Sequence[Question],
    config: MemoryConfig,
    consolidate: bool = True,
    extractor: Any | None = None,
) -> tuple[Row, dict[str, Any]]:
    stack = MemoryStack(config=config, extractor=extractor)
    for session in sessions:
        stack.remember(session.messages, session.scope, event_time=session.event_time)
    if consolidate:
        for user in USERS:
            stack.consolidate(Scope(user_id=user["name"], agent_id="assistant"))

    def context_fn(question: Question) -> str:
        scope = Scope(user_id=question.user, agent_id="assistant")
        return stack.recall(question.text, scope, token_budget=TOKEN_BUDGET).text

    row = evaluate_contexts(name, questions, context_fn)
    stats = stack.stats(Scope(user_id=USERS[0]["name"], agent_id="assistant"))
    # 汇总三个用户的库规模
    totals = {"episode": 0, "fact": 0, "preference": 0, "procedure": 0, "summary": 0}
    statuses = {"active": 0, "superseded": 0}
    for user in USERS:
        data = stack.stats(Scope(user_id=user["name"], agent_id="assistant"))
        for kind, count in data["by_kind"].items():
            if kind in totals:
                totals[kind] += count
        for status, count in data["by_status"].items():
            if status in statuses:
                statuses[status] += count
    stack.close()
    return row, {"kinds": totals, "statuses": statuses, "sample_config": stats["config"]}


def main() -> None:
    sessions, questions = build_dataset()
    embedder = HashingEmbedder()
    rows: list[Row] = []
    notes: dict[str, Any] = {}

    rows.append(evaluate_contexts("No memory", questions, lambda q: "", "无外部记忆"))
    rows.append(
        evaluate_contexts(
            "Full context (all)",
            questions,
            lambda q: full_context(user_messages(sessions, q.user), None),
            "不限预算，作为上界",
        )
    )
    rows.append(
        evaluate_contexts(
            f"Full context ({TOKEN_BUDGET} tok)",
            questions,
            lambda q: full_context(user_messages(sessions, q.user), TOKEN_BUDGET),
            "同预算下只保留最近消息",
        )
    )
    rows.append(
        evaluate_contexts(
            "Naive vector Top-4",
            questions,
            lambda q: naive_vector_context(q.text, user_messages(sessions, q.user), embedder, top_k=4),
            "原始消息窗口 + 纯向量检索",
        )
    )

    stack_variants = [
        ("MemStack (full)", base_config(), True, None, "全部机制"),
        ("  w/o temporal", base_config(use_temporal=False), True, None, "关闭时间感知"),
        ("  w/o graph", base_config(use_graph=False), True, None, "关闭图扩展"),
        ("  w/o routing", base_config(use_routing=False), True, None, "关闭粒度路由"),
        ("  w/o rerank", base_config(use_rerank=False), True, None, "关闭启发式重排"),
        ("  w/o consolidation", base_config(), False, None, "不做离线巩固"),
        ("  episodes only", base_config(), True, NullExtractor(), "只存原始 episode"),
    ]
    for name, config, consolidate, extractor, note in stack_variants:
        row, stats = run_stack(name, sessions, questions, config, consolidate, extractor)
        row.note = note
        rows.append(row)
        if name == "MemStack (full)":
            notes["full"] = stats

    print("\n" + "=" * 104)
    print("MemStack 合成基准（同一 500-token 上下文预算；3 个用户 × 4 个会话 × 11 个问题）")
    print("=" * 104)
    print(f"{'方法':<25}{'准确率':>8}{'token/题':>11}{'证据命中':>10}{'陈旧回答':>10}{'时延ms':>9}  说明")
    print("-" * 104)
    for row in rows:
        print(
            f"{row.name:<25}{row.accuracy * 100:>7.1f}%{row.tokens:>11.1f}"
            f"{row.evidence * 100:>9.1f}%{row.stale * 100:>9.1f}%{row.latency_ms:>9.2f}  {row.note}"
        )

    full = next(row for row in rows if row.name == "MemStack (full)")
    print("\nMemStack 分题型准确率：")
    for qtype, value in sorted(full.by_type.items()):
        print(f"  {qtype:<20} {value * 100:5.1f}%")

    if "full" in notes:
        kinds = notes["full"]["kinds"]
        statuses = notes["full"]["statuses"]
        print(
            "\nMemStack 记忆库规模（3 用户汇总）："
            f"episodes={kinds['episode']} facts={kinds['fact']} preferences={kinds['preference']} "
            f"procedures={kinds['procedure']} summaries={kinds['summary']} "
            f"| active={statuses['active']} superseded={statuses['superseded']}"
        )

    print(
        "\n说明：这是机制级合成评测，reader 是规则化模拟，数值不可与论文原表或生产 benchmark 直接比较。\n"
        "关键看三件事：同预算下准确率、陈旧事实抑制、每題上下文 token。"
    )


if __name__ == "__main__":
    main()
