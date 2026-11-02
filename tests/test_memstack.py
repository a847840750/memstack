"""MemStack 单元测试：只依赖标准库 unittest。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import calendar
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memstack import (  # noqa: E402
    MemoryConfig,
    MemoryKind,
    MemoryStack,
    MemoryStatus,
    Scope,
    SQLiteStore,
)
from memstack.extraction import RuleExtractor, normalize_relative_time  # noqa: E402


def ts(year: int, month: int, day: int) -> float:
    return float(calendar.timegm(datetime(year, month, day).timetuple()))


def scope(user: str = "u1", session: str | None = "s1") -> Scope:
    return Scope(user_id=user, agent_id="assistant", session_id=session)


class ExtractionTest(unittest.TestCase):
    def test_core_types(self) -> None:
        text = (
            "My name is Nora. I live in Lisbon. I work as a data analyst at Northwind Labs. "
            "I love Peruvian food. I am allergic to peanuts. I completed 3 courses on Coursera. "
            "To submit an expense report, first open the finance portal, then fill in the report form, "
            "then attach receipts, and finally submit it for approval."
        )
        items = RuleExtractor().extract(text, ts(2026, 4, 28))
        kinds = {item.kind for item in items}
        self.assertIn(MemoryKind.FACT, kinds)
        self.assertIn(MemoryKind.PREFERENCE, kinds)
        self.assertIn(MemoryKind.PROCEDURE, kinds)
        predicates = {item.data.get("predicate_key") for item in items}
        self.assertIn("location", predicates)
        self.assertIn("occupation", predicates)
        self.assertIn("employer", predicates)
        self.assertIn("allergy", predicates)
        self.assertIn("completed_count:courses:coursera", predicates)

    def test_chinese(self) -> None:
        items = RuleExtractor().extract("我叫李明，我住在上海，我对花生过敏，我喜欢川菜。", ts(2026, 1, 1))
        predicates = {item.data.get("predicate_key") for item in items}
        self.assertIn("name", predicates)
        self.assertIn("location", predicates)
        self.assertIn("allergy", predicates)


class RelativeTimeTest(unittest.TestCase):
    """写入侧相对时间归一化：时序问答依赖上下文里存在绝对日期。"""

    def test_patterns(self) -> None:
        ref = ts(2026, 5, 9)  # 周六
        self.assertIn("8 May 2026", normalize_relative_time("I went to the group yesterday.", ref))
        self.assertIn("2 May 2026", normalize_relative_time("I signed up last Saturday.", ref))
        self.assertIn("15 May 2026", normalize_relative_time("The concert is next Friday.", ref))
        self.assertIn("June 2026", normalize_relative_time("I'm going camping next month.", ref))
        self.assertIn("2023", normalize_relative_time("I ran a charity race three years ago.", ref))

    def test_normalized_text_reaches_context(self) -> None:
        stack = MemoryStack()
        try:
            stack.remember(
                "I went to the LGBTQ support group yesterday.",
                scope(), event_time=ts(2026, 5, 9),
            )
            bundle = stack.recall(
                "When did I go to the support group?", Scope(user_id="u1", agent_id="assistant"),
            )
            self.assertIn("8 May 2026", bundle.text)
        finally:
            stack.close()


class MemoryStackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = MemoryStack()

    def tearDown(self) -> None:
        self.stack.close()

    def test_basic_recall(self) -> None:
        self.stack.remember(
            "My name is Nora. I live in Lisbon. I work as a data analyst at Northwind Labs.",
            scope(), event_time=ts(2026, 1, 1),
        )
        bundle = self.stack.recall("Where does Nora live?", Scope(user_id="u1", agent_id="assistant"))
        self.assertIn("Lisbon", bundle.text)
        self.assertIn("user.location", bundle.text)
        self.assertGreater(bundle.confidence, 0.4)
        self.assertGreater(bundle.token_estimate, 0)

    def test_temporal_supersede(self) -> None:
        query_scope = Scope(user_id="u1", agent_id="assistant")
        self.stack.remember("I live in Lisbon.", scope(), event_time=ts(2026, 4, 28))
        self.stack.remember("I moved to Porto last month.", scope(session="s2"), event_time=ts(2026, 10, 28))

        current = self.stack.recall("Where do I live now?", query_scope)
        history = self.stack.recall("Where did I live before?", query_scope)
        self.assertIn("Porto", current.text)
        self.assertIn("Lisbon", history.text)
        self.assertTrue(history.trace["history_mode"])

        active = self.stack.store.find(query_scope, kinds=[MemoryKind.FACT], statuses=[MemoryStatus.ACTIVE])
        superseded = self.stack.store.find(query_scope, kinds=[MemoryKind.FACT], statuses=[MemoryStatus.SUPERSEDED])
        self.assertTrue(any(item.data.get("predicate_key") == "location" for item in active))
        self.assertTrue(any(item.data.get("predicate_key") == "location" for item in superseded))

    def test_preference_update(self) -> None:
        query_scope = Scope(user_id="u1", agent_id="assistant")
        self.stack.remember("I love Peruvian food.", scope(), event_time=ts(2026, 4, 1))
        self.stack.remember("I now prefer Brazilian food.", scope(session="s2"), event_time=ts(2026, 5, 1))
        current = self.stack.recall("What food do I prefer now?", query_scope)
        history = self.stack.recall("What food did I prefer before?", query_scope)
        self.assertIn("Brazilian", current.text)
        self.assertIn("Peruvian", history.text)

    def test_duplicate_merge(self) -> None:
        self.stack.remember("I am allergic to peanuts.", scope(), event_time=ts(2026, 4, 1))
        result = self.stack.remember("I am allergic to peanuts.", scope(session="s2"), event_time=ts(2026, 5, 1))
        self.assertTrue(any(d.action.value == "MERGE" for d in result.decisions))
        query_scope = Scope(user_id="u1", agent_id="assistant")
        allergies = [
            item for item in self.stack.store.find(query_scope, kinds=[MemoryKind.FACT])
            if item.data.get("predicate_key") == "allergy" and item.status == MemoryStatus.ACTIVE
        ]
        self.assertEqual(len(allergies), 1)
        self.assertEqual(len(allergies[0].source_ids), 2)

    def test_multi_value_counts(self) -> None:
        query_scope = Scope(user_id="u1", agent_id="assistant")
        self.stack.remember("I completed 3 courses on Coursera.", scope(), event_time=ts(2026, 1, 1))
        self.stack.remember("I completed 2 courses on edX.", scope(session="s2"), event_time=ts(2026, 2, 1))
        self.stack.consolidate(query_scope)
        bundle = self.stack.recall("How many online courses have I completed in total?", query_scope)
        self.assertIn("value=3", bundle.text)
        self.assertIn("value=2", bundle.text)
        self.assertEqual(bundle.trace["history_mode"], False)

    def test_scope_isolation(self) -> None:
        self.stack.remember("I live in Lisbon.", scope("alice"), event_time=ts(2026, 1, 1))
        self.stack.remember("I live in Osaka.", scope("bob"), event_time=ts(2026, 1, 1))
        alice = self.stack.recall("Where do I live?", Scope(user_id="alice", agent_id="assistant"))
        bob = self.stack.recall("Where do I live?", Scope(user_id="bob", agent_id="assistant"))
        self.assertIn("Lisbon", alice.text)
        self.assertNotIn("Osaka", alice.text)
        self.assertIn("Osaka", bob.text)
        self.assertNotIn("Lisbon", bob.text)

    def test_token_budget(self) -> None:
        self.stack.remember(
            " ".join([f"I live in Lisbon and I work as a data analyst. Filler sentence {i}." for i in range(20)]),
            scope(), event_time=ts(2026, 1, 1),
        )
        bundle = self.stack.recall("Where do I live?", Scope(user_id="u1", agent_id="assistant"), token_budget=200)
        self.assertLessEqual(bundle.token_estimate, 240)
        self.assertTrue(bundle.items)

    def test_forget_soft_and_hard(self) -> None:
        result = self.stack.remember("I live in Lisbon.", scope(), event_time=ts(2026, 1, 1))
        location_id = next(d.memory_id for d in result.decisions if d.kind == "fact" and "Lisbon" in d.text)
        query_scope = Scope(user_id="u1", agent_id="assistant")
        self.stack.forget([location_id], query_scope, hard=False)
        soft = self.stack.recall("Where do I live?", query_scope)
        self.assertNotIn("value=Lisbon", soft.text)
        self.assertIsNotNone(self.stack.store.get(location_id))
        self.stack.forget([location_id], query_scope, hard=True)
        self.assertIsNone(self.stack.store.get(location_id))

    def test_forget_episode_cascades(self) -> None:
        result = self.stack.remember("I live in Lisbon.", scope(), event_time=ts(2026, 1, 1))
        episode_id = result.episode_ids[0]
        query_scope = Scope(user_id="u1", agent_id="assistant")
        self.stack.forget([episode_id], query_scope, hard=False)
        bundle = self.stack.recall("Where do I live?", query_scope)
        self.assertNotIn("Lisbon", bundle.text)
        facts = [
            item for item in self.stack.store.find(query_scope, kinds=[MemoryKind.FACT])
            if item.data.get("predicate_key") == "location"
        ]
        self.assertTrue(all(item.status == MemoryStatus.DELETED for item in facts))

    def test_out_of_order_historical(self) -> None:
        query_scope = Scope(user_id="u1", agent_id="assistant")
        self.stack.remember("I moved to Porto last month.", scope(), event_time=ts(2026, 10, 28))
        self.stack.remember("I live in Lisbon.", scope(session="s2"), event_time=ts(2026, 1, 1))
        current = self.stack.recall("Where do I live now?", query_scope)
        self.assertIn("Porto", current.text)
        self.assertNotIn("value=Lisbon", current.text)

    def test_profile_block(self) -> None:
        self.stack.remember(
            "My name is Nora. I live in Lisbon. I work as a data analyst at Northwind Labs. I am allergic to peanuts.",
            scope(), event_time=ts(2026, 1, 1),
        )
        query_scope = Scope(user_id="u1", agent_id="assistant")
        report = self.stack.consolidate(query_scope)
        self.assertGreater(report["profile_chars"], 0)
        block = self.stack.get_block("user_profile", query_scope)
        self.assertIsNotNone(block)
        self.assertIn("Lisbon", block.value)
        self.assertIn("peanuts", block.value)

    def test_chinese_recall(self) -> None:
        self.stack.remember("我叫李明，我住在上海，我对花生过敏，我喜欢川菜。", scope(), event_time=ts(2026, 1, 1))
        bundle = self.stack.recall("李明住在哪里？", Scope(user_id="u1", agent_id="assistant"))
        self.assertIn("上海", bundle.text)
        allergy = self.stack.recall("我对什么过敏？", Scope(user_id="u1", agent_id="assistant"))
        self.assertIn("花生", allergy.text)

    def test_tool_dispatch(self) -> None:
        query_scope = Scope(user_id="u1", agent_id="assistant")
        tools = self.stack.as_tools()
        names = {tool["function"]["name"] for tool in tools}
        self.assertEqual(
            names,
            {"memory_search", "memory_add", "memory_forget", "memory_profile", "memory_consolidate"},
        )
        added = self.stack.dispatch_tool(
            "memory_add",
            {"messages": "I live in Lisbon.", "scope": query_scope.to_dict(), "event_time": "2026-01-01T00:00:00"},
        )
        self.assertTrue(added["ok"])
        found = self.stack.dispatch_tool(
            "memory_search", {"query": "Where do I live?", "scope": query_scope.to_dict()}
        )
        self.assertTrue(found["ok"])
        self.assertIn("Lisbon", found["result"]["context"])

    def test_export_import(self) -> None:
        self.stack.remember("I live in Lisbon.", scope(), event_time=ts(2026, 1, 1))
        query_scope = Scope(user_id="u1", agent_id="assistant")
        payload = self.stack.export(query_scope)
        self.assertGreater(len(payload["memories"]), 0)

        other = MemoryStack()
        try:
            imported = other.import_data(payload)
            self.assertGreater(imported, 0)
            bundle = other.recall("Where do I live?", query_scope)
            self.assertIn("Lisbon", bundle.text)
        finally:
            other.close()


class PersistenceTest(unittest.TestCase):
    def test_sqlite_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.db")
            store = SQLiteStore(path)
            stack = MemoryStack(store=store)
            stack.remember("I live in Lisbon.", scope(), event_time=ts(2026, 1, 1))
            stack.close()

            reopened = MemoryStack(store=SQLiteStore(path))
            try:
                bundle = reopened.recall("Where do I live?", Scope(user_id="u1", agent_id="assistant"))
                self.assertIn("Lisbon", bundle.text)
            finally:
                reopened.close()

    def test_summary_created(self) -> None:
        config = MemoryConfig(max_turns_per_episode=1, summary_min_episodes=2, max_similar_edges=0)
        stack = MemoryStack(config=config)
        try:
            messages = [
                {"role": "user", "content": "I live in Lisbon."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "I work as a data analyst."},
                {"role": "assistant", "content": "Noted."},
            ]
            stack.remember(messages, scope(), event_time=ts(2026, 1, 1))
            query_scope = Scope(user_id="u1", agent_id="assistant")
            report = stack.consolidate(query_scope)
            self.assertGreaterEqual(report["summaries_created"], 1)
            summaries = stack.store.find(query_scope, kinds=[MemoryKind.SUMMARY])
            self.assertGreaterEqual(len(summaries), 1)
        finally:
            stack.close()


if __name__ == "__main__":
    unittest.main()
