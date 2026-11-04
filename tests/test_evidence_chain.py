"""证据链检索（v4）单元测试：分解规则、分名额合并、防挤掉、端到端枚举召回。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import calendar
import sys
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memstack import (  # noqa: E402
    MemoryConfig,
    MemoryKind,
    MemoryStack,
    RecallHit,
    Scope,
)
from memstack.evidence_chain import (  # noqa: E402
    LLMQueryDecomposer,
    SubQuery,
    decompose_subqueries,
    merge_chain_hits,
    needs_chain,
    parse_subqueries_json,
    should_neighbor_expand,
)
from memstack.llm import SequenceLLM  # noqa: E402


def ts(year: int, month: int, day: int) -> float:
    return float(calendar.timegm(datetime(year, month, day).timetuple()))


def scope(session: str | None = None) -> Scope:
    return Scope(user_id="evan", agent_id="assistant", session_id=session)


def hit(memory_id: str, score: float) -> RecallHit:
    from memstack.types import MemoryItem

    item = MemoryItem(
        id=memory_id, kind=MemoryKind.EPISODE, text=f"memory {memory_id}",
        scope=scope(), data={},
    )
    return RecallHit(memory=item, score=score)


class DecomposeTest(unittest.TestCase):
    def test_enumeration(self) -> None:
        subs = decompose_subqueries("What hobbies does Evan pursue?", [], 2)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].reason, "enumeration")
        self.assertIn("evan", subs[0].text.lower())

    def test_quant(self) -> None:
        subs = decompose_subqueries("How many tournaments has Nate won?", [], 2)
        self.assertEqual([s.reason for s in subs], ["enumeration"])

    def test_verb_tail(self) -> None:
        # 只是"what did X VERB"的构型，不应误触枚举名词语
        subs = decompose_subqueries("What did Caroline research?", [], 2)
        self.assertEqual([s.reason for s in subs], ["enumeration"])
        self.assertIn("research", subs[0].text.lower())

    def test_bridge_priority_over_enumeration(self) -> None:
        # "team" 命中枚举词表，但 "lead of / employed by" 是强桥接词，必须优先 bridge
        subs = decompose_subqueries(
            "What company is the team lead of Project Atlas employed by?", [], 2,
        )
        self.assertTrue(subs)
        self.assertEqual(subs[0].reason, "bridge")
        self.assertIn("Project Atlas", subs[0].text)

    def test_plain_query_no_chain(self) -> None:
        self.assertFalse(needs_chain("Where does Nora live now?"))
        self.assertEqual(decompose_subqueries("Where does Nora live now?", [], 2), [])

    def test_unknown_type_no_subqueries(self) -> None:
        self.assertEqual(decompose_subqueries("", [], 2), [])


class MergeTest(unittest.TestCase):
    def test_keeps_first_round_evidence(self) -> None:
        first = [hit("a", 0.95), hit("b", 0.90), hit("c", 0.80), hit("d", 0.70)]
        sub = [hit("x", 0.85), hit("y", 0.60), hit("a", 0.99)]  # a 已在首轮，不算新证据
        merged = merge_chain_hits(first, [sub], top_k=6, slots_per_sub=2)
        ids = [h.memory.id for h in merged]
        self.assertEqual(len(ids), 6)
        # 首轮前 top_k - slots 名保序保留
        self.assertEqual(ids[:2], ["a", "b"])
        # 子查询补上新证据 x/y
        self.assertIn("x", ids)
        self.assertIn("y", ids)

    def test_sub_slots_limited(self) -> None:
        first = [hit(f"f{i}", 0.9 - i * 0.1) for i in range(4)]
        sub = [hit(f"s{i}", 0.7 - i * 0.05) for i in range(6)]
        merged = merge_chain_hits(first, [sub], top_k=8, slots_per_sub=2)
        sub_added = [h.memory.id for h in merged if h.memory.id.startswith("s")]
        self.assertEqual(len(sub_added), 2)

    def test_duplicate_sub_is_not_new_evidence(self) -> None:
        first = [hit("a", 0.9), hit("b", 0.8)]
        sub = [hit("a", 0.5), hit("b", 0.4)]
        merged = merge_chain_hits(first, [sub], top_k=4, slots_per_sub=2)
        self.assertEqual({h.memory.id for h in merged}, {"a", "b"})

    def test_neighbor_trigger(self) -> None:
        self.assertTrue(should_neighbor_expand(0.15, 0.30))
        self.assertFalse(should_neighbor_expand(0.45, 0.30))


class EndToEndEnumerationTest(unittest.TestCase):
    """多会话分散枚举：hop0 只能拿到部分取值的会话，证据链应补回长尾会话。"""

    def setUp(self) -> None:
        self.stack = MemoryStack()

    def tearDown(self) -> None:
        self.stack.close()

    def _write_hobby_sessions(self) -> None:
        # 目标会话：hobby 分散在 3 个 early 会话（发言带主语名，与 LoCoMo 口语体一致）
        self.stack.remember(
            "Evan: I started painting landscapes on weekends.", scope("s1"),
            event_time=ts(2026, 1, 3),
        )
        self.stack.remember(
            "Evan: I have been hiking in the mountains with friends.", scope("s2"),
            event_time=ts(2026, 1, 10),
        )
        self.stack.remember(
            "Evan: I read science fiction books at night.", scope("s3"),
            event_time=ts(2026, 1, 17),
        )
        # 干扰会话：居多、更新，会占据首轮 top 名额
        for idx in range(10):
            self.stack.remember(
                f"Just a status update about work item {idx} and nothing else.",
                scope(f"f{idx}"), event_time=ts(2026, 2, 1 + idx),
            )

    def test_chain_recovers_tail_evidence(self) -> None:
        self._write_hobby_sessions()
        query_scope = Scope(user_id="evan", agent_id="assistant")
        bundle = self.stack.recall(
            "What hobbies does Evan pursue?", query_scope, top_k=8, token_budget=1200,
        )
        hits_text = " | ".join(h.memory.id for h in bundle.items)
        # 首轮 + 枚举轮应覆盖 3 个 hobby 会话中的至少 2 个（budget 受限时>=2 即视为补齐长尾）
        hobby_hits = [
            mid for mid in ("s1", "s2", "s3")
            if mid in {h.memory.scope.session_id for h in bundle.items}
        ]
        self.assertGreaterEqual(len(hobby_hits), 2, f"hits={hits_text}")
        trace = bundle.trace.get("evidence_chain")
        self.assertIsNotNone(trace, "trace 应记录 evidence_chain")
        self.assertGreaterEqual(trace.get("subqueries", []).__len__(), 1)

    def test_chain_off_keeps_v2_path(self) -> None:
        self._write_hobby_sessions()
        legacy = MemoryStack(
            config=replace(MemoryConfig(), evidence_chain=False, multi_round=True),
            store=self.stack.store,
            embedder=self.stack.embedder,
            extractor=self.stack.extractor,
            summarizer=self.stack.summarizer,
            clock=self.stack.clock,
        )
        bundle = legacy.recall(
            "What hobbies does Evan pursue?",
            Scope(user_id="evan", agent_id="assistant"), top_k=8, token_budget=1200,
        )
        self.assertTrue(bundle.items)  # 旧路径仍可用
        self.assertNotIn("evidence_chain", bundle.trace)


class StatusPackingTest(unittest.TestCase):
    """生命周期感知打包：预算不足时 active（当前值）优先、历史值放尾部。"""

    def setUp(self) -> None:
        self.stack = MemoryStack()

    def tearDown(self) -> None:
        self.stack.close()

    def test_active_before_superseded_when_over_budget(self) -> None:
        scoped = Scope(user_id="nora", agent_id="assistant")
        self.stack.remember("I live in Lisbon.", scoped, event_time=ts(2026, 1, 1))
        self.stack.remember("I moved to Porto last month.", scoped, event_time=ts(2026, 9, 28))
        # 历史查询 + 极小预算：强制进入裁剪路径，active 的 Porto 必须先进上下文
        bundle = self.stack.recall(
            "Where did I live, before and now?", scoped,
            token_budget=60, include_history=True,
        )
        text = bundle.text.lower()
        porto_at = text.find("porto")
        lisbon_at = text.find("lisbon")
        self.assertGreaterEqual(porto_at, 0, f"当前值应优先进上下文: {bundle.text}")
        if lisbon_at >= 0:  # 预算可能放不下历史值；放进去了也必须在当前值之后
            self.assertGreater(lisbon_at, porto_at, f"当前值应排在历史值之前: {bundle.text}")

    def test_order_unchanged_within_budget(self) -> None:
        scoped = Scope(user_id="nora", agent_id="assistant")
        self.stack.remember("I live in Lisbon.", scoped, event_time=ts(2026, 1, 1))
        self.stack.remember("I moved to Porto last month.", scoped, event_time=ts(2026, 9, 28))
        bundle = self.stack.recall(
            "Where did I live before?", scoped, token_budget=2000, include_history=True,
        )
        self.assertIn("Lisbon", bundle.text)
        self.assertIn("Porto", bundle.text)


class LLMDecomposerTest(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        subs = parse_subqueries_json('[{"text": "Evan hobbies", "reason": "enumeration"}]')
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].reason, "enumeration")

    def test_parse_fenced_json(self) -> None:
        raw = 'Sure, here you go:\n```json\n[{"text": "X company", "reason": "bridge"}]\n```'
        subs = parse_subqueries_json(raw)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].reason, "bridge")

    def test_parse_invalid(self) -> None:
        self.assertEqual(parse_subqueries_json("not json at all"), [])
        self.assertEqual(parse_subqueries_json('[{"text": 123, "reason": "enumeration"}]'), [])
        self.assertEqual(parse_subqueries_json('[{"text": "ok", "reason": "unknown"}]'), [])
        self.assertEqual(parse_subqueries_json('{"text": "x", "reason": "bridge"}'), [])

    def test_llm_decompose_ok(self) -> None:
        dec = LLMQueryDecomposer(SequenceLLM([
            '[{"text": "Evan hobbies pursue", "reason": "enumeration"}]',
        ]))
        subs = dec.decompose("What hobbies does Evan pursue?", [], 2)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].text, "Evan hobbies pursue")
        self.assertEqual(dec.calls, 1)
        self.assertFalse(dec.last_fallback)

    def test_llm_decompose_falls_back_to_rules(self) -> None:
        dec = LLMQueryDecomposer(SequenceLLM(["garbage output"]))
        subs = dec.decompose("What hobbies does Evan pursue?", [], 2)
        self.assertTrue(dec.last_fallback)
        self.assertEqual(dec.fallbacks, 1)
        self.assertTrue(subs)  # 回退到规则分解，链仍可用
        self.assertEqual(subs[0].reason, "enumeration")

    def test_llm_exception_falls_back(self) -> None:
        class Boom:
            def complete(self, prompt, system=None):
                raise RuntimeError("boom")

        dec = LLMQueryDecomposer(Boom())
        subs = dec.decompose("How many tournaments has Nate won?", [], 2)
        self.assertTrue(dec.last_fallback)
        self.assertTrue(subs)

    def test_stack_integration_uses_llm_decomposer(self) -> None:
        stack = MemoryStack(
            query_decomposer=LLMQueryDecomposer(SequenceLLM([
                '[{"text": "Evan hobbies pursue", "reason": "enumeration"}]',
            ])),
        )
        scoped = Scope(user_id="evan", agent_id="assistant")
        for idx, hobby in enumerate(["painting", "hiking", "reading"]):
            stack.remember(f"Evan: I enjoy {hobby}.", scoped, event_time=ts(2026, 1, idx + 1))
        bundle = stack.recall("What hobbies does Evan pursue?", scoped, token_budget=600)
        self.assertTrue(bundle.items)
        trace = bundle.trace.get("evidence_chain") or {}
        self.assertEqual(trace.get("decomposer"), "llm")
        stack.close()


class ConfigSwapTest(unittest.TestCase):
    def test_validate_accepts_new_fields(self) -> None:
        MemoryConfig(evidence_chain=True).validate()
        MemoryConfig(evidence_chain=False).validate()


if __name__ == "__main__":
    unittest.main()