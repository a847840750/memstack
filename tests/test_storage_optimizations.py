"""存储与检索优化测试：float32 BLOB 往返、revision 失效、审计查询、缓存正确性。"""

import unittest

from memstack import MemoryStack, Scope


class TestStorageOptimizations(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = MemoryStack()
        self.scope = Scope(user_id="nora", agent_id="a")

    def test_embedding_blob_roundtrip(self) -> None:
        """向量经 float32 BLOB 存储往返后，与同一文本重新嵌入的结果在 1e-5 内一致。"""
        self.stack.remember([{"role": "user", "content": "the quick brown fox"}], self.scope)
        episode = self.stack.store.find(self.scope, kinds=["episode"])[0]
        self.assertIsNotNone(episode.embedding)
        expected = self.stack.embedder.embed([episode.text])[0]
        self.assertEqual(len(episode.embedding), len(expected))
        for a, b in zip(expected, episode.embedding):
            self.assertAlmostEqual(a, b, places=5)

    def test_legacy_json_embedding_still_readable(self) -> None:
        """旧库用 JSON 字符串存向量，升级后必须仍然可读。"""
        import json
        self.stack.remember([{"role": "user", "content": "legacy test row"}], self.scope)
        with self.stack.store._lock, self.stack.store._conn:
            self.stack.store._conn.execute(
                "UPDATE memories SET embedding=? WHERE tenant_id=? AND user_id=?",
                (json.dumps([0.1, 0.2, 0.3]), self.scope.tenant_id, self.scope.user_id),
            )
        items = self.stack.store.find(self.scope, kinds=["episode"])
        self.assertEqual(items[0].embedding, [0.1, 0.2, 0.3])

    def test_revision_bumps_on_write_not_on_read(self) -> None:
        base = self.stack.store.revision()
        self.stack.remember([{"role": "user", "content": "revision write 1"}], self.scope)
        after_write = self.stack.store.revision()
        self.assertGreater(after_write, base)
        self.stack.recall("revision write", self.scope)  # 读取（含 access bump）不应改变版本
        self.assertEqual(self.stack.store.revision(), after_write)

    def test_audit_roundtrip(self) -> None:
        self.stack.remember([{"role": "user", "content": "audit me"}], self.scope)
        logs = self.stack.store.list_audit(self.scope, limit=10)
        self.assertTrue(logs)
        self.assertTrue(any(log["op"].startswith("remember") for log in logs))
        filtered = self.stack.store.list_audit(self.scope, op="remember.episode", limit=10)
        self.assertTrue(all(log["op"] == "remember.episode" for log in filtered))

    def test_cache_invalidates_after_write(self) -> None:
        """写完新事实后，同 query 必须能检索到新内容（缓存不能被旧索引污染）。"""
        self.stack.remember(
            [{"role": "user", "content": "The budget is 3000 dollars."}], self.scope,
        )
        first = self.stack.recall("what is the budget", self.scope)
        self.assertIn("3000", first.text)
        # 同一 query 再查两次，让 BM25/PPR 缓存建立
        self.stack.recall("what is the budget", self.scope)
        self.stack.remember(
            [{"role": "user", "content": "The budget is now 9000 dollars."}], self.scope,
        )
        after = self.stack.recall("current budget", self.scope)
        self.assertIn("9000", after.text)

    def test_mmr_numpy_and_loop_agree(self) -> None:
        """numpy 路径与纯 Python 回退路径的 MMR 选择必须一致（结果等价性）。"""
        for i in range(6):
            self.stack.remember(
                [{"role": "user", "content": f"topic number {i}: unique detail {i}"}],
                self.scope,
            )
        items = self.stack.store.find(
            self.scope, kinds=["fact", "episode"], statuses=None, limit=100,
        )
        query = "topic number"
        qvec = self.stack.embedder.embed([query])[0]
        scores = {item.id: float(i % 5) / 5 for i, item in enumerate(items)}
        retriever = self.stack.retriever
        numpy_choice = retriever._select_mmr(items, scores, qvec, top_k=4)  # noqa: SLF001
        loop_choice = retriever._select_mmr_loop(sorted(items, key=lambda i: scores[i.id], reverse=True), scores, qvec, top_k=4)  # noqa: SLF001
        self.assertEqual(
            [m.id for m in numpy_choice], [m.id for m in loop_choice],
        )


if __name__ == "__main__":
    unittest.main()