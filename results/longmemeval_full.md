# MemStack 公开数据集评测结果

> 评测对象是记忆检索层；不是 LLM-as-judge QA 分数。

## longmemeval_oracle

- 问题数：500；预算：800 token；top_k：8；耗时：139.9s

| 方法 | Unit R@k | All-Unit | Turn Strict | Turn Token | Ans Strict | Ans Token | MRR | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MemStack | 100.0% | 81.8% | 20.8% | 55.7% | 38.6% | 46.6% | 1.000 | 753.0 |
| MemStack-ep | 100.0% | 95.4% | 7.0% | 56.4% | 42.0% | 52.3% | 1.000 | 409.0 |
| Dense only | 100.0% | 35.2% | 72.2% | 69.2% | 30.8% | 50.5% | 1.000 | 791.9 |
| BM25 only | 100.0% | 35.2% | 72.6% | 69.4% | 33.2% | 52.2% | 1.000 | 792.2 |
| Full context | 100.0% | 35.2% | 75.0% | 70.0% | 36.0% | 55.5% | 1.000 | 793.2 |

### MemStack 分题型

| 类别 | n | Unit R@k | Turn Token | Ans Token |
|---|---:|---:|---:|---:|
| knowledge-update | 78 | 100.0% | 62.4% | 63.0% |
| multi-session | 133 | 100.0% | 47.1% | 22.4% |
| single-session-assistant | 56 | 100.0% | 66.9% | 78.7% |
| single-session-preference | 30 | 100.0% | 64.2% | 36.3% |
| single-session-user | 70 | 100.0% | 63.3% | 70.9% |
| temporal-reasoning | 133 | 100.0% | 49.7% | 37.3% |
