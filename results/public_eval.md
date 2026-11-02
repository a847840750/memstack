# MemStack 公开数据集评测结果

> 评测对象是记忆检索层；不是 LLM-as-judge QA 分数。

## locomo

- 问题数：400；预算：800 token；top_k：8；耗时：354.9s

| 方法 | Unit R@k | All-Unit | Turn Strict | Turn Token | Ans Strict | Ans Token | MRR | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MemStack | 83.4% | 67.5% | 54.9% | 67.0% | 8.6% | 20.4% | 0.572 | 572.4 |
| MemStack-ep | 86.6% | 72.8% | 59.9% | 73.1% | 9.3% | 23.3% | 0.643 | 725.0 |
| Dense only | 22.2% | 15.9% | 20.7% | 48.9% | 5.3% | 13.8% | 0.222 | 641.5 |
| BM25 only | 58.2% | 46.6% | 54.9% | 69.3% | 9.1% | 19.9% | 0.582 | 658.1 |
| Full context | 2.3% | 1.8% | 2.3% | 35.3% | 1.3% | 6.8% | 0.023 | 518.5 |

### MemStack 分题型

| 类别 | n | Unit R@k | Turn Token | Ans Token |
|---|---:|---:|---:|---:|
| adversarial | 80 | 92.5% | 79.1% | 0.0% |
| multi-hop | 79 | 79.7% | 50.8% | 24.6% |
| open-domain | 78 | 61.5% | 45.6% | 14.5% |
| single-hop | 80 | 92.5% | 73.7% | 52.3% |
| temporal | 80 | 90.0% | 85.1% | 10.5% |

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
