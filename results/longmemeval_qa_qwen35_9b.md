# LongMemEval QA 层评测（LLM reader + judge）

- 问题数：500；上下文预算：800 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：3041.7s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 100.0% | 73.8% | 765.6 | 32.4% | 32.4% | — | 23.1% | 17.2% | 769 |
| memstack-mr | 100.0% | 74.0% | 768.1 | 31.2% | 31.2% | — | 21.9% | 16.2% | 550 |
| memstack-ep | 100.0% | 96.8% | 391.7 | 45.0% | 45.0% | — | 34.0% | 23.2% | 460 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 78 | 50.0% | 36.0% |
| multi-session | 133 | 14.3% | 4.7% |
| single-session-assistant | 56 | 78.6% | 65.1% |
| single-session-preference | 30 | 10.0% | 2.7% |
| single-session-user | 70 | 61.4% | 49.1% |
| temporal-reasoning | 133 | 10.5% | 7.1% |

### memstack-mr

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 78 | 46.2% | 32.0% |
| multi-session | 133 | 13.5% | 3.7% |
| single-session-assistant | 56 | 78.6% | 65.4% |
| single-session-preference | 30 | 10.0% | 2.7% |
| single-session-user | 70 | 60.0% | 48.3% |
| temporal-reasoning | 133 | 9.8% | 6.4% |

### memstack-ep

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 78 | 69.2% | 53.1% |
| multi-session | 133 | 33.1% | 20.6% |
| single-session-assistant | 56 | 66.1% | 53.4% |
| single-session-preference | 30 | 10.0% | 4.5% |
| single-session-user | 70 | 81.4% | 67.0% |
| temporal-reasoning | 133 | 22.6% | 17.4% |
