# LongMemEval QA 层评测（LLM reader + judge）

- 问题数：200；上下文预算：800 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：494.4s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 100.0% | 79.0% | 758.3 | 37.0% | 37.0% | — | 31.9% | 24.5% | 457 |
| memstack-ep | 100.0% | 96.5% | 404.8 | 44.0% | 44.0% | — | 39.7% | 29.0% | 332 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 34 | 52.9% | 44.8% |
| multi-session | 34 | 2.9% | 3.3% |
| single-session-assistant | 34 | 76.5% | 66.7% |
| single-session-preference | 30 | 13.3% | 4.2% |
| single-session-user | 34 | 58.8% | 51.7% |
| temporal-reasoning | 34 | 14.7% | 17.6% |

### memstack-ep

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 34 | 67.6% | 62.6% |
| multi-session | 34 | 8.8% | 5.4% |
| single-session-assistant | 34 | 64.7% | 54.9% |
| single-session-preference | 30 | 16.7% | 5.2% |
| single-session-user | 34 | 79.4% | 74.0% |
| temporal-reasoning | 34 | 23.5% | 32.3% |
