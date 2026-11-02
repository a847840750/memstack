# LongMemEval QA 层评测（LLM reader + judge）

- 问题数：200；上下文预算：6500 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：508.2s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 100.0% | 71.0% | 3733.0 | 48.5% | 48.5% | — | 42.7% | 32.0% | 293 |
| memstack | 100.0% | 99.0% | 2051.5 | 46.5% | 46.5% | — | 42.9% | 33.0% | 324 |

## 分题型准确率

### full

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 34 | 82.4% | 69.2% |
| multi-session | 34 | 0.0% | 2.1% |
| single-session-assistant | 34 | 94.1% | 83.2% |
| single-session-preference | 30 | 13.3% | 3.1% |
| single-session-user | 34 | 88.2% | 81.8% |
| temporal-reasoning | 34 | 8.8% | 12.0% |

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 34 | 67.6% | 58.9% |
| multi-session | 34 | 8.8% | 6.8% |
| single-session-assistant | 34 | 82.4% | 73.0% |
| single-session-preference | 30 | 6.7% | 3.4% |
| single-session-user | 34 | 82.4% | 75.0% |
| temporal-reasoning | 34 | 26.5% | 35.8% |
