# LongMemEval QA 层评测（LLM reader + judge）

- 问题数：200；上下文预算：6500 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：667.6s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 100.0% | 71.0% | 3733.0 | 51.0% | 51.0% | — | 43.0% | 31.0% | 282 |
| memstack | 100.0% | 100.0% | 2142.1 | 44.5% | 44.5% | — | 37.6% | 28.0% | 317 |

## 分题型准确率

### full

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 34 | 79.4% | 64.7% |
| multi-session | 34 | 5.9% | 4.9% |
| single-session-assistant | 34 | 94.1% | 79.7% |
| single-session-preference | 30 | 10.0% | 2.7% |
| single-session-user | 34 | 91.2% | 80.8% |
| temporal-reasoning | 34 | 20.6% | 20.7% |

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 34 | 61.8% | 51.6% |
| multi-session | 34 | 8.8% | 4.4% |
| single-session-assistant | 34 | 85.3% | 71.5% |
| single-session-preference | 30 | 10.0% | 3.0% |
| single-session-user | 34 | 79.4% | 71.6% |
| temporal-reasoning | 34 | 17.6% | 19.6% |
