# LongMemEval QA 层评测（LLM reader + judge）

- 问题数：500；上下文预算：800 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：1334.5s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 100.0% | 85.6% | 762.1 | 43.4% | 43.4% | — | 32.9% | 24.4% | 302 |
| memstack-ep | 100.0% | 96.4% | 430.9 | 46.8% | 46.8% | — | 36.3% | 25.0% | 254 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 78 | 69.2% | 52.0% |
| multi-session | 133 | 21.8% | 11.9% |
| single-session-assistant | 56 | 73.2% | 62.1% |
| single-session-preference | 30 | 6.7% | 3.6% |
| single-session-user | 70 | 87.1% | 73.0% |
| temporal-reasoning | 133 | 22.6% | 15.8% |

### memstack-ep

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| knowledge-update | 78 | 71.8% | 57.2% |
| multi-session | 133 | 31.6% | 22.8% |
| single-session-assistant | 56 | 66.1% | 52.8% |
| single-session-preference | 30 | 16.7% | 5.2% |
| single-session-user | 70 | 84.3% | 71.6% |
| temporal-reasoning | 133 | 26.3% | 19.0% |
