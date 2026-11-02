# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：1986；上下文预算：600 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：2726.1s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 84.2% | 74.1% | 553.9 | 54.3% | 47.2% | 78.7% | 37.4% | 20.3% | 219 |
| memstack-mr | 82.7% | 72.5% | 547.3 | 52.5% | 45.8% | 75.6% | 36.5% | 19.8% | 171 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 446 | 78.7% | 0.0% |
| multi-hop | 282 | 36.2% | 19.5% |
| open-domain | 96 | 13.5% | 10.3% |
| single-hop | 841 | 54.7% | 45.2% |
| temporal | 321 | 47.4% | 40.7% |

### memstack-mr

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 446 | 75.6% | 0.0% |
| multi-hop | 282 | 35.8% | 19.7% |
| open-domain | 96 | 11.5% | 9.9% |
| single-hop | 841 | 52.8% | 43.5% |
| temporal | 321 | 46.7% | 41.0% |
