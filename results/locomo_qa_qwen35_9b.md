# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：1986；上下文预算：600 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：3074.4s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 83.4% | 73.2% | 541.3 | 49.6% | 39.2% | 85.7% | 31.1% | 17.2% | 308 |
| memstack-mr | 81.3% | 70.7% | 533.5 | 48.1% | 37.9% | 83.2% | 29.6% | 16.0% | 301 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 446 | 85.7% | 0.0% |
| multi-hop | 282 | 38.7% | 21.8% |
| open-domain | 96 | 9.4% | 6.4% |
| single-hop | 841 | 54.8% | 45.0% |
| temporal | 321 | 7.5% | 10.2% |

### memstack-mr

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 446 | 83.2% | 0.0% |
| multi-hop | 282 | 35.1% | 19.4% |
| open-domain | 96 | 9.4% | 6.4% |
| single-hop | 841 | 54.0% | 43.3% |
| temporal | 321 | 6.9% | 9.6% |
