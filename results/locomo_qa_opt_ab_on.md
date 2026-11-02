# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：300；上下文预算：600 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：213.0s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 82.8% | 69.4% | 562.8 | 46.7% | 38.8% | 78.3% | 29.7% | 14.6% | 166 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 60 | 78.3% | 0.0% |
| multi-hop | 60 | 36.7% | 19.6% |
| open-domain | 60 | 8.3% | 4.6% |
| single-hop | 60 | 51.7% | 39.8% |
| temporal | 60 | 58.3% | 55.0% |
