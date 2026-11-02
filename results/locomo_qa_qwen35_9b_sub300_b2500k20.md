# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：300；上下文预算：2500 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：446.9s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 92.6% | 81.1% | 1628.0 | 45.0% | 34.6% | 86.7% | 24.4% | 9.6% | 431 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 60 | 86.7% | 0.0% |
| multi-hop | 60 | 48.3% | 24.9% |
| open-domain | 60 | 8.3% | 6.3% |
| single-hop | 60 | 68.3% | 53.3% |
| temporal | 60 | 13.3% | 13.0% |
