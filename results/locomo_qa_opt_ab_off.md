# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：300；上下文预算：600 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：646.5s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 81.8% | 68.4% | 562.7 | 46.7% | 38.8% | 78.3% | 26.2% | 10.4% | 761 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 60 | 78.3% | 0.0% |
| multi-hop | 60 | 38.3% | 19.1% |
| open-domain | 60 | 8.3% | 4.7% |
| single-hop | 60 | 55.0% | 43.0% |
| temporal | 60 | 53.3% | 38.1% |
