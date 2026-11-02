# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：300；上下文预算：600 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：262.3s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 82.8% | 69.4% | 562.8 | 52.7% | 46.2% | 78.3% | 33.2% | 16.7% | 220 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 60 | 78.3% | 0.0% |
| multi-hop | 60 | 36.7% | 18.7% |
| open-domain | 60 | 35.0% | 19.6% |
| single-hop | 60 | 53.3% | 40.1% |
| temporal | 60 | 60.0% | 54.5% |
