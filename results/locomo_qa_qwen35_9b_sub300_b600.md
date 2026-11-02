# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：300；上下文预算：600 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：361.2s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 82.2% | 68.0% | 560.0 | 38.3% | 25.8% | 88.3% | 18.2% | 8.3% | 390 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 60 | 88.3% | 0.0% |
| multi-hop | 60 | 35.0% | 17.2% |
| open-domain | 60 | 5.0% | 3.4% |
| single-hop | 60 | 53.3% | 42.5% |
| temporal | 60 | 10.0% | 9.6% |
