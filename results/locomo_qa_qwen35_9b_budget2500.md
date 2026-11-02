# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：300；上下文预算：2500 token
- Reader：`llm:qwen35-9b-local`；Judge：`judge:qwen35-9b-local`
- 模型基座：`qwen35-9b-local`
- 耗时：210.4s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memstack | 83.8% | 68.7% | 637.0 | 37.7% | 24.2% | 91.7% | 18.3% | 8.3% | 173 |

## 分题型准确率

### memstack

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 60 | 91.7% | 0.0% |
| multi-hop | 60 | 30.0% | 16.7% |
| open-domain | 60 | 5.0% | 3.9% |
| single-hop | 60 | 51.7% | 43.2% |
| temporal | 60 | 10.0% | 9.3% |
