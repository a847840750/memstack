# LoCoMo QA 层评测（LLM reader + judge）

- 问题数：50；上下文预算：600 token
- Reader：`hf:Qwen/Qwen3.5-0.8B`；Judge：`judge:hf:Qwen/Qwen3.5-0.8B`
- 耗时：613.2s

| 方法 | Unit R@k | All-Unit | Tokens | 总体准确率 | 可答题准确率 | 拒答准确率 | F1 | EM | QA 延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0 | 91.7% | 83.3% | 576.9 | 22.0% | 20.0% | 30.0% | 13.7% | 5.0% | 1971 |

## 分题型准确率

### mem0

| 类别 | n | 准确率 | F1 |
|---|---:|---:|---:|
| adversarial | 10 | 30.0% | 0.0% |
| multi-hop | 10 | 20.0% | 19.0% |
| open-domain | 10 | 30.0% | 14.6% |
| single-hop | 10 | 30.0% | 11.8% |
| temporal | 10 | 0.0% | 9.5% |
