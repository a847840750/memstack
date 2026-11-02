# MemStack 公开数据集评测

> **版本说明**：本文件保留 v1 评测（LoCoMo 400 题 / LongMemEval oracle 500 题）。
> v2 已加入多轮补证、证据回填、NDCG/延迟/bootstrap CI 与 SOTA 风格基线，
> 最新综合结果与分数对照见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

> 评测日期：2026-10-28
> 环境：Windows / Python 3.12 / CPU / 纯标准库 + SQLite
> 代码：[public_eval.py](public_eval.py)
> 原始结果：[results/locomo_400.json](results/locomo_400.json)、[results/longmemeval_full.json](results/longmemeval_full.json)
> 说明：本评测衡量**记忆检索层**，不是 LLM-as-judge QA 分数；所有方法使用同一 token 预算与同一指标。

---

## 0. 30 秒版

| 数据集 | 规模 | 关键结果 |
|---|---|---|
| **LoCoMo** | 1986 QA，分层抽取 400 题（397 题有证据） | MemStack Unit Recall **83.4%**，BM25 **58.2%**，Dense **22.2%**；MemStack-ep **86.6%**；平均上下文 **572 token** |
| **LongMemEval（oracle）** | 500 题 | MemStack **All-Unit Recall 81.8%**，MemStack-ep **95.4%**；Dense / BM25 / Full context 均 **35.2%**；MemStack-ep 只用 **409 token** |

**三条结论：**

1. **结构化记忆显著提升证据覆盖**：同 800-token 预算下，纯会话检索只能塞进 1–2 个会话，
   LongMemEval 的 All-Unit Recall 只有 35.2%；MemStack 能覆盖 81.8%–95.4% 的全部证据会话。
2. **覆盖与逐字证据是两种能力**：MemStack 把更多短 episode 打包进上下文，证据覆盖率更高，
   但严格逐字命中率低于“整段会话”基线；Turn Token Recall 为 55.7%–56.4% vs 69%–70%。
3. **多跳仍是弱项**：LoCoMo multi-hop 的 All-Unit Recall 只有 22.8%（MemStack）/ 29.1%（episode-only），
   说明需要跨会话组合证据时，单次 top-K + 800 token 不够；这是下一步要补的迭代检索。

---

## 1. 数据集与协议

### 1.1 LoCoMo

- 论文：Maharana et al., **Evaluating Very Long-Term Conversational Memory of LLM Agents**, ACL 2024（arXiv:2402.17753）
- 数据：10 个 conversation，共 **1986** 个 QA，平均约 300 轮/对话、约 9000 token/对话
- 题型：multi-hop 282、temporal 321、open-domain 96、single-hop 841、adversarial 446
- 证据：每个 QA 带 `evidence`（如 `D1:3`），可映射到具体对话轮
- 本次评测：
  - **分层抽取 400 题**（5 类各 80 题；其中 3 题无证据，指标按 397 题计），覆盖全部 10 个 conversation
  - 上下文预算 **800 token**，`top_k=8`
  - episode 最大 3 轮，进入上下文的 episode 最多 900 字符
  - 规则抽取器 + 512 维 hashing embedder（不调用 LLM）
  - 未跑全量 1986 题：CPU-only 下完整跑约 25 分钟；分层 400 题已覆盖所有题型与 conversation

### 1.2 LongMemEval（oracle）

- 论文：Wu et al., **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory**, ICLR 2025（arXiv:2410.10813）
- 数据：500 题；本次使用 `longmemeval_oracle`（**只含证据会话，不含 distractor**）
- 题型：temporal-reasoning 133、multi-session 133、knowledge-update 78、single-session-assistant 56、single-session-user 70、single-session-preference 30
- 每题 1–6 个证据会话；每个 turn 有 `has_answer` 标记
- 本次评测：
  - 全量 **500 题**
  - 上下文预算 **800 token**，`top_k=8`
  - episode 最大 2 轮，进入上下文的 episode 最多 700 字符
  - 规则抽取器 + 512 维 hashing embedder

> **重要口径**：oracle 版本中所有 haystack session 都是证据会话，所以 Unit Recall@k 对所有方法都是 100%；
> 有区分度的是 **All-Unit Recall**（是否覆盖全部证据会话）和 **Turn Token Recall**（证据轮内容覆盖）。

### 1.3 指标

| 指标 | 定义 | 说明 |
|---|---|---|
| **Unit R@k** | 至少一个证据会话出现在 top-k/上下文中 | 召回“有没有找到” |
| **All-Unit** | 所有证据会话都出现 | 多跳/多会话问题必需 |
| **Turn Strict** | 证据轮原文完整出现在上下文中 | 最严格，受压缩影响 |
| **Turn Token** | 证据轮内容 token 被上下文覆盖的比例 | 软匹配，衡量“证据可用性” |
| **Ans Strict** | 标准答案完整出现在上下文中 | 严格答案支持 |
| **Ans Token** | 标准答案 token 覆盖比例 | 软答案支持 |
| **MRR** | 第一个证据会话的倒数排名 | 排序质量 |
| **Tokens** | 上下文 token 估算 | 成本 |

**这些不是 QA 准确率**：LoCoMo 的很多答案需要日期运算、跨会话推理或归一化；LongMemEval 官方用 LLM judge。
本评测只回答“证据有没有被记忆系统找出来、放进去”。

### 1.4 对比方法

| 方法 | 说明 |
|---|---|
| **MemStack** | 完整写入：episode + fact/preference/procedure + 溯源 + 混合检索 + 时间 + 图 + 重排（LongMemEval 为控制 CPU 成本关闭实体图） |
| **MemStack-ep** | 只存 episode、不做事实抽取；LoCoMo 上同时关闭实体图扩展，用于隔离“抽取/图”的贡献 |
| **Dense only** | 原始会话窗口 + hashing 向量 top-k，按预算裁剪 |
| **BM25 only** | 原始会话窗口 + BM25 top-k，按预算裁剪 |
| **Full context** | 同预算下保留最近的历史会话（截断基线） |

所有方法使用同一 800-token 预算；Dense/BM25 的 top-k 结果也按预算裁剪，避免“多塞 token 赢”。

> 配置差异：LoCoMo 的 MemStack 开启实体图/PPR，MemStack-ep 关闭；LongMemEval 两者都关闭图（1500 个会话的实体链接在 CPU 上成本过高）。
> 因此 LongMemEval 的 MemStack 数字是**保守下界**，不代表图扩展的上限。

---

## 2. 结果

### 2.1 LoCoMo（400 题分层，397 题有证据）

| 方法 | Unit R@k | All-Unit | Turn Strict | Turn Token | Ans Strict | Ans Token | MRR | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MemStack** | **83.4%** | **67.5%** | 54.9% | 67.0% | 8.6% | 20.4% | **0.572** | **572.4** |
| **MemStack-ep** | **86.6%** | **72.8%** | **59.9%** | **73.1%** | **9.3%** | **23.3%** | **0.643** | 725.0 |
| Dense only | 22.2% | 15.9% | 20.7% | 48.9% | 5.3% | 13.8% | 0.222 | 641.5 |
| BM25 only | 58.2% | 46.6% | 54.9% | 69.3% | 9.1% | 19.9% | 0.582 | 658.1 |
| Full context | 2.3% | 1.8% | 2.3% | 35.3% | 1.3% | 6.8% | 0.023 | 518.5 |

**MemStack 分题型：**

| 类别 | n | Unit R@k | All-Unit | Turn Token | Ans Token | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 80 | 92.5% | 92.5% | 79.1% | 0.0% | 583.4 |
| multi-hop | 79 | 79.7% | **22.8%** | 50.8% | 24.6% | 575.2 |
| open-domain | 78 | 61.5% | 38.5% | 45.6% | 14.5% | 514.6 |
| single-hop | 80 | 92.5% | 92.5% | 73.7% | **52.3%** | 577.8 |
| temporal | 80 | 90.0% | 90.0% | **85.1%** | 10.5% | 609.5 |

**怎么读：**

- Unit Recall：MemStack / MemStack-ep 明显高于 Dense / BM25；Full context 只剩最近会话，几乎找不到早期证据。
- All-Unit：multi-hop 只有 22.8% / 29.1%，说明单次检索很难同时取回所有跨会话证据。
- Turn Token：temporal 85.1%、single-hop 73.7%；multi-hop 50.8%、open-domain 45.6% 是短板。
- Ans Token 低（尤其 temporal 10.5%）是因为答案常需要“昨天→具体日期”的运算，而不是原文里直接有答案；
  这不代表检索失败，代表**需要 LLM reader 做推理**。

### 2.2 LongMemEval oracle（500 题）

| 方法 | Unit R@k | All-Unit | Turn Strict | Turn Token | Ans Strict | Ans Token | MRR | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MemStack** | 100% | **81.8%** | 20.8% | 55.7% | 38.6% | 46.6% | 1.000 | 753.0 |
| **MemStack-ep** | 100% | **95.4%** | 7.0% | 56.4% | **42.0%** | **52.3%** | 1.000 | **409.0** |
| Dense only | 100% | 35.2% | **72.2%** | **69.2%** | 30.8% | 50.5% | 1.000 | 791.9 |
| BM25 only | 100% | 35.2% | 72.6% | 69.4% | 33.2% | 52.2% | 1.000 | 792.2 |
| Full context | 100% | 35.2% | **75.0%** | **70.0%** | 36.0% | **55.5%** | 1.000 | 793.2 |

**MemStack 分题型：**

| 类别 | n | All-Unit | Turn Token | Ans Token |
|---|---:|---:|---:|---:|
| knowledge-update | 78 | 89.7% | 62.4% | 63.0% |
| multi-session | 133 | **66.2%** | 47.1% | 22.4% |
| single-session-assistant | 56 | 100% | 66.9% | 78.7% |
| single-session-preference | 30 | 100% | 64.2% | 36.3% |
| single-session-user | 70 | 100% | 63.3% | 70.9% |
| temporal-reasoning | 133 | 71.4% | 49.7% | 37.3% |

**怎么读：**

- **All-Unit 是核心差异**：800 token 下 Dense / BM25 / Full context 只能装 1–2 个会话，
  因此 multi-session / knowledge-update 的 All-Unit 只有 0%–35.2%；MemStack-ep 达到 **95.4%**，且只用 **409 token**。
- **Turn Strict 是反向指标**：整段会话基线把 1 个会话完整塞入，逐字命中率自然高（72%–75%）；
  MemStack 把更多短 episode 打包，逐字命中率低（20.8%/7.0%），但 Turn Token 仍有 55.7%/56.4%。
- **MemStack-ep vs MemStack**：episode-only 覆盖更全（95.4% vs 81.8%）、成本更低（409 vs 753），
  但逐字命中率更低（7.0% vs 20.8%）。这说明“事实抽取”在当前规则抽取器下对开放对话有噪声；
  对结构化 Agent 记忆，事实层的价值在合成基准里更明显。
- **多会话仍是弱项**：multi-session All-Unit 66.2%（MemStack）/88.7%（ep），
  需要迭代检索或更大的证据预算。

---

## 3. 结论与设计含义

### 3.1 结构化记忆的价值在“覆盖”，不在“逐字”

- 同预算下，纯会话检索只能取 1–2 个会话；MemStack 的 episode/fact 粒度更细，能覆盖更多证据会话。
- 代价是上下文被切成很多短片段，逐字证据命中率下降；Turn Token Recall 仍可用。
- 工程含义：**上下文应同时包含“导航片段 + 原始证据”**，而不是二选一。当前实现已经保留 episode 原文，
  但上下文预算只够放压缩片段；下一步应做“证据回取”：先选 episode，再按预算回填原文。

### 3.2 多跳需要迭代检索

LoCoMo multi-hop All-Unit 22.8% / 29.1%，LongMemEval multi-session 66.2% / 88.7%：
一次 top-K 很难同时命中所有证据会话。路线：
- 先检索，再用已命中实体/项目做第二轮检索（RF-Mem / MA-RAG 的多轮思路）；
- 对 multi-hop 查询提高 `top_k` 或预算；
- 用 PPR 从已命中实体扩展到相邻证据（当前图已具备，但权重偏保守）。

### 3.3 时间能力

LoCoMo temporal Unit R 90%、Turn Token 85.1%，说明时间戳 + episode 检索有效；
但 Ans Token 只有 10.5%，因为答案常需要日期运算。记忆层应提供“事件时间”，推理层负责算术。

### 3.4 规则抽取器的边界

LoCoMo 是开放域闲聊，规则抽取器会产生噪声事实（“who.supports = you” 之类），
导致 MemStack 完整版在 Unit Recall 上略低于 episode-only（83.4% vs 86.6%）。
生产环境应使用 `LLMExtractor`；本次评测为了零 API、可复现，保留了规则抽取器。

### 3.5 成本

- LoCoMo：MemStack 572 token/题，BM25 658，Dense 641，Full context 518；MemStack 用更低成本获得更高覆盖。
- LongMemEval：MemStack 753 token/题，baseline 792；MemStack-ep 409 token/题，覆盖率最高。
- 所有上下文都受 800-token 预算约束；MemStack 的组装器实际使用 572/753/409，没有超预算。

---

## 4. 复现

```bash
cd 上下文记忆/memstack

# 下载数据（ModelScope / HF mirror，仓库不提交数据）
python public_eval.py --dataset locomo --limit 400 --budget 800 --turns 3 --episode-chars 900 \
  --out results/locomo_400.json

python public_eval.py --dataset longmemeval --budget 800 --turns 2 --episode-chars 700 \
  --out results/longmemeval_full.json
```

数据镜像：

- LoCoMo：`https://modelscope.cn/datasets/evalscope/locomo/resolve/master/locomo10.json`
- LongMemEval oracle：`https://hf-mirror.com/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle`

原始 JSON / Markdown 结果在 `results/`。

---

## 5. 效度威胁（必须一起读）

1. **LoCoMo 是 400 题分层子集**，不是全量 1986 题；子集覆盖 5 类题型和 10 个 conversation，但仍有抽样方差。
2. **LongMemEval 使用 oracle 版本**，只含证据会话；Unit Recall@k 对所有方法都是 100%，
   真实 `longmemeval_s` 有约 40 个 distractor session/题，难度更高。本次未跑，原因是 277MB 数据 + CPU 写入成本。
3. **无 LLM reader**：Ans Strict/Ans Token 只衡量“答案是否在上下文里”，不代表最终 QA 准确率；
   不能与 LoCoMo 论文的 F1/BLEU 或 LongMemEval 的 GPT-4o judge 直接比较。
4. **规则抽取器 + hashing embedder**：不是生产配置；换 `LLMExtractor` / BGE 后结果会变化。
5. **Token 估算是启发式**（英文词 ×1.3 + 中文字），与具体 tokenizer 有偏差；但所有方法用同一估算器，相对比较公平。
6. **Turn Strict 受压缩影响**：MemStack 的 episode 字符上限（700/900）会截断长轮次；
   Turn Token 是更稳健的软指标。
7. **无显著性检验**：未报告置信区间；400/500 题上的百分比差异应谨慎解读。

---

## 6. 引用

```bibtex
@inproceedings{maharana2024locomo,
  title={Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  booktitle={ACL}, year={2024}
}

@inproceedings{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and Chang, Kai-Wei and Yu, Dong},
  booktitle={ICLR}, year={2025}
}
```
