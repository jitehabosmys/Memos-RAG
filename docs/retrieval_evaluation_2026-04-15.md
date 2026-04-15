# Retrieval Evaluation 记录

日期：2026-04-15

这份文档记录 `memos-rag` 当前检索评测部分的设计与实现，包括：

- 评测集是怎么组织的
- 评测脚本统计了哪些指标
- 当前评测覆盖了哪些检索方案
- 结果应该如何解读

---

## 一、为什么要单独做检索评测

在 RAG 项目里，如果只靠“问几个问题看看回答好不好”，很难判断问题到底出在：

- 检索没召回到
- 候选召回到了，但排序不对
- LLM 生成阶段没有用好上下文

所以这次评测的目标，是把问题先拆到检索层，独立回答：

1. 正确 chunk 有没有被召回
2. 正确 memo 有没有被召回
3. 正确结果出现在第几名
4. dense、BM25、RRF、rerank 之间到底谁更有效

---

## 二、评测集放在哪里

当前评测集文件是：

```text
eval/retrieval_eval_set.json
```

它现在包含 30 条人工整理的问题，每条样本都显式标注了：

- `id`
- `question`
- `expected_chunk_ids`
- `expected_memo_ids`

样例结构如下：

```json
{
  "id": "q001",
  "question": "什么事情浪费时间？",
  "expected_chunk_ids": ["memo_4_0"],
  "expected_memo_ids": [4]
}
```

这个设计里有两个层次的“正确答案”：

### 1. Chunk Match

要求命中指定的 chunk id。

这个指标更严格，适合看：

- chunk 级别召回是否精准
- 排序是不是把真正相关的切片放到了前面

### 2. Memo Match

要求命中指定的 memo id。

这个指标更宽松，适合看：

- 即使不是最理想的 chunk，是否至少回到了正确 memo
- 在切片较多的 memo 上，系统是否仍有较强召回能力

这两个指标一起看，比只看“有没有沾边”更有用。

---

## 三、评测脚本实现在哪里

当前评测脚本是：

```text
src/evaluate_retrieval.py
```

它会：

1. 读取 `eval/retrieval_eval_set.json`
2. 初始化检索组件
3. 对每条问题分别运行多种检索方案
4. 统计 chunk match 和 memo match 两套指标
5. 输出 Top5 miss 样例，帮助排查失败原因

---

## 四、当前评测覆盖了哪些方法

### 默认评测方法

无论是否启用 query rewrite，都会评测这四种：

1. Dense Retrieval
2. BM25 Retrieval
3. RRF Fusion
4. Rerank

它们分别对应：

- `Dense Retrieval`
  - 只看 Chroma 的 dense 召回
- `BM25 Retrieval`
  - 只看 BM25 关键词召回
- `RRF Fusion`
  - dense + BM25 融合后的候选排序
- `Rerank`
  - dense + BM25 + RRF 后，再经过 cross-encoder reranker 的最终结果

### 可选评测方法

当 `RAG_USE_QUERY_REWRITE=true` 时，还会额外评测：

1. Multi-Query Global Fusion
2. Multi-Query Rerank

也就是：

- 先生成多个检索改写
- 每个 query 各自做 dense + BM25 + 局部融合
- 再做全局融合
- 最后再 rerank

不过当前默认主方案已经切回单查询，所以这部分属于可选对照，不是默认基线。

---

## 五、统计了哪些指标

评测脚本当前统计：

- `MRR`
- `Hit@1`
- `Hit@3`
- `Hit@5`

### 1. MRR 是什么

MRR 全称是 `Mean Reciprocal Rank`。

对于每个问题：

- 如果第 1 名就命中，得分是 `1`
- 如果第 2 名才命中，得分是 `1/2`
- 如果第 3 名才命中，得分是 `1/3`
- 以此类推

最后对所有问题求平均。

MRR 的含义是：

- 越强调“正确答案尽量靠前”
- 不只是看有没有命中，还看命中得早不早

### 2. Hit@K 是什么

`Hit@K` 表示：

- 正确结果是否出现在前 K 个结果里

例如：

- `Hit@1`
  - 是否第一名就对
- `Hit@3`
  - 是否前三里有对的
- `Hit@5`
  - 是否前五里有对的

对 RAG 来说，`Hit@5` 很重要，因为最终通常只会把少量上下文送给 LLM。

---

## 六、为什么现在只输出 Top5 miss

最早评测时，脚本会记录更广义的 miss。后来做了一个更实用的调整：

- 重点只关注 `Top5 miss`

原因是：

- 如果正确结果掉到第 30 名以后，当然也算 miss
- 但对于当前项目来说，更关键的是：
  - 最终传给 LLM 的上下文一般只有前几条
  - 所以“掉出前 5”才是对实际问答最有影响的失败

因此脚本现在会重点记录：

- 正确 chunk / memo 没有进入前 5
- 或者虽然出现了，但第一个正确结果排在第 6 名以后

这比“全量 miss 列表”更聚焦，也更适合调参。

---

## 七、脚本运行方式

本地运行命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/evaluate_retrieval.py
```

之所以常配合离线模式，是因为：

- embedding 模型和 reranker 模型已经缓存后
- 没必要每次重新尝试联网
- 这样能减少代理和网络波动带来的等待

如果要指定评测集路径，也可以通过环境变量：

```bash
RAG_EVAL_SET_PATH=eval/retrieval_eval_set.json .venv/bin/python src/evaluate_retrieval.py
```

如果想调节输出多少个 Top5 miss 样例：

```bash
RAG_EVAL_SHOW_TOP5_MISSES=3 .venv/bin/python src/evaluate_retrieval.py
```

---

## 八、脚本做了哪些可观测性增强

为了避免“一跑十几分钟，最后才一口气出结果”，脚本已经做了两项增强：

### 1. 逐题进度日志

每条问题都会输出：

```text
⏳ Evaluating q012 (12/30): ...
```

这样可以知道当前跑到哪了。

### 2. 周期性进度汇报

每 5 条问题会输出一次：

```text
✅ Progress: 5/30 questions evaluated.
```

这对 reranker 较慢、整轮评测要跑好几分钟的场景很重要。

---

## 九、当前这轮评测结果是什么

这轮结果保存在本地：

```text
test.txt
```

本次结果的关键数字如下。

### Chunk Match

- Dense
  - `MRR = 0.9214`
  - `Hit@1 = 0.8667`
  - `Hit@3 = 0.9667`
  - `Hit@5 = 0.9667`
- BM25
  - `MRR = 0.8500`
  - `Hit@1 = 0.7667`
  - `Hit@3 = 0.9333`
  - `Hit@5 = 0.9333`
- RRF
  - `MRR = 0.9020`
  - `Hit@1 = 0.8667`
  - `Hit@3 = 0.9000`
  - `Hit@5 = 0.9333`
- Rerank
  - `MRR = 0.9389`
  - `Hit@1 = 0.9000`
  - `Hit@3 = 1.0000`
  - `Hit@5 = 1.0000`
- Multi-Query Global Fusion
  - `MRR = 0.9039`
  - `Hit@1 = 0.8667`
  - `Hit@3 = 0.9000`
  - `Hit@5 = 0.9667`
- Multi-Query Rerank
  - `MRR = 0.9389`
  - `Hit@1 = 0.9000`
  - `Hit@3 = 1.0000`
  - `Hit@5 = 1.0000`

### Memo Match

- Dense
  - `MRR = 0.9667`
  - `Hit@1 = 0.9333`
  - `Hit@3 = 1.0000`
  - `Hit@5 = 1.0000`
- BM25
  - `MRR = 0.9000`
  - `Hit@1 = 0.8667`
  - `Hit@3 = 0.9333`
  - `Hit@5 = 0.9333`
- RRF
  - `MRR = 0.9103`
  - `Hit@1 = 0.8667`
  - `Hit@3 = 0.9333`
  - `Hit@5 = 0.9333`
- Rerank
  - `MRR = 0.9833`
  - `Hit@1 = 0.9667`
  - `Hit@3 = 1.0000`
  - `Hit@5 = 1.0000`
- Multi-Query Global Fusion
  - `MRR = 0.9139`
  - `Hit@1 = 0.8667`
  - `Hit@3 = 0.9333`
  - `Hit@5 = 0.9667`
- Multi-Query Rerank
  - `MRR = 0.9833`
  - `Hit@1 = 0.9667`
  - `Hit@3 = 1.0000`
  - `Hit@5 = 1.0000`

---

## 十、当前结果应该怎么解读

### 1. Dense 是很强的单路基线

从当前数据看，dense 已经相当强，尤其在：

- 语义改写
- 中英文不完全词面匹配
- 更自然语言式的问题表达

这些场景里表现不错。

### 2. BM25 是补充，不是主力

BM25 明显弱于 dense，尤其在两类问题上容易掉：

- 词面不完全重合
- 跨语言、跨表达的语义匹配

典型例子就是：

- memo 里是英文短语 `waste of time`
- query 是中文的 `什么事情浪费时间`

Dense 能召回，BM25 召不回来。

### 3. RRF 本身不是主要提升来源

从结果看，RRF 没有稳定超过 dense 基线。

这说明当前数据集上：

- 多路召回本身有价值
- 但单看融合排序，还不是决定性提升

RRF 更适合被理解为：

- 候选集合融合器

而不是最终最强排序器。

### 4. 真正拉开差距的是 rerank

当前最关键的提升来自 cross-encoder rerank。

因为它解决的是：

- 候选里已经有正确文档
- 但 dense / BM25 / RRF 没有把它排到足够前面

所以当前更准确的理解是：

- dense / BM25 / RRF 负责尽量把正确结果召回进候选池
- rerank 负责把真正最相关的候选推到最前面

### 5. Multi-query 暂时没有带来新的最终收益

在这 30 条样本上：

- `Multi-Query Rerank` 和 `Rerank` 打平

这意味着：

- 多查询改写并没有在当前评测集上进一步提高最终 top5 质量
- 现阶段默认主方案仍然更适合保持为单查询

这也是后来把默认主链路切回：

```text
single-query + hybrid recall + rerank
```

的重要原因之一。

---

## 十一、当前评测结论

基于这 30 条评测集，当前最稳的工程结论是：

1. Dense 是最强的单路召回基线
2. BM25 适合作为补充召回，而不是主力方案
3. RRF 适合作为候选融合层
4. Rerank 是当前最主要的性能增益来源
5. 当前默认主方案应保持：

```text
single-query + hybrid recall + rerank
```

---

## 十二、当前评测设计的边界

虽然这套评测已经很有用了，但仍然有几个明确边界：

### 1. 评测集规模还比较小

当前只有 30 条，适合作为第一版对照，但还不算严格的大规模 benchmark。

### 2. 目前不是时间专项评测

现在这批样本主要用于一般语义检索能力验证，不是专门为 metadata-aware retrieval 设计的。

如果后续要严谨评估时间感知检索，最好单独加入：

- `昨天`
- `去年`
- `最早`
- `最近一次`
- `最近一个月`

这类问题。

### 3. 当前结果还是本地单轮结果

目前结果来自一次本地运行，耗时较长，容易受：

- reranker CPU 开销
- query rewrite LLM 调用
- 代理和网络状态

等因素影响。

所以后续如果评测要常态化，最好考虑：

- 结果持久化
- 固定环境变量
- 固定模型缓存状态

---

## 十三、一句话总结

这套评测的核心价值，是把“RAG 好不好用”拆成了一个更可分析的问题：

- 是否召回到了
- 是否排进前 5
- 是 dense 更强，还是 BM25 更强
- rerank 到底有没有带来真实收益

到目前为止，这份评测已经支持我们比较有把握地得出结论：

```text
当前项目的默认最佳主方案是：
single-query + hybrid recall + rerank
```
