# Retrieval + Rerank 改进记录

日期：2026-04-14

这份文档记录本次对 `memos-rag` 检索链路的增强，包括多路召回、RRF 融合、cross-encoder reranker 接入，以及调试过程中观察到的一些现象。

---

## 一、本次改动的目标

原始项目的检索流程比较基础：

- 只使用 Chroma 的 dense retrieval
- 直接取 Top-K
- 不做多路召回
- 不做重排序

这次改动的目标是把它升级成更标准的两阶段检索链路：

1. dense + BM25 多路召回
2. 用 RRF 做候选融合
3. 用 cross-encoder reranker 做精排
4. 最终只把 rerank 后的前几条结果送给 LLM

---

## 二、本次实现了什么

### 1. 在 `src/rag.py` 中加入 BM25 检索器

项目中新增了一个轻量的 `SimpleBM25Retriever`，不依赖额外第三方 BM25 包，而是直接在现有代码中实现：

- 对 chunk 做 token 化
- 统计词频、文档频率、平均文档长度
- 按 BM25 公式打分

这样做的好处是：

- 不需要额外安装包
- 可以快速验证 dense + sparse 混合召回是否有价值
- 和现有 stable `chunk.id` 体系天然兼容

### 2. 实现了 Dense + BM25 的多路召回

当前检索不再只走一条路径，而是：

- Dense 路：Chroma 向量召回
- Sparse 路：BM25 关键词召回

这两路各自召回一定数量的候选，再进入融合阶段。

### 3. 使用 RRF 做多路结果融合

在两路召回之后，使用 RRF（Reciprocal Rank Fusion）做去重和排序融合。

RRF 的作用是：

- 不直接使用不同检索器的原始分数
- 只关注它们各自在本路中的排名
- 如果某条文档在多路检索中都排名靠前，就会获得更高融合分

这里 RRF 的角色不是“最终裁判”，而是“候选集合融合器”。

### 4. 接入 cross-encoder reranker

在 RRF 融合之后，新增了一个 rerank 阶段。

当前默认使用的 reranker 模型是：

```text
BAAI/bge-reranker-v2-m3
```

这个模型属于典型的 cross-encoder reranker：

- 输入是 `(query, document)` 对
- 输出是相关性分数
- 比单纯的向量相似度更适合做精细排序

新的检索链路可以表示为：

```text
query
-> dense recall
-> bm25 recall
-> RRF fusion
-> cross-encoder rerank
-> top-k docs
-> LLM
```

### 5. 调整了召回与精排参数

本次同时调整了各阶段 `top_k`，让 reranker 有足够的候选空间可以挑选：

- `RAG_DENSE_TOP_K = 12`
- `RAG_BM25_TOP_K = 12`
- `RAG_FUSION_TOP_K = 15`
- `RAG_FINAL_TOP_K = 5`
- `RAG_RRF_K = 60`
- `RAG_RERANK_BATCH_SIZE = 8`

同时增加了：

- `RAG_RERANKER_MODEL_NAME`
- `RAG_USE_RERANK`

这样后续可以很方便地：

- 切换 reranker 模型
- 临时关闭 rerank 做对照实验

---

## 三、调试工具也同步升级了

为了便于观察不同检索阶段的结果，本次还修改了 `src/query.py`，让它不再只打印 dense 检索结果，而是分别展示：

- Dense Retrieval (Chroma)
- BM25 Retrieval
- Hybrid Retrieval (RRF Fusion)
- Reranked Results (Cross-Encoder)

这样可以非常直观地比较：

- Dense 召回了什么
- BM25 召回了什么
- 融合后保留了什么
- reranker 最终把哪些结果排到了最前面

这对后续调参和分析检索行为非常有帮助。

---

## 四、这次改动中的一个有意思观察

测试时发现一个很典型的现象：

- 某条 memo 中有英文短语 `waste of time`
- 查询使用的是中文问题：`什么事情浪费时间`

结果是：

- Dense 检索把这条英文 memo 召回到了第一位
- BM25 没有召回到它

这说明：

### Dense 检索的优势

- 能做跨语言、跨表达的语义匹配
- 即使 query 和文档词面不一致，也可能因为语义接近而命中

### BM25 的局限

- 更依赖词面重合
- 不具备真正的语义互通能力
- 在跨语言场景下容易失效

这也再次说明：

- Dense 更适合语义理解
- BM25 更适合关键词精确命中
- 混合检索的价值在于两者互补，而不是互相替代

---

## 五、为什么还保留 RRF

在加入 reranker 之后，RRF 仍然保留，原因是：

- Dense 和 BM25 需要一个融合阶段
- RRF 很适合做“候选集融合”
- 它不依赖不同检索器分数量纲一致

但它的角色已经发生变化：

- 之前：RRF 可以充当最终排序器
- 现在：RRF 负责融合候选，最终排序交给 reranker

因此当前更准确的理解是：

- `RRF = 候选融合`
- `Cross-Encoder = 最终精排`

---

## 六、为什么要把召回池放大

在没有 rerank 的时候，Top-K 通常取得比较小。  
但一旦加入 reranker，就应该适当扩大候选池。

原因是：

- reranker 只会在“已经召回到的候选”里挑
- 如果召回候选太少，它就没有足够空间做精排
- 两阶段检索的思想本来就是：先尽量多召回，再精细筛选

所以本次参数调整就是围绕这个思路来的：

- 每路先多召回一些
- 融合后保留较大的候选池
- 最后 rerank 后只取少量高质量上下文给 LLM

---

## 七、工程层面的补充

### 1. 新增了本地一键刷新知识库脚本

新增脚本：

```text
refresh_local_kb.sh
```

它会串联执行：

1. `sync_data.sh`：从远程机器同步 SQLite 数据库
2. `src/ingest.py`：把本地数据库同步到 Chroma

这样本地调试流程就更顺手了。

### 2. 遇到过代理导致 Hugging Face 下载卡住的问题

在首次加载 reranker 模型时，出现过下载卡住和超时问题。  
最终排查发现：

- 问题主要是 WSL 和本机代理之间的连接状态不稳定
- 并非项目代码错误
- 之后网络恢复后，cross-encoder 模型下载成功

这说明后续如果要迁移到服务器部署，需要提前考虑：

- Hugging Face 模型缓存
- 代理或镜像源
- 首次冷启动时的模型下载问题

---

## 八、当前系统状态

当前项目的检索链路已经从原来的基础版：

```text
query -> dense retrieval -> LLM
```

升级为：

```text
query
-> dense retrieval
-> BM25 retrieval
-> RRF fusion
-> cross-encoder rerank
-> LLM
```

这已经是一个更接近真实工程实践的两阶段混合检索架构。

---

## 九、下一步建议

在这次改动基础上，接下来最值得做的方向有：

1. 构建一个小规模评测集，对比 rerank 前后的效果
2. 在 `query.py` 中加入“仅 dense 命中 / 仅 BM25 命中 / 双路命中”的标记
3. 尝试加入 metadata-aware retrieval
4. 研究是否需要 query rewrite / multi-query
5. 评估更轻量的 reranker 模型，以降低首次下载和推理成本

如果后续要继续打磨项目，这一轮改动可以作为一个非常清晰的里程碑：

- 从 baseline RAG 进入混合召回 + 精排阶段
