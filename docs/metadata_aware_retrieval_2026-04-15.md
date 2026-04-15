# Metadata-Aware Retrieval 实现记录

日期：2026-04-15

这份文档记录本次在 `memos-rag` 中加入时间感知检索（Metadata-Aware Retrieval）的具体实现方式。目标不是替换现有的 `single-query + hybrid recall + rerank` 主链路，而是在这个主链路上补上“时间意识”。

---

## 一、本次改动想解决什么问题

在原有检索链路里，系统已经能较好处理语义相关性，但还不够擅长处理这类问题：

- `最近我在想什么`
- `去年我怎么评价降A大调`
- `2026年1月我写过哪些和关系有关的内容`
- `我最早什么时候提到降A大调`

这些问题的共同点是：

- 不只是“找相关内容”
- 还包含了明确或模糊的时间约束

因此这次增强的目标是：

1. 识别 query 里的时间意图
2. 在检索阶段利用 `created_ts` 元数据
3. 让不同类型的时间表达走不同处理策略

---

## 二、这次没有重做数据层

这个增强能比较顺利接入，是因为项目原本就已经在 ETL 阶段把时间元数据写进 chunk 了。

在 [etl.py](/home/sytssmys/memos-rag/src/etl.py) 里，每个 chunk 的 metadata 已经包含：

- `memo_id`
- `created_ts`
- `date`
- `chunk_id`

所以这次没有新增数据库字段，也没有重新设计 Chroma schema，而是直接利用已有 metadata。

---

## 三、核心设计思路

这次实现把时间问题拆成三类：

### 1. 硬过滤 `hard filter`

适用于时间范围明确的问题，例如：

- `昨天我记了什么`
- `去年我怎么评价降A大调`
- `2026年1月我写过哪些和关系有关的内容`
- `最近一个月关于钢琴的记录`

这类 query 会先解析出时间范围，再在召回前做过滤。

### 2. 软偏好 `soft recent boost`

适用于模糊的最近性表达，例如：

- `最近我在想什么`
- `近期我都写了什么`
- `这段时间我在关注什么`

这类 query 不会被硬切到某个固定时间窗，而是在语义相关的候选上叠加“越新越加分”的偏置。

### 3. 时间排序 `final time ordering`

适用于显式要求先后顺序的问题，例如：

- `我最早什么时候提到降A大调`
- `我最近一次提到关系是什么时候`
- `最后一次写钢琴是什么时候`

这类 query 在语义召回和 rerank 完成后，再按时间做最终排序。

---

## 四、时间意图识别是怎么做的

目前实现已经升级为：

- `LLM 结构化解析` 作为主路径
- `规则解析` 作为兜底

这样做是为了同时兼顾：

- 自然语言时间表达的可扩展性
- 时间计算与执行逻辑的确定性
- API 异常时的可用性

实现位置在 [rag.py](/home/sytssmys/memos-rag/src/rag.py)。

### 1. 新增 `TimeIntent` 数据结构

使用 `dataclass` 定义了统一的时间意图对象，核心字段包括：

- `start_ts`
- `end_ts`
- `boost_recent`
- `sort_direction`
- `semantic_query`
- `matched_phrases`
- `reason`
- `parser_source`
- `retrieval_strategy`

它的职责是把各种中文时间表达统一转换成一个检索阶段可消费的结构。

### 2. 主路径：LLM 解析时间意图

现在新增了专门的时间意图解析链：

- `create_time_intent_chain()`
- `get_time_intent_chain()`
- `parse_time_intent_with_llm()`
- `build_time_intent_from_payload()`

LLM 的职责不是直接给出 Unix 时间戳，而是输出结构化语义，例如：

- 是不是 `soft_recent`
- 是不是 `relative_range`
- 是否存在 `sort_direction=oldest/recent`
- 命中了哪些原始时间短语

然后再由代码把这些结构转换成真正的：

- `start_ts`
- `end_ts`
- `boost_recent`
- `sort_direction`

这样做的好处是：

- 避免让 LLM 直接参与时间戳计算
- 让“第一条 / 第一篇 / 最开始 / 最近一次”这类表达更容易扩展
- 保留代码层对时区、边界和过滤逻辑的控制

此外，LLM 现在还可以额外输出：

- `retrieval_strategy = semantic_first`
- `retrieval_strategy = metadata_first`

不过现在它更准确的角色是：

- `路由建议`

最终是否走 `metadata_first`，还要经过代码侧复核。

代码会优先检查：

- `semantic_query` 里是否还有明确主题词

如果仍然存在明显主题，例如：

- `降A大调`
- `钢琴`
- `关系`

那么即使 LLM 提议 `metadata_first`，系统也会改判为：

- `semantic_first`

只有在 `semantic_query` 基本已经没有实际主题、只剩下：

- `memo`
- `笔记`
- `记录`
- `是什么`

这类泛词时，才会真正走：

- `metadata_first`

这样做是为了避免把这类 query 误判：

- `我在2026年一月发的第一条关于降A大调的memo`

它虽然带有“第一条”，但因为仍然存在明确主题 `降A大调`，所以应该走 `semantic_first`，而不是直接 metadata 排序。

它的作用是帮助系统区分两类问题：

- `我最早什么时候提到降A大调`
  - 这是“有主题 + 有排序”的问题，更适合 `semantic_first`
- `我发的第一条 memo 是什么`
  - 这是“主要靠元数据排序就能回答”的问题，更适合 `metadata_first`

### 3. 兜底路径：规则解析

如果出现下面这些情况：

- 时间意图 API 不可用
- LLM 输出不是合法 JSON
- LLM 没能解析出有效时间意图
- LLM 没有给出可用的 `semantic_query`

系统会自动回退到规则解析，也就是：

- `build_time_intent_rule_based()`

规则兜底现在仍然支持常见显式时间表达，也额外补上了：

- `第一条`
- `第一篇`
- `第一个`
- `最开始`
- `最先`

### 4. 统一入口：`build_time_intent(question)`

这个函数负责从用户问题中解析时间意图。

当前支持的类型包括：

- 相对日期
  - `今天`
  - `昨天`
  - `前天`
- 周/月/年范围
  - `上周`
  - `这周`
  - `上个月`
  - `这个月`
  - `今年`
  - `去年`
- 滚动时间窗
  - `最近7天`
  - `最近一个月`
  - `近三个月`
  - `最近两年`
- 绝对日期
  - `2026年`
  - `2026年1月`
  - `2026年1月12日`
  - `1月12日`
- 时间排序意图
  - `最新`
  - `最近一次`
  - `最后一次`
  - `最早`
  - `第一次`
  - `第一条`
  - `第一篇`
  - `第一个`
  - `最开始`
- 最近性偏好
  - `最近`
  - `近期`
  - `近来`
  - `这阵子`
  - `这段时间`

### 5. 为了避免时间词污染检索语义，新增 `build_retrieval_query(question, time_intent)`

现在这个函数优先使用 LLM 返回的：

- `semantic_query`

也就是“去掉时间约束后、保留主题内容的纯内容检索 query”。

不过现在的 fallback 已经做过一次收口：

1. 如果 LLM 解析成功
   - 直接使用 LLM 给出的 `semantic_query`
2. 如果 LLM 解析失败
   - 整体回退到 rule-based
   - 再由规则路径通过 `matched_phrases` 做 query 剥离

也就是说，现在不再走那种“LLM 成功一半，再局部 fallback 到字符串剥离”的混合路径。

例如：

```text
2026年1月我写过哪些和关系有关的内容
```

会被拆成：

- 时间过滤：`2026-01-01 00:00:00 ~ 2026-01-31 23:59:59`
- 检索 query：`我写过哪些和关系有关的内容`

这样做的目的是：

- 避免 embedding 和 BM25 把 `2026年1月` 当成正文关键词
- 让时间约束交给 metadata 层处理
- 保留语义检索对正文主题的专注

### 6. 新增 metadata-first 路由判断

现在系统还新增了：

- `decide_time_retrieval_strategy()`
- `strip_generic_query_words()`
- `get_metadata_sorted_candidates()`

它们负责在真正召回之前先判断：

- 这条 query 是不是本质上只需要“按时间排序”
- 是否应该跳过 dense / BM25 / rerank，直接做 metadata-first 返回

典型例子：

```text
我发的第一条memo是什么
```

这类 query 如果先走语义召回，很可能压根召不回真正最早的那条 memo。  
所以现在会直接：

1. 从文档池中按时间过滤
2. 再按 `oldest / recent` 排序
3. 直接返回排序后的候选

而像：

```text
我最早什么时候提到降A大调
```

这种带有明确主题约束的 query，仍然会走：

- `semantic_first`

也就是：

- 先做主题检索
- 再做时间排序

---

## 五、显式时间范围是怎么接进检索流程的

### 1. Dense 侧：转成 Chroma metadata filter

新增了 `build_chroma_time_filter(time_intent)`。

它会把时间范围转换成 Chroma 可接受的 filter，例如：

```python
{
    "$and": [
        {"created_ts": {"$gte": ...}},
        {"created_ts": {"$lte": ...}},
    ]
}
```

然后在 `vector_db.similarity_search_with_score(...)` 中传入 `filter=...`。

也就是说，明确时间范围的问题，会直接在向量召回前缩小候选范围。

### 2. BM25 侧：先做文档子集过滤，再计算 BM25

`SimpleBM25Retriever.retrieve()` 现在新增了 `time_intent` 参数。

处理方式是：

1. 先基于 `created_ts` 筛出符合时间范围的文档
2. 只在这个子集上重新计算：
   - `corpus_size`
   - `avg_doc_len`
   - `doc_freq`
3. 再做 BM25 打分

这样做比“先全量算 BM25 再过滤”更合理，因为：

- IDF 应该在当前候选子集语境中计算
- 明确时间查询本质上是在一个时间切片里检索

---

## 六、“最近”是怎么处理的

这里刻意没有把“最近”粗暴翻译成固定时间窗，而是做成 `soft recent boost`。

### 原因

如果把：

```text
最近我在想什么
```

硬切成“最近 7 天”，很容易出现：

- 最近 7 天刚好没写
- 但 10 天前写了一条特别相关
- 结果因为硬过滤直接丢失

所以更稳妥的做法是：

1. 先按语义召回
2. 再对新的文档加一点分

### 实现方式

新增了：

- `compute_recency_score()`
- `apply_recency_boost()`

它使用一个指数衰减公式：

```text
recency_score = exp(-age_days / half_life_days)
```

然后把这个值乘以一个较小权重，加到 RRF 融合分上。

当前可调参数：

- `RAG_TIME_RECENCY_BOOST_WEIGHT`
- `RAG_TIME_RECENCY_HALF_LIFE_DAYS`

默认设计原则是：

- 保持“语义相关”优先
- 让“越新”只是轻微偏置，而不是主导排序

---

## 七、“最早 / 最新”是怎么处理的

这种 query 的目标不是单纯找相关内容，而是带有明显的时间顺序要求。

本次实现采用两阶段思路：

1. 先按语义和 rerank 找到一批足够相关的候选
2. 再按时间做最终排序

不过现在又多了一条专门的快捷路径：

- 如果系统判断这是 `metadata_first` 查询
- 那么会在检索前直接按 metadata 过滤和排序
- 不再依赖 rerank 去“猜”最早或最后一条

对应实现：

- `apply_final_time_ordering()`
- `decide_time_retrieval_strategy()`
- `get_metadata_sorted_candidates()`

支持：

- `recent`
- `oldest`

例如：

- `最早`、`第一次` -> 从旧到新
- `最新`、`最近一次`、`最后一次` -> 从新到旧

为了避免 rerank 阶段只拿太小的候选池，本次还增加了：

- `RAG_TIME_SORT_CANDIDATE_POOL`

当 query 带有时间排序意图时，系统会先让 reranker 多保留一些候选，再做最终时间排序。

---

## 八、检索主链路具体变成了什么

原本默认主链路是：

```text
query
-> dense recall
-> bm25 recall
-> RRF fusion
-> cross-encoder rerank
-> top-k
```

现在如果启用了时间感知，它会变成：

```text
question
-> parse time intent
-> decide retrieval strategy
-> build retrieval query
-> if metadata_first:
     metadata filter + time sort
   else:
     dense recall with optional Chroma time filter
     bm25 recall with optional time-filtered document subset
     RRF fusion
     optional recency boost
     cross-encoder rerank
     optional final time ordering
-> top-k
```

注意这里仍然保持了原有主方案：

- `single-query`
- `hybrid recall`
- `rerank`

时间感知只是附着在现有主方案上的增强层。

---

## 九、这次对 `query.py` 做了什么

为了方便本地调试，[query.py](/home/sytssmys/memos-rag/src/query.py) 新增了时间意图输出。

现在你执行：

```bash
uv run src/query.py "去年我怎么评价降A大调"
```

会先看到：

- 实际用于检索的 query
- LLM 或规则给出的 `semantic_query`
- 时间意图来自 `llm` 还是 `rule`
- 最终路由是 `semantic_first` 还是 `metadata_first`
- 时间意图是否激活
- 是否有硬过滤
- 时间过滤范围
- 是否启用了 recent boost
- 是否有最终时间排序

这样本地验证就更容易了，不需要猜系统到底是怎么理解 query 的。

---

## 十、关键新增函数一览

本次时间感知检索主要新增了这些函数：

- `get_created_ts()`
- `get_now()`
- `make_day_bounds()`
- `make_week_bounds()`
- `make_month_bounds()`
- `make_year_bounds()`
- `create_time_intent_chain()`
- `get_time_intent_chain()`
- `extract_json_object()`
- `build_time_intent_from_payload()`
- `parse_time_intent_with_llm()`
- `build_time_intent_rule_based()`
- `build_time_intent()`
- `build_retrieval_query()`
- `build_chroma_time_filter()`
- `doc_matches_time_filter()`
- `compute_recency_score()`
- `apply_recency_boost()`
- `apply_final_time_ordering()`
- `serialize_time_intent()`

它们大多都集中在 [rag.py](/home/sytssmys/memos-rag/src/rag.py)。

---

## 十一、当前默认行为

目前相关默认开关是：

- `RAG_USE_TIME_AWARE_RETRIEVAL=true`
- `RAG_USE_LLM_TIME_PARSER=true`
- `RAG_USE_QUERY_REWRITE=false`

这意味着当前默认主方案是：

```text
single-query + hybrid recall + rerank + metadata-aware time handling
```

也就是：

- 不默认启用 multi-query
- 默认启用时间感知检索

---

## 十二、当前实现的边界

这次实现已经够做第一版落地，但仍然有一些明确边界：

### 1. LLM 解析并不等于“完全没有边界”

虽然现在主路径已经是 LLM 解析，但下面这些表达仍然可能需要更多样本观察：

- `那段时间`
- `当时`
- `前阵子`
- `去年年底`
- `春节前后`

这类问题现在比纯规则时代更容易扩展，但仍然可能需要继续打磨 prompt 或补少量规则。

### 2. reranker 本身没有直接读取时间 metadata

目前时间逻辑主要发生在：

- 检索前过滤
- RRF 后 recency boost
- rerank 后最终时间排序

也就是说，cross-encoder 仍然主要看正文内容，而不是显式把日期拼进 `(query, document)` 输入。

### 3. 目前还没有专门的时间评测集

虽然逻辑已经接上，但还没有建立一批专门围绕时间问题的评测样本，例如：

- `昨天`
- `去年`
- `最早`
- `最近一次`
- `最近一个月`

后续如果要严谨验证这一层，最好单独设计一小批 time-aware eval cases。

---

## 十三、我做过的验证

本次至少做了两类验证：

### 1. 语法检查

执行：

```bash
.venv/bin/python -m py_compile src/rag.py src/query.py
```

通过。

### 2. 时间解析样例验证

用样例检查过：

- `最近我在想什么`
- `去年我怎么评价降A大调`
- `我最早什么时候提到降A大调`
- `2026年1月我写过哪些和关系有关的内容`
- `2026年1月12日我写过什么`
- `最近一个月关于钢琴的记录`

确认：

- 时间意图能被正确识别
- 检索 query 能正确剥离时间短语
- 绝对月份和绝对日期解析正常

---

## 十四、一句话总结

这次 `Metadata-Aware Retrieval` 的本质不是重写整套 RAG，而是在现有 `single-query + hybrid recall + rerank` 的基础上，加了一层“时间理解 + 元数据利用”：

- 明确时间 -> 先过滤
- 最近性 -> 轻量加权
- 最早/最新 -> 最终排序

这是一个比较稳、比较工程化、也比较容易继续迭代的第一版实现。
