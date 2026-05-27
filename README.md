# Memos RAG — 个人第二大脑

基于 [Memos](https://usememos.com/) 笔记构建的智能问答系统，使用**混合检索（稠密向量 + BM25）+ Cross-Encoder 重排序 + 时间感知检索**实现高精度召回。

## 架构流程

```
Memos SQLite → ETL 清洗切分 → Embedding (bge-small-zh-v1.5) → Chroma DB
                                                                ↓
用户问题 → 时间意图解析 (LLM + 规则) → 语义 query 提取 → 混合检索 (Dense + BM25)
                                                                ↓
                                                    RRF 融合 → Cross-Encoder 重排序
                                                                ↓
                                                    时间排序 / Recency Boost → LLM 生成回答
```

## 核心特性

### 🔍 混合检索召回
- **稠密向量检索**：HuggingFace `bge-small-zh-v1.5` 语义检索，支持 Chroma `$and/$gte/$lte` 元数据过滤
- **BM25 关键词检索**：轻量内置 BM25（无需额外依赖），中英文混合 tokenization
- **RRF 融合**：`reciprocal_rank_fusion` 合并多路排序，消除单一检索偏差
- **多 Query 检索**：LLM 自动将问题改写为多条检索 query，各自独立检索再全局 RRF 融合

### 📊 Cross-Encoder 重排序
- 默认使用 `BAAI/bge-reranker-v2-m3`，对 RRF 融合后的候选文档精排，显著提升 TOP-K 准确率
- 可配置开关 `RAG_USE_RERANK`，batch 预测支持

### ⏰ 时间感知检索
- **时间意图解析**：LLM (`RAG_TIME_INTENT_MODEL_NAME`) 或规则正则，从问题中提取时间约束
  - 支持：今天/昨天/本周/上周/本月/上月/今年/去年、最近 N 天/周/月/年、绝对日期 `2024-01-01`、绝对年月、相对范围
  - 支持排序意图："最新"/"最早"/"第一条"/"最后一次"
  - 支持软时间偏好："最近"/"近期" 等模糊表达 → recency boost
- **时间过滤**：将时间约束转为 Chroma `created_ts` 元数据过滤条件，精确限定检索范围
- **Recency Boost**：对较新记录按指数衰减函数加权加分（半衰期可配置 `RAG_TIME_RECENCY_HALF_LIFE_DAYS`）
- **Metadata-First**：当问题仅涉及时间排序无实质主题时（如"我最后一条笔记是什么"），跳过语义检索，直接按时间元数据排序
- **最终排序**：rerank 后按 `sort_direction` (recent/oldest) 二次排列

### 🧩 差分同步
- 基于 `content_hash` (SHA256) 做差集同步：只处理新增/变更/删除的 chunk，避免全量重建
- 智能 HuggingFace 离线模式：本地有缓存时自动离线加载

### ⚡ 流式响应
- FastAPI SSE (`/chat/stream`) + Streamlit 打字机效果

### 📐 检索评估
- `eval/retrieval_eval_set.json`：30 组标注 Q&A pair，可用于 MRR / Hit@K 评估

## 项目结构

```text
memos-rag/
├── src/
│   ├── etl.py       # 从 SQLite 提取、清洗、切分 memo，生成 content_hash
│   ├── ingest.py    # 向量化 + 差分同步（content_hash 驱动）
│   ├── rag.py       # 核心：混合检索 / RRF 融合 / 重排序 / 时间感知 / RAG chain
│   ├── server.py    # FastAPI（/refresh /chat /chat/stream）
│   ├── app.py       # Streamlit 前端
│   └── query.py     # 命令行检索调试工具
├── eval/            # 标注检索评估集
├── data/            # SQLite 与 Chroma 持久化
├── docs/            # 设计文档
├── docker-compose.yml / Dockerfile
└── pyproject.toml
```

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量 (OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL_NAME)
echo "OPENAI_API_KEY=sk-..." >> .env

# 索引笔记
uv run src/ingest.py

# 启动后端
uv run uvicorn src.server:app

# 启动前端（另一终端）
uv run streamlit run src/app.py
```

## Docker 部署

```bash
docker compose up -d
# 访问 http://localhost:8501
```

## 可配置项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `CHROMA_DB_PATH` | `./data/chroma_db` | Chroma 持久化路径 |
| `RAG_RERANK_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-Encoder 重排序模型 |
| `RAG_USE_RERANK` | `true` | 是否启用重排序 |
| `RAG_USE_TIME_AWARE_RETRIEVAL` | `true` | 是否启用时间感知检索 |
| `RAG_USE_LLM_TIME_PARSER` | `true` | 是否使用 LLM 解析时间意图 |
| `RAG_USE_QUERY_REWRITE` | `false` | 是否启用多 Query 改写检索 |
| `RAG_DENSE_TOP_K` / `RAG_BM25_TOP_K` | `12` | 单路检索返回数 |
| `RAG_FUSION_TOP_K` | `15` | RRF 融合候选数 |
| `RAG_FINAL_TOP_K` | `5` | 最终送入 LLM 的文档数 |
| `RAG_RRF_K` | `60` | RRF 融合常数 |
| `RAG_TIME_RECENCY_WEIGHT` | `0.008` | Recency boost 权重 |
| `RAG_TIME_RECENCY_HALF_LIFE_DAYS` | `45` | 时间衰减半衰期（天） |
