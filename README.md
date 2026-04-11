# Memos RAG

一个基于 Memos 笔记构建的个人知识库问答系统。项目使用 SQLite 作为源数据，Chroma 作为向量数据库，LangChain 组织 RAG 流程，FastAPI 提供后端接口，Streamlit 提供前端聊天界面。

## 项目主流程

1. 从 Memos SQLite 数据库中提取笔记内容
2. 清洗文本并切分为可检索的 chunk
3. 为每个 chunk 分配 stable id，并同步写入 Chroma
4. 检索相关笔记片段并交给大模型生成回答
5. 通过 FastAPI 暴露 API，并由 Streamlit 提供交互界面

## 项目结构

```text
memos-rag/
├── src/
│   ├── etl.py       # 从 SQLite 提取、清洗、切分 memo，并生成 stable id
│   ├── ingest.py    # 向量化并同步到 Chroma，处理新增、更新、删除
│   ├── rag.py       # 构建检索 + 生成的 RAG chain
│   ├── server.py    # FastAPI 后端，提供 refresh/chat/stream 接口
│   ├── app.py       # Streamlit 前端聊天页面
│   └── query.py     # 命令行检索调试工具
├── data/            # SQLite 与 Chroma 持久化数据
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## `src/` 模块说明

- `src/etl.py`：负责 ETL，将原始 memo 转成 LangChain `Document`，并给每个 chunk 分配稳定 ID。
- `src/ingest.py`：负责 embedding 与向量库同步，基于 stable id 做差集同步。
- `src/rag.py`：负责加载向量库、定义 prompt，并组装 RAG chain。
- `src/server.py`：负责提供 HTTP API，并在刷新知识库后重载问答链。
- `src/app.py`：负责前端交互，支持健康检查、知识库刷新和流式聊天。
- `src/query.py`：负责独立验证向量检索效果，方便调试。


