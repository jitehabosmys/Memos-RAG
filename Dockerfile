FROM python:3.10-slim

WORKDIR /app

# 安装必要的系统工具 (比如 curl 用于健康检查)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install uv

# 拷贝依赖配置
COPY pyproject.toml uv.lock ./

# 安装依赖
# 1. --no-install-project: 只安装依赖，不安装当前项目(因为 src 还没拷进来)
# 2. 去掉 --system: uv sync 默认创建 .venv
RUN uv sync --frozen --no-dev --no-install-project

# 【关键】把 .venv 加入 PATH，这样后续命令就能找到 uvicorn/streamlit
ENV PATH="/app/.venv/bin:$PATH"

# 拷贝代码
COPY src ./src
COPY docker-entrypoint.sh ./

# 赋予脚本执行权限
RUN chmod +x docker-entrypoint.sh

# 环境变量默认值
ENV MEMOS_DB_PATH=/data/memos.db
ENV CHROMA_DB_PATH=/data/chroma_db
ENV OPENAI_API_BASE=https://api.deepseek.com/v1
ENV OPENAI_MODEL_NAME=glm-4.6

# 暴露端口
EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
