#!/bin/bash
set -e

# 1. 启动后端 API (后台运行)
echo "🚀 Starting FastAPI Backend..."
# 使用 uvicorn 直接启动，监听 0.0.0.0 (容器内必须监听 0.0.0.0 才能被外部访问)
uvicorn src.server:app --host 0.0.0.0 --port 8000 &

# 2. 启动前端 UI (前台运行)
echo "🎨 Starting Streamlit Frontend..."
# 等待几秒确保后端先启动
sleep 3
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
