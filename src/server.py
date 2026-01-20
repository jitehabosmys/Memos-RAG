import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import get_rag_chain
from ingest import ingest_data

load_dotenv()

app = FastAPI(title="Memos RAG API", version="1.0.0")

# --- 数据模型 ---
class QueryRequest(BaseModel):
    question: str

# --- 全局变量 ---
rag_chain = None
is_refreshing = False  # 🔒 全局锁，防止并发更新

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型，避免每次请求冷启动"""
    global rag_chain
    print("🚀 Starting Server: Loading RAG Chain...")
    try:
        rag_chain = get_rag_chain()
        print("✅ RAG Chain loaded successfully.")
    except Exception as e:
        # [Production Best Practice] Fail Fast
        # 如果核心组件加载失败，直接崩溃，让运维工具(Docker/K8s)感知并重启
        print(f"❌ Critical Error: Failed to load RAG Chain: {e}")
        raise RuntimeError(f"RAG Chain Initialization Failed: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Memos RAG Brain is Online"}

@app.post("/refresh")
async def refresh_knowledge_base():
    """触发知识库更新 (ETL + Ingest)"""
    global rag_chain, is_refreshing
    
    if is_refreshing:
        raise HTTPException(status_code=429, detail="⚠️ Update already in progress. Please wait.")
    
    is_refreshing = True
    try:
        print("🔄 Refresh request received. Starting ingestion...")
        # 1. 执行 ETL 和 向量化
        await asyncio.to_thread(ingest_data)
        
        # 2. 重新加载 RAG Chain
        print("♻️ Reloading RAG Chain...")
        rag_chain = get_rag_chain()
        
        return {"status": "success", "message": "Knowledge base updated and reloaded!"}
    except Exception as e:
        print(f"❌ Refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 无论成功失败，一定要释放锁
        is_refreshing = False

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """普通接口：等待生成完毕后一次性返回"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # [Production Best Practice] Async Invoke
        # 使用 ainvoke 而不是 invoke，防止阻塞 Event Loop
        response = await rag_chain.ainvoke(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def stream_chat_endpoint(request: QueryRequest):
    """流式接口：SSE (Server-Sent Events) 格式返回"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    async def event_generator():
        try:
            # [Production Best Practice] Async Streaming
            # 使用 async for 配合 astream，真正释放 CPU 给其他请求
            async for chunk in rag_chain.astream(request.question):
                # SSE 格式: data: <content>\n\n
                yield dict(data=chunk)
            yield dict(data="[DONE]") # 结束标记
        except Exception as e:
            yield dict(data=f"Error: {str(e)}")

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    # 本地开发使用 reload=True 实现热重载
    # 安全起见，默认只监听本地接口 (127.0.0.1)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
