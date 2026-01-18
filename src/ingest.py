import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from etl import fetch_all_memos, process_documents

# 设定数据持久化路径
PERSIST_DIRECTORY = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def ingest_data():
    print("🚀 Starting SMART ingestion pipeline...")
    
    # 1. 准备数据
    memos = fetch_all_memos()
    if not memos:
        print("⚠️ No memos found.")
        return
    
    new_chunks = process_documents(memos)
    new_ids = [chunk.id for chunk in new_chunks]
    
    # 2. 加载环境
    print(f"🧠 Loading embedding model...")
    
    # 智能离线模式控制
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        # 模糊匹配：只要缓存里包含模型名称的一部分，就认为存在
        model_cached = any(EMBEDDING_MODEL_NAME in str(repo.repo_id) for repo in cache_info.repos)
        
        if model_cached:
            print(f"✅ Model found locally. Offline Mode ENABLED.")
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            print(f"🌐 Model not found. Online Mode ENABLED for download...")
            os.environ["HF_HUB_OFFLINE"] = "0"
    except Exception:
        # 如果检测失败，默认开启下载，防止报错
        os.environ["HF_HUB_OFFLINE"] = "0"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"💾 Opening Vector Store...")
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="memos_rag"
    )

    # 3. 计算差异 (Diff)
    existing_data = vector_db.get() # 返回 {'ids': [...], 'embeddings': ..., 'documents': ...} 
    existing_ids = set(existing_data['ids'])
    incoming_ids = set(new_ids)

    to_add = incoming_ids - existing_ids
    to_update = incoming_ids.intersection(existing_ids)
    
    print("\n--- 📊 Sync Report ---")
    print(f"Total Source Chunks : {len(new_chunks)}")
    print(f"Existing in DB      : {len(existing_ids)}")
    print(f"---------------------")
    print(f"🆕 To Add           : {len(to_add)}")
    print(f"🔄 To Update        : {len(to_update)}")
    print(f"---------------------\n")

    if len(to_add) == 0 and len(to_update) == 0:
        print("✅ Nothing to change. Database is already in sync.")
        return

    print("⚡ Syncing...")
    vector_db.add_documents(documents=new_chunks, ids=new_ids)

    final_count = vector_db._collection.count()
    print(f"🎉 Done! Total documents in DB: {final_count}")

if __name__ == "__main__":
    ingest_data()