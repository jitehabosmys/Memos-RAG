import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from etl import DB_PATH, fetch_all_memos, process_documents

# 设定数据持久化路径
PERSIST_DIRECTORY = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def get_existing_hashes(vector_db: Chroma) -> dict[str, str | None]:
    """读取当前向量库中已有 chunk 的 content_hash。"""
    existing_data = vector_db.get(include=["metadatas"])
    ids = existing_data.get("ids", []) or []
    metadatas = existing_data.get("metadatas", []) or []

    existing_hashes = {}
    for chunk_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}
        existing_hashes[chunk_id] = metadata.get("content_hash")
    return existing_hashes

def ingest_data():
    print("🚀 Starting SMART ingestion pipeline...")
    
    # 1. 准备数据
    if not DB_PATH.exists():
        print(f"❌ Source database not found: {DB_PATH}. Sync aborted to avoid deleting valid index data.")
        return

    memos = fetch_all_memos()
    if not memos:
        print("⚠️ No valid memos found in source DB. Will remove stale chunks from vector store if needed.")
    
    new_chunks = process_documents(memos)
    new_ids = [chunk.id for chunk in new_chunks]
    chunk_by_id = {chunk.id: chunk for chunk in new_chunks}
    incoming_hashes = {
        chunk.id: chunk.metadata.get("content_hash")
        for chunk in new_chunks
    }
    
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
    existing_hashes = get_existing_hashes(vector_db)
    existing_ids = set(existing_hashes)
    incoming_ids = set(new_ids)

    to_add = incoming_ids - existing_ids
    overlapping_ids = incoming_ids.intersection(existing_ids)
    to_update = {
        chunk_id
        for chunk_id in overlapping_ids
        if existing_hashes.get(chunk_id) != incoming_hashes.get(chunk_id)
    }
    unchanged_ids = overlapping_ids - to_update
    to_delete = existing_ids - incoming_ids
    sync_ids = to_add | to_update
    chunks_to_sync = [chunk_by_id[chunk_id] for chunk_id in new_ids if chunk_id in sync_ids]
    ids_to_sync = [chunk.id for chunk in chunks_to_sync]
    
    print("\n--- 📊 Sync Report ---")
    print(f"Total Source Chunks : {len(new_chunks)}")
    print(f"Existing in DB      : {len(existing_ids)}")
    print(f"---------------------")
    print(f"🆕 To Add           : {len(to_add)}")
    print(f"🔄 To Update        : {len(to_update)}")
    print(f"⏭️ To Skip (Same)   : {len(unchanged_ids)}")
    print(f"🗑️ To Delete        : {len(to_delete)}")
    print(f"---------------------\n")

    if len(to_add) == 0 and len(to_update) == 0 and len(to_delete) == 0:
        print("✅ Nothing to change. Database is already in sync.")
        return

    print("⚡ Syncing...")
    if ids_to_sync:
        vector_db.add_documents(documents=chunks_to_sync, ids=ids_to_sync)

    if to_delete:
        vector_db.delete(ids=list(to_delete))

    final_count = vector_db._collection.count()
    print(f"🎉 Done! Total documents in DB: {final_count}")

if __name__ == "__main__":
    ingest_data()
