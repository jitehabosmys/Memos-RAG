import sys
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# -----------------------
PERSIST_DIRECTORY = "./data/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def query_vector_db(question: str):
    print(f"🔎 Searching for: '{question}'...")
    
    # 强制离线模式，消除网络请求延迟
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 1. 加载 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 加载现有的 Vector DB
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="memos_rag"
    )

    # 3. 检索 (Top 3)
    results = vector_db.similarity_search_with_score(question, k=3)

    print(f"\n✅ Found {len(results)} relevant notes:\n")
    for i, (doc, score) in enumerate(results):
        # Score 越小越相似 (L2 Distance) 或 越大越相似 (Cosine)，Chroma 默认L2，越小越好
        print(f"--- [Result {i+1}] (Score: {score:.4f}) ---")
        print(f"📅 Date: {doc.metadata.get('date', 'Unknown')}")
        print(f"📝 Content: {doc.page_content}...")
        print("")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/query.py 'Your question here'")
        sys.exit(1)
    
    question = sys.argv[1]
    query_vector_db(question)