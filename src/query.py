import sys
import os
from rag import (
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    FUSION_TOP_K,
    get_chunk_id,
    initialize_retrieval_components,
    run_hybrid_retrieval,
)


def print_scored_results(title: str, results):
    print(f"\n=== {title} ===")
    if not results:
        print("No results.")
        return

    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- [Result {i}] (Score: {score:.4f}) ---")
        print(f"🆔 Chunk ID: {get_chunk_id(doc)}")
        print(f"📅 Date: {doc.metadata.get('date', 'Unknown')}")
        print(f"📝 Content: {doc.page_content[:200]}...")
        print("")


def print_docs(title: str, docs):
    print(f"\n=== {title} ===")
    if not docs:
        print("No results.")
        return

    for i, doc in enumerate(docs, start=1):
        print(f"--- [Result {i}] ---")
        print(f"🆔 Chunk ID: {get_chunk_id(doc)}")
        print(f"📅 Date: {doc.metadata.get('date', 'Unknown')}")
        print(f"📝 Content: {doc.page_content[:200]}...")
        print("")


def print_reranked_results(title: str, results):
    print(f"\n=== {title} ===")
    if not results:
        print("No results.")
        return

    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- [Result {i}] (Rerank Score: {score:.4f}) ---")
        print(f"🆔 Chunk ID: {get_chunk_id(doc)}")
        print(f"📅 Date: {doc.metadata.get('date', 'Unknown')}")
        print(f"📝 Content: {doc.page_content[:200]}...")
        print("")


def query_vector_db(question: str):
    print(f"🔎 Searching for: '{question}'...")
    
    # 强制离线模式，消除网络请求延迟
    os.environ["HF_HUB_OFFLINE"] = "1"

    vector_db, bm25_retriever, reranker = initialize_retrieval_components()
    retrieval_result = run_hybrid_retrieval(
        question,
        vector_db,
        bm25_retriever,
        reranker=reranker,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        fusion_top_k=FUSION_TOP_K,
        final_top_k=FINAL_TOP_K,
    )

    print_scored_results("Dense Retrieval (Chroma)", retrieval_result["dense"])
    print_scored_results("BM25 Retrieval", retrieval_result["bm25"])
    print_docs("Hybrid Retrieval (RRF Fusion)", retrieval_result["fused"])
    print_reranked_results("Reranked Results (Cross-Encoder)", retrieval_result["reranked"])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/query.py 'Your question here'")
        sys.exit(1)
    
    question = sys.argv[1]
    query_vector_db(question)
