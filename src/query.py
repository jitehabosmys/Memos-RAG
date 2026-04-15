import sys
import os
from rag import (
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    FUSION_TOP_K,
    MULTI_QUERY_BM25_TOP_K,
    MULTI_QUERY_DENSE_TOP_K,
    MULTI_QUERY_FUSION_TOP_K,
    MULTI_QUERY_GLOBAL_TOP_K,
    QUERY_REWRITE_COUNT,
    USE_QUERY_REWRITE,
    get_chunk_id,
    initialize_retrieval_components,
    run_hybrid_retrieval,
    run_multi_query_retrieval,
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


def print_lines(title: str, lines: list[str]):
    print(f"\n=== {title} ===")
    if not lines:
        print("No items.")
        return

    for i, line in enumerate(lines, start=1):
        print(f"{i}. {line}")


def query_vector_db(question: str):
    print(f"🔎 Searching for: '{question}'...")
    
    # 强制离线模式，消除网络请求延迟
    os.environ["HF_HUB_OFFLINE"] = "1"

    vector_db, bm25_retriever, reranker, rewrite_chain = initialize_retrieval_components()

    single_query_result = run_hybrid_retrieval(
        question,
        vector_db,
        bm25_retriever,
        reranker=reranker,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        fusion_top_k=FUSION_TOP_K,
        final_top_k=FINAL_TOP_K,
    )

    print_scored_results("Single Query Dense Retrieval (Chroma)", single_query_result["dense"])
    print_scored_results("Single Query BM25 Retrieval", single_query_result["bm25"])
    print_docs("Single Query Hybrid Retrieval (RRF Fusion)", single_query_result["fused"])
    print_reranked_results("Single Query Reranked Results", single_query_result["reranked"])

    if USE_QUERY_REWRITE:
        multi_query_result = run_multi_query_retrieval(
            question,
            vector_db,
            bm25_retriever,
            reranker=reranker,
            rewrite_chain=rewrite_chain,
            rewrite_count=QUERY_REWRITE_COUNT,
            dense_top_k=MULTI_QUERY_DENSE_TOP_K,
            bm25_top_k=MULTI_QUERY_BM25_TOP_K,
            per_query_fusion_top_k=MULTI_QUERY_FUSION_TOP_K,
            global_fusion_top_k=MULTI_QUERY_GLOBAL_TOP_K,
            final_top_k=FINAL_TOP_K,
        )

        print_lines("Generated Query Rewrites", multi_query_result["rewrites"])
        for entry in multi_query_result["by_query"]:
            print_docs(f"Per-Query Fused Results: {entry['query']}", entry["fused"])
        print_docs("Multi-Query Global Fusion", multi_query_result["global_fused"])
        print_reranked_results("Multi-Query Reranked Results", multi_query_result["reranked"])
    else:
        print("\nℹ️ Multi-query rewrite is disabled. Set RAG_USE_QUERY_REWRITE=true to enable it.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/query.py 'Your question here'")
        sys.exit(1)
    
    question = sys.argv[1]
    query_vector_db(question)
