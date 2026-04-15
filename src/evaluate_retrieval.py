import json
import os
from pathlib import Path

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

EVAL_SET_PATH = Path(os.getenv("RAG_EVAL_SET_PATH", "eval/retrieval_eval_set.json"))
TOP_K_VALUES = (1, 3, 5)
MIN_EVAL_TOP_K = max(TOP_K_VALUES)
SHOW_TOP5_MISSES = int(os.getenv("RAG_EVAL_SHOW_TOP5_MISSES", "5"))


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def result_items_to_docs(items) -> list:
    docs = []
    for item in items:
        docs.append(item[0] if isinstance(item, tuple) else item)
    return docs


def memo_id_from_doc(doc):
    return doc.metadata.get("memo_id")


def find_first_match_rank(docs: list, expected_chunk_ids: set[str], expected_memo_ids: set[int], match_mode: str):
    for rank, doc in enumerate(docs, start=1):
        chunk_id = get_chunk_id(doc)
        memo_id = memo_id_from_doc(doc)
        if match_mode == "chunk" and chunk_id in expected_chunk_ids:
            return rank
        if match_mode == "memo" and memo_id in expected_memo_ids:
            return rank
    return None


def build_metrics():
    return {
        "count": 0,
        "hit": {k: 0 for k in TOP_K_VALUES},
        "mrr": 0.0,
        "top5_misses": [],
    }


def update_metrics(metrics: dict, sample: dict, docs: list, match_mode: str):
    expected_chunk_ids = set(sample["expected_chunk_ids"])
    expected_memo_ids = set(sample["expected_memo_ids"])
    first_rank = find_first_match_rank(docs, expected_chunk_ids, expected_memo_ids, match_mode)

    metrics["count"] += 1
    if first_rank is not None:
        metrics["mrr"] += 1.0 / first_rank
        for k in TOP_K_VALUES:
            if first_rank <= k:
                metrics["hit"][k] += 1
        if first_rank > MIN_EVAL_TOP_K:
            metrics["top5_misses"].append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "expected_chunk_ids": sample["expected_chunk_ids"],
                    "expected_memo_ids": sample["expected_memo_ids"],
                    "first_match_rank": first_rank,
                    "top_results": [
                        {
                            "chunk_id": get_chunk_id(doc),
                            "memo_id": memo_id_from_doc(doc),
                        }
                        for doc in docs[:MIN_EVAL_TOP_K]
                    ],
                }
            )
    else:
        metrics["top5_misses"].append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "expected_chunk_ids": sample["expected_chunk_ids"],
                "expected_memo_ids": sample["expected_memo_ids"],
                "top_results": [
                    {
                        "chunk_id": get_chunk_id(doc),
                        "memo_id": memo_id_from_doc(doc),
                    }
                    for doc in docs[:MIN_EVAL_TOP_K]
                ],
            }
        )


def finalize_metrics(metrics: dict):
    count = metrics["count"] or 1
    summary = {
        "count": metrics["count"],
        "mrr": metrics["mrr"] / count,
        "hit": {f"hit@{k}": metrics["hit"][k] / count for k in TOP_K_VALUES},
        "top5_misses": metrics["top5_misses"],
    }
    return summary


def print_summary(title: str, summary: dict):
    print(f"\n=== {title} ===")
    print(f"Samples: {summary['count']}")
    print(f"MRR: {summary['mrr']:.4f}")
    for k in TOP_K_VALUES:
        print(f"Hit@{k}: {summary['hit'][f'hit@{k}']:.4f}")

    top5_misses = summary["top5_misses"][:SHOW_TOP5_MISSES]
    if top5_misses:
        print("Sample Top5 misses:")
        for miss in top5_misses:
            top_results = ", ".join(item["chunk_id"] for item in miss["top_results"])
            print(f"- {miss['id']} {miss['question']}")
            print(f"  expected chunks: {miss['expected_chunk_ids']}")
            print(f"  top results: {top_results}")
            if miss.get("first_match_rank") is not None:
                print(f"  first match rank: {miss['first_match_rank']}")


def evaluate():
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Eval set not found: {EVAL_SET_PATH}")

    print(f"📂 Loading eval set from: {EVAL_SET_PATH}", flush=True)
    eval_set = load_eval_set()
    print(f"📌 Questions loaded: {len(eval_set)}", flush=True)

    print("🧠 Initializing retrieval components...", flush=True)
    vector_db, bm25_retriever, reranker, rewrite_chain = initialize_retrieval_components()
    print("✅ Retrieval components ready.", flush=True)

    methods = {
        "dense_chunk": build_metrics(),
        "bm25_chunk": build_metrics(),
        "rrf_chunk": build_metrics(),
        "rerank_chunk": build_metrics(),
        "dense_memo": build_metrics(),
        "bm25_memo": build_metrics(),
        "rrf_memo": build_metrics(),
        "rerank_memo": build_metrics(),
    }
    if USE_QUERY_REWRITE:
        methods["multiquery_rrf_chunk"] = build_metrics()
        methods["multiquery_rerank_chunk"] = build_metrics()
        methods["multiquery_rrf_memo"] = build_metrics()
        methods["multiquery_rerank_memo"] = build_metrics()

    total = len(eval_set)
    for idx, sample in enumerate(eval_set, start=1):
        print(f"⏳ Evaluating {sample['id']} ({idx}/{total}): {sample['question']}", flush=True)
        retrieval_result = run_hybrid_retrieval(
            sample["question"],
            vector_db,
            bm25_retriever,
            reranker=reranker,
            dense_top_k=max(DENSE_TOP_K, MIN_EVAL_TOP_K),
            bm25_top_k=max(BM25_TOP_K, MIN_EVAL_TOP_K),
            fusion_top_k=max(FUSION_TOP_K, MIN_EVAL_TOP_K),
            final_top_k=max(FINAL_TOP_K, MIN_EVAL_TOP_K),
        )

        dense_docs = result_items_to_docs(retrieval_result["dense"])
        bm25_docs = result_items_to_docs(retrieval_result["bm25"])
        rrf_docs = retrieval_result["fused"]
        rerank_docs = result_items_to_docs(retrieval_result["reranked"])

        update_metrics(methods["dense_chunk"], sample, dense_docs, "chunk")
        update_metrics(methods["bm25_chunk"], sample, bm25_docs, "chunk")
        update_metrics(methods["rrf_chunk"], sample, rrf_docs, "chunk")
        update_metrics(methods["rerank_chunk"], sample, rerank_docs, "chunk")

        update_metrics(methods["dense_memo"], sample, dense_docs, "memo")
        update_metrics(methods["bm25_memo"], sample, bm25_docs, "memo")
        update_metrics(methods["rrf_memo"], sample, rrf_docs, "memo")
        update_metrics(methods["rerank_memo"], sample, rerank_docs, "memo")

        if USE_QUERY_REWRITE:
            multi_query_result = run_multi_query_retrieval(
                sample["question"],
                vector_db,
                bm25_retriever,
                reranker=reranker,
                rewrite_chain=rewrite_chain,
                rewrite_count=QUERY_REWRITE_COUNT,
                dense_top_k=max(MULTI_QUERY_DENSE_TOP_K, MIN_EVAL_TOP_K),
                bm25_top_k=max(MULTI_QUERY_BM25_TOP_K, MIN_EVAL_TOP_K),
                per_query_fusion_top_k=max(MULTI_QUERY_FUSION_TOP_K, MIN_EVAL_TOP_K),
                global_fusion_top_k=max(MULTI_QUERY_GLOBAL_TOP_K, MIN_EVAL_TOP_K),
                final_top_k=max(FINAL_TOP_K, MIN_EVAL_TOP_K),
            )
            multi_rrf_docs = multi_query_result["global_fused"]
            multi_rerank_docs = result_items_to_docs(multi_query_result["reranked"])

            update_metrics(methods["multiquery_rrf_chunk"], sample, multi_rrf_docs, "chunk")
            update_metrics(methods["multiquery_rerank_chunk"], sample, multi_rerank_docs, "chunk")
            update_metrics(methods["multiquery_rrf_memo"], sample, multi_rrf_docs, "memo")
            update_metrics(methods["multiquery_rerank_memo"], sample, multi_rerank_docs, "memo")

        if idx % 5 == 0 or idx == total:
            print(f"✅ Progress: {idx}/{total} questions evaluated.", flush=True)

    summaries = {name: finalize_metrics(metrics) for name, metrics in methods.items()}

    print(f"\n📊 Loaded eval set: {EVAL_SET_PATH}")
    print(f"📌 Questions: {len(eval_set)}")
    print_summary("Dense Retrieval (Chunk Match)", summaries["dense_chunk"])
    print_summary("BM25 Retrieval (Chunk Match)", summaries["bm25_chunk"])
    print_summary("RRF Fusion (Chunk Match)", summaries["rrf_chunk"])
    print_summary("Rerank (Chunk Match)", summaries["rerank_chunk"])
    if USE_QUERY_REWRITE:
        print_summary("Multi-Query Global Fusion (Chunk Match)", summaries["multiquery_rrf_chunk"])
        print_summary("Multi-Query Rerank (Chunk Match)", summaries["multiquery_rerank_chunk"])
    print_summary("Dense Retrieval (Memo Match)", summaries["dense_memo"])
    print_summary("BM25 Retrieval (Memo Match)", summaries["bm25_memo"])
    print_summary("RRF Fusion (Memo Match)", summaries["rrf_memo"])
    print_summary("Rerank (Memo Match)", summaries["rerank_memo"])
    if USE_QUERY_REWRITE:
        print_summary("Multi-Query Global Fusion (Memo Match)", summaries["multiquery_rrf_memo"])
        print_summary("Multi-Query Rerank (Memo Match)", summaries["multiquery_rerank_memo"])


if __name__ == "__main__":
    evaluate()
