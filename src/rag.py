import os
import sys
import math
import re
from collections import Counter, defaultdict
from typing import Iterable
from dotenv import load_dotenv

# 加载环境变量 (.env)
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sentence_transformers import CrossEncoder

from etl import fetch_all_memos, process_documents

# 配置路径
PERSIST_DIRECTORY = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = os.getenv("RAG_RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
DENSE_TOP_K = int(os.getenv("RAG_DENSE_TOP_K", "12"))
BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "12"))
FUSION_TOP_K = int(os.getenv("RAG_FUSION_TOP_K", "15"))
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "5"))
QUERY_REWRITE_COUNT = int(os.getenv("RAG_QUERY_REWRITE_COUNT", "2"))
MULTI_QUERY_DENSE_TOP_K = int(os.getenv("RAG_MULTI_QUERY_DENSE_TOP_K", "8"))
MULTI_QUERY_BM25_TOP_K = int(os.getenv("RAG_MULTI_QUERY_BM25_TOP_K", "8"))
MULTI_QUERY_FUSION_TOP_K = int(os.getenv("RAG_MULTI_QUERY_FUSION_TOP_K", "8"))
MULTI_QUERY_GLOBAL_TOP_K = int(os.getenv("RAG_MULTI_QUERY_GLOBAL_TOP_K", "24"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", "8"))


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


USE_RERANK = env_flag("RAG_USE_RERANK", True)
USE_QUERY_REWRITE = env_flag("RAG_USE_QUERY_REWRITE", False)


def tokenize_for_bm25(text: str) -> list[str]:
    """将中英文混合文本切分为适合 BM25 的 token。"""
    return re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]", text.lower())


def get_chunk_id(doc) -> str:
    """统一获取检索文档的稳定 ID。"""
    return doc.id or doc.metadata.get("chunk_id", "")


class SimpleBM25Retriever:
    """一个轻量 BM25 检索器，避免为混合检索额外引入新依赖。"""

    def __init__(self, documents):
        self.documents = documents
        self.doc_tokens = [tokenize_for_bm25(doc.page_content) for doc in documents]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.doc_freqs = defaultdict(int)
        self.k1 = 1.5
        self.b = 0.75

        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] += 1

    def retrieve(self, query: str, top_k: int):
        if not self.documents:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = []
        corpus_size = len(self.documents)

        for idx, term_freqs in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lens[idx]
            score = 0.0

            for token in query_tokens:
                term_freq = term_freqs.get(token, 0)
                if term_freq == 0:
                    continue

                doc_freq = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
                score += idf * (term_freq * (self.k1 + 1)) / denom

            if score > 0:
                scores.append((self.documents[idx], score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(result_lists: Iterable[list], top_k: int):
    """使用 RRF 融合多路检索结果。"""
    fused_scores = defaultdict(float)
    doc_by_id = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            doc = item[0] if isinstance(item, tuple) else item
            chunk_id = get_chunk_id(doc)
            if not chunk_id:
                continue

            doc_by_id[chunk_id] = doc
            fused_scores[chunk_id] += 1.0 / (RRF_K + rank)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_by_id[chunk_id] for chunk_id in ranked_ids[:top_k]]


def set_hf_offline_mode(required_models: list[str]):
    """根据本地缓存情况决定是否启用 HuggingFace 离线模式。"""
    # 智能离线模式控制
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        cached_repo_ids = {str(repo.repo_id) for repo in cache_info.repos}
        all_cached = all(
            any(model_name in repo_id for repo_id in cached_repo_ids)
            for model_name in required_models
        )
        if all_cached:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ["HF_HUB_OFFLINE"] = "0"
    except:
        os.environ["HF_HUB_OFFLINE"] = "0"


def rerank_documents(query: str, docs: list, reranker, top_k: int = FINAL_TOP_K):
    """使用 cross-encoder reranker 对候选文档精排。"""
    if not docs or reranker is None:
        return [(doc, 0.0) for doc in docs[:top_k]]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]


def create_chat_model(streaming: bool, temperature: float = 0.0):
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "glm-4.6"),
        temperature=temperature,
        streaming=streaming,
    )


def create_query_rewrite_chain():
    prompt = ChatPromptTemplate.from_template(
        """你是一个检索改写助手。

请把用户问题改写为 {rewrite_count} 条适合知识库检索的查询语句。

要求：
1. 保持原意，不要添加原问题没有的新信息
2. 尽量使用不同表达方式
3. 一条可以偏自然语言，一条可以偏关键词
4. 每行输出一条，不要编号，不要解释

用户问题：
{question}
"""
    )
    return prompt | create_chat_model(streaming=False, temperature=0.0) | StrOutputParser()


def normalize_rewrite_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
    return line.strip("` ").strip()


def parse_query_rewrites(raw_text: str, original_question: str, rewrite_count: int):
    rewrites = []
    seen = {original_question.strip()}

    for raw_line in raw_text.splitlines():
        line = normalize_rewrite_line(raw_line)
        if not line:
            continue
        if line.lower().startswith("用户问题"):
            continue
        if line in seen:
            continue
        rewrites.append(line)
        seen.add(line)
        if len(rewrites) >= rewrite_count:
            break

    return rewrites


def generate_query_rewrites(question: str, rewrite_chain=None, rewrite_count: int = QUERY_REWRITE_COUNT):
    if rewrite_chain is None or rewrite_count <= 0:
        return []

    try:
        raw_text = rewrite_chain.invoke({"question": question, "rewrite_count": rewrite_count})
        return parse_query_rewrites(raw_text, question, rewrite_count)
    except Exception as exc:
        print(f"⚠️ Query rewrite failed: {exc}")
        return []


def initialize_retrieval_components():
    """初始化混合检索所需的 Embedding、向量库、BM25 检索器、reranker 和 rewrite chain。"""
    required_models = [EMBEDDING_MODEL_NAME]
    if USE_RERANK:
        required_models.append(RERANKER_MODEL_NAME)
    set_hf_offline_mode(required_models)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="memos_rag"
    )

    bm25_documents = process_documents(fetch_all_memos())
    bm25_retriever = SimpleBM25Retriever(bm25_documents)
    reranker = None
    if USE_RERANK:
        reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    rewrite_chain = None
    if USE_QUERY_REWRITE:
        rewrite_chain = create_query_rewrite_chain()
    return vector_db, bm25_retriever, reranker, rewrite_chain


def run_hybrid_retrieval(
    query: str,
    vector_db,
    bm25_retriever,
    reranker=None,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    fusion_top_k: int = FUSION_TOP_K,
    final_top_k: int = FINAL_TOP_K,
):
    """执行 dense + BM25 混合召回，并在需要时执行 rerank。"""
    dense_results = vector_db.similarity_search_with_score(query, k=dense_top_k)
    bm25_results = bm25_retriever.retrieve(query, top_k=bm25_top_k)
    fused_docs = reciprocal_rank_fusion([dense_results, bm25_results], top_k=fusion_top_k)
    reranked_results = rerank_documents(query, fused_docs, reranker, top_k=final_top_k)
    return {
        "dense": dense_results,
        "bm25": bm25_results,
        "fused": fused_docs,
        "reranked": reranked_results,
    }


def run_multi_query_retrieval(
    question: str,
    vector_db,
    bm25_retriever,
    reranker=None,
    rewrite_chain=None,
    rewrite_count: int = QUERY_REWRITE_COUNT,
    dense_top_k: int = MULTI_QUERY_DENSE_TOP_K,
    bm25_top_k: int = MULTI_QUERY_BM25_TOP_K,
    per_query_fusion_top_k: int = MULTI_QUERY_FUSION_TOP_K,
    global_fusion_top_k: int = MULTI_QUERY_GLOBAL_TOP_K,
    final_top_k: int = FINAL_TOP_K,
):
    rewrites = generate_query_rewrites(question, rewrite_chain, rewrite_count)
    queries = [question] + rewrites

    by_query = []
    per_query_fused_lists = []
    for query_text in queries:
        query_result = run_hybrid_retrieval(
            query_text,
            vector_db,
            bm25_retriever,
            reranker=None,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            fusion_top_k=per_query_fusion_top_k,
            final_top_k=per_query_fusion_top_k,
        )
        by_query.append({"query": query_text, **query_result})
        per_query_fused_lists.append(query_result["fused"])

    global_fused_docs = reciprocal_rank_fusion(per_query_fused_lists, top_k=global_fusion_top_k)
    reranked_results = rerank_documents(question, global_fused_docs, reranker, top_k=final_top_k)
    return {
        "question": question,
        "rewrites": rewrites,
        "queries": queries,
        "by_query": by_query,
        "global_fused": global_fused_docs,
        "reranked": reranked_results,
    }

def format_docs(docs):
    """将检索到的文档格式化为字符串，包含日期信息"""
    formatted = []
    for doc in docs:
        date = doc.metadata.get("date", "Unknown")
        content = doc.page_content
        formatted.append(f"[日期: {date}]\n{content}")
    return "\n\n---\n\n".join(formatted)

def get_rag_chain():
    """初始化并返回 RAG 处理链"""
    print("🧠 Initializing Second Brain Core...")
    vector_db, bm25_retriever, reranker, rewrite_chain = initialize_retrieval_components()

    def hybrid_retrieve(query: str):
        if USE_QUERY_REWRITE:
            retrieval_result = run_multi_query_retrieval(
                query,
                vector_db,
                bm25_retriever,
                reranker=reranker,
                rewrite_chain=rewrite_chain,
            )
            return [doc for doc, _ in retrieval_result["reranked"]]

        retrieval_result = run_hybrid_retrieval(query, vector_db, bm25_retriever, reranker=reranker)
        return [doc for doc, _ in retrieval_result["reranked"]]

    # 4. 初始化 LLM
    llm = create_chat_model(streaming=True, temperature=0.3)

    # 5. 定义 Prompt
    template = """你是一个基于我的 Memos 笔记构建的【个人第二大脑】。
    
    请根据以下【相关的笔记片段】来回答我的问题。
    如果笔记中没有相关内容，请诚实地告诉我“我的记忆库里没有相关记录”，不要编造。
    
    回答时请引用笔记中的日期，以证明你的来源。
    
    【相关的笔记片段】:
    {context}
    
    【我的问题】: {question}
    
    【你的回答】:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. 构建 RAG 链 (LCEL)
    rag_chain = (
        {"context": RunnableLambda(hybrid_retrieve) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    # 1. 检查 API Key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-...":
        print("⚠️  Warning: OPENAI_API_KEY not set in .env file.")
    
    # 2. 获取处理链
    rag_chain = get_rag_chain()

    print("✅ System Ready! (Type 'exit' to quit)")
    print("-" * 50)

    # 3. 聊天循环
    while True:
        try:
            user_input = input("\n🧑 You: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Bye!")
                break
            
            if not user_input.strip():
                continue

            print("🤖 Brain: ", end="", flush=True)
            
            # 流式输出
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
