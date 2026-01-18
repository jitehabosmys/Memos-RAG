import os
import sys
from dotenv import load_dotenv

# 加载环境变量 (.env)
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 配置路径
PERSIST_DIRECTORY = "./data/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

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

    # 强制离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 1. 加载 Embedding (用于检索)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 加载 Vector Store
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="memos_rag"
    )
    
    # 转换为 Retriever (检索器)                                                       │
    # search_kwargs{"k": 5} 表示每次检索前 5 条相关笔记
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 3. 初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
        temperature=0.3,
        streaming=True
    )

    # 4. 定义 Prompt
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

    # 5. 构建 RAG 链 (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
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