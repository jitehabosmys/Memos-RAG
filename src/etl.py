import sqlite3
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = Path(os.getenv("MEMOS_DB_PATH", "data/memos.db"))

def clean_text(text: str) -> str:
    """
    清洗 Memos 内容：去除 Markdown 图片和链接标记，保留纯文本。
    """
    if not text:
        return ""
    
    # 1. 去除图片标记 ![alt](url)
    text = re.sub(r'!\\[.*?\\]\(.*?\\]\)', '', text)
    
    # 2. 去除链接标记 [text](url) -> 只保留 text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 3. 去除多余的空白字符
    return text.strip()

def format_timestamp(ts: int) -> str:
    """将 Unix 时间戳转换为 YYYY-MM-DD HH:MM:SS 格式"""
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def fetch_all_memos() -> List[Dict[str, Any]]:
    """从 SQLite 数据库获取所有有效笔记，包含 ID"""
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        return []

    print(f"🔌 Connecting to database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # 获取 id, content, created_ts
        sql = "SELECT id, content, created_ts FROM memo ORDER BY created_ts DESC"
        cursor.execute(sql)
        rows = cursor.fetchall()

    data = []
    for id, content, created_ts in rows:
        cleaned_content = clean_text(content)
        # 过滤掉过短的无意义内容
        if len(cleaned_content) >= 5:
            data.append({
                "id": id,
                "content": cleaned_content,
                "created_ts": created_ts,
                "date_str": format_timestamp(created_ts)
            })
    
    return data

def process_documents(memos: List[Dict[str, Any]]) -> List[Document]:
    """将笔记转换为 LangChain Document 对象并进行切分，赋予稳定 ID"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    
    final_chunks = []
    
    for memo in memos:
        # 1. 构造单个 Document
        metadata = {
            "source": "memos",
            "memo_id": memo["id"],  # 记录原始 ID
            "created_ts": memo["created_ts"],
            "date": memo["date_str"]
        }
        raw_doc = Document(page_content=memo["content"], metadata=metadata)
        
        # 2. 对单篇笔记进行切分
        doc_chunks = text_splitter.split_documents([raw_doc])
        
        # 3. 为切片分配稳定的 ID
        for i, chunk in enumerate(doc_chunks):
            # 格式: memo_{原始ID}_{切片序号}
            # 例如: memo_123_0, memo_123_1
            stable_id = f"memo_{memo['id']}_{i}"
            
            # 将 ID 同时写入 metadata (可选) 和 id 属性 (关键)
            chunk.metadata["chunk_id"] = stable_id
            chunk.id = stable_id 
            
            final_chunks.append(chunk)

    print(f"✂️  Processed {len(memos)} memos into {len(final_chunks)} chunks with stable IDs.")
    return final_chunks

def main():
    # 1. ETL
    memos = fetch_all_memos()
    print(f"✅ Fetched {len(memos)} valid memos from DB.")
    
    if not memos:
        return

    # 2. Chunking
    chunks = process_documents(memos)

    # 3. Preview
    print("\n--- 🔍 Chunk Preview (Top 3) ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}] ID: {chunk.id}")
        print(f"📅 Date: {chunk.metadata['date']}")
        print(f"📝 Content: {chunk.page_content[:100]}..." if len(chunk.page_content) > 100 else f"📝 Content: {chunk.page_content}")
        print("-" * 40)

if __name__ == "__main__":
    main()
