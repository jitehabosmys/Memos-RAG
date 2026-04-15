import json
import os
from datetime import datetime
from pathlib import Path

from etl import DB_PATH, fetch_all_memos, process_documents

OUTPUT_DIR = Path(os.getenv("CHUNK_INDEX_OUTPUT_DIR", "data"))
JSON_OUTPUT_PATH = OUTPUT_DIR / "chunk_index.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "chunk_index.md"


def build_chunk_index():
    memos = fetch_all_memos()
    chunks = process_documents(memos)

    memo_index = {}
    flat_chunks = []

    for memo in memos:
        memo_index[memo["id"]] = {
            "memo_id": memo["id"],
            "date": memo["date_str"],
            "created_ts": memo["created_ts"],
            "full_content": memo["content"],
            "chunk_ids": [],
            "chunks": [],
        }

    for chunk in chunks:
        memo_id = chunk.metadata["memo_id"]
        chunk_id = chunk.id
        chunk_entry = {
            "chunk_id": chunk_id,
            "memo_id": memo_id,
            "date": chunk.metadata.get("date"),
            "chunk_content": chunk.page_content,
        }

        memo_index[memo_id]["chunk_ids"].append(chunk_id)
        memo_index[memo_id]["chunks"].append(chunk_entry)
        flat_chunks.append(chunk_entry)

    ordered_memos = list(memo_index.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_db": str(DB_PATH),
        "memo_count": len(ordered_memos),
        "chunk_count": len(flat_chunks),
        "memos": ordered_memos,
        "chunks": flat_chunks,
    }


def render_markdown(index_data: dict) -> str:
    lines = [
        "# Chunk Index",
        "",
        f"- Generated At: {index_data['generated_at']}",
        f"- Source DB: `{index_data['source_db']}`",
        f"- Memo Count: {index_data['memo_count']}",
        f"- Chunk Count: {index_data['chunk_count']}",
        "",
        "这份索引用于人工设计评测问题，方便查看每条 memo 被切成了哪些 chunk，以及每个 chunk 对应的稳定 ID。",
        "",
    ]

    for memo in index_data["memos"]:
        lines.extend(
            [
                f"## Memo {memo['memo_id']}",
                "",
                f"- Date: {memo['date']}",
                f"- Chunk IDs: {', '.join(memo['chunk_ids']) if memo['chunk_ids'] else 'None'}",
                "",
                "### Full Memo",
                "",
                memo["full_content"],
                "",
                "### Chunks",
                "",
            ]
        )

        if not memo["chunks"]:
            lines.append("No chunks.")
            lines.append("")
            continue

        for chunk in memo["chunks"]:
            lines.extend(
                [
                    f"#### {chunk['chunk_id']}",
                    "",
                    chunk["chunk_content"],
                    "",
                ]
            )

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_data = build_chunk_index()

    JSON_OUTPUT_PATH.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT_PATH.write_text(render_markdown(index_data), encoding="utf-8")

    print(f"✅ Exported chunk index JSON to: {JSON_OUTPUT_PATH}")
    print(f"✅ Exported chunk index Markdown to: {MARKDOWN_OUTPUT_PATH}")
    print(f"📊 Memos: {index_data['memo_count']}, Chunks: {index_data['chunk_count']}")


if __name__ == "__main__":
    main()
