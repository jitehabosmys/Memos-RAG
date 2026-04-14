#!/bin/bash
set -e

echo "📥 Step 1/2: Syncing SQLite database from remote..."
./sync_data.sh

echo "🧠 Step 2/2: Rebuilding local Chroma knowledge base..."
.venv/bin/python src/ingest.py

echo "✅ Local knowledge base is ready."
echo "💡 You can now test with: .venv/bin/python src/rag.py"
