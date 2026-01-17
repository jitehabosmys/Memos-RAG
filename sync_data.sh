#!/bin/bash

# --- 配置区 ---
# 请确保你的 ~/.ssh/config 中配置了名为 "memos" 的 Host
REMOTE_ALIAS="memos"
REMOTE_DB_PATH="/home/sytssmys/.memos/memos_prod.db"
LOCAL_BACKUP_DIR="$HOME/memos-rag/data"
TEMP_REMOTE_DB="/tmp/memos_backup.db"

# 1. 在服务器上创建热备份副本 (防止数据锁定)
ssh $REMOTE_ALIAS "sqlite3 $REMOTE_DB_PATH '.backup $TEMP_REMOTE_DB'"

# 2. 将副本拉回到本地
scp $REMOTE_ALIAS:$TEMP_REMOTE_DB $LOCAL_BACKUP_DIR/memos.db

# 3. 删除服务器上的临时文件
ssh $REMOTE_ALIAS "rm $TEMP_REMOTE_DB"

echo "✅ Successfully synced from remote server!"
