"""SQLite-backed persistent conversation memory."""

import sqlite3
import time
from pathlib import Path
from typing import Optional


class ConversationMemory:
    def __init__(self, persist_path: str = "~/.aarnxen/conversations.db"):
        self._path = Path(persist_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                provider TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id);
        """)

    def add_message(
        self, conversation_id: str, role: str, content: str,
        model: Optional[str] = None, provider: Optional[str] = None,
    ):
        self._conn.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
            (conversation_id, time.time()),
        )
        self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, model, provider, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, model, provider, time.time()),
        )
        self._conn.commit()

    def get_history(self, conversation_id: str, max_turns: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT role, content, model, provider, timestamp "
            "FROM messages WHERE conversation_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (conversation_id, max_turns),
        ).fetchall()
        rows.reverse()
        return [
            {"role": r[0], "content": r[1], "model": r[2], "provider": r[3]}
            for r in rows
        ]

    def close(self):
        self._conn.close()
