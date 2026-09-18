"""SQLite-backed conversation history service."""

from __future__ import annotations

import sqlite3
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "conversations.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _ensure_schema(conn)
        _local.conn = conn
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id   TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            thread_id   TEXT NOT NULL,
            role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content     TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES conversations(thread_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at);
        CREATE TABLE IF NOT EXISTS qa_conversations (
            id          TEXT PRIMARY KEY,
            topic       TEXT NOT NULL DEFAULT '',
            topic_title TEXT NOT NULL DEFAULT '',
            backend     TEXT NOT NULL DEFAULT 'auto',
            mode        TEXT NOT NULL DEFAULT 'mix',
            title       TEXT NOT NULL DEFAULT '',
            payload     TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_qa_conversations_updated ON qa_conversations(updated_at);
    """)
    conn.commit()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ── write ───────────────────────────────────────────────────────────────


def ensure_conversation(thread_id: str, *, title: str = "") -> dict[str, Any]:
    """Create a conversation record if it doesn't exist yet."""
    conn = _get_conn()
    now = _now()
    conn.execute(
        "INSERT OR IGNORE INTO conversations (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (thread_id, title or thread_id, now, now),
    )
    conn.commit()
    return _row_to_dict(conn.execute("SELECT * FROM conversations WHERE thread_id = ?", (thread_id,)).fetchone())


def save_message(thread_id: str, role: str, content: str) -> dict[str, Any]:
    """Append a message to a conversation, creating the conversation if needed."""
    conn = _get_conn()
    ensure_conversation(thread_id)
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = _now()
    conn.execute(
        "INSERT INTO messages (id, thread_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, thread_id, role, content, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE thread_id = ?",
        (now, thread_id),
    )
    conn.commit()
    return {"id": msg_id, "thread_id": thread_id, "role": role, "content": content, "created_at": now}


def update_conversation_title(thread_id: str, title: str) -> None:
    """Update the title of a conversation."""
    conn = _get_conn()
    conn.execute("UPDATE conversations SET title = ? WHERE thread_id = ?", (title, thread_id))
    conn.commit()


# ── read ────────────────────────────────────────────────────────────────


def list_conversations(*, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List conversations ordered by most recently updated."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (max(1, min(limit, 200)), max(0, offset)),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_conversation(thread_id: str) -> dict[str, Any] | None:
    """Get a single conversation by thread_id."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM conversations WHERE thread_id = ?", (thread_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_messages(thread_id: str, *, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    """Get messages for a conversation ordered by time."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?",
        (thread_id, max(1, min(limit, 500)), max(0, offset)),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_conversations() -> int:
    """Return total conversation count."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()
    return row["cnt"] if row else 0


def list_qa_conversations(*, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM qa_conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (max(1, min(limit, 200)), max(0, offset)),
    ).fetchall()
    result = []
    for row in rows:
        item = _row_to_dict(row)
        try:
            item["messages"] = json.loads(item.pop("payload") or "[]")
        except (TypeError, json.JSONDecodeError):
            item["messages"] = []
        result.append(item)
    return result


def count_qa_conversations() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) AS cnt FROM qa_conversations").fetchone()
    return int(row["cnt"]) if row else 0


def upsert_qa_conversation(payload: dict[str, Any]) -> dict[str, Any]:
    conn = _get_conn()
    now = _now()
    session_id = str(payload.get("id") or f"qa_{uuid.uuid4().hex[:16]}")
    created_at = str(payload.get("createdAt") or now)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    conn.execute(
        """INSERT INTO qa_conversations
        (id, topic, topic_title, backend, mode, title, payload, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET topic=excluded.topic, topic_title=excluded.topic_title,
        backend=excluded.backend, mode=excluded.mode, title=excluded.title,
        payload=excluded.payload, updated_at=excluded.updated_at""",
        (
            session_id,
            str(payload.get("topic") or ""),
            str(payload.get("topicTitle") or payload.get("topic") or ""),
            str(payload.get("backend") or "auto"),
            str(payload.get("mode") or "mix"),
            str(payload.get("title") or "未命名对话"),
            json.dumps(messages, ensure_ascii=False),
            created_at,
            str(payload.get("updatedAt") or now),
        ),
    )
    conn.commit()
    return next(item for item in list_qa_conversations(limit=200) if item["id"] == session_id)


def delete_qa_conversation(session_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM qa_conversations WHERE id = ?", (session_id,))
    conn.commit()
    return cursor.rowcount > 0


# ── delete ──────────────────────────────────────────────────────────────


def delete_conversation(thread_id: str) -> bool:
    """Delete a conversation and all its messages. Returns True if found."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
    conn.commit()
    return cursor.rowcount > 0


# ── helpers ─────────────────────────────────────────────────────────────


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)
