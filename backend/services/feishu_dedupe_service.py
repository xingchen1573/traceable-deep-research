"""Persistent de-duplication store for inbound Feishu message events.

Feishu's long-connection (WS) transport redelivers old events on its own schedule --
observed gaps range from 46 minutes to several hours, with no reconnect/ping-timeout
log anywhere near some of the redeliveries, so it is not just "reconnect replays the
last event". An in-memory sliding window (the previous `_SEEN_MESSAGE_IDS` dict in
feishu_service.py) cannot cover this: it resets on every worker restart and only
remembers the last `FEISHU_MESSAGE_DEDUPE_SECONDS` (600s) seconds, far shorter than
the redelivery gaps actually seen in production.

This module persists seen message_ids to the same on-disk SQLite database
channel_service.py/conversation_service.py already share (`data/conversations.db`,
mounted as the `app_data` volume in both docker-compose services), so the record
survives worker restarts and is visible across processes/containers on the same host.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "conversations.db"

_local = threading.local()

# "至少一天以上" per the redelivery gaps actually observed (up to several hours between
# redeliveries of the same message_id) -- 48h leaves headroom over any single observed gap.
DEFAULT_TTL_SECONDS = 172800.0


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(conn)
        _local.conn = conn
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feishu_seen_messages (
            message_id TEXT PRIMARY KEY,
            seen_at    REAL NOT NULL
        );
        """
    )
    conn.commit()


def mark_seen_or_duplicate(message_id: str | None, *, ttl_seconds: float = DEFAULT_TTL_SECONDS, now: float | None = None) -> bool:
    """Record message_id as seen; return True if it was already seen within ttl_seconds.

    Uses `INSERT ... ON CONFLICT DO NOTHING` so the check-and-record is a single atomic
    statement -- safe even if webhook and long-connection paths raced on the same
    message_id from different processes sharing this file.
    """
    if not message_id:
        return False
    conn = _get_conn()
    current = time.time() if now is None else float(now)
    cutoff = current - max(0.0, float(ttl_seconds))
    conn.execute("DELETE FROM feishu_seen_messages WHERE seen_at < ?", (cutoff,))
    cursor = conn.execute(
        "INSERT INTO feishu_seen_messages (message_id, seen_at) VALUES (?, ?) "
        "ON CONFLICT(message_id) DO NOTHING",
        (message_id, current),
    )
    conn.commit()
    return cursor.rowcount == 0


def clear_seen_messages() -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM feishu_seen_messages")
    conn.commit()


def count_seen_messages() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM feishu_seen_messages").fetchone()
    return int(row["cnt"] if row else 0)
