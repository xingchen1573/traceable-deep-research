"""Small persistent store for the latest structured conversational research plan."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "conversations.db"
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_plan_drafts (
                thread_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                user_request TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                job_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        _local.conn = conn
    return conn


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def save_latest_plan(thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
    confirmed_plan = result.get("confirmed_plan") if isinstance(result.get("confirmed_plan"), dict) else None
    plan_id = str(result.get("plan_id") or "").strip()
    if not thread_id or not confirmed_plan or not plan_id:
        return {}
    now = _now()
    payload = json.dumps(result, ensure_ascii=False)
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO research_plan_drafts
            (thread_id, plan_id, user_request, payload, status, job_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'draft', NULL, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            plan_id=excluded.plan_id,
            user_request=excluded.user_request,
            payload=excluded.payload,
            status='draft',
            job_id=NULL,
            updated_at=excluded.updated_at
        """,
        (
            thread_id,
            plan_id,
            str(confirmed_plan.get("user_request") or ""),
            payload,
            now,
            now,
        ),
    )
    conn.commit()
    return {"thread_id": thread_id, "plan_id": plan_id, "status": "draft", "updated_at": now}


def get_latest_plan(thread_id: str) -> dict[str, Any] | None:
    row = _get_conn().execute(
        "SELECT payload, status, job_id, updated_at FROM research_plan_drafts WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload["persistence_status"] = row["status"]
    payload["started_job_id"] = row["job_id"]
    payload["persisted_at"] = row["updated_at"]
    return payload


def mark_plan_started(thread_id: str, plan_id: str, job_id: str) -> None:
    if not thread_id or not plan_id or not job_id:
        return
    conn = _get_conn()
    conn.execute(
        """
        UPDATE research_plan_drafts
        SET status='started', job_id=?, updated_at=?
        WHERE thread_id=? AND plan_id=?
        """,
        (job_id, _now(), thread_id, plan_id),
    )
    conn.commit()
