"""SQLite-backed channel runtime state."""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "conversations.db"

_local = threading.local()
VALID_MODES = {"main", "kb_qa"}
VALID_MEMORY_SCOPES = {"channel"}
VALID_KNOWLEDGE_SCOPE_TYPES = {"knowledge_base", "theme"}

# docs/VISION.md 第5节: preset push-frequency tiers, mapped to a minimum elapsed-hours gap
# rather than fixed calendar weekdays -- self-healing after downtime, no cron expressions.
VALID_FREQUENCIES = {"daily", "weekly_3x", "weekly_1x"}
VALID_CONTENT_STRATEGIES = {"rotation", "new_only", "random"}
FREQUENCY_MIN_GAP_HOURS = {"daily": 20, "weekly_3x": 52, "weekly_1x": 150}
DEFAULT_PUSH_TIME = "09:00"
DEFAULT_PUSH_TIMEZONE = "Asia/Shanghai"


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        _ensure_schema(conn)
        _local.conn = conn
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id   TEXT PRIMARY KEY,
            platform     TEXT NOT NULL DEFAULT 'local',
            external_id  TEXT NOT NULL,
            title        TEXT NOT NULL DEFAULT '',
            mode         TEXT NOT NULL DEFAULT 'main',
            active_kb    TEXT,
            topic_binding TEXT,
            scope_type   TEXT,
            scope_id     TEXT,
            memory_scope TEXT NOT NULL DEFAULT 'channel',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_channels_updated_at ON channels(updated_at);
        CREATE TABLE IF NOT EXISTS channel_push_config (
            channel_id       TEXT PRIMARY KEY,
            enabled          INTEGER NOT NULL DEFAULT 0,
            frequency        TEXT NOT NULL DEFAULT 'daily',
            content_strategy TEXT NOT NULL DEFAULT 'rotation',
            push_time        TEXT NOT NULL DEFAULT '09:00',
            timezone         TEXT NOT NULL DEFAULT 'Asia/Shanghai',
            next_push_at     REAL,
            last_pushed_at   REAL,
            last_pushed_doc  TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );
    """)
    conn.commit()
    # topic_binding was added after channels shipped -- CREATE TABLE IF NOT EXISTS above is a
    # no-op against an already-existing data/conversations.db, so existing rows need an
    # explicit migration rather than just the literal column in the statement above.
    _ensure_column(conn, "channels", "topic_binding", "TEXT")
    _ensure_column(conn, "channels", "scope_type", "TEXT")
    _ensure_column(conn, "channels", "scope_id", "TEXT")
    _ensure_column(conn, "channel_push_config", "push_time", "TEXT NOT NULL DEFAULT '09:00'")
    _ensure_column(conn, "channel_push_config", "timezone", "TEXT NOT NULL DEFAULT 'Asia/Shanghai'")
    _ensure_column(conn, "channel_push_config", "next_push_at", "REAL")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        conn.commit()
    except sqlite3.OperationalError:
        # Separate processes/threads may open the shared SQLite database together during a
        # deployment restart. Only suppress the race when another connection actually added
        # the same column; unrelated migration failures must remain visible.
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            raise


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def build_channel_id(platform: str | None, external_id: str | None) -> str:
    safe_platform = _validate_segment(platform or "local", field="platform")
    safe_external = _validate_segment(external_id or "", field="external_id", allow_colon=True)
    if not safe_external:
        raise ValueError("external_id is required")
    return f"{safe_platform}:{safe_external}"


def thread_id_for(channel_id: str, mode: str = "main") -> str:
    safe_channel = _validate_channel_id(channel_id)
    safe_mode = _validate_mode(mode)
    return f"channel:{safe_channel}:{safe_mode}"


def ensure_channel(
    *,
    channel_id: str | None = None,
    platform: str | None = None,
    external_id: str | None = None,
    title: str | None = None,
    memory_scope: str = "channel",
) -> dict[str, Any]:
    safe_channel_id = _validate_channel_id(channel_id) if channel_id else build_channel_id(platform, external_id)
    if ":" in safe_channel_id:
        channel_platform, channel_external = safe_channel_id.split(":", 1)
    else:
        channel_platform, channel_external = platform or "local", external_id or safe_channel_id
    channel_platform = _validate_segment(platform or channel_platform or "local", field="platform")
    channel_external = _validate_segment(external_id or channel_external or safe_channel_id, field="external_id", allow_colon=True)
    safe_scope = _validate_memory_scope(memory_scope)
    now = now_text()
    display_title = (title or channel_external or safe_channel_id).strip()
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR IGNORE INTO channels
        (channel_id, platform, external_id, title, mode, active_kb, memory_scope, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'main', NULL, ?, ?, ?)
        """,
        (safe_channel_id, channel_platform, channel_external, display_title, safe_scope, now, now),
    )
    if title:
        conn.execute(
            "UPDATE channels SET title = ?, updated_at = ? WHERE channel_id = ?",
            (display_title, now, safe_channel_id),
        )
    conn.commit()
    return get_channel(safe_channel_id)


def get_channel(channel_id: str) -> dict[str, Any]:
    conn = _get_conn()
    safe_channel_id = _validate_channel_id(channel_id)
    row = conn.execute("SELECT * FROM channels WHERE channel_id = ?", (safe_channel_id,)).fetchone()
    if row is None:
        raise ValueError(f"channel not found: {safe_channel_id}")
    return _row_to_dict(row)


def list_channels(*, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM channels ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (max(1, min(limit, 200)), max(0, offset)),
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def count_channels() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM channels").fetchone()
    return int(row["cnt"] if row else 0)


def update_channel(
    channel_id: str,
    *,
    title: str | None = None,
    mode: str | None = None,
    active_kb: str | None = None,
    topic_binding: str | None = None,
    memory_scope: str | None = None,
) -> dict[str, Any]:
    safe_channel_id = _validate_channel_id(channel_id)
    updates: list[str] = []
    values: list[Any] = []
    if title is not None:
        updates.append("title = ?")
        values.append(str(title).strip() or safe_channel_id)
    if mode is not None:
        updates.append("mode = ?")
        values.append(_validate_mode(mode))
    if active_kb is not None:
        updates.append("active_kb = ?")
        values.append(str(active_kb).strip() or None)
    if topic_binding is not None:
        updates.append("topic_binding = ?")
        values.append(str(topic_binding).strip() or None)
        updates.extend(["scope_type = NULL", "scope_id = NULL"])
    if memory_scope is not None:
        updates.append("memory_scope = ?")
        values.append(_validate_memory_scope(memory_scope))
    if not updates:
        return get_channel(safe_channel_id)
    updates.append("updated_at = ?")
    values.append(now_text())
    values.append(safe_channel_id)
    conn = _get_conn()
    cursor = conn.execute(f"UPDATE channels SET {', '.join(updates)} WHERE channel_id = ?", values)
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError(f"channel not found: {safe_channel_id}")
    return get_channel(safe_channel_id)


def bind_topic(channel_id: str, topic: str) -> dict[str, Any]:
    """Persistently associate a channel with a knowledge-base topic (docs/VISION.md 第3节).

    Distinct from tools/knowledge_qa.py's `_ACTIVE_KB_BY_THREAD` (in-memory, per-thread,
    reset on restart, meant for one conversation's temporary override): this is the durable
    per-channel default a group's questions fall back to when no temporary override is set,
    and what cron push (VISION.md 第5节) will target once implemented.
    """
    safe_channel_id = _validate_channel_id(channel_id)
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE channels SET topic_binding = ?, scope_type = NULL, scope_id = NULL, updated_at = ? WHERE channel_id = ?",
        (str(topic).strip() or None, now_text(), safe_channel_id),
    )
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError(f"channel not found: {safe_channel_id}")
    return get_channel(safe_channel_id)


def unbind_topic(channel_id: str) -> dict[str, Any]:
    safe_channel_id = _validate_channel_id(channel_id)
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE channels SET topic_binding = NULL, updated_at = ? WHERE channel_id = ?",
        (now_text(), safe_channel_id),
    )
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError(f"channel not found: {safe_channel_id}")
    return get_channel(safe_channel_id)


def get_topic_binding(channel_id: str) -> str | None:
    safe_channel_id = _validate_channel_id(channel_id)
    conn = _get_conn()
    row = conn.execute("SELECT topic_binding FROM channels WHERE channel_id = ?", (safe_channel_id,)).fetchone()
    if row is None:
        return None
    return row["topic_binding"] or None


def bind_scope(channel_id: str, scope_type: str, scope_id: str) -> dict[str, Any]:
    safe_channel_id = _validate_channel_id(channel_id)
    normalized_type = str(scope_type or "").strip().lower()
    normalized_id = str(scope_id or "").strip()
    if normalized_type not in VALID_KNOWLEDGE_SCOPE_TYPES:
        raise ValueError("channel scope_type must be knowledge_base or theme")
    if not normalized_id:
        raise ValueError("scope_id is required")
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE channels SET scope_type = ?, scope_id = ?, topic_binding = NULL, updated_at = ? WHERE channel_id = ?",
        (normalized_type, normalized_id, now_text(), safe_channel_id),
    )
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError(f"channel not found: {safe_channel_id}")
    return get_channel(safe_channel_id)


def unbind_scope(channel_id: str) -> dict[str, Any]:
    safe_channel_id = _validate_channel_id(channel_id)
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE channels SET scope_type = NULL, scope_id = NULL, topic_binding = NULL, updated_at = ? WHERE channel_id = ?",
        (now_text(), safe_channel_id),
    )
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError(f"channel not found: {safe_channel_id}")
    return get_channel(safe_channel_id)


def get_scope_binding(channel_id: str) -> dict[str, Any] | None:
    channel = get_channel(channel_id)
    if channel.get("scope_type") and channel.get("scope_id"):
        return {"scope_type": channel["scope_type"], "scope_id": channel["scope_id"], "legacy": False}
    if channel.get("topic_binding"):
        return {"scope_type": "legacy_topic", "scope_id": channel["topic_binding"], "legacy": True}
    return None


_DEFAULT_PUSH_CONFIG = {
    "enabled": False,
    "frequency": "daily",
    "content_strategy": "rotation",
    "push_time": DEFAULT_PUSH_TIME,
    "timezone": DEFAULT_PUSH_TIMEZONE,
    "next_push_at": None,
    "last_pushed_at": None,
    "last_pushed_doc": None,
}


def _validate_push_time(value: str) -> str:
    safe = str(value or "").strip()
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", safe):
        raise ValueError("push_time must use 24-hour HH:MM format")
    return safe


def _validate_push_timezone(value: str) -> str:
    safe = str(value or "").strip()
    if not safe or len(safe) > 64:
        raise ValueError("timezone is required and must be at most 64 characters")
    try:
        ZoneInfo(safe)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone: {safe}") from exc
    return safe


def calculate_next_push_at(
    after_ts: float,
    *,
    frequency: str,
    push_time: str,
    timezone: str,
    respect_frequency_gap: bool,
) -> float:
    """Return the next wall-clock schedule slot strictly after the relevant threshold."""
    safe_frequency = frequency if frequency in VALID_FREQUENCIES else "daily"
    safe_time = _validate_push_time(push_time)
    safe_timezone = _validate_push_timezone(timezone)
    threshold = float(after_ts)
    if respect_frequency_gap:
        threshold += FREQUENCY_MIN_GAP_HOURS[safe_frequency] * 3600
    local_threshold = datetime.fromtimestamp(threshold, ZoneInfo(safe_timezone))
    hour, minute = (int(part) for part in safe_time.split(":"))
    candidate = local_threshold.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate.timestamp() <= threshold:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def _set_next_push_at(channel_id: str, next_push_at: float | None) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE channel_push_config SET next_push_at = ?, updated_at = ? WHERE channel_id = ?",
        (next_push_at, now_text(), _validate_channel_id(channel_id)),
    )
    conn.commit()


def initialize_next_push_at(channel_id: str, *, now_ts: float) -> float | None:
    """Populate a missing schedule for rows created before per-channel scheduling existed."""
    config = get_push_config(channel_id)
    if not config["enabled"]:
        return None
    if config["next_push_at"] is not None:
        return float(config["next_push_at"])
    last_pushed_at = config["last_pushed_at"]
    next_push_at = calculate_next_push_at(
        float(last_pushed_at) if last_pushed_at is not None else now_ts,
        frequency=config["frequency"],
        push_time=config["push_time"],
        timezone=config["timezone"],
        respect_frequency_gap=last_pushed_at is not None,
    )
    _set_next_push_at(channel_id, next_push_at)
    return next_push_at


def advance_push_schedule(channel_id: str, *, after_ts: float) -> float:
    config = get_push_config(channel_id)
    next_push_at = calculate_next_push_at(
        after_ts,
        frequency=config["frequency"],
        push_time=config["push_time"],
        timezone=config["timezone"],
        respect_frequency_gap=True,
    )
    _set_next_push_at(channel_id, next_push_at)
    return next_push_at


def get_push_config(channel_id: str) -> dict[str, Any]:
    """Return the channel's push config, or defaults (enabled=False) if never configured.

    Deliberately lenient (never raises for a missing row or an unregistered channel_id) --
    the scheduler tick calls this at high frequency across many channels and must not blow up
    just because a channel has never touched its push settings.
    """
    safe_channel_id = _validate_channel_id(channel_id)
    conn = _get_conn()
    row = conn.execute(
        "SELECT enabled, frequency, content_strategy, push_time, timezone, next_push_at, "
        "last_pushed_at, last_pushed_doc "
        "FROM channel_push_config WHERE channel_id = ?",
        (safe_channel_id,),
    ).fetchone()
    if row is None:
        return dict(_DEFAULT_PUSH_CONFIG)
    return {
        "enabled": bool(row["enabled"]),
        "frequency": row["frequency"],
        "content_strategy": row["content_strategy"],
        "push_time": row["push_time"],
        "timezone": row["timezone"],
        "next_push_at": row["next_push_at"],
        "last_pushed_at": row["last_pushed_at"],
        "last_pushed_doc": row["last_pushed_doc"],
    }


def upsert_push_config(
    channel_id: str,
    *,
    enabled: bool | None = None,
    frequency: str | None = None,
    content_strategy: str | None = None,
    push_time: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Partial update of a channel's push config; creates the row on first call.

    Raises ValueError if the channel itself isn't registered (same contract as bind_topic) --
    push config without a real channel behind it is meaningless.
    """
    safe_channel_id = _validate_channel_id(channel_id)
    get_channel(safe_channel_id)  # raises ValueError if the channel doesn't exist

    if frequency is not None and frequency not in VALID_FREQUENCIES:
        raise ValueError(f"frequency must be one of: {', '.join(sorted(VALID_FREQUENCIES))}")
    if content_strategy is not None and content_strategy not in VALID_CONTENT_STRATEGIES:
        raise ValueError(f"content_strategy must be one of: {', '.join(sorted(VALID_CONTENT_STRATEGIES))}")
    safe_push_time = _validate_push_time(push_time) if push_time is not None else None
    safe_timezone = _validate_push_timezone(timezone) if timezone is not None else None

    current = get_push_config(safe_channel_id)
    merged = {
        "enabled": current["enabled"] if enabled is None else bool(enabled),
        "frequency": current["frequency"] if frequency is None else frequency,
        "content_strategy": current["content_strategy"] if content_strategy is None else content_strategy,
        "push_time": current["push_time"] if safe_push_time is None else safe_push_time,
        "timezone": current["timezone"] if safe_timezone is None else safe_timezone,
    }
    schedule_changed = (
        (enabled is True and not current["enabled"])
        or frequency is not None
        or safe_push_time is not None
        or safe_timezone is not None
    )
    if not merged["enabled"]:
        next_push_at = None
    elif schedule_changed or current["next_push_at"] is None:
        schedule_threshold = time.time()
        if current["last_pushed_at"] is not None:
            minimum_gap_end = float(current["last_pushed_at"]) + FREQUENCY_MIN_GAP_HOURS[merged["frequency"]] * 3600
            schedule_threshold = max(schedule_threshold, minimum_gap_end)
        next_push_at = calculate_next_push_at(
            schedule_threshold,
            frequency=merged["frequency"],
            push_time=merged["push_time"],
            timezone=merged["timezone"],
            respect_frequency_gap=False,
        )
    else:
        next_push_at = current["next_push_at"]
    now = now_text()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO channel_push_config
            (channel_id, enabled, frequency, content_strategy, push_time, timezone,
             next_push_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
            enabled = excluded.enabled,
            frequency = excluded.frequency,
            content_strategy = excluded.content_strategy,
            push_time = excluded.push_time,
            timezone = excluded.timezone,
            next_push_at = excluded.next_push_at,
            updated_at = excluded.updated_at
        """,
        (
            safe_channel_id,
            int(merged["enabled"]),
            merged["frequency"],
            merged["content_strategy"],
            merged["push_time"],
            merged["timezone"],
            next_push_at,
            now,
            now,
        ),
    )
    conn.commit()
    return get_push_config(safe_channel_id)


def record_push(channel_id: str, *, document: str, pushed_at: float) -> None:
    safe_channel_id = _validate_channel_id(channel_id)
    config = get_push_config(safe_channel_id)
    next_push_at = calculate_next_push_at(
        pushed_at,
        frequency=config["frequency"],
        push_time=config["push_time"],
        timezone=config["timezone"],
        respect_frequency_gap=True,
    )
    now = now_text()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO channel_push_config
            (channel_id, last_pushed_at, last_pushed_doc, next_push_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
            last_pushed_at = excluded.last_pushed_at,
            last_pushed_doc = excluded.last_pushed_doc,
            next_push_at = excluded.next_push_at,
            updated_at = excluded.updated_at
        """,
        (safe_channel_id, pushed_at, document, next_push_at, now, now),
    )
    conn.commit()


def list_push_candidates() -> list[dict[str, Any]]:
    """Every channel eligible for a push tick to consider: has a topic binding and push enabled.

    Not paginated (unlike list_channels) -- the scheduler needs the full set every tick, and
    the expected scale (tens to low hundreds of bound+enabled channels) doesn't need it.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT channels.*, channel_push_config.frequency, channel_push_config.content_strategy,
               channel_push_config.push_time, channel_push_config.timezone,
               channel_push_config.next_push_at, channel_push_config.last_pushed_at,
               channel_push_config.last_pushed_doc
        FROM channels
        JOIN channel_push_config ON channel_push_config.channel_id = channels.channel_id
        WHERE (channels.scope_id IS NOT NULL OR channels.topic_binding IS NOT NULL)
          AND channel_push_config.enabled = 1
        """
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def set_mode(channel_id: str, mode: str, *, active_kb: str | None = None) -> dict[str, Any]:
    return update_channel(channel_id, mode=mode, active_kb=active_kb)


def _validate_channel_id(channel_id: str) -> str:
    text = str(channel_id or "").strip()
    if not text:
        raise ValueError("channel_id is required")
    if len(text) > 160:
        raise ValueError("channel_id is too long")
    if not re.match(r"^[A-Za-z0-9_.:-]+$", text):
        raise ValueError("channel_id may only contain letters, numbers, dot, underscore, colon and hyphen")
    return text


def _validate_segment(value: str, *, field: str, allow_colon: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    pattern = r"^[A-Za-z0-9_.:-]+$" if allow_colon else r"^[A-Za-z0-9_.-]+$"
    if len(text) > 120 or not re.match(pattern, text):
        raise ValueError(f"{field} may only contain letters, numbers, dot, underscore and hyphen")
    return text


def _validate_mode(mode: str) -> str:
    safe = str(mode or "").strip()
    if safe not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
    return safe


def _validate_memory_scope(scope: str) -> str:
    safe = str(scope or "channel").strip()
    if safe not in VALID_MEMORY_SCOPES:
        raise ValueError(f"memory_scope must be one of: {', '.join(sorted(VALID_MEMORY_SCOPES))}")
    return safe


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)
