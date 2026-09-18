"""JSONL runtime observability for Deep Research."""

from __future__ import annotations

import contextvars
import base64
import datetime as _dt
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OBSERVABILITY_DIR = ROOT / "logs" / "observability"
_SUBSCRIBERS_LOCK = threading.Lock()
_SUBSCRIBERS: dict[str, queue.Queue[dict[str, Any]]] = {}

_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "harness_session_id",
    default="default",
)
_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_run_id",
    default=None,
)
_JOB_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_job_id",
    default=None,
)
_UNSET = object()  # sentinel so set_observation_context(job_id=None) can explicitly clear


def _active_observability_dir() -> Path:
    configured = os.getenv("OBSERVABILITY_DIR", "").strip()
    return Path(configured).expanduser() if configured else OBSERVABILITY_DIR


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"


def set_observation_context(
    session_id: str | None = None,
    run_id: str | None = None,
    job_id: str | None | object = _UNSET,
) -> None:
    if session_id:
        _SESSION_ID.set(session_id)
    if run_id is not None:
        _RUN_ID.set(run_id)
    if job_id is not _UNSET:
        _JOB_ID.set(job_id)  # type: ignore[arg-type]


def current_session_id() -> str:
    return _SESSION_ID.get() or "default"


def current_run_id() -> str | None:
    return _RUN_ID.get()


def current_job_id() -> str | None:
    return _JOB_ID.get()


def emit_event(
    event_type: str,
    component: str,
    message: str,
    session_id: str | None = None,
    run_id: str | None = None,
    level: str = "info",
    data: dict[str, Any] | None = None,
) -> None:
    """Append one sanitized runtime event to the daily observability JSONL log."""
    try:
        now = _dt.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        date_text = now.strftime("%Y-%m-%d")
        safe_data = _sanitize(data or {})
        payload = {
            "timestamp": timestamp,
            "date": date_text,
            "session_id": session_id or current_session_id(),
            "run_id": run_id if run_id is not None else current_run_id(),
            "event_type": event_type,
            "level": level,
            "component": component,
            "message": message[:500],
            "data": safe_data,
        }
        _promote_filter_fields(payload)
        # Fall back to the ambient job_id context only at write time — never at
        # read time (query_events also calls _promote_filter_fields on historical
        # rows, and stamping those with *this* thread's current job_id would
        # corrupt old events with whatever job happens to be running right now).
        if not payload.get("job_id"):
            job_ctx = current_job_id()
            if job_ctx:
                payload["job_id"] = job_ctx
        observability_dir = _active_observability_dir()
        observability_dir.mkdir(parents=True, exist_ok=True)
        path = observability_dir / f"{date_text}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _publish(payload)
    except Exception:
        # Observability must never break the agent path.
        return


def subscribe_events() -> tuple[str, queue.Queue[dict[str, Any]]]:
    subscriber_id = f"sub_{uuid.uuid4().hex[:10]}"
    event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=200)
    with _SUBSCRIBERS_LOCK:
        _SUBSCRIBERS[subscriber_id] = event_queue
    return subscriber_id, event_queue


def unsubscribe_events(subscriber_id: str) -> None:
    with _SUBSCRIBERS_LOCK:
        _SUBSCRIBERS.pop(subscriber_id, None)


def replay_events(limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
    date_text = time.strftime("%Y-%m-%d", time.localtime())
    path = _active_observability_dir() / f"{date_text}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if session_id and str(payload.get("session_id")) != session_id:
            continue
        rows.append(payload)
    return rows[-max(0, min(limit, 200)) :]


def query_events(
    *,
    session_id: str | None = None,
    thread_id: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    event_type: str | None = None,
    level: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
    days: int = 7,
) -> dict[str, Any]:
    """Query JSONL observability logs by scanning newest events first."""
    limit = max(1, min(limit, 200))
    days = max(1, min(days, 90))
    cursor_state = _decode_cursor(cursor)
    events: list[dict[str, Any]] = []
    scanned = 0

    for date_text, path in _iter_log_files(days):
        if cursor_state:
            cursor_date = str(cursor_state.get("date") or "")
            if date_text > cursor_date:
                continue
        lines = path.read_text(encoding="utf-8").splitlines()
        start_index = len(lines) - 1
        if cursor_state and date_text == str(cursor_state.get("date") or ""):
            start_index = min(int(cursor_state.get("line", 0)) - 1, len(lines) - 1)
        if start_index < 0:
            continue
        for line_index in range(start_index, -1, -1):
            line = lines[line_index]
            if not line.strip():
                continue
            scanned += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            _promote_filter_fields(payload)
            if not _matches_event(
                payload,
                session_id=session_id,
                thread_id=thread_id,
                run_id=run_id,
                job_id=job_id,
                request_id=request_id,
                event_type=event_type,
                level=level,
            ):
                continue
            events.append(payload)
            if len(events) >= limit:
                return {
                    "events": events,
                    "next_cursor": _encode_cursor(date_text, line_index),
                    "has_more": True,
                    "limit": limit,
                    "scanned": scanned,
                    "days": days,
                }

    return {
        "events": events,
        "next_cursor": None,
        "has_more": False,
        "limit": limit,
        "scanned": scanned,
        "days": days,
    }


def list_runs(
    *,
    thread_id: str | None = None,
    job_id: str | None = None,
    days: int = 7,
    limit: int = 50,
) -> dict[str, Any]:
    """Aggregate recent events into a list of runs (one row per run_id).

    This is the backing query for the observability "run list" view: each
    run's events are rolled up into a single summary (title/status/duration)
    rather than shown as a flat log. Scans the JSONL window fresh on every
    call (no persistent index) — see docs/modules/observability-redesign-v0.1.md 3.10.
    """
    days = max(1, min(days, 90))
    limit = max(1, min(limit, 200))
    groups = _scan_grouped(days=days, group_key="run_id", thread_id=thread_id, job_id=job_id)
    runs = [_summarize_group(run_id, events) for run_id, events in groups.items()]
    runs.sort(key=lambda r: str(r.get("finished_at") or ""), reverse=True)
    return {
        "runs": runs[:limit],
        "days": days,
        "total_runs": len(runs),
        "truncated": len(runs) > limit,
    }


def get_run_detail(run_id: str, *, days: int = 30) -> dict[str, Any]:
    """Full detail for one run: rollup summary + its events in chronological order.

    Replaces the previous replay_events()-based run_detail, which only ever
    looked at *today's* log file and therefore returned nothing for any run
    from a prior day.
    """
    days = max(1, min(days, 90))
    groups = _scan_grouped(days=days, group_key="run_id", run_id=run_id)
    events = groups.get(run_id, [])
    summary = _summarize_group(run_id, events) if events else _empty_group_summary(run_id)
    return {
        **summary,
        "run_id": run_id,
        "events": events,
        "errors": [e for e in events if str(e.get("level")) == "error"],
    }


def get_job_trace(job_id: str, *, days: int = 30) -> dict[str, Any]:
    """Flattened chronological trace for every event tagged with a job_id.

    A research job spans many runs across background worker ticks that don't
    share one run context, so unlike get_run_detail this pulls by job_id
    across *all* runs/sessions and lays them out as one timeline rather than
    nesting by run (see docs/modules/observability-redesign-v0.1.md 3.11).
    """
    days = max(1, min(days, 90))
    groups = _scan_grouped(days=days, group_key="job_id", job_id=job_id)
    events = groups.get(job_id, [])
    summary = _summarize_group(job_id, events) if events else _empty_group_summary(job_id)
    run_ids = sorted({str(e.get("run_id")) for e in events if e.get("run_id")})
    return {
        **summary,
        "job_id": job_id,
        "run_ids": run_ids,
        "events": events,
        "errors": [e for e in events if str(e.get("level")) == "error"],
    }


def _iter_log_files(days: int) -> list[tuple[str, Path]]:
    today = _dt.date.today()
    files: list[tuple[str, Path]] = []
    for offset in range(days):
        date_text = (today - _dt.timedelta(days=offset)).isoformat()
        path = _active_observability_dir() / f"{date_text}.jsonl"
        if path.exists():
            files.append((date_text, path))
    return files


def _matches_event(
    payload: dict[str, Any],
    *,
    session_id: str | None,
    thread_id: str | None,
    run_id: str | None,
    job_id: str | None,
    request_id: str | None,
    event_type: str | None,
    level: str | None,
) -> bool:
    if session_id and str(payload.get("session_id")) != session_id:
        return False
    if thread_id and str(payload.get("thread_id") or payload.get("session_id")) != thread_id:
        return False
    if run_id and str(payload.get("run_id")) != run_id:
        return False
    if job_id and str(payload.get("job_id")) != job_id:
        return False
    if request_id and str(payload.get("request_id") or payload.get("run_id")) != request_id:
        return False
    if event_type and str(payload.get("event_type")) != event_type:
        return False
    if level and str(payload.get("level")) != level:
        return False
    return True


def _promote_filter_fields(payload: dict[str, Any]) -> None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    session_id = str(payload.get("session_id") or "")
    if not payload.get("thread_id") and session_id.startswith("thread_"):
        payload["thread_id"] = session_id
    for key in ("thread_id", "job_id", "request_id", "report_id"):
        if not payload.get(key) and data.get(key):
            payload[key] = data[key]


_TITLE_ID_LEN = 8
_TERMINAL_OK_SUFFIXES = (".finished", ".completed", ".succeeded")
_TERMINAL_FAIL_SUFFIXES = (".failed", ".cancelled")


def _scan_grouped(
    *,
    days: int,
    group_key: str,
    thread_id: str | None = None,
    job_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Scan the JSONL log window and bucket events by run_id or job_id.

    group_key is "run_id" or "job_id" — only events carrying that key are
    grouped; the optional thread_id/job_id/run_id args narrow the scan.
    Brute-force full-window scan by design (no persistent index), see
    docs/modules/observability-redesign-v0.1.md 3.10.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for _date_text, path in _iter_log_files(days):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            _promote_filter_fields(payload)
            key = payload.get(group_key)
            if not key:
                continue
            if thread_id and str(payload.get("thread_id") or payload.get("session_id")) != thread_id:
                continue
            if job_id and str(payload.get("job_id")) != job_id:
                continue
            if run_id and str(payload.get("run_id")) != run_id:
                continue
            groups.setdefault(str(key), []).append(payload)
    for events in groups.values():
        events.sort(key=lambda e: str(e.get("timestamp") or ""))
    return groups


def _infer_title(events: list[dict[str, Any]]) -> str:
    """Best-effort human-readable title for a group of events.

    Only the primary flows (chat / feishu / research job) get a dedicated
    rule per docs/modules/observability-redesign-v0.1.md 3.6/3.12; anything
    else falls back to the first event's raw message.
    """
    if not events:
        return "未知事件"
    first = events[0]
    components = {str(e.get("component") or "") for e in events}
    event_types = {str(e.get("event_type") or "") for e in events}
    session_id = str(first.get("session_id") or "")
    thread_id = str(first.get("thread_id") or "") or session_id
    short_id = thread_id[:_TITLE_ID_LEN] if thread_id else ""

    if "feishu" in components or session_id.startswith("feishu"):
        return f"飞书对话 · {short_id}" if short_id else "飞书对话"
    if session_id.startswith("dream-maintenance"):
        return "系统维护任务 (dream)"
    if session_id.startswith("dream"):
        return "系统复盘任务 (dream)"
    if "workers" in components or any(
        et.startswith("worker.") or et.startswith("retrieval.") for et in event_types
    ):
        return f"调研任务执行 · {short_id}" if short_id else "调研任务执行"
    if "server" in components and any(
        et.startswith("agent.turn") or et.startswith("chat.") for et in event_types
    ):
        return f"对话轮次 · {short_id}" if short_id else "对话轮次"
    return str(first.get("message") or first.get("event_type") or "运行事件")


def _infer_status(events: list[dict[str, Any]]) -> str:
    """Roll a group of events up into one status.

    See docs/modules/observability-redesign-v0.1.md 3.5.
    """
    if not events:
        return "unknown"

    # A job can recover from attempt-level failures by rotating credentials or
    # retrying a step. Its explicit final state is therefore authoritative.
    for event in reversed(events):
        event_type = str(event.get("event_type") or "")
        if event_type in {"job.succeeded", "job.waiting_approval"}:
            return "completed"
        if event_type in {"job.failed", "job.cancelled"}:
            return "error"
        if event_type == "job.interrupted":
            return "interrupted"

    event_types = [str(e.get("event_type") or "") for e in events]
    if "chat.interrupted" in event_types:
        return "waiting_user"

    has_error = any(str(e.get("level")) == "error" for e in events)
    has_fail_terminal = any(
        str(event.get("event_type") or "").endswith(_TERMINAL_FAIL_SUFFIXES)
        and str(event.get("level") or "") == "error"
        for event in events
    )
    has_ok_terminal = any(et.endswith(_TERMINAL_OK_SUFFIXES) for et in event_types)
    if has_error or has_fail_terminal:
        return "error"
    if has_ok_terminal:
        return "completed"

    # Receiving a transport event is not evidence of a still-running agent
    # turn. This most often appears in mocked/test Feishu adapters.
    if not any(et.startswith("agent.turn.") for et in event_types) and all(
        et.startswith("feishu.") for et in event_types
    ):
        return "completed"

    last_seen = _parse_ts(str(events[-1].get("timestamp") or ""))
    stale_seconds = float(os.getenv("OBSERVABILITY_RUN_STALE_SECONDS", "900"))
    if last_seen is not None and (_dt.datetime.now() - last_seen).total_seconds() > stale_seconds:
        return "interrupted"
    return "running"


def _parse_ts(ts: str) -> _dt.datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _duration_ms(events: list[dict[str, Any]]) -> int | None:
    if len(events) < 2:
        return None
    start = _parse_ts(str(events[0].get("timestamp") or ""))
    end = _parse_ts(str(events[-1].get("timestamp") or ""))
    if not start or not end:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def _summarize_group(key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = events[0], events[-1]
    return {
        "id": key,
        "title": _infer_title(events),
        "status": _infer_status(events),
        "session_id": first.get("session_id"),
        "thread_id": first.get("thread_id") or first.get("session_id"),
        "job_id": first.get("job_id"),
        "started_at": first.get("timestamp"),
        "finished_at": last.get("timestamp"),
        "duration_ms": _duration_ms(events),
        "event_count": len(events),
        "error_count": sum(1 for e in events if str(e.get("level")) == "error"),
        "components": sorted({str(e.get("component") or "") for e in events} - {""}),
    }


def _empty_group_summary(key: str) -> dict[str, Any]:
    return {
        "id": key,
        "title": "未找到匹配的观测事件",
        "status": "unknown",
        "session_id": None,
        "thread_id": None,
        "job_id": None,
        "started_at": None,
        "finished_at": None,
        "duration_ms": None,
        "event_count": 0,
        "error_count": 0,
        "components": [],
    }


def _encode_cursor(date_text: str, line_index: int) -> str:
    raw = json.dumps({"date": date_text, "line": line_index}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> dict[str, Any] | None:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if not isinstance(value, dict) or "date" not in value or "line" not in value:
            return None
        return value
    except Exception:
        return None


def _publish(payload: dict[str, Any]) -> None:
    with _SUBSCRIBERS_LOCK:
        subscribers = list(_SUBSCRIBERS.items())
    stale: list[str] = []
    for subscriber_id, event_queue in subscribers:
        try:
            event_queue.put_nowait(payload)
        except queue.Full:
            try:
                event_queue.get_nowait()
                event_queue.put_nowait(payload)
            except Exception:
                stale.append(subscriber_id)
        except Exception:
            stale.append(subscriber_id)
    if stale:
        with _SUBSCRIBERS_LOCK:
            for subscriber_id in stale:
                _SUBSCRIBERS.pop(subscriber_id, None)


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated-depth]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "...[truncated]"
    if isinstance(value, dict):
        return {str(key)[:80]: _sanitize(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [_sanitize(item, depth + 1) for item in items[:20]]
    return str(value)[:500]
