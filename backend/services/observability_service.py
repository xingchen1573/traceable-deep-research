"""Observability query helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from ...observability import get_job_trace, get_run_detail, list_runs, query_events
except ImportError:  # pragma: no cover
    from observability import get_job_trace, get_run_detail, list_runs, query_events

from .job_service import job_service


def events(
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
    return query_events(
        session_id=session_id,
        thread_id=thread_id,
        run_id=run_id,
        job_id=job_id,
        request_id=request_id,
        event_type=event_type,
        level=level,
        limit=limit,
        cursor=cursor,
        days=days,
    )


def runs(
    *,
    thread_id: str | None = None,
    job_id: str | None = None,
    days: int = 7,
    limit: int = 50,
) -> dict[str, Any]:
    """Aggregated run list — the observability page's default view.

    See docs/modules/observability-redesign-v0.1.md 3.3.
    """
    return list_runs(thread_id=thread_id, job_id=job_id, days=days, limit=limit)


def run_detail(run_id: str, *, days: int = 30) -> dict[str, Any]:
    """One run's rollup + full chronological event list.

    Replaces the previous replay_events(limit=200)-based implementation,
    which silently returned nothing for any run from before today.
    """
    detail = get_run_detail(run_id, days=days)
    events_list = detail.get("events") or []
    detail["tool_calls"] = [row for row in events_list if str(row.get("component")) in ("tools", "workers")]
    detail["artifacts"] = []
    return detail


def job_trace(job_id: str, *, days: int = 30) -> dict[str, Any]:
    """A research job's full trace, flattened across all runs it touched.

    Enriches the observability-log-derived summary with the job's own
    input (e.g. topic) from job_service, when the job is still tracked there —
    the observability log alone never carries a research topic string.
    See docs/modules/observability-redesign-v0.1.md 3.11.
    """
    trace = get_job_trace(job_id, days=days)
    try:
        job = job_service.get(job_id)
    except ValueError:
        job = None
    if job:
        trace["job_type"] = job.get("job_type")
        trace["job_status"] = job.get("status")
        topic = (job.get("input") or {}).get("topic")
        if topic:
            title_prefix = {
                "knowledge_index": "知识索引任务",
                "knowledge_qa_case_generation": "评测集生成任务",
                "research": "调研任务",
            }.get(str(job.get("job_type") or ""), "后台任务")
            trace["title"] = f"{title_prefix} · {topic}"
        # The job registry is authoritative for lifecycle state. Event-derived
        # aggregation can be stale when the detail request races with writes or
        # when a successful job contains recovered warning/error attempts.
        trace["status"] = str(job.get("status") or trace.get("status") or "unknown")
        trace["started_at"] = job.get("created_at") or trace.get("started_at")
        if trace["status"] in {"succeeded", "failed", "cancelled", "interrupted", "waiting_approval"}:
            trace["finished_at"] = job.get("finished_at") or job.get("updated_at") or trace.get("finished_at")
        else:
            trace["finished_at"] = None
        trace["duration_ms"] = _elapsed_ms(trace.get("started_at"), trace.get("finished_at"))
        trace["event_count"] = len(trace.get("events") or [])
        trace["error_count"] = len(trace.get("errors") or [])
        if not trace.get("events"):
            snapshot = _job_state_snapshot(job)
            trace["events"] = [snapshot]
            trace["errors"] = [snapshot] if snapshot["level"] == "error" else []
            trace["status"] = str(job.get("status") or "unknown")
            trace["session_id"] = snapshot["session_id"]
            trace["thread_id"] = snapshot["session_id"]
            trace["job_id"] = job.get("job_id")
            trace["started_at"] = job.get("created_at")
            trace["finished_at"] = job.get("updated_at")
            trace["event_count"] = 1
            trace["error_count"] = 1 if snapshot["level"] == "error" else 0
            trace["components"] = ["jobs"]
    return trace


def _elapsed_ms(started_at: Any, finished_at: Any) -> int | None:
    if not started_at or not finished_at:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            started = datetime.strptime(str(started_at), fmt)
            finished = datetime.strptime(str(finished_at), fmt)
            return max(0, int((finished - started).total_seconds() * 1000))
        except ValueError:
            continue
    return None


def _job_state_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Expose truthful state for jobs created before generic job events existed."""
    status = str(job.get("status") or "unknown")
    timestamp = str(job.get("updated_at") or job.get("created_at") or "")
    level = "error" if status == "failed" else "warning" if status in {"cancelled", "interrupted"} else "info"
    return {
        "timestamp": timestamp,
        "date": timestamp[:10],
        "session_id": "job_registry",
        "run_id": None,
        "job_id": job.get("job_id"),
        "event_type": "job.state_snapshot",
        "level": level,
        "component": "jobs",
        "message": "该任务创建时尚未记录通用观测事件；以下为任务注册表中的真实状态快照。",
        "data": {
            "derived_snapshot": True,
            "job_type": job.get("job_type"),
            "status": status,
            "progress": job.get("progress"),
            "steps": job.get("steps") or [],
            "error": job.get("error"),
        },
    }
