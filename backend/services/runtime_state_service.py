"""Lightweight process leases and stale-job reconciliation.

Runtime instances heartbeat into the shared conversations SQLite database. The
FastAPI process owns reconciliation; other processes such as the Feishu worker
only need to register and heartbeat their own instance.
"""

from __future__ import annotations

import datetime as dt
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from . import job_service as job_service_module
from .common import now_text

try:
    from ...observability import emit_event
except ImportError:  # pragma: no cover
    from observability import emit_event


DEFAULT_HEARTBEAT_SECONDS = 15.0
DEFAULT_RECONCILE_SECONDS = 60.0
DEFAULT_INSTANCE_STALE_SECONDS = 60.0
DEFAULT_LEGACY_GRACE_SECONDS = 24 * 60 * 60.0
DEFAULT_CHAT_STALLED_SECONDS = 5 * 60.0
DEFAULT_JOB_STALLED_SECONDS = 15 * 60.0

ACTIVE_JOB_STATES = {"queued", "running"}

_ACTIVE_MONITOR_LOCK = threading.Lock()
_ACTIVE_MONITOR: RuntimeMonitor | None = None
_CURRENT_INSTANCE_ID: str | None = None


def _ensure_schema() -> None:
    conn = job_service_module._get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_instances (
            instance_id   TEXT PRIMARY KEY,
            role          TEXT NOT NULL,
            environment   TEXT NOT NULL,
            pid           INTEGER,
            started_at    TEXT NOT NULL,
            heartbeat_at  TEXT NOT NULL,
            stopped_at    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_instances_heartbeat
            ON runtime_instances(heartbeat_at);
        CREATE INDEX IF NOT EXISTS idx_runtime_instances_role
            ON runtime_instances(role);
        """
    )
    conn.commit()


def current_instance_id() -> str | None:
    return _CURRENT_INSTANCE_ID


def runtime_environment() -> str:
    return str(os.getenv("HARNESS_ENV", "dev") or "dev").strip().lower()


def register_runtime_instance(role: str, *, instance_id: str | None = None) -> str:
    global _CURRENT_INSTANCE_ID

    _ensure_schema()
    resolved_id = instance_id or f"inst_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    now = now_text()
    conn = job_service_module._get_conn()
    conn.execute(
        "INSERT INTO runtime_instances "
        "(instance_id, role, environment, pid, started_at, heartbeat_at, stopped_at) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
        (resolved_id, role, runtime_environment(), os.getpid(), now, now),
    )
    conn.commit()
    _CURRENT_INSTANCE_ID = resolved_id
    emit_event(
        "runtime.instance.started",
        "runtime",
        "Runtime instance registered.",
        data={"instance_id": resolved_id, "role": role, "environment": runtime_environment()},
    )
    return resolved_id


def heartbeat_runtime_instance(instance_id: str) -> None:
    _ensure_schema()
    now = now_text()
    conn = job_service_module._get_conn()
    conn.execute(
        "UPDATE runtime_instances SET heartbeat_at = ?, stopped_at = NULL WHERE instance_id = ?",
        (now, instance_id),
    )
    # Job heartbeats follow their owning process. Progress timestamps remain
    # separate so the UI can distinguish alive-but-stalled from interrupted.
    conn.execute(
        "UPDATE jobs SET last_heartbeat_at = ? "
        "WHERE owner_instance_id = ? AND status IN ('queued', 'running')",
        (now, instance_id),
    )
    conn.commit()


def stop_runtime_instance(instance_id: str) -> None:
    global _CURRENT_INSTANCE_ID

    _ensure_schema()
    now = now_text()
    conn = job_service_module._get_conn()
    conn.execute(
        "UPDATE runtime_instances SET heartbeat_at = ?, stopped_at = ? WHERE instance_id = ?",
        (now, now, instance_id),
    )
    conn.commit()
    if _CURRENT_INSTANCE_ID == instance_id:
        _CURRENT_INSTANCE_ID = None
    emit_event(
        "runtime.instance.stopped",
        "runtime",
        "Runtime instance stopped.",
        data={"instance_id": instance_id},
    )


def reconcile_orphaned_jobs(
    *,
    instance_stale_seconds: float = DEFAULT_INSTANCE_STALE_SECONDS,
    legacy_grace_seconds: float = DEFAULT_LEGACY_GRACE_SECONDS,
    job_stalled_seconds: float = DEFAULT_JOB_STALLED_SECONDS,
    now: dt.datetime | None = None,
) -> dict[str, list[str]]:
    """Close jobs whose owner disappeared and flag alive jobs with no progress."""

    _ensure_schema()
    current = now or dt.datetime.now()
    conn = job_service_module._get_conn()
    rows = conn.execute(
        "SELECT job_id, status, owner_instance_id, updated_at, last_progress_at "
        "FROM jobs WHERE status IN ('queued', 'running')"
    ).fetchall()
    instances = {
        str(row["instance_id"]): row
        for row in conn.execute(
            "SELECT instance_id, heartbeat_at, stopped_at FROM runtime_instances"
        ).fetchall()
    }

    interrupted: list[str] = []
    stalled: list[str] = []
    healthy: list[str] = []
    interrupt_candidates: list[tuple[str, str | None, str]] = []
    health_updates: list[tuple[str, str]] = []
    for row in rows:
        job_id = str(row["job_id"])
        owner_id = str(row["owner_instance_id"] or "")
        reason: str | None = None
        if owner_id:
            instance = instances.get(owner_id)
            if instance is None:
                reason = "owner_instance_missing"
            elif instance["stopped_at"]:
                reason = "owner_instance_stopped"
            else:
                heartbeat_at = _parse_timestamp(instance["heartbeat_at"])
                if heartbeat_at is None or (current - heartbeat_at).total_seconds() > instance_stale_seconds:
                    reason = "owner_instance_lost"
        else:
            updated_at = _parse_timestamp(row["updated_at"])
            if updated_at is not None and (current - updated_at).total_seconds() > legacy_grace_seconds:
                reason = "legacy_state_without_owner"

        if reason:
            interrupt_candidates.append((job_id, owner_id or None, reason))
            continue

        last_progress_at = _parse_timestamp(row["last_progress_at"] or row["updated_at"])
        is_stalled = bool(
            last_progress_at is not None
            and (current - last_progress_at).total_seconds() > job_stalled_seconds
        )
        health_status = "stalled" if is_stalled else "healthy"
        health_updates.append((health_status, job_id))
        (stalled if is_stalled else healthy).append(job_id)

    # JobService._update() owns its own immediate transaction. Complete those
    # updates before issuing direct health writes on the shared connection.
    for job_id, owner_id, reason in interrupt_candidates:
        if _interrupt_job_if_active(job_id, owner_id=owner_id, reason=reason):
            interrupted.append(job_id)

    if health_updates:
        conn.executemany(
            "UPDATE jobs SET health_status = ? WHERE job_id = ? "
            "AND status IN ('queued', 'running')",
            health_updates,
        )
    conn.commit()
    return {"interrupted": interrupted, "stalled": stalled, "healthy": healthy}


def _interrupt_job_if_active(job_id: str, *, owner_id: str | None, reason: str) -> bool:
    changed = False

    def mutate(job: dict[str, Any]) -> dict[str, Any]:
        nonlocal changed
        if str(job.get("status") or "") not in ACTIVE_JOB_STATES:
            return job
        current_owner = str(job.get("owner_instance_id") or "") or None
        if owner_id is not None and current_owner != owner_id:
            return job
        now = now_text()
        for step in job.get("steps") or []:
            if step.get("status") == "running":
                step["status"] = "interrupted"
                step["summary"] = "执行进程已失联，任务未正常结束。"
                step["finished_at"] = now
        job["status"] = "interrupted"
        job["status_reason"] = reason
        job["health_status"] = "orphaned"
        job["current_step"] = None
        job["updated_at"] = now
        job["finished_at"] = now
        job["error"] = {
            "code": "execution_interrupted",
            "message": "执行进程已失联，任务未正常结束。",
            "retryable": True,
            "details": {"reason": reason, "owner_instance_id": owner_id},
        }
        progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        job["progress"] = {
            "current": int(progress.get("current") or 0),
            "total": max(1, int(progress.get("total") or 1)),
            "message": "interrupted",
        }
        changed = True
        return job

    job = job_service_module.job_service._update(job_id, mutate)
    if changed:
        if str(job.get("job_type") or "").startswith("research"):
            try:
                from ..research.audit import audit_run, write_stage_checkpoint

                input_data = job.get("input") if isinstance(job.get("input"), dict) else {}
                with audit_run(
                    job_id=job_id,
                    parent_job_id=str(input_data.get("parent_job_id") or "") or None,
                    attempt=int(job.get("attempt") or 1),
                    retry_of_job_id=str(job.get("retry_of_job_id") or "") or None,
                ):
                    write_stage_checkpoint(
                        "interrupt",
                        outputs={"job_status": "interrupted"},
                        transition_reason="execution owner lease was lost",
                        status="interrupted",
                        interrupt={
                            "kind": "owner_lost",
                            "reason": reason,
                            "owner_instance_id": owner_id,
                        },
                    )
            except Exception:
                pass
        emit_event(
            "job.interrupted",
            "jobs",
            "Background job owner was lost; job marked interrupted.",
            level="warning",
            data={
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "reason": reason,
                "owner_instance_id": owner_id,
            },
        )
    return changed


def _parse_timestamp(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _monitor_enabled() -> bool:
    configured = os.getenv("RUNTIME_STATE_MONITOR_ENABLED")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    # Pytest exercises the monitor explicitly; ordinary TestClient requests
    # must not leave session-wide background threads behind.
    return "PYTEST_CURRENT_TEST" not in os.environ


@dataclass
class RuntimeMonitor:
    instance_id: str | None
    role: str
    reconcile: bool
    stop_event: threading.Event
    thread: threading.Thread | None

    def stop(self) -> None:
        global _ACTIVE_MONITOR
        if self.instance_id is None:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        stop_runtime_instance(self.instance_id)
        if self.reconcile:
            reconcile_orphaned_jobs()
        with _ACTIVE_MONITOR_LOCK:
            if _ACTIVE_MONITOR is self:
                _ACTIVE_MONITOR = None


def start_runtime_monitor(
    *,
    role: str,
    reconcile: bool,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    reconcile_seconds: float = DEFAULT_RECONCILE_SECONDS,
    enabled: bool | None = None,
) -> RuntimeMonitor:
    global _ACTIVE_MONITOR

    should_enable = _monitor_enabled() if enabled is None else enabled
    if not should_enable:
        return RuntimeMonitor(None, role, reconcile, threading.Event(), None)

    with _ACTIVE_MONITOR_LOCK:
        if _ACTIVE_MONITOR is not None and _ACTIVE_MONITOR.thread is not None:
            if _ACTIVE_MONITOR.thread.is_alive():
                return _ACTIVE_MONITOR

        instance_id = register_runtime_instance(role)
        stop_event = threading.Event()
        monitor = RuntimeMonitor(instance_id, role, reconcile, stop_event, None)

        def run_loop() -> None:
            next_reconcile = 0.0
            while not stop_event.is_set():
                try:
                    heartbeat_runtime_instance(instance_id)
                    now_monotonic = time.monotonic()
                    if reconcile and now_monotonic >= next_reconcile:
                        reconcile_orphaned_jobs()
                        next_reconcile = now_monotonic + max(1.0, reconcile_seconds)
                except Exception as exc:
                    emit_event(
                        "runtime.monitor.error",
                        "runtime",
                        "Runtime state monitor iteration failed.",
                        level="warning",
                        data={"instance_id": instance_id, "error": str(exc)[:300]},
                    )
                stop_event.wait(max(0.1, heartbeat_seconds))

        thread = threading.Thread(
            target=run_loop,
            name=f"runtime-state-{role}",
            daemon=True,
        )
        monitor.thread = thread
        _ACTIVE_MONITOR = monitor
        thread.start()
        return monitor
