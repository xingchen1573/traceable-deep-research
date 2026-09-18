"""SQLite-backed job service for API v1.

Jobs used to live in a process-local `dict` (`JobService._jobs`). That was fine while
Feishu handling and the website API ran in the same process, but the Docker split into
`backend` and `feishu-worker` containers (see docker-compose.yml) put them in two
separate processes with separate memory -- a job created by the Feishu long-connection
worker was invisible to the website, and vice versa. This module persists jobs to the
same on-disk SQLite database channel_service.py/conversation_service.py already share
(`data/conversations.db`, mounted as the `app_data` volume in both docker-compose
services), so job state survives process restarts and is visible across
processes/containers on the same host.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import now_text

try:
    from ...observability import emit_event, set_observation_context
except ImportError:  # pragma: no cover
    from observability import emit_event, set_observation_context


TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted"}
STEP_TERMINAL_STATES = {"succeeded", "failed", "waiting_approval", "interrupted"}

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "conversations.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        _ensure_schema(conn)
        _local.conn = conn
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        conn.commit()
    except sqlite3.OperationalError:
        # Another thread's _ensure_schema() won the race and already added it.
        pass


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id       TEXT PRIMARY KEY,
            job_type     TEXT NOT NULL,
            status       TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            channel_id   TEXT,
            payload      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_job_type ON jobs(job_type);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_channel_id ON jobs(channel_id);
    """)
    conn.commit()
    _ensure_column(conn, "jobs", "archived", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "jobs", "owner_instance_id", "TEXT")
    _ensure_column(conn, "jobs", "status_reason", "TEXT")
    _ensure_column(conn, "jobs", "started_at", "TEXT")
    _ensure_column(conn, "jobs", "last_heartbeat_at", "TEXT")
    _ensure_column(conn, "jobs", "last_progress_at", "TEXT")
    _ensure_column(conn, "jobs", "finished_at", "TEXT")
    _ensure_column(conn, "jobs", "health_status", "TEXT")
    _ensure_column(conn, "jobs", "environment", "TEXT NOT NULL DEFAULT 'dev'")
    _ensure_column(conn, "jobs", "attempt", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "jobs", "retry_of_job_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_archived ON jobs(archived)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner_instance_id ON jobs(owner_instance_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_health_status ON jobs(health_status)")
    conn.commit()


class JobService:
    """SQLite-backed job registry, shared across every process on this host."""

    def create(
        self,
        *,
        job_type: str,
        input_data: dict[str, Any] | None = None,
        target: Callable[[dict[str, Any]], Any] | None = None,
        run_async: bool = True,
        steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = now_text()
        normalized_steps = _normalize_steps(steps or [])
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "progress": {"current": 0, "total": max(1, len(normalized_steps)), "message": "queued"},
            "current_step": None,
            "steps": normalized_steps,
            "input": input_data or {},
            "result": None,
            "error": None,
            "artifacts": [],
            "archived": False,
            "owner_instance_id": None,
            "status_reason": None,
            "started_at": None,
            "last_heartbeat_at": None,
            "last_progress_at": now,
            "finished_at": None,
            "health_status": "healthy",
            "environment": str(os.getenv("HARNESS_ENV", "dev") or "dev").strip().lower(),
            "attempt": 1,
            "retry_of_job_id": None,
        }
        self._insert(job)
        if target is not None:
            if run_async:
                thread = threading.Thread(target=self._run, args=(job_id, target), daemon=True)
                thread.start()
            else:
                self._run(job_id, target)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"job not found: {job_id}")
        return _row_to_job(row)

    def list(
        self,
        *,
        job_type: str | None = None,
        research_type: str | None = None,
        status: str | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
        channel_id: str | None = None,
        archived: bool | None = None,
    ) -> list[dict[str, Any]]:
        jobs, _, _ = self.list_page(
            job_type=job_type,
            research_type=research_type,
            status=status,
            statuses=statuses,
            limit=limit,
            channel_id=channel_id,
            archived=archived,
        )
        return jobs

    def list_page(
        self,
        *,
        job_type: str | None = None,
        research_type: str | None = None,
        status: str | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
        channel_id: str | None = None,
        archived: bool | None = None,
        query: str | None = None,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, int]:
        conn = _get_conn()
        clauses = []
        params: list[Any] = []
        if job_type:
            clauses.append("job_type = ?")
            params.append(job_type)
        pipeline_mode_sql = (
            "COALESCE(json_extract(payload, '$.result.pipeline_mode'), "
            "json_extract(payload, '$.input.pipeline_mode'), '')"
        )
        if research_type == "single_url":
            clauses.append(f"{pipeline_mode_sql} = ?")
            params.append("single_url")
        elif research_type == "deep_research":
            clauses.append(f"{pipeline_mode_sql} != ?")
            params.append("single_url")
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif statuses:
            normalized_statuses = [value for value in statuses if value]
            if normalized_statuses:
                clauses.append(f"status IN ({','.join('?' for _ in normalized_statuses)})")
                params.extend(normalized_statuses)
        if channel_id:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        if archived is not None:
            clauses.append("archived = ?")
            params.append(1 if archived else 0)
        if query and query.strip():
            pattern = f"%{query.strip().lower()}%"
            clauses.append("(LOWER(job_id) LIKE ? OR LOWER(job_type) LIKE ? OR LOWER(payload) LIKE ?)")
            params.extend([pattern, pattern, pattern])

        total_where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = int(conn.execute(f"SELECT COUNT(*) FROM jobs {total_where}", params).fetchone()[0])

        page_clauses = list(clauses)
        page_params = list(params)
        if cursor:
            priority, updated_at, job_id = _decode_list_cursor(cursor)
            page_clauses.append(
                "("
                "CASE WHEN status = 'waiting_approval' THEN 0 ELSE 1 END > ? "
                "OR (CASE WHEN status = 'waiting_approval' THEN 0 ELSE 1 END = ? AND updated_at < ?) "
                "OR (CASE WHEN status = 'waiting_approval' THEN 0 ELSE 1 END = ? AND updated_at = ? AND job_id < ?)"
                ")"
            )
            page_params.extend([priority, priority, updated_at, priority, updated_at, job_id])

        page_size = max(1, min(limit, 200))
        where = f"WHERE {' AND '.join(page_clauses)}" if page_clauses else ""
        offset_value = max(0, int(offset or 0)) if not cursor else 0
        rows = conn.execute(
            f"SELECT * FROM jobs {where} "
            "ORDER BY CASE WHEN status = 'waiting_approval' THEN 0 ELSE 1 END, updated_at DESC, job_id DESC "
            "LIMIT ? OFFSET ?",
            (*page_params, page_size + 1, offset_value),
        ).fetchall()
        has_more = len(rows) > page_size
        page_rows = rows[:page_size]
        next_cursor = _encode_list_cursor(page_rows[-1]) if has_more and page_rows else None
        return [_row_to_job(row) for row in page_rows], next_cursor, total

    def stats(self, *, archived: bool | None = False) -> dict[str, Any]:
        conn = _get_conn()
        clauses = []
        params: list[Any] = []
        if archived is not None:
            clauses.append("archived = ?")
            params.append(1 if archived else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        status_rows = conn.execute(
            f"SELECT status, COUNT(*) AS count FROM jobs {where} GROUP BY status",
            params,
        ).fetchall()
        type_rows = conn.execute(
            f"SELECT job_type, COUNT(*) AS count FROM jobs {where} GROUP BY job_type ORDER BY count DESC, job_type",
            params,
        ).fetchall()
        by_status = {str(row["status"]): int(row["count"]) for row in status_rows}
        by_type = {str(row["job_type"]): int(row["count"]) for row in type_rows}
        return {
            "total": sum(by_status.values()),
            "attention": by_status.get("waiting_approval", 0),
            "active": by_status.get("queued", 0) + by_status.get("running", 0),
            "failed": by_status.get("failed", 0) + by_status.get("interrupted", 0),
            "by_status": by_status,
            "by_type": by_type,
        }

    def latest(
        self,
        *,
        job_type: str | None = None,
        status: str | None = None,
        channel_id: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None:
        """Return the most recently updated matching job without status prioritization.

        ``list()`` intentionally puts ``waiting_approval`` jobs first for management UIs. That
        ordering is unsafe for conversational references such as "批准": a completed newer job
        must not be skipped in favor of an older pending job. Callers that need the actual latest
        activity should use this method instead of taking ``list(..., limit=1)[0]``.
        """
        conn = _get_conn()
        clauses = []
        params: list[Any] = []
        if job_type:
            clauses.append("job_type = ?")
            params.append(job_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if channel_id:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        if archived is not None:
            clauses.append("archived = ?")
            params.append(1 if archived else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY updated_at DESC, created_at DESC LIMIT 1",
            params,
        ).fetchone()
        return _row_to_job(row) if row is not None else None

    def cancel(self, job_id: str, reason: str | None = None) -> dict[str, Any]:
        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("status") in TERMINAL_STATES:
                return job
            now = now_text()
            job["status"] = "cancelled"
            job["status_reason"] = "user_cancelled"
            job["health_status"] = None
            job["updated_at"] = now
            job["finished_at"] = now
            job["error"] = {"code": "cancelled", "message": reason or "用户取消", "retryable": False}
            return job

        job = self._update(job_id, mutate)
        if str(job.get("job_type") or "").startswith("research") and job.get("status") == "cancelled":
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
                        inputs={"job_status_before_interrupt": "running_or_queued"},
                        outputs={"job_status": "cancelled"},
                        transition_reason="job cancellation requested",
                        status="cancelled",
                        interrupt={"kind": "user_cancelled", "reason": reason or "用户取消"},
                    )
            except Exception:
                pass
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        child_ids = [
            str(value).strip()
            for value in [
                result.get("active_child_job_id"),
                *(result.get("child_job_ids") or [] if isinstance(result.get("child_job_ids"), list) else []),
            ]
            if str(value or "").strip() and str(value).strip() != job_id
        ]
        for child_id in dict.fromkeys(child_ids):
            try:
                child = self.get(child_id)
            except ValueError:
                continue
            if child.get("status") not in TERMINAL_STATES | {"waiting_approval"}:
                self.cancel(child_id, f"父任务 {job_id} 已取消")
        emit_event(
            "job.cancelled",
            "jobs",
            "Background job was cancelled.",
            level="warning",
            data={"job_id": job_id, "job_type": job.get("job_type"), "reason": reason or "用户取消"},
        )
        return job

    def set_archived(self, job_id: str, archived: bool) -> dict[str, Any]:
        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            job["archived"] = archived
            job["updated_at"] = now_text()
            return job

        return self._update(job_id, mutate)

    def update_progress(self, job_id: str, *, current: int, total: int, message: str) -> None:
        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("status") in TERMINAL_STATES:
                return job
            now = now_text()
            job["progress"] = {"current": current, "total": total, "message": message}
            job["updated_at"] = now
            job["last_progress_at"] = now
            if job.get("status") in {"queued", "running"}:
                job["health_status"] = "healthy"
            return job

        job = self._update(job_id, mutate)
        if job.get("status") in TERMINAL_STATES:
            return
        emit_event(
            "job.progress.updated",
            "jobs",
            message or "Background job progress updated.",
            data={
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "current": current,
                "total": total,
            },
        )

    def set_steps(self, job_id: str, steps: list[dict[str, Any]]) -> None:
        now = now_text()
        normalized = _normalize_steps(steps)

        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            job["steps"] = normalized
            job["current_step"] = None
            job["updated_at"] = now
            job["last_progress_at"] = now
            job["progress"] = {"current": 0, "total": max(1, len(normalized)), "message": "queued"}
            return job

        job = self._update(job_id, mutate)
        emit_event(
            "job.steps.initialized",
            "jobs",
            "Background job steps initialized.",
            data={
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "steps": [{"id": step["id"], "title": step["title"]} for step in normalized],
            },
        )

    def update_step(
        self,
        job_id: str,
        step_id: str,
        status: str,
        *,
        summary: str | None = None,
        items: list[Any] | None = None,
        progress_message: str | None = None,
    ) -> None:
        now = now_text()

        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("status") in TERMINAL_STATES:
                return job
            steps = job.get("steps") or []
            for step in steps:
                if step.get("id") != step_id:
                    continue
                previous = step.get("status")
                step["status"] = status
                if summary is not None:
                    step["summary"] = summary
                if items is not None:
                    step["items"] = items
                if status == "running" and not step.get("started_at"):
                    step["started_at"] = now
                if status in STEP_TERMINAL_STATES and previous != status:
                    step["finished_at"] = now
                    step["elapsed_ms"] = _elapsed_ms(step.get("started_at"), now)
                break
            job["current_step"] = step_id if status == "running" else job.get("current_step")
            if status in STEP_TERMINAL_STATES and job.get("current_step") == step_id:
                job["current_step"] = None
            completed = sum(1 for step in steps if step.get("status") in STEP_TERMINAL_STATES)
            job["progress"] = {
                "current": completed,
                "total": max(1, len(steps)),
                "message": progress_message or summary or status,
            }
            job["updated_at"] = now
            job["last_progress_at"] = now
            if job.get("status") in {"queued", "running"}:
                job["health_status"] = "healthy"
            return job

        job = self._update(job_id, mutate)
        if job.get("status") in TERMINAL_STATES:
            return
        level = "error" if status == "failed" else "info"
        emit_event(
            f"job.step.{status}",
            "jobs",
            summary or progress_message or f"Job step {step_id} is {status}.",
            level=level,
            data={
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "step_id": step_id,
                "status": status,
                "progress": job.get("progress"),
            },
        )

    def update_result(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        progress_message: str | None = None,
    ) -> dict[str, Any]:
        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("status") in TERMINAL_STATES and status != job.get("status"):
                return job
            now = now_text()
            if status is not None:
                job["status"] = status
                job["status_reason"] = None
                if status == "running":
                    from .runtime_state_service import current_instance_id

                    job["owner_instance_id"] = current_instance_id()
                    job["started_at"] = job.get("started_at") or now
                    job["last_heartbeat_at"] = now
                    job["finished_at"] = None
                    job["health_status"] = "healthy"
                elif status in TERMINAL_STATES | {"waiting_approval"}:
                    job["finished_at"] = now
                    job["health_status"] = None
            if result is not None:
                job["result"] = result
            if artifacts is not None:
                job["artifacts"] = artifacts
            job["updated_at"] = now
            job["last_progress_at"] = now
            if progress_message:
                total = max(1, int((job.get("progress") or {}).get("total") or 1))
                current = total if status in TERMINAL_STATES | {"waiting_approval"} else int(
                    (job.get("progress") or {}).get("current") or 0
                )
                job["progress"] = {"current": current, "total": total, "message": progress_message}
            return job

        return self._update(job_id, mutate)

    def merge_running_result(
        self,
        job_id: str,
        *,
        result_patch: dict[str, Any],
        progress_message: str | None = None,
    ) -> dict[str, Any]:
        """Atomically publish a partial result while a job is still active.

        Long-running jobs can expose completed stage outputs without replacing fields
        written by another active workflow. Once a job reaches review or a terminal
        state, a late snapshot is ignored so it cannot overwrite the final result.
        """

        def mutate(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("status") not in {"queued", "running"}:
                return job
            current = job.get("result")
            merged = dict(current) if isinstance(current, dict) else {}
            merged.update(result_patch)
            now = now_text()
            job["result"] = merged
            job["updated_at"] = now
            job["last_progress_at"] = now
            if progress_message:
                progress = dict(job.get("progress") or {})
                progress["message"] = progress_message
                job["progress"] = progress
            return job

        return self._update(job_id, mutate)

    def _run(self, job_id: str, target: Callable[[dict[str, Any]], Any]) -> None:
        job = self.get(job_id)
        if job.get("status") in TERMINAL_STATES:
            return
        input_data = dict(job.get("input") or {})
        from .runtime_state_service import current_instance_id

        owner_instance_id = current_instance_id()

        def mutate_running(job: dict[str, Any]) -> dict[str, Any]:
            now = now_text()
            job["status"] = "running"
            job["status_reason"] = None
            job["owner_instance_id"] = owner_instance_id
            job["started_at"] = job.get("started_at") or now
            job["last_heartbeat_at"] = now
            job["last_progress_at"] = now
            job["finished_at"] = None
            job["health_status"] = "healthy"
            job["updated_at"] = now
            total = max(1, len(job.get("steps") or []) or int((job.get("progress") or {}).get("total") or 1))
            job["progress"] = {"current": 0, "total": total, "message": "running"}
            return job

        self._update(job_id, mutate_running)
        # Tag every observability event emitted while this job runs with its
        # job_id, so the "job 轨迹" view (get_job_trace) has something to show --
        # most worker/tool emit_event() calls never pass job_id explicitly.
        # See docs/modules/observability-redesign-v0.1.md 3.11.
        set_observation_context(job_id=job_id)
        emit_event(
            "job.started",
            "jobs",
            "Background job started.",
            data={"job_id": job_id, "job_type": job.get("job_type")},
        )
        try:
            try:
                result = target(input_data)
            except Exception as exc:

                def mutate_failed(job: dict[str, Any]) -> dict[str, Any]:
                    if job.get("status") in TERMINAL_STATES:
                        return job
                    now = now_text()
                    for step in job.get("steps") or []:
                        if step.get("status") == "running":
                            step["status"] = "failed"
                            step["summary"] = str(exc) or type(exc).__name__
                            step["finished_at"] = now
                    job["current_step"] = None
                    job["status"] = "failed"
                    job["status_reason"] = "unhandled_exception"
                    job["health_status"] = None
                    job["updated_at"] = now
                    job["last_progress_at"] = now
                    job["finished_at"] = now
                    error = _classify_job_error(exc)
                    job["error"] = {
                        **error,
                        "details": {
                            "type": type(exc).__name__,
                            **(getattr(exc, "details", {}) if isinstance(getattr(exc, "details", {}), dict) else {}),
                        },
                    }
                    total = max(1, len(job.get("steps") or []) or int((job.get("progress") or {}).get("total") or 1))
                    current = sum(
                        1
                        for step in job.get("steps") or []
                        if step.get("status") in {"succeeded", "failed", "waiting_approval"}
                    )
                    job["progress"] = {"current": max(1, current), "total": total, "message": "failed"}
                    return job

                failed_job = self._update(job_id, mutate_failed)
                if failed_job.get("status") != "failed":
                    return
                for step in failed_job.get("steps") or []:
                    if step.get("status") != "failed":
                        continue
                    emit_event(
                        "job.step.failed",
                        "jobs",
                        str(step.get("summary") or "Background job step failed."),
                        level="error",
                        data={
                            "job_id": job_id,
                            "job_type": failed_job.get("job_type"),
                            "step_id": step.get("id"),
                            "status": "failed",
                            "progress": failed_job.get("progress"),
                        },
                    )
                emit_event(
                    "job.failed",
                    "jobs",
                    "Background job failed.",
                    level="error",
                    data={
                        "job_id": job_id,
                        "job_type": failed_job.get("job_type"),
                        "error": failed_job.get("error"),
                    },
                )
                return
        finally:
            set_observation_context(job_id=None)

        def mutate_succeeded(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("status") in TERMINAL_STATES:
                return job
            result_status = result.get("status") if isinstance(result, dict) else None
            job["status"] = result_status if result_status in {"waiting_approval"} else "succeeded"
            now = now_text()
            job["status_reason"] = None
            job["health_status"] = None
            job["updated_at"] = now
            job["last_progress_at"] = now
            job["finished_at"] = now
            job["result"] = result
            if isinstance(result, dict) and result.get("artifacts"):
                job["artifacts"] = result["artifacts"]
            total = max(1, int((job.get("progress") or {}).get("total") or 1))
            job["progress"] = {"current": total, "total": total, "message": str(job["status"])}
            return job

        succeeded_job = self._update(job_id, mutate_succeeded)
        if succeeded_job.get("status") not in {"waiting_approval", "succeeded"}:
            return
        emit_event(
            "job.waiting_approval" if succeeded_job.get("status") == "waiting_approval" else "job.succeeded",
            "jobs",
            "Background job finished.",
            data={
                "job_id": job_id,
                "job_type": succeeded_job.get("job_type"),
                "status": succeeded_job.get("status"),
            },
        )

    def _insert(self, job: dict[str, Any]) -> None:
        conn = _get_conn()
        channel_id = str((job.get("input") or {}).get("channel_id") or "") or None
        conn.execute(
            "INSERT INTO jobs ("
            "job_id, job_type, status, created_at, updated_at, channel_id, payload, archived, "
            "owner_instance_id, status_reason, started_at, last_heartbeat_at, last_progress_at, "
            "finished_at, health_status, environment, attempt, retry_of_job_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job["job_id"],
                job["job_type"],
                job["status"],
                job["created_at"],
                job["updated_at"],
                channel_id,
                json.dumps(job, ensure_ascii=False),
                1 if job.get("archived") else 0,
                job.get("owner_instance_id"),
                job.get("status_reason"),
                job.get("started_at"),
                job.get("last_heartbeat_at"),
                job.get("last_progress_at"),
                job.get("finished_at"),
                job.get("health_status"),
                job.get("environment") or "dev",
                int(job.get("attempt") or 1),
                job.get("retry_of_job_id"),
            ),
        )
        conn.commit()

    def _update(self, job_id: str, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        """Read-modify-write a job row inside an IMMEDIATE transaction.

        BEGIN IMMEDIATE takes SQLite's write lock up front, so two processes racing to
        update the same job_id serialize on the lock (one blocks up to busy_timeout,
        then retries its read) instead of silently clobbering each other's writes the
        way a plain UPDATE after an unguarded SELECT would.
        """
        conn = _get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"job not found: {job_id}")
            job = _row_to_job(row)
            job = mutate(job)
            channel_id = str((job.get("input") or {}).get("channel_id") or "") or None
            conn.execute(
                "UPDATE jobs SET "
                "status = ?, updated_at = ?, channel_id = ?, payload = ?, archived = ?, "
                "owner_instance_id = ?, status_reason = ?, started_at = ?, last_heartbeat_at = ?, "
                "last_progress_at = ?, finished_at = ?, health_status = ?, environment = ?, "
                "attempt = ?, retry_of_job_id = ? WHERE job_id = ?",
                (
                    job["status"],
                    job["updated_at"],
                    channel_id,
                    json.dumps(job, ensure_ascii=False),
                    1 if job.get("archived") else 0,
                    job.get("owner_instance_id"),
                    job.get("status_reason"),
                    job.get("started_at"),
                    job.get("last_heartbeat_at"),
                    job.get("last_progress_at"),
                    job.get("finished_at"),
                    job.get("health_status"),
                    job.get("environment") or "dev",
                    int(job.get("attempt") or 1),
                    job.get("retry_of_job_id"),
                    job_id,
                ),
            )
            conn.commit()
            return job
        except BaseException:
            conn.rollback()
            raise


job_service = JobService()


def _encode_list_cursor(row: sqlite3.Row) -> str:
    payload = [0 if row["status"] == "waiting_approval" else 1, row["updated_at"], row["job_id"]]
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")


def _decode_list_cursor(cursor: str) -> tuple[int, str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        priority, updated_at, job_id = value
        if priority not in (0, 1) or not isinstance(updated_at, str) or not isinstance(job_id, str):
            raise ValueError
        return priority, updated_at, job_id
    except Exception as exc:
        raise ValueError("invalid jobs cursor") from exc


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    job = json.loads(row["payload"])
    keys = set(row.keys())
    for key in (
        "owner_instance_id",
        "status_reason",
        "started_at",
        "last_heartbeat_at",
        "last_progress_at",
        "finished_at",
        "health_status",
        "environment",
        "attempt",
        "retry_of_job_id",
    ):
        if key in keys:
            job[key] = row[key]
    if "archived" in keys:
        job["archived"] = bool(row["archived"])
    job["status"] = row["status"]
    job["updated_at"] = row["updated_at"]
    return job


def _elapsed_ms(started_at: str | None, finished_at: str | None) -> int | None:
    if not started_at or not finished_at:
        return None
    try:
        started = time.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        finished = time.strptime(finished_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return max(0, int((time.mktime(finished) - time.mktime(started)) * 1000))


def _normalize_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": step["id"],
            "title": step["title"],
            "status": step.get("status") or "pending",
            "summary": step.get("summary") or "",
            "started_at": step.get("started_at"),
            "finished_at": step.get("finished_at"),
            "elapsed_ms": step.get("elapsed_ms"),
            "items": step.get("items") or [],
        }
        for step in steps
    ]


def _classify_job_error(exc: Exception) -> dict[str, Any]:
    message = str(exc) or type(exc).__name__
    explicit_code = str(getattr(exc, "code", "") or "").strip()
    if explicit_code:
        return {
            "code": explicit_code,
            "message": message,
            "retryable": bool(getattr(exc, "retryable", False)),
        }
    lowered = message.lower()
    if isinstance(exc, TimeoutError) or "timeout" in lowered or "timed out" in lowered or "超时" in message:
        return {"code": "upstream_timeout", "message": message, "retryable": True}
    if (
        "429" in lowered
        or "rate limit" in lowered
        or "rate_limit" in lowered
        or "rate-limited" in lowered
        or "限流" in message
    ):
        return {"code": "upstream_rate_limited", "message": message, "retryable": True}
    return {"code": "internal_error", "message": message, "retryable": False}
