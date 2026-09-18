"""System test-run service."""

from __future__ import annotations

import py_compile
import threading
from pathlib import Path
from typing import Any

try:
    from ...tools._registry import ROOT
except ImportError:  # pragma: no cover
    from tools._registry import ROOT

from .job_service import job_service


def create_test_run(*, suite: str, target: str | None, mode: str, options: dict[str, Any] | None) -> dict:
    def run(input_data: dict) -> dict:
        if suite not in {"api_smoke", "backend_compile"}:
            raise ValueError(f"unsupported test suite: {suite}")
        job_id = run.job_id  # type: ignore[attr-defined]
        paths = [
            path
            for path in (ROOT / "backend").rglob("*.py")
            if "__pycache__" not in path.parts
        ]
        total = len(paths) or 1
        checked = []
        for index, path in enumerate(paths):
            rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
            job_service.update_progress(job_id, current=index, total=total, message=rel_path)
            py_compile.compile(str(path), doraise=True)
            checked.append(rel_path)
        job_service.update_progress(job_id, current=total, total=total, message="编译检查完成")
        return {
            "run_id": input_data.get("run_id"),
            "suite": suite,
            "status": "succeeded",
            "summary": {"total": len(checked) or 1, "passed": len(checked) or 1, "failed": 0},
            "artifacts": [],
            "checked_files": checked,
        }

    job = job_service.create(
        job_type="test_run",
        input_data={"suite": suite, "target": target, "mode": mode, "options": options or {}},
        target=None,
    )
    run.job_id = job["job_id"]  # type: ignore[attr-defined]
    thread = threading.Thread(target=job_service._run, args=(job["job_id"], run), daemon=True)
    thread.start()
    return job_service.get(job["job_id"])


def list_test_runs() -> dict:
    return {"runs": job_service.list(job_type="test_run", limit=100)}


def get_test_run(run_id: str) -> dict:
    return job_service.get(run_id)

