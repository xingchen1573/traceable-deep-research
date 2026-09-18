from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from backend.research.evidence_native import EvidenceNativeContractError
from scripts.demo_public_evidence_pipeline import DEFAULT_FIXTURE, _load_fixture, run_demo


ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = ROOT / "scripts" / "demo_public_evidence_pipeline.py"


def test_test_runtime_outputs_are_outside_checkout() -> None:
    import observability
    from backend.services import feishu_ws_service

    assert not observability.OBSERVABILITY_DIR.resolve().is_relative_to(ROOT)
    assert not feishu_ws_service.DEFAULT_HEARTBEAT_PATH.resolve().is_relative_to(ROOT)


def test_backend_package_import_is_inert(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import backend; "
                "assert 'backend.app' not in sys.modules; "
                "from backend import create_app; "
                "assert callable(create_app); "
                "assert 'backend.app' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_backend_package_lazily_preserves_fastapi_app_export(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import backend; "
                "assert 'backend.app' not in sys.modules; "
                "from backend import app; "
                "from fastapi import FastAPI; "
                "assert isinstance(app, FastAPI); "
                "assert backend.app is app; "
                "assert 'backend.app' in sys.modules"
            ),
        ],
        cwd=tmp_path,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
            "RECON_TOKEN_SECRET": "test-only-recon-secret",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_evidence_demo_is_deterministic_and_structurally_valid() -> None:
    fixture = _load_fixture(DEFAULT_FIXTURE)

    first = run_demo(fixture)
    second = run_demo(copy.deepcopy(fixture))

    assert first == second
    assert first["status"] == "passed"
    assert first["execution_mode"] == "offline_deterministic_no_model"
    assert first["citation_validation"]["status"] == "passed"
    assert first["semantic_validation"]["status"] == "not_run"
    assert set(first["evidence_context_packet"]["allowed_evidence_ids"]) == {
        "ev_alpha_checkpoint",
        "ev_beta_snapshot",
        "ev_alpha_partition",
    }
    assert first["context_diagnostics"]["selected_conflict_or_limitation_count"] == 1
    assert [item["number"] for item in first["citation_reader"]] == [1, 2]


def test_public_evidence_demo_rejects_untraceable_quote() -> None:
    fixture = _load_fixture(DEFAULT_FIXTURE)
    fixture["evidence_map"][0]["cleaned_quote"] = "原始 Chunk 中不存在的句子。"
    fixture["evidence_map"][0]["raw_quote"] = "原始 Chunk 中不存在的句子。"
    fixture["evidence_map"][0]["evidence_text"] = "原始 Chunk 中不存在的句子。"

    with pytest.raises(EvidenceNativeContractError, match="not traceable"):
        run_demo(fixture)


def test_public_evidence_demo_cli_does_not_read_env_or_open_network(tmp_path: Path) -> None:
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import builtins
            import io
            import os
            import socket
            from pathlib import Path

            _builtin_open = builtins.open
            _io_open = io.open
            _repo = Path(os.environ["DEMO_REPO_ROOT"]).resolve()
            _protected = tuple(
                (_repo / name).resolve()
                for name in ("data", "knowledge_base", "lightrag_workspaces", "logs", "memory")
            )

            def _guarded_open(original, file, mode="r", *args, **kwargs):
                try:
                    path = Path(os.fspath(file)).resolve()
                except (TypeError, ValueError):
                    return original(file, mode, *args, **kwargs)
                if path.name == ".env" or path.name.startswith(".env."):
                    raise RuntimeError(f"public demo attempted to read environment file: {path.name}")
                if any(flag in str(mode) for flag in "wax+") and any(
                    path == root or root in path.parents for root in _protected
                ):
                    raise RuntimeError(f"public demo attempted a protected write: {path}")
                return original(file, mode, *args, **kwargs)

            builtins.open = lambda file, mode="r", *args, **kwargs: _guarded_open(
                _builtin_open, file, mode, *args, **kwargs
            )
            io.open = lambda file, mode="r", *args, **kwargs: _guarded_open(
                _io_open, file, mode, *args, **kwargs
            )

            def _deny_network(*_args, **_kwargs):
                raise RuntimeError("public demo attempted network access")

            socket.create_connection = _deny_network
            socket.socket.connect = _deny_network
            socket.socket.connect_ex = _deny_network
            """
        ),
        encoding="utf-8",
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "COOKIE"))
    }
    environment.update(
        {
            "DEMO_REPO_ROOT": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(guard), str(ROOT))),
        }
    )
    command = [sys.executable, str(DEMO_SCRIPT), "--compact"]

    first = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert json.loads(first.stdout) == json.loads(second.stdout)
