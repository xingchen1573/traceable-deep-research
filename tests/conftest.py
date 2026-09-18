"""Session-wide test fixtures."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


_RUNTIME_ENV_NAMES = ("FEISHU_WS_HEARTBEAT_PATH",)
_PREVIOUS_RUNTIME_ENV = {name: os.environ.get(name) for name in _RUNTIME_ENV_NAMES}
_TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="deep-research-tests-"))
os.environ["FEISHU_WS_HEARTBEAT_PATH"] = str(_TEST_RUNTIME_ROOT / "feishu_ws_heartbeat.txt")

import observability as _observability

_PREVIOUS_OBSERVABILITY_DIR = _observability.OBSERVABILITY_DIR
_observability.OBSERVABILITY_DIR = _TEST_RUNTIME_ROOT / "observability"


@pytest.fixture(scope="session", autouse=True)
def isolate_runtime_outputs():
    """Keep logs and transport heartbeats outside the checkout during tests."""

    yield
    for name, previous in _PREVIOUS_RUNTIME_ENV.items():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    _observability.OBSERVABILITY_DIR = _PREVIOUS_OBSERVABILITY_DIR
    shutil.rmtree(_TEST_RUNTIME_ROOT, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def isolate_conversation_db(tmp_path_factory: pytest.TempPathFactory):
    """Redirect channel_service/conversation_service/job_service off the real data/conversations.db.

    These modules point at the exact same physical file in production
    (backend/services/channel_service.py, conversation_service.py and job_service.py all define
    `_DB_PATH = _DB_DIR / "conversations.db"`) -- channel_service owns the `channels` table,
    conversation_service owns `conversations`/`messages`, job_service owns `jobs` (added so job
    state survives the backend/feishu-worker container split -- see
    project_docker_deploy_cross_process_job_state_bug memory), sharing one SQLite file via WAL
    mode. This fixture must redirect ALL modules to the *same* temp file to keep that topology
    faithful; pointing them at different files would diverge from production for no reason.

    These modules also cache a thread-local sqlite3 connection (see each module's `_get_conn()`),
    so patching the path constant alone is not enough if a connection was already opened on this
    thread -- we must also drop any cached connection to force a fresh one against the redirected
    path. `_get_conn()` re-reads the module-level `_DB_PATH` every time it needs to open a *new*
    connection (not just once at import time), so this remains correct even for connections opened
    later from a different thread (e.g. a test's ThreadPoolExecutor worker).

    Session-scoped + autouse: pytest sets this up before the first test body in the whole run
    executes, and before that none of these modules' `_get_conn()` has ever been called (they
    only touch the database inside function bodies, never at import time), so there is no
    real-file connection to accidentally leak from before the redirect takes effect.

    Without this fixture, tests that exercise the real Feishu WS / channel chat flow (e.g.
    test_backend_api.py's test_feishu_ws_message_routes_without_webhook_token) silently write
    test fixture rows into the developer's real, currently-in-use data/conversations.db --
    this is exactly what produced the "长连接消息" / "ws ok channel:feishu:oc_ws_chat:main"
    message pairs a real user spotted in the actual Web channel console.
    """
    from backend.services import channel_service, conversation_service, feishu_dedupe_service, job_service, research_plan_store

    db_dir = tmp_path_factory.mktemp("shared_conversations_db")
    db_path = db_dir / "conversations.db"
    for module in (channel_service, conversation_service, feishu_dedupe_service, job_service, research_plan_store):
        if hasattr(module, "_DB_DIR"):
            module._DB_DIR = db_dir
        module._DB_PATH = db_path
        if hasattr(module._local, "conn"):
            module._local.conn.close()
            del module._local.conn
    yield
