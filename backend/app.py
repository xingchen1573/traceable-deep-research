"""FastAPI application initialization."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .dependencies import set_app_state
from .routes import register_routes
from .services.push_service import start_push_scheduler, stop_push_scheduler
from .services.runtime_state_service import start_runtime_monitor

try:
    from ..main import load_env_files
except ImportError:  # pragma: no cover - script mode fallback
    from main import load_env_files

APP_TITLE = "Deep Research API"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # docs/VISION.md 第5节 -- no-op unless PUSH_SCHEDULER_ENABLED is set, so importing this
    # module in dev/test never silently starts a background thread making real model/Feishu calls.
    start_push_scheduler()
    runtime_monitor = start_runtime_monitor(role="backend", reconcile=True)
    try:
        yield
    finally:
        stop_push_scheduler()
        runtime_monitor.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Populate os.environ from .env immediately, not just lazily on first agent build --
    # otherwise routes that read os.getenv() directly (models/config, models/check,
    # providers) report every credential as missing until someone happens to first talk
    # to the agent (which is what actually triggers main.py::load_env_files()).
    load_env_files()

    app = FastAPI(title=APP_TITLE, version="0.1.0", lifespan=_lifespan)

    # Initialize app state
    app.state.agent = None
    app.state.review_agent = None
    app.state.kb_qa_agent = None
    app.state.pending_interrupts = {}

    # Share state with dependencies module
    set_app_state(app.state)

    # Register all routes
    register_routes(app)

    return app


# Create the app instance
app = create_app()
