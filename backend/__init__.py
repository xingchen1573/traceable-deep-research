"""Server package for the Deep Research API.

Importing pure modules such as ``backend.research.evidence_native`` must not initialize
the server or read local configuration. The supported ASGI entrypoint is
``backend.app:app``; callers that need an application factory may use ``create_app``.
"""

from __future__ import annotations

from typing import Any


__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    """Resolve the historical package-level ASGI export only when it is requested."""

    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .app import app as application

    globals()["app"] = application
    return application


def create_app() -> Any:
    """Create the FastAPI application without making package import stateful."""

    from .app import create_app as _create_app

    return _create_app()
