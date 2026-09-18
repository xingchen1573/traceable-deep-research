"""Route registration for the API."""

from __future__ import annotations

from fastapi import FastAPI

from .chat import router as chat_router
from .chat_v1 import router as chat_v1_router
from .channels import router as channels_router
from .conversations import router as conversations_router
from .dream import router as dream_router
from .evaluations import router as evaluations_router
from .feishu import router as feishu_router
from .indexes import router as indexes_router
from .jobs import router as jobs_router
from .knowledge_bases import router as knowledge_bases_router
from .knowledge_tree import router as knowledge_tree_router
from .model_management import router as model_management_router
from .observability import router as observability_router
from .observability_v1 import router as observability_v1_router
from .research import router as research_router
from .system import router as system_router
from .tests import router as tests_router


def register_routes(app: FastAPI) -> None:
    """Register all route modules with the FastAPI app."""
    app.include_router(chat_router)
    app.include_router(channels_router)
    app.include_router(conversations_router)
    app.include_router(dream_router)
    app.include_router(observability_router)
    app.include_router(system_router)
    app.include_router(chat_v1_router)
    app.include_router(research_router)
    app.include_router(knowledge_bases_router)
    app.include_router(knowledge_tree_router)
    app.include_router(indexes_router)
    app.include_router(evaluations_router)
    app.include_router(feishu_router)
    app.include_router(tests_router)
    app.include_router(observability_v1_router)
    app.include_router(model_management_router)
    app.include_router(jobs_router)
