"""System/runtime service functions."""

from __future__ import annotations

import os

try:
    from ...tools._registry import KNOWLEDGE_BASE
except ImportError:  # pragma: no cover
    from tools._registry import KNOWLEDGE_BASE

from .common import now_text


def health() -> dict:
    return {
        "service": "Deep Research API",
        "version": "0.1.0",
        "time": now_text(),
    }


def capabilities() -> dict:
    return {
        "chat": True,
        "research_jobs": True,
        "knowledge_base_management": True,
        "knowledge_qa": True,
        "lightrag_index": True,
        "knowledge_evaluation": True,
        "observability": True,
        "model_check": True,
    }


def runtime_config() -> dict:
    return {
        "model_name": os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        "base_url_configured": bool(os.getenv("OPENAI_BASE_URL")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "1024")),
        "lightrag_enabled": True,
        "knowledge_base_root": str(KNOWLEDGE_BASE),
        "log_enabled": True,
    }
