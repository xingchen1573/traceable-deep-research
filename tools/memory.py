"""Memory-related tools."""

from __future__ import annotations

import json

from langchain_core.tools import tool

try:
    from .. import memory as memory_store
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    import memory as memory_store
    from observability import emit_event

from ._registry import mark_memory_surfaced


@tool
def search_memory(query: str) -> str:
    """Search long-term memory topics without loading every topic body."""
    return memory_store.search_memory(query)


@tool
def load_memory_topic(topic_id: str) -> str:
    """Load one long-term memory topic body by topic id."""
    content = memory_store.load_memory_topic(topic_id)
    if not content.startswith("Memory topic not found:"):
        safe_id = memory_store.sanitize_topic_id(topic_id)
        mark_memory_surfaced(safe_id)
        emit_event(
            "memory.topic.loaded",
            "tools",
            "Loaded memory topic through tool.",
            data={"topic_id": safe_id, "chars": len(content)},
        )
    else:
        emit_event(
            "memory.topic.loaded",
            "tools",
            "Memory topic tool load failed.",
            level="warning",
            data={"topic_id": topic_id, "status": "not_found"},
        )
    return content


@tool
def write_memory_topic(
    topic_id: str,
    title: str,
    summary: str,
    body: str,
    trust_level: str = "medium",
) -> str:
    """Write a formal memory topic file, then refresh MEMORY.md."""
    return memory_store.write_memory_topic(
        topic_id=topic_id,
        title=title,
        summary=summary,
        body=body,
        trust_level=trust_level,
    )


@tool
def read_memory_index() -> str:
    """Read the lightweight MEMORY.md index."""
    content = memory_store.read_memory_index()
    emit_event(
        "memory.index.loaded",
        "tools",
        "Loaded MEMORY.md through tool.",
        data={"chars": len(content)},
    )
    return content


@tool
def list_memory_topics() -> str:
    """List formal memory topic metadata as JSON."""
    records = memory_store.list_memory_topics()
    payload = [
        {
            "topic_id": record.topic_id,
            "title": record.title,
            "summary": record.summary,
            "path": record.path,
            "trust_level": record.trust_level,
            "updated_at": record.updated_at,
        }
        for record in records
    ]
    emit_event(
        "memory.topics.listed",
        "tools",
        "Listed formal memory topics.",
        data={"count": len(payload)},
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def refresh_memory_index() -> str:
    """Refresh MEMORY.md from current topic files."""
    content = memory_store.refresh_memory_index()
    emit_event(
        "memory.index.refreshed",
        "tools",
        "Refreshed MEMORY.md through tool.",
        data={"chars": len(content)},
    )
    return content


@tool
def read_dream_report() -> str:
    """Read the current dream_report.md content."""
    content = memory_store.read_dream_report()
    emit_event(
        "dream.report.loaded",
        "tools",
        "Loaded dream_report.md through tool.",
        data={"chars": len(content)},
    )
    return content
