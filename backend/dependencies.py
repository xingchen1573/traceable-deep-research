"""Agent factory functions and state management."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import Command

try:
    from .. import memory as memory_store
    from ..main import (
        DISPATCH_BRIDGE_TOOL_NAMES,
        build_agent,
        build_dream_model,
        build_kb_qa_agent,
        build_review_agent,
        extract_interrupts,
        extract_tool_results,
        final_message_text,
        latest_update_ai_text,
        normalize_stream_chunk,
    )
    from ..observability import emit_event, new_run_id, set_observation_context
    from ..tools import ensure_demo_files, set_current_thread_id
except ImportError:  # pragma: no cover
    import memory as memory_store
    from main import (
        DISPATCH_BRIDGE_TOOL_NAMES,
        build_agent,
        build_dream_model,
        build_kb_qa_agent,
        build_review_agent,
        extract_interrupts,
        extract_tool_results,
        final_message_text,
        latest_update_ai_text,
        normalize_stream_chunk,
    )
    from observability import emit_event, new_run_id, set_observation_context
    from tools import ensure_demo_files, set_current_thread_id

from .helpers import serialize_interrupt


# Global app state reference (set by app.py)
_app_state: Any = None


def set_app_state(state: Any) -> None:
    """Set the FastAPI app state reference."""
    global _app_state
    _app_state = state


def get_agent() -> Any:
    """Get or create the main agent (lazy initialization)."""
    if _app_state.agent is None:
        memory_store.ensure_memory_files()
        ensure_demo_files()
        _app_state.agent = build_agent()
    return _app_state.agent


def get_review_agent() -> Any:
    """Get or create the review agent (lazy initialization)."""
    if _app_state.review_agent is None:
        memory_store.ensure_memory_files()
        _app_state.review_agent = build_review_agent()
    return _app_state.review_agent


def get_kb_qa_agent() -> Any:
    """Get or create the dedicated kb_qa agent (lazy initialization).

    See docs/modules/kb-qa-channel-bypass-v0.1.md 4.2 -- this agent is persistent (has its own
    checkpointer) but is driven by whatever thread_id the caller passes in (the per-channel
    channel_service.thread_id_for(channel_id, "kb_qa") thread_id, not a fixed session id), so a
    single cached instance here serves every channel's kb_qa conversations, the same way
    get_agent() serves every channel's main-mode conversations.
    """
    if _app_state.kb_qa_agent is None:
        memory_store.ensure_memory_files()
        ensure_demo_files()
        _app_state.kb_qa_agent = build_kb_qa_agent()
    return _app_state.kb_qa_agent


def thread_config(thread_id: str) -> dict[str, Any]:
    """Create a LangGraph config for a thread."""
    return {"configurable": {"thread_id": thread_id}}


def new_thread_id() -> str:
    """Generate a new unique thread ID."""
    return f"thread_{uuid.uuid4().hex[:12]}"


def get_pending_interrupts() -> dict[str, Any]:
    """Get the pending interrupts dict from app state."""
    return _app_state.pending_interrupts


def set_pending_interrupt(thread_id: str, user_message: str, interrupts: list[Any], run_id: str) -> dict[str, Any]:
    """Store a pending interrupt for later resume."""
    record = {
        "original_user_message": user_message,
        "interrupts": interrupts,
        "created_at": _app_state.now_text if hasattr(_app_state, 'now_text') else "",
        "run_id": run_id,
    }
    _app_state.pending_interrupts[thread_id] = record
    return record


def clear_pending_interrupt(thread_id: str) -> None:
    """Clear a pending interrupt after resume."""
    _app_state.pending_interrupts.pop(thread_id, None)


def clear_agent_thread_state(agent: Any, thread_id: str) -> bool:
    """Discard a failed LangGraph turn so the next message starts cleanly.

    LangGraph checkpoints intermediate tool/model steps. If a later model call
    raises, leaving that checkpoint in place makes the next user message resume
    the failed plan instead of starting a new turn (for example, a plain ``hi``
    continued a failed research-plan tool chain). Conversation transcripts stay
    in the database; this only removes the in-memory graph execution state.
    """
    checkpointer = getattr(agent, "checkpointer", None)
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if not callable(delete_thread):
        return False
    delete_thread(thread_id)
    clear_pending_interrupt(thread_id)
    return True


def invoke_agent_once(agent: Any, payload: dict[str, Any] | Command, config: dict[str, Any]) -> dict[str, Any]:
    """Invoke the agent once and return the result.

    `tool_results` surfaces any DISPATCH_BRIDGE_TOOL_NAMES tool call's JSON result from this
    turn, keyed by tool name -- see main.py::extract_tool_results docstring for why
    chat_runtime_service.py needs this to reconstruct the Feishu progress/revision bridge
    status that channel_router_service.py used to set directly before 3.7/3.8 moved job
    creation and revision into the Agent's own tool choices.
    """
    final_text = ""
    interrupts: list[Any] = []
    tool_results: dict[str, dict[str, Any]] = {}
    for chunk in agent.stream(
        payload,
        config=config,
        stream_mode=["updates"],
        version="v2",
    ):
        chunk_type, data = normalize_stream_chunk(chunk)
        if chunk_type != "updates":
            continue
        update_interrupts = extract_interrupts(data)
        if update_interrupts:
            interrupts.extend(update_interrupts)
            continue
        tool_results.update(extract_tool_results(data, DISPATCH_BRIDGE_TOOL_NAMES))
        maybe_text = latest_update_ai_text(data)
        if maybe_text:
            final_text = maybe_text
    if interrupts:
        emit_event(
            "chat.interrupted",
            "server",
            "Chat request was interrupted by HITL.",
            data={"interrupt_count": len(interrupts)},
        )
        return {
            "status": "interrupted",
            "interrupts": [serialize_interrupt(interrupt) for interrupt in interrupts],
        }
    return {
        "status": "completed",
        "assistant_text": final_text.strip(),
        "tool_results": tool_results,
    }
