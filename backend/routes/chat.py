"""Chat-related API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from ... import memory as memory_store
    from ...observability import emit_event, new_run_id, set_observation_context
    from ...tools import set_current_thread_id
except ImportError:  # pragma: no cover
    import memory as memory_store
    from observability import emit_event, new_run_id, set_observation_context
    from tools import set_current_thread_id

from ..services import chat_runtime_service, conversation_service

from ..dependencies import (
    clear_pending_interrupt,
    get_agent,
    get_pending_interrupts,
    invoke_agent_once,
    new_thread_id,
    set_pending_interrupt,
    thread_config,
)
from ..helpers import (
    decision_payload,
    error_response,
    now_text,
    pending_interrupt_view,
    response_json,
)
from ..models import ChatRequest, ResumeRequest

router = APIRouter()


def _pending_review_response(status_code: int = 409) -> JSONResponse:
    """Return a response when a dream review is pending."""
    state = memory_store.read_dream_state()
    request_id = new_run_id()
    emit_event(
        "chat.blocked.pending_review",
        "server",
        "Chat request was blocked because a dream report is pending review.",
        session_id=state.get("logs_until") or "server",
        run_id=request_id,
        level="warning",
        data={"report_path": state.get("report_path")},
    )
    return response_json(
        {
            "status": "blocked",
            "reason": "pending_review",
            "report_path": state.get("report_path"),
            "dream_state": state,
        },
        request_id=request_id,
        status_code=status_code,
    )


@router.get("/chat/pending/{thread_id}")
def chat_pending(thread_id: str) -> JSONResponse:
    """Check if a thread has a pending HITL interrupt."""
    request_id = new_run_id()
    pending = get_pending_interrupts().get(thread_id)
    if pending is None:
        return response_json(
            {
                "status": "not_found",
                "thread_id": thread_id,
                "resume_supported": False,
            },
            request_id=request_id,
            status_code=404,
        )
    return response_json(
        pending_interrupt_view(thread_id, pending),
        request_id=request_id,
    )


@router.post("/chat")
def chat(request: ChatRequest) -> Any:
    """Process a chat message."""
    if memory_store.pending_dream_review_exists():
        return _pending_review_response()
    if request.channel_id or request.external_id:
        run_id = new_run_id()
        try:
            routed = chat_runtime_service.process_channel_chat(
                message=request.message,
                channel_id=request.channel_id,
                platform=request.platform,
                external_id=request.external_id,
                user_id=request.user_id,
                run_id=run_id,
            )
        except Exception as exc:
            return error_response(
                exc,
                default_message="The channel chat turn failed.",
                session_id=request.channel_id or request.external_id or "channel",
                event_type="agent.turn.failed",
                request_id=run_id,
            )
        return response_json(routed, request_id=run_id)
    agent = get_agent()
    thread_id = request.thread_id or new_thread_id()
    run_id = new_run_id()
    set_current_thread_id(thread_id)
    set_observation_context(session_id=thread_id, run_id=run_id)
    emit_event(
        "agent.turn.started",
        "server",
        "HTTP chat turn started.",
        session_id=thread_id,
        data={"user_chars": len(request.message)},
    )
    # Persist user message to conversation history
    conversation_service.save_message(thread_id, "user", request.message)
    try:
        result = invoke_agent_once(
            agent,
            {"messages": [{"role": "user", "content": request.message}]},
            thread_config(thread_id),
        )
    except Exception as exc:
        return error_response(
            exc,
            default_message="The chat turn failed.",
            session_id=thread_id,
            event_type="agent.turn.failed",
            request_id=run_id,
        )
    if result["status"] == "interrupted":
        pending = set_pending_interrupt(thread_id, request.message, result["interrupts"], run_id)
        return response_json(pending_interrupt_view(thread_id, pending), request_id=run_id)
    assistant_text = result["assistant_text"]
    memory_store.append_daily_log_turn(thread_id, request.message, assistant_text)
    # Persist assistant message to conversation history
    conversation_service.save_message(thread_id, "assistant", assistant_text)
    emit_event(
        "agent.turn.finished",
        "server",
        "HTTP chat turn finished.",
        session_id=thread_id,
        data={"assistant_chars": len(assistant_text)},
    )
    clear_pending_interrupt(thread_id)
    return response_json(
        {
            "status": "completed",
            "thread_id": thread_id,
            "run_id": run_id,
            "assistant_text": assistant_text,
        },
        request_id=run_id,
    )


@router.post("/chat/resume")
def chat_resume(request: ResumeRequest) -> dict[str, Any]:
    """Resume a chat after HITL interrupt."""
    agent = get_agent()
    pending = get_pending_interrupts().get(request.thread_id)
    if pending is None:
        raise HTTPException(status_code=400, detail="No pending interrupted chat for this thread_id")
    original_user_message = pending.get("original_user_message", "")
    run_id = new_run_id()
    set_current_thread_id(request.thread_id)
    set_observation_context(session_id=request.thread_id, run_id=run_id)
    decision = decision_payload(request)
    decision_type = str(decision.get("type", "unknown"))
    decision_event = {
        "approve": "hitl.decision.approved",
        "edit": "hitl.decision.edited",
        "reject": "hitl.decision.rejected",
    }.get(decision_type, "hitl.decision.unknown")
    emit_event(
        "chat.resumed",
        "server",
        "HTTP chat resume started.",
        session_id=request.thread_id,
        data={"decision": request.decision},
    )
    emit_event(
        decision_event,
        "server",
        "HTTP HITL decision received.",
        session_id=request.thread_id,
        data={"decision": decision_type},
    )
    try:
        from langgraph.types import Command

        result = invoke_agent_once(
            agent,
            Command(resume={"decisions": [decision]}),
            thread_config(request.thread_id),
        )
    except Exception as exc:
        return error_response(
            exc,
            default_message="The resumed chat turn failed.",
            session_id=request.thread_id,
            event_type="agent.turn.failed",
            request_id=run_id,
        )
    if result["status"] == "interrupted":
        pending = set_pending_interrupt(request.thread_id, original_user_message, result["interrupts"], run_id)
        return response_json(pending_interrupt_view(request.thread_id, pending), request_id=run_id)
    assistant_text = result["assistant_text"]
    memory_store.append_daily_log_turn(request.thread_id, original_user_message, assistant_text)
    # Persist assistant message to conversation history
    conversation_service.save_message(request.thread_id, "assistant", assistant_text)
    emit_event(
        "agent.turn.finished",
        "server",
        "HTTP resumed chat turn finished.",
        session_id=request.thread_id,
        data={"assistant_chars": len(assistant_text)},
    )
    clear_pending_interrupt(request.thread_id)
    return response_json(
        {
            "status": "completed",
            "thread_id": request.thread_id,
            "run_id": run_id,
            "assistant_text": assistant_text,
        },
        request_id=run_id,
    )
