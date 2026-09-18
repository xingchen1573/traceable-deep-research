"""Shared chat runtime for HTTP and platform adapters."""

from __future__ import annotations

import traceback
from typing import Any

try:
    from ... import memory as memory_store
    from ...observability import emit_event, new_run_id, set_observation_context
    from ...tools import set_current_thread_id
except ImportError:  # pragma: no cover
    import memory as memory_store
    from observability import emit_event, new_run_id, set_observation_context
    from tools import set_current_thread_id

from ..dependencies import (
    clear_agent_thread_state,
    clear_pending_interrupt,
    get_agent,
    get_pending_interrupts,
    invoke_agent_once,
    set_pending_interrupt,
    thread_config,
)
from ..helpers import pending_interrupt_view
from . import channel_router_service, channel_service, conversation_service
from .common import now_text

_APPROVE_KEYWORDS = ("存起来", "可以了", "批准", "确认", "同意", "没问题", "可以", "approve", "ok")
_REJECT_KEYWORDS = ("驳回", "打回", "拒绝", "不同意", "reject", "不行", "不可以")
_INTERRUPT_ACTION_LABELS = {
    "approve_research_job_tool": "批准调研报告落库",
    "revise_report_section_tool": "章节修改",
    "save_knowledge": "保存知识库文档",
    "save_knowledge_set": "保存知识库文档集",
    "write_file": "写文件",
    "delete_knowledge_base_index": "删除知识库索引",
}


def process_channel_chat(
    *,
    message: str,
    channel_id: str | None = None,
    platform: str | None = None,
    external_id: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Route one channel-scoped message through main, QA, or research runtime.

    Checks for a pending HITL interrupt on this channel's main-mode thread first -- see
    docs/modules/agent-dispatcher-role-v0.1.md 3.8 and the follow-up fix that added this
    check. approve_research_job_tool (and any other HumanInTheLoopMiddleware-gated tool) pauses
    the LangGraph run and previously left channel callers (Web channel chat + Feishu, both of
    which go through this function) with no way to actually submit an approve/reject decision
    and resume it -- the next message would just start a brand-new agent turn on a thread that's
    still paused mid-interrupt, which is unsupported. If a pending interrupt exists, this message
    is treated as a candidate decision instead of a fresh chat turn.
    """
    request_id = run_id or new_run_id()
    channel = channel_service.ensure_channel(channel_id=channel_id, platform=platform, external_id=external_id)
    main_thread_id = channel_service.thread_id_for(channel["channel_id"], "main")
    pending = get_pending_interrupts().get(main_thread_id)
    if pending is not None:
        return _resume_pending_interrupt(
            channel=channel,
            thread_id=main_thread_id,
            text=message,
            pending=pending,
            request_id=request_id,
            user_id=user_id,
        )

    def main_chat_handler(thread_id: str, text: str, channel: dict[str, Any]) -> dict[str, Any]:
        agent = get_agent()
        set_current_thread_id(thread_id)
        set_observation_context(session_id=thread_id, run_id=request_id)
        emit_event(
            "agent.turn.started",
            "server",
            "Channel chat turn started.",
            session_id=thread_id,
            data={"channel_id": channel.get("channel_id"), "user_chars": len(text)},
        )
        conversation_service.save_message(thread_id, "user", text)
        try:
            result = invoke_agent_once(
                agent,
                {"messages": [{"role": "user", "content": text}]},
                thread_config(thread_id),
            )
        except Exception as exc:
            assistant_text = _render_model_failure_text(exc)
            checkpoint_cleared = False
            try:
                checkpoint_cleared = clear_agent_thread_state(agent, thread_id)
            except Exception:
                checkpoint_cleared = False
            emit_event(
                "agent.turn.failed",
                "server",
                "Channel chat turn failed with an unhandled exception.",
                session_id=thread_id,
                level="error",
                data={
                    "channel_id": channel.get("channel_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                    "failed_checkpoint_cleared": checkpoint_cleared,
                    "traceback": traceback.format_exc()[-2000:],
                },
            )
            conversation_service.save_message(thread_id, "assistant", assistant_text)
            return {
                "status": "completed",
                "state": "completed",
                "channel_id": channel.get("channel_id"),
                "mode": channel.get("mode") or "main",
                "active_kb": channel.get("active_kb"),
                "thread_id": thread_id,
                "run_id": request_id,
                "assistant_text": assistant_text,
                "channel": channel,
            }
        if result["status"] == "interrupted":
            pending = set_pending_interrupt(thread_id, text, result["interrupts"], request_id)
            payload = pending_interrupt_view(thread_id, pending)
            payload["channel_id"] = channel.get("channel_id")
            payload["mode"] = channel.get("mode") or "main"
            payload["channel"] = channel
            payload["assistant_text"] = _render_pending_interrupt_text(result["interrupts"])
            conversation_service.save_message(thread_id, "assistant", payload["assistant_text"])
            return payload
        assistant_text = result["assistant_text"]
        memory_store.append_daily_log_turn(thread_id, text, assistant_text)
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        emit_event(
            "agent.turn.finished",
            "server",
            "Channel chat turn finished.",
            session_id=thread_id,
            data={"channel_id": channel.get("channel_id"), "assistant_chars": len(assistant_text)},
        )
        clear_pending_interrupt(thread_id)
        response = {
            "status": "completed",
            "state": "completed",
            "channel_id": channel.get("channel_id"),
            "mode": channel.get("mode") or "main",
            "active_kb": channel.get("active_kb"),
            "thread_id": thread_id,
            "run_id": request_id,
            "assistant_text": assistant_text,
            "channel": channel,
        }
        _apply_dispatch_bridge_status(response, result.get("tool_results") or {})
        return response

    return channel_router_service.route_channel_message(
        message=message,
        channel_id=channel_id,
        platform=platform,
        external_id=external_id,
        user_id=user_id,
        main_chat_handler=main_chat_handler,
    )


def _apply_dispatch_bridge_status(response: dict[str, Any], tool_results: dict[str, Any]) -> None:
    """Reconstruct route_channel_message's old "research_started"/"section_revision_started"
    status from this turn's tool-call results, so backend/services/feishu_service.py's existing
    progress/revision bridges keep firing exactly as before, even though job creation and
    section revision now happen inside the Agent's own tool choice instead of a
    channel_router_service.py hard route -- see docs/modules/agent-dispatcher-role-v0.1.md
    3.7/3.8. Mutates `response` in place; leaves it untouched if neither tool was called (the
    common case for ordinary chat turns).
    """
    started = tool_results.get("start_research_job_tool") or {}
    if started.get("job_id"):
        response["status"] = "research_started"
        response["research_job_id"] = started["job_id"]
        return

    single_url = tool_results.get("start_single_url_brief_tool") or {}
    if single_url.get("job_id"):
        # Reuses the exact same "research_started" status feishu_service.py's progress bridge
        # already listens for -- see docs/modules/single-url-brief-pipeline-v0.1.md 3. The
        # bridge itself is generic over job_service.get(job_id), so no bridge code changes.
        response["status"] = "research_started"
        response["research_job_id"] = single_url["job_id"]
        return

    revised = tool_results.get("revise_report_section_tool") or {}
    if revised.get("status") == "revision_started" and revised.get("job_id"):
        response["status"] = "section_revision_started"
        response["research_job_id"] = revised["job_id"]
        response["revision_id"] = revised.get("revision_id")


def _render_pending_interrupt_text(interrupts: list[dict[str, Any]]) -> str:
    """Render a HITL interrupt (see helpers.py::serialize_interrupt) as human-readable text.

    Used both for the initial interrupt (main_chat_handler above) and for
    _resume_pending_interrupt below (re-interrupt after a resume attempt, or the nudge shown
    when a follow-up message can't be classified as a decision) -- one renderer, three call
    sites, so the wording stays consistent.
    """
    action_requests: list[dict[str, Any]] = []
    for item in interrupts or []:
        if isinstance(item, dict):
            action_requests.extend(a for a in item.get("action_requests") or [] if isinstance(a, dict))
    if not action_requests:
        return "有一个操作正在等待人工审批，请回复“批准”或“驳回”处理。"
    lines = ["⏸️ 有操作等待你确认", "━━━━━━━━━━━━"]
    for action in action_requests:
        name = str(action.get("name") or "")
        label = _INTERRUPT_ACTION_LABELS.get(name, name)
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        job_id = args.get("job_id")
        line = f"• {label}"
        if job_id:
            line += f"（job_id={job_id}）"
        lines.append(line)
    lines.extend(["", "回复“批准”/“存起来”完成审批，或回复“驳回”+原因打回。"])
    return "\n".join(lines)


def _render_model_failure_text(exc: Exception) -> str:
    """Render an unhandled model/graph exception as a user-facing Feishu/Web message.

    Added alongside main.py's build_model/build_recall_model/build_dream_model timeout+
    max_retries config (see docs/modules/feishu-hitl-approve-hang-v0.1.md) -- before this, a
    model call that ultimately failed (timeout, connection error, rate limit) after exhausting
    retries would propagate all the way up through process_channel_chat / _resume_pending_interrupt
    with no catch anywhere in the call chain (channel_router_service.py, feishu_service.py,
    feishu_ws_service.py all have no try/except around this), leaving the user with total silence
    and no way to tell a real failure apart from "still thinking". This gives an honest, specific-
    where-possible explanation instead.

    ValueError is handled separately from network/model failures: tools like
    approve_research_job_tool raise ValueError for *deterministic* business-rule rejections (e.g.
    quality gate not passed -- see research_service.py::approve_research), which will fail again
    on every retry with the exact same message. Telling the user "retry later, probably transient"
    for those is actively misleading (caught during 2026-07-19 verification: a real
    "Validation 阶段预算耗尽" rejection got the generic transient-failure wording). ValueError's
    str(exc) is human-written business text (not a raw exception message/stack trace), so it's
    safe and useful to show directly.
    """
    name = type(exc).__name__
    error_text = str(exc).lower()
    if isinstance(exc, ValueError):
        message = str(exc).strip() or "请求被拒绝，但没有给出具体原因。"
        return f"抱歉，这个操作没有成功：{message}\n这不是网络或模型服务的临时问题，重新发送同一个操作大概率还是会失败，需要先处理提示里说的具体原因。"
    lowered = name.lower()
    route_evidence = any(marker in error_text for marker in ("404", "model", "route"))
    not_found_evidence = any(marker in error_text for marker in ("not found", "not_found"))
    if route_evidence and ("notfound" in lowered or not_found_evidence):
        return (
            "抱歉，这一轮处理失败了：当前配置的模型在目标服务中不存在，或模型路由尚未配置。\n"
            "这是模型名称与服务网关不匹配一类的配置问题，重复发送通常不会恢复，请检查模型配置。"
        )
    if "credential pool is saturated" in error_text or "rate limit cooldown" in error_text:
        reason = "模型 Key 均处于限流冷却期"
    elif "timeout" in lowered:
        reason = "模型响应超时"
    elif "connection" in lowered or "connect" in lowered:
        reason = "连接模型服务失败"
    elif "ratelimit" in lowered or "rate_limit" in lowered:
        reason = "模型服务商限流"
    else:
        reason = f"处理过程中出现异常（{name}）"
    return (
        f"抱歉，这一轮处理失败了：{reason}。已自动重试但仍未成功，大概率是模型服务商侧的临时波动，不是消息没收到。\n"
        "请稍后重新发一遍这条消息；如果涉及调研任务，可以问我“XX 任务现在什么状态”来确认有没有受影响。"
    )


def _classify_channel_decision(text: str) -> dict[str, Any] | None:
    """Classify a follow-up channel message as an approve/reject HITL decision.

    Simple substring keyword matching -- same style/precision trade-off as
    channel_router_service.py's ENTER_KB_KEYWORDS/FOLLOWUP_KEYWORDS, not a model classifier.
    Reject is checked first so a negated sentence that still contains an approve keyword (e.g.
    "不批准") lands on reject. "edit" decisions need a structured edited_action that free chat
    text can't express -- unsupported here, use POST /api/v1/chat/resume directly for that.
    Returns None when the message doesn't look like a decision, so the caller can leave the
    interrupt pending instead of misinterpreting an unrelated message.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    if any(keyword in stripped or keyword in lowered for keyword in _REJECT_KEYWORDS):
        return {"type": "reject", "message": stripped}
    if any(keyword in stripped or keyword in lowered for keyword in _APPROVE_KEYWORDS):
        return {"type": "approve"}
    return None


def _resume_pending_interrupt(
    *,
    channel: dict[str, Any],
    thread_id: str,
    text: str,
    pending: dict[str, Any],
    request_id: str,
    user_id: str | None,
) -> dict[str, Any]:
    """Handle a channel message that arrived while thread_id has a pending HITL interrupt.

    See process_channel_chat's docstring for why this branch exists. Mirrors
    backend/routes/chat.py::chat_resume's Command(resume=...) mechanics, adapted to (a) classify
    the decision from free text instead of a structured API field, and (b) return the same
    channel-response shape main_chat_handler uses (channel_id/mode/channel/assistant_text), so
    Feishu/Web channel callers don't need to special-case this path.
    """
    conversation_service.save_message(thread_id, "user", text)
    decision = _classify_channel_decision(text)
    if decision is None:
        assistant_text = _render_pending_interrupt_text(pending.get("interrupts") or [])
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        return {
            "status": "completed",
            "state": "completed",
            "channel_id": channel.get("channel_id"),
            "mode": channel.get("mode") or "main",
            "active_kb": channel.get("active_kb"),
            "thread_id": thread_id,
            "run_id": request_id,
            "assistant_text": assistant_text,
            "channel": channel,
            "user_id": user_id,
            "updated_at": now_text(),
        }

    from langgraph.types import Command

    set_current_thread_id(thread_id)
    set_observation_context(session_id=thread_id, run_id=request_id)
    emit_event(
        "hitl.decision.received",
        "server",
        "Channel HITL decision received.",
        session_id=thread_id,
        data={"channel_id": channel.get("channel_id"), "decision": decision.get("type")},
    )
    agent = get_agent()
    try:
        result = invoke_agent_once(
            agent,
            Command(resume={"decisions": [decision]}),
            thread_config(thread_id),
        )
    except Exception as exc:
        assistant_text = _render_model_failure_text(exc)
        emit_event(
            "agent.turn.failed",
            "server",
            "Channel HITL resume failed with an unhandled exception.",
            session_id=thread_id,
            level="error",
            data={
                "channel_id": channel.get("channel_id"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "traceback": traceback.format_exc()[-2000:],
            },
        )
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        return {
            "status": "completed",
            "state": "completed",
            "channel_id": channel.get("channel_id"),
            "mode": channel.get("mode") or "main",
            "active_kb": channel.get("active_kb"),
            "thread_id": thread_id,
            "run_id": request_id,
            "assistant_text": assistant_text,
            "channel": channel,
            "user_id": user_id,
            "updated_at": now_text(),
        }
    original_user_message = str(pending.get("original_user_message") or "")
    if result["status"] == "interrupted":
        new_pending = set_pending_interrupt(thread_id, original_user_message, result["interrupts"], request_id)
        assistant_text = _render_pending_interrupt_text(result["interrupts"])
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        return {
            "status": "interrupted",
            "state": "interrupted",
            "channel_id": channel.get("channel_id"),
            "mode": channel.get("mode") or "main",
            "active_kb": channel.get("active_kb"),
            "thread_id": thread_id,
            "run_id": new_pending.get("run_id"),
            "interrupts": new_pending.get("interrupts") or [],
            "resume_supported": True,
            "assistant_text": assistant_text,
            "channel": channel,
            "user_id": user_id,
            "updated_at": now_text(),
        }

    assistant_text = result["assistant_text"]
    memory_store.append_daily_log_turn(thread_id, original_user_message, assistant_text)
    conversation_service.save_message(thread_id, "assistant", assistant_text)
    emit_event(
        "agent.turn.finished",
        "server",
        "Channel HITL resume turn finished.",
        session_id=thread_id,
        data={"channel_id": channel.get("channel_id"), "assistant_chars": len(assistant_text)},
    )
    clear_pending_interrupt(thread_id)
    response = {
        "status": "completed",
        "state": "completed",
        "channel_id": channel.get("channel_id"),
        "mode": channel.get("mode") or "main",
        "active_kb": channel.get("active_kb"),
        "thread_id": thread_id,
        "run_id": request_id,
        "assistant_text": assistant_text,
        "channel": channel,
        "user_id": user_id,
        "updated_at": now_text(),
    }
    _apply_dispatch_bridge_status(response, result.get("tool_results") or {})
    return response
