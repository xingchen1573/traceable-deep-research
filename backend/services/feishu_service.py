"""Minimal Feishu adapter for channel runtime."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from ...observability import emit_event, new_run_id
except ImportError:  # pragma: no cover
    from observability import emit_event, new_run_id

from . import channel_service, chat_runtime_service, feishu_dedupe_service
from .common import now_text


_REPLIES: list[dict[str, Any]] = []
_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}
_CHAT_TITLE_CACHE: dict[str, dict[str, Any]] = {}
_PROGRESS_BRIDGES: dict[str, dict[str, Any]] = {}
_REVISION_BRIDGES: dict[str, dict[str, Any]] = {}
FEISHU_OPEN_API_BASE = "https://open.feishu.cn/open-apis"
ROOT = Path(__file__).resolve().parent.parent.parent
CHAT_TITLE_CACHE_TTL_SECONDS = 300.0
CHAT_TITLE_FAILURE_CACHE_TTL_SECONDS = 60.0
FEISHU_PROGRESS_TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted", "waiting_approval"}
REVISION_ACTION_TERMINAL_STATES = {"resolved", "failed"}
REVISION_REJECT_REASON_LABELS = {
    "fewer_evidence_used": "修改后引用证据变少了",
    "content_shrank": "修改后正文变短了",
    "revision_fell_back_to_template": "模型改写失败，回退到了模板占位内容",
    "empty_revision": "模型返回了空内容",
    "regenerated_overreach": "修改后重新引入了已被判不支持的论断",
}


def handle_event(payload: dict[str, Any], *, validate_token: bool = True) -> dict[str, Any]:
    """Handle one Feishu webhook payload.

    The first version supports local verification and text message events. Real
    Feishu API sending is intentionally deferred; replies are recorded in memory
    and returned for local verification.
    """
    _load_env()
    if payload.get("type") == "url_verification" or payload.get("challenge"):
        if validate_token:
            _validate_verification_token(payload)
        return {"challenge": payload.get("challenge", "")}

    if validate_token:
        _validate_verification_token(payload)

    event = _extract_event(payload)
    if not event:
        return {"status": "ignored", "reason": "unsupported_event"}

    message = event.get("message") or {}
    message_id = _root_id(message)
    if _is_bot_sender(event):
        return {"status": "ignored", "reason": "bot_sender", "message_id": message_id}
    if _is_duplicate_message(message_id):
        return {"status": "ignored", "reason": "duplicate_message", "message_id": message_id}

    if not _is_text_message(message):
        return {"status": "ignored", "reason": "non_text_message"}

    chat_id = _chat_id(message)
    if not chat_id:
        return {"status": "ignored", "reason": "missing_chat_id"}

    text = _message_text(message)
    if not text:
        return {"status": "ignored", "reason": "empty_text"}

    sender = event.get("sender") or {}
    user_id = _sender_id(sender)
    channel_id = channel_service.build_channel_id("feishu", chat_id)
    channel = channel_service.ensure_channel(
        channel_id=channel_id,
        platform="feishu",
        external_id=chat_id,
        title=_chat_title(message, chat_id),
    )
    run_id = new_run_id()
    emit_event(
        "feishu.event.received",
        "feishu",
        "Received Feishu text message event.",
        session_id=channel_id,
        run_id=run_id,
        data={"chat_id": chat_id, "user_id": user_id, "message_chars": len(text)},
    )
    routed = chat_runtime_service.process_channel_chat(
        message=text,
        channel_id=channel["channel_id"],
        user_id=user_id,
        run_id=run_id,
    )
    reply = send_reply(chat_id=chat_id, text=str(routed.get("assistant_text") or ""), root_id=_root_id(message))
    if str(routed.get("status") or "") == "research_started":
        _start_research_progress_bridge(
            chat_id=chat_id,
            job_id=str(routed.get("research_job_id") or ""),
            message_id=str(reply.get("message_id") or ""),
        )
    elif str(routed.get("status") or "") == "section_revision_started":
        _start_section_revision_bridge(
            chat_id=chat_id,
            job_id=str(routed.get("research_job_id") or ""),
            revision_id=str(routed.get("revision_id") or ""),
        )
    return {
        "status": "handled",
        "channel_id": channel["channel_id"],
        "chat_id": chat_id,
        "user_id": user_id,
        "route": routed,
        "reply": reply,
    }


def send_reply(*, chat_id: str, text: str, root_id: str | None = None) -> dict[str, Any]:
    """Send or record a reply depending on FEISHU_ENABLE_REAL_SEND."""
    _load_env()
    msg_type, content = _reply_message_payload(text)
    if os.getenv("FEISHU_ENABLE_REAL_SEND", "").lower() in {"1", "true", "yes"}:
        return _send_feishu_message(chat_id=chat_id, text=text, msg_type=msg_type, content=content)
    reply = {
        "status": "recorded",
        "chat_id": chat_id,
        "root_id": root_id,
        "msg_type": msg_type,
        "text": _trim_reply(text),
        "content": content,
        "created_at": now_text(),
    }
    _REPLIES.append(reply)
    emit_event(
        "feishu.reply.recorded",
        "feishu",
        "Recorded Feishu reply locally.",
        session_id=f"feishu:{chat_id}",
        data={"text_chars": len(reply["text"])},
    )
    return reply


def config_status() -> dict[str, Any]:
    _load_env()
    return {
        "app_id_configured": bool(os.getenv("FEISHU_APP_ID")),
        "app_secret_configured": bool(os.getenv("FEISHU_APP_SECRET")),
        "verification_token_configured": bool(os.getenv("FEISHU_VERIFICATION_TOKEN")),
        "encrypt_key_configured": bool(os.getenv("FEISHU_ENCRYPT_KEY")),
        "real_send_enabled": os.getenv("FEISHU_ENABLE_REAL_SEND", "").lower() in {"1", "true", "yes"},
        "reply_format": os.getenv("FEISHU_REPLY_FORMAT", "post").strip().lower() or "post",
        "api_base": os.getenv("FEISHU_OPEN_API_BASE") or FEISHU_OPEN_API_BASE,
    }


def credentials() -> dict[str, str]:
    """Return Feishu credentials for local adapters without exposing them via API."""
    _load_env()
    return {
        "app_id": os.getenv("FEISHU_APP_ID", "").strip(),
        "app_secret": os.getenv("FEISHU_APP_SECRET", "").strip(),
        "verification_token": os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip(),
        "encrypt_key": os.getenv("FEISHU_ENCRYPT_KEY", "").strip(),
    }


def list_replies(limit: int = 50) -> list[dict[str, Any]]:
    return list(_REPLIES[-max(1, min(limit, 200)) :])


def clear_replies() -> None:
    _REPLIES.clear()


def clear_seen_messages() -> None:
    feishu_dedupe_service.clear_seen_messages()


def clear_chat_title_cache() -> None:
    _CHAT_TITLE_CACHE.clear()


def sync_channel_titles() -> dict[str, Any]:
    """Refresh persisted Feishu channel titles from Feishu's group information API."""
    channels = [item for item in channel_service.list_channels(limit=200) if item.get("platform") == "feishu"]
    updated = 0
    resolved = 0
    for channel in channels:
        chat_id = str(channel.get("external_id") or "").strip()
        if not chat_id:
            continue
        title = _chat_title({}, chat_id, allow_api=True, force_refresh=True)
        if not title:
            continue
        resolved += 1
        if title != str(channel.get("title") or ""):
            channel_service.update_channel(str(channel["channel_id"]), title=title)
            updated += 1
    return {
        "attempted": len(channels),
        "resolved": resolved,
        "updated": updated,
        "failed": len(channels) - resolved,
    }


def clear_progress_bridges() -> None:
    _PROGRESS_BRIDGES.clear()


def clear_revision_bridges() -> None:
    _REVISION_BRIDGES.clear()


def _validate_verification_token(payload: dict[str, Any]) -> None:
    _load_env()
    expected = os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip()
    if not expected:
        return
    actual = str(payload.get("token") or "").strip()
    if actual != expected:
        raise ValueError("invalid Feishu verification token")


def _send_feishu_message(*, chat_id: str, text: str, msg_type: str, content: str) -> dict[str, Any]:
    token = _tenant_access_token()
    body = {
        "receive_id": chat_id,
        "msg_type": msg_type,
        "content": content,
    }
    response = _post_json(
        f"{_api_base()}/im/v1/messages?receive_id_type=chat_id",
        body,
        headers={"Authorization": f"Bearer {token}"},
    )
    if int(response.get("code") or 0) != 0:
        raise RuntimeError(f"Feishu send message failed: {response.get('msg') or response}")
    reply = {
        "status": "sent",
        "chat_id": chat_id,
        "message_id": ((response.get("data") or {}).get("message_id")),
        "msg_type": msg_type,
        "text": _trim_reply(text),
        "created_at": now_text(),
    }
    _REPLIES.append(reply)
    emit_event(
        "feishu.reply.sent",
        "feishu",
        "Sent Feishu reply.",
        session_id=f"feishu:{chat_id}",
        data={"message_id": reply.get("message_id"), "msg_type": msg_type},
    )
    return reply


def _update_feishu_message(*, message_id: str, text: str) -> dict[str, Any]:
    token = _tenant_access_token()
    msg_type, content = _reply_message_payload(text)
    body = {
        "msg_type": msg_type,
        "content": content,
    }
    response = _request_json(
        "PUT",
        f"{_api_base()}/im/v1/messages/{quote(message_id, safe='')}",
        body,
        headers={"Authorization": f"Bearer {token}"},
    )
    if int(response.get("code") or 0) != 0:
        raise RuntimeError(f"Feishu update message failed: {response.get('msg') or response}")
    emit_event(
        "feishu.reply.updated",
        "feishu",
        "Updated Feishu reply message.",
        session_id=f"feishu_message:{message_id}",
        data={"message_id": message_id, "msg_type": msg_type},
    )
    return {
        "status": "updated",
        "message_id": message_id,
        "msg_type": msg_type,
        "text": _trim_reply(text),
        "updated_at": now_text(),
        "response": response,
    }


def _start_research_progress_bridge(*, chat_id: str, job_id: str, message_id: str) -> None:
    if not job_id or not message_id:
        return
    if os.getenv("FEISHU_RESEARCH_PROGRESS_PUSH", "true").strip().lower() in {"0", "false", "no"}:
        return
    if job_id in _PROGRESS_BRIDGES:
        return
    state = {
        "job_id": job_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "started_at": time.time(),
        "last_text": "",
        "last_signature": None,
        "last_update_at": 0.0,
        "last_error": None,
    }
    _PROGRESS_BRIDGES[job_id] = state
    thread = threading.Thread(target=_run_research_progress_bridge, args=(job_id,), daemon=True)
    thread.start()


def _run_research_progress_bridge(job_id: str) -> None:
    from .job_service import job_service

    state = _PROGRESS_BRIDGES.get(job_id)
    if not state:
        return
    interval = max(1.0, float(os.getenv("FEISHU_PROGRESS_POLL_SECONDS", "2.5") or 2.5))
    min_update_seconds = max(1.0, float(os.getenv("FEISHU_PROGRESS_MIN_UPDATE_SECONDS", "12") or 12))
    heartbeat_seconds = max(15.0, float(os.getenv("FEISHU_PROGRESS_HEARTBEAT_SECONDS", "30") or 30))
    max_seconds = max(30.0, float(os.getenv("FEISHU_PROGRESS_MAX_SECONDS", "7200") or 7200))
    while True:
        try:
            job = job_service.get(job_id)
            text = _render_research_progress_message(job)
            terminal = str(job.get("status") or "") in FEISHU_PROGRESS_TERMINAL_STATES
            signature = _research_progress_signature(job)
            now = time.time()
            should_update = _should_push_research_progress(
                state=state,
                signature=signature,
                terminal=terminal,
                now=now,
                min_update_seconds=min_update_seconds,
                heartbeat_seconds=heartbeat_seconds,
            )
            if should_update:
                _push_research_progress_update(state, text)
                state["last_text"] = text
                state["last_signature"] = signature
                state["last_update_at"] = now
            if terminal:
                state["finished_at"] = time.time()
                return
        except Exception as exc:
            state["last_error"] = str(exc) or type(exc).__name__
            emit_event(
                "feishu.progress_bridge.error",
                "feishu",
                "Failed to update Feishu research progress.",
                session_id=f"research:{job_id}",
                data={"job_id": job_id, "error": state["last_error"]},
            )
        if time.time() - float(state.get("started_at") or time.time()) > max_seconds:
            state["timeout_at"] = time.time()
            return
        time.sleep(interval)


def _start_section_revision_bridge(*, chat_id: str, job_id: str, revision_id: str) -> None:
    """Notify the Feishu chat once a manual section revision (triggered by _handle_section_revision_request
    in channel_router_service.py) finishes, instead of leaving the user with only the initial
    "已发起修改" ack and a link they have to click through to. Unlike _start_research_progress_bridge
    (which continuously edits one message across many stages of a full report generation), a section
    revision is a single short operation — one writer call plus at most one retry — so this only
    needs a one-shot "started -> done" notification, sent as a brand-new message rather than an edit.
    """
    if not job_id or not revision_id:
        return
    if revision_id in _REVISION_BRIDGES:
        return
    _REVISION_BRIDGES[revision_id] = {
        "job_id": job_id,
        "revision_id": revision_id,
        "chat_id": chat_id,
        "started_at": time.time(),
    }
    thread = threading.Thread(target=_run_section_revision_bridge, args=(revision_id,), daemon=True)
    thread.start()


def _run_section_revision_bridge(revision_id: str) -> None:
    from .job_service import job_service

    state = _REVISION_BRIDGES.get(revision_id)
    if not state:
        return
    job_id = str(state["job_id"])
    interval = max(1.0, float(os.getenv("FEISHU_REVISION_POLL_SECONDS", "3") or 3))
    max_seconds = max(30.0, float(os.getenv("FEISHU_REVISION_MAX_SECONDS", "600") or 600))
    while True:
        try:
            job = job_service.get(job_id)
            result = job.get("result") if isinstance(job.get("result"), dict) else {}
            action = next(
                (
                    item
                    for item in result.get("revision_actions") or []
                    if isinstance(item, dict) and str(item.get("id") or "") == revision_id
                ),
                None,
            )
            if action and str(action.get("status") or "") in REVISION_ACTION_TERMINAL_STATES:
                text = _render_section_revision_result_message(job, action)
                send_reply(chat_id=str(state["chat_id"]), text=text)
                _REVISION_BRIDGES.pop(revision_id, None)
                return
        except Exception as exc:
            emit_event(
                "feishu.revision_bridge.error",
                "feishu",
                "Failed to check section revision result.",
                session_id=f"research:{job_id}",
                data={"revision_id": revision_id, "error": str(exc) or type(exc).__name__},
            )
        if time.time() - float(state.get("started_at") or time.time()) > max_seconds:
            _REVISION_BRIDGES.pop(revision_id, None)
            return
        time.sleep(interval)


def _render_section_revision_result_message(job: dict[str, Any], action: dict[str, Any]) -> str:
    job_id = str(job.get("job_id") or "")
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    section_ids = [str(item) for item in action.get("section_ids") or []]
    per_section_outcome = action.get("per_section_outcome") if isinstance(action.get("per_section_outcome"), dict) else {}
    documents_by_section = {
        str(doc.get("section_id") or ""): doc for doc in result.get("documents") or [] if isinstance(doc, dict)
    }
    editorial = result.get("editorial") if isinstance(result.get("editorial"), dict) else {}
    lines = ["✏️ 章节修改结果", "━━━━━━━━━━━━", f"🧾 Job：{job_id}"]
    if str(action.get("status") or "") == "failed":
        lines.append(f"❌ 修改失败：{action.get('error') or '未知错误'}")
    else:
        for section_id in section_ids:
            if section_id == "editorial":
                # editorial (executive summary + conclusion) is not a `documents` entry -- it
                # lives in result["editorial"] -- so it needs its own title/content lookup
                # instead of documents_by_section's.
                title = "执行摘要与结论"
                content = str(editorial.get("executive_summary") or "") + " " + str(editorial.get("conclusion") or "")
                diagnostics = per_section_outcome.get(section_id) or {}
            else:
                doc = documents_by_section.get(section_id) or {}
                title = str(doc.get("title") or section_id)
                content = str(doc.get("content") or "")
                diagnostics = per_section_outcome.get(section_id) or doc.get("revision_diagnostics") or {}
            if diagnostics.get("accepted"):
                lines.append(f"✅ {title}：已按要求修改")
                excerpt = _document_excerpt(content, limit=150)
                if excerpt:
                    lines.append(f"   {excerpt}")
            else:
                reason = _revision_reject_reason_label(str(diagnostics.get("rejected_reason") or ""))
                lines.append(f"⚠️ {title}：修改被拒绝（{reason}）")
        # Sections dropped for lacking evidence never reach revise_report_sections at all, so
        # they have no per_section_outcome / documents_by_section entry and would otherwise be
        # completely invisible in this async push -- see docs/modules/deep-research-step4-
        # approval-and-delivery-discussion-v0.1.md 商榷点二. Keep this to a single summary line
        # per section (no evidence_gaps detail) to match this message's existing terseness; the
        # full gap list stays available via check_research_job_tool.
        skipped = action.get("skipped_evidence_insufficient") if isinstance(action.get("skipped_evidence_insufficient"), dict) else {}
        skipped_section_ids = [str(item) for item in skipped.get("section_ids") or []]
        blueprint_titles = {
            str(section.get("section_id") or ""): str(section.get("title") or "")
            for section in (result.get("report_blueprint") or {}).get("sections") or []
            if isinstance(section, dict)
        }
        for section_id in skipped_section_ids:
            title = blueprint_titles.get(section_id) or str((documents_by_section.get(section_id) or {}).get("title") or section_id)
            lines.append(f"⛔ {title}：证据不足，本次未处理")
    base = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5173").rstrip("/")
    delivery = result.get("delivery") if isinstance(result.get("delivery"), dict) else {}
    feishu_doc_url = str(delivery.get("feishu_doc_url") or "").strip()
    if feishu_doc_url:
        # Only present when _run_section_revision actually recreated the doc (i.e. at least one
        # targeted section was accepted) — the doc created at initial delivery is create-only and
        # is never edited in place, so a fresh URL here means a brand-new document reflecting the
        # post-revision content, not the same link as before.
        lines.append(f"📄 飞书文档（已更新）：{feishu_doc_url}")
    lines.extend(["", f"🔗 Research 详情：{base}/research?job_id={job_id}"])
    return "\n".join(lines)


def _revision_reject_reason_label(reason: str) -> str:
    return REVISION_REJECT_REASON_LABELS.get(reason, reason or "未达到质量护栏要求")


def _should_push_research_progress(
    *,
    state: dict[str, Any],
    signature: tuple[Any, ...],
    terminal: bool,
    now: float,
    min_update_seconds: float,
    heartbeat_seconds: float,
) -> bool:
    last_signature = state.get("last_signature")
    last_update_at = float(state.get("last_update_at") or 0)
    since_last_update = now - last_update_at
    signature_changed = signature != last_signature
    if signature_changed and (last_signature is None or terminal or since_last_update >= min_update_seconds):
        return True
    return bool(last_update_at and not terminal and since_last_update >= heartbeat_seconds)


def _push_research_progress_update(state: dict[str, Any], text: str) -> None:
    try:
        _update_feishu_message(message_id=str(state["message_id"]), text=text)
        return
    except Exception as exc:
        if not _is_feishu_message_edit_limit_error(exc):
            raise
        _send_research_progress_rollover(state, text, reason="edit_limit")


def _send_research_progress_rollover(state: dict[str, Any], text: str, *, reason: str) -> None:
    rollover_text = "调研进度接力\n\n" + text
    msg_type, content = _reply_message_payload(rollover_text)
    reply = _send_feishu_message(
        chat_id=str(state["chat_id"]),
        text=rollover_text,
        msg_type=msg_type,
        content=content,
    )
    state["message_id"] = str(reply.get("message_id") or state.get("message_id") or "")
    state["rollover_at"] = time.time()
    emit_event(
        "feishu.progress_bridge.rollover",
        "feishu",
        "Started a new Feishu progress message.",
        session_id=f"research:{state.get('job_id')}",
        data={
            "job_id": state.get("job_id"),
            "message_id": state.get("message_id"),
            "reason": reason,
        },
    )


def _is_feishu_message_edit_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "230072" in text or "number of times it can be edited" in text


def _research_progress_signature(job: dict[str, Any]) -> tuple[Any, ...]:
    progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
    steps = job.get("steps") if isinstance(job.get("steps"), list) else []
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    step_signature = tuple(
        (
            str(step.get("id") or ""),
            str(step.get("status") or ""),
            str(step.get("summary") or ""),
        )
        for step in steps
        if isinstance(step, dict)
    )
    return (
        str(job.get("status") or ""),
        int(progress.get("current") or 0),
        int(progress.get("total") or 0),
        str(progress.get("message") or ""),
        step_signature,
        str(validation.get("status") or ""),
        str(validation.get("reason") or ""),
        str(quality_gate.get("status") or ""),
        bool(quality_gate.get("approval_ready")),
    )


def _render_research_progress_message(job: dict[str, Any]) -> str:
    job_id = str(job.get("job_id") or "")
    status = str(job.get("status") or "unknown")
    input_data = job.get("input") if isinstance(job.get("input"), dict) else {}
    request = str(input_data.get("user_request") or "调研任务").strip()
    progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
    current = int(progress.get("current") or 0)
    total = max(1, int(progress.get("total") or 1))
    message = str(progress.get("message") or status)
    display_message = _job_status_label(message) if message in FEISHU_PROGRESS_TERMINAL_STATES | {"queued", "running"} else message
    steps = job.get("steps") if isinstance(job.get("steps"), list) else []
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    elapsed = _job_elapsed_label(job)

    lines = [
        "🔎 调研任务进度",
        "━━━━━━━━━━━━",
        f"📌 任务：{_short_text(request, 200)}",
        f"🧾 Job：{job_id}",
        f"📍 状态：{_job_status_label(status)} · {current}/{total}",
        f"⏱ 耗时：{elapsed}",
        f"🧭 当前：{display_message}",
        "",
        "🪜 阶段进度",
    ]
    for step in steps[:12]:
        if not isinstance(step, dict):
            continue
        lines.extend(_research_step_lines(step))

    if status == "waiting_approval":
        lines.extend(_research_result_summary_lines(result, job_id=job_id))
        lines.extend(["", "✅ 已完成质量检查，等待审批/落库。"])
    elif status == "succeeded":
        artifacts = job.get("artifacts") if isinstance(job.get("artifacts"), list) else []
        lines.extend(_research_result_summary_lines(result, job_id=job_id))
        lines.extend(["", "✅ 调研已完成。"])
        if artifacts:
            lines.append("产物：" + "、".join(str(item) for item in artifacts[:3]))
    elif status == "failed":
        error = job.get("error") if isinstance(job.get("error"), dict) else {}
        lines.extend(["", "❌ 调研失败：" + str(error.get("message") or "未知错误")])
    elif status == "cancelled":
        lines.extend(["", "✖️ 调研已取消。"])
    elif status == "interrupted":
        lines.extend(["", "⚠️ 调研执行进程已中断，可在任务详情中查看并重新发起。"])
    else:
        lines.extend(["", f"🟢 最近更新：{now_text()}（任务仍在执行）", *_research_links(job_id)])

    return "\n".join(lines).strip()


def _job_status_label(status: str) -> str:
    return {
        "queued": "排队中",
        "running": "进行中",
        "waiting_approval": "等待审批",
        "succeeded": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
        "interrupted": "已中断",
    }.get(status, status)


def _step_status_icon(status: str) -> str:
    return {
        "pending": "▫️",
        "queued": "▫️",
        "running": "▶️",
        "succeeded": "✅",
        "waiting_approval": "🟡",
        "failed": "⚠️",
        "cancelled": "✖️",
        "interrupted": "⚠️",
    }.get(status, "▫️")


def _research_step_lines(step: dict[str, Any]) -> list[str]:
    title = str(step.get("title") or step.get("id") or "").strip()
    step_status = str(step.get("status") or "pending")
    summary = str(step.get("summary") or "").strip()
    elapsed_label = _step_elapsed_label(step)
    line = f"{_step_status_icon(step_status)} {title}"
    if elapsed_label:
        line += f"（{elapsed_label}）"
    summary_lines = _step_summary_lines(title, summary)
    if not summary_lines:
        return [line]
    if len(summary_lines) == 1 and len(summary_lines[0]) <= 58:
        return [f"{line}：{summary_lines[0]}"]
    return [line, *[f"   {item}" for item in summary_lines]]


def _step_summary_lines(title: str, summary: str) -> list[str]:
    if not summary:
        return []
    if "目标文档" in summary:
        return _plan_summary_lines(summary)
    if "；" in summary or ";" in summary:
        return [
            _short_text(part.strip(), 180)
            for part in re.split(r"[；;]\s*", summary)
            if part.strip()
        ][:6]
    return [_short_text(summary, 180)]


def _plan_summary_lines(summary: str) -> list[str]:
    before, sep, after = summary.partition("目标文档:")
    if not sep:
        before, sep, after = summary.partition("目标文档：")
    lines: list[str] = []
    prefix = before.strip(" ；;")
    if prefix:
        lines.append(prefix)
    if not after.strip():
        return lines
    lines.append("目标文档：")
    docs = [item.strip() for item in re.split(r"、(?=\d{2}[_＿])", after.strip()) if item.strip()]
    for index, doc in enumerate(docs[:8], start=1):
        lines.append(f"{index}. {_short_text(doc, 180)}")
    if len(docs) > 8:
        lines.append(f"... 其余 {len(docs) - 8} 个章节见审核包")
    return lines


def _research_result_summary_lines(result: dict[str, Any], *, job_id: str) -> list[str]:
    if not result:
        return []
    lines: list[str] = []
    documents = [item for item in result.get("documents") or [] if isinstance(item, dict)]
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    if documents:
        lines.extend(["", "📚 文档清单"])
        for doc in documents[:6]:
            metrics = doc.get("evidence_metrics") if isinstance(doc.get("evidence_metrics"), dict) else {}
            evidence_count = metrics.get("evidence_count")
            source_count = metrics.get("source_count")
            suffix = f"（证据{evidence_count or 0}/来源{source_count or 0}）" if metrics else ""
            lines.append(f"• {doc.get('title')}{suffix}")
        lines.extend(_research_document_preview_lines(documents))
    lines.extend(_research_delivery_lines(result))
    if quality_gate:
        gate_status = str(quality_gate.get("status") or "")
        approval_ready = "可审批" if quality_gate.get("approval_ready") else "暂不建议审批"
        lines.extend(["", f"🧪 质量门：{gate_status or '-'}，{approval_ready}"])
    if validation and validation.get("status") != "passed":
        lines.extend(_validation_issue_lines(validation))
    lines.extend(_research_advice_lines(result))
    lines.extend(["", *_research_links(job_id)])
    return lines


def _research_document_preview_lines(documents: list[dict[str, Any]]) -> list[str]:
    lines = ["", "📝 内容预览"]
    for doc in documents[:3]:
        title = str(doc.get("title") or "文档").strip()
        excerpt = _document_excerpt(str(doc.get("content") or ""), limit=150)
        if excerpt:
            lines.append(f"• {title}")
            lines.append(f"  {excerpt}")
    if len(documents) > 3:
        lines.append(f"• 其余 {len(documents) - 3} 篇请查看审核包或 Research 详情。")
    return lines if len(lines) > 1 else []


def _document_excerpt(content: str, *, limit: int) -> str:
    lines: list[str] = []
    current_section = ""
    for raw in str(content or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = line.strip("# ").strip()
            continue
        if current_section in {"来源", "证据索引", "边界"}:
            continue
        if line.startswith("- ") or line.startswith("• ") or line.startswith("|"):
            continue
        if line.startswith("来源：") or line.startswith("证据"):
            continue
        lines.append(line)
        if len(" ".join(lines)) >= limit:
            break
    return _short_text(" ".join(lines), limit)


def _research_delivery_lines(result: dict[str, Any]) -> list[str]:
    delivery = result.get("delivery") if isinstance(result.get("delivery"), dict) else {}
    paths = [str(item) for item in delivery.get("paths") or result.get("artifacts") or [] if str(item)]
    feishu_doc_url = str(delivery.get("feishu_doc_url") or "").strip()
    lines: list[str] = []
    if paths or feishu_doc_url:
        lines.extend(["", "📎 审核包"])
        if feishu_doc_url:
            lines.append(f"• 飞书文档：{feishu_doc_url}")
        for path in paths[:2]:
            lines.append(f"• 本地 Markdown：{path}")
    return lines


def _research_advice_lines(result: dict[str, Any]) -> list[str]:
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    lines = ["", "🧭 下一步建议"]
    if quality_gate.get("approval_ready") and validation.get("status") == "passed":
        lines.append("• 可以审批落库；建议先抽查每章 1-2 条证据原文。")
        lines.append("• 如果用户觉得章节不够，再按章节发起补查。")
    else:
        lines.append("• 暂不建议落库；优先对质量问题章节发起补查。")
        reason = str(quality_gate.get("reason") or validation.get("reason") or "").strip()
        if reason:
            lines.append("• 主要原因：" + _short_text(reason, 120))
    return lines


def _validation_issue_lines(validation: dict[str, Any]) -> list[str]:
    reason = str(validation.get("reason") or "").strip()
    lines = ["", "⚠️ 质量问题"]
    if reason:
        lines.append("• " + _short_text(reason, 140))
    issue_count = 0
    for item in validation.get("per_document") or []:
        if not isinstance(item, dict):
            continue
        blocking = [row for row in item.get("blocking_unsupported_claims") or [] if isinstance(row, dict)]
        missing_refs = [row for row in item.get("missing_inline_refs") or [] if isinstance(row, dict)]
        unsupported_refs = [row for row in item.get("unsupported_refs") or [] if isinstance(row, dict)]
        if not blocking and not missing_refs and not unsupported_refs:
            continue
        title = str(item.get("title") or "document")
        ref_count = item.get("evidence_ref_count") or 0
        matched = item.get("matched_evidence_ref_count") or 0
        lines.append(f"• {title}：引用匹配 {matched}/{ref_count}，强事实未引用 {len(blocking)}，正文缺引用 {len(missing_refs)}")
        for claim in blocking[:2]:
            text = str(claim.get("text") or claim.get("claim") or "").strip()
            section = str(claim.get("section") or "").strip()
            line = f"  - {_short_text(text, 110)}"
            if section:
                line += f"（{section}）"
            lines.append(line)
            candidates = claim.get("candidate_evidence") if isinstance(claim.get("candidate_evidence"), list) else []
            if candidates:
                candidate = candidates[0]
                lines.append(
                    "    候选证据："
                    + _short_text(str(candidate.get("id") or "") + " " + str(candidate.get("claim") or ""), 100)
                )
        if missing_refs:
            missing_ids = [
                str(row.get("evidence_id") or row.get("id") or "").strip()
                for row in missing_refs[:4]
                if str(row.get("evidence_id") or row.get("id") or "").strip()
            ]
            if missing_ids:
                lines.append("  - 缺少正文标注：" + "、".join(missing_ids))
        issue_count += 1
        if issue_count >= 3:
            break
    return lines


def _evidence_reader_html_path(job_id: str) -> Path | None:
    """Return the on-disk brief.html path for job_id if it has been rendered.

    Only single_url_brief jobs produce this artifact today (see
    evidence_reader_service.render_single_url_brief_html / VISION.md 第2节);
    checking file existence rather than job_type keeps this from needing to
    know about pipeline internals, and naturally covers the day question_loop
    also gets an HTML artifact.
    """
    if not job_id:
        return None
    try:
        from .research_service import RESEARCH_ARTIFACT_DIR, _safe_artifact_name
    except ImportError:  # pragma: no cover
        return None
    path = RESEARCH_ARTIFACT_DIR / _safe_artifact_name(job_id) / "brief.html"
    return path if path.exists() else None


def _research_links(job_id: str) -> list[str]:
    base = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5173").rstrip("/")
    lines = ["🔗 核查入口"]
    if job_id:
        lines.append(f"• Research 详情：{base}/research?job_id={job_id}")
        if _evidence_reader_html_path(job_id) is not None:
            lines.append(f"• 证据阅读版（附引用溯源）：{base}/api/v1/research/jobs/{job_id}/evidence-reader")
    lines.append(f"• 群运行台：{base}/channels")
    return lines


def _job_elapsed_label(job: dict[str, Any]) -> str:
    started = str(job.get("created_at") or "")
    status = str(job.get("status") or "")
    end = str(job.get("updated_at") or "") if status in FEISHU_PROGRESS_TERMINAL_STATES else now_text()
    ms = _elapsed_ms(started, end)
    return _format_duration(ms) if ms is not None else "-"


def _step_elapsed_label(step: dict[str, Any]) -> str:
    elapsed_ms = step.get("elapsed_ms")
    if elapsed_ms is not None:
        try:
            return _format_duration(int(elapsed_ms))
        except (TypeError, ValueError):
            return ""
    if str(step.get("status") or "") == "running":
        started = str(step.get("started_at") or "")
        ms = _elapsed_ms(started, now_text())
        return _format_duration(ms) if ms is not None else ""
    return ""


def _elapsed_ms(started_at: str | None, finished_at: str | None) -> int | None:
    if not started_at or not finished_at:
        return None
    try:
        started = time.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        finished = time.strptime(finished_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return max(0, int((time.mktime(finished) - time.mktime(started)) * 1000))


def _format_duration(ms: int | None) -> str:
    if ms is None:
        return ""
    seconds = max(0, int(round(ms / 1000)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _short_text(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 1)].rstrip() + "…"


def _reply_message_payload(text: str) -> tuple[str, str]:
    mode = os.getenv("FEISHU_REPLY_FORMAT", "post").strip().lower()
    if mode == "text":
        return "text", json.dumps({"text": _markdown_to_plain_text(_trim_reply(text))}, ensure_ascii=False)
    return "post", json.dumps(_markdown_to_feishu_post(_trim_reply(text)), ensure_ascii=False)


def _markdown_to_feishu_post(text: str) -> dict[str, Any]:
    lines = _markdown_to_plain_text(text).splitlines()
    content: list[list[dict[str, str]]] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            content.append([{"tag": "text", "text": ""}])
            continue
        content.append([{"tag": "text", "text": line}])
    if not content:
        content = [[{"tag": "text", "text": ""}]]
    return {"zh_cn": {"title": "", "content": content}}


def _markdown_to_plain_text(text: str) -> str:
    lines: list[str] = []
    in_code_block = False
    for raw in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
            if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
                line = ""
            line = re.sub(r"^\s*[-*+]\s+", "• ", line)
            line = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", line)
            line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            line = re.sub(r"__(.+?)__", r"\1", line)
            line = re.sub(r"`([^`]+)`", r"\1", line)
            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", line)
        lines.append(line)
    return "\n".join(lines).strip()


def _tenant_access_token() -> str:
    _load_env()
    now = time.time()
    cached = str(_TOKEN_CACHE.get("token") or "")
    if cached and float(_TOKEN_CACHE.get("expires_at") or 0) > now + 60:
        return cached
    app_id = os.getenv("FEISHU_APP_ID", "").strip()
    app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required for real send")
    response = _post_json(
        f"{_api_base()}/auth/v3/tenant_access_token/internal",
        {"app_id": app_id, "app_secret": app_secret},
    )
    if int(response.get("code") or 0) != 0:
        raise RuntimeError(f"Feishu tenant_access_token failed: {response.get('msg') or response}")
    token = str((response.get("tenant_access_token") or response.get("data", {}).get("tenant_access_token") or "")).strip()
    if not token:
        raise RuntimeError("Feishu tenant_access_token response did not include a token")
    expire = int(response.get("expire") or 7200)
    _TOKEN_CACHE.update({"token": token, "expires_at": now + max(60, expire)})
    return token


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    return _request_json("POST", url, payload, headers=headers)


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return _request_json("GET", url, None, headers=headers)


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("FEISHU_HTTP_TIMEOUT", "15"))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu HTTP {exc.code}: {raw[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Feishu request failed: {exc}") from exc
    return json.loads(raw or "{}")


def _api_base() -> str:
    _load_env()
    return (os.getenv("FEISHU_OPEN_API_BASE") or FEISHU_OPEN_API_BASE).rstrip("/")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _extract_event(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event")
    if isinstance(event, dict):
        return event
    schema = payload.get("schema")
    if schema == "2.0" and isinstance(payload.get("header"), dict):
        event = payload.get("event")
        return event if isinstance(event, dict) else {}
    return {}


def _is_text_message(message: dict[str, Any]) -> bool:
    return str(message.get("message_type") or "").lower() == "text"


def _is_duplicate_message(message_id: str | None) -> bool:
    """Persistent, cross-process/cross-restart dedup (see feishu_dedupe_service.py).

    Feishu's long-connection transport redelivers the same message_id at arbitrary
    intervals (46 minutes to several hours observed in production), including cases
    with no ping-timeout/reconnect log nearby -- so a short in-memory sliding window
    cannot catch it. FEISHU_MESSAGE_DEDUPE_SECONDS still tunes the retention window,
    but now defaults to a long TTL instead of Feishu's short retry cadence.
    """
    if not message_id:
        return False
    ttl_seconds = float(os.getenv("FEISHU_MESSAGE_DEDUPE_SECONDS", "") or feishu_dedupe_service.DEFAULT_TTL_SECONDS)
    return feishu_dedupe_service.mark_seen_or_duplicate(message_id, ttl_seconds=ttl_seconds)


def _is_bot_sender(event: dict[str, Any]) -> bool:
    sender = event.get("sender") or {}
    sender_type = str(sender.get("sender_type") or sender.get("type") or "").lower()
    if sender_type in {"app", "bot"}:
        return True
    sender_id = sender.get("sender_id") or {}
    if isinstance(sender_id, dict):
        id_type = str(sender_id.get("id_type") or "").lower()
        if id_type in {"app_id", "bot_id"}:
            return True
        value = " ".join(str(v).lower() for v in sender_id.values())
        if "cli_" in value or "bot" in value:
            return True
    return False


def _chat_id(message: dict[str, Any]) -> str:
    return str(message.get("chat_id") or "").strip()


def _root_id(message: dict[str, Any]) -> str | None:
    return str(message.get("message_id") or message.get("root_id") or "").strip() or None


def _chat_title(
    message: dict[str, Any],
    chat_id: str,
    *,
    allow_api: bool | None = None,
    force_refresh: bool = False,
) -> str | None:
    """Resolve a human-readable group name without treating ``chat_type`` as a title.

    ``im.message.receive_v1`` identifies group conversations with ``chat_type=group`` but
    does not normally include the group's display name. When real Feishu API access is
    enabled, resolve that name from ``GET /im/v1/chats/{chat_id}`` and cache it briefly so
    an active group does not trigger one lookup per message.
    """
    inline_title = str(message.get("chat_name") or "").strip()
    if inline_title:
        return inline_title

    api_enabled = (
        os.getenv("FEISHU_ENABLE_REAL_SEND", "").lower() in {"1", "true", "yes"}
        if allow_api is None
        else allow_api
    )
    if not api_enabled:
        return None

    now = time.time()
    cached = _CHAT_TITLE_CACHE.get(chat_id)
    if (
        not force_refresh
        and cached is not None
        and float(cached.get("expires_at") or 0) > now
    ):
        return str(cached.get("title") or "").strip() or None

    try:
        title = _fetch_feishu_chat_title(chat_id)
    except Exception as exc:
        _CHAT_TITLE_CACHE[chat_id] = {
            "title": None,
            "expires_at": now + CHAT_TITLE_FAILURE_CACHE_TTL_SECONDS,
        }
        emit_event(
            "feishu.chat_title.lookup_failed",
            "feishu",
            "Failed to resolve Feishu chat title; preserving the existing channel title.",
            session_id=f"feishu:{chat_id}",
            data={"error_type": type(exc).__name__},
        )
        return None

    _CHAT_TITLE_CACHE[chat_id] = {
        "title": title,
        "expires_at": now
        + (
            CHAT_TITLE_CACHE_TTL_SECONDS
            if title
            else CHAT_TITLE_FAILURE_CACHE_TTL_SECONDS
        ),
    }
    return title


def _fetch_feishu_chat_title(chat_id: str) -> str | None:
    token = _tenant_access_token()
    response = _get_json(
        f"{_api_base()}/im/v1/chats/{quote(chat_id, safe='')}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if int(response.get("code") or 0) != 0:
        raise RuntimeError(f"Feishu get chat failed: {response.get('msg') or response}")
    title = str((response.get("data") or {}).get("name") or "").strip()
    return title or None


def _sender_id(sender: dict[str, Any]) -> str | None:
    sender_id = sender.get("sender_id") or {}
    if isinstance(sender_id, dict):
        return str(sender_id.get("open_id") or sender_id.get("user_id") or sender_id.get("union_id") or "").strip() or None
    return str(sender_id or "").strip() or None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, dict):
        return str(content.get("text") or "").strip()
    text = str(content or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return str(parsed.get("text") or "").strip()
    except json.JSONDecodeError:
        pass
    return text


def _trim_reply(text: str, limit: int | None = None) -> str:
    if limit is None:
        try:
            limit = int(os.getenv("FEISHU_REPLY_CHAR_LIMIT", "3200") or 3200)
        except ValueError:
            limit = 3200
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 20].rstrip() + "\n...（已截断）"
