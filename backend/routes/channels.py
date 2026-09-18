"""Channel runtime API routes."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Query

from ..schemas.channel import ChannelCreateRequest, ChannelPushConfigRequest, ChannelScopeBindingRequest, ChannelUpdateRequest
from ..services import channel_service, conversation_service, feishu_service, knowledge_scope_service, push_service
from ..services.common import map_exception, ok
from ..services.job_service import job_service

router = APIRouter(prefix="/api/v1/channels", tags=["channels"])


@router.get("")
def list_channels(limit: int = Query(default=100, ge=1, le=200), offset: int = Query(default=0, ge=0)):
    try:
        return ok({
            "channels": channel_service.list_channels(limit=limit, offset=offset),
            "total": channel_service.count_channels(),
        })
    except Exception as exc:
        return map_exception(exc)


@router.post("")
def create_channel(request: ChannelCreateRequest):
    try:
        channel = channel_service.ensure_channel(
            platform=request.platform,
            external_id=request.external_id,
            title=request.title,
            memory_scope=request.memory_scope,
        )
        return ok(channel)
    except Exception as exc:
        return map_exception(exc)


@router.post("/sync-feishu-titles")
def sync_feishu_titles():
    try:
        return ok(feishu_service.sync_channel_titles())
    except Exception as exc:
        return map_exception(exc)


@router.get("/{channel_id}")
def get_channel(channel_id: str):
    try:
        channel = channel_service.get_channel(channel_id)
        main_thread = channel_service.thread_id_for(channel_id, "main")
        qa_thread = channel_service.thread_id_for(channel_id, "kb_qa")
        return ok({
            **channel,
            "threads": {
                "main": main_thread,
                "kb_qa": qa_thread,
            },
            "conversations": {
                "main": conversation_service.get_conversation(main_thread),
                "kb_qa": conversation_service.get_conversation(qa_thread),
            },
        })
    except Exception as exc:
        return map_exception(exc)


@router.put("/{channel_id}")
def update_channel(channel_id: str, request: ChannelUpdateRequest):
    try:
        return ok(
            channel_service.update_channel(
                channel_id,
                title=request.title,
                mode=request.mode,
                active_kb=request.active_kb,
                topic_binding=request.topic_binding,
                memory_scope=request.memory_scope,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.get("/{channel_id}/messages")
def get_channel_messages(channel_id: str, mode: str = Query(default="main"), limit: int = Query(default=200, ge=1, le=500)):
    try:
        thread_id = channel_service.thread_id_for(channel_id, mode)
        return ok({
            "channel_id": channel_id,
            "mode": mode,
            "thread_id": thread_id,
            "conversation": conversation_service.get_conversation(thread_id),
            "messages": conversation_service.get_messages(thread_id, limit=limit),
        })
    except Exception as exc:
        return map_exception(exc)


@router.get("/{channel_id}/scope-binding")
def get_channel_scope_binding(channel_id: str):
    try:
        return ok(channel_service.get_scope_binding(channel_id))
    except Exception as exc:
        return map_exception(exc)


@router.put("/{channel_id}/scope-binding")
def bind_channel_scope(channel_id: str, request: ChannelScopeBindingRequest):
    try:
        knowledge_scope_service.resolve_scope(request.scope_type, request.scope_id)
        channel = channel_service.bind_scope(channel_id, request.scope_type, request.scope_id)
        return ok({**channel, "scope_binding": channel_service.get_scope_binding(channel_id)})
    except Exception as exc:
        return map_exception(exc)


@router.delete("/{channel_id}/scope-binding")
def unbind_channel_scope(channel_id: str):
    try:
        return ok(channel_service.unbind_scope(channel_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{channel_id}/jobs")
def get_channel_jobs(
    channel_id: str,
    job_type: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    try:
        return ok({
            "channel_id": channel_id,
            "jobs": job_service.list(job_type=job_type, status=status, limit=limit, channel_id=channel_id),
        })
    except Exception as exc:
        return map_exception(exc)


def _format_push_config(config: dict) -> dict:
    timezone_name = str(config.get("timezone") or channel_service.DEFAULT_PUSH_TIMEZONE)
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        timezone = ZoneInfo(channel_service.DEFAULT_PUSH_TIMEZONE)

    def format_timestamp(value):
        return datetime.fromtimestamp(float(value), timezone).strftime("%Y-%m-%d %H:%M") if value is not None else None

    scheduler = push_service.scheduler_status()
    return {
        **config,
        "last_pushed_at": format_timestamp(config.get("last_pushed_at")),
        "next_push_at": format_timestamp(config.get("next_push_at")),
        "scheduler_enabled": scheduler["enabled"],
        "scheduler_running": scheduler["running"],
        "scheduler_tick_interval_seconds": scheduler["tick_interval_seconds"],
    }


@router.get("/{channel_id}/push-config")
def get_channel_push_config(channel_id: str):
    try:
        config = channel_service.get_push_config(channel_id)
        if config["enabled"] and config["next_push_at"] is None:
            channel_service.initialize_next_push_at(channel_id, now_ts=datetime.now().timestamp())
            config = channel_service.get_push_config(channel_id)
        return ok(_format_push_config(config))
    except Exception as exc:
        return map_exception(exc)


@router.put("/{channel_id}/push-config")
def update_channel_push_config(channel_id: str, request: ChannelPushConfigRequest):
    try:
        config = channel_service.upsert_push_config(
            channel_id,
            enabled=request.enabled,
            frequency=request.frequency,
            content_strategy=request.content_strategy,
            push_time=request.push_time,
            timezone=request.timezone,
        )
        return ok(_format_push_config(config))
    except Exception as exc:
        return map_exception(exc)
