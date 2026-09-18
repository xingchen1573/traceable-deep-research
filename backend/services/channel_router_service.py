"""Channel-aware chat routing."""

from __future__ import annotations

from typing import Any

try:
    from ...tools import set_current_thread_id
except ImportError:  # pragma: no cover
    from tools import set_current_thread_id

from . import channel_service, conversation_service, knowledge_base_service
from .common import now_text


ENTER_KB_KEYWORDS = (
    "切换到知识库问答",
    "进入知识库问答",
    "开启知识库问答",
    "知识库问答模式",
    "切换为知识库",
    "进入知识库模式",
)
EXIT_KB_KEYWORDS = (
    "退出知识库",
    "退出问答模式",
    "回到日常模式",
    "回到主模式",
    "切回主模式",
    "切回日常",
)
FOLLOWUP_KEYWORDS = ("补查", "补充证据", "完善章节", "重新完善")


def route_channel_message(
    *,
    message: str,
    channel_id: str | None = None,
    platform: str | None = None,
    external_id: str | None = None,
    user_id: str | None = None,
    main_chat_handler: Any,
) -> dict[str, Any]:
    """Route one channel-scoped message.

    RESEARCH_KEYWORDS hard-routing and the standalone section-revision pre-filter (which used
    to call find_revisable_job_for_channel + extract_section_revision_intent directly here,
    bypassing the main Agent entirely) have both been retired -- see
    docs/modules/agent-dispatcher-role-v0.1.md 3.7/3.8. Both立项 (research job creation) and
    批准/返修 (approve/revise) decisions now live in the main Agent's own tool choices
    (start_research_job_tool / check_research_job_tool / approve_research_job_tool /
    revise_report_section_tool), reached uniformly through main_chat_handler below. This keeps
    exactly one entry point ("does this channel's message go through the Agent or not") instead
    of some intents bypassing the Agent and others not.

    kb_qa mode switching (enter/exit keyword matching below) and the followup-request dead-end
    message are unrelated to that decision and stay as they were. What *did* change for kb_qa:
    once inside kb_qa mode, _route_qa() below now hands each message to a dedicated
    kb_qa_agent (persistent, multi-turn) instead of a single deterministic
    knowledge_base_service.query() call -- see docs/modules/kb-qa-channel-bypass-v0.1.md 4.1/4.2.
    That's a separate agent from the main one built by main_chat_handler's build_agent(); the
    entry/exit keyword gate below is what decides which of the two a message ever reaches.
    """
    channel = channel_service.ensure_channel(channel_id=channel_id, platform=platform, external_id=external_id)
    text = str(message or "").strip()
    if not text:
        raise ValueError("message is required")

    if _is_exit_kb(text):
        channel = channel_service.update_channel(channel["channel_id"], mode="main")
        thread_id = channel_service.thread_id_for(channel["channel_id"], "main")
        conversation_service.save_message(thread_id, "user", text)
        assistant_text = "已回到日常主管理模式。"
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        return _completed(channel=channel, thread_id=thread_id, assistant_text=assistant_text, user_id=user_id)

    if _is_enter_kb(text):
        topic = _extract_topic(text) or channel.get("active_kb")
        if topic:
            knowledge_base_service.set_active(topic, channel_service.thread_id_for(channel["channel_id"], "kb_qa"))
            channel = channel_service.update_channel(channel["channel_id"], mode="kb_qa", active_kb=topic)
            assistant_text = f"已切换到知识库问答模式，当前知识库：{topic}。"
        else:
            channel = channel_service.update_channel(channel["channel_id"], mode="kb_qa")
            listing = knowledge_base_service.list_knowledge_bases(include_index=False, include_stats=False)
            topics = [item.get("topic") for item in listing.get("topics", []) if item.get("topic")]
            assistant_text = "已切换到知识库问答模式。请先选择知识库 topic。"
            if topics:
                assistant_text += " 可选：" + "、".join(str(item) for item in topics[:8])
        thread_id = channel_service.thread_id_for(channel["channel_id"], "kb_qa")
        conversation_service.save_message(thread_id, "user", text)
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        return _completed(channel=channel, thread_id=thread_id, assistant_text=assistant_text, user_id=user_id)

    if channel.get("mode") == "kb_qa":
        return _route_qa(channel=channel, message=text, user_id=user_id)

    if _is_followup_request(text):
        thread_id = channel_service.thread_id_for(channel["channel_id"], "main")
        conversation_service.save_message(thread_id, "user", text)
        assistant_text = "请在当前群的调研任务中选择具体 job 或章节后发起补查。第一期已支持 channel 级 followup 任务展示。"
        conversation_service.save_message(thread_id, "assistant", assistant_text)
        return _completed(channel=channel, thread_id=thread_id, assistant_text=assistant_text, user_id=user_id)

    thread_id = channel_service.thread_id_for(channel["channel_id"], "main")
    return main_chat_handler(thread_id, text, channel)


def _route_qa(*, channel: dict[str, Any], message: str, user_id: str | None) -> dict[str, Any]:
    """Drive one kb_qa-mode turn through the dedicated, stateful kb_qa_agent instead of a single
    deterministic knowledge_base_service.query() call -- see
    docs/modules/kb-qa-channel-bypass-v0.1.md 4.2. thread_id_for(channel_id, "kb_qa") is the same
    per-channel, per-mode thread_id this function has always used for message logging; it now
    also drives the agent's own LangGraph checkpointer, giving this mode real multi-turn memory
    across separate messages instead of re-answering from scratch every time.

    No Python-level "no active topic" response short-circuit here on purpose: kb_qa_agent's own
    tool set (query_knowledge_base / list_knowledge_bases / set_active_knowledge_base) is exactly
    what it needs to detect a missing topic (query_knowledge_base returns need_topic) and resolve
    it itself, per KB_QA_AGENT_PROMPT -- that used to be duplicated Python logic here, now it's
    the agent's own reasoning, consistent with how the rest of this module already defers intent
    judgment to an Agent rather than routing it in Python (see module docstring). The one bit of
    Python-level state handling that *is* still needed is syncing channel["active_kb"] (SQL) into
    the in-memory active-topic-by-thread state kb_qa_agent's tools actually read from -- see the
    comment below.

    get_kb_qa_agent/invoke_agent_once/thread_config are imported lazily here (not at module
    import time) to avoid a load-time cycle with backend.dependencies -> ..main -> ...tools, the
    same pattern tools/research_jobs.py already uses for research_service/job_service.
    """
    from ..dependencies import get_kb_qa_agent, invoke_agent_once, thread_config

    thread_id = channel_service.thread_id_for(channel["channel_id"], "kb_qa")
    conversation_service.save_message(thread_id, "user", message)
    set_current_thread_id(thread_id)

    # channel["active_kb"] (SQL) can be set two ways: the enter-kb-mode keyword path above (which
    # already calls knowledge_base_service.set_active(), keeping the two in sync) or a direct
    # PUT /channels/{id} API call (backend/routes/channels.py::update_channel touches only the
    # SQL row, it does not call set_active()). kb_qa_agent's tools read the *in-memory*
    # active-topic-by-thread state (tools/knowledge_qa.py::_ACTIVE_KB_BY_THREAD), not this SQL
    # column, so without this sync a channel pinned to a topic only via PUT would make
    # kb_qa_agent spuriously think there's no active topic. Only seed when nothing is tracked yet
    # for this thread -- once set, leave it alone so a mid-conversation set_active_knowledge_base
    # tool call (4.2's "对话中途换 topic") isn't clobbered back to the stale SQL value on the next
    # turn.
    channel_active_kb = str(channel.get("active_kb") or "").strip()
    if channel_active_kb and not knowledge_base_service.get_active(thread_id).get("active_topic"):
        knowledge_base_service.set_active(channel_active_kb, thread_id)

    agent = get_kb_qa_agent()
    result = invoke_agent_once(
        agent,
        {"messages": [{"role": "user", "content": message}]},
        thread_config(thread_id),
    )
    if result["status"] == "interrupted":
        # kb_qa_agent's tools are all read-only (query_knowledge_base / set_active_knowledge_base
        # / list_knowledge_bases / inspect_knowledge_base / summarize_knowledge_base) -- see
        # docs/modules/kb-qa-channel-bypass-v0.1.md 4.2 "不需要 HITL". This branch should be
        # unreachable; if it somehow fires, don't leave the turn hanging on an approval flow this
        # mode has no UI for -- surface a plain-language fallback instead.
        assistant_text = "知识库问答遇到了需要人工确认的操作，但当前模式不支持审批流程；请退出知识库问答模式后重试。"
    else:
        assistant_text = result["assistant_text"] or "知识库没有返回可用答案。"

    # Read back the live active topic *after* the turn (not channel["active_kb"], which is only
    # written by _is_enter_kb below and would go stale the moment kb_qa_agent calls
    # set_active_knowledge_base mid-conversation to switch topics -- see 4.2's "对话中途换 topic").
    active_kb = str((knowledge_base_service.get_active(thread_id) or {}).get("active_topic") or "").strip()
    conversation_service.save_message(thread_id, "assistant", assistant_text)
    return {
        "status": "completed",
        "state": "completed",
        "channel_id": channel["channel_id"],
        "mode": "kb_qa",
        "active_kb": active_kb or None,
        "thread_id": thread_id,
        "run_id": None,
        "assistant_text": assistant_text,
        "qa": {
            "channel_id": channel["channel_id"],
            "mode": "kb_qa",
            "active_kb": active_kb or None,
            "answer": assistant_text,
            # kb_qa_agent answers with free-form prose (citations are embedded in the text per
            # KB_QA_AGENT_PROMPT, not a separate structured field the way the old single
            # query_knowledge_base call's JSON payload was) -- no downstream consumer reads these
            # two fields today (checked: not used by frontend), kept for response-shape stability.
            "citations": [],
            "gaps": [],
        },
        "channel": channel,
        "user_id": user_id,
        "updated_at": now_text(),
    }


def _completed(*, channel: dict[str, Any], thread_id: str, assistant_text: str, user_id: str | None) -> dict[str, Any]:
    return {
        "status": "completed",
        "state": "completed",
        "channel_id": channel["channel_id"],
        "mode": channel.get("mode") or "main",
        "active_kb": channel.get("active_kb"),
        "thread_id": thread_id,
        "run_id": None,
        "assistant_text": assistant_text,
        "channel": channel,
        "user_id": user_id,
        "updated_at": now_text(),
    }


def _is_enter_kb(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in text for keyword in ENTER_KB_KEYWORDS) or "kb_qa" in lowered


def _is_exit_kb(text: str) -> bool:
    return any(keyword in text for keyword in EXIT_KB_KEYWORDS)


def _is_followup_request(text: str) -> bool:
    return any(keyword in text for keyword in FOLLOWUP_KEYWORDS)


def _extract_topic(text: str) -> str:
    markers = ("知识库：", "知识库:", "topic：", "topic:", "使用 ")
    for marker in markers:
        if marker not in text:
            continue
        tail = text.split(marker, 1)[1].strip()
        if not tail:
            continue
        for sep in (" ", "，", "。", "\n", "\t"):
            if sep in tail:
                tail = tail.split(sep, 1)[0]
        return tail.strip()
    return ""
