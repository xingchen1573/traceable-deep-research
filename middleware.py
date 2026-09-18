"""LangChain v1 middleware for Deep Research."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware

try:
    from langchain.messages import RemoveMessage
except Exception:  # pragma: no cover
    RemoveMessage = None  # type: ignore

try:
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
except Exception:  # pragma: no cover
    REMOVE_ALL_MESSAGES = "__remove_all__"

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
except Exception:  # pragma: no cover
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore

try:
    from . import memory as memory_store
    from . import tools as harness_tools
    from .observability import current_session_id, emit_event
    from .prompts import STATE_COMPACT_PROMPT
except ImportError:  # pragma: no cover - script mode fallback
    import memory as memory_store
    import tools as harness_tools
    from observability import current_session_id, emit_event
    from prompts import STATE_COMPACT_PROMPT


def _content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "")
    return str(getattr(message, "type", "") or getattr(message, "role", ""))


def _clone_with_content(message: Any, content: str) -> Any:
    if isinstance(message, dict):
        updated = dict(message)
        updated["content"] = content
        return updated
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    if hasattr(message, "copy"):
        return message.copy(update={"content": content})
    return message


def _replace_messages(messages: list[Any]) -> dict[str, Any]:
    if RemoveMessage is None:
        return {"messages": messages}
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        kind = _message_type(message).lower()
        class_name = message.__class__.__name__.lower()
        if kind in {"human", "user"} or class_name.startswith("human"):
            return _content(message)
    return ""


def _looks_like_context_overflow(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in [
            "context length",
            "maximum context",
            "token limit",
            "too many tokens",
            "context window",
        ]
    )


# Tool names whose ToolMessage output must survive ContextSnipMiddleware/MicroCompactMiddleware/
# ContextCompactMiddleware unaltered -- see
# docs/modules/deep-research-step1-clarify-and-dispatch-discussion-v0.1.md "基建改动": a
# clarify_concept_tool/draft_research_plan_tool exchange is the regulatory plan the user is
# actively confirming turn-by-turn, not disposable tool noise -- snipping, placeholder-izing, or
# summarizing it away mid-negotiation would silently invalidate the draft the user is looking at.
# ReactiveCompactMiddleware is deliberately NOT in scope: it only fires as a last-resort retry
# after a real context-overflow exception, and forcing protected messages to survive there could
# defeat the trim that was needed to recover from the overflow in the first place.
_PROTECTED_TOOL_NAMES = frozenset({"clarify_concept_tool", "draft_research_plan_tool"})


def _tool_message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "")
    return str(getattr(message, "name", "") or "")


def _tool_message_call_id(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("tool_call_id") or "")
    return str(getattr(message, "tool_call_id", "") or "")


def _ai_tool_call_ids(message: Any) -> set[str]:
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    ids: set[str] = set()
    for call in calls:
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if call_id:
            ids.add(str(call_id))
    return ids


def _is_protected_tool_message(message: Any) -> bool:
    kind = _message_type(message).lower()
    is_tool = kind == "tool" or message.__class__.__name__.lower().startswith("tool")
    return is_tool and _tool_message_name(message) in _PROTECTED_TOOL_NAMES


def _protected_message_indices(messages: list[Any]) -> set[int]:
    """Indices of messages that must not be snipped/compacted/summarized away.

    A ToolMessage can't be protected in isolation: it is linked to the AIMessage that issued
    the tool call via tool_call_id, and (per OpenAI/Anthropic tool-calling contracts) every
    tool_call on an AIMessage needs its matching ToolMessage present, or the next model call is
    rejected as an invalid message sequence. So protecting a clarify_concept_tool/
    draft_research_plan_tool ToolMessage means protecting its issuing AIMessage AND, if that
    AIMessage batched other tool_calls in the same turn, their sibling ToolMessages too --
    otherwise removing/altering the sibling would orphan a tool_call the model itself made.
    """
    protected_call_ids: set[str] = {
        _tool_message_call_id(message)
        for message in messages
        if _is_protected_tool_message(message) and _tool_message_call_id(message)
    }
    if protected_call_ids:
        for message in messages:
            call_ids = _ai_tool_call_ids(message)
            if call_ids & protected_call_ids:
                protected_call_ids |= call_ids
    protected: set[int] = set()
    for index, message in enumerate(messages):
        if _is_protected_tool_message(message):
            protected.add(index)
        elif _ai_tool_call_ids(message) & protected_call_ids:
            protected.add(index)
        elif _tool_message_call_id(message) and _tool_message_call_id(message) in protected_call_ids:
            protected.add(index)
    return protected


class ContextSnipMiddleware(AgentMiddleware):
    """Hard-bound unusually large message blocks before model calls."""

    def __init__(self, max_chars: int = 5000, head_chars: int = 1800, tail_chars: int = 800) -> None:
        self.max_chars = max_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        protected = _protected_message_indices(messages)
        changed = False
        snipped_count = 0
        updated: list[Any] = []
        for index, message in enumerate(messages):
            text = _content(message)
            if len(text) <= self.max_chars or index in protected:
                updated.append(message)
                continue
            omitted = len(text) - self.head_chars - self.tail_chars
            snipped = (
                f"{text[:self.head_chars]}\n\n"
                f"[snipped: 已省略 {omitted} 个字符]\n\n"
                f"{text[-self.tail_chars:]}"
            )
            updated.append(_clone_with_content(message, snipped))
            changed = True
            snipped_count += 1
        if changed:
            emit_event(
                "context.snip.triggered",
                "ContextSnipMiddleware",
                "Snipped oversized message blocks before model call.",
                session_id=_thread_id(runtime),
                data={"snipped_count": snipped_count, "max_chars": self.max_chars},
            )
            return _replace_messages(updated)
        return None


class MicroCompactMiddleware(AgentMiddleware):
    """Replace older non-file tool outputs with compact placeholders."""

    def __init__(self, keep_recent: int = 6) -> None:
        self.keep_recent = keep_recent

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        if len(messages) <= self.keep_recent:
            return None
        cutoff = max(0, len(messages) - self.keep_recent)
        protected = _protected_message_indices(messages)
        changed = False
        compacted_count = 0
        updated: list[Any] = []
        for index, message in enumerate(messages):
            text = _content(message)
            kind = _message_type(message).lower()
            is_tool = kind == "tool" or message.__class__.__name__.lower().startswith("tool")
            looks_like_file = "```" in text or "path=" in text.lower() or "file:" in text.lower()
            if index < cutoff and is_tool and not looks_like_file and len(text) > 400 and index not in protected:
                compacted = (
                    "[micro-compact: 较早的非文件类工具结果已被消费并压缩。"
                    f"原始输出长度={len(text)} 字符。]"
                )
                updated.append(_clone_with_content(message, compacted))
                changed = True
                compacted_count += 1
            else:
                updated.append(message)
        if changed:
            emit_event(
                "context.microcompact.triggered",
                "MicroCompactMiddleware",
                "Micro-compacted older non-file tool outputs.",
                session_id=_thread_id(runtime),
                data={"compacted_count": compacted_count, "keep_recent": self.keep_recent},
            )
            return _replace_messages(updated)
        return None


class SkillSelectionMiddleware(AgentMiddleware):
    """Inject lightweight skill-routing guidance, not full skill bodies."""

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        latest = _latest_user_text(messages).lower()
        hints: list[str] = []
        if any(token in latest for token in ["repo", "code", "file", "project", "仓库", "代码", "文件"]):
            hints.append("repo_reader")
        if any(token in latest for token in ["url", "http", "web", "网页", "网站"]):
            hints.append("web_reader")
        if not hints:
            return None
        emit_event(
            "skill.selection.triggered",
            "SkillSelectionMiddleware",
            "Selected lightweight skill routing hints.",
            session_id=_thread_id(runtime),
            data={"hints": hints},
        )
        summaries = harness_tools.skill_summaries()
        guidance = (
            "技能路由提示：当前任务可能适合使用 "
            f"{', '.join(hints)}。可用技能摘要：\n{summaries}\n"
            "只有在确实需要时，才通过 load_skill(skill_name) 加载完整技能正文。"
        )
        if SystemMessage is not None:
            return {"messages": [SystemMessage(content=guidance)]}
        return {"messages": [{"role": "system", "content": guidance}]}


class MemoryWritePolicyMiddleware(AgentMiddleware):
    """Nudge hot-path semantic memory writes for clear, low-risk user facts."""

    NAME_PATTERN = re.compile(r"(?:我叫|叫我|my name is|call me)\s*([^\s,，。.!！?？]{1,40})", re.I)
    PREFERENCE_PATTERN = re.compile(r"(?:我喜欢|我不喜欢|以后|默认|偏好|prefer|preference)", re.I)

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        latest = _latest_user_text(messages)
        if not latest:
            return None

        name_match = self.NAME_PATTERN.search(latest)
        preference_match = self.PREFERENCE_PATTERN.search(latest)
        if not name_match and not preference_match:
            return None

        if name_match:
            fact = f"用户偏好的称呼/代号是 {name_match.group(1).strip()}。"
            guidance = (
                "记忆写入策略：用户提供了一个稳定、低风险的身份事实。"
                "在给出最终回复前，调用 write_memory_topic，topic_id='user_profile'，"
                "title='User Profile'，附上简洁的 summary，并在 body 中记录该事实。"
                f"需要写入的事实：{fact} "
                "只记录用户在对话中明确陈述的内容，不要推断或补全时间线、来源或历史叙事。"
                "日期与时间戳由系统通过 updated_at 自动维护，不要在 body 中自行编写日期。"
                "/dream 仅用于复盘，会从每日日志生成 dream_report.md，并不会保存该事实。"
            )
        else:
            guidance = (
                "记忆写入策略：用户可能提供了一条持久偏好或工作规则。"
                "如果它稳定、跨会话有用且不敏感，调用 write_memory_topic 更新合适的 topic。"
                "如果它是临时或敏感的，则不要存储。"
                "只记录用户明确陈述的内容，不要推断或补全时间线与来源；"
                "日期由系统通过 updated_at 自动维护，不要在 body 中自行编写日期。"
                "/dream 仅用于复盘，会从每日日志生成 dream_report.md。"
            )

        if SystemMessage is not None:
            return {"messages": [SystemMessage(content=guidance)]}
        return {"messages": [{"role": "system", "content": guidance}]}


class RelevantMemoryRecallMiddleware(AgentMiddleware):
    """Preload a few likely-relevant long-term topic bodies before model calls."""

    def __init__(
        self,
        recall_model: Any,
        surfaced_registry: memory_store.SurfacedMemoryRegistry,
        max_topics: int = 3,
    ) -> None:
        self.recall_model = recall_model
        self.surfaced_registry = surfaced_registry
        self.max_topics = max(1, max_topics)

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        query = self._query_from_messages(messages)
        if not query.strip():
            return None

        session_id = _thread_id(runtime)
        topics = memory_store.list_memory_topics()
        already_surfaced = self.surfaced_registry.surfaced(session_id)
        emit_event(
            "memory.recall.started",
            "RelevantMemoryRecallMiddleware",
            "Relevant long-term memory recall started.",
            session_id=session_id,
            data={"topic_count": len(topics), "already_surfaced": sorted(already_surfaced)},
        )
        candidates = [
            topic
            for topic in topics
            if not self.surfaced_registry.is_surfaced(session_id, topic.topic_id)
        ]
        if not candidates:
            return None

        selected_ids = self._select_topic_ids(query, candidates)
        if not selected_ids:
            return None
        emit_event(
            "memory.recall.selected",
            "RelevantMemoryRecallMiddleware",
            "Selected relevant long-term memory topics.",
            session_id=session_id,
            data={"topic_ids": selected_ids},
        )

        loaded = memory_store.load_memory_topics(selected_ids)
        if not loaded:
            return None

        surfaced_ids = [record.topic_id for record, _body in loaded]
        self.surfaced_registry.mark_many(session_id, surfaced_ids)
        emit_event(
            "memory.recall.injected",
            "RelevantMemoryRecallMiddleware",
            "Injected relevant long-term memory topic bodies.",
            session_id=session_id,
            data={"topic_ids": surfaced_ids, "count": len(surfaced_ids)},
        )
        guidance = self._render_loaded_topics(loaded)
        if SystemMessage is not None:
            return {"messages": [SystemMessage(content=guidance)]}
        return {"messages": [{"role": "system", "content": guidance}]}

    def _query_from_messages(self, messages: list[Any]) -> str:
        parts: list[str] = []
        latest = _latest_user_text(messages)
        if latest:
            parts.append(f"latest user request: {latest[:1200]}")
        for message in messages[-6:]:
            kind = _message_type(message).lower()
            if kind not in {"human", "user", "ai", "assistant"}:
                continue
            text = _content(message).strip()
            if text:
                parts.append(f"{kind}: {text[:500]}")
        return "\n".join(parts)[-3000:]

    def _select_topic_ids(self, query: str, candidates: list[memory_store.TopicRecord]) -> list[str]:
        candidate_ids = {topic.topic_id for topic in candidates}
        try:
            selected = self._select_with_model(query, candidates)
        except Exception:
            selected = []
        valid = self._valid_topic_ids(selected, candidate_ids)
        if valid:
            return valid
        return self._fallback_topic_ids(query, candidates)

    def _select_with_model(self, query: str, candidates: list[memory_store.TopicRecord]) -> list[str]:
        index_lines = [
            (
                f"- topic_id={topic.topic_id}; title={topic.title}; "
                f"trust={topic.trust_level}; updated={topic.updated_at}; "
                f"path={topic.path}; summary={topic.summary}"
            )
            for topic in candidates
        ]
        system = (
            "你是一个低成本的相关记忆选择器。"
            "从 topic 索引中选出最多 3 个对当前请求直接有用的 topic_id。"
            "只能依据 topic 索引元数据，不要臆想 topic 正文。"
            "只返回 JSON，格式严格为：{\"topic_ids\": [\"topic_id\"]}。"
            "如果没有相关 topic，返回 {\"topic_ids\": []}。"
        )
        user = "Topic 索引:\n" + "\n".join(index_lines) + "\n\n当前查询:\n" + query
        if SystemMessage is not None and HumanMessage is not None:
            response = self.recall_model.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        else:
            response = self.recall_model.invoke(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        data = _json_from_text(_content(response))
        topic_ids = data.get("topic_ids", [])
        if not isinstance(topic_ids, list):
            return []
        return [str(topic_id) for topic_id in topic_ids]

    def _valid_topic_ids(self, topic_ids: list[str], candidate_ids: set[str]) -> list[str]:
        valid: list[str] = []
        seen: set[str] = set()
        for topic_id in topic_ids:
            safe_id = memory_store.sanitize_topic_id(topic_id)
            if safe_id in candidate_ids and safe_id not in seen:
                valid.append(safe_id)
                seen.add(safe_id)
            if len(valid) >= self.max_topics:
                break
        return valid

    def _fallback_topic_ids(self, query: str, candidates: list[memory_store.TopicRecord]) -> list[str]:
        lowered = query.lower()
        scored: list[tuple[int, str]] = []
        name_keywords = ["name", "call me", "preferred name", "名字", "叫", "称呼"]
        memory_keywords = ["memory", "remember", "记忆", "长期", "topic", "autodream"]
        for topic in candidates:
            haystack = f"{topic.topic_id} {topic.title} {topic.summary}".lower()
            score = 0
            if any(keyword in lowered for keyword in name_keywords) and any(
                marker in haystack for marker in ["user", "profile", "name", "用户", "称呼"]
            ):
                score += 10
            if any(keyword in lowered for keyword in memory_keywords) and "memory" in haystack:
                score += 8
            query_terms = _simple_terms(lowered)
            score += sum(1 for term in query_terms if term in haystack)
            if score:
                scored.append((score, topic.topic_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [topic_id for _score, topic_id in scored[: self.max_topics]]

    def _render_loaded_topics(self, loaded: list[tuple[memory_store.TopicRecord, str]]) -> str:
        sections = [
            "相关长期记忆",
            "",
            "以下 topic 正文由相关记忆召回机制预加载。",
            "记忆只是线索，不是真相；行动前请先验证。",
            "如果需要的 topic 正文已经在这里展示，除非需要验证或补充缺失细节，否则不要再次调用 load_memory_topic。",
        ]
        for record, body in loaded:
            sections.extend(
                [
                    "",
                    f"## topic_id: {record.topic_id}",
                    f"path: {record.path}",
                    f"trust_level: {record.trust_level}",
                    f"updated_at: {record.updated_at}",
                    "",
                    body.strip(),
                ]
            )
        return "\n".join(sections).strip()


class FrustrationMiddleware(AgentMiddleware):
    """Detect user frustration and bias the next step toward verification."""

    PATTERN = re.compile(
        r"(不对|还是错|又错|你在干嘛|别瞎|wrong|still wrong|try again)",
        re.I,
    )

    def __init__(self) -> None:
        self.frustration_level: defaultdict[str, int] = defaultdict(int)

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        latest = _latest_user_text(messages)
        if not self.PATTERN.search(latest):
            return None
        thread_id = _thread_id(runtime)
        self.frustration_level[thread_id] += 1
        emit_event(
            "guard.frustration.hit",
            "FrustrationMiddleware",
            "User frustration signal detected.",
            session_id=thread_id,
            data={"frustration_level": self.frustration_level[thread_id]},
        )
        guidance = (
            "运行时守护：检测到用户出现挫败情绪。切换到聚焦验证模式。"
            "不要盲目重试。先简要说明可能哪里出了问题，验证假设，"
            "再选择一个更低噪声的下一步行动。"
        )
        if SystemMessage is not None:
            return {"messages": [SystemMessage(content=guidance)]}
        return {"messages": [{"role": "system", "content": guidance}]}


class ReactiveCompactMiddleware(AgentMiddleware):
    """Emergency trim + one retry if a model call overflows the context window."""

    def __init__(self, keep_recent: int = 8) -> None:
        self.keep_recent = keep_recent

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            return handler(request)
        except Exception as exc:
            if not _looks_like_context_overflow(exc):
                raise
            messages = list(getattr(request, "messages", []) or [])
            if len(messages) <= self.keep_recent:
                raise
            emit_event(
                "context.reactive_compact.retry",
                "ReactiveCompactMiddleware",
                "Context overflow detected; trimmed older context and retried once.",
                data={"message_count": len(messages), "keep_recent": self.keep_recent},
            )
            summary_text = (
                "[reactive-compact: 发生上下文溢出后，较早的上下文已被裁剪。"
                "请从最新的用户请求和近期证据继续。]"
            )
            compacted = messages[-self.keep_recent :]
            if SystemMessage is not None:
                compacted.insert(0, SystemMessage(content=summary_text))
            else:
                compacted.insert(0, {"role": "system", "content": summary_text})
            if hasattr(request, "model_copy"):
                request = request.model_copy(update={"messages": compacted})
            else:
                request.messages = compacted
            return handler(request)


class LoopGuardMiddleware(AgentMiddleware):
    """Block repeated tool calls and open a simple circuit breaker on failures."""

    def __init__(self, max_repeats: int = 2, max_failures: int = 3) -> None:
        self.max_repeats = max_repeats
        self.max_failures = max_failures
        self.recent_signatures: defaultdict[str, list[str]] = defaultdict(list)
        self.failure_counts: defaultdict[str, int] = defaultdict(int)
        self.breaker_open: defaultdict[str, bool] = defaultdict(bool)

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        thread_id = _thread_id(runtime)
        if not self.breaker_open[thread_id]:
            return None
        emit_event(
            "guard.breaker.opened",
            "LoopGuardMiddleware",
            "Loop/failure circuit breaker halt summary emitted.",
            session_id=thread_id,
            level="warning",
            data={"failure_count": self.failure_counts[thread_id]},
        )
        summary = (
            "## 运行时中止摘要\n"
            "- 循环/失败守护已打开熔断器。\n"
            "- Agent 应停止盲目重试，转而请求澄清或选择一条已验证的路径。\n"
            f"- 连续工具失败次数：{self.failure_counts[thread_id]}\n"
        )
        self.breaker_open[thread_id] = False
        if AIMessage is not None:
            return {"messages": [AIMessage(content=summary)]}
        return {"messages": [{"role": "assistant", "content": summary}]}

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        thread_id = _thread_id(getattr(request, "runtime", None))
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = getattr(request, "tool_name", None) or tool_call.get("name") or "unknown_tool"
        args = getattr(request, "args", None) or tool_call.get("args") or {}
        signature = f"{tool_name}:{str(args)[:500]}"
        recent = self.recent_signatures[thread_id]
        repeats = sum(1 for item in recent[-4:] if item == signature)
        if repeats >= self.max_repeats:
            self.breaker_open[thread_id] = True
            emit_event(
                "guard.loop.blocked",
                "LoopGuardMiddleware",
                "Blocked repeated tool call with the same arguments.",
                session_id=thread_id,
                level="warning",
                data={"tool_name": tool_name, "repeat_count": repeats + 1},
            )
            return _tool_message(
                request,
                f"运行时中止摘要：已拦截对 `{tool_name}` 的重复调用（参数完全相同）。",
            )
        recent.append(signature)
        self.recent_signatures[thread_id] = recent[-10:]

        emit_event(
            "tool.call.started",
            "LoopGuardMiddleware",
            "Tool call started.",
            session_id=thread_id,
            data={"tool_name": tool_name},
        )
        result = handler(request)
        output = _content(result)
        if _looks_failed(output):
            self.failure_counts[thread_id] += 1
            emit_event(
                "tool.call.failed",
                "LoopGuardMiddleware",
                "Tool call output looked failed.",
                session_id=thread_id,
                level="warning",
                data={"tool_name": tool_name, "failure_count": self.failure_counts[thread_id]},
            )
            if self.failure_counts[thread_id] >= self.max_failures:
                self.breaker_open[thread_id] = True
                emit_event(
                    "guard.breaker.opened",
                    "LoopGuardMiddleware",
                    "Tool failure threshold opened circuit breaker.",
                    session_id=thread_id,
                    level="warning",
                    data={"tool_name": tool_name, "failure_count": self.failure_counts[thread_id]},
                )
        else:
            self.failure_counts[thread_id] = 0
            emit_event(
                "tool.call.finished",
                "LoopGuardMiddleware",
                "Tool call finished.",
                session_id=thread_id,
                data={"tool_name": tool_name, "output_chars": len(output)},
            )
        return result


class ContextCompactMiddleware(AgentMiddleware):
    """Structured state-preserving context compression.

    Keeps the most recent N messages intact and replaces older messages with a
    structured YAML state summary extracted by a cheap model call. Timestamps and
    version are generated by the system, never by the model.
    """

    def __init__(self, max_recent: int = 8, trigger_at: int = 16, model: Any = None) -> None:
        self.max_recent = max_recent
        self.trigger_at = trigger_at
        self.model = model

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        if len(messages) <= self.trigger_at:
            return None

        # 1. Split recent (preserved) vs old (to be compressed)
        recent = messages[-self.max_recent :]
        old = messages[: -self.max_recent]

        # 1b. Pull protected clarify_concept_tool/draft_research_plan_tool exchanges out of
        # `old` before they're summarized away -- they're the plan the user is actively
        # confirming, not disposable history. Kept in their original relative order and spliced
        # back in unmodified below; `old` (fed to _extract_state) is left as-is on purpose so the
        # summary can still reference them for context, this is just a belt-and-suspenders
        # duplication, not a correctness requirement.
        protected = _protected_message_indices(messages)
        preserved_old = [messages[index] for index in sorted(protected) if index < len(old)]

        # 2. Build or update structured state from old messages
        new_state = self._extract_state(old)

        # 3. Render state as YAML-like system message
        state_text = self._render_state(new_state)

        emit_event(
            "context.compact.triggered",
            "ContextCompactMiddleware",
            "Compressed older context into structured state.",
            session_id=_thread_id(runtime),
            data={
                "total_messages": len(messages),
                "preserved_recent": len(recent),
                "compressed_old": len(old),
                "preserved_protected": len(preserved_old),
                "state_version": new_state.get("version"),
            },
        )

        if SystemMessage is not None:
            return {"messages": [SystemMessage(content=state_text), *preserved_old, *recent]}
        return {"messages": [{"role": "system", "content": state_text}, *preserved_old, *recent]}

    def _extract_state(self, old_messages: list[Any]) -> dict[str, Any]:
        """Use a cheap model call to extract structured state from old messages."""
        if not self.model:
            return self._fallback_state(old_messages)

        # Build a lightweight context for extraction
        parts: list[str] = []
        for msg in old_messages:
            kind = _message_type(msg).lower()
            text = _content(msg).strip()
            if text:
                parts.append(f"[{kind}] {text[:400]}")
        context = "\n".join(parts)

        prompt = STATE_COMPACT_PROMPT + "\n\n以下是需要提取状态的老对话消息：\n" + context[:3000]
        try:
            response = self.model.invoke([{"role": "user", "content": prompt}])
            text = _content_from_response(response)
            extracted = _json_from_text(text)
        except Exception:
            extracted = {}

        return self._normalize_state(extracted, old_messages)

    def _normalize_state(self, extracted: dict[str, Any], old_messages: list[Any]) -> dict[str, Any]:
        """Sanitize AI-extracted state and add system-managed fields."""
        now = datetime.now(timezone.utc).isoformat()

        def _as_list(value: Any) -> list[Any]:
            return value if isinstance(value, list) else []

        # Try to infer plan step from messages if AI didn't provide it
        current_step = extracted.get("current_step")
        if current_step is None:
            current_step = self._infer_current_step(old_messages)

        def _with_timestamp(items: Any) -> list[dict[str, Any]]:
            return [{"timestamp": now, **item} for item in _as_list(items) if isinstance(item, dict)]

        return {
            "version": 1,
            "generated_at": now,
            "current_task": str(extracted.get("current_task") or "").strip() or None,
            "current_plan": [str(s) for s in _as_list(extracted.get("current_plan")) if s],
            "current_step": current_step,
            "key_decisions": _with_timestamp(extracted.get("key_decisions", [])),
            "verified_facts": _with_timestamp(extracted.get("verified_facts", [])),
            "errors_fixed": _with_timestamp(extracted.get("errors_fixed", [])),
            "open_questions": [str(q) for q in _as_list(extracted.get("open_questions")) if q],
        }

    def _infer_current_step(self, messages: list[Any]) -> int | None:
        """Best-effort: look for plan step markers in tool results or AI replies."""
        for msg in reversed(messages):
            text = _content(msg)
            if "[进行中]" in text or "in_progress" in text.lower():
                # Try to find step number
                match = re.search(r"(?:步骤|step)[\s:]?(\d+)", text, re.I)
                if match:
                    return int(match.group(1))
            if "[completed]" in text or "已完成" in text:
                match = re.search(r"(?:步骤|step)[\s:]?(\d+)", text, re.I)
                if match:
                    return int(match.group(1)) + 1
        return None

    def _fallback_state(self, old_messages: list[Any]) -> dict[str, Any]:
        """When no model is available, create a minimal state."""
        now = datetime.now(timezone.utc).isoformat()
        # Look for user messages as task hint
        user_texts = [_content(m) for m in old_messages if _message_type(m).lower() in {"human", "user"}]
        task = user_texts[-1][:120].strip() if user_texts else None
        return {
            "version": 1,
            "generated_at": now,
            "current_task": task,
            "current_plan": [],
            "current_step": None,
            "key_decisions": [],
            "verified_facts": [],
            "errors_fixed": [],
            "open_questions": [],
        }

    def _render_state(self, state: dict[str, Any]) -> str:
        """Render state dict as a YAML-like system message."""
        lines = [
            "--- 会话状态摘要 ---",
            f"version: {state.get('version', 1)}",
            f"generated_at: {state.get('generated_at', '')}",
            "",
            f"current_task: {state.get('current_task') or 'null'}",
        ]

        plan = state.get("current_plan", [])
        step = state.get("current_step")
        if plan:
            lines.append("current_plan:")
            for i, p in enumerate(plan, 1):
                marker = ""
                if step is not None:
                    if i < step:
                        marker = " [completed]"
                    elif i == step:
                        marker = " [in_progress]"
                lines.append(f"  - {p}{marker}")
            lines.append(f"current_step: {step}")
        else:
            lines.append("current_plan: []")
            lines.append("current_step: null")

        for section in ["key_decisions", "verified_facts", "errors_fixed"]:
            items = state.get(section, [])
            lines.append("")
            lines.append(f"{section}:")
            if items:
                for item in items:
                    fact = item.get("fact") or item.get("error", "")
                    src = item.get("source", "unknown")
                    ts = item.get("timestamp", "")[:19]
                    lines.append(f"  - [{src}] {fact} ({ts})")
            else:
                lines.append("  (none)")

        questions = state.get("open_questions", [])
        lines.append("")
        lines.append("open_questions:")
        if questions:
            for q in questions:
                lines.append(f"  - {q}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("note: 本摘要仅供参考，最近完整消息中的用户明确指令优先级更高。")
        return "\n".join(lines)


def _content_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "".join(pieces)
    return str(content)


def _thread_id(runtime: Any) -> str:
    if runtime is None:
        return current_session_id()
    config = getattr(runtime, "config", {}) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    return str(configurable.get("thread_id") or current_session_id())


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _simple_terms(text: str) -> list[str]:
    return [
        part
        for part in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", text.lower())
        if len(part) >= 2
    ]


def _looks_failed(output: str) -> bool:
    lowered = output.strip().lower()
    if not lowered:
        return False
    failure_prefixes = (
        "denied:",
        "not found:",
        "error:",
        "failed:",
        "runtime halt",
        "运行时中止摘要",
    )
    if lowered.startswith(failure_prefixes):
        return True
    try:
        data = json.loads(output)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    status = str(data.get("status") or "").lower()
    if status in {"error", "failed", "denied"}:
        return True
    return any(bool(data.get(key)) for key in ["error", "fetch_error"]) and not data.get("results")


def _tool_message(request: Any, content: str) -> Any:
    tool_call = getattr(request, "tool_call", None) or {}
    tool_call_id = tool_call.get("id") or getattr(request, "tool_call_id", None) or "blocked-tool-call"
    if ToolMessage is not None:
        return ToolMessage(content=content, tool_call_id=tool_call_id)
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}
