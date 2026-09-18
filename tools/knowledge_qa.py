"""Local Markdown knowledge-base QA tools."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event

from ._registry import (
    KNOWLEDGE_BASE,
    conversation_channel_id,
    current_thread_id,
    discover_topic_dirs,
    ensure_knowledge_base,
    safe_knowledge_path,
    validate_topic_shape,
)
from .lightrag_qa import build_lightrag_index, lightrag_index_state, query_lightrag, query_lightrag_stream


MAX_CONTEXT_CHARS = 7000
MAX_SNIPPETS = 6
_ACTIVE_KB_BY_THREAD: dict[str, str] = {}


class KnowledgeQAModelUnavailableError(RuntimeError):
    pass


def get_active_knowledge_base_topic() -> str | None:
    """Return the active knowledge-base topic for the current thread, if any."""
    return _ACTIVE_KB_BY_THREAD.get(current_thread_id())


def _channel_topic_binding() -> str:
    """Persistent per-channel default topic (docs/VISION.md 第3节), if this thread belongs to
    a registered channel and that channel has been explicitly bound via
    tools/knowledge_manager.py::bind_channel_to_topic_tool. Any failure (bare CLI/dev thread
    with no channel row, channel service unavailable, etc.) resolves to "" rather than raising --
    this is a best-effort fallback tier, never a hard requirement."""
    try:
        from ..backend.services import channel_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import channel_service
    try:
        return str(channel_service.get_topic_binding(conversation_channel_id()) or "").strip()
    except Exception:
        return ""


def _channel_scope_binding() -> dict[str, str] | None:
    try:
        from ..backend.services import channel_service
    except ImportError:  # pragma: no cover - top-level package fallback
        try:
            from backend.services import channel_service
        except ImportError:
            return None
    try:
        binding = channel_service.get_scope_binding(conversation_channel_id())
    except ValueError:
        return None
    if not binding or binding.get("legacy"):
        return None
    return {
        "scope_type": str(binding.get("scope_type") or ""),
        "scope_id": str(binding.get("scope_id") or ""),
    }


def _query_bound_scope(question: str, binding: dict[str, str], *, mode: str) -> str:
    try:
        from ..backend.services import knowledge_scope_service
    except ImportError:  # pragma: no cover - top-level package fallback
        from backend.services import knowledge_scope_service
    result = knowledge_scope_service.query_scope(
        binding["scope_type"],
        binding["scope_id"],
        question,
        mode=mode,
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def list_knowledge_bases() -> str:
    """列出本地 knowledge_base 下可用于问答的知识库 topic，包含扁平 topic 和 theme/subtopic 两种形态。"""
    ensure_knowledge_base()
    topics: list[dict[str, Any]] = []
    for topic, path in sorted(
        discover_topic_dirs(KNOWLEDGE_BASE), key=lambda item: item[1].stat().st_mtime, reverse=True
    ):
        files = sorted(path.glob("*.md"))
        entry = _entry_document(files)
        topics.append(
            {
                "topic": topic,
                "file_count": len(files),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "entry_document": entry.name if entry else None,
            }
        )
    active = _ACTIVE_KB_BY_THREAD.get(current_thread_id())
    return json.dumps(
        {
            "status": "ok",
            "active_topic": active,
            "topics": topics,
        },
        ensure_ascii=False,
    )


@tool
def set_active_knowledge_base(topic: str) -> str:
    """为当前会话设置默认知识库 topic，用于后续本地知识库问答。"""
    safe_topic = _validate_topic(topic)
    _ACTIVE_KB_BY_THREAD[current_thread_id()] = safe_topic
    emit_event(
        "knowledge_qa.active_topic_set",
        "tools",
        "Set active knowledge-base topic.",
        data={"thread_id": current_thread_id(), "topic": safe_topic},
    )
    return json.dumps(
        {
            "status": "active_topic_set",
            "active_topic": safe_topic,
            "message": f"已将当前知识库设置为 {safe_topic}。",
        },
        ensure_ascii=False,
    )


@tool
def index_knowledge_base_lightrag(topic: str, force: bool = False) -> str:
    """Build or refresh the local LightRAG index for one knowledge_base topic."""
    result = build_lightrag_index(topic, force=force)
    return json.dumps(result, ensure_ascii=False)


@tool
def query_knowledge_base(
    question: str,
    topic: str = "",
    backend: str = "auto",
    mode: str = "mix",
    include_retrieval_trace: bool = False,
) -> str:
    """基于本地 knowledge_base Markdown 文档进行问答。

    当用户明确说“知识库问答”“查知识库”“基于当前知识库”“在知识库里问”，
    或明确要求调用 query_knowledge_base 时，主 Agent 应使用本工具，而不是自行回答。

    只读取 knowledge_base/<topic> 下的 Markdown，不联网，不读取项目源码。
    如果 topic 为空，则使用当前会话的 active knowledge base。
    """
    question_text = _clean_question(str(question or "").strip())
    if not question_text:
        raise ValueError("question must not be empty")

    active_topic = _ACTIVE_KB_BY_THREAD.get(current_thread_id(), "")
    bound_topic = _channel_topic_binding()
    bound_scope = _channel_scope_binding()
    if not str(topic or "").strip() and not active_topic and bound_scope:
        return _query_bound_scope(question_text, bound_scope, mode=str(mode or "mix").strip().lower())
    topic_name = str(topic or "").strip() or active_topic or bound_topic
    if not topic_name:
        return _need_topic_response()
    try:
        safe_topic = _validate_topic(topic_name)
    except ValueError:
        fallback_topic = active_topic or bound_topic
        if fallback_topic and str(topic or "").strip():
            safe_topic = _validate_topic(fallback_topic)
        else:
            subtopics = _theme_subtopics(topic_name)
            if subtopics:
                return _query_theme_aggregate(
                    question_text,
                    topic_name,
                    subtopics,
                    _validate_backend(backend),
                    str(mode or "mix").strip().lower(),
                )
            return json.dumps(
                {
                    "status": "topic_not_found",
                    "topic": topic_name,
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "gaps": [f"知识库 topic 不存在：{topic_name}"],
                },
                ensure_ascii=False,
            )

    backend_name = _validate_backend(backend)
    mode_name = str(mode or "mix").strip().lower()
    fallback_reason = ""
    if backend_name in {"auto", "lightrag"}:
        state = lightrag_index_state(safe_topic)
        if state.available:
            try:
                output = query_lightrag(
                    question_text,
                    safe_topic,
                    mode=mode_name,
                    include_retrieval_trace=include_retrieval_trace,
                )
                if output.get("status") != "answered" or not str(output.get("answer") or "").strip():
                    if backend_name == "lightrag":
                        return json.dumps(
                            _with_backend_diagnostics(output, backend_name, "lightrag"),
                            ensure_ascii=False,
                        )
                    raise RuntimeError("LightRAG query returned no answer.")
                output = _with_backend_diagnostics(output, backend_name, "lightrag")
                emit_event(
                    "knowledge_qa.answered",
                    "tools",
                    "Answered from LightRAG knowledge base.",
                    data={
                        "topic": safe_topic,
                        "backend": "lightrag",
                        "mode": mode_name,
                        "citation_count": len(output.get("citations", [])),
                    },
                )
                return json.dumps(output, ensure_ascii=False)
            except Exception as exc:
                if backend_name == "lightrag":
                    return json.dumps(
                        _with_backend_diagnostics({
                            "status": "error",
                            "topic": safe_topic,
                            "backend": "lightrag",
                            "mode": mode_name,
                            "answer": "",
                            "citations": [],
                            "confidence": "low",
                            "gaps": [f"LightRAG 查询失败：{str(exc)[:300]}"],
                        }, backend_name, "lightrag"),
                        ensure_ascii=False,
                    )
                fallback_reason = f"LightRAG 查询失败：{str(exc)[:300]}"
                emit_event(
                    "knowledge_qa.lightrag_fallback",
                    "tools",
                    "LightRAG query failed; falling back to local Markdown QA.",
                    data={"topic": safe_topic, "reason": str(exc)[:200]},
                )
        elif backend_name == "lightrag":
            return json.dumps(
                _with_backend_diagnostics({
                    "status": "index_missing",
                    "topic": safe_topic,
                    "backend": "lightrag",
                    "mode": mode_name,
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "gaps": [f"LightRAG 索引不可用：{state.reason}"],
                }, backend_name, "lightrag"),
                ensure_ascii=False,
            )
        else:
            fallback_reason = f"LightRAG 索引不可用：{state.reason}"

    docs = _load_topic_documents(safe_topic)
    snippets = _retrieve_snippets(question_text, docs)
    if not snippets:
        output = {
            "status": "no_match",
            "topic": safe_topic,
            "backend": "local",
            "mode": "markdown",
            "answer": "当前知识库中没有找到足够相关的内容，无法可靠回答这个问题。",
            "citations": [],
            "confidence": "low",
            "gaps": ["未召回到相关 Markdown 片段。"],
        }
        return json.dumps(
            _with_backend_diagnostics(output, backend_name, "local", fallback_reason=fallback_reason),
            ensure_ascii=False,
        )

    answer = _generate_answer(question_text, safe_topic, snippets)
    output = {
        "status": "answered",
        "topic": safe_topic,
        "backend": "local",
        "mode": "markdown",
        "answer": answer,
        "citations": [
            {
                "title": item["title"],
                "path": item["path"],
                "snippet": item["snippet"][:500],
            }
            for item in snippets[:MAX_SNIPPETS]
        ],
        "confidence": _confidence(snippets),
        "gaps": [] if len(snippets) >= 2 else ["相关片段较少，答案置信度有限。"],
    }
    if include_retrieval_trace:
        output["retrieval_trace"] = [
            {
                "rank": index,
                "chunk_id": "",
                "reference_id": "",
                "title": item["title"],
                "path": item["path"],
                "content": item["snippet"],
                "score": item.get("score"),
                "score_type": "local_lexical",
            }
            for index, item in enumerate(snippets, start=1)
        ]
    emit_event(
        "knowledge_qa.answered",
        "tools",
        "Answered from local knowledge base.",
        data={"topic": safe_topic, "snippet_count": len(snippets), "confidence": output["confidence"]},
    )
    return json.dumps(
        _with_backend_diagnostics(output, backend_name, "local", fallback_reason=fallback_reason),
        ensure_ascii=False,
    )


def _validate_topic(topic: str) -> str:
    safe = str(topic or "").strip().replace("\\", "/").strip("/")
    if not safe or ".." in safe:
        raise ValueError("topic must be a single knowledge_base directory name")
    segments = validate_topic_shape(safe)
    safe = "/".join(segments)
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"knowledge base topic not found: {safe}")
    if not list(path.glob("*.md")):
        raise ValueError(f"knowledge base topic has no Markdown documents: {safe}")
    return safe


def _theme_subtopics(theme: str) -> list[str]:
    """If `theme` is a bare (single-segment) theme container -- has no Markdown docs of its own
    but has subdirectories that do -- return its "theme/subtopic" topic strings; otherwise [].
    Same structural rule as tools/knowledge_manager.py::_theme_subtopic_dirs /
    tools/_registry.py::discover_topic_dirs (docs/VISION.md 第1节). Duplicated locally rather than
    imported from knowledge_manager, which itself imports from this module -- importing back
    would be circular."""
    safe = str(theme or "").strip().replace("\\", "/").strip("/")
    if not safe or "/" in safe or ".." in safe:
        return []
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        return []
    subtopic_dirs = sorted(sub for sub in path.iterdir() if sub.is_dir() and list(sub.glob("*.md")))
    return [f"{safe}/{sub.name}" for sub in subtopic_dirs]


def _query_theme_aggregate(
    question: str, theme: str, subtopics: list[str], backend: str, mode: str
) -> str:
    """Channel bound to a bare theme: query every subtopic under it (LightRAG where available,
    local Markdown otherwise) and merge into one answer, instead of asking the user to pick a
    subtopic. Mirrors backend/services/push_service.py::_rotation_pool's "aggregate across
    subtopics" convention, applied to retrieval instead of rotation."""
    lightrag_snippets: list[dict[str, Any]] = []
    local_docs: list[dict[str, str]] = []
    contributing_subtopics: set[str] = set()
    skipped_subtopics: list[str] = []
    used_lightrag_any = False
    used_local_any = False
    fallback_reasons: list[str] = []

    for sub in subtopics:
        used_lightrag = False
        if backend in {"auto", "lightrag"}:
            state = lightrag_index_state(sub)
            if state.available:
                try:
                    output = query_lightrag(question, sub, mode=mode)
                except Exception:
                    output = None
                if output and output.get("status") == "answered" and str(output.get("answer") or "").strip():
                    for item in output.get("citations", []):
                        lightrag_snippets.append(
                            {
                                "title": item.get("title", ""),
                                "path": item.get("path", ""),
                                "snippet": item.get("snippet", ""),
                                "score": 1000.0,
                            }
                        )
                        contributing_subtopics.add(sub)
                    used_lightrag = True
                    used_lightrag_any = True
            elif backend == "lightrag":
                skipped_subtopics.append(sub)
        if not used_lightrag and backend != "lightrag":
            local_docs.extend(_load_topic_documents(sub))
            used_local_any = True
            if backend == "auto":
                fallback_reasons.append(f"{sub} 的 LightRAG 索引不可用或未返回答案")

    local_snippets = _retrieve_snippets(question, local_docs) if local_docs else []
    for item in local_snippets:
        subtopic = "/".join(str(item.get("path", "")).split("/")[-3:-1])
        if subtopic:
            contributing_subtopics.add(subtopic)

    snippets = sorted(lightrag_snippets + local_snippets, key=lambda item: item.get("score", 0), reverse=True)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in snippets:
        key = (item["path"], item["snippet"][:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= MAX_SNIPPETS:
            break

    gaps: list[str] = []
    if skipped_subtopics:
        gaps.append(f"以下子主题因 LightRAG 索引不可用被跳过：{', '.join(skipped_subtopics)}")

    if not deduped:
        output = {
            "status": "no_match",
            "topic": theme,
            "backend": backend,
            "mode": mode,
            "answer": "当前大主题下没有找到足够相关的内容，无法可靠回答这个问题。",
            "citations": [],
            "confidence": "low",
            "subtopics_used": [],
            "gaps": gaps or [f"{theme} 下 {len(subtopics)} 个子主题均未召回相关内容。"],
        }
        actual_backend = "mixed" if used_lightrag_any and used_local_any else "lightrag" if used_lightrag_any else "local"
        return json.dumps(
            _with_backend_diagnostics(
                output,
                backend,
                actual_backend,
                fallback_reason="；".join(fallback_reasons[:5]),
            ),
            ensure_ascii=False,
        )

    answer = _generate_answer(question, theme, deduped)
    output = {
        "status": "answered",
        "topic": theme,
        "backend": backend,
        "mode": mode,
        "answer": answer,
        "citations": [
            {"title": item["title"], "path": item["path"], "snippet": item["snippet"][:500]}
            for item in deduped
        ],
        "confidence": _confidence(deduped),
        "subtopics_used": sorted(contributing_subtopics),
        "gaps": gaps if len(deduped) >= 2 else gaps + ["相关片段较少，答案置信度有限。"],
    }
    actual_backend = "mixed" if used_lightrag_any and used_local_any else "lightrag" if used_lightrag_any else "local"
    output = _with_backend_diagnostics(
        output,
        backend,
        actual_backend,
        fallback_reason="；".join(fallback_reasons[:5]),
    )
    emit_event(
        "knowledge_qa.answered",
        "tools",
        "Answered from theme-aggregated knowledge base.",
        data={
            "theme": theme,
            "subtopic_count": len(subtopics),
            "subtopics_used": output["subtopics_used"],
            "snippet_count": len(deduped),
            "confidence": output["confidence"],
        },
    )
    return json.dumps(output, ensure_ascii=False)


def _validate_backend(backend: str) -> str:
    name = str(backend or "auto").strip().lower()
    if name not in {"auto", "local", "lightrag"}:
        raise ValueError("backend must be one of: auto, local, lightrag")
    return name


def _with_backend_diagnostics(
    output: dict[str, Any],
    requested_backend: str,
    actual_backend: str,
    *,
    fallback_reason: str = "",
) -> dict[str, Any]:
    result = dict(output)
    result["backend"] = actual_backend
    result["requested_backend"] = requested_backend
    result["actual_backend"] = actual_backend
    result["fallback_occurred"] = bool(requested_backend == "auto" and actual_backend != "lightrag")
    result["fallback_reason"] = fallback_reason if result["fallback_occurred"] else ""
    return result


def _need_topic_response() -> str:
    bases = json.loads(list_knowledge_bases.invoke({}))
    return json.dumps(
        {
            "status": "need_topic",
            "available_topics": [item["topic"] for item in bases.get("topics", [])],
            "message": "需要先选择知识库 topic。请调用 set_active_knowledge_base(topic) 或在 query_knowledge_base 中传入 topic。",
        },
        ensure_ascii=False,
    )


def _entry_document(files: list[Path]) -> Path | None:
    for file in files:
        if file.name.startswith("00_") or "overview" in file.stem.lower() or "总览" in file.stem:
            return file
    return files[0] if files else None


def _load_topic_documents(topic: str) -> list[dict[str, str]]:
    topic_path = safe_knowledge_path(topic)
    docs: list[dict[str, str]] = []
    for path in sorted(topic_path.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(raw)
        title = frontmatter.get("title") or path.stem
        docs.append(
            {
                "title": title,
                "path": str(path.relative_to(KNOWLEDGE_BASE.parent)),
                "body": body.strip(),
            }
        )
    return docs


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    frontmatter: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, parts[2]


def _retrieve_snippets(question: str, docs: list[dict[str, str]]) -> list[dict[str, Any]]:
    terms = _query_terms(question)
    candidates: list[dict[str, Any]] = []
    for doc in docs:
        for chunk in _chunk_markdown(doc["body"]):
            score = _score_chunk(question, terms, doc["title"], doc["path"], chunk)
            if score <= 0:
                continue
            candidates.append(
                {
                    "title": doc["title"],
                    "path": doc["path"],
                    "snippet": chunk,
                    "score": score,
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    total_chars = 0
    for item in candidates:
        key = (item["path"], item["snippet"][:80])
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        total_chars += len(item["snippet"])
        if len(selected) >= MAX_SNIPPETS or total_chars >= MAX_CONTEXT_CHARS:
            break
    return selected


def _chunk_markdown(body: str) -> list[str]:
    sections = re.split(r"\n(?=#{1,4}\s+)", body)
    chunks: list[str] = []
    for section in sections:
        text = section.strip()
        if not text:
            continue
        if len(text) <= 1200:
            chunks.append(text)
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        buf = ""
        for paragraph in paragraphs:
            if len(buf) + len(paragraph) > 1000 and buf:
                chunks.append(buf.strip())
                buf = paragraph
            else:
                buf = f"{buf}\n\n{paragraph}".strip()
        if buf:
            chunks.append(buf.strip())
    return chunks


def _query_terms(question: str) -> list[str]:
    lowered = question.lower()
    terms = re.findall(r"[a-zA-Z0-9_\-]{2,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    for token in chinese:
        if len(token) <= 6:
            terms.append(token)
        terms.extend(token[i : i + 2] for i in range(max(0, len(token) - 1)))
    stop = {"什么", "如何", "怎么", "一下", "这个", "哪些", "为什么", "以及", "进行"}
    cleaned = [term for term in terms if term and term not in stop]
    return list(dict.fromkeys(cleaned))[:40]


def _clean_question(question: str) -> str:
    text = question.strip()
    for marker in ["新问题：", "新问题:", "回答：", "回答:", "问题：", "问题:"]:
        if marker in text:
            text = text.rsplit(marker, 1)[-1].strip()
    cleanup_patterns = [
        r"这是知识库问答任务[。.!！]?",
        r"请使用\s*query_knowledge_base\s*",
        r"使用\s*query_knowledge_base\s*",
        r"基于当前知识库",
        r"请给出引用[。.!！]?",
        r"不要联网[，,、]?",
        r"不要读取源码[。.!！]?",
        r"不要读取项目源码[。.!！]?",
    ]
    for pattern in cleanup_patterns:
        text = re.sub(pattern, "", text, flags=re.I).strip()
    return text.strip(" ：:。.!！")


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in (history or [])[-6:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content[:1200]})
    return normalized


def _contextual_retrieval_question(question: str, history: list[dict[str, str]]) -> str:
    prior_questions = [item["content"] for item in history if item["role"] == "user"][-3:]
    return "\n".join([question, *prior_questions])


def _score_chunk(question: str, terms: list[str], title: str, path: str, chunk: str) -> float:
    haystack = f"{title}\n{path}\n{chunk}".lower()
    score = 0.0
    if question.lower() in haystack:
        score += 8
    for term in terms:
        if term in title.lower():
            score += 5
        if term in path.lower():
            score += 3
        hits = haystack.count(term.lower())
        score += min(hits, 5)
    return score


def _generate_answer(
    question: str,
    topic: str,
    snippets: list[dict[str, Any]],
    history_messages: list[dict[str, str]] | None = None,
) -> str:
    context = "\n\n".join(
        f"[{index}] {item['title']} ({item['path']})\n{item['snippet']}"
        for index, item in enumerate(snippets[:MAX_SNIPPETS], start=1)
    )
    prompt = (
        "你是本地 Markdown 知识库问答助手。只能根据给定知识库片段回答；"
        "如果片段不足以回答，要明确说明不足。回答使用中文，并在关键结论后标注引用编号。\n\n"
        f"当前知识库：{topic}\n"
        f"用户问题：{question}\n\n"
        f"知识库片段：\n{context[:MAX_CONTEXT_CHARS]}\n\n"
        "请给出简洁但有依据的回答。"
    )
    try:
        model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            api_key=os.getenv("OPENAI_API_KEY") or None,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            temperature=0,
            streaming=False,
        )
        response = model.invoke([*(history_messages or []), {"role": "user", "content": prompt}])
        content = getattr(response, "content", "")
        if content:
            return str(content).strip()
        raise RuntimeError("model returned an empty response")
    except Exception as exc:
        emit_event(
            "knowledge_qa.answer_unavailable",
            "tools",
            "Knowledge QA model call failed; no fallback answer was produced.",
            level="error",
            data={"reason": str(exc)[:200]},
        )
        raise KnowledgeQAModelUnavailableError("模型不可用，无法生成知识库回答。") from exc


def _generate_answer_stream(
    question: str,
    topic: str,
    snippets: list[dict[str, Any]],
    on_token: Any,
    history_messages: list[dict[str, str]] | None = None,
) -> None:
    context = "\n\n".join(
        f"[{index}] {item['title']} ({item['path']})\n{item['snippet']}"
        for index, item in enumerate(snippets[:MAX_SNIPPETS], start=1)
    )
    prompt = (
        "你是本地 Markdown 知识库问答助手。只能根据给定知识库片段回答；"
        "如果片段不足以回答，要明确说明不足。回答使用中文，并在关键结论后标注引用编号。\n\n"
        f"当前知识库：{topic}\n"
        f"用户问题：{question}\n\n"
        f"知识库片段：\n{context[:MAX_CONTEXT_CHARS]}\n\n"
        "请给出简洁但有依据的回答。"
    )
    try:
        model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            api_key=os.getenv("OPENAI_API_KEY") or None,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            temperature=0,
            streaming=True,
        )
        produced_content = False
        for chunk in model.stream([*(history_messages or []), {"role": "user", "content": prompt}]):
            content = getattr(chunk, "content", "")
            if content:
                produced_content = True
                on_token(str(content))
        if not produced_content:
            raise RuntimeError("model returned an empty response")
    except Exception as exc:
        emit_event(
            "knowledge_qa.answer_unavailable",
            "tools",
            "Knowledge QA model call failed; no fallback answer was produced.",
            level="error",
            data={"reason": str(exc)[:200]},
        )
        raise KnowledgeQAModelUnavailableError("模型不可用，无法生成知识库回答。") from exc


def stream_query_knowledge_base(
    topic: str,
    *,
    question: str,
    history: list[dict[str, Any]] | None = None,
    backend: str,
    mode: str,
    on_token: Any,
) -> dict[str, Any]:
    """Run the selected QA backend and forward model-level answer tokens."""
    safe_topic = _validate_topic(topic)
    question_text = _clean_question(str(question or ""))
    if not question_text:
        raise ValueError("question must not be empty")
    history_messages = _normalize_history(history)
    backend_name = _validate_backend(backend)
    mode_name = str(mode or "mix").strip().lower()
    fallback_reason = ""
    if backend_name in {"auto", "lightrag"}:
        state = lightrag_index_state(safe_topic)
        if state.available:
            try:
                return _with_backend_diagnostics(
                    _query_lightrag_stream_with_timeout(
                        question_text,
                        safe_topic,
                        mode=mode_name,
                        history_messages=history_messages,
                        on_token=on_token,
                    ),
                    backend_name,
                    "lightrag",
                )
            except Exception as exc:
                if backend_name == "lightrag":
                    raise
                fallback_reason = f"LightRAG 查询失败：{str(exc)[:300]}"
        elif backend_name == "lightrag":
            raise RuntimeError(f"LightRAG index is not available for topic '{safe_topic}': {state.reason}")
        else:
            fallback_reason = f"LightRAG 索引不可用：{state.reason}"

    docs = _load_topic_documents(safe_topic)
    snippets = _retrieve_snippets(
        _contextual_retrieval_question(question_text, history_messages),
        docs,
    )
    if not snippets:
        answer = "当前知识库中没有找到足够相关的内容，无法可靠回答这个问题。"
        on_token(answer)
        return _with_backend_diagnostics({
            "status": "no_match", "topic": safe_topic, "backend": "local", "mode": "markdown",
            "answer": answer, "citations": [], "confidence": "low", "gaps": ["未召回到相关 Markdown 片段。"],
        }, backend_name, "local", fallback_reason=fallback_reason)
    pieces: list[str] = []
    _generate_answer_stream(
        question_text,
        safe_topic,
        snippets,
        lambda text: (pieces.append(text), on_token(text)),
        history_messages=history_messages,
    )
    answer = "".join(pieces).strip()
    return _with_backend_diagnostics({
        "status": "answered", "topic": safe_topic, "backend": "local", "mode": "markdown",
        "answer": answer,
        "citations": [{"title": item["title"], "path": item["path"], "snippet": item["snippet"][:500]} for item in snippets[:MAX_SNIPPETS]],
        "confidence": _confidence(snippets), "gaps": [] if len(snippets) >= 2 else ["相关片段较少，答案置信度有限。"],
    }, backend_name, "local", fallback_reason=fallback_reason)


def _query_lightrag_stream_with_timeout(
    question: str,
    topic: str,
    *,
    mode: str,
    history_messages: list[dict[str, str]],
    on_token: Any,
) -> dict[str, Any]:
    """Keep auto QA responsive when LightRAG never produces its first token."""
    updates: queue.Queue[tuple[str, Any]] = queue.Queue()

    def run() -> None:
        try:
            result = query_lightrag_stream(
                question,
                topic,
                mode=mode,
                conversation_history=history_messages,
                on_token=lambda text: updates.put(("delta", text)),
            )
            updates.put(("done", result))
        except Exception as exc:
            updates.put(("error", exc))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    started_at = time.monotonic()
    first_token_timeout = max(
        0.1,
        float(os.getenv("KNOWLEDGE_QA_LIGHTRAG_FIRST_TOKEN_TIMEOUT_SECONDS", "30")),
    )
    saw_token = False

    while worker.is_alive() or not updates.empty():
        elapsed = time.monotonic() - started_at
        if not saw_token and elapsed >= first_token_timeout:
            raise TimeoutError(f"LightRAG first token timeout after {first_token_timeout:g}s")
        try:
            wait_for = 0.1 if saw_token else min(0.1, first_token_timeout - elapsed)
            event, payload = updates.get(timeout=wait_for)
        except queue.Empty:
            continue
        if event == "delta":
            text = str(payload or "")
            if text:
                saw_token = True
                on_token(text)
        elif event == "done":
            return payload
        else:
            raise payload

    raise RuntimeError("LightRAG stream ended without a result")


def _confidence(snippets: list[dict[str, Any]]) -> str:
    if len(snippets) >= 3 and snippets[0].get("score", 0) >= 8:
        return "high"
    if len(snippets) >= 2:
        return "medium"
    return "low"
