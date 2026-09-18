"""Read-only knowledge-base management tools."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ._registry import (
    KNOWLEDGE_BASE,
    LIGHTRAG_WORKSPACES,
    conversation_channel_id,
    discover_topic_dirs,
    ensure_knowledge_base,
    safe_knowledge_path,
    validate_topic_shape,
)
from .knowledge_qa import get_active_knowledge_base_topic
from .lightrag_qa import _build_metadata as _build_lightrag_metadata
from .lightrag_qa import lightrag_index_state


KB_METADATA_FILE = "kb_metadata.json"


def _channel_service():
    # Lazy import, matching tools/research_jobs.py's convention for backend.services.* --
    # channel_service.py has no local imports of its own so it isn't actually part of the
    # tools/__init__.py -> backend.services.research_service -> workers -> tools load cycle,
    # but importing it lazily here keeps this module consistent with that project-wide pattern.
    try:
        from ..backend.services import channel_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import channel_service
    return channel_service


def _knowledge_tree_service():
    try:
        from ..backend.services import knowledge_tree_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import knowledge_tree_service
    return knowledge_tree_service


def _knowledge_scope_service():
    try:
        from ..backend.services import knowledge_scope_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import knowledge_scope_service
    return knowledge_scope_service


def _research_service():
    try:
        from ..backend.services import research_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import research_service
    return research_service


@tool
def list_personal_knowledge_tree_tool(include_legacy: bool = True) -> str:
    """列出新的个人知识树：KnowledgeBase、单层 Theme、ResearchItem，以及可选的旧测试 topic。"""
    return json.dumps(_knowledge_tree_service().list_tree(include_legacy=include_legacy), ensure_ascii=False)


@tool
def preview_knowledge_tree_change_tool(action: str, payload_json: str) -> str:
    """为知识树结构变更生成持久化预览，但不执行变更。

    action 支持 create_knowledge_base、create_theme、rename_node、move_research、
    update_research_status、trash_node、trash_legacy、restore_node。payload_json 必须是对应
    动作的 JSON 对象。把返回的 preview_id 和 changes 展示给用户，只有用户确认后才可调用
    confirm_knowledge_tree_change_tool。
    """
    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload_json must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("payload_json must decode to an object")
    preview = _knowledge_tree_service().create_preview(action, payload)
    return json.dumps(preview, ensure_ascii=False)


@tool
def confirm_knowledge_tree_change_tool(preview_id: str) -> str:
    """确认并执行一个已展示给用户的知识树变更预览；本工具必须经过人工审批。"""
    return json.dumps(_knowledge_tree_service().confirm_preview(preview_id), ensure_ascii=False)


@tool
def query_personal_knowledge_scope_tool(scope_type: str, scope_id: str, question: str, include_history: bool = False) -> str:
    """查询一个稳定 KnowledgeScope。scope_type 为 knowledge_base、theme 或 research。"""
    result = _knowledge_scope_service().query_scope(
        scope_type,
        scope_id,
        question,
        include_history=include_history,
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def bind_channel_to_scope_tool(scope_type: str, scope_id: str) -> str:
    """把当前频道持久绑定到稳定 KnowledgeBase 或 Theme ID，供默认问答和主动推送使用。"""
    if scope_type not in {"knowledge_base", "theme"}:
        raise ValueError("channel scope_type must be knowledge_base or theme")
    _knowledge_scope_service().resolve_scope(scope_type, scope_id)
    channel_id = conversation_channel_id()
    binding = _channel_service().bind_scope(channel_id, scope_type, scope_id)
    return json.dumps({"status": "bound", "channel_id": channel_id, "scope_binding": binding}, ensure_ascii=False)


@tool
def preview_research_archive_tool(
    job_id: str,
    knowledge_base_id: str = "",
    theme_id: str = "",
    new_knowledge_base_title: str = "",
    new_theme_title: str = "",
    title: str = "",
    tags_csv: str = "",
) -> str:
    """为已完成调研生成三层知识体系归档预览，不执行落库。

    目标可留空让系统推荐；也可指定已有 knowledge_base_id/theme_id，或传
    new_knowledge_base_title/new_theme_title 新建位置。将预览完整展示给用户后，才可调用
    confirm_research_archive_tool。
    """
    payload: dict[str, Any] = {}
    if knowledge_base_id:
        payload["knowledge_base_id"] = knowledge_base_id
    if theme_id:
        payload["theme_id"] = theme_id
    if new_knowledge_base_title:
        payload["new_knowledge_base"] = {"title": new_knowledge_base_title}
    if new_theme_title:
        payload["new_theme"] = {"title": new_theme_title}
    if title:
        payload["title"] = title
    if tags_csv:
        payload["tags"] = [item.strip() for item in re.split(r"[,，]", tags_csv) if item.strip()]
    return json.dumps(_research_service().preview_research_archive(job_id, payload), ensure_ascii=False)


@tool
def confirm_research_archive_tool(job_id: str, preview_id: str) -> str:
    """确认调研归档预览并原子写入 ResearchItem、manifest 与两级总览；必须经过人工审批。"""
    return json.dumps(_research_service().confirm_research_archive(job_id, preview_id), ensure_ascii=False)


@tool
def list_knowledge_base_status() -> str:
    """列出本地知识库及其 LightRAG 索引、向量化和图谱化状态，包含扁平 topic 和 theme/subtopic 两种形态。"""
    ensure_knowledge_base()
    topics: list[dict[str, Any]] = [
        _topic_status(topic, include_documents=False)
        for topic, _path in sorted(
            discover_topic_dirs(KNOWLEDGE_BASE), key=lambda item: item[1].stat().st_mtime, reverse=True
        )
    ]
    return json.dumps(
        {
            "status": "ok",
            "active_topic": get_active_knowledge_base_topic(),
            "topic_count": len(topics),
            "topics": topics,
        },
        ensure_ascii=False,
    )


@tool
def inspect_knowledge_base(topic: str) -> str:
    """查看单个知识库 topic 的文档、frontmatter、hash、索引和向量化状态。"""
    safe_topic = _validate_topic(topic)
    return json.dumps(
        {
            "status": "ok",
            "topic": safe_topic,
            "knowledge_base": _topic_status(safe_topic, include_documents=True),
        },
        ensure_ascii=False,
    )


@tool
def summarize_knowledge_base(topic: str) -> str:
    """生成单个知识库的只读管理概览，不进行内容问答，不联网。"""
    safe_topic = _validate_topic(topic)
    status = _topic_status(safe_topic, include_documents=True)
    kb_metadata = status.get("metadata") or {}
    overview = _overview_excerpt(safe_topic)
    doc_titles = [doc["title"] for doc in status.get("documents", [])]
    index_status = status.get("index_status")
    graph = status.get("graph_stats") or {}
    vectors = status.get("vector_stats") or {}
    lines = [
        f"知识库 `{kb_metadata.get('title') or safe_topic}` 当前包含 {status.get('document_count', 0)} 篇 Markdown 文档。",
        f"topic：{safe_topic}；版本：{kb_metadata.get('version') or '未标注'}；来源模式：{kb_metadata.get('source_mode') or '未标注'}。",
        f"入口文档：{status.get('entry_document') or '未识别'}。",
        f"文档结构：{', '.join(doc_titles) if doc_titles else '未识别标题'}。",
        (
            "LightRAG 索引状态："
            f"{index_status}（{status.get('index_reason')}），"
            f"图谱 {graph.get('nodes', 0)} 个节点 / {graph.get('edges', 0)} 条边，"
            f"向量 chunks={vectors.get('chunks', 0)}、entities={vectors.get('entities', 0)}、relationships={vectors.get('relationships', 0)}。"
        ),
        f"当前 embedding：{status.get('embedding_model') or '未知'} / dim={status.get('embedding_dim') or '未知'}。",
    ]
    if overview:
        lines.append(f"总览摘录：{overview}")
    lines.append("适合回答的问题：围绕上述文档主题、概念、结构、实践步骤和已落库证据的问题。")
    lines.append("不适合回答的问题：知识库未覆盖的最新事实、外部网页实时信息、项目源码实现细节或未建索引 topic 的语义问答。")
    return json.dumps(
        {
            "status": "ok",
            "topic": safe_topic,
            "summary": "\n".join(lines),
            "knowledge_base": status,
        },
        ensure_ascii=False,
    )


@tool
def check_knowledge_base_index(topic: str) -> str:
    """诊断单个知识库的 LightRAG 索引状态，并说明是否需要重建及原因。"""
    safe_topic = _validate_topic(topic)
    diagnosis = _index_diagnosis(safe_topic)
    return json.dumps(
        {
            "status": "ok",
            "topic": safe_topic,
            "diagnosis": diagnosis,
        },
        ensure_ascii=False,
    )


@tool
def delete_knowledge_base_index(topic: str) -> str:
    """删除单个知识库的本地 LightRAG 索引；不会删除 knowledge_base Markdown 文档。"""
    safe_topic = _validate_topic(topic)
    workspace_dir = LIGHTRAG_WORKSPACES / safe_topic
    if not workspace_dir.exists():
        return json.dumps(
            {
                "status": "missing",
                "topic": safe_topic,
                "message": "该知识库没有可删除的 LightRAG 索引目录。",
            },
            ensure_ascii=False,
        )
    shutil.rmtree(workspace_dir)
    return json.dumps(
        {
            "status": "deleted",
            "topic": safe_topic,
            "deleted_path": str(workspace_dir.relative_to(LIGHTRAG_WORKSPACES.parent)).replace("\\", "/"),
            "message": "已删除 LightRAG 索引目录；Markdown 知识库文档未删除，可通过重建索引恢复。",
        },
        ensure_ascii=False,
    )


@tool
def inspect_theme_tool(theme: str) -> str:
    """只读扫描一个大主题（theme）下的所有子主题知识库，生成主题总览草稿。

    docs/VISION.md 第1节：大主题层需要一份总览/索引文件，回答"这个主题目前覆盖了哪些方面、还缺
    什么"。本工具只生成草稿数据，不写入任何文件——如果主 Agent 认可这份草稿，需要自己调用
    save_knowledge(title="00_主题总览", topic=theme, content=...) 落库，这是"提议/执行分离"的
    只读边界，和 delete_knowledge_base_index 的设计原则一致。

    theme 必须是单段目录名（不含 "/"），且必须已经是一个"主题容器"：即
    knowledge_base/<theme>/ 下至少有一个子目录自身含有 Markdown 文档（子主题）。
    """
    safe_theme = _validate_theme(theme)
    subtopics = _theme_subtopics(safe_theme)
    return json.dumps(
        {
            "status": "ok",
            "theme": safe_theme,
            "subtopic_count": len(subtopics),
            "subtopics": subtopics,
            "draft_overview": _draft_theme_overview(safe_theme, subtopics),
        },
        ensure_ascii=False,
    )


@tool
def bind_channel_to_topic_tool(topic: str) -> str:
    """把当前会话所在的频道（如飞书群）持久绑定到一个知识库 topic，作为该频道之后默认问答的知识库。

    与 set_active_knowledge_base 的临时切换是两回事：绑定是持久化的（重启后依然生效），
    set_active_knowledge_base 仍然可以在本次会话里临时切到别的 topic 而不影响绑定关系。

    topic 可以是叶子级 topic（扁平 topic 或 "theme/subtopic"），也可以是一个大主题（theme）本身
    ——绑定到 theme 后，推送轮询（第5节）和知识库问答（query_knowledge_base）都会跨该主题下所有
    子主题聚合。
    """
    safe_topic = _validate_topic_or_theme(topic)
    channel_id = conversation_channel_id()
    try:
        _channel_service().bind_topic(channel_id, safe_topic)
    except ValueError:
        return json.dumps(
            {
                "status": "channel_not_registered",
                "channel_id": channel_id,
                "message": "当前会话还没有注册的 channel 记录，暂时无法绑定；先在这个频道发一条普通消息建立会话后再试。",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "bound",
            "channel_id": channel_id,
            "topic": safe_topic,
            "message": f"已将当前频道绑定到知识库 {safe_topic}，之后本频道的知识库问答默认使用这个 topic。",
        },
        ensure_ascii=False,
    )


@tool
def unbind_channel_topic_tool() -> str:
    """取消当前会话所在频道的知识库绑定；取消后该频道的问答需要重新指定 topic 或临时切换。"""
    channel_id = conversation_channel_id()
    try:
        _channel_service().unbind_topic(channel_id)
    except ValueError:
        return json.dumps(
            {
                "status": "channel_not_registered",
                "channel_id": channel_id,
                "message": "当前会话还没有注册的 channel 记录，没有绑定可取消。",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {"status": "unbound", "channel_id": channel_id, "message": "已取消当前频道的知识库绑定。"},
        ensure_ascii=False,
    )


@tool
def get_channel_topic_binding_tool() -> str:
    """查询当前会话所在频道目前绑定的知识库 topic；没有绑定时 topic_binding 为 null。"""
    channel_id = conversation_channel_id()
    topic = _channel_service().get_topic_binding(channel_id)
    return json.dumps(
        {"status": "ok", "channel_id": channel_id, "topic_binding": topic},
        ensure_ascii=False,
    )


@tool
def get_channel_push_config_tool() -> str:
    """查询当前频道的知识推送配置：开关、频率、时间、时区、内容策略及上下次推送。"""
    channel_id = conversation_channel_id()
    config = _channel_service().get_push_config(channel_id)
    return json.dumps({"status": "ok", "channel_id": channel_id, "push_config": config}, ensure_ascii=False)


@tool
def set_channel_push_config_tool(
    enabled: bool | None = None,
    frequency: str = "",
    content_strategy: str = "",
    push_time: str = "",
    timezone: str = "",
) -> str:
    """配置当前频道的知识主动推送。只传需要修改的参数，未传的保持原值不变。

    - enabled：是否开启推送。
    - frequency：推送频率档位，daily（每天）| weekly_3x（每周三次）| weekly_1x（每周一次）。
    - content_strategy：内容策略，rotation（轮询复习+新增插队，默认）| new_only（只推新增）| random（随机）。
    - push_time：频道当地时间，24 小时制 HH:MM，例如 09:00。
    - timezone：IANA 时区，例如 Asia/Shanghai、America/Los_Angeles。

    开启推送（enabled=True）前，当前频道必须已经绑定 KnowledgeBase、Theme 或兼容的旧 topic。
    """
    channel_id = conversation_channel_id()
    channel_svc = _channel_service()
    try:
        channel_svc.get_channel(channel_id)
    except ValueError:
        return json.dumps(
            {
                "status": "channel_not_registered",
                "channel_id": channel_id,
                "message": "当前会话还没有注册的 channel 记录，暂时无法配置推送；先在这个频道发一条普通消息建立会话后再试。",
            },
            ensure_ascii=False,
        )
    if enabled and not channel_svc.get_scope_binding(channel_id):
        return json.dumps(
            {
                "status": "topic_not_bound",
                "channel_id": channel_id,
                "message": "开启推送前需要先绑定 KnowledgeBase、Theme 或知识库 topic。",
            },
            ensure_ascii=False,
        )
    config = channel_svc.upsert_push_config(
        channel_id,
        enabled=enabled,
        frequency=frequency or None,
        content_strategy=content_strategy or None,
        push_time=push_time or None,
        timezone=timezone or None,
    )
    return json.dumps(
        {"status": "updated", "channel_id": channel_id, "push_config": config, "message": "已更新推送配置。"},
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


def _validate_theme(theme: str) -> str:
    safe = str(theme or "").strip().replace("\\", "/").strip("/")
    if not safe or "/" in safe or ".." in safe:
        raise ValueError("theme must be a single knowledge_base directory name")
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"theme not found: {safe}")
    if not _theme_subtopic_dirs(path):
        raise ValueError(f"{safe} is not a theme container (no subtopic directory has Markdown documents): {safe}")
    return safe


def _theme_subtopic_dirs(theme_path: Path) -> list[Path]:
    return sorted(sub for sub in theme_path.iterdir() if sub.is_dir() and list(sub.glob("*.md")))


def _validate_topic_or_theme(value: str) -> str:
    """Accepts either a leaf topic (flat, or "theme/subtopic") or a bare theme container name --
    used only by bind_channel_to_topic_tool (docs/VISION.md 第5节 主题级推送聚合), since binding a
    channel to a theme is meaningful (push rotation aggregates across its subtopics) in a way
    that e.g. inspect_knowledge_base's single-directory document listing is not. Those callers
    stay on leaf-only _validate_topic."""
    try:
        return _validate_topic(value)
    except ValueError as leaf_error:
        try:
            return _validate_theme(value)
        except ValueError:
            pass
        raise leaf_error


def _theme_subtopics(theme: str) -> list[dict[str, Any]]:
    theme_path = safe_knowledge_path(theme)
    subtopics: list[dict[str, Any]] = []
    for sub_path in _theme_subtopic_dirs(theme_path):
        topic = f"{theme}/{sub_path.name}"
        status = _topic_status(topic, include_documents=False)
        kb_metadata = status.get("metadata") or {}
        subtopics.append(
            {
                "subtopic": sub_path.name,
                "topic": topic,
                "title": kb_metadata.get("title") or sub_path.name,
                "description": kb_metadata.get("description") or _overview_excerpt(topic),
                "document_count": status.get("document_count", 0),
                "updated_at": status.get("updated_at"),
            }
        )
    return subtopics


def _draft_theme_overview(theme: str, subtopics: list[dict[str, Any]]) -> str:
    lines = [f"# {theme} 主题总览", ""]
    if not subtopics:
        lines.append("当前主题下暂无子主题。")
        return "\n".join(lines)
    lines.append("本主题目前包含以下子主题调研：")
    lines.append("")
    for item in subtopics:
        summary = item["description"] or "（暂无摘要）"
        lines.append(f"- **{item['title']}**（`{item['topic']}`）：{summary}")
    lines.append("")
    lines.append("## 待补充方向")
    lines.append("")
    lines.append("以上为草稿，由主 Agent 结合用户需求判断本主题还缺哪些方面，补充到这里。")
    return "\n".join(lines)


def _topic_markdown_documents(topic: str) -> list[Path]:
    return sorted(safe_knowledge_path(topic).glob("*.md"))


def _topic_status(topic: str, *, include_documents: bool) -> dict[str, Any]:
    docs = _topic_markdown_documents(topic)
    topic_path = safe_knowledge_path(topic)
    kb_metadata = _load_or_infer_kb_metadata(topic, docs)
    entry = _entry_document(docs)
    index_state = lightrag_index_state(topic)
    index_metadata = index_state.metadata or {}
    storage_dir = _storage_dir(topic, index_metadata)
    # LightRAG can leave partial JSON files behind after an interrupted build.
    # They are diagnostics, not a usable downstream vector/graph stage.
    graph_stats = (
        _graph_stats(storage_dir / "graph_chunk_entity_relation.graphml")
        if index_state.available
        else {"nodes": 0, "edges": 0}
    )
    vector_stats = (
        {
            "chunks": _vdb_count(storage_dir / "vdb_chunks.json"),
            "entities": _vdb_count(storage_dir / "vdb_entities.json"),
            "relationships": _vdb_count(storage_dir / "vdb_relationships.json"),
        }
        if index_state.available
        else {"chunks": 0, "entities": 0, "relationships": 0}
    )
    result: dict[str, Any] = {
        "topic": topic,
        "metadata": kb_metadata,
        "metadata_status": "explicit" if _kb_metadata_path(topic).exists() else "inferred",
        "document_count": len(docs),
        "entry_document": entry.name if entry else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(topic_path.stat().st_mtime)),
        "index_status": _index_status(index_state.available, index_state.reason),
        "index_reason": index_state.reason,
        "index_diagnosis": _index_diagnosis(topic),
        "embedding_model": index_metadata.get("embedding_model"),
        "embedding_dim": index_metadata.get("embedding_dim"),
        "chunk_size": index_metadata.get("chunk_size"),
        "chunk_overlap_size": index_metadata.get("chunk_overlap_size"),
        "process_options": index_metadata.get("process_options"),
        "indexed_at": index_metadata.get("indexed_at"),
        "lightrag_workspace": index_metadata.get("lightrag_workspace"),
        "graph_stats": graph_stats,
        "vector_stats": vector_stats,
    }
    if include_documents:
        result["documents"] = [_document_info(path) for path in docs]
    return result


def _kb_metadata_path(topic: str) -> Path:
    return safe_knowledge_path(topic) / KB_METADATA_FILE


def _load_or_infer_kb_metadata(topic: str, docs: list[Path]) -> dict[str, Any]:
    path = _kb_metadata_path(topic)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "topic": topic,
                "title": topic,
                "status": "metadata_invalid",
                "notes": f"kb_metadata.json 解析失败：{exc}",
            }
        if isinstance(payload, dict):
            return payload
    entry = _entry_document(docs)
    title = topic
    description = ""
    if entry:
        text = entry.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        entry_title = str(frontmatter.get("title") or "").strip()
        heading_title = str(_first_heading(body) or "").strip()
        if entry_title and not _is_generic_entry_title(entry_title):
            title = entry_title
        elif heading_title and not _is_generic_entry_title(heading_title):
            title = heading_title
        description = _first_paragraph(body)
    return {
        "topic": topic,
        "title": title,
        "version": "",
        "description": description[:240],
        "source_mode": "",
        "status": "active",
        "created_at": "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(safe_knowledge_path(topic).stat().st_mtime)),
        "notes": "未发现 kb_metadata.json，当前为工具推断信息。",
    }


def _is_generic_entry_title(title: str) -> bool:
    normalized = re.sub(r"^[0-9]+[._\-\s]*", "", str(title or "").strip().lower())
    normalized = re.sub(r"[\s_\-]+", "", normalized)
    return normalized in {"总览", "主题总览", "overview", "readme", "index", "目录"}


def _index_diagnosis(topic: str) -> dict[str, Any]:
    state = lightrag_index_state(topic)
    current = state.metadata or {}
    expected = _build_lightrag_metadata(topic, _topic_markdown_documents(topic))
    differences = _metadata_differences(current, expected) if current else []
    missing_files = _missing_storage_files(topic, current)
    reasons: list[str] = []
    if state.reason == "missing_metadata":
        reasons.append("缺少 index_metadata.json，尚未建立 LightRAG 索引。")
    elif state.reason.startswith("invalid_metadata"):
        reasons.append(f"index_metadata.json 无法解析：{state.reason}")
    elif differences:
        reasons.extend(_humanize_metadata_differences(differences))
    elif missing_files:
        reasons.append(f"索引存储文件不完整：{', '.join(missing_files)}。")
    elif state.available:
        reasons.append("索引可用，当前 Markdown、embedding、chunk 和 process_options 配置与索引元数据一致。")
    else:
        reasons.append(f"索引不可用：{state.reason}。")
    return {
        "index_status": _index_status(state.available, state.reason),
        "available": state.available,
        "reason": state.reason,
        "needs_rebuild": not state.available,
        "rebuild_reasons": reasons,
        "metadata_differences": differences,
        "missing_storage_files": missing_files,
        "expected_metadata": _compact_expected_metadata(expected),
        "current_metadata": _compact_expected_metadata(current) if current else {},
    }


def _metadata_differences(current: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "topic",
        "embedding_model",
        "embedding_dim",
        "embedding_token_limit",
        "embedding_send_dim",
        "embedding_batch_num",
        "embedding_input_chars",
        "chunk_size",
        "chunk_overlap_size",
        "process_options",
        "documents",
    ]
    differences = []
    for key in keys:
        if current.get(key) != expected.get(key):
            differences.append(
                {
                    "field": key,
                    "current": _compact_value(current.get(key)),
                    "expected": _compact_value(expected.get(key)),
                }
            )
    return differences


def _compact_value(value: Any) -> Any:
    if isinstance(value, list):
        return {"count": len(value), "sha256": hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()}
    return value


def _humanize_metadata_differences(differences: list[dict[str, Any]]) -> list[str]:
    labels = {
        "documents": "Markdown 文档内容、数量或 hash 发生变化",
        "embedding_model": "embedding 模型发生变化",
        "embedding_dim": "embedding 维度发生变化",
        "embedding_token_limit": "embedding token 限制发生变化",
        "embedding_send_dim": "embedding dimensions 发送配置发生变化",
        "embedding_batch_num": "embedding 批量参数发生变化",
        "embedding_input_chars": "embedding 输入截断长度发生变化",
        "chunk_size": "chunk_size 发生变化",
        "chunk_overlap_size": "chunk_overlap_size 发生变化",
        "process_options": "LightRAG process_options 发生变化",
        "topic": "topic 不一致",
    }
    return [f"{labels.get(item['field'], item['field'])}，建议重建索引。" for item in differences]


def _compact_expected_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        "topic": metadata.get("topic"),
        "embedding_model": metadata.get("embedding_model"),
        "embedding_dim": metadata.get("embedding_dim"),
        "chunk_size": metadata.get("chunk_size"),
        "chunk_overlap_size": metadata.get("chunk_overlap_size"),
        "process_options": metadata.get("process_options"),
        "document_count": len(metadata.get("documents") or []),
        "indexed_at": metadata.get("indexed_at"),
        "lightrag_workspace": metadata.get("lightrag_workspace"),
    }


def _missing_storage_files(topic: str, metadata: dict[str, Any]) -> list[str]:
    if not metadata:
        return []
    storage_dir = _storage_dir(topic, metadata)
    markers = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml",
    ]
    return [name for name in markers if not (storage_dir / name).exists()]


def _entry_document(files: list[Path]) -> Path | None:
    for file in files:
        if file.name.startswith("00_") or "overview" in file.stem.lower() or "总览" in file.stem:
            return file
    return files[0] if files else None


def _document_info(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    title = frontmatter.get("title") or _first_heading(body) or path.stem
    return {
        "title": title,
        "path": str(path.relative_to(KNOWLEDGE_BASE.parent)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "frontmatter": frontmatter,
    }


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    frontmatter: dict[str, Any] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, parts[2]


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _first_paragraph(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith(">"):
            if lines:
                break
            continue
        if stripped.startswith("- ") or re.match(r"^\d+\.", stripped):
            if lines:
                break
            continue
        lines.append(stripped)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _storage_dir(topic: str, metadata: dict[str, Any]) -> Path:
    workspace = str(metadata.get("lightrag_workspace") or "").strip()
    if workspace:
        return LIGHTRAG_WORKSPACES / topic / workspace
    return LIGHTRAG_WORKSPACES / topic / _workspace_name(topic)


def _workspace_name(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", topic).strip("_") or "topic"
    digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]
    return f"kb_{slug[:40]}_{digest}"


def _index_status(available: bool, reason: str) -> str:
    if available:
        return "ready"
    if reason == "missing_metadata":
        return "missing"
    if reason.startswith("invalid_metadata"):
        return "invalid"
    if reason.startswith("missing_storage_files"):
        return "incomplete"
    if reason == "stale_metadata":
        return "stale"
    return "error"


def _graph_stats(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"nodes": 0, "edges": 0}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {"nodes": 0, "edges": 0}
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    return {
        "nodes": len(root.findall(".//g:node", namespace)),
        "edges": len(root.findall(".//g:edge", namespace)),
    }


def _vdb_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    data = payload.get("data") if isinstance(payload, dict) else None
    return len(data) if isinstance(data, list) else 0


def _overview_excerpt(topic: str) -> str:
    docs = _topic_markdown_documents(topic)
    entry = _entry_document(docs)
    if not entry:
        return ""
    text = entry.read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    cleaned = re.sub(r"\s+", " ", re.sub(r"^#+\s+.*$", "", body, flags=re.MULTILINE)).strip()
    return cleaned[:500]
