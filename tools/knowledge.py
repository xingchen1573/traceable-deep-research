"""Knowledge-base tools for saving research findings as structured documents."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event

from ._registry import ensure_knowledge_base, safe_knowledge_path, validate_topic_shape


@tool
def save_knowledge(
    title: str,
    content: str,
    topic: str = "",
    sources: str | list[str] | None = None,
    keywords: str | list[str] | None = None,
) -> str:
    """将调研结论保存为知识库文档。

    由主 Agent 在整理完调研内容后调用，生成带 frontmatter 的 Markdown 文件。

    title 规则（必须严格遵守）：
    - title 就是最终 Markdown 文件名，不要包含 .md 后缀。
    - 如果用户明确指定文件名、标题、子文档主题，或说“至少包含 A、B、C”，必须把对应名称原样传入 title。
    - 不要把用户指定标题改写、合并或替换成同义标题、概括标题、泛化标题。
    - 如果用户指定了编号前缀（如 01_<用户指定标题>），title 必须保留该编号前缀。

    topic 规则（必须严格遵守）：
    - 只有用户明确要求归档到某个专题时，才填写 topic。
    - 如果用户没有明确指定专题名称，必须传空字符串，让文档平铺在 knowledge_base/ 根目录。
    - 不要自行提炼主题。例如：用户说"了解 AI 趋势"，不要推断 topic="ai_trends"。

    写入路径：
    - topic 非空 → knowledge_base/<topic>/<title>.md
    - topic 为空 → knowledge_base/<title>.md

    sources 和 keywords 可以是列表，也可以是 JSON 字符串（如 '["a","b"]'）。
    日期由系统自动维护，不要在 content 中自行编写日期。
    """
    result = _write_knowledge_document(
        title=title,
        content=content,
        topic=topic,
        sources=sources,
        keywords=keywords,
        event_type="knowledge.saved",
    )
    return (
        f"已保存知识文档 {result['path']} "
        f"(topic={result['topic'] or '无'}, title={result['title']})."
    )


@tool
def save_knowledge_set(
    topic: str,
    documents: list[dict[str, Any]] | str,
    sources: str | list[str] | None = None,
    keywords: str | list[str] | None = None,
) -> str:
    """批量保存同一个文档集下的多篇知识库 Markdown。

    用于深度调研、系统学习、详细报告、报告文件夹、系列文档或文档集。

    必须严格遵守：
    - topic 是文档集文件夹名；用户指定 topic 时必须原样传入。
    - documents 是文档数组，每项必须包含 title 和 content。
    - title 是最终 Markdown 文件名，不要包含 .md 后缀。
    - 如果用户明确指定标题清单，documents 必须逐项使用这些标题，不得改写、合并或替换。
    - 所有文档会在一次工具调用中写入同一 topic；不要逐篇调用 save_knowledge。
    - 文档集必须包含 00_总览 或 overview 文档。
    - 非总览文档不强制固定章节名，但必须有足够篇幅、来源、连续分析和不确定性/边界说明。

    写入路径：
    - knowledge_base/<topic>/<title>.md

    sources 和 keywords 是文档集级元数据，可以是列表或 JSON 字符串。
    """
    saved = _save_knowledge_set_impl(
        topic=topic,
        documents=documents,
        sources=sources,
        keywords=keywords,
    )
    return json.dumps(saved, ensure_ascii=False)


def _save_knowledge_set_impl(
    *,
    topic: str,
    documents: list[dict[str, Any]] | str,
    sources: str | list[str] | None = None,
    keywords: str | list[str] | None = None,
    source_job_id: str | None = None,
    source_url: str | None = None,
    skip_quality_gate: bool = False,
    knowledge_base_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """save_knowledge_set 的实现主体，供 @tool 版本和内部调用方（如 approve_research，
    save_single_url_brief_tool）共用。source_job_id/source_url 只在这里作为参数存在，
    不出现在 @tool 签名里，避免 agent 自己填一个可能不存在/编造的 job_id。

    skip_quality_gate 同理只在内部调用可见：用于已经通过上游专用质量门的产物。当前包括：
    single_url_brief（有独立的引用真实性/实质性判据，见 scripts/run_single_url_brief_eval.py）
    和 quality_gate.approval_ready=true 的 question_loop 调研报告。后者已经完成覆盖度、引用、
    grounding、claim entailment 和报告质量校验，不应在写盘时再被低精度关键词启发式二次否决。
    普通知识文档和没有上游质量门的旧 job 仍使用 _validate_knowledge_set_quality。"""
    ensure_knowledge_base()
    safe_topic = _validate_topic(topic)
    docs = _coerce_documents(documents)
    sources_list = _coerce_list(sources)
    keywords_list = _coerce_list(keywords)

    prepared: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    seen_paths: set[str] = set()
    for doc in docs:
        raw_title = str(doc.get("title", "")).strip()
        raw_content = str(doc.get("content", "")).strip()
        safe_title = _validate_title(raw_title)
        if not raw_content:
            raise ValueError(f"Document content is empty for title: {raw_title or '<empty>'}")
        if raw_title in seen_titles:
            raise ValueError(f"Duplicate document title: {raw_title}")
        seen_titles.add(raw_title)
        relative_path = f"{safe_topic}/{safe_title}.md"
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate document path after sanitization: {relative_path}")
        seen_paths.add(relative_path)
        prepared.append({
            "title": raw_title,
            "content": raw_content,
            "safe_title": safe_title,
            "evidence_metrics": doc.get("evidence_metrics") or {},
            "evidence_table": doc.get("evidence_table") or [],
        })

    if not skip_quality_gate:
        _validate_knowledge_set_quality(prepared, sources_list)

    results = [
        _write_knowledge_document(
            title=item["title"],
            content=item["content"],
            topic=safe_topic,
            sources=sources_list,
            keywords=keywords_list,
            event_type="knowledge.set_document_saved",
            source_job_id=source_job_id,
            source_url=source_url,
        )
        for item in prepared
    ]
    metadata = _write_knowledge_base_metadata(
        safe_topic,
        {
            "title": safe_topic,
            **(knowledge_base_metadata or {}),
            "source_job_id": source_job_id or (knowledge_base_metadata or {}).get("source_job_id"),
        },
    )

    emit_event(
        "knowledge.set_saved",
        "tools",
        "Saved knowledge document set.",
        data={
            "topic": safe_topic,
            "document_count": len(results),
            "paths": [item["path"] for item in results],
            "metadata_path": metadata["path"],
        },
    )
    return {
        "status": "saved",
        "topic": safe_topic,
        "document_count": len(results),
        "paths": [item["path"] for item in results],
        "titles": [item["title"] for item in results],
        "metadata_path": metadata["path"],
        "metadata": metadata["metadata"],
    }


def _write_knowledge_base_metadata(topic: str, metadata: dict[str, Any]) -> dict[str, Any]:
    path = safe_knowledge_path(f"{topic}/kb_metadata.json")
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing = payload
        except (OSError, json.JSONDecodeError):
            existing = {}

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    merged = {
        **existing,
        **{key: value for key, value in metadata.items() if value not in (None, "")},
        "topic": topic,
        "status": str(metadata.get("status") or existing.get("status") or "active"),
        "created_at": str(existing.get("created_at") or metadata.get("created_at") or now),
        "updated_at": now,
    }
    merged["title"] = str(merged.get("title") or topic).strip() or topic
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    relative_path = _relative_knowledge_path(path)
    emit_event(
        "knowledge.metadata.saved",
        "tools",
        "Saved knowledge-base metadata.",
        data={"topic": topic, "title": merged["title"], "path": relative_path},
    )
    return {"path": relative_path, "metadata": merged}


@tool
def research_coverage_check(
    required_titles: str | list[str],
    retrieval_result: str | dict[str, Any],
    user_request: str = "",
    round_index: int = 1,
    max_rounds: int = 2,
) -> str:
    """检查调研结果是否足够写入文档集。

    这是只读判断工具，不检索、不写文件。

    用于文档集调研流程：
    - web_worker_tool 或 repo_worker_tool 返回证据包后调用本工具。
    - 如果 ready_to_write=true，下一步应进入 save_knowledge_set。
    - 如果 ready_to_write=false 且 round_index < max_rounds，只能按 followup_queries 做一次针对性补查。
    - 如果达到 max_rounds，即使仍有缺口，也应写入文档并在内容中标注不确定性。
    """
    titles = _coerce_titles(required_titles)
    result = _coerce_result(retrieval_result)
    round_index = max(int(round_index or 1), 1)
    max_rounds = max(int(max_rounds or 2), 1)

    sources = _coerce_items(result.get("sources"))
    documents = _coerce_items(result.get("documents"))
    evidences = _coerce_items(result.get("evidences"))
    key_facts = _coerce_items(result.get("key_facts"))
    evidence_pack = _coerce_items(result.get("evidence_pack"))
    evidence_map = _coerce_items(result.get("evidence_map"))
    selected_chunks = _coerce_items(result.get("selected_chunks"))
    gaps = [str(item) for item in _coerce_items(result.get("gaps")) if str(item).strip()]

    source_count = len(_unique_urls(sources))
    document_count = len(_unique_urls(documents))
    evidence_count = len(evidences)
    key_fact_count = len(key_facts)
    evidence_pack_count = max(len(evidence_pack), len(evidence_map))
    evidence_text = _coverage_text(sources, documents, evidences, key_facts, evidence_pack, evidence_map, selected_chunks)

    section_by_title = _section_id_by_title(titles)
    covered_titles, missing_titles = _classify_title_coverage(
        titles=titles,
        evidence_text=evidence_text,
        evidence_items=[*evidence_pack, *evidence_map, *selected_chunks],
        section_by_title=section_by_title,
    )
    per_title_status = _per_title_status(
        titles=titles,
        sources=sources,
        documents=documents,
        evidences=evidences,
        key_facts=key_facts,
        evidence_pack=[*evidence_pack, *evidence_map, *selected_chunks],
        section_by_title=section_by_title,
    )
    depth_ready = bool(per_title_status) and all(
        item["status"] == "ready"
        for item in per_title_status
        if not _is_overview_title(str(item.get("title") or ""))
    )
    hard_ready = (
        source_count >= 2
        and (document_count >= 1 or evidence_count >= 3 or key_fact_count >= 3 or evidence_pack_count >= 3)
        and len(covered_titles) >= max(1, min(len(titles), 2))
        and (depth_ready or not titles)
    )
    has_fatal_gap = _has_fatal_gap(gaps)
    max_reached = round_index >= max_rounds
    ready = hard_ready and not has_fatal_gap

    followup_queries: list[str] = []
    if not ready:
        weak_titles = [
            str(item["title"])
            for item in per_title_status
            if item.get("status") in {"thin", "missing"}
            and not _is_overview_title(str(item.get("title") or ""))
        ] or missing_titles
        for title in weak_titles[:2]:
            topic_hint = f" {user_request}" if user_request else ""
            followup_queries.append(f"{_strip_title_prefix(title)}{topic_hint}".strip())
        if not followup_queries and source_count < 3:
            followup_queries.append(str(user_request or "补充权威来源").strip())

    output = {
        "ready_to_write": ready,
        "round_index": round_index,
        "max_rounds": max_rounds,
        "source_count": source_count,
        "document_count": document_count,
        "evidence_count": evidence_count,
        "key_fact_count": key_fact_count,
        "evidence_pack_count": evidence_pack_count,
        "depth_ready": depth_ready,
        "per_title_status": per_title_status,
        "covered_titles": covered_titles,
        "missing_titles": missing_titles,
        "fatal_gaps": [gap for gap in gaps if _has_fatal_gap([gap])],
        "followup_queries": [] if ready or max_reached else followup_queries[:2],
        "reason": _coverage_reason(
            ready=ready,
            hard_ready=hard_ready,
            max_reached=max_reached,
            has_fatal_gap=has_fatal_gap,
            missing_titles=missing_titles,
        ),
    }
    return json.dumps(output, ensure_ascii=False)


def _coerce_list(value: str | list[str] | None) -> list[str]:
    """Accept list, JSON string, or plain string and return a clean list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    # Fallback: plain string treated as single-item list
    return [text]


def _coerce_titles(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    titles: list[str] = []
    for line in text.splitlines():
        title = line.strip().strip("-*`'\" ")
        title = re.sub(r"^\d+[.、)]\s*", "", title)
        if title:
            titles.append(title)
    return titles or [text]


def _coerce_result(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _unique_urls(items: list[Any]) -> set[str]:
    urls: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("source_url") or "").strip()
            if url:
                urls.add(url)
            source_urls = item.get("source_urls")
            if isinstance(source_urls, list):
                urls.update(str(value).strip() for value in source_urls if str(value).strip())
    return urls


def _coverage_text(*groups: list[Any]) -> str:
    pieces: list[str] = []
    for group in groups:
        for item in group:
            if isinstance(item, dict):
                pieces.extend(str(value) for value in item.values() if value)
            elif item:
                pieces.append(str(item))
    return "\n".join(pieces).lower()


def _classify_title_coverage(
    *,
    titles: list[str],
    evidence_text: str,
    evidence_items: list[Any] | None = None,
    section_by_title: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    covered: list[str] = []
    missing: list[str] = []
    if not titles:
        return covered, missing
    section_counts = _section_item_counts(evidence_items or [])
    section_by_title = section_by_title or _section_id_by_title(titles)
    has_any_evidence = bool(evidence_text.strip())
    for title in titles:
        section_id = section_by_title.get(title)
        if section_id and section_counts.get(section_id, 0) > 0:
            covered.append(title)
            continue
        normalized = _strip_title_prefix(title)
        if normalized in {"总览", "overview", "summary"}:
            if has_any_evidence:
                covered.append(title)
            else:
                missing.append(title)
            continue
        keywords = _title_keywords(normalized)
        if not keywords:
            if has_any_evidence:
                covered.append(title)
            else:
                missing.append(title)
            continue
        if any(keyword.lower() in evidence_text for keyword in keywords):
            covered.append(title)
        else:
            missing.append(title)
    return covered, missing


def _per_title_status(
    *,
    titles: list[str],
    sources: list[Any],
    documents: list[Any],
    evidences: list[Any],
    key_facts: list[Any],
    evidence_pack: list[Any],
    section_by_title: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not titles:
        return []
    evidence_items = [*evidence_pack, *evidences, *key_facts]
    section_by_title = section_by_title or _section_id_by_title(titles)
    global_source_count = len(_unique_urls([*sources, *documents, *evidence_items]))
    global_evidence_count = len(evidence_items)
    non_overview_count = len([title for title in titles if not _is_overview_title(title)])
    statuses: list[dict[str, Any]] = []
    for title in titles:
        if _is_overview_title(title):
            status = "ready" if global_source_count >= 2 and global_evidence_count >= 3 else "thin"
            statuses.append({
                "title": title,
                "source_count": global_source_count,
                "evidence_count": global_evidence_count,
                "status": status,
                "gaps": [] if status == "ready" else ["总览缺少足够的全局证据支撑。"],
            })
            continue

        section_id = section_by_title.get(title)
        relevant_items = _items_for_section(evidence_items, section_id) if section_id else []
        if not relevant_items:
            relevant_items = _relevant_items_for_title(title, evidence_items)
        if not relevant_items and non_overview_count == 1:
            relevant_items = evidence_items
        relevant_source_count = len(_unique_urls(relevant_items))
        relevant_evidence_count = len(relevant_items)
        gaps: list[str] = []
        if relevant_source_count < 2:
            gaps.append("相关来源少于 2 个。")
        if relevant_evidence_count < 3:
            gaps.append("相关证据少于 3 条。")
        if relevant_source_count >= 2 and relevant_evidence_count >= 3:
            status = "ready"
        elif relevant_source_count == 0 and relevant_evidence_count == 0:
            status = "missing"
        else:
            status = "thin"
        statuses.append({
            "title": title,
            "source_count": relevant_source_count,
            "evidence_count": relevant_evidence_count,
            "status": status,
            "gaps": gaps,
        })
    return statuses


def _section_id_by_title(titles: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index, title in enumerate(titles):
        match = re.match(r"^\s*(\d{2})[_\-]", str(title))
        mapping[title] = f"sec_{match.group(1)}" if match else f"sec_{index:02d}"
    return mapping


def _section_item_counts(items: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id") or "").strip()
        if section_id:
            counts[section_id] = counts.get(section_id, 0) + 1
    return counts


def _items_for_section(items: list[Any], section_id: str | None) -> list[Any]:
    if not section_id:
        return []
    return [
        item for item in items
        if isinstance(item, dict) and str(item.get("section_id") or "").strip() == section_id
    ]


def _relevant_items_for_title(title: str, items: list[Any]) -> list[Any]:
    keywords = [keyword.lower() for keyword in _title_keywords(_strip_title_prefix(title))]
    if not keywords:
        return items
    relevant: list[Any] = []
    for item in items:
        text = _coverage_text([item])
        suggested_titles: list[str] = []
        if isinstance(item, dict):
            raw_titles = item.get("suggested_titles") or item.get("suggested_sections") or []
            if isinstance(raw_titles, str):
                suggested_titles = [raw_titles]
            elif isinstance(raw_titles, list):
                suggested_titles = [str(value) for value in raw_titles]
        suggested_text = " ".join(suggested_titles).lower()
        if any(keyword in text or keyword in suggested_text for keyword in keywords):
            relevant.append(item)
    return relevant


def _is_overview_title(title: str) -> bool:
    if re.match(r"^\s*00[_\-]", str(title).strip()):
        return True
    normalized = _strip_title_prefix(title).lower()
    return normalized in {"总览", "overview", "summary"} or "总览" in normalized or "overview" in normalized


def _strip_title_prefix(title: str) -> str:
    text = str(title).strip()
    text = re.sub(r"^\d{2}_", "", text)
    return re.sub(r"_[A-Fa-f0-9]{4,}$", "", text).strip()


def _title_keywords(title: str) -> list[str]:
    normalized = re.sub(r"[_\-]+", " ", title).strip()
    if not normalized:
        return []
    keywords = [part for part in re.split(r"\s+", normalized) if len(part) >= 3]
    if any("\u4e00" <= ch <= "\u9fff" for ch in normalized):
        chunks = [part for part in re.split(r"[与和及、/ ]+", normalized) if part]
        keywords.extend(chunks)
    return list(dict.fromkeys(keywords))


def _has_fatal_gap(gaps: list[str]) -> bool:
    fatal_markers = ["无结果", "没有找到", "无法获取", "全部失败", "no results", "not found", "failed"]
    return any(any(marker in gap.lower() for marker in fatal_markers) for gap in gaps)


def _coverage_reason(
    *,
    ready: bool,
    hard_ready: bool,
    max_reached: bool,
    has_fatal_gap: bool,
    missing_titles: list[str],
) -> str:
    if ready and max_reached and missing_titles:
        return "已达到最大调研轮次，应进入写作并在缺口处标注不确定性。"
    if ready and hard_ready:
        return "来源、正文、证据和文档级深度达到默认阈值，可以进入文档集写作。"
    if has_fatal_gap:
        return "存在阻塞性证据缺口，需要针对性补查。"
    if missing_titles:
        return "部分指定文档标题缺少明显覆盖证据，需要针对缺口补查。"
    return "证据数量不足，需要补充来源。"


def _validate_knowledge_set_quality(prepared: list[dict[str, Any]], sources: list[str]) -> None:
    if not any(_is_overview_title(item["title"]) for item in prepared):
        raise ValueError("Document set must include a 00_* or overview entry document.")
    for item in prepared:
        title = item["title"]
        content = item["content"]
        if _is_overview_title(title):
            continue
        issues = _knowledge_quality_issues(content, sources, item)
        if issues:
            raise ValueError(f"Document quality check failed for {title}: {'; '.join(issues)}")


def _knowledge_quality_issues(
    content: str,
    docset_sources: list[str],
    document: dict[str, Any] | None = None,
) -> list[str]:
    issues: list[str] = []
    text = content.strip()
    evidence_ready = _evidence_bound_document_ready(document or {})
    if len(text) < 1500 and not evidence_ready:
        issues.append("content is shorter than 1500 characters")
    urls = _extract_urls(text)
    if len(set(urls or docset_sources)) < 2:
        issues.append("fewer than 2 source URLs or source entries")
    if _looks_like_list_only(text):
        issues.append("content is mostly list items and lacks continuous analysis paragraphs")
    if not _has_analysis_signal(text):
        issues.append("missing clear analysis or interpretation")
    if not _has_uncertainty_signal(text):
        issues.append("missing uncertainty, limitation, or follow-up verification note")
    return issues


def _evidence_bound_document_ready(document: dict[str, Any]) -> bool:
    metrics = document.get("evidence_metrics") or {}
    table = document.get("evidence_table") or []
    evidence_count = max(
        int(metrics.get("evidence_count") or 0),
        len([row for row in table if isinstance(row, dict)]),
    )
    source_count = int(metrics.get("source_count") or 0)
    domain_count = int(metrics.get("domain_count") or 0)
    weak_question_count = int(metrics.get("weak_question_count") or 0)
    weak_matrix_count = int(metrics.get("weak_matrix_cell_count") or 0)
    paragraph_status = str(metrics.get("paragraph_validation_status") or "")
    if paragraph_status and paragraph_status != "passed":
        return False
    return (
        evidence_count >= 3
        and source_count >= 2
        and domain_count >= 2
        and weak_question_count == 0
        and weak_matrix_count == 0
    )


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)\]>'\"]+", text)


def _looks_like_list_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    list_lines = [
        line for line in lines
        if re.match(r"^([-*+]|\d+[.)、]|#{1,6}\s)", line)
    ]
    paragraph_lines = [
        line for line in lines
        if len(line) >= 80 and not re.match(r"^([-*+]|\d+[.)、]|#{1,6}\s|\|)", line)
    ]
    return len(list_lines) / max(len(lines), 1) > 0.65 and len(paragraph_lines) < 2


def _has_analysis_signal(text: str) -> bool:
    markers = [
        "因此", "意味着", "可以看出", "相比", "对比", "原因", "影响",
        "判断", "分析", "取舍", "风险", "适合", "不适合", "核心问题",
    ]
    return any(marker in text for marker in markers)


def _has_uncertainty_signal(text: str) -> bool:
    markers = [
        "不确定", "限制", "局限", "缺口", "待确认", "待核验", "后续核验",
        "证据不足", "无法确认", "可能", "仍需", "边界",
    ]
    return any(marker in text for marker in markers)


def _coerce_documents(value: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        docs = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("documents must not be empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("documents must be a list or JSON array string") from exc
        if not isinstance(parsed, list):
            raise ValueError("documents JSON must be an array")
        docs = parsed
    if not docs:
        raise ValueError("documents must contain at least one document")
    normalized: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        if not isinstance(doc, dict):
            raise ValueError(f"Document #{index} must be an object")
        normalized.append(doc)
    return normalized


def _write_knowledge_document(
    *,
    title: str,
    content: str,
    topic: str = "",
    sources: str | list[str] | None = None,
    keywords: str | list[str] | None = None,
    event_type: str,
    source_job_id: str | None = None,
    source_url: str | None = None,
) -> dict[str, str]:
    ensure_knowledge_base()
    raw_title = str(title).strip()
    raw_content = str(content).strip()
    safe_title = _validate_title(raw_title)
    if not raw_content:
        raise ValueError("content must not be empty")
    safe_topic = _sanitize_filename(topic).strip("-_") if topic else ""
    if safe_topic:
        validate_topic_shape(safe_topic)

    if safe_topic:
        full = safe_knowledge_path(f"{safe_topic}/{safe_title}.md")
    else:
        full = safe_knowledge_path(f"{safe_title}.md")
    full.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(
        title=raw_title,
        safe_title=safe_title,
        content=raw_content,
        safe_topic=safe_topic,
        sources=_coerce_list(sources),
        keywords=_coerce_list(keywords),
        source_job_id=source_job_id,
        source_url=source_url,
    )
    full.write_text(payload, encoding="utf-8")

    relative_path = _relative_knowledge_path(full)
    emit_event(
        event_type,
        "tools",
        "Saved knowledge document.",
        data={
            "path": relative_path,
            "title": raw_title,
            "topic": safe_topic,
            "chars": len(raw_content),
        },
    )
    return {
        "path": relative_path,
        "title": raw_title,
        "topic": safe_topic,
    }


def _build_payload(
    *,
    title: str,
    safe_title: str,
    content: str,
    safe_topic: str,
    sources: list[str],
    keywords: list[str],
    source_job_id: str | None = None,
    source_url: str | None = None,
) -> str:
    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    frontmatter_lines = ["---", f"title: {title[:120] or safe_title}"]
    if safe_topic:
        frontmatter_lines.append(f"topic: {safe_topic}")
    if sources:
        frontmatter_lines.append(f"sources: {sources}")
    if keywords:
        frontmatter_lines.append(f"keywords: {keywords}")
    if source_job_id:
        frontmatter_lines.append(f"source_job_id: {source_job_id}")
    if source_url:
        frontmatter_lines.append(f"source_url: {source_url}")
    frontmatter_lines.append(f"created_at: {created_at}")
    frontmatter_lines.append("---\n")
    return "\n".join(frontmatter_lines) + "\n" + content + "\n"


def _validate_title(title: str) -> str:
    if not title:
        raise ValueError("title must not be empty")
    if title.endswith(".md"):
        raise ValueError("title must not include the .md suffix")
    safe_title = _sanitize_filename(title)
    if not safe_title:
        raise ValueError("title is not a valid filename")
    return safe_title


def _validate_topic(topic: str) -> str:
    safe_topic = _sanitize_filename(topic).strip("-_") if topic else ""
    if not safe_topic:
        raise ValueError("topic must not be empty for save_knowledge_set")
    validate_topic_shape(safe_topic)
    return safe_topic


def _relative_knowledge_path(path: Path) -> str:
    base = safe_knowledge_path("").parent
    return str(path.relative_to(base))


def _sanitize_filename(name: str) -> str:
    """Strip/replace characters unsafe for filesystem paths."""
    safe = name.strip().replace("\\", "/").replace("..", "")
    safe = "".join(ch if ch.isalnum() or ch in "-_./一-鿿" else "_" for ch in safe)
    return safe.strip("/_-").strip()
