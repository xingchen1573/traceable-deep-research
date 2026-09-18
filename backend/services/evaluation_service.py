"""Evaluation service functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

try:
    from ...tools._registry import ROOT, safe_knowledge_path
    from ...tools.knowledge_eval import _score_case, _summarize_results, evaluate_knowledge_base_qa
except ImportError:  # pragma: no cover
    from tools._registry import ROOT, safe_knowledge_path
    from tools.knowledge_eval import _score_case, _summarize_results, evaluate_knowledge_base_qa

try:
    from ...observability import emit_event
except ImportError:  # pragma: no cover
    from observability import emit_event

from . import eval_case_store, knowledge_scope_service
from .common import invoke_tool
from .job_service import job_service
from ..schemas.research import ResearchJobCreateRequest

EVALUATION_DIR = ROOT / "docs" / "evaluations"
RESEARCH_PROBE_OUTPUT_DIR = ROOT / "logs" / "research_e2e"
SINGLE_URL_BRIEF_OUTPUT_DIR = ROOT / "logs" / "single_url_eval"
CASE_GENERATION_STEPS = [
    {"id": "catalog", "title": "读取知识库材料"},
    {"id": "generate", "title": "生成候选评测问题"},
    {"id": "validate", "title": "验证问题与标准证据"},
    {"id": "draft", "title": "形成评测集草稿"},
]

RESEARCH_CONTENT_REVIEW_DIMENSIONS = (
    "answer_coverage",
    "content_richness",
    "readability",
    "scope_alignment",
)
RESEARCH_CONTENT_REVIEW_LABELS = {"良好", "一般", "需改进", "未评估"}


def _production_research_max_rounds() -> int:
    """Return the same request default used by real research jobs."""
    return int(ResearchJobCreateRequest.model_fields["max_rounds"].default)


def create_knowledge_qa_eval_job(topic: str, *, backend: str = "auto", mode: str = "mix") -> dict:
    def target(_: dict) -> dict:
        return invoke_tool(evaluate_knowledge_base_qa, {"topic": topic, "backend": backend, "mode": mode})

    return job_service.create(
        job_type="knowledge_qa_evaluation",
        input_data={"topic": topic, "backend": backend, "mode": mode},
        target=target,
    )


def create_knowledge_scope_eval_job(
    scope_type: str,
    scope_id: str,
    *,
    mode: str = "mix",
    include_history: bool = False,
) -> dict:
    """Run the fixed QA case set against a stable KnowledgeScope."""

    scope = knowledge_scope_service.resolve_scope(scope_type, scope_id, include_history=include_history)
    case_key = knowledge_scope_service.scope_case_key(scope_type, scope_id)

    def target(_: dict) -> dict:
        cases = eval_case_store.load_cases("knowledge_qa", topic=case_key)
        if not cases:
            raise ValueError(f"knowledge scope has no evaluation cases: {case_key}")
        rows = []
        for case in cases:
            output = knowledge_scope_service.query_scope(
                scope_type,
                scope_id,
                str(case.get("question") or ""),
                mode=mode,
                include_history=include_history,
            )
            output["topic"] = ""
            output["backend"] = "knowledge_scope"
            output["requested_backend"] = "knowledge_scope"
            output["actual_backend"] = "knowledge_scope"
            rows.append(_score_case(case, output))
        summary = _summarize_results(rows)
        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        report_id = f"knowledge-scope-{scope_type}-{scope_id[-12:]}-{stamp}"
        payload = {
            "schema_version": 1,
            "report_id": report_id,
            "created_at": created_at,
            "scope": {key: scope[key] for key in ("type", "id", "title", "path")},
            "scope_case_key": case_key,
            "scope_content_hash": knowledge_scope_service.scope_content_hash(
                scope_type, scope_id, include_history=include_history
            ),
            "include_history": include_history,
            "backend": "knowledge_scope",
            "mode": mode,
            "summary": summary,
            "cases": rows,
        }
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        json_path = EVALUATION_DIR / f"{report_id}.json"
        md_path = EVALUATION_DIR / f"{report_id}.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_render_scope_evaluation_report(payload), encoding="utf-8")
        return {
            **payload,
            "report_path": _display_report_path(md_path),
            "report_data_path": _display_report_path(json_path),
        }

    return job_service.create(
        job_type="knowledge_scope_evaluation",
        input_data={
            "scope_type": scope_type,
            "scope_id": scope_id,
            "scope_case_key": case_key,
            "mode": mode,
            "include_history": include_history,
        },
        target=target,
    )


def _display_report_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _render_scope_evaluation_report(payload: dict[str, Any]) -> str:
    scope = payload["scope"]
    summary = payload["summary"]
    lines = [
        f"# KnowledgeScope 问答评估：{scope['title']}",
        "",
        f"评估时间：{payload['created_at']}",
        "",
        f"- scope：`{scope['type']}:{scope['id']}`",
        f"- 内容指纹：`{payload['scope_content_hash']}`",
        f"- 用例数：{summary['total']}",
        f"- 通过：{summary['passed']}",
        f"- 未通过：{summary['failed']}",
        f"- 通过率：{summary['pass_rate']}%",
        f"- 平均分：{summary['average_score']}",
        "",
        "## 用例",
        "",
    ]
    for item in payload["cases"]:
        lines.extend(
            [
                f"### {item['id']} - {'通过' if item['passed'] else '未通过'}",
                "",
                f"- 问题：{item['question']}",
                f"- 分数：{item['score']}",
                f"- 失败检查：{', '.join(item['failed_checks']) or '无'}",
                "",
            ]
        )
    return "\n".join(lines)


def list_knowledge_scope_eval_cases(scope_type: str, scope_id: str) -> dict[str, Any]:
    knowledge_scope_service.resolve_scope(scope_type, scope_id)
    key = knowledge_scope_service.scope_case_key(scope_type, scope_id)
    has_cases = eval_case_store.has_topic_cases("knowledge_qa", key)
    return {"scope_key": key, "has_scope_cases": has_cases, "cases": eval_case_store.load_cases("knowledge_qa", topic=key)}


def upsert_knowledge_scope_eval_case(scope_type: str, scope_id: str, case: dict[str, Any]) -> dict[str, Any]:
    knowledge_scope_service.resolve_scope(scope_type, scope_id)
    key = knowledge_scope_service.scope_case_key(scope_type, scope_id)
    return eval_case_store.upsert_case("knowledge_qa", case, topic=key)


def delete_knowledge_scope_eval_case(scope_type: str, scope_id: str, case_id: str) -> None:
    knowledge_scope_service.resolve_scope(scope_type, scope_id)
    key = knowledge_scope_service.scope_case_key(scope_type, scope_id)
    eval_case_store.delete_case("knowledge_qa", case_id, topic=key)


def create_knowledge_qa_case_generation_job(topic: str) -> dict:
    safe_topic = _validate_eval_topic(topic)
    job = job_service.create(
        job_type="knowledge_qa_case_generation",
        input_data={"topic": safe_topic},
        target=None,
        steps=CASE_GENERATION_STEPS,
    )

    def target(_: dict[str, Any]) -> dict[str, Any]:
        job_id = job["job_id"]
        job_service.update_step(job_id, "catalog", "running", summary="正在读取知识库章节和证据")
        title, documents, catalog = _build_evidence_catalog(safe_topic)
        knowledge_content_hash = _knowledge_content_hash(safe_topic)
        if not catalog:
            raise ValueError("知识库中没有可用于生成评测集的正文材料")
        job_service.update_step(
            job_id,
            "catalog",
            "succeeded",
            summary=f"已读取 {len(documents)} 篇文档和 {len(catalog)} 条稳定证据",
        )

        target_count = _target_case_count(len(documents), len(catalog))
        minimum_count = min(8, target_count)
        version = _next_eval_set_version(safe_topic)
        eval_set_id = f"evalset_{time.strftime('%Y%m%d_%H%M%S')}_{hashlib.sha1(job_id.encode()).hexdigest()[:6]}"
        valid_cases: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        feedback = ""
        for attempt in range(2):
            job_service.update_step(
                job_id,
                "generate",
                "running",
                summary=f"正在生成专属评测问题（第 {attempt + 1} 次）",
            )
            raw_cases = _generate_eval_case_candidates(
                safe_topic,
                title,
                catalog,
                target_count=target_count,
                feedback=feedback,
            )
            job_service.update_step(
                job_id,
                "generate",
                "succeeded",
                summary=f"模型返回 {len(raw_cases)} 条候选问题",
            )
            job_service.update_step(job_id, "validate", "running", summary="正在验证证据定位和问题结构")
            valid_cases, rejected = _validate_generated_cases(
                raw_cases,
                catalog,
                eval_set_id=eval_set_id,
                version=version,
            )
            if len(valid_cases) >= minimum_count:
                break
            feedback = (
                f"上一次只有 {len(valid_cases)} 条候选通过服务端验证，至少需要 {minimum_count} 条。"
                f"主要拒绝原因：{'; '.join(str(item.get('reason') or '') for item in rejected[:6])}。"
                "请重新生成完整候选集，所有可回答问题的参考要点都必须绑定提供的证据 ID。"
            )
        if len(valid_cases) < minimum_count:
            raise ValueError(
                f"评测集草稿只有 {len(valid_cases)} 条问题通过证据验证，至少需要 {minimum_count} 条"
            )

        selected = _select_generated_cases(valid_cases, target_count)
        case_hash = _case_set_hash(selected)
        job_service.update_step(
            job_id,
            "validate",
            "succeeded",
            summary=f"准入 {len(selected)} 条问题，隔离 {len(rejected)} 条候选",
            items=rejected[:20],
        )
        job_service.update_step(job_id, "draft", "running", summary="正在整理评测集草稿")
        draft = {
            "status": "draft_ready",
            "approval_status": "pending",
            "topic": safe_topic,
            "topic_title": title,
            "eval_set_id": eval_set_id,
            "eval_set_version": version,
            "case_set_hash": case_hash,
            "knowledge_content_hash": knowledge_content_hash,
            "target_case_count": target_count,
            "case_count": len(selected),
            "answerable_count": sum(1 for case in selected if case.get("answerability") == "answerable"),
            "unanswerable_count": sum(1 for case in selected if case.get("answerability") == "unanswerable"),
            "document_count": len(documents),
            "evidence_count": len(catalog),
            "cases": selected,
            "rejected_candidates": rejected,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        job_service.update_step(
            job_id,
            "draft",
            "succeeded",
            summary=f"评测集草稿已生成，共 {len(selected)} 题",
        )
        emit_event(
            "knowledge_evaluation.case_set.generated",
            "evaluation_service",
            "Generated a topic-specific knowledge evaluation case set draft.",
            data={
                "job_id": job_id,
                "topic": safe_topic,
                "eval_set_id": eval_set_id,
                "case_count": len(selected),
                "rejected_count": len(rejected),
            },
        )
        return draft

    thread = threading.Thread(target=job_service._run, args=(job["job_id"], target), daemon=True)
    thread.start()
    return job_service.get(job["job_id"])


def approve_knowledge_qa_case_generation(job_id: str) -> dict[str, Any]:
    job = job_service.get(job_id)
    if job.get("job_type") != "knowledge_qa_case_generation":
        raise ValueError(f"job is not a knowledge QA case-generation job: {job_id}")
    if job.get("status") != "succeeded":
        raise ValueError(f"case-generation job is not ready for approval: {job.get('status')}")
    result = dict(job.get("result") or {})
    if result.get("approval_status") == "approved":
        return result
    topic = _validate_eval_topic(str(result.get("topic") or ""))
    cases = [dict(case) for case in result.get("cases") or [] if isinstance(case, dict)]
    if not cases:
        raise ValueError("case-generation job has no validated cases")

    previous_cases = (
        eval_case_store.load_cases("knowledge_qa", topic=topic)
        if eval_case_store.has_topic_cases("knowledge_qa", topic)
        else []
    )
    if previous_cases and eval_case_store.get_active_topic_version(topic) is None:
        previous_versions = []
        previous_ids = []
        for case in previous_cases:
            try:
                previous_versions.append(int(case.get("eval_set_version") or 0))
            except (TypeError, ValueError):
                pass
            if case.get("eval_set_id"):
                previous_ids.append(str(case["eval_set_id"]))
        previous_version = max(previous_versions or [max(1, int(result.get("eval_set_version") or 2) - 1)])
        eval_case_store.activate_topic_version(
            topic,
            previous_cases,
            eval_set_id=previous_ids[0] if previous_ids else f"legacy-{topic}-v{previous_version}",
            version=previous_version,
            case_set_hash=_case_set_hash(previous_cases),
            knowledge_content_hash="",
            approved_at="",
        )
    approved_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    eval_case_store.activate_topic_version(
        topic,
        cases,
        eval_set_id=str(result.get("eval_set_id") or ""),
        version=int(result.get("eval_set_version") or 1),
        case_set_hash=str(result.get("case_set_hash") or _case_set_hash(cases)),
        knowledge_content_hash=str(result.get("knowledge_content_hash") or _knowledge_content_hash(topic)),
        approved_at=approved_at,
    )
    approved = {
        **result,
        "approval_status": "approved",
        "approved_at": approved_at,
        "previous_case_count": len(previous_cases),
        "active_case_count": len(cases),
    }
    job_service.update_result(job_id, status="succeeded", result=approved, progress_message="approved")
    emit_event(
        "knowledge_evaluation.case_set.approved",
        "evaluation_service",
        "Approved and activated a topic-specific knowledge evaluation case set.",
        data={
            "job_id": job_id,
            "topic": topic,
            "eval_set_id": result.get("eval_set_id"),
            "case_count": len(cases),
            "previous_case_count": len(previous_cases),
        },
    )
    return approved


def _validate_eval_topic(topic: str) -> str:
    safe = str(topic or "").strip().replace("\\", "/").strip("/")
    if not safe or ".." in safe:
        raise ValueError("topic must identify an existing knowledge-base directory")
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"knowledge base topic not found: {safe}")
    if not list(path.glob("*.md")):
        raise ValueError(f"knowledge base topic has no Markdown documents: {safe}")
    return safe


def _build_evidence_catalog(topic: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    topic_path = safe_knowledge_path(topic)
    title = topic
    metadata_path = topic_path / "kb_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            title = str(metadata.get("title") or topic).strip() or topic
        except (OSError, json.JSONDecodeError):
            pass

    documents: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    for path in sorted(topic_path.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter_text(raw)
        document_title = str(frontmatter.get("title") or path.stem).strip() or path.stem
        blocks = _section_evidence_blocks(body, default_heading=document_title)
        documents.append(
            {
                "document_path": path.name,
                "title": document_title,
                "section_count": len({block[0] for block in blocks}),
                "evidence_count": min(len(blocks), 12),
            }
        )
        for heading, quote in blocks[:12]:
            normalized_quote = _normalize_evidence_quote(quote)
            if len(normalized_quote) < 40:
                continue
            digest_source = f"{path.name}\n{heading}\n{normalized_quote}".encode("utf-8")
            ref_id = f"evref_{hashlib.sha256(digest_source).hexdigest()[:12]}"
            catalog.append(
                {
                    "id": ref_id,
                    "document_path": path.name,
                    "document_title": document_title,
                    "section_title": heading,
                    "evidence_quote": quote,
                    "quote_hash": f"sha256:{hashlib.sha256(normalized_quote.encode('utf-8')).hexdigest()}",
                }
            )
            if len(catalog) >= 60:
                break
        if len(catalog) >= 60:
            break
    return title, documents, catalog


def _knowledge_content_hash(topic: str) -> str:
    topic_path = safe_knowledge_path(topic)
    digest = hashlib.sha256()
    for path in sorted(topic_path.glob("*.md"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _split_frontmatter_text(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    frontmatter: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("'\"")
    return frontmatter, parts[2]


def _section_evidence_blocks(body: str, *, default_heading: str) -> list[tuple[str, str]]:
    heading = default_heading
    section_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        if section_lines:
            sections.append((heading, "\n".join(section_lines).strip()))

    for line in body.splitlines():
        match = re.match(r"^#{1,4}\s+(.+?)\s*$", line)
        if match:
            flush()
            heading = match.group(1).strip() or default_heading
            section_lines = []
        else:
            section_lines.append(line)
    flush()

    blocks: list[tuple[str, str]] = []
    for section_title, section_body in sections:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_body) if part.strip()]
        for paragraph in paragraphs:
            if len(paragraph) <= 1200:
                blocks.append((section_title, paragraph))
                continue
            start = 0
            while start < len(paragraph):
                blocks.append((section_title, paragraph[start : start + 1000]))
                start += 1000
    return blocks


def _normalize_evidence_quote(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _target_case_count(document_count: int, evidence_count: int) -> int:
    if evidence_count < 8:
        return 8
    return min(12, max(8, int(document_count) + 4))


def _next_eval_set_version(topic: str) -> int:
    stored_versions = eval_case_store.list_topic_versions(topic)
    if stored_versions:
        return max(int(item.get("version") or 0) for item in stored_versions) + 1
    if not eval_case_store.has_topic_cases("knowledge_qa", topic):
        return 1
    versions = []
    for case in eval_case_store.load_cases("knowledge_qa", topic=topic):
        try:
            versions.append(int(case.get("eval_set_version") or 0))
        except (TypeError, ValueError):
            continue
    return max(versions or [0]) + 1


def _generate_eval_case_candidates(
    topic: str,
    topic_title: str,
    catalog: list[dict[str, Any]],
    *,
    target_count: int,
    feedback: str = "",
) -> list[dict[str, Any]]:
    try:
        from ...main import build_model
    except ImportError:  # pragma: no cover - script mode fallback
        from main import build_model

    evidence_payload = [
        {
            "id": item["id"],
            "document_path": item["document_path"],
            "document_title": item["document_title"],
            "section_title": item["section_title"],
            "quote": item["evidence_quote"],
        }
        for item in catalog
    ]
    prompt = {
        "task": "基于给定知识库证据，生成一套可重复回归的知识库问答评测集。只返回 JSON。",
        "topic": topic,
        "topic_title": topic_title,
        "candidate_count": min(15, target_count + 3),
        "rules": [
            "所有问题必须是自然、完整、自包含的中文问题，不得使用‘这个知识库’或‘上述材料’作为唯一指代。",
            "覆盖定义与定位、核心事实、使用方式或流程、跨章节综合、优势限制与场景、来源定位等不同题型。",
            "问题数量由材料决定，不得为了凑数量改写出语义重复的问题。",
            "answerability 只能是 answerable 或 unanswerable。",
            "answerable 问题必须提供 reference_answer 和 expected_points；每个 expected_point 必须绑定一条或多条给定 evidence ID。",
            "unanswerable 问题必须具体，并与材料中明确提到的公开资料缺口或覆盖边界相关；不得生成抽象的‘请回答一个未覆盖问题’。",
            "不得使用知识库外部事实补充参考答案，不得编造 evidence ID。",
            "reference_answer 用自然中文概括，不要求未来实际回答逐字复述。",
            "id 使用简短英文小写下划线；category 使用中文；difficulty 只能是 easy、medium 或 hard。",
        ],
        "schema": {
            "cases": [
                {
                    "id": "architecture_plugins",
                    "category": "核心架构",
                    "question": "完整中文问题",
                    "answerability": "answerable",
                    "difficulty": "medium",
                    "reference_answer": "参考回答",
                    "expected_points": [
                        {"id": "point_01", "text": "必须覆盖的事实要点", "evidence_refs": ["evref_xxx"]}
                    ],
                    "forbidden_claims": [],
                    "evidence_refs": ["evref_xxx"],
                }
            ]
        },
        "evidence_catalog": evidence_payload,
    }
    if feedback:
        prompt["retry_feedback"] = feedback
    model = build_model(streaming=False).bind(max_tokens=8192)
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "你是知识库评测集设计器。只能依据提供的证据目录工作，输出必须是合法 JSON。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
    )
    parsed = _json_object_from_text(_model_response_text(response))
    cases = parsed.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("评测集生成模型没有返回 cases 数组")
    return [dict(case) for case in cases if isinstance(case, dict)]


def _model_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    return str(content or "")


def _json_object_from_text(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("评测集生成模型返回了无法解析的 JSON")


def _research_report_sections(summary: dict[str, Any]) -> list[dict[str, str]]:
    child_job_id = str(summary.get("job_id") or "").strip()
    if not child_job_id:
        return []
    try:
        child = job_service.get(child_job_id)
    except ValueError:
        return []
    result = child.get("result") if isinstance(child.get("result"), dict) else {}
    sections: list[dict[str, str]] = []
    total_chars = 0
    for index, document in enumerate(result.get("documents") or []):
        if not isinstance(document, dict):
            continue
        title = str(document.get("title") or f"章节 {index + 1}").strip()
        section_id = str(document.get("section_id") or document.get("id") or title).strip()
        content = str(document.get("content") or "").strip()
        if not content:
            continue
        remaining = max(0, 48000 - total_chars)
        if remaining <= 0:
            break
        content = content[: min(12000, remaining)]
        total_chars += len(content)
        sections.append({"section_id": section_id, "title": title, "content": content})
    return sections


def _normalize_research_content_review(
    raw: dict[str, Any],
    *,
    valid_section_ids: set[str],
) -> dict[str, Any]:
    raw_dimensions = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    dimensions: dict[str, dict[str, Any]] = {}
    invalid_section_ids: list[str] = []
    for dimension in RESEARCH_CONTENT_REVIEW_DIMENSIONS:
        item = raw_dimensions.get(dimension) if isinstance(raw_dimensions.get(dimension), dict) else {}
        label = str(item.get("label") or "未评估").strip()
        if label not in RESEARCH_CONTENT_REVIEW_LABELS:
            label = "未评估"
        section_ids = []
        for value in item.get("section_ids") or []:
            section_id = str(value or "").strip()
            if not section_id:
                continue
            if section_id in valid_section_ids:
                section_ids.append(section_id)
            else:
                invalid_section_ids.append(section_id)
        dimensions[dimension] = {
            "label": label,
            "reason": re.sub(r"\s+", " ", str(item.get("reason") or "")).strip()[:500],
            "section_ids": list(dict.fromkeys(section_ids)),
        }

    suggestions: list[dict[str, Any]] = []
    for item in raw.get("suggestions") or []:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id") or "").strip()
        if section_id and section_id not in valid_section_ids:
            invalid_section_ids.append(section_id)
            continue
        reason = re.sub(r"\s+", " ", str(item.get("reason") or "")).strip()
        if not reason:
            continue
        severity = str(item.get("severity") or "low").strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "low"
        suggestions.append(
            {
                "section_id": section_id or None,
                "severity": severity,
                "reason": reason[:500],
            }
        )
        if len(suggestions) >= 8:
            break

    has_advice = bool(suggestions) or any(
        item.get("label") in {"一般", "需改进"} for item in dimensions.values()
    )
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence"))))
    except (TypeError, ValueError):
        confidence = None
    return {
        "status": "completed_with_advice" if has_advice else "completed",
        "blocking": False,
        "dimensions": dimensions,
        "suggestions": suggestions,
        "confidence": confidence,
        "diagnostics": {"invalid_section_ids": list(dict.fromkeys(invalid_section_ids))},
    }


def _review_research_content(case: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    sections = _research_report_sections(summary)
    if not sections:
        return {
            "status": "content_review_unavailable",
            "blocking": False,
            "reason": "调研任务没有可用于内容评审的章节正文。",
        }
    if os.getenv("RESEARCH_E2E_CONTENT_REVIEW", "1").strip().lower() in {"0", "false", "no", "off"}:
        return {
            "status": "content_review_unavailable",
            "blocking": False,
            "reason": "内容质量建议评审已通过配置关闭。",
        }
    try:
        from ...main import build_model
    except ImportError:  # pragma: no cover - script mode fallback
        from main import build_model

    valid_section_ids = {section["section_id"] for section in sections}
    prompt = {
        "task": "从用户阅读价值角度评审调研报告。只返回合法 JSON。本评审只提供建议，不决定审批、落库或生产质量门。",
        "user_request": case.get("user_request"),
        "core_intents": case.get("core_intents") or [],
        "optional_topics": case.get("optional_topics") or [],
        "scope_notes": case.get("scope_notes") or "",
        "rules": [
            "判断报告是否回答用户核心问题，不要求穷尽所有可能细节。",
            "optional_topics 缺失不得判为缺陷，除非正文对该内容作出明显错误承诺。",
            "不要要求固定标题或固定文章模板；根据实际主题评价结构和可读性。",
            "事实安全和引用真实性由生产质量门负责，本次不要重复裁决事实真伪。",
            "每个判断必须给出简短理由；章节级建议必须使用提供的 section_id。",
            "label 只能是 良好、一般、需改进、未评估；severity 只能是 low、medium、high。",
        ],
        "schema": {
            "dimensions": {
                dimension: {"label": "良好", "reason": "判断理由", "section_ids": ["section_id"]}
                for dimension in RESEARCH_CONTENT_REVIEW_DIMENSIONS
            },
            "suggestions": [
                {"section_id": "section_id 或空字符串", "severity": "medium", "reason": "可操作建议"}
            ],
            "confidence": 0.8,
        },
        "sections": sections,
    }
    try:
        model = build_model(streaming=False).bind(max_tokens=4096)
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": "你是调研报告内容质量评审员。评审结果是非阻断建议，必须输出 JSON。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ]
        )
        parsed = _json_object_from_text(_model_response_text(response))
        review = _normalize_research_content_review(parsed, valid_section_ids=valid_section_ids)
        metadata = getattr(response, "response_metadata", None) or {}
        review["judge"] = {
            "status": "succeeded",
            "model": metadata.get("model_name") or metadata.get("model") or os.getenv("MODEL_NAME"),
            "provider": metadata.get("harness_provider") or "primary",
        }
        return review
    except Exception as exc:
        return {
            "status": "content_review_unavailable",
            "blocking": False,
            "reason": re.sub(r"\s+", " ", str(exc) or type(exc).__name__)[:500],
            "judge": {"status": "unavailable"},
        }


def _validate_generated_cases(
    raw_cases: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    eval_set_id: str,
    version: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_by_id = {str(item["id"]): item for item in catalog}
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_ids: set[str] = set()
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for index, raw in enumerate(raw_cases, start=1):
        question = re.sub(r"\s+", " ", str(raw.get("question") or "")).strip()
        normalized_question = re.sub(r"[\W_]+", "", question.lower())
        if len(question) < 8:
            rejected.append({"index": index, "reason": "问题为空或过短"})
            continue
        if normalized_question in seen_questions:
            rejected.append({"index": index, "question": question, "reason": "问题语义文本重复"})
            continue
        answerability = str(raw.get("answerability") or "answerable").strip().lower()
        if answerability not in {"answerable", "unanswerable"}:
            rejected.append({"index": index, "question": question, "reason": "answerability 非法"})
            continue
        reference_answer = re.sub(r"\s+", " ", str(raw.get("reference_answer") or "")).strip()
        if len(reference_answer) < 20:
            rejected.append({"index": index, "question": question, "reason": "参考回答为空或过短"})
            continue

        case_id = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id") or "").strip().lower()).strip("_")
        case_id = case_id or f"case_{index:02d}"
        if case_id in seen_ids:
            case_id = f"{case_id}_{index:02d}"
        expected_points: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        point_error = ""
        for point_index, point in enumerate(raw.get("expected_points") or [], start=1):
            if not isinstance(point, dict):
                continue
            point_text = re.sub(r"\s+", " ", str(point.get("text") or "")).strip()
            point_refs = [str(item) for item in point.get("evidence_refs") or [] if str(item) in catalog_by_id]
            if not point_text:
                continue
            if answerability == "answerable" and not point_refs:
                point_error = f"参考要点未绑定有效证据：{point_text[:60]}"
                break
            point_id = re.sub(r"[^a-z0-9_]+", "_", str(point.get("id") or "").lower()).strip("_")
            expected_points.append(
                {
                    "id": point_id or f"point_{point_index:02d}",
                    "text": point_text,
                    "evidence_refs": list(dict.fromkeys(point_refs)),
                }
            )
            evidence_ids.extend(point_refs)
        if point_error:
            rejected.append({"index": index, "question": question, "reason": point_error})
            continue
        if answerability == "answerable" and not expected_points:
            rejected.append({"index": index, "question": question, "reason": "可回答问题缺少参考要点"})
            continue

        raw_refs = raw.get("evidence_refs") or []
        for item in raw_refs:
            ref_id = str(item.get("id") if isinstance(item, dict) else item)
            if ref_id in catalog_by_id:
                evidence_ids.append(ref_id)
        evidence_ids = list(dict.fromkeys(evidence_ids))
        if answerability == "answerable" and not evidence_ids:
            rejected.append({"index": index, "question": question, "reason": "可回答问题没有有效标准证据"})
            continue

        difficulty = str(raw.get("difficulty") or "medium").strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        forbidden_claims = [
            re.sub(r"\s+", " ", str(item)).strip()
            for item in raw.get("forbidden_claims") or []
            if str(item).strip()
        ][:8]
        valid.append(
            {
                "id": case_id,
                "category": str(raw.get("category") or "通用问题").strip() or "通用问题",
                "question": question,
                "answerability": answerability,
                "difficulty": difficulty,
                "reference_answer": reference_answer,
                "expected_points": expected_points,
                "forbidden_claims": forbidden_claims,
                "evidence_refs": [dict(catalog_by_id[ref_id]) for ref_id in evidence_ids],
                "eval_set_id": eval_set_id,
                "eval_set_version": version,
                "generation": {"source": "ai_generated", "generated_at": generated_at},
                # Old deterministic scorer remains compatible until the semantic judge phase.
                "must_include_all": [],
                "must_not_include": forbidden_claims,
                "min_answer_chars": 80 if answerability == "unanswerable" else 120,
                "expect_insufficient": answerability == "unanswerable",
                "failure_type_hint": "knowledge_gap" if answerability == "unanswerable" else "generation",
            }
        )
        seen_questions.add(normalized_question)
        seen_ids.add(case_id)
    return valid, rejected


def _select_generated_cases(cases: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        buckets.setdefault(str(case.get("category") or "通用问题"), []).append(case)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count and any(buckets.values()):
        for category in list(buckets):
            if buckets[category] and len(selected) < target_count:
                selected.append(buckets[category].pop(0))
    return selected


def _case_set_hash(cases: list[dict[str, Any]]) -> str:
    stable = [
        {
            "id": case.get("id"),
            "question": case.get("question"),
            "answerability": case.get("answerability"),
            "expected_points": case.get("expected_points"),
            "evidence_ids": [item.get("id") for item in case.get("evidence_refs") or [] if isinstance(item, dict)],
        }
        for case in cases
    ]
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def create_research_probe_eval_job(
    case_ids: list[str] | None = None,
    *,
    strategy: str = "hv_analysis",
    depth: str = "normal",
    case_interval_seconds: float = 30.0,
) -> dict:
    from scripts.run_research_e2e_probe import run_probe, summary_failed
    from scripts.run_research_multi_topic_probe import (
        _aggregate,
        _case_args,
        _error_row,
        _flatten_case_result,
        _write_reports,
        _case_set_version,
        classify_summary,
    )

    cases = eval_case_store.load_cases("research_probe")
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case["id"] in wanted]
    if not cases:
        raise ValueError("no research_probe cases matched case_ids")

    base_args = argparse.Namespace(
        strategy=strategy,
        depth=depth,
        force_depth=None,
        force_max_rounds=None,
        max_rounds=_production_research_max_rounds(),
        timeout_seconds=max(3600, int(os.getenv("RESEARCH_E2E_TIMEOUT_SECONDS", "3600") or 3600)),
        poll_interval=5.0,
        via_api=False,
        verbose=False,
    )

    def target(input_data: dict[str, Any]) -> dict:
        job_id = target.job_id  # type: ignore[attr-defined]
        started = time.time()
        rows: list[dict[str, Any]] = []
        full_results: list[dict[str, Any]] = []
        child_job_ids: list[str] = []

        def parent_cancelled() -> bool:
            return job_service.get(job_id).get("status") == "cancelled"

        def register_child(child_job_id: str) -> None:
            if child_job_id not in child_job_ids:
                child_job_ids.append(child_job_id)
            job_service.update_result(
                job_id,
                result={
                    "evaluation_type": "research_probe",
                    "active_child_job_id": child_job_id,
                    "child_job_ids": list(child_job_ids),
                    "rows": list(rows),
                },
                progress_message=f"正在运行子调研 {child_job_id}",
            )

        for index, case in enumerate(cases, start=1):
            if parent_cancelled():
                break
            job_service.update_step(job_id, case["id"], "running", summary=f"运行中：{case['title']}")
            case_args = _case_args(case, base_args)
            case_args.parent_job_id = job_id
            case_args.on_child_created = register_child
            try:
                summary = run_probe(case_args)
                failed = summary_failed(summary)
                row = _flatten_case_result(case, summary, failed)
                classification = classify_summary(summary)
                if classification["execution_status"] == "succeeded":
                    row["content_review"] = _review_research_content(case, summary)
                else:
                    row["content_review"] = {"status": "not_run", "blocking": False}
            except Exception as exc:
                failed = True
                summary = {"error": str(exc)}
                row = _error_row(case, {"error": str(exc)})
            rows.append(row)
            full_results.append(
                {
                    "case": case,
                    "summary": summary,
                    "outcome": row.get("outcome"),
                    "content_review": row.get("content_review"),
                }
            )
            current_parent = job_service.get(job_id)
            if current_parent.get("status") == "cancelled":
                break
            execution_status = row.get("execution_status")
            production_status = row.get("production_gate_status")
            if execution_status == "failed":
                step_status = "failed"
                step_summary = f"执行失败：{case['title']}"
            elif execution_status == "cancelled":
                step_status = "interrupted"
                step_summary = f"已取消：{case['title']}"
            elif production_status == "failed":
                step_status = "succeeded"
                step_summary = f"执行完成，生产质量门未通过：{case['title']}"
            else:
                step_status = "succeeded"
                step_summary = f"执行完成，生产质量门通过：{case['title']}"
            job_service.update_step(
                job_id,
                case["id"],
                step_status,
                summary=step_summary,
            )
            job_service.update_result(
                job_id,
                result={
                    "evaluation_type": "research_probe",
                    "active_child_job_id": None,
                    "child_job_ids": list(child_job_ids),
                    "rows": list(rows),
                },
            )
            if index < len(cases) and case_interval_seconds > 0:
                sleep_deadline = time.time() + case_interval_seconds
                while time.time() < sleep_deadline:
                    if parent_cancelled():
                        break
                    time.sleep(min(1.0, max(0.0, sleep_deadline - time.time())))
                if parent_cancelled():
                    break
        payload = {
            "schema_version": 2,
            "evaluation_type": "research_probe",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "configuration": {
                "case_set_version": _case_set_version(cases),
                "strategy": strategy,
                "depth": depth,
                "timeout_seconds": base_args.timeout_seconds,
                "case_interval_seconds": case_interval_seconds,
            },
            "aggregate": _aggregate(rows, started),
            "rows": rows,
            "results": full_results,
            "child_job_ids": child_job_ids,
        }
        report_paths = _write_reports(RESEARCH_PROBE_OUTPUT_DIR, payload)
        return {
            "evaluation_type": "research_probe",
            "aggregate": payload["aggregate"],
            "rows": rows,
            "results": full_results,
            "child_job_ids": child_job_ids,
            "report_paths": report_paths,
        }

    job = job_service.create(
        job_type="research_probe_evaluation",
        input_data={"case_ids": case_ids, "strategy": strategy, "depth": depth},
        target=None,
    )
    target.job_id = job["job_id"]  # type: ignore[attr-defined]
    job_service.set_steps(job["job_id"], [{"id": case["id"], "title": case["title"]} for case in cases])

    import threading

    thread = threading.Thread(target=job_service._run, args=(job["job_id"], target), daemon=True)
    thread.start()
    return job_service.get(job["job_id"])


def create_single_url_brief_eval_job(case_ids: list[str] | None = None) -> dict:
    from scripts.run_single_url_brief_eval import _markdown_report, run_case

    cases = eval_case_store.load_cases("single_url_brief")
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case["id"] in wanted]
    if not cases:
        raise ValueError("no single_url_brief cases matched case_ids")

    def target(input_data: dict[str, Any]) -> dict:
        job_id = target.job_id  # type: ignore[attr-defined]
        rows: list[dict[str, Any]] = []
        for case in cases:
            job_service.update_step(job_id, case["id"], "running", summary=f"运行中：{case['id']}")
            row = run_case(case, verbose=False)
            rows.append(row)
            job_service.update_step(
                job_id,
                case["id"],
                "succeeded" if row["passed"] else "failed",
                summary=f"{'通过' if row['passed'] else '失败'}：{case['id']}",
            )
        SINGLE_URL_BRIEF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        md_path = SINGLE_URL_BRIEF_OUTPUT_DIR / f"single-url-brief-eval-{stamp}.md"
        md_path.write_text(_markdown_report(rows), encoding="utf-8")
        return {
            "total_cases": len(rows),
            "passed_cases": sum(1 for row in rows if row["passed"]),
            "rows": rows,
            "report_paths": {"markdown": str(md_path)},
        }

    job = job_service.create(
        job_type="single_url_brief_evaluation",
        input_data={"case_ids": case_ids},
        target=None,
    )
    target.job_id = job["job_id"]  # type: ignore[attr-defined]
    job_service.set_steps(job["job_id"], [{"id": case["id"], "title": case.get("url", case["id"])} for case in cases])

    import threading

    thread = threading.Thread(target=job_service._run, args=(job["job_id"], target), daemon=True)
    thread.start()
    return job_service.get(job["job_id"])


def get_eval_job(job_id: str) -> dict:
    return job_service.get(job_id)


def list_eval_cases(eval_type: str, *, topic: str | None = None) -> dict:
    has_topic_cases = eval_case_store.has_topic_cases(eval_type, topic)
    cases = eval_case_store.load_cases(eval_type, topic=topic)
    versions = []
    eval_set_ids = []
    for case in cases:
        try:
            versions.append(int(case.get("eval_set_version") or 0))
        except (TypeError, ValueError):
            pass
        eval_set_id = str(case.get("eval_set_id") or "").strip()
        if eval_set_id:
            eval_set_ids.append(eval_set_id)
    response = {
        "eval_type": eval_type,
        "topic": topic,
        "scopes": eval_case_store.list_scopes(eval_type),
        "scope": "topic" if has_topic_cases else "default",
        "has_topic_cases": has_topic_cases,
        "eval_set_id": eval_set_ids[0] if eval_set_ids else None,
        "eval_set_version": max(versions or [0]) or None,
        "cases": cases,
    }
    if eval_type == "knowledge_qa" and topic and has_topic_cases:
        active = eval_case_store.get_active_topic_version(topic)
        current_hash = _knowledge_content_hash(topic)
        stored_hash = str((active or {}).get("knowledge_content_hash") or "")
        current_case_hash = _case_set_hash(cases)
        stored_case_hash = str((active or {}).get("case_set_hash") or "")
        response.update(
            {
                "versions": eval_case_store.list_topic_versions(topic),
                "knowledge_content_hash": current_hash,
                "eval_set_content_hash": stored_hash or None,
                "is_stale": bool(stored_hash and stored_hash != current_hash),
                "case_set_modified": bool(stored_case_hash and stored_case_hash != current_case_hash),
                "stale_reason": (
                    "知识库 Markdown 内容已在当前评测集启用后发生变化，请检查标准证据并生成新版本。"
                    if stored_hash and stored_hash != current_hash
                    else ""
                ),
            }
        )
    return response


def switch_knowledge_qa_case_version(topic: str, version: int) -> dict[str, Any]:
    safe_topic = _validate_eval_topic(topic)
    selected = eval_case_store.switch_topic_version(safe_topic, version)
    current_hash = _knowledge_content_hash(safe_topic)
    stored_hash = str(selected.get("knowledge_content_hash") or "")
    return {
        **selected,
        "topic": safe_topic,
        "active": True,
        "is_stale": bool(stored_hash and stored_hash != current_hash),
        "stale_reason": (
            "该历史评测集对应的知识库内容与当前版本不同，运行结果应按历史基准解释。"
            if stored_hash and stored_hash != current_hash
            else ""
        ),
    }


def upsert_eval_case(eval_type: str, case: dict[str, Any], *, topic: str | None = None) -> dict:
    return eval_case_store.upsert_case(eval_type, case, topic=topic)


def delete_eval_case(eval_type: str, case_id: str, *, topic: str | None = None) -> None:
    eval_case_store.delete_case(eval_type, case_id, topic=topic)


def list_reports(topic: str | None = None, limit: int = 50) -> dict:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for path in sorted(EVALUATION_DIR.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
        summary = _summary_from_report(path)
        structured_path = path.with_suffix(".json")
        structured: dict[str, Any] = {}
        if structured_path.exists():
            try:
                structured = json.loads(structured_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                structured = {}
            if not isinstance(structured, dict):
                structured = {}
            if isinstance(structured, dict) and isinstance(structured.get("summary"), dict):
                summary = structured["summary"]
        structured_scope = structured.get("scope") if isinstance(structured.get("scope"), dict) else None
        report_topic = (
            knowledge_scope_service.scope_case_key(
                str(structured_scope.get("type") or ""),
                str(structured_scope.get("id") or ""),
            )
            if structured_scope
            else _topic_from_report_name(path.stem)
        )
        if topic and report_topic != topic:
            continue
        reports.append(
            {
                "report_id": path.stem,
                "topic": report_topic,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "summary": summary,
                "backend": structured.get("backend"),
                "mode": structured.get("mode"),
                "eval_set_id": structured.get("eval_set_id"),
                "eval_set_version": structured.get("eval_set_version"),
                "knowledge_content_hash": structured.get("knowledge_content_hash"),
                "knowledge_content_stale": structured.get("knowledge_content_stale", False),
                "case_set_modified": structured.get("case_set_modified", False),
                "scope": structured_scope,
            }
        )
        if len(reports) >= max(1, min(limit, 200)):
            break
    return {"reports": reports}


def list_research_probe_reports(limit: int = 50) -> dict[str, Any]:
    RESEARCH_PROBE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for path in sorted(
        RESEARCH_PROBE_OUTPUT_DIR.glob("research-multi-topic-e2e-*.md"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        structured: dict[str, Any] = {}
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                parsed = json.loads(json_path.read_text(encoding="utf-8"))
                structured = parsed if isinstance(parsed, dict) else {}
            except (OSError, json.JSONDecodeError):
                structured = {}
        rows = _normalize_research_probe_rows(structured.get("rows") or [])
        aggregate = _normalize_research_probe_aggregate(structured.get("aggregate") or {}, rows)
        reports.append(
            {
                "report_id": path.stem,
                "evaluation_type": "research_probe",
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "created_at": structured.get("finished_at")
                or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "started_at": structured.get("started_at"),
                "schema_version": structured.get("schema_version", 1),
                "configuration": structured.get("configuration") or {},
                "summary": aggregate,
            }
        )
        if len(reports) >= max(1, min(limit, 200)):
            break
    return {"reports": reports}


def get_research_probe_report(report_id: str) -> dict[str, Any]:
    safe = Path(str(report_id)).name
    if safe.endswith(".md"):
        safe = safe[:-3]
    if not safe.startswith("research-multi-topic-e2e-"):
        raise ValueError(f"invalid research probe report id: {report_id}")
    path = RESEARCH_PROBE_OUTPUT_DIR / f"{safe}.md"
    if not path.exists():
        raise ValueError(f"document not found: {report_id}")
    response: dict[str, Any] = {
        "report_id": safe,
        "evaluation_type": "research_probe",
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "content": path.read_text(encoding="utf-8"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "summary": {},
        "rows": [],
        "results": [],
    }
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return response
    try:
        structured = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return response
    if not isinstance(structured, dict):
        return response
    rows = _normalize_research_probe_rows(structured.get("rows") or [])
    aggregate = _normalize_research_probe_aggregate(structured.get("aggregate") or {}, rows)
    response.update(
        {
            "created_at": structured.get("finished_at") or response["created_at"],
            "started_at": structured.get("started_at"),
            "finished_at": structured.get("finished_at"),
            "schema_version": structured.get("schema_version", 1),
            "configuration": structured.get("configuration") or {},
            "summary": aggregate,
            "rows": rows,
            "results": structured.get("results") or [],
            "child_job_ids": structured.get("child_job_ids") or [],
        }
    )
    return response


def _normalize_research_probe_rows(rows: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        quality_status = str(row.get("quality_gate_status") or "")
        validation_status = str(row.get("validation_status") or "")
        if not row.get("execution_status"):
            execution_failed = quality_status == "error" or validation_status == "error" or not row.get("job_id")
            row["execution_status"] = "failed" if execution_failed else "succeeded"
        if not row.get("production_gate_status"):
            if row["execution_status"] != "succeeded":
                row["production_gate_status"] = "not_evaluated"
            else:
                row["production_gate_status"] = "passed" if not row.get("failed") else "failed"
        if not row.get("outcome"):
            if row["execution_status"] == "failed":
                row["outcome"] = "execution_failed"
            elif row["production_gate_status"] == "failed":
                row["outcome"] = "pipeline_gate_failed"
            else:
                row["outcome"] = "completed"
        if row.get("hybrid_scored_chunks") is None:
            row["hybrid_scored_chunks"] = (
                int(row.get("selected_chunks") or 0) if row.get("hybrid_score_range") else int(row.get("hybrid_embedding_chunks") or 0)
            )
        if not isinstance(row.get("content_review"), dict):
            row["content_review"] = {"status": "not_run", "blocking": False}
        normalized.append(row)
    return normalized


def _normalize_research_probe_aggregate(
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output = dict(aggregate)
    total = len(rows) or int(output.get("total_cases") or 0)
    execution_succeeded = sum(1 for row in rows if row.get("execution_status") == "succeeded")
    execution_failed = sum(1 for row in rows if row.get("execution_status") == "failed")
    cancelled = sum(1 for row in rows if row.get("execution_status") == "cancelled")
    production_passed = sum(1 for row in rows if row.get("production_gate_status") == "passed")
    production_failed = sum(1 for row in rows if row.get("production_gate_status") == "failed")
    output.update(
        {
            "total_cases": total,
            "execution_succeeded_cases": execution_succeeded,
            "execution_failed_cases": execution_failed,
            "cancelled_cases": cancelled,
            "production_gate_passed_cases": production_passed,
            "production_gate_failed_cases": production_failed,
            "production_gate_not_evaluated_cases": total - production_passed - production_failed,
            "production_gate_pass_rate": round(production_passed / execution_succeeded, 3)
            if execution_succeeded
            else 0,
            "content_review_completed_cases": sum(
                1
                for row in rows
                if (row.get("content_review") or {}).get("status") in {"completed", "completed_with_advice"}
            ),
            "content_review_advisory_cases": sum(
                1 for row in rows if (row.get("content_review") or {}).get("status") == "completed_with_advice"
            ),
            "content_review_unavailable_cases": sum(
                1
                for row in rows
                if (row.get("content_review") or {}).get("status") == "content_review_unavailable"
            ),
            "hybrid_scored_total": sum(int(row.get("hybrid_scored_chunks") or 0) for row in rows),
            "cited_claim_total": sum(int(row.get("cited_claim_count") or 0) for row in rows),
            "supported_cited_claim_total": sum(
                int(row.get("supported_cited_claim_count") or 0) for row in rows
            ),
            "unsupported_cited_claim_total": sum(
                int(row.get("unsupported_cited_claim_count") or 0) for row in rows
            ),
            "core_unsupported_cited_claim_total": sum(
                int(row.get("core_unsupported_cited_claim_count") or 0) for row in rows
            ),
            "unsupported_marker_total": sum(
                int(row.get("unsupported_marker_count") or 0) for row in rows
            ),
            "semantic_review_unavailable_total": sum(
                int(row.get("semantic_review_unavailable_count") or 0) for row in rows
            ),
            "atomic_boundary_case_count": sum(
                str(row.get("entailment_payload_profile") or "") == "atomic_boundary_v2"
                for row in rows
            ),
            "provider_fallback_case_count": sum(
                int(row.get("provider_fallback_triggered_count") or 0) > 0 for row in rows
            ),
            "unresolved_internal_claim_ref_total": sum(
                int(row.get("unresolved_internal_claim_ref_count") or 0) for row in rows
            ),
            "unresolved_internal_claim_ref_case_count": sum(
                int(row.get("unresolved_internal_claim_ref_count") or 0) > 0 for row in rows
            ),
        }
    )
    cited_claim_total = int(output.get("cited_claim_total") or 0)
    supported_cited_claim_total = int(output.get("supported_cited_claim_total") or 0)
    output["cited_claim_support_rate"] = (
        round(supported_cited_claim_total / cited_claim_total, 4)
        if cited_claim_total
        else None
    )
    return output


def compare_reports(report_id: str, baseline_report_id: str) -> dict[str, Any]:
    current = _report_for_comparison(report_id)
    baseline = _report_for_comparison(baseline_report_id)
    if current.get("topic") != baseline.get("topic"):
        raise ValueError("reports must belong to the same knowledge-base topic")
    current_summary = current.get("summary") or {}
    baseline_summary = baseline.get("summary") or {}
    metric_paths = {
        "基础通过率": ("pass_rate",),
        "平均分": ("average_score",),
        "标准证据命中率": ("retrieval_quality", "metrics", "evidence_recall"),
        "参考要点覆盖度": ("retrieval_quality", "metrics", "point_coverage"),
        "前 3 条关键证据命中率": ("retrieval_quality", "metrics", "hit_at_3"),
        "答案正确性": ("semantic_quality", "scores", "correctness"),
        "答案完整性": ("semantic_quality", "scores", "completeness"),
        "答案忠实度": ("semantic_quality", "scores", "faithfulness"),
    }
    metrics = []
    for label, path in metric_paths.items():
        current_value = _nested_number(current_summary, path)
        baseline_value = _nested_number(baseline_summary, path)
        metrics.append(
            {
                "key": path[-1],
                "label": label,
                "current": current_value,
                "baseline": baseline_value,
                "delta": round(current_value - baseline_value, 4)
                if current_value is not None and baseline_value is not None
                else None,
            }
        )
    return {
        "topic": current.get("topic"),
        "current_report_id": current.get("report_id"),
        "baseline_report_id": baseline.get("report_id"),
        "current_created_at": current.get("created_at"),
        "baseline_created_at": baseline.get("created_at"),
        "configuration_changed": any(
            current.get(key) != baseline.get(key)
            for key in ("backend", "mode", "eval_set_version", "case_set_hash", "knowledge_content_hash")
        ),
        "current_configuration": _report_configuration(current),
        "baseline_configuration": _report_configuration(baseline),
        "metrics": metrics,
    }


def _report_for_comparison(report_id: str) -> dict[str, Any]:
    safe = Path(str(report_id)).name
    if safe.endswith(".md"):
        safe = safe[:-3]
    path = EVALUATION_DIR / f"{safe}.md"
    if not path.exists():
        raise ValueError(f"document not found: {report_id}")
    payload: dict[str, Any] = {
        "report_id": safe,
        "topic": _topic_from_report_name(safe),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "summary": _summary_from_report(path),
    }
    structured_path = path.with_suffix(".json")
    if structured_path.exists():
        try:
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            structured = {}
        if isinstance(structured, dict):
            for key in (
                "created_at",
                "summary",
                "backend",
                "mode",
                "eval_set_id",
                "eval_set_version",
                "case_set_hash",
                "knowledge_content_hash",
                "knowledge_content_stale",
            ):
                if structured.get(key) is not None:
                    payload[key] = structured[key]
    return payload


def _nested_number(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _report_configuration(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": report.get("backend"),
        "mode": report.get("mode"),
        "eval_set_id": report.get("eval_set_id"),
        "eval_set_version": report.get("eval_set_version"),
        "case_set_hash": report.get("case_set_hash"),
        "knowledge_content_hash": report.get("knowledge_content_hash"),
        "knowledge_content_stale": report.get("knowledge_content_stale", False),
    }


def get_report(report_id: str) -> dict:
    safe = Path(str(report_id)).name
    if safe.endswith(".md"):
        safe = safe[:-3]
    path = EVALUATION_DIR / f"{safe}.md"
    if not path.exists():
        raise ValueError(f"document not found: {report_id}")
    topic = _topic_from_report_name(safe)
    response = {
        "report_id": safe,
        "topic": topic,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "content": path.read_text(encoding="utf-8"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "summary": _summary_from_report(path),
    }
    structured_path = path.with_suffix(".json")
    if structured_path.exists():
        try:
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            structured = {}
        if isinstance(structured, dict):
            response.update(
                {
                    key: structured[key]
                    for key in (
                        "created_at",
                        "summary",
                        "cases",
                        "recommended_actions",
                        "run_id",
                        "job_id",
                        "backend",
                        "mode",
                        "eval_set_id",
                        "eval_set_version",
                        "case_set_hash",
                        "eval_set_case_hash",
                        "case_set_modified",
                        "knowledge_content_hash",
                        "knowledge_content_stale",
                    )
                    if structured.get(key) is not None
                }
            )
            structured_scope = structured.get("scope") if isinstance(structured.get("scope"), dict) else None
            if structured_scope:
                response["scope"] = structured_scope
                response["topic"] = knowledge_scope_service.scope_case_key(
                    str(structured_scope.get("type") or ""),
                    str(structured_scope.get("id") or ""),
                )

    job = _job_for_report(response["path"])
    if job:
        result = job.get("result") or {}
        response["job_id"] = job.get("job_id")
        if result.get("run_id"):
            response["run_id"] = result.get("run_id")
        if not response.get("cases"):
            response["cases"] = _enrich_report_cases(result.get("results") or [], topic)
        if not response.get("recommended_actions") and response.get("cases"):
            response["recommended_actions"] = _recommended_report_actions(response["cases"])
        if result.get("summary"):
            response["summary"] = result["summary"]
    if response.get("cases"):
        # Recompute from structured case state so reports created before a status-rule
        # correction do not keep showing stale or contradictory action text.
        response["recommended_actions"] = _recommended_report_actions(response["cases"])
    return response


def _job_for_report(report_path: str) -> dict[str, Any] | None:
    normalized = str(report_path or "").replace("\\", "/")
    for job in job_service.list(job_type="knowledge_qa_evaluation", limit=200):
        result_path = str((job.get("result") or {}).get("report_path") or "").replace("\\", "/")
        if result_path == normalized:
            return job
    return None


def _enrich_report_cases(rows: list[Any], topic: str) -> list[dict[str, Any]]:
    case_catalog = {
        str(case.get("id") or ""): case
        for case in eval_case_store.load_cases("knowledge_qa", topic=topic)
        if isinstance(case, dict)
    }
    cases: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        source_case = case_catalog.get(str(item.get("id") or ""), {})
        item["question"] = str(item.get("question") or source_case.get("question") or "")
        item["answer"] = str(item.get("answer") or "")
        item["category"] = str(item.get("category") or source_case.get("category") or "未分类")
        item["backend"] = str(item.get("backend") or item.get("actual_backend") or "auto")
        item["mode"] = str(item.get("mode") or "mix")
        item["failed_checks"] = list(item.get("failed_checks") or [])
        item["likely_failure_type"] = str(item.get("likely_failure_type") or "none")
        item["recommended_action"] = str(item.get("recommended_action") or "")
        cases.append(item)
    return cases


def _recommended_report_actions(cases: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for item in cases:
        action = str(item.get("recommended_action") or "").strip()
        if not item.get("passed") and action and action not in actions:
            actions.append(action)
        semantic = item.get("semantic_evaluation") or {}
        semantic_action = str(semantic.get("recommended_action") or "").strip()
        if semantic.get("quality_status") == "needs_review" and semantic_action and semantic_action not in actions:
            actions.append(semantic_action)
        retrieval = item.get("retrieval_evaluation") or {}
        retrieval_action = str(retrieval.get("recommended_action") or "").strip()
        if retrieval.get("quality_status") == "needs_review" and retrieval_action and retrieval_action not in actions:
            actions.append(retrieval_action)
    if actions:
        return actions
    unavailable_count = sum(
        1
        for item in cases
        if (item.get("semantic_evaluation") or {}).get("status") == "unavailable"
    )
    if unavailable_count:
        return [
            f"{unavailable_count} 个用例仅完成基础规则检查，AI 语义评测未完成；模型恢复后请重新运行评估。"
        ]
    retrieval_semantic_unavailable = sum(
        1
        for item in cases
        if (item.get("retrieval_evaluation") or {}).get("semantic_judge_status") == "unavailable"
    )
    if retrieval_semantic_unavailable:
        return [
            f"{retrieval_semantic_unavailable} 个用例已完成确定性检索检查，但检索语义评测未完成；模型恢复后请重新运行评估。"
        ]
    retrieval_unavailable = sum(
        1
        for item in cases
        if (item.get("retrieval_evaluation") or {}).get("status") == "unavailable"
    )
    if retrieval_unavailable:
        return [
            f"{retrieval_unavailable} 个用例没有可用的检索上下文快照；请重新运行评估以生成检索质量指标。"
        ]
    return ["当前用例全部通过；继续扩展更难用例并观察多轮运行稳定性。"]


def _topic_from_report_name(stem: str) -> str:
    prefix = "knowledge-qa-"
    if not stem.startswith(prefix):
        return ""
    value = stem[len(prefix) :]
    match = re.match(r"(.+)-v\d+\.\d+-\d{8}-\d{6}$", value)
    return match.group(1) if match else value


def _summary_from_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")[:3000]
    summary: dict[str, Any] = {}
    for label, key in [
        ("通过率", "pass_rate"),
        ("用例数", "total"),
        ("通过", "passed"),
        ("未通过", "failed"),
        ("平均分", "average_score"),
        ("自动回退次数", "fallback_count"),
    ]:
        match = re.search(rf"{label}：([0-9.]+)%?", text)
        if match:
            value = float(match.group(1))
            summary[key] = value if "." in match.group(1) else int(value)
    return summary
