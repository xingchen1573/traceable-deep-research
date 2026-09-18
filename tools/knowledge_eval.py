"""Knowledge-base QA evaluation tools."""

from __future__ import annotations

import json
import hashlib
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ._registry import ROOT, safe_knowledge_path
from .knowledge_qa import query_knowledge_base
from .knowledge_manager import inspect_knowledge_base

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event


DEFAULT_TOPIC = "harness-agent-v03-e2e-1781143504"
EVALUATION_DIR = ROOT / "docs" / "evaluations"
SEMANTIC_SCORE_KEYS = (
    "correctness",
    "completeness",
    "faithfulness",
    "relevance",
    "boundary_awareness",
    "citation_support",
)
SEMANTIC_THRESHOLDS = {
    "correctness": 0.75,
    "completeness": 0.70,
    "faithfulness": 0.75,
    "relevance": 0.75,
    "boundary_awareness": 0.70,
    "citation_support": 0.65,
}
RETRIEVAL_THRESHOLDS = {
    "evidence_recall": 0.60,
    "point_coverage": 0.70,
    "noise_ratio_max": 0.50,
    "boundary_safety": 0.70,
}
RETRIEVAL_ADVISORY_THRESHOLDS = {
    "context_relevance": 0.65,
    "noise_ratio_max": 0.35,
}


@tool
def evaluate_knowledge_base_qa(topic: str = DEFAULT_TOPIC, backend: str = "auto", mode: str = "mix") -> str:
    """运行知识库问答质量评估，并在 docs/evaluations 下生成 Markdown 报告。"""
    safe_topic = _validate_topic(topic or DEFAULT_TOPIC)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cases = _eval_set_for_topic(safe_topic)
    evaluation_context = _evaluation_context(safe_topic, cases)
    results = []
    for case in cases:
        output = _query_case(case, safe_topic, backend=backend, mode=mode)
        results.append(_score_case(case, output))
    _apply_retrieval_evaluations(safe_topic, cases, results)
    _apply_semantic_judgments(safe_topic, cases, results)
    summary = _summarize_results(results)
    report_path = _write_report(safe_topic, started_at, backend, mode, summary, results)
    public_results = [_public_result(item) for item in results]
    report_data_path = _write_structured_report(
        report_path,
        topic=safe_topic,
        started_at=started_at,
        backend=backend,
        mode=mode,
        summary=summary,
        results=public_results,
        evaluation_context=evaluation_context,
    )
    return json.dumps(
        {
            "status": "ok",
            "topic": safe_topic,
            "backend": backend,
            "mode": mode,
            "started_at": started_at,
            "summary": summary,
            "report_path": str(report_path.relative_to(ROOT)).replace("\\", "/"),
            "report_data_path": str(report_data_path.relative_to(ROOT)).replace("\\", "/"),
            **evaluation_context,
            "results": public_results,
        },
        ensure_ascii=False,
    )


def _public_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": item["answer"],
        "passed": item["passed"],
        "score": item["score"],
        "failed_checks": item["failed_checks"],
        "likely_failure_type": item["likely_failure_type"],
        "recommended_action": item["recommended_action"],
        "backend": item["backend"],
        "requested_backend": item["requested_backend"],
        "actual_backend": item["actual_backend"],
        "fallback_occurred": item["fallback_occurred"],
        "fallback_reason": item["fallback_reason"],
        "mode": item["mode"],
        "citation_count": item["citation_count"],
        "retrieval_evaluation": item.get("retrieval_evaluation"),
        "semantic_evaluation": item.get("semantic_evaluation"),
    }


def _write_structured_report(
    report_path: Path,
    *,
    topic: str,
    started_at: str,
    backend: str,
    mode: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    evaluation_context: dict[str, Any] | None = None,
) -> Path:
    actions = []
    for item in results:
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
    if not actions:
        unavailable_count = sum(
            1
            for item in results
            if (item.get("semantic_evaluation") or {}).get("status") == "unavailable"
        )
        if unavailable_count:
            actions.append(
                f"{unavailable_count} 个用例仅完成基础规则检查，AI 语义评测未完成；模型恢复后请重新运行评估。"
            )
        else:
            retrieval_semantic_unavailable = sum(
                1
                for item in results
                if (item.get("retrieval_evaluation") or {}).get("semantic_judge_status") == "unavailable"
            )
            if retrieval_semantic_unavailable:
                actions.append(
                    f"{retrieval_semantic_unavailable} 个用例已完成确定性检索检查，但检索语义评测未完成；模型恢复后请重新运行评估。"
                )
            else:
                actions.append("当前用例全部通过；继续扩展更难用例并观察多轮运行稳定性。")
    payload = {
        "schema_version": 3,
        "report_id": report_path.stem,
        "topic": topic,
        "created_at": started_at,
        "backend": backend,
        "mode": mode,
        "summary": summary,
        "cases": results,
        "recommended_actions": actions,
        **(evaluation_context or {}),
    }
    data_path = report_path.with_suffix(".json")
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_path


def _validate_topic(topic: str) -> str:
    safe = str(topic or "").strip().replace("\\", "/").strip("/")
    if not safe or "/" in safe or ".." in safe:
        raise ValueError("topic must be a single knowledge_base directory name")
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"knowledge base topic not found: {safe}")
    if not list(path.glob("*.md")):
        raise ValueError(f"knowledge base topic has no Markdown documents: {safe}")
    return safe


def _eval_set_for_topic(topic: str) -> list[dict[str, Any]]:
    try:
        from ..backend.services import eval_case_store
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import eval_case_store
    return eval_case_store.load_cases("knowledge_qa", topic=topic)


def _evaluation_context(topic: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from ..backend.services import eval_case_store
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import eval_case_store
    active = eval_case_store.get_active_topic_version(topic) or {}
    current_hash = _knowledge_content_hash(topic)
    stored_hash = str(active.get("knowledge_content_hash") or "")
    current_case_hash = _case_set_hash(cases)
    stored_case_hash = str(active.get("case_set_hash") or "")
    versions = []
    eval_set_ids = []
    for case in cases:
        try:
            versions.append(int(case.get("eval_set_version") or 0))
        except (TypeError, ValueError):
            pass
        if case.get("eval_set_id"):
            eval_set_ids.append(str(case["eval_set_id"]))
    return {
        "eval_set_id": active.get("eval_set_id") or (eval_set_ids[0] if eval_set_ids else None),
        "eval_set_version": active.get("version") or max(versions or [0]) or None,
        "case_set_hash": current_case_hash,
        "eval_set_case_hash": stored_case_hash or None,
        "case_set_modified": bool(stored_case_hash and stored_case_hash != current_case_hash),
        "knowledge_content_hash": current_hash,
        "eval_set_content_hash": stored_hash or None,
        "knowledge_content_stale": bool(stored_hash and stored_hash != current_hash),
    }


def _knowledge_content_hash(topic: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(safe_knowledge_path(topic).glob("*.md"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _case_set_hash(cases: list[dict[str, Any]]) -> str:
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _query_case(case: dict[str, Any], topic: str, *, backend: str, mode: str) -> dict[str, Any]:
    raw = query_knowledge_base.invoke(
        {
            "question": case["question"],
            "topic": topic,
            "backend": backend,
            "mode": mode,
            "include_retrieval_trace": True,
        }
    )
    try:
        return json.loads(raw)
    except Exception as exc:
        return {
            "status": "invalid_json",
            "topic": topic,
            "backend": backend,
            "mode": mode,
            "answer": "",
            "citations": [],
            "confidence": "low",
            "gaps": [f"query_knowledge_base returned invalid JSON: {exc}"],
            "raw": str(raw)[:1000],
        }


def _score_case(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    answer = str(output.get("answer") or "")
    citations = output.get("citations") or []
    gaps = output.get("gaps") or []
    must_include_all = case.get("must_include_all")
    if must_include_all is None:
        must_include_all = case.get("expected_keywords") or []
    must_include_any = case.get("must_include_any") or []
    must_not_include = case.get("must_not_include")
    if must_not_include is None:
        must_not_include = case.get("forbidden_keywords") or []
    checks = {
        "answered": output.get("status") == "answered" and bool(answer.strip()),
        "has_citations": len(citations) > 0,
        "topic_citations": _citations_match_topic(citations, str(output.get("topic") or "")),
        "must_include_all": _contains_all(answer, must_include_all),
        "must_include_any": _contains_any(answer, must_include_any) if must_include_any else True,
        "must_not_include": not _contains_any(answer, must_not_include),
        "answer_length": len(answer.strip()) >= int(case.get("min_answer_chars") or 0),
    }
    if case.get("expect_insufficient"):
        checks["acknowledged_insufficient"] = _acknowledges_insufficient(answer, gaps)
    failed = [name for name, passed in checks.items() if not passed]
    score = round(100 * (len(checks) - len(failed)) / max(1, len(checks)), 1)
    failure_type = _classify_failure(case, output, failed)
    recommended_action = _recommended_action(failure_type, failed)
    return {
        "id": case["id"],
        "category": case.get("category") or "未分类",
        "question": case["question"],
        "passed": not failed,
        "score": score,
        "checks": checks,
        "failed_checks": failed,
        "must_include_all": must_include_all,
        "must_include_any": must_include_any,
        "must_not_include": must_not_include,
        "likely_failure_type": failure_type,
        "diagnosis": _diagnosis_text(failure_type, failed),
        "recommended_action": recommended_action,
        "status": output.get("status"),
        "backend": output.get("backend"),
        "requested_backend": output.get("requested_backend") or output.get("backend"),
        "actual_backend": output.get("actual_backend") or output.get("backend"),
        "fallback_occurred": bool(output.get("fallback_occurred")),
        "fallback_reason": str(output.get("fallback_reason") or ""),
        "mode": output.get("mode"),
        "answer": answer,
        "citations": citations,
        "citation_count": len(citations),
        "confidence": output.get("confidence"),
        "gaps": gaps,
        "retrieval_trace": output.get("retrieval_trace") or [],
    }


def _apply_retrieval_evaluations(
    topic: str,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    case_by_id = {str(case.get("id") or ""): case for case in cases}
    eligible: list[dict[str, Any]] = []
    for result in results:
        case = case_by_id.get(str(result.get("id") or ""), {})
        evaluation = _deterministic_retrieval_evaluation(case, result)
        result["retrieval_evaluation"] = evaluation
        if evaluation.get("status") != "unavailable":
            eligible.append(result)
    if not eligible:
        return

    emit_event(
        "knowledge_evaluation.retrieval_judge.started",
        "knowledge_eval",
        "Started retrieval-context quality evaluation.",
        data={"topic": topic, "case_count": len(eligible)},
    )
    payload_cases = [
        _retrieval_judge_payload(case_by_id[str(result["id"])], result)
        for result in eligible
    ]
    try:
        raw_judgments = _invoke_retrieval_judge(topic, payload_cases)
    except Exception as exc:
        reason = f"检索语义裁判不可用：{type(exc).__name__}: {str(exc)[:240]}"
        needs_review = 0
        for result in eligible:
            evaluation = result["retrieval_evaluation"]
            evaluation["status"] = "partial"
            evaluation["semantic_judge_status"] = "unavailable"
            evaluation["reason"] = reason
            _finalize_retrieval_quality(evaluation, answerability=str(case_by_id[str(result["id"])].get("answerability") or "answerable"))
            needs_review += 1 if evaluation.get("quality_status") == "needs_review" else 0
        emit_event(
            "knowledge_evaluation.retrieval_judge.degraded",
            "knowledge_eval",
            "Retrieval semantic evaluation was unavailable; deterministic retrieval metrics were retained.",
            level="warning",
            data={"topic": topic, "case_count": len(eligible), "needs_review_cases": needs_review, "reason": reason},
        )
        return

    raw_by_id = {
        str(item.get("id") or ""): item
        for item in raw_judgments
        if isinstance(item, dict) and item.get("id")
    }
    completed = 0
    unavailable = 0
    needs_review = 0
    for result in eligible:
        case = case_by_id[str(result["id"])]
        evaluation = result["retrieval_evaluation"]
        raw = raw_by_id.get(str(result.get("id") or ""))
        if not raw:
            evaluation["status"] = "partial"
            evaluation["semantic_judge_status"] = "unavailable"
            evaluation["reason"] = "检索语义裁判没有返回该用例的判断。"
            unavailable += 1
        else:
            _merge_retrieval_judgment(evaluation, raw, case)
            completed += 1
        _finalize_retrieval_quality(
            evaluation,
            answerability=str(case.get("answerability") or "answerable"),
        )
        needs_review += 1 if evaluation.get("quality_status") == "needs_review" else 0
    emit_event(
        "knowledge_evaluation.retrieval_judge.completed",
        "knowledge_eval",
        "Completed retrieval-context quality evaluation.",
        data={
            "topic": topic,
            "completed_cases": completed,
            "unavailable_cases": unavailable,
            "needs_review_cases": needs_review,
        },
    )


def _deterministic_retrieval_evaluation(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    trace = [dict(item) for item in result.get("retrieval_trace") or [] if isinstance(item, dict)]
    if not trace:
        return {
            "status": "unavailable",
            "quality_status": "not_evaluated",
            "semantic_judge_status": "not_started",
            "reason": "问答结果没有保留检索上下文快照。",
            "metrics": {},
            "chunks": [],
        }

    evidence_refs = [item for item in case.get("evidence_refs") or [] if isinstance(item, dict)]
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_refs}
    matched_evidence_ids: set[str] = set()
    matched_ranks: list[int] = []
    chunks: list[dict[str, Any]] = []
    for fallback_rank, raw in enumerate(trace, start=1):
        rank = int(raw.get("rank") or fallback_rank)
        matches: list[tuple[str, float]] = []
        for evidence_id, evidence in evidence_by_id.items():
            score = _evidence_chunk_match_score(evidence, raw)
            if score >= 0.28:
                matches.append((evidence_id, score))
                matched_evidence_ids.add(evidence_id)
                matched_ranks.append(rank)
        chunks.append(
            {
                "rank": rank,
                "chunk_id": str(raw.get("chunk_id") or ""),
                "reference_id": str(raw.get("reference_id") or ""),
                "title": str(raw.get("title") or ""),
                "path": str(raw.get("path") or ""),
                "content": str(raw.get("content") or ""),
                "score": raw.get("score"),
                "score_type": str(raw.get("score_type") or "not_provided"),
                "matched_evidence_ids": [item[0] for item in sorted(matches, key=lambda item: item[1], reverse=True)],
                "evidence_match_score": round(max((item[1] for item in matches), default=0.0), 3),
            }
        )

    point_ids = [str(item.get("id") or "") for item in case.get("expected_points") or [] if isinstance(item, dict)]
    deterministic_point_ids: list[str] = []
    for point in case.get("expected_points") or []:
        if not isinstance(point, dict):
            continue
        refs = {str(item) for item in point.get("evidence_refs") or []}
        if refs & matched_evidence_ids:
            deterministic_point_ids.append(str(point.get("id") or ""))

    answerable = str(case.get("answerability") or "answerable") == "answerable"
    evidence_recall = round(len(matched_evidence_ids) / len(evidence_refs), 3) if answerable and evidence_refs else None
    point_coverage = round(len(deterministic_point_ids) / len(point_ids), 3) if answerable and point_ids else None
    first_rank = min(matched_ranks) if matched_ranks else None
    metrics = {
        "evidence_recall": evidence_recall,
        "point_coverage": point_coverage,
        "hit_at_3": bool(first_rank and first_rank <= 3) if answerable else None,
        "first_key_evidence_rank": first_rank,
        "reciprocal_rank": round(1 / first_rank, 3) if first_rank else (0.0 if answerable else None),
        "source_concentration": _source_concentration(chunks),
        "duplicate_ratio": _duplicate_chunk_ratio(chunks),
        "context_relevance": None,
        "noise_ratio": None,
        "boundary_safety": None,
    }
    return {
        "status": "completed",
        "quality_status": "not_evaluated",
        "semantic_judge_status": "pending",
        "metrics": metrics,
        "matched_evidence_ids": sorted(matched_evidence_ids),
        "matched_point_ids": deterministic_point_ids,
        "missing_point_ids": [point_id for point_id in point_ids if point_id not in deterministic_point_ids],
        "chunks": chunks,
        "reason": "",
        "recommended_action": "",
    }


def _evidence_chunk_match_score(evidence: dict[str, Any], chunk: dict[str, Any]) -> float:
    expected_path = str(evidence.get("document_path") or "").replace("\\", "/").lower()
    actual_path = str(chunk.get("path") or "").replace("\\", "/").lower()
    if expected_path and actual_path:
        if Path(expected_path).name != Path(actual_path).name and not actual_path.endswith(expected_path):
            return 0.0
    expected = _normalize_retrieval_text(evidence.get("evidence_quote"))
    actual = _normalize_retrieval_text(chunk.get("content"))
    if not expected or not actual:
        return 0.0
    if expected in actual or actual in expected:
        return round(min(len(expected), len(actual)) / max(1, len(expected)), 3)
    size = 4
    if len(expected) < size or len(actual) < size:
        return 0.0
    expected_grams = {expected[index:index + size] for index in range(len(expected) - size + 1)}
    actual_grams = {actual[index:index + size] for index in range(len(actual) - size + 1)}
    return round(len(expected_grams & actual_grams) / max(1, len(expected_grams)), 3)


def _normalize_retrieval_text(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _source_concentration(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    counts: dict[str, int] = {}
    for item in chunks:
        path = str(item.get("path") or "unknown")
        counts[path] = counts.get(path, 0) + 1
    return round(max(counts.values()) / len(chunks), 3)


def _duplicate_chunk_ratio(chunks: list[dict[str, Any]]) -> float:
    if len(chunks) < 2:
        return 0.0
    seen: list[set[str]] = []
    duplicates = 0
    for item in chunks:
        text = _normalize_retrieval_text(item.get("content"))
        grams = {text[index:index + 5] for index in range(max(0, len(text) - 4))}
        if grams and any(len(grams & previous) / max(1, len(grams | previous)) >= 0.80 for previous in seen):
            duplicates += 1
        seen.append(grams)
    return round(duplicates / len(chunks), 3)


def _retrieval_judge_payload(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    evaluation = result.get("retrieval_evaluation") or {}
    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "answerability": case.get("answerability") or "answerable",
        "expected_points": [
            {"id": item.get("id"), "text": item.get("text")}
            for item in case.get("expected_points") or []
            if isinstance(item, dict)
        ],
        "chunks": [
            {
                "rank": item.get("rank"),
                "path": item.get("path"),
                "content": str(item.get("content") or "")[:1800],
            }
            for item in evaluation.get("chunks") or []
        ],
    }


def _invoke_retrieval_judge(topic: str, payload_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from ..main import build_model
    except ImportError:  # pragma: no cover - script mode fallback
        from main import build_model

    prompt = {
        "task": "评估检索上下文与问题的关系。只返回 JSON，不评价最终回答。",
        "topic": topic,
        "rules": [
            "逐条判断 chunk 是否直接支撑问题、仅背景相关、无关噪声或可能误导。",
            "matched_point_ids 只能使用输入中的参考要点 ID；没有直接支撑时不要勾选。",
            "relevance_score 表示该 chunk 对回答问题的直接价值，范围 0 到 1。",
            "label 只能是 supporting、related、noise、misleading。",
            "boundary_risk 表示整组上下文诱导模型越过证据边界的风险，范围 0 到 1。",
            "不可回答问题不要求召回为空；重点判断上下文是否被误解为足以支持确定答案。",
            "不要因为表述方式不同而扣分，只判断实际含义。",
        ],
        "schema": {
            "judgments": [
                {
                    "id": "case_id",
                    "chunks": [
                        {
                            "rank": 1,
                            "relevance_score": 0.0,
                            "label": "supporting|related|noise|misleading",
                            "matched_point_ids": [],
                            "reason": "简短理由",
                        }
                    ],
                    "boundary_risk": 0.0,
                    "reason": "整组检索上下文的简短判断",
                }
            ]
        },
        "cases": payload_cases,
    }
    model = build_model(streaming=False).bind(max_tokens=8192)
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "你是严格、克制的 RAG 检索质量裁判。输出必须是合法 JSON，不得输出 Markdown。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
    )
    parsed = _json_object_from_model_text(_model_response_text(response))
    judgments = parsed.get("judgments") or []
    if not isinstance(judgments, list):
        raise ValueError("检索裁判没有返回 judgments 数组")
    return [dict(item) for item in judgments if isinstance(item, dict)]


def _merge_retrieval_judgment(
    evaluation: dict[str, Any],
    raw: dict[str, Any],
    case: dict[str, Any],
) -> None:
    chunks = evaluation.get("chunks") or []
    by_rank = {int(item.get("rank") or 0): item for item in chunks}
    point_ids = {
        str(item.get("id") or "")
        for item in case.get("expected_points") or []
        if isinstance(item, dict)
    }
    semantic_points: set[str] = set(evaluation.get("matched_point_ids") or [])
    relevance_scores: list[float] = []
    noise_count = 0
    for item in raw.get("chunks") or []:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item.get("rank") or 0)
        except (TypeError, ValueError):
            continue
        target = by_rank.get(rank)
        if not target:
            continue
        score = _clamp_score(item.get("relevance_score"))
        label = str(item.get("label") or "related").strip().lower()
        if label not in {"supporting", "related", "noise", "misleading"}:
            label = "related"
        matched = [str(value) for value in item.get("matched_point_ids") or [] if str(value) in point_ids]
        semantic_points.update(matched)
        target["relevance_score"] = score
        target["relevance_label"] = label
        target["semantic_matched_point_ids"] = list(dict.fromkeys(matched))
        target["judge_reason"] = str(item.get("reason") or "").strip()[:500]
        relevance_scores.append(score)
        noise_count += 1 if label in {"noise", "misleading"} else 0

    metrics = evaluation["metrics"]
    metrics["context_relevance"] = round(sum(relevance_scores) / max(1, len(relevance_scores)), 3)
    metrics["noise_ratio"] = round(noise_count / max(1, len(chunks)), 3)
    metrics["boundary_safety"] = round(1 - _clamp_score(raw.get("boundary_risk")), 3)
    all_point_ids = [str(item.get("id") or "") for item in case.get("expected_points") or [] if isinstance(item, dict)]
    matched_points = [point_id for point_id in all_point_ids if point_id in semantic_points]
    if str(case.get("answerability") or "answerable") == "answerable" and all_point_ids:
        metrics["point_coverage"] = round(len(matched_points) / len(all_point_ids), 3)
    evaluation["matched_point_ids"] = matched_points
    evaluation["missing_point_ids"] = [point_id for point_id in all_point_ids if point_id not in semantic_points]
    evaluation["status"] = "completed"
    evaluation["semantic_judge_status"] = "completed"
    evaluation["reason"] = str(raw.get("reason") or "").strip()[:1200]


def _finalize_retrieval_quality(evaluation: dict[str, Any], *, answerability: str) -> None:
    metrics = evaluation.get("metrics") or {}
    below: list[str] = []
    if answerability == "answerable":
        for key in ("evidence_recall", "point_coverage"):
            value = metrics.get(key)
            if value is not None and float(value) < RETRIEVAL_THRESHOLDS[key]:
                below.append(key)
    advisory: list[str] = []
    if answerability == "answerable" and metrics.get("hit_at_3") is False:
        advisory.append("hit_at_3")
    relevance = metrics.get("context_relevance")
    if relevance is not None and float(relevance) < RETRIEVAL_ADVISORY_THRESHOLDS["context_relevance"]:
        advisory.append("context_relevance")
    noise = metrics.get("noise_ratio")
    if noise is not None and float(noise) > RETRIEVAL_THRESHOLDS["noise_ratio_max"]:
        below.append("noise_ratio")
    elif noise is not None and float(noise) > RETRIEVAL_ADVISORY_THRESHOLDS["noise_ratio_max"]:
        advisory.append("noise_ratio")
    boundary = metrics.get("boundary_safety")
    if boundary is not None and float(boundary) < RETRIEVAL_THRESHOLDS["boundary_safety"]:
        below.append("boundary_safety")
    evaluation["below_threshold"] = below
    evaluation["advisory"] = advisory
    evaluation["quality_status"] = "needs_review" if below else "passed"
    evaluation["recommended_action"] = _retrieval_recommended_action(below, advisory)


def _retrieval_recommended_action(below: list[str], advisory: list[str] | None = None) -> str:
    items = set(below)
    advisory_items = set(advisory or [])
    actions: list[str] = []
    if items & {"evidence_recall", "point_coverage"}:
        actions.append("检查关键证据是否被切分、漏索引或未进入最终上下文")
    if "hit_at_3" in items:
        actions.append("优化关键证据的前排排序")
    if "context_relevance" in items or "noise_ratio" in items:
        actions.append("减少与问题仅弱相关或重复的检索片段")
    if "boundary_safety" in items:
        actions.append("避免把背景材料当作足以支持确定结论的证据")
    if advisory_items & {"hit_at_3", "context_relevance"}:
        actions.append("关键证据已覆盖，可继续优化前排排序和上下文精简")
    if "noise_ratio" in advisory_items:
        actions.append("可减少弱相关背景片段，提高上下文利用效率")
    return "；".join(actions) + "。" if actions else "检索质量达到当前诊断阈值，无需立即处理。"


def _apply_semantic_judgments(
    topic: str,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    case_by_id = {str(case.get("id") or ""): case for case in cases}
    eligible = [
        result
        for result in results
        if case_by_id.get(str(result.get("id") or ""), {}).get("expected_points")
        or case_by_id.get(str(result.get("id") or ""), {}).get("evidence_refs")
    ]
    if not eligible:
        for result in results:
            result["semantic_evaluation"] = {
                "status": "not_applicable",
                "quality_status": "not_evaluated",
                "reason": "该旧评测用例没有参考要点或稳定标准证据。",
            }
        return

    emit_event(
        "knowledge_evaluation.semantic_judge.started",
        "knowledge_eval",
        "Started semantic answer-quality evaluation.",
        data={"topic": topic, "case_count": len(eligible)},
    )
    payload_cases = [
        _semantic_judge_payload(case_by_id[str(result["id"])], result)
        for result in eligible
    ]
    try:
        raw_judgments = _invoke_semantic_judge(topic, payload_cases)
    except Exception as exc:
        reason = f"语义裁判不可用：{type(exc).__name__}: {str(exc)[:240]}"
        for result in eligible:
            result["semantic_evaluation"] = {
                "status": "unavailable",
                "quality_status": "not_evaluated",
                "reason": reason,
            }
        emit_event(
            "knowledge_evaluation.semantic_judge.degraded",
            "knowledge_eval",
            "Semantic answer-quality evaluation was unavailable; deterministic checks were retained.",
            level="warning",
            data={"topic": topic, "case_count": len(eligible), "reason": reason},
        )
        return

    raw_by_id = {
        str(item.get("id") or ""): item
        for item in raw_judgments
        if isinstance(item, dict) and item.get("id")
    }
    completed = 0
    unavailable = 0
    needs_review = 0
    for result in results:
        case = case_by_id.get(str(result.get("id") or ""), {})
        if result not in eligible:
            result["semantic_evaluation"] = {
                "status": "not_applicable",
                "quality_status": "not_evaluated",
                "reason": "该用例没有可供语义裁判使用的标准证据。",
            }
            continue
        raw = raw_by_id.get(str(result.get("id") or ""))
        if not raw:
            result["semantic_evaluation"] = {
                "status": "unavailable",
                "quality_status": "not_evaluated",
                "reason": "语义裁判没有返回该用例的判断。",
            }
            unavailable += 1
            continue
        semantic = _normalize_semantic_judgment(raw, case)
        result["semantic_evaluation"] = semantic
        completed += 1
        needs_review += 1 if semantic["quality_status"] == "needs_review" else 0
    emit_event(
        "knowledge_evaluation.semantic_judge.completed",
        "knowledge_eval",
        "Completed semantic answer-quality evaluation.",
        data={
            "topic": topic,
            "completed_cases": completed,
            "unavailable_cases": unavailable,
            "needs_review_cases": needs_review,
        },
    )


def _semantic_judge_payload(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for item in case.get("evidence_refs") or []:
        if not isinstance(item, dict):
            continue
        evidence.append(
            {
                "id": item.get("id"),
                "document_path": item.get("document_path"),
                "section_title": item.get("section_title"),
                "quote": str(item.get("evidence_quote") or "")[:1400],
            }
        )
    citations = []
    for index, item in enumerate(result.get("citations") or [], start=1):
        if not isinstance(item, dict):
            continue
        citations.append(
            {
                "index": index,
                "path": item.get("path"),
                "snippet": str(item.get("snippet") or "")[:900],
            }
        )
    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "answerability": case.get("answerability") or "answerable",
        "reference_answer": str(case.get("reference_answer") or "")[:3000],
        "expected_points": case.get("expected_points") or [],
        "forbidden_claims": case.get("forbidden_claims") or [],
        "standard_evidence": evidence,
        "actual_answer": str(result.get("answer") or "")[:7000],
        "actual_citations": citations,
        "reported_gaps": result.get("gaps") or [],
    }


def _invoke_semantic_judge(topic: str, payload_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from ..main import build_model
    except ImportError:  # pragma: no cover - script mode fallback
        from main import build_model

    prompt = {
        "task": "依据标准事实要点、标准证据、实际回答和实际引用，评估知识库问答质量。只返回 JSON。",
        "topic": topic,
        "rules": [
            "只能依据输入中的标准证据和实际引用判断，不得补充外部知识。",
            "correctness 判断实际结论是否与标准事实一致。",
            "completeness 判断 expected_points 的覆盖情况；matched_point_ids 只能使用输入中的 point id。",
            "faithfulness 判断实际回答中的事实陈述是否得到 actual_citations 支持。",
            "relevance 判断回答是否直接回应问题，避免无关扩写。",
            "boundary_awareness 判断回答是否尊重材料边界；不可回答问题尤其要明确说明不足。",
            "citation_support 判断实际引用片段是否支撑回答中的主要结论，而不只是路径存在。",
            "所有分数范围为 0 到 1。unsupported_claims 和 citation_issues 必须给出具体、简短的中文描述。",
            "不要因为文风、措辞与参考答案不同而扣分；比较事实含义和证据支撑。",
        ],
        "schema": {
            "judgments": [
                {
                    "id": "case_id",
                    "scores": {
                        "correctness": 0.0,
                        "completeness": 0.0,
                        "faithfulness": 0.0,
                        "relevance": 0.0,
                        "boundary_awareness": 0.0,
                        "citation_support": 0.0,
                    },
                    "matched_point_ids": [],
                    "missing_point_ids": [],
                    "unsupported_claims": [],
                    "citation_issues": [],
                    "reason": "简要解释评分依据",
                }
            ]
        },
        "cases": payload_cases,
    }
    model = build_model(streaming=False).bind(max_tokens=8192)
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "你是严格但不过度机械的 RAG 回答质量裁判。输出必须是合法 JSON，不得输出 Markdown。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
    )
    parsed = _json_object_from_model_text(_model_response_text(response))
    judgments = parsed.get("judgments") or []
    if not isinstance(judgments, list):
        raise ValueError("语义裁判没有返回 judgments 数组")
    return [dict(item) for item in judgments if isinstance(item, dict)]


def _model_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") if isinstance(item, dict) else item)
            for item in content
            if (isinstance(item, str) and item) or (isinstance(item, dict) and item.get("text"))
        )
    return str(content or "")


def _json_object_from_model_text(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
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
    raise ValueError("语义裁判返回了无法解析的 JSON")


def _normalize_semantic_judgment(raw: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    raw_scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    scores = {key: _clamp_score(raw_scores.get(key)) for key in SEMANTIC_SCORE_KEYS}
    point_ids = [str(item.get("id") or "") for item in case.get("expected_points") or [] if isinstance(item, dict)]
    matched = [str(item) for item in raw.get("matched_point_ids") or [] if str(item) in point_ids]
    matched = list(dict.fromkeys(matched))
    missing = [point_id for point_id in point_ids if point_id not in matched]
    if point_ids:
        scores["completeness"] = round(len(matched) / len(point_ids), 3)
    below_threshold = [
        key for key, threshold in SEMANTIC_THRESHOLDS.items() if scores[key] < threshold
    ]
    quality_status = "needs_review" if below_threshold else "passed"
    semantic = {
        "status": "completed",
        "quality_status": quality_status,
        "scores": scores,
        "matched_point_ids": matched,
        "missing_point_ids": missing,
        "unsupported_claims": _string_list(raw.get("unsupported_claims"), limit=8),
        "citation_issues": _string_list(raw.get("citation_issues"), limit=8),
        "reason": str(raw.get("reason") or "").strip()[:1200],
        "below_threshold": below_threshold,
    }
    semantic["recommended_action"] = _semantic_recommended_action(semantic)
    return semantic


def _clamp_score(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:500] for item in value if str(item).strip()][:limit]


def _semantic_recommended_action(semantic: dict[str, Any]) -> str:
    below = set(semantic.get("below_threshold") or [])
    actions = []
    if below & {"correctness", "completeness"}:
        actions.append("对照参考要点补齐或修正回答内容")
    if below & {"faithfulness", "citation_support"}:
        actions.append("检查引用片段与主要结论的支撑关系")
    if "relevance" in below:
        actions.append("收紧回答范围并直接回应问题")
    if "boundary_awareness" in below:
        actions.append("强化证据边界和不确定性说明")
    return "；".join(actions) + "。" if actions else "语义质量达到当前阈值，无需立即处理。"


def _classify_failure(case: dict[str, Any], output: dict[str, Any], failed_checks: list[str]) -> str:
    if not failed_checks:
        return "none"
    if "answered" in failed_checks:
        if output.get("status") in {"no_match", "index_missing"}:
            return "retrieval"
        return "generation"
    if "has_citations" in failed_checks or "topic_citations" in failed_checks:
        return "citation"
    if "acknowledged_insufficient" in failed_checks:
        return "boundary"
    if case.get("expect_insufficient") and ("must_include_all" in failed_checks or "must_include_any" in failed_checks):
        return "knowledge_gap"
    if "must_not_include" in failed_checks:
        return "boundary"
    if "must_include_all" in failed_checks or "must_include_any" in failed_checks:
        return str(case.get("failure_type_hint") or "generation")
    if "answer_length" in failed_checks:
        return "generation"
    return str(case.get("failure_type_hint") or "evaluation_rule")


def _diagnosis_text(failure_type: str, failed_checks: list[str]) -> str:
    if failure_type == "none":
        return "当前规则检查全部通过。"
    labels = {
        "retrieval": "可能是召回覆盖不足或检索上下文未命中关键片段。",
        "generation": "可能是生成组织不充分，或回答未覆盖必要要点。",
        "citation": "可能是引用抽取、引用路径或来源映射存在问题。",
        "knowledge_gap": "可能是知识库内容本身缺少该问题所需证据。",
        "boundary": "可能是边界约束不足，模型没有正确拒答、澄清或避免错误前提。",
        "evaluation_rule": "可能是评估规则本身过严或与实际引用/答案格式不匹配。",
    }
    return f"{labels.get(failure_type, labels['evaluation_rule'])} 失败检查项：{', '.join(failed_checks)}。"


def _recommended_action(failure_type: str, failed_checks: list[str]) -> str:
    if failure_type == "none":
        return "无需立即处理；可在后续更难用例中继续观察。"
    actions = {
        "retrieval": "检查召回片段、实体/关系抽取和 chunk 设置；必要时补充文档或重建索引。",
        "generation": "检查 query prompt、回答结构和最低详略要求；必要时增加针对性用例。",
        "citation": "检查 LightRAG references 解析、citation path 映射和报告规则。",
        "knowledge_gap": "优先补充知识库内容；不要通过 prompt 强行生成无证据答案。",
        "boundary": "强化证据不足、错误前提和同名概念澄清约束。",
        "evaluation_rule": "复核评估规则是否过严，必要时调整 must_include/must_not_include。",
    }
    action = actions.get(failure_type, actions["evaluation_rule"])
    if "answer_length" in failed_checks and failure_type != "generation":
        action += " 同时检查回答长度阈值是否合理。"
    return action


def _citations_match_topic(citations: list[Any], topic: str) -> bool:
    if not citations:
        return False
    if not topic:
        return True
    topic_docs = {path.name for path in safe_knowledge_path(topic).glob("*.md")}
    for item in citations:
        if not isinstance(item, dict):
            continue
        citation_path = str(item.get("path") or "").replace("\\", "/")
        if topic in citation_path:
            continue
        if Path(citation_path).name in topic_docs:
            continue
        return False
    return True


def _contains_all(text: str, keywords: list[Any]) -> bool:
    """Every item must be found in text. An item that is itself a list/tuple is a
    synonym group -- it counts as found if ANY one of its alternatives appears.

    Plain string keywords still require an exact literal substring match, same as
    before. This lets eval cases require a semantic point be present without
    pinning it to one specific wording the model has to reproduce verbatim.
    """
    lowered = text.lower()
    for keyword in keywords:
        if isinstance(keyword, (list, tuple)):
            if not any(str(alt).lower() in lowered for alt in keyword):
                return False
        elif str(keyword).lower() not in lowered:
            return False
    return True


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _acknowledges_insufficient(answer: str, gaps: list[Any]) -> bool:
    text = f"{answer}\n{json.dumps(gaps, ensure_ascii=False)}"
    return bool(re.search(r"不足|没有|未覆盖|无法|未找到|证据不足|资料较少", text))


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    avg_score = round(sum(float(item["score"]) for item in results) / max(1, total), 1)
    failed_checks: dict[str, int] = {}
    categories: dict[str, dict[str, Any]] = {}
    failure_types: dict[str, int] = {}
    backend_usage: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    for item in results:
        for check in item["failed_checks"]:
            failed_checks[check] = failed_checks.get(check, 0) + 1
        category = item.get("category") or "未分类"
        bucket = categories.setdefault(category, {"total": 0, "passed": 0, "failed": 0, "average_score": 0.0})
        bucket["total"] += 1
        bucket["passed"] += 1 if item["passed"] else 0
        bucket["failed"] += 0 if item["passed"] else 1
        if item["likely_failure_type"] != "none":
            failure_types[item["likely_failure_type"]] = failure_types.get(item["likely_failure_type"], 0) + 1
        actual_backend = str(item.get("actual_backend") or item.get("backend") or "unknown")
        backend_usage[actual_backend] = backend_usage.get(actual_backend, 0) + 1
        fallback_reason = str(item.get("fallback_reason") or "").strip()
        if item.get("fallback_occurred") and fallback_reason:
            fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1
    for category, bucket in categories.items():
        category_scores = [float(item["score"]) for item in results if (item.get("category") or "未分类") == category]
        bucket["average_score"] = round(sum(category_scores) / max(1, len(category_scores)), 1)
        bucket["pass_rate"] = round(100 * bucket["passed"] / max(1, bucket["total"]), 1)
    semantic_rows = [
        item.get("semantic_evaluation") or {}
        for item in results
        if (item.get("semantic_evaluation") or {}).get("status") == "completed"
    ]
    semantic_unavailable = sum(
        1 for item in results if (item.get("semantic_evaluation") or {}).get("status") == "unavailable"
    )
    semantic_scores = {
        key: round(
            sum(float((row.get("scores") or {}).get(key) or 0) for row in semantic_rows)
            / max(1, len(semantic_rows)),
            3,
        )
        for key in SEMANTIC_SCORE_KEYS
    }
    if semantic_rows and semantic_unavailable:
        semantic_status = "partial"
    elif semantic_rows:
        semantic_status = "completed"
    elif semantic_unavailable:
        semantic_status = "unavailable"
    else:
        semantic_status = "not_applicable"
    retrieval_rows = [
        item.get("retrieval_evaluation") or {}
        for item in results
        if (item.get("retrieval_evaluation") or {}).get("status") in {"completed", "partial"}
    ]
    retrieval_unavailable = sum(
        1
        for item in results
        if (item.get("retrieval_evaluation") or {}).get("status") == "unavailable"
    )
    retrieval_metrics: dict[str, float | None] = {}
    for key in (
        "evidence_recall",
        "point_coverage",
        "hit_at_3",
        "reciprocal_rank",
        "context_relevance",
        "noise_ratio",
        "source_concentration",
        "duplicate_ratio",
        "boundary_safety",
    ):
        values = [
            float((row.get("metrics") or {}).get(key))
            for row in retrieval_rows
            if (row.get("metrics") or {}).get(key) is not None
        ]
        retrieval_metrics[key] = round(sum(values) / len(values), 3) if values else None
    retrieval_semantic_unavailable = sum(
        1 for row in retrieval_rows if row.get("semantic_judge_status") == "unavailable"
    )
    if retrieval_rows and retrieval_unavailable:
        retrieval_status = "partial"
    elif retrieval_rows and retrieval_semantic_unavailable:
        retrieval_status = "partial"
    elif retrieval_rows:
        retrieval_status = "completed"
    else:
        retrieval_status = "unavailable" if retrieval_unavailable else "not_applicable"
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(100 * passed / max(1, total), 1),
        "average_score": avg_score,
        "failed_checks": failed_checks,
        "categories": categories,
        "failure_types": failure_types,
        "backend_usage": backend_usage,
        "fallback_count": sum(1 for item in results if item.get("fallback_occurred")),
        "fallback_reasons": fallback_reasons,
        "semantic_quality": {
            "status": semantic_status,
            "judged_cases": len(semantic_rows),
            "unavailable_cases": semantic_unavailable,
            "needs_review_cases": sum(1 for row in semantic_rows if row.get("quality_status") == "needs_review"),
            "scores": semantic_scores,
        },
        "retrieval_quality": {
            "status": retrieval_status,
            "evaluated_cases": len(retrieval_rows),
            "unavailable_cases": retrieval_unavailable,
            "semantic_unavailable_cases": retrieval_semantic_unavailable,
            "needs_review_cases": sum(
                1 for row in retrieval_rows if row.get("quality_status") == "needs_review"
            ),
            "advisory_cases": sum(1 for row in retrieval_rows if row.get("advisory")),
            "metrics": retrieval_metrics,
        },
    }


def _write_report(
    topic: str,
    started_at: str,
    backend: str,
    mode: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> Path:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    path = EVALUATION_DIR / f"knowledge-qa-{topic}-v0.1-{stamp}.md"
    path.write_text(_render_report(topic, started_at, backend, mode, summary, results), encoding="utf-8")
    return path


def _render_report(
    topic: str,
    started_at: str,
    backend: str,
    mode: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    kb_info = json.loads(inspect_knowledge_base.invoke({"topic": topic}))
    kb = kb_info.get("knowledge_base") or {}
    metadata = kb.get("metadata") or {}
    lines = [
        f"# 知识库问答质量评估：{metadata.get('title') or topic}",
        "",
        f"评估时间：{started_at}",
        "",
        "## 1. 评估对象",
        "",
        f"- topic：`{topic}`",
        f"- backend：`{backend}`",
        f"- mode：`{mode}`",
        f"- 文档数量：{kb.get('document_count')}",
        f"- 索引状态：{kb.get('index_status')} / {kb.get('index_reason')}",
        f"- 图谱规模：{(kb.get('graph_stats') or {}).get('nodes', 0)} 节点 / {(kb.get('graph_stats') or {}).get('edges', 0)} 边",
        f"- 向量规模：chunks={(kb.get('vector_stats') or {}).get('chunks', 0)}，entities={(kb.get('vector_stats') or {}).get('entities', 0)}，relationships={(kb.get('vector_stats') or {}).get('relationships', 0)}",
        "",
        "## 2. 总体结果",
        "",
        f"- 用例数：{summary['total']}",
        f"- 通过：{summary['passed']}",
        f"- 未通过：{summary['failed']}",
        f"- 通过率：{summary['pass_rate']}%",
        f"- 平均分：{summary['average_score']}",
        f"- 失败检查项：{json.dumps(summary['failed_checks'], ensure_ascii=False)}",
        f"- 失败类型：{json.dumps(summary.get('failure_types') or {}, ensure_ascii=False)}",
        f"- 实际问答后端：{json.dumps(summary.get('backend_usage') or {}, ensure_ascii=False)}",
        f"- 自动回退次数：{summary.get('fallback_count') or 0}",
        f"- 自动回退原因：{json.dumps(summary.get('fallback_reasons') or {}, ensure_ascii=False)}",
        f"- 检索评测状态：{(summary.get('retrieval_quality') or {}).get('status') or 'not_applicable'}",
        f"- 检索评测用例：{(summary.get('retrieval_quality') or {}).get('evaluated_cases') or 0}",
        f"- 检索需复核用例：{(summary.get('retrieval_quality') or {}).get('needs_review_cases') or 0}",
        f"- 检索指标：{json.dumps((summary.get('retrieval_quality') or {}).get('metrics') or {}, ensure_ascii=False)}",
        f"- 语义评测状态：{(summary.get('semantic_quality') or {}).get('status') or 'not_applicable'}",
        f"- 语义评测用例：{(summary.get('semantic_quality') or {}).get('judged_cases') or 0}",
        f"- 语义需复核用例：{(summary.get('semantic_quality') or {}).get('needs_review_cases') or 0}",
        f"- 语义指标：{json.dumps((summary.get('semantic_quality') or {}).get('scores') or {}, ensure_ascii=False)}",
        "",
        "## 3. 分类统计",
        "",
        "| 类别 | 用例数 | 通过 | 未通过 | 通过率 | 平均分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, bucket in (summary.get("categories") or {}).items():
        lines.append(
            f"| {category} | {bucket['total']} | {bucket['passed']} | {bucket['failed']} | {bucket['pass_rate']}% | {bucket['average_score']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 失败分析矩阵",
            "",
            "| 用例 | 类别 | 是否通过 | 失败类型 | 失败检查 | 建议动作 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in results:
        if item["passed"]:
            continue
        lines.append(
            f"| {item['id']} | {item['category']} | 否 | {item['likely_failure_type']} | {', '.join(item['failed_checks'])} | {item['recommended_action']} |"
        )
    if all(item["passed"] for item in results):
        lines.append("| 无 | - | 全部通过 | none | 无 | 继续扩展更难用例 |")
    lines.extend(
        [
            "",
            "## 5. 用例明细",
        "",
        ]
    )
    for item in results:
        lines.extend(_render_case(item))
    lines.extend(
        [
            "## 6. 初步判断",
            "",
            _overall_judgement(summary),
            "",
            "## 7. 覆盖不足与后续优化方向",
            "",
            "- 当前评估仍以规则检查为主，无法替代人工语义审阅。",
            "- 当前专项评测集围绕 Harness Engineering，其他 topic 仍需要独立扩展。",
            "- 通过率高说明当前规则覆盖的质量项满足要求，不代表不存在更隐蔽的事实或推理问题。",
            "- 若 `must_include_all` 或 `must_include_any` 失败较多，优先检查检索召回和回答 prompt。",
            "- 若 `has_citations` 或 `topic_citations` 失败，优先修复引用抽取与来源路径。",
            "- 若 `answer_length` 失败较多，优先调整回答结构和最低详略要求。",
            "- 若不足承认失败，优先强化“证据不足必须说明”的约束。",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_case(item: dict[str, Any]) -> list[str]:
    answer = re.sub(r"\s+", " ", item["answer"]).strip()
    answer_excerpt = answer[:800] + ("..." if len(answer) > 800 else "")
    fallback_label = "是" if item["fallback_occurred"] else "否"
    if item["fallback_reason"]:
        fallback_label += f"（{item['fallback_reason']}）"
    citations = item["citations"][:4]
    citation_lines = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        citation_lines.append(f"- `{citation.get('path')}`：{str(citation.get('snippet') or '')[:160]}")
    semantic = item.get("semantic_evaluation") or {}
    retrieval = item.get("retrieval_evaluation") or {}
    retrieval_lines = [
        f"- 检索评测状态：{retrieval.get('status') or 'unavailable'} / {retrieval.get('quality_status') or 'not_evaluated'}",
    ]
    if retrieval.get("status") in {"completed", "partial"}:
        retrieval_lines.extend(
            [
                f"- 检索指标：{json.dumps(retrieval.get('metrics') or {}, ensure_ascii=False)}",
                f"- 检索命中标准证据：{', '.join(retrieval.get('matched_evidence_ids') or []) or '无'}",
                f"- 检索覆盖参考要点：{', '.join(retrieval.get('matched_point_ids') or []) or '无'}",
                f"- 检索缺失参考要点：{', '.join(retrieval.get('missing_point_ids') or []) or '无'}",
                f"- 检索判断理由：{retrieval.get('reason') or '无'}",
                f"- 检索建议：{retrieval.get('recommended_action') or '无'}",
            ]
        )
        for chunk in (retrieval.get("chunks") or [])[:6]:
            retrieval_lines.append(
                f"- 检索片段 #{chunk.get('rank')} `{chunk.get('path')}`："
                f"{chunk.get('relevance_label') or '未判定'} / "
                f"相关性={chunk.get('relevance_score') if chunk.get('relevance_score') is not None else '未提供'} / "
                f"证据命中={', '.join(chunk.get('matched_evidence_ids') or []) or '无'}"
            )
    elif retrieval.get("reason"):
        retrieval_lines.append(f"- 检索评测说明：{retrieval.get('reason')}")
    semantic_lines = [
        f"- 语义评测状态：{semantic.get('status') or 'not_applicable'} / {semantic.get('quality_status') or 'not_evaluated'}",
    ]
    if semantic.get("status") == "completed":
        semantic_lines.extend(
            [
                f"- 语义评分：{json.dumps(semantic.get('scores') or {}, ensure_ascii=False)}",
                f"- 已覆盖参考要点：{', '.join(semantic.get('matched_point_ids') or []) or '无'}",
                f"- 缺失参考要点：{', '.join(semantic.get('missing_point_ids') or []) or '无'}",
                f"- 未获支持的陈述：{'；'.join(semantic.get('unsupported_claims') or []) or '无'}",
                f"- 引用支撑问题：{'；'.join(semantic.get('citation_issues') or []) or '无'}",
                f"- 语义判断理由：{semantic.get('reason') or '无'}",
                f"- 语义建议：{semantic.get('recommended_action') or '无'}",
            ]
        )
    elif semantic.get("reason"):
        semantic_lines.append(f"- 语义评测说明：{semantic.get('reason')}")
    return [
        f"### {item['id']} - {'通过' if item['passed'] else '未通过'}",
        "",
        f"- 类别：{item['category']}",
        f"- 问题：{item['question']}",
        f"- 分数：{item['score']}",
        f"- 状态：{item['status']} / 请求后端={item['requested_backend']} / 实际后端={item['actual_backend']} / mode={item['mode']} / confidence={item['confidence']}",
        f"- 自动回退：{fallback_label}",
        f"- 引用数：{item['citation_count']}",
        f"- 失败检查：{', '.join(item['failed_checks']) if item['failed_checks'] else '无'}",
        f"- 失败类型：{item['likely_failure_type']}",
        f"- 诊断：{item['diagnosis']}",
        f"- 建议：{item['recommended_action']}",
        f"- 检查项：{json.dumps(item['checks'], ensure_ascii=False)}",
        *retrieval_lines,
        *semantic_lines,
        "",
        "回答摘录：",
        "",
        f"> {answer_excerpt}",
        "",
        "引用摘录：",
        "",
        *(citation_lines or ["- 无"]),
        "",
    ]


def _overall_judgement(summary: dict[str, Any]) -> str:
    pass_rate = float(summary.get("pass_rate") or 0)
    if pass_rate >= 80:
        return "当前知识库问答质量达到 MVP 可用水平，后续应围绕失败用例做定向优化。"
    if pass_rate >= 50:
        return "当前知识库问答链路可用，但稳定性和回答质量仍需要根据失败用例继续优化。"
    return "当前知识库问答质量不足，建议优先修复检索召回、引用生成和回答约束。"
