"""Final-output adjudication for the isolated Finding architecture experiment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable


SCHEMA_VERSION = "finding-architecture-absolute-output-adjudication/v1"

SUPPORTED_VERDICTS = {
    "entailed",
    "reasonable_synthesis",
    "partial_support",
    "missing_citation",
    "unsupported_addition",
    "scope_mismatch",
    "subject_mismatch",
    "comparison_axis_mismatch",
    "source_role_mismatch",
    "contradicted_by_citation",
    "conflict_or_limitation_omission",
    "non_atomic",
    "non_claim_narration",
}
SUPPORTED_SEVERITIES = {"none", "minor", "major", "critical"}
SUPPORTED_CRITERION_VERDICTS = {
    "covered_correctly",
    "covered_but_overstated",
    "partially_covered",
    "not_covered",
    "not_applicable",
}
PASS_VERDICTS = {"entailed", "reasonable_synthesis", "non_claim_narration"}
FIDELITY_ERROR_VERDICTS = SUPPORTED_VERDICTS - PASS_VERDICTS


def _clip(value: Any, limit: int = 3500) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}\n[truncated]"


def _review_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = row.get("original_evidence") or row.get("adjacent_original_evidence") or []
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "atomic_claim": _clip(item.get("atomic_claim"), 1000),
            "quote": _clip(item.get("quote")),
            "source_title": item.get("source_title"),
            "source_type": item.get("source_type"),
            "publisher_relation": item.get("publisher_relation"),
            "subject": item.get("subject"),
            "subject_relation": item.get("subject_relation"),
            "comparison_axis": item.get("comparison_axis"),
            "expected_comparison_axis": item.get("expected_comparison_axis"),
            "scope": item.get("scope"),
            "stance": item.get("stance"),
            "support_relation": item.get("support_relation"),
            "target_fit": item.get("target_fit"),
            "conflict_or_limitation": bool(item.get("conflict_or_limitation")),
        }
        for item in source_rows
        if isinstance(item, dict)
    ]


def claim_review_payload(
    *,
    opaque_candidate_id: str,
    rows: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "task": "盲审最终报告中的事实单元与其原始 Evidence。只返回 JSON。",
        "opaque_candidate_id": opaque_candidate_id,
        "rules": [
            "不得推测候选属于哪个实验组，不得使用外部知识，只依据给出的 claim、上下文和原始 quote。",
            "逐个 review_id 判断完整陈述是否被引用 Evidence 支持；Evidence 只支持其中一部分时必须判 partial_support。",
            "正文增加原文没有的因果、性能、目的、适用范围、主体、比较结论或评价时，按最具体的错误类型判定。",
            "引用与正文正面冲突时判 contradicted_by_citation；不要把一般性的证据不足误判成正面矛盾。",
            "claim_ledger_gap 的 cited_evidence_ids 为空，但 adjacent Evidence 可能是同一句句末引用；必须结合 claim_context 判断引用作用域。",
            "non_claim_narration 只用于标题式、过渡式或不包含可核验事实的文字，不得用它放过评价性或机制性陈述。",
            "severity=critical：错误改变核心结论、关键比较方向或安全/决策建议；major：影响主要论证或 hard Criterion；minor：局部限定、措辞或非核心细节；无错误为 none。",
            "每条必须给出简短、可复核的中文 reason，并列出实际支持该判断的 evidence_ids。",
        ],
        "review_units": [
            {
                "review_id": f"unit_{index:04d}",
                "origin": row.get("review_unit_origin"),
                "claim": row.get("claim"),
                "claim_context": row.get("claim_context"),
                "claim_type": row.get("claim_type"),
                "support_scope": row.get("support_scope"),
                "cited_evidence_ids": row.get("cited_evidence_ids") or [],
                "adjacent_cited_evidence_ids": row.get("adjacent_cited_evidence_ids") or [],
                "evidence": _review_evidence(row),
            }
            for index, row in rows
        ],
        "schema": {
            "judgments": [
                {
                    "review_id": "unit_0000",
                    "verdict": sorted(SUPPORTED_VERDICTS),
                    "severity": sorted(SUPPORTED_SEVERITIES),
                    "supported_evidence_ids": ["ev_001"],
                    "reason": "简短中文理由",
                }
            ]
        },
    }


def parse_claim_review(
    result: dict[str, Any],
    rows: list[tuple[int, dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    expected = {f"unit_{index:04d}": index for index, _ in rows}
    parsed: dict[int, dict[str, Any]] = {}
    for item in result.get("judgments") or []:
        if not isinstance(item, dict) or str(item.get("review_id") or "") not in expected:
            continue
        verdict = str(item.get("verdict") or "")
        severity = str(item.get("severity") or "")
        if verdict not in SUPPORTED_VERDICTS or severity not in SUPPORTED_SEVERITIES:
            continue
        if verdict in PASS_VERDICTS:
            severity = "none"
        elif severity == "none":
            severity = "minor"
        parsed[expected[str(item["review_id"])]] = {
            "status": "reviewed",
            "verdict": verdict,
            "severity": severity,
            "supported_evidence_ids": sorted({
                str(value) for value in item.get("supported_evidence_ids") or [] if str(value)
            }),
            "reason": _clip(item.get("reason"), 1000),
        }
    return parsed


def review_claim_rows(
    *,
    opaque_candidate_id: str,
    claim_audit: dict[str, Any],
    model: Callable[[dict[str, Any]], dict[str, Any]],
    batch_size: int = 6,
    max_attempts: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(item) for item in claim_audit.get("rows") or [] if isinstance(item, dict)]
    traces: list[dict[str, Any]] = []
    pending = set(range(len(rows)))
    for attempt in range(max(1, max_attempts)):
        if not pending:
            break
        ordered = sorted(pending, reverse=bool(attempt % 2))
        for offset in range(0, len(ordered), max(1, batch_size)):
            indices = ordered[offset:offset + max(1, batch_size)]
            batch = [(index, rows[index]) for index in indices]
            payload = claim_review_payload(opaque_candidate_id=opaque_candidate_id, rows=batch)
            result = model(payload)
            traces.append({"attempt": attempt + 1, "indices": indices, "result": result})
            for index, judgment in parse_claim_review(result, batch).items():
                rows[index]["human_review"] = judgment
                pending.discard(index)
    for index in pending:
        rows[index]["human_review"] = {
            "status": "unresolved",
            "verdict": None,
            "severity": None,
            "supported_evidence_ids": [],
            "reason": "review model did not return a valid judgment after retries",
        }
    return rows, traces


def _criterion_evidence(
    criterion: dict[str, Any],
    evidence_by_cluster: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    cluster_ids = {
        str(value)
        for key in ("required_claim_cluster_ids", "optional_claim_cluster_ids", "counter_claim_cluster_ids")
        for value in criterion.get(key) or []
        if str(value)
    }
    rows = []
    for cluster_id in sorted(cluster_ids):
        for item in evidence_by_cluster.get(cluster_id, []):
            rows.append({
                "claim_cluster_id": cluster_id,
                "evidence_id": item.get("id"),
                "atomic_claim": _clip(item.get("atomic_claim") or item.get("claim"), 1000),
                "quote": _clip(item.get("cleaned_quote") or item.get("raw_quote") or item.get("evidence_text")),
                "scope": item.get("scope"),
                "stance": item.get("stance"),
                "subject": item.get("subject"),
                "comparison_axis": item.get("comparison_axis"),
            })
    return rows


def criterion_review_payload(
    *,
    opaque_candidate_id: str,
    report_content: str,
    inventory: dict[str, Any],
    evidence_map: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_map:
        cluster_id = str(item.get("claim_cluster_id") or "")
        if cluster_id:
            evidence_by_cluster.setdefault(cluster_id, []).append(item)
    criteria = []
    criterion_ordinal = 0
    for section in inventory.get("sections") or []:
        for criterion in section.get("criteria") or []:
            if str(criterion.get("level") or "") not in {"hard", "soft"}:
                continue
            criterion_ordinal += 1
            criteria.append({
                "criterion_key": (
                    f"{section.get('section_id')}::{criterion.get('criterion_id')}::{criterion_ordinal:03d}"
                ),
                "criterion_id": criterion.get("criterion_id"),
                "section_id": section.get("section_id"),
                "description": criterion.get("description"),
                "level": criterion.get("level"),
                "answerability": criterion.get("answerability"),
                "required_claim_cluster_ids": criterion.get("required_claim_cluster_ids") or [],
                "counter_claim_cluster_ids": criterion.get("counter_claim_cluster_ids") or [],
                "reference_evidence": _criterion_evidence(criterion, evidence_by_cluster),
            })
    return {
        "task": "盲审最终报告是否正确满足冻结的研究 Criterion，并评价整体成文质量。只返回 JSON。",
        "opaque_candidate_id": opaque_candidate_id,
        "report": _clip(report_content, 30000),
        "criteria": criteria,
        "rules": [
            "不得推测实验组，不得使用外部知识；只以报告、Criterion 和原始 Evidence 为依据。",
            "只有报告明确且正确表达 Criterion 所需内容才算 covered_correctly；只出现关键词或引用 cluster 不算覆盖。",
            "报告遗漏关键部分判 partially_covered 或 not_covered；扩大 Evidence 范围或隐藏冲突判 covered_but_overstated。",
            "answerability=conflicted 的 Criterion 只有同时呈现支持与限制边界才可 covered_correctly。",
            "整体质量评价关注：是否直接回答问题、结构是否清楚、综合是否克制、限制是否明确、是否可供用户实际使用。",
            "quality score 使用 1-5 整数；release_ready 只有在无明显事实/引用风险且 hard Criterion 完整时才可为 true。",
        ],
        "schema": {
            "criteria": [
                {
                    "criterion_key": "sec_00::criterion_01::001",
                    "criterion_id": "criterion id",
                    "verdict": sorted(SUPPORTED_CRITERION_VERDICTS),
                    "severity": sorted(SUPPORTED_SEVERITIES),
                    "reason": "简短中文理由",
                }
            ],
            "overall_quality": {
                "directness": 1,
                "coherence": 1,
                "synthesis": 1,
                "boundary_handling": 1,
                "usefulness": 1,
                "release_ready": False,
                "reason": "简短中文理由",
            },
        },
    }


def parse_criterion_review(result: dict[str, Any]) -> dict[str, Any]:
    criteria = []
    for item in result.get("criteria") or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "")
        severity = str(item.get("severity") or "")
        if verdict not in SUPPORTED_CRITERION_VERDICTS or severity not in SUPPORTED_SEVERITIES:
            continue
        if verdict in {"covered_correctly", "not_applicable"}:
            severity = "none"
        elif severity == "none":
            severity = "major" if verdict == "not_covered" else "minor"
        criteria.append({
            "criterion_key": str(item.get("criterion_key") or ""),
            "criterion_id": str(item.get("criterion_id") or ""),
            "verdict": verdict,
            "severity": severity,
            "reason": _clip(item.get("reason"), 1000),
        })
    quality_raw = result.get("overall_quality") if isinstance(result.get("overall_quality"), dict) else {}
    quality: dict[str, Any] = {}
    for key in ("directness", "coherence", "synthesis", "boundary_handling", "usefulness"):
        try:
            quality[key] = max(1, min(5, int(quality_raw.get(key))))
        except (TypeError, ValueError):
            quality[key] = None
    quality["release_ready"] = bool(quality_raw.get("release_ready"))
    quality["reason"] = _clip(quality_raw.get("reason"), 1200)
    values = [value for key, value in quality.items() if key not in {"release_ready", "reason"} and isinstance(value, int)]
    quality["average"] = round(sum(values) / len(values), 2) if values else None
    return {"criteria": criteria, "overall_quality": quality}


def summarize_adjudication(
    *,
    rows: list[dict[str, Any]],
    criterion_review: dict[str, Any],
) -> dict[str, Any]:
    reviewed = [row for row in rows if (row.get("human_review") or {}).get("status") == "reviewed"]
    unresolved = len(rows) - len(reviewed)
    verdicts = Counter(str((row.get("human_review") or {}).get("verdict") or "") for row in reviewed)
    severities = Counter(str((row.get("human_review") or {}).get("severity") or "") for row in reviewed)
    factual_rows = [row for row in reviewed if (row.get("human_review") or {}).get("verdict") != "non_claim_narration"]
    supported = sum((row.get("human_review") or {}).get("verdict") in {"entailed", "reasonable_synthesis"} for row in factual_rows)
    errors = sum((row.get("human_review") or {}).get("verdict") in FIDELITY_ERROR_VERDICTS for row in factual_rows)
    criteria = criterion_review.get("criteria") or []
    hard = [
        row for row in criteria
        if str(row.get("level") or "") == "hard"
        and str(row.get("answerability") or "") in {"supported", "partial", "conflicted"}
    ]
    hard_correct = sum(row.get("verdict") == "covered_correctly" for row in hard)
    quality = criterion_review.get("overall_quality") or {}
    blocking_verdicts = {
        "missing_citation",
        "unsupported_addition",
        "scope_mismatch",
        "subject_mismatch",
        "comparison_axis_mismatch",
        "source_role_mismatch",
        "contradicted_by_citation",
        "conflict_or_limitation_omission",
        "partial_support",
        "non_atomic",
    }
    major_or_critical = sum(severities[level] for level in ("major", "critical"))
    blocking_claims = sum(verdicts[item] for item in blocking_verdicts)
    effective_release_ready = bool(
        quality.get("release_ready") is True
        and errors == 0
        and unresolved == 0
        and major_or_critical == 0
    )
    quality = {
        **quality,
        "judge_declared_release_ready": bool(quality.get("release_ready")),
        "release_ready": effective_release_ready,
    }
    strict_release_pass = bool(
        rows
        and not unresolved
        and severities["critical"] == 0
        and severities["major"] == 0
        and blocking_claims == 0
        and hard_correct == len(hard)
        and errors == 0
        and quality.get("release_ready") is True
        and (quality.get("average") or 0) >= 4.0
    )
    return {
        "review_unit_count": len(rows),
        "reviewed_unit_count": len(reviewed),
        "unresolved_unit_count": unresolved,
        "factual_unit_count": len(factual_rows),
        "verdict_counts": dict(sorted(verdicts.items())),
        "severity_counts": dict(sorted(severities.items())),
        "supported_count": supported,
        "fidelity_error_count": errors,
        "support_rate": round(supported / len(factual_rows), 3) if factual_rows else None,
        "fidelity_error_rate": round(errors / len(factual_rows), 3) if factual_rows else None,
        "major_or_critical_error_count": major_or_critical,
        "hard_criterion_count": len(hard),
        "hard_criterion_correct_count": hard_correct,
        "hard_criterion_correct_rate": round(hard_correct / len(hard), 3) if hard else None,
        "overall_quality": quality,
        "strict_release_pass": strict_release_pass,
        "strict_release_rule": (
            "all units resolved; zero fidelity errors of any severity; zero critical/major errors; "
            "zero blocking fidelity verdicts; "
            "all hard criteria covered correctly; overall release_ready=true and average>=4.0"
        ),
    }


def adjudicate_candidate(
    *,
    opaque_candidate_id: str,
    candidate: dict[str, Any],
    claim_audit: dict[str, Any],
    inventory: dict[str, Any],
    evidence_map: list[dict[str, Any]],
    model: Callable[[dict[str, Any]], dict[str, Any]],
    batch_size: int = 6,
) -> dict[str, Any]:
    rows, claim_traces = review_claim_rows(
        opaque_candidate_id=opaque_candidate_id,
        claim_audit=claim_audit,
        model=model,
        batch_size=batch_size,
    )
    criterion_payload = criterion_review_payload(
        opaque_candidate_id=opaque_candidate_id,
        report_content=str((candidate.get("report_document") or {}).get("content") or ""),
        inventory=inventory,
        evidence_map=evidence_map,
    )
    criterion_result = model(criterion_payload)
    criterion_review = parse_criterion_review(criterion_result)
    criterion_contract = criterion_review_payload(
        opaque_candidate_id=opaque_candidate_id,
        report_content="",
        inventory=inventory,
        evidence_map=evidence_map,
    )["criteria"]
    level_by_key = {
        str(criterion.get("criterion_key") or ""): str(criterion.get("level") or "")
        for criterion in criterion_contract
    }
    answerability_by_key = {
        str(criterion.get("criterion_key") or ""): str(criterion.get("answerability") or "")
        for criterion in criterion_contract
    }
    for row in criterion_review["criteria"]:
        key = str(row.get("criterion_key") or "")
        row["level"] = level_by_key.get(key)
        row["answerability"] = answerability_by_key.get(key)
    return {
        "schema_version": SCHEMA_VERSION,
        "opaque_candidate_id": opaque_candidate_id,
        "review_basis": "final report versus frozen original Evidence Quote and Reference Evidence Inventory",
        "external_knowledge_allowed": False,
        "arm_visible_to_judge": False,
        "rows": rows,
        "criterion_review": criterion_review,
        "summary": summarize_adjudication(rows=rows, criterion_review=criterion_review),
        "judge_traces": {
            "claim_batches": claim_traces,
            "criterion_result": criterion_result,
        },
    }
