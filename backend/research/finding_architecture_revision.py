"""Shared final-output revision layer for the isolated architecture experiment."""

from __future__ import annotations

import re
from typing import Any

from .claims import split_atomic_units
from .finding_architecture_eval import (
    DirectEvidenceContextManager,
    EvaluationContractError,
    FrozenJob,
    RecordingJsonModel,
    _minimal_report_document,
    _packet_evidence_ids,
)
from .writer import (
    _annotate_claim_ledger,
    _call_writer_model,
    _citation_map,
    _evidence_metrics,
    _evidence_refs,
    _evidence_table_row,
    _render_internal_citations,
)


REVISION_SCHEMA_VERSION = "finding-architecture-shared-final-revision/v1"
REVISION_PROMPT_VERSION = "finding-architecture-shared-final-revision-prompt/v1"
_MARKER_RE = re.compile(r"\[\[(?:ev_\d{3,4})(?:\s*,\s*ev_\d{3,4})*\]\]")
_VISIBLE_CITATION_RE = re.compile(r"\[\d+\]")


def _normalize_internal_citation_syntax(content: str) -> tuple[str, list[dict[str, Any]]]:
    changes = []

    def replace(match: re.Match[str]) -> str:
        body = match.group(1)
        evidence_ids = re.findall(r"ev_\d{3,4}", body)
        residue = re.sub(r"ev_\d{3,4}|[\[\],\s]", "", body)
        if not evidence_ids or residue:
            return match.group(0)
        normalized = f"[[{', '.join(dict.fromkeys(evidence_ids))}]]"
        if normalized != match.group(0):
            changes.append({"original": match.group(0), "normalized": normalized})
        return normalized

    normalized = re.sub(r"\[\[(.*?)\]\]", replace, str(content or ""))

    # The revision contract uses postfix Evidence markers. Models commonly emit
    # ``claim。[[ev_001]] next claim``; the shared atomic splitter otherwise treats
    # that marker as a prefix of the next claim, which can corrupt pruning and the
    # claim ledger. Attach it to the sentence it follows before any coverage check.
    trailing_marker = re.compile(
        r"([。！？!?；;])([ \t]*)(\[\[(?:ev_\d{3,4})(?:\s*,\s*ev_\d{3,4})*\]\])"
    )

    def attach_to_preceding_sentence(match: re.Match[str]) -> str:
        original = match.group(0)
        attached = f"{match.group(3)}{match.group(1)}{match.group(2)}"
        changes.append({
            "original": original,
            "normalized": attached,
            "reason": "attach_postfix_marker_to_preceding_atomic_unit",
        })
        return attached

    return trailing_marker.sub(attach_to_preceding_sentence, normalized), changes


def _drop_uncovered_atomic_units(content: str) -> tuple[str, list[dict[str, Any]]]:
    kept_lines = []
    removed = []
    for line_number, raw_line in enumerate(str(content or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            kept_lines.append(raw_line)
            continue
        kept_units = []
        for unit in split_atomic_units(line):
            prose = _MARKER_RE.sub("", unit)
            prose = re.sub(r"^[\s>*+\-\d.()]+", "", prose).strip()
            prose = re.sub(r"[`*_]", "", prose).strip()
            substantive = len(re.sub(r"\s+", "", prose)) >= 4
            if substantive and not _MARKER_RE.search(unit):
                removed.append({"line_number": line_number, "unit": unit})
                continue
            kept_units.append(unit)
        if kept_units:
            kept_lines.append("".join(kept_units))
    return "\n".join(kept_lines).strip(), removed


def internal_citation_coverage(content: str) -> dict[str, Any]:
    """Require every substantive atomic prose unit to carry an internal Evidence marker."""
    units = []
    uncovered = []
    for line_number, raw_line in enumerate(str(content or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line):
            continue
        line_units = line.split("|") if line.startswith("|") else split_atomic_units(line)
        for unit in line_units:
            prose = _MARKER_RE.sub("", str(unit or ""))
            prose = re.sub(r"^[\s>*+\-\d.()]+", "", prose).strip()
            prose = re.sub(r"[`*_]", "", prose).strip()
            if len(re.sub(r"\s+", "", prose)) < 4:
                continue
            row = {
                "line_number": line_number,
                "unit": str(unit).strip(),
                "prose": prose,
                "has_internal_citation": bool(_MARKER_RE.search(str(unit))),
            }
            units.append(row)
            if not row["has_internal_citation"]:
                uncovered.append(row)
    return {
        "status": "passed" if units and not uncovered else "failed",
        "unit_count": len(units),
        "covered_unit_count": len(units) - len(uncovered),
        "uncovered_unit_count": len(uncovered),
        "coverage_rate": round((len(units) - len(uncovered)) / len(units), 3) if units else 0.0,
        "uncovered_units": uncovered,
    }


def _strip_reader_citations(content: str) -> str:
    return _VISIBLE_CITATION_RE.sub("", str(content or "")).strip()


def _issue_packet(adjudication: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(adjudication.get("rows") or []):
        review = row.get("human_review") or {}
        verdict = str(review.get("verdict") or "")
        if verdict in {"entailed", "reasonable_synthesis", "non_claim_narration"}:
            continue
        output.append({
            "review_unit_index": index,
            "claim": row.get("claim"),
            "claim_context": row.get("claim_context"),
            "verdict": verdict,
            "severity": review.get("severity"),
            "reason": review.get("reason"),
            "cited_evidence_ids": row.get("cited_evidence_ids") or [],
            "adjacent_cited_evidence_ids": row.get("adjacent_cited_evidence_ids") or [],
        })
    return output


def build_revision_payload(
    *,
    opaque_candidate_id: str,
    section: dict[str, Any],
    original_content: str,
    adjudication: dict[str, Any],
    evidence_packet: dict[str, Any],
    allowed_evidence_ids: set[str],
) -> dict[str, Any]:
    return {
        "task": "依据冻结 Evidence 修订一节不合格的中文研究报告。只返回 JSON。",
        "prompt_version": REVISION_PROMPT_VERSION,
        "opaque_candidate_id": opaque_candidate_id,
        "section_brief": {
            key: value for key, value in section.items()
            if key not in {"dossier", "finding_ids", "evidence_ids"}
        },
        "original_draft_without_reader_citations": _strip_reader_citations(original_content),
        "known_fidelity_issues": _issue_packet(adjudication),
        "allowed_evidence_ids": sorted(allowed_evidence_ids),
        "evidence_packet": evidence_packet,
        "rules": [
            "只使用 evidence_packet；不得使用外部知识，也不得推测候选来自哪个实验组。",
            "这是重写而不是润色：删除所有 Evidence 不完整支持的事实、因果、目的、性能、范围、比较或建议。",
            "每个可核验的事实性子句和分析结论都必须在同一原子单元内带 [[ev_XXX]]；句末引用不得跨越前一句。",
            "综合多条 Evidence 时列出全部桥接 ID，唯一合法格式是 [[ev_001, ev_002]]；不得写成 [[ev_001], [ev_002]]，且综合不得新增 Evidence 中不存在的机制或评价。",
            "主体、比较轴、来源角色和适用范围必须保持一致；来源方自述必须明确归因。",
            "冲突、限制和不确定性不得省略；证据不足时明确缩短或删除，不能用常识补齐。",
            "只可逐字使用 allowed_evidence_ids；禁止 Finding ID、读者引用 [N]、Sources 章节和原始 Quote 转录。",
            "使用连贯中文段落，不使用 Markdown 表格或项目符号，以避免引用作用域不清。",
            "目标不是保持原稿长度，而是得到可以逐句回溯原始 Evidence 的最短完整答案。",
        ],
        "schema": {"content": "带 [[ev_XXX]] 内部引用标记的修订后 Markdown 正文"},
    }


def revise_candidate(
    *,
    opaque_candidate_id: str,
    candidate: dict[str, Any],
    adjudication: dict[str, Any],
    job: FrozenJob,
    model: RecordingJsonModel,
    context_manager: DirectEvidenceContextManager,
    max_attempts: int = 3,
) -> dict[str, Any]:
    documents = [item for item in candidate.get("documents") or [] if isinstance(item, dict)]
    if len(documents) != 1:
        raise EvaluationContractError("shared final revision currently requires one section-stage document")
    original = documents[0]
    section_id = str(original.get("section_id") or "")
    section = next(
        (
            item for item in job.report_blueprint.get("sections") or []
            if str(item.get("section_id") or "") == section_id
        ),
        None,
    )
    if not isinstance(section, dict):
        raise EvaluationContractError(f"revision section missing from frozen Blueprint: {section_id}")
    direct_inputs = {"evidence_map": job.evidence_map, "question_states": job.question_states}
    packet, context_diagnostics = context_manager.build(direct_inputs, section)
    allowed_ids = _packet_evidence_ids(packet)
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in job.evidence_map
        if str(item.get("id") or "") in allowed_ids
    }
    payload = build_revision_payload(
        opaque_candidate_id=opaque_candidate_id,
        section=section,
        original_content=str(original.get("content") or ""),
        adjudication=adjudication,
        evidence_packet=packet,
        allowed_evidence_ids=allowed_ids,
    )
    content = ""
    coverage = internal_citation_coverage("")
    attempts = []
    normalization_changes = []
    deterministic_pruning = {"applied": False, "removed_units": []}
    active_payload = payload
    for attempt in range(1, max(1, max_attempts) + 1):
        content, meta = _call_writer_model(
            model,
            active_payload,
            evidence_by_id,
            allowed_finding_ids=set(),
        )
        content, changes = _normalize_internal_citation_syntax(content)
        normalization_changes.extend({"attempt": attempt, **row} for row in changes)
        coverage = internal_citation_coverage(content)
        attempts.append({
            "attempt": attempt,
            "model_meta": meta,
            "citation_coverage": coverage,
        })
        if content and coverage["status"] == "passed":
            break
        active_payload = {
            **payload,
            "task": "修复修订稿中无引用、引用作用域或 Evidence ID 问题。只返回 JSON。",
            "invalid_revision": content or str(meta.get("invalid_content") or ""),
            "uncovered_units": coverage.get("uncovered_units") or [],
            "invalid_markers": meta.get("invalid_markers") or [],
            "rules": [
                *payload["rules"],
                "上一稿未通过确定性全文引用覆盖检查。必须让每个 substantive atomic unit 在本单元内拥有真实支持它的 [[ev_XXX]]，否则删除该单元。",
            ],
        }
    if content and coverage["status"] != "passed":
        pruned_content, removed_units = _drop_uncovered_atomic_units(content)
        pruned_coverage = internal_citation_coverage(pruned_content)
        deterministic_pruning = {
            "applied": bool(removed_units),
            "removed_units": removed_units,
            "coverage_after_pruning": pruned_coverage,
        }
        if pruned_content and pruned_coverage["status"] == "passed":
            content = pruned_content
            coverage = pruned_coverage
    if not content or coverage["status"] != "passed":
        return {
            "arm": candidate.get("arm"),
            "documents": [],
            "editorial": {},
            "report_document": {"content": "", "claim_ledger": []},
            "revision_diagnostics": {
                "status": "failed",
                "schema_version": REVISION_SCHEMA_VERSION,
                "attempts": attempts,
                "normalization_changes": normalization_changes,
                "deterministic_pruning": deterministic_pruning,
                "context_diagnostics": context_diagnostics,
            },
            "model_calls": model.calls,
        }
    citation_map = _citation_map(job.evidence_map)
    rendered, ledger = _render_internal_citations(content, evidence_by_id, citation_map)
    ledger = _annotate_claim_ledger(ledger, [], evidence_by_id, section_id)
    used_ids = list(dict.fromkeys(
        evidence_id for row in ledger for evidence_id in row.get("evidence_ids") or []
    ))
    evidence_table = [
        _evidence_table_row(evidence_by_id[evidence_id])
        for evidence_id in used_ids if evidence_id in evidence_by_id
    ]
    document = {
        "title": original.get("title") or section.get("title"),
        "section_id": section_id,
        "content": rendered.strip(),
        "evidence_refs": _evidence_refs([
            evidence_by_id[evidence_id] for evidence_id in used_ids if evidence_id in evidence_by_id
        ]),
        "evidence_table": evidence_table,
        "claim_ledger": ledger,
        "section_brief": section,
        "paragraph_validation": {"status": "passed", "failed_group_count": 0, "groups": []},
        "evidence_metrics": _evidence_metrics(evidence_table, ledger),
        "synthesis_diagnostics": {
            "mode": "shared_final_output_revision",
            "model_attempts": len(attempts),
            "input_evidence_count": len(allowed_ids),
            "used_evidence_count": len(used_ids),
        },
    }
    return {
        "arm": candidate.get("arm"),
        "documents": [document],
        "editorial": {},
        "report_document": _minimal_report_document(document, job.report_blueprint),
        "revision_diagnostics": {
            "status": "passed",
            "schema_version": REVISION_SCHEMA_VERSION,
            "prompt_version": REVISION_PROMPT_VERSION,
            "attempts": attempts,
            "normalization_changes": normalization_changes,
            "deterministic_pruning": deterministic_pruning,
            "final_internal_citation_coverage": coverage,
            "context_diagnostics": context_diagnostics,
            "allowed_evidence_ids": sorted(allowed_ids),
            "used_evidence_ids": used_ids,
            "original_issue_count": len(_issue_packet(adjudication)),
        },
        "model_calls": model.calls,
    }
