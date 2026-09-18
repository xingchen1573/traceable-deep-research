"""Evidence-native report writing without a Finding intermediary.

This module is production-neutral: it does not read jobs, databases, or experiment
artifacts. Callers provide frozen accepted Evidence and model callables explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from .audit import audit_scope, submit_with_audit_context
from .claims import split_atomic_units
from .quality import (
    ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION,
    ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
    validate_claim_entailment,
)
from .synthesis import JsonModel
from .text_utils import normalize_for_substring
from .writer import (
    _annotate_claim_ledger,
    _assemble_report,
    _call_writer_model,
    _citation_map,
    _delivery_completeness,
    _evidence_metrics,
    _evidence_refs,
    _evidence_table_row,
    _excluded_from_assembled_body,
    _is_reference_title,
    _material_utilization,
    _normalize_internal_citation_markers,
    _reference_title_from_blueprint,
    _render_evidence_gap,
    _render_internal_citations,
    _report_label,
    _section_is_underdeveloped,
    _target_length_chars,
)


ARCHITECTURE = "evidence_native_eplus"
ARCHITECTURE_VERSION = "eplus/v3"
CONTEXT_VERSION = "eplus-context/2"
PROMPT_VERSION = "eplus-writer/2"


class EvidenceNativeContractError(ValueError):
    """Raised when E+ inputs cannot be used without weakening provenance."""


class EvidenceNativeCheckpointStore:
    """Content-addressed, architecture-scoped checkpoints for resumable E+ writing."""

    schema_version = "evidence-native-checkpoint/v1"

    def __init__(self, root: Path, *, contract: dict[str, Any]) -> None:
        self.root = Path(root)
        self.contract = copy.deepcopy(contract)
        self.contract_hash = stable_hash(self.contract)
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "global"))[:120]

    def _path(self, stage: str, scope: str) -> Path:
        return self.root / self._safe(stage) / f"{self._safe(scope)}.json"

    def load(self, stage: str, scope: str, inputs: Any) -> dict[str, Any] | None:
        path = self._path(stage, scope)
        if not path.is_file():
            self.events.append({"stage": stage, "scope": scope, "outcome": "miss"})
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.events.append(
                {"stage": stage, "scope": scope, "outcome": "rejected", "reason": type(exc).__name__}
            )
            return None
        expected_document_hash = str(document.get("checkpoint_sha256") or "")
        unhashed = {key: value for key, value in document.items() if key != "checkpoint_sha256"}
        expected = {
            "schema_version": self.schema_version,
            "architecture": ARCHITECTURE,
            "architecture_version": ARCHITECTURE_VERSION,
            "contract_sha256": self.contract_hash,
            "stage": stage,
            "scope": scope,
            "input_sha256": stable_hash(inputs),
        }
        mismatches = [key for key, value in expected.items() if document.get(key) != value]
        if not expected_document_hash or stable_hash(unhashed) != expected_document_hash:
            mismatches.append("checkpoint_sha256")
        payload = document.get("payload")
        if not isinstance(payload, dict) or stable_hash(payload) != str(document.get("output_sha256") or ""):
            mismatches.append("output_sha256")
        if mismatches:
            self.events.append(
                {
                    "stage": stage,
                    "scope": scope,
                    "outcome": "rejected",
                    "reason": "hash_or_contract_mismatch",
                    "mismatches": sorted(set(mismatches)),
                    "path": str(path),
                }
            )
            return None
        self.events.append({"stage": stage, "scope": scope, "outcome": "resumed", "path": str(path)})
        return copy.deepcopy(payload)

    def save(self, stage: str, scope: str, inputs: Any, payload: dict[str, Any]) -> str:
        path = self._path(stage, scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": self.schema_version,
            "architecture": ARCHITECTURE,
            "architecture_version": ARCHITECTURE_VERSION,
            "context_manager_version": CONTEXT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "contract_sha256": self.contract_hash,
            "stage": stage,
            "scope": scope,
            "input_sha256": stable_hash(inputs),
            "output_sha256": stable_hash(payload),
            "payload": copy.deepcopy(payload),
        }
        document["checkpoint_sha256"] = stable_hash(document)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        self.events.append({"stage": stage, "scope": scope, "outcome": "saved", "path": str(path)})
        return str(path)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def estimate_tokens(value: Any) -> int:
    # Deterministic conservative approximation. Selection only needs stable ordering/bounds.
    return max(1, (len(_stable_json(value)) + 2) // 3)


def _criterion_level(value: dict[str, Any]) -> str:
    level = str(value.get("level") or "soft").lower()
    return level if level in {"hard", "soft", "optional"} else "soft"


def _source_key(evidence: dict[str, Any]) -> str:
    return str(
        evidence.get("source_url")
        or evidence.get("canonical_project")
        or evidence.get("source_title")
        or ""
    )


def _evidence_question_ids(evidence: dict[str, Any]) -> set[str]:
    values = evidence.get("question_ids")
    shared_ids = values if isinstance(values, (list, tuple, set)) else [values]
    return {
        str(value)
        for value in [evidence.get("question_id"), *shared_ids]
        if str(value or "").strip()
    }


def _is_conflict(evidence: dict[str, Any]) -> bool:
    value = " ".join(
        [
            str(evidence.get("stance") or ""),
            str(evidence.get("evidence_type") or ""),
            str(evidence.get("claim_type") or ""),
            *[str(item) for item in evidence.get("quality_flags") or []],
        ]
    ).casefold()
    return any(token in value for token in ("counter", "conflict", "limitation", "contradict", "negative"))


def _is_citable(evidence: dict[str, Any]) -> bool:
    quote = str(
        evidence.get("cleaned_quote")
        or evidence.get("raw_quote")
        or evidence.get("evidence_text")
        or ""
    ).strip()
    return bool(
        str(evidence.get("id") or "").strip()
        and str(evidence.get("source_url") or "").strip()
        and quote
        and str(evidence.get("admission_status") or "accepted") in {"accepted", "inference_only"}
    )


def _sort_key(evidence: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if _is_conflict(evidence) else 1,
        -float(evidence.get("directness_score") or 0),
        -float(evidence.get("evidence_score") or 0),
        -float(evidence.get("authority_score") or 0),
        _source_key(evidence),
        str(evidence.get("id") or ""),
    )


def _cluster_key(evidence: dict[str, Any]) -> tuple[str, str, str, str]:
    evidence_id = str(evidence.get("id") or "")
    return (
        str(evidence.get("question_id") or ""),
        str(evidence.get("subject") or "__unknown_subject__"),
        str(
            evidence.get("comparison_axis")
            or evidence.get("expected_comparison_axis")
            or "__unknown_axis__"
        ),
        str(evidence.get("claim_cluster_id") or evidence_id),
    )


def _cell_key(evidence: dict[str, Any]) -> tuple[str, str, str]:
    subject = str(evidence.get("subject") or "")
    axis = str(evidence.get("comparison_axis") or evidence.get("expected_comparison_axis") or "")
    if subject or axis:
        return ("subject_axis", subject, axis)
    return ("matrix_cell", str(evidence.get("matrix_cell_id") or ""), "")


def _validate_evidence(evidence_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(item) for item in evidence_map if isinstance(item, dict)]
    invalid = [
        str(item.get("id") or f"row_{index:04d}")
        for index, item in enumerate(rows)
        if not _is_citable(item)
    ]
    if invalid:
        raise EvidenceNativeContractError(
            "evidence_map contains non-citable or non-admitted Evidence: " + ", ".join(invalid[:12])
        )
    ids = [str(item.get("id") or "") for item in rows]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise EvidenceNativeContractError(f"duplicate citable Evidence ids: {duplicates[:12]}")
    return sorted(rows, key=_sort_key)


def validate_evidence_traceability(
    evidence_map: list[dict[str, Any]], selected_chunks: list[dict[str, Any]]
) -> None:
    """Fail closed when citable Evidence cannot be traced to its frozen source Chunk."""
    chunks = {
        str(item.get("id") or ""): item
        for item in selected_chunks
        if isinstance(item, dict) and str(item.get("id") or "")
    }

    def comparable(value: Any) -> str:
        normalized = normalize_for_substring(unicodedata.normalize("NFKC", str(value or "")))
        return "".join(character for character in normalized if unicodedata.category(character) != "Cf")

    failures: list[str] = []
    for evidence in _validate_evidence(evidence_map):
        evidence_id = str(evidence.get("id") or "")
        chunk = chunks.get(str(evidence.get("chunk_id") or ""))
        quote = str(
            evidence.get("cleaned_quote")
            or evidence.get("raw_quote")
            or evidence.get("evidence_text")
            or ""
        )
        if not chunk or not comparable(quote) or comparable(quote) not in comparable(chunk.get("text")):
            failures.append(evidence_id)
    if failures:
        raise EvidenceNativeContractError(
            "citable Evidence is not traceable to selected Chunk content: " + ", ".join(failures[:12])
        )


def _plan_questions(section: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in section.get("questions") or section.get("research_questions") or []
        if isinstance(item, dict)
    ]


def _normalized_question_states(question_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for item in question_states:
        if not isinstance(item, dict):
            continue
        state = copy.deepcopy(item)
        if isinstance(state.get("criteria"), list):
            state["criteria"] = sorted(
                (row for row in state["criteria"] if isinstance(row, dict)),
                key=lambda row: str(row.get("criterion_id") or ""),
            )
        if isinstance(state.get("comparison_cells"), list):
            state["comparison_cells"] = sorted(
                (row for row in state["comparison_cells"] if isinstance(row, dict)),
                key=lambda row: str(row.get("cell_id") or row.get("id") or ""),
            )
        states.append(state)
    return sorted(
        states,
        key=lambda row: (str(row.get("section_id") or ""), str(row.get("question_id") or "")),
    )


def build_evidence_native_blueprint(
    *,
    normalized_request: dict[str, Any],
    research_plan: dict[str, Any],
    question_states: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]] | None = None,
    research_mode: str | None = None,
    evidence_landscape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic Blueprint using only plan, question states, and Evidence."""
    if selected_chunks is not None:
        validate_evidence_traceability(evidence_map, selected_chunks)
    evidence = _validate_evidence(evidence_map)
    states = _normalized_question_states(question_states)
    state_by_id = {str(item.get("question_id") or ""): item for item in states}
    evidence_by_section: dict[str, list[dict[str, Any]]] = {}
    for row in evidence:
        evidence_by_section.setdefault(str(row.get("section_id") or ""), []).append(row)

    sections: list[dict[str, Any]] = []
    core_ids: list[str] = []
    nonempty_core_ids: list[str] = []
    limited_ids: list[str] = []
    supplemental_empty_ids: list[str] = []
    for index, planned in enumerate(research_plan.get("sections") or []):
        if not isinstance(planned, dict):
            continue
        section_id = str(planned.get("section_id") or planned.get("id") or "").strip()
        if not section_id:
            continue
        questions: list[dict[str, Any]] = []
        known_gaps: list[str] = []
        hard_criterion_ids: set[str] = set()
        covered_hard_ids: set[str] = set()
        comparison_requested = False
        hard_question = False
        hard_state_requirements: list[dict[str, Any]] = []
        required_comparison_cells: dict[str, dict[str, Any]] = {}
        question_ids: set[str] = set()
        for q_index, planned_question in enumerate(_plan_questions(planned)):
            question_id = str(
                planned_question.get("question_id")
                or planned_question.get("id")
                or f"{section_id}_q{q_index + 1}"
            )
            question_ids.add(question_id)
            state = state_by_id.get(question_id, {})
            coverage = state.get("coverage") if isinstance(state.get("coverage"), dict) else {}
            criteria: list[dict[str, Any]] = []
            for criterion in state.get("criteria") or []:
                if not isinstance(criterion, dict):
                    continue
                criterion_id = str(criterion.get("criterion_id") or "")
                level = _criterion_level(criterion)
                status = str(criterion.get("status") or "uncovered")
                if level == "hard" and criterion_id:
                    hard_criterion_ids.add(criterion_id)
                    if status == "covered":
                        covered_hard_ids.add(criterion_id)
                criteria.append(
                    {
                        "criterion_id": criterion_id,
                        "description": criterion.get("description"),
                        "level": level,
                        "status": status,
                        "evidence_ids": [str(item) for item in criterion.get("evidence_ids") or []],
                        "preferred_source_types": copy.deepcopy(criterion.get("preferred_source_types") or []),
                    }
                )
            answer_level = str(
                planned_question.get("answer_contract_level")
                or state.get("answer_contract_level")
                or ""
            ).lower()
            hard_question = hard_question or answer_level == "hard" or any(
                item.get("level") == "hard" for item in criteria
            )
            if answer_level != "soft":
                hard_state_requirements.append(
                    {
                        "question_id": question_id,
                        "min_sources": int(state.get("min_sources") or 0),
                        "status": str(state.get("status") or ""),
                        "coverage": coverage,
                    }
                )
            for cell in state.get("comparison_cells") or []:
                if not isinstance(cell, dict):
                    continue
                cell_id = str(cell.get("cell_id") or cell.get("id") or "").strip()
                if cell_id:
                    required_comparison_cells[cell_id] = cell
            comparison_mode = str(planned_question.get("comparison_axis_mode") or "")
            comparison_requested = comparison_requested or comparison_mode not in {"", "not_applicable"}
            for gap in [
                *(coverage.get("missing_question_points") or []),
                *(coverage.get("enrichment_gaps") or []),
                *(coverage.get("gaps") or []),
            ]:
                text = str(
                    gap.get("missing_fact") or gap.get("description") or gap.get("gap_type") or ""
                    if isinstance(gap, dict)
                    else gap or ""
                ).strip()
                if text and text not in known_gaps:
                    known_gaps.append(text)
            questions.append(
                {
                    "question_id": question_id,
                    "question": state.get("question") or planned_question.get("text") or planned_question.get("question"),
                    "purpose": state.get("purpose") or planned_question.get("purpose"),
                    "answer_contract_level": answer_level or ("hard" if hard_question else "soft"),
                    "research_status": state.get("status"),
                    "coverage_status": coverage.get("coverage_status"),
                    "coverage_sufficient": coverage.get("sufficient"),
                    "criteria": criteria,
                }
            )

        rows = [
            row
            for row in evidence
            if str(row.get("section_id") or "") == section_id
            or bool(_evidence_question_ids(row) & question_ids)
        ]
        source_shortfalls: list[dict[str, Any]] = []
        for requirement in hard_state_requirements:
            question_id = str(requirement.get("question_id") or "")
            coverage = requirement.get("coverage") if isinstance(requirement.get("coverage"), dict) else {}
            checks = coverage.get("checks") if isinstance(coverage.get("checks"), dict) else {}
            if checks.get("source_floor") is False:
                source_shortfalls.append(
                    {
                        "question_id": question_id,
                        "required": int(
                            coverage.get("source_floor_target")
                            or requirement.get("min_sources")
                            or 0
                        ),
                        "actual": int(coverage.get("source_count") or 0),
                        "mode": coverage.get("source_floor_mode"),
                    }
                )
        missing_comparison_cell_ids = sorted(
            cell_id
            for cell_id, cell in required_comparison_cells.items()
            if str(cell.get("status") or "").lower() not in {"complete", "covered"}
        )
        overview = section_id == "sec_00" or "总览" in str(planned.get("title") or "")
        reference = _is_reference_title(str(planned.get("title") or ""))
        core = bool(hard_question and not overview and not reference)
        hard_complete = not hard_criterion_ids or hard_criterion_ids.issubset(covered_hard_ids)
        state_contract_complete = not any(
            (
                requirement.get("status") != "sufficient"
                if requirement.get("status")
                else requirement.get("coverage", {}).get("sufficient") is not True
            )
            for requirement in hard_state_requirements
        )
        comparison_complete = not missing_comparison_cell_ids
        if not rows:
            reportability_status = "empty"
        elif hard_complete and state_contract_complete and not source_shortfalls and comparison_complete:
            reportability_status = "ready"
        else:
            reportability_status = "limited"
        if core:
            core_ids.append(section_id)
            if reportability_status != "empty":
                nonempty_core_ids.append(section_id)
        if reportability_status == "limited":
            limited_ids.append(section_id)
        if reportability_status == "empty" and not core:
            supplemental_empty_ids.append(section_id)
        render_as = "executive_summary" if overview else "narrative"
        if comparison_requested and research_plan.get("subjects") and research_plan.get("comparison_axes"):
            render_as = "comparison_matrix"
        sections.append(
            {
                "section_id": section_id,
                "title": planned.get("title") or f"{index:02d}_{section_id}",
                "purpose": planned.get("purpose") or planned.get("intent") or "",
                "render_as": render_as,
                "writing_strategy": "evidence_native_cluster_guarded",
                "research_questions": questions,
                "known_gaps": known_gaps,
                "evidence_ids": [str(item.get("id") or "") for item in rows],
                "reportability": {
                    "status": reportability_status,
                    "evidence_count": len(rows),
                    "source_count": len({_source_key(item) for item in rows}),
                    "core_section": core,
                    "hard_criterion_count": len(hard_criterion_ids),
                    "covered_hard_criterion_count": len(covered_hard_ids),
                    "missing_hard_criterion_ids": sorted(hard_criterion_ids - covered_hard_ids),
                    "source_shortfalls": source_shortfalls,
                    "required_comparison_cell_ids": sorted(required_comparison_cells),
                    "missing_comparison_cell_ids": missing_comparison_cell_ids,
                    "hard_question_states_sufficient": state_contract_complete,
                    "reason": (
                        "no citable accepted Evidence is available"
                        if not rows
                        else "hard Criterion, source, or comparison coverage is incomplete"
                        if not (hard_complete and state_contract_complete and not source_shortfalls and comparison_complete)
                        else "citable accepted Evidence covers the planned hard contract"
                    ),
                },
            }
        )

    core_empty = sorted(set(core_ids) - set(nonempty_core_ids))
    resolved_mode = str(
        research_mode
        or normalized_request.get("research_mode")
        or research_plan.get("research_mode")
        or "domain_practice_review"
    )
    blueprint = {
        "blueprint_profile": ARCHITECTURE,
        "schema_version": "evidence-native-blueprint/v2",
        "blueprint_inputs": ["normalized_request", "research_plan", "question_states", "accepted_evidence"],
        "audience": {"language": "zh-CN", "reader": "research_user"},
        "completeness": {
            "status": "complete_report" if core_ids and not core_empty else "partial_report",
            "required_section_count": len(core_ids),
            "ready_section_count": len(nonempty_core_ids),
            "core_empty_section_ids": core_empty,
            "limited_section_ids": sorted(set(limited_ids)),
            "supplemental_empty_section_ids": sorted(set(supplemental_empty_ids)),
            "reason": "Evidence-native completeness projected from finalized Question State and citable Evidence",
        },
        "evidence_landscape": copy.deepcopy(evidence_landscape or {}),
        "report_type": research_plan.get("research_type") or "research_report",
        "research_mode": resolved_mode,
        "style": {"language": "zh-CN", "format": "multi_section_research_report"},
        "sections": sections,
    }
    blueprint["input_hash"] = stable_hash(
        {
            "normalized_request": normalized_request,
            "research_plan": research_plan,
            "question_states": states,
            "evidence": evidence,
        }
    )
    return blueprint


class EvidenceContextManager:
    def __init__(self, *, max_input_tokens: int = 1_000_000, prompt_reserve_tokens: int = 6500) -> None:
        self.max_input_tokens = max(2000, int(max_input_tokens))
        self.prompt_reserve_tokens = max(500, int(prompt_reserve_tokens))
        if self.prompt_reserve_tokens >= self.max_input_tokens:
            raise EvidenceNativeContractError("prompt reserve leaves no Evidence context budget")

    def build(
        self,
        *,
        section: dict[str, Any],
        question_states: list[dict[str, Any]],
        evidence_map: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = _validate_evidence(evidence_map)
        section_id = str(section.get("section_id") or "")
        question_ids = {
            str(item.get("question_id") or "")
            for item in section.get("research_questions") or []
            if isinstance(item, dict)
        }
        candidates = [
            item
            for item in evidence
            if str(item.get("section_id") or "") == section_id
            or bool(_evidence_question_ids(item) & question_ids)
        ]
        states = [
            item
            for item in _normalized_question_states(question_states)
            if isinstance(item, dict)
            and (
                str(item.get("section_id") or "") == section_id
                or str(item.get("question_id") or "") in question_ids
            )
        ]
        base = {
            "context_profile": "cluster_guarded",
            "context_manager_version": CONTEXT_VERSION,
            "section": {key: value for key, value in section.items() if key != "dossier"},
            "question_states": [
                {
                    "question_id": state.get("question_id"),
                    "question": state.get("question"),
                    "criteria": [
                        {
                            "criterion_id": criterion.get("criterion_id"),
                            "description": criterion.get("description"),
                            "level": _criterion_level(criterion),
                            "status": criterion.get("status"),
                        }
                        for criterion in state.get("criteria") or []
                        if isinstance(criterion, dict)
                    ],
                }
                for state in states
            ],
        }
        budget = self.max_input_tokens - self.prompt_reserve_tokens
        consumed = estimate_tokens(base)
        if consumed >= budget:
            raise EvidenceNativeContractError(
                f"section {section_id} fixed prompt/context overhead exceeds input budget"
            )

        by_id = {str(item.get("id") or ""): item for item in candidates}
        criterion_ids_by_evidence: dict[str, set[str]] = {}
        for state in states:
            for criterion in state.get("criteria") or []:
                if not isinstance(criterion, dict):
                    continue
                criterion_id = str(criterion.get("criterion_id") or "")
                if not criterion_id:
                    continue
                for evidence_id in criterion.get("evidence_ids") or []:
                    normalized_id = str(evidence_id or "")
                    if normalized_id in by_id:
                        criterion_ids_by_evidence.setdefault(normalized_id, set()).add(criterion_id)
        selected: list[str] = []
        selected_set: set[str] = set()
        selected_sources: set[str] = set()

        def add(item: dict[str, Any]) -> bool:
            nonlocal consumed
            evidence_id = str(item.get("id") or "")
            if not evidence_id or evidence_id in selected_set:
                return False
            cost = estimate_tokens(
                self._packet_row(
                    item,
                    criterion_ids=criterion_ids_by_evidence.get(evidence_id, set()),
                )
            )
            if consumed + cost > budget:
                return False
            selected.append(evidence_id)
            selected_set.add(evidence_id)
            selected_sources.add(_source_key(item))
            consumed += cost
            return True

        criteria: list[tuple[str, dict[str, Any]]] = []
        for state in states:
            question_id = str(state.get("question_id") or "")
            for criterion in state.get("criteria") or []:
                if isinstance(criterion, dict):
                    criteria.append((question_id, criterion))

        # 1. Hard Criterion breadth, round-robin across Criteria before adding depth.
        hard_groups: list[list[dict[str, Any]]] = []
        for question_id, criterion in sorted(criteria, key=lambda item: (item[0], str(item[1].get("criterion_id") or ""))):
            if _criterion_level(criterion) != "hard":
                continue
            criterion_id = str(criterion.get("criterion_id") or "")
            declared_ids = {str(item) for item in criterion.get("evidence_ids") or []}
            pool = [
                item
                for item in candidates
                if str(item.get("question_id") or "") == question_id
                and str(item.get("id") or "") in declared_ids
            ]
            representatives = [
                sorted(rows, key=_sort_key)[0]
                for _key, rows in sorted(
                    {
                        key: [item for item in pool if _cluster_key(item) == key]
                        for key in {_cluster_key(item) for item in pool}
                    }.items()
                )
                if rows
            ][:8]
            hard_groups.append(representatives)
        offset = 0
        while any(offset < len(group) for group in hard_groups):
            for group in hard_groups:
                if offset < len(group):
                    add(group[offset])
            offset += 1

        # 2. Comparison breadth.
        cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for item in candidates:
            key = _cell_key(item)
            if any(key[1:]):
                cells.setdefault(key, []).append(item)
        for key in sorted(cells):
            add(sorted(cells[key], key=_sort_key)[0])

        # 3. Conflicts and limitations.
        for item in sorted((item for item in candidates if _is_conflict(item)), key=_sort_key):
            add(item)
            support_rows = [
                row
                for row in candidates
                if _cluster_key(row) == _cluster_key(item) and not _is_conflict(row)
            ]
            if support_rows:
                add(sorted(support_rows, key=_sort_key)[0])

        # 4. One representative per guarded cluster, preferring new sources.
        clusters: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for item in candidates:
            clusters.setdefault(_cluster_key(item), []).append(item)
        for key in sorted(clusters):
            rows = sorted(
                clusters[key],
                key=lambda item: (1 if _source_key(item) in selected_sources else 0, _sort_key(item)),
            )
            if rows:
                add(rows[0])

        # 5. Fill remaining capacity source-diversely.
        for item in sorted(
            (item for item in candidates if str(item.get("id") or "") not in selected_set),
            key=lambda row: (1 if _source_key(row) in selected_sources else 0, _sort_key(row)),
        ):
            add(item)

        if candidates and not selected:
            raise EvidenceNativeContractError(
                f"section {section_id} has citable Evidence but none fits the configured context budget"
            )
        def assemble_packet() -> dict[str, Any]:
            packet_section = copy.deepcopy(base["section"])
            packet_section["evidence_ids"] = list(selected)
            return {
                **base,
                "section": packet_section,
                "schema_version": "evidence-context-packet/v2",
                "allowed_evidence_ids": list(selected),
                "claim_clusters": [
                    {
                        "synthesis_boundary_id": "boundary_"
                        + hashlib.sha256("\x1f".join(key).encode("utf-8")).hexdigest()[:12],
                        "question_id": key[0],
                        "criterion_ids": sorted(
                            {
                                str(criterion_id)
                                for item in rows
                                if str(item.get("id") or "") in selected_set
                                for criterion_id in criterion_ids_by_evidence.get(
                                    str(item.get("id") or ""), set()
                                )
                                if str(criterion_id)
                            }
                        ),
                        "subject": "unknown" if key[1].startswith("__unknown") else key[1],
                        "comparison_axis": "unknown" if key[2].startswith("__unknown") else key[2],
                        "claim_cluster_id": key[3],
                        "stances": sorted(
                            {
                                str(item.get("stance") or "unknown")
                                for item in rows
                                if str(item.get("id") or "") in selected_set
                            }
                        ),
                        "sources": sorted(
                            {
                                _source_key(item)
                                for item in rows
                                if str(item.get("id") or "") in selected_set
                            }
                        ),
                        "cross_boundary_synthesis": "forbidden_without_relational_quote",
                        "evidence": [
                            self._packet_row(
                                item,
                                criterion_ids=criterion_ids_by_evidence.get(
                                    str(item.get("id") or ""), set()
                                ),
                            )
                            for item in sorted(rows, key=_sort_key)
                            if str(item.get("id") or "") in selected_set
                        ],
                    }
                    for key, rows in sorted(clusters.items())
                    if any(str(item.get("id") or "") in selected_set for item in rows)
                ],
            }

        packet = assemble_packet()
        # Account for the two fixed-length hashes added below as part of the actual Writer payload.
        while selected and estimate_tokens(packet) + 64 > budget:
            removed = selected.pop()
            selected_set.remove(removed)
            packet = assemble_packet()
        if candidates and not selected:
            raise EvidenceNativeContractError(
                f"section {section_id} Evidence packet structure exceeds the configured context budget"
            )
        selected_rows = [by_id[item] for item in selected if item in by_id]
        packet["input_hash"] = stable_hash(
            {"section": packet["section"], "question_states": base["question_states"], "evidence": selected_rows}
        )
        packet["packet_hash"] = stable_hash(packet)
        selected_criterion_ids = {
            criterion_id
            for evidence_id in selected
            for criterion_id in criterion_ids_by_evidence.get(evidence_id, set())
        }
        diagnostics = {
            "context_manager_version": CONTEXT_VERSION,
            "max_input_tokens": self.max_input_tokens,
            "prompt_reserve_tokens": self.prompt_reserve_tokens,
            "material_budget_tokens": budget,
            "estimated_input_tokens": estimate_tokens(packet),
            "candidate_evidence_count": len(candidates),
            "candidate_evidence_ids": sorted(by_id),
            "selected_evidence_count": len(selected),
            "selected_evidence_ids": selected,
            "cropped_evidence_ids": sorted(set(by_id) - selected_set),
            "selected_source_count": len({_source_key(item) for item in selected_rows}),
            "selected_claim_cluster_count": len({_cluster_key(item) for item in selected_rows}),
            "selected_comparison_cells": [list(item) for item in sorted({_cell_key(item) for item in selected_rows if any(_cell_key(item)[1:])})],
            "selected_conflict_or_limitation_count": sum(_is_conflict(item) for item in selected_rows),
            "criterion_coverage": [
                {
                    "question_id": question_id,
                    "criterion_id": str(criterion.get("criterion_id") or ""),
                    "level": _criterion_level(criterion),
                    "selected": str(criterion.get("criterion_id") or "") in selected_criterion_ids,
                }
                for question_id, criterion in sorted(
                    criteria, key=lambda item: (item[0], str(item[1].get("criterion_id") or ""))
                )
            ],
            "cropping_reasons": {
                evidence_id: "context_budget_or_lower_selection_priority"
                for evidence_id in sorted(set(by_id) - selected_set)
            },
            "duplicate_evidence_id_count": 0,
            "source_dedupe_count": 0,
            "packet_hash": packet["packet_hash"],
        }
        return packet, diagnostics

    @staticmethod
    def _packet_row(
        evidence: dict[str, Any], *, criterion_ids: Iterable[str] | None = None
    ) -> dict[str, Any]:
        row = {
            "id": evidence.get("id"),
            "question_id": evidence.get("question_id"),
            "criterion_ids": sorted({str(item) for item in criterion_ids or [] if str(item)}),
            "extracted_criterion_ids": [
                str(item) for item in evidence.get("criterion_ids") or [] if str(item)
            ],
            "subject": evidence.get("subject"),
            "comparison_axis": evidence.get("comparison_axis") or evidence.get("expected_comparison_axis"),
            "matrix_cell_id": evidence.get("matrix_cell_id"),
            "claim_cluster_id": evidence.get("claim_cluster_id"),
            "atomic_claim": evidence.get("atomic_claim") or evidence.get("claim"),
            "quote": evidence.get("cleaned_quote") or evidence.get("raw_quote") or evidence.get("evidence_text"),
            "scope": evidence.get("scope") or "unknown",
            "claim_type": evidence.get("claim_type"),
            "stance": evidence.get("stance"),
            "support_relation": evidence.get("support_relation"),
            "target_fit": evidence.get("target_fit"),
            "source_title": evidence.get("source_title"),
            "source_url": evidence.get("source_url"),
            "source_type": evidence.get("source_type"),
            "publisher_relation": evidence.get("publisher_relation"),
            "conflict_or_limitation": _is_conflict(evidence),
        }
        row["content_sha256"] = stable_hash(row)
        return row


def _packet_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for cluster in packet.get("claim_clusters") or []
        if isinstance(cluster, dict)
        for row in cluster.get("evidence") or []
        if isinstance(row, dict)
    ]


def evidence_with_writer_context(
    evidence_map: list[dict[str, Any]], packets_by_section: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    context_by_evidence: dict[str, dict[str, str]] = {}
    for section_id, packet in packets_by_section.items():
        for row in _packet_rows(packet):
            evidence_id = str(row.get("id") or "")
            if evidence_id:
                context_by_evidence.setdefault(evidence_id, {})[section_id] = _stable_json(row)[:3200]
    return [
        {
            **item,
            "validation_context_by_section": context_by_evidence.get(str(item.get("id") or ""), {}),
        }
        for item in evidence_map
    ]


def _writer_rules(target_length: int) -> list[str]:
    return [
        "<hard_constraints>",
        "1. 只使用 evidence_packet 作为外部可核查事实的来源，不使用外部知识，不补写 Quote 没有表达的事实、因果、目的、性能、排序、适用场景或评价。",
        "2. Quote 是事实权威；atomic_claim 只用于定位。范围或强度不一致时采用 Quote 能支持的较窄表述。",
        "3. 来源明确陈述的外部事实、数字、主体属性和来源归因必须分别带自己的 [[ev_XXX]]；标记紧跟 Evidence 真正支持的最小事实子句，且 Evidence 必须支持其全部主体、关系、条件、强度和范围。可以在引用事实之后另起明确分开的解释、综合或推断，这类未引用分析不强制挂引用，但不得引入新的外部事实，也不得把引用放在分析之后让 Evidence 看似支持整个推断。",
        "4. 只能在同一 synthesis_boundary_id 内综合；跨 boundary 的因果、机制、比较或总括必须有明确关系 Evidence。",
        "5. 不得把多个局部事实自动提升为统一目的、效果保证、最佳实践或普遍推荐。",
        "6. 保留主体、比较维度、适用范围、来源角色和不确定性；来源方判断必须明确归因。",
        "7. conflict_or_limitation=true 的材料与支持材料共同保留；存在冲突时并列呈现，不强行裁决。",
        "8. Evidence 不足时缩短、明确未知或不回答，不得为目标长度补充常识、背景、案例或推断。",
        "9. 只能使用 allowed_evidence_ids；不得输出 Finding ID、原始 Quote 全文、Sources 章节或审计旁白。",
        "</hard_constraints>",
        "<writing_style>",
        "10. 用中文写连贯、克制的分析正文，产品名和代码标识保留原文。",
        f"11. 目标长度约 {target_length} 字，但证据充分性优先，宁可更短也不扩写。",
        "12. research questions 是覆盖地图，不是固定标题模板；严格遵守当前章节边界。",
        "13. 只有材料明确支持多维比较或多步流程时才使用 Markdown 表格；每个事实单元也独立引用。",
        "</writing_style>",
    ]


def _writer_few_shot_examples() -> list[dict[str, Any]]:
    return [
        {
            "name": "split_partially_supported_compound_claim",
            "evidence": {
                "id": "ev_example_1",
                "quote": "对象甲的索引阶段依赖模型调用，因此计算开销较高。",
            },
            "incorrect": "对象甲采用层级机制，其索引依赖模型调用且计算开销较高 [[ev_example_1]]。",
            "why_incorrect": "Evidence 只支持索引依赖和计算开销，不支持对象甲采用层级机制；句末引用会被理解为支持整个复合陈述。",
            "correct": "对象甲的索引阶段依赖模型调用，计算开销较高 [[ev_example_1]]。",
        },
        {
            "name": "keep_uncited_analysis_outside_cited_span",
            "evidence": {
                "id": "ev_example_2",
                "quote": "对象乙支持按子单元检索，再返回对应父单元。",
            },
            "correct": "对象乙支持先检索子单元、再返回对应父单元 [[ev_example_2]]。这提示实现时可以分别设计匹配粒度与返回粒度。",
            "explanation": "第一句是 Evidence 支持的 cited span；引用后的第二句是明确分开的分析，不得把第二句并入该引用的支持范围。",
        },
    ]


def _section_payload(
    *,
    user_request: str,
    section: dict[str, Any],
    blueprint: dict[str, Any],
    packet: dict[str, Any],
    target_length: int,
) -> dict[str, Any]:
    return {
        "task": "基于结构化 Evidence Packet 撰写一节中文分析报告。只返回 JSON。",
        "prompt_version": PROMPT_VERSION,
        "writing_architecture": ARCHITECTURE,
        "user_request": user_request,
        "section_brief": {key: value for key, value in section.items() if key != "dossier"},
        "report_outline": [
            {"section_id": item.get("section_id"), "title": item.get("title"), "purpose": item.get("purpose")}
            for item in blueprint.get("sections") or []
        ],
        "research_mode": blueprint.get("research_mode"),
        "target_length_chars": target_length,
        "allowed_evidence_ids": list(packet.get("allowed_evidence_ids") or []),
        "evidence_packet": packet,
        "rules": _writer_rules(target_length),
        "few_shot_examples": _writer_few_shot_examples(),
        "few_shot_policy": "示例只演示引用边界，禁止复制示例实体、措辞或事实；输出仍只能使用 evidence_packet。",
        "schema": {"content": "不含章节标题的实质性 Markdown 正文"},
    }


_MARKER_RE = re.compile(r"\[\[([^\]]+)\]\]|\[\s*((?:ev_\d{3,4})(?:\s*[,;、]\s*ev_\d{3,4})*)\s*\]")


def _clean_span(value: str) -> str:
    value = re.sub(r"\[\d+\]", "", str(value or ""))
    value = re.sub(r"\b(?:ev|finding)_\d{3,4}\b", "", value)
    return re.sub(r"\[\s*\]", "", value).strip().lstrip("，,；;| ").rstrip("| ")


def _render_visible_citations(
    content: str,
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
) -> str:
    """Render each E+ claim marker independently, even when sources repeat nearby.

    E+ keeps a cited-span ledger finer than a full atomic prose unit. Deduplicating numeric
    references across the whole unit can therefore remove the visible citation from a later
    claim while its ledger still records Evidence. References are deduplicated only inside one
    marker here; separate claim markers remain separately visible.
    """
    normalized = _normalize_internal_citation_markers(str(content or ""))

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1) or match.group(2) or ""
        evidence_ids = [
            item
            for item in re.findall(r"\bev_\d{3,4}\b", raw)
            if item in evidence_by_id
        ]
        refs = []
        for evidence_id in evidence_ids:
            ref = citation_map.get(str(evidence_by_id[evidence_id].get("source_url") or ""))
            if ref and ref not in refs:
                refs.append(ref)
        return " ".join(f"[{ref}]" for ref in refs)

    rendered = _MARKER_RE.sub(replace, normalized)
    rendered = re.sub(r"\b(?:ev|finding)_\d{3,4}\b", "", rendered)
    rendered = re.sub(r"\[\s*\]", "", rendered)
    # Multiple Evidence rows can support one cited span while resolving to the same source.
    # Collapse only adjacent identical numeric refs. Citations attached to distinct prose
    # claims remain visible, preserving the fine-grained claim ledger contract.
    rendered = re.sub(r"(\[(\d+)\])(?:[ \t]*\[\2\])+", r"\1", rendered)
    rendered = re.sub(r"(\[\d+\])(?=[A-Za-z0-9\u4e00-\u9fff])", r"\1 ", rendered)
    return re.sub(r"[ \t]{2,}", " ", rendered)


def _render_citations_with_spans(
    content: str,
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    rendered_content = _render_visible_citations(content, evidence_by_id, citation_map)
    normalized = _normalize_internal_citation_markers(str(content or ""))
    ledger: list[dict[str, Any]] = []
    previous_claim = ""
    for line in normalized.splitlines():
        is_table = line.lstrip().startswith("|")
        units = line.split("|") if is_table else split_atomic_units(line) if line.strip() and not line.lstrip().startswith(("#", "- ", "* ")) else [line]
        for unit in units:
            full_rendered, _ = _render_internal_citations(unit, evidence_by_id, citation_map)
            full_atomic = _clean_span(full_rendered)
            cursor = 0
            last: dict[str, Any] | None = None
            for match in _MARKER_RE.finditer(unit):
                raw = match.group(1) or match.group(2) or ""
                ids = [item for item in re.findall(r"\bev_\d{3,4}\b", raw) if item in evidence_by_id]
                if not ids:
                    continue
                cited_span = _clean_span(unit[cursor : match.start()])
                if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", cited_span):
                    raw_with_markers = unit[cursor : match.end()]
                    rendered_span = _render_visible_citations(raw_with_markers, evidence_by_id, citation_map)
                    last = {
                        "text": cited_span,
                        "cited_span": cited_span,
                        "uncited_analysis_span": "",
                        "full_atomic_unit": full_atomic,
                        "claim_context": previous_claim[:1000] if not is_table else "",
                        "evidence_ids": list(dict.fromkeys(ids)),
                        "rendered_text": rendered_span.strip(),
                        "_raw": raw_with_markers,
                    }
                    ledger.append(last)
                elif last is not None:
                    last["evidence_ids"] = list(dict.fromkeys([*last["evidence_ids"], *ids]))
                    last["_raw"] += unit[cursor : match.end()]
                    rendered_span = _render_visible_citations(last["_raw"], evidence_by_id, citation_map)
                    last["rendered_text"] = rendered_span.strip()
                cursor = match.end()
            trailing = _clean_span(unit[cursor:])
            if last is not None and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", trailing):
                last["uncited_analysis_span"] = trailing
            if not is_table and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", full_atomic):
                previous_claim = full_atomic
    for row in ledger:
        row.pop("_raw", None)
    return rendered_content, ledger


def _gap_document(section: dict[str, Any]) -> dict[str, Any]:
    content = _render_evidence_gap(section)
    return {
        "title": section.get("title"),
        "section_id": section.get("section_id"),
        "content": content,
        "evidence_refs": [],
        "evidence_table": [],
        "claim_ledger": [],
        "section_brief": section,
        "paragraph_validation": {"status": "passed", "failed_group_count": 0, "groups": []},
        "evidence_metrics": _evidence_metrics([], []),
        "synthesis_diagnostics": {
            "mode": "evidence_gap",
            "gap_reason": "no_reportable_findings_or_evidence",
            "model_attempts": 0,
            "finding_count": 0,
            "evidence_count": 0,
            "input_support_unit_count": 0,
            "used_support_unit_count": 0,
            "reportability": section.get("reportability") or {},
        },
    }


def _write_section(
    *,
    user_request: str,
    section: dict[str, Any],
    blueprint: dict[str, Any],
    evidence_map: list[dict[str, Any]],
    packet: dict[str, Any],
    context_diagnostics: dict[str, Any],
    model: JsonModel,
) -> dict[str, Any]:
    reportability = section.get("reportability") if isinstance(section.get("reportability"), dict) else {}
    if str(reportability.get("status") or "") == "empty":
        return _gap_document(section)
    selected_ids = [str(item) for item in packet.get("allowed_evidence_ids") or []]
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence_map
        if str(item.get("id") or "") in set(selected_ids)
    }
    if not evidence_by_id:
        raise EvidenceNativeContractError(
            f"section {section.get('section_id')} is reportable but selected Evidence is empty"
        )
    target_length = _target_length_chars(section, [{} for _ in selected_ids])
    payload = _section_payload(
        user_request=user_request,
        section=section,
        blueprint=blueprint,
        packet=packet,
        target_length=target_length,
    )
    with audit_scope(stage="section_writer", section_id=str(section.get("section_id") or "") or None):
        content, meta = _call_writer_model(model, payload, evidence_by_id, allowed_finding_ids=set())
        max_retries = max(0, int(os.getenv("RESEARCH_WRITER_MAX_RETRIES", "3")))
        marker_repair_attempted = False
        retries = 0
        active_payload = payload
        while not content and retries < max_retries:
            if meta.get("failure_reason") == "evidence_validation_failed" and not marker_repair_attempted:
                marker_repair_attempted = True
                active_payload = {
                    **payload,
                    "task": "修复引用标记无效的中文章节，只返回 JSON。",
                    "invalid_draft": meta.get("invalid_content") or "",
                    "invalid_markers": meta.get("invalid_markers") or [],
                    "rules": [
                        *payload["rules"],
                        "上一稿用了 allowlist 外的 ID。只能逐字复制 allowed_evidence_ids；找不到真实支持时删除对应子句。",
                    ],
                }
            content, retry_meta = _call_writer_model(
                model, active_payload, evidence_by_id, allowed_finding_ids=set()
            )
            meta["attempts"] = int(meta.get("attempts") or 1) + 1
            meta["failure_reason"] = None if content else retry_meta.get("failure_reason")
            meta["invalid_content"] = retry_meta.get("invalid_content")
            meta["invalid_markers"] = retry_meta.get("invalid_markers") or []
            retries += 1
        if content and _section_is_underdeveloped(content, section, [{} for _ in selected_ids]):
            expanded, _ = _call_writer_model(
                model,
                {
                    **payload,
                    "task": "仅基于同一 Evidence Packet 扩展过短章节，只返回 JSON。",
                    "existing_draft": content,
                    "rules": [*payload["rules"], "只增加 Evidence 已支持的差异、限制和冲突，不补充外部事实。"],
                },
                evidence_by_id,
                allowed_finding_ids=set(),
            )
            meta["attempts"] = int(meta.get("attempts") or 1) + 1
            if expanded and len(expanded) > len(content):
                content = expanded
    citation_map = _citation_map(evidence_map)
    rendered, ledger = _render_citations_with_spans(content, evidence_by_id, citation_map)
    ledger = _annotate_claim_ledger(
        ledger, [], evidence_by_id, str(section.get("section_id") or "")
    )
    if content and not ledger:
        rendered = ""
        meta["failure_reason"] = "missing_required_citations"
    used_ids = list(dict.fromkeys(item for row in ledger for item in row.get("evidence_ids") or []))
    evidence_table = [_evidence_table_row(evidence_by_id[item]) for item in used_ids if item in evidence_by_id]
    return {
        "title": section.get("title"),
        "section_id": section.get("section_id"),
        "content": rendered.strip(),
        "rendered_in_report": not _excluded_from_assembled_body(str(section.get("title") or "")),
        "evidence_refs": _evidence_refs([evidence_by_id[item] for item in used_ids if item in evidence_by_id]),
        "evidence_table": evidence_table,
        "claim_ledger": ledger,
        "section_brief": section,
        "paragraph_validation": {"status": "passed", "failed_group_count": 0, "groups": []},
        "evidence_metrics": _evidence_metrics(evidence_table, ledger),
        "synthesis_diagnostics": {
            "mode": "evidence_native_writer" if rendered else "model_failed",
            "fallback_reason": meta.get("failure_reason"),
            "model_attempts": int(meta.get("attempts") or 0),
            "input_finding_count": 0,
            "used_finding_count": 0,
            "input_support_unit_count": len(selected_ids),
            "used_support_unit_count": len(used_ids),
            "dossier_evidence_count": len(selected_ids),
            "used_evidence_count": len(used_ids),
            "evidence_utilization_ratio": round(len(used_ids) / len(selected_ids), 3) if selected_ids else 1.0,
            "dossier_chunk_count": 0,
            "dossier_context_chars": len(_stable_json(packet)),
            "selected_evidence_ids": selected_ids,
            "cropped_evidence_ids": list(context_diagnostics.get("cropped_evidence_ids") or []),
            "context_packet_hash": packet.get("packet_hash"),
            "reportability": reportability,
        },
    }


def _visible_ledger(documents: list[dict[str, Any]], editorial: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [
        *[
            row
            for document in documents
            if not _excluded_from_assembled_body(str(document.get("title") or ""))
            for row in document.get("claim_ledger") or []
        ],
        *((editorial or {}).get("claim_ledger") or []),
    ]


def _combine_section_entailments(
    documents: list[dict[str, Any]], results_by_index: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    judgments: list[dict[str, Any]] = []
    offset = 0
    for index, document in enumerate(documents):
        result = results_by_index.get(index) or {}
        for row in result.get("judgments") or []:
            if not isinstance(row, dict):
                continue
            adjusted = copy.deepcopy(row)
            adjusted["claim_index"] = int(row.get("claim_index") or 0) + offset
            judgments.append(adjusted)
        offset += len(document.get("claim_ledger") or [])
    judgments.sort(key=lambda item: int(item.get("claim_index") or 0))
    blocking = [
        item
        for item in judgments
        if item.get("status") in {"unsupported", "label_mismatch", "non_atomic"}
    ]
    unavailable = [item for item in judgments if item.get("status") == "validation_unavailable"]
    return {
        "status": "failed" if blocking or unavailable else "passed",
        "claim_count": len(judgments),
        "entailed_count": sum(item.get("status") == "entailed" for item in judgments),
        "inference_count": sum(item.get("status") == "reasonable_inference" for item in judgments),
        "unsupported_count": sum(item.get("status") == "unsupported" for item in judgments),
        "label_mismatch_count": sum(item.get("status") == "label_mismatch" for item in judgments),
        "non_atomic_count": sum(item.get("status") == "non_atomic" for item in judgments),
        "unsupported_inference_count": len(blocking),
        "model_reviewed_count": sum(item.get("mode") == "model" for item in judgments),
        "semantic_review_unavailable_count": len(unavailable),
        "deterministic_fallback_count": 0,
        "reused_judgment_count": sum(bool(item.get("reused")) for item in judgments),
        "judgments": judgments,
    }


def _prevalidate_sections(
    *,
    documents: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    model: Callable[[dict[str, Any]], dict[str, Any]] | None,
    parallelism: int,
    checkpoint_store: EvidenceNativeCheckpointStore | None,
    entailment_payload_profile: str,
) -> dict[str, Any]:
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map}
    results_by_index: dict[int, dict[str, Any]] = {}
    pending: dict[Any, tuple[int, dict[str, Any]]] = {}

    def inputs_for(document: dict[str, Any]) -> dict[str, Any]:
        evidence_ids = {
            str(evidence_id)
            for row in document.get("claim_ledger") or []
            for evidence_id in row.get("evidence_ids") or []
        }
        return {
            "claim_ledger": document.get("claim_ledger") or [],
            "evidence": [evidence_by_id[item] for item in sorted(evidence_ids) if item in evidence_by_id],
            "entailment_payload_profile": entailment_payload_profile,
            "entailment_prompt_version": (
                ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION
                if entailment_payload_profile == ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2
                else None
            ),
        }

    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(documents) or 1))) as executor:
        for index, document in enumerate(documents):
            section_id = str(document.get("section_id") or f"section_{index:03d}")
            checkpoint_input = inputs_for(document)
            cached = (
                checkpoint_store.load("section_prevalidation", section_id, checkpoint_input)
                if checkpoint_store
                else None
            )
            if isinstance((cached or {}).get("entailment"), dict):
                results_by_index[index] = copy.deepcopy(cached["entailment"])
                continue
            future = submit_with_audit_context(
                executor,
                validate_claim_entailment,
                report_document={"claim_ledger": document.get("claim_ledger") or []},
                evidence_map=evidence_map,
                model=model,
                parallelism=max(1, int(os.getenv("RESEARCH_ENTAILMENT_PARALLELISM", "6"))),
                payload_profile=entailment_payload_profile,
            )
            pending[future] = (index, checkpoint_input)
        for future in as_completed(pending):
            index, checkpoint_input = pending[future]
            result = future.result()
            results_by_index[index] = result
            if checkpoint_store and int(result.get("semantic_review_unavailable_count") or 0) == 0:
                section_id = str(documents[index].get("section_id") or f"section_{index:03d}")
                checkpoint_store.save(
                    "section_prevalidation", section_id, checkpoint_input, {"entailment": result}
                )
    return _combine_section_entailments(documents, results_by_index)


def build_editorial_claim_bank(
    documents: list[dict[str, Any]], section_entailment: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judgments = {
        int(item.get("claim_index")): item
        for item in section_entailment.get("judgments") or []
        if isinstance(item, dict) and str(item.get("claim_index", "")).lstrip("-").isdigit()
    }
    bank: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    visible_index = 0
    for document in documents:
        if _excluded_from_assembled_body(str(document.get("title") or "")):
            continue
        section_id = str(document.get("section_id") or "section")
        source_section_hash = stable_hash(
            {
                "section_id": section_id,
                "content": document.get("content"),
                "claim_ledger": document.get("claim_ledger") or [],
            }
        )
        for row in document.get("claim_ledger") or []:
            judgment = judgments.get(visible_index) or {}
            visible_index += 1
            evidence_ids = [str(item) for item in row.get("evidence_ids") or [] if str(item)]
            cited_span = str(row.get("cited_span") or row.get("text") or "").strip()
            status = str(judgment.get("status") or "validation_unavailable")
            inference_qualified = status != "reasonable_inference" or any(
                marker in cited_span.casefold()
                for marker in (
                    "推断", "可能", "或许", "可视为", "可作为", "表明", "提示", "暗示",
                    "infer", "may", "might", "could", "likely", "suggest", "indicate",
                )
            )
            admitted = bool(
                evidence_ids
                and cited_span
                and row.get("atomic") is not False
                and status in {"entailed", "reasonable_inference"}
                and inference_qualified
            )
            if not admitted:
                excluded.append(
                    {
                        "section_id": section_id,
                        "cited_span": cited_span,
                        "evidence_ids": evidence_ids,
                        "validation_status": status,
                        "validation_reason": judgment.get("reason"),
                        "admission_reason": (
                            "reasonable_inference_missing_explicit_qualification"
                            if status == "reasonable_inference" and not inference_qualified
                            else "prevalidation_not_admitted"
                        ),
                    }
                )
                continue
            signature = stable_hash(
                {
                    "section_id": section_id,
                    "cited_span": cited_span,
                    "claim_context": row.get("claim_context"),
                    "evidence_ids": evidence_ids,
                    "claim_type": row.get("claim_type"),
                }
            )
            bank.append(
                {
                    "claim_id": f"claim_{signature[:16]}",
                    "section_id": section_id,
                    "section_title": document.get("title"),
                    "cited_span": cited_span,
                    "claim_context": row.get("claim_context"),
                    "evidence_ids": evidence_ids,
                    "direct_evidence_ids": [str(item) for item in row.get("direct_evidence_ids") or []],
                    "inference_evidence_ids": [str(item) for item in row.get("inference_evidence_ids") or []],
                    "claim_type": row.get("claim_type"),
                    "support_scope": row.get("support_scope"),
                    "prevalidation_status": status,
                    "prevalidation_reason": judgment.get("reason"),
                    "validation_status": status,
                    "validation_reason": judgment.get("reason"),
                    "signature_hash": signature,
                    "source_section_hash": source_section_hash,
                    "scope_contract": "摘要复用不得扩大对象、领域、条件、时间、强度或 support_scope。",
                }
            )
    return bank, excluded


def _expand_claim_markers(content: str, claim_by_id: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    normalized = _normalize_internal_citation_markers(str(content or ""))
    invalid: list[str] = re.findall(r"\b(?:ev|finding)_\d{3,4}\b", normalized)

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        tokens = [item for item in re.split(r"\s*[,，;；、]\s*", raw) if item]
        unknown = [item for item in tokens or [raw] if item not in claim_by_id]
        if unknown:
            invalid.extend(unknown)
            return ""
        ids = list(
            dict.fromkeys(
                evidence_id
                for token in tokens
                for evidence_id in claim_by_id[token].get("evidence_ids") or []
            )
        )
        return "".join(f"[[{item}]]" for item in ids)

    expanded = re.sub(r"\[\[([^\]]+)\]\]", replace, normalized)
    # Final deterministic invariant: no internal Claim Bank id may survive into visible prose,
    # even if the model emitted a malformed wrapper outside the recognized marker grammar.
    invalid.extend(re.findall(r"\bclaim_[A-Za-z0-9_-]+\b", expanded))
    return expanded, list(dict.fromkeys(invalid))


def _write_editorial(
    *,
    user_request: str,
    blueprint: dict[str, Any],
    claim_bank: list[dict[str, Any]],
    excluded_claims: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    model: JsonModel,
) -> dict[str, Any]:
    if not claim_bank:
        disclosure = "当前没有通过引用支持预检查、可用于摘要或综合结论的章节声明。"
        return {
            "report_title": "阶段性研究结果",
            "executive_summary": disclosure,
            "conclusion": disclosure,
            "claim_ledger": [],
            "diagnostics": {
                "mode": "empty_claim_bank",
                "input_claim_count": 0,
                "excluded_section_claims": excluded_claims,
                "model_called": False,
            },
        }
    claim_by_id = {str(item.get("claim_id") or ""): item for item in claim_bank}
    evidence_ids = {
        str(evidence_id)
        for claim in claim_bank
        for evidence_id in claim.get("evidence_ids") or []
    }
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence_map
        if str(item.get("id") or "") in evidence_ids
    }
    payload = {
        "task": "只基于 section_claim_bank 写中文执行摘要、综合结论和短标题。只返回 JSON。",
        "prompt_version": PROMPT_VERSION,
        "writing_architecture": ARCHITECTURE,
        "user_request": user_request,
        "research_mode": blueprint.get("research_mode"),
        "section_claim_bank": claim_bank,
        "rules": [
            "1. 只使用 section_claim_bank，不读取原始 Evidence，不假设 Finding。",
            "2. 每个事实性子句紧跟一个或多个合法 [[claim_xxx]]；不得直接输出 Evidence 或 Finding ID。",
            "3. 只能压缩或重述 cited_span，不得新增主体、关系、因果、效果、比较、推荐或评价。",
            "4. 保留对象、领域、条件、时间、强度和 support_scope；材料不足时缩短或明确未知。",
            "5. 中文连贯分析，不添加 Sources 章节、Quote 全文或审计说明。",
        ],
        "schema": {"report_title": "", "executive_summary": "", "conclusion": ""},
    }
    with audit_scope(stage="editorial", section_id="editorial"):
        result = model(payload)
        result = result if isinstance(result, dict) else {"_failure_reason": "invalid_response_type"}
        missing = [field for field in ("executive_summary", "conclusion") if not str(result.get(field) or "").strip()]
        if missing:
            completion = model(
                {
                    **payload,
                    "task": "只补齐缺失的 Editorial 字段，只返回 JSON。",
                    "existing_editorial": {field: result.get(field) or "" for field in ("executive_summary", "conclusion")},
                    "missing_fields": missing,
                    "schema": {field: "" for field in missing},
                }
            )
            if isinstance(completion, dict):
                for field in missing:
                    if completion.get(field):
                        result[field] = completion[field]

    citation_map = _citation_map(evidence_map)
    output: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    invalid_by_field: dict[str, list[str]] = {}
    for field in ("executive_summary", "conclusion"):
        raw = str(result.get(field) or "")
        expanded, invalid = _expand_claim_markers(raw, claim_by_id)
        if invalid:
            invalid_by_field[field] = invalid
            repair_payload = {
                **payload,
                "task": f"只修复 {field} 中的非法 claim marker，只返回该字段 JSON。",
                "invalid_text": raw,
                "invalid_markers": invalid,
                "allowed_claim_ids": sorted(claim_by_id),
                "schema": {field: ""},
            }
            with audit_scope(stage="editorial", section_id="editorial"):
                repair = model(repair_payload)
            repaired_raw = str((repair or {}).get(field) or "") if isinstance(repair, dict) else ""
            expanded, invalid = _expand_claim_markers(repaired_raw, claim_by_id)
        if invalid:
            expanded = ""
        rendered, rows = _render_citations_with_spans(expanded, evidence_by_id, citation_map)
        output[field] = rendered
        ledger.extend(rows)
    failed_fields = [field for field in ("executive_summary", "conclusion") if not str(output.get(field) or "").strip()]
    output["report_title"] = str(result.get("report_title") or "").strip()[:80]
    output["claim_ledger"] = _annotate_claim_ledger(ledger, [], evidence_by_id, "editorial")
    output["diagnostics"] = {
        "mode": "model_failed" if failed_fields else "evidence_native_editorial",
        "failed_fields": failed_fields,
        "input_claim_count": len(claim_bank),
        "excluded_section_claim_count": len(excluded_claims),
        "excluded_section_claims": excluded_claims,
        "invalid_claim_markers": invalid_by_field,
        "claim_bank_sha256": stable_hash(claim_bank),
        "upstream_failure_reason": result.get("_failure_reason"),
    }
    return output


def write_evidence_native_report(
    *,
    user_request: str,
    blueprint: dict[str, Any],
    question_states: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]] | None = None,
    writer_model: JsonModel,
    prevalidation_model: Callable[[dict[str, Any]], dict[str, Any]] | None,
    parallelism: int = 3,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_store: EvidenceNativeCheckpointStore | None = None,
    entailment_payload_profile: str = ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
) -> dict[str, Any]:
    started = time.perf_counter()
    if selected_chunks is not None:
        validate_evidence_traceability(evidence_map, selected_chunks)
    manager = EvidenceContextManager(
        # No experiment-era 48k cap in production. Configure the real provider/model context
        # explicitly when one exists; otherwise this high guard only catches pathological input.
        max_input_tokens=int(os.getenv("RESEARCH_EPLUS_MAX_INPUT_TOKENS", "1000000")),
        prompt_reserve_tokens=int(os.getenv("RESEARCH_EPLUS_PROMPT_RESERVE_TOKENS", "6500")),
    )
    all_sections = [copy.deepcopy(item) for item in blueprint.get("sections") or [] if isinstance(item, dict)]
    sections = [
        item
        for item in all_sections
        if not _is_reference_title(str(item.get("title") or ""))
        and not (str(item.get("section_id") or "") == "sec_00" or "总览" in str(item.get("title") or ""))
    ]
    packets: dict[str, dict[str, Any]] = {}
    context_diagnostics_by_section: dict[str, dict[str, Any]] = {}
    context_diagnostics: list[dict[str, Any]] = []
    section_by_id: dict[str, dict[str, Any]] = {}
    for section in sections:
        section_id = str(section.get("section_id") or "")
        if str((section.get("reportability") or {}).get("status") or "") == "empty":
            packet = {"allowed_evidence_ids": [], "claim_clusters": [], "packet_hash": stable_hash({"section_id": section_id, "empty": True})}
            diagnostics = {
                "context_manager_version": CONTEXT_VERSION,
                "candidate_evidence_count": 0,
                "selected_evidence_count": 0,
                "selected_evidence_ids": [],
                "cropped_evidence_ids": [],
                "packet_hash": packet["packet_hash"],
            }
        else:
            packet, diagnostics = manager.build(
                section=section, question_states=question_states, evidence_map=evidence_map
            )
        packets[section_id] = packet
        context_diagnostics_by_section[section_id] = diagnostics
        section["dossier"] = {
            "evidence": _packet_rows(packet),
            "source_notes": [],
            "relevant_chunks": [],
            "unused_evidence_ids": list(packet.get("allowed_evidence_ids") or []),
            "context_chars": len(_stable_json(packet)),
            "context_packet_hash": packet.get("packet_hash"),
        }
        section_by_id[section_id] = section
        context_diagnostics.append({"section_id": section_id, **diagnostics})

    documents_by_index: dict[int, dict[str, Any]] = {}
    section_checkpoint_inputs: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(sections) or 1))) as executor:
        futures: dict[Any, int] = {}
        for index, section in enumerate(sections):
            section_id = str(section.get("section_id") or "")
            checkpoint_input = {
                "section": section,
                "report_outline": [
                    {"section_id": item.get("section_id"), "title": item.get("title"), "purpose": item.get("purpose")}
                    for item in blueprint.get("sections") or []
                ],
                "packet": packets[section_id],
                "prompt_version": PROMPT_VERSION,
            }
            section_checkpoint_inputs[index] = checkpoint_input
            cached = (
                checkpoint_store.load("section_draft", section_id, checkpoint_input)
                if checkpoint_store
                else None
            )
            if isinstance((cached or {}).get("document"), dict):
                documents_by_index[index] = copy.deepcopy(cached["document"])
                if progress_callback:
                    progress_callback(
                        {
                            "completed": len(documents_by_index),
                            "total": len(sections),
                            "title": documents_by_index[index].get("title"),
                            "resumed": True,
                        }
                    )
                continue
            future = submit_with_audit_context(
                executor,
                _write_section,
                user_request=user_request,
                section=section,
                blueprint=blueprint,
                evidence_map=evidence_map,
                packet=packets[section_id],
                context_diagnostics=context_diagnostics_by_section[section_id],
                model=writer_model,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            document = future.result()
            documents_by_index[index] = document
            mode = str((document.get("synthesis_diagnostics") or {}).get("mode") or "")
            if checkpoint_store and mode not in {"model_failed", "fallback_template"}:
                checkpoint_store.save(
                    "section_draft",
                    str(document.get("section_id") or f"section_{index:03d}"),
                    section_checkpoint_inputs[index],
                    {"document": document},
                )
            if progress_callback:
                try:
                    progress_callback(
                        {
                            "completed": len(documents_by_index),
                            "total": len(sections),
                            "title": document.get("title"),
                        }
                    )
                except Exception:
                    pass
    documents = [documents_by_index[index] for index in sorted(documents_by_index)]
    validation_evidence = evidence_with_writer_context(evidence_map, packets)
    section_entailment = _prevalidate_sections(
        documents=documents,
        evidence_map=validation_evidence,
        model=prevalidation_model,
        parallelism=parallelism,
        checkpoint_store=checkpoint_store,
        entailment_payload_profile=entailment_payload_profile,
    )
    claim_bank, excluded_claims = build_editorial_claim_bank(documents, section_entailment)
    claim_bank_input = {
        "documents": [
            {"section_id": item.get("section_id"), "content": item.get("content"), "claim_ledger": item.get("claim_ledger") or []}
            for item in documents
        ],
        "section_prevalidation": section_entailment,
    }
    if checkpoint_store:
        checkpoint_store.save(
            "editorial_claim_bank",
            "global",
            claim_bank_input,
            {"claim_bank": claim_bank, "excluded_claims": excluded_claims},
        )
    editorial_input = {
        "user_request": user_request,
        "blueprint_structure": {
            "research_mode": blueprint.get("research_mode"),
            "completeness": blueprint.get("completeness"),
            "sections": [
                {
                    "section_id": item.get("section_id"),
                    "title": item.get("title"),
                    "reportability": item.get("reportability"),
                }
                for item in blueprint.get("sections") or []
            ],
        },
        "claim_bank": claim_bank,
        "excluded_claims": excluded_claims,
        "prompt_version": PROMPT_VERSION,
    }
    cached_editorial = (
        checkpoint_store.load("editorial", "global", editorial_input)
        if checkpoint_store
        else None
    )
    if isinstance((cached_editorial or {}).get("editorial"), dict):
        editorial = copy.deepcopy(cached_editorial["editorial"])
    else:
        editorial = _write_editorial(
            user_request=user_request,
            blueprint=blueprint,
            claim_bank=claim_bank,
            excluded_claims=excluded_claims,
            evidence_map=evidence_map,
            model=writer_model,
        )
        if checkpoint_store and str((editorial.get("diagnostics") or {}).get("mode") or "") != "model_failed":
            checkpoint_store.save("editorial", "global", editorial_input, {"editorial": editorial})
    completeness = _delivery_completeness(
        blueprint.get("completeness") or {"status": "partial_report"}, documents
    )
    research_mode = str(blueprint.get("research_mode") or "domain_practice_review")
    report_content, sanitized_count = _assemble_report(
        user_request,
        documents,
        editorial,
        _citation_map(evidence_map),
        evidence_map,
        completeness=completeness,
        research_mode=research_mode,
        reference_title=_reference_title_from_blueprint(blueprint),
    )
    report_status = str(completeness.get("status") or "partial_report")
    updated_sections = [section_by_id.get(str(item.get("section_id") or ""), item) for item in all_sections]
    updated_blueprint = {**blueprint, "sections": updated_sections}
    return {
        "documents": documents,
        "report_status": report_status,
        "report_blueprint": updated_blueprint,
        "report_document": {
            "title": editorial.get("report_title") or f"{user_request} {_report_label(report_status == 'complete_report', research_mode)}",
            "content": report_content,
            "sanitized_evidence_id_count": sanitized_count,
            "delivery_status": report_status,
            "completeness": completeness,
            "research_mode": research_mode,
            "evidence_refs": _evidence_refs(evidence_map),
            "claim_ledger": _visible_ledger(documents, editorial),
        },
        "editorial": editorial,
        "section_prevalidation": section_entailment,
        "editorial_claim_bank": claim_bank,
        "evidence_context_packets": packets,
        "validation_evidence_map": validation_evidence,
        "diagnostics": {
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "section_count": len(documents),
            "parallelism": parallelism,
            "model_mode": "model",
            "writing_architecture": ARCHITECTURE,
            "writing_architecture_version": ARCHITECTURE_VERSION,
            "context_manager_version": CONTEXT_VERSION,
            "context_sections": context_diagnostics,
            "prevalidation": {
                key: section_entailment.get(key)
                for key in ("status", "claim_count", "entailed_count", "inference_count", "unsupported_count", "label_mismatch_count")
            },
            "entailment_payload_profile": entailment_payload_profile,
            "claim_bank_count": len(claim_bank),
            "excluded_claim_count": len(excluded_claims),
            "material_utilization": _material_utilization(documents),
            "checkpoint": {
                "enabled": checkpoint_store is not None,
                "contract_sha256": checkpoint_store.contract_hash if checkpoint_store else None,
                "events": copy.deepcopy(checkpoint_store.events) if checkpoint_store else [],
            },
        },
    }


def architecture_capabilities() -> dict[str, bool]:
    # Revision/follow-up stay explicitly closed until their architecture-aware paths land.
    return {
        "section_revision": False,
        "targeted_followup": False,
        "approval": True,
        "archive": True,
    }
