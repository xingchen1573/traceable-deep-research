#!/usr/bin/env python3
"""Replay question coverage against saved research checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from backend.research.coverage import assess_question_coverage
from backend.research.models import (
    LEGACY_CRITERION_LEVEL_POLICY,
    QuestionBudget,
    ResearchCriterionState,
    ResearchQuestionState,
    _criterion_states,
)
from backend.research.source_policy import (
    build_source_requirements,
    coverage_requirements,
    normalize_question_kind,
)


def _state_from_dict(raw: dict[str, Any]) -> ResearchQuestionState:
    criteria = [ResearchCriterionState(**item) for item in raw.get("criteria") or []]
    budget = QuestionBudget(**raw["budget"])
    state = ResearchQuestionState(
        question_id=str(raw["question_id"]),
        section_id=str(raw["section_id"]),
        section_title=str(raw["section_title"]),
        question=str(raw["question"]),
        purpose=str(raw.get("purpose") or ""),
        success_criteria=[str(item) for item in raw.get("success_criteria") or []],
        preferred_source_types=[str(item) for item in raw.get("preferred_source_types") or []],
        budget=budget,
        criteria=criteria,
        criterion_level_policy=str(
            raw.get("criterion_level_policy") or LEGACY_CRITERION_LEVEL_POLICY
        ),
    )
    excluded = {
        "question_id", "section_id", "section_title", "question", "purpose",
        "success_criteria", "preferred_source_types", "budget", "criteria",
    }
    state_fields = {item.name for item in fields(ResearchQuestionState)}
    for key, value in raw.items():
        if key in state_fields and key not in excluded:
            setattr(state, key, value)
    return state


def _apply_current_policy(state: ResearchQuestionState) -> None:
    old_requirements = dict(state.coverage_requirements)
    old_min_evidence = int(old_requirements.get("min_claim_clusters") or 0)
    old_min_sources = int(old_requirements.get("min_domains") or 0)
    preserve_evidence_override = state.min_evidence > old_min_evidence
    preserve_source_override = state.min_sources > old_min_sources
    state.question_kind = normalize_question_kind(
        None,
        question=state.question,
        purpose=state.purpose,
        success_criteria=state.success_criteria,
    )
    state.coverage_requirements = coverage_requirements(state.question_kind, question=state.question)
    if not preserve_evidence_override:
        state.min_evidence = int(state.coverage_requirements["min_claim_clusters"])
    if not preserve_source_override:
        state.min_sources = int(state.coverage_requirements["min_domains"])
    state.source_requirements = build_source_requirements(
        question=state.question,
        purpose=state.purpose,
        success_criteria=state.success_criteria,
        preferred_source_types=state.preferred_source_types,
        question_kind=state.question_kind,
    )
    state.criteria = _criterion_states(
        state.success_criteria,
        state.preferred_source_types,
        question=state.question,
        level_policy=state.criterion_level_policy,
    )


def replay(path: Path, *, rejudge: bool, report_quality: bool) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    states = [_state_from_dict(item) for item in checkpoint["payload"]["question_states"]]
    if rejudge:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
        from backend.services.research_service import _judge_question_coverage

    rows = []
    rejudge_success_count = 0
    for state in states:
        old_kind = state.question_kind
        old_boundary = bool(state.coverage_requirements.get("requires_boundary"))
        _apply_current_policy(state)
        if rejudge:
            semantic = _judge_question_coverage(state)
            if semantic:
                state.diagnostics["semantic_coverage"] = semantic
                rejudge_success_count += 1
        coverage = assess_question_coverage(state)
        rows.append({
            "question_id": state.question_id,
            "old_kind": old_kind,
            "new_kind": state.question_kind,
            "old_boundary": old_boundary,
            "new_boundary": bool(state.coverage_requirements.get("requires_boundary")),
            "boundary_evidence_count": coverage["boundary_evidence_count"],
            "boundary_check_passed": coverage["checks"]["boundary_evidence"],
            "semantic_status": coverage["semantic_coverage_status"],
            "semantic_counts": coverage["semantic_coverage_counts"],
            "semantic_raw_counts": coverage["semantic_coverage_raw_counts"],
            "criterion_level_policy": coverage["criterion_level_policy"],
            "criterion_level_migrations": coverage["criterion_level_migrations"],
            "sufficient": coverage["sufficient"],
            "answer_status": coverage["answer_status"],
            "answered_points": coverage["answered_points"],
            "missing_question_points": coverage["missing_question_points"],
            "enrichment_gap_count": len(coverage["enrichment_gaps"]),
            "failed_checks": [
                name for name, passed in coverage["checks"].items()
                if name != "source_mix" and not passed
            ],
            "advisory_source_mix_satisfied": coverage["checks"]["source_mix"],
            "blocking_gap_types": sorted({
                str(item.get("gap_type") or "")
                for item in coverage["gaps"]
                if item.get("blocking")
            }),
        })
    result = {
        "job_id": checkpoint.get("job_id") or path.parent.parent.name,
        "question_count": len(rows),
        "sufficient_count": sum(row["sufficient"] for row in rows),
        # Keep partial_count for older automation, but make it mean the new reader-facing status.
        "partial_count": sum(row["answer_status"] == "partial" for row in rows),
        "answer_status_counts": {
            status: sum(row["answer_status"] == status for row in rows)
            for status in ("complete", "partial", "uncovered")
        },
        "kind_changed_count": sum(row["old_kind"] != row["new_kind"] for row in rows),
        "boundary_removed_count": sum(row["old_boundary"] and not row["new_boundary"] for row in rows),
        "boundary_required_count": sum(row["new_boundary"] for row in rows),
        "boundary_satisfied_count": sum(row["new_boundary"] and row["boundary_check_passed"] for row in rows),
        "rejudge_success_count": rejudge_success_count if rejudge else None,
        "rejudge_failure_count": len(rows) - rejudge_success_count if rejudge else None,
        "rows": rows,
    }
    if report_quality:
        from backend.research.quality import evaluate_report_quality

        draft_path = path.with_name("draft.json")
        draft = json.loads(draft_path.read_text(encoding="utf-8"))["payload"]
        validation_path = path.with_name("validation.json")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))["payload"]
        saved_quality = validation.get("report_quality") or {}
        question_rows = []
        for state, row in zip(states, rows):
            state.status = "sufficient" if row["sufficient"] else "exhausted"
            question_rows.append(state.to_dict())
        quality = evaluate_report_quality(
            report_document=draft["report_document"],
            question_states=question_rows,
            findings=draft.get("findings") or [],
            rejected_evidence=checkpoint["payload"].get("rejected_evidence") or [],
            documents=draft.get("documents") or [],
            model=lambda _payload: {
                "scores": saved_quality.get("scores") or {},
                "issues": saved_quality.get("issues") or [],
                "summary": saved_quality.get("critic_summary") or "",
            },
        )
        result["report_quality"] = {
            "status": quality["status"],
            "average_score": quality["average_score"],
            "blocking_issues": quality["blocking_issues"],
            "sufficient_question_count": quality["sufficient_question_count"],
            "question_count": quality["question_count"],
            "transparent_degradation_accepted": quality["transparent_degradation_accepted"],
            "limitation_disclosed": quality["limitation_disclosed"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--rejudge", action="store_true", help="Call the current semantic judge on saved evidence")
    parser.add_argument("--report-quality", action="store_true", help="Also replay report quality from the sibling draft checkpoint")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    results = [
        replay(path, rejudge=args.rejudge, report_quality=args.report_quality)
        for path in args.checkpoints
    ]
    if args.summary_only:
        for result in results:
            result.pop("rows", None)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
