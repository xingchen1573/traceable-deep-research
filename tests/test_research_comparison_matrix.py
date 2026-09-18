from __future__ import annotations

from dataclasses import asdict

import pytest

from backend.research.evidence import admit_evidence
from backend.research.models import ComparisonBudget, question_states_from_plan
from backend.research.orchestrator import QuestionResearchOrchestrator, _update_comparison_cells
from backend.research.source_pool import GlobalSourcePool
from backend.services import research_service


def _comparison_plan(*, subjects: list[str], axes: list[str], depth: str = "normal") -> dict:
    budget = asdict(ComparisonBudget.for_depth(depth))
    sections = []
    cells = []
    for axis_index, axis in enumerate(axes, start=1):
        section_id = f"sec_{axis_index:02d}"
        sections.append({
            "id": section_id,
            "title": axis,
            "questions": [{
                "id": f"{section_id}_q1",
                "text": f"各对象在{axis}上的差异是什么？",
                "comparison_axis": axis,
                "comparison_subjects": subjects,
                "success_criteria": [f"覆盖{axis}的关键事实"],
            }],
        })
        for subject_index, subject in enumerate(subjects, start=1):
            cells.append({
                "id": f"cell_{axis_index:02d}_{subject_index:02d}",
                "subject": subject,
                "axis": axis,
            })
    return {
        "research_type": "comparison",
        "subjects": subjects,
        "comparison_axes": axes,
        "matrix_cells": cells,
        "comparison_budget": budget,
        "sections": sections,
    }


def test_comparison_budget_defaults_and_preflight_limits() -> None:
    assert asdict(ComparisonBudget.for_depth("normal")) == {
        "max_subjects": 3,
        "max_axes": 4,
        "max_required_cells": 12,
        "initial_queries_per_cell": 1,
        "max_gap_queries_per_cell": 1,
        "comparative_queries_per_axis": 1,
        "max_queries_per_wave": 8,
        "max_fetch_urls_per_wave": 10,
        "max_rounds": 3,
    }
    assert ComparisonBudget.for_depth("deep").max_required_cells == 25

    plan = _comparison_plan(subjects=["A", "B", "C", "D"], axes=["机制"])
    with pytest.raises(ValueError, match="subjects=4>3"):
        research_service._validate_comparison_plan_budget(
            plan,
            asdict(ComparisonBudget.for_depth("normal")),
        )


def test_comparison_cells_are_nested_under_axis_questions() -> None:
    subjects = ["A", "B", "C"]
    axes = ["机制", "成本", "生态", "风险"]
    states = question_states_from_plan(_comparison_plan(subjects=subjects, axes=axes), depth="normal")

    assert len(states) == len(axes)
    assert sum(len(state.comparison_cells) for state in states) == len(subjects) * len(axes)
    assert {state.comparison_axis for state in states} == set(axes)
    assert all(len(state.comparison_cells) == len(subjects) for state in states)
    assert all(state.budget.max_rounds == 3 for state in states)


def test_open_axis_question_is_canonicalized_and_owns_all_subject_cells() -> None:
    plan = {
        "research_type": "comparison",
        "subjects": ["LightRAG", "GraphRAG"],
        "comparison_axes": ["局限"],
        "matrix_cells": [
            {"id": "cell_001", "subject": "LightRAG", "axis": "局限"},
            {"id": "cell_002", "subject": "GraphRAG", "axis": "局限"},
        ],
        "comparison_budget": asdict(ComparisonBudget.for_depth("normal")),
        "sections": [
            {
                "id": "sec_04",
                "title": "04_局限对比",
                "questions": [{
                    "id": "sec_04_q1",
                    "text": "LightRAG 和 GraphRAG 各自有哪些有证据支持的局限？",
                    "comparison_axis_mode": "open",
                    "comparison_axis": "",
                    "comparison_subjects": ["LightRAG", "GraphRAG"],
                    "success_criteria": ["确定两者各自有证据支持的局限或证据缺口"],
                }],
            },
            {
                "id": "sec_05",
                "title": "05_补充信息",
                "questions": [{
                    "id": "sec_05_q1",
                    "text": "还有哪些性能或社区信息值得补充？",
                    "comparison_axis_mode": "open",
                    "comparison_axis": "",
                    "comparison_subjects": ["LightRAG", "GraphRAG"],
                    "success_criteria": [],
                }],
            },
        ],
    }

    states = question_states_from_plan(plan, depth="normal")
    limitation = next(state for state in states if state.question_id == "sec_04_q1")
    supplemental = next(state for state in states if state.question_id == "sec_05_q1")

    assert limitation.comparison_axis == "局限"
    assert [cell.cell_id for cell in limitation.comparison_cells] == ["cell_001", "cell_002"]
    assert all(cell.criteria for cell in limitation.comparison_cells)
    assert supplemental.comparison_cells == []


def test_comparison_matrix_rejects_plan_without_unique_axis_question() -> None:
    plan = {
        "research_type": "comparison",
        "subjects": ["A", "B"],
        "comparison_axes": ["局限"],
        "matrix_cells": [
            {"id": "cell_001", "subject": "A", "axis": "局限"},
            {"id": "cell_002", "subject": "B", "axis": "局限"},
        ],
        "comparison_budget": asdict(ComparisonBudget.for_depth("normal")),
        "sections": [{
            "id": "sec_01",
            "title": "补充信息",
            "questions": [{
                "id": "sec_01_q1",
                "text": "还有哪些性能数据值得补充？",
                "comparison_axis_mode": "open",
                "comparison_subjects": ["A", "B"],
                "success_criteria": ["列出性能数据"],
            }],
        }],
    }

    with pytest.raises(ValueError, match="requires exactly one Axis Question"):
        question_states_from_plan(plan, depth="normal")


def test_cross_axis_evidence_cannot_fill_required_cell() -> None:
    quote = "Google Gemini Deep Research 提供多步骤研究计划，用户可以确认或调整后执行。"
    accepted, rejected = admit_evidence(
        [{
            "claim": quote,
            "evidence_text": quote,
            "source_url": "https://example.com/gemini",
            "source_title": "Gemini Deep Research",
            "chunk_id": "chunk_1",
            "criterion_ids": ["criterion_01"],
            "subject": "Google Gemini Deep Research",
            "subject_relation": "target",
            "comparison_axis": "证据引用",
            "scope": "domain_specific",
            "claim_type": "direct_fact",
            "directness_score": 0.9,
            "applicability_score": 0.9,
        }],
        chunks=[{
            "id": "chunk_1",
            "text": quote,
            "source_url": "https://example.com/gemini",
            "source_title": "Gemini Deep Research",
        }],
        question_id="sec_01_q1",
        section_id="sec_01",
        question="比较 Elicit、Perplexity 和 Gemini Deep Research 的证据引用。",
        comparison_subjects=["Elicit", "Perplexity", "Gemini Deep Research"],
        expected_comparison_axis="证据引用",
    )

    assert accepted == []
    assert rejected[0]["criterion_ids"] == []
    assert rejected[0]["comparison_axis_binding_status"] == "incompatible_axis"
    assert rejected[0]["rejection_reason"] == "comparison_axis_semantic_mismatch"


def test_subject_alias_can_complete_the_correct_matrix_cell() -> None:
    plan = _comparison_plan(subjects=["Gemini Deep Research"], axes=["适用场景"])
    state = question_states_from_plan(plan, depth="normal")[0]
    quote = "Gemini 适用于市场分析和竞品研究场景。"
    accepted, rejected = admit_evidence(
        [{
            "id": "ev_001",
            "claim": quote,
            "evidence_text": quote,
            "source_url": "https://example.com/gemini",
            "source_title": "Gemini Deep Research",
            "chunk_id": "chunk_1",
            "criterion_ids": ["criterion_01"],
            "subject": "Gemini",
            "subject_relation": "target",
            "comparison_axis": "适用场景",
            "scope": "domain_specific",
            "claim_type": "direct_fact",
            "directness_score": 0.9,
            "applicability_score": 0.9,
        }],
        chunks=[{
            "id": "chunk_1",
            "text": quote,
            "source_url": "https://example.com/gemini",
            "source_title": "Gemini Deep Research",
        }],
        question_id=state.question_id,
        section_id=state.section_id,
        question=state.question,
        comparison_subjects=state.comparison_subjects,
        expected_comparison_axis=state.comparison_axis,
    )

    assert rejected == []
    assert accepted[0]["criterion_binding_status"] == "verified_target"
    state.accepted_evidence = accepted
    _update_comparison_cells(state, force_terminal=True)
    assert state.comparison_cells[0].status == "complete"
    assert state.comparison_cells[0].evidence_ids == ["ev_001"]


def test_use_case_fit_language_can_complete_the_correct_matrix_cell() -> None:
    plan = _comparison_plan(subjects=["Perplexity"], axes=["适用场景"])
    state = question_states_from_plan(plan, depth="normal")[0]
    quote = "Choose Perplexity when you need current sources and want to inspect leads quickly."
    accepted, rejected = admit_evidence(
        [{
            "id": "ev_001",
            "claim": "Perplexity Research 适合快速搜索当前网络来源。",
            "evidence_text": quote,
            "source_url": "https://example.com/perplexity",
            "source_title": "Perplexity Research",
            "chunk_id": "chunk_1",
            "criterion_ids": ["criterion_01"],
            "subject": "Perplexity",
            "subject_relation": "target",
            "comparison_axis": "适用场景",
            "scope": "domain_specific",
            "claim_type": "direct_fact",
            "directness_score": 0.9,
            "applicability_score": 0.9,
        }],
        chunks=[{
            "id": "chunk_1",
            "text": quote,
            "source_url": "https://example.com/perplexity",
            "source_title": "Perplexity Research",
        }],
        question_id=state.question_id,
        section_id=state.section_id,
        question=state.question,
        comparison_subjects=state.comparison_subjects,
        expected_comparison_axis=state.comparison_axis,
    )

    assert rejected == []
    assert accepted[0]["comparison_axis_binding_status"] == "verified_axis"
    state.accepted_evidence = accepted
    _update_comparison_cells(state, force_terminal=True)
    assert state.comparison_cells[0].status == "complete"


def test_comparison_normalization_preserves_fifth_axis_question_only_for_comparison() -> None:
    subjects = ["A", "B"]
    axes = ["轴一", "轴二", "轴三", "轴四", "轴五"]
    raw = {
        "research_type": "comparison",
        "subjects": subjects,
        "comparison_axes": axes,
        "sections": [
            {"title": "总览", "questions": []},
            {
                "title": "横向比较",
                "questions": [
                    {
                        "text": f"比较{axis}",
                        "comparison_axis": axis,
                        "comparison_subjects": subjects,
                        "success_criteria": [f"回答{axis}"],
                    }
                    for axis in axes
                ],
            },
        ],
    }

    normalized = research_service._normalize_research_plan(
        raw,
        user_request="比较 A 与 B",
        topic="A vs B",
        strategy="standard",
        planner_mode="authored_plan",
    )
    questions = normalized["sections"][1]["questions"]

    assert len(questions) == 5
    assert questions[-1]["comparison_axis"] == "轴五"
    assert questions[-1]["comparison_subjects"] == subjects


def test_comparison_scheduler_finishes_all_initial_cells_before_gap_queries() -> None:
    plan = _comparison_plan(
        subjects=["A", "B", "C"],
        axes=["机制", "成本", "生态", "风险"],
    )
    states = question_states_from_plan(plan, depth="normal")
    initial_queries = research_service._question_initial_queries({}, states)
    initial_query_set = {query for queries in initial_queries.values() for query in queries}
    calls: list[list[str]] = []

    def batch(active_states, queries_by_id, _round_index):
        queries = [query for state in active_states for query in queries_by_id.get(state.question_id, [])]
        calls.append(queries)
        return {
            "results": {
                state.question_id: {
                    "sources": [],
                    "materials": [],
                    "selected_chunks": [],
                    "evidence_map": [],
                    "provider_attempts": [{"status": "empty", "provider": "fake"}],
                }
                for state in active_states
            },
            "diagnostics": {},
        }

    result = QuestionResearchOrchestrator(
        lambda *_args: {},
        batch_retrieval=batch,
        parallelism=1,
    ).run(states, initial_queries=initial_queries)

    assert calls
    assert all(len(call) <= 8 for call in calls)
    seen: set[str] = set()
    first_non_initial_wave = len(calls)
    for index, call in enumerate(calls):
        if any(query not in initial_query_set for query in call):
            first_non_initial_wave = index
            break
        seen.update(call)
    assert seen == initial_query_set
    assert first_non_initial_wave >= 2
    assert result["pipeline_diagnostics"]["question_count"] == 4


def test_matrix_coverage_keeps_unavailable_distinct_from_exhausted() -> None:
    plan = _comparison_plan(subjects=["A", "B"], axes=["机制"])
    states = question_states_from_plan(plan, depth="normal")
    rows = [state.to_dict() for state in states]
    rows[0]["comparison_cells"][0]["status"] = "unavailable"
    rows[0]["comparison_cells"][0]["unavailable_reasons"] = ["provider timeout"]
    rows[0]["comparison_cells"][1]["status"] = "exhausted"

    coverage = research_service._comparison_matrix_coverage(rows, plan)

    assert coverage["status"] == "partial"
    assert coverage["status_counts"]["unavailable"] == 1
    assert coverage["status_counts"]["exhausted"] == 1
    assert coverage["requires_gap_acknowledgement"] is True


def test_source_pool_round_robins_subject_slots_across_axes() -> None:
    plan = _comparison_plan(subjects=["Alpha", "Beta"], axes=["机制", "成本"])
    states = question_states_from_plan(plan, depth="normal")
    pool = GlobalSourcePool()
    for state in states:
        pool.add([{
            "url": f"https://example.com/{state.question_id}/alpha",
            "title": f"Alpha {state.comparison_axis} official evidence",
            "snippet": f"Alpha {state.comparison_axis}",
            "source_type": "official_docs",
        }], state, match_states=states)

    selected = pool.select(states, max_urls=2)

    assert len(selected) == 2
    assert {next(iter(item.discovered_by_question_ids)) for item in selected} == {
        state.question_id for state in states
    }


def test_partial_matrix_disclosure_is_embedded_in_report_and_archive_document() -> None:
    matrix = {
        "enabled": True,
        "complete": False,
        "complete_cell_count": 1,
        "required_cell_count": 2,
        "gaps": [{
            "cell_id": "cell_002",
            "subject": "Beta",
            "axis": "机制",
            "status": "unavailable",
            "criteria": [{"description": "官方机制说明", "status": "uncovered"}],
            "unavailable_reasons": ["provider timeout"],
        }],
    }
    documents, report = research_service._attach_matrix_gap_disclosure(
        [
            {"title": "00_总览", "content": "总览"},
            {"title": "01_机制", "content": "机制正文"},
        ],
        {"title": "整合报告", "content": "整合正文"},
        matrix,
    )

    assert "比较矩阵缺口" in documents[1]["content"]
    assert "`unavailable`" in documents[1]["content"]
    assert "provider timeout" in report["content"]
