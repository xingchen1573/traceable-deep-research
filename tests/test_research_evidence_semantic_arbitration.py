from __future__ import annotations

import copy
import json
from typing import Any

import pytest

import workers
from backend.research import evidence as evidence_module
from backend.research.evidence import (
    admit_evidence,
    apply_semantic_arbitration,
    is_direct_exact_evidence,
)
from backend.research.models import (
    ComparisonCellState,
    QuestionBudget,
    ResearchCriterionState,
    ResearchQuestionState,
)
from backend.research.orchestrator import (
    QuestionResearchOrchestrator,
    _global_evidence,
    _update_comparison_cells,
)
from backend.services.research_service import _coverage_evidence_digest
from scripts.replay_research_coverage import _state_from_dict


def _state(*, comparison: bool = False) -> ResearchQuestionState:
    criterion = ResearchCriterionState(
        criterion_id="criterion_01",
        description="目标对象的路线规划事实" if not comparison else "覆盖能力事实",
        level="hard",
    )
    state = ResearchQuestionState(
        question_id="q1",
        section_id="sec_01",
        section_title="能力",
        question="旅游 Agent 的路线规划能力如何？" if not comparison else "比较 A 与 B 的能力",
        purpose="核验目标对象能力",
        success_criteria=[criterion.description],
        preferred_source_types=["official_docs"],
        budget=QuestionBudget(
            max_rounds=1,
            max_sources=4,
            timeout_seconds=30,
            max_model_calls=6,
        ),
        criteria=[criterion],
        topic_anchor="旅游 Agent" if not comparison else "A B",
        comparison_subjects=["A", "B"] if comparison else [],
        comparison_axis="能力" if comparison else "",
        min_evidence=1,
        min_sources=1,
    )
    if comparison:
        state.comparison_cells = [
            ComparisonCellState(
                cell_id="cell_a_ability",
                subject="A",
                axis="能力",
                criteria=[copy.deepcopy(criterion)],
            )
        ]
        state.comparison_budget = {"max_gap_queries_per_cell": 0}
    return state


def _candidate(
    *,
    chunk_id: str = "c1",
    quote: str = "多个自治模块通过共享状态协同完成复杂任务。",
    claim: str = "多个自治模块可以协同完成复杂任务。",
    source_url: str = "https://example.com/travel-agent",
    subject: str = "旅游 Agent",
    support_relation: str = "direct_support",
    target_fit: str = "exact_target",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "claim": claim,
        "evidence_text": quote,
        "source_url": source_url,
        "source_title": f"{subject} documentation",
        "criterion_ids": ["criterion_01"],
        "subject": subject,
        "subject_relation": "target",
        "scope": "domain_specific",
        "claim_type": "direct_fact",
        "directness_score": 0.9,
        "applicability_score": 0.9,
        "support_relation": support_relation,
        "target_fit": target_fit,
        "semantic_reason": "初步语义判断",
    }


def _retrieval(items: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_chunks": chunks,
        "evidence_map": items,
        "sources": [],
        "materials": [],
        "provider_attempts": [],
        "_diagnostics": {"model_call_count": 1},
    }


def _run_once(state: ResearchQuestionState, retrieval: dict[str, Any]) -> ResearchQuestionState:
    orchestrator = QuestionResearchOrchestrator(lambda *_args: retrieval, coverage_judge=None)
    orchestrator.run([state], initial_queries={state.question_id: [state.question]})
    return state


def test_grounding_failure_does_not_call_arbitration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.1] * len(kwargs["texts"]))
    calls = []
    monkeypatch.setattr(workers, "arbitrate_evidence_semantics", lambda **kwargs: calls.append(kwargs))
    item = _candidate(quote="这段 quote 不在 chunk 中。")
    _run_once(_state(), _retrieval([item], [{
        "id": "c1",
        "text": "完全不同的原文片段，无法定位模型返回的 quote。",
        "source_url": item["source_url"],
        "source_title": item["source_title"],
    }]))
    assert calls == []


def test_illegal_subject_does_not_call_arbitration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.1] * len(kwargs["texts"]))
    calls = []
    monkeypatch.setattr(workers, "arbitrate_evidence_semantics", lambda **kwargs: calls.append(kwargs))
    item = _candidate(subject="C", quote="C 可以完成能力任务。", claim="C 可以完成当前能力任务。")
    item["comparison_axis"] = "能力"
    _run_once(_state(comparison=True), _retrieval([item], [{
        "id": "c1",
        "text": item["evidence_text"],
        "source_url": item["source_url"],
        "source_title": item["source_title"],
    }]))
    assert calls == []


def test_graphrag_correct_evidence_is_not_lexically_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.74])
    quote = "Community reports are generated after hierarchical clustering and can be used during retrieval."
    accepted, rejected = admit_evidence(
        [_candidate(
            quote=quote,
            claim="GraphRAG 在分层聚类后生成社区报告，并在检索时使用这些报告。",
            subject="GraphRAG",
            source_url="https://example.com/graphrag",
        )],
        chunks=[{
            "id": "c1",
            "text": quote,
            "source_url": "https://example.com/graphrag",
            "source_title": "GraphRAG documentation",
        }],
        question_id="q1",
        section_id="sec_01",
        question="GraphRAG 与 LightRAG 的检索延迟和社区结构如何？",
        target_subject="GraphRAG",
        valid_criterion_ids={"criterion_01"},
    )
    assert rejected == []
    evidence_id = accepted[0]["semantic_arbitration"]["evidence_id"]
    accepted, semantic_rejected = apply_semantic_arbitration(
        accepted,
        verdicts={evidence_id: {
            "support_relation": "direct_support",
            "target_fit": "exact_target",
            "criterion_supported": True,
            "requires_inference": False,
            "reason": "原文直接讨论 GraphRAG 的社区报告与检索用途",
            "confidence": "high",
        }},
    )
    assert semantic_rejected == []
    assert accepted[0]["support_relation"] == "direct_support"
    assert accepted[0]["target_fit"] == "exact_target"
    assert accepted[0]["directness_score"] == 0.9


def test_general_multi_agent_content_cannot_become_travel_direct_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.1])
    quote = "多个自治模块通过共享状态协同完成复杂任务。"
    accepted, _ = admit_evidence(
        [_candidate(quote=quote)],
        chunks=[{
            "id": "c1",
            "text": quote,
            "source_url": "https://example.com/travel-agent",
            "source_title": "旅游 Agent documentation",
        }],
        question_id="q1",
        section_id="sec_01",
        question="旅游 Agent 的路线规划能力如何？",
        target_subject="旅游 Agent",
        valid_criterion_ids={"criterion_01"},
    )
    evidence_id = accepted[0]["semantic_arbitration"]["evidence_id"]
    accepted, rejected = apply_semantic_arbitration(
        accepted,
        verdicts={evidence_id: {
            "support_relation": "unrelated",
            "target_fit": "general",
            "criterion_supported": False,
            "requires_inference": False,
            "reason": "原文只讨论一般多 Agent 协作，没有路线规划事实",
            "confidence": "high",
        }},
    )
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "semantic_unrelated"


def test_conflicts_are_batched_once_per_question_round(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.1] * len(kwargs["texts"]))
    calls: list[dict[str, Any]] = []

    def arbitrate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "resolved",
            "reason": "",
            "verdicts": {
                item["semantic_arbitration"]["evidence_id"]: {
                    "support_relation": "inference_support",
                    "target_fit": "related_target",
                    "criterion_supported": True,
                    "requires_inference": True,
                    "reason": "需要从一般协作机制推断到目标能力",
                    "confidence": "medium",
                }
                for item in kwargs["evidence"]
            },
        }

    monkeypatch.setattr(workers, "arbitrate_evidence_semantics", arbitrate)
    first = _candidate(chunk_id="c1")
    second = _candidate(
        chunk_id="c2",
        quote="多个智能体通过消息协议分配任务并汇总结果。",
        claim="多个智能体可以分配任务并汇总结果。",
        source_url="https://example.org/travel-agent",
    )
    state = _run_once(_state(), _retrieval([first, second], [
        {"id": "c1", "text": first["evidence_text"], "source_url": first["source_url"], "source_title": first["source_title"]},
        {"id": "c2", "text": second["evidence_text"], "source_url": second["source_url"], "source_title": second["source_title"]},
    ]))
    assert len(calls) == 1
    assert len(calls[0]["evidence"]) == 2
    assert state.model_calls == 2


def test_no_conflict_adds_no_arbitration_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.1] * len(kwargs["texts"]))
    calls = []
    monkeypatch.setattr(workers, "arbitrate_evidence_semantics", lambda **kwargs: calls.append(kwargs))
    item = _candidate(
        support_relation="context_only",
        target_fit="general",
    )
    item.update({"scope": "general", "claim_type": "design_principle", "directness_score": 0.4, "applicability_score": 0.4})
    _run_once(_state(), _retrieval([item], [{
        "id": "c1",
        "text": item["evidence_text"],
        "source_url": item["source_url"],
        "source_title": item["source_title"],
    }]))
    assert calls == []


@pytest.mark.parametrize(
    ("response_or_error", "expected_status"),
    [
        ("not valid json", "invalid_response"),
        (TimeoutError("slow"), "unavailable"),
        (RuntimeError("provider down"), "unavailable"),
    ],
)
def test_arbitration_invalid_timeout_and_provider_failures_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    response_or_error: Any,
    expected_status: str,
) -> None:
    class Response:
        content = response_or_error
        response_metadata: dict[str, Any] = {}

    def invoke(*_args: Any, **_kwargs: Any) -> Any:
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return Response()

    monkeypatch.setattr(workers, "_invoke_evidence_chat", invoke)
    result = workers.arbitrate_evidence_semantics(
        question_id="q1",
        section_id="sec_01",
        question="旅游 Agent 的路线规划能力如何？",
        purpose="核验能力",
        criteria=[{"criterion_id": "criterion_01", "description": "路线规划事实", "level": "hard"}],
        target_subject="旅游 Agent",
        comparison_subjects=[],
        comparison_axis="",
        evidence=[{
            "claim": "多个自治模块可以协同完成复杂任务。",
            "raw_quote": "多个自治模块通过共享状态协同完成复杂任务。",
            "semantic_arbitration": {
                "evidence_id": "candidate_001",
                "initial": {"support_relation": "direct_support", "target_fit": "exact_target"},
                "program_signals": {"lexical_scope_signal": "general"},
                "conflict_reasons": ["model_direct_exact_but_lexical_general"],
            },
        }],
    )
    assert result["status"] == expected_status
    assert result["verdicts"] == {}


def test_unresolved_conflict_is_conservatively_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    item = _candidate()
    item["semantic_arbitration"] = {
        "version": "v1",
        "mode": "live",
        "status": "pending",
        "evidence_id": "candidate_001",
        "initial": {"support_relation": "direct_support", "target_fit": "exact_target"},
        "conflict_reasons": ["model_direct_exact_but_lexical_general"],
    }
    accepted, rejected = apply_semantic_arbitration(
        [item],
        failure_status="invalid_response",
        failure_reason="malformed JSON",
    )
    assert rejected == []
    assert accepted[0]["support_relation"] == "inference_support"
    assert accepted[0]["semantic_arbitration"]["status"] == "invalid_response"
    assert not is_direct_exact_evidence(accepted[0])


def test_off_mode_preserves_legacy_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "off")
    quote = "GraphRAG uses community reports during retrieval."
    raw = _candidate(quote=quote, claim="GraphRAG 在检索阶段使用社区报告。", subject="GraphRAG")
    accepted, rejected = admit_evidence(
        [raw],
        chunks=[{"id": "c1", "text": quote, "source_url": raw["source_url"], "source_title": raw["source_title"]}],
        question_id="q1",
        section_id="sec_01",
        question="GraphRAG 如何使用社区报告？",
    )
    assert rejected == []
    assert "semantic_arbitration" not in accepted[0]
    assert accepted[0]["scope"] in {"domain_specific", "cross_domain", "general"}


def test_shadow_records_difference_without_changing_business_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    quote = "多个自治模块通过共享状态协同完成复杂任务。"
    raw = _candidate(quote=quote)
    chunk = {"id": "c1", "text": quote, "source_url": raw["source_url"], "source_title": raw["source_title"]}
    monkeypatch.setattr(evidence_module, "_semantic_rescue_scores", lambda **kwargs: [0.9])
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "off")
    legacy, _ = admit_evidence([raw], chunks=[chunk], question_id="q1", section_id="sec_01", question="旅游 Agent 的路线规划能力如何？")
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "shadow")
    shadow, _ = admit_evidence([raw], chunks=[chunk], question_id="q1", section_id="sec_01", question="旅游 Agent 的路线规划能力如何？")
    evidence_id = shadow[0]["semantic_arbitration"]["evidence_id"]
    shadow, rejected = apply_semantic_arbitration(
        shadow,
        verdicts={evidence_id: {
            "support_relation": "unrelated",
            "target_fit": "general",
            "criterion_supported": False,
            "requires_inference": False,
            "reason": "一般协作知识不回答路线规划",
            "confidence": "high",
        }},
    )
    assert rejected == []
    for field in ("scope", "directness_score", "applicability_score", "claim_type", "admission_status", "evidence_score"):
        assert shadow[0][field] == legacy[0][field]
    assert "support_relation" not in shadow[0]
    assert shadow[0]["semantic_arbitration"]["would_change"] is True


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_non_live_coverage_digest_preserves_legacy_shape(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", mode)
    state = _state()
    state.accepted_evidence = [{"id": "ev_001", **_candidate()}]
    row = _coverage_evidence_digest(state)["evidence_registry"][0]
    assert "support_relation" not in row
    assert "target_fit" not in row
    assert "directness_score" not in row
    assert "applicability_score" not in row


def test_live_coverage_digest_exposes_semantic_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    state = _state()
    state.accepted_evidence = [{"id": "ev_001", **_candidate()}]
    row = _coverage_evidence_digest(state)["evidence_registry"][0]
    assert row["support_relation"] == "direct_support"
    assert row["target_fit"] == "exact_target"
    assert row["directness_score"] == 0.9
    assert row["applicability_score"] == 0.9


def test_inference_evidence_cannot_complete_comparison_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    state = _state(comparison=True)
    state.accepted_evidence = [{
        "id": "ev_001",
        "subject": "A",
        "comparison_axis": "能力",
        "criterion_ids": ["criterion_01"],
        "criterion_binding_status": "verified_target",
        "comparison_axis_binding_status": "verified_axis",
        "support_relation": "inference_support",
        "target_fit": "exact_target",
    }]
    _update_comparison_cells(state, force_terminal=True)
    assert state.comparison_cells[0].status == "partial"
    assert state.comparison_cells[0].criteria[0].status == "partial"


def test_direct_exact_evidence_can_complete_comparison_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "live")
    state = _state(comparison=True)
    state.accepted_evidence = [{
        "id": "ev_001",
        "subject": "A",
        "comparison_axis": "能力",
        "criterion_ids": ["criterion_01"],
        "criterion_binding_status": "verified_target",
        "comparison_axis_binding_status": "verified_axis",
        "support_relation": "direct_support",
        "target_fit": "exact_target",
    }]
    _update_comparison_cells(state, force_terminal=True)
    assert state.comparison_cells[0].status == "complete"
    assert state.comparison_cells[0].criteria[0].status == "covered"


def test_global_evidence_ids_remap_comparison_cells() -> None:
    state = _state(comparison=True)
    state.accepted_evidence = [{
        **_candidate(),
        "id": "",
        "_coverage_evidence_id": "q1:coverage_ev_001",
        "criterion_binding_status": "verified_target",
        "comparison_axis_binding_status": "verified_axis",
        "comparison_axis": "能力",
        "subject": "A",
    }]
    state.criteria[0].evidence_ids = ["q1:coverage_ev_001"]
    state.comparison_cells[0].evidence_ids = ["q1:coverage_ev_001"]
    state.comparison_cells[0].criteria[0].evidence_ids = ["q1:coverage_ev_001"]

    evidence_map = _global_evidence([state])

    assert evidence_map[0]["id"] == "ev_001"
    assert state.criteria[0].evidence_ids == ["ev_001"]
    assert state.comparison_cells[0].evidence_ids == ["ev_001"]
    assert state.comparison_cells[0].criteria[0].evidence_ids == ["ev_001"]


def test_old_checkpoint_without_semantic_fields_restores() -> None:
    raw = _state().to_dict()
    raw["accepted_evidence"] = [{
        "id": "ev_001",
        "scope": "domain_specific",
        "claim_type": "direct_fact",
        "directness_score": 0.9,
        "applicability_score": 0.9,
    }]
    restored = _state_from_dict(json.loads(json.dumps(raw, ensure_ascii=False)))
    assert restored.accepted_evidence[0].get("support_relation") is None
    assert is_direct_exact_evidence(restored.accepted_evidence[0])
