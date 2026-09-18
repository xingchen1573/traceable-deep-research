from __future__ import annotations

from backend.research.coverage import assess_question_coverage
from backend.research.evidence import admit_evidence
from backend.research.material_store import JobMaterialStore
from backend.research.models import QuestionBudget, ResearchCriterionState, ResearchQuestionState
from backend.research.orchestrator import _plan_gap_actions
from backend.research.source_policy import entity_match_metadata


def _state(question: str, *, kind: str, min_sources: int = 2) -> ResearchQuestionState:
    return ResearchQuestionState(
        question_id="q1",
        section_id="s1",
        section_title="section",
        question=question,
        purpose="answer",
        success_criteria=["one fact"],
        preferred_source_types=["github"],
        budget=QuestionBudget(max_rounds=2, max_sources=6, timeout_seconds=30, max_model_calls=4),
        question_kind=kind,
        min_evidence=1,
        min_sources=min_sources,
    )


def test_same_github_repository_has_multiple_source_identities():
    state = _state("DeepSeek Harness 的架构是什么？", kind="implementation")
    state.accepted_evidence = [
        {"source_url": "https://github.com/deepseek-ai/deepseek-harness/blob/main/README.md", "claim_cluster_id": "a", "grounded_exactly": True},
        {"source_url": "https://github.com/deepseek-ai/deepseek-harness/blob/main/docs/architecture.md", "claim_cluster_id": "b", "grounded_exactly": True},
    ]
    coverage = assess_question_coverage(state)
    assert coverage["source_floor_mode"] == "source_identity"
    assert coverage["source_identity_count"] == 2
    assert coverage["domain_count"] == 1
    assert coverage["checks"]["source_floor"] is True


def test_comparison_does_not_require_cross_domain_sources_by_question_kind():
    state = _state("比较两个系统的实现差异", kind="comparison")
    state.accepted_evidence = [
        {"source_url": "https://github.com/a/project/blob/main/README.md", "claim_cluster_id": "a", "grounded_exactly": True},
        {"source_url": "https://github.com/b/project/blob/main/docs/design.md", "claim_cluster_id": "b", "grounded_exactly": True},
    ]
    coverage = assess_question_coverage(state)
    assert coverage["source_floor_mode"] == "source_identity"
    assert coverage["source_identity_count"] == 2
    assert coverage["domain_count"] == 1
    assert coverage["checks"]["source_floor"] is True
    assert coverage["source_diversity_required"] is False


def test_confirmed_hard_cross_validation_criterion_requires_domains():
    state = _state("比较两个系统的实现差异", kind="comparison")
    state.criteria = [
        ResearchCriterionState("criterion_01", "用至少两个独立来源交叉验证核心差异", level="hard")
    ]
    state.accepted_evidence = [
        {"source_url": "https://github.com/a/project/blob/main/README.md", "claim_cluster_id": "a", "grounded_exactly": True},
        {"source_url": "https://github.com/b/project/blob/main/docs/design.md", "claim_cluster_id": "b", "grounded_exactly": True},
    ]

    coverage = assess_question_coverage(state)

    assert coverage["source_floor_mode"] == "domain"
    assert coverage["source_floor_target"] == 2
    assert coverage["checks"]["source_floor"] is False
    assert coverage["source_diversity_required"] is True
    assert next(item for item in coverage["gaps"] if item["gap_type"] == "source_floor")["blocking"] is True


def test_unrelated_projects_fail_entity_gate_before_semantic_rescue():
    assert entity_match_metadata(
        url="https://github.com/browser-use/browser-harness/blob/main/README.md",
        title="browser harness",
        topic="DeepSeek Harness",
    )["entity_mismatch_reason"] == "unrelated_project"
    quote = "The browser harness installs with pip and starts a local browser automation server."
    chunks = [{"id": "c1", "text": quote, "source_url": "https://github.com/browser-use/browser-harness/blob/main/README.md", "source_title": "browser harness"}]
    accepted, rejected = admit_evidence(
        [{"chunk_id": "c1", "atomic_claim": "DeepSeek Harness can be installed with pip.", "raw_quote": quote}],
        chunks=chunks,
        question_id="q1",
        section_id="s1",
        question="DeepSeek Harness 如何安装？",
    )
    assert not accepted
    assert rejected[0]["rejection_reason"] == "entity_mismatch"


def test_comparison_question_allows_primary_source_for_other_subject():
    question = "DeepSeek Harness 与 LangChain 的 Agent 架构有什么差异？"
    metadata = entity_match_metadata(
        url="https://python.langchain.com/docs/concepts/agents/",
        title="LangChain agents",
        snippet="Agents combine language models with tools and choose which tools to use.",
        topic=question,
    )
    assert metadata["entity_match_score"] == 1.0
    assert metadata["entity_mismatch_reason"] == ""

    quote = "LangChain agents combine language models with tools and choose which tools to use iteratively."
    chunks = [{
        "id": "c1",
        "text": quote,
        "source_url": "https://python.langchain.com/docs/concepts/agents/",
        "source_title": "LangChain agents",
    }]
    accepted, rejected = admit_evidence(
        [{"chunk_id": "c1", "atomic_claim": "LangChain agents let models select and use tools iteratively.", "raw_quote": quote}],
        chunks=chunks,
        question_id="q1",
        section_id="s1",
        question=question,
    )
    assert len(accepted) == 1
    assert rejected == []


def test_generic_entity_gate_keeps_uncertain_source_without_confirming_it():
    metadata = entity_match_metadata(
        url="https://example.com/article",
        title="Architecture notes",
        snippet="A plugin system with configurable tools.",
        topic="LightRAG",
    )
    assert metadata["entity_match_score"] == 0.5
    assert metadata["entity_mismatch_reason"] == "entity_uncertain"


def test_generic_entity_gate_confirms_anchor_in_title():
    metadata = entity_match_metadata(
        url="https://example.com/lightrag-guide",
        title="LightRAG Architecture Guide",
        snippet="A plugin system with configurable tools.",
        topic="LightRAG",
    )
    assert metadata["entity_match_score"] == 0.75
    assert metadata["entity_mismatch_reason"] == ""


def test_material_store_capability_circuit_breaker_records_first_reason():
    store = JobMaterialStore()
    assert store.disable_capability("context7", "HTTP 404 from context endpoint") is True
    assert store.disable_capability("context7", "HTTP 404 from another key") is False
    diagnostics = store.diagnostics()
    assert diagnostics["capabilities"]["context7"]["reason"] == "HTTP 404 from context endpoint"


def test_advisory_source_floor_does_not_generate_same_project_followup():
    state = _state("某个项目的架构是什么？", kind="implementation")
    state.topic_anchor = "某个项目"
    state.coverage = {
        "source_floor_mode": "source_identity",
        "gaps": [{"gap_type": "source_floor", "blocking": False}],
    }
    state.accepted_evidence = [{
        "source_url": "https://github.com/acme/example/blob/main/README.md",
    }]
    actions = _plan_gap_actions(state)
    assert actions == []
