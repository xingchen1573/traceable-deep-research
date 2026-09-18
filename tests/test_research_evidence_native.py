from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.research import evidence_native
from backend.research.evidence_native import (
    ARCHITECTURE,
    EvidenceContextManager,
    EvidenceNativeCheckpointStore,
    EvidenceNativeContractError,
    build_editorial_claim_bank,
    build_evidence_native_blueprint,
    validate_evidence_traceability,
    write_evidence_native_report,
)
from backend.research.quality import (
    ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
    evaluate_report_quality,
    validate_report_citations,
)
from backend.services.research_service import _research_writing_architecture


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "source_url": f"https://{chunk_id}.example.com/doc",
        "source_title": chunk_id,
    }


def _evidence(
    evidence_id: str,
    *,
    chunk_id: str,
    quote: str,
    criterion_ids: list[str] | None = None,
    subject: str = "",
    axis: str = "",
    matrix_cell_id: str = "",
    stance: str = "support",
    admission_status: str = "accepted",
) -> dict:
    return {
        "id": evidence_id,
        "chunk_id": chunk_id,
        "section_id": "sec_01",
        "question_id": "q1",
        "criterion_ids": criterion_ids or [],
        "subject": subject,
        "comparison_axis": axis,
        "matrix_cell_id": matrix_cell_id,
        "claim_cluster_id": f"cluster_{evidence_id}",
        "atomic_claim": quote,
        "claim": quote,
        "cleaned_quote": quote,
        "raw_quote": quote,
        "evidence_text": quote,
        "source_url": f"https://{chunk_id}.example.com/doc",
        "source_title": chunk_id,
        "source_type": "official_docs",
        "admission_status": admission_status,
        "support_relation": "direct_support" if admission_status == "accepted" else "inference_support",
        "target_fit": "exact_target" if admission_status == "accepted" else "related_target",
        "scope": "domain_specific" if admission_status == "accepted" else "cross_domain",
        "claim_type": "direct_fact" if admission_status == "accepted" else "design_principle",
        "stance": stance,
        "grounded_exactly": True,
        "directness_score": 1.0 if admission_status == "accepted" else 0.6,
        "evidence_score": 0.9,
        "authority_score": 0.9,
    }


def _plan() -> dict:
    return {
        "research_type": "comparison",
        "subjects": ["A"],
        "comparison_axes": ["可靠性"],
        "sections": [
            {
                "id": "sec_00",
                "title": "00_总览",
                "questions": [],
            },
            {
                "id": "sec_01",
                "title": "01_可靠性",
                "purpose": "比较恢复与限制",
                "questions": [
                    {
                        "id": "q1",
                        "text": "系统如何恢复？",
                        "answer_contract_level": "hard",
                        "comparison_axis_mode": "single_axis",
                    }
                ],
            },
            {"id": "sec_99", "title": "来源索引", "questions": []},
        ],
    }


def _states(*, evidence_ids: list[str] | None = None) -> list[dict]:
    ids = evidence_ids or ["ev_001"]
    return [
        {
            "section_id": "sec_01",
            "question_id": "q1",
            "question": "系统如何恢复？",
            "answer_contract_level": "hard",
            "status": "sufficient",
            "min_sources": 1,
            "criteria": [
                {
                    "criterion_id": "crit_hard",
                    "description": "恢复机制",
                    "level": "hard",
                    "status": "covered",
                    "evidence_ids": ids,
                }
            ],
            "comparison_cells": [
                {
                    "cell_id": "cell_a_reliability",
                    "subject": "A",
                    "axis": "可靠性",
                    "status": "covered",
                    "evidence_ids": ids,
                }
            ],
            "coverage": {"sufficient": True},
        }
    ]


def test_traceability_and_blueprint_accept_inference_only_without_treating_limited_as_empty() -> None:
    quote = "系统会从最近的合法检查点恢复执行。"
    chunks = [_chunk("chunk_1", f"前文。{quote}后文。")]
    evidence = [
        _evidence(
            "ev_001",
            chunk_id="chunk_1",
            quote=quote,
            criterion_ids=["crit_hard"],
            subject="A",
            axis="可靠性",
            matrix_cell_id="cell_a_reliability",
            admission_status="inference_only",
        )
    ]

    validate_evidence_traceability(evidence, chunks)
    blueprint = build_evidence_native_blueprint(
        normalized_request={"user_request": "检查恢复能力"},
        research_plan=_plan(),
        question_states=_states(),
        evidence_map=evidence,
        selected_chunks=chunks,
    )

    section = next(item for item in blueprint["sections"] if item["section_id"] == "sec_01")
    assert section["reportability"]["status"] == "ready"
    assert blueprint["completeness"]["core_empty_section_ids"] == []
    assert blueprint["completeness"]["status"] == "complete_report"

    limited_states = _states()
    limited_states[0]["status"] = "exhausted"
    limited = build_evidence_native_blueprint(
        normalized_request={"user_request": "检查恢复能力"},
        research_plan=_plan(),
        question_states=limited_states,
        evidence_map=evidence,
        selected_chunks=chunks,
    )
    limited_section = next(item for item in limited["sections"] if item["section_id"] == "sec_01")
    assert limited_section["reportability"]["status"] == "limited"
    assert limited["completeness"]["core_empty_section_ids"] == []
    assert limited["completeness"]["status"] == "complete_report"


def test_blueprint_uses_final_question_state_instead_of_rederiving_coverage_from_rows() -> None:
    evidence = [_evidence(
        "ev_001",
        chunk_id="chunk_1",
        quote="系统支持恢复。",
        criterion_ids=["crit_hard"],
        subject="A",
        axis="可靠性",
        matrix_cell_id="cell_a_reliability",
    )]
    states = _states()
    states[0]["criteria"][0]["status"] = "uncovered"
    states[0]["criteria"][0]["evidence_ids"] = ["ev_001"]
    states[0]["comparison_cells"][0]["status"] = "partial"

    blueprint = build_evidence_native_blueprint(
        normalized_request={"user_request": "检查恢复能力"},
        research_plan=_plan(),
        question_states=states,
        evidence_map=evidence,
    )

    section = next(item for item in blueprint["sections"] if item["section_id"] == "sec_01")
    assert blueprint["schema_version"] == "evidence-native-blueprint/v2"
    assert section["reportability"]["status"] == "limited"
    assert section["reportability"]["missing_hard_criterion_ids"] == ["crit_hard"]
    assert section["reportability"]["missing_comparison_cell_ids"] == ["cell_a_reliability"]


def test_blueprint_uses_question_state_source_floor_instead_of_recounting_domains() -> None:
    evidence = [
        _evidence("ev_001", chunk_id="chunk_1", quote="证据一。", criterion_ids=["crit_hard"]),
        _evidence("ev_002", chunk_id="chunk_1", quote="证据二。", criterion_ids=["crit_hard"]),
    ]
    states = _states(evidence_ids=["ev_001", "ev_002"])
    states[0]["min_sources"] = 2
    states[0]["coverage"] = {
        "sufficient": True,
        "checks": {"source_floor": True},
        "source_count": 2,
        "source_floor_target": 2,
        "source_floor_mode": "source_identity",
    }

    blueprint = build_evidence_native_blueprint(
        normalized_request={"user_request": "检查恢复能力"},
        research_plan=_plan(),
        question_states=states,
        evidence_map=evidence,
    )

    section = next(item for item in blueprint["sections"] if item["section_id"] == "sec_01")
    assert section["reportability"]["status"] == "ready"
    assert section["reportability"]["source_shortfalls"] == []


def test_traceability_rejects_quote_not_present_in_frozen_chunk() -> None:
    evidence = [_evidence("ev_001", chunk_id="chunk_1", quote="不存在的原句")]
    with pytest.raises(EvidenceNativeContractError, match="not traceable"):
        validate_evidence_traceability(evidence, [_chunk("chunk_1", "另一段内容")])


def test_blueprint_rejects_non_admitted_evidence_instead_of_silently_dropping_it() -> None:
    evidence = [_evidence("ev_001", chunk_id="chunk_1", quote="证据", admission_status="accepted")]
    evidence[0]["admission_status"] = "quarantined"
    with pytest.raises(EvidenceNativeContractError, match="non-citable or non-admitted"):
        build_evidence_native_blueprint(
            normalized_request={"user_request": "测试"},
            research_plan=_plan(),
            question_states=_states(),
            evidence_map=evidence,
        )


def test_context_selection_is_order_independent_and_preserves_priority_classes() -> None:
    rows = [
        _evidence(
            "ev_001",
            chunk_id="hard",
            quote="硬性标准证据。" * 20,
            criterion_ids=["crit_hard"],
        ),
        _evidence(
            "ev_002",
            chunk_id="cell",
            quote="比较单元证据。" * 20,
            subject="A",
            axis="可靠性",
            matrix_cell_id="cell_a_reliability",
        ),
        _evidence(
            "ev_003",
            chunk_id="conflict",
            quote="该机制在网络中断时存在限制。" * 20,
            stance="limitation",
        ),
        _evidence(
            "ev_004",
            chunk_id="filler",
            quote="补充材料。" * 900,
        ),
    ]
    section = {
        "section_id": "sec_01",
        "title": "可靠性",
        "research_questions": [{"question_id": "q1"}],
    }
    manager = EvidenceContextManager(max_input_tokens=2600, prompt_reserve_tokens=500)

    packet_a, diagnostics_a = manager.build(
        section=section, question_states=_states(evidence_ids=["ev_001"]), evidence_map=rows
    )
    packet_b, diagnostics_b = manager.build(
        section=copy.deepcopy(section),
        question_states=list(reversed(_states(evidence_ids=["ev_001"]))),
        evidence_map=list(reversed(rows)),
    )

    assert packet_a == packet_b
    assert diagnostics_a["selected_evidence_ids"] == diagnostics_b["selected_evidence_ids"]
    assert {"ev_001", "ev_002", "ev_003"}.issubset(packet_a["allowed_evidence_ids"])
    assert "ev_004" in diagnostics_a["cropped_evidence_ids"]
    assert packet_a["section"]["evidence_ids"] == packet_a["allowed_evidence_ids"]
    assert diagnostics_a["candidate_evidence_ids"] == ["ev_001", "ev_002", "ev_003", "ev_004"]
    assert diagnostics_a["estimated_input_tokens"] <= diagnostics_a["material_budget_tokens"]


def test_context_keeps_unbound_evidence_available_without_calling_it_criterion_support() -> None:
    rows = [
        _evidence(
            "ev_001",
            chunk_id="candidate",
            quote="候选材料。",
            criterion_ids=["crit_hard"],
        ),
        _evidence("ev_002", chunk_id="other", quote="其他材料。"),
    ]
    states = _states()
    states[0]["criteria"][0]["evidence_ids"] = []
    section = {
        "section_id": "sec_01",
        "title": "可靠性",
        "research_questions": [{"question_id": "q1"}],
        "evidence_ids": ["ev_001", "ev_002"],
    }

    packet, diagnostics = EvidenceContextManager().build(
        section=section,
        question_states=states,
        evidence_map=rows,
    )

    assert packet["schema_version"] == "evidence-context-packet/v2"
    assert set(packet["allowed_evidence_ids"]) == {"ev_001", "ev_002"}
    assert packet["section"]["evidence_ids"] == packet["allowed_evidence_ids"]
    assert diagnostics["candidate_evidence_ids"] == ["ev_001", "ev_002"]
    assert diagnostics["criterion_coverage"][0]["selected"] is False
    packet_rows = [row for cluster in packet["claim_clusters"] for row in cluster["evidence"]]
    ev_001 = next(row for row in packet_rows if row["id"] == "ev_001")
    assert ev_001["criterion_ids"] == []
    assert ev_001["extracted_criterion_ids"] == ["crit_hard"]


def test_blueprint_and_context_include_evidence_shared_with_a_later_question() -> None:
    shared = _evidence(
        "ev_001",
        chunk_id="shared",
        quote="共享证据也回答当前问题。",
        criterion_ids=["crit_hard"],
    )
    shared.update({
        "section_id": "sec_other",
        "question_id": "q_other",
        "question_ids": ["q_other", "q1"],
    })
    states = _states()

    blueprint = build_evidence_native_blueprint(
        normalized_request={"user_request": "检查恢复能力"},
        research_plan=_plan(),
        question_states=states,
        evidence_map=[shared],
    )
    section = next(item for item in blueprint["sections"] if item["section_id"] == "sec_01")
    assert section["evidence_ids"] == ["ev_001"]
    assert section["reportability"]["status"] == "ready"

    packet, diagnostics = EvidenceContextManager().build(
        section=section,
        question_states=states,
        evidence_map=[shared],
    )
    assert packet["allowed_evidence_ids"] == ["ev_001"]
    assert diagnostics["criterion_coverage"] == [{
        "question_id": "q1",
        "criterion_id": "crit_hard",
        "level": "hard",
        "selected": True,
    }]


def test_context_fails_when_fixed_prompt_overhead_exceeds_budget() -> None:
    section = {
        "section_id": "sec_01",
        "title": "可靠性",
        "purpose": "超长结构说明" * 2000,
        "research_questions": [{"question_id": "q1"}],
    }
    with pytest.raises(EvidenceNativeContractError, match="overhead exceeds"):
        EvidenceContextManager(max_input_tokens=2000, prompt_reserve_tokens=500).build(
            section=section,
            question_states=_states(),
            evidence_map=[_evidence("ev_001", chunk_id="chunk_1", quote="证据")],
        )


def test_cited_span_and_uncited_analysis_are_kept_separate() -> None:
    evidence = _evidence("ev_001", chunk_id="chunk_1", quote="系统支持恢复。")
    rendered, ledger = evidence_native._render_citations_with_spans(
        "系统支持恢复[[ev_001]]，这有助于定位失败。",
        {"ev_001": evidence},
        {evidence["source_url"]: 1},
    )

    assert "[1]" in rendered
    assert ledger[0]["cited_span"] == "系统支持恢复"
    assert ledger[0]["uncited_analysis_span"] == "这有助于定位失败。"


def test_eplus_keeps_repeated_source_citation_visible_for_each_claim_marker() -> None:
    evidence = {
        "ev_001": _evidence("ev_001", chunk_id="chunk_1", quote="第一条事实。"),
        "ev_002": _evidence("ev_002", chunk_id="chunk_1", quote="第二条事实。"),
    }
    citation_map = {"https://chunk_1.example.com/doc": 1}

    rendered, ledger = evidence_native._render_citations_with_spans(
        "第一条事实[[ev_001]]，例如第二条事实[[ev_002]]。",
        evidence,
        citation_map,
    )

    assert rendered == "第一条事实[1]，例如第二条事实[1]。"
    assert [row["rendered_text"] for row in ledger] == [
        "第一条事实[1]",
        "，例如第二条事实[1]",
    ]


def test_eplus_visible_text_and_ledger_share_multi_source_formatting() -> None:
    evidence = {
        "ev_001": _evidence("ev_001", chunk_id="chunk_1", quote="第一来源。"),
        "ev_002": _evidence("ev_002", chunk_id="chunk_2", quote="第二来源。"),
    }
    citation_map = {
        "https://chunk_1.example.com/doc": 1,
        "https://chunk_2.example.com/doc": 2,
    }

    rendered, ledger = evidence_native._render_citations_with_spans(
        "组合声明[[ev_001]] [[ev_002]]。",
        evidence,
        citation_map,
    )

    assert rendered == "组合声明[1] [2]。"
    assert ledger[0]["rendered_text"] == "组合声明[1] [2]"
    assert ledger[0]["rendered_text"] in rendered


def test_eplus_collapses_adjacent_duplicate_refs_for_one_cited_span() -> None:
    evidence = {
        "ev_001": _evidence("ev_001", chunk_id="chunk_1", quote="第一条支撑。"),
        "ev_002": _evidence("ev_002", chunk_id="chunk_1", quote="第二条支撑。"),
    }
    citation_map = {"https://chunk_1.example.com/doc": 1}

    rendered, ledger = evidence_native._render_citations_with_spans(
        "组合声明[[ev_001]][[ev_002]]。",
        evidence,
        citation_map,
    )

    assert rendered == "组合声明[1]。"
    assert ledger[0]["evidence_ids"] == ["ev_001", "ev_002"]
    assert ledger[0]["rendered_text"] == "组合声明[1]"
    assert ledger[0]["rendered_text"] in rendered


def test_editorial_compact_claim_markers_are_normalized_and_expanded() -> None:
    expanded, invalid = evidence_native._expand_claim_markers(
        "组合结论[[claim_a1][claim_b2]]。",
        {
            "claim_a1": {"evidence_ids": ["ev_001"]},
            "claim_b2": {"evidence_ids": ["ev_002"]},
        },
    )

    assert expanded == "组合结论[[ev_001]][[ev_002]]。"
    assert invalid == []


def test_editorial_residual_claim_id_is_rejected() -> None:
    expanded, invalid = evidence_native._expand_claim_markers(
        "不完整标记 [claim_a1]。",
        {"claim_a1": {"evidence_ids": ["ev_001"]}},
    )

    assert expanded == "不完整标记 [claim_a1]。"
    assert invalid == ["claim_a1"]


def test_final_citation_gate_rejects_internal_claim_id() -> None:
    validation = validate_report_citations(
        documents=[],
        report_document={"content": "用户可见正文残留 [claim_a1]。"},
        evidence_map=[{"id": "ev_001", "source_url": "https://example.com/source"}],
    )

    assert validation["status"] == "failed"
    assert "report exposes internal claim ids" in validation["reason"]


def test_claim_bank_only_admits_supported_atomic_and_qualified_inferences() -> None:
    documents = [
        {
            "section_id": "sec_01",
            "title": "可靠性",
            "content": "",
            "claim_ledger": [
                {
                    "text": "系统支持恢复",
                    "cited_span": "系统支持恢复",
                    "claim_context": "",
                    "evidence_ids": ["ev_001"],
                    "direct_evidence_ids": ["ev_001"],
                    "inference_evidence_ids": [],
                    "atomic": True,
                    "claim_type": "fact",
                    "support_scope": "domain_specific",
                },
                {
                    "text": "系统一定最可靠",
                    "cited_span": "系统一定最可靠",
                    "evidence_ids": ["ev_002"],
                    "atomic": True,
                    "claim_type": "inference",
                    "support_scope": "cross_domain",
                },
                {
                    "text": "这可能减少恢复时间",
                    "cited_span": "这可能减少恢复时间",
                    "evidence_ids": ["ev_003"],
                    "atomic": True,
                    "claim_type": "inference",
                    "support_scope": "cross_domain",
                },
            ],
        }
    ]
    entailment = {
        "judgments": [
            {"claim_index": 0, "status": "entailed", "reason": "direct"},
            {"claim_index": 1, "status": "reasonable_inference", "reason": "synthesis"},
            {"claim_index": 2, "status": "reasonable_inference", "reason": "synthesis"},
        ]
    }

    bank, excluded = build_editorial_claim_bank(documents, entailment)

    assert [item["cited_span"] for item in bank] == ["系统支持恢复", "这可能减少恢复时间"]
    assert bank[0]["claim_id"].startswith("claim_")
    assert bank[0]["source_section_hash"]
    assert excluded[0]["admission_reason"] == "reasonable_inference_missing_explicit_qualification"


def test_checkpoint_refuses_tampered_output_and_records_reason(tmp_path: Path) -> None:
    store = EvidenceNativeCheckpointStore(tmp_path, contract={"model": "mock"})
    store.save("section_draft", "sec_01", {"input": 1}, {"document": {"content": "ok"}})
    path = tmp_path / "section_draft" / "sec_01.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["document"]["content"] = "tampered"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    assert store.load("section_draft", "sec_01", {"input": 1}) is None
    assert store.events[-1]["reason"] == "hash_or_contract_mismatch"


def test_evidence_native_report_is_finding_isolated_and_resumes_without_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = "系统会从最近的合法检查点恢复执行。"
    chunks = [_chunk("chunk_1", quote)]
    evidence = [
        _evidence(
            "ev_001",
            chunk_id="chunk_1",
            quote=quote,
            criterion_ids=["crit_hard"],
            subject="A",
            axis="可靠性",
            matrix_cell_id="cell_a_reliability",
        )
    ]
    blueprint = build_evidence_native_blueprint(
        normalized_request={"user_request": "检查恢复能力", "poison": "POISON_FINDING_TEXT"},
        research_plan=_plan(),
        question_states=_states(),
        evidence_map=evidence,
        selected_chunks=chunks,
    )
    monkeypatch.setattr(evidence_native, "_section_is_underdeveloped", lambda *_args, **_kwargs: False)
    writer_payloads: list[dict] = []
    validation_payloads: list[dict] = []

    def writer_model(payload: dict) -> dict:
        writer_payloads.append(copy.deepcopy(payload))
        if "section_claim_bank" in payload:
            claim_id = payload["section_claim_bank"][0]["claim_id"]
            return {
                "report_title": "检查点恢复研究",
                "executive_summary": f"系统支持检查点恢复[[{claim_id}]]。",
                "conclusion": f"现有材料表明系统可恢复[[{claim_id}]]。",
            }
        return {"content": "系统会从最近的合法检查点恢复执行[[ev_001]]，这有助于定位失败。"}

    def validation_model(payload: dict, timeout_override: float | None = None) -> dict:
        validation_payloads.append(copy.deepcopy(payload))
        return {
            "judgments": [
                {
                    "claim_index": item["claim_index"],
                    "status": "entailed",
                    "claim_scope": "domain_fact",
                    "reason": "引用与声明一致",
                }
                for item in payload.get("claims") or []
            ]
        }

    first_store = EvidenceNativeCheckpointStore(tmp_path / "checkpoints", contract={"model": "mock"})
    first = write_evidence_native_report(
        user_request="检查恢复能力",
        blueprint=blueprint,
        question_states=_states(),
        evidence_map=evidence,
        selected_chunks=chunks,
        writer_model=writer_model,
        prevalidation_model=validation_model,
        checkpoint_store=first_store,
    )

    serialized_payloads = json.dumps(writer_payloads + validation_payloads, ensure_ascii=False)
    assert "POISON_FINDING_TEXT" not in serialized_payloads
    assert first["editorial_claim_bank"]
    assert first["report_document"]["claim_ledger"]
    assert first["documents"][0]["synthesis_diagnostics"]["selected_evidence_ids"] == ["ev_001"]

    def unexpected_model(_payload: dict, **_kwargs: object) -> dict:
        pytest.fail("valid E+ checkpoints should suppress repeated Writer/Validator calls")

    second_store = EvidenceNativeCheckpointStore(tmp_path / "checkpoints", contract={"model": "mock"})
    second = write_evidence_native_report(
        user_request="检查恢复能力",
        blueprint=blueprint,
        question_states=_states(),
        evidence_map=evidence,
        selected_chunks=chunks,
        writer_model=unexpected_model,
        prevalidation_model=unexpected_model,
        checkpoint_store=second_store,
    )

    assert second["report_document"] == first["report_document"]
    assert any(event["outcome"] == "resumed" for event in second_store.events)


def test_prevalidation_checkpoint_isolated_by_entailment_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = {
        "section_id": "sec_01",
        "claim_ledger": [{
            "section_id": "sec_01",
            "text": "系统支持检查点恢复。",
            "claim_type": "fact",
            "evidence_ids": ["ev_001"],
        }],
    }
    evidence = [{"id": "ev_001", "claim": "系统支持检查点恢复。"}]
    profiles: list[str] = []

    def fake_validate(**kwargs: object) -> dict:
        profile = str(kwargs.get("payload_profile") or "")
        profiles.append(profile)
        return {
            "status": "passed",
            "payload_profile": profile,
            "claim_count": 1,
            "entailed_count": 1,
            "inference_count": 0,
            "unsupported_count": 0,
            "label_mismatch_count": 0,
            "non_atomic_count": 0,
            "model_reviewed_count": 1,
            "semantic_review_unavailable_count": 0,
            "deterministic_fallback_count": 0,
            "judgments": [{
                "claim_index": 0,
                "section_id": "sec_01",
                "status": "entailed",
                "mode": "model",
            }],
        }

    monkeypatch.setattr(evidence_native, "validate_claim_entailment", fake_validate)
    store = EvidenceNativeCheckpointStore(tmp_path / "checkpoints", contract={"model": "mock"})

    for profile in (
        ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
        ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
        ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ):
        evidence_native._prevalidate_sections(
            documents=[document],
            evidence_map=evidence,
            model=lambda _payload: {},
            parallelism=1,
            checkpoint_store=store,
            entailment_payload_profile=profile,
        )

    assert profiles == [
        ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
        ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ]
    assert any(event["outcome"] == "resumed" for event in store.events)


def test_quality_uses_eplus_support_units_instead_of_empty_findings() -> None:
    report = {
        "content": "# 标题\n\n## 一\n事实[1]\n\n## 二\n分析[1]\n\n## 三\n限制[1]",
        "delivery_status": "complete_report",
        "completeness": {"status": "complete_report"},
    }
    quality = evaluate_report_quality(
        report_document=report,
        question_states=[{"status": "sufficient"}],
        findings=[],
        rejected_evidence=[],
        support_units=[{"claim_id": "claim_1", "section_id": "sec_01"}],
        support_unit_kind="validated_section_claim",
    )

    assert quality["scores"]["synthesis"] == 4
    assert quality["scores"]["evidence_quality"] == 5


def test_eplus_writer_prompt_includes_compound_claim_and_uncited_analysis_few_shots() -> None:
    payload = evidence_native._section_payload(
        user_request="比较两个系统",
        section={"section_id": "sec_01", "title": "比较"},
        blueprint={"sections": []},
        packet={"allowed_evidence_ids": ["ev_001"], "claim_clusters": []},
        target_length=600,
    )

    examples = {item["name"]: item for item in payload["few_shot_examples"]}
    rules = "\n".join(payload["rules"])
    assert "split_partially_supported_compound_claim" in examples
    assert "keep_uncited_analysis_outside_cited_span" in examples
    assert "禁止复制" in payload["few_shot_policy"]
    assert "每个陈述性子句都带自己的" not in rules
    assert "未引用分析不强制挂引用" in rules
    assert "不得把引用放在分析之后" in rules


def test_writing_architecture_defaults_to_eplus_and_keeps_explicit_finding_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_WRITING_ARCHITECTURE", raising=False)
    assert _research_writing_architecture({}) == ARCHITECTURE
    assert _research_writing_architecture({"writing_architecture": "eplus"}) == ARCHITECTURE
    assert _research_writing_architecture({"writing_architecture": "finding"}) == "finding"
    with pytest.raises(ValueError, match="unsupported research writing architecture"):
        _research_writing_architecture({"writing_architecture": "unknown"})


def test_question_loop_service_runs_eplus_without_finding_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.services import research_service

    plan = {
        "strategy": "standard",
        "research_type": "implementation_design",
        "sections": [
            {"id": "sec_00", "title": "00_总览", "questions": [{"text": "总体结论"}]},
            {
                "id": "sec_01",
                "title": "01_恢复与风险",
                "intent": "分析恢复能力与限制",
                "questions": [{"text": "如何恢复并控制风险？", "answer_contract_level": "hard"}],
                "success_criteria": ["覆盖恢复与风险"],
                "evidence_budget": {"min_evidence": 4, "min_sources": 3},
            },
        ],
    }
    monkeypatch.setattr(research_service, "RESEARCH_ARTIFACT_DIR", tmp_path / "research_artifacts")
    monkeypatch.setattr(research_service, "_plan_research", lambda **_kwargs: plan)
    monkeypatch.setattr(
        research_service,
        "_research_reconnaissance",
        lambda _request: {
            "summary": "检查点恢复",
            "entity_type": "concept",
            "ambiguities": [],
            "disambiguation_terms": [],
        },
    )
    monkeypatch.setattr(
        research_service,
        "_build_query_plan",
        lambda **_kwargs: {
            "strategy": "standard",
            "depth": "normal",
            "queries": [
                {
                    "section_id": "sec_01",
                    "question_id": "sec_01_q1",
                    "question": "如何恢复并控制风险？",
                    "query": "checkpoint recovery risk official",
                }
            ],
        },
    )

    def retrieve(**kwargs: object) -> dict:
        state = kwargs["state"]
        rows = []
        chunks = []
        materials = []
        sources = []
        for index, domain in enumerate(
            ("docs.example.com", "case.example.org", "paper.example.net", "ops.example.io"),
            start=1,
        ):
            text = f"检查点恢复证据 {index} 表明系统保留失败边界和恢复记录。"
            url = f"https://{domain}/{state.question_id}/{index}"
            chunk_id = f"chunk_{index}"
            chunks.append(
                {"id": chunk_id, "text": text, "source_url": url, "source_title": domain}
            )
            rows.append(
                {
                    "claim": text,
                    "evidence_text": text,
                    "chunk_id": chunk_id,
                    "source_url": url,
                    "source_title": domain,
                    "stance": "limitation" if index == 4 else "support",
                }
            )
            materials.append({"id": f"material_{index}", "title": domain, "url": url})
            sources.append({"title": domain, "url": url})
        return {
            "sources": sources,
            "materials": materials,
            "selected_chunks": chunks,
            "evidence_map": rows,
        }

    monkeypatch.setattr(research_service, "_retrieve_question_round", retrieve)
    monkeypatch.setattr(research_service, "_research_json_model", lambda _payload: {})

    def forbidden_finding_model(_payload: dict) -> dict:
        pytest.fail("E+ must not invoke Finding synthesis or admission")

    monkeypatch.setattr(research_service, "_research_finding_json_model", forbidden_finding_model)
    monkeypatch.setattr(evidence_native, "_section_is_underdeveloped", lambda *_args, **_kwargs: False)

    def writer_model(payload: dict) -> dict:
        if "section_claim_bank" in payload:
            claim_ids = [item["claim_id"] for item in payload["section_claim_bank"]]
            markers = "".join(f"[[{item}]]" for item in claim_ids[:2])
            return {
                "report_title": "检查点恢复与风险",
                "executive_summary": f"现有材料表明恢复过程可审计{markers}。",
                "conclusion": f"现有材料提示应保留失败边界{markers}。",
            }
        selected = list(payload.get("allowed_evidence_ids") or [])
        return {
            "content": "\n\n".join(
                f"现有证据表明恢复记录覆盖第 {index} 类运行边界[[{evidence_id}]]。"
                for index, evidence_id in enumerate(selected, start=1)
            )
        }

    monkeypatch.setattr(research_service, "_research_writer_json_model", writer_model)

    entailment_prompt_versions: list[str] = []

    def validation(payload: dict, timeout_override: float | None = None) -> dict:
        if payload.get("cases"):
            entailment_prompt_versions.append(str(payload.get("prompt_version") or ""))
            return {
                "judgments": [
                    {
                        "case_id": item["case_id"],
                        "status": "entailed",
                        "claim_scope": "domain_fact",
                        "reason": "mock Evidence 支持声明",
                    }
                    for item in payload["cases"]
                ]
            }
        return {}

    monkeypatch.setattr(
        research_service,
        "_validation_model",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(research_service, "_write_research_review_artifacts", lambda **_kwargs: [])

    job = research_service.job_service.create(job_type="research-eplus-test", input_data={}, target=None)
    research_service.job_service.set_steps(job["job_id"], research_service._initial_steps())
    result = research_service._run_question_loop_research(
        job["job_id"],
        {
            "user_request": "调研检查点恢复与风险",
            "source_mode": "web",
            "depth": "normal",
            "research_strategy": "standard",
            "writing_architecture": "eplus",
        },
    )

    assert result["writing_architecture"] == ARCHITECTURE
    assert result["findings"] == []
    assert result["capabilities"]["section_revision"] is False
    assert result["capabilities"]["targeted_followup"] is False
    assert result["editorial_claim_bank"]
    assert result["section_prevalidation"]["claim_count"] > 0
    assert result["validation"]["entailment"]["claim_count"] > 0
    assert entailment_prompt_versions
    assert set(entailment_prompt_versions) == {"claim-entailment/atomic-boundary-isolation/v2"}
    assert result["validation"]["entailment"]["unsupported_marker_policy"] == "advisory_only"
    assert "available_findings" not in json.dumps(result["pipeline_diagnostics"], ensure_ascii=False)
    checkpoint_root = (
        research_service.RESEARCH_ARTIFACT_DIR
        / job["job_id"]
        / "checkpoints"
        / "evidence_native"
    )
    assert (checkpoint_root / "section_draft").is_dir()
    assert (checkpoint_root / "section_prevalidation").is_dir()
    assert (checkpoint_root / "editorial" / "global.json").is_file()


def test_eplus_unimplemented_operations_fail_before_mutating_job() -> None:
    from backend.services import research_service

    job = research_service.job_service.create(job_type="research", input_data={}, target=None)
    initial_result = {
        "status": "waiting_approval",
        "pipeline_mode": "question_loop",
        "writing_architecture": ARCHITECTURE,
        "capabilities": {
            "section_revision": False,
            "targeted_followup": False,
            "approval": True,
            "archive": True,
        },
        "documents": [],
    }
    research_service.job_service.update_result(
        job["job_id"],
        status="waiting_approval",
        result=initial_result,
        progress_message="waiting",
    )

    with pytest.raises(ValueError, match="targeted follow-up is not enabled"):
        research_service.followup_research(job["job_id"], {"query": "more evidence"})
    with pytest.raises(ValueError, match="section revision is not enabled"):
        research_service.revise_research_sections(
            job["job_id"], {"section_ids": ["sec_01"], "instruction": "修改"}
        )

    current = research_service.job_service.get(job["job_id"])
    assert current["result"] == initial_result
    artifacts = research_service.research_artifacts(job["job_id"])
    assert artifacts["writing_architecture"] == ARCHITECTURE
    assert artifacts["capabilities"]["section_revision"] is False
