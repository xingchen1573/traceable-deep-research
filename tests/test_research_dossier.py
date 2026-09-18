from __future__ import annotations

from backend.research.dossier import attach_section_dossiers
from backend.research.quality import classify_unused_dossier_evidence
from backend.research.writer import (
    _accept_revision,
    _call_writer_model,
    _reintroduces_forbidden_text,
    _render_internal_citations,
    _sanitize_leaked_evidence_ids,
    revise_report_sections,
    write_report,
)


def _research_inputs():
    evidence = [
        {
            "id": "ev_001",
            "section_id": "sec_01",
            "question_id": "sec_01_q1",
            "atomic_claim": "监督者负责协调专业角色。",
            "cleaned_quote": "监督者负责协调专业角色。",
            "source_url": "https://docs.example.org/architecture",
            "source_title": "Architecture docs",
            "source_type": "official_docs",
            "scope": "general",
            "claim_type": "direct_fact",
            "publisher_relation": "first_party",
            "subject": "监督者机制",
            "subject_relation": "target_component",
            "criterion_ids": ["criterion_01"],
            "declared_criterion_ids": ["criterion_01"],
            "criterion_binding_status": "verified_target_component",
        },
        {
            "id": "ev_002",
            "section_id": "sec_01",
            "question_id": "sec_01_q1",
            "atomic_claim": "检查点允许失败任务从已保存状态恢复。",
            "cleaned_quote": "检查点允许失败任务从已保存状态恢复。",
            "source_url": "https://case.example.org/recovery",
            "source_title": "Recovery case",
            "source_type": "case_study",
            "scope": "cross_domain",
            "claim_type": "direct_fact",
        },
    ]
    findings = [{
        "finding_id": "finding_001",
        "section_id": "sec_01",
        "question_id": "sec_01_q1",
        "finding": "监督者可以协调专业角色。",
        "finding_type": "inference",
        "evidence_ids": ["ev_001"],
        "limitations": [],
    }]
    blueprint = {
        "thesis": "协调与恢复共同构成可靠执行机制。",
        "sections": [{
            "section_id": "sec_01",
            "title": "01_架构与恢复",
            "purpose": "解释协调和失败恢复机制",
            "research_questions": [{
                "question_id": "sec_01_q1",
                "question": "系统如何协调角色并从失败中恢复？",
            }],
            "finding_ids": ["finding_001"],
            "known_gaps": ["缺少目标领域的公开部署数据"],
            "visual_specs": [],
        }],
        "visual_specs": [],
    }
    chunks = [{
        "id": "chunk_002",
        "section_id": "sec_01",
        "question_id": "sec_01_q1",
        "source_url": "https://case.example.org/recovery",
        "source_title": "Recovery case",
        "hybrid_score": 0.91,
        "text": "该案例使用持久化检查点保存执行状态，并在工具调用失败后恢复未完成任务。",
    }]
    materials = [{
        "url": "https://case.example.org/recovery",
        "title": "Recovery case",
        "source_type": "case_study",
    }]
    return blueprint, findings, evidence, chunks, materials


def _add_recovery_finding(blueprint, findings):
    findings.append({
        "finding_id": "finding_002",
        "section_id": "sec_01",
        "question_id": "sec_01_q1",
        "finding": "检查点允许失败任务从已保存状态恢复。",
        "finding_type": "fact",
        "evidence_ids": ["ev_002"],
        "limitations": [],
    })
    blueprint["sections"][0]["finding_ids"].append("finding_002")


def test_section_dossier_preserves_admitted_evidence_omitted_from_findings():
    blueprint, findings, evidence, chunks, materials = _research_inputs()

    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )

    dossier = attached["sections"][0]["dossier"]
    assert {item["id"] for item in dossier["evidence"]} == {"ev_001", "ev_002"}
    assert dossier["unused_evidence_ids"] == ["ev_002"]
    assert dossier["relevant_chunks"][0]["excerpt"].startswith("该案例使用持久化检查点")
    recovery_source = next(item for item in dossier["source_notes"] if "recovery" in item["source_url"])
    assert recovery_source["evidence_role"] == "transferable_engineering_evidence"
    assert recovery_source["evidence_ids"] == ["ev_002"]
    assert dossier["known_gaps"] == ["缺少目标领域的公开部署数据"]
    assert attached["dossier_diagnostics"][0]["context_chars"] > 0
    first = next(item for item in dossier["evidence"] if item["id"] == "ev_001")
    assert first["subject"] == "监督者机制"
    assert first["criterion_ids"] == ["criterion_01"]
    assert first["publisher_relation"] == "first_party"


def test_writer_does_not_bypass_findings_with_dossier_only_evidence():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    section_payloads = []

    def model(payload):
        if payload["task"].startswith("Write one analytical"):
            section_payloads.append(payload)
            return {"content": "监督者机制可用于协调专业角色 [[ev_001]]。"}
        return {
            "executive_summary": "协调与恢复需要分层设计 [[ev_001]]。",
        "conclusion": "当前只确认监督者协调机制 [[ev_001]]。",
            "editor_notes": [],
        }

    result = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        model=model,
    )

    assert section_payloads
    payload = section_payloads[0]
    assert "unused_evidence_ids" not in payload["section_dossier"]
    assert {item["id"] for item in payload["evidence"]} == {"ev_001"}
    first_evidence = next(item for item in payload["evidence"] if item["id"] == "ev_001")
    assert first_evidence["subject"] == "监督者机制"
    assert first_evidence["criterion_ids"] == ["criterion_01"]
    assert any("Evidence sufficiency overrides length" in rule for rule in payload["rules"])
    assert any("分别核对主体" in rule for rule in payload["rules"])
    rules_blob = "\n".join(payload["rules"])
    assert "每个陈述性子句都要带自己的" not in rules_blob
    assert "未引用分析不强制挂引用" in rules_blob
    assert "不要为了满足引用数量" in rules_blob
    document = result["documents"][0]
    assert {item for row in document["claim_ledger"] for item in row["evidence_ids"]} == {"ev_001"}
    assert document["synthesis_diagnostics"]["evidence_utilization_ratio"] == 0.5
    utilization = result["diagnostics"]["material_utilization"]
    assert utilization["input_evidence_count"] == 2
    assert utilization["used_evidence_count"] == 1
    assert utilization["evidence_utilization_ratio"] == 0.5


def test_material_utilization_excludes_overview_reuse_from_aggregate_ratio():
    """Regression test for docs/evaluations/deep-research-section-dossier-e2e-20260715.md
    section 3: the overview section (00_总览) has no dossier of its own -- _write_section reuses
    the global first-6 findings for it when it has no finding_ids -- so its
    dossier_evidence_count is always 0 while its used_evidence_count reflects those reused
    findings' evidence. Folding it into `_material_utilization`'s aggregate sum inflates the
    ratio's numerator without a matching denominator contribution, so a body section that only
    used half its own dossier evidence could still read as a misleading 1.0+ overall.
    """
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    evidence = [
        *evidence,
        {
            "id": "ev_003",
            "section_id": "sec_02",
            "question_id": "sec_02_q1",
            "atomic_claim": "扩展场景下需要独立的重试策略。",
            "cleaned_quote": "扩展场景下需要独立的重试策略。",
            "source_url": "https://docs.example.org/retry",
            "source_title": "Retry docs",
            "source_type": "official_docs",
            "scope": "general",
            "claim_type": "direct_fact",
        },
        {
            "id": "ev_004",
            "section_id": "sec_02",
            "question_id": "sec_02_q1",
            "atomic_claim": "重试上限应结合下游系统的限流策略配置。",
            "cleaned_quote": "重试上限应结合下游系统的限流策略配置。",
            "source_url": "https://docs.example.org/ratelimit",
            "source_title": "Rate limit docs",
            "source_type": "official_docs",
            "scope": "general",
            "claim_type": "direct_fact",
        },
    ]
    findings = [
        *findings,
        {
            "finding_id": "finding_002",
            "section_id": "sec_02",
            "question_id": "sec_02_q1",
            "finding": "扩展场景需要独立的重试策略。",
            "finding_type": "fact",
            "evidence_ids": ["ev_003"],
            "limitations": [],
        },
    ]
    blueprint["sections"] = [
        {
            "section_id": "sec_00",
            "title": "00_总览",
            "purpose": "总览全文结论",
            "research_questions": [],
            "finding_ids": [],
            "known_gaps": [],
            "visual_specs": [],
        },
        blueprint["sections"][0],
        {
            "section_id": "sec_02",
            "title": "02_扩展场景",
            "purpose": "说明扩展场景下的重试策略",
            "research_questions": [{"question_id": "sec_02_q1", "question": "扩展场景如何配置重试？"}],
            "finding_ids": ["finding_002"],
            "known_gaps": [],
            "visual_specs": [],
        },
    ]

    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )

    def model(payload):
        if not payload["task"].startswith("Write one analytical"):
            return {
                "executive_summary": "总体结论 [[ev_001]]。",
                "conclusion": "后续需持续验证 [[ev_003]]。",
                "editor_notes": [],
            }
        title = str(payload["section_brief"].get("title") or "")
        if title.startswith("00_总览"):
            return {"content": "全文结论汇总：协调机制已验证 [[ev_001]]，扩展场景重试策略已确认 [[ev_003]]。"}
        if title.startswith("01_"):
            return {
                "content": "监督者机制可用于协调专业角色 [[ev_001]]。"
            }
        # sec_02: deliberately leaves ev_004 uncited, so this section's true utilization is 1/2.
        return {"content": "扩展场景下需要独立的重试策略 [[ev_003]]。"}

    result = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        model=model,
    )

    utilization = result["diagnostics"]["material_utilization"]
    overview_row = next(row for row in utilization["sections"] if "总览" in str(row["title"]))
    body_rows = [row for row in utilization["sections"] if "总览" not in str(row["title"])]

    assert overview_row["dossier_evidence_count"] == 0
    assert overview_row["used_evidence_count"] == 2
    assert len(body_rows) == 2
    assert utilization["input_evidence_count"] == sum(row["dossier_evidence_count"] for row in body_rows) == 4
    assert utilization["used_evidence_count"] == sum(row["used_evidence_count"] for row in body_rows) == 2
    assert utilization["evidence_utilization_ratio"] == 0.5


def test_writer_does_not_run_for_evidence_only_dossier_when_findings_are_empty():
    blueprint, _, evidence, chunks, materials = _research_inputs()
    blueprint["sections"][0]["finding_ids"] = []
    attached = attach_section_dossiers(
        blueprint,
        findings=[],
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    section_calls = []

    def model(payload):
        if payload["task"].startswith("Write one analytical"):
            section_calls.append(payload)
            return {"content": "公开资料记录了监督者协调机制 [[ev_001]]，检查点可恢复失败任务 [[ev_002]]。"}
        return {
            "executive_summary": "公开资料提供了协调与恢复证据 [[ev_001]]。",
            "conclusion": "检查点机制仍需在目标场景验证 [[ev_002]]。",
        }

    result = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=[],
        evidence_map=evidence,
        model=model,
    )

    assert section_calls == []
    assert result["documents"][0]["content"] == ""
    assert result["documents"][0]["synthesis_diagnostics"]["mode"] == "model_failed"
    assert result["documents"][0]["synthesis_diagnostics"]["fallback_reason"] == "findings_missing_for_available_evidence"


def test_writer_renders_explicit_evidence_gap_without_calling_section_model():
    blueprint, _, evidence, chunks, materials = _research_inputs()
    blueprint["sections"][0].update({
        "finding_ids": [],
        "evidence_ids": [],
        "reportability": {
            "status": "empty",
            "finding_count": 0,
            "evidence_count": 0,
            "source_count": 0,
            "core_section": True,
            "reason": "section has no body-worthy finding",
        },
    })
    blueprint["sections"][0]["research_questions"][0]["criteria"] = [{
        "criterion_id": "criterion_01",
        "description": "确定目标系统的恢复能力",
        "level": "hard",
        "status": "uncovered",
    }]
    blueprint["completeness"] = {
        "status": "partial_report",
        "required_section_count": 1,
        "ready_section_count": 0,
        "limited_section_ids": [],
        "core_empty_section_ids": ["sec_01"],
        "supplemental_empty_section_ids": [],
    }
    attached = attach_section_dossiers(
        blueprint,
        findings=[],
        evidence_map=[],
        selected_chunks=chunks,
        materials=materials,
    )
    section_calls = []

    def model(payload):
        if payload["task"].startswith("Write one analytical"):
            section_calls.append(payload)
            raise AssertionError("an evidence-gap section must not call the section writer")
        return {"executive_summary": "阶段性摘要。", "conclusion": "阶段性结论。"}

    result = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=[],
        evidence_map=[],
        model=model,
    )

    document = result["documents"][0]
    diagnostics = document["synthesis_diagnostics"]
    assert section_calls == []
    assert diagnostics["mode"] == "evidence_gap"
    assert diagnostics["fallback_reason"] is None
    assert diagnostics["model_attempts"] == 0
    assert diagnostics["gap_question_ids"] == ["sec_01_q1"]
    assert diagnostics["gap_criterion_ids"] == ["criterion_01"]
    assert "不据此推断相反结论" in document["content"]
    assert "确定目标系统的恢复能力" in document["content"]
    assert document["evidence_table"] == []
    assert document["claim_ledger"] == []
    assert result["report_status"] == "partial_report"
    assert "## 01 架构与恢复" in result["report_document"]["content"]
    assert "本轮仅形成证据缺口说明的章节：01 架构与恢复" in result["report_document"]["content"]


def test_writer_does_not_hide_missing_findings_when_evidence_exists():
    blueprint, _, evidence, chunks, materials = _research_inputs()
    blueprint["sections"][0].update({
        "finding_ids": [],
        "evidence_ids": ["ev_001"],
        "reportability": {
            "status": "empty",
            "finding_count": 0,
            "evidence_count": 0,
            "source_count": 0,
            "core_section": True,
            "reason": "section has no body-worthy finding",
        },
    })
    attached = attach_section_dossiers(
        blueprint,
        findings=[],
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )

    result = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=[],
        evidence_map=evidence,
        model=lambda _payload: {"executive_summary": "摘要。", "conclusion": "结论。"},
    )

    diagnostics = result["documents"][0]["synthesis_diagnostics"]
    assert diagnostics["mode"] == "model_failed"
    assert diagnostics["fallback_reason"] == "findings_missing_for_available_evidence"
    assert result["documents"][0]["content"] == ""


def test_write_section_reports_model_writer_mode_when_model_succeeds():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    _add_recovery_finding(blueprint, findings)
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )

    def model(payload):
        if payload["task"].startswith("Write one analytical"):
            return {"content": "监督者机制可用于协调专业角色 [[ev_001]]，检查点机制支持恢复 [[ev_002]]。"}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。", "editor_notes": []}

    result = write_report(user_request="调研可靠执行机制", blueprint=attached, findings=findings, evidence_map=evidence, model=model)

    diagnostics = result["documents"][0]["synthesis_diagnostics"]
    assert diagnostics["mode"] == "model_writer"
    assert diagnostics["fallback_reason"] is None


def test_write_section_reports_model_failed_mode_and_reason_without_fallback_content():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    attached["sections"][0]["reportability"] = {"core_section": True, "status": "ready"}
    attached["completeness"] = {
        "status": "complete_report",
        "required_section_count": 1,
        "ready_section_count": 1,
        "limited_section_ids": [],
        "core_empty_section_ids": [],
        "supplemental_empty_section_ids": [],
    }

    def failing_model(payload):
        if payload["task"].startswith("Write one analytical"):
            return {"_failure_reason": "timeout"}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。", "editor_notes": []}

    result = write_report(user_request="调研可靠执行机制", blueprint=attached, findings=findings, evidence_map=evidence, model=failing_model)

    diagnostics = result["documents"][0]["synthesis_diagnostics"]
    assert diagnostics["mode"] == "model_failed"
    assert diagnostics["fallback_reason"] == "timeout"
    # A model that fails every attempt should be retried once, not zero times.
    assert diagnostics["model_attempts"] >= 1
    assert result["documents"][0]["content"] == ""
    assert "## 01 架构与恢复" not in result["report_document"]["content"]
    assert "本轮未形成可交付正文的章节：01 架构与恢复" in result["report_document"]["content"]
    assert result["report_status"] == "partial_report"


def test_write_section_repairs_unknown_marker_once_without_content_fallback():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    _add_recovery_finding(blueprint, findings)
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    section_payloads = []

    def model(payload):
        if payload["task"].startswith("Write one analytical"):
            section_payloads.append(payload)
            return {"content": "检查点支持恢复 [[ev_999]]。"}
        if payload["task"].startswith("Repair one analytical"):
            section_payloads.append(payload)
            assert payload["invalid_markers"] == ["ev_999"]
            assert "[[ev_999]]" in payload["invalid_draft"]
            return {"content": "检查点支持恢复 [[ev_002]]。"}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。"}

    result = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        model=model,
    )

    document = result["documents"][0]
    assert len(section_payloads) >= 2
    assert "检查点支持恢复 [2]" in document["content"]
    assert document["synthesis_diagnostics"]["mode"] == "model_writer"
    assert document["synthesis_diagnostics"]["model_attempts"] >= 2


def test_accept_revision_rejects_when_new_version_is_worse():
    old = {
        "content": "旧稿包含充分的分析内容并引用两条证据。" * 3,
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    fewer_evidence = {**old, "evidence_table": [{"id": "ev_001"}]}
    accepted, reason = _accept_revision(old, fewer_evidence)
    assert accepted is False
    assert reason == "fewer_evidence_used"

    shorter = {**old, "content": "太短"}
    accepted, reason = _accept_revision(old, shorter)
    assert accepted is False
    assert reason == "content_shrank"

    fell_back = {**old, "synthesis_diagnostics": {"mode": "fallback_template"}}
    accepted, reason = _accept_revision(old, fell_back)
    assert accepted is False
    assert reason == "revision_fell_back_to_template"

    genuinely_better = {**old, "content": old["content"] + "补充了更多分析。"}
    accepted, reason = _accept_revision(old, genuinely_better)
    assert accepted is True
    assert reason == ""


def test_accept_revision_allows_shrink_when_manual_instruction_asks_for_it():
    """A user who explicitly asks "把这节写得更简洁一些" should not have that request rejected
    purely for succeeding at being shorter — see research-manual-section-revision-feature memory
    for the real E2E run that surfaced this gap."""
    old = {
        "content": "旧稿包含充分的分析内容并引用两条证据。" * 5,
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    shorter_same_evidence = {**old, "content": "精简后的稿子。" * 2}
    manual_shrink_issue = [{"section_id": "sec_01", "type": "manual_revision", "instruction": "把这节写得更简洁一些，去掉重复的表述。"}]

    accepted, reason = _accept_revision(old, shorter_same_evidence, manual_shrink_issue)
    assert accepted is True
    assert reason == ""

    # Without the bypass (no issues, or issues that don't ask for shrinkage), it's still rejected.
    accepted, reason = _accept_revision(old, shorter_same_evidence)
    assert accepted is False
    assert reason == "content_shrank"

    other_issue = [{"section_id": "sec_01", "type": "manual_revision", "instruction": "让这几节的结论互相呼应一下。"}]
    accepted, reason = _accept_revision(old, shorter_same_evidence, other_issue)
    assert accepted is False
    assert reason == "content_shrank"

    # An automated quality-gate issue type never gets the shrink-keyword bypass, even if its
    # instruction text happens to contain a shrink keyword — that bypass is gated on type ==
    # manual_revision, not text content. Use `readability` (a non-grounding automated type) so
    # neither the shrink-keyword bypass nor the grounding-correction bypass applies, leaving
    # content_shrank to fire. (entailment_label_mismatch can't be used here anymore: it is now a
    # grounding-correction type, so it legitimately bypasses content_shrank too — see
    # test_accept_revision_allows_label_mismatch_correction_to_drop_overreaching_claim.)
    automated_issue = [{"section_id": "sec_01", "type": "readability", "instruction": "请精简这句推断表达"}]
    accepted, reason = _accept_revision(old, shorter_same_evidence, automated_issue)
    assert accepted is False
    assert reason == "content_shrank"

    # A shrink request still can't smuggle in fewer citations.
    fewer_evidence_and_shorter = {**shorter_same_evidence, "evidence_table": [{"id": "ev_001"}]}
    accepted, reason = _accept_revision(old, fewer_evidence_and_shorter, manual_shrink_issue)
    assert accepted is False
    assert reason == "fewer_evidence_used"


def test_accept_revision_allows_grounding_correction_to_remove_text_and_evidence():
    old = {
        "content": "有依据内容。无依据扩写。" * 10,
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    corrected = {
        "content": "有依据内容。",
        "evidence_table": [{"id": "ev_001"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    issue = [{"section_id": "sec_01", "type": "entailment_unsupported", "instruction": "删除无依据扩写"}]

    assert _accept_revision(old, corrected, issue) == (True, "")


def test_accept_revision_allows_label_mismatch_correction_to_drop_overreaching_claim():
    """A label_mismatch verdict (claim overreaches what the cited evidence supports -- e.g. an
    unqualified attribution like "Osmani cited Peter Steinberger" when the evidence only shows
    Steinberger said it himself) is, like unsupported, a *grounding* defect: the fix is to delete
    or qualify the over-reaching clause, which can legitimately drop the citation it was
    over-reaching on. The `_accept_revision` evidence-count floor must not veto that fix the way
    it vetoes a generic "rewrote this and lost a citation" regression.

    See case-grounding-citation-drop idx=35/38: both label_mismatch claims needed the writer to
    remove an unsupported attribution; without this bypass the auto-revision loop ran but the
    writer's correction was rejected as fewer_evidence_used, so the claim survived unchanged and
    the quality_gate stayed needs_more_evidence -- the "stuck in waiting_approval" pattern.

    The corrected section keeps roughly the same content length (only the attribution clause is
    reworded), so the content_shrank floor never fires -- this isolates the fewer_evidence floor
    as the one that must be bypassed for grounding corrections.
    """
    old = {
        "content": "有依据内容。Osmani 引用了 Peter Steinberger 的实践经验，这是循环设计的关键论据。" * 5,
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    corrected = {
        "content": "有依据内容。Peter Steinberger 认为应设计循环来提示代理，这是循环设计的关键论据。" * 5,
        "evidence_table": [{"id": "ev_001"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    issue = [{"section_id": "sec_01", "type": "entailment_label_mismatch", "instruction": "claim 归属错误，改写为证据实际支撑的归属"}]

    assert _accept_revision(old, corrected, issue) == (True, "")

    # Without the bypass, the same correction (fewer citations) is still rejected -- the bypass
    # is gated on issue type, not on the fact that a correction happened.
    assert _accept_revision(old, corrected) == (False, "fewer_evidence_used")

    # A non-grounding issue type (pure readability) still can't smuggle in fewer citations.
    other_issue = [{"section_id": "sec_01", "type": "readability", "instruction": "改写得更清楚"}]
    assert _accept_revision(old, corrected, other_issue) == (False, "fewer_evidence_used")


def test_accept_revision_allows_critic_reduce_direction_to_drop_a_redundant_table():
    """2026-08-13 (see docs/evaluations/deep-research-critic-content-direction-20260813.md): a
    critic issue self-classified content_direction="reduce" (e.g. "this table restates the
    prose, delete it") is not an entailment_* type, but the fix it asks for is the same shape as
    a grounding correction -- deleting content legitimately drops evidence markers. Without this
    bypass, real job data (case-critic-direction-padding sec_03) showed the writer padding in new
    unsupported synthesis claims just to keep evidence-count from tripping fewer_evidence_used,
    because no existing issue type recognized this as a legitimate-to-shrink fix.
    """
    old = {
        "content": "正文描述。" * 10 + "\n\n| 机制 | 作用 |\n|---|---|\n| A | 做X |\n| B | 做Y |",
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    corrected = {
        "content": "正文描述。" * 10,
        "evidence_table": [{"id": "ev_001"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    issue = [{
        "section_id": "sec_01", "type": "information_design",
        "instruction": "表格与前文重复，删掉", "content_direction": "reduce",
    }]

    assert _accept_revision(old, corrected, issue) == (True, "")

    # Without the reduce classification, the identical correction is still rejected.
    unclassified_issue = [{"section_id": "sec_01", "type": "information_design", "instruction": "表格与前文重复，删掉"}]
    assert _accept_revision(old, corrected, unclassified_issue) == (False, "fewer_evidence_used")

    # content_direction="add" must never bypass the floor, even if present alongside other issues.
    add_issue = [{"section_id": "sec_01", "type": "limitations", "instruction": "补充权衡论证", "content_direction": "add"}]
    assert _accept_revision(old, corrected, add_issue) == (False, "fewer_evidence_used")


def test_accept_revision_reports_model_unavailable_when_revision_fell_back_on_operational_failure():
    """Layer D fix for the bounded auto-revision loop's `revision_fell_back_to_template`
    death-spiral (server case-revision-fallback-loop sec_01): when the OLD draft was real
    model-written prose and the NEW candidate fell back to the template purely because the
    revision model call failed/timed out (operational failure, not a content problem),
    `_accept_revision` must NOT accept the template -- that would erase real content with a
    generic stub -- and must signal this with the `revision_model_unavailable` reason so the
    caller keeps the old draft and the loop can try again next pass, instead of treating the
    section as permanently un-revisable.

    The legacy `revision_fell_back_to_template` reason is preserved for the genuine content
    case: when the OLD draft was ALSO a fallback_template, accepting another fallback isn't a
    regression so the reason stays as before (and the caller's retry feedback differs).
    """
    old_model_written = {
        "content": "真实模型撰写的分析正文，引用两条证据。" * 5,
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "model_writer", "fallback_reason": None},
    }
    # Candidate fell back to template because the revision model call timed out -- the writer
    # never produced real content, so this is operational, not a content regression.
    candidate_template_operational = {
        "content": "（模板兜底正文）",
        "evidence_table": [{"id": "ev_001"}, {"id": "ev_002"}],
        "synthesis_diagnostics": {"mode": "fallback_template", "fallback_reason": "timeout"},
    }
    issue = [{"section_id": "sec_01", "type": "entailment_unsupported", "instruction": "删除无依据扩写"}]

    # Operational failure: reject the template, but with the retryable model-unavailable signal
    # (not the permanent revision_fell_back_to_template reason).
    accepted, reason = _accept_revision(old_model_written, candidate_template_operational, issue)
    assert accepted is False
    assert reason == "revision_model_unavailable"

    # The same operational signal fires for the other model-failure fallback_reasons.
    for failure_reason in ("empty_response", "parse_failure", "empty_content", "unknown", "model_not_configured"):
        candidate = {**candidate_template_operational,
                     "synthesis_diagnostics": {"mode": "fallback_template", "fallback_reason": failure_reason}}
        accepted, reason = _accept_revision(old_model_written, candidate, issue)
        assert (accepted, reason) == (False, "revision_model_unavailable"), failure_reason

    # Content regression (old was ALSO template) keeps the legacy reason -- accepting another
    # fallback isn't replacing real content with a stub here, so the loop's retry feedback for
    # "write real prose" still applies. Same evidence count + same length, so the only branch
    # that can fire is the fallback-mode check; old_mode==new_mode==fallback_template means the
    # template branch doesn't reject at all -> accepted. That's the legacy behavior and the
    # loop already handles it (a section that was a template and stayed a template doesn't need
    # a model-unavailable retry), so just assert acceptance here rather than a specific reason.
    old_template = {**old_model_written,
                    "synthesis_diagnostics": {"mode": "fallback_template", "fallback_reason": "timeout"}}
    accepted, reason = _accept_revision(old_template, candidate_template_operational, issue)
    assert accepted is True

    # When the candidate genuinely produced real model prose, accept it normally (sanity).
    candidate_real = {**old_model_written, "content": "修改后的真实正文。" * 5,
                      "synthesis_diagnostics": {"mode": "model_writer", "fallback_reason": None}}
    assert _accept_revision(old_model_written, candidate_real, issue) == (True, "")


def test_reintroduces_forbidden_text_detects_instruction_quote_paste():
    """Target 1 of the regeneration guard: a critic/revision issue's instruction may quote an
    EXAMPLE sentence (the form _CRITIC_RULES since 2026-08-15 no longer produces, but a regression
    or hand-authored issue could). If the writer pasted that quoted span verbatim into the returned
    section, it smuggles in an assertion no cited evidence was validated against. The check must
    catch the verbatim paste after surface normalization (full-width punctuation/whitespace
    differences must not hide it) and must NOT fire when the section only addresses the direction.
    """
    instruction = (
        "建议在开头用1-2句话概括本节核心论点，"
        "例如'验证检查架构的有效性取决于权限回路、状态管理和部署流程的协同设计'。"
        "当前第一句直接进入细节，弱化了结构引导。"
    )
    issues = [{"section_id": "sec_00", "type": "structure", "instruction": instruction}]
    # Verbatim paste (same surface form).
    pasted = {
        "content": "验证检查架构的有效性取决于权限回路、状态管理和部署流程的协同设计[10][6]。其余正文略。",
        "claim_ledger": [],
    }
    hits = _reintroduces_forbidden_text(
        pasted, issues, overlap_threshold=0.6, require_evidence_intersect=False
    )
    assert any(h["source"] == "instruction_quote" for h in hits)
    # Paraphrase (different wording) must NOT trip the verbatim instruction-quote target.
    paraphrased = {
        "content": "本节从权限回路、状态管理与部署流程三个维度展开，说明验证检查架构的有效性来源。",
        "claim_ledger": [],
    }
    hits = _reintroduces_forbidden_text(
        paraphrased, issues, overlap_threshold=0.6, require_evidence_intersect=False
    )
    assert not any(h["source"] == "instruction_quote" for h in hits)


def test_reintroduces_forbidden_text_detects_regenerated_entailment_claim():
    """Target 2 of the regeneration guard (the a98ba6 dead loop): the writer deleted an
    entailment-rejected claim then rewrote the same proposition reworded, re-attached the
    citation, and marked the issue fixed. The token overlap of the new claim against the prior
    (rejected) claim's tokens must trip at threshold 0.5 but a legitimately-different rewrite
    that entailment accepted (sec_00 S3) must stay below threshold.
    """
    prior = "验证检查作为自动反馈回路嵌入整体流程，为代理输出提供持续校验。"
    issues = [{
        "section_id": "editorial",
        "type": "entailment_label_mismatch",
        "claim": prior,
        "evidence_ids": ["ev_016"],
    }]
    # Regeneration: same proposition reworded, same evidence id.
    regenerated = {
        "content": "验证检查构成自动反馈回路，用于校验代理输出。",
        "claim_ledger": [{"text": "验证检查构成自动反馈回路，用于校验代理输出。", "evidence_ids": ["ev_016"]}],
    }
    hits = _reintroduces_forbidden_text(
        regenerated, issues, overlap_threshold=0.5, require_evidence_intersect=True
    )
    assert any(h["source"] == "regenerated_claim" for h in hits)

    # Legitimate rewrite (sec_00 S3, accepted by entailment in pass_3): different proposition,
    # low token overlap -- must NOT trip even at the lowest threshold.
    s3_prior = (
        "这一框架通过将权限评估、状态持久化和环境配置整合为一个闭环，"
        "为解决AI代理行为一致性问题提供了可复用的参考设计。"
    )
    s3_new = (
        "整体而言，验证检查架构并非孤立组件，而是与项目规则、权限管道、状态管理紧密耦合的系统工程方案，"
        "其有效性依赖于各环节的协调与边界条件的清晰定义。"
    )
    s3_issues = [{
        "section_id": "sec_00",
        "type": "entailment_unsupported",
        "claim": s3_prior,
        "evidence_ids": ["ev_016", "ev_017"],
    }]
    s3_new_doc = {
        "content": s3_new,
        "claim_ledger": [{"text": s3_new, "evidence_ids": ["ev_016", "ev_017"]}],
    }
    for threshold in (0.5, 0.6, 0.7):
        hits = _reintroduces_forbidden_text(
            s3_new_doc, s3_issues, overlap_threshold=threshold, require_evidence_intersect=True
        )
        assert not any(h["source"] == "regenerated_claim" for h in hits), (
            f"legitimate S3 rewrite must not trip at threshold {threshold}"
        )

    # require_evidence_intersect must suppress a high-overlap hit when the new claim cites
    # DIFFERENT evidence than the rejected one (a real rewrite to a new source).
    diff_ev_doc = {
        "content": regenerated["claim_ledger"][0]["text"],
        "claim_ledger": [{"text": regenerated["claim_ledger"][0]["text"], "evidence_ids": ["ev_099"]}],
    }
    hits = _reintroduces_forbidden_text(
        diff_ev_doc, issues, overlap_threshold=0.5, require_evidence_intersect=True
    )
    assert not any(h["source"] == "regenerated_claim" for h in hits)


def test_accept_revision_rejects_regenerated_overreach_only_when_guard_enabled(monkeypatch):
    """The regeneration guard is env-gated OFF by default (its overlap threshold is not free to
    set -- see _reintroduces_forbidden_text). With the guard OFF, a regenerating revision is
    accepted as before (the dead loop this guard targets is real but the threshold must be
    calibrated first). With the guard ON and overlap low enough to trip, the revision is rejected
    with `regenerated_overreach` and the OLD draft is what the caller keeps.
    """
    old = {
        "content": "旧稿包含充分分析内容，引用两条证据。" * 5,
        "evidence_table": [{"id": "ev_016"}, {"id": "ev_017"}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    prior = "验证检查作为自动反馈回路嵌入整体流程，为代理输出提供持续校验。"
    regenerated = {
        "content": "验证检查构成自动反馈回路，用于校验代理输出。" + "后续分析展开。" * 5,
        "evidence_table": [{"id": "ev_016"}, {"id": "ev_017"}],
        "claim_ledger": [{"text": "验证检查构成自动反馈回路，用于校验代理输出。", "evidence_ids": ["ev_016"]}],
        "synthesis_diagnostics": {"mode": "model_writer"},
    }
    issue = [{
        "section_id": "editorial",
        "type": "entailment_label_mismatch",
        "claim": prior,
        "evidence_ids": ["ev_016"],
    }]
    # Guard OFF (default): accepted.
    monkeypatch.delenv("RESEARCH_REGENERATION_CHECK", raising=False)
    assert _accept_revision(old, regenerated, issue) == (True, "")

    # Guard ON, overlap 0.5 trips: rejected, old draft preserved by the caller.
    monkeypatch.setenv("RESEARCH_REGENERATION_CHECK", "1")
    monkeypatch.setenv("RESEARCH_REGENERATION_OVERLAP", "0.5")
    accepted, reason = _accept_revision(old, regenerated, issue)
    assert accepted is False
    assert reason == "regenerated_overreach"

    # Guard ON but a clean (non-regenerating) rewrite is still accepted.
    clean = {**regenerated, "claim_ledger": [{"text": "本节围绕权限评估与状态管理展开分析。", "evidence_ids": ["ev_016"]}]}
    accepted, reason = _accept_revision(old, clean, issue)
    assert accepted is True
    assert reason == ""


def test_writer_model_rejects_evidence_or_finding_markers_outside_current_payload():
    model = lambda _payload: {"content": "越权引用 [[ev_002]] 和 [[finding_002]]。"}

    content, meta = _call_writer_model(
        model,
        {"task": "write"},
        {"ev_001": {"id": "ev_001"}},
        allowed_finding_ids={"finding_001"},
    )

    assert content == ""
    assert meta["failure_reason"] == "evidence_validation_failed"


def test_table_claim_ledger_is_split_by_cited_cell_and_drops_punctuation_only_claims():
    evidence = {
        "ev_001": {"id": "ev_001", "source_url": "https://example.com/a"},
        "ev_002": {"id": "ev_002", "source_url": "https://example.com/b"},
    }
    rendered, ledger = _render_internal_citations(
        "| 模式 | 特点 |\n|---|---|\n| Retry [[ev_001]] | 失败后重试 [[ev_002]] |\n； [[ev_001]]",
        evidence,
        {"https://example.com/a": 1, "https://example.com/b": 2},
    )

    assert "|Retry [1]|失败后重试 [2]|" in rendered
    assert [item["text"] for item in ledger] == ["Retry", "失败后重试"]


def test_revise_report_sections_retries_once_after_rejection_and_can_recover():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    _add_recovery_finding(blueprint, findings)
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    good_content = (
        "监督者机制可用于协调专业角色 [[ev_001]]。"
        "作为可迁移的工程证据，检查点机制表明失败任务能够从已保存状态恢复 [[ev_002]]。"
    )

    def good_model(payload):
        if payload["task"].startswith("Write one analytical"):
            return {"content": good_content}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。", "editor_notes": []}

    written = write_report(user_request="调研可靠执行机制", blueprint=attached, findings=findings, evidence_map=evidence, model=good_model)

    call_count = {"first": 0, "retry": 0}

    def recovers_on_retry_model(payload):
        if payload["task"].startswith("Write one analytical"):
            revision_issues = payload["section_brief"].get("revision_issues") or []
            is_retry = any(issue.get("type") == "revision_rejected_retry" for issue in revision_issues)
            if is_retry:
                call_count["retry"] += 1
                # Second attempt: cites both evidence ids again, matching the original length, fixing
                # the earlier regression without tripping the separate content_shrank check.
                return {"content": good_content}
            call_count["first"] += 1
            # First attempt: strictly worse, cites only one evidence id.
            return {"content": "监督者机制可用于协调专业角色 [[ev_001]]。"}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。", "editor_notes": []}

    revised = revise_report_sections(
        user_request="调研可靠执行机制",
        written=written,
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        issues=[{"section_id": "sec_01", "type": "manual_revision", "instruction": "让表述更清楚一些"}],
        model=recovers_on_retry_model,
    )

    document = revised["documents"][0]
    assert call_count["first"] >= 1
    assert call_count["retry"] >= 1, "expected a retry attempt after the first rejection"
    assert document["revision_diagnostics"]["accepted"] is True
    assert document["revision_diagnostics"]["retried"] is True
    assert "ev_002" in [row["id"] for row in document["evidence_table"]]






def test_grounding_revision_rebuilds_internal_markers_from_existing_claim_ledger():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    _add_recovery_finding(blueprint, findings)
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    content = (
        "监督者机制可用于协调专业角色 [[ev_001]]。"
        "检查点机制表明失败任务能够恢复 [[ev_002]]。"
    )

    def initial_model(payload):
        if payload["task"].startswith("Write one analytical"):
            return {"content": content}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。"}

    written = write_report(
        user_request="调研可靠执行机制",
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        model=initial_model,
    )
    captured = []

    def revision_model(payload):
        captured.append(payload)
        return {"content": content}

    revised = revise_report_sections(
        user_request="调研可靠执行机制",
        written=written,
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        issues=[{
            "section_id": "sec_01",
            "type": "entailment_unsupported",
            "instruction": "删除无依据扩写",
        }],
        model=revision_model,
    )

    payload = next(item for item in captured if item["task"].startswith("Write one analytical"))
    assert payload["section_brief"]["existing_draft"] == written["documents"][0]["content"]
    assert payload["section_brief"]["existing_claim_ledger"] == written["documents"][0]["claim_ledger"]
    assert "source_notes" not in payload["section_dossier"]
    assert "relevant_chunks" not in payload["section_dossier"]
    assert any("Never copy numeric citations" in rule for rule in payload["rules"])
    assert revised["documents"][0]["revision_diagnostics"]["accepted"] is True


def test_revise_report_sections_keeps_old_document_when_revision_is_worse():
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    _add_recovery_finding(blueprint, findings)
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    good_content = (
        "监督者机制可用于协调专业角色 [[ev_001]]。"
        "作为可迁移的工程证据，检查点机制表明失败任务能够从已保存状态恢复 [[ev_002]]。"
    )

    def good_model(payload):
        if payload["task"].startswith("Write one analytical"):
            return {"content": good_content}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。", "editor_notes": []}

    written = write_report(user_request="调研可靠执行机制", blueprint=attached, findings=findings, evidence_map=evidence, model=good_model)
    original_content = written["documents"][0]["content"]
    assert original_content

    def worse_model(payload):
        if payload["task"].startswith("Write one analytical"):
            # Cites only one evidence id where the original cited two: strictly worse.
            return {"content": "监督者机制可用于协调专业角色 [[ev_001]]。"}
        return {"executive_summary": "总结 [[ev_001]]。", "conclusion": "结论 [[ev_002]]。", "editor_notes": []}

    revised = revise_report_sections(
        user_request="调研可靠执行机制",
        written=written,
        blueprint=attached,
        findings=findings,
        evidence_map=evidence,
        issues=[{"section_id": "sec_01", "type": "readability", "instruction": "improve readability"}],
        model=worse_model,
    )

    document = revised["documents"][0]
    assert document["content"] == original_content
    assert document["revision_diagnostics"]["accepted"] is False
    assert document["revision_diagnostics"]["rejected_reason"] == "fewer_evidence_used"
    assert document["revision_diagnostics"]["retried"] is True
    assert revised["diagnostics"]["revision_rejected_count"] == 1


def test_revision_rules_forbid_inventing_claims_for_neutral_or_add_issues():
    """2026-08-15 (case-critic-example-copy sec_00, pass_1): a critic structure issue marked
    content_direction="add" suggested an example topic sentence in its instruction text. The
    Writer copied that quoted example as a literal new claim, attached citations, and marked the
    issue "fixed" -- but the pasted sentence was an unsupported synthesis, so entailment later
    flagged it. An isolated 3-run A/B
    (scripts/replays/legacy/replay_a98ba6_example_copy.py) showed the writer
    copied the example sentence in 3/3 baseline runs and 2/3 "stronger anti-copy prompt" runs;
    the root cause is the critic QUOTING a finished sentence. That is fixed on the critic side
    (quality.py _CRITIC_RULES no longer lets the critic supply candidate sentences). This test
    pins the writer-side matching guard: whenever revision_issues are present, a rule must tell
    the writer that any sentence quoted inside an issue instruction is illustration-only and
    must not be pasted into the section, with the reject-don't-pad escape hatch available."""
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    attached = attach_section_dossiers(
        blueprint,
        findings=findings,
        evidence_map=evidence,
        selected_chunks=chunks,
        materials=materials,
    )
    existing = (
        "监督者机制可用于协调专业角色 [[ev_001]]。"
        "作为可迁移的工程证据，检查点机制表明失败任务能够从已保存状态恢复 [[ev_002]]。"
    )
    written = write_report(
        user_request="调研可靠执行机制", blueprint=attached,
        findings=findings, evidence_map=evidence,
        model=lambda p: {"content": existing} if p["task"].startswith("Write one analytical")
        else {"executive_summary": "s [[ev_001]]。", "conclusion": "c [[ev_002]]。", "editor_notes": []},
    )

    captured = []

    def revision_model(payload):
        captured.append(payload)
        return {"content": existing, "issue_responses": [{"issue_index": 0, "action": "fixed", "reason": "added topic sentence"}]}

    revise_report_sections(
        user_request="调研可靠执行机制", written=written, blueprint=attached,
        findings=findings, evidence_map=evidence,
        issues=[{
            "section_id": "sec_01", "type": "structure",
            "instruction": "段首缺少结论句，例如'可靠性取决于协调与恢复的协同设计'。",
            "content_direction": "neutral",
        }],
        model=revision_model,
    )
    payload = next(item for item in captured if item["task"].startswith("Write one analytical"))
    rules_blob = "\n".join(payload["rules"])
    # The anti-copy guard is unconditional on revision: any quoted sentence inside an issue
    # instruction is illustration-only and must not be pasted into the returned section.
    assert "ILLUSTRATION ONLY" in rules_blob
    assert "never paste it" in rules_blob
    # And the writer is pointed at the existing reject-don't-pad escape hatch.
    assert "action=\"rejected\"" in rules_blob


def test_revision_rules_omit_neutral_add_anti_fabrication_rule_when_only_reduce_issues():
    """The unconditional anti-copy guard above is cheap (one extra rule string on every revision),
    so there is no per-issue-type gating to test for -- it's always present when revising. This
    test instead pins that a pure grounding/reduce revision still gets the two-branch grounding
    rules and the anti-copy guard, without any neutral/add-specific fabrication rule that the
    08-15 isolation test proved too soft to rely on (2/3 stronger-prompt runs still copied)."""
    blueprint, findings, evidence, chunks, materials = _research_inputs()
    attached = attach_section_dossiers(
        blueprint, findings=findings, evidence_map=evidence,
        selected_chunks=chunks, materials=materials,
    )
    existing = "监督者机制可用于协调专业角色 [[ev_001]]。检查点恢复 [[ev_002]]。"
    written = write_report(
        user_request="调研可靠执行机制", blueprint=attached,
        findings=findings, evidence_map=evidence,
        model=lambda p: {"content": existing} if p["task"].startswith("Write one analytical")
        else {"executive_summary": "s [[ev_001]]。", "conclusion": "c [[ev_002]]。", "editor_notes": []},
    )
    captured = []

    def revision_model(payload):
        captured.append(payload)
        return {"content": "监督者机制可用于协调专业角色 [[ev_001]]。"}

    revise_report_sections(
        user_request="调研可靠执行机制", written=written, blueprint=attached,
        findings=findings, evidence_map=evidence,
        issues=[{
            "section_id": "sec_01", "type": "entailment_unsupported",
            "instruction": "删除无依据扩写",
        }],
        model=revision_model,
    )
    payload = next(item for item in captured if item["task"].startswith("Write one analytical"))
    rules_blob = "\n".join(payload["rules"])
    # The grounding-reduce branch is taken (entailment_unsupported issue), not the neutral/add
    # anti-fabrication branch that was removed -- the removed rule's specific phrasing must be gone.
    assert "never invent a new factual or synthesis claim" not in rules_blob
    # The unconditional anti-copy guard is still present (it applies to every revision).
    assert "ILLUSTRATION ONLY" in rules_blob


def _document_with_dossier(*, section_id, title, dossier_evidence, used_ids, research_questions=None, content=""):
    return {
        "section_id": section_id,
        "title": title,
        "content": content,
        "evidence_table": [{"id": item_id} for item_id in used_ids],
        "section_brief": {
            "research_questions": research_questions or [],
            "dossier": {"evidence": dossier_evidence},
        },
    }


def test_classify_unused_dossier_evidence_skips_sections_with_no_unused_items():
    document = _document_with_dossier(
        section_id="sec_01",
        title="01_机制",
        dossier_evidence=[{"id": "ev_001", "claim": "a"}],
        used_ids=["ev_001"],
    )

    result = classify_unused_dossier_evidence(documents=[document], model=lambda payload: {})

    assert result == {
        "unused_evidence_count": 0,
        "missed_relevant_count": 0,
        "irrelevant_count": 0,
        "classifications": [],
    }


def test_classify_unused_dossier_evidence_without_model_marks_unknown():
    document = _document_with_dossier(
        section_id="sec_01",
        title="01_机制",
        dossier_evidence=[{"id": "ev_001", "claim": "a"}, {"id": "ev_002", "claim": "b"}],
        used_ids=["ev_001"],
    )

    result = classify_unused_dossier_evidence(documents=[document], model=None)

    assert result["unused_evidence_count"] == 1
    assert result["classifications"][0]["evidence_id"] == "ev_002"
    assert result["classifications"][0]["verdict"] == "unknown"
    assert result["classifications"][0]["mode"] == "no_model"


def test_classify_unused_dossier_evidence_splits_irrelevant_and_missed_relevant():
    document = _document_with_dossier(
        section_id="sec_01",
        title="01_检查点机制",
        research_questions=[{"question_id": "sec_01_q1", "question": "系统如何从失败中恢复？"}],
        dossier_evidence=[
            {"id": "ev_001", "claim": "监督者负责协调专业角色。"},
            {"id": "ev_002", "claim": "检查点允许失败任务从已保存状态恢复。"},
            {"id": "ev_003", "claim": "该产品在 2019 年发布首个版本。"},
        ],
        used_ids=["ev_001"],
        content="监督者机制可用于协调专业角色 [1]。",
    )

    def model(payload):
        assert payload["section_title"] == "01_检查点机制"
        assert {item["evidence_id"] for item in payload["unused_evidence"]} == {"ev_002", "ev_003"}
        return {
            "classifications": [
                {"evidence_id": "ev_002", "verdict": "missed_relevant", "reason": "直接回答了失败恢复问题"},
                {"evidence_id": "ev_003", "verdict": "irrelevant", "reason": "发布时间与本节主题无关"},
            ]
        }

    result = classify_unused_dossier_evidence(documents=[document], model=model)

    assert result["unused_evidence_count"] == 2
    assert result["missed_relevant_count"] == 1
    assert result["irrelevant_count"] == 1
    by_id = {item["evidence_id"]: item for item in result["classifications"]}
    assert by_id["ev_002"]["verdict"] == "missed_relevant"
    assert by_id["ev_002"]["mode"] == "model"
    assert by_id["ev_003"]["verdict"] == "irrelevant"


def test_classify_unused_dossier_evidence_defaults_invalid_verdict_to_irrelevant():
    document = _document_with_dossier(
        section_id="sec_01",
        title="01_机制",
        dossier_evidence=[{"id": "ev_002", "claim": "b"}],
        used_ids=[],
    )

    def model(payload):
        return {"classifications": [{"evidence_id": "ev_002", "verdict": "maybe"}]}

    result = classify_unused_dossier_evidence(documents=[document], model=model)

    assert result["classifications"][0]["verdict"] == "irrelevant"
    assert result["classifications"][0]["mode"] == "no_verdict_defaulted"


def test_sanitize_leaked_evidence_ids_strips_bare_markers_and_reports_count():
    sanitized, count = _sanitize_leaked_evidence_ids("正文提到 ev_003 且残留标记 [[ev_011]] 未被渲染。")
    assert "ev_003" not in sanitized
    assert "[[ev_011]]" not in sanitized
    assert count == 2

    clean_sanitized, clean_count = _sanitize_leaked_evidence_ids("正文引用编号 [1] 且不包含内部标记。")
    assert clean_sanitized == "正文引用编号 [1] 且不包含内部标记。"
    assert clean_count == 0


def test_render_internal_citations_recovers_single_bracket_marker():
    """Regression: the writer is told to use [[ev_001]] but sometimes writes single-bracket
    [ev_001] instead. Before this fix, the stray-token cleanup pass deleted the id from inside
    the single brackets but left the brackets themselves, producing a literal empty "[]" in the
    delivered report (reported by the user from a real job's rendered content). The single-bracket
    form must now be recognized and converted into a real numbered citation instead."""
    evidence_by_id = {"ev_003": {"id": "ev_003", "source_url": "https://example.com/a"}}
    citation_map = {"https://example.com/a": 4}
    content = "可区分为直接注入与间接注入两类[ev_003]：直接注入是用户通过恶意指令直接操纵。"

    rendered, ledger = _render_internal_citations(content, evidence_by_id, citation_map)

    assert "[]" not in rendered
    assert "[4]" in rendered
    assert any(row["evidence_ids"] == ["ev_003"] for row in ledger)


def test_render_internal_citations_strips_any_leftover_empty_brackets():
    evidence_by_id: dict[str, dict] = {}
    citation_map: dict[str, int] = {}
    content = "这是一句包含空标记的正文[]，不应留下裸括号。"

    rendered, _ledger = _render_internal_citations(content, evidence_by_id, citation_map)

    assert "[]" not in rendered


def test_render_internal_citations_strips_orphaned_enumeration_punctuation():
    """Regression: reported from a real job's rendered executive summary. The writer model
    sometimes lists internal markers with a trailing/连接 Chinese enumeration comma, e.g.
    "（[[ev_004]]、[[ev_004]]、[[ev_004]]）" or "（[[ev_001]]、）" for a single marker. Once
    markers are rendered to numbered refs and duplicate refs on the same line are collapsed,
    the leftover 、punctuation used to survive verbatim, producing malformed citations like
    "（[4]、、）" or "（[1]、）" in the delivered report."""
    evidence_by_id = {"ev_004": {"id": "ev_004", "source_url": "https://example.com/a"}}
    citation_map = {"https://example.com/a": 4}

    single, _ = _render_internal_citations(
        "支持流式响应（[[ev_004]]、）。", evidence_by_id, citation_map
    )
    assert "、" not in single
    assert "[4]" in single

    repeated, _ = _render_internal_citations(
        "默认模型配置（[[ev_004]]、[[ev_004]]、[[ev_004]]）。", evidence_by_id, citation_map
    )
    assert "、" not in repeated
    assert repeated.count("[4]") == 1


def test_render_internal_citations_repairs_malformed_numeric_wrappers():
    rendered, _ = _render_internal_citations(
        "组合引用[[5], [4]]，以及残缺引用[[3], ]和[[10], ]。",
        {},
        {},
    )

    assert "[[" not in rendered
    assert "[5] [4]" in rendered
    assert "[3]" in rendered
    assert "[10]" in rendered


def test_render_internal_citations_repairs_malformed_internal_id_lists():
    evidence_by_id = {
        "ev_024": {"id": "ev_024", "source_url": "https://example.com/a"},
        "ev_025": {"id": "ev_025", "source_url": "https://example.com/b"},
        "ev_155": {"id": "ev_155", "source_url": "https://example.com/c"},
    }
    rendered, ledger = _render_internal_citations(
        "评分有差异[[ev_024], [ev_025]]，另一处也是如此[[ev_024], [ev_025], [ev_155]]。",
        evidence_by_id,
        {
            "https://example.com/a": 1,
            "https://example.com/b": 2,
            "https://example.com/c": 3,
        },
    )

    assert "[[" not in rendered
    assert "ev_" not in rendered
    assert "[1][2]" in rendered
    assert "[3]" in rendered
    assert ledger[0]["evidence_ids"] == ["ev_024", "ev_025", "ev_155"]


def test_render_internal_citations_keeps_preceding_sentence_as_claim_context():
    rendered, ledger = _render_internal_citations(
        "Elicit 更侧重学术研究。它采用系统综述式筛选 [[ev_175]]。",
        {"ev_175": {"id": "ev_175", "source_url": "https://example.com/elicit"}},
        {"https://example.com/elicit": 1},
    )

    assert rendered.endswith("它采用系统综述式筛选 [1]。")
    assert ledger[0]["text"] == "它采用系统综述式筛选 。"
    assert ledger[0]["claim_context"] == "Elicit 更侧重学术研究。"


def test_render_internal_citations_separates_reference_from_following_text():
    rendered, _ = _render_internal_citations(
        "钩子机制由事件监听器实现[[ev_001]]内置工具另行注册。",
        {"ev_001": {"id": "ev_001", "source_url": "https://example.com/a"}},
        {"https://example.com/a": 10},
    )

    assert "[10] 内置工具" in rendered
