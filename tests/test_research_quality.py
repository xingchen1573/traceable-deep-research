from __future__ import annotations

import time

from backend.research.quality import (
    ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION,
    ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    carry_forward_stale_critic_issues,
    combined_quality_gate,
    evaluate_report_quality,
    validate_claim_entailment,
)
from backend.research.quality import _recheck_low_confidence_issues
from backend.research.quality import _run_section_critics
from backend.research.quality import _CRITIC_RULES
from backend.services.research_service import (
    _documents_with_editorial_pseudo_document,
    _entailment_evidence_with_writer_context,
    _entailment_revision_issues,
    _revision_budget_split,
    _section_critic_budgets,
)


def _document(section_id: str, title: str = "章节标题", content: str = "正文内容。" * 20) -> dict:
    return {"section_id": section_id, "title": title, "content": content}


def _report_document(content: str) -> dict:
    return {"content": content, "delivery_status": "complete_report"}


def _question_states(count: int) -> list[dict]:
    return [{"status": "sufficient"} for _ in range(count)]


def test_critic_contract_separates_uncited_analysis_from_citation_errors():
    rules = "\n".join(_CRITIC_RULES)

    assert "每条事实性从句都要带引用标记" not in rules
    assert "不要仅因解释、综合或推断没有引用" in rules
    assert "Citation Validator 与 Claim Entailment Validator" in rules


def test_entailment_context_matches_writer_dossier_by_section_and_source():
    evidence = [{"id": "ev_001", "source_url": "https://example.com/a", "atomic_claim": "原子结论"}]
    blueprint = {"sections": [
        {
            "section_id": "sec_01",
            "dossier": {
                "source_notes": [{
                    "source_url": "https://example.com/a",
                    "claims": ["来源结论"],
                    "excerpts": ["本节 Writer 可见原文"],
                }],
                "relevant_chunks": [],
            },
        },
        {
            "section_id": "sec_02",
            "dossier": {
                "source_notes": [{"source_url": "https://example.com/b", "excerpts": ["其他来源"]}],
                "relevant_chunks": [],
            },
        },
    ]}

    enriched = _entailment_evidence_with_writer_context(evidence, blueprint)

    contexts = enriched[0]["validation_context_by_section"]
    assert "本节 Writer 可见原文" in contexts["sec_01"]
    assert "sec_02" not in contexts


def test_section_critic_receives_only_findings_from_its_section():
    seen: dict[str, list[str]] = {}

    def model(payload, timeout_override=None):
        seen[payload["report"]] = [item["finding_id"] for item in payload["available_findings"]]
        return {
            "scores": {
                key: 4
                for key in (
                    "structure", "synthesis", "evidence_quality", "readability",
                    "information_design", "limitations",
                )
            },
            "issues": [],
            "summary": "ok",
        }

    _run_section_critics(
        [_document("sec_01", content="第一章正文"), _document("sec_03", content="第三章正文")],
        [
            {"finding_id": "finding_001", "section_id": "sec_01"},
            {"finding_id": "finding_003", "section_id": "sec_03"},
        ],
        model,
        None,
        {},
        parallelism=1,
    )

    assert seen["第一章正文"] == ["finding_001"]
    assert seen["第三章正文"] == ["finding_003"]


def test_section_critic_skips_empty_fallback_sections():
    calls = []

    def model(payload, timeout_override=None):
        calls.append(payload)
        return {"scores": {}, "issues": [], "summary": ""}

    result = _run_section_critics(
        [{
            "section_id": "sec_03",
            "title": "03_最佳实践",
            "content": "",
            "synthesis_diagnostics": {"mode": "fallback_template", "fallback_reason": "no_findings_or_evidence"},
        }],
        [],
        model,
        None,
        {},
        parallelism=1,
    )

    assert calls == []
    assert result["section_diagnostics"] == [{
        "section_id": "sec_03", "title": "03_最佳实践", "outcome": "not_deliverable",
    }]


def test_entailment_revision_issues_include_substantive_blockers_but_not_punctuation():
    issues = _entailment_revision_issues({"judgments": [
        {"section_id": "sec_01", "status": "unsupported", "claim": "数据库列表没有证据。", "reason": "missing"},
        {"section_id": "sec_02", "status": "non_atomic", "claim": "两个事实；另一个事实。", "reason": "composite"},
        {"section_id": "sec_03", "status": "unsupported", "claim": "；", "reason": "punctuation"},
        {"section_id": "editorial", "status": "unsupported", "claim": "编辑结论", "reason": "editorial"},
    ]})

    assert [item["type"] for item in issues] == [
        "entailment_unsupported",
        "entailment_non_atomic",
        "entailment_unsupported",
    ]
    assert "数据库列表没有证据" in issues[0]["instruction"]




def test_entailment_batches_claims_so_one_timeout_does_not_degrade_whole_section(monkeypatch):
    """Regression test for case-timeout-and-page-chrome: sec_01 had 17 claims judged in a
    single model call. That one call hit the 75s role-model timeout and returned nothing,
    so all 17 claims fell back to _deterministic_entailment at once, producing 13 false
    label_mismatch verdicts. validate_claim_entailment now batches claims
    (RESEARCH_ENTAILMENT_BATCH_SIZE, default 6), so a single timed-out batch only
    degrades that batch's claims -- the rest still get a real model review.
    """
    from backend.research import quality

    # Build 8 claims in one section, all inference-typed (so none are pre-decided
    # unsupported/non_atomic deterministically and all reach the model path).
    claims = []
    for index in range(8):
        claims.append({
            "claim_index": index,
            "section_id": "sec_01",
            "text": f"这是第 {index} 条推断性陈述。",
            "claim_type": "inference",
            "evidence_ids": ["ev_001"],
        })
    report_document = {"claim_ledger": claims}
    evidence_map = [{"id": "ev_001", "atomic_claim": "证据文本", "evidence_text": "证据文本"}]

    # Track which batches the model was called with. Make the FIRST call time out
    # (return {} -- what _validation_model returns on timeout), and subsequent calls
    # succeed and entailing every claim.
    call_batches: list[int] = []

    def fake_model(payload):
        n = len(payload.get("claims") or [])
        call_batches.append(n)
        if len(call_batches) == 1:
            return {}  # first batch times out, returns nothing
        return {"judgments": [
            {"claim_index": int(item.get("claim_index")), "status": "entailed", "reason": "ok"}
            for item in payload.get("claims") or []
        ]}

    result = validate_claim_entailment(
        report_document=report_document,
        evidence_map=evidence_map,
        model=fake_model,
        parallelism=1,  # deterministic ordering so "first call" is well-defined
    )
    # With batch_size=6, 8 claims -> batches of [6, 2]. Both batches were sent to the model
    # (batching is in place), and the second batch's claims got a real verdict. The first
    # batch timed out (6 fallback claims); layer C (2026-08-05) then re-reviews those 6
    # fallback claims in batches of 2 (RESEARCH_ENTAILMENT_FALLBACK_BATCH_SIZE), so 3 more
    # model calls of size 2 follow and those claims also get a real verdict. Total call
    # sequence: [6, 2, 2, 2, 2].
    assert call_batches[:2] == [6, 2], f"expected first-pass batches of 6 and 2, got {call_batches}"
    assert len(call_batches) == 5, f"expected 2 first-pass + 3 re-review batches, got {call_batches}"
    # Every claim ends up with a real model verdict -- no claim is left on the deterministic
    # fallback. That is the layer-C guarantee the original batching fix alone couldn't make.
    assert result["deterministic_fallback_count"] == 0
    assert result["model_reviewed_count"] == 8
    assert all(row["status"] == "entailed" for row in result["judgments"])

    # Layer C re-reviewed the 6 first-batch fallback claims and the model entailed all of
    # them on the second pass, so they end up mode=model too. Every claim is now a real
    # model verdict -- the original batching fix alone left the first batch on fallback.
    judgments = {j["claim_index"]: j for j in result["judgments"]}
    model_judged = sum(1 for j in judgments.values() if j.get("mode") == "model")
    fallback_judged = sum(1 for j in judgments.values() if j.get("mode") == "deterministic_fallback")
    assert model_judged == 8, f"layer C should have upgraded all 8 to model, got {model_judged} model"
    assert fallback_judged == 0, f"no fallback should remain after layer C, got {fallback_judged} fallback"

    # The model_judged claims are entailed (not the keyword-matching fallback verdict).
    for index in (6, 7):
        assert judgments[index]["status"] == "entailed", (
            f"claim {index} should be entailed (model reviewed it), got {judgments[index]['status']}"
        )


def test_aggregate_component_verdict_rescues_when_all_components_backed():
    """Layer H aggregate: a synthesis claim judged unsupported whose every component the model
    finds backed gets upgraded to reasonable_inference. This is the a98ba6 idx0/idx35 '协同设计'
    case -- each underlying component has cited evidence, the connective word does not, and the
    default judge's near-verbatim standard had flagged the whole claim unsupported."""
    from backend.research.quality import _aggregate_component_verdict

    original = {
        "status": "unsupported",
        "reason": "evidence does not mention 协同",
        "claim_index": 0,
        "section_id": "sec_00",
        "evidence_ids": ["ev_017", "ev_011", "ev_016"],
    }
    component_result = {
        "components": [
            {"component": "权限回路", "backed": True, "backing_evidence_ids": ["ev_017"], "reason": "cascading through policy engines"},
            {"component": "状态管理", "backed": True, "backing_evidence_ids": ["ev_011"], "reason": "session state"},
            {"component": "部署流程", "backed": True, "backing_evidence_ids": ["ev_016"], "reason": "flow diagram"},
        ],
        "status": "reasonable_inference",
        "reason": "all components backed",
    }
    new_status, new_reason, rescued = _aggregate_component_verdict(original, component_result)
    assert rescued is True
    assert new_status == "reasonable_inference"
    assert "component-wise rescue" in new_reason


def test_aggregate_component_verdict_keeps_unsupported_when_any_component_unbacked():
    """Layer H must NOT rescue a real fabrication (a98ba6 idx5 '编译器优化': ev_011 is about
    session state and never mentions compiler optimization). An unbacked component leaves the
    original unsupported verdict untouched. Only-upgrade-never-downgrade."""
    from backend.research.quality import _aggregate_component_verdict

    original = {
        "status": "unsupported",
        "reason": "evidence does not mention compiler optimization",
        "claim_index": 5,
        "section_id": "sec_00",
        "evidence_ids": ["ev_011"],
    }
    component_result = {
        "components": [
            {"component": "建议来源于相邻领域", "backed": False, "backing_evidence_ids": [], "reason": "ev_011 never mentions adjacent fields"},
            {"component": "迁移需要适配", "backed": True, "backing_evidence_ids": ["ev_011"], "reason": "context-dependent"},
        ],
        "status": "unsupported",
        "reason": "compiler optimization unbacked",
    }
    new_status, new_reason, rescued = _aggregate_component_verdict(original, component_result)
    assert rescued is False
    assert new_status == "unsupported"
    # reason unchanged from the original
    assert new_reason == original["reason"]


def test_aggregate_component_verdict_leaves_verdict_untouched_when_no_components():
    """Safe default: if the model returned no components list (timeout/malformed), Layer H does
    nothing -- the original verdict stands. This is the 'do no harm' guarantee: even if the
    rescue re-judge fails, the worst case is no rescue, never a wrong change."""
    from backend.research.quality import _aggregate_component_verdict

    original = {"status": "unsupported", "reason": "orig", "claim_index": 1, "section_id": "sec_00", "evidence_ids": ["ev_001"]}
    # missing components
    new_status, _, rescued = _aggregate_component_verdict(original, {"status": "unsupported", "reason": "x"})
    assert rescued is False
    assert new_status == "unsupported"
    # empty components list
    new_status, _, rescued = _aggregate_component_verdict(original, {"components": [], "status": "unsupported", "reason": "x"})
    assert rescued is False
    assert new_status == "unsupported"
    # non-dict result
    new_status, _, rescued = _aggregate_component_verdict(original, None)
    assert rescued is False
    assert new_status == "unsupported"
    # a claim that is NOT unsupported/label_mismatch is never touched even if components all backed
    entailed = {"status": "entailed", "reason": "ok", "claim_index": 2, "section_id": "sec_00", "evidence_ids": ["ev_001"]}
    new_status, _, rescued = _aggregate_component_verdict(
        entailed, {"components": [{"component": "x", "backed": True, "backing_evidence_ids": ["ev_001"]}], "status": "reasonable_inference"}
    )
    assert rescued is False
    assert new_status == "entailed"


def test_entailment_component_payload_forces_decomposition_schema():
    """The Layer H payload must ask the model to return per-component verdicts (the structured
    signal) rather than a bare overall status. The schema is what makes this a structural
    enforcement instead of a soft prompt rule (the project's repeatedly-validated pattern)."""
    from backend.research.quality import _entailment_component_payload

    rows = [(0, {"text": "claim", "claim_type": "inference", "support_scope": "cross_domain", "evidence_ids": ["ev_001"]})]
    evidence_by_id = {"ev_001": {"id": "ev_001", "cleaned_quote": "q", "atomic_claim": "c"}}
    payload = _entailment_component_payload("sec_00", rows, evidence_by_id)
    schema_judgment = payload["schema"]["judgments"][0]
    assert "components" in schema_judgment
    assert schema_judgment["components"][0]["backed"] is True
    # rules must tell the model connective wording is NOT a component needing its own evidence
    rules_blob = " ".join(payload["rules"])
    assert "协同" in rules_blob or "connective" in rules_blob.lower() or "glue" in rules_blob.lower()
    assert "component" in rules_blob.lower()


def test_validate_claim_entailment_component_rescue_upgrades_synthesis_when_enabled(monkeypatch):
    """End-to-end of Layer H wiring: with RESEARCH_ENTAILMENT_COMPONENT_RESCUE=1, a model-judged
    unsupported claim whose rescue re-judge finds all components backed is upgraded to
    reasonable_inference (component_rescue=True). With the flag off (default), no rescue runs and
    the unsupported verdict stands."""
    from backend.research import quality

    claims = [{
        "claim_index": 0,
        "section_id": "sec_00",
        "text": "验证检查架构的有效性取决于权限回路、状态管理和部署流程的协同设计。",
        "claim_type": "inference",
        "evidence_ids": ["ev_017", "ev_011", "ev_016"],
    }]
    report_document = {"claim_ledger": claims}
    evidence_map = [
        {"id": "ev_017", "cleaned_quote": "Evaluates tool call permissions by cascading through policy engines.", "atomic_claim": "tool call permissions"},
        {"id": "ev_011", "cleaned_quote": "Avoid storing session state directly in the harness memory.", "atomic_claim": "session state"},
        {"id": "ev_016", "cleaned_quote": "A Mermaid diagram illustrating the flow.", "atomic_claim": "flow diagram"},
    ]

    def fake_model(payload):
        task = str(payload.get("task") or "")
        if "Decompose" in task or "components" in str(payload.get("schema") or ""):
            # Layer H rescue re-judge: all components backed.
            return {"judgments": [{
                "claim_index": 0,
                "components": [
                    {"component": "权限回路", "backed": True, "backing_evidence_ids": ["ev_017"]},
                    {"component": "状态管理", "backed": True, "backing_evidence_ids": ["ev_011"]},
                    {"component": "部署流程", "backed": True, "backing_evidence_ids": ["ev_016"]},
                ],
                "status": "reasonable_inference",
                "reason": "all backed",
            }]}
        # default first-pass judge: flags the synthesis unsupported (the near-verbatim bug)
        return {"judgments": [{"claim_index": 0, "status": "unsupported", "reason": "no 协同 in evidence"}]}

    # flag OFF (default): unsupported stands, no rescue.
    monkeypatch.delenv("RESEARCH_ENTAILMENT_COMPONENT_RESCUE", raising=False)
    result = validate_claim_entailment(report_document=report_document, evidence_map=evidence_map, model=fake_model, parallelism=1)
    assert result["unsupported_count"] == 1
    assert result["judgments"][0]["status"] == "unsupported"
    assert not result["judgments"][0].get("component_rescue")

    # flag ON: rescue upgrades to reasonable_inference.
    monkeypatch.setenv("RESEARCH_ENTAILMENT_COMPONENT_RESCUE", "1")
    result = validate_claim_entailment(report_document=report_document, evidence_map=evidence_map, model=fake_model, parallelism=1)
    assert result["unsupported_count"] == 0
    assert result["inference_count"] == 1
    assert result["judgments"][0]["status"] == "reasonable_inference"
    assert result["judgments"][0].get("component_rescue") is True


def test_atomic_boundary_profile_never_falls_through_to_component_rescue(monkeypatch):
    prompt_versions: list[str] = []

    def fake_model(payload):
        prompt_versions.append(str(payload.get("prompt_version") or ""))
        return {
            "judgments": [{
                "case_id": "0",
                "status": "unsupported",
                "claim_scope": "domain_fact",
                "reason": "Evidence only supports part of the cited span.",
            }]
        }

    monkeypatch.setenv("RESEARCH_ENTAILMENT_COMPONENT_RESCUE", "1")
    result = validate_claim_entailment(
        report_document={
            "claim_ledger": [{
                "section_id": "sec_01",
                "text": "该机制提高了响应速度。",
                "claim_type": "inference",
                "evidence_ids": ["ev_001"],
            }]
        },
        evidence_map=[{
            "id": "ev_001",
            "atomic_claim": "响应速度有所提高。",
            "cleaned_quote": "测试显示响应速度有所提高。",
        }],
        model=fake_model,
        parallelism=1,
        payload_profile=ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    )

    assert result["unsupported_count"] == 1
    assert prompt_versions == [ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION]


def test_validate_claim_entailment_component_rescue_keeps_real_fabrication(monkeypatch):
    """Layer H must NOT upgrade a real fabrication (a98ba6 idx5 编译器优化). Even when the rescue
    re-judge runs, an unbacked component means the original unsupported verdict stays."""
    from backend.research import quality

    claims = [{
        "claim_index": 0,
        "section_id": "sec_00",
        "text": "该建议来源于相邻领域（如编译器优化）的工程经验。",
        "claim_type": "inference",
        "evidence_ids": ["ev_011"],
    }]
    report_document = {"claim_ledger": claims}
    evidence_map = [{"id": "ev_011", "cleaned_quote": "Avoid storing session state directly in the harness memory.", "atomic_claim": "session state"}]

    def fake_model(payload):
        if "components" in str(payload.get("schema") or ""):
            # rescue re-judge correctly finds the compiler-optimization component UNbacked
            return {"judgments": [{
                "claim_index": 0,
                "components": [
                    {"component": "编译器优化/相邻领域", "backed": False, "backing_evidence_ids": [], "reason": "ev_011 never mentions it"},
                ],
                "status": "unsupported",
                "reason": "unbacked component",
            }]}
        return {"judgments": [{"claim_index": 0, "status": "unsupported", "reason": "no compiler in evidence"}]}

    monkeypatch.setenv("RESEARCH_ENTAILMENT_COMPONENT_RESCUE", "1")
    result = validate_claim_entailment(report_document=report_document, evidence_map=evidence_map, model=fake_model, parallelism=1)
    assert result["unsupported_count"] == 1
    assert result["judgments"][0]["status"] == "unsupported"
    assert not result["judgments"][0].get("component_rescue")


def test_validate_claim_entailment_preserves_model_claim_scope():
    captured = {}
    report_document = {"claim_ledger": [{
        "section_id": "sec_01",
        "text": "现有资料尚未提供具体构建步骤。",
        "claim_type": "limitation",
        "evidence_ids": ["ev_001"],
    }]}
    evidence_map = [{
        "id": "ev_001",
        "atomic_claim": "The repository can be built locally.",
        "cleaned_quote": "Clone the repository, install dependencies, and build the project.",
    }]

    def fake_model(payload):
        captured.update(payload)
        return {"judgments": [{
            "claim_index": 0,
            "status": "unsupported",
            "claim_scope": "research_process_disclosure",
            "reason": "The sentence describes what available materials did not provide.",
        }]}

    result = validate_claim_entailment(
        report_document=report_document,
        evidence_map=evidence_map,
        model=fake_model,
        parallelism=1,
    )

    assert "claim_scope" in captured["schema"]["judgments"][0]
    assert result["judgments"][0]["status"] == "unsupported"
    assert result["judgments"][0]["claim_scope"] == "research_process_disclosure"


def test_section_critic_budgets_even_split_when_generous():
    documents = [_document(f"sec_{i:02d}") for i in range(3)]
    deadline = time.monotonic() + 150.0
    budgets = _section_critic_budgets(documents, deadline)
    assert set(budgets) == {doc["section_id"] for doc in documents}
    for value in budgets.values():
        assert 49.0 <= value <= 50.0


def test_section_critic_budgets_excludes_reference_section():
    documents = [_document("sec_01"), _document("sec_ref", title="05_证据与参考")]
    deadline = time.monotonic() + 40.0
    budgets = _section_critic_budgets(documents, deadline)
    assert set(budgets) == {"sec_01"}


def test_section_critic_budgets_caps_to_affordable_k_when_budget_thin():
    # 5 sections, only ~65s left: even split (13s) is below the 30s floor, so only the
    # first floor(65/30)=2 sections get a real budget; the rest are pre-scheduled as skipped.
    documents = [_document(f"sec_{i:02d}") for i in range(5)]
    deadline = time.monotonic() + 65.0
    budgets = _section_critic_budgets(documents, deadline)
    affordable = [sid for sid, budget in budgets.items() if budget > 0]
    skipped = [sid for sid, budget in budgets.items() if budget <= 0]
    assert len(affordable) == 2
    assert len(skipped) == 3
    for sid in affordable:
        assert budgets[sid] >= 30.0


def test_documents_with_editorial_pseudo_document_appends_joined_editorial():
    """Regression for docs/modules/deep-research-step4-approval-and-delivery-discussion-v0.1.md
    商榷点四: the critic (`_run_section_critics` via `evaluate_report_quality`) only reads
    `documents`, and editorial has no single `content` field -- it's split into
    `executive_summary`/`conclusion`. The pseudo-document must join both into one `content` and
    tag it `section_id="editorial"` so it's shaped exactly like a real document."""
    documents = [_document("sec_01")]
    editorial = {"executive_summary": "这是执行摘要。", "conclusion": "这是结论。"}

    result = _documents_with_editorial_pseudo_document(documents, editorial)

    assert len(result) == 2
    assert result[0] is documents[0], "the original sec_01 document must be untouched, not copied"
    pseudo = result[1]
    assert pseudo["section_id"] == "editorial"
    assert "这是执行摘要。" in pseudo["content"]
    assert "这是结论。" in pseudo["content"]
    # The original `documents` list itself must never be mutated -- other readers (report
    # assembly, knowledge-base save, citation validation) expect only real sec_XX documents.
    assert documents == [_document("sec_01")]


def test_documents_with_editorial_pseudo_document_skips_when_editorial_empty():
    documents = [_document("sec_01")]
    assert _documents_with_editorial_pseudo_document(documents, {}) == documents
    assert _documents_with_editorial_pseudo_document(documents, None) == documents


def test_evaluate_report_quality_scores_editorial_pseudo_document():
    """End-to-end proof that once editorial is wrapped as a pseudo-document and passed to
    evaluate_report_quality, the existing generic _run_section_critics machinery (budgets,
    concurrency, low-confidence dimension merge) picks it up with zero changes to quality.py --
    exactly the "只需要伪装成这个形状喂进去" claim the discussion doc's feasibility check made."""
    documents = [_document("sec_01")]
    editorial_pseudo = _documents_with_editorial_pseudo_document(
        documents, {"executive_summary": "摘要正文。", "conclusion": "结论正文。"}
    )
    report_document = _report_document("## 一\n\n正文内容。" * 5)

    seen_section_ids = []

    def fake_model(payload, timeout_override=None):
        seen_section_ids.append(payload.get("report"))
        if "摘要正文" in str(payload.get("report") or ""):
            return {
                "scores": {"structure": 2, "synthesis": 2, "evidence_quality": 2, "readability": 2, "information_design": 2, "limitations": 2},
                "issues": [{"section_id": "editorial", "type": "weak_narrative", "instruction": "执行摘要论证薄弱", "confidence": 90}],
                "summary": "editorial weak",
            }
        return {
            "scores": {"structure": 5, "synthesis": 5, "evidence_quality": 5, "readability": 5, "information_design": 5, "limitations": 5},
            "issues": [],
            "summary": "sec_01 fine",
        }

    result = evaluate_report_quality(
        report_document=report_document,
        question_states=_question_states(1),
        findings=[{"finding": "x"}] * 4,
        rejected_evidence=[],
        documents=editorial_pseudo,
        model=fake_model,
        section_critic_budgets={"sec_01": 30.0, "editorial": 30.0},
    )

    section_ids_scored = {item["section_id"] for item in result["section_critic_diagnostics"]}
    assert "editorial" in section_ids_scored, "editorial must actually reach the critic model, not be silently skipped"
    editorial_issues = [issue for issue in result["issues"] if issue.get("section_id") == "editorial"]
    assert editorial_issues, "a critic issue targeting editorial must survive into report_quality.issues"


def test_evaluate_report_quality_low_critic_dimension_score_no_longer_blocks():
    """2026-08-17 (architecture decision, see docs/plans/writer-critic-loop-redesign): the
    6-dimension critic is a subjective model judgment that used to gate approval via
    report_quality_dimension_below_3 and drive an unproductive writer<->critic revision loop.
    It still scores/reports issues (used for the report's appended 局限性声明), but a low
    dimension score must no longer appear in blocking_issues or flip status to failed."""

    def fake_model(payload, timeout_override=None):
        return {
            "scores": {"structure": 1, "synthesis": 1, "evidence_quality": 1, "readability": 1, "information_design": 1, "limitations": 1},
            "issues": [{"section_id": "sec_01", "type": "structure", "instruction": "结构需要改进", "confidence": 90}],
            "summary": "low scores across the board",
        }

    result = evaluate_report_quality(
        report_document=_report_document("## 一\n\n正文内容 [1]。" * 5),
        question_states=_question_states(1),
        findings=[{"finding": "x"}] * 4,
        rejected_evidence=[],
        documents=[_document("sec_01", title="一")],
        model=fake_model,
        section_critic_budgets={"sec_01": 30.0},
    )

    assert "report_quality_dimension_below_3" not in result["blocking_issues"]
    assert result["status"] == "passed"
    # The low scores and issue text still surface -- they're now purely material for the
    # report's limitations disclosure, not a gate.
    assert result["scores"]["structure"] == 1
    assert result["issues"]


def test_evaluate_report_quality_records_evidence_gap_without_critic_call():
    calls = []

    def fake_model(payload, timeout_override=None):
        calls.append(payload)
        raise AssertionError("evidence-gap text is not analytical prose for the critic")

    result = evaluate_report_quality(
        report_document={
            "content": "# 阶段性报告\n\n## 局限差异\n\n本轮检索未形成合格证据，现记录证据缺口。",
            "delivery_status": "partial_report",
            "completeness": {
                "status": "partial_report",
                "required_section_count": 1,
                "ready_section_count": 0,
                "core_empty_section_ids": ["sec_04"],
            },
        },
        question_states=[{"status": "exhausted"}],
        findings=[],
        rejected_evidence=[],
        documents=[{
            "section_id": "sec_04",
            "title": "04_局限差异",
            "content": "本轮检索未形成合格证据，现记录证据缺口。",
            "synthesis_diagnostics": {"mode": "evidence_gap"},
        }],
        model=fake_model,
        section_critic_budgets={"sec_04": 30.0},
    )

    assert calls == []
    assert result["status"] == "passed"
    assert result["report_status"] == "partial_report"
    assert result["section_critic_diagnostics"] == [{
        "section_id": "sec_04",
        "title": "04_局限差异",
        "outcome": "evidence_gap",
    }]


def test_run_section_critics_overwrites_model_hallucinated_section_id():
    """Regression for docs/evaluations/deep-research-revision-loop-critic-writer-transparency-
    20260811-handoff.md: `_model_critic_section` is called once per section with only raw
    `content` (no section_id/title in the payload -- see quality.py's `_model_critic_section`),
    so the model has no grounding for the `section_id` it self-reports on each issue. A real-job
    audit (10 jobs, 293 issues) found 41.6% of critic issues came back with a section_id matching
    no real document section (e.g. "report", "main", "para_1", or a Chinese section title), which
    `revise_report_sections`'s exact-match `targets` routing then silently drops before the
    writer ever sees them. The caller already knows the true section_id via `document`, so it
    must stamp that onto every issue rather than trust the model's guess."""
    documents = [_document("sec_02", title="真实标题")]

    def fake_model(payload, timeout_override=None):
        return {
            "scores": {"structure": 3, "synthesis": 3, "evidence_quality": 3, "readability": 3, "information_design": 3, "limitations": 3},
            "issues": [
                {"section_id": "report", "type": "evidence_quality", "instruction": "缺少引用", "confidence": 80},
                {"section_id": "para_1", "type": "structure", "instruction": "段首无结论", "confidence": 70},
            ],
            "summary": "hallucinated section_id",
        }

    result = evaluate_report_quality(
        report_document=_report_document("## 一\n\n正文内容。" * 5),
        question_states=_question_states(1),
        findings=[{"finding": "x"}] * 4,
        rejected_evidence=[],
        documents=documents,
        model=fake_model,
        section_critic_budgets={"sec_02": 30.0},
    )

    assert len(result["issues"]) == 2
    assert all(issue["section_id"] == "sec_02" for issue in result["issues"]), (
        "the model's hallucinated section_id ('report'/'para_1') must be overwritten with the "
        "true section_id known by the caller, or these issues get silently dropped by "
        "revise_report_sections's exact-match routing"
    )
    # editorial's poor score (2) must pull the aggregate dimension down via the existing
    # min-across-sections merge -- no special-casing needed in quality.py.
    assert result["scores"]["structure"] == 2


_FULL_SCORES = {"structure": 4, "synthesis": 4, "evidence_quality": 4, "readability": 4, "information_design": 4, "limitations": 4}


def test_critic_confirms_once_then_reuses_on_unchanged_content():
    """A-class cross-pass memory (2026-08-12, mirrors validate_claim_entailment's 'confirm
    twice' rule): a section's content unchanged from the prior pass isn't reuse-eligible on a
    single agreeing look -- only once two consecutive independent scores agree does the second
    pass's `confirmed=True` diagnostic make the THIRD pass eligible to skip the model call.
    Pass A here starts with no prior (fresh, unconfirmed). Pass B's content is identical and its
    fresh score agrees -- confirmed, but still hit the model (this is only the second look
    agreeing). Pass C reuses without calling the model."""
    section = "sec_01"
    content = "正文内容不变。" * 20
    documents = [_document(section, content=content)]
    call_count = {"n": 0}

    def fake_model(payload, timeout_override=None):
        call_count["n"] += 1
        return {"scores": _FULL_SCORES, "issues": [], "summary": ""}

    pass_a = evaluate_report_quality(
        report_document=_report_document(content), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents, model=fake_model,
        section_critic_budgets={section: 30.0},
    )
    diag_a = pass_a["section_critic_diagnostics"][0]
    assert diag_a["confirmed"] is False, "first ever look has no prior to agree with"
    assert call_count["n"] == 1

    pass_b = evaluate_report_quality(
        report_document=_report_document(content), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents, model=fake_model,
        section_critic_budgets={section: 30.0}, prior_report_quality=pass_a,
    )
    diag_b = pass_b["section_critic_diagnostics"][0]
    assert call_count["n"] == 2, "second look must still call the model -- only one prior look exists"
    assert diag_b["confirmed"] is True, "content unchanged and score agrees with the prior pass"

    pass_c = evaluate_report_quality(
        report_document=_report_document(content), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents, model=fake_model,
        section_critic_budgets={section: 30.0}, prior_report_quality=pass_b,
    )
    assert call_count["n"] == 2, "third pass on unchanged, twice-confirmed content must reuse, not call the model"
    assert pass_c["scores"] == pass_b["scores"]


def test_critic_does_not_reuse_when_content_changed():
    """Content changing between passes must never trigger A-class reuse, regardless of how
    confirmed the prior pass was -- reuse means 'this exact content was already judged twice',
    which is false once the writer has actually revised the section."""
    section = "sec_01"
    call_count = {"n": 0}

    def fake_model(payload, timeout_override=None):
        call_count["n"] += 1
        return {"scores": _FULL_SCORES, "issues": [], "summary": ""}

    content_v1 = "第一版正文。" * 20
    documents_v1 = [_document(section, content=content_v1)]
    pass_a = evaluate_report_quality(
        report_document=_report_document(content_v1), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents_v1, model=fake_model,
        section_critic_budgets={section: 30.0},
    )

    content_v2 = "第二版正文，内容已修改。" * 20
    documents_v2 = [_document(section, content=content_v2)]
    pass_b = evaluate_report_quality(
        report_document=_report_document(content_v2), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents_v2, model=fake_model,
        section_critic_budgets={section: 30.0}, prior_report_quality=pass_a,
    )

    assert call_count["n"] == 2, "changed content must always hit the model, never reuse"
    diag_b = pass_b["section_critic_diagnostics"][0]
    assert diag_b.get("reused") is not True


def test_critic_receives_prior_issues_as_reversal_check_when_content_changes():
    """B-class cross-pass memory (2026-08-12): when content changed since the prior pass, the
    critic has no exact-match anchor to reuse against, but real job data
    (case-critic-pass-reversal sec_03) showed the critic flip-flop on the same rubric point
    pass to pass with no self-awareness ('delete this table' -> 'add a table here' -> insist
    again). The prior pass's own issues for this section must reach the payload as
    `prior_pass_reviewer_notes`, containing only type/instruction -- never `scores`, to avoid
    anchoring this pass's numeric score toward the previous one."""
    section = "sec_03"
    content_v1 = "包含一张表格的正文。" * 15
    documents_v1 = [_document(section, content=content_v1)]

    def model_v1(payload, timeout_override=None):
        return {
            "scores": _FULL_SCORES,
            "issues": [{"section_id": section, "type": "information_design", "instruction": "表格与正文重复，删掉", "confidence": 80}],
            "summary": "",
        }

    pass_a = evaluate_report_quality(
        report_document=_report_document(content_v1), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents_v1, model=model_v1,
        section_critic_budgets={section: 30.0},
    )

    seen_payload = {}
    content_v2 = "删除表格后的正文。" * 15
    documents_v2 = [_document(section, content=content_v2)]

    def model_v2(payload, timeout_override=None):
        seen_payload.update(payload)
        return {"scores": _FULL_SCORES, "issues": [], "summary": ""}

    evaluate_report_quality(
        report_document=_report_document(content_v2), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents_v2, model=model_v2,
        section_critic_budgets={section: 30.0}, prior_report_quality=pass_a,
    )

    notes = seen_payload.get("prior_pass_reviewer_notes")
    assert notes is not None, "content changed -- prior issues for this section must be passed as a reversal check"
    prior_issues = notes["previous_reviewer_issues"]
    assert len(prior_issues) == 1
    assert prior_issues[0]["type"] == "information_design"
    assert prior_issues[0]["instruction"] == "表格与正文重复，删掉"
    assert "scores" not in notes, "prior scores must never be exposed -- would anchor this pass's own scoring"


def test_critic_prior_issues_omitted_for_a_section_with_no_prior_pass():
    """A section with no prior pass at all (e.g. it's new this revision, or the report only just
    started) must not get a `prior_pass_reviewer_notes` field -- there's nothing to check."""
    seen_payload = {}

    def fake_model(payload, timeout_override=None):
        seen_payload.update(payload)
        return {"scores": _FULL_SCORES, "issues": [], "summary": ""}

    documents = [_document("sec_01")]
    evaluate_report_quality(
        report_document=_report_document("正文。" * 20), question_states=_question_states(1),
        findings=[], rejected_evidence=[], documents=documents, model=fake_model,
        section_critic_budgets={"sec_01": 30.0},
    )
    assert "prior_pass_reviewer_notes" not in seen_payload


def test_revision_budget_split_reserves_first_validate_cost_for_recheck():
    # First validate took 90s -- that's the best estimate for what the recheck will cost too, so
    # revision must not be allowed to eat into that reserved slice.
    now = time.monotonic()
    deadline = now + 400.0
    split = _revision_budget_split(deadline, first_validate_elapsed_seconds=90.0, now=now)
    assert split["recheck_reserve"] == 90.0
    assert split["revision_budget_available"] == 400.0 - 90.0


def test_revision_budget_split_uses_floor_when_first_validate_was_fast():
    # First validate only took 5s, well below the RESEARCH_VALIDATION_RECHECK_MIN_REMAINING_SECONDS
    # floor (default 90s) -- the reserve should not shrink below that floor.
    now = time.monotonic()
    deadline = now + 400.0
    split = _revision_budget_split(deadline, first_validate_elapsed_seconds=5.0, now=now)
    assert split["recheck_reserve"] == 90.0
    assert split["revision_budget_available"] == 400.0 - 90.0


def test_revision_budget_split_returns_zero_when_remaining_below_reserve():
    # Remaining (50s) is smaller than even the bare RESEARCH_VALIDATION_RECHECK_MIN_REMAINING_SECONDS
    # floor (90s) plus a minimally useful revision slice (45s) -- 50 < 90+45 -- so there is no
    # split that lets both phases run usefully. This is the genuinely-unsalvageable case: revision
    # must still get 0, not a degraded slice. Contrast with
    # test_revision_budget_split_degrades_reserve_instead_of_starving_revision below, where the
    # ideal reserve doesn't fit but the *floor* reserve does -- that case no longer returns 0
    # after the 2026-07-31 fix (see docstring on _revision_budget_split).
    now = time.monotonic()
    deadline = now + 50.0
    split = _revision_budget_split(deadline, first_validate_elapsed_seconds=90.0, now=now)
    assert split["recheck_reserve"] == 90.0
    assert split["revision_budget_available"] == 0.0
    assert split["degraded_reserve"] is False


def test_revision_budget_split_degrades_reserve_instead_of_starving_revision():
    # Motivated by case-revision-reserve-starvation (docs/evaluations/
    # deep-research-revision-budget-reserve-starvation-fix-e2e-20260731.md): a real job with 5
    # sections + 75 claims measured first_validate_ms=227175 against the default 360s budget.
    # Before the fix, recheck_reserve == the "ideal" first-validate-cost estimate would leave
    # revision_budget_available at exactly 0.0 even when report_quality/entailment had already
    # collected real revision_issues. The fix degrades the reserve down to the configured floor
    # once the ideal reserve no longer fits, so revision gets a real (if reduced) budget.
    # 2026-08-12: the floor rose 60 -> 90 (must stay >= the critic evidence reserve's own ideal,
    # see _revision_budget_split's recheck_min_remaining comment), so this fixture's numbers are
    # adjusted from the original incident's exact values to still land in the degrade band under
    # the new floor -- the mechanism being tested (degrade, don't skip) is unchanged.
    validation_started = time.monotonic()
    deadline = validation_started + 340.0
    now = validation_started + 200.0  # first validate took 200s, remaining = 140s
    split = _revision_budget_split(deadline, first_validate_elapsed_seconds=200.0, now=now)
    assert split["recheck_reserve"] == 90.0
    assert round(split["revision_budget_available"], 3) == 50.0
    assert split["degraded_reserve"] is True


def test_revision_budget_split_never_degrades_reserve_below_the_floor():
    # Even in the degraded branch, the recheck must never be promised less than the configured
    # floor -- that floor is what keeps this fix from reopening the original regression (docs/
    # evaluations/deep-research-multi-topic-question-loop-first-run-e2e-20260725.md section 4.5:
    # revision consuming the recheck's entire budget). No matter how large first_validate_elapsed
    # is, recheck_reserve should be either the (capped) ideal estimate or exactly the floor --
    # never something in between that a slow revision could still erode below the floor.
    now = time.monotonic()
    deadline = now + 150.0
    split = _revision_budget_split(deadline, first_validate_elapsed_seconds=500.0, now=now)
    assert split["recheck_reserve"] == 90.0
    assert split["degraded_reserve"] is True
    assert round(split["revision_budget_available"], 3) == 60.0


def test_revision_budget_split_reserve_never_shrinks_when_ideal_reserve_already_fits():
    # Sanity check that the new degrade branch is inert whenever the original (pre-fix) formula
    # already had room -- i.e. this fix changes nothing for the common/healthy case. Uses an
    # elapsed cost (100s) above the RESEARCH_VALIDATION_RECHECK_MIN_REMAINING_SECONDS floor (90s)
    # so the reserve is driven by the measured cost, not by the floor.
    now = time.monotonic()
    deadline = now + 200.0
    split = _revision_budget_split(deadline, first_validate_elapsed_seconds=100.0, now=now)
    assert split["recheck_reserve"] == 100.0
    assert split["revision_budget_available"] == 200.0 - 100.0
    assert split["degraded_reserve"] is False


def test_entailment_reserve_reserves_two_sections_when_budget_generous(monkeypatch):
    # 5 sections, 300s left: ideal reserve = min(2, 5) * 30s + 10s jitter buffer = 70s, and
    # 300 - 70 = 230 is well above the 60s entailment floor -- the common/healthy case gets the
    # full ideal reserve (floor + jitter buffer).
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document(f"sec_{i:02d}") for i in range(5)]
    now = time.monotonic()
    deadline = now + 300.0
    result = research_service._entailment_deadline_with_critic_reserve(documents, deadline, now=now)
    assert result["critic_reserve"] == 70.0
    assert result["entailment_deadline"] == deadline - 70.0
    assert result["degraded_reserve"] is False


def test_entailment_reserve_degrades_to_one_section_when_tight(monkeypatch):
    # remaining=105s: ideal reserve (70s) leaves 35s < 60s entailment floor -- degrade to the
    # 1-section floor + jitter (30s + 10s = 40s), which leaves 65s >= 60s and fits.
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document(f"sec_{i:02d}") for i in range(5)]
    now = time.monotonic()
    deadline = now + 105.0
    result = research_service._entailment_deadline_with_critic_reserve(documents, deadline, now=now)
    assert result["critic_reserve"] == 40.0
    assert result["entailment_deadline"] == deadline - 40.0
    assert result["degraded_reserve"] is True


def test_entailment_reserve_gives_up_when_even_floor_starves_entailment(monkeypatch):
    # remaining=80s: even the 1-section floor + jitter (40s) leaves only 40s < 60s entailment
    # floor -- no reserve can fit without starving entailment itself, so reserve must be 0.
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document(f"sec_{i:02d}") for i in range(5)]
    now = time.monotonic()
    deadline = now + 80.0
    result = research_service._entailment_deadline_with_critic_reserve(documents, deadline, now=now)
    assert result["critic_reserve"] == 0.0
    assert result["entailment_deadline"] == deadline
    assert result["degraded_reserve"] is False


def test_entailment_reserve_caps_to_available_sections_when_fewer_than_min_sections(monkeypatch):
    # Only 1 scoreable section -- the ideal reserve should be 1*30s + 10s jitter, not 2*30s + 10s.
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document("sec_01")]
    now = time.monotonic()
    deadline = now + 300.0
    result = research_service._entailment_deadline_with_critic_reserve(documents, deadline, now=now)
    assert result["critic_reserve"] == 40.0


def test_entailment_reserve_excludes_reference_section_from_count(monkeypatch):
    # The reference/citation-index section is excluded from _section_critic_budgets too -- the
    # reserve calculation must count sections the same way so the two stay consistent.
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document("sec_01"), _document("sec_ref", title="05_证据与参考")]
    now = time.monotonic()
    deadline = now + 300.0
    result = research_service._entailment_deadline_with_critic_reserve(documents, deadline, now=now)
    assert result["critic_reserve"] == 40.0


def test_entailment_reserve_recomputed_independently_against_later_now(monkeypatch):
    # The second (post-revision) validate pass must recompute against its own `now`, not reuse
    # the first pass's result -- revision has consumed wall time in between, so `remaining` is
    # smaller the second time even though `deadline` is unchanged.
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document(f"sec_{i:02d}") for i in range(5)]
    validation_started = time.monotonic()
    deadline = validation_started + 360.0
    first_pass = research_service._entailment_deadline_with_critic_reserve(
        documents, deadline, now=validation_started
    )
    assert first_pass["critic_reserve"] == 70.0
    later_now = validation_started + 300.0  # revision ate 300s of the 360s budget
    second_pass = research_service._entailment_deadline_with_critic_reserve(
        documents, deadline, now=later_now
    )
    assert second_pass["critic_reserve"] == 0.0
    assert second_pass["entailment_deadline"] == deadline


def test_entailment_reserve_jitter_buffer_survives_near_deadline_overshoot(monkeypatch):
    """Real case-validation-reserve-jitter (RESEARCH_VALIDATION_BUDGET_SECONDS=90, 6 sections):
    the reserve computed a 20s floor (no jitter buffer, pre-fix), so entailment's deadline was
    validation_deadline - 20s. Entailment ran right up to that deadline and finished 123ms past
    it (normal scheduling/network slop -- `_validation_model` schedules its timeout right to the
    wire). By the time `_section_critic_budgets` re-checked the SAME 20s floor against what was
    actually left (19.877s), it fell just short and every section came back budget_exhausted --
    the reserve existed on paper but delivered nothing. This test reproduces that near-miss
    overshoot and confirms the jitter buffer leaves enough real headroom for at least one
    section's floor budget to still clear afterwards.

    2026-08-12: RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS's default rose 20 -> 30 (critic
    calls now carry real evidence rows), so the original job's exact 90s/6-section numbers no
    longer reserve anything at all under the new floor (ideal and degraded tiers both fail the
    60s entailment floor -- see the module docstring's three-tier degrade). Widened to 110s so
    the degraded 1-section tier (30s + 10s jitter = 40s) is the one actually exercised, keeping
    the same near-miss-overshoot scenario meaningful under the new default.
    """
    from backend.services import research_service

    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", raising=False)
    documents = [_document(f"sec_{i:02d}") for i in range(6)]
    now = time.monotonic()
    validation_deadline = now + 110.0
    reserve_info = research_service._entailment_deadline_with_critic_reserve(
        documents, validation_deadline, now=now
    )
    # Entailment finishes 123ms past its own (narrowed) deadline -- the exact real-world overshoot.
    entailment_finished_at = reserve_info["entailment_deadline"] + 0.123
    budgets = research_service._section_critic_budgets(documents, validation_deadline)
    remaining_for_critic = validation_deadline - entailment_finished_at
    section_min_budget = float(research_service.os.getenv("RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS", "30"))
    assert remaining_for_critic >= section_min_budget, (
        "jitter buffer must leave enough headroom that a same-magnitude overshoot doesn't erase "
        "the entire reserve"
    )
    # _section_critic_budgets itself takes a deadline, not a `now` override, so simulate its
    # remaining-time check directly against the post-overshoot clock to prove at least one
    # section would be schedulable in the real call path.
    even_split = remaining_for_critic / len(
        [d for d in documents if not research_service._is_reference_title(str(d.get("title") or ""))]
    )
    affordable = even_split if even_split >= section_min_budget else (
        section_min_budget if int(remaining_for_critic // section_min_budget) >= 1 else 0.0
    )
    assert affordable > 0.0


def test_validation_model_retries_once_on_transient_connection_error(monkeypatch):
    """A dropped TLS connection (proxy hiccup, SSL EOF) should get one quick retry instead of
    permanently failing that critic call -- see docs/evaluations/ 20260724 real E2E observation
    that critic calls routinely never finish even with a dedicated per-section budget, traced to
    this exact failure mode having zero retries."""
    from backend.services import research_service

    calls: list[str] = []

    class FakeResponse:
        content = '{"ok": true}'

    def flaky_then_ok(_messages, **_kwargs):
        calls.append("call")
        if len(calls) == 1:
            raise ConnectionError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
        return FakeResponse()

    monkeypatch.setattr(research_service, "invoke_research_chat", flaky_then_ok)
    diagnostics: dict = {}
    model = research_service._validation_model(time.monotonic() + 30.0, diagnostics)
    result = model({"report": "x"})
    assert result == {"ok": True}
    assert len(calls) == 2
    assert diagnostics.get("transient_connection_retries") == 1
    assert not diagnostics.get("model_failures")


def test_validation_model_does_not_retry_non_transient_errors(monkeypatch):
    """Rate-limit/other real failures must keep failing fast -- retrying those re-runs the full
    timeout and can blow the validation budget (see the max_rate_limit_retries=0 rationale)."""
    from backend.services import research_service

    calls: list[str] = []

    def always_rate_limited(_messages, **_kwargs):
        calls.append("call")
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr(research_service, "invoke_research_chat", always_rate_limited)
    diagnostics: dict = {}
    model = research_service._validation_model(time.monotonic() + 30.0, diagnostics)
    result = model({"report": "x"})
    assert result == {}
    assert len(calls) == 1
    assert not diagnostics.get("transient_connection_retries")
    assert diagnostics.get("model_failures") == 1


def test_evaluate_report_quality_partial_timeout_only_degrades_timed_out_sections():
    """The core regression this refactor targets: a validation-budget timeout must degrade only
    the sections whose critic call didn't finish, not every dimension for the whole report --
    see docs/evaluations/deep-research-dossier-utilization-and-evidence-truncation-fix-e2e-
    20260724.md item 5 ("validation 超时 critic 打分整体回落，无按章节部分降级")."""
    documents = [
        _document("sec_01", title="一、背景", content="背景正文。" * 20),
        _document("sec_02", title="二、方案", content="方案正文。" * 20),
    ]
    report_document = _report_document("## 一、背景\n\n" + documents[0]["content"] + "\n\n## 二、方案\n\n" + documents[1]["content"])

    def flaky_model(payload, timeout_override=None):
        # sec_01's content is embedded in the payload; time it out. sec_02 succeeds with a
        # real (lower) critic score for every dimension.
        if documents[0]["content"] in str(payload.get("report") or ""):
            return {}
        return {"scores": {
            "structure": 3, "synthesis": 3, "evidence_quality": 3,
            "readability": 3, "information_design": 3, "limitations": 3,
        }, "issues": [], "summary": "ok"}

    budgets = {"sec_01": 15.0, "sec_02": 15.0}
    result = evaluate_report_quality(
        report_document=report_document,
        question_states=_question_states(2),
        findings=[{"finding": "x"}] * 4,
        rejected_evidence=[],
        documents=documents,
        model=flaky_model,
        section_critic_budgets=budgets,
    )
    diagnostics = result["section_critic_diagnostics"]
    outcomes = {item["section_id"]: item["outcome"] for item in diagnostics}
    assert outcomes["sec_01"] == "model_error_or_timeout"
    assert outcomes["sec_02"] == "critic_scored"
    # sec_02's real critic score (3) must still pull dimensions down -- proof the timed-out
    # section didn't erase the successful one's contribution.
    assert result["scores"]["structure"] <= 3


def test_evaluate_report_quality_all_sections_budget_exhausted_keeps_heuristics_not_defaults():
    """Before this refactor, a fully-timed-out critic dropped every dimension to whatever the
    single joined-report call would have produced (nothing -- i.e. pure heuristic). Confirm the
    per-section path still lands on the deterministic heuristic rather than some artificial
    all-5s or all-1s default when literally no section could be scored."""
    documents = [_document("sec_01"), _document("sec_02")]
    report_document = _report_document(documents[0]["content"] + documents[1]["content"])

    def never_called(payload, timeout_override=None):
        raise AssertionError("model should not be invoked when every section is pre-scheduled as skipped")

    budgets = {"sec_01": 0.0, "sec_02": 0.0}
    result = evaluate_report_quality(
        report_document=report_document,
        question_states=_question_states(2),
        findings=[],
        rejected_evidence=[],
        documents=documents,
        model=never_called,
        section_critic_budgets=budgets,
    )
    outcomes = {item["outcome"] for item in result["section_critic_diagnostics"]}
    assert outcomes == {"budget_exhausted"}
    # Heuristic-only scores computed deterministically from report content/findings, e.g.
    # synthesis with zero findings floors to 1 per the existing rubric.
    assert result["scores"]["synthesis"] == 1


def test_carry_forward_falls_back_when_second_pass_fully_starved():
    # Real case-shared-validation-budget shape: first pass scored sections and produced real
    # issues; second (post-revision) pass ran entirely out of budget and produced none. The
    # second pass's empty issues must not silently replace the first pass's real ones.
    previous = {
        "issues": [{"section_id": "sec_01", "type": "x", "instruction": "fix x"}],
        "critic_summary": "sec_01: needs work",
        "section_critic_diagnostics": [{"section_id": "sec_01", "outcome": "critic_scored", "scores": {}}],
    }
    current = {
        "issues": [],
        "critic_summary": "",
        "section_critic_diagnostics": [{"section_id": "sec_01", "outcome": "budget_exhausted"}],
        "scores": {"structure": 4},
        "status": "passed",
    }
    result = carry_forward_stale_critic_issues(current, previous)
    assert result["issues"] == previous["issues"]
    assert result["critic_summary"] == previous["critic_summary"]
    assert result["issues_stale_pre_revision"] is True
    # Scores/status must stay the current (post-revision) pass's, never the stale one's.
    assert result["scores"] == {"structure": 4}
    assert result["status"] == "passed"


def test_carry_forward_no_op_when_second_pass_scored_something():
    # Even a partial second-pass success (some sections scored, some budget_exhausted) must not
    # be overwritten -- it's not "fully starved", it's a real (if partial) re-evaluation.
    previous = {
        "issues": [{"section_id": "sec_01", "type": "x", "instruction": "fix x"}],
        "section_critic_diagnostics": [{"section_id": "sec_01", "outcome": "critic_scored", "scores": {}}],
    }
    current = {
        "issues": [{"section_id": "sec_02", "type": "y", "instruction": "fix y"}],
        "section_critic_diagnostics": [{"section_id": "sec_02", "outcome": "critic_scored", "scores": {}}],
    }
    result = carry_forward_stale_critic_issues(current, previous)
    assert result is current
    assert "issues_stale_pre_revision" not in result


def test_carry_forward_no_op_when_first_pass_also_starved():
    # If the first pass never scored anything either, there is nothing real to fall back to --
    # carrying forward an equally-empty previous result would just relabel the same emptiness.
    previous = {
        "issues": [],
        "section_critic_diagnostics": [{"section_id": "sec_01", "outcome": "budget_exhausted"}],
    }
    current = {
        "issues": [],
        "section_critic_diagnostics": [{"section_id": "sec_01", "outcome": "budget_exhausted"}],
    }
    result = carry_forward_stale_critic_issues(current, previous)
    assert result is current


def test_carry_forward_no_op_when_no_previous_pass():
    # First-validate-only path (no revision ever ran) has no "previous" to carry forward from.
    current = {"issues": [], "section_critic_diagnostics": []}
    assert carry_forward_stale_critic_issues(current, None) is current


def test_combined_quality_gate_flags_stale_critic_issues_without_blocking():
    report_quality = {"status": "passed", "issues_stale_pre_revision": True}
    result = combined_quality_gate(
        grounding={"status": "passed"},
        validation={"status": "passed"},
        report_quality=report_quality,
    )
    assert result["approval_ready"] is True
    assert any("陈旧问题清单" in warning for warning in result.get("warnings") or [])
    assert result["diagnostics"]["critic_issues_stale_pre_revision"] is True


def test_combined_quality_gate_flags_low_confidence_pass_without_blocking():
    report_quality = {
        "status": "passed",
        "section_critic_diagnostics": [
            {"section_id": "sec_01", "outcome": "budget_exhausted"},
            {"section_id": "sec_02", "outcome": "budget_exhausted"},
            {"section_id": "sec_03", "outcome": "critic_scored", "scores": {}},
        ],
    }
    gate = combined_quality_gate(
        grounding={"status": "passed"},
        validation={"status": "passed"},
        report_quality=report_quality,
    )
    assert gate["approval_ready"] is True
    assert gate["status"] == "passed"
    assert gate["diagnostics"]["low_confidence_pass"] is True
    assert gate["diagnostics"]["critic_coverage_ratio"] < 0.5
    assert any("低置信度" in warning for warning in gate["warnings"])


def test_combined_quality_gate_no_warning_when_critic_coverage_is_high():
    report_quality = {
        "status": "passed",
        "section_critic_diagnostics": [
            {"section_id": "sec_01", "outcome": "critic_scored", "scores": {}},
            {"section_id": "sec_02", "outcome": "critic_scored", "scores": {}},
            {"section_id": "sec_03", "outcome": "budget_exhausted"},
        ],
    }
    gate = combined_quality_gate(
        grounding={"status": "passed"},
        validation={"status": "passed"},
        report_quality=report_quality,
    )
    assert "diagnostics" not in gate
    assert "warnings" not in gate


def test_combined_quality_gate_blocks_when_semantic_review_is_unavailable():
    entailment = {
        "status": "failed",
        "model_reviewed_count": 1,
        "semantic_review_unavailable_count": 2,
        "deterministic_fallback_count": 0,
    }
    gate = combined_quality_gate(
        grounding={"status": "passed"},
        validation={"status": "passed"},
        report_quality={"status": "passed"},
        entailment=entailment,
    )
    assert gate["approval_ready"] is False
    assert gate["checks"]["semantic_review_complete"] is False
    assert gate["diagnostics"]["entailment_semantic_review_unavailable_count"] == 2
    assert gate["diagnostics"]["entailment_model_reviewed_count"] == 1
    assert any("不可审批" in warning for warning in gate["warnings"])


def test_combined_quality_gate_no_entailment_fallback_warning_when_fully_model_reviewed():
    entailment = {
        "status": "passed",
        "model_reviewed_count": 3,
        "deterministic_fallback_count": 0,
    }
    gate = combined_quality_gate(
        grounding={"status": "passed"},
        validation={"status": "passed"},
        report_quality={"status": "passed"},
        entailment=entailment,
    )
    assert "diagnostics" not in gate
    assert "warnings" not in gate


def test_combined_quality_gate_blocks_historical_deterministic_fallback_results():
    gate = combined_quality_gate(
        grounding={"status": "passed"},
        validation={"status": "passed"},
        report_quality={"status": "passed"},
        entailment={
            "status": "passed",
            "model_reviewed_count": 1,
            "deterministic_fallback_count": 2,
        },
    )

    assert gate["approval_ready"] is False
    assert gate["checks"]["semantic_review_complete"] is False
    assert gate["diagnostics"]["entailment_semantic_review_unavailable_count"] == 2


def test_recheck_drops_low_confidence_issue_the_model_explicitly_refutes():
    issues = [{"section_id": "sec_01", "type": "missing_tradeoffs", "instruction": "x", "confidence": 20}]

    def model(payload):
        assert payload["issue"]["type"] == "missing_tradeoffs"
        return {"issue_confirmed": False, "reason": "not real"}

    kept, diagnostics = _recheck_low_confidence_issues(issues, {"sec_01": "正文"}, model)
    assert kept == []
    assert diagnostics == {
        "low_confidence_count": 1, "rechecked_count": 1,
        "confirmed_count": 0, "refuted_count": 1, "no_verdict_count": 0,
    }


def test_recheck_keeps_high_confidence_issue_without_calling_model():
    issues = [{"section_id": "sec_01", "type": "weak_narrative", "instruction": "x", "confidence": 90}]

    def model(payload):
        raise AssertionError("a >=70 confidence issue must not trigger a recheck call")

    kept, diagnostics = _recheck_low_confidence_issues(issues, {"sec_01": "正文"}, model)
    assert kept == issues
    assert diagnostics["low_confidence_count"] == 0
    assert diagnostics["rechecked_count"] == 0


def test_recheck_treats_missing_confidence_as_low_and_keeps_issue_on_no_verdict():
    issues = [{"section_id": "sec_01", "type": "evidence_dumping", "instruction": "x"}]

    def model(payload):
        return {}  # budget exhausted / timeout -- no verdict

    kept, diagnostics = _recheck_low_confidence_issues(issues, {"sec_01": "正文"}, model)
    assert kept == issues
    assert diagnostics["low_confidence_count"] == 1
    assert diagnostics["no_verdict_count"] == 1
    assert diagnostics["refuted_count"] == 0


def test_recheck_keeps_issue_the_model_explicitly_confirms():
    issues = [{"section_id": "sec_01", "type": "unsupported_synthesis", "instruction": "x", "confidence": 40}]

    def model(payload):
        return {"issue_confirmed": True, "reason": "real"}

    kept, diagnostics = _recheck_low_confidence_issues(issues, {"sec_01": "正文"}, model)
    assert kept == issues
    assert diagnostics["confirmed_count"] == 1
    assert diagnostics["refuted_count"] == 0


def test_evaluate_report_quality_surfaces_confidence_recheck_diagnostics():
    documents = [_document("sec_01", title="一、背景")]
    report_document = _report_document("## 一、背景\n\n" + documents[0]["content"])

    def critic_model(payload, timeout_override=None):
        if "issue" in payload:  # this is the recheck call, not the section-scoring call
            return {"issue_confirmed": False}
        return {
            "scores": {"structure": 4, "synthesis": 4, "evidence_quality": 4, "readability": 4, "information_design": 4, "limitations": 4},
            "issues": [{"section_id": "sec_01", "type": "weak_narrative", "instruction": "x", "confidence": 10}],
            "summary": "ok",
        }

    result = evaluate_report_quality(
        report_document=report_document,
        question_states=_question_states(1),
        findings=[{"finding": "x"}],
        rejected_evidence=[],
        documents=documents,
        model=critic_model,
        section_critic_budgets={"sec_01": 15.0},
    )
    assert result["issues"] == []
    assert result["critic_confidence_recheck"] == {
        "low_confidence_count": 1, "rechecked_count": 1,
        "confirmed_count": 0, "refuted_count": 1, "no_verdict_count": 0,
    }


def test_evidence_refuted_critic_issue_is_hidden_but_kept_for_audit():
    document = {
        **_document("sec_01", title="一、对比"),
        "evidence_table": [{"id": "ev_001"}],
    }
    report_document = _report_document("## 一、对比\n\n" + document["content"])
    seen_rules: list[str] = []

    def critic_model(payload, timeout_override=None):
        if payload["task"].startswith("A report critic flagged"):
            return {"issue_confirmed": False, "reason": "数字来源序号不是 evidence id"}
        seen_rules.extend(payload["rules"])
        return {
            "scores": {
                "structure": 4,
                "synthesis": 4,
                "evidence_quality": 4,
                "readability": 4,
                "information_design": 4,
                "limitations": 4,
            },
            "issues": [{
                "section_id": "sec_01",
                "type": "evidence_quality",
                "instruction": "正文 [3] 与 ev_001 不匹配。",
                "confidence": 90,
                "evidence_ids": ["ev_001"],
            }],
            "summary": "ok",
        }

    result = evaluate_report_quality(
        report_document=report_document,
        question_states=_question_states(1),
        findings=[{"finding": "x", "section_id": "sec_01"}],
        rejected_evidence=[],
        documents=[document],
        evidence_map=[{
            "id": "ev_001",
            "atomic_claim": "报告包含来源引用。",
            "raw_quote": "报告包含来源引用。",
            "source_url": "https://example.com/source",
        }],
        model=critic_model,
        section_critic_budgets={"sec_01": 15.0},
    )

    assert result["issues"] == []
    assert result["critic_refuted_issue_count"] == 1
    assert result["critic_refuted_issues"][0]["self_check"] == "refuted"
    assert result["critic_summary"] == "评论完成：保留 0 条建议；1 条证据类建议经独立复核排除。"
    assert result["critic_raw_summary"] == "一、对比: ok"
    assert result["critic_evidence_recheck"]["refuted_count"] == 1
    assert any("[N]" in rule and "不是内部 evidence id" in rule for rule in seen_rules)


def test_section_critic_evidence_carries_writer_context_from_enriched_map():
    """In `case-validator-context-parity`, the critic reviewed the Writer with LESS context than
    the Writer had. The Writer used a same-source dossier excerpt containing a deployment
    recommendation, while the critic received only a short quote without that recommendation and
    therefore flagged a correctly-grounded claim as unsupported
    and the Writer deleted a true assertion to obey it. `_entailment_evidence_with_writer_context`
    already attaches that same dossier excerpt onto each evidence row as
    `validation_context_by_section`; _section_critic_evidence must resolve it into the same
    `writer_context` field the entailment judge gets, so the reviewer sees at least as much as
    the author it is reviewing. `section_id` drives the section-scoped lookup (context is
    pinned to the section the evidence was cited in, never leaking across sections)."""
    from backend.research.quality import _section_critic_evidence

    document = {"section_id": "sec_02", "title": "对比", "evidence_table": [{"id": "ev_001"}]}
    evidence_by_id = {
        "ev_001": {
            "id": "ev_001",
            "atomic_claim": "ExampleGraph 建议为高吞吐场景启用硬件加速。",
            "cleaned_quote": "索引策略使用分层图结构",  # short quote omits deployment guidance
            "source_url": "https://example.com/aibi1",
            "validation_context_by_section": {
                "sec_02": "部署要求 可在通用处理器上运行；高吞吐场景建议启用硬件加速",
                "sec_03": "其他章节的同源上下文，不得泄漏",
            },
            "scope": "domain_specific",
            "stance": "limitation",
            "source_type": "web_article",
            "source_title": "ExampleGraph 部署指南",
        }
    }

    rows = _section_critic_evidence(document, evidence_by_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "ev_001"
    # The critic must see the fuller dossier context, not just the bare quote fragment.
    assert row["writer_context"] == "部署要求 可在通用处理器上运行；高吞吐场景建议启用硬件加速"
    # Section-scoped: the other section's context must NOT leak into this one's critic call.
    assert "其他章节" not in row["writer_context"]


def test_section_critic_evidence_writer_context_empty_when_not_enriched():
    """Plain evidence_map rows (no validation_context_by_section attached -- e.g. a unit test or a
    path that didn't run _entailment_evidence_with_writer_context) must degrade to an empty
    writer_context string, never a missing field or a KeyError, so callers that haven't been
    migrated don't crash."""
    from backend.research.quality import _section_critic_evidence

    document = {"section_id": "sec_01", "evidence_table": [{"id": "ev_001"}]}
    evidence_by_id = {"ev_001": {"id": "ev_001", "cleaned_quote": "仅 quote，无 dossier 上下文。"}}
    rows = _section_critic_evidence(document, evidence_by_id)
    assert rows[0]["writer_context"] == ""


def test_evidence_backed_issue_recheck_receives_writer_context():
    """The evidence-backed issue recheck (_recheck_evidence_backed_issues) used to see the same
    bare quote the critic did -- so even its independent second look was blind to the dossier
    context that justified the Writer's claim. It must get the same writer_context so a
    correctly-grounded claim it is re-checking isn't refuted for the same reason the critic did
    in `case-validator-context-parity`."""
    from backend.research.quality import _recheck_evidence_backed_issues

    seen_payload = {}

    def model(payload, timeout_override=None):
        seen_payload.update(payload)
        return {"issue_confirmed": False}

    issues = [{
        "section_id": "sec_02", "type": "evidence_quality",
        "instruction": "无凭据的推断", "confidence": 80, "evidence_ids": ["ev_001"],
    }]
    evidence_by_id = {
        "ev_001": {
            "id": "ev_001", "cleaned_quote": "索引策略使用分层图结构",
            "validation_context_by_section": {"sec_02": "高吞吐场景建议启用硬件加速"},
            "source_url": "https://example.com/a", "source_title": "T",
        }
    }
    _recheck_evidence_backed_issues(
        issues, {"sec_02": "正文。"}, evidence_by_id, model, parallelism=1,
    )
    cited = seen_payload.get("cited_evidence") or []
    assert cited, "recheck must be called with the cited evidence"
    assert cited[0]["writer_context"] == "高吞吐场景建议启用硬件加速"
