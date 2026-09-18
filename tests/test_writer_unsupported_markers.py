"""Tests for the one-shot post-processing step that replaced the writer<->critic<->entailment
revision loop (see docs/plans/writer-critic-loop-redesign, 2026-08-17): claims
validate_claim_entailment flags as unsupported are marked in place with [待核实] rather than
rewritten or deleted, and the 6-dimension critic's output becomes a non-blocking "局限性声明"
appended to the report instead of driving revision."""

from __future__ import annotations

from backend.research.writer import _build_limitations_disclosure, _splice_unsupported_markers


def _document(section_id: str, title: str, content: str, ledger: list[dict]) -> dict:
    return {
        "title": title,
        "section_id": section_id,
        "content": content,
        "claim_ledger": ledger,
    }


def _evidence_map() -> list[dict]:
    return [
        {"id": "ev_001", "source_url": "https://a.example/1", "source_title": "来源 A"},
        {"id": "ev_002", "source_url": "https://b.example/2", "source_title": "来源 B"},
    ]


def test_splice_unsupported_markers_inserts_marker_via_claim_index():
    documents = [
        _document(
            "sec_01",
            "一",
            "LightRAG 使用双层索引结构进行检索 [1]。",
            [{
                "text": "LightRAG 使用双层索引结构进行检索",
                "rendered_text": "LightRAG 使用双层索引结构进行检索 [1]",
                "evidence_ids": ["ev_001"],
                "section_id": "sec_01",
            }],
        ),
    ]
    editorial = {"executive_summary": "摘要。", "conclusion": "结论。", "claim_ledger": []}
    entailment = {
        "judgments": [
            {
                "claim_index": 0,
                "section_id": "sec_01",
                "claim": "LightRAG 使用双层索引结构进行检索",
                "status": "unsupported",
            }
        ]
    }

    documents_out, editorial_out, content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=None,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 1
    assert "LightRAG 使用双层索引结构进行检索 [1] [待核实]" in documents_out[0]["content"]
    assert "LightRAG 使用双层索引结构进行检索 [1] [待核实]" in content
    # The claim's own sentence stays byte-identical apart from the appended marker -- it must
    # never be rewritten or deleted.
    assert "LightRAG 使用双层索引结构进行检索 [1]" in documents[0]["content"]


def test_splice_unsupported_markers_ignores_entailed_and_reasonable_inference():
    documents = [
        _document(
            "sec_01",
            "一",
            "GraphRAG 依赖社区检测算法构建图谱 [1]。",
            [{
                "text": "GraphRAG 依赖社区检测算法构建图谱",
                "rendered_text": "GraphRAG 依赖社区检测算法构建图谱 [1]",
                "evidence_ids": ["ev_001"],
                "section_id": "sec_01",
            }],
        ),
    ]
    editorial = {"executive_summary": "", "conclusion": "", "claim_ledger": []}
    entailment = {
        "judgments": [
            {"claim_index": 0, "section_id": "sec_01", "claim": "GraphRAG 依赖社区检测算法构建图谱", "status": "entailed"},
        ]
    }

    documents_out, _editorial_out, content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=None,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 0
    assert "[待核实]" not in content
    assert "[待核实]" not in documents_out[0]["content"]


def test_splice_unsupported_markers_falls_back_to_section_and_text_match_when_index_stale():
    documents = [
        _document(
            "sec_01",
            "一",
            "某系统在生产环境中被广泛采用 [1]。",
            [{
                "text": "某系统在生产环境中被广泛采用",
                "rendered_text": "某系统在生产环境中被广泛采用 [1]",
                "evidence_ids": ["ev_001"],
                "section_id": "sec_01",
            }],
        ),
    ]
    editorial = {"executive_summary": "", "conclusion": "", "claim_ledger": []}
    # claim_index deliberately wrong (out of range) -- must still resolve via section_id + text.
    entailment = {
        "judgments": [
            {
                "claim_index": 99,
                "section_id": "sec_01",
                "claim": "某系统在生产环境中被广泛采用",
                "status": "label_mismatch",
            }
        ]
    }

    documents_out, _editorial_out, _content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=None,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 1
    assert "某系统在生产环境中被广泛采用 [1] [待核实]" in documents_out[0]["content"]


def test_splice_unsupported_markers_never_silently_drops_unmatched_claim():
    """If the claim's rendered_text can't be found verbatim in the live section content (e.g. the
    ledger snapshot diverged from the content), the flag must still surface -- appended to the
    end of the section -- rather than disappearing."""
    documents = [
        _document("sec_01", "一", "这段正文已经被后续处理改写，不再包含原始措辞。", []),
    ]
    editorial = {"executive_summary": "", "conclusion": "", "claim_ledger": []}
    entailment = {
        "judgments": [
            {
                "claim_index": 0,
                "section_id": "sec_01",
                "claim": "一个找不到匹配原文的声明",
                "status": "unsupported",
            }
        ]
    }

    documents_out, _editorial_out, content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=None,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 1
    assert "[待核实] 未验证声明：一个找不到匹配原文的声明" in documents_out[0]["content"]
    assert "[待核实] 未验证声明：一个找不到匹配原文的声明" in content


def test_splice_unsupported_markers_redirects_overview_section_flag_to_editorial():
    """_assemble_report skips any document whose title contains '总览' (folded into the executive
    summary instead) and the references section (rendered from citation_map, not `content`) --
    see _assemble_report's own title-skip loop. A marker spliced into one of those documents'
    `content` would silently vanish from the delivered report, so it must be redirected to the
    editorial conclusion, which _assemble_report always renders."""
    documents = [
        _document(
            "sec_00",
            "00_总览",
            "总体结论：系统已完成量产部署 [1]。",
            [{
                "text": "总体结论：系统已完成量产部署",
                "rendered_text": "总体结论：系统已完成量产部署 [1]",
                "evidence_ids": ["ev_001"],
                "section_id": "sec_00",
            }],
        ),
    ]
    editorial = {"executive_summary": "摘要。", "conclusion": "结论。", "claim_ledger": []}
    entailment = {
        "judgments": [
            {
                "claim_index": 0,
                "section_id": "sec_00",
                "claim": "总体结论：系统已完成量产部署",
                "status": "unsupported",
            }
        ]
    }

    documents_out, editorial_out, content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=None,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 1
    # The overview document's own content is untouched -- it's never rendered into the body.
    assert "[待核实]" not in documents_out[0]["content"]
    assert "[待核实]" in editorial_out["conclusion"]
    assert "[待核实]" in content


def test_splice_unsupported_markers_handles_editorial_section():
    documents = [_document("sec_01", "一", "正文。", [])]
    editorial = {
        "executive_summary": "本报告总结了 A 与 B 的核心差异 [1]。",
        "conclusion": "综上，A 更适合小规模场景。",
        "claim_ledger": [{
            "text": "本报告总结了 A 与 B 的核心差异",
            "rendered_text": "本报告总结了 A 与 B 的核心差异 [1]",
            "evidence_ids": ["ev_001"],
            "section_id": "editorial",
        }],
    }
    entailment = {
        "judgments": [
            {
                "claim_index": 0,
                "section_id": "editorial",
                "claim": "本报告总结了 A 与 B 的核心差异",
                "status": "unsupported",
            }
        ]
    }

    _documents_out, editorial_out, content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=None,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 1
    assert "本报告总结了 A 与 B 的核心差异 [1] [待核实]" in editorial_out["executive_summary"]
    assert "本报告总结了 A 与 B 的核心差异 [1] [待核实]" in content


def test_splice_unsupported_markers_appends_limitations_section():
    documents = [_document("sec_01", "一", "正文内容 [1]。", [])]
    editorial = {"executive_summary": "摘要。", "conclusion": "结论。", "claim_ledger": []}
    entailment = {"judgments": []}
    report_quality = {
        "issues": [{"section_id": "sec_01", "type": "structure", "instruction": "结构可以更清晰"}],
        "critic_summary": "整体结构清晰，个别地方可以加强论证。",
    }

    _documents_out, _editorial_out, content, marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=editorial,
        entailment=entailment,
        report_quality=report_quality,
        user_request="测试课题",
        evidence_map=_evidence_map(),
    )

    assert marker_count == 0
    assert "## 局限性声明" in content
    assert "本节为 AI 自评，未经交叉验证" in content
    assert "结构可以更清晰" in content
    assert "整体结构清晰，个别地方可以加强论证。" in content
    # Appears after the conclusion but before the reference list.
    assert content.index("## 局限性声明") > content.index("## 综合结论")
    assert content.index("## 局限性声明") < content.index("## 来源")


def test_build_limitations_disclosure_reports_marker_density_without_threshold():
    disclosure = _build_limitations_disclosure(None, marker_count=3, total_claims=12)
    assert "本节为 AI 自评，未经交叉验证" in disclosure
    assert "3 / 12" in disclosure
    assert "25.0%" in disclosure
    assert "未设强制拦截阈值" in disclosure


def test_build_limitations_disclosure_handles_missing_report_quality_and_zero_claims():
    disclosure = _build_limitations_disclosure(None, marker_count=0, total_claims=0)
    assert "本节为 AI 自评，未经交叉验证" in disclosure
    assert "0 / 0" in disclosure


def test_build_limitations_disclosure_does_not_cut_critic_summary_mid_sentence():
    complete_sentence = "本节证据充分，但信息密度偏高。"
    dangling_fragment = "下一节仍然缺乏统领性论点"
    report_quality = {
        "critic_summary": complete_sentence * 40 + dangling_fragment,
    }

    disclosure = _build_limitations_disclosure(
        report_quality,
        marker_count=0,
        total_claims=1,
    )

    visible_summary = disclosure.split("**评论模型总体评价：** ", 1)[1].split("\n", 1)[0]
    assert len(visible_summary) <= 500
    assert visible_summary.endswith("。")
    assert dangling_fragment not in visible_summary
