"""Unit tests for question_loop's evidence-annotated reading HTML.

See docs/modules/question-loop-report-evidence-reader-html-v0.1.md for the design this
implements. Two independent things need coverage:

1. render_question_loop_report_html() itself (pure function, no I/O) -- markdown chapter
   parsing, [N] citation markers resolving to source-level cards (not per-evidence-id, see
   design doc 1.1 关键修正: by the time a report reaches approve_research() the writer has
   already collapsed [[ev_001]] into [N] source-numbered references), multi-quote cards,
   unresolvable numbers left alone, the trailing "来源" chapter dropped.
2. approve_research()'s wiring -- only renders once the job is genuinely approved (决策4/5),
   and a rendering failure must never block save_knowledge_set's result or the job's terminal
   status (决策10).
"""

from __future__ import annotations

import backend.services.research_service as research_service
from backend.services.evidence_reader_service import (
    build_question_loop_citation_sources,
    render_question_loop_document_html,
    render_question_loop_documents_html,
    render_question_loop_report_html,
)
from backend.services.job_service import job_service

# ---------------------------------------------------------------------------
# render_question_loop_report_html: pure unit tests
# ---------------------------------------------------------------------------

_EVIDENCE_MAP = [
    {"id": "ev_001", "source_url": "https://a.example/1", "source_title": "来源 A", "quote": "A 站的原话一。"},
    {"id": "ev_002", "source_url": "https://a.example/1", "source_title": "来源 A", "quote": "A 站的原话二。"},
    {"id": "ev_003", "source_url": "https://b.example/2", "source_title": "来源 B", "quote": "B 站的原话。"},
]

_REPORT_CONTENT = """# 测试课题 深度研究报告

## 执行摘要

这是摘要段落，不含引用。

## 背景

背景段落引用了 A 站的说法[1]，也引用了 B 站的说法[2]。

> 当前研究已形成完整章节框架，但部分章节的公开证据广度仍不足。

## 综合结论

结论段落再次引用 A 站[1]。

## 来源

[1] 来源 A: https://a.example/1
[2] 来源 B: https://b.example/2
"""


def test_render_question_loop_report_html_converts_numbered_citations_to_source_cards():
    html = render_question_loop_report_html(title="测试课题", report_content=_REPORT_CONTENT, evidence_map=_EVIDENCE_MAP)

    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # [1] appears twice in the body (背景 + 综合结论); both must become clickable markers pointing
    # at the *same* card -- source-level grouping, not one card per occurrence (决策8).
    assert html.count('data-note="note-1"') == 2
    assert 'id="note-1"' in html
    assert 'id="note-2"' in html
    # The source-1 card must show both quotes attributed to that source (决策3: a quote list, not
    # just the first one -- otherwise the second quote's grounding would silently disappear).
    assert "A 站的原话一。" in html and "A 站的原话二。" in html
    assert "B 站的原话。" in html


def test_render_question_loop_report_html_drops_trailing_reference_chapter():
    html = render_question_loop_report_html(title="测试课题", report_content=_REPORT_CONTENT, evidence_map=_EVIDENCE_MAP)
    # The raw "[1] 来源 A: https://a.example/1" reference-list line must not be reproduced as
    # ordinary body text -- that information is superseded by the source cards, keeping it would
    # just duplicate it in a less useful form.
    assert "[1] 来源 A: https://a.example/1" not in html


def test_render_question_loop_report_html_preserves_partial_report_notice():
    """writer.py's _assemble_report emits a '> ...' blockquote when completeness is partial --
    losing it silently would hide a disclosure the report is explicitly supposed to surface."""
    html = render_question_loop_report_html(title="测试课题", report_content=_REPORT_CONTENT, evidence_map=_EVIDENCE_MAP)
    assert "当前研究已形成完整章节框架" in html


def test_render_question_loop_report_html_leaves_unresolvable_marker_as_plain_text():
    """决策9: a [N] that doesn't resolve to any source card must not fabricate one, and must not
    take down the rest of the render either."""
    content = "## 章节\n\n正文引用了一个不存在的编号[99]。\n"
    html = render_question_loop_report_html(title="t", report_content=content, evidence_map=_EVIDENCE_MAP)
    assert "[99]" in html
    assert 'data-note="note-99"' not in html


def test_render_question_loop_report_html_escapes_and_does_not_execute_injected_markup():
    content = "## 章节\n\n包含 <script>alert(1)</script> 和 & 的正文。\n"
    html = render_question_loop_report_html(title="t & <b>标题</b>", report_content=content, evidence_map=[])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "t &amp; &lt;b&gt;标题&lt;/b&gt;" in html


def test_render_question_loop_report_html_handles_empty_report_gracefully():
    html = render_question_loop_report_html(title="空报告", report_content="", evidence_map=[])
    assert html.startswith("<!DOCTYPE html>")
    assert "本次报告未产生可核查的来源引用" in html


def test_render_question_loop_document_html_preserves_global_numbers_and_filters_unused_sources():
    html = render_question_loop_document_html(
        title="单篇章节",
        document_content="本章只引用 B 站[2]。",
        evidence_map=_EVIDENCE_MAP,
    )

    assert 'data-note="note-2"' in html
    assert 'id="note-2"' in html
    assert 'id="note-1"' not in html
    assert "来源 B" in html


def test_render_question_loop_documents_html_combines_documents_and_keeps_global_numbers():
    html = render_question_loop_documents_html(
        title="完整报告",
        documents=[
            {"title": "00_总览", "content": "总览引用 A 站[1]。"},
            {"title": "01_机制", "content": "机制引用 B 站[2]。"},
        ],
        evidence_map=_EVIDENCE_MAP,
    )

    assert "00_总览" in html
    assert "01_机制" in html
    assert 'data-note="note-1"' in html
    assert 'data-note="note-2"' in html
    assert 'id="note-1"' in html
    assert 'id="note-2"' in html
    assert ".reveal{opacity:1;transform:none" in html


def test_build_question_loop_citation_sources_keeps_document_usage_and_quotes():
    evidence = [
        {
            **_EVIDENCE_MAP[0],
            "used_by_documents": ["00_总览"],
            "atomic_claim": "A 站支撑总览。",
        }
    ]
    sources = build_question_loop_citation_sources(evidence)

    assert sources == [
        {
            "number": 1,
            "title": "来源 A",
            "url": "https://a.example/1",
            "evidence": [
                {
                    "evidence_id": "ev_001",
                    "quote": "A 站的原话一。",
                    "claim": "A 站支撑总览。",
                    "used_by_documents": ["00_总览"],
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# approve_research(): wiring + failure isolation
# ---------------------------------------------------------------------------


def _fake_save_knowledge_set_impl(*, topic, **_kwargs):
    return {"topic": topic, "paths": ["/fake/kb/path.md"]}


def _make_waiting_approval_job(*, channel_id: str = "thread_question_loop_html_test") -> str:
    job = job_service.create(job_type="research", input_data={"channel_id": channel_id}, target=None)
    job_id = job["job_id"]
    job_service.update_result(
        job_id,
        status="waiting_approval",
        result={
            "quality_gate": {"approval_ready": True},
            "documents": [{"title": "章节一", "content": "内容"}],
            "report_document": {"title": "整合报告", "content": _REPORT_CONTENT},
            "evidence_map": _EVIDENCE_MAP,
            "suggested_topic": "测试课题",
            "sources": [],
        },
    )
    return job_id


def test_approve_research_writes_brief_html_for_question_loop_job(tmp_path, monkeypatch):
    """Only in approve_research() -- the one place a question_loop job actually becomes
    'succeeded' (决策4/5) -- should this HTML get rendered and written."""
    monkeypatch.setattr(research_service, "RESEARCH_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(research_service, "_save_knowledge_set_impl", _fake_save_knowledge_set_impl)

    job_id = _make_waiting_approval_job()
    research_service.approve_research(job_id, {"topic": "测试课题", "documents": [{"title": "章节一", "content": "内容"}]})

    html_path = tmp_path / research_service._safe_artifact_name(job_id) / "brief.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert 'id="note-1"' in html

    job = job_service.get(job_id)
    assert job["status"] == "succeeded"


def test_approve_research_succeeds_even_if_html_render_raises(tmp_path, monkeypatch):
    """决策10: the evidence reader HTML is a sideshow to the approval -- a rendering bug must
    never take down save_knowledge_set's result or the job's terminal status with it."""
    monkeypatch.setattr(research_service, "RESEARCH_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(research_service, "_save_knowledge_set_impl", _fake_save_knowledge_set_impl)

    def boom(**kwargs):
        raise RuntimeError("rendering exploded")

    monkeypatch.setattr(research_service, "render_question_loop_report_html", boom)

    job_id = _make_waiting_approval_job()
    saved = research_service.approve_research(
        job_id, {"topic": "测试课题", "documents": [{"title": "章节一", "content": "内容"}]}
    )

    assert saved["paths"] == ["/fake/kb/path.md"]
    job = job_service.get(job_id)
    assert job["status"] == "succeeded"
    html_path = tmp_path / research_service._safe_artifact_name(job_id) / "brief.html"
    assert not html_path.exists()


def test_approve_research_skips_generic_save_gate_after_upstream_quality_approval(monkeypatch):
    captured = {}

    def fake_save(*, topic, skip_quality_gate=False, **_kwargs):
        captured["skip_quality_gate"] = skip_quality_gate
        return {"topic": topic, "paths": ["/fake/kb/path.md"]}

    monkeypatch.setattr(research_service, "_save_knowledge_set_impl", fake_save)

    job_id = _make_waiting_approval_job()
    research_service.approve_research(
        job_id,
        {"topic": "测试课题", "documents": [{"title": "章节一", "content": "没有关键词但已通过上游质量门。"}]},
    )

    assert captured["skip_quality_gate"] is True


def test_approve_research_passes_report_title_as_knowledge_base_metadata(monkeypatch):
    captured = {}

    def fake_save(*, topic, knowledge_base_metadata=None, **_kwargs):
        captured["topic"] = topic
        captured["metadata"] = knowledge_base_metadata
        return {"topic": topic, "paths": ["/fake/kb/path.md"]}

    monkeypatch.setattr(research_service, "_save_knowledge_set_impl", fake_save)
    job_id = _make_waiting_approval_job()

    research_service.approve_research(job_id, {"topic": "测试课题"})

    assert captured["metadata"]["title"] == "整合报告"
    assert captured["metadata"]["source_job_id"] == job_id
    assert captured["metadata"]["notes"] == "由调研审批落库自动生成。"
