from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import workers
from backend.research.models import QuestionBudget, ResearchQuestionState
from backend.research.source_pool import SourceCandidate
from backend.schemas.research import ResearchJobCreateRequest
from backend.services import job_service as job_service_module
from backend.services import research_service


def _state() -> ResearchQuestionState:
    return ResearchQuestionState(
        question_id="sec_01_q1",
        section_id="sec_01",
        section_title="01_机制",
        question="系统如何工作？",
        purpose="说明机制",
        success_criteria=["说明关键步骤"],
        preferred_source_types=["official_docs"],
        budget=QuestionBudget(max_rounds=3, max_sources=6, timeout_seconds=30, max_model_calls=7),
        min_evidence=1,
        min_sources=1,
    )


def test_removed_legacy_mode_is_rejected_by_request_schema():
    with pytest.raises(ValidationError):
        ResearchJobCreateRequest(user_request="调研测试主题", pipeline_mode="legacy")


def test_successful_empty_extraction_returns_no_chunk_fallback_without_failing_job(monkeypatch):
    chunk = {
        "id": "chk_001",
        "section_id": "sec_01",
        "question_id": "sec_01_q1",
        "question": "系统如何工作？",
        "text": "候选材料不能在模型失败后自动变成证据。",
        "source_url": "https://example.com/a",
    }
    monkeypatch.setattr(workers, "_extract_single_section_evidence_model", lambda *args: ([], 1))
    monkeypatch.setattr(
        workers,
        "_extract_question_level_evidence_with_budget",
        lambda **kwargs: {
            "evidence": [],
            "stats": {
                "question_retry_count": 1,
                "question_retry_evidence_count": 0,
                "question_retry_timeout_count": 0,
                "question_retry_error_count": 0,
                "failed_question_ids": [],
                "empty_question_ids": ["sec_01_q1"],
            },
        },
    )

    result = workers._extract_single_section_evidence_with_budget(
        task="调研",
        section={"id": "sec_01", "title": "01_机制", "questions": []},
        model_chunks=[chunk],
        all_chunks=[chunk],
    )

    assert result["evidence"] == []
    assert result["stats"]["status"] == "empty"
    assert result["stats"]["retry_exhausted"] is False
    assert result["stats"]["empty_question_ids"] == ["sec_01_q1"]


def test_extraction_model_failure_after_retry_remains_exhausted(monkeypatch):
    chunk = {
        "id": "chk_001",
        "section_id": "sec_01",
        "question_id": "sec_01_q1",
        "question": "系统如何工作？",
        "text": "候选材料不能在模型失败后自动变成证据。",
        "source_url": "https://example.com/a",
    }
    monkeypatch.setattr(workers, "_extract_single_section_evidence_model", lambda *args: ([], 1))
    monkeypatch.setattr(
        workers,
        "_extract_question_level_evidence_with_budget",
        lambda **kwargs: {
            "evidence": [],
            "stats": {
                "question_retry_count": 1,
                "question_retry_evidence_count": 0,
                "question_retry_timeout_count": 0,
                "question_retry_error_count": 1,
                "failed_question_ids": ["sec_01_q1"],
                "empty_question_ids": [],
            },
        },
    )

    result = workers._extract_single_section_evidence_with_budget(
        task="调研",
        section={"id": "sec_01", "title": "01_机制", "questions": []},
        model_chunks=[chunk],
        all_chunks=[chunk],
    )

    assert result["evidence"] == []
    assert result["stats"]["status"] == "failed"
    assert result["stats"]["retry_exhausted"] is True
    assert result["stats"]["failed_question_ids"] == ["sec_01_q1"]


def test_material_extraction_failure_saves_full_material_checkpoint(monkeypatch):
    captured = {}

    class FakeWorker:
        def invoke(self, payload):
            return json.dumps({
                "sources": [],
                "materials": [],
                "selected_chunks": [],
                "evidence_map": [],
                "_diagnostics": {
                    "section_evidence_extraction": {
                        "sections": [{
                            "section_id": "sec_01",
                            "status": "failed",
                            "retry_exhausted": True,
                            "failed_question_ids": ["sec_01_q1"],
                            "empty_question_ids": [],
                            "question_retry_count": 1,
                        }]
                    }
                },
            })

    def fake_checkpoint(job_id, stage, payload):
        captured.update({"job_id": job_id, "stage": stage, "payload": payload})
        return "/tmp/evidence-checkpoint.json"

    monkeypatch.setattr(research_service, "web_worker_tool", FakeWorker())
    monkeypatch.setattr(research_service, "_save_question_loop_checkpoint", fake_checkpoint)
    document = {"url": "https://example.com/a", "title": "A", "content": "完整抓取正文"}

    with pytest.raises(research_service.EvidenceExtractionExhaustedError) as exc_info:
        research_service._extract_question_from_material_store(
            job_id="job_checkpoint",
            user_request="调研测试主题",
            strategy="standard",
            depth="normal",
            state=_state(),
            queries=["测试主题 机制"],
            round_index=1,
            candidates=[SourceCandidate(url="https://example.com/a", title="A")],
            documents=[document],
            provider_attempts=[],
        )

    assert captured["payload"]["documents"] == [document]
    assert captured["payload"]["state"]["question_id"] == "sec_01_q1"
    assert exc_info.value.details["checkpoint_path"] == "/tmp/evidence-checkpoint.json"


def test_extraction_error_is_classified_as_retryable():
    error = research_service.EvidenceExtractionExhaustedError(
        "证据抽取重试已耗尽",
        details={"checkpoint_path": "/tmp/evidence-checkpoint.json"},
    )

    classified = job_service_module._classify_job_error(error)

    assert classified == {
        "code": "evidence_extraction_exhausted",
        "message": "证据抽取重试已耗尽",
        "retryable": True,
    }
