"""End-to-end API tests for research approval into the new knowledge tree."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.services import index_service, knowledge_tree_service, research_service


client = TestClient(app)


@pytest.fixture()
def archive_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "knowledge_base"
    monkeypatch.setattr(knowledge_tree_service, "KNOWLEDGE_ROOT", root)
    job = {
        "job_id": "job_archive_001",
        "job_type": "research",
        "status": "waiting_approval",
        "updated_at": "2026-08-27 12:00:00",
        "input": {"user_request": "调研 Harness 与 Loop 的差异", "topic": None},
        "result": {
            "suggested_topic": "Harness vs Loop",
            "documents": [
                {"title": "00_总览", "content": "# Harness vs Loop\n\nHarness 负责组织执行[1]。"},
                {"title": "01_关键机制", "content": "# 关键机制\n\nLoop 负责迭代。"},
            ],
            "sources": [{"title": "Source A", "url": "https://example.com/a"}],
            "evidence_map": [
                {
                    "id": "ev_001",
                    "source_title": "Source A",
                    "source_url": "https://example.com/a",
                    "raw_quote": "Harness 负责组织执行。",
                    "atomic_claim": "Harness 负责组织执行。",
                    "used_by_documents": ["00_总览"],
                }
            ],
            "quality_gate": {"approval_ready": True},
            "report_document": {"title": "Harness vs Loop 调研"},
        },
    }

    def get_job(job_id: str):
        assert job_id == job["job_id"]
        return job

    def update_result(job_id: str, *, status=None, result=None, artifacts=None, progress_message=None):
        assert job_id == job["job_id"]
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if artifacts is not None:
            job["artifacts"] = artifacts
        return job

    monkeypatch.setattr(research_service.job_service, "get", get_job)
    monkeypatch.setattr(research_service.job_service, "update_result", update_result)
    monkeypatch.setattr(
        index_service,
        "create_research_index_job",
        lambda research_id, force=False: {
            "job_id": "job_index_001",
            "job_type": "research_index",
            "status": "queued",
            "input": {"research_id": research_id, "force": force},
        },
    )
    return root, job


def test_research_archive_preview_confirm_and_index_status(archive_context):
    root, job = archive_context
    preview_response = client.post(f"/api/v1/research/jobs/{job['job_id']}/archive-preview", json={})
    assert preview_response.status_code == 200, preview_response.text
    preview = preview_response.json()["data"]
    assert preview["state"] == "pending"
    assert preview["recommendation"]["mode"] == "new_location"
    assert preview["archive"]["knowledge_base"]["title"] == "Agent 学习"
    assert preview["archive"]["theme"]["title"] == "Harness"
    assert not (root / "Agent 学习").exists()

    confirm_response = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-confirm",
        json={"preview_id": preview["preview_id"]},
    )
    assert confirm_response.status_code == 200, confirm_response.text
    saved = confirm_response.json()["data"]
    assert saved["status"] == "saved"
    assert saved["index_job_id"] == "job_index_001"
    assert job["status"] == "succeeded"
    assert (root / saved["manifest_path"]).exists()

    document_response = client.get(
        f"/api/v1/knowledge-tree/research/{saved['research_id']}/documents/00_%E6%80%BB%E8%A7%88.md"
    )
    assert document_response.status_code == 200, document_response.text
    document = document_response.json()["data"]
    assert document["has_html"] is True
    assert document["citations"][0]["number"] == 1
    assert document["citations"][0]["evidence"][0]["quote"] == "Harness 负责组织执行。"

    html_response = client.get(
        f"/api/v1/knowledge-tree/research/{saved['research_id']}/documents/00_%E6%80%BB%E8%A7%88.md/html"
    )
    assert html_response.status_code == 200, html_response.text
    assert 'data-note="note-1"' in html_response.text

    full_html_response = client.get(f"/api/v1/knowledge-tree/research/{saved['research_id']}/reader")
    assert full_html_response.status_code == 200, full_html_response.text
    assert "00_总览" in full_html_response.text
    assert "01_关键机制" in full_html_response.text
    assert 'data-note="note-1"' in full_html_response.text

    index_response = client.get(f"/api/v1/knowledge-tree/research/{saved['research_id']}/index")
    assert index_response.status_code == 200
    index_status = index_response.json()["data"]["index_status"]
    assert index_status["status"] == "pending"
    assert index_status["job_id"] == "job_index_001"


def test_archive_preview_rejects_quality_gate_failure(archive_context):
    _root, job = archive_context
    job["result"]["quality_gate"] = {"approval_ready": False, "reason": "引用不足"}
    response = client.post(f"/api/v1/research/jobs/{job['job_id']}/archive-preview", json={})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "引用不足"


def test_archive_quality_acknowledgement_is_not_recorded_for_passing_gate(archive_context):
    _root, job = archive_context

    response = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-preview",
        json={"acknowledge_quality_warnings": True},
    )

    assert response.status_code == 200, response.text
    quality_review = response.json()["data"]["archive"]["quality_review"]
    assert quality_review["automatic_approval_ready"] is True
    assert quality_review["human_acknowledged"] is False
    assert quality_review["acknowledged_at"] is None


def test_archive_quality_warning_acknowledgement_is_explicit_and_auditable(
    archive_context,
    monkeypatch: pytest.MonkeyPatch,
):
    root, job = archive_context
    quality_gate = {
        "status": "needs_more_evidence",
        "approval_ready": False,
        "reason": "语义复核未全部完成",
        "warnings": ["3 条声明待人工核实", "部分语义复核因渠道压力未完成"],
        "checks": {"semantic_review_complete": False},
    }
    job["result"]["quality_gate"] = quality_gate
    original_quality_gate = deepcopy(quality_gate)
    monkeypatch.setattr(research_service, "now_text", lambda: "2026-09-10 18:00:00")

    rejected = client.post(f"/api/v1/research/jobs/{job['job_id']}/archive-preview", json={})
    assert rejected.status_code == 400
    assert rejected.json()["error"]["message"] == "语义复核未全部完成"

    job["status"] = "running"
    invalid_state = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-preview",
        json={"acknowledge_quality_warnings": True},
    )
    assert invalid_state.status_code == 400
    job["status"] = "waiting_approval"

    preview_response = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-preview",
        json={"acknowledge_quality_warnings": True},
    )
    assert preview_response.status_code == 200, preview_response.text
    preview = preview_response.json()["data"]
    assert preview["archive"]["quality_review"] == {
        "automatic_status": "needs_more_evidence",
        "automatic_approval_ready": False,
        "reason": "语义复核未全部完成",
        "warnings": ["3 条声明待人工核实", "部分语义复核因渠道压力未完成"],
        "human_acknowledged": True,
        "acknowledged_at": "2026-09-10 18:00:00",
    }

    confirm_response = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-confirm",
        json={"preview_id": preview["preview_id"]},
    )
    assert confirm_response.status_code == 200, confirm_response.text
    saved = confirm_response.json()["data"]
    manifest = json.loads((root / saved["manifest_path"]).read_text(encoding="utf-8"))

    assert saved["index_job_id"] == "job_index_001"
    assert manifest["source_job_id"] == job["job_id"]
    assert manifest["quality_review"] == preview["archive"]["quality_review"]
    assert job["result"]["quality_gate"] == original_quality_gate


def test_partial_comparison_archive_requires_ack_and_persists_gaps(archive_context):
    root, job = archive_context
    job["result"]["comparison_budget"] = {"max_required_cells": 12}
    job["result"]["matrix_coverage"] = {
        "enabled": True,
        "status": "partial",
        "complete": False,
        "requires_gap_acknowledgement": True,
        "gaps": [{"cell_id": "cell_01", "subject": "Harness", "axis": "机制", "status": "exhausted"}],
    }

    rejected = client.post(f"/api/v1/research/jobs/{job['job_id']}/archive-preview", json={})
    assert rejected.status_code == 400
    assert "Matrix Gap" in rejected.json()["error"]["message"]

    preview = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-preview",
        json={"acknowledge_matrix_gaps": True},
    ).json()["data"]
    saved = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-confirm",
        json={"preview_id": preview["preview_id"]},
    ).json()["data"]
    manifest = json.loads((root / saved["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["matrix_gaps_acknowledged"] is True
    assert manifest["matrix_coverage"]["gaps"][0]["status"] == "exhausted"


def test_research_index_retry_endpoint_uses_stable_id(archive_context, monkeypatch: pytest.MonkeyPatch):
    root, job = archive_context
    preview = client.post(f"/api/v1/research/jobs/{job['job_id']}/archive-preview", json={}).json()["data"]
    saved = client.post(
        f"/api/v1/research/jobs/{job['job_id']}/archive-confirm",
        json={"preview_id": preview["preview_id"]},
    ).json()["data"]

    calls = []

    def create_job(research_id: str, *, force: bool = False):
        calls.append((research_id, force))
        return {"job_id": "job_retry_001", "job_type": "research_index", "status": "queued"}

    monkeypatch.setattr(index_service, "create_research_index_job", create_job)
    response = client.post(
        f"/api/v1/knowledge-tree/research/{saved['research_id']}/index/jobs",
        json={"force": True, "mode": "lightrag"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "accepted"
    assert calls == [(saved["research_id"], True)]
    assert (root / saved["manifest_path"]).exists()
