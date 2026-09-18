"""Run a real Research Studio E2E probe and print a compact quality summary."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT / ".env", override=False)


def _compact_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _doc_summary(document: dict[str, Any], validation_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence_table = document.get("evidence_table") or []
    content = str(document.get("content") or "")
    metrics = document.get("evidence_metrics") or {}
    paragraph_validation = document.get("paragraph_validation") or {}
    synthesis_diagnostics = document.get("synthesis_diagnostics") or {}
    validation_doc = validation_doc or {}
    unsupported_refs = validation_doc.get("unsupported_refs") or []
    unsupported_claims = validation_doc.get("unsupported_claims") or []
    blocking_unsupported_claims = validation_doc.get("blocking_unsupported_claims") or []
    missing_inline_refs = validation_doc.get("missing_inline_refs") or []
    return {
        "title": document.get("title"),
        "content_chars": len(content),
        "has_inline_ev_refs": "[ev_" in content,
        "evidence_table_count": len(evidence_table),
        "evidence_metrics": {
            "evidence_count": metrics.get("evidence_count"),
            "source_count": metrics.get("source_count"),
            "domain_count": metrics.get("domain_count"),
            "question_count": metrics.get("question_count"),
            "used_evidence_count": metrics.get("used_evidence_count"),
            "risk_flag_count": metrics.get("risk_flag_count"),
            "risk_flags": metrics.get("risk_flags") or {},
            "weak_question_count": metrics.get("weak_question_count"),
            "composer": metrics.get("composer") or {},
            "paragraph_validation_status": metrics.get("paragraph_validation_status"),
        },
        "paragraph_validation": {
            "status": paragraph_validation.get("status"),
            "failed_group_count": paragraph_validation.get("failed_group_count"),
            "composer": paragraph_validation.get("composer") or {},
        },
        "synthesis_diagnostics": {
            "elapsed_ms": synthesis_diagnostics.get("elapsed_ms"),
            "group_count": synthesis_diagnostics.get("group_count"),
            "evidence_count": synthesis_diagnostics.get("evidence_count"),
            "source_count": synthesis_diagnostics.get("source_count"),
            "composer": synthesis_diagnostics.get("composer") or {},
            "slow_groups": (synthesis_diagnostics.get("groups") or [])[:6],
        },
        "validation": {
            "status": "passed" if not unsupported_refs and not blocking_unsupported_claims and not missing_inline_refs else "failed",
            "evidence_ref_count": validation_doc.get("evidence_ref_count"),
            "matched_evidence_ref_count": validation_doc.get("matched_evidence_ref_count"),
            "match_ratio": validation_doc.get("match_ratio"),
            "unsupported_refs": len(unsupported_refs),
            "unsupported_claims": len(unsupported_claims),
            "blocking_unsupported_claims": len(blocking_unsupported_claims),
            "unsupported_claim_audit": _unsupported_claim_audit(unsupported_claims),
            "unsupported_claim_categories": _unsupported_claim_categories(unsupported_claims),
            "missing_inline_refs": len(missing_inline_refs),
        },
        "first_evidence_row": _first_evidence_row(evidence_table),
        "content_preview": _compact_text(content),
    }


def _first_evidence_row(rows: list[Any]) -> dict[str, Any] | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        return {
            "id": row.get("id"),
            "section_title": row.get("section_title"),
            "source_domain": row.get("source_domain"),
            "used_in_document": row.get("used_in_document"),
            "risk_flags": row.get("risk_flags") or [],
            "claim": _compact_text(row.get("claim"), 120),
        }
    return None


def _unsupported_claim_audit(items: list[Any], limit: int = 6) -> list[dict[str, Any]]:
    output = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "text": _compact_text(item.get("text"), 220),
                "reason": item.get("reason"),
                "section": item.get("section"),
                "block_type": item.get("block_type"),
                "line_index": item.get("line_index"),
                "trigger_patterns": item.get("trigger_patterns") or [],
                "nearby_evidence_ids": item.get("nearby_evidence_ids") or [],
                "inherited_evidence_ids": item.get("inherited_evidence_ids") or [],
                "initial_category": item.get("initial_category"),
                "llm_review": item.get("llm_review"),
                "previous_sentences": [_compact_text(text, 140) for text in (item.get("previous_sentences") or [])],
                "next_sentences": [_compact_text(text, 140) for text in (item.get("next_sentences") or [])],
            }
        )
    return output


def _unsupported_claim_categories(items: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(((item.get("llm_review") or {}).get("category") if isinstance(item.get("llm_review"), dict) else "") or item.get("initial_category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


def _step_summary(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": step.get("id"),
            "status": step.get("status"),
            "elapsed_ms": step.get("elapsed_ms"),
            "summary": _compact_text(step.get("summary"), 140),
        }
        for step in job.get("steps") or []
    ]


def _chunk_rerank_summary(chunks: list[Any]) -> dict[str, Any]:
    rows = [item for item in chunks if isinstance(item, dict)]
    modes: dict[str, int] = {}
    semantic_scores = []
    hybrid_scores = []
    preview = []
    for item in rows:
        mode = str(item.get("rerank_mode") or "unknown")
        modes[mode] = modes.get(mode, 0) + 1
        if item.get("semantic_score") is not None:
            semantic_scores.append(float(item.get("semantic_score")))
        if item.get("hybrid_score") is not None:
            hybrid_scores.append(float(item.get("hybrid_score")))
        if len(preview) < 5:
            preview.append(
                {
                    "id": item.get("id"),
                    "question_id": item.get("question_id"),
                    "rank": item.get("rank"),
                    "rerank_mode": item.get("rerank_mode"),
                    "lexical_score": item.get("lexical_score"),
                    "semantic_score": item.get("semantic_score"),
                    "hybrid_score": item.get("hybrid_score"),
                    "source_domain": _domain(item.get("source_url")),
                    "source_title": _compact_text(item.get("source_title"), 80),
                }
            )
    return {
        "modes": modes,
        "semantic_scored_count": len(semantic_scores),
        "hybrid_scored_count": len(hybrid_scores),
        "semantic_score_range": _range(semantic_scores),
        "hybrid_score_range": _range(hybrid_scores),
        "preview": preview,
    }


def _synthesis_summary(documents: list[Any]) -> dict[str, Any]:
    rows = []
    total_elapsed = 0
    total_groups = 0
    modes: dict[str, int] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        diagnostics = document.get("synthesis_diagnostics") or {}
        elapsed_ms = int(diagnostics.get("elapsed_ms") or 0)
        total_elapsed += elapsed_ms
        total_groups += int(diagnostics.get("group_count") or 0)
        composer = diagnostics.get("composer") or {}
        for mode, count in (composer.get("modes") or {}).items():
            modes[str(mode)] = modes.get(str(mode), 0) + int(count or 0)
        rows.append({
            "title": document.get("title"),
            "elapsed_ms": elapsed_ms,
            "group_count": diagnostics.get("group_count"),
            "evidence_count": diagnostics.get("evidence_count"),
            "source_count": diagnostics.get("source_count"),
            "composer": composer,
        })
    rows.sort(key=lambda item: int(item.get("elapsed_ms") or 0), reverse=True)
    return {
        "elapsed_ms_total": total_elapsed,
        "group_count": total_groups,
        "modes": modes,
        "slow_documents": rows[:6],
    }


def _domain(url: Any) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(str(url or "")).netloc
    except Exception:
        return ""


def _range(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [round(min(values), 4), round(max(values), 4)]


def _model_execution_summary(job_id: Any, pipeline_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Prefer per-job audit traces over process-global model-pool counters."""
    model_pool = pipeline_diagnostics.get("model_pool") or {}
    preferred_provider = model_pool.get("preferred_provider") or {}
    provider_fallback = model_pool.get("provider_fallback") or {}
    fallback = {
        "source": "pipeline_diagnostics",
        "preferred_model": preferred_provider.get("model"),
        "preferred_call_count": int(preferred_provider.get("call_count") or 0),
        "preferred_failure_count": int(preferred_provider.get("failure_count") or 0),
        "preferred_rate_limit_count": int(preferred_provider.get("rate_limit_count") or 0),
        "fallback_enabled": bool(provider_fallback.get("enabled")),
        "fallback_model": provider_fallback.get("fallback_model"),
        "fallback_triggered_count": int(provider_fallback.get("triggered_count") or 0),
        "fallback_success_count": int(provider_fallback.get("success_count") or 0),
        "fallback_failure_count": int(provider_fallback.get("failure_count") or 0),
        "providers": {},
        "models": {},
    }

    audit_path = ROOT / "data" / "research_artifacts" / str(job_id or "") / "audit" / "model_calls.jsonl"
    if not audit_path.is_file():
        return fallback

    calls: list[dict[str, Any]] = []
    try:
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if isinstance(item, dict) and item.get("event") == "model_call":
                calls.append(item)
    except (OSError, ValueError, TypeError):
        return fallback
    if not calls:
        return fallback

    providers: dict[str, int] = {}
    models: dict[str, int] = {}
    preferred_calls: list[dict[str, Any]] = []
    fallback_calls: list[dict[str, Any]] = []
    for call in calls:
        provider = str(call.get("actual_provider") or call.get("requested_provider") or "unknown")
        model = str(call.get("actual_model") or (call.get("parameters") or {}).get("model") or "unknown")
        providers[provider] = providers.get(provider, 0) + 1
        models[model] = models.get(model, 0) + 1
        if provider == "preferred":
            preferred_calls.append(call)
        if "fallback" in provider:
            fallback_calls.append(call)

    def most_common_model(rows: list[dict[str, Any]]) -> str | None:
        counts: dict[str, int] = {}
        for row in rows:
            model = str(row.get("actual_model") or "").strip()
            if model:
                counts[model] = counts.get(model, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def failed(row: dict[str, Any]) -> bool:
        return str(row.get("status") or "") != "success"

    def rate_limited(row: dict[str, Any]) -> bool:
        status = str(row.get("status") or "").lower()
        error = str(row.get("error") or row.get("error_message") or "").lower()
        return "rate_limit" in status or "rate limit" in error or "429" in error

    return {
        "source": "job_audit",
        "preferred_model": most_common_model(preferred_calls) or fallback.get("preferred_model"),
        "preferred_call_count": len(preferred_calls),
        "preferred_failure_count": sum(1 for row in preferred_calls if failed(row)),
        "preferred_rate_limit_count": sum(1 for row in preferred_calls if rate_limited(row)),
        "fallback_enabled": bool(provider_fallback.get("enabled")) or bool(fallback_calls),
        "fallback_model": most_common_model(fallback_calls) or provider_fallback.get("fallback_model"),
        "fallback_triggered_count": len(fallback_calls),
        "fallback_success_count": sum(1 for row in fallback_calls if not failed(row)),
        "fallback_failure_count": sum(1 for row in fallback_calls if failed(row)),
        "providers": providers,
        "models": models,
    }


def _report_output_summary(report_document: Any) -> dict[str, Any]:
    document = report_document if isinstance(report_document, dict) else {}
    content = str(document.get("content") or "")
    return {
        "content_chars": len(content),
        "unresolved_internal_claim_ref_count": len(re.findall(r"\[claim_[0-9a-f]+\]", content)),
        "unresolved_internal_claim_group_count": content.count("[[claim_"),
        "unsupported_marker_count": content.count("[待核实]"),
    }


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    titles = [item.strip() for item in args.titles.split(",") if item.strip()]
    payload = {
        "user_request": args.user_request,
        "pipeline_mode": "question_loop",
        "plan_policy": "system_generated",
        "source_mode": "web",
        "research_strategy": args.strategy,
        "depth": args.depth,
        "output_mode": "document_set",
        "required_titles": titles,
        "max_rounds": args.max_rounds,
        "auto_save": False,
    }
    parent_job_id = str(getattr(args, "parent_job_id", "") or "").strip()
    if parent_job_id:
        payload["parent_job_id"] = parent_job_id
    return payload


def _validation_by_title(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in validation.get("per_document") or []:
        if isinstance(item, dict):
            output[str(item.get("title") or "")] = item
    return output


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    _load_dotenv()
    os.environ.setdefault("RESEARCH_PARAGRAPH_COMPOSER", "1")
    os.environ.setdefault("RESEARCH_MODEL_SECTION_EVIDENCE", "1")
    # Eval runs exist to find problems, so default to full revision-loop observability here --
    # unlike production, where this stays opt-in (real per-pass content roughly doubles each
    # checkpoint's size; see _debug_revision_content_snapshot in research_service.py). A probe
    # run's checkpoint volume is a non-issue, and without this flag a failed case's writer<->critic
    # interaction (why revision didn't converge) is unrecoverable after the fact.
    os.environ.setdefault("RESEARCH_DEBUG_REVISION_TRACE", "1")
    os.environ.setdefault("RESEARCH_CHECKPOINTS_ENABLED", "1")
    os.environ.setdefault("RESEARCH_AUDIT_ENABLED", "1")
    os.environ.setdefault("RESEARCH_MODEL_TRACE_ENABLED", "1")

    if args.via_api:
        return _run_probe_via_api(args)
    return _run_probe_via_service(args)


def _notify_child_created(args: argparse.Namespace, job_id: str) -> None:
    callback = getattr(args, "on_child_created", None)
    if callable(callback):
        callback(job_id)


def _parent_cancelled(args: argparse.Namespace) -> bool:
    parent_job_id = str(getattr(args, "parent_job_id", "") or "").strip()
    if not parent_job_id:
        return False
    try:
        from backend.services.job_service import job_service

        return job_service.get(parent_job_id).get("status") == "cancelled"
    except Exception:
        return False


def _cancel_child_job(job_id: str, reason: str) -> None:
    try:
        from backend.services.job_service import job_service

        job_service.cancel(job_id, reason)
    except Exception:
        pass


def _run_probe_via_service(args: argparse.Namespace) -> dict[str, Any]:
    from backend.services import research_service

    started = time.time()
    job = research_service.create_research_job(_build_payload(args))
    job_id = str(job["job_id"])
    _notify_child_created(args, job_id)
    last_message = ""
    deadline = time.time() + args.timeout_seconds

    while time.time() < deadline:
        if _parent_cancelled(args):
            _cancel_child_job(job_id, "父评测任务已取消")
            job = research_service.get_research_job(job_id)
            return _summarize_job(job, started)
        job = research_service.get_research_job(job_id)
        message = str((job.get("progress") or {}).get("message") or "")
        if args.verbose and message and message != last_message:
            print(f"[research-e2e] {job.get('status')} - {message}", flush=True)
            last_message = message
        if job.get("status") in {"waiting_approval", "succeeded", "failed", "cancelled"}:
            break
        time.sleep(args.poll_interval)
    else:
        job = research_service.get_research_job(job_id)
        _cancel_child_job(job_id, f"调研 E2E 超过 {args.timeout_seconds} 秒")
        raise TimeoutError(f"research job did not finish within {args.timeout_seconds}s: {job_id}")

    return _summarize_job(job, started)


def _run_probe_via_api(args: argparse.Namespace) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from backend.app import app

    client = TestClient(app)
    started = time.time()
    response = client.post("/api/v1/research/jobs", json=_build_payload(args))
    response.raise_for_status()
    envelope = response.json()
    if envelope.get("status") != "accepted":
        raise RuntimeError(f"unexpected create response: {envelope}")
    job_id = envelope["data"]["job_id"]
    _notify_child_created(args, str(job_id))
    last_message = ""
    deadline = time.time() + args.timeout_seconds

    while time.time() < deadline:
        if _parent_cancelled(args):
            _cancel_child_job(str(job_id), "父评测任务已取消")
            response = client.get(f"/api/v1/research/jobs/{job_id}")
            response.raise_for_status()
            return _summarize_job(response.json()["data"], started)
        response = client.get(f"/api/v1/research/jobs/{job_id}")
        response.raise_for_status()
        envelope = response.json()
        if envelope.get("status") != "ok":
            raise RuntimeError(f"unexpected poll response: {envelope}")
        job = envelope["data"]
        message = str((job.get("progress") or {}).get("message") or "")
        if args.verbose and message and message != last_message:
            print(f"[research-e2e-api] {job.get('status')} - {message}", flush=True)
            last_message = message
        if job.get("status") in {"waiting_approval", "succeeded", "failed", "cancelled"}:
            return _summarize_job(job, started)
        time.sleep(args.poll_interval)
    _cancel_child_job(str(job_id), f"调研 E2E 超过 {args.timeout_seconds} 秒")
    raise TimeoutError(f"research API job did not finish within {args.timeout_seconds}s: {job_id}")


def _summarize_job(job: dict[str, Any], started: float) -> dict[str, Any]:
    result = job.get("result") or {}
    validation = result.get("validation") or {}
    validation_docs = _validation_by_title(validation)
    documents = result.get("documents") or []
    retrieval = result.get("retrieval") or {}
    selected_chunks = result.get("selected_chunks") or []
    diagnostics = retrieval.get("diagnostics") or {}
    section_diag = (retrieval.get("evidence_diagnostics") or {})
    pipeline_diagnostics = result.get("pipeline_diagnostics") or {}
    summary = {
        "job_id": job.get("job_id"),
        "pipeline_mode": result.get("pipeline_mode") or "unknown",
        "writing_architecture": result.get("writing_architecture") or "unknown",
        "writing_architecture_version": result.get("writing_architecture_version"),
        "capabilities": result.get("capabilities") or {},
        "elapsed_seconds": round(time.time() - started, 1),
        "job_status": job.get("status"),
        "result_status": result.get("status"),
        "quality_gate": {
            "status": (result.get("quality_gate") or {}).get("status"),
            "approval_ready": (result.get("quality_gate") or {}).get("approval_ready"),
            "reason": _compact_text((result.get("quality_gate") or {}).get("reason")),
            "checks": (result.get("quality_gate") or {}).get("checks") or {},
        },
        "validation": {
            "status": validation.get("status"),
            "reason": _compact_text(validation.get("reason")),
            "entailment": validation.get("entailment") or {},
        },
        "report_quality": result.get("report_quality") or {},
        "report_output": _report_output_summary(result.get("report_document")),
        "pipeline_diagnostics": pipeline_diagnostics,
        "model_execution": _model_execution_summary(job.get("job_id"), pipeline_diagnostics),
        "shadow_comparison": result.get("shadow_comparison") or {},
        "question_states": [
            {
                "question_id": item.get("question_id"),
                "question": _compact_text(item.get("question"), 240),
                "question_kind": item.get("question_kind"),
                "status": item.get("status"),
                "stop_reason": item.get("stop_reason"),
                "rounds_used": item.get("rounds_used"),
                "evidence_count": len(item.get("accepted_evidence") or []),
                "rejected_evidence_count": len(item.get("rejected_evidence") or []),
                "source_requirements": item.get("source_requirements") or [],
                "gap_actions": item.get("gap_actions") or [],
                "access_issue_count": len(item.get("access_issues") or []),
                "stagnant_rounds": item.get("stagnant_rounds"),
                "budget": item.get("budget") or {},
                "round_budgets": item.get("round_budgets") or [],
                "attempted_fetch_url_count": len(item.get("attempted_fetch_urls") or []),
                "gap_execution_log": item.get("gap_execution_log") or [],
                "coverage": item.get("coverage") or {},
            }
            for item in result.get("question_states") or []
            if isinstance(item, dict)
        ],
        "counts": {
            "sources": len(result.get("sources") or []),
            "materials": len(result.get("materials") or []),
            "selected_chunks": len(selected_chunks),
            "evidence_map": len(result.get("evidence_map") or []),
            "evidence_pack": len(result.get("evidence_pack") or []),
            "documents": len(documents),
        },
        "diagnostics": {
            "chunk_rerank": _chunk_rerank_summary(selected_chunks),
            "synthesis": _synthesis_summary(documents),
            "section_evidence": {
                "section_count": section_diag.get("section_count"),
                "evidence_count": section_diag.get("evidence_count"),
                "model_evidence_count": section_diag.get("model_evidence_count"),
                "timeout_count": section_diag.get("timeout_count"),
                "error_count": section_diag.get("error_count"),
                "retry_exhausted_count": section_diag.get("retry_exhausted_count"),
                "rejected_count": section_diag.get("rejected_count"),
                "sections": [
                    {
                        "section_id": item.get("section_id"),
                        "title": item.get("title"),
                        "evidence_count": item.get("evidence_count"),
                        "source_count": item.get("source_count"),
                        "domain_count": item.get("domain_count"),
                        "weak_question_count": item.get("weak_question_count"),
                        "weak_matrix_cell_count": item.get("weak_matrix_cell_count"),
                        "matrix_rows": [
                            {
                                "matrix_cell_id": row.get("matrix_cell_id"),
                                "subject": row.get("subject"),
                                "comparison_axis": row.get("comparison_axis"),
                                "evidence_count": row.get("evidence_count"),
                                "source_count": row.get("source_count"),
                            }
                            for row in (item.get("matrix_rows") or [])[:8]
                            if isinstance(row, dict)
                        ],
                        "risk_single_source_count": item.get("risk_single_source_count"),
                    }
                    for item in (section_diag.get("sections") or [])[:6]
                    if isinstance(item, dict)
                ],
            },
            "material_pool": (diagnostics.get("material_pool") or {}),
            "global_synthesis": (diagnostics.get("global_synthesis") or {}),
        },
        "documents": [
            _doc_summary(document, validation_docs.get(str(document.get("title") or "")))
            for document in documents
            if isinstance(document, dict)
        ],
        "steps": _step_summary(job),
    }
    return summary


def summary_failed(summary: dict[str, Any]) -> bool:
    return (
        summary["job_status"] not in {"waiting_approval", "succeeded"}
        or summary["result_status"] != "waiting_approval"
        or summary["quality_gate"]["status"] != "passed"
        or summary["quality_gate"]["approval_ready"] is not True
        or summary["validation"]["status"] != "passed"
        # Production's real gate (research_service.py's quality_gate/validation) only blocks on
        # blocking_unsupported_claims -- unsupported_claims is the pre-filter candidate set before
        # _blocking_unsupported_claims() drops known false-positive categories (citation_inheritance_supported,
        # validation_false_positive, sources/evidence_index/boundary blocks). This probe used to check the
        # unfiltered unsupported_claims, which is stricter than what production actually requires for
        # approval -- see docs/evaluations/deep-research-plan-enrichment-topic-anchor-loss-fix-e2e-20260727.md
        # section 6, where a real report passed production's quality_gate/validation but was still flagged
        # failed here over a single citation_inheritance_supported false positive.
        or any(doc["validation"]["blocking_unsupported_claims"] for doc in summary["documents"])
        or any(doc["validation"]["missing_inline_refs"] for doc in summary["documents"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-request",
        default="调研 Harness Engineering 循环工程的定义和关键机制，要求内容严格来自网页证据",
    )
    parser.add_argument("--titles", default="00_总览,03_关键机制")
    parser.add_argument("--strategy", default="hv_analysis", choices=["standard", "hv_analysis"])
    parser.add_argument("--depth", default="normal", choices=["normal", "deep"])
    parser.add_argument("--max-rounds", type=int, default=1)
    # See run_research_multi_topic_probe.py: a real question_loop job can take ~1000-1100s
    # (retrieval + writing + entailment/critic/revision), so 900s reports finished jobs as
    # timeouts. 1800 -> 3600 (2026-08-12): RESEARCH_VALIDATION_BUDGET_SECONDS itself rose to
    # 2400s (critic-evidence-grounding fix added real evidence payloads + two new model-call
    # stages to validation, see docs/evaluations/deep-research-critic-evidence-grounding-
    # 20260812.md), so 1800s alone could no longer even cover the validation stage's own ceiling,
    # let alone retrieval+writing on top of it -- case-probe-running-timeout hit exactly this and
    # was misreported as a probe failure while still genuinely running. 3600s covers the new
    # validation ceiling plus retrieval/writing headroom.
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--via-api", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    summary = run_probe(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary_failed(summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
