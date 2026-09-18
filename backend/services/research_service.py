"""Research job service."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
import openai

from ..research.audit import (
    audit_run,
    audit_scope,
    code_version,
    configuration_summary,
    invoke_with_model_trace,
    payload_audit_scope,
    record_parse_result,
    submit_with_audit_context,
    write_stage_checkpoint,
)
from ..research.dossier import attach_section_dossiers
from ..research.evidence import (
    evidence_semantic_arbitration_mode,
    evidence_support_relation,
    evidence_target_fit,
    is_direct_exact_evidence,
    semantic_relation_allows_criterion,
)
from ..research.evidence_native import (
    ARCHITECTURE as EVIDENCE_NATIVE_ARCHITECTURE,
    ARCHITECTURE_VERSION as EVIDENCE_NATIVE_ARCHITECTURE_VERSION,
    CONTEXT_VERSION as EVIDENCE_NATIVE_CONTEXT_VERSION,
    PROMPT_VERSION as EVIDENCE_NATIVE_PROMPT_VERSION,
    EvidenceNativeCheckpointStore,
    architecture_capabilities as evidence_native_capabilities,
    build_evidence_native_blueprint,
    stable_hash as evidence_native_hash,
    write_evidence_native_report,
)
from ..research.model_runtime import ProviderFallbackExhaustedError, invoke_research_chat, model_pool_diagnostics
from ..research.material_store import JobMaterialStore
from ..research.models import (
    CRITERION_LEVEL_POLICY_V1,
    ComparisonBudget,
    ResearchQuestionState,
    question_states_from_plan,
)
from ..research.orchestrator import QuestionResearchOrchestrator
from ..research.quality import (
    ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
    classify_unused_dossier_evidence,
    combined_quality_gate,
    evaluate_report_quality,
    validate_claim_entailment,
    validate_report_citations,
)
from ..research.synthesis import build_report_blueprint as build_question_report_blueprint
from ..research.synthesis import (
    FindingSynthesisError,
    classify_evidence_landscape,
    review_finding_admission,
    synthesize_findings,
)
from ..research.source_policy import compile_research_query, normalize_source_type, source_query
from ..research.source_pool import GlobalSourcePool, SourceCandidate, normalize_url
from ..research.text_utils import extract_key_terms, is_non_research_section_title
from ..research.writer import (
    _is_reference_title,
    _splice_unsupported_markers,
    revise_report_sections,
    write_report as write_question_report,
)

try:
    from ...tools.knowledge import _save_knowledge_set_impl, research_coverage_check
    from ...tools.web import broad_discovery, fetch_context7_context, fetch_url_with_chain
    from ...workers import web_worker_tool
    from ...observability import emit_event
except ImportError:  # pragma: no cover
    from tools.knowledge import _save_knowledge_set_impl, research_coverage_check
    from tools.web import broad_discovery, fetch_context7_context, fetch_url_with_chain
    from workers import web_worker_tool
    from observability import emit_event

from .common import invoke_tool, now_text, parse_tool_json
from .evidence_reader_service import (
    build_question_loop_citation_sources,
    render_question_loop_document_html,
    render_question_loop_documents_html,
    render_question_loop_report_html,
    render_single_url_brief_html,
)
from .job_service import job_service
from . import index_service, knowledge_tree_service


STANDARD_TITLES = ["00_总览", "01_核心概念", "02_实操使用指南"]
HV_ANALYSIS_TITLES = ["00_总览", "01_纵向演化", "02_横向对比", "03_关键机制", "04_实操路径", "05_证据与参考"]
RESEARCH_STEPS = [
    ("plan", "规划调研"),
    ("query_plan", "生成查询计划"),
    ("search", "检索来源"),
    ("fetch", "抓取材料"),
    ("material_pool", "建立素材池"),
    ("chunk_select", "筛选关键片段"),
    ("evidence", "整理证据"),
    ("coverage", "覆盖度检查"),
    ("draft", "生成文档"),
    ("validation", "证据验收"),
    ("approval", "等待审批"),
]
ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_ARTIFACT_DIR = ROOT / "data" / "research_artifacts"


def _research_writing_architecture(input_data: dict[str, Any]) -> str:
    raw = str(
        input_data.get("writing_architecture")
        or os.getenv("RESEARCH_WRITING_ARCHITECTURE", EVIDENCE_NATIVE_ARCHITECTURE)
    ).strip().lower()
    aliases = {
        "finding": "finding",
        "finding_mediated": "finding",
        "eplus": EVIDENCE_NATIVE_ARCHITECTURE,
        "evidence_native": EVIDENCE_NATIVE_ARCHITECTURE,
        EVIDENCE_NATIVE_ARCHITECTURE: EVIDENCE_NATIVE_ARCHITECTURE,
    }
    if raw not in aliases:
        raise ValueError(
            "unsupported research writing architecture: "
            f"{raw}; expected finding or {EVIDENCE_NATIVE_ARCHITECTURE}"
        )
    return aliases[raw]


class EvidenceExtractionExhaustedError(RuntimeError):
    code = "evidence_extraction_exhausted"
    retryable = True

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def _write_question_loop_checkpoint(job_id: str, stage: str, payload: dict[str, Any]) -> str:
    """Persist a question-loop stage snapshot for inspection and targeted retry input."""
    if os.getenv("RESEARCH_CHECKPOINTS_ENABLED", "1") != "1":
        return ""
    checkpoint_dir = RESEARCH_ARTIFACT_DIR / _safe_artifact_name(job_id) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{_safe_artifact_name(stage)}.json"
    temporary = path.with_suffix(".json.tmp")
    audit_checkpoint_path = write_stage_checkpoint(
        stage,
        outputs={"question_loop_snapshot": payload},
        transition_reason="question-loop business checkpoint persisted",
        job_id=job_id,
    )
    document = {
        "job_id": job_id,
        "stage": stage,
        "created_at": now_text(),
        "pipeline_mode": "question_loop",
        "payload": payload,
    }
    if audit_checkpoint_path:
        document["schema_version"] = "question-loop-checkpoint/v2"
        document["audit_checkpoint_path"] = audit_checkpoint_path
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return str(path)


def _save_question_loop_checkpoint(job_id: str, stage: str, payload: dict[str, Any]) -> str:
    try:
        return _write_question_loop_checkpoint(job_id, stage, payload)
    except Exception:
        return ""


def _save_revision_pass_snapshot(job_id: str, pass_number: int, data: dict[str, Any]) -> str:
    """Persist one revision-loop pass's input issues + recheck output.

    Debugging aid only (2026-08-11): before this, only the first-pass and final entailment
    judgments were ever visible (`first_pass_entailment` was a plain local variable, discarded
    once the loop finished) -- there was no way to tell whether a claim that survives all the
    way to the final quality_gate reason was flagged-but-missed in an earlier pass (evidence of
    entailment judgment noise across passes) or genuinely never caught until the last pass. Each
    call writes checkpoints/revision_pass_{N}.json so a real job's full revision history can be
    replayed pass-by-pass instead of only comparing the first and last snapshot. Never read by
    production code -- swallow write failures the same way _save_question_loop_checkpoint does.
    """
    return _save_question_loop_checkpoint(job_id, f"revision_pass_{pass_number}", data)


def _debug_revision_content_snapshot(
    documents: list[dict[str, Any]], editorial: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Optional, expensive addition to a revision-pass snapshot: each section's actual content
    and per-pass accept/reject outcome for this pass (rationale included when the writer was
    asked for one -- see RESEARCH_DEBUG_REVISION_TRACE in writer.py::_model_section).

    Gated behind RESEARCH_DEBUG_REVISION_TRACE=1 (2026-08-11, see docs/evaluations/deep-research-
    revision-loop-critic-writer-transparency-20260811-handoff.md): `_save_revision_pass_snapshot`
    already always records issues/diagnostics, but never the section prose itself, and
    `revision_diagnostics` (accepted/rejected_reason/retried) gets overwritten by `written[
    "documents"]` every pass, so only the LAST pass's accept/reject history survives to
    validation.json. This makes pass-1/pass-2's intermediate content and outcomes visible without
    changing that always-on default -- real per-pass content roughly doubles each snapshot's
    size, so it stays opt-in rather than becoming the production default until proven useful on a
    real job.

    editorial (executive summary + conclusion) goes through a separate function (`_edit_report`,
    not `_model_section`), so its own rationale lives in `editor_notes` rather than
    `revision_rationale` -- a real job (case-editorial-rationale-routing) initially showed editorial
    issues being fixed with zero rationale captured, which looked like "editorial wasn't routed"
    until checking `revalidate_section_ids` proved it was routed and revised; the rationale was
    just never asked for on this second code path. Pass `editorial` explicitly to close that gap.
    """
    if os.getenv("RESEARCH_DEBUG_REVISION_TRACE") != "1":
        return {}
    result = {
        "documents_content": [
            {
                "section_id": document.get("section_id"),
                "title": document.get("title"),
                "content": document.get("content"),
                "revision_diagnostics": document.get("revision_diagnostics"),
                "revision_rationale": document.get("revision_rationale"),
            }
            for document in documents
        ],
    }
    if editorial:
        result["editorial_content"] = {
            "executive_summary": editorial.get("executive_summary"),
            "conclusion": editorial.get("conclusion"),
            "editor_notes": editorial.get("editor_notes"),
        }
    return result


_TRANSIENT_CONNECTION_ERROR_MARKERS = (
    "connecterror",
    "connectionreset",
    "connectionerror",
    "readerror",
    "remoteprotocolerror",
    "eof occurred",
    "connection reset",
    "connection aborted",
)


def _is_transient_connection_error(exc: Exception) -> bool:
    """A dropped TCP/TLS connection (proxy hiccup, SSL EOF mid-handshake), not a slow
    response or a rate limit. These fail almost instantly -- unlike a rate-limit retry
    (which re-runs the full timeout and can blow the validation budget, see the
    `max_rate_limit_retries=0` comment below), one extra attempt here costs at most a
    couple of seconds, so it's safe to retry even under a tight remaining budget.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_CONNECTION_ERROR_MARKERS)


def _validation_model(
    deadline: float,
    diagnostics: dict[str, Any],
    *,
    temperature: float = 0,
    system_prompt: str | None = None,
    model_name: str | None = None,
    evidence: bool = True,
    budget_exhausted_key: str = "budget_exhausted",
):
    # validate_claim_entailment() already calls this closure concurrently (parallelism=3 across
    # sections), and _run_section_critics() now does too -- the counter increments below
    # (`diagnostics["model_calls"] = ... + 1` etc.) are a classic unlocked read-modify-write race
    # when hit from multiple threads at once. This never corrupted report content (each call's
    # own `invoke_research_chat` result stays thread-local), only the aggregate diagnostic
    # counters, which could silently undercount -- a lock costs nothing on the hot path (the
    # network call itself dominates) and makes those counters trustworthy again.
    diagnostics_lock = threading.Lock()

    def invoke(payload: dict[str, Any], timeout_override: float | None = None) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            with diagnostics_lock:
                diagnostics[budget_exhausted_key] = True
                diagnostics["skipped_model_calls"] = int(diagnostics.get("skipped_model_calls") or 0) + 1
            return {}
        timeout = min(float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")), remaining)
        if timeout_override is not None:
            # Callers that split the shared validation budget across independent units of work
            # (e.g. per-section critic scoring, see _section_critic_budgets) pass a tighter cap
            # so one slow section can't eat another section's share of the remaining budget.
            timeout = min(timeout, max(1.0, timeout_override))
        with diagnostics_lock:
            diagnostics["model_calls"] = int(diagnostics.get("model_calls") or 0) + 1
        retry_parent_call_id: str | None = None
        for attempt in range(2):
            remaining_for_attempt = deadline - time.monotonic()
            try:
                with payload_audit_scope(payload, fallback_stage="validation"):
                    response = invoke_research_chat(
                        [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                        timeout_seconds=max(1.0, min(timeout, remaining_for_attempt)),
                        # A single validation-budget call must never exceed `remaining`: retrying on
                        # rate limit would re-run the full timeout again and can blow the budget (this
                        # is what produced the observed 214s-vs-180s overshoot). Transient connection
                        # drops are retried once regardless -- see _is_transient_connection_error.
                        max_rate_limit_retries=0,
                        # This validation wrapper already owns its one explicit connection retry
                        # and its diagnostic counter; avoid stacking the shared runtime retry.
                        max_connection_retries=0,
                        temperature=temperature,
                        system_prompt=system_prompt,
                        model_name=model_name,
                        evidence=evidence,
                        allow_provider_fallback=True,
                        fallback_stage="validation",
                        deadline_monotonic=deadline,
                        parent_call_id=retry_parent_call_id,
                        call_kind="retry" if attempt else "validation",
                    )
            except Exception as exc:
                retry_parent_call_id = str(
                    getattr(exc, "harness_call_id", "") or retry_parent_call_id or ""
                ) or None
                # Transient connection drops (TCP/TLS hiccup) fail almost instantly, so one extra
                # attempt costs at most a couple of seconds -- safe to retry regardless of budget.
                if attempt == 0 and _is_transient_connection_error(exc) and remaining_for_attempt > 2.0:
                    with diagnostics_lock:
                        diagnostics["transient_connection_retries"] = (
                            int(diagnostics.get("transient_connection_retries") or 0) + 1
                        )
                    continue
                # APITimeoutError / RateLimitError also collapse a whole entailment batch to the
                # keyword-matching _deterministic_entailment at once -- see job_20260804_064115:
                # sec_03's first batch of 6 inference claims timed out and 4 of them (direct
                # translations of the cited evidence, which the model would have called entailed)
                # came back as false label_mismatch. Retry once when the validation budget still
                # has room for a second full-timeout attempt; a retry re-runs the whole timeout,
                # so when remaining is thin we DON'T retry (the 214s-vs-180s overshoot that
                # motivated max_rate_limit_retries=0 still applies) and degrade to fallback
                # instead of stealing the slice reserved for later sections.
                if attempt == 0 and _is_retryable_model_error(exc) and remaining_for_attempt > timeout * 2 + 5.0:
                    with diagnostics_lock:
                        diagnostics["timeout_retries"] = int(diagnostics.get("timeout_retries") or 0) + 1
                    continue
                with diagnostics_lock:
                    diagnostics["model_failures"] = int(diagnostics.get("model_failures") or 0) + 1
                    diagnostics["last_error"] = type(exc).__name__
                    if time.monotonic() >= deadline:
                        diagnostics[budget_exhausted_key] = True
                return {}
            parsed = _json_from_text(str(getattr(response, "content", response) or ""))
            call_id = str((getattr(response, "response_metadata", None) or {}).get("harness_call_id") or "") or None
            record_parse_result(
                call_id,
                parsed_result=parsed or None,
                status="success" if parsed else "parse_error",
            )
            return parsed
        return {}

    return invoke


def _is_retryable_model_error(exc: Exception) -> bool:
    """APITimeoutError and RateLimitError are transient provider-side failures whose only
    alternative is degrading a whole entailment batch to keyword matching. They are retried
    once by the caller -- but ONLY when the validation budget can absorb a second full-timeout
    attempt (the retry budget gate lives in invoke(), not here). Keep this narrower than the
    SDK's own retry surface: connection-level drops go through _is_transient_connection_error
    instead (cheaper retry, no full timeout re-run)."""
    return isinstance(exc, (openai.APITimeoutError, openai.RateLimitError))


def _section_critic_budgets(documents: list[dict[str, Any]], deadline: float) -> dict[str, float]:
    """Split whatever validation budget remains evenly across sections for critic scoring.

    Sequential per-section critic calls would otherwise use a first-come-first-served remaining
    budget (like the rest of validation), letting early sections exhaust the deadline and leaving
    every later section skipped -- see docs/evaluations/deep-research-dossier-utilization-and-
    evidence-truncation-fix-e2e-20260724.md item 5. An even split degrades more predictably: if
    the per-section share would fall below a floor (RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS,
    default 30s -- below this a model call is unlikely to ever finish), only the first K sections
    that can afford the floor get a real critic call; the rest are pre-scheduled as skipped
    (budget 0.0) rather than each getting a token sliver of time that can't complete anyway.
    """
    # The reference/citation-index section's content is discarded by _assemble_report and never
    # shown to the reader (see quality.py's _is_harmless_reference_fallback), so it shouldn't
    # consume a budget slot or count toward the even split -- quality.py's critic loop skips it
    # via the same _is_reference_title check.
    section_ids = [
        str(document.get("section_id") or "")
        for document in documents
        if not _is_reference_title(str(document.get("title") or ""))
    ]
    if not section_ids:
        return {}
    remaining = max(0.0, deadline - time.monotonic())
    # 20 -> 30 (2026-08-12): critic calls now carry the section's real evidence rows (quote/
    # scope/stance/source) alongside the rendered prose, so a section with many evidence entries
    # needs more time to reason over than the bare-prose-only baseline this floor was tuned for.
    min_budget = max(1.0, float(os.getenv("RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS", "30")))
    even_split = remaining / len(section_ids)
    if even_split >= min_budget:
        return {section_id: even_split for section_id in section_ids}
    affordable = int(remaining // min_budget)
    return {
        section_id: (min_budget if index < affordable else 0.0)
        for index, section_id in enumerate(section_ids)
    }


def _documents_with_editorial_pseudo_document(
    documents: list[dict[str, Any]], editorial: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Return `documents` plus a synthetic {section_id, title, content} entry for editorial
    (executive summary + conclusion) so `_run_section_critics`/`evaluate_report_quality` -- which
    only ever reads `documents` and only knows the `section_id`/`title`/`content` shape -- can
    score editorial too. editorial itself has no single `content` field (it's split into
    `executive_summary`/`conclusion`), so the two are joined here purely for the critic call.

    This is intentionally local to each `evaluate_report_quality` call site -- it must NOT be
    assigned back into `written`/`result["documents"]`, which every other reader (report assembly,
    knowledge-base save, citation validation) expects to contain only real sec_XX report
    sections. See docs/modules/deep-research-step4-approval-and-delivery-discussion-v0.1.md 商榷点四.

    Also attaches an `evidence_table` stub (id-only rows) built from editorial's own
    `claim_ledger`, mirroring what a real section document carries -- `evaluate_report_quality`'s
    critic call (quality.py `_section_critic_evidence`) resolves these ids against `evidence_map`
    to give the critic the same raw evidence real sections get. Without this, editorial would be
    the only critic-scored "section" with no evidence grounding at all.
    """
    editorial = editorial or {}
    content = "\n\n".join(
        part for part in (str(editorial.get("executive_summary") or "").strip(), str(editorial.get("conclusion") or "").strip()) if part
    )
    editorial_diagnostics = editorial.get("diagnostics") if isinstance(editorial.get("diagnostics"), dict) else {}
    if not content and str(editorial_diagnostics.get("mode") or "") != "model_failed":
        return documents
    editorial_evidence_ids = {
        str(evidence_id)
        for row in editorial.get("claim_ledger") or []
        if isinstance(row, dict)
        for evidence_id in row.get("evidence_ids") or []
        if str(evidence_id)
    }
    return [*documents, {
        "section_id": "editorial",
        "title": "执行摘要与结论",
        "content": content,
        "evidence_table": [{"id": evidence_id} for evidence_id in editorial_evidence_ids],
        "synthesis_diagnostics": editorial_diagnostics,
    }]


def _writer_generation_failures(
    documents: list[dict[str, Any]], editorial: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Return model-generation failures that make a production report non-deliverable."""
    failures = []
    for document in documents:
        diagnostics = document.get("synthesis_diagnostics") if isinstance(document.get("synthesis_diagnostics"), dict) else {}
        mode = str(diagnostics.get("mode") or "")
        if mode == "evidence_gap":
            reportability = diagnostics.get("reportability") if isinstance(diagnostics.get("reportability"), dict) else {}
            valid_gap = (
                bool(str(document.get("content") or "").strip())
                and str(diagnostics.get("gap_reason") or "") == "no_reportable_findings_or_evidence"
                and str(reportability.get("status") or "") == "empty"
                and int(diagnostics.get("finding_count") or 0) == 0
                and int(diagnostics.get("evidence_count") or 0) == 0
                and not (document.get("evidence_table") or [])
                and not (document.get("claim_ledger") or [])
            )
            if not valid_gap:
                failures.append({
                    "section_id": str(document.get("section_id") or ""),
                    "title": str(document.get("title") or ""),
                    "failure_reason": "invalid_evidence_gap",
                    "model_attempts": int(diagnostics.get("model_attempts") or 0),
                })
            continue
        if mode != "model_failed":
            continue
        if _is_reference_title(str(document.get("title") or "")) and str(
            diagnostics.get("fallback_reason") or ""
        ) == "no_findings_or_evidence":
            continue
        failures.append({
            "section_id": str(document.get("section_id") or ""),
            "title": str(document.get("title") or ""),
            "failure_reason": str(diagnostics.get("fallback_reason") or "unknown"),
            "model_attempts": int(diagnostics.get("model_attempts") or 0),
        })
    editorial_diagnostics = (editorial or {}).get("diagnostics") if isinstance((editorial or {}).get("diagnostics"), dict) else {}
    if str(editorial_diagnostics.get("mode") or "") == "model_failed":
        failures.append({
            "section_id": "editorial",
            "title": "执行摘要与结论",
            "failure_reason": str(
                editorial_diagnostics.get("retry_failure_reason")
                or editorial_diagnostics.get("upstream_failure_reason")
                or "empty_editorial"
            ),
            "failed_fields": [str(value) for value in editorial_diagnostics.get("failed_fields") or []],
        })
    return failures


def _admit_research_findings(
    question_states: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    *,
    model: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the independent finding reviewer before any finding reaches the writer."""
    admission = review_finding_admission(
        question_states,
        findings,
        model=model,
        parallelism=max(1, int(os.getenv("RESEARCH_FINDING_ADMISSION_PARALLELISM", "3"))),
    )
    if admission.get("status") != "passed":
        raise FindingSynthesisError(list(admission.get("failures") or [{
            "section_id": "",
            "question_id": "",
            "failure_reason": "finding_admission_failed",
        }]))
    return list(admission.get("admitted_findings") or []), admission


def _revision_budget_split(
    validation_deadline: float,
    first_validate_elapsed_seconds: float,
    *,
    now: float,
    revision_min_remaining: float | None = None,
) -> dict[str, float]:
    """Reserve enough of the remaining validation budget for the post-revision recheck so a slow
    revision pass can't starve it (docs/evaluations/deep-research-multi-topic-question-loop-
    first-run-e2e-20260725.md section 4.5: revision alone consumed 60-70% of the shared 180s
    budget in two real cases, leaving the recheck ~0s and forcing claim_entailment into its
    deterministic fallback for nearly every claim). Uses the first validate pass's own measured
    cost as the recheck estimate -- it's the same computation on the same-sized report, so it's a
    better predictor than a fixed constant.

    That "ideal" reserve (recheck_reserve == first_validate_elapsed_seconds, floored) starves
    revision entirely once the first validate pass itself is slow enough to approach the total
    budget -- see docs/evaluations/deep-research-revision-budget-reserve-starvation-fix-e2e-
    20260731.md: a real job (5 sections, 75 claims) had first_validate take 227s out of a 360s
    budget, leaving only 133s remaining -- less than the 227s the ideal reserve wants -- so
    revision_budget_available floored at 0.0 and 8 real, non-empty revision_issues never got
    applied. The report shipped to the user exactly as the (flawed) first draft.

    Fix: when the *ideal* reserve doesn't fit, degrade it down to the bare
    RESEARCH_VALIDATION_RECHECK_MIN_REMAINING_SECONDS floor instead of refusing to run revision
    at all. A recheck capped at the floor may not finish every model call (some claims/sections
    can still fall back to deterministic scoring -- see
    _annotate_entailment_degradation_transparency, which already makes that visible on
    quality_gate), but a degraded-but-real recheck plus a revision pass that actually ran beats a
    revision that never got a chance to fix issues that were already known to exist. Only when
    even the bare floor plus a minimally useful revision slice (revision_min_remaining) can't both
    fit does this fall back to skipping revision outright -- that remains the correct call when
    the total remaining budget is smaller than the recheck's own unavoidable minimum.
    """
    recheck_min_remaining = max(
        # 60 -> 90 (2026-08-12): must stay >= _entailment_deadline_with_critic_reserve's own ideal
        # reserve (min_sections(2) * RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS(30) + jitter(10)
        # = 70 by default). If this floor were smaller than that, revision_deadline (validation_
        # deadline - recheck_min_remaining) could sit LATER than the second-pass entailment's own
        # carved deadline (validation_deadline - critic_reserve) -- letting revision run right up
        # to a point where the post-revision entailment recheck's deadline has already passed,
        # so it would immediately see itself as budget_exhausted and never actually re-check
        # anything. Caught by test_revision_deadline_is_capped_and_never_shrinks_shared_validation_
        # deadline when the critic-evidence-grounding fix raised the section floor.
        10.0, float(os.getenv("RESEARCH_VALIDATION_RECHECK_MIN_REMAINING_SECONDS", "90"))
    )
    if revision_min_remaining is None:
        revision_min_remaining = max(
            5.0, float(os.getenv("RESEARCH_VALIDATION_REVISION_MIN_REMAINING_SECONDS", "45"))
        )
    remaining = max(0.0, validation_deadline - now)
    ideal_reserve = max(recheck_min_remaining, first_validate_elapsed_seconds)
    if remaining - ideal_reserve >= revision_min_remaining:
        # Common case: plenty of budget left, the ideal (first-validate-cost-sized) reserve
        # fits with room to spare for revision too. Unchanged from the original behavior.
        recheck_reserve = ideal_reserve
        degraded_reserve = False
    elif remaining - recheck_min_remaining >= revision_min_remaining:
        # The ideal reserve doesn't fit, but the bare floor does, with enough left over for a
        # minimally useful revision pass. Degrade the reserve rather than starve revision.
        recheck_reserve = recheck_min_remaining
        degraded_reserve = True
    else:
        # Not even the floor reserve plus a minimal revision slice fit in what's left --
        # no split can let both phases run usefully; skip revision (same as before the fix).
        recheck_reserve = ideal_reserve
        degraded_reserve = False
    revision_budget_available = max(0.0, remaining - recheck_reserve)
    return {
        "recheck_reserve": recheck_reserve,
        "revision_budget_available": revision_budget_available,
        "degraded_reserve": degraded_reserve,
    }


def _entailment_deadline_with_critic_reserve(
    documents: list[dict[str, Any]],
    validation_deadline: float,
    *,
    now: float,
    entailment_min_remaining: float | None = None,
) -> dict[str, Any]:
    """Carve an un-stealable slice of the shared validation budget for section-critic scoring
    out of the deadline entailment/citation validation is allowed to run against.

    entailment and _section_critic_budgets currently share one deadline with no reservation --
    whichever runs first gets to spend all of it. A real job (case-shared-validation-budget)
    showed the failure mode: entailment hit rate limiting and first_validate alone took 174.5s
    of a 360s budget; by the time the post-revision second_validate pass reached section
    critics, nothing was left and all 5 sections came back "budget_exhausted" -- report_quality
    .issues ended up empty (no per-section feedback for the user) even though the report
    "passed" (critic scores can only lower the deterministic defaults via min-merge, never
    raise them, so a critic that never ran leaves every dimension at its lenient default).

    Mirrors _revision_budget_split's three-tier degrade, but critic-shaped, not revision-shaped:
    - Ideal reserve: enough for min(RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS, scoreable sections)
      sections to each clear the existing per-section floor
      (RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS). Default 2, not "all sections": one
      scored section is too noisy a sample (and still trips
      _annotate_low_confidence_critic_coverage's 50% threshold, so it wouldn't even fix that
      signal); reserving for every section would make this reserve grow with report length and
      start meaningfully competing with recheck_reserve on long reports.
    - A fixed jitter buffer (RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS, default
      10s, independent of section count) is added on top of every non-zero reserve tier. Without
      it, this reserve is a cliff: `_section_critic_budgets` re-checks the *same*
      RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS floor against however much time entailment
      actually left behind, and entailment routinely runs right up to (and briefly past) its own
      deadline -- `_validation_model`'s timeout is `min(role_timeout, remaining)`, so it schedules
      right to the wire, and network/retry slop can push the real finish a few hundred ms later
      still. A real job (case-validation-reserve-jitter, RESEARCH_VALIDATION_BUDGET_SECONDS=90)
      reserved the 20s floor tier for critic, but entailment finished 123ms past its carved-out
      deadline -- leaving `_section_critic_budgets` 19.877s, just under its own 20s floor, so
      every section came back budget_exhausted despite the reserve "succeeding". The jitter
      buffer absorbs exactly this kind of near-miss overshoot; it isn't scaled by report length
      or task complexity because the overshoot it protects against (scheduling/network slop, not
      genuine model work) doesn't scale with either.
    - entailment's own floor: entailment must keep at least
      RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS (default 60s -- same order of
      magnitude as RESEARCH_VALIDATION_RECHECK_MIN_REMAINING_SECONDS, since both bound "how long
      a single degraded-but-real model-review pass over the whole claim ledger needs to not be
      pure deterministic fallback"). If the ideal critic reserve (floor sections + jitter) would
      push entailment below this floor, degrade to a 1-section floor reserve (still +jitter)
      instead; if even that doesn't fit, reserve nothing -- entailment (the harder,
      approval-blocking gate) wins outright, same as before this change.

    Returns the narrowed entailment-facing deadline plus the reserve itself, so the caller can
    build a separate model closure for entailment while citation/critic keep using the original
    closure bound to `validation_deadline` -- critic still gets everything left after
    entailment/citation actually finish, not just its reserved slice; the reserve only
    guarantees a floor, it never caps critic's upside if entailment finishes early.
    """
    entailment_min_remaining = max(
        10.0,
        entailment_min_remaining
        if entailment_min_remaining is not None
        else float(os.getenv("RESEARCH_VALIDATION_ENTAILMENT_MIN_REMAINING_SECONDS", "60")),
    )
    section_min_budget = max(1.0, float(os.getenv("RESEARCH_VALIDATION_SECTION_MIN_BUDGET_SECONDS", "30")))
    min_sections = max(0, int(os.getenv("RESEARCH_VALIDATION_CRITIC_MIN_SECTIONS", "2")))
    jitter_buffer = max(0.0, float(os.getenv("RESEARCH_VALIDATION_CRITIC_RESERVE_JITTER_BUFFER_SECONDS", "10")))
    scoreable = [
        document for document in documents if not _is_reference_title(str(document.get("title") or ""))
    ]
    remaining = max(0.0, validation_deadline - now)
    if not scoreable or min_sections <= 0:
        return {"critic_reserve": 0.0, "entailment_deadline": validation_deadline, "degraded_reserve": False}

    ideal_reserve = min(len(scoreable), min_sections) * section_min_budget + jitter_buffer
    floor_reserve = section_min_budget + jitter_buffer

    if remaining - ideal_reserve >= entailment_min_remaining:
        critic_reserve = ideal_reserve
        degraded_reserve = False
    elif remaining - floor_reserve >= entailment_min_remaining:
        critic_reserve = floor_reserve
        degraded_reserve = True
    else:
        critic_reserve = 0.0
        degraded_reserve = False

    return {
        "critic_reserve": critic_reserve,
        "entailment_deadline": validation_deadline - critic_reserve,
        "degraded_reserve": degraded_reserve,
    }


def _entailment_evidence_with_writer_context(
    evidence_map: list[dict[str, Any]],
    report_blueprint: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach the same section-scoped, same-source context that the Writer could inspect."""
    context_by_section_source: dict[tuple[str, str], list[str]] = {}
    for section in report_blueprint.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id") or "")
        dossier = section.get("dossier") if isinstance(section.get("dossier"), dict) else {}
        for source in dossier.get("source_notes") or []:
            if not isinstance(source, dict):
                continue
            source_url = str(source.get("source_url") or "")
            values = [
                *[str(item) for item in source.get("claims") or [] if str(item).strip()],
                *[str(item) for item in source.get("excerpts") or [] if str(item).strip()],
            ]
            context_by_section_source.setdefault((section_id, source_url), []).extend(values)
        for chunk in dossier.get("relevant_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            source_url = str(chunk.get("source_url") or "")
            excerpt = str(chunk.get("excerpt") or "").strip()
            if excerpt:
                context_by_section_source.setdefault((section_id, source_url), []).append(excerpt)

    output = []
    for evidence in evidence_map:
        source_url = str(evidence.get("source_url") or "")
        by_section: dict[str, str] = {}
        for (section_id, context_url), values in context_by_section_source.items():
            if context_url != source_url:
                continue
            unique_values = list(dict.fromkeys(value for value in values if value.strip()))
            if unique_values:
                by_section[section_id] = "\n".join(unique_values)[:3200]
        output.append({**evidence, "validation_context_by_section": by_section})
    return output


def _entailment_revision_issues(entailment: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    for item in entailment.get("judgments") or []:
        status = str(item.get("status") or "")
        section_id = str(item.get("section_id") or "")
        claim = str(item.get("claim") or "")
        if (
            status not in {"unsupported", "label_mismatch", "non_atomic"}
            or not (section_id.startswith("sec_") or section_id == "editorial")
            or not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", claim)
        ):
            continue
        issues.append({
            "section_id": section_id,
            "type": f"entailment_{status}",
            "claim": claim[:500],
            "evidence_ids": [str(value) for value in item.get("evidence_ids") or []],
            "validation_reason": str(item.get("reason") or "")[:300],
            "instruction": (
                f"Validation rejected this claim: {claim[:500]} "
                f"Reason: {str(item.get('reason') or '')[:300]} "
                "Remove it or rewrite only the portion explicitly supported by the cited evidence. "
                "Do not add replacement facts."
            ),
        })
    return issues


def _annotate_validation_budget(quality_gate: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    if not diagnostics.get("budget_exhausted"):
        return
    quality_gate.setdefault("warnings", []).append(
        "Validation 模型增强未在阶段预算内全部完成；审批结论基于已完成的确定性校验。"
    )
    quality_gate.setdefault("diagnostics", {})["validation_model_enhancements_complete"] = False


def _target_section_evidence_count(title: str = "", section: dict[str, Any] | None = None) -> int:
    """Default Deep Research floor for body-worthy evidence, not raw snippets."""
    if _is_reference_or_overview_title(title or str((section or {}).get("title") or "")):
        default = 4
    else:
        default = 6
    budget = (section or {}).get("evidence_budget") if isinstance((section or {}).get("evidence_budget"), dict) else {}
    try:
        planned = int(budget.get("min_evidence") or 0)
    except (TypeError, ValueError):
        planned = 0
    try:
        configured = int(os.getenv("RESEARCH_TARGET_EVIDENCE_PER_SECTION", str(default)))
    except ValueError:
        configured = default
    return max(default, configured, planned)


def _target_section_source_count(title: str = "", section: dict[str, Any] | None = None) -> int:
    budget = (section or {}).get("evidence_budget") if isinstance((section or {}).get("evidence_budget"), dict) else {}
    try:
        planned = int(budget.get("min_sources") or 0)
    except (TypeError, ValueError):
        planned = 0
    try:
        configured = int(os.getenv("RESEARCH_TARGET_SOURCES_PER_SECTION", "3"))
    except ValueError:
        configured = 3
    return max(2, configured, planned)


def create_research_job(payload: dict[str, Any]) -> dict:
    if payload.get("source_mode", "web") != "web":
        raise ValueError("首版 Research Studio 仅支持 source_mode=web")
    prepared_payload = dict(payload)
    if str(prepared_payload.get("pipeline_mode") or "question_loop") == "question_loop":
        plan_policy = str(prepared_payload.get("plan_policy") or "confirmed_required")
        if plan_policy == "confirmed_required":
            prepared_payload = _prepare_confirmed_research_payload(prepared_payload)

    def target(input_data: dict[str, Any]) -> dict:
        return _run_research_pipeline(job_id=target.job_id, input_data=input_data)

    job = job_service.create(job_type="research", input_data=prepared_payload, target=None)
    target.job_id = job["job_id"]  # type: ignore[attr-defined]
    job_service.set_steps(job["job_id"], _initial_steps())

    import threading

    thread = threading.Thread(target=job_service._run, args=(job["job_id"], target), daemon=True)
    thread.start()
    return job_service.get(job["job_id"])


def _prepare_confirmed_research_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize an interactive, user-confirmed plan before a job is created."""
    from . import research_planning_service

    user_request = str(payload.get("user_request") or "").strip()
    confirmed_plan = payload.get("confirmed_plan") if isinstance(payload.get("confirmed_plan"), dict) else None
    plan_token_value = str(payload.get("plan_token") or "").strip()
    if confirmed_plan is not None:
        if not plan_token_value or not research_planning_service.verify_plan_token(confirmed_plan, plan_token_value):
            raise ValueError("确认计划签名无效或计划内容已被修改，请重新生成并确认调研计划")
        expected_plan_id = research_planning_service.plan_id(confirmed_plan)
        supplied_plan_id = str(payload.get("plan_id") or "").strip()
        if not supplied_plan_id or supplied_plan_id != expected_plan_id:
            raise ValueError("确认计划 ID 缺失或与计划内容不一致，请重新生成并确认调研计划")
        plan_request = str(confirmed_plan.get("user_request") or "").strip()
        if plan_request and plan_request != user_request:
            raise ValueError("确认计划对应的调研请求与当前请求不一致，请重新生成计划")
        strategy = research_planning_service.resolve_research_strategy(
            user_request,
            str(confirmed_plan.get("strategy") or "auto"),
        )
        raw_plan = confirmed_plan
    else:
        raise ValueError("创建调研任务前必须提供带签名的用户确认计划")

    normalized = _normalize_research_plan(
        raw_plan,
        user_request=user_request,
        topic=_suggest_topic(payload, user_request),
        strategy=strategy,
        planner_mode="authored_plan",
    )
    if not normalized:
        raise ValueError("用户确认的调研计划结构无效，已拒绝创建任务")
    executable_questions = sum(
        len(section.get("questions") or [])
        for section in normalized.get("sections") or []
        if isinstance(section, dict) and not is_non_research_section_title(
            str(section.get("title") or ""), has_questions=bool(section.get("questions"))
        )
    )
    if executable_questions < 1:
        raise ValueError("用户确认的调研计划没有可执行研究问题，已拒绝创建任务")

    comparison_budget: dict[str, Any] = {}
    if _comparison_plan_enabled(normalized):
        comparison_budget = _normalized_comparison_budget(str(payload.get("depth") or "normal"))
        _validate_comparison_plan_budget(normalized, comparison_budget)

    canonical_confirmed_plan = {
        "version": "v1",
        "user_request": user_request,
        "topic": str(normalized.get("topic") or "").strip(),
        "strategy": strategy,
        "research_type": str(normalized.get("research_type") or "").strip(),
        "subjects": normalized.get("subjects") or [],
        "comparison_axes": normalized.get("comparison_axes") or [],
        "matrix_cells": normalized.get("matrix_cells") or [],
        "sections": normalized.get("sections") or [],
    }
    prepared = dict(payload)
    prepared.update(
        {
            "research_strategy": strategy,
            "planned_sections": canonical_confirmed_plan["sections"],
            "confirmed_plan": canonical_confirmed_plan,
            "plan_id": research_planning_service.plan_id(canonical_confirmed_plan),
            "plan_token": research_planning_service.plan_token(canonical_confirmed_plan),
            "comparison_budget": comparison_budget,
        }
    )
    return prepared


def _comparison_plan_enabled(plan: dict[str, Any]) -> bool:
    if str(plan.get("research_type") or "").strip().lower() == "comparison":
        return True
    if len([item for item in plan.get("subjects") or [] if str(item).strip()]) >= 2:
        return bool([item for item in plan.get("comparison_axes") or [] if str(item).strip()])
    return any(
        isinstance(question, dict) and len(question.get("comparison_subjects") or []) >= 2
        for section in plan.get("sections") or [] if isinstance(section, dict)
        for question in section.get("questions") or []
    )


def _normalized_comparison_budget(depth: str) -> dict[str, int]:
    resolved_depth = depth if depth in {"normal", "deep"} else "normal"
    budget = asdict(ComparisonBudget.for_depth(resolved_depth))
    prefix = f"RESEARCH_COMPARISON_{resolved_depth.upper()}_"
    for key, default in list(budget.items()):
        raw = os.getenv(prefix + key.upper())
        if raw is None:
            continue
        try:
            budget[key] = max(1, int(raw))
        except ValueError:
            budget[key] = default
    return budget


def _validate_comparison_plan_budget(plan: dict[str, Any], budget: dict[str, Any]) -> None:
    subjects = [str(item).strip() for item in plan.get("subjects") or [] if str(item).strip()]
    axes = [str(item).strip() for item in plan.get("comparison_axes") or [] if str(item).strip()]
    required_cells = len(subjects) * len(axes)
    violations = []
    if len(subjects) > int(budget["max_subjects"]):
        violations.append(f"subjects={len(subjects)}>{budget['max_subjects']}")
    if len(axes) > int(budget["max_axes"]):
        violations.append(f"axes={len(axes)}>{budget['max_axes']}")
    if required_cells > int(budget["max_required_cells"]):
        violations.append(f"required_cells={required_cells}>{budget['max_required_cells']}")
    if violations:
        raise ValueError("比较矩阵超过当前深度预算，未创建任务：" + "，".join(violations))


def get_research_job(job_id: str) -> dict:
    return job_service.get(job_id)


def research_artifacts(job_id: str) -> dict:
    job = job_service.get(job_id)
    result = job.get("result") or {}
    input_data = job.get("input") or {}
    return {
        "channel_id": result.get("channel_id") or input_data.get("channel_id"),
        "sources": result.get("sources", []),
        "evidence_pack": result.get("evidence_pack", []),
        "materials": result.get("materials", []),
        "selected_chunks": result.get("selected_chunks", []),
        "evidence_map": result.get("evidence_map", []),
        "validation": result.get("validation", {}),
        "research_plan": result.get("research_plan", {}),
        "query_plan": result.get("query_plan", {}),
        "coverage": result.get("coverage", {}),
        "comparison_budget": result.get("comparison_budget", {}),
        "matrix_coverage": result.get("matrix_coverage", {}),
        "report_document": result.get("report_document", {}),
        "question_states": result.get("question_states", []),
        "findings": result.get("findings", []),
        "rejected_evidence": result.get("rejected_evidence", []),
        "report_quality": result.get("report_quality", {}),
        "quality_gate": result.get("quality_gate", {}),
        "pipeline_diagnostics": result.get("pipeline_diagnostics", {}),
        "writing_architecture": result.get("writing_architecture") or "finding",
        "writing_architecture_version": result.get("writing_architecture_version"),
        "capabilities": result.get("capabilities") or {},
        "evidence_native_blueprint": result.get("evidence_native_blueprint") or {},
        "evidence_context_diagnostics": result.get("evidence_context_diagnostics") or [],
        "section_prevalidation": result.get("section_prevalidation") or {},
        "editorial_claim_bank": result.get("editorial_claim_bank") or [],
        "documents": result.get("documents", []),
        "delivery": result.get("delivery") or _delivery_payload(result),
        "suggested_topic": result.get("suggested_topic") or (job.get("input") or {}).get("topic"),
    }


def _publish_running_research_snapshot(job_id: str, result_patch: dict[str, Any]) -> None:
    """Expose completed research stages to polling clients before drafting finishes."""

    job_service.merge_running_result(
        job_id,
        result_patch={"status": "running", **result_patch},
    )


def _delivery_payload(
    result: dict[str, Any], input_data: dict[str, Any] | None = None, *, title_suffix: str = ""
) -> dict[str, Any]:
    input_data = input_data or {}
    payload = {
        "type": "local_markdown",
        "paths": result.get("paths") or [],
        "feishu_doc_url": None,
    }
    if not _should_deliver_feishu_doc(input_data, payload["paths"]):
        return payload
    try:
        from . import feishu_doc_service

        created = feishu_doc_service.create_research_review_doc(
            review_path=str(payload["paths"][0]),
            title=_feishu_review_doc_title(result=result, input_data=input_data) + title_suffix,
            folder_token=os.getenv("FEISHU_DOC_FOLDER_TOKEN") or None,
        )
    except Exception as exc:
        return {
            **payload,
            "diagnostics": {
                "feishu_doc_delivery": "failed",
                "error": str(exc),
            },
        }
    return {
        **payload,
        "type": "feishu_doc",
        "feishu_doc_url": created.get("url"),
        "feishu_document_id": created.get("document_id"),
        "feishu_doc_title": created.get("title"),
        "diagnostics": {"feishu_doc_delivery": "succeeded"},
    }


def _should_deliver_feishu_doc(input_data: dict[str, Any], paths: list[Any]) -> bool:
    if not paths:
        return False
    enabled = os.getenv("FEISHU_DOC_DELIVERY_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no"}:
        return False
    delivery_target = str(input_data.get("delivery_target") or "").strip()
    if delivery_target == "feishu_doc":
        return True
    channel_id = str(input_data.get("channel_id") or "").strip()
    auto_feishu = os.getenv("FEISHU_DOC_DELIVERY_FOR_FEISHU_CHANNELS", "true").strip().lower()
    return channel_id.startswith("feishu:") and auto_feishu not in {"0", "false", "no"}


def _feishu_review_doc_title(*, result: dict[str, Any], input_data: dict[str, Any]) -> str:
    request = str(input_data.get("user_request") or "深度调研").strip()
    base = _integrated_report_title(user_request=request, result=result, input_data=input_data)
    return f"调研审核包 - {base[:42]}"


def _write_research_review_artifacts(*, job_id: str, result: dict[str, Any], input_data: dict[str, Any]) -> list[str]:
    if os.getenv("RESEARCH_WRITE_REVIEW_ARTIFACTS", "true").strip().lower() in {"0", "false", "no"}:
        return []
    artifact_dir = RESEARCH_ARTIFACT_DIR / _safe_artifact_name(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    review_path = artifact_dir / "review.md"
    review_path.write_text(_render_research_review_markdown(job_id=job_id, result=result, input_data=input_data), encoding="utf-8")
    return [str(review_path)]


def _safe_artifact_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return clean or f"research_{int(time.time())}"


def _render_research_review_markdown(*, job_id: str, result: dict[str, Any], input_data: dict[str, Any]) -> str:
    documents = [doc for doc in result.get("documents") or [] if isinstance(doc, dict)]
    evidence_map = [row for row in result.get("evidence_map") or [] if isinstance(row, dict)]
    sources = [row for row in result.get("sources") or [] if isinstance(row, dict)]
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    report_document = result.get("report_document") if isinstance(result.get("report_document"), dict) else {}
    if not report_document:
        report_document = _build_integrated_report_document(
            user_request=str(input_data.get("user_request") or result.get("suggested_topic") or job_id),
            result=result,
            input_data=input_data,
        )
    report_content = str(report_document.get("content") or "").strip()
    lines = [
        report_content or f"# {input_data.get('user_request') or result.get('suggested_topic') or job_id}\n\n暂无完整报告内容。",
        "",
        "---",
        "",
        "# 审计材料",
        "",
        "## 任务信息",
        "",
        f"- Job: `{job_id}`",
        f"- Channel: `{result.get('channel_id') or input_data.get('channel_id') or '-'}`",
        f"- Topic: `{result.get('suggested_topic') or input_data.get('topic') or '-'}`",
        f"- Status: `{result.get('status') or '-'}`",
        f"- Quality Gate: `{quality_gate.get('status') or '-'}`; approval_ready=`{bool(quality_gate.get('approval_ready'))}`",
        f"- Validation: `{validation.get('status') or '-'}`",
        "",
        "## 审核建议",
        "",
        *_research_review_recommendation_lines(result),
        "",
        "## 覆盖与质量",
        "",
        f"- 覆盖度：{coverage.get('reason') or '-'}",
        f"- 验证结论：{validation.get('reason') or '-'}",
        f"- 来源数：{len(sources)}",
        f"- 证据数：{len(evidence_map)}",
        f"- 内部章节数：{len(documents)}",
        "",
        "## 内部章节目录",
        "",
    ]
    for doc in documents:
        metrics = doc.get("evidence_metrics") if isinstance(doc.get("evidence_metrics"), dict) else {}
        lines.append(
            f"- {doc.get('title') or 'Untitled'}"
            f"（证据 {metrics.get('evidence_count', 0)} / 来源 {metrics.get('source_count', 0)}）"
        )
    lines.extend(["", "## 断言—证据审计", ""])
    lines.extend(_render_claim_evidence_audit_lines(documents=documents, result=result, evidence_map=evidence_map))
    lines.extend(["", "<details>", "<summary>技术附录：原始来源、检索诊断与内部章节稿</summary>", "", "以下内容保留给工程排查和机器复核；普通审核优先阅读上面的断言—证据审计。", "", "## 来源摘要", ""])
    for source in sources[:20]:
        title = str(source.get("title") or source.get("source_title") or source.get("url") or "source").strip()
        url = str(source.get("url") or source.get("source_url") or "").strip()
        lines.append(f"- {title}: {url}" if url else f"- {title}")
    question_states = [row for row in result.get("question_states") or [] if isinstance(row, dict)]
    if question_states:
        lines.extend(["", "## 问题级检索诊断", ""])
        for state in question_states:
            state_coverage = state.get("coverage") if isinstance(state.get("coverage"), dict) else {}
            source_mix = state_coverage.get("source_type_counts") or {}
            lines.append(
                f"- `{state.get('question_id') or '-'}` {state.get('status') or '-'}；"
                f"kind={state.get('question_kind') or '-'}；rounds={state.get('rounds_used') or 0}；stop={state.get('stop_reason') or '-'}；"
                f"clusters={state_coverage.get('claim_cluster_count') or 0}；sources={state_coverage.get('source_count') or 0}；"
                f"source_mix={json.dumps(source_mix, ensure_ascii=False)}；followups={len(state.get('gap_execution_log') or [])}；"
                f"fetch_urls={len(state.get('attempted_fetch_urls') or [])}；access_issues={len(state.get('access_issues') or [])}"
            )
    lines.extend(["", "## 证据表", ""])
    for row in evidence_map[:80]:
        evidence_id = str(row.get("id") or row.get("evidence_id") or row.get("chunk_id") or "").strip()
        section = str(row.get("section_id") or "").strip()
        claim = _clean_evidence_claim(str(row.get("claim") or row.get("evidence_text") or ""))
        url = str(row.get("source_url") or row.get("url") or "").strip()
        lines.append(f"- `{evidence_id or '-'}` `{section or '-'}` {claim}。来源：{url}")
    lines.extend(["", "## 内部章节稿", ""])
    lines.append("以下内容用于审计生成过程，不作为用户阅读版报告主体。")
    lines.append("")
    for doc in documents:
        title = str(doc.get("title") or "Untitled").strip()
        content = str(doc.get("content") or "").strip()
        lines.extend([f"---", "", f"## {title}", "", content or "（空文档）", ""])
    lines.extend(["", "</details>"])
    return "\n".join(lines).strip() + "\n"


def _render_claim_evidence_audit_lines(
    *, documents: list[dict[str, Any]], result: dict[str, Any], evidence_map: list[dict[str, Any]]
) -> list[str]:
    """Render a deterministic claim-to-evidence audit ledger for human review.

    The delivered report intentionally uses source-level numeric citations. This audit section
    keeps the internal evidence IDs and adds the missing judgment context: what the claim says,
    which evidence rows it binds to, what the source actually quotes, and how entailment judged it.
    """
    claims: list[dict[str, Any]] = []
    for document in documents:
        section_id = str(document.get("section_id") or document.get("title") or "unknown")
        for row in document.get("claim_ledger") or []:
            if isinstance(row, dict):
                claims.append({**row, "section_id": row.get("section_id") or section_id})
    editorial = result.get("editorial") if isinstance(result.get("editorial"), dict) else {}
    for row in editorial.get("claim_ledger") or []:
        if isinstance(row, dict):
            claims.append({**row, "section_id": row.get("section_id") or "editorial"})
    evidence_by_id = {
        str(row.get("id") or row.get("evidence_id") or row.get("chunk_id") or "").strip(): row
        for row in evidence_map
        if str(row.get("id") or row.get("evidence_id") or row.get("chunk_id") or "").strip()
    }
    entailment = result.get("validation", {}).get("entailment", {}) if isinstance(result.get("validation"), dict) else {}
    judgments = [row for row in entailment.get("judgments") or [] if isinstance(row, dict)]
    lines: list[str] = [
        "> 这张表把最终正文中的断言与内部 evidence_id 对齐；状态是验证器判断，不等同于事实真值，人工复核应优先看 quote 和支持范围。",
        "",
    ]
    if not claims:
        return lines + ["暂无可用的 claim ledger。"]
    status_counts: dict[str, int] = {}
    for index, claim in enumerate(claims):
        text = _clean_evidence_claim(str(claim.get("text") or claim.get("claim") or "")) or "（未提供断言文本）"
        evidence_ids = [str(value).strip() for value in claim.get("evidence_ids") or [] if str(value).strip()]
        judgment = next((row for row in judgments if row.get("claim_index") == index), None)
        status = str((judgment or {}).get("status") or "未判定")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = str((judgment or {}).get("reason") or "").strip()
        lines.extend([
            f"### {index + 1}. {text}",
            "",
            f"- 位置：`{claim.get('section_id') or 'unknown'}`；类型：`{claim.get('claim_type') or '未标注'}`；支持范围：`{claim.get('support_scope') or '未标注'}`",
            f"- 审计状态：`{status}`" + (f"；判断理由：{reason}" if reason else ""),
            f"- 绑定证据：{', '.join(f'`{item}`' for item in evidence_ids) if evidence_ids else '无'}",
        ])
        if not evidence_ids:
            lines.append("- 支持判断：没有绑定证据，不能据此确认该断言。")
        for evidence_id in evidence_ids[:8]:
            evidence = evidence_by_id.get(evidence_id) or {}
            quote = _clean_evidence_claim(str(evidence.get("quote") or evidence.get("cleaned_quote") or evidence.get("evidence_text") or ""))
            source = str(evidence.get("source_url") or evidence.get("url") or "").strip()
            scope = str(evidence.get("scope") or evidence.get("claim_type") or "未标注")
            directness = evidence.get("directness_score")
            quality = f"；直接性：{directness}" if directness is not None else ""
            lines.append(f"  - `{evidence_id}`（范围：`{scope}`{quality}）{quote or '未提供原文摘录'}。来源：{source or '-'}")
        lines.append("")
    summary = "；".join(f"{key} {value}" for key, value in sorted(status_counts.items()))
    return [f"审计条目：{len(claims)}；状态分布：{summary}。", ""] + lines


def _research_review_recommendation_lines(result: dict[str, Any]) -> list[str]:
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    followup_actions = [row for row in result.get("followup_actions") or [] if isinstance(row, dict)]
    lines: list[str] = []
    if quality_gate.get("approval_ready") and validation.get("status") == "passed":
        lines.append("- 建议：可以进入人工抽查和审批落库。")
        lines.append("- 人工重点：抽查每章 1-2 条证据原文是否支撑结论，确认主题范围是否满足预期。")
    else:
        lines.append("- 建议：暂不直接落库，先按质量问题发起 targeted follow-up。")
        reason = str(quality_gate.get("reason") or validation.get("reason") or "").strip()
        if reason:
            lines.append(f"- 主要原因：{reason}")
    if followup_actions:
        pending = [row for row in followup_actions if str(row.get("status") or "") not in {"completed", "resolved"}]
        if pending:
            lines.append("- 已有未完成补查：" + "；".join(str(row.get("query") or row.get("reason") or "") for row in pending[:3]))
    return lines


def _build_integrated_report_document(*, user_request: str, result: dict[str, Any], input_data: dict[str, Any]) -> dict[str, Any]:
    documents = [doc for doc in result.get("documents") or [] if isinstance(doc, dict)]
    evidence_map = [row for row in result.get("evidence_map") or [] if isinstance(row, dict)]
    title = _integrated_report_title(user_request=user_request, result=result, input_data=input_data)
    report_blueprint = result.get("report_blueprint") if isinstance(result.get("report_blueprint"), dict) else {}
    report_type = str(report_blueprint.get("report_type") or _infer_report_type(
        user_request=user_request,
        research_strategy=str(result.get("research_strategy") or input_data.get("research_strategy") or ""),
        research_plan=result.get("research_plan") if isinstance(result.get("research_plan"), dict) else {},
    ))
    body_rows = _integrated_body_evidence_rows(documents=documents, evidence_map=evidence_map)
    lines: list[str] = [
        f"# {title}",
        "",
        f"> 研究问题：{str(user_request or input_data.get('user_request') or '').strip()}",
        "",
        "## 摘要",
        "",
        *_integrated_summary_lines(documents=documents, evidence_map=body_rows),
        "",
        "## 研究主线",
        "",
        *_integrated_thesis_lines(user_request=user_request, documents=documents, evidence_map=body_rows, report_type=report_type),
        "",
        "## 一页结论",
        "",
        *_integrated_key_finding_lines(documents=documents, evidence_map=body_rows),
        "",
        "## 证据地图",
        "",
        *_integrated_evidence_map_table(documents),
        "",
    ]
    structure = _integrated_structure_block(documents=documents, report_type=report_type)
    if structure:
        lines.extend(structure)
        lines.append("")
    lines.extend(["## 深度分析", ""])
    for doc in documents:
        section_title = str(doc.get("title") or "未命名章节").strip()
        lines.extend([f"### {section_title}", ""])
        lines.extend(_reader_section_body(doc, report_type=report_type))
        lines.append("")
    lines.extend([
        "## 结论、风险与后续核查",
        "",
        *_integrated_next_step_lines(result),
    ])
    return {
        "title": title,
        "content": "\n".join(lines).strip() + "\n",
        "evidence_refs": _integrated_evidence_refs(documents, body_rows),
        "source": "long_form_report_composer",
        "report_type": report_type,
        "diagnostics": {
            "document_count": len(documents),
            "body_worthy_evidence_count": len(body_rows),
            "composer": "deterministic_report_level_synthesis",
        },
    }


def _integrated_report_title(*, user_request: str, result: dict[str, Any], input_data: dict[str, Any]) -> str:
    """Build a short reader-facing title from any research request.

    This is deliberately centralized: report type metadata supplies the semantic suffix when
    available, while the request is only used to extract the subject and preserve the original
    wording in the separate ``研究问题`` subtitle.
    """
    editorial = result.get("editorial") if isinstance(result.get("editorial"), dict) else {}
    model_title = re.sub(r"\s+", " ", str(editorial.get("report_title") or "")).strip(" #\n\r\t:：。.!！")
    if model_title and len(model_title) <= 60 and not any(token in model_title for token in ("用户问题", "研究问题", "执行摘要")):
        return model_title
    value = str(user_request or input_data.get("user_request") or result.get("suggested_topic") or "深度调研报告")
    value = re.sub(r"\s+", " ", value).strip(" ：:，,。")
    original = value
    value = re.sub(
        r"^(?:请(?:帮我|协助我)?|帮我)?(?:调研|研究|分析|了解|梳理|对比|比较|如何实现)(?:一下|下)?[：:\s]*",
        "",
        value,
        flags=re.I,
    ).strip()
    value = re.split(
        r"[，,；;](?:包括|包含|涵盖|重点|主要|需要|希望|围绕|从|并(?:且)?|以及)",
        value,
        maxsplit=1,
    )[0].strip(" ：:，,。")
    value = re.sub(
        r"(?:公开实践与参考设计|公开实践|参考设计|深度)?(?:调研|研究|分析)?报告$",
        "",
        value,
    ).strip(" ：:，,。")
    if not value:
        value = str(result.get("suggested_topic") or input_data.get("topic") or original or "深度调研").strip()
    if len(value) > 42:
        shortened = re.split(r"[，,；;：:]|\s+-\s+", value, maxsplit=1)[0].strip()
        value = shortened if len(shortened) >= 6 else value[:42].rstrip()
    lowered = original.lower()
    blueprint = result.get("report_blueprint") if isinstance(result.get("report_blueprint"), dict) else {}
    report_type = str(blueprint.get("report_type") or result.get("report_type") or "").strip().lower()
    if report_type in {"comparison", "compare", "dual_compare", "multi_compare"} or any(
        token in original for token in ("对比", "比较", "区别", "差异")
    ) or " vs " in f" {lowered} ":
        suffix = "：对比研究"
    elif report_type in {"implementation_guide", "implementation", "how_to", "实践指南"} or any(
        token in original for token in ("怎么做", "如何实现", "实施方案", "落地方案", "搭建指南")
    ):
        suffix = "：实践指南"
    elif any(token in original for token in ("定义", "用途", "架构", "使用方法", "公开实践", "参考设计")):
        suffix = "：架构与实践"
    else:
        suffix = "：调研"
    return value if value.endswith(("报告", "指南", "研究", "调研")) else f"{value}{suffix}"


def _integrated_summary_lines(*, documents: list[dict[str, Any]], evidence_map: list[dict[str, Any]]) -> list[str]:
    findings = _top_evidence_claims(documents=documents, evidence_map=evidence_map, limit=4)
    if not findings:
        return ["本报告当前没有足够可引用证据，建议先补查后再形成阅读版报告。"]
    evidence_count = len(evidence_map)
    source_count = len({str(row.get("source_url") or "") for row in evidence_map if row.get("source_url")})
    section_count = len(documents)
    lines = [
        (
            f"本报告基于 {section_count} 个研究章节、{evidence_count} 条正文级证据和 "
            f"{source_count} 个可追溯来源综合形成。正文不直接堆叠证据表，而是先归纳主线，"
            "再按章节展开机制、场景、工程取舍、风险和后续核查点。"
        ),
    ]
    if findings:
        first = "；".join(f"{claim} [{evidence_id}]" for claim, evidence_id in findings[:3])
        lines.append(f"从现有证据看，最直接的几条结论是：{first}。")
    return lines


def _integrated_key_finding_lines(*, documents: list[dict[str, Any]], evidence_map: list[dict[str, Any]]) -> list[str]:
    findings = _top_evidence_claims(documents=documents, evidence_map=evidence_map, limit=8)
    if not findings:
        return ["- 当前证据不足，无法形成关键结论。"]
    rows = [["判断", "证据依据", "阅读提示"]]
    for index, (claim, evidence_id) in enumerate(findings[:8], start=1):
        rows.append([f"结论 {index}", f"{claim} [{evidence_id}]", _finding_reader_hint(claim)])
    return _markdown_table("", rows[0], rows[1:])


def _integrated_body_evidence_rows(*, documents: list[dict[str, Any]], evidence_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in documents:
        table = doc.get("evidence_table")
        if not isinstance(table, list):
            continue
        for row in table:
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("id") or row.get("evidence_id") or "").strip()
            key = evidence_id or f"{row.get('source_url')}:{row.get('claim')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if rows:
        return rows
    for row in evidence_map:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _integrated_thesis_lines(
    *,
    user_request: str,
    documents: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    report_type: str,
) -> list[str]:
    findings = _top_evidence_claims(documents=documents, evidence_map=evidence_map, limit=6)
    if not findings:
        return ["当前材料尚不足以形成完整研究主线。建议先补充来源，再进行综合判断。"]
    section_titles = "、".join(str(doc.get("title") or "未命名章节") for doc in documents[:5])
    lines = [
        (
            f"这次调研的问题不是简单罗列资料，而是回答“{user_request}”在实际判断中该如何理解、"
            f"如何拆解以及如何落地。报告按 {section_titles} 等章节组织，先处理范围和核心结论，"
            "再进入结构、流程、工程约束和风险。"
        ),
        _report_type_thesis(report_type, findings),
    ]
    return lines


def _report_type_thesis(report_type: str, findings: list[tuple[str, str]]) -> str:
    first = "；".join(f"{claim} [{evidence_id}]" for claim, evidence_id in findings[:3])
    if report_type == "comparison":
        return f"对比型报告的主线是先确认对象和维度，再用证据解释差异边界。当前最关键的证据线索包括：{first}。"
    if report_type == "implementation_guide":
        return f"实现型报告的主线是从场景进入架构，再落到工具、状态和评估。当前最关键的证据线索包括：{first}。"
    if report_type == "diagnosis":
        return f"诊断型报告的主线是先定位表现，再拆出原因、影响和修复路径。当前最关键的证据线索包括：{first}。"
    return f"综合型报告的主线是用证据先固定事实边界，再解释这些事实之间的关系。当前最关键的证据线索包括：{first}。"


def _finding_reader_hint(claim: str) -> str:
    text = str(claim or "").lower()
    if any(token in text for token in ("api", "工具", "调用", "集成", "mcp", "sdk")):
        return "适合放入工程实现与依赖清单核查。"
    if any(token in text for token in ("风险", "限制", "隐私", "失败", "成本", "评估")):
        return "适合放入风险、成本或验收指标核查。"
    if any(token in text for token in ("架构", "模块", "agent", "流程", "工作流")):
        return "适合放入系统结构和流程设计核查。"
    return "适合抽查原文后作为正文结论依据。"


def _integrated_evidence_map_table(documents: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for doc in documents:
        metrics = doc.get("evidence_metrics") if isinstance(doc.get("evidence_metrics"), dict) else {}
        title = str(doc.get("title") or "未命名章节")
        rows.append([
            title,
            str(metrics.get("evidence_count") or len(doc.get("evidence_table") or [])),
            str(metrics.get("source_count") or "-"),
            str(metrics.get("weak_question_count") or 0),
            _section_depth_label(metrics),
        ])
    if not rows:
        return ["当前没有可展示的章节证据地图。"]
    return _markdown_table("", ["章节", "正文级证据", "来源", "薄弱问题", "深度状态"], rows)


def _section_depth_label(metrics: dict[str, Any]) -> str:
    evidence_count = int(metrics.get("evidence_count") or 0)
    source_count = int(metrics.get("source_count") or 0)
    weak_question_count = int(metrics.get("weak_question_count") or 0)
    if evidence_count >= 10 and source_count >= 3 and not weak_question_count:
        return "充实，可支撑长文展开"
    if evidence_count >= 6 and source_count >= 3 and not weak_question_count:
        return "可展开，建议抽查关键引用"
    return "偏薄，建议补查后再落库"


def _integrated_structure_block(*, documents: list[dict[str, Any]], report_type: str) -> list[str]:
    if report_type == "comparison":
        return _integrated_comparison_block(documents)
    if report_type == "implementation_guide":
        return _integrated_implementation_block(documents)
    return _integrated_dimension_block(documents)


def _integrated_implementation_block(documents: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for doc in documents:
        title = str(doc.get("title") or "章节")
        rows.append([
            title,
            _section_primary_claim(doc),
            _section_primary_refs(doc),
        ])
    if not rows:
        return []
    return [
        "## 实现拆解",
        "",
        *_markdown_table("", ["层面", "当前证据支撑的关注点", "引用"], rows),
    ]


def _integrated_comparison_block(documents: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for doc in documents:
        for row in (doc.get("evidence_table") or [])[:5]:
            if not isinstance(row, dict):
                continue
            subject = str(row.get("subject") or row.get("section_title") or doc.get("title") or "对象")
            axis = str(row.get("comparison_axis") or "综合维度")
            claim = _clean_evidence_sentence(str(row.get("claim") or ""))
            evidence_id = str(row.get("id") or "")
            rows.append([subject, axis, f"{claim} [{evidence_id}]", str(row.get("source_domain") or "-")])
    if not rows:
        return _integrated_dimension_block(documents)
    return [
        "## 对比视图",
        "",
        *_markdown_table("", ["对象", "维度", "证据化观察", "来源域名"], rows[:12]),
    ]


def _integrated_dimension_block(documents: list[dict[str, Any]]) -> list[str]:
    rows = [
        [str(doc.get("title") or "章节"), _section_primary_claim(doc), _section_primary_refs(doc)]
        for doc in documents
    ]
    if not rows:
        return []
    return [
        "## 分析维度",
        "",
        *_markdown_table("", ["维度", "主要观察", "引用"], rows),
    ]


def _top_evidence_claims(*, documents: list[dict[str, Any]], evidence_map: list[dict[str, Any]], limit: int) -> list[tuple[str, str]]:
    rows: list[dict[str, Any]] = []
    for doc in documents:
        table = doc.get("evidence_table")
        if isinstance(table, list):
            rows.extend(row for row in table if isinstance(row, dict))
    if not rows:
        rows = evidence_map
    findings: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        evidence_id = str(row.get("id") or row.get("evidence_id") or "").strip()
        claim = _clean_evidence_sentence(_clean_evidence_claim(str(row.get("claim") or row.get("evidence_text") or "")))
        if not evidence_id or not claim:
            continue
        key = re.sub(r"\W+", "", claim.lower())[:160]
        if key in seen:
            continue
        seen.add(key)
        findings.append((claim, evidence_id))
        if len(findings) >= limit:
            break
    return findings


def _section_primary_claim(doc: dict[str, Any]) -> str:
    table = [row for row in doc.get("evidence_table") or [] if isinstance(row, dict)]
    if not table:
        return "当前章节证据不足，暂不形成强结论。"
    claims = []
    for row in table[:2]:
        claim = _clean_evidence_sentence(str(row.get("claim") or ""))
        evidence_id = str(row.get("id") or "")
        if claim and evidence_id:
            claims.append(f"{claim} [{evidence_id}]")
    return "；".join(claims) or "当前章节证据不足，暂不形成强结论。"


def _section_primary_refs(doc: dict[str, Any]) -> str:
    refs = []
    for row in (doc.get("evidence_table") or [])[:4]:
        if isinstance(row, dict) and row.get("id"):
            refs.append(f"[{row.get('id')}]")
    return " ".join(refs) or "-"


def _reader_section_body(doc: dict[str, Any], *, report_type: str) -> list[str]:
    table = [row for row in doc.get("evidence_table") or [] if isinstance(row, dict)]
    if not table:
        return ["本节暂无可展开正文。"]
    lines: list[str] = []
    questions = _section_questions_from_document(doc)
    if questions:
        lines.extend(["#### 本节回答的问题", ""])
        for question in questions[:4]:
            lines.append(f"- {question}")
        lines.append("")
    lines.extend(["#### 证据化结论", ""])
    for claim, evidence_id in _section_key_claims(table, limit=5):
        lines.append(f"- {claim} [{evidence_id}]")
    lines.append("")
    table_block = _section_interpretation_table(doc, report_type=report_type)
    if table_block:
        lines.extend(table_block)
        lines.append("")
    lines.extend(["#### 展开分析", ""])
    lines.extend(_section_longform_paragraphs(doc, report_type=report_type))
    lines.append("")
    lines.extend(["#### 本节小结与核查点", ""])
    lines.extend(_section_takeaway_lines(doc))
    return _trim_blank_lines(lines)


def _section_questions_from_document(doc: dict[str, Any]) -> list[str]:
    questions = []
    for row in doc.get("evidence_table") or []:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question") or "").strip()
        if question and question not in questions:
            questions.append(question)
    if questions:
        return questions
    content = str(doc.get("content") or "")
    output: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("### ") and stripped[4:].strip() not in {"来源", "证据边界", "证据索引"}:
            output.append(stripped[4:].strip())
    return output[:4]


def _section_key_claims(table: list[dict[str, Any]], *, limit: int) -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in table:
        evidence_id = str(row.get("id") or "").strip()
        claim = _clean_evidence_sentence(str(row.get("claim") or ""))
        key = re.sub(r"\W+", "", claim.lower())[:120]
        if not evidence_id or not claim or key in seen:
            continue
        seen.add(key)
        claims.append((claim, evidence_id))
        if len(claims) >= limit:
            break
    return claims


def _section_interpretation_table(doc: dict[str, Any], *, report_type: str) -> list[str]:
    rows: list[list[str]] = []
    grouped = _section_rows_grouped_for_report(doc)
    for title, evidence_rows in grouped[:5]:
        claims = []
        refs = []
        domains = []
        for row in evidence_rows[:3]:
            claim = _clean_evidence_sentence(str(row.get("claim") or ""))
            evidence_id = str(row.get("id") or "")
            if claim and evidence_id:
                claims.append(f"{claim} [{evidence_id}]")
                refs.append(f"[{evidence_id}]")
            domain = str(row.get("source_domain") or "")
            if domain and domain not in domains:
                domains.append(domain)
        rows.append([
            title,
            "；".join(claims) or "当前证据不足。",
            _dimension_meaning(title, report_type),
            "、".join(domains[:3]) or "-",
            " ".join(refs[:4]) or "-",
        ])
    if not rows:
        return []
    return [
        "#### 证据解释表",
        "",
        *_markdown_table("", ["议题", "证据化观察", "为什么重要", "主要来源", "引用"], rows),
    ]


def _section_rows_grouped_for_report(doc: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in doc.get("evidence_table") or []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("question") or row.get("comparison_axis") or row.get("section_title") or "关键证据").strip()
        grouped.setdefault(title, []).append(row)
    return [(title, rows) for title, rows in grouped.items()]


def _dimension_meaning(title: str, report_type: str) -> str:
    text = str(title or "").lower()
    if report_type == "implementation_guide":
        if any(token in text for token in ("架构", "模块", "agent", "分工", "流程")):
            return "决定系统如何拆分职责、传递状态和定位失败点。"
        if any(token in text for token in ("工具", "api", "集成", "数据")):
            return "决定实现复杂度、实时性、权限边界和降级方案。"
        if any(token in text for token in ("评估", "风险", "成本", "验证")):
            return "决定是否能从原型进入可运营系统。"
        return "帮助把用户需求转化为可实现的产品和工程边界。"
    if report_type == "comparison":
        return "用于判断对象差异是否有公开证据支撑，而不是只凭印象选型。"
    return "用于把分散事实放回本节问题，避免只罗列来源。"


def _section_longform_paragraphs(doc: dict[str, Any], *, report_type: str) -> list[str]:
    paragraphs: list[str] = []
    grouped = _section_rows_grouped_for_report(doc)
    for group_index, (title, rows) in enumerate(grouped[:6], start=1):
        if not rows:
            continue
        claims = _section_key_claims(rows, limit=4)
        if not claims:
            continue
        paragraph = _compose_report_paragraph(title=title, claims=claims, rows=rows, group_index=group_index, report_type=report_type)
        paragraphs.append(paragraph)
        source_note = _source_diversity_sentence(rows)
        if source_note:
            paragraphs.append(source_note)
    if paragraphs:
        return paragraphs
    return [_section_primary_claim(doc)]


def _compose_report_paragraph(
    *,
    title: str,
    claims: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    group_index: int,
    report_type: str,
) -> str:
    claim_text = "；".join(f"{claim} [{evidence_id}]" for claim, evidence_id in claims[:4])
    prefix = "首先" if group_index == 1 else "进一步看"
    if report_type == "implementation_guide":
        return (
            f"{prefix}，“{title}”这一议题把报告从概念层推进到可实现层。"
            f"现有证据显示：{claim_text}。这些证据共同说明，本节不能只停留在抽象描述，"
            "还需要把角色、工具、状态、失败路径和评估方式放在同一个实现链路里审视。"
        )
    if report_type == "comparison":
        return (
            f"{prefix}，“{title}”这一维度适合用来判断对象差异。"
            f"现有证据显示：{claim_text}。因此，本节的重点不是给出单一排名，"
            "而是明确哪些差异有证据支撑、哪些仍需要继续查证。"
        )
    return (
        f"{prefix}，“{title}”提供了本节的一组关键事实。"
        f"现有证据显示：{claim_text}。这组材料的价值在于，它把分散来源中的事实收束到同一个问题下，"
        "让后续判断可以围绕证据边界展开。"
    )


def _source_diversity_sentence(rows: list[dict[str, Any]]) -> str:
    domains = []
    refs = []
    for row in rows:
        domain = str(row.get("source_domain") or "")
        if domain and domain not in domains:
            domains.append(domain)
        if row.get("id"):
            refs.append(f"[{row.get('id')}]")
    if len(domains) >= 2:
        return f"来源上，本组证据覆盖 {', '.join(domains[:3])} 等来源域名，适合做交叉抽查；对应引用包括 {' '.join(refs[:4])}。"
    if domains:
        return f"来源上，本组证据主要来自 {domains[0]}，后续如果用于强结论，建议继续补充第二来源；对应引用包括 {' '.join(refs[:4])}。"
    return ""


def _section_takeaway_lines(doc: dict[str, Any]) -> list[str]:
    table = [row for row in doc.get("evidence_table") or [] if isinstance(row, dict)]
    if not table:
        return ["- 当前章节证据不足，应先补查再形成结论。"]
    refs = _section_primary_refs(doc)
    metrics = doc.get("evidence_metrics") if isinstance(doc.get("evidence_metrics"), dict) else {}
    lines = [
        f"- 可以保留的结论边界：本节主要依据 {refs} 等证据，不应脱离这些来源扩展强事实。",
    ]
    weak_question_count = int(metrics.get("weak_question_count") or 0)
    if weak_question_count:
        lines.append(f"- 需要补查：仍有 {weak_question_count} 个研究问题证据偏薄，应优先做 targeted follow-up。")
    else:
        lines.append("- 人工核查建议：抽查本节表格中的核心引用，确认原文语境是否支持正文表述。")
    return lines


def _trim_blank_lines(lines: list[str]) -> list[str]:
    output: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        output.append(line)
        previous_blank = blank
    while output and not output[0].strip():
        output.pop(0)
    while output and not output[-1].strip():
        output.pop()
    return output


def _integrated_next_step_lines(result: dict[str, Any]) -> list[str]:
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    if quality_gate.get("approval_ready") and validation.get("status") == "passed":
        return [
            "- 质量门和引用验收已通过，可以进入人工抽查。",
            "- 人工抽查重点应放在核心结论的原文证据、来源权威性和是否贴合用户问题。",
            "- 如果用户需要更像长篇报告，应继续补查核心章节，而不是只增加格式化内容。",
        ]
    return [
        "- 当前报告仍存在证据或引用问题，不建议直接落库。",
        "- 应优先补查薄弱章节，并清理标题型、残句型或低相关证据。",
    ]


def _integrated_evidence_refs(documents: list[dict[str, Any]], evidence_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in documents:
        for ref in doc.get("evidence_refs") or []:
            if not isinstance(ref, dict):
                continue
            evidence_id = str(ref.get("evidence_id") or ref.get("id") or "").strip()
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                refs.append(ref)
    if refs:
        return refs
    for row in evidence_map:
        evidence_id = str(row.get("id") or row.get("evidence_id") or "").strip()
        if evidence_id and evidence_id not in seen:
            seen.add(evidence_id)
            refs.append(_evidence_ref(row))
    return refs


def preview_research_archive(job_id: str, payload: dict[str, Any]) -> dict:
    """Recommend a hierarchy location and build an exact, confirmable archive preview."""

    job = job_service.get(job_id)
    result = job.get("result") or {}
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    automatic_approval_ready = bool(quality_gate.get("approval_ready"))
    quality_acknowledgement_requested = payload.get("acknowledge_quality_warnings") is True
    quality_acknowledgement_applied = bool(
        quality_gate and not automatic_approval_ready and quality_acknowledgement_requested
    )
    if quality_gate and not automatic_approval_ready:
        if not quality_acknowledgement_requested:
            raise ValueError(str(quality_gate.get("reason") or "research documents did not pass quality gate"))
        if job.get("status") != "waiting_approval":
            raise ValueError("quality warnings can only be acknowledged while the research job is waiting approval")
    raw_quality_warnings = quality_gate.get("warnings") or []
    if not isinstance(raw_quality_warnings, list):
        raw_quality_warnings = [raw_quality_warnings]
    quality_review = {
        "automatic_status": str(quality_gate.get("status") or "unknown").strip() or "unknown",
        "automatic_approval_ready": automatic_approval_ready,
        "reason": str(quality_gate.get("reason") or "").strip(),
        "warnings": [str(item).strip() for item in raw_quality_warnings if str(item).strip()],
        "human_acknowledged": quality_acknowledgement_applied,
        "acknowledged_at": now_text() if quality_acknowledgement_applied else None,
    }
    _require_matrix_gap_acknowledgement(result, payload)
    documents = payload.get("documents") or result.get("documents") or []
    if not documents:
        raise ValueError("research job has no documents to archive")

    title = str(
        payload.get("title")
        or (result.get("report_document") or {}).get("title")
        or (result.get("report_blueprint") or {}).get("title")
        or (result.get("research_plan") or {}).get("topic_anchor")
        or result.get("suggested_topic")
        or (job.get("input") or {}).get("user_request")
        or "未命名调研"
    ).strip()

    evidence_map = result.get("evidence_map") or []
    archive_documents: list[dict[str, Any]] = []
    for document in documents:
        prepared = dict(document) if isinstance(document, dict) else {}
        title_value = str(prepared.get("title") or "未命名文档").strip()
        content_value = str(prepared.get("content") or "")
        try:
            prepared["html"] = render_question_loop_document_html(
                title=title_value,
                document_content=content_value,
                evidence_map=evidence_map,
            )
        except Exception as exc:  # HTML is derived and must not block the knowledge archive.
            emit_event(
                "research.archive_document_html_failed",
                "research_service",
                "Rendering a ResearchItem document HTML asset failed; Markdown archive continues.",
                level="warning",
                data={"job_id": job_id, "document_title": title_value, "error": str(exc) or type(exc).__name__},
            )
        archive_documents.append(prepared)

    research_html = ""
    try:
        research_html = render_question_loop_documents_html(
            title=title,
            documents=archive_documents,
            evidence_map=evidence_map,
        )
    except Exception as exc:  # Full HTML is derived and must not block the knowledge archive.
        emit_event(
            "research.archive_full_html_failed",
            "research_service",
            "Rendering the complete ResearchItem HTML asset failed; Markdown archive continues.",
            level="warning",
            data={"job_id": job_id, "error": str(exc) or type(exc).__name__},
        )

    location, recommendation = _recommend_archive_location(job, result, payload)
    summary = _archive_summary(result, documents)
    tags = payload.get("tags") or payload.get("keywords") or result.get("keywords") or []
    archive_payload = {
        **location,
        "source_job_id": job_id,
        "title": title,
        "summary": summary,
        "researched_at": result.get("completed_at") or job.get("updated_at") or now_text(),
        "documents": archive_documents,
        "html": research_html,
        "sources": payload.get("sources") or result.get("sources") or [],
        "citation_sources": build_question_loop_citation_sources(evidence_map),
        "tags": tags,
        "evidence_count": len(evidence_map or result.get("evidence_pack") or []),
        "comparison_budget": result.get("comparison_budget") or {},
        "matrix_coverage": result.get("matrix_coverage") or {},
        "matrix_gaps_acknowledged": bool(payload.get("acknowledge_matrix_gaps")),
        "quality_review": quality_review,
    }
    preview = knowledge_tree_service.create_preview("archive_research", archive_payload)
    preview["recommendation"] = recommendation
    next_result = {
        **result,
        "archive_preview": {
            "preview_id": preview["preview_id"],
            "state": preview["state"],
            "recommendation": recommendation,
            "archive": preview.get("archive"),
        },
    }
    job_service.update_result(
        job_id,
        status="waiting_approval",
        result=next_result,
        progress_message="archive preview ready",
    )
    return preview


def confirm_research_archive(job_id: str, preview_id: str) -> dict:
    """Commit one ResearchItem archive and enqueue its independent index."""

    job = job_service.get(job_id)
    result = dict(job.get("result") or {})
    existing = result.get("archive_result") if isinstance(result.get("archive_result"), dict) else {}
    if job.get("status") == "succeeded" and existing.get("preview_id") == preview_id:
        return existing

    committed = knowledge_tree_service.confirm_research_archive(preview_id, source_job_id=job_id)
    archive = committed.get("result") or {}
    research_id = str(archive.get("research_id") or "")
    if not research_id:
        raise RuntimeError("archive confirmation returned no research_id")

    index_job_id: str | None = None
    index_error: str | None = None
    try:
        index_job = index_service.create_research_index_job(research_id)
        index_job_id = str(index_job.get("job_id") or "") or None
        if index_job_id:
            knowledge_tree_service.attach_research_index_job(research_id, index_job_id)
    except Exception as exc:  # The knowledge asset is committed even if derived indexing cannot start.
        index_error = str(exc) or type(exc).__name__
        knowledge_tree_service.update_research_index_status(research_id, "failed", error=index_error)

    saved = {
        "status": "saved",
        "preview_id": preview_id,
        **archive,
        "index_job_id": index_job_id,
        "index_error": index_error,
    }
    next_result = {
        **result,
        "status": "succeeded",
        "saved_topic": None,
        "paths": archive.get("document_paths") or [],
        "delivery": _delivery_payload({**result, "paths": archive.get("document_paths") or []}),
        "archive_preview": {**(result.get("archive_preview") or {}), "state": "committed"},
        "archive_result": saved,
        "save_result": saved,
    }
    job_service.update_result(
        job_id,
        status="succeeded",
        result=next_result,
        artifacts=archive.get("document_paths") or [],
        progress_message="succeeded",
    )
    return saved


def _recommend_archive_location(
    job: dict[str, Any], result: dict[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    explicit_keys = {"knowledge_base_id", "theme_id", "new_knowledge_base", "new_theme"}
    if any(payload.get(key) for key in explicit_keys):
        return (
            {key: payload.get(key) for key in explicit_keys if payload.get(key)},
            {"mode": "user_selected", "reason": "使用用户在审批界面指定的归档位置。"},
        )

    topic = str(result.get("suggested_topic") or (job.get("input") or {}).get("topic") or "").strip()
    if "/" in topic:
        segments = [part.strip() for part in topic.replace("\\", "/").split("/") if part.strip()]
        if segments:
            location: dict[str, Any] = {"new_knowledge_base": {"title": segments[0]}}
            if len(segments) > 1:
                location["new_theme"] = {"title": segments[1]}
            return location, {"mode": "topic_path", "reason": "沿用调研任务中已经确认的知识路径。"}

    tree = knowledge_tree_service.list_tree(include_legacy=False)
    knowledge_bases = tree.get("knowledge_bases") or []
    query = " ".join(
        str(value or "")
        for value in [
            topic,
            (job.get("input") or {}).get("user_request"),
            (result.get("report_document") or {}).get("title"),
            (result.get("research_plan") or {}).get("topic_anchor"),
        ]
    )
    candidates: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
    for kb in knowledge_bases:
        kb_score = _archive_match_score(query, kb)
        candidates.append((kb_score, kb, None))
        for theme in kb.get("themes") or []:
            candidates.append((_archive_match_score(query, theme) + 2, kb, theme))
    if candidates:
        score, kb, theme = max(candidates, key=lambda item: item[0])
        if score > 2 or len(knowledge_bases) == 1:
            location = {"knowledge_base_id": kb["id"]}
            if theme is not None and score > 2:
                location["theme_id"] = theme["id"]
            target_name = f"{kb['title']} / {theme['title']}" if theme else str(kb["title"])
            return location, {"mode": "existing_match", "score": score, "reason": f"现有位置“{target_name}”与本次调研主题最接近。"}

    kb_title, theme_title = _default_archive_bucket(query)
    location = {"new_knowledge_base": {"title": kb_title}}
    if theme_title:
        location["new_theme"] = {"title": theme_title}
    return location, {"mode": "new_location", "reason": "现有知识树没有足够接近的位置，建议新建学习空间。"}


def _archive_match_score(query: str, node: dict[str, Any]) -> int:
    haystack = re.sub(r"\s+", "", str(query or "")).casefold()
    values = [node.get("title"), node.get("description"), *(node.get("tags") or [])]
    score = 0
    for value in values:
        needle = re.sub(r"\s+", "", str(value or "")).casefold()
        if not needle:
            continue
        if needle in haystack:
            score += 6
        for token in re.findall(r"[a-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", needle):
            if token in haystack:
                score += 1
    return score


def _default_archive_bucket(query: str) -> tuple[str, str | None]:
    lowered = str(query or "").casefold()
    if any(token in lowered for token in ("agent", "智能体", "codex", "harness", "loop", "memory", "记忆")):
        if "harness" in lowered or "loop" in lowered:
            return "Agent 学习", "Harness"
        if "memory" in lowered or "记忆" in lowered:
            return "Agent 学习", "Memory"
        return "Agent 学习", None
    if any(token in lowered for token in ("投资", "股票", "基金", "finance")):
        return "个人投资", None
    if any(token in lowered for token in ("产品", "设计", "用户体验", "product")):
        return "产品学习", None
    return "个人学习", None


def _archive_summary(result: dict[str, Any], documents: list[dict[str, Any]]) -> str:
    direct = str(result.get("summary") or (result.get("report_document") or {}).get("summary") or "").strip()
    if direct:
        return re.sub(r"\s+", " ", direct)[:1000]
    for document in documents:
        content = str(document.get("content") or "")
        for paragraph in re.split(r"\n\s*\n", content):
            cleaned = re.sub(r"^[#>*\-\s]+", "", paragraph).strip()
            if cleaned and not cleaned.startswith("---"):
                return re.sub(r"\s+", " ", cleaned)[:1000]
    return ""


def approve_research(job_id: str, payload: dict[str, Any]) -> dict:
    job = job_service.get(job_id)
    result = job.get("result") or {}
    quality_gate = result.get("quality_gate") or {}
    if quality_gate and not quality_gate.get("approval_ready"):
        raise ValueError(str(quality_gate.get("reason") or "research documents did not pass quality gate"))
    _require_matrix_gap_acknowledgement(result, payload)
    documents = payload.get("documents") or result.get("documents") or []
    if not documents:
        raise ValueError("research job has no documents to approve")

    topic = str(payload.get("topic") or result.get("suggested_topic") or "").strip()
    if not topic:
        raise ValueError("topic is required to approve research documents")

    is_single_url = result.get("pipeline_mode") == "single_url"
    upstream_quality_approved = bool(quality_gate and quality_gate.get("approval_ready"))
    sources = payload.get("sources") or _source_strings(result.get("sources") or [])
    saved = _save_knowledge_set_impl(
        topic=topic,
        documents=documents,
        sources=sources,
        keywords=payload.get("keywords") or [],
        source_job_id=job_id,
        source_url=result.get("url") if is_single_url else None,
        # Both branches already passed a purpose-built upstream quality gate. Re-running the
        # generic knowledge-document keyword heuristics here can only create a contradictory
        # second approval decision (for example, rejecting a fully analytical section merely
        # because it does not contain one of a short list of Chinese marker words). Jobs without
        # an upstream approval signal retain the generic save-time gate for backward compatibility.
        skip_quality_gate=is_single_url or upstream_quality_approved,
        knowledge_base_metadata=_research_knowledge_base_metadata(job, result, topic),
    )
    next_result = {
        **result,
        "status": "succeeded",
        "saved_topic": saved.get("topic") or topic,
        "paths": saved.get("paths") or [],
        "delivery": _delivery_payload({**result, "paths": saved.get("paths") or []}),
        "save_result": saved,
    }
    # 证据阅读版 HTML（VISION.md 第2节 / question-loop-report-evidence-reader-html-v0.1.md）：
    # only render once the report is truly final (this is the one place a question_loop job
    # transitions to "succeeded" -- see design doc 决策4/5), using the exact documents that just
    # got saved so the HTML can never diverge from what's in the knowledge base. Rendering is a
    # sideshow to the approval itself: any failure here must not block save_knowledge_set's result
    # from reaching the caller (决策10), so it's isolated in its own try/except and only logged.
    # single_url jobs already wrote their brief.html during _run_single_url_research (using the
    # real annotations/synthesis data) -- this block's report_document/evidence_map fields don't
    # exist on a single_url result, so re-running it here would silently overwrite the correct
    # HTML with an empty/garbage one. Skip entirely for single_url.
    if not is_single_url:
        try:
            report_document = next_result.get("report_document") or {}
            report_html = render_question_loop_report_html(
                title=topic,
                report_content=str(report_document.get("content") or ""),
                evidence_map=next_result.get("evidence_map") or [],
            )
            _write_single_url_brief_html_artifact(job_id=job_id, html=report_html)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see comment above
            emit_event(
                "research.question_loop_evidence_reader_html_failed",
                "research_service",
                "Rendering the question_loop evidence reader HTML failed; approval proceeds without it.",
                level="warning",
                data={"job_id": job_id, "error": str(exc) or type(exc).__name__},
            )
    job_service.update_result(
        job_id,
        status="succeeded",
        result=next_result,
        artifacts=saved.get("paths") or [],
        progress_message="succeeded",
    )
    return saved


def _research_knowledge_base_metadata(
    job: dict[str, Any],
    result: dict[str, Any],
    topic: str,
) -> dict[str, Any]:
    report_document = result.get("report_document") or {}
    research_plan = result.get("research_plan") or {}
    job_input = job.get("input") or {}
    title = str(
        report_document.get("title")
        or (result.get("report_blueprint") or {}).get("title")
        or research_plan.get("topic_anchor")
        or topic
    ).strip()
    description = str(research_plan.get("user_request") or job_input.get("user_request") or "").strip()
    return {
        "title": title or topic,
        "description": re.sub(r"\s+", " ", description)[:500],
        "source_mode": str(job_input.get("source_mode") or result.get("source_mode") or "").strip(),
        "source_job_id": job.get("job_id"),
        "notes": "由调研审批落库自动生成。",
        "comparison_budget": result.get("comparison_budget") or {},
        "matrix_coverage": result.get("matrix_coverage") or {},
        "matrix_gaps_acknowledged": bool(
            (result.get("matrix_coverage") or {}).get("requires_gap_acknowledgement")
        ),
    }


def _require_matrix_gap_acknowledgement(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    matrix_coverage = result.get("matrix_coverage") if isinstance(result.get("matrix_coverage"), dict) else {}
    if not matrix_coverage.get("enabled") or matrix_coverage.get("complete"):
        return
    if payload.get("acknowledge_matrix_gaps") is True:
        return
    gap_count = len(matrix_coverage.get("gaps") or [])
    raise ValueError(
        f"该比较报告仍有 {gap_count} 个 Matrix Gap；请在查看缺口后明确确认 acknowledge_matrix_gaps"
    )


def followup_research(job_id: str, payload: dict[str, Any]) -> dict:
    job = job_service.get(job_id)
    result = dict(job.get("result") or {})
    if not result:
        raise ValueError("research job has no result to follow up")
    if (
        result.get("writing_architecture") == EVIDENCE_NATIVE_ARCHITECTURE
        and not bool((result.get("capabilities") or {}).get("targeted_followup"))
    ):
        raise ValueError(
            "targeted follow-up is not enabled for evidence_native_eplus jobs; "
            "the request was not started and the existing report was not modified"
        )

    query = _followup_query_from_payload(payload, job.get("input") or {})
    action_id = f"followup_{uuid.uuid4().hex[:8]}"
    action = {
        "id": action_id,
        "issue_id": payload.get("issue_id"),
        "issue_type": payload.get("issue_type"),
        "evidence_id": payload.get("evidence_id"),
        "subject": payload.get("subject"),
        "comparison_axis": payload.get("comparison_axis"),
        "query": query,
        "reason": payload.get("reason"),
        "status": "running",
        "requested_at": now_text(),
    }
    coverage = dict(result.get("coverage") or {})
    followup_queries = _dedupe_strings([*(coverage.get("followup_queries") or []), query])
    coverage["followup_queries"] = followup_queries
    coverage["reason"] = f"已启动 targeted follow-up：{query}"
    pending_result = {
        **result,
        "message": "补查已启动，正在后台执行 targeted follow-up。",
        "coverage": coverage,
        "followup_actions": [*(result.get("followup_actions") or []), action],
    }
    job_service.update_result(
        job_id,
        status="running",
        result=pending_result,
        progress_message="followup running",
    )
    thread = threading.Thread(
        target=_run_targeted_followup_background,
        kwargs={
            "job_id": job_id,
            "job": job,
            "result": result,
            "payload": payload,
            "query": query,
            "action": action,
            "coverage": coverage,
            "followup_queries": followup_queries,
        },
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job_id,
        "followup_job_id": action_id,
        "status": "followup_started",
        "message": "补查已启动，正在后台执行。",
        "followup_queries": followup_queries,
    }


def _run_targeted_followup_background(
    *,
    job_id: str,
    job: dict[str, Any],
    result: dict[str, Any],
    payload: dict[str, Any],
    query: str,
    action: dict[str, Any],
    coverage: dict[str, Any],
    followup_queries: list[str],
) -> None:
    try:
        _run_targeted_followup(
            job_id=job_id,
            job=job,
            result=result,
            payload=payload,
            query=query,
            action=action,
            coverage=coverage,
            followup_queries=followup_queries,
        )
    except Exception as exc:
        _record_followup_failure(
            job_id=job_id,
            job=job,
            result=result,
            action=action,
            coverage=coverage,
            followup_queries=followup_queries,
            error=exc,
        )


def _record_followup_failure(
    *,
    job_id: str,
    job: dict[str, Any],
    result: dict[str, Any],
    action: dict[str, Any],
    coverage: dict[str, Any],
    followup_queries: list[str],
    error: Exception,
) -> None:
    coverage["followup_queries"] = followup_queries
    coverage["reason"] = f"已记录补查请求，但 targeted follow-up 执行失败：{error}"
    failed_action = {**action, "status": "failed", "error": str(error), "resolved_at": now_text()}
    next_result = {
        **result,
        "message": f"补查失败：{error}",
        "coverage": coverage,
        "followup_actions": _replace_followup_action(result.get("followup_actions") or [], failed_action),
    }
    fallback_status = job.get("status") if job.get("status") in {"waiting_approval", "succeeded"} else "waiting_approval"
    job_service.update_result(
        job_id,
        status=fallback_status,
        result=next_result,
        progress_message="followup failed",
    )


def _run_targeted_followup(
    *,
    job_id: str,
    job: dict[str, Any],
    result: dict[str, Any],
    payload: dict[str, Any],
    query: str,
    action: dict[str, Any],
    coverage: dict[str, Any],
    followup_queries: list[str],
) -> dict:
    input_data = dict(job.get("input") or {})
    channel_id = str(input_data.get("channel_id") or "").strip()
    user_request = str(input_data.get("user_request") or result.get("research_plan", {}).get("user_request") or "").strip()
    if not user_request:
        raise ValueError("research job input is missing user_request")
    strategy = str(result.get("research_strategy") or _research_strategy(input_data))
    depth = str(result.get("depth") or _research_depth(input_data))
    research_plan = result.get("research_plan") if isinstance(result.get("research_plan"), dict) else {}
    titles = _followup_titles(input_data=input_data, result=result, strategy=strategy)
    topic = str(result.get("suggested_topic") or input_data.get("topic") or _suggest_topic(input_data, user_request))
    if not research_plan:
        research_plan = _build_research_plan(user_request=user_request, topic=topic, titles=titles, strategy=strategy)
    query_plan = result.get("query_plan") if isinstance(result.get("query_plan"), dict) else {}
    if not query_plan:
        query_plan = _build_query_plan(user_request=user_request, research_plan=research_plan, strategy=strategy, depth=depth)

    target = _followup_target_context(payload=payload, result=result, research_plan=research_plan, query=query)
    task = _build_targeted_followup_task(
        user_request=user_request,
        topic=topic,
        titles=titles,
        strategy=strategy,
        depth=depth,
        research_plan=research_plan,
        query_plan=query_plan,
        target=target,
    )

    _step(job_id, "search", "running", f"正在补查：{query}")
    _step(job_id, "fetch", "running", "正在抓取补查材料")
    raw_retrieval = web_worker_tool.invoke({"task_or_url": task})
    incoming = parse_tool_json(raw_retrieval)
    retrieval = _merge_retrieval(_retrieval_from_result(result), incoming)

    sources = _coerce_list(retrieval.get("sources"))
    materials = _coerce_list(retrieval.get("materials"))
    selected_chunks = _coerce_list(retrieval.get("selected_chunks"))
    evidence_pack = _coerce_list(retrieval.get("evidence_pack"))
    evidence_map = _coerce_list(retrieval.get("evidence_map"))
    evidences = _coerce_list(retrieval.get("evidences"))
    key_facts = _coerce_list(retrieval.get("key_facts"))
    diagnostics = incoming.get("_diagnostics") or {}
    documents_raw = _coerce_list(retrieval.get("documents"))
    _step(job_id, "search", "succeeded", f"补查后累计找到 {len(_source_items(sources))} 个候选来源", items=_source_items(sources[:8]))
    _step(job_id, "fetch", "succeeded", f"补查后累计抓取 {len(documents_raw)} 份网页材料", items=_document_items(documents_raw[:6]))
    _step(
        job_id,
        "material_pool",
        "succeeded",
        f"补查后素材池包含 {len(materials)} 份材料",
        items=[*_diagnostic_items(diagnostics, "material_pool"), *_material_items(materials[:6])],
    )
    _step(
        job_id,
        "chunk_select",
        "succeeded",
        f"补查后筛选 {len(selected_chunks)} 个正文片段",
        items=[*_diagnostic_items(diagnostics, "material_pool"), *_chunk_items(selected_chunks[:6])],
    )
    _step(
        job_id,
        "evidence",
        "succeeded",
        f"补查后累计整理 {len(evidence_map) or len(evidence_pack) or len(evidences) or len(key_facts)} 条证据",
        items=[*_diagnostic_items(diagnostics, "section_evidence_extraction"), *_evidence_items(evidence_map or evidence_pack or evidences or key_facts)],
    )

    previous_rounds = _coerce_list((result.get("retrieval") or {}).get("rounds") if isinstance(result.get("retrieval"), dict) else [])
    round_index = len(previous_rounds) + 1
    max_rounds = max(round_index, int(input_data.get("max_rounds") or coverage.get("max_rounds") or round_index or 1))
    _step(job_id, "coverage", "running", "正在重新检查补查后的覆盖度")
    next_coverage = invoke_tool(
        research_coverage_check,
        {
            "required_titles": titles,
            "retrieval_result": retrieval,
            "user_request": user_request,
            "round_index": round_index,
            "max_rounds": max_rounds,
        },
    )
    next_coverage = _augment_coverage_with_evidence_diagnostics(
        coverage={**next_coverage, "followup_queries": followup_queries},
        retrieval=retrieval,
        research_plan=research_plan,
        user_request=user_request,
        round_index=round_index,
        max_rounds=max_rounds,
    )
    if query not in [str(item) for item in next_coverage.get("followup_queries") or []]:
        next_coverage["followup_queries"] = _dedupe_strings([*(next_coverage.get("followup_queries") or []), query])
    _step(job_id, "coverage", "succeeded", str(next_coverage.get("reason") or "补查覆盖度检查完成"), items=_coverage_items(next_coverage))

    _step(job_id, "draft", "running", "正在用补查证据重新生成文档")
    report_blueprint = _build_report_blueprint(
        user_request=user_request,
        research_strategy=str(input_data.get("research_strategy") or ""),
        titles=titles,
        research_plan=research_plan,
        retrieval=retrieval,
    )
    documents = _synthesize_documents(user_request, topic, titles, retrieval, next_coverage, research_plan, report_blueprint)
    _step(job_id, "validation", "running", "正在重新校验证据引用")
    initial_validation = _validate_documents_against_evidence(documents, evidence_map or evidence_pack)
    if _needs_citation_repair(initial_validation):
        documents = _repair_document_citations(documents, retrieval, research_plan, initial_validation)
    grounding = _validate_synthesis_grounding(documents, retrieval)
    validation = _validate_documents_against_evidence(documents, evidence_map or evidence_pack)
    evidence_map = _mark_evidence_usage(evidence_map, documents)
    retrieval["evidence_map"] = evidence_map
    quality_gate = _quality_gate(documents, next_coverage, grounding, validation)
    _step(job_id, "draft", "succeeded", f"补查后生成 {len(documents)} 篇待审批文档", items=[{"label": doc["title"], "value": f"{len(doc.get('content', ''))} chars"} for doc in documents])
    _step(job_id, "validation", "succeeded" if validation.get("status") == "passed" else "failed", str(validation.get("reason") or "补查证据验收完成"), items=_validation_items(validation))

    resolved = bool(quality_gate.get("approval_ready"))
    unresolved_weak_sections = [
        item for item in (quality_gate.get("checks") or {}).get("weak_sections") or []
        if isinstance(item, dict)
    ]
    completed_action = {
        **action,
        "status": "resolved" if resolved else "needs_more_evidence",
        "resolved_at": now_text(),
        "added_sources": max(0, len(sources) - len(_coerce_list(result.get("sources")))),
        "added_evidence": max(0, len(evidence_map) - len(_coerce_list(result.get("evidence_map")))),
        "target": target,
        "quality_checks": quality_gate.get("checks") or {},
        "unresolved_weak_sections": unresolved_weak_sections,
    }
    research_rounds = [
        *previous_rounds,
        {
            "round_index": round_index,
            "type": "targeted_followup",
            "query": query,
            "coverage": next_coverage,
            "source_count": len(sources),
            "document_count": len(documents_raw),
            "evidence_pack_count": len(evidence_pack),
            "diagnostics": incoming.get("_diagnostics") or {},
        },
    ]
    next_result = {
        **result,
        "status": "waiting_approval",
        "channel_id": channel_id or result.get("channel_id"),
        "message": "补查完成，质量门已通过。" if resolved else "补查完成，但仍需要更多证据。",
        "coverage": next_coverage,
        "quality_gate": quality_gate,
        "grounding": grounding,
        "validation": validation,
        "initial_validation": initial_validation,
        "documents": documents,
        "sources": sources,
        "evidence_pack": evidence_pack,
        "materials": materials,
        "selected_chunks": selected_chunks,
        "evidence_map": evidence_map,
        "suggested_topic": topic,
        "retrieval": {
            "key_facts": key_facts,
            "evidences": evidences,
            "evidence_diagnostics": retrieval.get("evidence_diagnostics") or {},
            "gaps": _coerce_list(retrieval.get("gaps")),
            "provider_attempts": _coerce_list(retrieval.get("provider_attempts")),
            "rounds": research_rounds,
            "diagnostics": retrieval.get("_diagnostics") or {},
        },
        "report_blueprint": report_blueprint,
        "followup_actions": _replace_followup_action(result.get("followup_actions") or [], completed_action),
        "document_titles": [str(doc.get("title") or "") for doc in documents],
    }
    next_result["report_document"] = _build_integrated_report_document(
        user_request=user_request,
        result=next_result,
        input_data=job.get("input") or {},
    )
    artifact_paths = _write_research_review_artifacts(job_id=job_id, result=next_result, input_data=job.get("input") or {})
    next_result["artifacts"] = artifact_paths
    next_result["delivery"] = _delivery_payload({**next_result, "paths": artifact_paths}, job.get("input") or {})
    job_service.update_result(
        job_id,
        status="waiting_approval",
        result=next_result,
        artifacts=next_result["artifacts"],
        progress_message="followup resolved" if resolved else "followup needs_more_evidence",
    )
    _step(
        job_id,
        "approval",
        "waiting_approval",
        "补查后文档可审批落库" if resolved else "补查后仍有证据质量问题，暂不建议落库",
        items=[
            {"label": "documents", "value": len(documents)},
            {"label": "topic", "value": topic},
            {"label": "quality", "value": quality_gate["status"]},
        ],
    )
    return {
        "job_id": job_id,
        "status": "resolved" if resolved else "needs_more_evidence",
        "message": next_result["message"],
        "followup_queries": _coerce_list(next_coverage.get("followup_queries")),
        "unresolved_weak_sections": unresolved_weak_sections,
    }


def _replace_followup_action(actions: list[Any], next_action: dict[str, Any]) -> list[Any]:
    action_id = str(next_action.get("id") or "").strip()
    replaced = False
    output = []
    for action in actions:
        if isinstance(action, dict) and action_id and str(action.get("id") or "").strip() == action_id:
            output.append({**action, **next_action})
            replaced = True
        else:
            output.append(action)
    if not replaced:
        output.append(next_action)
    return output


def _followup_query_from_payload(payload: dict[str, Any], input_data: dict[str, Any]) -> str:
    explicit = str(payload.get("query") or "").strip()
    if explicit:
        return explicit
    subject = str(payload.get("subject") or "").strip()
    axis = str(payload.get("comparison_axis") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    user_request = str(input_data.get("user_request") or "").strip()
    parts = [subject, axis, reason or user_request, "权威资料", "官方文档", "案例", "evidence"]
    return " ".join(part for part in parts if part).strip() or "补查证据 权威资料 官方文档"


def _followup_titles(*, input_data: dict[str, Any], result: dict[str, Any], strategy: str) -> list[str]:
    documents = [doc for doc in result.get("documents") or [] if isinstance(doc, dict)]
    titles = [str(doc.get("title") or "").strip() for doc in documents if str(doc.get("title") or "").strip()]
    if titles:
        return titles
    result_titles = [
        str(section.get("title") or "").strip()
        for section in (result.get("research_plan") or {}).get("sections") or []
        if isinstance(section, dict) and str(section.get("title") or "").strip()
    ]
    if result_titles:
        return result_titles
    return _required_titles(input_data, strategy)


def _retrieval_from_result(result: dict[str, Any]) -> dict[str, Any]:
    retrieval_meta = result.get("retrieval") if isinstance(result.get("retrieval"), dict) else {}
    return {
        "sources": _coerce_list(result.get("sources")),
        "documents": _coerce_list(retrieval_meta.get("documents") or result.get("source_documents")),
        "materials": _coerce_list(result.get("materials")),
        "selected_chunks": _coerce_list(result.get("selected_chunks")),
        "evidences": _coerce_list(retrieval_meta.get("evidences")),
        "key_facts": _coerce_list(retrieval_meta.get("key_facts")),
        "evidence_pack": _coerce_list(result.get("evidence_pack")),
        "evidence_map": _coerce_list(result.get("evidence_map")),
        "gaps": _coerce_list(retrieval_meta.get("gaps")),
        "provider_attempts": _coerce_list(retrieval_meta.get("provider_attempts")),
        "evidence_diagnostics": retrieval_meta.get("evidence_diagnostics") or {},
        "_diagnostics": retrieval_meta.get("diagnostics") or {},
    }


def _followup_target_context(
    *,
    payload: dict[str, Any],
    result: dict[str, Any],
    research_plan: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    subject = str(payload.get("subject") or "").strip()
    axis = str(payload.get("comparison_axis") or "").strip()
    evidence_id = str(payload.get("evidence_id") or "").strip()
    target_section = _followup_section_for_target(subject=subject, axis=axis, evidence_id=evidence_id, result=result, research_plan=research_plan)
    target_cell = _followup_matrix_cell(subject=subject, axis=axis, research_plan=research_plan)
    question = str((target_cell or {}).get("question") or "").strip()
    if not question:
        question = " ".join(part for part in [subject, axis, str(payload.get("reason") or "").strip()] if part).strip()
    return {
        "issue_id": payload.get("issue_id"),
        "issue_type": payload.get("issue_type"),
        "evidence_id": evidence_id,
        "subject": subject,
        "comparison_axis": axis,
        "query": query,
        "reason": payload.get("reason"),
        "section": target_section,
        "matrix_cell": target_cell,
        "question": question or query,
    }


def _followup_section_for_target(
    *,
    subject: str,
    axis: str,
    evidence_id: str,
    result: dict[str, Any],
    research_plan: dict[str, Any],
) -> dict[str, Any]:
    evidence_rows = [
        item for item in [
            *_coerce_list(result.get("evidence_map")),
            *_coerce_list(result.get("evidence_pack")),
            *_coerce_list(result.get("selected_chunks")),
        ]
        if isinstance(item, dict)
    ]
    section_id = ""
    for item in evidence_rows:
        ids = {str(item.get(key) or "").strip() for key in ("id", "evidence_id", "chunk_id")}
        if evidence_id and evidence_id in ids:
            section_id = str(item.get("section_id") or "").strip()
            break
    if not section_id and (subject or axis):
        for item in evidence_rows:
            if subject and str(item.get("subject") or "").strip() != subject:
                continue
            item_axis = str(item.get("comparison_axis") or item.get("axis") or "").strip()
            if axis and item_axis != axis:
                continue
            section_id = str(item.get("section_id") or "").strip()
            if section_id:
                break
    sections = [item for item in research_plan.get("sections") or [] if isinstance(item, dict)]
    if section_id:
        for section in sections:
            if str(section.get("id") or "").strip() == section_id:
                return section
    if subject or axis:
        comparison = _comparison_section(research_plan)
        if comparison:
            return comparison
    return sections[0] if sections else {}


def _followup_matrix_cell(*, subject: str, axis: str, research_plan: dict[str, Any]) -> dict[str, Any]:
    if not subject and not axis:
        return {}
    for cell in research_plan.get("matrix_cells") or []:
        if not isinstance(cell, dict):
            continue
        if subject and str(cell.get("subject") or "").strip() != subject:
            continue
        cell_axis = str(cell.get("axis") or cell.get("comparison_axis") or "").strip()
        if axis and cell_axis != axis:
            continue
        return cell
    return {}


def _build_targeted_followup_task(
    *,
    user_request: str,
    topic: str,
    titles: list[str],
    strategy: str,
    depth: str,
    research_plan: dict[str, Any],
    query_plan: dict[str, Any],
    target: dict[str, Any],
) -> str:
    section = target.get("section") if isinstance(target.get("section"), dict) else {}
    matrix_cell = target.get("matrix_cell") if isinstance(target.get("matrix_cell"), dict) else {}
    target_payload = {
        "research_plan": research_plan,
        "query_plan": {
            **query_plan,
            "queries": [
                {
                    "section_id": section.get("id") or "",
                    "section_title": section.get("title") or "",
                    "question_id": matrix_cell.get("id") or "",
                    "question": target.get("question") or target.get("query"),
                    "query": target.get("query"),
                    "subject": target.get("subject") or matrix_cell.get("subject"),
                    "comparison_axis": target.get("comparison_axis") or matrix_cell.get("axis"),
                    "matrix_cell_id": matrix_cell.get("id"),
                }
            ],
        },
        "required_titles": titles,
        "followup_target": target,
    }
    target_lines = [
        f"issue_type={target.get('issue_type') or ''}",
        f"section={section.get('title') or section.get('id') or ''}",
        f"subject={target.get('subject') or ''}",
        f"comparison_axis={target.get('comparison_axis') or ''}",
        f"question={target.get('question') or ''}",
        f"query={target.get('query') or ''}",
        f"reason={target.get('reason') or ''}",
    ]
    fetch_limit = 8 if depth == "deep" else 5
    return (
        "source_mode=web targeted follow-up research.\n"
        f"用户需求：{user_request}\n"
        f"topic 使用：{topic}\n"
        f"research_strategy={strategy}\n"
        f"depth={depth}\n"
        f"fetch_url_limit={fetch_limit}\n"
        f"必须保留文档标题：{'、'.join(titles)}\n"
        "本轮只解决这个薄弱问题：\n"
        + "\n".join(f"- {line}" for line in target_lines if line.split("=", 1)[-1])
        + "\n"
        "检索要求：优先官方文档、产品文档、学术/权威资料和可引用案例；围绕 subject + comparison_axis 找直接证据。\n"
        "证据要求：至少尝试提供 2 条 evidence_map；每条必须包含 id、claim、evidence_text、source_url、section_id、question_id。\n"
        "如果是对比矩阵单元，必须保留 subject、comparison_axis、matrix_cell_id，避免把证据归到泛化章节。\n"
        "结构化研究计划 JSON（worker 必须优先使用）：\n"
        "<<<RESEARCH_PAYLOAD_JSON\n"
        f"{json.dumps(target_payload, ensure_ascii=False)}\n"
        "RESEARCH_PAYLOAD_JSON>>>\n"
        "请输出 JSON retrieval result，只包含本轮新增或高度相关的 sources/materials/selected_chunks/evidence_map/evidence_pack。"
    )


def revise_research_sections(job_id: str, payload: dict[str, Any]) -> dict:
    """Manually revise one or more report sections, leaving every other section untouched.

    Unlike `followup_research` (single evidence issue, may fetch new material, ultimately
    rewrites the *entire* document set via the legacy `_synthesize_documents`), this reuses
    `revise_report_sections` — the question_loop-native, section-isolated rewrite path already
    used by the automatic quality-gate revision pass. Sections not named in `section_ids` come
    back as the exact same object `revise_report_sections` was given, so they cannot drift.
    Only supports jobs produced by `pipeline_mode="question_loop"`; legacy-pipeline jobs raise
    a clear error instead of silently doing the wrong thing (see the
    feishu-channel-router-legacy-pipeline-bug lesson: never let a "should be safe" op quietly
    fall onto the untested legacy path).
    """
    job = job_service.get(job_id)
    if job.get("status") != "waiting_approval":
        raise ValueError(f"research job is not waiting for review (status={job.get('status')}); cannot revise sections now")

    result = dict(job.get("result") or {})
    if not result:
        raise ValueError("research job has no result to revise")
    if result.get("pipeline_mode") != "question_loop":
        raise ValueError("this research job does not support section-level revision (only question_loop jobs do)")
    if (
        result.get("writing_architecture") == EVIDENCE_NATIVE_ARCHITECTURE
        and not bool((result.get("capabilities") or {}).get("section_revision"))
    ):
        raise ValueError(
            "section revision is not enabled for evidence_native_eplus jobs; "
            "the request was not started and the existing report was not modified"
        )

    blueprint = result.get("report_blueprint") or {}
    findings = result.get("findings") or []
    evidence_map = result.get("evidence_map") or []
    if not blueprint or not findings or not evidence_map:
        raise ValueError("research job is missing report_blueprint/findings/evidence_map required for section revision")

    section_ids = _dedupe_strings([str(item) for item in payload.get("section_ids") or [] if str(item).strip()])
    if not section_ids:
        raise ValueError("section_ids is required")
    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("instruction is required")

    # "editorial" (executive_summary + conclusion) is not a blueprint section -- it's the
    # separate _edit_report output -- but it's still a valid manual-revision target: the
    # automatic quality-gate revision loop already reaches it (revalidate_section_ids.add(
    # "editorial") above), and _entailment_revision_issues already tags entailment issues on
    # it. See docs/modules/deep-research-step4-approval-and-delivery-discussion-v0.1.md 商榷点一.
    valid_section_ids = {str(section.get("section_id") or "") for section in blueprint.get("sections") or []}
    valid_section_ids.add("editorial")
    invalid_ids = [sid for sid in section_ids if sid not in valid_section_ids]
    if invalid_ids:
        raise ValueError(f"unknown section_ids: {', '.join(invalid_ids)}")

    evidence_gaps = _section_revision_evidence_gaps(result, section_ids)
    fixable_issues = _section_revision_fixable_issues(result, section_ids)
    fixable_by_section: dict[str, list[dict[str, Any]]] = {}
    for issue in fixable_issues:
        fixable_sid = str(issue.get("section_id") or "")
        if fixable_sid:
            fixable_by_section.setdefault(fixable_sid, []).append(issue)
    # A section is genuinely evidence-blocked only when it has a coverage gap that wording
    # revision cannot close AND no entailment/critic issue targeting it that wording revision
    # CAN fix without new evidence. Mirrors the 08-03 validation-stage fix at line ~2641: a
    # retrieval-time coverage gap must not suppress revision-fixable prose defects
    # (label_mismatch / unsupported / critic writing issues). In
    # `case-targeted-wording-revision`, a targeted wording-only revision bounced as
    # evidence_insufficient even though the label_mismatch claims were pure wording
    # defects the writer can fix by qualifying or deleting the over-reaching clause -- the
    # section-level guard here is the user-side path the 08-03 validation-stage fix did not
    # cover, so the same "coverage gap skips fixable issues" failure still reproduces here.
    blocked_section_ids = {
        str(gap.get("section_id") or "")
        for gap in evidence_gaps
        if str(gap.get("section_id") or "") not in fixable_by_section
    }
    revisable_section_ids = [sid for sid in section_ids if sid not in blocked_section_ids]
    if evidence_gaps and not revisable_section_ids:
        # Every requested section is evidence-blocked -- nothing left to write, reject outright
        # rather than starting a revision run with zero targets.
        return {
            "job_id": job_id,
            "status": "evidence_insufficient",
            "section_ids": section_ids,
            "message": (
                "目标章节的原始证据尚未满足研究问题覆盖门槛；当前章节返修只会重复使用原证据，"
                "因此已停止文字改写。请先针对这些问题补查证据，再重新生成章节。"
            ),
            "recommended_action": "targeted_retrieval",
            "evidence_gaps": evidence_gaps,
        }

    revision_id = f"revision_{uuid.uuid4().hex[:8]}"
    # A single batch request can legitimately mix a wording-only section (e.g. "01章重复内容需要
    # 合并精简") with an evidence-blocked one (e.g. "03章缺乏案例代码"). Rejecting the whole batch
    # just because one target lacks evidence made the wording-only section collateral damage --
    # in `case-mixed-revision-source-policy`, a multi-section revision bounced entirely because
    # one target was evidence-insufficient, even though at least one other
    # section's issue (duplicate content) had nothing to do with evidence. Revise whatever has
    # enough evidence now and report the rest as a partial block instead of an all-or-nothing one.
    # Each revisable section gets the user's manual_revision instruction plus any fixable
    # entailment/critic issues targeting it, so the writer can correct wording defects it can
    # fix without new evidence (label_mismatch / unsupported claims) alongside the user's
    # requested change -- without this, the manual_revision rules would tell the writer to
    # preserve every unflagged sentence, leaving the label_mismatch claims untouched.
    issues: list[dict[str, Any]] = []
    for sid in revisable_section_ids:
        issues.append({"section_id": sid, "type": "manual_revision", "instruction": instruction})
        issues.extend(fixable_by_section.get(sid, []))
    # Computed once, up front, and written into `action` itself (not just the synchronous
    # response below) so the async Feishu revision-result bridge can also report which sections
    # were skipped for lacking evidence -- see docs/modules/deep-research-step4-approval-and-
    # delivery-discussion-v0.1.md 商榷点二. Only surface sections that were actually blocked (no
    # fixable issues targeted them) as skipped -- sections with fixable entailment/critic issues
    # were revised in place, so they must not appear here, otherwise the tool layer/Feishu message
    # would tell the user they were "跳过未返修" when in fact they were just rewritten.
    skipped_evidence_insufficient: dict[str, Any] | None = None
    if evidence_gaps:
        skipped_gaps = [gap for gap in evidence_gaps if str(gap.get("section_id") or "") in blocked_section_ids]
        if skipped_gaps:
            skipped_evidence_insufficient = {
                "section_ids": sorted(blocked_section_ids),
                "message": (
                    "以下章节因原始证据不足被跳过，未参与本次返修，仅重写了证据充分的目标章节；"
                    "如需处理这些章节，请先做针对性补查。"
                ),
                "evidence_gaps": skipped_gaps,
            }
    action = {
        "id": revision_id,
        "section_ids": revisable_section_ids,
        "instruction": instruction,
        "status": "running",
        "requested_at": now_text(),
        "skipped_evidence_insufficient": skipped_evidence_insufficient,
    }
    pending_result = {
        **result,
        "message": f"章节修改已启动，正在后台执行（{len(revisable_section_ids)} 节）。",
        "revision_actions": [*(result.get("revision_actions") or []), action],
    }
    job_service.update_result(
        job_id,
        status="running",
        result=pending_result,
        progress_message="section revision running",
    )
    thread = threading.Thread(
        target=_run_section_revision_background,
        kwargs={
            "job_id": job_id,
            "result": result,
            "input_data": job.get("input") or {},
            "blueprint": blueprint,
            "findings": findings,
            "evidence_map": evidence_map,
            "issues": issues,
            "action": action,
        },
        daemon=True,
    )
    thread.start()
    response = {
        "job_id": job_id,
        "revision_id": revision_id,
        "status": "revision_started",
        "section_ids": revisable_section_ids,
    }
    if skipped_evidence_insufficient:
        response["skipped_evidence_insufficient"] = skipped_evidence_insufficient
    return response


def _section_revision_evidence_gaps(
    result: dict[str, Any], section_ids: list[str]
) -> list[dict[str, Any]]:
    """Return target-section question gaps that wording-only revision cannot repair.

    Older/legacy result payloads may not contain question_states; those retain the previous
    behavior. Current question_loop jobs always persist them, so a hard coverage failure can be
    surfaced before the writer turns weak or off-topic evidence into more polished prose.
    """
    states = [item for item in result.get("question_states") or [] if isinstance(item, dict)]
    if not states:
        return []
    output: list[dict[str, Any]] = []
    for section_id in section_ids:
        for state in states:
            if str(state.get("section_id") or "") != section_id:
                continue
            coverage = state.get("coverage") if isinstance(state.get("coverage"), dict) else {}
            if coverage.get("sufficient") is True or str(state.get("status") or "") == "sufficient":
                continue
            blocking_gaps = [
                {
                    key: gap.get(key)
                    for key in ("gap_type", "criterion_id", "missing_fact", "current", "target")
                    if gap.get(key) is not None
                }
                for gap in coverage.get("gaps") or []
                if isinstance(gap, dict) and gap.get("blocking") is not False
            ]
            output.append({
                "section_id": section_id,
                "question_id": str(state.get("question_id") or ""),
                "question": str(state.get("question") or ""),
                "reason": str(coverage.get("reason") or state.get("stop_reason") or "insufficient coverage"),
                "evidence_count": int(coverage.get("evidence_count") or 0),
                "source_count": int(coverage.get("source_count") or 0),
                "blocking_gaps": blocking_gaps,
            })
    return output


def _section_revision_fixable_issues(
    result: dict[str, Any], section_ids: list[str]
) -> list[dict[str, Any]]:
    """Entailment and critic issues targeting the requested sections that wording revision CAN
    fix without any new evidence -- label_mismatch / unsupported / non_atomic claims (delete or
    qualify the over-reaching clause) and report_quality critic writing issues
    (information_design / missing_tradeoffs / weak_narrative / structure / synthesis).

    These are the same issue classes `_entailment_revision_issues` and the validation-stage
    `report_quality["issues"]` collection already feed into the automatic revision pass at line
    ~2641. This helper exposes them on the user-side `revise_research_sections` path so a
    section whose question_state is `exhausted` (a retrieval-time gap that rewriting cannot
    close) is NOT hard-blocked when it also carries fixable prose defects -- the writer fixes
    what it can fix without new evidence, and the retrieval gap stays as a diagnostic. See
    `case-targeted-wording-revision`: a targeted wording-only revision bounced as
    evidence_insufficient even though the label_mismatch claims needed no new material.

    `target_ids` (derived from the caller's `section_ids`) can now include `"editorial"` --
    once the top-of-function whitelist accepts it, the final filter below matches it the same
    way it matches any `sec_XX` id, so a manual "editorial" revision request automatically picks
    up any entailment issue `_entailment_revision_issues` already tagged `section_id="editorial"`
    with, aligning editorial with existing sec_XX behavior.
    """
    target_ids = set(section_ids)
    issues: list[dict[str, Any]] = []
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    entailment = validation.get("entailment") if isinstance(validation.get("entailment"), dict) else {}
    issues.extend(_entailment_revision_issues(entailment))
    report_quality = result.get("report_quality") if isinstance(result.get("report_quality"), dict) else {}
    if report_quality.get("status") != "passed":
        for issue in report_quality.get("issues") or []:
            if not isinstance(issue, dict) or issue.get("self_check") == "refuted":
                continue
            section_id = str(issue.get("section_id") or "")
            if section_id in target_ids or section_id in {"full_report", "overall"}:
                issues.append(issue)
    # Filter to issues targeting the requested sections (or whole-report issues that apply to
    # every revised section). full_report/overall critic issues (e.g. "add a limitations
    # paragraph") are revision-fixable and apply to the report as a whole, so they attach to
    # every requested section the writer touches -- drop them only when the user named zero
    # sections, which can't happen here (section_ids is required and validated above).
    return [issue for issue in issues if str(issue.get("section_id") or "") in target_ids or str(issue.get("section_id") or "") in {"full_report", "overall"}]


def _run_section_revision_background(
    *,
    job_id: str,
    result: dict[str, Any],
    input_data: dict[str, Any],
    blueprint: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    action: dict[str, Any],
) -> None:
    try:
        _run_section_revision(
            job_id=job_id,
            result=result,
            input_data=input_data,
            blueprint=blueprint,
            findings=findings,
            evidence_map=evidence_map,
            issues=issues,
            action=action,
        )
    except Exception as exc:
        _record_section_revision_failure(job_id=job_id, result=result, action=action, error=exc)


def _record_section_revision_failure(*, job_id: str, result: dict[str, Any], action: dict[str, Any], error: Exception) -> None:
    failed_action = {**action, "status": "failed", "error": str(error), "resolved_at": now_text()}
    next_result = {
        **result,
        "message": f"章节修改失败：{error}",
        "revision_actions": _replace_followup_action(result.get("revision_actions") or [], failed_action),
    }
    job_service.update_result(
        job_id,
        status="waiting_approval",
        result=next_result,
        progress_message="section revision failed",
    )


def _run_section_revision(
    *,
    job_id: str,
    result: dict[str, Any],
    input_data: dict[str, Any],
    blueprint: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    action: dict[str, Any],
) -> None:
    user_request = str(input_data.get("user_request") or (result.get("research_plan") or {}).get("user_request") or "").strip()
    written = {
        "documents": result.get("documents") or [],
        "report_document": result.get("report_document") or {},
        "report_status": result.get("report_status"),
        "editorial": result.get("editorial") or {},
        "diagnostics": (result.get("pipeline_diagnostics") or {}).get("writer") or {},
    }
    revised = revise_report_sections(
        user_request=user_request,
        written=written,
        blueprint=blueprint,
        findings=findings,
        evidence_map=evidence_map,
        issues=issues,
        model=_research_writer_json_model,
    )
    documents = revised["documents"]
    report_document = revised["report_document"]
    report_status = str(revised.get("report_status") or result.get("report_status") or "partial_report")
    per_section_outcome = {
        str(doc.get("section_id") or doc.get("title") or ""): doc.get("revision_diagnostics") or {}
        for doc in documents
        if doc.get("revision_diagnostics")
    }
    if (revised.get("diagnostics") or {}).get("editorial_revised"):
        # _edit_report (unlike the per-section _accept_revision gate) has no reject path -- it
        # always returns something to use, even a template fallback -- so a manual editorial
        # revision that actually ran must be reported as accepted. editorial is not a
        # `documents` entry, so without this it never lands in per_section_outcome, and
        # _render_section_revision_result_message's fallback lookup resolves to `{}`, which
        # renders as a false "⚠️ editorial：修改被拒绝（未达到质量护栏要求）" even though the
        # revision succeeded. See docs/modules/deep-research-step4-approval-and-delivery-
        # discussion-v0.1.md 商榷点一.
        per_section_outcome["editorial"] = {"attempted": True, "accepted": True}
    resolved_action = {
        **action,
        "status": "resolved",
        "resolved_at": now_text(),
        "per_section_outcome": per_section_outcome,
    }
    next_result = {
        **result,
        "status": "waiting_approval",
        "documents": documents,
        "report_document": report_document,
        "report_status": report_status,
        "editorial": revised.get("editorial") or result.get("editorial"),
        "message": "章节修改已完成，请重新审核。",
        "revision_actions": _replace_followup_action(result.get("revision_actions") or [], resolved_action),
        "revision_history": [
            *(result.get("revision_history") or []),
            {
                "revision_id": action.get("id"),
                "section_ids": action.get("section_ids"),
                "instruction": action.get("instruction"),
                "requested_at": action.get("requested_at"),
                "resolved_at": resolved_action["resolved_at"],
                "per_section_outcome": per_section_outcome,
            },
        ],
    }
    # If at least one targeted section was actually accepted, the delivered artifacts (local
    # review.md and, if applicable, the Feishu doc created at first delivery) are now stale — the
    # original _delivery_payload call only ever ran once, at initial waiting_approval, and
    # nothing else in this revision path ever touched it. Regenerate the local markdown and
    # recreate a fresh Feishu doc from it (create-only API — no in-place block replacement — so a
    # new doc/URL each time a revision actually changes something; skipped when everything was
    # rejected, since regenerating an identical doc would just be noise).
    if any(bool(outcome.get("accepted")) for outcome in per_section_outcome.values()):
        artifact_paths = _write_research_review_artifacts(job_id=job_id, result=next_result, input_data=input_data)
        if artifact_paths:
            next_result["artifacts"] = list(dict.fromkeys([*(next_result.get("artifacts") or []), *artifact_paths]))
            revision_number = len(next_result.get("revision_history") or [])
            next_result["delivery"] = _delivery_payload(
                {**next_result, "paths": artifact_paths}, input_data, title_suffix=f" · 修改 v{revision_number}"
            )
    job_service.update_result(
        job_id,
        status="waiting_approval",
        result=next_result,
        progress_message="section revision applied",
    )


def find_revisable_job_for_channel(channel_id: str) -> dict[str, Any] | None:
    """Most recent waiting_approval, question_loop research job for this channel, or None.

    Used by the Feishu natural-language revision entry point to figure out which job a chat
    message like "总览那节写得啰嗦，精简一下" refers to, without requiring the user to name a
    job id. A channel can have a legacy-pipeline job also stuck at waiting_approval (e.g. from
    before Feishu was routed to question_loop, see feishu-channel-router-legacy-pipeline-bug
    memory) — skip those rather than returning one revise_research_sections would just reject.
    """
    candidates = job_service.list(job_type="research", channel_id=channel_id, status="waiting_approval", limit=10)
    for job in candidates:
        result = job.get("result") or {}
        if (
            isinstance(result, dict)
            and result.get("pipeline_mode") == "question_loop"
            and bool((result.get("capabilities") or {"section_revision": True}).get("section_revision"))
        ):
            return job
    return None


def extract_section_revision_intent(job: dict[str, Any], message: str) -> dict[str, Any]:
    """Classify a chat message as report-revision or not, and if so, resolve which section(s).

    Returns {"is_revision_request": bool, "section_ids": [...], "instruction": str}.

    Originally this only resolved section_ids and was gated behind a keyword allowlist in
    channel_router_service.py (e.g. "精简"/"改一下"). Real Feishu testing hit two misses in a
    row ("能不能加些表格呢", "让04...更加充实") that named no section-scope keyword and used verbs
    not on the list — no fixed keyword list can cover every natural way to ask for a change.
    `is_revision_request` replaces that keyword gate: the caller now runs this for any message in
    a channel with a resolvable job (see find_revisable_job_for_channel), and trusts the model's
    own judgment of whether the message is about revising the report at all, not just which
    section. `is_revision_request=False` means treat this as ordinary chat, not even a
    clarification reply. `is_revision_request=True` with empty `section_ids` means it IS a
    revision request but couldn't be confidently mapped to a specific section — ask the user to
    be more specific. Any section id the model invents that isn't actually in this job's
    blueprint is dropped rather than trusted, same defensive posture as revise_research_sections'
    own section_ids validation.
    """
    sections = [
        {"section_id": str(section.get("section_id") or ""), "title": str(section.get("title") or "")}
        for section in ((job.get("result") or {}).get("report_blueprint") or {}).get("sections") or []
        if section.get("section_id")
    ]
    valid_ids = {section["section_id"] for section in sections}
    if not sections or not str(message or "").strip():
        return {"is_revision_request": False, "section_ids": [], "instruction": ""}
    report_thesis = str(((job.get("result") or {}).get("report_blueprint") or {}).get("thesis") or "").strip()
    payload = {
        "task": (
            "This chat channel has a research report that was just delivered and is currently "
            "awaiting the user's review — the message below is a new reply in that same channel, "
            "so it should be read as a plausible follow-up about that report first, not judged in "
            "isolation. Decide whether it is asking to revise or improve part of the report (e.g. "
            "make a section more detailed/concise, add a table, fix tone, add missing content), as "
            "opposed to being something clearly unrelated (small talk, a brand-new topic with no "
            "connection to the report). If it is a revision request, also identify which section(s) "
            "of the report it means. Return JSON only."
        ),
        "report_thesis": report_thesis,
        "sections": sections,
        "user_message": message,
        "rules": [
            "Default to is_revision_request=true when the message plausibly refers to changing "
            "something about the report (tone, detail level, structure, missing content, "
            "formatting like tables) and there is no clear signal it is about something else "
            "entirely — do not require the message to explicitly say \"report\" or name a section.",
            "Only set is_revision_request=false when the message is clearly unrelated to the "
            "report, e.g. small talk, a greeting, or a request about a completely different topic.",
            "Only use section_id values that appear in the sections list; never invent one.",
            "If is_revision_request is true but the message does not clearly point to any specific "
            "section, return an empty section_ids list — do not guess a section.",
            "If the message clearly refers to two or more sections (e.g. asking them to be consistent "
            "with each other), include all of them in section_ids.",
            "instruction should restate what change the user wants, in Chinese, concise and actionable.",
        ],
        "schema": {
            "is_revision_request": False,
            "section_ids": ["section_id, ..."],
            "instruction": "restated revision request",
        },
    }
    parsed = _research_json_model(payload)
    is_revision_request = bool(parsed.get("is_revision_request"))
    section_ids = (
        _dedupe_strings([str(item) for item in parsed.get("section_ids") or [] if str(item) in valid_ids])
        if is_revision_request
        else []
    )
    instruction = str(parsed.get("instruction") or "").strip() or str(message or "").strip()
    return {"is_revision_request": is_revision_request, "section_ids": section_ids, "instruction": instruction}


def _run_research_pipeline(job_id: str, input_data: dict[str, Any]) -> dict[str, Any]:
    try:
        job = job_service.get(job_id)
    except Exception:
        job = {}
    with audit_run(
        job_id=job_id,
        parent_job_id=str(input_data.get("parent_job_id") or "") or None,
        run_id=str(input_data.get("run_id") or "") or None,
        attempt=int(job.get("attempt") or 1),
        retry_of_job_id=str(job.get("retry_of_job_id") or "") or None,
    ):
        mode = str(input_data.get("pipeline_mode") or "question_loop").strip().lower()
        write_stage_checkpoint(
            "request_normalization",
            inputs={"raw_request": input_data},
            outputs={
                "normalized_request": {
                    "user_request": str(input_data.get("user_request") or "").strip(),
                    "pipeline_mode": mode,
                    "source_mode": str(input_data.get("source_mode") or "web"),
                    "research_strategy": str(input_data.get("research_strategy") or "standard"),
                    "depth": str(input_data.get("depth") or "normal"),
                }
            },
            transition_reason="research job dispatched",
        )
        try:
            if mode == "single_url":
                # Independent short-circuit path -- see docs/modules/single-url-brief-pipeline-v0.1.md.
                result = _run_single_url_research(job_id, input_data)
            elif mode == "question_loop":
                result = _run_question_loop_research(job_id, input_data)
            elif mode == "agentic_retrieval":
                result = _run_question_loop_research(job_id, input_data, agentic=True)
            else:
                raise ValueError(f"unsupported research pipeline_mode: {mode}")
        except BaseException as exc:
            write_stage_checkpoint(
                "final_status",
                inputs={"request": input_data},
                outputs={},
                transition_reason="research pipeline raised",
                status="failed",
                error=exc,
            )
            raise
        write_stage_checkpoint(
            "final_status",
            inputs={"request": input_data},
            outputs={
                "task_status": {
                    "status": result.get("status"),
                    "report_status": result.get("report_status"),
                    "delivery_status": result.get("delivery_status"),
                    "quality_gate": result.get("quality_gate"),
                }
            },
            transition_reason="research pipeline returned",
            status=str(result.get("status") or "succeeded"),
        )
        return result


def _apply_requested_max_rounds(states: list[Any], input_data: dict[str, Any]) -> None:
    """Let an explicit request-level max_rounds only RAISE the per-question round budget.

    只能上调，不能下调 (2026-08-13 fix): this used to be `min(requested_max_rounds,
    state.budget.max_rounds)`, which silently SHRANK QuestionBudget.for_depth()'s
    depth-appropriate per-question round budget (normal=3, deep=4) whenever the caller's
    payload carried any max_rounds value -- including the API schema's own default of 2
    (backend/schemas/research.py's `max_rounds: int = 2`), which every caller that goes
    through Pydantic's model_dump() sends whether they intended to override anything or
    not. That covers both the Web frontend (ResearchStudioPage.tsx never sets max_rounds,
    so every job silently ran depth-normal questions at 2 rounds instead of 3, deep at 2
    instead of 4) and the eval harness (run_research_multi_topic_probe.py/run_research_e2e_
    probe.py default to `_PROD_MAX_ROUNDS` = this same schema default of 2, believing it
    tracked "the production budget" when it was actually clamping it down) -- see the long
    string of research_probe 0/5-1/5 e2e reports in docs/evaluations/. Only tools/research_
    jobs.py's Feishu/conversational path avoided this (it deliberately omits max_rounds --
    see that file's own comment). Changed to `max()` so an explicit request can only RAISE
    the per-question round budget above the depth default, never silently lower it below
    what QuestionBudget.for_depth() already decided was appropriate for this depth.
    """
    requested_max_rounds = int(input_data.get("max_rounds") or 0)
    if requested_max_rounds > 0:
        for state in states:
            if state.comparison_cells:
                continue
            state.budget.max_rounds = max(requested_max_rounds, state.budget.max_rounds)


def _run_question_loop_research(job_id: str, input_data: dict[str, Any], *, agentic: bool = False) -> dict[str, Any]:
    user_request = str(input_data.get("user_request") or "").strip()
    if not user_request:
        raise ValueError("user_request is required")
    channel_id = str(input_data.get("channel_id") or "").strip()
    strategy = _research_strategy(input_data)
    depth = _research_depth(input_data)
    topic = _suggest_topic(input_data, user_request)
    writing_architecture = _research_writing_architecture(input_data)

    _step(job_id, "plan", "running", "正在生成问题级研究计划")
    has_authored_plan = (
        isinstance(input_data.get("confirmed_plan"), dict)
        or (isinstance(input_data.get("planned_sections"), list) and bool(input_data.get("planned_sections")))
    )
    if has_authored_plan:
        # The user already confirmed a plan through clarify_concept_tool ->
        # draft_research_plan_tool before start_research_job_tool was called -- re-running
        # _research_reconnaissance here would both waste a network+model call the tool already
        # made and risk a different (non-deterministic search) disambiguation verdict than what
        # the user actually confirmed. Reuse the terms carried on the request instead (决策16:
        # 消歧词静默随 plan 进 job). recon_context stays empty: _plan_research's authored_plan
        # branch below doesn't call the planner model, so there's nothing to feed it to.
        recon_context = ""
        disambiguation_terms = [str(item) for item in input_data.get("disambiguation_terms") or [] if str(item).strip()]
    else:
        reconnaissance = _research_reconnaissance(user_request)
        recon_context = str(reconnaissance.get("summary") or "")
        disambiguation_terms = [str(item) for item in reconnaissance.get("disambiguation_terms") or [] if str(item).strip()]
    research_plan = _plan_research(
        user_request=user_request,
        topic=topic,
        input_data=input_data,
        strategy=strategy,
        recon_context=recon_context,
        force_dynamic_questions=True,
    )
    comparison_budget = {}
    if _comparison_plan_enabled(research_plan):
        comparison_budget = (
            dict(input_data.get("comparison_budget") or {})
            or _normalized_comparison_budget(depth)
        )
    if comparison_budget:
        _validate_comparison_plan_budget(research_plan, comparison_budget)
        research_plan["comparison_budget"] = comparison_budget
    query_plan = _build_query_plan(
        user_request=user_request,
        research_plan=research_plan,
        strategy=strategy,
        depth=depth,
        extra_focus_terms=disambiguation_terms,
    )
    states = question_states_from_plan(research_plan, depth=depth)
    if not states:
        raise ValueError("research planner did not produce executable questions")
    _apply_requested_max_rounds(states, input_data)
    query_expansion = _expand_initial_query_candidates(states)
    initial_queries = _question_initial_queries(
        query_plan,
        states,
        expanded_queries=query_expansion["queries_by_id"],
    )
    write_stage_checkpoint(
        "planner",
        inputs={
            "normalized_request": {
                "user_request": user_request,
                "topic": topic,
                "strategy": strategy,
                "depth": depth,
                "comparison_budget": comparison_budget,
                "recon_context": recon_context,
            }
        },
        outputs={"research_plan": research_plan, "question_states_initial": [state.to_dict() for state in states]},
        transition_reason="planner produced executable question states",
    )
    write_stage_checkpoint(
        "query_generation",
        inputs={"research_plan": research_plan},
        outputs={
            "query_plan": query_plan,
            "query_expansion": query_expansion,
            "initial_queries": initial_queries,
        },
        transition_reason="initial queries compiled from the accepted plan",
    )
    _step(
        job_id,
        "plan",
        "succeeded",
        f"生成 {len(states)} 个可执行问题，采用 question-loop",
        items=[{"label": state.section_title, "value": state.question} for state in states[:10]],
    )
    _step(
        job_id,
        "query_plan",
        "succeeded",
        f"为 {len(initial_queries)} 个问题准备初始查询",
        items=[{"label": key, "value": values[0] if values else ""} for key, values in list(initial_queries.items())[:8]],
    )
    _publish_running_research_snapshot(
        job_id,
        {
            "channel_id": channel_id or None,
            "pipeline_mode": "question_loop",
            "writing_architecture": writing_architecture,
            "writing_architecture_version": (
                EVIDENCE_NATIVE_ARCHITECTURE_VERSION
                if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
                else "finding/current"
            ),
            "capabilities": (
                evidence_native_capabilities()
                if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
                else {
                    "section_revision": True,
                    "targeted_followup": True,
                    "approval": True,
                    "archive": True,
                }
            ),
            "research_strategy": strategy,
            "depth": depth,
            "comparison_budget": comparison_budget,
            "research_plan": research_plan,
            "query_plan": query_plan,
            "question_states": [state.to_dict() for state in states],
            "suggested_topic": topic,
        },
    )

    _step(job_id, "search", "running", "正在按问题执行增量检索")
    source_pool = GlobalSourcePool()
    material_store = JobMaterialStore()
    routed_cache: dict[str, list[dict[str, Any]]] = {state.question_id: [] for state in states}
    use_global_pool = bool(comparison_budget) or (
        os.getenv("RESEARCH_GLOBAL_SOURCE_POOL", "1") == "1"
        and getattr(_retrieve_question_round, "__module__", "") == __name__
    )
    question_progress: dict[str, dict[str, Any]] = {}
    question_progress_lock = threading.Lock()

    def report_question_progress(event: dict[str, Any]) -> None:
        question_id = str(event.get("question_id") or "")
        with question_progress_lock:
            question_progress[question_id] = dict(event)
            completed = sum(
                str(item.get("status") or "") in {"sufficient", "exhausted"}
                for item in question_progress.values()
            )
        question = str(event.get("question") or "").strip()
        summary = (
            f"问题进度 {completed}/{len(states)} · 第 {int(event.get('rounds_used') or 0)} 轮"
            f" · 证据 {int(event.get('evidence_count') or 0)} 条"
        )
        if question:
            summary += f" · {question[:52]}"
        _step(job_id, "search", "running", summary)

    orchestrator = QuestionResearchOrchestrator(
        lambda state, queries, round_index: _retrieve_question_round(
            job_id=job_id,
            user_request=user_request,
            strategy=strategy,
            depth=depth,
            state=state,
            queries=queries,
            round_index=round_index,
        ),
        batch_retrieval=(lambda active_states, queries_by_id, round_index: _retrieve_question_wave(
            job_id=job_id,
            user_request=user_request,
            strategy=strategy,
            depth=depth,
            all_states=states,
            active_states=active_states,
            queries_by_id=queries_by_id,
            round_index=round_index,
            source_pool=source_pool,
            material_store=material_store,
            routed_cache=routed_cache,
        )) if use_global_pool else None,
        coverage_judge=_judge_question_coverage,
        progress_callback=report_question_progress,
        parallelism=max(1, int(os.getenv("RESEARCH_QUESTION_LOOP_PARALLELISM", "3"))),
        agentic=agentic,
        source_pool=source_pool,
        material_store=material_store,
        job_id=job_id,
        user_request=user_request,
        research_depth=depth,
    )
    question_result = orchestrator.run(states, initial_queries=initial_queries)
    model_json = _research_finding_json_model
    evidence_landscape = classify_evidence_landscape(question_result["evidence_map"])
    research_mode = str(evidence_landscape.get("mode") or "evidence_informed_reference_design")
    if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE:
        findings: list[dict[str, Any]] = []
        finding_admission: dict[str, Any] = {
            "status": "not_applicable",
            "admitted_count": 0,
            "rejected_count": 0,
            "review_unavailable_count": 0,
        }
        report_blueprint = build_evidence_native_blueprint(
            normalized_request={
                "user_request": user_request,
                "topic": topic,
                "strategy": strategy,
                "depth": depth,
                "research_mode": research_mode,
            },
            research_plan=research_plan,
            question_states=question_result["question_states"],
            evidence_map=question_result["evidence_map"],
            selected_chunks=question_result["selected_chunks"],
            research_mode=research_mode,
            evidence_landscape=evidence_landscape,
        )
        question_result.setdefault("pipeline_diagnostics", {})["finding_admission"] = {
            "status": "not_applicable_evidence_native"
        }
    else:
        try:
            findings = synthesize_findings(
                question_result["question_states"],
                model=model_json,
                research_mode=research_mode,
            )
            findings, finding_admission = _admit_research_findings(
                question_result["question_states"], findings, model=model_json
            )
        except FindingSynthesisError as exc:
            write_stage_checkpoint(
                "finding_synthesis",
                inputs={
                    "question_states": question_result["question_states"],
                    "evidence_map": question_result["evidence_map"],
                },
                outputs={"findings": [], "failures": exc.failures},
                transition_reason="finding synthesis or independent admission failed closed; no deterministic fallback was produced",
            )
            raise
        question_result.setdefault("pipeline_diagnostics", {})["finding_admission"] = {
            key: finding_admission.get(key)
            for key in ("status", "admitted_count", "rejected_count", "review_unavailable_count")
        }
        report_blueprint = build_question_report_blueprint(
            user_request=user_request,
            research_plan=research_plan,
            findings=findings,
            evidence_map=question_result["evidence_map"],
            question_states=question_result["question_states"],
            research_mode=research_mode,
            evidence_landscape=evidence_landscape,
        )
        report_blueprint = attach_section_dossiers(
            report_blueprint,
            findings=findings,
            evidence_map=question_result["evidence_map"],
            selected_chunks=question_result["selected_chunks"],
            materials=question_result["materials"],
        )
    rescue_section_ids = {
        state.section_id
        for state in states
        if any(
            bool(gap.get("blocking"))
            and str(gap.get("gap_type") or "") in {"semantic_coverage", "evidence_floor", "source_floor"}
            for gap in state.coverage.get("gaps") or []
        )
    }
    rescue_states = [state for state in states if state.section_id in rescue_section_ids]
    rescue_enabled = (
        not comparison_budget
        and os.getenv("RESEARCH_REPORTABILITY_RESCUE", "1").strip().lower() not in {"0", "false", "no"}
    )
    rescue_diagnostics = {
        "enabled": rescue_enabled,
        "disabled_reason": "comparison_budget_hard_cap" if comparison_budget else "",
        "section_ids": sorted(rescue_section_ids),
        "question_count": len(rescue_states),
        "executed": False,
        "added_rounds": 0,
        "generated_gap_action_count": 0,
    }
    if rescue_enabled and rescue_states:
        rescue_queries: dict[str, list[str]] = {}
        for state in rescue_states:
            state.budget.max_rounds = max(state.budget.max_rounds, state.rounds_used + 1)
            state.budget.max_materials += max(1, state.budget.followup_fetch_limit)
            state.budget.max_fetch_urls += max(1, state.budget.followup_fetch_limit)
            state.budget.max_model_calls += 2
            state.budget.timeout_seconds += 60.0
            state.status = "pending"
            state.stop_reason = ""
            state.stagnant_rounds = 0
            uncovered_criteria = sorted(
                (
                    criterion for criterion in state.criteria
                    if criterion.status != "covered"
                ),
                key=lambda criterion: (
                    bool(criterion.attempts),
                    {"hard": 0, "soft": 1, "optional": 2}.get(criterion.level, 3),
                    len(criterion.attempts),
                    criterion.criterion_id,
                ),
            )
            rescue_queries[state.question_id] = [
                compile_research_query(
                    topic_anchor=state.topic_anchor or state.question,
                    content=criterion.query_hint or criterion.description,
                    source_type=(criterion.preferred_source_types or state.preferred_source_types or [""])[0],
                )
                for criterion in uncovered_criteria[:2]
            ]
            if not rescue_queries[state.question_id]:
                rescue_queries[state.question_id] = [compile_research_query(
                    topic_anchor=state.topic_anchor or state.question,
                    content=state.question,
                    source_type=(state.preferred_source_types or [""])[0],
                )]
            generated_gap_actions = _register_scheduled_gap_actions(
                state,
                rescue_queries[state.question_id],
            )
            rescue_diagnostics["generated_gap_action_count"] += generated_gap_actions
        before_rounds = sum(state.rounds_used for state in states)
        previous_elapsed = int((question_result.get("pipeline_diagnostics") or {}).get("elapsed_ms") or 0)
        rescue_pass = orchestrator.run(rescue_states, initial_queries=rescue_queries)
        rescue_elapsed = int((rescue_pass.get("pipeline_diagnostics") or {}).get("elapsed_ms") or 0)
        question_result = orchestrator.snapshot(states, elapsed_ms=previous_elapsed + rescue_elapsed)
        rescue_diagnostics["executed"] = True
        rescue_diagnostics["added_rounds"] = sum(state.rounds_used for state in states) - before_rounds
        evidence_landscape = classify_evidence_landscape(question_result["evidence_map"])
        research_mode = str(evidence_landscape.get("mode") or research_mode)
        if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE:
            findings = []
            report_blueprint = build_evidence_native_blueprint(
                normalized_request={
                    "user_request": user_request,
                    "topic": topic,
                    "strategy": strategy,
                    "depth": depth,
                    "research_mode": research_mode,
                },
                research_plan=research_plan,
                question_states=question_result["question_states"],
                evidence_map=question_result["evidence_map"],
                selected_chunks=question_result["selected_chunks"],
                research_mode=research_mode,
                evidence_landscape=evidence_landscape,
            )
        else:
            try:
                findings = synthesize_findings(
                    question_result["question_states"],
                    model=model_json,
                    research_mode=research_mode,
                )
                findings, finding_admission = _admit_research_findings(
                    question_result["question_states"], findings, model=model_json
                )
            except FindingSynthesisError as exc:
                write_stage_checkpoint(
                    "finding_synthesis",
                    inputs={
                        "question_states": question_result["question_states"],
                        "evidence_map": question_result["evidence_map"],
                        "reportability_rescue": rescue_diagnostics,
                    },
                    outputs={"findings": [], "failures": exc.failures},
                    transition_reason="post-rescue finding synthesis or independent admission failed closed; no deterministic fallback was produced",
                )
                raise
            question_result.setdefault("pipeline_diagnostics", {})["finding_admission"] = {
                key: finding_admission.get(key)
                for key in ("status", "admitted_count", "rejected_count", "review_unavailable_count")
            }
            report_blueprint = build_question_report_blueprint(
                user_request=user_request,
                research_plan=research_plan,
                findings=findings,
                evidence_map=question_result["evidence_map"],
                question_states=question_result["question_states"],
                research_mode=research_mode,
                evidence_landscape=evidence_landscape,
            )
            report_blueprint = attach_section_dossiers(
                report_blueprint,
                findings=findings,
                evidence_map=question_result["evidence_map"],
                selected_chunks=question_result["selected_chunks"],
                materials=question_result["materials"],
            )
    question_result["pipeline_diagnostics"]["reportability_rescue"] = rescue_diagnostics
    candidate_sources = question_result["sources"]
    materials = question_result["materials"]
    selected_chunks = question_result["selected_chunks"]
    evidence_map = question_result["evidence_map"]
    sources = _used_sources_from_evidence(
        candidate_sources=candidate_sources,
        materials=materials,
        evidence_map=evidence_map,
    )
    rejected_evidence = question_result["rejected_evidence"]
    question_states = question_result["question_states"]
    write_stage_checkpoint(
        "search",
        inputs={"query_plan": query_plan},
        outputs={"sources": candidate_sources, "question_states": question_states},
        transition_reason="question-loop retrieval completed",
    )
    write_stage_checkpoint(
        "fetch",
        inputs={"sources": candidate_sources},
        outputs={"materials": materials},
        transition_reason="source bodies fetched into the job material store",
    )
    write_stage_checkpoint(
        "chunk_generation",
        inputs={"materials": materials},
        outputs={"selected_chunks": selected_chunks},
        transition_reason="chunks generated, filtered, deduplicated, and scored",
    )
    write_stage_checkpoint(
        "evidence_admission",
        inputs={"selected_chunks": selected_chunks},
        outputs={"evidence_map": evidence_map, "rejected_evidence": rejected_evidence},
        transition_reason="atomic evidence admission completed",
    )
    write_stage_checkpoint(
        "evidence_native_blueprint" if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE else "finding_synthesis",
        inputs={"question_states": question_states, "evidence_map": evidence_map},
        outputs={"findings": findings, "report_blueprint": report_blueprint},
        transition_reason=(
            "Evidence-native report blueprint built without Findings"
            if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
            else "findings and report blueprint synthesized"
        ),
    )
    _step(job_id, "search", "succeeded", f"问题级检索累计发现 {len(sources)} 个来源", items=_source_items(sources[:8]))
    _step(job_id, "fetch", "succeeded", f"抓取并保留 {len(materials)} 份唯一材料", items=_material_items(materials[:8]))
    _step(job_id, "material_pool", "succeeded", f"建立 {len(materials)} 份清洗后材料", items=_material_items(materials[:8]))
    _step(job_id, "chunk_select", "succeeded", f"混合检索选中 {len(selected_chunks)} 个问题相关片段", items=_chunk_items(selected_chunks[:8]))
    _step(
        job_id,
        "evidence",
        "succeeded",
        f"准入 {len(evidence_map)} 条 atomic evidence，隔离 {len(rejected_evidence)} 条",
        items=_evidence_items(evidence_map[:8]),
    )

    coverage = _question_loop_coverage(question_states)
    matrix_coverage = _comparison_matrix_coverage(question_states, research_plan)
    if matrix_coverage.get("enabled"):
        coverage["matrix_coverage"] = matrix_coverage
        coverage["depth_ready"] = bool(matrix_coverage.get("complete")) and not coverage.get("safety_blocked_count")
        coverage["matrix_gap_count"] = len(matrix_coverage.get("gaps") or [])
        coverage["fatal_gaps"] = [
            *(coverage.get("fatal_gaps") or []),
            *[
                f"{cell.get('subject')} × {cell.get('axis')}: {cell.get('status')}"
                for cell in matrix_coverage.get("gaps") or []
            ],
        ]
        coverage["reason"] = (
            f"比较矩阵 {matrix_coverage.get('complete_cell_count')}/{matrix_coverage.get('required_cell_count')} "
            f"个 Required Cell 完整；整体为 {matrix_coverage.get('status')}。"
        )
    write_stage_checkpoint(
        "coverage",
        inputs={"question_states": question_states, "evidence_map": evidence_map},
        outputs={"coverage": coverage},
        transition_reason=str(coverage.get("reason") or "coverage evaluated"),
    )
    _step(job_id, "coverage", "succeeded", coverage["reason"], items=_coverage_items(coverage))
    checkpoint_paths = [path for path in [_save_question_loop_checkpoint(job_id, "coverage", {
        "research_plan": research_plan,
        "query_plan": query_plan,
        "question_states": question_states,
        "coverage": coverage,
        "matrix_coverage": matrix_coverage,
        "sources": sources,
        "materials": materials,
        "selected_chunks": selected_chunks,
        "evidence_map": evidence_map,
        "rejected_evidence": rejected_evidence,
    })] if path]
    running_pipeline_diagnostics = {
        **question_result["pipeline_diagnostics"],
        "candidate_source_count": len(candidate_sources),
        "used_source_count": len(sources),
        "writing_architecture": writing_architecture,
        "checkpoint_paths": checkpoint_paths,
    }
    _publish_running_research_snapshot(
        job_id,
        {
            "question_states": question_states,
            "coverage": coverage,
            "matrix_coverage": matrix_coverage,
            "findings": findings,
            "report_blueprint": report_blueprint,
            "evidence_native_blueprint": (
                report_blueprint if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE else None
            ),
            "sources": sources,
            "materials": materials,
            "selected_chunks": selected_chunks,
            "evidence_map": evidence_map,
            "evidence_pack": evidence_map,
            "rejected_evidence": rejected_evidence,
            "pipeline_diagnostics": running_pipeline_diagnostics,
            "retrieval": {
                "evidence_diagnostics": {
                    "mode": "question_loop",
                    "evidence_count": len(evidence_map),
                    "rejected_count": len(rejected_evidence),
                },
                "diagnostics": question_result["pipeline_diagnostics"],
                "gaps": coverage.get("fatal_gaps") or [],
                "rounds": [
                    {
                        "question_id": state.get("question_id"),
                        "rounds_used": state.get("rounds_used"),
                        "stop_reason": state.get("stop_reason"),
                    }
                    for state in question_states
                ],
            },
        },
    )

    _step(
        job_id,
        "draft",
        "running",
        "正在按 Evidence Packet 并行撰写章节"
        if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
        else "正在综合 findings 并并行撰写章节",
    )

    def report_writer_progress(event: dict[str, Any]) -> None:
        _step(
            job_id,
            "draft",
            "running",
            f"章节撰写 {int(event.get('completed') or 0)}/{int(event.get('total') or 0)} · {str(event.get('title') or '')[:64]}",
        )

    eplus_prevalidation_diagnostics: dict[str, Any] = {}
    if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE:
        prevalidation_budget_seconds = max(
            10.0, float(os.getenv("RESEARCH_EPLUS_PREVALIDATION_BUDGET_SECONDS", "600"))
        )
        eplus_prevalidation_diagnostics = {
            "budget_seconds": prevalidation_budget_seconds,
            "budget_exhausted": False,
            "model_calls": 0,
            "model_failures": 0,
            "skipped_model_calls": 0,
            "timeout_retries": 0,
        }
        prevalidation_model = _validation_model(
            time.monotonic() + prevalidation_budget_seconds,
            eplus_prevalidation_diagnostics,
            budget_exhausted_key="prevalidation_budget_exhausted",
        )
        eplus_checkpoint_store = EvidenceNativeCheckpointStore(
            RESEARCH_ARTIFACT_DIR
            / _safe_artifact_name(job_id)
            / "checkpoints"
            / "evidence_native",
            contract={
                "writing_architecture": EVIDENCE_NATIVE_ARCHITECTURE,
                "writing_architecture_version": EVIDENCE_NATIVE_ARCHITECTURE_VERSION,
                "context_manager_version": EVIDENCE_NATIVE_CONTEXT_VERSION,
                "prompt_version": EVIDENCE_NATIVE_PROMPT_VERSION,
                "blueprint_input_hash": report_blueprint.get("input_hash"),
                "question_states_sha256": evidence_native_hash(question_states),
                "accepted_evidence_sha256": evidence_native_hash(evidence_map),
                "selected_chunks_sha256": evidence_native_hash(selected_chunks),
                "configuration": configuration_summary(),
                "code_version": code_version(),
                "writer": {
                    "model_name": os.getenv("RESEARCH_WRITER_MODEL_NAME") or None,
                    "temperature": float(os.getenv("RESEARCH_WRITER_TEMPERATURE", "0.3")),
                    "timeout_seconds": float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
                    "max_retries": int(os.getenv("RESEARCH_WRITER_MAX_RETRIES", "3")),
                },
                "prevalidation": {
                    "budget_seconds": prevalidation_budget_seconds,
                    "parallelism": max(1, int(os.getenv("RESEARCH_ENTAILMENT_PARALLELISM", "6"))),
                },
            },
        )
        written = write_evidence_native_report(
            user_request=user_request,
            blueprint=report_blueprint,
            question_states=question_states,
            evidence_map=evidence_map,
            selected_chunks=selected_chunks,
            writer_model=_research_writer_json_model,
            prevalidation_model=prevalidation_model,
            parallelism=max(1, int(os.getenv("RESEARCH_SECTION_WRITER_PARALLELISM", "3"))),
            progress_callback=report_writer_progress,
            checkpoint_store=eplus_checkpoint_store,
            entailment_payload_profile=ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
        )
        report_blueprint = written.get("report_blueprint") or report_blueprint
    else:
        written = write_question_report(
            user_request=user_request,
            blueprint=report_blueprint,
            findings=findings,
            evidence_map=evidence_map,
            model=_research_writer_json_model,
            parallelism=max(1, int(os.getenv("RESEARCH_SECTION_WRITER_PARALLELISM", "3"))),
            progress_callback=report_writer_progress,
        )
    documents = written["documents"]
    report_document = written["report_document"]
    report_status = str(written.get("report_status") or "partial_report")
    report_completeness = report_blueprint.get("completeness") or {"status": report_status}
    if matrix_coverage.get("enabled") and not matrix_coverage.get("complete"):
        report_status = "partial_report"
        report_completeness = {
            **report_completeness,
            "status": "partial_report",
            "matrix_status": "partial",
            "matrix_gap_count": len(matrix_coverage.get("gaps") or []),
        }
        report_document["delivery_status"] = "partial_report"
        report_document["matrix_coverage"] = matrix_coverage
        documents, report_document = _attach_matrix_gap_disclosure(
            documents,
            report_document,
            matrix_coverage,
        )
    evidence_map = _mark_evidence_usage(evidence_map, documents)
    if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE:
        for stage, payload in (
            (
                "evidence_context",
                {
                    "report_blueprint_input_hash": report_blueprint.get("input_hash"),
                    "context_packets": written.get("evidence_context_packets") or {},
                    "context_diagnostics": (written.get("diagnostics") or {}).get("context_sections") or [],
                },
            ),
            (
                "section_prevalidation",
                {
                    "section_prevalidation": written.get("section_prevalidation") or {},
                    "prevalidation_diagnostics": eplus_prevalidation_diagnostics,
                },
            ),
            (
                "editorial_claim_bank",
                {
                    "editorial_claim_bank": written.get("editorial_claim_bank") or [],
                    "editorial": written.get("editorial") or {},
                },
            ),
        ):
            checkpoint_path = _save_question_loop_checkpoint(job_id, stage, payload)
            if checkpoint_path:
                checkpoint_paths.append(checkpoint_path)
    draft_checkpoint = _save_question_loop_checkpoint(job_id, "draft", {
        "writing_architecture": writing_architecture,
        "findings": findings,
        "report_blueprint": report_blueprint,
        "documents": documents,
        "report_document": report_document,
        "editorial": written.get("editorial") or {},
        "section_prevalidation": written.get("section_prevalidation") or {},
        "editorial_claim_bank": written.get("editorial_claim_bank") or [],
        "evidence_context_packets": written.get("evidence_context_packets") or {},
        "evidence_map": evidence_map,
    })
    if draft_checkpoint:
        checkpoint_paths.append(draft_checkpoint)
    writer_failures = _writer_generation_failures(documents, written.get("editorial"))
    if writer_failures:
        write_stage_checkpoint(
            "writer_generation",
            inputs={
                "report_blueprint": report_blueprint,
                "finding_count": len(findings),
                "evidence_count": len(evidence_map),
            },
            outputs={"failures": writer_failures, "draft_checkpoint": draft_checkpoint},
            transition_reason="writer retries exhausted; no deterministic report content was produced",
            status="failed",
        )
        _step(job_id, "draft", "failed", "写作模型重试耗尽，未生成兜底正文")
        failure_summary = ", ".join(
            f"{item.get('section_id') or 'unknown'}:{item.get('failure_reason') or 'unknown'}"
            for item in writer_failures
        )
        raise RuntimeError(f"writer generation failed: {failure_summary}")

    _step(job_id, "validation", "running", "正在校验 claim ledger、引用和报告质量")
    validation_started = time.monotonic()
    # Default raised 180 -> 360 -> 720 -> 1080 -> 2400 (2026-08-12: the critic-evidence-grounding
    # fix adds real evidence payloads to every critic call plus two new model-call stages --
    # evidence-backed issue self-check (quality.py _recheck_evidence_backed_issues) and writer
    # rejection arbitration, including a forced follow-up write when arbitration overrules the
    # writer (writer.py _settle_issue_rejections) -- on top of the existing critic scoring +
    # entailment + revision loop that alone measured ~710s against the prior 720s budget
    # (e2e-eval-sweep-20260809 §3.4). Per user decision: raise generously and let quality-first
    # validation run to completion rather than tune this from a guess -- see
    # docs/evaluations/deep-research-critic-evidence-grounding-20260812.md.
    validation_budget_seconds = max(10.0, float(os.getenv("RESEARCH_VALIDATION_BUDGET_SECONDS", "2400")))
    validation_deadline = validation_started + validation_budget_seconds
    validation_diagnostics: dict[str, Any] = {
        "budget_seconds": validation_budget_seconds,
        "budget_exhausted": False,
        "model_calls": 0,
        "model_failures": 0,
        "skipped_model_calls": 0,
        "timeout_retries": 0,
        "revision_skipped": False,
        # Root-cause investigation instrumentation (docs/evaluations/
        # deep-research-multi-topic-question-loop-first-run-e2e-20260725.md section 5 item 1):
        # per-phase wall time so a revision-eats-the-budget hypothesis can be confirmed or refuted
        # from real timings instead of inferred from revision_passes + total elapsed alone.
        "phase_timings_ms": {},
    }
    validation_model = _validation_model(validation_deadline, validation_diagnostics)
    entailment_evidence_map = (
        written.get("validation_evidence_map")
        if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
        else None
    ) or _entailment_evidence_with_writer_context(evidence_map, report_blueprint)
    grounding = {
        "status": "passed" if all(bool(item.get("grounded_exactly")) for item in evidence_map) else "failed",
        "evidence_url_count": len({str(item.get("source_url") or "") for item in evidence_map}),
        "grounded_evidence_count": sum(bool(item.get("grounded_exactly")) for item in evidence_map),
        "evidence_count": len(evidence_map),
        "per_document": [],
    }
    _first_validate_started = time.monotonic()
    validation = validate_report_citations(documents=documents, report_document=report_document, evidence_map=evidence_map)
    critic_reserve_info = _entailment_deadline_with_critic_reserve(
        documents, validation_deadline, now=time.monotonic()
    )
    validation_diagnostics["critic_reserve_first_validate"] = critic_reserve_info
    entailment_model = _validation_model(
        critic_reserve_info["entailment_deadline"],
        validation_diagnostics,
        budget_exhausted_key="entailment_budget_exhausted",
    )
    entailment = validate_claim_entailment(
        report_document=report_document,
        evidence_map=entailment_evidence_map,
        model=entailment_model,
        parallelism=max(1, int(os.getenv("RESEARCH_ENTAILMENT_PARALLELISM", "6"))),
        payload_profile=(
            ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2
            if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
            else ENTAILMENT_PAYLOAD_PROFILE_DEFAULT
        ),
    )
    entailment = _annotate_entailment_core_scope(
        entailment,
        report_blueprint,
        unsupported_blocks_quality_gate=writing_architecture != EVIDENCE_NATIVE_ARCHITECTURE,
    )
    validation["entailment"] = entailment
    # editorial (executive summary + conclusion) is not a `documents` entry, so without this it
    # is invisible to the critic -- entailment already covers it (_entailment_revision_issues
    # tags section_id="editorial"), but critic ("is this well-argued", not "is this supported")
    # never has. Build the pseudo-document ONLY for this call; `documents` itself (used by report
    # assembly, knowledge-base save, citation validation elsewhere in this function) stays
    # untouched. See docs/modules/deep-research-step4-approval-and-delivery-discussion-v0.1.md 商榷点四.
    documents_for_critic = _documents_with_editorial_pseudo_document(documents, written.get("editorial"))
    report_quality = evaluate_report_quality(
        report_document=report_document,
        question_states=question_states,
        findings=findings,
        rejected_evidence=rejected_evidence,
        documents=documents_for_critic,
        model=validation_model,
        section_critic_budgets=_section_critic_budgets(documents_for_critic, validation_deadline),
        section_critic_parallelism=max(1, int(os.getenv("RESEARCH_SECTION_CRITIC_PARALLELISM", "6"))),
        # 2026-08-15: the critic is a reviewer of the Writer, so it must see at least the
        # evidence context the Writer saw -- the section-scoped dossier excerpts the Writer
        # was allowed to use, which only exist on the enriched entailment_evidence_map (each
        # row carries `validation_context_by_section`). Passing the plain evidence_map here
        # left the critic reviewing with strictly LESS context than the author it was reviewing
        # (see quality.py::_section_critic_evidence). In `case-validator-context-parity`, a short
        # quote omitted a deployment recommendation that appeared in the same-source dossier
        # excerpt, so the critic flagged the Writer's
        # correctly-grounded claim as 无凭据 and the Writer dutifully deleted a true assertion.
        evidence_map=entailment_evidence_map,
        support_units=(
            written.get("editorial_claim_bank") or []
            if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
            else None
        ),
        support_unit_kind=(
            "validated_section_claim"
            if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
            else "finding"
        ),
    )
    # 2026-08-17: first_pass_entailment/latest_entailment/latest_report_quality used to track
    # cross-pass state for the revision loop's claim-reuse cache and stale-critic-issue
    # carry-forward. There is only one pass now (see the one-shot note below), so those
    # variables no longer have a second pass to compare against -- removed rather than left as
    # unread dead assignments.
    validation_diagnostics["phase_timings_ms"]["first_validate_ms"] = int(
        (time.monotonic() - _first_validate_started) * 1000
    )
    # `insufficient_question_coverage` is a *retrieval-time* gap (question-level evidence is
    # thin) and is still surfaced as a report_quality blocking issue / limitations-disclosure
    # input below -- it just no longer triggers a writer revision pass (see the 2026-08-17 note
    # below: there is no revision pass anymore).
    # 2026-08-17 (architecture decision, see docs/plans/writer-critic-loop-redesign): the
    # multi-pass writer<->critic<->entailment revision loop that used to run here (recompute
    # revision_issues, call revise_report_sections, re-validate, repeat up to
    # RESEARCH_REVISION_MAX_PASSES times) proved to oscillate rather than converge -- a claim
    # rejected as unsupported would get reworded and re-rejected pass after pass (the a98ba6 dead
    # loop; a 10-job audit found 54-73% cross-pass recurrence of the same issue type). The
    # underlying problem was structural, not a tuning gap: using one model to adjudicate another
    # model's subjective writing/grounding output does not converge just because you give it more
    # passes. This is now a single one-shot post-processing step instead of a loop: claims
    # entailment flagged as unsupported are marked in place with an inline [待核实] marker (never
    # rewritten, never deleted -- see writer._splice_unsupported_markers), and the critic's issues
    # become non-blocking material for the report's appended 局限性声明 (see quality.py's
    # evaluate_report_quality, which no longer gates on report_quality_dimension_below_3. E+
    # treats final entailment markers as advisory; the explicit Finding compatibility path keeps
    # its historical core-unsupported gate.
    documents, editorial, spliced_content, unsupported_marker_count = _splice_unsupported_markers(
        documents=documents,
        editorial=written.get("editorial") or {},
        entailment=entailment,
        report_quality=report_quality,
        user_request=user_request,
        evidence_map=evidence_map,
        completeness=report_completeness,
        research_mode=report_document.get("research_mode"),
    )
    report_document["content"] = spliced_content
    written["documents"] = documents
    written["editorial"] = editorial
    # No per-pass revision snapshots in the one-shot flow -- nothing to append to
    # checkpoint_paths below, but the name stays so that append site doesn't need its own
    # conditional.
    revision_pass_checkpoint_paths: list[str] = []
    validation_diagnostics["revision_skipped"] = True
    validation_diagnostics["revision_skipped_reason"] = "one_shot_flow_no_revision_loop"
    validation_diagnostics["revision_converged"] = True
    validation_diagnostics["revision_passes"] = 0
    validation_diagnostics["unsupported_marker_count"] = unsupported_marker_count
    # Diagnostic-only Dossier coverage check (docs/evaluations/deep-research-section-dossier-e2e-20260715.md
    # section 6 item 2): classifies evidence the Writer had access to but never cited as
    # irrelevant vs missed_relevant. Never feeds quality_gate or revision_issues -- surfaced for a
    # human/editor to act on, not auto-applied (see classify_unused_dossier_evidence docstring for
    # why). Uses the final (possibly revised) `documents`, so it runs once, after the revision
    # block above. Skipped outright once the validation budget is nearly spent so it can never be
    # the reason validation overruns its budget.
    coverage_min_remaining = max(5.0, float(os.getenv("RESEARCH_VALIDATION_COVERAGE_MIN_REMAINING_SECONDS", "20")))
    if validation_deadline - time.monotonic() >= coverage_min_remaining:
        validation["dossier_coverage"] = classify_unused_dossier_evidence(
            documents=documents,
            model=validation_model,
            parallelism=max(1, int(os.getenv("RESEARCH_ENTAILMENT_PARALLELISM", "6"))),
        )
    else:
        validation["dossier_coverage"] = {"skipped": True, "skipped_reason": "insufficient_validation_budget_remaining"}
    if not validation_diagnostics["budget_exhausted"] and time.monotonic() >= validation_deadline:
        # Defensive backstop: catches budget exhaustion that happened between model calls
        # (no single invoke() saw remaining<=1.0) rather than during one.
        validation_diagnostics["budget_exhausted"] = True
    validation_diagnostics["elapsed_ms"] = int((time.monotonic() - validation_started) * 1000)
    validation_diagnostics["remaining_seconds"] = round(max(0.0, validation_deadline - time.monotonic()), 3)
    validation["stage_budget"] = validation_diagnostics
    # A writer revision_issue rejection that an independent arbitration overruled, and a forced
    # follow-up still didn't resolve (writer.py's _settle_issue_rejections/_disputed_issue_entry)
    # -- two rounds of real model judgment failing to reconcile writer and critic/entailment. Per
    # the 2026-08-12 grilling decision this is surfaced as a blocking quality-gate entry for human
    # review rather than silently dropped or auto-resolved a third time.
    writer_disputed_issues = (written.get("diagnostics") or {}).get("writer_disputed_issues") or []
    if writer_disputed_issues:
        report_quality.setdefault("blocking_issues", [])
        if "writer_critic_dispute_unresolved" not in report_quality["blocking_issues"]:
            report_quality["blocking_issues"].append("writer_critic_dispute_unresolved")
        report_quality["status"] = "failed"
        report_quality["writer_disputed_issues"] = writer_disputed_issues
    quality_gate = combined_quality_gate(grounding=grounding, validation=validation, report_quality=report_quality, entailment=entailment)
    core_empty_section_ids = [str(item) for item in report_completeness.get("core_empty_section_ids") or [] if str(item)]
    core_section_count = int(report_completeness.get("required_section_count") or 0)
    core_structure_ready = core_section_count > 0 and not core_empty_section_ids
    quality_gate["checks"]["report_completeness"] = core_structure_ready
    quality_gate["checks"]["core_empty_section_ids"] = core_empty_section_ids
    if matrix_coverage.get("enabled"):
        quality_gate["checks"]["matrix_complete"] = bool(matrix_coverage.get("complete"))
        quality_gate["matrix_gap_count"] = len(matrix_coverage.get("gaps") or [])
        quality_gate["archive_requires_gap_acknowledgement"] = not bool(matrix_coverage.get("complete"))
        if not matrix_coverage.get("complete"):
            quality_gate.setdefault("warnings", []).append(
                f"比较矩阵仍有 {len(matrix_coverage.get('gaps') or [])} 个 Required Cell 未完整覆盖；"
                "报告已标记为 partial，归档前需人工确认这些缺口。"
            )
    if not core_structure_ready:
        quality_gate["status"] = "needs_more_evidence"
        quality_gate["approval_ready"] = False
        quality_gate["reason"] = (
            "未通过：核心章节为空（" + "、".join(core_empty_section_ids) + "）"
            if core_empty_section_ids
            else "未通过：报告没有可交付的核心章节"
        )
    limited_section_ids = [str(item) for item in report_completeness.get("limited_section_ids") or [] if str(item)]
    if limited_section_ids:
        quality_gate.setdefault("warnings", []).append(
            "材料广度提示：以下章节内容可交付但材料较少：" + "、".join(limited_section_ids)
        )
    # Keep operational timeout diagnostics visible. Semantic entailment validation itself is
    # fail-closed: if its model review remains unavailable after retry, combined_quality_gate()
    # blocks approval instead of accepting a deterministic semantic guess.
    _annotate_validation_budget(quality_gate, validation_diagnostics)
    # evaluate_report_quality() is the canonical classifier for blocking writer fallbacks. It
    # deliberately excludes a structurally empty reference section whose body is discarded in
    # favor of the assembled source index. Re-scanning raw document diagnostics here used to
    # undo that classification and falsely fail otherwise valid reports.
    has_fallback_sections = bool(report_quality.get("fallback_sections"))
    quality_gate["checks"]["fallback_sections_absent"] = not has_fallback_sections
    if has_fallback_sections:
        quality_gate["status"] = "needs_more_evidence"
        quality_gate["approval_ready"] = False
        quality_gate["reason"] = "未通过：报告包含 fallback 模板章节"
    # `report_status` is purely a section-coverage signal (see _report_completeness) and
    # `quality_gate.approval_ready` is purely a validation-gate signal; they are computed
    # independently and can legitimately disagree. Neither should
    # be renamed/repurposed since other code and tests key off their existing meaning. This new
    # `delivery_status` is the single field the user-facing message and frontend should trust for
    # "is this actually done" instead of reading `report_status` alone.
    if report_status == "complete_report" and quality_gate.get("approval_ready") and not has_fallback_sections:
        delivery_status = "complete_report"
    elif report_status == "complete_report":
        delivery_status = "draft_pending_review"
    else:
        delivery_status = "partial_report"
    _step(job_id, "draft", "succeeded", f"生成 {len(documents)} 篇章节稿和 1 篇整合报告", items=[{"label": doc.get("title"), "value": f"{len(str(doc.get('content') or ''))} chars"} for doc in documents])
    validation_step_passed = validation["status"] == "passed"
    validation_summary = (
        "引用校验已完成；Validation 模型增强预算耗尽，已采用确定性校验结果"
        if validation_diagnostics["budget_exhausted"]
        else validation["reason"]
    )
    _step(job_id, "validation", "succeeded" if validation_step_passed else "failed", validation_summary, items=_validation_items(validation))

    result: dict[str, Any] = {
        "status": "waiting_approval",
        "channel_id": channel_id or None,
        "message": {
            "complete_report": "完整研究报告已生成，等待审批落库。",
            "draft_pending_review": "研究报告章节已完整生成，但仍有质量校验问题待处理（如模板化正文或引用校验未过），暂不建议直接批准，请查看 quality_gate 详情。",
        }.get(delivery_status, "阶段性研究报告已生成，部分章节仍需补充研究。"),
        "report_status": report_status,
        "delivery_status": delivery_status,
        "report_completeness": report_completeness,
        "research_mode": research_mode,
        "evidence_landscape": evidence_landscape,
        "pipeline_mode": "question_loop",
        "writing_architecture": writing_architecture,
        "writing_architecture_version": (
            EVIDENCE_NATIVE_ARCHITECTURE_VERSION
            if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
            else "finding/current"
        ),
        "capabilities": (
            evidence_native_capabilities()
            if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE
            else {
                "section_revision": True,
                "targeted_followup": True,
                "approval": True,
                "archive": True,
            }
        ),
        "research_strategy": strategy,
        "depth": depth,
        "comparison_budget": comparison_budget,
        "matrix_coverage": matrix_coverage,
        "research_plan": research_plan,
        "query_plan": query_plan,
        "question_states": question_states,
        "coverage": coverage,
        "findings": findings,
        "report_blueprint": report_blueprint,
        "evidence_native_blueprint": (
            report_blueprint if writing_architecture == EVIDENCE_NATIVE_ARCHITECTURE else None
        ),
        "evidence_context_diagnostics": (
            (written.get("diagnostics") or {}).get("context_sections") or []
        ),
        "section_prevalidation": written.get("section_prevalidation") or {},
        "editorial_claim_bank": written.get("editorial_claim_bank") or [],
        "report_document": report_document,
        "editorial": written.get("editorial") or {},
        "report_quality": report_quality,
        "quality_gate": quality_gate,
        "grounding": grounding,
        "validation": validation,
        "initial_validation": validation,
        "documents": documents,
        "sources": sources,
        "materials": materials,
        "selected_chunks": selected_chunks,
        "evidence_map": evidence_map,
        "evidence_pack": evidence_map,
        "rejected_evidence": rejected_evidence,
        "suggested_topic": topic,
        "pipeline_diagnostics": {
            **question_result["pipeline_diagnostics"],
            "candidate_source_count": len(candidate_sources),
            "used_source_count": len(sources),
            "writer": written["diagnostics"],
            "writing_architecture": writing_architecture,
            "eplus_prevalidation": eplus_prevalidation_diagnostics,
            "report_status": report_status,
            "report_completeness": report_completeness,
            "research_mode": research_mode,
            "evidence_landscape": evidence_landscape,
            "targeted_followup_generated_count": sum(int(state.get("generated_gap_action_count") or 0) for state in question_states),
            "targeted_followup_executed_count": sum(len(state.get("gap_execution_log") or []) for state in question_states),
            "targeted_followup_executed_rate": _targeted_followup_rate(question_states),
            "model_pool": model_pool_diagnostics(),
            "validation_stage": validation_diagnostics,
            "checkpoint_paths": checkpoint_paths,
            "domain_specific_evidence_ratio": round(sum(str(item.get("scope") or "") == "domain_specific" for item in evidence_map) / len(evidence_map), 3) if evidence_map else 0.0,
            "unsupported_inference_count": int(entailment.get("unsupported_inference_count") or 0),
            "entailment": {key: entailment.get(key) for key in ("status", "claim_count", "entailed_count", "inference_count", "unsupported_count", "label_mismatch_count")},
        },
        "retrieval": {
            "evidence_diagnostics": {
                "mode": "question_loop",
                "evidence_count": len(evidence_map),
                "rejected_count": len(rejected_evidence),
            },
            "diagnostics": question_result["pipeline_diagnostics"],
            "gaps": coverage.get("fatal_gaps") or [],
            "rounds": [{"question_id": state.get("question_id"), "rounds_used": state.get("rounds_used"), "stop_reason": state.get("stop_reason")} for state in question_states],
        },
        "document_titles": [str(doc.get("title") or "") for doc in documents],
    }
    validation_checkpoint = _save_question_loop_checkpoint(job_id, "validation", {
        "quality_gate": quality_gate,
        "grounding": grounding,
        "validation": validation,
        "report_quality": report_quality,
        "report_document": report_document,
        "report_status": report_status,
        "report_completeness": report_completeness,
        "matrix_coverage": matrix_coverage,
        "documents": documents,
        "editorial": written.get("editorial") or {},
        "evidence_map": evidence_map,
        "pipeline_diagnostics": result["pipeline_diagnostics"],
    })
    if validation_checkpoint:
        checkpoint_paths.append(validation_checkpoint)
    checkpoint_paths.extend(revision_pass_checkpoint_paths)
    result["pipeline_diagnostics"]["checkpoint_paths"] = checkpoint_paths
    artifact_paths = _write_research_review_artifacts(job_id=job_id, result=result, input_data=input_data)
    artifact_paths = list(dict.fromkeys([*artifact_paths, *checkpoint_paths]))
    result["artifacts"] = artifact_paths
    result["delivery"] = _delivery_payload({**result, "paths": artifact_paths}, input_data)
    _step(
        job_id,
        "approval",
        "waiting_approval",
        (
            "报告质量门通过，可审批落库"
            if quality_gate["approval_ready"]
            else "阶段性报告已生成，仍需补充研究"
            if report_status != "complete_report"
            else "报告已生成，但质量门未通过"
        ),
        items=[{"label": "quality", "value": quality_gate["status"]}, {"label": "report_score", "value": report_quality["average_score"]}],
    )
    return result


# ---------------------------------------------------------------------------
# single_url pipeline -- see docs/modules/single-url-brief-pipeline-v0.1.md
#
# Independent short path for "summarize/analyze this one link" requests: fetch one URL with
# the shared fetch_url_with_chain, run a two-stage generation (structured extraction with
# citation locators, then narrative synthesis), splice verbatim quotes back in from the raw
# fetched text by locator (never trust model-generated quote text), and deliver directly
# without HumanInTheLoopMiddleware approval. Does not touch question_loop's evidence chain,
# coverage gate, or multi-source retrieval -- those stay exclusively in
# _run_question_loop_research / backend/research/*.
# ---------------------------------------------------------------------------

_SINGLE_URL_STEPS = [
    ("fetch", "抓取网页正文"),
    ("extract", "结构化抽取论点与数据"),
    ("synthesize", "综合生成简报"),
    ("deliver", "交付简报"),
]

_SINGLE_URL_MAX_CONTENT_CHARS = 60000
_SINGLE_URL_MAX_PARAGRAPHS = 220
_SINGLE_URL_PROVIDER_CHAIN_DEFAULT = ["tavily", "firecrawl", "basic"]


def _single_url_initial_steps() -> list[dict[str, Any]]:
    return [{"id": step_id, "title": title, "status": "pending"} for step_id, title in _SINGLE_URL_STEPS]


def _single_url_provider_chain() -> list[str]:
    """Scoped override of fetch_url_with_chain's provider order, single_url-only.

    The global URL_FETCH_PROVIDER_CHAIN env var (used by question_loop's much higher-volume
    multi-source retrieval) defaults to trying the free local `basic` (BeautifulSoup) fetch
    before the paid Tavily/Firecrawl extractors. Real A/B testing against several live URLs
    (see docs/modules/single-url-brief-pipeline-v0.2.md 7) showed `basic` frequently returns
    text with zero real paragraph breaks, forcing this pipeline's citation splice-back to slice
    from large, noisy packed blocks instead of the source's real paragraph units -- Tavily's
    extraction reliably preserves real paragraph structure, producing materially cleaner
    verbatim quotes, for a sub-2-second latency cost that's negligible for a one-URL brief
    (unlike a multi-source research job fetching dozens of sources per run). Scoping this
    override to single_url only, rather than changing URL_FETCH_PROVIDER_CHAIN globally, keeps
    question_loop's fetch cost/volume profile untouched.
    """
    raw = os.getenv("SINGLE_URL_FETCH_PROVIDER_CHAIN", "").strip()
    if not raw:
        return list(_SINGLE_URL_PROVIDER_CHAIN_DEFAULT)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _run_single_url_research(job_id: str, input_data: dict[str, Any]) -> dict[str, Any]:
    url = str(input_data.get("url") or "").strip()
    if not url:
        raise ValueError("url is required for pipeline_mode=single_url")
    user_request = str(input_data.get("user_request") or "").strip() or f"总结分析这个链接：{url}"
    channel_id = str(input_data.get("channel_id") or "").strip()

    job_service.set_steps(job_id, _single_url_initial_steps())

    _step(job_id, "fetch", "running", f"正在抓取 {url}")
    fetch_result = fetch_url_with_chain(url, provider_chain=_single_url_provider_chain())
    document = fetch_result.get("document") if isinstance(fetch_result, dict) else None
    content = str((document or {}).get("content") or "").strip()
    if not document or not content:
        attempts = fetch_result.get("provider_attempts") if isinstance(fetch_result, dict) else []
        reason = "; ".join(
            str(attempt.get("reason") or attempt.get("status") or "") for attempt in (attempts or []) if isinstance(attempt, dict)
        )
        _step(job_id, "fetch", "failed", f"抓取失败：{reason or '未获取到正文'}")
        raise ValueError(f"无法抓取链接正文：{url}（{reason or '未获取到正文'}）")
    if _looks_like_unreadable_content(content):
        # fetch_url_with_chain's own success/quality heuristics (tools/web.py::_content_quality)
        # only check length/error-page/CSS-boilerplate markers -- they don't detect raw binary
        # (e.g. an undecoded PDF byte stream when a provider mistakes a PDF response for text,
        # observed for real arxiv-style PDF URLs without a literal .pdf suffix). Left unchecked,
        # this pipeline would otherwise "succeed" with an empty/degenerate brief that looks like
        # a normal thin-content result instead of surfacing that the fetch itself was unusable.
        # This check lives entirely in single_url's own code, not in fetch_url_with_chain.
        _step(job_id, "fetch", "failed", "抓取到的内容疑似无法正常解析（可能是二进制/未解码内容）")
        raise ValueError(f"抓取到的内容无法正常解析为文本，暂不支持这个链接：{url}")
    source_title = str(document.get("title") or url).strip()
    truncated = len(content) > _SINGLE_URL_MAX_CONTENT_CHARS
    if truncated:
        content = content[:_SINGLE_URL_MAX_CONTENT_CHARS]
    _step(
        job_id,
        "fetch",
        "succeeded",
        f"抓取成功：{source_title}（{len(content)} 字符{'，已截断' if truncated else ''}）",
        items=[{"label": "provider", "value": document.get("provider") or ""}],
    )

    paragraphs = _segment_source_paragraphs(content)
    if not paragraphs:
        _step(job_id, "extract", "failed", "正文为空，无法抽取")
        raise ValueError(f"链接正文为空，无法生成简报：{url}")

    _step(job_id, "extract", "running", "正在结构化抽取核心论点与数据")
    extraction = _extract_single_url_structure(
        user_request=user_request, url=url, title=source_title, paragraphs=paragraphs
    )
    key_points = extraction.get("key_points") or []
    key_data = extraction.get("key_data") or []
    _step(
        job_id,
        "extract",
        "succeeded",
        f"抽取 {len(key_points)} 条核心论点、{len(key_data)} 条关键数据",
    )

    _step(job_id, "synthesize", "running", "正在综合生成简报正文")
    synthesis = _synthesize_single_url_brief(
        user_request=user_request, url=url, title=source_title, key_points=key_points, key_data=key_data
    )
    brief_title = _single_url_brief_title(source_title=source_title, url=url)
    brief_markdown, citation_count = _render_single_url_brief_markdown(
        title=brief_title, url=url, synthesis=synthesis, paragraphs=paragraphs
    )
    _step(
        job_id,
        "synthesize",
        "succeeded",
        f"简报生成完成（{len(brief_markdown)} 字，{citation_count} 条原文引用）",
    )

    annotations, annotation_diagnostics = _compute_single_url_brief_annotations(synthesis=synthesis, paragraphs=paragraphs)
    if annotation_diagnostics is not None:
        brief_markdown = _insert_annotation_failure_notice(brief_markdown)

    _step(job_id, "deliver", "running", "正在交付简报")
    artifact_paths = _write_single_url_brief_artifact(job_id=job_id, title=brief_title, markdown=brief_markdown)
    annotations_artifact_path = _write_single_url_brief_annotations_artifact(job_id=job_id, annotations=annotations)
    if annotations_artifact_path:
        artifact_paths.append(annotations_artifact_path)
    brief_html = render_single_url_brief_html(
        title=brief_title,
        url=url,
        synthesis=synthesis,
        paragraphs=paragraphs,
        annotations=annotations,
        annotation_diagnostics=annotation_diagnostics,
    )
    html_artifact_path = _write_single_url_brief_html_artifact(job_id=job_id, html=brief_html)
    if html_artifact_path:
        artifact_paths.append(html_artifact_path)
    delivery = _deliver_single_url_brief(markdown=brief_markdown, title=brief_title, input_data=input_data)
    deliver_summary = "已生成飞书文档" if delivery.get("feishu_doc_url") else "已生成简报文档"
    _step(
        job_id,
        "deliver",
        "succeeded",
        f"{deliver_summary}：{delivery.get('feishu_doc_url') or (artifact_paths[0] if artifact_paths else '')}",
    )

    return {
        # Deliberately no "status": "waiting_approval" -- job_service._run() defaults any
        # other/missing status to "succeeded", so this pipeline never enters the
        # question_loop-only waiting_approval/approve_research_job_tool HITL gate (see
        # docs/modules/single-url-brief-pipeline-v0.1.md 7).
        "pipeline_mode": "single_url",
        "channel_id": channel_id or None,
        "message": "单 URL 简报已生成并交付，未落库；如需存入知识库请另行确认。",
        "url": url,
        "source_title": source_title,
        "suggested_topic": "",
        "documents": [{"title": brief_title, "content": brief_markdown}],
        "document_titles": [brief_title],
        "brief_markdown": brief_markdown,
        "citation_count": citation_count,
        "artifacts": artifact_paths,
        "delivery": delivery,
        "diagnostics": annotation_diagnostics or {},
    }


_SINGLE_URL_MAX_PARAGRAPH_CHARS = 900


_BINARY_MAGIC_PREFIXES = ("%PDF-", "PK\x03\x04", "\x89PNG", "\xff\xd8\xff", "GIF87a", "GIF89a")


def _looks_like_unreadable_content(content: str) -> bool:
    """Best-effort detection of raw binary/undecoded content masquerading as fetched text.

    Observed for real arxiv-style PDF URLs (no literal .pdf suffix, so
    fetch_url_with_chain's PDF adapter never triggers): a text-oriented provider ends up
    returning the undecoded PDF byte stream, which passes tools/web.py's own quality checks
    (length/error-page/CSS-boilerplate heuristics only) but is unusable for extraction --
    without this check the pipeline would silently produce an empty, degenerate-looking
    brief instead of surfacing that the fetch itself didn't yield readable text.

    A naive printable-character ratio turned out too lenient on real data: decoding a
    compressed PDF byte stream as UTF-8 (with decode errors replaced) yields mostly U+FFFD
    replacement characters and control bytes, both of which Python still counts as
    "printable" often enough to slip past a plain ratio check. Checking for the replacement
    character and control-byte ratios directly, plus common binary magic-byte prefixes,
    catches the real failure mode this was built for without needing a fragile ratio tuned
    against one sample.
    """
    sample = content[:2000]
    if not sample:
        return True
    if any(sample.lstrip().startswith(prefix) for prefix in _BINARY_MAGIC_PREFIXES):
        return True
    replacement_ratio = sample.count("�") / len(sample)
    if replacement_ratio > 0.02:
        return True
    control_chars = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\r\t")
    return (control_chars / len(sample)) > 0.05


def _segment_source_paragraphs(content: str) -> list[dict[str, str]]:
    """Split raw fetched text into numbered paragraphs that double as citation locators.

    Each paragraph's text is an exact (stripped) substring of the original fetched content --
    this list is the single source of truth _render_single_url_brief_markdown slices verbatim
    quotes from. The model is only ever shown these ids, never asked to reproduce paragraph
    text, so a hallucinated quote is architecturally impossible (see design doc 5.2).

    Blank-line splitting is the ideal case, but real fetches often don't have it -- e.g. the
    BeautifulSoup fallback in fetch_url_with_chain joins extracted text with single newlines
    and no blank lines at all, so naive blank-line splitting yields exactly one "paragraph"
    spanning the entire document. Oversized blocks are recursively split (by line, then by
    sentence, then by hard character slice as a last resort) instead of being truncated --
    truncating here would silently drop everything past the cap from both extraction and
    citation, which for a single-blob fetch means losing the entire article body.
    """
    text = str(content or "").replace("\r\n", "\n")
    raw_blocks = re.split(r"\n\s*\n+", text)
    paragraphs: list[dict[str, str]] = []
    for raw in raw_blocks:
        candidate = raw.strip()
        if not candidate:
            continue
        for chunk in _split_oversized_text(candidate, _SINGLE_URL_MAX_PARAGRAPH_CHARS):
            if not chunk.strip():
                continue
            paragraphs.append({"id": f"P{len(paragraphs) + 1}", "text": chunk.strip()})
            if len(paragraphs) >= _SINGLE_URL_MAX_PARAGRAPHS:
                return paragraphs
    return paragraphs


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return [
            piece
            for chunk in _pack_lines(lines, max_chars, joiner="\n")
            for piece in _split_oversized_text(chunk, max_chars)
        ]
    sentences = [item for item in re.split(r"(?<=[。！？.!?])\s+", text) if item.strip()]
    if len(sentences) > 1:
        return _pack_lines(sentences, max_chars, joiner=" ")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _pack_lines(items: list[str], max_chars: int, *, joiner: str) -> list[str]:
    """Greedily pack consecutive items into <=max_chars chunks without dropping any content."""
    chunks: list[str] = []
    buffer = ""
    for item in items:
        candidate = f"{buffer}{joiner}{item}" if buffer else item
        if len(candidate) > max_chars and buffer:
            chunks.append(buffer)
            buffer = item
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return chunks


def _extract_single_url_structure(
    *, user_request: str, url: str, title: str, paragraphs: list[dict[str, str]]
) -> dict[str, Any]:
    valid_ids = {item["id"] for item in paragraphs}
    payload = {
        "task": (
            "Read the numbered source paragraphs below (from one fetched web page) and extract "
            "structured findings for a brief: (1) the core arguments/claims the source makes, "
            "(2) key data points, statistics, or figures it reports. For EVERY item, attach a "
            "locator that is the exact paragraph id where that argument or figure appears. "
            "Paraphrase in your own words -- do not copy sentences verbatim from the source "
            "paragraphs into your output; the system handles verbatim quoting separately."
        ),
        "user_request": user_request,
        "source_url": url,
        "source_title": title,
        "paragraphs": paragraphs,
        "rules": [
            "locator must be one of the given paragraph ids exactly as written; never invent an id.",
            "Paraphrase, do not quote -- reproducing original sentences here defeats the point.",
            "If the source has little usable content, return fewer items rather than padding.",
            "Both lists may be empty if genuinely absent; do not fabricate items to fill them.",
        ],
        "schema": {
            "key_points": [{"claim": "paraphrased core argument", "locator": "P#"}],
            "key_data": [{"stat": "paraphrased data point or figure", "locator": "P#"}],
        },
    }
    parsed = _research_json_model(payload)
    return {
        "key_points": _filter_valid_locator_items(parsed.get("key_points"), "claim", valid_ids),
        "key_data": _filter_valid_locator_items(parsed.get("key_data"), "stat", valid_ids),
    }


def _filter_valid_locator_items(raw_items: Any, text_key: str, valid_ids: set[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get(text_key) or "").strip()
        locator = str(item.get("locator") or "").strip()
        if not text or locator not in valid_ids:
            continue
        items.append({text_key: text, "locator": locator})
    return items


_SINGLE_URL_WRITER_SYSTEM_PROMPT = (
    "你是资深行业分析师，负责把已抽取的结构化发现写成一篇连贯的单来源简报正文。"
    "只能使用 payload 中给出的 key_points/key_data，不得引入材料之外的事实；"
    "涉及引用原文时只输出 locator（段落编号），绝不能自己写出该段原文内容，"
    "系统会在你的输出之外，从原始正文按 locator 精确回填引用文字。"
    "你的输出是几个独立的 JSON 字段值，不是一整篇带标题的成品报告——"
    "报告的标题、小节结构和最终排版由系统组装，你只需要按 schema 把每个字段的纯文字内容填好，"
    "不要在任何字段里自己加『摘要』『点评』这类小标题，也不要把多个字段的内容合并写进同一个字段。"
)


def _build_single_url_synthesis_payload(
    *,
    user_request: str,
    url: str,
    title: str,
    key_points: list[dict[str, str]],
    key_data: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "task": (
            "Using ONLY the structured findings below (already extracted from one fetched "
            "document), fill in the JSON fields described in `schema`. Do not introduce facts "
            "beyond what is in key_points/key_data."
        ),
        "user_request": user_request,
        "source_url": url,
        "source_title": title,
        "key_points": key_points,
        "key_data": key_data,
        "length_guidance": (
            "全文中文合计约 1000-2000 字，作为参考基准按信息量浮动：信息量小的来源不要为了凑字数硬写，"
            "信息量大的来源也不要为了控制篇幅生硬截断。"
        ),
        "rules": [
            "最重要的格式约束：输出必须是拆分好的独立字段值，每个字段只装该字段自己的内容，用纯文字"
            "（可以分段落，但不要用 Markdown 标题/列表符号）；绝不能把 summary/key_points_narrative/"
            "citations/commentary 四块内容合并写成一整篇带标题的完整报告塞进某一个字段——那是系统的"
            "组装工作，不是你的输出格式。",
            "summary：3-5 句摘要，先说清楚这篇来源整体在讲什么、结论是什么；只写摘要本身，不要在这个"
            "字段里再写核心论点、引用或点评的内容。",
            "key_points_narrative：只写核心论点与数据的叙述本身；把 key_points 和 key_data 组织成有"
            "论证脉络的连续段落（可分多段），不要逐条罗列成清单，也不要在这里重复 summary 或写出"
            "『摘要』『点评』这类小标题。",
            "citations 必须是一个 JSON 数组，每个元素形如 {\"locator\": \"P3\", \"reason\": \"...\"}；"
            "挑 2-6 条最值得在『关键原文引用』小节展示的 locator；citations 里只给 locator 和 reason，"
            "绝不能给出该段原文文字本身，也不要把 citations 内容写成文字塞进别的字段。",
            "commentary 必须是一个 JSON 对象，包含 supporting_evidence（支持证据）、counter_evidence"
            "（反驳或制约证据；如果来源本身没有反面信息，如实说明，不要编造）、limitations（局限性）、"
            "applicable_scenarios（适用场景）四个字符串字段，以及一句话的 overall_credibility"
            "（整体可信度判断）；不要把这些内容写进 key_points_narrative。",
            "全部字段使用中文；不要在任何字段中直接写出某个 locator 对应段落的原文文字。",
        ],
        "schema": {
            "summary": "（纯文字，3-5 句摘要，不含标题/列表）",
            "key_points_narrative": "（纯文字，连续段落，不含标题/列表，不重复摘要或点评内容）",
            "citations": [{"locator": "P3", "reason": "为何值得引用这段原文（一句话）"}],
            "commentary": {
                "supporting_evidence": "（纯文字）",
                "counter_evidence": "（纯文字）",
                "limitations": "（纯文字）",
                "applicable_scenarios": "（纯文字）",
                "overall_credibility": "（一句话）",
            },
        },
    }


def _synthesis_fields_collapsed(synthesis: dict[str, Any]) -> bool:
    """Detect the observed failure mode where the model merges summary/citations/commentary
    into `key_points_narrative` instead of filling each schema field separately (see
    docs/evaluations/e2e-eval-sweep-20260726.md section 3): `key_points_narrative` ends up long
    while `summary`/`citations`/`commentary` are left empty. A field-by-field length/emptiness
    check catches this without depending on the exact markdown markers the model used to fake
    the merged sections.
    """
    summary = str(synthesis.get("summary") or "").strip()
    narrative = str(synthesis.get("key_points_narrative") or "").strip()
    citations = synthesis.get("citations")
    commentary = synthesis.get("commentary")
    commentary_has_content = isinstance(commentary, dict) and any(
        str(commentary.get(field) or "").strip() for field in
        ("supporting_evidence", "limitations", "applicable_scenarios", "overall_credibility")
    )
    citations_present = isinstance(citations, list) and len(citations) > 0
    return bool(narrative) and not summary and not citations_present and not commentary_has_content


def _single_url_synthesis_missing_fields(synthesis: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not str(synthesis.get("summary") or "").strip():
        missing.append("summary")
    if not str(synthesis.get("key_points_narrative") or "").strip():
        missing.append("key_points_narrative")
    if not isinstance(synthesis.get("citations"), list) or not synthesis.get("citations"):
        missing.append("citations")
    commentary = synthesis.get("commentary") if isinstance(synthesis.get("commentary"), dict) else {}
    for field in ("supporting_evidence", "limitations", "applicable_scenarios", "overall_credibility"):
        if not str(commentary.get(field) or "").strip():
            missing.append(f"commentary.{field}")
    return missing


def _synthesize_single_url_brief(
    *,
    user_request: str,
    url: str,
    title: str,
    key_points: list[dict[str, str]],
    key_data: list[dict[str, str]],
) -> dict[str, Any]:
    payload = _build_single_url_synthesis_payload(
        user_request=user_request, url=url, title=title, key_points=key_points, key_data=key_data
    )
    synthesis = _research_writer_json_model(payload)
    missing_fields = _single_url_synthesis_missing_fields(synthesis)
    if not missing_fields:
        return synthesis

    failure_kind = "collapsed" if _synthesis_fields_collapsed(synthesis) else "incomplete"
    emit_event(
        f"research.single_url_synthesis_fields_{failure_kind}",
        "research_service",
        "Single-URL brief synthesis missed required output fields; retrying with a corrective prompt.",
        level="warning",
        data={
            "source_url": url,
            "failure_kind": failure_kind,
            "missing_fields": missing_fields,
            "narrative_chars": len(str(synthesis.get("key_points_narrative") or "")),
        },
    )
    retry_payload = dict(payload)
    retry_payload["rules"] = [
        "你上一次的输出缺少这些必填字段：" + "、".join(missing_fields) + "。schema 中每个字段必须"
        "独立填写，不能留空，也不能把多个字段合并进 key_points_narrative。citations 必须引用输入中"
        "真实存在的 locator 和 quote。",
        *payload["rules"],
    ]
    retried = _research_writer_json_model(retry_payload)
    retry_missing_fields = _single_url_synthesis_missing_fields(retried)
    if retry_missing_fields:
        raise RuntimeError(
            "single_url synthesis incomplete after retry: " + ", ".join(retry_missing_fields)
        )
    return retried


def _single_url_brief_title(*, source_title: str, url: str) -> str:
    base = re.sub(r"\s+", " ", str(source_title or "")).strip() or url
    return f"链接简报 - {base[:60]}"


def _render_single_url_brief_markdown(
    *, title: str, url: str, synthesis: dict[str, Any], paragraphs: list[dict[str, str]]
) -> tuple[str, int]:
    paragraph_by_id = {item["id"]: item["text"] for item in paragraphs}
    lines = [f"# {title}", "", f"来源：{url}", "", "## 摘要", "", str(synthesis.get("summary") or "").strip() or "（模型未返回摘要）", ""]
    lines += [
        "## 核心论点与数据",
        "",
        str(synthesis.get("key_points_narrative") or "").strip() or "（模型未返回核心论点与数据）",
        "",
        "## 关键原文引用",
        "",
    ]
    citation_count = 0
    for item in synthesis.get("citations") or []:
        if not isinstance(item, dict):
            continue
        locator = str(item.get("locator") or "").strip()
        # Never fabricate a quote: only ever splice text that exists verbatim in the original
        # fetched paragraphs, looked up by the model's locator. An unresolvable locator is
        # silently dropped rather than backfilled with model-authored text.
        paragraph_text = paragraph_by_id.get(locator)
        if not paragraph_text:
            continue
        reason = str(item.get("reason") or "").strip()
        citation_count += 1
        lines.append(f"> {paragraph_text}")
        lines.append("")
        lines.append(f"（原文 {locator}）" + (f"：{reason}" if reason else ""))
        lines.append("")
    if citation_count == 0:
        lines.append("（原文未提供可精确定位的引用片段。）")
        lines.append("")
    commentary = synthesis.get("commentary") if isinstance(synthesis.get("commentary"), dict) else {}
    lines += [
        "## 点评",
        "",
        f"- 支持证据：{commentary.get('supporting_evidence') or '未提供'}",
        f"- 反驳/制约证据：{commentary.get('counter_evidence') or '原文未提供明显的反面信息'}",
        f"- 局限性：{commentary.get('limitations') or '未提供'}",
        f"- 适用场景：{commentary.get('applicable_scenarios') or '未提供'}",
        f"- 整体可信度：{commentary.get('overall_credibility') or '未提供'}",
        "",
    ]
    return "\n".join(lines).strip() + "\n", citation_count


_NARRATIVE_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？；]|\n+")

# Cosine-similarity threshold below which a narrative sentence is left unannotated rather than
# force-bound to its best-scoring candidate locator. This is an initial empirical guess (design
# doc single-url-brief-citation-binding-v0.1.md 3.3) -- it still needs calibration against a
# real distribution of scores from actual briefs before being treated as final.
_ANNOTATION_SIMILARITY_THRESHOLD = 0.5


def _split_narrative_into_sentences(text: str) -> list[dict[str, Any]]:
    """Split narrative text into sentences on terminal punctuation/newlines, keeping char offsets.

    Offsets point into the original `text` (not the stripped sentence), so downstream renderers
    can locate a sentence precisely without relying on substring search (design doc 3.2/5).
    """
    sentences: list[dict[str, Any]] = []
    cursor = 0
    boundaries = [match.end() for match in _NARRATIVE_SENTENCE_BOUNDARY_RE.finditer(text)]
    boundaries.append(len(text))
    for boundary in boundaries:
        if boundary <= cursor:
            continue
        raw = text[cursor:boundary]
        stripped = raw.strip()
        if stripped:
            lstrip_len = len(raw) - len(raw.lstrip())
            rstrip_len = len(raw) - len(raw.rstrip())
            start = cursor + lstrip_len
            end = boundary - rstrip_len
            sentences.append({"sentence": text[start:end], "start": start, "end": end})
        cursor = boundary
    return sentences


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _resolve_embedding_credentials() -> tuple[str | None, str | None]:
    """Resolve (api_key, base_url) as one atomic provider pair, not independently-chosen fields.

    Real .env has EMBEDDING_API_KEY/EMBEDDING_BASE_URL unset, the real dedicated embedding
    credentials under the LightRAG-style EMBEDDING_BINDING_API_KEY/EMBEDDING_BINDING_HOST
    (same fallback workers.py's rerank embedding lookup already relies on), and OPENAI_API_KEY/
    OPENAI_BASE_URL pointing at a completely different, embedding-incapable chat provider.
    Picking api_key and base_url via independent `or` chains previously mixed
    EMBEDDING_BINDING_API_KEY with OPENAI_BASE_URL -- confirmed via real verification to 404
    against the wrong provider -- so candidates here are tried as matched pairs instead."""
    candidates = [
        (os.getenv("EMBEDDING_API_KEY"), os.getenv("EMBEDDING_BASE_URL")),
        (os.getenv("EMBEDDING_BINDING_API_KEY"), os.getenv("EMBEDDING_BINDING_HOST")),
        (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")),
    ]
    for api_key, base_url in candidates:
        if api_key:
            return api_key, base_url or None
    return None, None


def _embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """Batch-embed `texts` in a single request, reusing the same env config as
    model_service.py::_check_embedding (EMBEDDING_MODEL/EMBEDDING_API_KEY/EMBEDDING_BASE_URL)."""
    api_key, base_url = _resolve_embedding_credentials()
    if not api_key:
        raise RuntimeError("EMBEDDING_API_KEY (or OPENAI_API_KEY/EMBEDDING_BINDING_API_KEY) is not configured")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model_name, input=texts)
    return [item.embedding for item in response.data]


def _bind_narrative_sentences_to_citations(
    *, narrative: str, citations: list[dict[str, Any]], paragraph_by_id: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Non-generative sentence<->locator binding (design doc single-url-brief-citation-binding-v0.1.md).

    Returns (annotations, diagnostics). diagnostics is None on a normal outcome -- including a
    normal outcome that happens to produce zero annotations (no sentences, no citations, or every
    sentence scored below threshold) -- and is only set when the embedding call itself could not
    be made or failed, per the design's "silent degrade but leave a trace" contract.
    """
    sentences = _split_narrative_into_sentences(narrative)
    if not sentences:
        return [], None

    # Candidate scope is strictly the paragraphs already selected by citations -- never the full
    # paragraph set (design doc 3.1) -- so any annotation the reader clicks always resolves to a
    # locator already surfaced in the "关键原文引用" section.
    candidate_locators: list[str] = []
    seen_locators: set[str] = set()
    for item in citations or []:
        locator = str(item.get("locator") or "").strip() if isinstance(item, dict) else ""
        if locator and locator in paragraph_by_id and locator not in seen_locators:
            seen_locators.add(locator)
            candidate_locators.append(locator)
    if not candidate_locators:
        return [], None

    try:
        texts = [item["sentence"] for item in sentences] + [paragraph_by_id[locator] for locator in candidate_locators]
        vectors = _embed_texts_batch(texts)
        sentence_vectors = vectors[: len(sentences)]
        candidate_vectors = vectors[len(sentences):]
    except Exception as exc:
        return [], {"annotation_binding": "failed", "error": str(exc)}

    annotations: list[dict[str, Any]] = []
    for sentence, vector in zip(sentences, sentence_vectors):
        best_locator: str | None = None
        best_score = -1.0
        for locator, candidate_vector in zip(candidate_locators, candidate_vectors):
            score = _cosine_similarity(vector, candidate_vector)
            if score > best_score:
                best_score = score
                best_locator = locator
        if best_locator is not None and best_score >= _ANNOTATION_SIMILARITY_THRESHOLD:
            annotations.append({
                "sentence": sentence["sentence"],
                "start": sentence["start"],
                "end": sentence["end"],
                "locator": best_locator,
                "score": round(best_score, 4),
            })
    return annotations, None


def _compute_single_url_brief_annotations(
    *, synthesis: dict[str, Any], paragraphs: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    narrative = str(synthesis.get("key_points_narrative") or "")
    paragraph_by_id = {item["id"]: item["text"] for item in paragraphs}
    return _bind_narrative_sentences_to_citations(
        narrative=narrative, citations=synthesis.get("citations") or [], paragraph_by_id=paragraph_by_id
    )


_ANNOTATION_FAILURE_NOTICE = "本次未生成正文引用标注。"


def _insert_annotation_failure_notice(markdown: str) -> str:
    """Insert the annotation-binding-failed notice into the delivered markdown body itself, so
    every delivery channel (local file, Feishu doc, ...) carries it -- not just one branch."""
    lines = markdown.split("\n")
    if lines and lines[0].startswith("# "):
        lines.insert(1, "")
        lines.insert(2, _ANNOTATION_FAILURE_NOTICE)
        return "\n".join(lines)
    return f"{_ANNOTATION_FAILURE_NOTICE}\n\n{markdown}"


def _write_single_url_brief_annotations_artifact(*, job_id: str, annotations: list[dict[str, Any]]) -> str | None:
    if os.getenv("RESEARCH_WRITE_REVIEW_ARTIFACTS", "true").strip().lower() in {"0", "false", "no"}:
        return None
    artifact_dir = RESEARCH_ARTIFACT_DIR / _safe_artifact_name(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "brief.annotations.json"
    path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_single_url_brief_artifact(*, job_id: str, title: str, markdown: str) -> list[str]:
    if os.getenv("RESEARCH_WRITE_REVIEW_ARTIFACTS", "true").strip().lower() in {"0", "false", "no"}:
        return []
    artifact_dir = RESEARCH_ARTIFACT_DIR / _safe_artifact_name(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "brief.md"
    path.write_text(markdown, encoding="utf-8")
    return [str(path)]


def _write_single_url_brief_html_artifact(*, job_id: str, html: str) -> str | None:
    """Write the evidence-annotated reading HTML (VISION.md 第2节) alongside brief.md/
    brief.annotations.json -- same artifact dir, same write-toggle, same not-fabricated-content
    guarantee (the HTML is rendered from the same synthesis/paragraphs/annotations data, not
    regenerated). See backend/services/evidence_reader_service.py::render_single_url_brief_html."""
    if os.getenv("RESEARCH_WRITE_REVIEW_ARTIFACTS", "true").strip().lower() in {"0", "false", "no"}:
        return None
    artifact_dir = RESEARCH_ARTIFACT_DIR / _safe_artifact_name(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "brief.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def _deliver_single_url_brief(*, markdown: str, title: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Deliver the finished brief directly -- no HITL, no waiting_approval.

    Reuses feishu_doc_service.create_document/append_markdown directly (not the
    question_loop-flavored create_research_review_doc, whose "调研审核包" framing implies a
    pending-review artifact this pipeline does not have).
    """
    payload: dict[str, Any] = {"type": "local_markdown", "feishu_doc_url": None}
    if not _should_deliver_feishu_doc(input_data, [markdown]):
        return payload
    try:
        from . import feishu_doc_service

        document = feishu_doc_service.create_document(title=title, folder_token=os.getenv("FEISHU_DOC_FOLDER_TOKEN") or None)
        document_id = str(document["document_id"])
        try:
            feishu_doc_service.append_markdown(document_id=document_id, markdown=markdown)
        except Exception:
            feishu_doc_service.append_markdown(document_id=document_id, markdown=markdown, text_only=True)
    except Exception as exc:
        return {**payload, "diagnostics": {"feishu_doc_delivery": "failed", "error": str(exc)}}
    base = os.getenv("FEISHU_DOC_URL_BASE", "https://www.feishu.cn/docx").strip() or "https://www.feishu.cn/docx"
    return {
        **payload,
        "type": "feishu_doc",
        "feishu_doc_url": f"{base.rstrip('/')}/{document_id}",
        "feishu_document_id": document_id,
        "feishu_doc_title": document.get("title"),
        "diagnostics": {"feishu_doc_delivery": "succeeded"},
    }


def _retrieve_question_round(
    *,
    job_id: str,
    user_request: str,
    strategy: str,
    depth: str,
    state: ResearchQuestionState,
    queries: list[str],
    round_index: int,
) -> dict[str, Any]:
    payload = _question_retrieval_payload(
        [state],
        {state.question_id: queries},
    )
    query_template = dict((payload.get("query_plan") or {}).get("queries", [{}])[0])
    query_rows = [
        {**query_template, "query": query}
        for query in queries
    ] or [{**query_template, "query": state.question}]
    payload["query_plan"] = {"query_limit": len(query_rows), "queries": query_rows}
    remaining_materials = max(0, state.budget.max_materials - len(state.materials))
    remaining_urls = max(0, state.budget.max_fetch_urls - len(state.attempted_fetch_urls))
    round_limit = state.budget.initial_fetch_limit if round_index == 1 else state.budget.followup_fetch_limit
    fetch_limit = max(1, min(round_limit, remaining_materials or 1, remaining_urls or 1))
    task = "\n".join([
        "source_mode=web 问题级 Deep Research。",
        f"用户需求：{user_request}",
        f"topic={job_id}_{state.question_id}_r{round_index}",
        f"research_strategy={strategy}",
        f"depth={depth}",
        f"query_limit={max(1, len(query_rows))}",
        f"fetch_url_limit={fetch_limit}",
        "required_capabilities=broad_discovery,docs_search,url_fetch",
        "必须抓取正文；仅为当前问题抽取可直接引用的 atomic evidence。",
        "优先来源类型：" + "、".join(state.preferred_source_types),
        "来源组合要求：" + json.dumps(state.source_requirements, ensure_ascii=False),
        "当前定向补检索动作：" + json.dumps(state.gap_actions, ensure_ascii=False),
        "成功标准：" + "；".join(state.success_criteria),
        "<<<RESEARCH_PAYLOAD_JSON",
        json.dumps(payload, ensure_ascii=False),
        "RESEARCH_PAYLOAD_JSON>>>",
    ])
    retrieval = parse_tool_json(web_worker_tool.invoke({"task_or_url": task}))
    gaps = [str(item).strip() for item in retrieval.get("gaps") or [] if str(item).strip()]
    retrieval["suggested_queries"] = [item for item in gaps if len(item) <= 240][:3]
    return retrieval


def _retrieve_question_wave(
    *,
    job_id: str,
    user_request: str,
    strategy: str,
    depth: str,
    all_states: list[ResearchQuestionState],
    active_states: list[ResearchQuestionState],
    queries_by_id: dict[str, list[str]],
    round_index: int,
    source_pool: GlobalSourcePool,
    material_store: JobMaterialStore | None = None,
    routed_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Share discovery/fetch globally while each question consumes materials independently."""
    wave_started = time.perf_counter()
    material_store = material_store or JobMaterialStore()
    material_store_before = material_store.diagnostics()
    discoveries: dict[str, dict[str, Any]] = {}
    discovery_started = time.perf_counter()
    max_workers = max(1, min(len(active_states), int(os.getenv("RESEARCH_DISCOVERY_PARALLELISM", "4"))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            submit_with_audit_context(
                executor,
                _discover_question_sources,
                job_id=job_id,
                user_request=user_request,
                strategy=strategy,
                depth=depth,
                state=state,
                queries=queries_by_id.get(state.question_id) or [],
                round_index=round_index,
            ): state
            for state in active_states
        }
        for future in as_completed(futures):
            state = futures[future]
            try:
                discovery = future.result()
            except Exception as exc:  # pragma: no cover - provider dependent
                discovery = {"sources": [], "provider_attempts": [{
                    "capability": "broad_discovery",
                    "provider": "question_wave",
                    "status": "error",
                    "reason": str(exc)[:240],
                    "degraded": True,
                }]}
            discoveries[state.question_id] = discovery
            source_pool.add(discovery.get("sources") or [], state, match_states=all_states)
    discovery_elapsed_ms = int((time.perf_counter() - discovery_started) * 1000)

    # Context7 docs channel: library candidates (URL `context7://library/<id>`, carrying
    # `library_id`) were synthesized in `SourcePool.add` so the path-style IDs Context7's
    # search returns don't get silently dropped by `normalize_url`. Fetch their docs via
    # /api/v2/context out-of-band -- these candidates never enter `select`'s ranking
    # competition (they're not http-URL fetchable) and don't count against `global_cap`.
    # Without this pass the official docs Context7 already indexed never reach the material
    # pool (see project_sec03_official_sources_discovered_not_fetched_20260804: sec_03_q1
    # had `/crewaiinc/crewai` etc. in `sources` but zero official URLs fetched).
    context7_started = time.perf_counter()
    context7_library_candidates = source_pool.context7_library_candidates()
    context7_doc_count = 0
    max_context7_libraries = 0
    context7_selected_ids: list[str] = []
    context7_selected_questions: set[str] = set()
    context7_status = material_store.capability_status("context7")
    context7_disabled_reason = str(context7_status.get("reason") or "") if context7_status.get("disabled") else ""
    if context7_library_candidates and not context7_status.get("disabled"):
        state_by_id = {state.question_id: state for state in all_states}
        # Round-robin the fetch cap across active questions instead of taking the first
        # N by pool insertion order (which is thread-finish order -- non-deterministic).
        # In case-question-slot-fairness a flat cap let sec_01_q1's six loop-engineering
        # libraries fill all 6 slots before sec_03_q1's `/significant-gravitas/autogpt`
        # (discovered only by sec_03_q1) was reached, so AutoGPT official docs were
        # never fetched. See _select_context7_libraries_for_fetch for the algorithm.
        env_cap = max(1, int(os.getenv("RESEARCH_CONTEXT7_MAX_LIBRARIES_PER_ROUND", "6")))
        max_context7_per_question = max(1, int(os.getenv("RESEARCH_CONTEXT7_MAX_LIBRARIES_PER_QUESTION", "2")))
        # Dynamic cap: guarantee one slot per active question when the env cap is smaller
        # than the active-question count (job_20260805_011442 had cap=6 with 10 active
        # questions -- round-robin filled 6 slots from the first 6 questions, leaving
        # sec_04_q3 /crewaiinc/crewai starved across BOTH waves). `len(active_states)` is
        # an upper bound: questions with no context7 candidates yield their slot back to
        # the round-robin (cursor hits an empty queue). Clamp by env_cap so a small job
        # (3 questions) keeps the env floor of 6 (2 per question) instead of shrinking.
        max_context7_libraries = max(env_cap, len(active_states))
        selected_context7 = _select_context7_libraries_for_fetch(
            context7_library_candidates,
            active_states=active_states,
            all_states=all_states,
            max_total=max_context7_libraries,
            max_per_question=max_context7_per_question,
            previously_selected_questions=source_pool.context7_served_questions,
        )
        for candidate in selected_context7:
            # Merge ALL discovering questions' text into the context-retrieval query, not
            # just the alphabetically-first question's. A library shared across questions
            # (e.g. `/langchain-ai/docs` discovered by sec_02_q1 AgentExecutor +
            # sec_02_q2 with_structured_output + sec_02_q3 LangGraph) is fetched ONCE
            # (deduped by library_id); fetching with only sec_02_q1's question made
            # Context7's rerank return only AgentExecutor snippets, so sec_02_q2 got
            # nothing about structured output. The merged query makes the single fetch
            # cover every sub-topic the library was discovered for.
            questions = [
                state_by_id[qid].question
                for qid in sorted(candidate.discovered_by_question_ids)
                if qid in state_by_id
            ]
            context_query = (" ".join(questions)[:500]) if questions else (user_request or candidate.library_id)
            context_result = fetch_context7_context(candidate.library_id, context_query)
            attempts = context_result.get("provider_attempts") or []
            first_404 = next(
                (
                    str(item.get("reason") or "")
                    for item in attempts
                    if isinstance(item, dict) and "404" in str(item.get("reason") or "")
                ),
                "",
            )
            if first_404:
                material_store.disable_capability("context7", first_404)
                context7_disabled_reason = first_404
            docs_stored = material_store.record_context7_documents(
                candidate,
                context_result.get("documents") or [],
                attempts,
            )
            context7_doc_count += docs_stored

            # Only mark as fetched if we actually got content; failed attempts (API error,
            # rate limit, timeout → empty documents) should remain fetchable in later rounds.
            if docs_stored > 0:
                candidate.fetched = True
                candidate.fetch_count += 1
                context7_selected_ids.append(candidate.library_id)
                # Record this library's discovering questions as served so the NEXT wave's
                # selector deprioritizes them in favor of still-starved questions.
                context7_selected_questions.update(
                    qid for qid in candidate.discovered_by_question_ids if qid in state_by_id
                )
            else:
                # Zero content stored — API error/rate-limit/empty result. Merge the attempt
                # records into access_issues so diagnostics show what went wrong, and leave
                # candidate.fetched=False so it can be retried in a future round.
                from backend.research.orchestrator import _merge_access_issues
                for qid in candidate.discovered_by_question_ids:
                    if qid in state_by_id:
                        state = state_by_id[qid]
                        state.access_issues = _merge_access_issues(
                            state.access_issues,
                            attempts,
                        )
            # A 404 is a capability-level failure, not a library-level miss. Stop the current
            # round immediately so the remaining questions/libraries do not multiply the same
            # dead-end attempt; the job falls back to normal web/GitHub discovery.
            if first_404:
                break
        # Persist the cross-wave served set onto the pool (survives across waves within
        # one job because the orchestrator reuses the same source_pool instance).
        source_pool.context7_served_questions.update(context7_selected_questions)
    if context7_disabled_reason:
        for state in active_states:
            state.diagnostics["context7_disabled_reason"] = context7_disabled_reason
    context7_elapsed_ms = int((time.perf_counter() - context7_started) * 1000)

    # Re-score any Context7 library candidate that just got its real document body fetched
    # (or was fetched in an earlier wave but never re-scored) -- runs after the fetch loop
    # above so this wave's own new fetches benefit immediately, and before the ranker block
    # below so relevant_candidates()'s ranking uses body-content scores, not the stale
    # discovery-time library blurb. See _apply_context7_body_boost's docstring.
    context7_body_boost_started = time.perf_counter()
    context7_body_boost_diagnostics = _apply_context7_body_boost(source_pool, material_store, active_states)
    context7_body_boost_elapsed_ms = int((time.perf_counter() - context7_body_boost_started) * 1000)

    round_budget = sum(_question_round_fetch_limit(state, round_index) for state in active_states)
    # This cap applies to unique URLs fetched by the shared pool, not to the sum of per-question
    # material consumption. A fetched URL can serve several questions, so those two counters are
    # deliberately different. Production comparison case-fixed-source-fetch-cap fetched 40 unique
    # URLs across four waves with the default cap of 10; case-scaled-source-fetch-cap fetched 110
    # after the cap was scaled by active question count, but still used 34 question rounds and took
    # longer. Keep the conservative default and allow an explicit operator override.
    comparison_fetch_cap = next(
        (
            int(state.comparison_budget.get("max_fetch_urls_per_wave") or 0)
            for state in active_states
            if state.comparison_budget
        ),
        0,
    )
    global_cap = comparison_fetch_cap or max(
        1,
        int(os.getenv("RESEARCH_GLOBAL_FETCH_URLS_PER_ROUND", "10")),
    )
    selection_limit = min(round_budget, global_cap)
    ranker_mode = str(os.getenv("RESEARCH_SOURCE_RANKER_MODE", "scoped")).strip().lower()
    if ranker_mode not in {"legacy", "scoped"}:
        ranker_mode = "scoped"
    consumer_mode = str(os.getenv("RESEARCH_MATERIAL_CONSUMER_MODE", "question")).strip().lower()
    if consumer_mode not in {"batch", "question"}:
        consumer_mode = "question"
    llm_boost_mode = _llm_boost_enabled_mode()
    llm_boost_diagnostics: dict[str, Any] | None = None
    ranker_status = material_store.capability_status("ranker_llm")
    ranker_started = time.perf_counter()
    if llm_boost_mode != "0" and not ranker_status.get("disabled"):
        boost_candidates = _llm_boost_candidate_pool(source_pool, active_states)
        boost_result = _llm_boost_source_candidates(boost_candidates, active_states)
        llm_boost_diagnostics = {"mode": llm_boost_mode, **boost_result["diagnostics"]}
        failed_calls = [
            item for item in boost_result["diagnostics"].get("calls") or []
            if str(item.get("status") or "") == "failed"
        ]
        if failed_calls:
            reasons = [str(item.get("reason") or "") for item in failed_calls]
            if any("rate" in reason.lower() or "limit" in reason.lower() for reason in reasons):
                material_store.disable_capability("ranker_llm", "rate_limited; deterministic lexical/entity-aware fallback")
                ranker_status = material_store.capability_status("ranker_llm")
        if llm_boost_mode == "shadow":
            llm_boost_diagnostics["preview"] = source_pool.preview_llm_boost(boost_result["scores"], active_states)
        elif llm_boost_mode == "live":
            llm_boost_diagnostics["boosted_pair_count"] = source_pool.apply_llm_boost(boost_result["scores"], active_states)
    # Embedding rescue runs AFTER llm_boost (the cheaper pass runs second only after the boost's
    # relevance merges are in place, so rescue decides what's STILL lexically-generic), and BEFORE
    # the shadow/preview_select so the shadow diagnostics reflect the rescued relationships. See
    # GlobalSourcePool.apply_semantic_rescue for the batched/env-gated/fail-open contract.
    source_semantic_rescue_count = source_pool.apply_semantic_rescue(active_states)
    if ranker_status.get("disabled"):
        for state in active_states:
            state.diagnostics["ranker_degraded"] = True
            state.diagnostics["ranker_degraded_reason"] = str(ranker_status.get("reason") or "")
    selection_shadow = _source_selection_shadow(source_pool, active_states, max_urls=selection_limit)
    selected = source_pool.select(active_states, max_urls=selection_limit, mode=ranker_mode)
    ranker_elapsed_ms = int((time.perf_counter() - ranker_started) * 1000)
    results: dict[str, dict[str, Any]] = {}
    fetch_elapsed_ms = 0
    extraction_elapsed_ms = 0

    if consumer_mode == "batch":
        fetch_started = time.perf_counter()
        fetched = _fetch_source_batch(
            job_id=job_id,
            user_request=user_request,
            strategy=strategy,
            depth=depth,
            all_states=all_states,
            active_queries=queries_by_id,
            selected=selected,
            round_index=round_index,
        ) if selected else {}
        fetch_elapsed_ms = int((time.perf_counter() - fetch_started) * 1000)
        cache = routed_cache if routed_cache is not None else {}
        if fetched:
            for state in all_states:
                routed = _route_batch_retrieval(fetched, state, selected)
                if routed.get("selected_chunks") or routed.get("evidence_map"):
                    cache.setdefault(state.question_id, []).append(routed)
        for state in active_states:
            results[state.question_id] = _merge_retrieval_results(cache.get(state.question_id) or [])
            cache[state.question_id] = []
    else:
        fetch_started = time.perf_counter()
        if selected:
            _fetch_source_candidates(selected, material_store)
        fetch_elapsed_ms = int((time.perf_counter() - fetch_started) * 1000)
        extraction_started = time.perf_counter()
        extraction_parallelism = max(1, min(
            len(active_states),
            int(os.getenv("RESEARCH_EVIDENCE_CONSUMER_PARALLELISM", "3")),
        ))
        with ThreadPoolExecutor(max_workers=extraction_parallelism) as executor:
            futures = {}
            for state in active_states:
                consume_limit = _question_round_fetch_limit(state, round_index)
                candidates = source_pool.relevant_candidates(
                    state,
                    exclude_urls=state.attempted_fetch_urls,
                    limit=max(1, consume_limit),
                    mode=ranker_mode,
                )
                documents = material_store.documents(candidates)
                futures[submit_with_audit_context(
                    executor,
                    _extract_question_from_material_store,
                    job_id=job_id,
                    user_request=user_request,
                    strategy=strategy,
                    depth=depth,
                    state=state,
                    queries=queries_by_id.get(state.question_id) or [],
                    round_index=round_index,
                    candidates=candidates,
                    documents=documents,
                    provider_attempts=material_store.provider_attempts(candidates),
                )] = state
            for future in as_completed(futures):
                state = futures[future]
                try:
                    results[state.question_id] = future.result()
                except EvidenceExtractionExhaustedError:
                    raise
                except Exception as exc:  # pragma: no cover - model/provider dependent
                    results[state.question_id] = {
                        "sources": [],
                        "materials": [],
                        "selected_chunks": [],
                        "evidence_map": [],
                        "provider_attempts": [{
                            "capability": "material_extract",
                            "provider": "question_consumer",
                            "status": "error",
                            "reason": str(exc)[:240],
                            "degraded": True,
                        }],
                        "_diagnostics": {"chargeable_elapsed_seconds": 0.0, "model_call_count": 0},
                    }
        extraction_elapsed_ms = int((time.perf_counter() - extraction_started) * 1000)

    for state in active_states:
        result = results.setdefault(state.question_id, {})
        if consumer_mode == "batch":
            result["_diagnostics"] = {
                **(result.get("_diagnostics") or {}),
                "chargeable_elapsed_seconds": round(time.perf_counter() - wave_started, 3),
            }
        discovery = discoveries.get(state.question_id) or {}
        result["sources"] = _dedupe_result_rows(
            [*(result.get("sources") or []), *(discovery.get("sources") or [])],
            ("url", "source_url"),
        )
        result["provider_attempts"] = [
            *(result.get("provider_attempts") or []),
            *(discovery.get("provider_attempts") or []),
        ]
        results[state.question_id] = result
    attempted_urls = [
        str(url)
        for result in results.values()
        for url in result.get("attempted_urls") or []
        if str(url).startswith("http")
    ]
    material_store_after = material_store.diagnostics()
    wave_elapsed_ms = int((time.perf_counter() - wave_started) * 1000)
    wave_output = {
        "results": results,
        "diagnostics": {
            "retrieval_execution": "global_source_pool",
            "source_ranker_mode": ranker_mode,
            "material_consumer_mode": consumer_mode,
            "source_selection_shadow": selection_shadow,
            "source_ranker_llm_boost": llm_boost_diagnostics,
            "ranker_degraded": bool(ranker_status.get("disabled")),
            "ranker_degraded_reason": str(ranker_status.get("reason") or ""),
            "source_ranker_semantic_rescue": source_semantic_rescue_count,
            "source_pool": source_pool.diagnostics(),
            "material_store": material_store.diagnostics(),
            "last_global_fetch_url_count": len(selected),
            "last_shared_wave_elapsed_ms": wave_elapsed_ms,
            "context7_docs": {
                "library_candidate_count": len(context7_library_candidates),
                "document_count": context7_doc_count,
                "selected_library_ids": context7_selected_ids,
                "selected_question_ids": sorted(context7_selected_questions),
                "served_question_ids": sorted(source_pool.context7_served_questions),
                "max_total": max_context7_libraries,
                "elapsed_ms": context7_elapsed_ms,
                "body_boost": context7_body_boost_diagnostics,
                "body_boost_elapsed_ms": context7_body_boost_elapsed_ms,
                "disabled_reason": context7_disabled_reason,
            },
            "retrieval_wave": {
                "round_index": round_index,
                "active_question_count": len(active_states),
                "round_budget": round_budget,
                "global_fetch_cap": global_cap,
                "selection_limit": selection_limit,
                "selected_unique_url_count": len(selected),
                "question_attempted_url_count": len(attempted_urls),
                "question_unique_attempted_url_count": len(set(attempted_urls)),
                "discovered_source_count": sum(len(item.get("sources") or []) for item in discoveries.values()),
                "new_document_count": max(
                    0,
                    int(material_store_after.get("document_count") or 0)
                    - int(material_store_before.get("document_count") or 0),
                ),
                "discovery_elapsed_ms": discovery_elapsed_ms,
                "ranker_elapsed_ms": ranker_elapsed_ms,
                "fetch_elapsed_ms": fetch_elapsed_ms,
                "extraction_elapsed_ms": extraction_elapsed_ms,
                "wave_elapsed_ms": wave_elapsed_ms,
            },
        },
    }
    with audit_scope(stage="retrieval_round", round=round_index):
        write_stage_checkpoint(
            "retrieval_round",
            inputs={
                "queries_by_question": queries_by_id,
                "active_questions": [state.to_dict() for state in active_states],
            },
            outputs={
                "discoveries": discoveries,
                "selected_sources": selected,
                "results": results,
                "diagnostics": wave_output["diagnostics"],
            },
            transition_reason=f"retrieval round {round_index} completed",
        )
    return wave_output


def _source_selection_shadow(
    source_pool: GlobalSourcePool,
    states: list[ResearchQuestionState],
    *,
    max_urls: int,
) -> dict[str, Any]:
    if max_urls <= 0:
        return {"legacy_count": 0, "scoped_count": 0, "overlap_count": 0, "overlap_ratio": 1.0}
    legacy = source_pool.preview_select(states, max_urls=max_urls, mode="legacy")
    scoped = source_pool.preview_select(states, max_urls=max_urls, mode="scoped")
    legacy_urls = {item.url for item in legacy}
    scoped_urls = {item.url for item in scoped}
    overlap = legacy_urls & scoped_urls
    return {
        "legacy_count": len(legacy),
        "scoped_count": len(scoped),
        "overlap_count": len(overlap),
        "overlap_ratio": round(len(overlap) / max(1, len(legacy_urls | scoped_urls)), 3),
        "legacy_domains": sorted({urlparse(item.url).netloc for item in legacy}),
        "scoped_domains": sorted({urlparse(item.url).netloc for item in scoped}),
        "scoped_relationship_counts": {
            relation: sum(
                1 for item in scoped
                if any(item.relationship_by_question.get(state.question_id) == relation for state in states)
            )
            for relation in ("direct", "adjacent", "generic")
        },
    }


_LLM_BOOST_MODE_ENV = "RESEARCH_SOURCE_RANKER_LLM_BOOST"


def _llm_boost_enabled_mode() -> str:
    """"0" (default/off), "shadow" (compute + record diagnostics, never affects selection),
    or "live" (merged into relevance and actually changes what gets fetched). Any other value
    falls back to "0" -- an unrecognized value must never silently enable a live behavior
    change."""
    mode = str(os.getenv(_LLM_BOOST_MODE_ENV, "0")).strip().lower()
    return mode if mode in {"0", "shadow", "live"} else "0"


def _llm_boost_candidate_pool(
    source_pool: GlobalSourcePool, states: list[ResearchQuestionState]
) -> list[SourceCandidate]:
    """Candidates worth spending an LLM call on: not yet fetched, and not already `direct` for
    every active question -- a candidate the lexical scorer already admits needs no rescue call.
    This keeps the boost pass's cost proportional to what lexical scoring actually missed,
    instead of re-scoring every candidate every round."""
    return [
        candidate
        for candidate in source_pool.unfetched_candidates()
        if any(
            candidate.relationship_by_question.get(state.question_id) != "direct"
            for state in states
        )
    ]


def _llm_boost_source_candidates(
    candidates: list[SourceCandidate],
    states: list[ResearchQuestionState],
) -> dict[str, Any]:
    """Batch-score (candidate title+snippet) x (question criteria) relevance with a cheap LLM
    call per question, to catch semantically-relevant-but-lexically-invisible candidates the
    keyword-overlap scorer in source_pool.py can never admit -- e.g. an English "Recursion
    Limits" source against a Chinese "最大迭代次数" criterion (same meaning, zero lexical
    overlap; see project_source_pool_extract_key_terms_relevance_fix_20260801). Deliberately
    independent of QuestionBudget (backend/research/models.py): this never touches a question's
    max_rounds/timeout_seconds/max_model_calls, so a slow or failed boost call can only ever
    cost its own isolated timeout, never eat into the question's own retrieval budget. Returns
    {"scores": {url: {"question_id:criterion_id": relevance}}, "diagnostics": {...}} --
    diagnostics is populated even when scores is empty (e.g. all calls failed) so shadow-mode
    callers can still see why nothing was boosted.
    """
    timeout_seconds = float(os.getenv("RESEARCH_RANKER_TIMEOUT_SECONDS", "45"))
    max_candidates_per_call = max(1, int(os.getenv("RESEARCH_RANKER_MAX_CANDIDATES_PER_CALL", "40")))
    scores: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, Any] = {
        "candidate_count": len(candidates),
        "question_count": len(states),
        "calls": [],
    }
    if not candidates or not states:
        return {"scores": scores, "diagnostics": diagnostics}

    def score_question(state: ResearchQuestionState) -> tuple[str, dict[str, Any]]:
        criteria = [item for item in state.criteria if item.description]
        if not criteria:
            return state.question_id, {"status": "skipped", "reason": "no_criteria"}
        relevant_candidates = [
            candidate for candidate in candidates
            if candidate.relationship_by_question.get(state.question_id) != "direct"
        ][:max_candidates_per_call]
        if not relevant_candidates:
            return state.question_id, {"status": "skipped", "reason": "no_candidates"}
        candidate_by_id = {f"c{index:03d}": candidate for index, candidate in enumerate(relevant_candidates)}
        payload = {
            "task": (
                "Score how relevant each candidate search result is to each research criterion. "
                "Return JSON only."
            ),
            "question": state.question,
            "purpose": state.purpose,
            "criteria": [
                {"criterion_id": item.criterion_id, "description": item.description}
                for item in criteria
            ],
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "title": candidate.title[:300],
                    "snippet": candidate.snippet[:500],
                }
                for candidate_id, candidate in candidate_by_id.items()
            ],
            "rules": [
                "Judge relevance by meaning, not by shared words -- a candidate in a different "
                "language from the question/criteria can still be highly relevant (e.g. an "
                "English source about 'recursion limits' is a direct match for a Chinese "
                "criterion asking about '最大迭代次数/循环终止机制').",
                "relevance is 0.0-1.0: 0.0 means unrelated, 1.0 means the candidate clearly and "
                "directly answers the criterion.",
                "Only include a score for a (candidate_id, criterion_id) pair when relevance > 0.1 "
                "-- omit pairs you judge irrelevant rather than returning a 0.0 row for every one.",
                "Judge from the title and snippet only; do not assume unstated content.",
            ],
            "schema": {"scores": [{"candidate_id": "c000", "criterion_id": "criterion_01", "relevance": 0.0}]},
        }
        result = _invoke_research_json(payload, timeout_seconds=timeout_seconds, role="ranker")
        if result.get("_failure_reason"):
            return state.question_id, {"status": "failed", "reason": result["_failure_reason"]}
        rows = [item for item in result.get("scores") or [] if isinstance(item, dict)]
        scored_count = 0
        for row in rows:
            candidate = candidate_by_id.get(str(row.get("candidate_id") or ""))
            criterion_id = str(row.get("criterion_id") or "")
            if candidate is None or not criterion_id:
                continue
            try:
                relevance = max(0.0, min(1.0, float(row.get("relevance"))))
            except (TypeError, ValueError):
                continue
            if relevance <= 0.1:
                continue
            key = f"{state.question_id}:{criterion_id}"
            scores.setdefault(candidate.url, {})[key] = max(scores.get(candidate.url, {}).get(key, 0.0), relevance)
            scored_count += 1
        return state.question_id, {
            "status": "scored",
            "candidate_count": len(relevant_candidates),
            "scored_pair_count": scored_count,
        }

    with ThreadPoolExecutor(max_workers=max(1, min(len(states), 6))) as executor:
        futures = {submit_with_audit_context(executor, score_question, state): state for state in states}
        for future in as_completed(futures):
            state = futures[future]
            try:
                question_id, call_diagnostics = future.result()
            except Exception as exc:  # pragma: no cover -- defensive, mirrors other wave-level guards
                question_id, call_diagnostics = state.question_id, {"status": "failed", "reason": str(exc)[:240]}
            diagnostics["calls"].append({"question_id": question_id, **call_diagnostics})

    return {"scores": scores, "diagnostics": diagnostics}


def _apply_context7_body_boost(
    source_pool: GlobalSourcePool,
    material_store: JobMaterialStore,
    active_states: list[ResearchQuestionState],
) -> dict[str, Any]:
    """Re-score fetched Context7 library candidates using their real fetched document body
    instead of the short library-level title+description used at discovery time.

    A context7 library candidate's `relevance_by_criterion` is set once at discovery
    (`_match()`, source_pool.py) from `docs_search`'s short library blurb -- before its real
    docs are fetched via `/api/v2/context`. Once fetched, the candidate bypasses
    `relevant_candidates()`'s eligibility gate (source_pool.py's `item.library_id or ...`
    check) but still competes for ranking position using that stale discovery-time score. A
    same-named-but-unrelated library can out-rank the real one on library-name lexical
    overlap alone even after the real one's body content unambiguously answers the question
    -- see project_sec02_context7_source_pool_scoring_pollution_20260819 for the production
    incident and the isolated real-model verification (7/8 win rate with body content vs
    2/5 with title-only, once the verification script's initial `docs[:2]` truncation --
    which cut off the two documents that actually answered the question -- was corrected to
    `docs[:5]`).

    Reuses `_llm_boost_source_candidates` unmodified (it only reads candidate.title/snippet,
    agnostic to what's in them) and `GlobalSourcePool.apply_llm_boost` unmodified (raise-only
    merge, never lowers a score the lexical matcher already found). The only new logic here
    is: select fetched-but-unscored-for-this-question library candidates, overwrite their
    `snippet` in place with a bounded join of their fetched document bodies, score, merge.
    """
    if os.getenv("RESEARCH_CONTEXT7_BODY_RESCORE_ENABLED", "1") != "1":
        return {"status": "disabled"}
    candidates_by_url: dict[str, SourceCandidate] = {}
    for candidate in source_pool.fetched_context7_candidates():
        pending_states = [
            state for state in active_states
            if state.question_id in candidate.discovered_by_question_ids
            and state.question_id not in candidate.body_scored_by_question
        ]
        if not pending_states:
            continue
        documents = material_store.documents([candidate])
        body_text = " ".join(
            str(doc.get("content") or "")[:500] for doc in documents[:5]
        ).strip()
        if not body_text:
            continue
        candidate.snippet = body_text[:1200]
        candidates_by_url[candidate.url] = candidate
    if not candidates_by_url:
        return {"status": "skipped", "reason": "no_candidates"}
    candidates = list(candidates_by_url.values())
    result = _llm_boost_source_candidates(candidates, active_states)
    boosted_count = source_pool.apply_llm_boost(result["scores"], active_states)
    for candidate in candidates:
        for state in active_states:
            if state.question_id in candidate.discovered_by_question_ids:
                candidate.body_scored_by_question.add(state.question_id)
    return {
        "status": "scored",
        "candidate_count": len(candidates),
        "boosted_pair_count": boosted_count,
        **result["diagnostics"],
    }


def _fetch_source_candidates(selected: list[SourceCandidate], material_store: JobMaterialStore) -> None:
    max_workers = max(1, min(len(selected), int(os.getenv("RESEARCH_GLOBAL_FETCH_PARALLELISM", "6"))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            submit_with_audit_context(executor, fetch_url_with_chain, candidate.url): candidate
            for candidate in selected
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - provider dependent
                result = {"document": None, "provider_attempts": [{
                    "capability": "url_fetch",
                    "provider": "global_fetch_queue",
                    "status": "error",
                    "reason": str(exc)[:240],
                    "degraded": True,
                    "url": candidate.url,
                }]}
            material_store.record_fetch(candidate, result)


def _select_context7_libraries_for_fetch(
    candidates: list[SourceCandidate],
    *,
    active_states: list[ResearchQuestionState],
    all_states: list[ResearchQuestionState],
    max_total: int,
    max_per_question: int,
    previously_selected_questions: set[str] | None = None,
) -> list[SourceCandidate]:
    """Pick which Context7 library candidates to fetch docs for this wave.

    Discovery runs concurrently across active questions (`ThreadPoolExecutor` in
    `_retrieve_question_wave`), so the candidate pool's insertion order is thread-finish
    order -- non-deterministic. A flat `max_total` cap on `context7_library_candidates()`
    (pool insertion order) therefore lets whichever question's discovery finishes first
    monopolize the slots: in production case-question-slot-fairness, sec_01_q1's six
    loop-engineering libraries filled all 6 slots before sec_03_q1's
    `/significant-gravitas/autogpt` (discovered only by sec_03_q1) was reached, so
    AutoGPT official docs were never fetched and sec_03_q1 got only off-topic
    cobusgreyling docs.

    Round-robin across `active_states` instead: each active question gets up to
    `max_per_question` slots before any gets a second (mirroring
    `_round_robin_hard_criteria` in source_pool.py). Candidates whose primary
    discovering question isn't in `active_states` (e.g. a prior-wave question that
    has since completed, or a hand-constructed candidate with empty
    `discovered_by_question_ids`) fall through to `orphans` and fill any remaining
    cap. Same `library_id` deduped (a library discovered by multiple questions is
    one candidate with many `discovered_by_question_ids`, fetched once with a
    merged query -- see the caller).

    `previously_selected_questions` carries the set of questions that already claimed
    a context7 fetch slot in a PRIOR wave of this same job. The round-robin orders
    questions that have NOT yet claimed a slot ahead of those that have, so a question
    starved in round 1 (because `max_total < len(active_states)` -- e.g. cap=6 with
    10 active questions, questions 7-10 got nothing) gets priority in round 2 before
    any round-1 question claims a second slot. Without this, round 2 would re-rank
    by plan order and the same front-of-plan questions would monopolize again (see
    job_20260805_011442: round 2's 6 slots went to the same 6 questions as round 1,
    sec_04_q3 stayed starved across both waves).
    """
    active_qids = {state.question_id for state in active_states}
    by_question: dict[str, list[SourceCandidate]] = {qid: [] for qid in active_qids}
    orphans: list[SourceCandidate] = []
    seen_libraries: set[str] = set()
    for candidate in candidates:
        if not candidate.library_id or candidate.library_id in seen_libraries:
            continue
        seen_libraries.add(candidate.library_id)
        primary = next(
            (qid for qid in sorted(candidate.discovered_by_question_ids) if qid in active_qids),
            None,
        )
        if primary is not None:
            by_question.setdefault(primary, []).append(candidate)
        else:
            orphans.append(candidate)

    # Order questions so those NOT yet served by context7 in a prior wave come first
    # (within the same `max_per_question` round they still interleave fairly via the
    # cursor loop below -- this only decides who gets the first slot when the cap
    # can't fit everyone). `previously_selected_questions` is None in the helper's
    # unit tests and in wave 1; both fall back to plain plan order.
    prior = previously_selected_questions or set()
    question_order = sorted(
        (state.question_id for state in active_states),
        key=lambda qid: (qid in prior, qid),
    )

    # E1: max_per_question >= max_total would let one question drain the whole cap
    # (re-introducing the original monopolization bug). Clamp.
    max_per_question = max(1, min(max_per_question, max_total))

    selected: list[SourceCandidate] = []
    taken_by_question: dict[str, int] = {qid: 0 for qid in question_order}
    # Per-question cursor so each round-robin pass resumes where the last left off,
    # not always from index 0 (fairness when one question's queue is shorter).
    cursor: dict[str, int] = {qid: 0 for qid in question_order}

    while len(selected) < max_total:
        progressed = False
        for qid in question_order:
            if len(selected) >= max_total:
                break
            if taken_by_question[qid] >= max_per_question:
                continue
            queue = by_question.get(qid) or []
            # Advance cursor past already-selected candidates.
            while cursor[qid] < len(queue) and queue[cursor[qid]] in selected:
                cursor[qid] += 1
            if cursor[qid] >= len(queue):
                continue
            chosen = queue[cursor[qid]]
            cursor[qid] += 1
            taken_by_question[qid] += 1
            selected.append(chosen)
            progressed = True
        if not progressed:
            break

    # Fill remaining cap from orphans (candidates whose primary discovering question
    # isn't in active_states -- e.g. a prior-wave question).
    for candidate in orphans:
        if len(selected) >= max_total:
            break
        if candidate in selected:
            continue
        selected.append(candidate)

    return selected


def _extract_question_from_material_store(
    *,
    job_id: str,
    user_request: str,
    strategy: str,
    depth: str,
    state: ResearchQuestionState,
    queries: list[str],
    round_index: int,
    candidates: list[SourceCandidate],
    documents: list[dict[str, Any]],
    provider_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not documents:
        return {
            "sources": [item.to_dict() for item in candidates],
            "materials": [],
            "selected_chunks": [],
            "evidence_map": [],
            "provider_attempts": provider_attempts,
            "attempted_urls": [item.url for item in candidates],
            "_diagnostics": {"chargeable_elapsed_seconds": 0.0, "model_call_count": 0},
        }
    payload = _question_retrieval_payload([state], {state.question_id: queries})
    payload["documents"] = documents
    payload["source_candidates"] = [item.to_dict() for item in candidates]
    payload["provider_attempts"] = provider_attempts
    task = "\n".join([
        "source_mode=web 问题级共享材料抽证。",
        "retrieval_mode=material_extract",
        f"用户需求：{user_request}",
        f"topic={job_id}_{state.question_id}_r{round_index}_material_extract",
        f"research_strategy={strategy}",
        f"depth={depth}",
        f"query_limit={max(1, len(queries))}",
        f"fetch_url_limit={max(1, len(documents))}",
        "只使用 payload 中已经抓取的 documents；不得搜索或再次抓取 URL。",
        "<<<RESEARCH_PAYLOAD_JSON",
        json.dumps(payload, ensure_ascii=False),
        "RESEARCH_PAYLOAD_JSON>>>",
    ])
    started = time.perf_counter()
    result = parse_tool_json(web_worker_tool.invoke({"task_or_url": task}))
    diagnostics = dict(result.get("_diagnostics") or {})
    diagnostics["chargeable_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    extraction = diagnostics.get("section_evidence_extraction") or {}
    section_rows = [item for item in extraction.get("sections") or [] if isinstance(item, dict)]
    estimated_calls = len(section_rows) + sum(int(item.get("question_retry_count") or 0) for item in section_rows)
    diagnostics["model_call_count"] = max(1, int(diagnostics.get("model_call_count") or estimated_calls or 0))
    result["_diagnostics"] = diagnostics
    result["attempted_urls"] = [str(item.get("url") or "") for item in documents if str(item.get("url") or "")]
    exhausted_rows = [item for item in section_rows if item.get("retry_exhausted")]
    if exhausted_rows:
        checkpoint_path = _save_question_loop_checkpoint(
            job_id,
            f"evidence_extraction_failed_{state.question_id}_round_{round_index}",
            {
                "state": state.to_dict(),
                "queries": list(queries),
                "round_index": round_index,
                "source_candidates": [item.to_dict() for item in candidates],
                "documents": documents,
                "provider_attempts": provider_attempts,
                "partial_result": result,
                "exhausted_sections": exhausted_rows,
            },
        )
        raise EvidenceExtractionExhaustedError(
            f"证据抽取重试已耗尽：{state.question_id}",
            details={
                "question_id": state.question_id,
                "section_id": state.section_id,
                "round_index": round_index,
                "checkpoint_path": checkpoint_path,
                "failed_question_ids": list(dict.fromkeys(
                    str(question_id)
                    for row in exhausted_rows
                    for question_id in [*(row.get("failed_question_ids") or []), *(row.get("empty_question_ids") or [])]
                    if str(question_id)
                )),
            },
        )
    return result


def _discover_question_sources(
    *,
    job_id: str,
    user_request: str,
    strategy: str,
    depth: str,
    state: ResearchQuestionState,
    queries: list[str],
    round_index: int,
) -> dict[str, Any]:
    payload = _question_retrieval_payload([state], {state.question_id: queries})
    task = "\n".join([
        "source_mode=web 问题级 Deep Research 候选发现。",
        "retrieval_mode=discovery_only",
        f"用户需求：{user_request}",
        f"topic={job_id}_{state.question_id}_r{round_index}_discovery",
        f"research_strategy={strategy}",
        f"depth={depth}",
        f"query_limit={max(1, len(queries))}",
        "required_capabilities=broad_discovery,docs_search",
        "只发现候选来源，不抓取正文。",
        "<<<RESEARCH_PAYLOAD_JSON",
        json.dumps(payload, ensure_ascii=False),
        "RESEARCH_PAYLOAD_JSON>>>",
    ])
    return parse_tool_json(web_worker_tool.invoke({"task_or_url": task}))


def _fetch_source_batch(
    *,
    job_id: str,
    user_request: str,
    strategy: str,
    depth: str,
    all_states: list[ResearchQuestionState],
    active_queries: dict[str, list[str]],
    selected: list[SourceCandidate],
    round_index: int,
) -> dict[str, Any]:
    queries = {
        state.question_id: active_queries.get(state.question_id)
        or [state.criteria[0].description if state.criteria else state.question]
        for state in all_states
    }
    payload = _question_retrieval_payload(all_states, queries)
    task = "\n".join([
        "source_mode=web 任务级 Deep Research 全局抓取。",
        "retrieval_mode=explicit_fetch",
        f"用户需求：{user_request}",
        f"topic={job_id}_global_r{round_index}_fetch_{uuid.uuid4().hex[:6]}",
        f"research_strategy={strategy}",
        f"depth={depth}",
        f"query_limit={max(1, len(all_states))}",
        f"fetch_url_limit={len(selected)}",
        "required_capabilities=url_fetch",
        "只抓取下列全局排序后的 URL；同一正文同时服务所有相关问题：",
        *[item.url for item in selected],
        "<<<RESEARCH_PAYLOAD_JSON",
        json.dumps(payload, ensure_ascii=False),
        "RESEARCH_PAYLOAD_JSON>>>",
    ])
    return parse_tool_json(web_worker_tool.invoke({"task_or_url": task}))


def _question_retrieval_payload(
    states: list[ResearchQuestionState],
    queries_by_id: dict[str, list[str]],
) -> dict[str, Any]:
    sections: dict[str, dict[str, Any]] = {}
    query_rows: list[dict[str, Any]] = []
    for state in states:
        section = sections.setdefault(state.section_id, {
            "id": state.section_id,
            "title": state.section_title,
            "preferred_source_types": [],
            "source_requirements": [],
            "questions": [],
        })
        section["preferred_source_types"] = list(dict.fromkeys([
            *section["preferred_source_types"], *state.preferred_source_types,
        ]))
        section["source_requirements"].extend(state.source_requirements)
        section["questions"].append({
            "id": state.question_id,
            "text": state.question,
            "purpose": state.purpose,
            "topic_anchor": state.topic_anchor,
            "question_kind": state.question_kind,
            "comparison_axis": state.comparison_axis,
            "comparison_subjects": list(state.comparison_subjects),
            "success_criteria": list(state.success_criteria),
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "description": criterion.description,
                    "query_hint": criterion.query_hint,
                    "level": criterion.level,
                    "status": criterion.status,
                }
                for criterion in state.criteria
            ],
            "comparison_cells": [
                {
                    "cell_id": cell.cell_id,
                    "subject": cell.subject,
                    "axis": cell.axis,
                    "status": cell.status,
                    "criteria": [
                        {
                            "criterion_id": criterion.criterion_id,
                            "description": criterion.description,
                            "query_hint": criterion.query_hint,
                            "level": criterion.level,
                            "status": criterion.status,
                        }
                        for criterion in cell.criteria
                    ],
                }
                for cell in state.comparison_cells
            ],
            "active_queries": list(queries_by_id.get(state.question_id) or []),
        })
        query = " ".join(queries_by_id.get(state.question_id) or [state.question])[:160]
        query_rows.append({
            "section_id": state.section_id,
            "section_title": state.section_title,
            "question_id": state.question_id,
            "question": state.question,
            "query": query,
            "purpose": state.purpose,
            "topic_anchor": state.topic_anchor,
            "question_kind": state.question_kind,
            "comparison_axis": state.comparison_axis,
            "comparison_subjects": list(state.comparison_subjects),
            "answer_contract_level": state.answer_contract_level,
            "success_criteria": list(state.success_criteria),
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "description": criterion.description,
                    "query_hint": criterion.query_hint,
                    "level": criterion.level,
                    "status": criterion.status,
                }
                for criterion in state.criteria
            ],
            "comparison_cells": [
                {
                    "cell_id": cell.cell_id,
                    "subject": cell.subject,
                    "axis": cell.axis,
                    "status": cell.status,
                    "criteria": [
                        {
                            "criterion_id": criterion.criterion_id,
                            "description": criterion.description,
                            "query_hint": criterion.query_hint,
                            "level": criterion.level,
                            "status": criterion.status,
                        }
                        for criterion in cell.criteria
                    ],
                }
                for cell in state.comparison_cells
            ],
            "active_queries": list(queries_by_id.get(state.question_id) or []),
            "preferred_source_types": list(state.preferred_source_types),
            "source_requirements": state.source_requirements,
        })
    return {
        "research_plan": {"sections": list(sections.values())},
        "query_plan": {"query_limit": len(query_rows), "queries": query_rows},
        "required_titles": [item["title"] for item in sections.values()],
    }


def _question_round_fetch_limit(state: ResearchQuestionState, round_index: int) -> int:
    remaining_materials = max(0, state.budget.max_materials - len(state.materials))
    remaining_urls = max(0, state.budget.max_fetch_urls - len(state.attempted_fetch_urls))
    requested = state.budget.initial_fetch_limit if round_index == 1 else state.budget.followup_fetch_limit
    return max(0, min(requested, remaining_materials, remaining_urls))


def _route_batch_retrieval(
    retrieval: dict[str, Any],
    state: ResearchQuestionState,
    selected: list[SourceCandidate],
) -> dict[str, Any]:
    canonical_by_url = {normalize_url(item.url): item.source_type for item in selected}
    chunks = [
        {
            **item,
            "source_type": canonical_by_url.get(normalize_url(str(item.get("source_url") or "")), item.get("source_type")),
        }
        for item in retrieval.get("selected_chunks") or []
        if str(item.get("question_id") or "") == state.question_id
    ]
    chunk_ids = {str(item.get("id") or "") for item in chunks}
    evidence = [
        {
            **item,
            "source_type": canonical_by_url.get(normalize_url(str(item.get("source_url") or "")), item.get("source_type")),
        }
        for item in (retrieval.get("evidence_map") or retrieval.get("evidence_pack") or [])
        if str(item.get("question_id") or "") == state.question_id
        or str(item.get("chunk_id") or "") in chunk_ids
    ]
    urls = {str(item.get("source_url") or "") for item in chunks}
    normalized_urls = {normalize_url(url) for url in urls}
    materials = [{
        **item,
        "source_type": canonical_by_url.get(normalize_url(str(item.get("url") or "")), item.get("source_type")),
    } for item in retrieval.get("materials") or [] if normalize_url(str(item.get("url") or "")) in normalized_urls]
    selected_rows = [item.to_dict() for item in selected if normalize_url(item.url) in normalized_urls]
    retrieval_diagnostics = retrieval.get("_diagnostics") or {}
    return {
        "sources": selected_rows,
        "materials": materials,
        "selected_chunks": chunks,
        "evidence_map": evidence,
        "provider_attempts": retrieval.get("provider_attempts") or [],
        "attempted_urls": sorted(urls),
        "_diagnostics": {
            "model_call_count": int(retrieval_diagnostics.get("model_call_count") or 0),
            "chunk_ranker": _chunk_ranker_diagnostics_for_chunks(chunks),
        },
    }


def _merge_retrieval_results(items: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_ranker = _merge_chunk_ranker_diagnostics([
        (item.get("_diagnostics") or {}).get("chunk_ranker")
        or ((item.get("_diagnostics") or {}).get("material_pool") or {}).get("chunk_ranker")
        or {}
        for item in items
    ])
    return {
        "sources": _dedupe_result_rows([row for item in items for row in item.get("sources") or []], ("url", "source_url")),
        "materials": _dedupe_result_rows([row for item in items for row in item.get("materials") or []], ("url", "id")),
        "selected_chunks": _dedupe_result_rows([row for item in items for row in item.get("selected_chunks") or []], ("id",)),
        "evidence_map": [row for item in items for row in item.get("evidence_map") or []],
        "provider_attempts": [row for item in items for row in item.get("provider_attempts") or []],
        "attempted_urls": list(dict.fromkeys(row for item in items for row in item.get("attempted_urls") or [])),
        "_diagnostics": {
            "model_call_count": sum(int((item.get("_diagnostics") or {}).get("model_call_count") or 0) for item in items),
            "chunk_ranker": chunk_ranker,
        },
    }


def _chunk_ranker_diagnostics_for_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    per_question: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        question_id = str(chunk.get("question_id") or "")
        diagnostics = chunk.get("chunk_ranker_diagnostics")
        if question_id and isinstance(diagnostics, dict) and question_id not in per_question:
            per_question[question_id] = dict(diagnostics)
    totals = {
        "lexical_candidate_count": 0,
        "semantic_candidate_count": 0,
        "rrf_union_count": 0,
        "entity_rejected_count": 0,
        "strong_match_count": 0,
        "protected_reinsert_count": 0,
    }
    reasons: list[str] = []
    for diagnostics in per_question.values():
        for key in totals:
            totals[key] += int(diagnostics.get(key) or 0)
        reason = str(diagnostics.get("ranker_degraded_reason") or "")
        if reason and reason not in reasons:
            reasons.append(reason)
    modes = [str(item.get("ranker_mode") or "") for item in per_question.values() if item.get("ranker_mode")]
    return {
        "mode": modes[0] if modes and all(mode == modes[0] for mode in modes) else ("mixed" if modes else ""),
        "question_count": len(per_question),
        **totals,
        "ranker_degraded": any(bool(item.get("ranker_degraded")) for item in per_question.values()),
        "ranker_degraded_reasons": reasons,
        "questions": per_question,
    }


def _merge_chunk_ranker_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in items if isinstance(item, dict) and item]
    totals = {
        "question_count": 0,
        "lexical_candidate_count": 0,
        "semantic_candidate_count": 0,
        "rrf_union_count": 0,
        "entity_rejected_count": 0,
        "strong_match_count": 0,
        "protected_reinsert_count": 0,
    }
    reasons: list[str] = []
    questions: dict[str, dict[str, Any]] = {}
    modes: list[str] = []
    for item in valid:
        mode = str(item.get("mode") or "")
        if mode and mode not in modes:
            modes.append(mode)
        for key in totals:
            totals[key] += int(item.get(key) or 0)
        for reason in item.get("ranker_degraded_reasons") or []:
            value = str(reason or "")
            if value and value not in reasons:
                reasons.append(value)
        for question_id, diagnostics in (item.get("questions") or {}).items():
            if isinstance(diagnostics, dict):
                questions[str(question_id)] = dict(diagnostics)
    return {
        "mode": modes[0] if len(modes) == 1 else ("mixed" if modes else ""),
        **totals,
        "ranker_degraded": any(bool(item.get("ranker_degraded")) for item in valid),
        "ranker_degraded_reasons": reasons,
        "questions": questions,
    }


def _dedupe_result_rows(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = next((str(item.get(name) or "") for name in keys if str(item.get(name) or "")), "")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _classify_model_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message:
        return "timeout"
    if "429" in message or "rate limit" in message or "rate_limited" in message:
        return "rate_limited"
    return "api_error"


def _invoke_research_json(
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    temperature: float = 0,
    system_prompt: str | None = None,
    model_name: str | None = None,
    evidence: bool = True,
    role: str | None = None,
    allow_provider_fallback: bool = False,
    fallback_stage: str = "research",
    force_fallback_provider: bool = False,
    skip_preferred_provider: bool = False,
    parent_call_id: str | None = None,
) -> dict[str, Any]:
    """Call the research model and return parsed JSON.

    On any failure this returns ``{"_failure_reason": "..."}`` instead of a
    bare ``{}`` so callers that care (currently the section writer) can tell
    a timeout apart from an empty response or a JSON parse failure, instead
    of every failure mode collapsing into an indistinguishable empty dict.
    Existing callers that only read specific keys via ``.get(...)`` are
    unaffected: they still see those keys as absent.

    ``evidence`` selects which independently-configurable research role (see
    docs/modules/model-config-completion-v0.1.md 3.3/3.4) backs this call. It
    defaults to ``True`` because every call site except the section writer/editorial
    passes (``_research_writer_json_model``, which passes ``evidence=False``) is a
    classification/judging/extraction call.

    ``role``, when given, takes precedence over ``evidence`` -- lets a caller select
    any ``ResearchRole`` directly (e.g. ``"ranker"`` for the source-candidate LLM
    boost) without needing its own boolean flag.
    """
    with payload_audit_scope(payload, fallback_stage=fallback_stage):
        try:
            response = invoke_research_chat(
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                system_prompt=system_prompt,
                model_name=model_name,
                evidence=evidence,
                role=role,
                allow_provider_fallback=allow_provider_fallback,
                force_fallback_provider=force_fallback_provider,
                skip_preferred_provider=skip_preferred_provider,
                fallback_stage=fallback_stage,
                provider_fallback_reason="invalid_json_or_empty_response" if force_fallback_provider else "",
                parent_call_id=parent_call_id,
                call_kind="fallback" if force_fallback_provider else "model",
            )
        except Exception as exc:
            failure = {"_failure_reason": _classify_model_failure(exc)}
            if force_fallback_provider or isinstance(exc, ProviderFallbackExhaustedError):
                failure["_provider_fallback_attempted"] = True
            return failure
    response_provider = str((getattr(response, "response_metadata", None) or {}).get("harness_provider") or "primary")
    call_id = str((getattr(response, "response_metadata", None) or {}).get("harness_call_id") or "") or None
    content = str(getattr(response, "content", response) or "")
    if not content.strip():
        record_parse_result(call_id, status="empty_response")
        if response_provider == "preferred" and not skip_preferred_provider:
            return _invoke_research_json(
                payload,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                system_prompt=system_prompt,
                model_name=model_name,
                evidence=evidence,
                role=role,
                allow_provider_fallback=allow_provider_fallback,
                fallback_stage=fallback_stage,
                skip_preferred_provider=True,
                parent_call_id=call_id,
            )
        if allow_provider_fallback and not force_fallback_provider and response_provider != "mimo_fallback":
            return _invoke_research_json(
                payload,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                system_prompt=system_prompt,
                model_name=model_name,
                evidence=evidence,
                role=role,
                fallback_stage=fallback_stage,
                force_fallback_provider=True,
                parent_call_id=call_id,
            )
        failure = {"_failure_reason": "empty_response"}
        if force_fallback_provider or response_provider == "mimo_fallback":
            failure["_provider_fallback_attempted"] = True
        return failure
    parsed, parse_error = _parse_json_or_error(content)
    if not parsed:
        record_parse_result(call_id, status="parse_error", error=parse_error)
        # Re-asking the exact same question tends to reproduce the exact same formatting
        # mistake (e.g. an unescaped quote from a quoted term, or output cut off before the
        # closing brace) — so before giving up, make one repair call that hands the model its
        # own broken output plus the concrete JSONDecodeError and asks it to fix ONLY the JSON
        # syntax, not regenerate the content.
        parsed = _repair_json_response(
            content,
            parse_error,
            schema=payload.get("schema") if isinstance(payload.get("schema"), dict) else None,
            temperature=temperature,
            system_prompt=system_prompt,
            model_name=model_name,
            evidence=evidence,
            role=role,
            allow_provider_fallback=allow_provider_fallback,
            fallback_stage=fallback_stage,
            force_fallback_provider=force_fallback_provider or response_provider == "mimo_fallback",
            skip_preferred_provider=skip_preferred_provider,
            parent_call_id=call_id,
        )
        if not parsed:
            if response_provider == "preferred" and not skip_preferred_provider:
                return _invoke_research_json(
                    payload,
                    timeout_seconds=timeout_seconds,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    evidence=evidence,
                    role=role,
                    allow_provider_fallback=allow_provider_fallback,
                    fallback_stage=fallback_stage,
                    skip_preferred_provider=True,
                    parent_call_id=call_id,
                )
            if allow_provider_fallback and not force_fallback_provider and response_provider != "mimo_fallback":
                return _invoke_research_json(
                    payload,
                    timeout_seconds=timeout_seconds,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    evidence=evidence,
                    role=role,
                    fallback_stage=fallback_stage,
                    force_fallback_provider=True,
                    parent_call_id=call_id,
                )
            emit_event(
                "research.writer_json_parse_failed",
                "research_service",
                "Research model response could not be parsed as JSON, even after a repair attempt.",
                level="warning",
                data={
                    "raw_response_excerpt": content[:2000],
                    "parse_error": str(parse_error) if parse_error else "",
                    "task_shape": _research_json_task_shape(payload),
                },
            )
            failure = {"_failure_reason": "parse_failure"}
            if force_fallback_provider or response_provider == "mimo_fallback":
                failure["_provider_fallback_attempted"] = True
            return failure
    record_parse_result(call_id, parsed_result=parsed, status="success")
    return parsed


def _research_json_task_shape(payload: dict[str, Any]) -> str:
    task = str(payload.get("task") or "").lower()
    if "executive summary" in task or "editorial fields" in task:
        return "editorial"
    if "analytical report section" in task:
        return "section_writer"
    return "other"


def _repair_json_response(
    raw_content: str,
    error: JSONDecodeError | None,
    *,
    schema: dict[str, Any] | None = None,
    temperature: float,
    system_prompt: str | None,
    model_name: str | None,
    evidence: bool = True,
    role: str | None = None,
    allow_provider_fallback: bool = False,
    fallback_stage: str = "research",
    force_fallback_provider: bool = False,
    skip_preferred_provider: bool = False,
    parent_call_id: str | None = None,
) -> dict[str, Any]:
    """Ask the model to turn its own malformed/non-JSON output into valid JSON, instead of
    blindly re-asking the original question — mirrors the existing
    `_repair_draft_json`/`_parse_draft_json` pattern used by the legacy pipeline's draft
    synthesis, adapted for the question_loop writer/model call path.

    Two distinct failure shapes land here, and the rules below address both: (1) the response
    is JSON-like but has a syntax bug (unescaped quote, missing bracket, truncated ending), or
    (2) the response has NO JSON structure at all — the model ignored "return JSON only" and
    just wrote plain prose/Markdown directly (real-world example: `json_error` was consistently
    "Expecting value: line 1 column 1", i.e. the raw output didn't start with `{` at all).
    Naively telling the model to "fix JSON syntax" doesn't help case (2) since there's no syntax
    to fix — it needs to be told to wrap the prose into the schema instead.
    """
    repair_payload = {
        "task": "将一段模型输出转换成合法 JSON，使其匹配给定的 schema。只返回修复后的 JSON，不要添加解释或代码块围栏。",
        "rules": [
            "如果 raw_output 已经是 JSON 但存在语法问题（如未转义的引号、缺失的括号、多余的逗号、被截断的结尾），只修复语法，不改变字段内容含义；若某个字符串在中途被截断，请在保留已有内容的基础上为该字符串和外层结构补上合法收尾。",
            "如果 raw_output 完全不是 JSON（例如忽略了指令、直接输出了纯文本或 Markdown 正文），不要丢弃这些内容：将 raw_output 原文一字不改地放入 schema 中最合适的正文字段（通常是唯一的长文本字段），再按 schema 补全其余字段（找不到对应内容时留空字符串或空数组）。",
            "不得新增事实、不得改写正文内容含义，只做格式转换。",
        ],
        "json_error": str(error) if error else "invalid or unparsable JSON",
        "raw_output": raw_content[:24000],
    }
    if schema:
        repair_payload["schema"] = schema
    try:
        response = invoke_research_chat(
            [{"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)}],
            timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
            temperature=temperature,
            system_prompt=system_prompt,
            model_name=model_name,
            evidence=evidence,
            role=role,
            allow_provider_fallback=allow_provider_fallback,
            force_fallback_provider=force_fallback_provider,
            skip_preferred_provider=skip_preferred_provider,
            fallback_stage=fallback_stage,
            provider_fallback_reason="json_repair",
            parent_call_id=parent_call_id,
            call_kind="json_repair",
        )
    except Exception:
        return {}
    repaired_text = str(getattr(response, "content", response) or "")
    repaired = _json_from_text(repaired_text)
    repair_call_id = str((getattr(response, "response_metadata", None) or {}).get("harness_call_id") or "") or None
    record_parse_result(
        repair_call_id,
        parsed_result=repaired or None,
        status="success" if repaired else "parse_error",
    )
    return repaired


def _research_json_model(payload: dict[str, Any]) -> dict[str, Any]:
    return _invoke_research_json(
        payload,
        timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
    )


def _research_finding_json_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Finding synthesis model with the configured provider fallback enabled.

    Evidence admission is already complete at this point, so an unavailable primary model should
    not silently collapse a well-populated question into the old three-row deterministic fallback.
    Reuse the same evidence-role model and credentials, but allow the existing MiMo fallback path
    before deterministic synthesis is considered.
    """
    return _invoke_research_json(
        payload,
        timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
        evidence=True,
        allow_provider_fallback=True,
        fallback_stage="finding_synthesis",
    )


def _research_coverage_json_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Coverage judge with provider fallback for transient or malformed responses."""
    return _invoke_research_json(
        payload,
        timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
        evidence=True,
        allow_provider_fallback=True,
        fallback_stage="coverage_judge",
    )


_QUERY_EXPANSION_SYSTEM_PROMPT = (
    "你是研究检索查询编辑器。你的唯一任务是把中性的研究问题改写成搜索引擎可用的短查询。"
    "查询是探索性探针，不是答案，也不进入研究结论。"
)


def _research_query_json_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate retrieval-only query probes with the existing ranker model role."""
    return _invoke_research_json(
        payload,
        timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
        system_prompt=_QUERY_EXPANSION_SYSTEM_PROMPT,
        role="ranker",
        allow_provider_fallback=True,
        fallback_stage="query_expansion",
    )


_WRITER_SYSTEM_PROMPT = (
    "你是资深行业分析师和技术编辑，负责把结构化的研究证据改写成连贯、可读的分析性报告正文。"
    "撰写时要有清晰的段落论点、自然的论证递进和恰当的过渡，避免逐条罗列孤立事实；"
    "同时必须严格遵守 payload 中的 rules 和引用要求，不能编造材料之外的内容。"
    "写作前必须分别核对证据主体、当前问题或比较维度、以及来源角色：主体不同不能互换，"
    "相邻指标不能替代目标指标，厂商或项目方自评必须明确归因且不能冒充独立评测。"
)


_COVERAGE_CONTRACT_VERSION = "coverage-quote-binding/v1"


def _research_writer_json_model(payload: dict[str, Any]) -> dict[str, Any]:
    """JsonModel used only for section writing / editorial passes.

    Distinct from ``_research_json_model`` (used for evidence extraction,
    judging, and entailment, which must stay deterministic at temperature=0)
    so writing-quality tuning never destabilizes the deterministic checks
    that other tests and the quality gate rely on.
    """
    return _invoke_research_json(
        payload,
        timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
        temperature=float(os.getenv("RESEARCH_WRITER_TEMPERATURE", "0.3")),
        system_prompt=_WRITER_SYSTEM_PROMPT,
        model_name=os.getenv("RESEARCH_WRITER_MODEL_NAME") or None,
        evidence=False,
        allow_provider_fallback=True,
        fallback_stage="writer",
    )


def _coverage_evidence_digest(state: ResearchQuestionState) -> dict[str, Any]:
    """Select compact, criterion-balanced evidence for the semantic coverage judge.

    Evidence extraction may attach up to two concrete ``criterion_ids``. Historical rows and
    deterministic fallbacks do not have that field, so they receive a fresh lexical attribution
    instead of reusing the criterion ledger (which is itself derived from prior coverage output).
    The synthetic umbrella
    criterion (``实质回答研究问题``) deliberately receives no private evidence bucket: its status
    must be derived from the concrete dimensions and the question-level answer, not compete with
    those dimensions for a fixed input window.
    """
    concrete_criteria = [
        criterion
        for criterion in state.criteria
        if not criterion.description.startswith("实质回答研究问题")
    ]
    valid_criterion_ids = {criterion.criterion_id for criterion in concrete_criteria}
    criterion_term_sets = {
        criterion.criterion_id: _coverage_criterion_terms(criterion)
        for criterion in concrete_criteria
    }
    term_frequency: dict[str, int] = {}
    for terms in criterion_term_sets.values():
        for term in terms:
            term_frequency[term] = term_frequency.get(term, 0) + 1
    shared_criterion_terms = {
        term
        for term, count in term_frequency.items()
        if len(concrete_criteria) > 1 and count == len(concrete_criteria)
    }
    ignored_criterion_terms = shared_criterion_terms | {
        str(term).lower()
        for term in extract_key_terms(state.topic_anchor, top_k=8)
        if str(term)
    }
    rows: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(state.accepted_evidence):
        if not isinstance(item, dict):
            continue
        evidence_id = str(
            item.get("id")
            or item.get("_coverage_evidence_id")
            or f"{state.question_id}:coverage_ev_{index + 1:03d}"
        )
        item["_coverage_evidence_id"] = evidence_id
        rows.append((evidence_id, item))

    assigned: dict[str, dict[str, float]] = {evidence_id: {} for evidence_id, _item in rows}
    attribution_modes: dict[str, str] = {}
    criterion_by_id = {criterion.criterion_id: criterion for criterion in concrete_criteria}
    for evidence_id, item in rows:
        for value in item.get("criterion_ids") or []:
            criterion_id = str(value)
            if criterion_id in valid_criterion_ids:
                # The extractor's explicit assignment determines membership, while the existing
                # deterministic relevance score orders representatives inside that group. Giving
                # every explicit row a flat 1.0 let a high-authority but generic architecture
                # claim outrank an actual npm/npx command inside an installation criterion.
                assigned[evidence_id][criterion_id] = _coverage_criterion_relevance(
                    item,
                    criterion_by_id[criterion_id],
                    ignored_terms=ignored_criterion_terms,
                )
                attribution_modes[evidence_id] = "explicit"

    for evidence_id, item in rows:
        if assigned[evidence_id] or not concrete_criteria:
            continue
        if str(item.get("criterion_binding_status") or "") in {
            "incompatible_subject",
            "unverified_subject",
        }:
            continue
        scored = sorted(
            (
                (
                    _coverage_criterion_relevance(
                        item,
                        criterion,
                        ignored_terms=ignored_criterion_terms,
                    ),
                    criterion.criterion_id,
                )
                for criterion in concrete_criteria
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < 0.15:
            continue
        assigned[evidence_id][scored[0][1]] = scored[0][0]
        attribution_modes[evidence_id] = "lexical_fallback"
        if len(scored) > 1 and scored[1][0] >= max(0.16, scored[0][0] * 0.8):
            assigned[evidence_id][scored[1][1]] = scored[1][0]

    selected_ids: set[str] = set()
    criterion_groups: list[dict[str, Any]] = []
    for criterion in concrete_criteria:
        candidates = [
            (evidence_id, item)
            for evidence_id, item in rows
            if criterion.criterion_id in assigned[evidence_id]
        ]
        directly_relevant_candidates = [
            row
            for row in candidates
            if assigned[row[0]][criterion.criterion_id] >= 0.1
        ]
        if len(directly_relevant_candidates) >= 2:
            candidates = directly_relevant_candidates
        representatives = _coverage_representative_evidence(
            candidates,
            limit=3,
            relevance_scores={
                evidence_id: assigned[evidence_id][criterion.criterion_id]
                for evidence_id, _item in candidates
            },
            prefer_source_diversity=False,
        )
        evidence_ids = [evidence_id for evidence_id, _item in representatives]
        selected_ids.update(evidence_ids)
        criterion_groups.append({
            "criterion_id": criterion.criterion_id,
            "description": criterion.description,
            "level": criterion.level,
            "evidence_ids": evidence_ids,
        })

    unassigned_candidates = [
        (evidence_id, item)
        for evidence_id, item in rows
        if not assigned[evidence_id]
    ]
    unassigned = _coverage_representative_evidence(unassigned_candidates, limit=4)
    unassigned_ids = [evidence_id for evidence_id, _item in unassigned]
    selected_ids.update(unassigned_ids)

    if not selected_ids:
        fallback = _coverage_representative_evidence(rows, limit=12)
        selected_ids.update(evidence_id for evidence_id, _item in fallback)

    evidence_registry = []
    evidence_attribution = []
    for evidence_id, item in rows:
        if evidence_id not in selected_ids:
            continue
        criterion_ids = sorted(assigned[evidence_id])
        evidence_row = {
            "id": evidence_id,
            "claim": item.get("atomic_claim") or item.get("claim"),
            "quote": (
                item.get("cleaned_quote")
                or item.get("raw_quote")
                or item.get("evidence_text")
            ),
            "criterion_ids": criterion_ids,
            "source_type": item.get("source_type"),
            "stance": item.get("stance"),
            "source_url": item.get("source_url"),
            "source_title": item.get("source_title"),
            "subject": item.get("subject"),
            "subject_relation": item.get("subject_relation"),
            "criterion_binding_status": item.get("criterion_binding_status"),
            "scope": item.get("scope"),
        }
        if evidence_semantic_arbitration_mode() == "live":
            evidence_row.update({
                "directness_score": item.get("directness_score"),
                "applicability_score": item.get("applicability_score"),
                "support_relation": evidence_support_relation(item),
                "target_fit": evidence_target_fit(item),
            })
        evidence_registry.append(evidence_row)
        evidence_attribution.append({
            "evidence_id": evidence_id,
            "criterion_ids": criterion_ids,
            "mode": attribution_modes.get(evidence_id, "unassigned"),
        })
    return {
        "evidence_registry": evidence_registry,
        "criterion_groups": criterion_groups,
        "unassigned_evidence_ids": unassigned_ids,
        "evidence_attribution": evidence_attribution,
        "input_evidence_count": len(rows),
        "selected_evidence_count": len(evidence_registry),
    }


def _coverage_criterion_terms(criterion: Any) -> set[str]:
    target = " ".join([
        str(getattr(criterion, "description", "") or ""),
        str(getattr(criterion, "query_hint", "") or ""),
    ])
    return {str(term).lower() for term in extract_key_terms(target, top_k=18) if str(term)}


def _coverage_criterion_relevance(
    item: dict[str, Any],
    criterion: Any,
    *,
    ignored_terms: set[str] | None = None,
) -> float:
    evidence_text = " ".join([
        str(item.get("atomic_claim") or item.get("claim") or ""),
        str(item.get("source_title") or ""),
    ])
    target_terms = _coverage_criterion_terms(criterion) - set(ignored_terms or set())
    evidence_terms = {str(term).lower() for term in extract_key_terms(evidence_text, top_k=24) if str(term)}
    operation_score = _coverage_operation_relevance(
        str(getattr(criterion, "description", "") or ""),
        evidence_text,
    )
    if not target_terms or not evidence_terms:
        return operation_score
    matched_target_terms = {
        target
        for target in target_terms
        if any(_coverage_terms_match(target, evidence) for evidence in evidence_terms)
    }
    lexical_score = len(matched_target_terms) / max(2, min(len(target_terms), len(evidence_terms)))
    return min(1.0, lexical_score + operation_score)


def _coverage_operation_relevance(criterion_text: str, evidence_text: str) -> float:
    """Bridge common bilingual task intents when coverage evidence is selected lexically.

    This is only a representative-selection feature. It does not admit evidence or mark a
    criterion complete; the semantic judge still makes that decision from the selected claims.
    """
    criterion_value = criterion_text.lower()
    evidence_value = evidence_text.lower()
    operation_signals = (
        (
            (
                "安装", "获取", "命令", "步骤", "启动", "运行", "使用", "install", "setup",
                "quickstart", "getting started", "start", "run", "web", "sdk",
            ),
            (
                "安装", "启动", "运行", "install", "setup", "quickstart", "npm ", "npx ",
                "pnpm ", "yarn ", "pip ", "docker ", "git clone", "源码安装", "一键安装",
                "python ", " run ", " start ",
            ),
        ),
        (
            ("配置", "参数", "环境变量", "模型", "工具", "存储", "config", "configuration"),
            (
                "配置", "参数", "环境变量", "config", "configuration", "api_key", "api key",
                "model", "tools", "storage", "yaml", "json", ".env",
            ),
        ),
        (
            ("示例", "例子", "代码", "脚本", "sample", "hello world"),
            ("示例", "例子", "example", "sample", "hello world", "```", "python ", "javascript"),
        ),
    )
    score = 0.0
    for criterion_markers, evidence_markers in operation_signals:
        if not any(marker in criterion_value for marker in criterion_markers):
            continue
        evidence_hits = sum(1 for marker in evidence_markers if marker in evidence_value)
        if evidence_hits:
            score += min(0.55, 0.2 + (0.15 * min(evidence_hits, 3)))
    return min(0.7, score)


def _coverage_terms_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left.isascii() or not right.isascii():
        return False
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _coverage_representative_evidence(
    rows: list[tuple[str, dict[str, Any]]],
    *,
    limit: int,
    relevance_scores: dict[str, float] | None = None,
    prefer_source_diversity: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    relevance_scores = relevance_scores or {}
    ordered = sorted(
        rows,
        key=lambda row: (
            float(relevance_scores.get(row[0], 0.0)),
            *_coverage_evidence_quality(row[1]),
        ),
        reverse=True,
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_clusters: set[str] = set()
    selected_sources: set[str] = set()

    def add(row: tuple[str, dict[str, Any]], *, require_new_source: bool) -> bool:
        evidence_id, item = row
        cluster = str(item.get("claim_cluster_id") or item.get("claim_fingerprint") or evidence_id)
        source_url = str(item.get("source_url") or "")
        if cluster in selected_clusters:
            return False
        if require_new_source and source_url and source_url in selected_sources:
            return False
        selected.append(row)
        selected_clusters.add(cluster)
        if source_url:
            selected_sources.add(source_url)
        return True

    diversity_passes = (True, False) if prefer_source_diversity else (False,)
    for require_new_source in diversity_passes:
        for row in ordered:
            add(row, require_new_source=require_new_source)
            if len(selected) >= limit:
                return selected
    return selected


def _coverage_evidence_quality(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        1.0 if item.get("grounded_exactly") else 0.0,
        float(item.get("entity_match_score") or 0.0),
        1.0 if is_direct_exact_evidence(item) else 0.5 if evidence_support_relation(item) == "inference_support" else 0.0,
        float(item.get("evidence_score") or 0.0),
        float(item.get("authority_score") or 0.0),
    )


def _judge_question_coverage(state: ResearchQuestionState) -> dict[str, Any]:
    evidence_digest = _coverage_evidence_digest(state)
    selection_diagnostics = {
        "coverage_contract_version": _COVERAGE_CONTRACT_VERSION,
        "input_evidence_count": evidence_digest["input_evidence_count"],
        "selected_evidence_count": evidence_digest["selected_evidence_count"],
        "criterion_evidence_groups": evidence_digest["criterion_groups"],
        "unassigned_evidence_ids": evidence_digest["unassigned_evidence_ids"],
        "evidence_attribution": evidence_digest["evidence_attribution"],
    }
    state.diagnostics["coverage_evidence_selection"] = selection_diagnostics
    payload = {
        "task": "Judge semantic coverage for one research question. Return JSON only.",
        "coverage_contract_version": _COVERAGE_CONTRACT_VERSION,
        "question": state.question,
        "purpose": state.purpose,
        "answer_contract_level": state.answer_contract_level,
        "success_criteria": state.success_criteria,
        "criterion_ledger": [{
            "criterion_id": item.criterion_id,
            "description": item.description,
            "level": item.level,
        } for item in state.criteria],
        "evidence": evidence_digest["evidence_registry"],
        "criterion_evidence_groups": evidence_digest["criterion_groups"],
        "unassigned_evidence_ids": evidence_digest["unassigned_evidence_ids"],
        "evidence_selection_diagnostics": {
            "input_evidence_count": selection_diagnostics["input_evidence_count"],
            "selected_evidence_count": selection_diagnostics["selected_evidence_count"],
        },
        "rules": [
            "Use only supplied evidence.",
            "Quote is the factual authority; claim is only a compact locator. When claim and Quote differ in scope, strength, subject, condition, or causality, judge only what Quote supports.",
            "Evidence is stored once in evidence and mapped to concrete criteria by criterion_evidence_groups. Read every concrete criterion group before judging answer_status. unassigned_evidence_ids may also support a criterion when its claim is relevant.",
            "Preserve each evidence item's subject and subject_relation. Evidence marked comparison, general, unknown, or incompatible_subject cannot establish a target-specific fact unless the research question itself is a comparison and the finding keeps the original subject explicit.",
            "A synthetic criterion whose description starts with 实质回答研究问题 has no private evidence group. Judge it only after evaluating the concrete criterion groups and the research question as a whole.",
            "Judge answer_status only from the research question: complete means the evidence gives a useful, coherent answer to the central request and covers its main named dimensions; it does not require exhaustive low-level mechanics. partial means a material part of the central request is still unanswered. uncovered means the evidence does not substantively answer it.",
            "Every missing_question_points item must name a material dimension that is explicit in the question. Do not copy a planner-added example, diagram, benchmark, parameter list, case count, or other enrichment into this field unless the question itself explicitly asks for it.",
            "A missing implementation detail, exact metric, exhaustive inventory, example, or historical datapoint should be disclosed as an enrichment gap or limitation rather than downgrading an otherwise useful complete answer, unless that precision is itself the question's central request.",
            "Assess soft/optional criteria separately as enrichment_gaps. They must never change answer_status.",
            "A synthetic criterion whose description starts with 实质回答研究问题 must use a status consistent with answer_status. When answer_contract_level=soft, this question remains useful enrichment and its status must not be treated as a whole-report completion gate.",
            "Judge each criterion_id from criterion_ledger independently.",
            "For every criterion_coverage row, return supporting_evidence_ids containing only Evidence IDs actually used for that criterion. The IDs must exist in evidence. Return an empty list when no supplied Evidence supports the criterion; do not copy the whole candidate group by default.",
            "Classify every criterion as covered, partial, or uncovered.",
            "covered means the criterion's core request is substantively answered by evidence; a missing narrow detail does not downgrade an otherwise answered core criterion.",
            "partial means a material part of the criterion's core request is still unanswered. Do not use partial merely to record a missing example, parameter enumeration, extra case, or quantitative detail.",
            "uncovered means the core request is absent, off-topic, or unsupported; do not call it partial merely because related evidence exists.",
            "A hard criterion must be covered (not partial) for sufficient=true.",
            "Soft and optional criteria may be reported as gaps but must not make sufficient=false by themselves.",
            "Do not turn source-count, link-count, visualization, or source-mix preferences into a hard content gap.",
            "For every uncovered criterion return one structured gap, not a prose query.",
            "Write missing_fact in the same language as the question and keep it under 70 characters.",
        ],
        "schema": {
            "answer_status": "complete|partial|uncovered",
            "answered_points": [],
            "missing_question_points": [],
            "enrichment_gaps": [{
                "criterion_id": "criterion_02",
                "missing_fact": "",
            }],
            "sufficient": False,
            "covered_aspects": [],
            "criterion_coverage": [{
                "criterion_id": "criterion_01",
                "status": "covered|partial|uncovered",
                "reason": "",
                "supporting_evidence_ids": ["ev_001"],
            }],
            "gaps": [{
                "criterion_id": "criterion_01",
                "gap_code": "missing_implementation_detail",
                "missing_fact": "",
                "preferred_source_types": [],
            }],
        },
    }
    if evidence_semantic_arbitration_mode() == "live":
        payload["rules"].insert(
            3,
            "Use support_relation and target_fit as the primary semantic facts. Only direct_support + exact_target may independently cover a target-specific hard factual criterion. inference_support may make it partial but never covered by itself. context_only and unrelated do not fill a criterion.",
        )
    with audit_scope(
        stage="coverage_judge",
        question_id=state.question_id,
        section_id=state.section_id,
        round=state.rounds_used,
    ):
        result = _research_coverage_json_model(payload)
    answer_status = str(result.get("answer_status") or "").strip().lower()
    if not isinstance(result.get("sufficient"), bool) and answer_status not in {"complete", "partial", "uncovered"}:
        return {}
    valid_criterion_ids = {item.criterion_id for item in state.criteria}
    valid_evidence_ids = {
        str(item.get("id") or "")
        for item in evidence_digest["evidence_registry"]
        if str(item.get("id") or "")
    }
    criterion_coverage = []
    binding_issues: list[dict[str, Any]] = []
    for item in result.get("criterion_coverage") or []:
        if not isinstance(item, dict):
            continue
        criterion_id = str(item.get("criterion_id") or "")
        status = str(item.get("status") or "").strip().lower()
        if criterion_id not in valid_criterion_ids or status not in {"covered", "partial", "uncovered"}:
            continue
        raw_supporting = item.get("supporting_evidence_ids")
        supplied_ids = (
            [str(value) for value in raw_supporting if str(value)]
            if isinstance(raw_supporting, list)
            else []
        )
        unknown_ids = sorted(set(supplied_ids) - valid_evidence_ids)
        supporting_evidence_ids = list(dict.fromkeys(
            evidence_id for evidence_id in supplied_ids if evidence_id in valid_evidence_ids
        ))
        if status == "uncovered" and supporting_evidence_ids:
            binding_issues.append({
                "criterion_id": criterion_id,
                "issue": "uncovered_with_supporting_evidence",
                "evidence_ids": supporting_evidence_ids,
            })
            supporting_evidence_ids = []
        if unknown_ids:
            binding_issues.append({
                "criterion_id": criterion_id,
                "issue": "unknown_supporting_evidence_ids",
                "evidence_ids": unknown_ids,
            })
        if status in {"covered", "partial"} and not supporting_evidence_ids:
            binding_issues.append({
                "criterion_id": criterion_id,
                "issue": "status_without_supporting_evidence",
                "status": status,
            })
        criterion_coverage.append({
            "criterion_id": criterion_id,
            "status": status,
            "reason": str(item.get("reason") or "").strip()[:160],
            "supporting_evidence_ids": supporting_evidence_ids,
        })
    synthetic_ids = {
        item.criterion_id
        for item in state.criteria
        if item.level == "hard" and item.description.startswith("实质回答研究问题")
    }
    gaps = []
    for index, item in enumerate(result.get("gaps") or []):
        if not isinstance(item, dict):
            continue
        missing_fact = str(item.get("missing_fact") or "").strip()[:100]
        if not missing_fact:
            continue
        gaps.append({
            "criterion_id": str(item.get("criterion_id") or f"criterion_{index + 1:02d}"),
            "gap_code": str(item.get("gap_code") or "semantic_gap")[:80],
            "missing_fact": missing_fact,
            "preferred_source_types": [
                normalize_source_type(value)
                for value in _coerce_string_list(item.get("preferred_source_types"))[:2]
            ],
        })
    enrichment_gaps = []
    for item in result.get("enrichment_gaps") or []:
        if not isinstance(item, dict):
            continue
        missing_fact = str(item.get("missing_fact") or "").strip()[:100]
        if not missing_fact:
            continue
        enrichment_gaps.append({
            "criterion_id": str(item.get("criterion_id") or ""),
            "missing_fact": missing_fact,
        })
    criterion_levels = {item.criterion_id: item.level for item in state.criteria}
    soft_gap_facts = [
        item["missing_fact"]
        for item in [*gaps, *enrichment_gaps]
        if criterion_levels.get(item.get("criterion_id")) in {"soft", "optional"}
    ]
    missing_question_points = _coerce_string_list(result.get("missing_question_points"))
    core_gaps = []
    for item in gaps:
        criterion_id = item["criterion_id"]
        level = criterion_levels.get(criterion_id, "hard")
        if level in {"soft", "optional"}:
            if not any(
                existing["criterion_id"] == criterion_id
                and _coverage_gap_text_matches(existing["missing_fact"], item["missing_fact"])
                for existing in enrichment_gaps
            ):
                enrichment_gaps.append({
                    "criterion_id": criterion_id,
                    "missing_fact": item["missing_fact"],
                })
            continue
        if criterion_id in synthetic_ids and any(
            _coverage_gap_text_matches(item["missing_fact"], fact) for fact in soft_gap_facts
        ):
            continue
        core_gaps.append(item)
    answered_points = _coerce_string_list(result.get("answered_points"))
    if answer_status in {"complete", "partial", "uncovered"}:
        expected_status = {"complete": "covered", "partial": "partial", "uncovered": "uncovered"}[answer_status]
        criterion_by_id = {item["criterion_id"]: item for item in criterion_coverage}
        for criterion_id in synthetic_ids:
            synthetic_row = criterion_by_id.get(criterion_id, {"criterion_id": criterion_id})
            synthetic_row.update({
                "status": expected_status,
                "reason": "与问题级 answer_status 保持一致",
                "supporting_evidence_ids": list(
                    synthetic_row.get("supporting_evidence_ids") or []
                ),
            })
            criterion_by_id[criterion_id] = synthetic_row
        criterion_coverage = list(criterion_by_id.values())
    binding_issue_keys = {
        (str(item.get("criterion_id") or ""), str(item.get("issue") or ""))
        for item in binding_issues
    }
    for row in criterion_coverage:
        criterion_id = str(row.get("criterion_id") or "")
        if (
            str(row.get("status") or "") in {"covered", "partial"}
            and not row.get("supporting_evidence_ids")
            and (criterion_id, "status_without_supporting_evidence") not in binding_issue_keys
        ):
            binding_issues.append({
                "criterion_id": criterion_id,
                "issue": "status_without_supporting_evidence",
                "status": str(row.get("status") or ""),
            })
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence_digest["evidence_registry"]
        if str(item.get("id") or "")
    }
    group_ids = {
        str(group.get("criterion_id") or ""): [str(value) for value in group.get("evidence_ids") or []]
        for group in evidence_digest["criterion_groups"]
    }
    criterion_levels = {item.criterion_id: item.level for item in state.criteria}
    semantic_downgrades: list[dict[str, Any]] = []
    for row in criterion_coverage if evidence_semantic_arbitration_mode() == "live" else []:
        criterion_id = str(row.get("criterion_id") or "")
        if (
            row.get("status") != "covered"
            or criterion_levels.get(criterion_id) != "hard"
            or criterion_id in synthetic_ids
        ):
            continue
        support_ids = (
            [str(value) for value in row.get("supporting_evidence_ids") or []]
            if "supporting_evidence_ids" in row
            else group_ids.get(criterion_id, [])
        )
        supporting = [evidence_by_id[evidence_id] for evidence_id in support_ids if evidence_id in evidence_by_id]
        if any(is_direct_exact_evidence(item) for item in supporting):
            continue
        row["status"] = "partial" if any(semantic_relation_allows_criterion(item) for item in supporting) else "uncovered"
        row["reason"] = "硬 Criterion 缺少 direct_support + exact_target 证据"
        semantic_downgrades.append({
            "criterion_id": criterion_id,
            "status": row["status"],
            "evidence_ids": support_ids,
        })
    if semantic_downgrades and answer_status == "complete":
        answer_status = "partial"
        for criterion_id in synthetic_ids:
            for row in criterion_coverage:
                if row.get("criterion_id") == criterion_id:
                    row["status"] = "partial"
                    row["reason"] = "具体硬 Criterion 缺少直接且目标精确的证据"
        missing_question_points = list(dict.fromkeys([
            *missing_question_points,
            *[f"{item['criterion_id']} 缺少直接目标证据" for item in semantic_downgrades],
        ]))
    normalized_sufficient = (
        answer_status == "complete"
        if answer_status in {"complete", "partial", "uncovered"}
        else bool(result.get("sufficient"))
        if isinstance(result.get("sufficient"), bool)
        else False
    )
    return {
        "coverage_contract_version": _COVERAGE_CONTRACT_VERSION,
        "sufficient": normalized_sufficient,
        "answer_status": answer_status,
        "answered_points": answered_points,
        "missing_question_points": missing_question_points,
        "enrichment_gaps": enrichment_gaps,
        "covered_aspects": _coerce_string_list(result.get("covered_aspects")),
        "weak_aspects": [item["missing_fact"] for item in core_gaps],
        "gaps": core_gaps,
        "criterion_coverage": criterion_coverage,
        "criterion_binding_issues": binding_issues,
        "semantic_relation_downgrades": semantic_downgrades,
        "suggested_queries": [],
    }


def _coverage_gap_text_matches(left: str, right: str) -> bool:
    ignored = {
        "缺少", "未提供", "没有", "尚未", "说明", "描述", "基本", "完整", "具体",
        "missing", "provide", "provided", "describe", "basic", "complete", "specific",
    }
    left_terms = {
        str(term).lower()
        for term in extract_key_terms(left, top_k=16)
        if str(term).lower() not in ignored
    }
    right_terms = {
        str(term).lower()
        for term in extract_key_terms(right, top_k=16)
        if str(term).lower() not in ignored
    }
    if not left_terms or not right_terms:
        return False
    matched = {
        term
        for term in left_terms
        if any(_coverage_terms_match(term, candidate) for candidate in right_terms)
    }
    return len(matched) / max(1, min(len(left_terms), len(right_terms))) >= 0.6


def _expand_initial_query_candidates(states: list[ResearchQuestionState]) -> dict[str, Any]:
    pending = [
        state
        for state in states
        if not any(str(criterion.query_hint or "").strip() for criterion in state.criteria)
    ]
    if not pending:
        return {"status": "not_needed", "queries_by_id": {}, "failure_reason": ""}
    result = _research_query_json_model({
        "task": "为每个研究问题生成两个互补、可直接执行的检索查询。只返回 JSON。",
        "questions": [
            {
                "question_id": state.question_id,
                "topic_anchor": state.topic_anchor,
                "question": state.question,
                "preferred_source_types": state.preferred_source_types,
            }
            for state in pending
        ],
        "rules": [
            "每个查询 4-12 个词、最多 100 个字符，去掉‘确定’‘有证据支持’‘证据缺口’等验收措辞。",
            "保留用户点名的主体和问题轴；技术主题优先使用官方文档常见的英文术语。",
            "可以加入标准名称、协议操作名、schema 或文档工件名作为候选检索词，但不得把它们写成已经成立的结论。",
            "不得加入优势、缺陷、因果、性能、成本、适用结论或未经用户给出的比较结论。",
            "第一个查询面向核心机制；第二个查询明确面向 specification、official documentation 或 reference。",
            "不同问题的查询必须体现各自问题轴，不能复用同一句通用模板。",
        ],
        "schema": {
            "questions": [{
                "question_id": "",
                "queries": ["core mechanism query", "primary documentation query"],
            }]
        },
    })
    failure_reason = str(result.get("_failure_reason") or "").strip()
    known_ids = {state.question_id for state in pending}
    queries_by_id: dict[str, list[str]] = {}
    forbidden_meta = ("有证据支持", "证据缺口", "确定", "证明")
    for item in result.get("questions") or []:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or "").strip()
        if question_id not in known_ids:
            continue
        queries = []
        for raw in item.get("queries") or []:
            query = compile_research_query(content=str(raw or "").strip(), limit=100)
            if not query or any(marker in query for marker in forbidden_meta) or query in queries:
                continue
            queries.append(query)
            if len(queries) >= 2:
                break
        if queries:
            queries_by_id[question_id] = queries
    status = "succeeded" if len(queries_by_id) == len(pending) else "partial" if queries_by_id else "degraded"
    if status == "succeeded":
        diagnostic_reason = ""
    else:
        diagnostic_reason = failure_reason or (
            "partial_query_expansion" if queries_by_id else "empty_or_invalid_queries"
        )
    return {
        "status": status,
        "queries_by_id": queries_by_id,
        "failure_reason": diagnostic_reason,
    }


def _question_initial_queries(
    query_plan: dict[str, Any],
    states: list[ResearchQuestionState],
    *,
    expanded_queries: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    # Planner queries remain diagnostic input; executable queries are compiled
    # from the criterion ledger so every first-round search has a coverage owner.
    expanded_queries = expanded_queries or {}
    output: dict[str, list[str]] = {}
    for state in states:
        if state.comparison_cells:
            output[state.question_id] = [
                cell.initial_query
                for cell in state.comparison_cells
                if cell.initial_query
            ]
            continue
        ordered = [item for item in state.criteria if item.level == "hard"]
        ordered.extend(item for item in state.criteria if item.level == "soft")
        ordered.extend(item for item in state.criteria if item.level == "optional")
        authored_hints = [str(criterion.query_hint or "").strip() for criterion in ordered if criterion.query_hint]
        if authored_hints:
            compiled = [
                compile_research_query(
                    topic_anchor=state.topic_anchor or state.question,
                    content=hint,
                    source_type=(state.preferred_source_types or [""])[0],
                )
                for hint in authored_hints[:2]
            ]
        else:
            compiled = [
                compile_research_query(content=query, limit=100)
                for query in expanded_queries.get(state.question_id, [])[:2]
            ]
            if not compiled and ordered and any(
                criterion.level_source != "question_dimensions_v2" for criterion in ordered[:2]
            ):
                compiled = [
                    compile_research_query(
                        topic_anchor=state.topic_anchor or state.question,
                        content=criterion.description,
                        source_type=(criterion.preferred_source_types or state.preferred_source_types or [""])[0],
                    )
                    for criterion in ordered[:2]
                ]
        output[state.question_id] = [item for item in compiled if item]
        if not output[state.question_id]:
            output[state.question_id] = [compile_research_query(
                topic_anchor=state.topic_anchor or state.question, content=state.question,
            )]
            for source_type in state.preferred_source_types[:2]:
                targeted = source_query(
                    state.topic_anchor or state.question,
                    source_type,
                    content=state.question,
                )
                if targeted not in output[state.question_id]:
                    output[state.question_id].append(targeted)
    return output


def _annotate_entailment_core_scope(
    entailment: dict[str, Any],
    report_blueprint: dict[str, Any],
    *,
    unsupported_blocks_quality_gate: bool = True,
) -> dict[str, Any]:
    """Classify unsupported claims by existing section criterion levels."""
    core_section_ids = {
        str(section.get("section_id") or "")
        for section in report_blueprint.get("sections") or []
        if bool((section.get("reportability") or {}).get("core_section"))
    }
    judgments = []
    blocking = []
    advisory = []
    noncore_unsupported = []
    core_unsupported = []
    for raw in entailment.get("judgments") or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        section_id = str(item.get("section_id") or "")
        claim_scope = str(item.get("claim_scope") or "").strip().lower()
        if claim_scope == "research_process_disclosure":
            research_process_limitation = True
            scope_classification_source = "model"
        elif claim_scope in {"domain_fact", "product_limitation", "unknown"}:
            research_process_limitation = False
            scope_classification_source = "model"
        else:
            research_process_limitation = _is_research_process_limitation_claim(
                str(item.get("claim") or item.get("text") or "")
            )
            scope_classification_source = "keyword_fallback"
        core_scope = (
            not research_process_limitation
            and (not section_id or section_id in {"editorial", "sec_00"} or section_id in core_section_ids)
        )
        item["core_scope"] = core_scope
        item["research_process_limitation"] = research_process_limitation
        item["scope_classification_source"] = scope_classification_source
        status = str(item.get("status") or "")
        if status == "unsupported" and core_scope:
            core_unsupported.append(item)
        item["quality_gate_blocking"] = (
            unsupported_blocks_quality_gate and status == "unsupported" and core_scope
        )
        if item["quality_gate_blocking"]:
            blocking.append(item)
        elif status in {"unsupported", "label_mismatch", "non_atomic"}:
            advisory.append(item)
            if status == "unsupported" and not core_scope:
                noncore_unsupported.append(item)
        judgments.append(item)
    return {
        **entailment,
        "judgments": judgments,
        "blocking_core_unsupported_count": len(blocking),
        "blocking_core_unsupported_claims": blocking[:12],
        "core_unsupported_count": len(core_unsupported),
        "core_unsupported_claims": core_unsupported[:12],
        "unsupported_marker_policy": (
            "blocking_core" if unsupported_blocks_quality_gate else "advisory_only"
        ),
        "advisory_nonblocking_count": len(advisory),
        "advisory_nonblocking_claims": advisory[:12],
        "noncore_unsupported_count": len(noncore_unsupported),
        "noncore_unsupported_claims": noncore_unsupported[:12],
    }


def _is_research_process_limitation_claim(claim: str) -> bool:
    """Recognize disclosures about this run's evidence limits, not external-world facts."""
    text = re.sub(r"\s+", "", str(claim or "")).lower()
    if not text:
        return False
    scope_markers = (
        "本次检索", "本次调研", "本次材料", "本次研究", "检索范围内", "现有材料",
        "当前材料", "公开材料", "公开资料", "目前公开", "现有公开", "所获证据",
        "thissearch", "thisresearch", "availablematerials", "publiclyavailable",
    )
    absence_markers = (
        "未发现", "未提供", "未检索到", "没有找到", "不足", "缺失", "有限",
        "didnotfind", "doesnotprovide", "didnotprovide", "insufficient", "missing", "limited",
    )
    return any(marker in text for marker in scope_markers) and any(
        marker in text for marker in absence_markers
    )


def _question_loop_coverage(states: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {state.get("question_id"): _question_answer_status(state) for state in states}
    core_states = [
        state for state in states
        if str(state.get("answer_contract_level") or "hard").lower() != "soft"
    ]
    advisory_states = [state for state in states if state not in core_states]
    complete = [state for state in core_states if statuses.get(state.get("question_id")) == "complete"]
    partial = [state for state in core_states if statuses.get(state.get("question_id")) == "partial"]
    uncovered = [state for state in core_states if statuses.get(state.get("question_id")) == "uncovered"]
    advisory_status_counts = {
        status: sum(statuses.get(state.get("question_id")) == status for state in advisory_states)
        for status in ("complete", "partial", "uncovered")
    }
    weak = [*partial, *uncovered]
    safety_blocked = [
        state for state in states
        if str((state.get("coverage") or {}).get("coverage_status") or "") == "blocked_by_safety"
    ]
    coverage_status_counts: dict[str, int] = {}
    for state in states:
        label = str((state.get("coverage") or {}).get("coverage_status") or "insufficient")
        coverage_status_counts[label] = coverage_status_counts.get(label, 0) + 1
    return {
        "ready_to_write": bool(states) and bool(complete or partial) and not safety_blocked,
        "round_index": max((int(state.get("rounds_used") or 0) for state in states), default=0),
        "max_rounds": max((int((state.get("budget") or {}).get("max_rounds") or 0) for state in states), default=0),
        "source_count": len({str(item.get("source_url") or "") for state in states for item in state.get("accepted_evidence") or []}),
        "document_count": sum(len(state.get("materials") or []) for state in states),
        "evidence_count": sum(len(state.get("accepted_evidence") or []) for state in states),
        "key_fact_count": 0,
        "evidence_pack_count": sum(len(state.get("accepted_evidence") or []) for state in states),
        "depth_ready": not weak and not safety_blocked,
        "covered_titles": list(dict.fromkeys(str(state.get("section_title") or "") for state in complete)),
        "partial_titles": list(dict.fromkeys(str(state.get("section_title") or "") for state in partial)),
        "missing_titles": list(dict.fromkeys(str(state.get("section_title") or "") for state in uncovered)),
        "fatal_gaps": [f"{state.get('question')}: {', '.join((state.get('coverage') or {}).get('weak_aspects') or [])}" for state in weak],
        "followup_queries": [query for state in weak for query in state.get("suggested_queries") or []][:8],
        "weak_sections": [{"section_id": state.get("section_id"), "title": state.get("section_title"), "reason": (state.get("coverage") or {}).get("reason"), "evidence_count": len(state.get("accepted_evidence") or [])} for state in weak],
        "coverage_status_counts": coverage_status_counts,
        "answer_status_counts": {
            "complete": len(complete),
            "partial": len(partial),
            "uncovered": len(uncovered),
        },
        "core_question_count": len(core_states),
        "advisory_question_count": len(advisory_states),
        "advisory_answer_status_counts": advisory_status_counts,
        "safety_blocked_count": len(safety_blocked),
        "reason": (
            (
                f"{len(complete)} 个核心研究问题完整覆盖，{len(partial)} 个部分覆盖，{len(uncovered)} 个未覆盖。"
                f"另有 {len(advisory_states)} 个补充问题（不影响整体完整性）。"
                if advisory_states
                else f"{len(complete)} 个研究问题完整覆盖，{len(partial)} 个部分覆盖，{len(uncovered)} 个未覆盖。"
            )
            + (f"另有 {len(safety_blocked)} 个问题存在事实安全阻断。" if safety_blocked else "")
        ),
    }


def _comparison_matrix_coverage(
    states: list[dict[str, Any]],
    research_plan: dict[str, Any],
) -> dict[str, Any]:
    if not _comparison_plan_enabled(research_plan):
        return {"enabled": False, "status": "not_applicable", "complete": True}

    cells_by_id: dict[str, dict[str, Any]] = {}
    for state in states:
        state_cells = state.get("comparison_cells") or (state.get("coverage") or {}).get("comparison_cells") or []
        for cell in state_cells:
            if not isinstance(cell, dict):
                continue
            cell_id = str(cell.get("cell_id") or cell.get("id") or "").strip()
            if cell_id:
                cells_by_id[cell_id] = dict(cell)

    required_cells: list[dict[str, Any]] = []
    for raw in research_plan.get("matrix_cells") or []:
        if not isinstance(raw, dict):
            continue
        cell_id = str(raw.get("id") or raw.get("cell_id") or "").strip()
        subject = str(raw.get("subject") or "").strip()
        axis = str(raw.get("axis") or raw.get("comparison_axis") or "").strip()
        if not cell_id or not subject or not axis:
            continue
        observed = cells_by_id.get(cell_id) or {}
        status = str(observed.get("status") or "uncovered").strip().lower()
        if status not in {"complete", "partial", "uncovered", "exhausted", "unavailable"}:
            status = "uncovered"
        criteria = [item for item in observed.get("criteria") or [] if isinstance(item, dict)]
        required_cells.append({
            "cell_id": cell_id,
            "subject": subject,
            "axis": axis,
            "status": status,
            "evidence_ids": [str(item) for item in observed.get("evidence_ids") or [] if str(item)],
            "criteria": criteria,
            "attempt_count": int(observed.get("attempt_count") or len(observed.get("attempts") or [])),
            "unavailable_reasons": [
                str(item) for item in observed.get("unavailable_reasons") or [] if str(item)
            ],
        })

    axes = [str(item).strip() for item in research_plan.get("comparison_axes") or [] if str(item).strip()]
    if not axes:
        axes = list(dict.fromkeys(cell["axis"] for cell in required_cells))
    axis_rows = []
    for axis in axes:
        axis_cells = [cell for cell in required_cells if cell["axis"] == axis]
        axis_complete = bool(axis_cells) and all(cell["status"] == "complete" for cell in axis_cells)
        axis_rows.append({
            "axis": axis,
            "status": "complete" if axis_complete else "partial",
            "required_cell_count": len(axis_cells),
            "complete_cell_count": sum(cell["status"] == "complete" for cell in axis_cells),
        })

    complete = bool(required_cells) and bool(axis_rows) and all(row["status"] == "complete" for row in axis_rows)
    status_counts = {
        status: sum(cell["status"] == status for cell in required_cells)
        for status in ("complete", "partial", "uncovered", "exhausted", "unavailable")
    }
    gaps = [cell for cell in required_cells if cell["status"] != "complete"]
    return {
        "enabled": True,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "required_cell_count": len(required_cells),
        "complete_cell_count": status_counts["complete"],
        "status_counts": status_counts,
        "axes": axis_rows,
        "cells": required_cells,
        "gaps": gaps,
        "requires_gap_acknowledgement": not complete,
    }


def _attach_matrix_gap_disclosure(
    documents: list[dict[str, Any]],
    report_document: dict[str, Any],
    matrix_coverage: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gaps = [item for item in matrix_coverage.get("gaps") or [] if isinstance(item, dict)]
    if not gaps:
        return documents, report_document
    marker = "<!-- comparison-matrix-gaps-v1 -->"
    lines = [
        marker,
        "## 比较矩阵缺口",
        "",
        (
            f"本报告为 partial：{matrix_coverage.get('complete_cell_count', 0)} / "
            f"{matrix_coverage.get('required_cell_count', 0)} 个 Required Cell 完整覆盖。"
        ),
        "以下状态描述本次执行结果，不将系统故障解释为公开证据不足：",
        "",
    ]
    for cell in gaps:
        status = str(cell.get("status") or "uncovered")
        missing_criteria = [
            str(item.get("description") or item.get("criterion_id") or "").strip()
            for item in cell.get("criteria") or []
            if isinstance(item, dict) and str(item.get("status") or "") != "covered"
        ]
        suffix = f"；缺失 Hard Criterion：{'、'.join(missing_criteria)}" if missing_criteria else ""
        reasons = [str(item) for item in cell.get("unavailable_reasons") or [] if str(item)]
        if status == "unavailable" and reasons:
            suffix += f"；系统不可用原因：{'；'.join(reasons)}"
        lines.append(f"- {cell.get('subject')} × {cell.get('axis')}：`{status}`{suffix}")
    disclosure = "\n".join(lines)

    next_documents = [dict(document) for document in documents]
    target = next((
        document
        for document in next_documents
        if "参考" not in str(document.get("title") or "")
        and "总览" not in str(document.get("title") or "")
    ), next_documents[0] if next_documents else None)
    if target is not None and marker not in str(target.get("content") or ""):
        target["content"] = str(target.get("content") or "").rstrip() + "\n\n" + disclosure

    next_report = dict(report_document)
    if marker not in str(next_report.get("content") or ""):
        next_report["content"] = str(next_report.get("content") or "").rstrip() + "\n\n" + disclosure
    next_report["matrix_coverage"] = matrix_coverage
    return next_documents, next_report


def _question_answer_status(state: dict[str, Any]) -> str:
    """Read the new content status while remaining useful for historical checkpoints."""
    coverage = state.get("coverage") if isinstance(state.get("coverage"), dict) else {}
    value = str(coverage.get("answer_status") or "").strip().lower()
    if value in {"complete", "partial", "uncovered"}:
        return value
    coverage_status = str(coverage.get("coverage_status") or "").strip().lower()
    if coverage_status in {"sufficient", "sufficient_with_gaps"} or str(state.get("status") or "") == "sufficient":
        return "complete"
    if int(coverage.get("evidence_count") or len(state.get("accepted_evidence") or [])) > 0:
        return "partial"
    return "uncovered"


def _targeted_followup_rate(states: list[dict[str, Any]]) -> float:
    generated = sum(int(state.get("generated_gap_action_count") or 0) for state in states)
    executed = sum(len(state.get("gap_execution_log") or []) for state in states)
    return round(executed / generated, 3) if generated else 1.0


def _register_scheduled_gap_actions(state: Any, queries: list[str]) -> int:
    """Account for gap actions explicitly scheduled by the reportability rescue pass."""
    scheduled_queries = set(list(dict.fromkeys(query for query in queries if query))[:2])
    generated = sum(
        str(item.get("query") or "") in scheduled_queries
        for item in getattr(state, "gap_actions", [])
        if isinstance(item, dict)
    )
    state.generated_gap_action_count += generated
    return generated


def _initial_steps() -> list[dict[str, Any]]:
    return [{"id": step_id, "title": title, "status": "pending"} for step_id, title in RESEARCH_STEPS]


def _step(job_id: str, step_id: str, status: str, summary: str, items: list[Any] | None = None) -> None:
    job_service.update_step(job_id, step_id, status, summary=summary, items=items, progress_message=summary)


_EMPTY_RECONNAISSANCE: dict[str, Any] = {"summary": "", "entity_type": "", "ambiguities": [], "disambiguation_terms": []}


def _unavailable_reconnaissance(reason: str) -> dict[str, Any]:
    return {**_EMPTY_RECONNAISSANCE, "verification_status": "unavailable", "failure_reason": reason}


def _research_reconnaissance(
    user_request: str,
    *,
    fetch_document_limit: int = 2,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Identify the concept/entity behind a short or ambiguous user request before planning.

    Returns a dict (not just the summary string) because a short query like "调研 harness
    企业级实现方案" can collide with an unrelated but prominent real entity (e.g. the Harness.io
    CI/CD product vs. the "AI Agent Harness" architecture pattern the user meant). The model
    call below already asked for `ambiguities`, but the caller used to discard everything except
    `summary` — so a real, same-named entity's docs could get pulled into the evidence pool with
    nothing downstream ever having been told the request was ambiguous in the first place.
    `disambiguation_terms` are short qualifier terms (e.g. "AI agent", "智能体", "LLM") that the
    caller should append to every subsequent search query so retrieval stays on the concept the
    user meant instead of drifting to a same-named but unrelated real product/company.
    """
    if os.getenv("RESEARCH_CONCEPT_RECON", "1").lower() in {"0", "false", "no"}:
        return _unavailable_reconnaissance("concept_recon_disabled")
    try:
        discovery = broad_discovery(user_request, num_results=5)
    except Exception:
        return _unavailable_reconnaissance("concept_search_failed")
    sources = [item for item in discovery.get("sources") or [] if isinstance(item, dict)]
    if not sources:
        return _unavailable_reconnaissance("concept_search_empty")
    documents = []
    for source in sources[:max(0, fetch_document_limit)]:
        try:
            fetched = fetch_url_with_chain(str(source.get("url") or ""))
        except Exception:
            continue
        document = fetched.get("document") if isinstance(fetched, dict) else None
        if isinstance(document, dict) and document.get("content"):
            documents.append({"title": document.get("title"), "url": document.get("url"), "content": str(document.get("content"))[:2200]})
    payload = {
        "task": "Identify the exact concept/entity in the request before research planning. Return JSON only.",
        "user_request": user_request,
        "search_results": [{"title": item.get("title"), "url": item.get("url"), "snippet": str(item.get("snippet") or "")[:300]} for item in sources[:5]],
        "documents": documents,
        "rules": [
            "Use only supplied material.",
            "Set needs_user_clarification=true only when the request leaves the intended entity or meaning unresolved and the user must choose before research can proceed.",
            "Do not treat unrelated dictionary senses or same-spelled entities as blocking ambiguity when the request already names the domain, expands the acronym, defines the intended meaning, or explicitly excludes those alternatives.",
            "When needs_user_clarification=false, ambiguities must be empty. When it is true, state only the unresolved choices the user must decide between.",
            "Keep summary under 250 Chinese characters.",
            "If a term collides with another real entity and the request does not resolve which one is intended, list that collision in ambiguities.",
            "When ambiguities is non-empty, disambiguation_terms must contain 2-5 short search-query qualifier terms (English or Chinese) that would keep a web search on the concept the user meant and away from the colliding entity; otherwise leave it empty.",
        ],
        "schema": {
            "summary": "",
            "entity_type": "concept|technology|company|product|method|other",
            "needs_user_clarification": False,
            "ambiguities": [],
            "disambiguation_terms": [],
        },
    }
    result = _invoke_research_json(
        payload,
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75"))
        ),
    )
    if result.get("_failure_reason"):
        source_labels = [
            str(item.get("title") or item.get("url") or "").strip()
            for item in sources[:5]
            if str(item.get("title") or item.get("url") or "").strip()
        ]
        return {
            "summary": (
                f"调研请求：{user_request.strip()}。"
                + (f"初步检索候选：{'；'.join(source_labels)}。" if source_labels else "")
            )[:500],
            "entity_type": "",
            "ambiguities": [],
            "disambiguation_terms": [],
            "verification_status": "unavailable",
            "failure_reason": str(result.get("_failure_reason") or "concept_judge_failed"),
        }
    ambiguities = [str(item).strip() for item in result.get("ambiguities") or [] if str(item).strip()][:5]
    needs_user_clarification = result.get("needs_user_clarification")
    if needs_user_clarification is False:
        ambiguities = []
    normalized = {
        "summary": str(result.get("summary") or "").strip()[:500],
        "entity_type": str(result.get("entity_type") or "").strip(),
        "needs_user_clarification": bool(ambiguities),
        "ambiguities": ambiguities,
        "disambiguation_terms": [str(item).strip() for item in result.get("disambiguation_terms") or [] if str(item).strip()][:5],
    }
    if not normalized["summary"] or not isinstance(result.get("ambiguities"), list):
        return _unavailable_reconnaissance("invalid_concept_judgement")
    return {**normalized, "verification_status": "verified", "failure_reason": ""}


def _agentic_recon(user_request: str) -> dict[str, Any]:
    """Second-stage deep topic exploration for `clarify_concept_tool`, run only when the first
    stage (`_research_reconnaissance`) found no ambiguity -- see
    docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md. Reuses the exact same per-
    question agentic retrieval framework `question_loop`'s search phase already runs
    (`QuestionResearchOrchestrator._run_question_agent` / `agentic_model.py` / `agentic_tools.py`)
    against one synthetic "question" whose actual content is "what's out there on this topic",
    not a real research question -- see `run_single_agent_session`'s docstring for why this is a
    thin public wrapper rather than a second implementation of the retry/thread-id/finalize
    plumbing `_run_question_agent` already has.

    Convergence is delegated to the model's own `done()` call (决策: 单一完整 agent session +
    终结合成总结) inside a budget sized at roughly the "~3 rounds" the design doc describes
    (`AgenticBudget.for_recon()`) -- there is no outer Python round-loop here, unlike
    `_run_agentic`'s per-question retrieval which is one call per round; this is one continuous
    session. A user-pasted reference URL is expected to already be part of `user_request`'s text
    (决策: 不改接口，拼进 user_request 文本让模型自己判断) -- the agent has a `fetch_url` tool
    (`include_fetch_url=True`) to retrieve it directly instead of hoping `search_web`'s keyword
    search resurfaces the same page.

    Returns the same shape `_research_reconnaissance` does (`summary`/`entity_type`/
    `ambiguities`/`disambiguation_terms`) plus `evidence_count`/`source_count` diagnostics, so
    `clarify_concept_tool` can swap this in as `recon_context`'s source without changing its own
    contract. On any failure (model pool exhausted, agent invocation error) this falls back to
    `_research_reconnaissance`'s lightweight result rather than blocking clarify_concept_tool
    entirely -- a shallower recon_context is better than clarify_concept_tool failing outright.
    """
    if os.getenv("RESEARCH_AGENTIC_RECON", "1").lower() in {"0", "false", "no"}:
        return dict(_EMPTY_RECONNAISSANCE)
    try:
        state = _recon_question_state(
            question=user_request,
            question_id="recon_q1",
            purpose="探索该调研主题网上有哪些可用材料、方向和成熟度，为后续调研规划做准备",
        )
        orchestrator = _recon_orchestrator(job_id=f"recon_{uuid.uuid4().hex[:12]}", user_request=user_request)
        state = orchestrator.run_single_agent_session(
            state,
            system_prompt=_recon_agent_prompt(),
            initial_message=_recon_agent_briefing(user_request),
        )
        return _synthesize_recon_summary(user_request, state)
    except Exception:
        return dict(_EMPTY_RECONNAISSANCE)


def _recon_agent_prompt() -> str:
    from prompts import RECON_AGENT_PROMPT  # local: mirrors QUESTION_RESEARCH_AGENT_PROMPT's lazy import site

    return RECON_AGENT_PROMPT


def _recon_agent_briefing(user_request: str) -> str:
    return (
        f"要探索的调研主题（可能包含用户直接给出的参考链接/资料，请一并当作高置信度材料处理）：{user_request}\n"
        "目标：摸清楚这个主题网上实际有哪些可用材料、覆盖了哪些子方向、处于什么成熟度，"
        "为后续把这个主题拆解成一份有实质内容的调研规划做准备，而不是回答某个具体问题。"
    )


def _recon_question_state(*, question: str, question_id: str, purpose: str) -> ResearchQuestionState:
    from ..research.models import QuestionBudget

    return ResearchQuestionState(
        question_id=question_id,
        section_id="recon",
        section_title="主题探索",
        question=question,
        purpose=purpose,
        success_criteria=[purpose],
        preferred_source_types=["official_docs", "web_article"],
        budget=QuestionBudget.for_depth("normal"),
        min_evidence=1,
        min_sources=1,
    )


def _recon_orchestrator(
    *, job_id: str, user_request: str, budget: Any = None
) -> QuestionResearchOrchestrator:
    from ..research.agentic_tools import AgenticBudget

    return QuestionResearchOrchestrator(
        lambda state, queries, round_index: {},
        agentic=True,
        source_pool=GlobalSourcePool(),
        material_store=JobMaterialStore(),
        job_id=job_id,
        user_request=user_request,
        research_depth="normal",
        agentic_budget=budget or AgenticBudget.for_recon(),
        agentic_include_fetch_url=True,
    )


def _synthesize_recon_summary(user_request: str, state: ResearchQuestionState) -> dict[str, Any]:
    """One-shot synthesis over the full accumulated evidence/source pool after the recon session
    ends -- this is what "主题解读整体推翻重写" (决策4) means in a single-session design: rather
    than an outer per-round merge step, the whole accumulated pool is handed to the model ONCE at
    the end and it writes a fresh summary from scratch, so late-round findings can freely override
    an early-round framing instead of being appended alongside it.
    """
    evidence_rows = [
        {"claim": str(item.get("atomic_claim") or "")[:300], "source_url": item.get("source_url")}
        for item in state.accepted_evidence[:40]
    ]
    source_rows = [
        {"title": item.get("title"), "url": item.get("url")}
        for item in state.sources[:20]
        if isinstance(item, dict)
    ]
    if not evidence_rows and not source_rows:
        return dict(_EMPTY_RECONNAISSANCE)
    payload = {
        "task": "Synthesize a topic reconnaissance summary from accumulated research evidence. Return JSON only.",
        "user_request": user_request,
        "accumulated_evidence": evidence_rows,
        "accumulated_sources": source_rows,
        "rules": [
            "Base the summary only on the supplied evidence/sources, not prior knowledge.",
            "Describe what sub-directions/aspects of the topic actually have material available, "
            "and roughly how mature/well-documented the topic is online.",
            "Keep summary under 600 Chinese characters.",
            "If a term in the request collides with a different, unrelated real entity, list that "
            "collision in ambiguities; otherwise leave ambiguities empty.",
        ],
        "schema": {
            "summary": "",
            "entity_type": "concept|technology|company|product|method|other",
            "ambiguities": [],
            "disambiguation_terms": [],
        },
    }
    result = _research_json_model(payload)
    summary = str(result.get("summary") or "").strip()
    if not summary:
        return dict(_EMPTY_RECONNAISSANCE)
    return {
        "summary": summary[:800],
        "entity_type": str(result.get("entity_type") or "").strip(),
        "ambiguities": [str(item).strip() for item in result.get("ambiguities") or [] if str(item).strip()][:5],
        "disambiguation_terms": [str(item).strip() for item in result.get("disambiguation_terms") or [] if str(item).strip()][:5],
        "evidence_count": len(state.accepted_evidence),
        "source_count": len(state.sources),
    }


def _agentic_recon_for_section(*, user_request: str, section_title: str, section_axis: str = "") -> str:
    """Focused supplemental agentic exploration for one redo'd section
    (docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md 线1的延伸) --
    `draft_research_plan_tool`'s redo path calls this instead of reusing the same-turn
    `recon_context` verbatim, so "用户不满意 → 要求重新调研 → 改大纲" actually fetches new
    evidence for that section's direction rather than just rephrasing stale material.

    Deliberately narrower than `_agentic_recon`: the briefing anchors the exploration on
    `section_title`/`section_axis`, not the whole topic, and the budget
    (`AgenticBudget.for_section_focus()`) is sized down to match. Returns a plain text block (not
    the full `_research_reconnaissance`-shaped dict `_agentic_recon` returns) meant to be appended
    to the existing `recon_context` before `_enrich_explicit_plan_questions` regenerates the
    section -- there is no separate ambiguity/entity_type concept at section granularity.

    On any failure this returns an empty string so the redo path falls back to the original
    `recon_context` unchanged rather than blocking the redo entirely.
    """
    if os.getenv("RESEARCH_AGENTIC_RECON", "1").lower() in {"0", "false", "no"}:
        return ""
    try:
        question = f"{user_request}：{section_title}" if user_request else section_title
        purpose = f"补充探索「{section_title}」这个章节方向下网上实际有哪些可用材料和新方向"
        if section_axis:
            purpose += f"（方向提示：{section_axis}）"
        state = _recon_question_state(question=question, question_id="recon_section_q1", purpose=purpose)
        from ..research.agentic_tools import AgenticBudget

        orchestrator = _recon_orchestrator(
            job_id=f"recon_section_{uuid.uuid4().hex[:12]}",
            user_request=user_request,
            budget=AgenticBudget.for_section_focus(),
        )
        state = orchestrator.run_single_agent_session(
            state,
            system_prompt=_recon_agent_prompt(),
            initial_message=(
                f"要补充探索的章节方向：{section_title}\n"
                f"所属调研主题：{user_request}\n"
                + (f"方向提示：{section_axis}\n" if section_axis else "")
                + "目标：只针对这一节的方向找新材料/新角度，不需要覆盖整个调研主题。"
            ),
        )
        synthesized = _synthesize_recon_summary(question, state)
        summary = str(synthesized.get("summary") or "").strip()
        if not summary:
            return ""
        return f"[针对「{section_title}」的补充探索]{summary}"
    except Exception:
        return ""


def _extract_topic_anchor(user_request: str) -> str:
    """Extract the stable per-job topic anchor (e.g. "Harness Engineering", "LightRAG GraphRAG")
    from user_request ONCE per job. Every question/round in the job shares this same value
    (see ResearchQuestionState.topic_anchor) rather than each of the 4-6+ separate model calls a
    job makes re-deriving its own phrasing, which risked drift -- see compile_research_query's
    docstring for the truncation-order incidents a wobbling anchor previously caused.

    Unlike _research_reconnaissance, this must NOT depend on broad_discovery succeeding: the
    anchor is load-bearing infrastructure for every query the job compiles, not a best-effort
    disambiguation aid, so it has to be attempted even when web search is unavailable. A bare
    model call over user_request's own text is sufficient -- the request already reads like
    "调研 X 的...，要求...网页证据", and the model only needs to strip that boilerplate.
    """
    result = _research_json_model({
        "task": "Extract the core topic anchor from this research request. Return JSON only.",
        "user_request": user_request,
        "rules": [
            "Output ONLY the core subject phrase, e.g. 'Harness Engineering', 'LightRAG GraphRAG'.",
            "Strip boilerplate such as '调研/对比调研/深度调研' and '要求...网页证据' and '...的...和...'.",
            "Keep it under 30 characters. If the request names multiple subjects, list them space-separated.",
            "If no clear topic anchor can be identified, return an empty string.",
        ],
        "schema": {"topic_anchor": ""},
    })
    return str(result.get("topic_anchor") or "").strip()[:40]


def _plan_research(
    *,
    user_request: str,
    topic: str,
    input_data: dict[str, Any],
    strategy: str,
    recon_context: str = "",
    force_dynamic_questions: bool = False,
) -> dict[str, Any]:
    topic_anchor = _extract_topic_anchor(user_request)
    confirmed_plan_raw = input_data.get("confirmed_plan")
    if isinstance(confirmed_plan_raw, dict):
        plan = _normalize_research_plan(
            confirmed_plan_raw,
            user_request=user_request,
            topic=topic,
            strategy=strategy,
            planner_mode="authored_plan",
        )
        if plan:
            plan["planner"] = {"mode": "authored_plan", "source": "confirmed_plan"}
            plan["topic_anchor"] = topic_anchor
            return plan
        if str(input_data.get("plan_policy") or "") == "confirmed_required":
            raise ValueError("用户确认的完整调研计划无法规范化，已停止任务")
    planned_sections_raw = input_data.get("planned_sections")
    if isinstance(planned_sections_raw, list) and planned_sections_raw:
        # Full sections+questions plan the user already confirmed through
        # clarify_concept_tool -> draft_research_plan_tool before start_research_job_tool was
        # called -- see docs/modules/deep-research-step1-clarify-and-dispatch-discussion-
        # v0.1.md "接通那根断线". _normalize_research_plan/_normalize_planned_sections is the
        # same pair draft_research_plan_tool already ran this data through, so feeding it in
        # again here is idempotent normalization, not a re-plan: no new model call happens.
        plan = _normalize_research_plan(
            {"sections": planned_sections_raw},
            user_request=user_request,
            topic=topic,
            strategy=strategy,
            planner_mode="authored_plan",
        )
        if plan:
            plan["planner"] = {"mode": "authored_plan", "source": "conversation_confirmed"}
            plan["topic_anchor"] = topic_anchor
            return plan
        if str(input_data.get("plan_policy") or "") == "confirmed_required":
            raise ValueError("用户确认的调研计划无法规范化，已停止任务")
        # Falls through to required_titles/model/fallback below only if normalization somehow
        # dropped below 3 sections (defensive -- the plan was already validated once when the
        # user confirmed it in draft_research_plan_tool).

    explicit_titles = [str(item).strip() for item in input_data.get("required_titles") or [] if str(item).strip()]
    if explicit_titles:
        plan = _build_research_plan(user_request=user_request, topic=topic, titles=explicit_titles, strategy=strategy)
        if force_dynamic_questions:
            # strict=True here (job-internal path): a planning-model failure must stop the job,
            # not silently degrade to _normalize_section_question_items's generic template
            # questions ("XX 的关键问题是什么？" etc.) -- those are meta-questions no web article
            # ever literally answers, so evidence extraction later comes back empty for them and
            # the failure only surfaces much further downstream as an unexplained coverage gap.
            # Mirrors 决策8's draft_research_plan_tool convention (see _enrich_explicit_plan_questions'
            # own docstring) that never applied to this job-internal call site before.
            enriched = _enrich_explicit_plan_questions(
                plan=plan,
                user_request=user_request,
                recon_context=recon_context,
                strict=True,
            )
            if enriched.get("_planning_unavailable"):
                raise ValueError(
                    "研究问题规划失败："
                    + str(enriched.get("_failure_reason") or "unknown")
                    + "（规划模型多次尝试后仍不可用，已停止，不降级为通用模板问题继续调研）"
                )
            plan = enriched
        plan["planner"] = {"mode": "explicit_titles", "source": "request"}
        plan["plan_quality"] = _plan_quality(plan, user_request=user_request)
        plan["topic_anchor"] = topic_anchor
        return plan

    model_plan = _model_research_plan(user_request=user_request, topic=topic, strategy=strategy, recon_context=recon_context)
    if model_plan.get("_planning_unavailable"):
        raise ValueError(
            "研究问题规划失败："
            + str(model_plan.get("_failure_reason") or "unknown")
            + "（已停止，不降级为固定模板继续调研）"
        )
    if model_plan:
        plan = _normalize_research_plan(model_plan, user_request=user_request, topic=topic, strategy=strategy, planner_mode="model")
        if plan:
            plan["topic_anchor"] = topic_anchor
            return plan

    fallback = _fallback_research_plan(user_request=user_request, topic=topic, strategy=strategy)
    plan = _normalize_research_plan(fallback, user_request=user_request, topic=topic, strategy=strategy, planner_mode="fallback")
    if plan:
        plan["topic_anchor"] = topic_anchor
        return plan

    titles = _required_titles(input_data, strategy)
    plan = _build_research_plan(user_request=user_request, topic=topic, titles=titles, strategy=strategy)
    plan["planner"] = {"mode": "static_last_resort", "source": "legacy"}
    plan["plan_quality"] = _plan_quality(plan, user_request=user_request)
    plan["topic_anchor"] = topic_anchor
    return plan


def _enrich_explicit_plan_questions(
    *,
    plan: dict[str, Any],
    user_request: str,
    recon_context: str,
    timeout_seconds: float | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """``timeout_seconds`` override lets draft_research_plan_tool's redo/add adjustment path use
    PLAN_DRAFT_TIMEOUT_SECONDS instead of the job-internal RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS
    default -- same reasoning as _model_research_plan's timeout_seconds param above.

    ``strict=True`` is draft_research_plan_tool's opt-in only (job callers never pass it, so the
    job-internal fallback described below is unchanged). Per
    docs/modules/deep-research-step1-clarify-and-dispatch-discussion-v0.1.md decision 8,
    draft_research_plan_tool must not accept the generic-template fallback this function uses for
    the job's own resilience -- it must tell the user the model is unavailable instead. When
    strict and the model call fails, this returns ``{"_planning_unavailable": True, "_failure_reason": ...}``
    with no ``sections`` key, instead of falling through to the template fallback below.
    """
    sections = [item for item in plan.get("sections") or [] if isinstance(item, dict)]
    payload = {
        "task": "Create executable research questions for fixed report sections. Return JSON only.",
        "user_request": user_request,
        "concept_reconnaissance": recon_context,
        "fixed_sections": [{"id": item.get("id"), "title": item.get("title"), "axis": item.get("axis")} for item in sections],
        "rules": [
            "Do not add, remove, rename, or reorder sections.",
            "Overview and evidence/reference-index sections need no executable questions.",
            "Create 1-2 specific questions per substantive analysis section; normal mode must stay at 6 questions or fewer.",
            "Questions must explicitly mention the subject/domain from the user request.",
            "Each question needs purpose, success_criteria, preferred_source_types, and dependencies.",
            "Keep success_criteria focused on answer substance: core criteria are only what must be answered for basic understanding or use. Put examples, parameter enumerations, activity metrics, background, and other enrichment in separate criteria prefixed with 补充：; prefix genuinely optional presentation extras with 可选：.",
            "Do not require multiple sources merely because a question is a comparison or evaluation. Only write an explicit multi-source/cross-validation/multiple-case criterion when that corroboration is itself necessary to answer the user's confirmed scope.",
            "Each question also needs criteria_queries: one short search-ready phrase per entry in success_criteria, aligned by index (criteria_queries[i] answers success_criteria[i]). Each phrase must be short search terms, not a sentence -- 3-8 words, under 40 characters, e.g. 'feedback loop mechanisms Harness Engineering' not a restatement of the full criterion prose.",
        ],
        "schema": {"sections": [{"id": "sec_01", "questions": [{"text": "", "purpose": "", "success_criteria": [], "preferred_source_types": [], "dependencies": [], "criteria_queries": ["short query for criterion 1", "short query for criterion 2"]}]}]},
    }
    resolved_timeout = (
        timeout_seconds if timeout_seconds is not None else float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75"))
    )
    # Retry a few times before treating the call as unavailable -- a single transient timeout/
    # rate-limit hit must not immediately read as "the planner is down" (see the strict=True
    # caller below, which now stops the job entirely on that verdict).
    max_attempts = max(1, int(os.getenv("RESEARCH_PLAN_ENRICH_MAX_ATTEMPTS", "3")))
    result: dict[str, Any] = {}
    for _attempt in range(max_attempts):
        result = _invoke_research_json(
            payload,
            timeout_seconds=resolved_timeout,
            allow_provider_fallback=True,
            fallback_stage="planner",
        )
        if not result.get("_failure_reason"):
            break
        if result.get("_provider_fallback_attempted"):
            break
    if strict and result.get("_failure_reason"):
        return {"_planning_unavailable": True, "_failure_reason": result["_failure_reason"]}
    generated = {str(item.get("id") or ""): item for item in result.get("sections") or [] if isinstance(item, dict)}
    # No early return on a failed/empty model call (e.g. RateLimitError): falling through lets
    # the per-section loop below hit _normalize_section_question_items()'s fallback, which embeds
    # user_request in the generated questions. Returning `plan` here instead left the raw
    # generic-template questions ("核心概念", "工作机制", ...) with zero topic anchor, so gap-filling
    # rounds searched on bare template words and pulled in unrelated evidence (see
    # docs/evaluations/ for the LightRAG case that surfaced this).
    output_sections = []
    for section in sections:
        generated_section = generated.get(str(section.get("id") or "")) or {}
        questions = _normalize_section_question_items(
            generated_section.get("questions"),
            title=str(section.get("title") or ""),
            user_request=user_request,
        )
        if is_non_research_section_title(
            str(section.get("title") or ""),
            has_questions=bool(section.get("questions")),
        ):
            questions = _normalize_section_question_items(
                section.get("questions"),
                title=str(section.get("title") or ""),
                user_request=user_request,
            )
            if any(token in str(section.get("title") or "").lower() for token in ("证据", "参考", "reference")):
                questions = []
        for index, question in enumerate(questions):
            if isinstance(question, dict):
                question["id"] = f"{section.get('id')}_q{index + 1}"
        output_sections.append({
            **section,
            "questions": questions,
            "preferred_source_types": _coerce_string_list(generated_section.get("preferred_source_types")) or section.get("preferred_source_types") or ["official_docs", "case_study"],
        })
    return {**plan, "sections": output_sections}


def _research_plan_prompt(
    *,
    user_request: str,
    topic: str,
    strategy: str,
    recon_context: str = "",
    variant: str = "current",
    required_evidence_obligations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build an inspectable planner prompt for production and prompt A/B probes."""
    prompt = {
        "task": "Create a dynamic Deep Research plan. Do not write the report.",
        "user_request": user_request,
        "topic": topic,
        "strategy": strategy,
        "concept_reconnaissance": recon_context,
        "requirements": [
            "Return JSON only.",
            "Use only as many sections as the user request needs. A normal plan often has 3-6 sections, but this is guidance rather than a quota; never create a section merely to reach a count.",
            "Section titles must be specific to the user request, not generic templates.",
            "Always include 00_总览 as the first section.",
            "Each substantive non-overview section needs at least one concrete research question. Add another question only when it covers a distinct part of the confirmed scope, not to fill a quota.",
            "For a broad or novice request, use the topic and reconnaissance to select a compact set of essential dimensions that makes the report independently useful. This is professional scope selection, not permission to turn every potentially interesting detail into a completion requirement.",
            "Each question needs purpose, answer_contract_level, success_criteria, preferred_source_types, and optional dependencies.",
            "Set answer_contract_level=hard only for a dimension explicitly requested by the user, clearly implied by the requested deliverable, or essential to a useful basic explanation (definition, core mechanism/capabilities, basic usage/scenarios, major limitations). Set answer_contract_level=soft for planner-added enrichment such as ecosystem/community activity, contributor counts, release trends/roadmaps, benchmarks, broad competitor surveys, or extra historical detail when the user did not ask for it. Soft questions are still researched and may appear in the report, but their gaps do not make the research incomplete.",
            "Treat the research question itself as the core answer contract. success_criteria are retrieval and enrichment guidance, not separate reasons to fail the whole question.",
            "Distinguish confirmed scope from planner-added enrichment. A dimension explicitly requested by the user may be part of the research question and its hard answer contract; a useful dimension introduced by the planner must stay out of the question's core wording and be written as a success criterion prefixed with 补充： or 可选：.",
            "Keep each question centered on one answerable idea. When a proposed question mixes several independently researchable dimensions, split them only when each dimension belongs to the user's confirmed scope; otherwise keep the extra dimensions as enrichment criteria.",
            "For latest/current requests, infer only the temporal scope needed to answer the user's wording. Historical depth, release counts, trends, metrics, and change categories that the user did not request are enrichment rather than hard completion requirements.",
            "For comparison requests, use enough representative alternatives and decision dimensions to answer the user's purpose. Targets or dimensions selected by the planner, rather than named or clearly implied by the user, are enrichment rather than hard completion requirements.",
            "Any concrete count, threshold, benchmark baseline, sample size, or time window chosen by the planner must be prefixed with 补充： or 可选： unless the user explicitly specified it. The planner may still recommend such a measurement, but its absence must not redefine the whole research as incomplete.",
            "Keep success_criteria focused on answer substance. Put examples, parameter enumerations, activity metrics, background, and other enrichment in separate criteria prefixed with 补充：; prefix genuinely optional presentation extras with 可选：.",
            "Do not require multiple sources merely because a question is a comparison or evaluation. Only write an explicit multi-source/cross-validation/multiple-case criterion when that corroboration is itself necessary to answer the user's confirmed scope.",
            "Each question also needs criteria_queries: one short search-ready phrase per entry in success_criteria, aligned by index (criteria_queries[i] answers success_criteria[i]). Each phrase must be short search terms, not a sentence -- 3-8 words, under 40 characters, e.g. 'feedback loop mechanisms Harness Engineering' not a restatement of the full criterion prose.",
            "Use source types such as official_docs, github, paper, case_study, product_page, benchmark, and community.",
            "Each section needs evidence_budget with min_evidence and min_sources; non-overview sections should target at least 6 body-worthy evidence rows and 3 sources.",
        ],
        "scope_examples": [
            {
                "user_request": "调研一下某个我不熟悉的新型数据库",
                "good_scope_decision": (
                    "Planner 应自主选择足以形成独立、有用报告的核心范围，例如它是什么、关键机制、"
                    "基本使用方式、适用场景与主要限制；不要为了显得深入而把固定版本数量、社区指标、"
                    "完整参数表或基准数字变成完成条件。具体章节应根据侦察材料生成，不照抄本例。"
                ),
            },
            {
                "user_request": "比较 A、B、C 三个方案的延迟、成本和私有化部署，并整理最近三个版本的变化",
                "good_scope_decision": (
                    "A/B/C、延迟/成本/私有化部署、最近三个版本均由用户明确指定，应保留为核心问题；"
                    "不能因为通常建议控制范围就删减这些要求。"
                ),
            },
            {
                "user_request": "调研某智能体框架，包括架构、功能、最新动态和同类对比",
                "good_scope_decision": (
                    "四个点都是确认范围。最新动态只需形成对当前状态和重要近期变化的有用回答；"
                    "同类对比应选足以帮助理解定位的代表对象和关键维度。用户未指定的历史版本数量、"
                    "社区活跃度明细和性能指标可作为补充，但不能反过来决定整项调研失败。"
                ),
            },
        ],
        "schema": {
            "research_type": "implementation_design | comparison | concept_explainer | market_landscape | technical_architecture | best_practices | literature_review",
            "topic": topic,
            "subjects": ["explicit subject named by the user"],
            "comparison_axes": ["explicit comparison axis named by the user"],
            "sections": [
                {
                    "title": "00_总览",
                    "intent": "why this section exists",
                    "preferred_source_types": ["official_docs", "case_study"],
                    "questions": [{
                        "text": "question",
                        "purpose": "why this question matters",
                        "answer_contract_level": "hard|soft",
                        "success_criteria": ["observable coverage requirement"],
                        "preferred_source_types": ["official_docs", "case_study"],
                        "dependencies": [],
                        "criteria_queries": ["short query for criterion 1", "short query for criterion 2"],
                    }],
                    "evidence_budget": {"min_evidence": 4, "min_sources": 3},
                }
            ],
        },
    }
    if variant in {"scope_aligned_v2", "neutral_contract_v3"}:
        prompt["requirements"] = [
            requirement
            for requirement in prompt["requirements"]
            if not requirement.startswith("Set answer_contract_level=hard")
        ]
        prompt["requirements"].insert(
            7,
            "Set answer_contract_level=hard for dimensions explicitly requested by the user or clearly required to answer those words. For a broad novice request, definition, core mechanism/capabilities, and basic usage/scenarios may be selected as hard foundations. Do not automatically make major limitations, advantages, ecosystem, history, benchmarks, comparisons, or operational metrics hard unless the user requested them or omitting them would make the requested answer materially misleading. Planner-added dimensions must be soft.",
        )
        prompt["requirements"].insert(
            10,
            "Before creating each non-overview section, compare it with the user's wording. If the section has no direct phrase or necessary answer relationship in the request, keep its questions soft and do not let its evidence budget block completion.",
        )
        prompt["scope_examples"].append({
            "user_request": "调研一下 Harness Engineering，讲清楚它是什么、核心工作方式，以及团队通常怎样实践，整理成一份中文报告并附上来源。",
            "good_scope_decision": (
                "定义、核心工作方式、团队实践和来源是确认范围，可作为 hard；优势与局限、生态、"
                "历史和竞品如果由 Planner 主动补充，应标成 soft，不得因为这些补充证据较少而判整项失败。"
            ),
        })
        if variant == "neutral_contract_v3":
            prompt["requirements"].extend([
                "The planner defines what must be discovered; it must not write candidate answers into questions, success_criteria, or criteria_queries. A concrete product fact, causal relation, advantage, limitation, performance result, or comparison conclusion may appear only when the user explicitly supplied it as a premise to verify.",
                "Treat concept reconnaissance as orientation and possible terminology, not as evidence. Do not promote a reconnaissance statement into a fact or completion criterion.",
                "Write each success criterion as an unanswered, observable fact slot: what subject, attribute, condition, comparison axis, or evidence boundary must be established. It must allow the eventual answer to be positive, negative, conditional, conflicting, or unknown.",
                "Do not fill an unanswered slot with candidate values. Unless the user supplied them, success_criteria must not name possible components, algorithms, data flows, use cases, failure modes, costs, strengths, or limitations, including parenthetical examples introduced by phrases such as '如', '例如', 'including', or 'e.g.'.",
                "Use the swap test for comparisons: a criterion must remain a valid research requirement if the eventual properties of the compared subjects are exchanged. If it stops making sense, it probably contains a preselected answer and must be rewritten as an open slot.",
                "Write discovery-oriented criteria_queries. A query may name the subject and requested axis, but must not embed an unverified answer that biases retrieval toward confirmation.",
                "For every question, return contract_source=user_explicit|deliverable_required|essential_foundation|planner_enrichment and a short contract_reason. planner_enrichment must use answer_contract_level=soft.",
                "00_总览 is synthesized from substantive sections and must have an empty questions array. Do not create an overview-only research question or criterion.",
            ])
            prompt["planning_examples"] = [{
                "user_request": "比较 A 和 B 的架构、检索流程、适用场景和局限。",
                "bad_plan_fragment": {
                    "question": "为什么 A 不擅长复杂关系，而 B 的索引成本更高？",
                    "success_criteria": ["证明 A 的复杂关系建模能力有限"],
                    "criteria_queries": ["A 复杂关系建模限制"],
                    "why_bad": "It turns unverified candidate conclusions into the research contract and confirmation-biased retrieval.",
                },
                "good_plan_fragment": {
                    "question": "A 和 B 在用户指定的架构、检索流程、适用场景与局限维度上分别有哪些有证据支持的差异？",
                    "answer_contract_level": "hard",
                    "contract_source": "user_explicit",
                    "contract_reason": "The user explicitly requested both subjects and all four comparison axes.",
                    "success_criteria": [
                        "确定 A 与 B 各自有文档支持的架构组成和数据流",
                        "确定 A 与 B 各自有文档支持的适用条件、已知限制或证据不足之处",
                    ],
                    "criteria_queries": [
                        "A B architecture retrieval documentation",
                        "A B use cases limitations evidence",
                    ],
                },
                "bad_criterion_examples": [
                    "确定 A 的架构组成（如图索引、向量检索和双级检索）",
                    "确定 B 的已知局限，如社区检测开销或内存需求",
                    "确定 A 的适用场景（如动态数据、实时查询和多跳推理）",
                ],
                "why_bad_criteria": (
                    "The named components, costs, and scenarios are candidate answers not supplied by the user. "
                    "A neutral criterion asks which documented components, limitations, or scenarios exist."
                ),
            }]
            question_schema = prompt["schema"]["sections"][0]["questions"][0]
            question_schema["contract_source"] = (
                "user_explicit | deliverable_required | essential_foundation | planner_enrichment"
            )
            question_schema["contract_reason"] = "why this dimension belongs in the answer contract"
    elif variant in {"neutral_contract_v4", "question_contract_v5", "question_contract_v6"}:
        prompt["requirements"] = [
            "Return JSON only and follow the supplied schema.",
            "Preserve every subject, comparison axis, time boundary, deliverable, and evidence requirement explicitly requested by the user.",
            "For a broad request, choose only the definition, core mechanism/capabilities, and basic usage/scenarios needed for an independently useful answer. Other planner-added dimensions are soft enrichment.",
            "00_总览 must have questions=[] because it synthesizes the substantive sections; it does not own a separate research question.",
            "Each substantive section should contain one neutral, answerable question per distinct requested dimension. Return contract_source=user_explicit|deliverable_required|essential_foundation|planner_enrichment and a short contract_reason for every question.",
            "A question and its success_criteria define unknown evidence slots, not likely answers. Do not insert any component, algorithm, mechanism, scenario, advantage, limitation, cost, performance result, causal relation, or parenthetical example that the user did not state.",
            "A neutral slot must remain valid whether the evidence supports, rejects, qualifies, conflicts with, or cannot answer a candidate conclusion. For comparisons, it must remain valid if the subjects' eventual properties are exchanged.",
            "criteria_queries must contain only the named subject plus the user's requested axis or neutral evidence terms; never search for an unverified conclusion.",
            "Use answer_contract_level=hard for user_explicit, deliverable_required, or genuinely essential foundations. Use soft for planner_enrichment. Missing soft enrichment must not make the research incomplete.",
            "Use source types such as official_docs, github, paper, case_study, product_page, benchmark, and community. Each substantive section needs a realistic evidence_budget; do not turn arbitrary counts into answer criteria.",
        ]
        prompt["scope_examples"] = []
        prompt["planning_examples"] = [{
            "user_request": "比较 A 和 B 的架构、检索流程、适用场景和局限。",
            "bad": {
                "question": "为什么 A 不擅长复杂关系，而 B 的索引成本更高？",
                "success_criteria": [
                    "确定 A 的架构组成（如图索引、向量检索和双级检索）",
                    "证明 B 存在社区检测开销和内存压力",
                ],
                "criteria_queries": ["A 复杂关系限制", "B 社区检测内存开销"],
                "problem": "Every highlighted technical property is a candidate answer that the user did not provide.",
            },
            "good": {
                "question": "A 与 B 在架构、检索流程、适用场景和局限方面分别有哪些有证据支持的差异？",
                "answer_contract_level": "hard",
                "contract_source": "user_explicit",
                "contract_reason": "The user explicitly named both subjects and all four axes.",
                "success_criteria": [
                    "确定 A 与 B 各自有证据支持的架构组成与数据流",
                    "确定 A 与 B 各自有证据支持的适用条件、局限或证据缺口",
                ],
                "criteria_queries": ["A B architecture evidence", "A B use cases limitations evidence"],
            },
        }]
        question_schema = prompt["schema"]["sections"][0]["questions"][0]
        question_schema["contract_source"] = (
            "user_explicit | deliverable_required | essential_foundation | planner_enrichment"
        )
        question_schema["contract_reason"] = "why this dimension belongs in the answer contract"
        if variant in {"question_contract_v5", "question_contract_v6"}:
            prompt["requirements"] = [
                "Return JSON only and follow the supplied schema.",
                "Preserve every subject, comparison axis, time boundary, deliverable, and evidence requirement explicitly requested by the user.",
                "For a broad request, create only the definition, core mechanism/capabilities, and basic usage/scenarios sections needed for an independently useful answer. Planner-added enrichment must be a separate soft section.",
                "00_总览 must have questions=[] because it synthesizes substantive sections.",
                "A source, citation, evidence-index, or reference section is report metadata and must also have questions=[]. Satisfy a user's source requirement through preferred_source_types and cited evidence across substantive questions, not through a standalone executable question.",
                "Each substantive section must contain exactly one neutral, answerable question. If the user explicitly requested multiple distinct axes, create separate sections instead of multiple hard questions in one section.",
                "For a comparison with multiple named subjects, organize substantive sections by the user's requested axes and keep all compared subjects in each axis question. Do not split one shared axis into separate subject-profile sections unless the user explicitly requested standalone profiles.",
                "For research_type=comparison, return subjects as the exact named comparison set and comparison_axes as the user's explicit axes. Do not add representative alternatives or leave either list empty.",
                "For every comparison question, classify its axis contract explicitly. Use comparison_axis_mode=exact only when the question is restricted to one concrete evidence dimension, and set comparison_axis to that exact dimension. Use comparison_axis_mode=open and comparison_axis='' for open-ended synthesis goals whose answer may legitimately contain several dimensions. Do not turn a section label or rhetorical task into an exact evidence axis.",
                "The question may name the requested subject and axis, but must not contain a candidate component, mechanism, scenario, advantage, limitation, cost, result, causal relation, or parenthetical answer example that the user did not state.",
                "For every question, return answer_contract_level and contract_source=user_explicit|deliverable_required|essential_foundation|planner_enrichment with a short contract_reason. planner_enrichment must be soft.",
                "A user_explicit hard question must keep the user's requested axis at the same granularity. Do not embed planner-selected subdimensions, example categories, audiences, scenarios, metrics, or evaluation criteria inside that hard question. Put useful added detail in a separate planner_enrichment soft section.",
                "For each question, return criteria_contract=question_dimensions and exactly one success_criterion: a concise neutral restatement of that question's single user-requested axis. Do not decompose it into planner-selected subcategories, relationships, audiences, stages, tools, examples, metrics, or likely answers. If multiple explicit axes are needed, split them into separate substantive sections. Leave criteria_queries=[]; query compilation and evidence-gap expansion happen downstream.",
                "Use preferred_source_types to express source expectations without naming candidate answers. Each substantive section needs a realistic evidence_budget.",
            ]
            prompt["planning_examples"] = [{
                "user_request": "比较 A 和 B 的架构、检索流程、适用场景和局限。",
                "bad_questions": [
                    "为什么 A 不擅长复杂关系，而 B 的索引成本更高？",
                    "A 的双级检索与 B 的社区检测有什么差异？",
                ],
                "good_questions": [
                    {
                        "text": "A 与 B 在架构设计上有哪些有证据支持的差异？",
                        "comparison_axis_mode": "exact",
                        "comparison_axis": "架构",
                        "answer_contract_level": "hard",
                        "contract_source": "user_explicit",
                        "contract_reason": "The user explicitly requested the architecture axis.",
                        "criteria_contract": "question_dimensions",
                        "success_criteria": [
                            "确定 A 与 B 在架构设计上有证据支持的差异或证据缺口",
                        ],
                        "criteria_queries": [],
                    },
                    {
                        "text": "A 与 B 各自有哪些有证据支持的局限或证据缺口？",
                        "comparison_axis_mode": "open",
                        "comparison_axis": "",
                        "answer_contract_level": "hard",
                        "contract_source": "user_explicit",
                        "contract_reason": "The user explicitly requested limitations.",
                        "criteria_contract": "question_dimensions",
                        "success_criteria": [
                            "确定 A 与 B 各自有证据支持的局限或证据缺口",
                        ],
                        "criteria_queries": [],
                    },
                ],
            }, {
                "user_request": "讲清楚 X 是什么、核心工作方式，以及团队通常怎样实践。",
                "bad_questions": [
                    "X 的定义、起源时间和主要提出者是什么？",
                    "X 如何通过约束、测试、反馈和工作流控制形成闭环？",
                    "团队通常使用 ToolOne 和 ToolTwo 怎样实践 X？",
                ],
                "why_bad": (
                    "Origin history, named mechanisms, a closed-loop conclusion, and named tools were not supplied "
                    "by the user. They are candidate answers or planner-added scope."
                ),
                "good_questions": [
                    {
                        "text": "X 的定义和概念边界是什么？",
                        "answer_contract_level": "hard",
                        "contract_source": "user_explicit",
                        "contract_reason": "The user asked what X is.",
                        "criteria_contract": "question_dimensions",
                        "success_criteria": ["确定 X 的定义和概念边界或证据缺口"],
                        "criteria_queries": [],
                    },
                    {
                        "text": "X 有哪些有证据支持的核心工作机制或流程？",
                        "answer_contract_level": "hard",
                        "contract_source": "user_explicit",
                        "contract_reason": "The user requested the core working method.",
                        "criteria_contract": "question_dimensions",
                        "success_criteria": ["确定 X 有证据支持的核心工作方式或证据缺口"],
                        "criteria_queries": [],
                    },
                    {
                        "text": "团队通常怎样实践 X？有哪些有证据支持的组织或流程做法？",
                        "answer_contract_level": "hard",
                        "contract_source": "user_explicit",
                        "contract_reason": "The user requested common team practices.",
                        "criteria_contract": "question_dimensions",
                        "success_criteria": ["确定团队实践 X 时有证据支持的做法或证据缺口"],
                        "criteria_queries": [],
                    },
                ],
            }]
            question_schema["success_criteria"] = ["one neutral restatement of the question axis"]
            question_schema["criteria_queries"] = []
            question_schema["criteria_contract"] = "question_dimensions"
            question_schema["comparison_axis_mode"] = "exact | open | not_applicable"
            question_schema["comparison_axis"] = "exact evidence dimension when mode=exact; otherwise empty"
            if variant == "question_contract_v6":
                required_obligations = [
                    {
                        "obligation_id": str(item.get("obligation_id") or "").strip(),
                        "description": str(item.get("description") or "").strip(),
                        "user_quote": str(item.get("user_quote") or "").strip(),
                    }
                    for item in required_evidence_obligations or []
                    if str(item.get("obligation_id") or "").strip()
                    and str(item.get("description") or "").strip()
                ]
                prompt["requirements"].insert(
                    6,
                    "Apply the evidence-independence test before finalizing every hard question: imagine that retrieval fully supports one requested obligation but returns no relevant evidence for another. Those obligations may remain in one coherent section and question, but they must be separate success_criteria so evidence can be admitted and coverage judged independently. This is about evidence contracts, not conjunction words.",
                )
                prompt["requirements"].insert(
                    7,
                    "Keep a comparison axis, a causal relationship, a sequence, or an explicit choice among alternatives as one criterion when the relationship or decision is itself the requested evidence object. Create separate criteria only when each obligation can be supported and reported independently while the other remains unknown.",
                )
                prompt["requirements"].insert(
                    8,
                    "The supplied required_evidence_obligations were produced by an independent scope-decomposition step. Preserve every obligation_id exactly once in scope_obligation_mappings. Map each obligation_id to exactly one success_criterion_path under a user_explicit hard question. Related obligations may share a question, but never share one criterion. Do not invent, merge, or omit required obligation IDs.",
                )
                prompt["requirements"].insert(
                    9,
                    "Each required obligation must also produce exactly one success_criterion. If one obligation description names several subjects, components, steps, or aspects that collectively define the requested axis, keep them together in one neutral criterion. Do not split one obligation into one criterion per subject, component, step, or likely answer unless the independent scope decomposition supplied separate obligation IDs.",
                )
                prompt["requirements"] = [
                    (
                        "For each user_explicit hard question, use criteria_contract=explicit_scope_obligations and return one neutral success_criterion per mapped required obligation. Related obligations may share a question, so multiple criteria are allowed, but planner-selected subcategories or likely answers are not. For essential foundations or soft enrichment that have no required obligation ID, use criteria_contract=question_dimensions and exactly one neutral criterion. Leave criteria_queries=[]; query compilation and evidence-gap expansion happen downstream."
                        if requirement.startswith("For each question, return criteria_contract=question_dimensions")
                        else requirement
                    )
                    for requirement in prompt["requirements"]
                ]
                prompt["required_evidence_obligations"] = required_obligations
                prompt["schema"]["scope_obligation_mappings"] = [{
                    "obligation_id": "o1",
                    "success_criterion_path": "sections[1].questions[0].success_criteria[0]",
                }]
                question_schema["criteria_contract"] = "explicit_scope_obligations | question_dimensions"
                question_schema["success_criteria"] = [
                    "one neutral criterion per mapped required obligation",
                ]
                prompt["planning_examples"].append({
                    "user_request": "说明系统如何检测并修复数据与索引之间的不一致。",
                    "bad_questions": [{
                        "text": "系统如何检测并修复数据与索引之间的不一致？",
                        "problem": (
                            "Detection evidence and remediation evidence are independently useful and independently missing. "
                            "Binding both to one hard question can cause valid remediation evidence to be rejected when detection is unsupported."
                        ),
                    }],
                    "good_section": {
                        "title": "不一致检测与修复",
                        "questions": [{
                            "text": "系统如何检测并修复数据与索引之间的不一致？",
                            "criteria_contract": "explicit_scope_obligations",
                            "success_criteria": [
                                "确定检测数据与索引不一致的有证据支持的方法或证据缺口",
                                "确定发现不一致后的修复方法或证据缺口",
                            ],
                        }],
                    },
                    "good_scope_obligation_mappings": [
                        {
                            "obligation_id": "o1",
                            "success_criterion_path": "sections[1].questions[0].success_criteria[0]",
                        },
                        {
                            "obligation_id": "o2",
                            "success_criterion_path": "sections[1].questions[0].success_criteria[1]",
                        },
                    ],
                    "do_not_over_split": {
                        "user_request": "比较 A 与 B 在架构上的差异，并说明何时选择增量更新或全量重建。",
                        "keep_as_single_contracts": [
                            "A 与 B 在架构上有哪些有证据支持的差异？",
                            "在什么条件下应选择增量更新或全量重建？",
                        ],
                        "reason": "Each question asks for one relationship or decision axis even though answering it may require evidence about multiple subjects or alternatives.",
                    },
                })
    elif variant != "current":
        raise ValueError(f"unknown planner prompt variant: {variant}")
    prompt["planner_prompt_variant"] = variant
    return prompt


def _question_contract_plan_shape_issues(
    plan: dict[str, Any],
    *,
    required_obligation_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Check the deterministic question-contract shape without semantic topic judgments."""
    issues: list[dict[str, str]] = []
    substantive_question_count = 0
    hard_question_paths: set[str] = set()
    hard_criterion_paths: set[str] = set()
    valid_sources = {"user_explicit", "deliverable_required", "essential_foundation", "planner_enrichment"}
    if str(plan.get("research_type") or "").strip() == "comparison":
        if len(_coerce_string_list(plan.get("subjects"))) < 2:
            issues.append({"path": "subjects", "type": "comparison_subjects_missing"})
        if not _coerce_string_list(plan.get("comparison_axes")):
            issues.append({"path": "comparison_axes", "type": "comparison_axes_missing"})
    for section_index, section in enumerate(plan.get("sections") or []):
        if not isinstance(section, dict):
            issues.append({"path": f"sections[{section_index}]", "type": "invalid_section"})
            continue
        questions = [item for item in section.get("questions") or [] if isinstance(item, dict)]
        if _is_reference_or_overview_title(
            str(section.get("title") or ""),
            has_questions=bool(questions),
        ):
            if questions:
                issues.append({"path": f"sections[{section_index}].questions", "type": "executable_non_research_section"})
            continue
        if len(questions) != 1:
            issues.append({
                "path": f"sections[{section_index}].questions",
                "type": "question_count_not_one",
            })
            continue
        substantive_question_count += 1
        question = questions[0]
        question_path = f"sections[{section_index}].questions[0]"
        if str(question.get("answer_contract_level") or "hard").strip().lower() != "soft":
            hard_question_paths.add(question_path)
        if question.get("criteria_queries"):
            issues.append({
                "path": f"sections[{section_index}].questions[0].criteria_queries",
                "type": "planner_query_present",
            })
        valid_criteria_contracts = (
            {"question_dimensions", "explicit_scope_obligations"}
            if required_obligation_ids is not None
            else {"question_dimensions"}
        )
        if str(question.get("criteria_contract") or "").strip() not in valid_criteria_contracts:
            issues.append({
                "path": f"sections[{section_index}].questions[0].criteria_contract",
                "type": "invalid_criteria_contract",
            })
        success_criteria = [item for item in question.get("success_criteria") or [] if str(item).strip()]
        if (
            (required_obligation_ids is None and len(success_criteria) != 1)
            or (required_obligation_ids is not None and not success_criteria)
        ):
            issues.append({
                "path": f"sections[{section_index}].questions[0].success_criteria",
                "type": "criterion_count_invalid",
            })
        if question_path in hard_question_paths:
            for criterion_index, _criterion in enumerate(success_criteria):
                hard_criterion_paths.add(
                    f"{question_path}.success_criteria[{criterion_index}]"
                )
        contract_source = str(question.get("contract_source") or "").strip()
        if contract_source not in valid_sources:
            issues.append({
                "path": f"sections[{section_index}].questions[0].contract_source",
                "type": "invalid_contract_source",
            })
        if contract_source == "planner_enrichment" and str(question.get("answer_contract_level") or "").lower() != "soft":
            issues.append({
                "path": f"sections[{section_index}].questions[0].answer_contract_level",
                "type": "planner_enrichment_not_soft",
            })
    if substantive_question_count == 0:
        issues.append({"path": "sections", "type": "no_substantive_questions"})
    if required_obligation_ids is not None:
        obligations = [item for item in plan.get("scope_obligation_mappings") or [] if isinstance(item, dict)]
        mapped_counts: dict[str, int] = {}
        seen_obligation_ids: set[str] = set()
        if not obligations and required_obligation_ids:
            issues.append({"path": "scope_obligation_mappings", "type": "scope_obligation_mappings_missing"})
        for obligation_index, obligation in enumerate(obligations):
            obligation_id = str(obligation.get("obligation_id") or "").strip()
            criterion_path = str(obligation.get("success_criterion_path") or "").strip()
            if (
                obligation_id not in required_obligation_ids
                or obligation_id in seen_obligation_ids
                or criterion_path not in hard_criterion_paths
            ):
                issues.append({
                    "path": f"scope_obligation_mappings[{obligation_index}]",
                    "type": "invalid_scope_obligation_mapping",
                })
                continue
            seen_obligation_ids.add(obligation_id)
            mapped_counts[criterion_path] = mapped_counts.get(criterion_path, 0) + 1
        for obligation_id in sorted(required_obligation_ids - seen_obligation_ids):
            issues.append({"path": f"scope_obligation_mappings.{obligation_id}", "type": "required_obligation_missing"})
        for criterion_path, count in sorted(mapped_counts.items()):
            if count > 1:
                issues.append({"path": criterion_path, "type": "compound_evidence_contract"})
    return issues


def _normalize_same_question_obligation_mappings(plan: dict[str, Any]) -> int:
    """Remove structurally redundant mappings without changing research criteria.

    A planner may map one broad obligation once per subject/component even though all of
    those hard criteria live under the same question. The criteria still remain and are
    independently covered downstream; only the duplicate bookkeeping rows are removed.
    Cross-question duplicates are preserved so deterministic validation still fails closed.
    """
    mappings = [item for item in plan.get("scope_obligation_mappings") or [] if isinstance(item, dict)]
    if not mappings:
        return 0
    normalized: list[dict[str, Any]] = []
    first_question_path: dict[str, str] = {}
    removed = 0
    for mapping in mappings:
        obligation_id = str(mapping.get("obligation_id") or "").strip()
        criterion_path = str(mapping.get("success_criterion_path") or "").strip()
        match = re.fullmatch(r"(.+\.questions\[\d+\])\.success_criteria\[\d+\]", criterion_path)
        question_path = match.group(1) if match else ""
        first_path = first_question_path.get(obligation_id)
        if obligation_id and question_path and first_path == question_path:
            removed += 1
            continue
        normalized.append(mapping)
        if obligation_id and question_path and first_path is None:
            first_question_path[obligation_id] = question_path
    if removed:
        plan["scope_obligation_mappings"] = normalized
        plan["scope_obligation_mapping_normalized_count"] = removed
    return removed


def _research_scope_obligations_prompt(user_request: str) -> dict[str, Any]:
    """Build the narrow pre-planning task that separates independently satisfiable scope."""
    return {
        "task": "Decompose the user's substantive research scope into the minimum independently satisfiable evidence obligations. Return JSON only.",
        "user_request": user_request,
        "rules": [
            "An obligation is independent when evidence could fully support it while another requested obligation remains unknown, and the supported result would still be useful to report.",
            "Keep a comparison, causal relationship, sequence, or explicit choice among alternatives as one obligation when that relationship or decision is itself what the user asks to establish.",
            "Do not split by conjunction words or grammar alone. Use evidence satisfiability and report usefulness.",
            "Preserve separately requested scope even when a later clause refers back to it. If the user asks both to identify or characterize a phenomenon and to diagnose, remediate, evaluate, or monitor it, keep the phenomenon inventory and the action method as separate obligations because either evidence set can be useful while the other is missing.",
            "Include only substantive evidence needs explicitly requested by the user. Exclude formatting, language, citation presentation, source preference, and delivery requirements.",
            "Use stable sequential IDs o1, o2, ... in request order. Do not answer the topic or name candidate findings.",
        ],
        "calibration": [
            {
                "request": "说明系统如何检测并修复数据与索引之间的不一致",
                "obligations": ["检测不一致的方法", "发现不一致后的修复方法"],
            },
            {
                "request": "说明常见错误以及如何诊断",
                "obligations": ["常见错误或失效模式", "诊断这些错误的方法"],
            },
            {
                "request": "比较 A 与 B 的架构差异，并说明何时选择增量更新或全量重建",
                "obligations": ["A 与 B 在架构上的差异", "增量更新与全量重建之间的选择条件"],
            },
        ],
        "schema": {
            "obligations": [{
                "obligation_id": "o1",
                "description": "one independently satisfiable substantive evidence need",
                "user_quote": "the supporting phrase from the request",
            }],
        },
    }


def _research_scope_obligations_system_prompt() -> str:
    return (
        "You are an independent research-scope analyst, not a planner or report writer. "
        "Identify the minimum independently satisfiable evidence obligations in the user's own request. "
        "Do not answer the topic, infer likely findings, or split text from grammar alone. Return only JSON."
    )


def _model_research_scope_obligations(
    user_request: str,
    *,
    timeout_seconds: float,
) -> tuple[list[dict[str, str]], str]:
    parsed = _invoke_research_json(
        _research_scope_obligations_prompt(user_request),
        timeout_seconds=timeout_seconds,
        system_prompt=_research_scope_obligations_system_prompt(),
        allow_provider_fallback=True,
        fallback_stage="planner_scope_decomposition",
    )
    if parsed.get("_failure_reason"):
        return [], str(parsed.get("_failure_reason") or "scope_decomposition_failed")
    obligations: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(parsed.get("obligations") or [], start=1):
        if not isinstance(item, dict):
            continue
        obligation_id = str(item.get("obligation_id") or f"o{index}").strip()
        description = str(item.get("description") or "").strip()
        if not obligation_id or obligation_id in seen_ids or not description:
            continue
        seen_ids.add(obligation_id)
        obligations.append({
            "obligation_id": obligation_id,
            "description": description,
            "user_quote": str(item.get("user_quote") or "").strip(),
        })
    if not obligations:
        return [], "invalid_scope_decomposition"
    return obligations, ""


def _research_plan_system_prompt(variant: str) -> str | None:
    """Return stable planner-role rules for prompt variants that explicitly test role separation."""
    if variant not in {
        "neutral_contract_v3",
        "neutral_contract_v4",
        "question_contract_v5",
        "question_contract_v6",
    }:
        return None
    return (
        "You are a research planner, not a report writer or subject-matter answerer. "
        "Preserve the user's requested scope, identify unanswered evidence needs, and produce only JSON matching "
        "the supplied schema. Never turn assumptions, reconnaissance notes, common beliefs, or likely conclusions "
        "into established facts. Questions and retrieval criteria must remain neutral until evidence is collected."
    )


def _research_plan_neutrality_audit_prompt(
    *,
    user_request: str,
    recon_context: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Build an isolated semantic acceptance check for a proposed research plan."""
    return {
        "task": "Audit whether this research plan is neutral, scope-faithful, and safe to use for retrieval. Return JSON only.",
        "user_request": user_request,
        "concept_reconnaissance": recon_context,
        "candidate_plan": plan,
        "rules": [
            "The user request is the scope authority. Reconnaissance may disambiguate terminology but is not evidence for a product fact or conclusion.",
            "Reject a question, success criterion, or query that assigns an unverified component, mechanism, scenario, advantage, limitation, cost, performance result, causal relation, or comparison outcome to a named subject.",
            "Neutral decomposition may use abstract analysis categories such as components, data flow, roles, process, or tools. Reject named or enumerated candidate values tied to the subject, including mechanisms such as constraints/testing/feedback, products or frameworks, proposed origin dates, concrete failure modes, or suggested scenarios, unless the user explicitly stated them.",
            "Reject planner-added hard scope that the user did not request and that is not essential to make a broad answer independently useful. Soft enrichment is allowed.",
            "For contract_source=user_explicit, reject a hard question that narrows or expands the user's axis with planner-selected subdimensions, example categories, audiences, scenarios, metrics, or evaluation criteria. Those additions must be separate planner_enrichment soft questions.",
            "Apply the evidence-independence test to each hard question. Reject a compound contract when evidence could fully support one requested obligation while being irrelevant to another independently required obligation, causing useful evidence to fail the shared question. Do not reject a comparison, causal relationship, sequence, or alternative-choice question merely because it mentions multiple subjects or operations; those remain one contract when the relationship or decision is the requested evidence object.",
            "For a broad 'what is it' request, the definition may be an essential hard foundation. Origin/history, ecosystem, named tools, limitations, benchmarks, and use cases are not automatically hard unless the user requested them or their omission would make the requested answer materially misleading.",
            "Definition/concept boundary, origin/history/timeline, named proponents, use cases, limitations, and ecosystem are distinct axes. A request asking what something is does not by itself justify origin, history, dates, or proponents as hard scope.",
            "Reject omission or softening of a subject, axis, time boundary, deliverable, or evidence requirement explicitly requested by the user.",
            "Queries may include neutral disambiguation terms, subject names, requested axes, and evidence/source terms. Reject queries that search only for a preselected answer.",
            "00_总览 must not own executable questions.",
            "accepted=true only when no material violation remains. Report precise JSON paths and minimal reasons; do not rewrite the plan in this call.",
            "Split explicit_scope_checks into atomic user requirements. Do not combine several user phrases into one aggregate row, because that can hide one omitted or expanded axis.",
        ],
        "calibration_examples": [
            {
                "user_request": "讲清楚 X 是什么、核心工作方式和团队实践。",
                "reject": "X 的核心工作方式如何通过约束、测试、反馈和工作流控制实现？",
                "reason": "The named mechanisms are candidate answers not supplied by the user.",
                "accept": "X 的核心工作方式有哪些有证据支持的机制或流程？",
            },
            {
                "user_request": "讲清楚 X 是什么。",
                "reject": "X 的定义、提出背景、起源时间和推动者是什么？",
                "reason": "Definition is requested; origin history and named proponents are planner-added scope and cannot silently become hard.",
                "accept": "X 的定义和概念边界是什么？",
            },
        ],
        "schema": {
            "accepted": True,
            "summary": "short audit conclusion",
            "explicit_scope_checks": [{
                "user_requirement": "one explicit subject, axis, boundary, deliverable, or source requirement",
                "user_quote": "verbatim supporting phrase from the user request",
                "covered": True,
                "plan_paths": ["sections[1].questions[0]"],
            }],
            "hard_question_checks": [{
                "path": "sections[1].questions[0]",
                "contract_source": "user_explicit | deliverable_required | essential_foundation | planner_enrichment",
                "justified_as_hard": True,
                "user_quote_or_foundation_reason": "exact user phrase or why this is indispensable to a broad basic answer",
            }],
            "violations": [{
                "path": "sections[1].questions[0].success_criteria[0]",
                "type": "preseeded_answer | hard_scope_expansion | explicit_scope_omission | compound_evidence_contract | reconnaissance_promoted | executable_overview | other",
                "reason": "what is wrong and why it materially biases or weakens the research",
            }],
        },
    }


def _research_plan_neutrality_audit_system_prompt() -> str:
    return (
        "You are an independent research-plan auditor. Judge the proposed plan against the user's request, "
        "not against your own knowledge of the topic. Distinguish neutral analytical dimensions from claims "
        "about what a named subject actually does. Treat definition, origin/history, use cases, limitations, and "
        "ecosystem as separate scope axes; topical relatedness is not proof that the user requested an axis. "
        "Return only the requested JSON verdict."
    )


def _model_research_plan(
    *,
    user_request: str,
    topic: str,
    strategy: str,
    recon_context: str = "",
    timeout_seconds: float | None = None,
    strict: bool = False,
    prompt_variant: str | None = None,
) -> dict[str, Any]:
    """Generate a dynamic sections+questions plan via the research model pool.

    Uses ``invoke_research_chat(evidence=True)`` (through ``_invoke_research_json``), not the
    plain single-key ``_model()`` this used to call -- see
    docs/modules/deep-research-step1-clarify-and-dispatch-discussion-v0.1.md 工具二: planning
    must share the research model pool's multi-key rotation + rate-limit cooldown/retry so a
    single exhausted key doesn't block job startup, exactly like ``_research_reconnaissance``.
    ``timeout_seconds`` lets ``draft_research_plan_tool`` (main-agent conversation, wants a
    short-ish wait) override the job-internal default (``RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS``)
    without touching job callers that pass nothing.

    ``strict=True`` is draft_research_plan_tool's opt-in only, same convention as
    ``_enrich_explicit_plan_questions``: on failure it returns
    ``{"_planning_unavailable": True, "_failure_reason": ...}`` instead of ``{}`` so the tool can
    tell the user the model is unavailable (决策8) instead of silently reporting "no plan".
    """
    if os.getenv("RESEARCH_DYNAMIC_PLANNER", "1").lower() in {"0", "false", "no"}:
        return {"_planning_unavailable": True, "_failure_reason": "planner_disabled"} if strict else {}
    resolved_variant = prompt_variant or os.getenv("RESEARCH_PLANNER_PROMPT_VARIANT", "question_contract_v6")
    resolved_timeout = (
        timeout_seconds if timeout_seconds is not None else float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75"))
    )
    required_evidence_obligations: list[dict[str, str]] = []
    if resolved_variant == "question_contract_v6":
        required_evidence_obligations, decomposition_error = _model_research_scope_obligations(
            user_request,
            timeout_seconds=resolved_timeout,
        )
        if decomposition_error:
            return {"_planning_unavailable": True, "_failure_reason": decomposition_error}
    prompt = _research_plan_prompt(
        user_request=user_request,
        topic=topic,
        strategy=strategy,
        recon_context=recon_context,
        variant=resolved_variant,
        required_evidence_obligations=required_evidence_obligations,
    )
    contract_attempts = (
        max(1, int(os.getenv("RESEARCH_PLANNER_CONTRACT_ATTEMPTS", "2")))
        if resolved_variant in {"question_contract_v5", "question_contract_v6"}
        else 1
    )
    last_shape_issues: list[dict[str, str]] = []
    for attempt_index in range(contract_attempts):
        attempt_prompt = prompt
        if attempt_index > 0 and last_shape_issues:
            attempt_prompt = {
                **prompt,
                "requirements": [
                    *prompt.get("requirements", []),
                    "The previous plan was rejected by deterministic contract validation. Repair the listed paths and issue types; do not repeat the rejected mapping shape.",
                ],
                "contract_retry_feedback": {
                    "previous_output_rejected": True,
                    "shape_issues": last_shape_issues,
                },
            }
        parsed = _invoke_research_json(
            attempt_prompt,
            timeout_seconds=resolved_timeout,
            system_prompt=_research_plan_system_prompt(resolved_variant),
            allow_provider_fallback=True,
            fallback_stage="planner",
        )
        if parsed.get("_failure_reason"):
            if strict or resolved_variant in {"question_contract_v5", "question_contract_v6"}:
                return {"_planning_unavailable": True, "_failure_reason": parsed["_failure_reason"]}
            return {}
        plan = parsed["research_plan"] if isinstance(parsed.get("research_plan"), dict) else parsed
        if resolved_variant not in {"question_contract_v5", "question_contract_v6"}:
            return plan

        if resolved_variant == "question_contract_v6":
            _normalize_same_question_obligation_mappings(plan)

        shape_issues = _question_contract_plan_shape_issues(
            plan,
            required_obligation_ids={
                item["obligation_id"] for item in required_evidence_obligations
            } if resolved_variant == "question_contract_v6" else None,
        )
        if not shape_issues:
            return plan
        last_shape_issues = shape_issues
        if attempt_index + 1 < contract_attempts:
            emit_event(
                "research.planner.contract_retry",
                "research",
                "Planner returned an invalid question contract; retrying the same model request once.",
                level="warning",
                data={
                    "attempt": attempt_index + 1,
                    "max_attempts": contract_attempts,
                    "issue_types": sorted(
                        {str(issue.get("type") or "unknown") for issue in shape_issues}
                    ),
                },
            )

    return {
        "_planning_unavailable": True,
        "_failure_reason": "invalid_question_contract",
        "_shape_issues": last_shape_issues,
    }


def _normalize_research_plan(
    raw: dict[str, Any],
    *,
    user_request: str,
    topic: str,
    strategy: str,
    planner_mode: str,
) -> dict[str, Any]:
    research_type = _normalize_research_type(
        raw.get("research_type") or raw.get("type") or _classify_research_type(user_request, strategy)
    )
    sections = _normalize_planned_sections(
        raw.get("sections"),
        user_request=user_request,
        strategy=strategy,
        question_limit=5 if research_type == "comparison" else 4,
    )
    # A compact user-confirmed plan may legitimately contain only overview + one substantive
    # section. Requiring three sections here silently discarded that plan and triggered a second
    # model planning call, contradicting the planner's "only as many sections as needed" contract.
    if len(sections) < 2:
        return {}
    fallback_profile = _build_research_profile(user_request=user_request, sections=sections, strategy=strategy)
    subjects = _coerce_string_list(raw.get("subjects")) or fallback_profile["subjects"]
    comparison_axes = _coerce_string_list(raw.get("comparison_axes")) or fallback_profile["comparison_axes"]
    profile = _comparison_profile(subjects=subjects, axes=comparison_axes, strategy=strategy)
    plan = {
        "strategy": strategy,
        "research_type": research_type,
        "strategy_profile": profile["profile"],
        "topic": str(raw.get("topic") or topic).strip() or topic,
        "user_request": user_request,
        "subjects": profile["subjects"],
        "comparison_axes": profile["comparison_axes"],
        "matrix_cells": profile["matrix_cells"],
        "judgement_questions": profile["judgement_questions"],
        "vertical": [item for item in sections if item["axis"] == "vertical"],
        "horizontal": [item for item in sections if item["axis"] == "horizontal"],
        "intersections": [
            "历史选择如何塑造当前生态位置",
            "当前优势或短板的形成原因",
            "未来可能演化路径和验证缺口",
        ] if strategy == "hv_analysis" else [],
        "sections": sections,
        "planner": {"mode": planner_mode, "source": "dynamic_research_planner"},
        # Old checkpoints had no criterion-level provenance and therefore restored every string
        # as hard. New plans stamp the policy so the visible research question is the hard answer
        # contract while planner-expanded detail remains retrieval guidance.
        "criterion_level_policy": CRITERION_LEVEL_POLICY_V1,
    }
    plan["plan_quality"] = _plan_quality(plan, user_request=user_request)
    return plan


def _normalize_planned_sections(
    raw_sections: Any,
    *,
    user_request: str,
    strategy: str,
    question_limit: int = 4,
) -> list[dict[str, Any]]:
    if not isinstance(raw_sections, list):
        return []
    sections: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for index, raw in enumerate(raw_sections[:8]):
        if isinstance(raw, str):
            raw = {"title": raw}
        if not isinstance(raw, dict):
            continue
        title = _normalize_section_title(str(raw.get("title") or raw.get("name") or ""), len(sections))
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        axis = _normalize_section_axis(str(raw.get("axis") or ""), title, strategy)
        raw_questions = raw.get("questions")
        questions = _normalize_section_question_items(
            raw_questions,
            title=title,
            user_request=user_request,
            question_limit=question_limit,
        )
        if is_non_research_section_title(
            title,
            has_questions=bool(raw_questions) if isinstance(raw_questions, list) else None,
        ):
            questions = []
        section_id = _section_id(title, len(sections))
        sections.append({
            "id": section_id,
            "title": title,
            "axis": axis,
            "intent": str(raw.get("intent") or raw.get("description") or "").strip(),
            "questions": questions,
            "success_criteria": _coerce_string_list(raw.get("success_criteria")) or _section_success_criteria(axis),
            "preferred_source_types": _coerce_string_list(raw.get("preferred_source_types")) or ["official_docs", "case_study"],
            "evidence_budget": _normalize_evidence_budget(raw.get("evidence_budget"), axis=axis),
        })
    if not sections:
        return []
    if not _is_overview_title_for_plan(sections[0]["title"]):
        sections.insert(0, _overview_section(user_request))
    # The input loop already enforces the planner's eight-section ceiling. Truncating once more
    # to seven here silently dropped a valid final section after contract validation, so the
    # normalized plan no longer preserved the user's confirmed scope.
    sections = sections[:8]
    for index, section in enumerate(sections):
        section["title"] = _normalize_section_title(section["title"], index)
        section["id"] = _section_id(section["title"], index)
        section["axis"] = _normalize_section_axis(str(section.get("axis") or ""), section["title"], strategy)
        if is_non_research_section_title(
            section["title"],
            has_questions=bool(section.get("questions")),
        ):
            section["questions"] = []
        elif not section.get("questions"):
            section["questions"] = _normalize_section_question_items(
                [],
                title=section["title"],
                user_request=user_request,
                question_limit=question_limit,
            )
        for question_index, question in enumerate(section.get("questions") or []):
            if isinstance(question, dict):
                question["id"] = str(question.get("id") or f"{section['id']}_q{question_index + 1}")
    return sections


def _normalize_section_title(title: str, index: int) -> str:
    text = re.sub(r"\s+", " ", str(title or "").strip(" ：:，,。"))
    if not text:
        return ""
    text = re.sub(r"^\d{1,2}[_\-.、\s]+", "", text).strip()
    prefix = f"{index:02d}"
    return f"{prefix}_{text}"


def _normalize_section_axis(axis: str, title: str, strategy: str) -> str:
    safe_axis = str(axis or "").strip().lower()
    if safe_axis in {"overview", "vertical", "horizontal", "mechanism", "practice", "evidence", "standard"}:
        return safe_axis
    return _section_axis(title, strategy)


def _normalize_section_question_items(
    raw_questions: Any,
    *,
    title: str,
    user_request: str,
    question_limit: int = 4,
) -> list[dict[str, Any]]:
    questions = []
    if isinstance(raw_questions, list):
        for raw in raw_questions[:question_limit]:
            if isinstance(raw, dict):
                text = _question_text(raw)
                row: dict[str, Any] = {"text": text}
                for key in (
                    "purpose",
                    "id",
                    "question_id",
                    "contract_source",
                    "contract_reason",
                    "criteria_contract",
                    "comparison_axis_mode",
                    "comparison_axis",
                    "subject",
                    "matrix_cell_id",
                ):
                    value = str(raw.get(key) or "").strip()
                    if value:
                        row["id" if key == "question_id" else key] = value
                row["answer_contract_level"] = _normalize_answer_contract_level(
                    raw.get("answer_contract_level"),
                    question=text,
                    user_request=user_request,
                )
                for key in ("success_criteria", "preferred_source_types", "dependencies", "criteria_queries"):
                    values = _coerce_string_list(raw.get(key))
                    if values:
                        row[key] = values
                comparison_subjects = _coerce_string_list(raw.get("comparison_subjects"))
                if comparison_subjects:
                    row["comparison_subjects"] = comparison_subjects
            else:
                text = str(raw or "").strip()
                row = {
                    "text": text,
                    "answer_contract_level": _normalize_answer_contract_level(
                        None,
                        question=text,
                        user_request=user_request,
                    ),
                }
            if text:
                questions.append(row)
    if not questions:
        stripped = _strip_title_prefix(title)
        questions = [
            {"text": f"{stripped} 的关键问题是什么？"},
            {"text": f"{stripped} 有哪些可靠证据或案例？"},
            {"text": f"{stripped} 对“{user_request}”的结论是什么？"},
        ]
    return questions[:question_limit]


def _normalize_answer_contract_level(value: Any, *, question: str, user_request: str) -> str:
    requested = str(value or "").strip().lower()
    if requested == "soft":
        return "soft"
    question_text = str(question or "").lower()
    request_text = str(user_request or "").lower()
    enrichment_dimensions = (
        ("生态", "社区", "贡献者", "活跃度", "issue", "pull request", "github activity", "community", "contributor"),
        ("版本发布", "发布频率", "路线图", "roadmap", "release trend", "版本趋势"),
        ("基准", "benchmark", "吞吐", "延迟测试", "性能排名"),
        ("竞品", "同类对比", "竞争格局", "横向比较", "comparison", "competitor"),
        ("历史演进", "发展历程", "时间线", "版本历史", "timeline"),
    )
    for terms in enrichment_dimensions:
        if any(term in question_text for term in terms) and not any(term in request_text for term in terms):
            return "soft"
    return "hard"


def _normalize_evidence_budget(raw: Any, *, axis: str) -> dict[str, int]:
    min_evidence = 4 if axis in {"overview", "evidence"} else 6
    min_sources = 3
    if isinstance(raw, dict):
        try:
            min_evidence = max(min_evidence, int(raw.get("min_evidence") or min_evidence))
        except (TypeError, ValueError):
            pass
        try:
            min_sources = max(min_sources, int(raw.get("min_sources") or min_sources))
        except (TypeError, ValueError):
            pass
    return {"min_evidence": min_evidence, "min_sources": min_sources}


def _overview_section(user_request: str) -> dict[str, Any]:
    return {
        "id": "sec_00",
        "title": "00_总览",
        "axis": "overview",
        "intent": "概括调研目标、范围、主要结论和证据边界。",
        "questions": [],
        "success_criteria": _section_success_criteria("overview"),
        "evidence_budget": {"min_evidence": 4, "min_sources": 3},
    }


def _fallback_research_plan(*, user_request: str, topic: str, strategy: str) -> dict[str, Any]:
    research_type = _classify_research_type(user_request, strategy)
    sections = _fallback_sections_for_type(research_type, user_request=user_request, strategy=strategy)
    return {
        "research_type": research_type,
        "topic": topic,
        "sections": sections,
        "planner": {"mode": "fallback", "source": "rule_templates_by_research_type"},
    }


def _classify_research_type(user_request: str, strategy: str) -> str:
    text = str(user_request or "").lower()
    if strategy == "hv_analysis" or any(key in text for key in ("对比", "比较", "差异", "vs", "横向")):
        return "comparison"
    if any(key in text for key in ("实现", "搭建", "构建", "开发", "落地", "方案", "接入", "runtime")):
        return "implementation_design"
    if any(key in text for key in ("指南", "规范", "最佳实践", "书写", "写法", "教程")):
        return "best_practices"
    if any(key in text for key in ("架构", "机制", "原理", "流程", "模块")):
        return "technical_architecture"
    if any(key in text for key in ("市场", "产品", "竞品", "行业", "生态", "趋势")):
        return "market_landscape"
    if any(key in text for key in ("论文", "文献", "综述", "研究进展")):
        return "literature_review"
    return "concept_explainer"


def _normalize_research_type(value: Any) -> str:
    text = str(value or "").strip()
    allowed = {
        "implementation_design",
        "comparison",
        "concept_explainer",
        "market_landscape",
        "technical_architecture",
        "best_practices",
        "literature_review",
    }
    return text if text in allowed else "concept_explainer"


def _fallback_sections_for_type(research_type: str, *, user_request: str, strategy: str) -> list[dict[str, Any]]:
    subject = _topic_label_from_request(user_request)
    if research_type == "comparison":
        titles = [
            "00_总览",
            "01_对象定义与评价维度",
            "02_核心能力对比",
            "03_架构机制与实现差异",
            "04_适用场景、成本与风险",
            "05_选择建议与证据缺口",
        ]
    elif research_type == "implementation_design":
        agent_label = "Agent 分工与协作架构" if "agent" in user_request.lower() else "系统架构与模块分工"
        titles = [
            "00_总览",
            "01_目标场景与用户流程",
            f"02_{agent_label}",
            "03_工具调用、数据流与外部集成",
            "04_状态管理、记忆与上下文",
            "05_工程实现路径、评估与风险",
        ]
    elif research_type == "best_practices":
        titles = [
            "00_总览",
            "01_适用对象与使用场景",
            "02_核心原则与质量标准",
            "03_书写结构、模板与示例",
            "04_评审流程、反例与常见错误",
            "05_落地检查清单与持续优化",
        ]
    elif research_type == "technical_architecture":
        titles = [
            "00_总览",
            "01_问题背景与核心概念",
            "02_架构组成与数据流",
            "03_关键机制与实现约束",
            "04_工程落地方式与依赖",
            "05_局限、风险与验证方法",
        ]
    elif research_type == "market_landscape":
        titles = [
            "00_总览",
            "01_市场背景与用户需求",
            "02_主要产品、玩家与定位",
            "03_能力维度、差异与趋势",
            "04_商业模式、成本与采用门槛",
            "05_机会、风险与后续观察",
        ]
    elif research_type == "literature_review":
        titles = [
            "00_总览",
            "01_研究问题与检索范围",
            "02_主要方法与代表工作",
            "03_关键发现、共识与分歧",
            "04_应用场景、局限与评估",
            "05_证据缺口与后续阅读路线",
        ]
    else:
        titles = [
            "00_总览",
            f"01_{subject}的定义与问题背景",
            "02_核心组成、工作机制与边界",
            "03_典型场景、案例与实践路径",
            "04_价值、限制与判断标准",
        ]
    sections = []
    for index, title in enumerate(titles):
        sections.append({
            "title": title,
            "intent": _fallback_section_intent(title, research_type),
            "axis": _section_axis(title, strategy),
            "questions": _fallback_questions_for_title(title, user_request=user_request, research_type=research_type),
            "evidence_budget": {"min_evidence": 4 if index == 0 else 6, "min_sources": 3},
        })
    return sections


def _topic_label_from_request(user_request: str) -> str:
    text = str(user_request or "").strip()
    text = re.sub(r"^(?:帮我|请|麻烦)?(?:深度)?(?:调研|研究|分析|看看)(?:一下)?", "", text).strip(" ：:，,。")
    return text[:24] or "主题"


def _fallback_section_intent(title: str, research_type: str) -> str:
    stripped = _strip_title_prefix(title)
    return f"围绕 {research_type} 调研类型，回答“{stripped}”相关的核心问题并收集可引用证据。"


def _fallback_questions_for_title(title: str, *, user_request: str, research_type: str) -> list[str]:
    stripped = _strip_title_prefix(title)
    if "总览" in stripped:
        return [f"{user_request} 的核心结论是什么？", "哪些结论证据充分，哪些存在缺口？"]
    if research_type == "implementation_design":
        if "目标场景" in stripped:
            return ["目标用户和主要使用场景是什么？", "端到端用户流程如何展开？", "哪些边界需求不应放入 MVP？"]
        if "架构" in stripped or "分工" in stripped:
            return ["系统应拆成哪些模块或 agent？", "这些模块如何协作和交接状态？", "协作失败或冲突时如何处理？"]
        if "工具" in stripped or "集成" in stripped:
            return ["需要接入哪些外部工具或服务？", "工具调用的数据流和权限边界是什么？", "工具失败时如何降级？"]
        if "状态" in stripped or "记忆" in stripped:
            return ["需要保存哪些用户偏好和上下文？", "多轮修改如何保持一致性？", "隐私和数据保留有什么风险？"]
        return ["MVP 如何分阶段落地？", "如何评估效果和成本？", "主要工程风险是什么？"]
    if research_type == "comparison":
        return [f"{stripped} 的评价维度是什么？", "各对象在这些维度上的公开证据是什么？", "差异背后的原因和适用边界是什么？"]
    if research_type == "best_practices":
        return [f"{stripped} 的核心原则是什么？", "有哪些可复用模板或示例？", "常见错误和评审标准是什么？"]
    return [f"{stripped} 的核心问题是什么？", f"{stripped} 有哪些权威来源或案例？", f"{stripped} 对“{user_request}”的判断是什么？"]


def _titles_from_research_plan(research_plan: dict[str, Any]) -> list[str]:
    titles = [
        str(section.get("title") or "").strip()
        for section in research_plan.get("sections") or []
        if isinstance(section, dict) and str(section.get("title") or "").strip()
    ]
    return titles or STANDARD_TITLES


def _plan_quality(plan: dict[str, Any], *, user_request: str) -> dict[str, Any]:
    sections = [item for item in plan.get("sections") or [] if isinstance(item, dict)]
    titles = [str(item.get("title") or "") for item in sections]
    generic_titles = [
        title for title in titles
        if _strip_title_prefix(title) in {"核心概念", "实操使用指南", "关键机制", "证据与参考"}
    ]
    question_count = sum(len(item.get("questions") or []) for item in sections)
    non_overview = [title for title in titles if not _is_overview_title_for_plan(title)]
    issues = []
    if len(sections) < 3:
        issues.append("section_count_low")
    if generic_titles and len(generic_titles) >= max(2, len(non_overview)):
        issues.append("generic_template_sections")
    if question_count < max(1, len(non_overview)):
        issues.append("question_count_low")
    searchable = "\n".join([
        *titles,
        *[
            _question_text(question)
            for section in sections
            for question in (section.get("questions") or [])
        ],
    ]).lower()
    if not any(term and term.lower() in searchable for term in _topic_terms(user_request)):
        issues.append("topic_specificity_low")
    return {
        "status": "passed" if not issues else "weak",
        "issues": issues,
        "section_count": len(sections),
        "question_count": question_count,
        "generic_titles": generic_titles,
    }


def _topic_terms(user_request: str) -> list[str]:
    text = _topic_label_from_request(user_request)
    stopwords = {"帮我", "调研", "研究", "分析", "一下", "深度", "实现", "对比", "比较", "方案"}
    return [term for term in extract_key_terms(text, top_k=8) if term not in stopwords]


def _is_overview_title_for_plan(title: str) -> bool:
    stripped = _strip_title_prefix(title).lower()
    return stripped in {"总览", "overview", "summary"} or "总览" in stripped


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _research_strategy(input_data: dict[str, Any]) -> str:
    strategy = str(input_data.get("research_strategy") or "standard").strip()
    return strategy if strategy in {"standard", "hv_analysis"} else "standard"


def _research_depth(input_data: dict[str, Any]) -> str:
    depth = str(input_data.get("depth") or "normal").strip()
    return depth if depth in {"normal", "deep"} else "normal"


def _strategy_label(strategy: str) -> str:
    return "横纵分析法" if strategy == "hv_analysis" else "标准调研"


def _required_titles(input_data: dict[str, Any], strategy: str = "standard") -> list[str]:
    titles = [str(item).strip() for item in input_data.get("required_titles") or [] if str(item).strip()]
    if titles:
        return titles
    return HV_ANALYSIS_TITLES if strategy == "hv_analysis" else STANDARD_TITLES


def _suggest_topic(input_data: dict[str, Any], user_request: str) -> str:
    topic = str(input_data.get("topic") or "").strip()
    if topic:
        return topic
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", user_request.lower()).strip("_")
    return (text[:40] or "research_topic").strip("_")


def _build_research_plan(*, user_request: str, topic: str, titles: list[str], strategy: str) -> dict[str, Any]:
    sections = []
    for index, title in enumerate(titles):
        section_id = _section_id(title, index)
        axis = _section_axis(title, strategy)
        questions = _section_questions(title, strategy)
        sections.append({
            "id": section_id,
            "title": title,
            "axis": axis,
            "questions": questions,
            "success_criteria": _section_success_criteria(axis),
        })
    profile = _build_research_profile(user_request=user_request, sections=sections, strategy=strategy)
    return {
        "strategy": strategy,
        "strategy_profile": profile["profile"],
        "topic": topic,
        "user_request": user_request,
        "subjects": profile["subjects"],
        "comparison_axes": profile["comparison_axes"],
        "matrix_cells": profile["matrix_cells"],
        "judgement_questions": profile["judgement_questions"],
        "vertical": [item for item in sections if item["axis"] == "vertical"],
        "horizontal": [item for item in sections if item["axis"] == "horizontal"],
        "intersections": [
            "历史选择如何塑造当前生态位置",
            "当前优势或短板的形成原因",
            "未来可能演化路径和验证缺口",
        ] if strategy == "hv_analysis" else [],
        "sections": sections,
        "criterion_level_policy": CRITERION_LEVEL_POLICY_V1,
    }


def _build_research_profile(*, user_request: str, sections: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    subjects = _extract_comparison_subjects(user_request)
    axes = _extract_comparison_axes(user_request, sections)
    return _comparison_profile(subjects=subjects, axes=axes, strategy=strategy)


def _comparison_profile(*, subjects: list[str], axes: list[str], strategy: str) -> dict[str, Any]:
    is_multi_subject = len(subjects) >= 2 and bool(axes)
    matrix_cells: list[dict[str, Any]] = []
    if is_multi_subject:
        for subject in subjects[:5]:
            for axis in axes[:5]:
                cell_id = f"cell_{len(matrix_cells) + 1:03d}"
                matrix_cells.append({
                    "id": cell_id,
                    "subject": subject,
                    "axis": axis,
                    "question": f"{subject} 在“{axis}”上的公开证据是什么？",
                    "min_evidence": 1,
                    "min_sources": 1,
                })
    return {
        "profile": "multi_subject_compare" if is_multi_subject else strategy,
        "subjects": subjects if is_multi_subject else [],
        "comparison_axes": axes if is_multi_subject else [],
        "matrix_cells": matrix_cells,
        "judgement_questions": _comparison_judgement_questions(subjects, axes) if is_multi_subject else [],
    }


def _extract_comparison_subjects(user_request: str) -> list[str]:
    text = str(user_request or "")
    patterns = [
        r"对比调研\s+(.+?)\s+在",
        r"对比\s+(.+?)\s+在",
        r"比较\s+(.+?)\s+在",
        r"调研\s+(.+?)\s+的定位、差异",
        r"调研\s+(.+?)\s+在\s+.+?中",
    ]
    raw = ""
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1)
            break
    if not raw:
        return []
    raw = re.sub(r"^(?:一下|关于)?", "", raw).strip(" ：:，,。")
    parts = re.split(r"\s*(?:、|，|,|/| vs | VS | 与 | 和 |及|以及)\s*", raw)
    subjects = []
    for part in parts:
        value = part.strip(" ：:，,。")
        if value and len(value) <= 80 and value not in subjects:
            subjects.append(value)
    return subjects[:6]


def _extract_comparison_axes(user_request: str, sections: list[dict[str, Any]]) -> list[str]:
    text = str(user_request or "")
    raw = ""
    match = re.search(r"在(.+?)(?:上的差异|方面的差异|上的不同|方面|上)", text)
    if match:
        raw = match.group(1)
    axes: list[str] = []
    if raw:
        for part in re.split(r"\s*(?:、|，|,|/|与|和|及|以及)\s*", raw):
            value = part.strip(" ：:，,。")
            if value and len(value) <= 60 and value not in axes:
                axes.append(value)
    if axes:
        return axes[:5]
    fallback = []
    for section in sections:
        title = str(section.get("title") or "")
        if "对比" in title:
            fallback.extend(["架构/机制", "适用场景", "局限"])
        for question in section.get("questions") or []:
            value = _question_text(question)
            if value and value not in fallback:
                fallback.append(value)
    for value in fallback:
        if value and value not in axes and not _is_reference_or_overview_title(value):
            axes.append(value)
        if len(axes) >= 5:
            break
    return axes[:5]


def _comparison_judgement_questions(subjects: list[str], axes: list[str]) -> list[str]:
    subject_text = "、".join(subjects[:4])
    output = []
    if subject_text and axes:
        output.append(f"{subject_text} 在{axes[0]}上的关键差异是什么？")
    if len(axes) > 1 and subject_text:
        output.append(f"{subject_text} 在不同维度上的适用边界分别是什么？")
    output.append("哪些对比结论仍缺少直接公开证据？")
    return output[:3]


def _build_query_plan(
    *,
    user_request: str,
    research_plan: dict[str, Any],
    strategy: str,
    depth: str,
    extra_focus_terms: list[str] | None = None,
) -> dict[str, Any]:
    per_section = 4 if depth == "deep" else 3
    # `extra_focus_terms` carries reconnaissance-detected disambiguation qualifiers (see
    # _research_reconnaissance) — e.g. for "调研 harness 企业级实现方案", appending "AI agent"/
    # "智能体" keeps search results on the AI-agent-harness concept instead of drifting to the
    # unrelated, more heavily-indexed Harness.io CI/CD product that happens to share the name.
    focus_terms = list(dict.fromkeys(extra_focus_terms or []))
    buckets: list[list[dict[str, Any]]] = []
    matrix_cells = [item for item in research_plan.get("matrix_cells") or [] if isinstance(item, dict)]
    cell_queries: list[dict[str, Any]] = []
    if matrix_cells:
        section = _comparison_section(research_plan) or ((research_plan.get("sections") or [{}])[0])
        section_id = str(section.get("id") or "sec_02")
        section_title = str(section.get("title") or "02_横向对比")
        for cell in matrix_cells:
            subject = str(cell.get("subject") or "").strip()
            axis = str(cell.get("axis") or "").strip()
            if not subject or not axis:
                continue
            cell_queries.append({
                "section_id": section_id,
                "section_title": section_title,
                "question_id": f"{section_id}_{cell.get('id')}",
                "question": str(cell.get("question") or f"{subject} {axis}"),
                "query": _compose_search_query([subject, axis, *focus_terms, "evidence official documentation case study"]),
                "subject": subject,
                "comparison_axis": axis,
                "matrix_cell_id": cell.get("id"),
            })
    for section in research_plan.get("sections") or []:
        section_title = str(section.get("title") or "")
        bucket: list[dict[str, Any]] = []
        for question in (section.get("questions") or [])[:per_section]:
            question_text = _question_text(question)
            if not question_text:
                continue
            preferred_types = (
                _coerce_string_list(question.get("preferred_source_types"))
                if isinstance(question, dict)
                else []
            ) or _coerce_string_list(section.get("preferred_source_types"))
            source_terms = [item.replace("_", " ") for item in preferred_types]
            query = _compose_search_query([user_request, section_title, question_text, *focus_terms, *source_terms])
            bucket.append({
                "section_id": section.get("id"),
                "section_title": section_title,
                "question_id": (
                    str(question.get("id") or question.get("question_id") or "")
                    if isinstance(question, dict)
                    else ""
                ) or f"{section.get('id')}_q{len(bucket) + 1}",
                "question": question_text,
                "query": query,
            })
        if bucket:
            buckets.append(bucket)
    base_limit = 12 if strategy == "hv_analysis" and depth == "deep" else 6 if strategy == "hv_analysis" else 5
    limit = max(base_limit, len(cell_queries)) if cell_queries else base_limit
    queries = []
    queries.extend(cell_queries[:limit])
    for offset in range(per_section):
        for bucket in buckets:
            if offset < len(bucket):
                queries.append(bucket[offset])
            if len(queries) >= limit:
                break
        if len(queries) >= limit:
            break
    return {
        "strategy": strategy,
        "depth": depth,
        "focus_terms": focus_terms,
        "query_limit": limit,
        "wave_query_limit": int(
            (research_plan.get("comparison_budget") or {}).get("max_queries_per_wave") or limit
        ),
        "queries": queries[:limit],
    }


def _compose_search_query(parts: list[Any]) -> str:
    tokens: list[str] = []
    for part in parts:
        text = re.sub(r"\s+", " ", str(part or "")).strip()
        if text:
            tokens.append(text)
    return " ".join(_dedupe_strings(tokens))[:760]


def _comparison_section(research_plan: dict[str, Any]) -> dict[str, Any] | None:
    for section in research_plan.get("sections") or []:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "")
        axis = str(section.get("axis") or "")
        if axis == "horizontal" or "对比" in title:
            return section
    return None


def _section_id(title: str, index: int) -> str:
    match = re.match(r"^\s*(\d{2})[_\-]", str(title))
    return f"sec_{match.group(1)}" if match else f"sec_{index:02d}"


def _question_text(question: Any) -> str:
    if isinstance(question, dict):
        return str(question.get("text") or question.get("question") or question.get("title") or "").strip()
    return str(question or "").strip()


def _strip_title_prefix(title: str) -> str:
    text = str(title).strip()
    text = re.sub(r"^\d{2}[_\-]", "", text)
    return text.strip() or str(title)


def _section_axis(title: str, strategy: str) -> str:
    if strategy != "hv_analysis":
        return "standard"
    text = str(title)
    if "纵向" in text or "演化" in text:
        return "vertical"
    if "横向" in text or "对比" in text:
        return "horizontal"
    if "证据" in text or "参考" in text:
        return "evidence"
    if "实操" in text or "路径" in text:
        return "practice"
    if "机制" in text or "概念" in text:
        return "mechanism"
    return "overview"


def _section_questions(title: str, strategy: str) -> list[str]:
    if strategy != "hv_analysis":
        stripped = _strip_title_prefix(title)
        return [stripped, f"{stripped} 来源", f"{stripped} 实践"]
    axis = _section_axis(title, strategy)
    if axis == "vertical":
        return ["起源和定义", "关键时间线", "重要节点与转折", "爆发或变化原因"]
    if axis == "horizontal":
        return ["同类方案或相邻范式", "核心差异", "用户选择原因", "生态位置与优劣势"]
    if axis == "mechanism":
        return ["核心概念", "工作机制", "组成部分", "约束与边界"]
    if axis == "practice":
        return ["实践步骤", "工作流", "案例或模板", "风险和注意事项"]
    if axis == "evidence":
        return ["来源清单", "证据强弱", "未验证缺口", "后续阅读路线"]
    return ["定义", "背景", "关键结论", "价值与边界"]


def _section_success_criteria(axis: str) -> list[str]:
    common = ["至少 3 个独立来源", "至少 6 条正文级可追溯证据", "明确不确定性"]
    if axis == "vertical":
        return [*common, "包含起源、阶段和关键节点"]
    if axis == "horizontal":
        return [*common, "包含至少 2 个对比对象或相邻范式"]
    if axis == "evidence":
        return ["列出主要来源", "说明证据缺口", "不新增无证据结论"]
    return common


def _build_web_task(
    *,
    user_request: str,
    topic: str,
    titles: list[str],
    max_rounds: int,
    strategy: str,
    depth: str,
    research_plan: dict[str, Any],
    query_plan: dict[str, Any],
) -> str:
    query_lines = "\n".join(f"- [{item['section_id']}] {item['query']}" for item in query_plan.get("queries") or [])
    section_lines = "\n".join(
        f"- {item['title']} ({item['id']}): {'；'.join(_question_text(question) for question in (item.get('questions') or []) if _question_text(question))}；证据预算 {_target_section_evidence_count(str(item.get('title') or ''), item)} 条 / 来源 {_target_section_source_count(str(item.get('title') or ''), item)} 个"
        for item in research_plan.get("sections") or []
    )
    fetch_limit = 15 if strategy == "hv_analysis" and depth == "deep" else 8 if strategy == "hv_analysis" else 8
    structured_payload = {
        "research_plan": research_plan,
        "query_plan": query_plan,
        "required_titles": titles,
    }
    return (
        "source_mode=web 文档集调研。\n"
        f"用户需求：{user_request}\n"
        f"topic 使用：{topic}\n"
        f"research_strategy={strategy}\n"
        f"depth={depth}\n"
        f"fetch_url_limit={fetch_limit}\n"
        f"必须产出这些标题的知识库文档：{'、'.join(titles)}\n"
        "研究框架：\n"
        f"{section_lines}\n"
        "优先执行这些查询：\n"
        f"{query_lines}\n"
        "检索要求：必须分别覆盖每个非总览标题；不要只检索总主题。\n"
        "每个非总览标题至少需要 3 个相关来源、6 条正文级相关证据；总览至少 4 条证据，证据需带 source_url。\n"
        "正文级证据不能是标题、导航、目录、残句或泛泛概念，优先收集 API/代码/架构/流程/案例/评估/风险等可支撑长文分析的事实。\n"
        "证据要求：输出 evidence_pack/evidence_map 时必须包含 section_id、question_id，并优先来自 selected chunks。\n"
        "结构化研究计划 JSON（worker 必须优先使用，不要重新猜 section/questions）：\n"
        "<<<RESEARCH_PAYLOAD_JSON\n"
        f"{json.dumps(structured_payload, ensure_ascii=False)}\n"
        "RESEARCH_PAYLOAD_JSON>>>\n"
        f"最多调研轮次：{max_rounds}\n"
        "请优先查找官方文档、权威资料和可引用来源，输出 JSON evidence/source/documents。"
    )


def _build_followup_task(
    *,
    user_request: str,
    topic: str,
    titles: list[str],
    coverage: dict[str, Any],
    round_index: int,
    max_rounds: int,
) -> str:
    weak_sections = [
        item for item in coverage.get("weak_sections") or []
        if isinstance(item, dict)
    ]
    weak_titles = [
        str(item.get("title") or "")
        for item in coverage.get("per_title_status") or []
        if item.get("status") in {"thin", "missing"} and str(item.get("title") or "").strip()
    ]
    followups = coverage.get("followup_queries") or []
    weak_section_lines = [
        f"{item.get('title') or item.get('section_id')}: {item.get('reason') or '证据不足'}"
        for item in weak_sections[:3]
    ]
    targets = weak_section_lines or weak_titles or [str(item) for item in followups if str(item).strip()] or titles
    return (
        "source_mode=web 文档集补查。\n"
        f"用户需求：{user_request}\n"
        f"topic 使用：{topic}\n"
        f"当前轮次：{round_index}/{max_rounds}\n"
        f"必须保留文档标题：{'、'.join(titles)}\n"
        f"本轮只补查这些薄弱标题/查询：{'、'.join(targets[:3])}\n"
        f"优先查询：{'；'.join(str(item) for item in followups[:5])}\n"
        "请为每个薄弱标题寻找更直接的官方文档、API 文档、教程正文或权威资料。\n"
        "输出 evidence_pack/evidence_map 时必须在 suggested_titles 中标明对应标题，并保留 claim、evidence_text、source_url、section_id、question_id。"
    )


def _coverage_sufficient_for_draft(coverage: dict[str, Any]) -> bool:
    return bool(coverage.get("depth_ready")) and not coverage.get("fatal_gaps")


def _augment_coverage_with_evidence_diagnostics(
    *,
    coverage: dict[str, Any],
    retrieval: dict[str, Any],
    research_plan: dict[str, Any],
    user_request: str,
    round_index: int,
    max_rounds: int,
) -> dict[str, Any]:
    diagnostics = retrieval.get("evidence_diagnostics") or {}
    sections = diagnostics.get("sections") or []
    if not isinstance(sections, list) or not sections:
        return coverage
    section_meta = {
        str(section.get("id") or ""): section
        for section in research_plan.get("sections") or []
        if isinstance(section, dict)
    }
    weak_sections = []
    followups = [str(item) for item in coverage.get("followup_queries") or [] if str(item).strip()]
    for item in sections:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id") or "")
        meta = section_meta.get(section_id, {})
        title = str(item.get("title") or meta.get("title") or section_id)
        if _is_reference_or_overview_title(title):
            continue
        target_evidence = _target_section_evidence_count(title, meta)
        target_sources = _target_section_source_count(title, meta)
        body_table = _build_section_evidence_table(retrieval, meta or {"id": section_id, "title": title}, title)
        evidence_count = len(body_table) if body_table else int(item.get("evidence_count") or 0)
        source_count = len({str(row.get("source_url") or "") for row in body_table if row.get("source_url")}) if body_table else int(item.get("source_count") or 0)
        domain_count = len({str(row.get("source_domain") or "") for row in body_table if row.get("source_domain")}) if body_table else int(item.get("domain_count") or 0)
        question_count = int(item.get("question_count") or 0)
        covered_question_count = int(item.get("covered_question_count") or 0)
        question_rows = [
            row for row in item.get("question_rows") or []
            if isinstance(row, dict)
        ]
        matrix_rows = [
            row for row in item.get("matrix_rows") or []
            if isinstance(row, dict)
        ]
        weak_question_rows = _weak_question_rows(question_rows)
        weak_matrix_rows = _weak_matrix_rows(matrix_rows)
        risk_question_rows = _risk_single_source_question_rows(question_rows)
        risk_single_source_count = int(item.get("risk_single_source_count") or 0)
        reasons = []
        if evidence_count < target_evidence:
            reasons.append(f"evidence<{target_evidence}")
        if source_count < target_sources:
            reasons.append(f"source<{target_sources}")
        if domain_count < min(2, target_sources):
            reasons.append(f"domain<{min(2, target_sources)}")
        if question_count and covered_question_count < min(question_count, 2):
            reasons.append("question_coverage_low")
        if weak_question_rows:
            reasons.append("question_evidence<2")
        if weak_matrix_rows:
            reasons.append("matrix_cell_coverage_low")
        if risk_single_source_count and (source_count < 2 or domain_count < 2 or weak_question_rows):
            reasons.append("risk_claim_single_source")
        if not reasons:
            continue
        weak_sections.append({
            "section_id": section_id,
            "title": title,
            "evidence_count": evidence_count,
            "target_evidence_count": target_evidence,
            "source_count": source_count,
            "target_source_count": target_sources,
            "domain_count": domain_count,
            "question_count": question_count,
            "covered_question_count": covered_question_count,
            "weak_question_count": len(weak_question_rows),
            "weak_questions": weak_question_rows[:6],
            "weak_matrix_cell_count": len(weak_matrix_rows),
            "weak_matrix_cells": weak_matrix_rows[:8],
            "risk_single_source_count": risk_single_source_count,
            "risk_single_source_questions": risk_question_rows[:6],
            "reason": ",".join(reasons),
        })
        followup_questions = weak_matrix_rows or weak_question_rows or [
            {"question": _question_text(question)}
            for question in (meta.get("questions") or [_strip_title_prefix(title)])[:2]
        ]
        for question in followup_questions[:3]:
            question_text = str(
                question.get("question")
                or " ".join(
                    value for value in [
                        str(question.get("subject") or "").strip(),
                        str(question.get("comparison_axis") or "").strip(),
                    ]
                    if value
                )
            ).strip()
            if question_text:
                followups.append(f"{user_request} {title} {question_text} 权威资料 官方文档 案例".strip())
    if not weak_sections:
        return coverage
    next_coverage = dict(coverage)
    next_coverage["weak_sections"] = weak_sections
    if round_index < max_rounds:
        next_coverage["depth_ready"] = False
        next_coverage["ready_to_write"] = False
        next_coverage["followup_queries"] = _dedupe_strings(followups)[:6]
    next_coverage["reason"] = (
        str(coverage.get("reason") or "覆盖度检查完成")
        + "；证据诊断显示薄弱章节: "
        + "、".join(f"{item['title']}({item['reason']})" for item in weak_sections[:3])
    )
    return next_coverage


def _weak_question_rows(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = max(1, int(os.getenv("RESEARCH_TARGET_EVIDENCE_PER_QUESTION", "2")))
    output = []
    for row in question_rows:
        evidence_count = int(row.get("evidence_count") or 0)
        reasons = []
        if evidence_count < target:
            reasons.append(f"evidence<{target}")
        if not reasons:
            continue
        output.append({
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "evidence_count": evidence_count,
            "source_count": int(row.get("source_count") or 0),
            "domain_count": int(row.get("domain_count") or 0),
            "risk_flag_count": int(row.get("risk_flag_count") or 0),
            "reason": ",".join(reasons),
        })
    return output


def _weak_matrix_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not matrix_rows:
        return []
    min_evidence = max(1, int(os.getenv("RESEARCH_TARGET_EVIDENCE_PER_MATRIX_CELL", "1")))
    min_sources = max(1, int(os.getenv("RESEARCH_TARGET_SOURCES_PER_MATRIX_CELL", "1")))
    output = []
    for row in matrix_rows:
        evidence_count = int(row.get("evidence_count") or 0)
        source_count = int(row.get("source_count") or 0)
        reasons = []
        if evidence_count < min_evidence:
            reasons.append(f"evidence<{min_evidence}")
        if source_count < min_sources:
            reasons.append(f"source<{min_sources}")
        if not reasons:
            continue
        output.append({
            "matrix_cell_id": row.get("matrix_cell_id"),
            "subject": row.get("subject"),
            "comparison_axis": row.get("comparison_axis"),
            "evidence_count": evidence_count,
            "source_count": source_count,
            "domain_count": int(row.get("domain_count") or 0),
            "selected_chunk_count": int(row.get("selected_chunk_count") or 0),
            "reason": ",".join(reasons),
        })
    return output


def _risk_single_source_question_rows(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in question_rows:
        risk_count = int(row.get("risk_flag_count") or 0)
        source_count = int(row.get("source_count") or 0)
        if not risk_count or source_count >= 2:
            continue
        output.append({
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "evidence_count": int(row.get("evidence_count") or 0),
            "source_count": source_count,
            "domain_count": int(row.get("domain_count") or 0),
            "risk_flag_count": risk_count,
            "reason": "risk_single_source",
        })
    return output


def _is_reference_or_overview_title(title: str, *, has_questions: bool | None = None) -> bool:
    return is_non_research_section_title(title, has_questions=has_questions)


def _dedupe_strings(items: list[str]) -> list[str]:
    output = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _url_domain(url: str) -> str:
    try:
        parsed = urlparse(str(url))
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def _merge_retrieval(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    if not base:
        first = dict(incoming)
        if incoming.get("_diagnostics"):
            first["_diagnostics"] = {**incoming["_diagnostics"], "rounds": [incoming["_diagnostics"]]}
        return first
    merged = dict(base)
    for key in (
        "sources",
        "documents",
        "materials",
        "selected_chunks",
        "evidences",
        "key_facts",
        "evidence_pack",
        "evidence_map",
        "gaps",
        "provider_attempts",
    ):
        merged[key] = _dedupe_retrieval_items([
            *_coerce_list(base.get(key)),
            *_coerce_list(incoming.get(key)),
        ])
    if incoming.get("suggested_next_step"):
        merged["suggested_next_step"] = incoming.get("suggested_next_step")
    if incoming.get("_diagnostics"):
        previous = (base.get("_diagnostics") or {}).get("rounds") or []
        merged["_diagnostics"] = {**incoming["_diagnostics"], "rounds": [*previous, incoming["_diagnostics"]]}
    return merged


def _dedupe_retrieval_items(items: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = _retrieval_item_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _retrieval_item_key(item: Any) -> str:
    if isinstance(item, dict):
        item_id = str(item.get("id") or item.get("evidence_id") or "").strip()
        if item_id:
            return f"id:{item_id}"
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id:
            return f"chunk_id:{chunk_id}"
        material_id = str(item.get("material_id") or "").strip()
        chunk_index = item.get("chunk_index")
        section_id = str(item.get("section_id") or "").strip()
        if material_id and chunk_index is not None:
            return f"chunk:{material_id}:{chunk_index}:{section_id}"
        for key in ("url", "source_url", "claim", "evidence_text", "title"):
            value = str(item.get(key) or "").strip()
            if value:
                return f"{key}:{value}"
        return json.dumps(item, ensure_ascii=False, sort_keys=True)[:500]
    return str(item)[:500]


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
        timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
    )


def _invoke_legacy_research_model(payload: dict[str, Any], *, call_kind: str = "legacy_model") -> Any:
    messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    model = _model()
    with payload_audit_scope(payload, fallback_stage=call_kind):
        return invoke_with_model_trace(
            lambda: model.invoke(messages),
            messages=messages,
            requested_provider="primary",
            requested_model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            actual_provider="primary",
            call_kind=call_kind,
            parameters={
                "temperature": 0,
                "timeout_seconds": float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
                "base_url": os.getenv("OPENAI_BASE_URL") or None,
            },
        )


def _synthesize_documents(
    user_request: str,
    topic: str,
    titles: list[str],
    retrieval: dict[str, Any],
    coverage: dict[str, Any],
    research_plan: dict[str, Any] | None = None,
    report_blueprint: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    normalized = []
    sections = {str(item.get("title") or ""): item for item in (research_plan or {}).get("sections") or []}
    blueprint_sections = {
        str(item.get("title") or ""): item
        for item in (report_blueprint or {}).get("sections") or []
        if isinstance(item, dict)
    }
    for title in titles:
        section = sections.get(title) or {"id": _section_id(title, len(normalized)), "title": title, "questions": []}
        section_blueprint = blueprint_sections.get(title) or _section_report_blueprint(
            title=title,
            section=section,
            report_type=str((report_blueprint or {}).get("report_type") or "research_brief"),
        )
        doc = _synthesize_evidence_bound_document(
            user_request=user_request,
            topic=topic,
            title=title,
            section=section,
            retrieval=retrieval,
            coverage=coverage,
            section_blueprint=section_blueprint,
        )
        normalized.append(doc)
    return normalized


def _build_report_blueprint(
    *,
    user_request: str,
    research_strategy: str,
    titles: list[str],
    research_plan: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    report_type = _infer_report_type(user_request=user_request, research_strategy=research_strategy, research_plan=research_plan)
    section_map = {
        str(section.get("title") or ""): section
        for section in research_plan.get("sections") or []
        if isinstance(section, dict)
    }
    sections = [
        _section_report_blueprint(
            title=title,
            section=section_map.get(title) or {"title": title},
            report_type=report_type,
        )
        for title in titles
    ]
    evidence_count = len(_coerce_list(retrieval.get("evidence_map")) or _coerce_list(retrieval.get("evidence_pack")))
    return {
        "report_type": report_type,
        "audience": _infer_report_audience(user_request),
        "style": "dynamic_evidence_bound_report",
        "evidence_count": evidence_count,
        "sections": sections,
        "appendix": ["sources", "evidence_table", "validation"],
    }


def _infer_report_type(*, user_request: str, research_strategy: str, research_plan: dict[str, Any]) -> str:
    text = " ".join([
        user_request,
        research_strategy,
        " ".join(str(section.get("title") or "") for section in research_plan.get("sections") or [] if isinstance(section, dict)),
    ]).lower()
    if research_strategy == "hv_analysis" or any(token in text for token in ["对比", "比较", "vs", "横向", "竞品"]):
        return "comparison"
    if any(token in text for token in ["实现", "开发", "搭建", "架构", "方案", "落地", "agent"]):
        return "implementation_guide"
    if any(token in text for token in ["问题", "原因", "诊断", "为什么", "失败", "优化", "质量"]):
        return "diagnosis"
    if any(token in text for token in ["趋势", "市场", "行业", "玩家", "格局"]):
        return "trend"
    if any(token in text for token in ["论文", "文献", "研究综述", "综述"]):
        return "literature_review"
    return "research_brief"


def _infer_report_audience(user_request: str) -> str:
    text = str(user_request or "")
    if any(token in text for token in ["实现", "开发", "架构", "代码", "工程"]):
        return "工程实现与方案评审"
    if any(token in text for token in ["对比", "选型", "选择", "采购"]):
        return "决策与选型"
    if any(token in text for token in ["学习", "指南", "教程"]):
        return "学习与知识沉淀"
    return "研究阅读与审核"


def _section_report_blueprint(*, title: str, section: dict[str, Any], report_type: str) -> dict[str, Any]:
    text = " ".join([
        title,
        str(section.get("intent") or ""),
        " ".join(_question_text(question) for question in section.get("questions") or []),
    ]).lower()
    render_as = "narrative"
    if "总览" in title or "overview" in text:
        render_as = "executive_summary"
    elif str(section.get("axis") or "") == "horizontal" or report_type == "comparison" or any(token in text for token in ["对比", "比较", "差异", "选择"]):
        render_as = "comparison_table"
    elif any(token in text for token in ["架构", "模块", "组成", "角色", "分工", "数据流", "通信"]):
        render_as = "architecture_table"
    elif any(token in text for token in ["路线", "实施", "落地", "步骤", "mvp", "roadmap"]):
        render_as = "roadmap"
    elif any(token in text for token in ["风险", "挑战", "限制", "局限", "验证", "评估"]):
        render_as = "risk_matrix"
    purpose_by_render = {
        "executive_summary": "先给出可读结论，再标明证据边界。",
        "comparison_table": "用对比矩阵承载差异，再补充解释。",
        "architecture_table": "用结构表拆解模块、职责和证据。",
        "roadmap": "用路线图组织阶段、动作和依据。",
        "risk_matrix": "用风险矩阵展示问题、影响和依据。",
        "narrative": "按研究问题组织证据约束分析。",
    }
    budget = section.get("evidence_budget") if isinstance(section.get("evidence_budget"), dict) else {}
    return {
        "title": title,
        "purpose": purpose_by_render.get(render_as, purpose_by_render["narrative"]),
        "render_as": render_as,
        "evidence_requirements": {
            "min_evidence": _target_section_evidence_count(title, section),
            "min_sources": _target_section_source_count(title, section),
        },
    }


def _synthesize_evidence_bound_document(
    *,
    user_request: str,
    topic: str,
    title: str,
    section: dict[str, Any],
    retrieval: dict[str, Any],
    coverage: dict[str, Any],
    section_blueprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    table = _build_section_evidence_table(retrieval, section, title)
    if not table:
        return _synthesize_minimal_gap_document(title=title, section=section, coverage=coverage)

    source_lines = _document_source_lines(table)
    question_groups = _evidence_groups_by_question(table, section)
    matrix_groups = _evidence_groups_by_matrix(table)
    section_blueprint = section_blueprint or _section_report_blueprint(
        title=title,
        section=section,
        report_type="research_brief",
    )
    group_blocks: list[dict[str, Any]] = []
    body_parts = [
        f"# {title}",
        "",
        *_report_lead_markdown(title=title, table=table, section_blueprint=section_blueprint),
        "",
    ]
    structure_block = _structured_report_markdown(
        section_blueprint=section_blueprint,
        table=table,
        question_groups=question_groups,
        matrix_groups=matrix_groups,
    )
    if structure_block:
        body_parts.extend(structure_block)
        body_parts.append("")
    body_parts.extend(["## 详细分析", ""])
    if matrix_groups and str(section.get("axis") or "") == "horizontal":
        matrix_block = _matrix_evidence_markdown(matrix_groups)
        group_blocks.extend(matrix_block["group_blocks"])
        body_parts.extend(matrix_block["markdown"])
    else:
        for group in question_groups:
            group_block = _evidence_bound_group_markdown(group)
            group_blocks.append(group_block)
            body_parts.extend(group_block["markdown"])
    body_parts.extend([
        "## 审计附录",
        "",
        "### 来源",
        "",
        *source_lines,
        "",
        "### 证据边界",
        "",
        _evidence_bound_boundary_text(table),
        "",
        "### 证据索引",
        "",
        *_evidence_index_lines(table),
    ])
    content = "\n".join(body_parts).strip()
    synthesis_diagnostics = _document_synthesis_diagnostics(
        title=title,
        table=table,
        group_blocks=group_blocks,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return {
        "title": title,
        "content": content,
        "evidence_refs": [_evidence_ref(row) for row in table],
        "evidence_table": table,
        "paragraph_validation": _document_paragraph_validation(group_blocks),
        "evidence_metrics": _document_evidence_metrics(table, group_blocks),
        "synthesis_diagnostics": synthesis_diagnostics,
        "report_blueprint": section_blueprint,
    }


def _build_section_evidence_table(retrieval: dict[str, Any], section: dict[str, Any], title: str) -> list[dict[str, Any]]:
    section_id = str(section.get("id") or "")
    evidence_items = _section_items(_coerce_list(retrieval.get("evidence_map")), section_id, title)
    if not evidence_items:
        evidence_items = _section_items(_coerce_list(retrieval.get("evidence_pack")), section_id, title)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(evidence_items):
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id") or item.get("evidence_id") or "").strip() or f"ev_{index + 1:03d}"
        claim = str(item.get("claim") or item.get("fact") or "").strip()
        quote = str(item.get("evidence_text") or item.get("evidence") or item.get("text") or "").strip()
        source_url = str(item.get("source_url") or item.get("url") or "").strip()
        if not claim and quote:
            claim = _sentence_from_text(quote)
        claim = _clean_evidence_claim(claim)
        if not quote and claim:
            quote = claim
        if not claim or not source_url:
            continue
        quality_issue = _body_worthy_evidence_issue(
            claim=claim,
            quote=quote,
            source_url=source_url,
            section_title=str(section.get("title") or title),
            question=str(item.get("question") or ""),
        )
        if quality_issue:
            continue
        key = f"{evidence_id}:{source_url}:{claim[:80]}"
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "id": evidence_id,
            "section_id": str(item.get("section_id") or section_id),
            "section_title": str(section.get("title") or title),
            "question_id": str(item.get("question_id") or ""),
            "question": str(item.get("question") or ""),
            "subject": str(item.get("subject") or ""),
            "comparison_axis": str(item.get("comparison_axis") or item.get("axis") or ""),
            "matrix_cell_id": str(item.get("matrix_cell_id") or ""),
            "claim": claim[:280],
            "quote": quote[:900],
            "source_url": source_url,
            "source_title": str(item.get("source_title") or item.get("title") or source_url)[:180],
            "source_domain": _url_domain(source_url),
            "confidence": str(item.get("confidence") or "medium"),
            "chunk_id": str(item.get("chunk_id") or ""),
            "used_in_document": True,
            "risk_flags": _evidence_risk_flags(
                claim=claim,
                quote=quote,
                confidence=str(item.get("confidence") or "medium"),
                source_url=source_url,
            ),
        })
    return rows[:32]


def _body_worthy_evidence_issue(
    *,
    claim: str,
    quote: str,
    source_url: str,
    section_title: str,
    question: str = "",
) -> str:
    claim_text = _clean_evidence_sentence(claim)
    quote_text = _clean_evidence_sentence(quote)
    combined = f"{claim_text} {quote_text}".strip()
    if not claim_text or not quote_text:
        return "empty_claim_or_quote"
    if len(claim_text) < 10 or len(quote_text) < 16:
        return "too_short"
    if _looks_like_heading_or_question(claim_text):
        return "title_or_question_like_claim"
    if _looks_like_navigation_text(quote_text):
        return "navigation_or_toc_text"
    if _looks_like_markdown_heading(quote_text) or _looks_like_markdown_heading(claim_text):
        return "markdown_heading"
    if not _url_domain(source_url):
        return "missing_source_domain"
    return ""


def _looks_like_heading_or_question(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if value.endswith(("?", "？")):
        return True
    if len(value) <= 42 and not re.search(r"[。.!；;，,：:]", value):
        heading_markers = ("什么", "如何", "为何", "为什么", "什么时候", "哪些", "指南", "概览", "介绍", "目录", "参考")
        return any(marker in value for marker in heading_markers)
    return False


def _looks_like_navigation_text(text: str) -> bool:
    value = str(text or "").lower()
    nav_markers = [
        "toggle", "subscribe", "cookie", "read more", "sign in", "login", "menu",
        "目录", "首页", "导航", "上一篇", "下一篇", "返回顶部", "版权", "扫码", "关注公众号",
    ]
    marker_count = sum(1 for marker in nav_markers if marker in value)
    return marker_count >= 2 and len(value) < 260


def _looks_like_markdown_heading(text: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S+", str(text or "")))


def _topic_overlap_count(text: str, target: str) -> int:
    text_tokens = set(_text_tokens(text))
    target_tokens = {token for token in _text_tokens(target) if len(token) >= 2}
    if not target_tokens:
        return 1
    return len(text_tokens & target_tokens)


def _has_substantive_technical_signal(text: str) -> bool:
    value = str(text or "").lower()
    signals = [
        "api", "sdk", "rag", "mcp", "agent", "workflow", "architecture", "database", "vector",
        "github", "代码", "架构", "流程", "模块", "工具", "调用", "评估", "成本", "风险", "案例",
        "数据流", "状态", "记忆", "权限", "集成", "失败", "降级", "指标",
    ]
    return any(signal in value for signal in signals)


def _report_lead_markdown(*, title: str, table: list[dict[str, Any]], section_blueprint: dict[str, Any]) -> list[str]:
    render_as = str(section_blueprint.get("render_as") or "narrative")
    lines = [
        "## 本章结论",
        "",
    ]
    for row in table[:4]:
        claim = _clean_evidence_sentence(str(row.get("claim") or ""))
        evidence_id = str(row.get("id") or "")
        if claim and evidence_id:
            lines.append(f"- {claim} [{evidence_id}]")
    if len(table) > 4:
        lines.append("- 更多证据详见下方分析与审计附录。")
    return lines


def _structured_report_markdown(
    *,
    section_blueprint: dict[str, Any],
    table: list[dict[str, Any]],
    question_groups: list[dict[str, Any]],
    matrix_groups: list[dict[str, Any]],
) -> list[str]:
    render_as = str(section_blueprint.get("render_as") or "narrative")
    if render_as == "comparison_table":
        return _comparison_report_table(matrix_groups=matrix_groups, question_groups=question_groups, table=table)
    if render_as == "architecture_table":
        return _group_report_table(
            title="## 架构拆解",
            headers=["模块/主题", "证据化职责或机制", "引用"],
            question_groups=question_groups,
            table=table,
        )
    if render_as == "roadmap":
        return _group_report_table(
            title="## 实施路线图",
            headers=["阶段/任务", "证据化动作", "引用"],
            question_groups=question_groups,
            table=table,
        )
    if render_as == "risk_matrix":
        return _group_report_table(
            title="## 风险与验证矩阵",
            headers=["风险/问题", "证据化影响或约束", "引用"],
            question_groups=question_groups,
            table=table,
        )
    if render_as == "executive_summary":
        return _executive_summary_table(table)
    return []


def _comparison_report_table(
    *,
    matrix_groups: list[dict[str, Any]],
    question_groups: list[dict[str, Any]],
    table: list[dict[str, Any]],
) -> list[str]:
    rows: list[list[str]] = []
    if matrix_groups:
        for group in matrix_groups:
            subject = str(group.get("subject") or "对象")
            axis = str(group.get("comparison_axis") or "维度")
            evidence_rows = [row for row in group.get("rows") or [] if isinstance(row, dict)]
            rows.append([subject, axis, _joined_claims(evidence_rows), _joined_refs(evidence_rows)])
    else:
        for group in question_groups:
            evidence_rows = [row for row in group.get("rows") or [] if isinstance(row, dict)]
            rows.append([str(group.get("title") or "主题"), "研究问题", _joined_claims(evidence_rows), _joined_refs(evidence_rows)])
    if not rows:
        rows = [["-", "-", _joined_claims(table[:3]), _joined_refs(table[:3])]]
    return _markdown_table("## 对比矩阵", ["对象/主题", "维度", "证据化结论", "引用"], rows)


def _group_report_table(
    *,
    title: str,
    headers: list[str],
    question_groups: list[dict[str, Any]],
    table: list[dict[str, Any]],
) -> list[str]:
    rows: list[list[str]] = []
    for group in question_groups:
        evidence_rows = [row for row in group.get("rows") or [] if isinstance(row, dict)]
        rows.append([
            str(group.get("title") or "主题"),
            _joined_claims(evidence_rows),
            _joined_refs(evidence_rows),
        ])
    if not rows:
        rows.append(["-", _joined_claims(table[:3]), _joined_refs(table[:3])])
    return _markdown_table(title, headers, rows)


def _executive_summary_table(table: list[dict[str, Any]]) -> list[str]:
    rows = [
        [
            _clean_evidence_sentence(str(row.get("claim") or "")),
            str(row.get("source_title") or row.get("source_domain") or ""),
            f"[{row.get('id')}]",
        ]
        for row in table[:6]
    ]
    return _markdown_table("## 关键证据摘要", ["结论", "来源", "引用"], rows)


def _markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [title, ""] if title else []
    lines.append("| " + " | ".join(_escape_table_cell(header) for header in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        padded = [*row, *[""] * max(0, len(headers) - len(row))]
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in padded[: len(headers)]) + " |")
    return lines


def _joined_claims(rows: list[dict[str, Any]]) -> str:
    claims = []
    for row in rows[:3]:
        claim = _clean_evidence_sentence(str(row.get("claim") or ""))
        evidence_id = str(row.get("id") or "")
        if claim and evidence_id:
            claims.append(f"{claim} [{evidence_id}]")
    return "；".join(claims) or "当前证据不足。"


def _joined_refs(rows: list[dict[str, Any]]) -> str:
    refs = [f"[{row.get('id')}]" for row in rows[:4] if row.get("id")]
    return " ".join(refs) or "-"


def _escape_table_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|")
    return re.sub(r"\s+", " ", text).strip()


def _synthesize_minimal_gap_document(
    *,
    title: str,
    section: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    reason = str(coverage.get("reason") or "当前章节没有可绑定证据。")
    content = "\n".join([
        f"# {title}",
        "",
        "## 来源",
        "",
        "- 暂无可绑定来源。",
        "",
        "## 证据状态",
        "",
        f"当前章节未形成可审计证据表，原因：{reason}",
        "",
        "## 边界",
        "",
        f"章节 `{section.get('id') or title}` 暂不生成事实性结论。",
    ])
    return {
        "title": title,
        "content": content,
        "evidence_refs": [],
        "synthesis_diagnostics": {
            "title": title,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "evidence_count": 0,
            "source_count": 0,
            "group_count": 0,
            "groups": [],
        },
    }


def _document_source_lines(table: list[dict[str, Any]]) -> list[str]:
    lines = []
    seen: set[str] = set()
    for row in table:
        url = str(row.get("source_url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(row.get("source_title") or url)
        lines.append(f"- {title}: {url}")
    return lines or ["- 暂无可绑定来源。"]


def _evidence_groups_by_question(table: list[dict[str, Any]], section: dict[str, Any]) -> list[dict[str, Any]]:
    question_labels = _question_labels(section)
    grouped: dict[str, dict[str, Any]] = {}
    for row in table:
        question_id = str(row.get("question_id") or "")
        label = question_labels.get(question_id) or str(row.get("question") or "").strip() or "关键证据"
        bucket = grouped.setdefault(question_id or label, {"question_id": question_id, "title": label, "rows": []})
        bucket["rows"].append(row)
    return list(grouped.values())


def _evidence_groups_by_matrix(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in table:
        subject = str(row.get("subject") or "").strip()
        axis = str(row.get("comparison_axis") or "").strip()
        cell_id = str(row.get("matrix_cell_id") or "").strip()
        if not subject or not axis:
            continue
        key = (axis, subject, cell_id)
        bucket = grouped.setdefault(
            key,
            {
                "title": f"{subject} - {axis}",
                "subject": subject,
                "comparison_axis": axis,
                "matrix_cell_id": cell_id,
                "rows": [],
            },
        )
        bucket["rows"].append(row)
    return list(grouped.values())


def _question_labels(section: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, question in enumerate(section.get("questions") or []):
        if isinstance(question, dict):
            question_id = str(question.get("id") or question.get("question_id") or f"{section.get('id')}_q{index + 1}")
            labels[question_id] = _question_text(question)
        else:
            labels[f"{section.get('id')}_q{index + 1}"] = str(question or "").strip()
    return labels


def _matrix_evidence_markdown(matrix_groups: list[dict[str, Any]]) -> dict[str, Any]:
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for group in matrix_groups:
        axis = str(group.get("comparison_axis") or "对比维度")
        by_axis.setdefault(axis, []).append(group)
    markdown: list[str] = []
    group_blocks: list[dict[str, Any]] = []
    for axis, groups in by_axis.items():
        markdown.extend([f"### {axis}", ""])
        for group in sorted(groups, key=lambda item: str(item.get("subject") or "")):
            started = time.perf_counter()
            rows = [row for row in group.get("rows") or [] if isinstance(row, dict)]
            subject = str(group.get("subject") or "对象")
            sentence = _matrix_subject_sentence(subject, rows)
            validation = _validate_evidence_bound_paragraph(sentence, rows)
            block = {
                "title": f"{subject} - {axis}",
                "rows": rows,
                "paragraph": sentence,
                "composition": {
                    "mode": "matrix_deterministic",
                    "row_count": len(rows),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
                "validation": validation,
                "markdown": [sentence, ""],
            }
            group_blocks.append(block)
            markdown.extend(block["markdown"])
    return {"markdown": markdown, "group_blocks": group_blocks}


def _matrix_subject_sentence(subject: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"- {subject}: 当前没有可绑定证据。"
    snippets = []
    for row in rows:
        claim = _clean_evidence_sentence(str(row.get("claim") or ""))
        evidence_id = str(row.get("id") or "")
        snippets.append(f"{claim} [{evidence_id}]")
    return f"- {subject}: " + "；".join(snippets) + "。"


def _evidence_bound_group_markdown(group: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    title = str(group.get("title") or "关键证据").strip() or "关键证据"
    rows = [row for row in group.get("rows") or [] if isinstance(row, dict)]
    parts = [f"### {title}", ""]
    composition = _compose_evidence_bound_paragraph(group)
    paragraph = composition["paragraph"]
    validation = _validate_evidence_bound_paragraph(paragraph, rows)
    if validation["status"] != "passed":
        composition = {
            **composition,
            "mode": "fallback_after_validation",
            "fallback_reason": validation.get("reason"),
        }
        paragraph = "\n\n".join(_evidence_bound_paragraph(row) for row in rows)
        validation = _validate_evidence_bound_paragraph(paragraph, rows)
    parts.append(paragraph)
    parts.append("")
    if validation["status"] != "passed":
        for row in rows:
            parts.append(_evidence_bound_paragraph(row))
            parts.append("")
    return {
        "title": title,
        "rows": rows,
        "paragraph": paragraph,
        "composition": {
            **composition,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
        "validation": validation,
        "markdown": parts,
    }


def _compose_evidence_bound_paragraph(group: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in group.get("rows") or [] if isinstance(row, dict)]
    if not rows:
        return {"paragraph": "", "mode": "empty", "row_count": 0}
    fallback = "\n\n".join(_evidence_bound_paragraph(row) for row in rows)
    if os.getenv("RESEARCH_PARAGRAPH_COMPOSER", "1") == "0":
        return {"paragraph": fallback, "mode": "fallback_disabled", "row_count": len(rows)}
    compact_rows = [
        {
            "id": row.get("id"),
            "claim": row.get("claim"),
            "quote": row.get("quote"),
            "source_title": row.get("source_title"),
            "source_url": row.get("source_url"),
        }
        for row in rows[:3]
    ]
    prompt = {
        "task": "把 evidence rows 改写成一段更自然的中文研究小结。",
        "rules": [
            "只返回 JSON，不要解释，形状为 {\"paragraph\":\"...\"}。",
            "只能使用 rows 中已有 claim/quote/source 信息。",
            "不得新增数字、年份、版本、产品能力、建议或外部判断。",
            "每个被使用的 evidence id 必须以内联形式出现，例如 [ev_001]。",
            "如果无法合成，直接逐条改写，但仍保留 evidence id。",
        ],
        "question": group.get("title"),
        "rows": compact_rows,
    }
    data = _invoke_research_json(
        prompt,
        timeout_seconds=float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
        temperature=float(os.getenv("RESEARCH_WRITER_TEMPERATURE", "0.3")),
        system_prompt=_WRITER_SYSTEM_PROMPT,
        model_name=os.getenv("RESEARCH_WRITER_MODEL_NAME") or None,
        evidence=False,
        allow_provider_fallback=True,
        fallback_stage="paragraph_composer",
    )
    paragraph = str(data.get("paragraph") or "").strip()
    if paragraph:
        return {"paragraph": paragraph, "mode": "model_composed", "row_count": len(rows)}
    failure_reason = str(data.get("_failure_reason") or "empty_model_output")
    if failure_reason != "empty_model_output":
        return {
            "paragraph": fallback,
            "mode": "fallback_exception",
            "row_count": len(rows),
            "fallback_reason": failure_reason,
            "provider_fallback_attempted": bool(data.get("_provider_fallback_attempted")),
        }
    return {
        "paragraph": fallback,
        "mode": "fallback_empty_model_output",
        "row_count": len(rows),
    }


def _validate_evidence_bound_paragraph(paragraph: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(paragraph or "")
    allowed_ids = {str(row.get("id") or "").strip() for row in rows if row.get("id")}
    used_ids = set(re.findall(r"\[(ev_\d{3,4})\]", text, re.I))
    unknown_ids = sorted(used_ids - allowed_ids)
    missing_ids = sorted(allowed_ids - used_ids)
    unsupported = _unsupported_strong_claims(text)
    leaked_urls = [
        url for url in re.findall(r"https?://[^\s)\]>'\"]+", text)
        if url not in {str(row.get("source_url") or "") for row in rows}
    ]
    reasons = []
    if unknown_ids:
        reasons.append("unknown_evidence_ids")
    if missing_ids:
        reasons.append("missing_evidence_ids")
    if unsupported:
        reasons.append("unsupported_strong_claims")
    if leaked_urls:
        reasons.append("unknown_urls")
    return {
        "status": "passed" if not reasons else "failed",
        "reason": "ok" if not reasons else ",".join(reasons),
        "used_ids": sorted(used_ids),
        "missing_ids": missing_ids,
        "unknown_ids": unknown_ids,
        "unsupported_claims": unsupported[:4],
        "unknown_urls": leaked_urls[:4],
    }


def _document_paragraph_validation(group_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    groups = [
        {
            "title": block.get("title"),
            **(block.get("validation") or {}),
        }
        for block in group_blocks
    ]
    failed = [item for item in groups if item.get("status") != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "failed_group_count": len(failed),
        "composer": _document_composer_metrics(group_blocks),
        "groups": groups,
    }


def _document_composer_metrics(group_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, int] = {}
    fallback_reasons: list[dict[str, Any]] = []
    elapsed_values: list[int] = []
    slow_groups: list[dict[str, Any]] = []
    for block in group_blocks:
        composition = block.get("composition") or {}
        mode = str(composition.get("mode") or "unknown")
        modes[mode] = modes.get(mode, 0) + 1
        elapsed_ms = int(composition.get("elapsed_ms") or 0)
        elapsed_values.append(elapsed_ms)
        slow_groups.append({
            "title": block.get("title"),
            "mode": mode,
            "elapsed_ms": elapsed_ms,
            "row_count": composition.get("row_count") or len(block.get("rows") or []),
            "validation_status": (block.get("validation") or {}).get("status"),
            "validation_reason": (block.get("validation") or {}).get("reason"),
        })
        if "fallback" in mode:
            fallback_reasons.append({
                "title": block.get("title"),
                "mode": mode,
                "reason": composition.get("fallback_reason") or "",
            })
    slow_groups.sort(key=lambda item: int(item.get("elapsed_ms") or 0), reverse=True)
    return {
        "group_count": len(group_blocks),
        "modes": modes,
        "fallback_count": sum(count for mode, count in modes.items() if "fallback" in mode),
        "model_composed_count": modes.get("model_composed", 0),
        "fallback_reasons": fallback_reasons[:6],
        "elapsed_ms_total": sum(elapsed_values),
        "elapsed_ms_max": max(elapsed_values) if elapsed_values else 0,
        "slow_groups": slow_groups[:6],
    }


def _document_synthesis_diagnostics(
    *,
    title: str,
    table: list[dict[str, Any]],
    group_blocks: list[dict[str, Any]],
    elapsed_ms: int,
) -> dict[str, Any]:
    source_urls = {str(row.get("source_url") or "") for row in table if row.get("source_url")}
    groups = []
    for block in group_blocks:
        composition = block.get("composition") or {}
        validation = block.get("validation") or {}
        groups.append({
            "title": block.get("title"),
            "mode": composition.get("mode"),
            "row_count": composition.get("row_count") or len(block.get("rows") or []),
            "elapsed_ms": int(composition.get("elapsed_ms") or 0),
            "validation_status": validation.get("status"),
            "validation_reason": validation.get("reason"),
            "fallback_reason": composition.get("fallback_reason"),
        })
    groups.sort(key=lambda item: int(item.get("elapsed_ms") or 0), reverse=True)
    return {
        "title": title,
        "elapsed_ms": elapsed_ms,
        "evidence_count": len(table),
        "source_count": len(source_urls),
        "group_count": len(group_blocks),
        "composer": _document_composer_metrics(group_blocks),
        "groups": groups,
    }


def _document_evidence_metrics(table: list[dict[str, Any]], group_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    used_ids = {
        evidence_id
        for block in group_blocks
        for evidence_id in (block.get("validation") or {}).get("used_ids", [])
    }
    source_urls = {str(row.get("source_url") or "") for row in table if row.get("source_url")}
    domains = {str(row.get("source_domain") or "") for row in table if row.get("source_domain")}
    questions = {str(row.get("question_id") or row.get("question") or "") for row in table if row.get("question_id") or row.get("question")}
    risk_count = sum(len(row.get("risk_flags") or []) for row in table)
    question_rows = _document_question_evidence_rows(table)
    weak_questions = _weak_question_rows(question_rows)
    matrix_rows = _document_matrix_evidence_rows(table)
    weak_matrix_rows = _weak_matrix_rows(matrix_rows)
    risk_by_flag: dict[str, int] = {}
    for row in table:
        for flag in row.get("risk_flags") or []:
            risk_by_flag[str(flag)] = risk_by_flag.get(str(flag), 0) + 1
    return {
        "evidence_count": len(table),
        "body_worthy_evidence_count": len(table),
        "target_evidence_count": _target_section_evidence_count(str(table[0].get("section_title") or "") if table else ""),
        "target_source_count": _target_section_source_count(str(table[0].get("section_title") or "") if table else ""),
        "source_count": len(source_urls),
        "domain_count": len(domains),
        "question_count": len(questions),
        "used_evidence_count": len(used_ids),
        "risk_flag_count": risk_count,
        "risk_flags": risk_by_flag,
        "question_rows": question_rows,
        "weak_question_count": len(weak_questions),
        "weak_questions": weak_questions[:6],
        "matrix_rows": matrix_rows,
        "matrix_cell_count": len(matrix_rows),
        "covered_matrix_cell_count": sum(1 for row in matrix_rows if int(row.get("evidence_count") or 0) > 0),
        "weak_matrix_cell_count": len(weak_matrix_rows),
        "weak_matrix_cells": weak_matrix_rows[:8],
        "subject_count": len({str(row.get("subject") or "") for row in matrix_rows if row.get("subject")}),
        "axis_count": len({str(row.get("comparison_axis") or "") for row in matrix_rows if row.get("comparison_axis")}),
        "composer": _document_composer_metrics(group_blocks),
        "paragraph_validation_status": "passed" if all((block.get("validation") or {}).get("status") == "passed" for block in group_blocks) else "failed",
    }


def _document_question_evidence_rows(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in table:
        key = str(row.get("question_id") or row.get("question") or "关键证据")
        bucket = grouped.setdefault(
            key,
            {
                "question_id": row.get("question_id"),
                "question": row.get("question") or row.get("question_id") or "关键证据",
                "evidence_ids": set(),
                "source_urls": set(),
                "domains": set(),
                "risk_flags": [],
            },
        )
        if row.get("id"):
            bucket["evidence_ids"].add(str(row.get("id")))
        if row.get("source_url"):
            bucket["source_urls"].add(str(row.get("source_url")))
        if row.get("source_domain"):
            bucket["domains"].add(str(row.get("source_domain")))
        bucket["risk_flags"].extend(str(flag) for flag in row.get("risk_flags") or [])
    rows = []
    for item in grouped.values():
        risk_flags = sorted(set(item["risk_flags"]))
        rows.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "evidence_count": len(item["evidence_ids"]),
            "source_count": len(item["source_urls"]),
            "domain_count": len(item["domains"]),
            "risk_flag_count": len(item["risk_flags"]),
            "risk_flags": risk_flags,
        })
    return rows


def _document_matrix_evidence_rows(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in table:
        subject = str(row.get("subject") or "").strip()
        axis = str(row.get("comparison_axis") or "").strip()
        cell_id = str(row.get("matrix_cell_id") or "").strip()
        if not subject or not axis:
            continue
        key = cell_id or f"{subject}::{axis}"
        bucket = grouped.setdefault(
            key,
            {
                "matrix_cell_id": cell_id,
                "subject": subject,
                "comparison_axis": axis,
                "evidence_ids": set(),
                "source_urls": set(),
                "domains": set(),
            },
        )
        if row.get("id"):
            bucket["evidence_ids"].add(str(row.get("id")))
        if row.get("source_url"):
            bucket["source_urls"].add(str(row.get("source_url")))
        if row.get("source_domain"):
            bucket["domains"].add(str(row.get("source_domain")))
    rows = []
    for item in grouped.values():
        rows.append({
            "matrix_cell_id": item["matrix_cell_id"],
            "subject": item["subject"],
            "comparison_axis": item["comparison_axis"],
            "evidence_count": len(item["evidence_ids"]),
            "source_count": len(item["source_urls"]),
            "domain_count": len(item["domains"]),
            "selected_chunk_count": 0,
        })
    return rows


def _evidence_bound_paragraph(row: dict[str, Any]) -> str:
    evidence_id = str(row.get("id") or "")
    claim = _clean_evidence_sentence(str(row.get("claim") or ""))
    source_title = str(row.get("source_title") or row.get("source_url") or "来源")
    return f"{claim} [{evidence_id}]。来源：{source_title}。"


def _evidence_risk_flags(*, claim: str, quote: str, confidence: str, source_url: str) -> list[str]:
    text = f"{claim}\n{quote}"
    flags: list[str] = []
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|倍|个|项|次|万|亿|ms|s|秒|分钟|小时|天|GB|MB|KB)", text):
        flags.append("number_claim")
    if re.search(r"\b\d{4}\b", text):
        flags.append("date_claim")
    if re.search(r"(?:API|SDK|版本|模型|产品|发布|推出|支持|内置|默认)", text, re.I):
        flags.append("product_claim")
    if str(confidence or "").lower() == "low":
        flags.append("low_confidence")
    if not _url_domain(source_url):
        flags.append("missing_domain")
    return flags


def _evidence_bound_boundary_text(table: list[dict[str, Any]]) -> str:
    source_count = len({str(row.get("source_url") or "") for row in table if row.get("source_url")})
    evidence_ids = "、".join(f"[{row.get('id')}]" for row in table[:8] if row.get("id"))
    return (
        f"本章节仅覆盖证据表中的 {len(table)} 条证据和 {source_count} 个来源；"
        f"未在证据表中出现的背景、建议或外部判断不纳入正文结论。主要证据包括：{evidence_ids}。"
    )


def _evidence_index_lines(table: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in table:
        evidence_id = str(row.get("id") or "")
        claim = _clean_evidence_sentence(str(row.get("claim") or ""))
        url = str(row.get("source_url") or "")
        lines.append(f"- [{evidence_id}] {claim}。来源：{url}")
    return lines


def _evidence_ref(row: dict[str, Any]) -> dict[str, str]:
    return {
        "evidence_id": str(row.get("id") or ""),
        "chunk_id": str(row.get("chunk_id") or ""),
        "claim": str(row.get("claim") or ""),
        "source_url": str(row.get("source_url") or ""),
    }


def _sentence_from_text(text: str) -> str:
    for sentence in re.split(r"(?<=[。！？.!?])\s+", str(text).strip()):
        cleaned = _clean_evidence_sentence(sentence)
        if cleaned:
            return cleaned
    return _clean_evidence_sentence(text)


def _clean_evidence_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -_*#\n\t。")
    return cleaned[:500]


def _clean_evidence_claim(text: str) -> str:
    cleaned = _clean_evidence_sentence(text)
    if not cleaned:
        return ""
    dict_prefix = re.match(r"^\{['\"]text['\"]:\s*['\"](.+?)['\"]\}\s*[:：]\s*(.+)$", cleaned)
    if dict_prefix:
        cleaned = dict_prefix.group(2).strip()
    question_prefix = re.match(r"^(.{4,180}?[？?])\s*[:：]\s*(.+)$", cleaned)
    if question_prefix:
        return _clean_evidence_sentence(question_prefix.group(2))
    label_prefix = re.match(r"^(.{4,120}?)\s*[:：]\s*(.+)$", cleaned)
    if label_prefix:
        prefix = label_prefix.group(1)
        suffix = label_prefix.group(2)
        question_markers = ("如何", "什么", "哪些", "怎样", "为何", "为什么", "是否", "相比", "目前有", "从用户", "在旅游助手", "主流")
        if len(suffix) >= 12 and any(marker in prefix for marker in question_markers):
            return _clean_evidence_sentence(suffix)
    return cleaned


def _trim_retrieval_for_section(retrieval: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    section_id = str(section.get("id") or "")
    title = str(section.get("title") or "")
    evidence_map = _section_items(_coerce_list(retrieval.get("evidence_map")), section_id, title)
    evidence_pack = _section_items(_coerce_list(retrieval.get("evidence_pack")), section_id, title)
    selected_chunks = _section_items(_coerce_list(retrieval.get("selected_chunks")), section_id, title)
    if "证据" in title or "参考" in title:
        evidence_map = _coerce_list(retrieval.get("evidence_map"))[:24]
        evidence_pack = _coerce_list(retrieval.get("evidence_pack"))[:24]
        selected_chunks = _coerce_list(retrieval.get("selected_chunks"))[:24]
    return {
        "sources": [_trim_item(item) for item in _coerce_list(retrieval.get("sources"))[:10]],
        "materials": [_trim_item(item) for item in _coerce_list(retrieval.get("materials"))[:10]],
        "selected_chunks": [_trim_item(item) for item in selected_chunks[:12]],
        "evidence_map": [_trim_item(item) for item in evidence_map[:12]],
        "evidence_pack": [_trim_item(item) for item in evidence_pack[:12]],
        "gaps": [_trim_text(item, 300) for item in _coerce_list(retrieval.get("gaps"))[:8]],
    }


def _needs_citation_repair(validation: dict[str, Any]) -> bool:
    for item in validation.get("per_document") or []:
        if item.get("blocking_unsupported_claims") or item.get("missing_inline_refs"):
            return True
    return False


def _mark_evidence_usage(evidence_map: list[Any], documents: list[dict[str, Any]]) -> list[Any]:
    usage: dict[str, set[str]] = {}
    for document in documents:
        title = str(document.get("title") or "")
        for ref in document.get("evidence_refs") or []:
            if not isinstance(ref, dict):
                continue
            for key in ("evidence_id", "id", "chunk_id"):
                value = str(ref.get(key) or "").strip()
                if value:
                    usage.setdefault(value, set()).add(title)
    updated = []
    for item in evidence_map:
        if not isinstance(item, dict):
            updated.append(item)
            continue
        ids = [
            str(item.get("id") or "").strip(),
            str(item.get("evidence_id") or "").strip(),
            str(item.get("chunk_id") or "").strip(),
        ]
        used_by = sorted({title for value in ids for title in usage.get(value, set()) if title})
        updated.append({**item, "used_by_documents": used_by})
    return updated


def _repair_document_citations(
    documents: list[dict[str, Any]],
    retrieval: dict[str, Any],
    research_plan: dict[str, Any] | None,
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    validation_by_title = {
        str(item.get("title") or ""): item
        for item in validation.get("per_document") or []
        if isinstance(item, dict)
    }
    sections = {str(item.get("title") or ""): item for item in (research_plan or {}).get("sections") or []}
    repaired: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        title = str(document.get("title") or "")
        issue = validation_by_title.get(title) or {}
        if not issue.get("unsupported_claims") and not issue.get("missing_inline_refs"):
            repaired.append(document)
            continue
        section = sections.get(title) or {"id": _section_id(title, index), "title": title, "questions": []}
        scoped_retrieval = _trim_retrieval_for_section(retrieval, section)
        prompt = {
            "task": "修复 Markdown 文档的证据引用标记。",
            "rules": [
                "只返回 JSON，不要解释。",
                "JSON 形状必须是 {\"document\":{\"title\":\"...\",\"content\":\"...\",\"evidence_refs\":[...]}}。",
                "不得新增事实，不得新增来源。",
                "只能使用 scoped_retrieval 中已有 evidence_map/evidence_pack/selected_chunks。",
                "为已有强事实结论补上正确的 [evidence_id]；如果找不到证据，改写为“证据不足/待核验”，不要继续保留强断言。",
                "保留或修正 evidence_refs，使每个 evidence_id 都在正文中以 [ev_xxx] 出现。",
                "source_url 必须来自 scoped_retrieval。",
            ],
            "document": document,
            "validation_issue": issue,
            "scoped_retrieval": scoped_retrieval,
        }
        response = _invoke_legacy_research_model(prompt, call_kind="citation_repair")
        content = getattr(response, "content", response)
        data = _parse_draft_json(str(content))
        next_doc = data.get("document") if isinstance(data, dict) else None
        if not isinstance(next_doc, dict):
            repaired.append(document)
            continue
        text = str(next_doc.get("content") or "").strip()
        refs = next_doc.get("evidence_refs") or document.get("evidence_refs") or []
        if not text:
            repaired.append(document)
            continue
        next_refs = refs if isinstance(refs, list) else []
        repaired_doc = {
            **document,
            "title": title,
            "content": text,
            "evidence_refs": next_refs,
        }
        if not repaired_doc.get("evidence_table") and next_refs:
            repaired_doc = _restore_document_audit_fields(repaired_doc, retrieval, section)
        repaired.append(repaired_doc)
    return repaired


def _restore_document_audit_fields(
    document: dict[str, Any],
    retrieval: dict[str, Any],
    section: dict[str, Any],
) -> dict[str, Any]:
    table = _build_section_evidence_table(retrieval, section, str(document.get("title") or ""))
    if not table:
        return document
    ref_ids = {
        str(ref.get("evidence_id") or ref.get("id") or ref.get("chunk_id") or "").strip()
        for ref in document.get("evidence_refs") or []
        if isinstance(ref, dict)
    }
    if ref_ids:
        filtered = [
            row for row in table
            if str(row.get("id") or "").strip() in ref_ids or str(row.get("chunk_id") or "").strip() in ref_ids
        ]
        if filtered:
            table = filtered
    group_blocks = []
    for group in _evidence_groups_by_question(table, section):
        group_block = _evidence_bound_group_markdown(group)
        group_blocks.append(group_block)
    return {
        **document,
        "evidence_table": table,
        "paragraph_validation": _document_paragraph_validation(group_blocks),
        "evidence_metrics": _document_evidence_metrics(table, group_blocks),
    }


def _section_items(items: list[Any], section_id: str, title: str) -> list[Any]:
    matched = []
    title_key = _strip_title_prefix(title)
    for item in items:
        if not isinstance(item, dict):
            continue
        if section_id and str(item.get("section_id") or "") == section_id:
            matched.append(item)
            continue
        suggested = item.get("suggested_titles") or item.get("suggested_sections") or []
        if isinstance(suggested, str):
            suggested = [suggested]
        if any(str(value) == title or title_key in str(value) for value in suggested if value):
            matched.append(item)
    return matched or items[:8]


def _parse_draft_json(raw: str) -> dict[str, Any]:
    try:
        return parse_tool_json(raw)
    except JSONDecodeError as first_error:
        repaired = _repair_draft_json(raw, first_error)
        try:
            return parse_tool_json(repaired)
        except JSONDecodeError as second_error:
            excerpt = _json_error_excerpt(raw, first_error.pos)
            raise ValueError(
                "draft synthesis returned invalid JSON after repair attempt: "
                f"{second_error}. raw_excerpt={excerpt}"
            ) from second_error


def _parse_json_or_error(text: str) -> tuple[dict[str, Any], JSONDecodeError | None]:
    stripped = str(text or "").strip()
    try:
        value = json.loads(stripped)
        return (value if isinstance(value, dict) else {}), None
    except JSONDecodeError as first_error:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            return {}, first_error
        try:
            value = json.loads(match.group(0))
            return (value if isinstance(value, dict) else {}), None
        except JSONDecodeError as second_error:
            return {}, second_error


def _json_from_text(text: str) -> dict[str, Any]:
    parsed, _error = _parse_json_or_error(text)
    return parsed


def _repair_draft_json(raw: str, error: JSONDecodeError) -> str:
    repair_prompt = {
        "task": "修复一段模型输出，使其成为合法 JSON。",
        "rules": [
            "只返回修复后的 JSON，不要解释。",
            "不得新增事实、不得改写中文正文含义。",
            "保留 documents/title/content/evidence_refs 字段。",
            "修复常见问题：未转义换行、缺少逗号、错误引号、Markdown 代码围栏。",
        ],
        "json_error": str(error),
        "raw_output": raw[:24000],
    }
    response = _invoke_legacy_research_model(repair_prompt, call_kind="json_repair")
    return str(getattr(response, "content", response) or "")


def _json_error_excerpt(raw: str, pos: int, radius: int = 260) -> str:
    start = max(0, pos - radius)
    end = min(len(raw), pos + radius)
    return raw[start:end].replace("\n", "\\n")


def _trim_retrieval(retrieval: dict[str, Any]) -> dict[str, Any]:
    trimmed: dict[str, Any] = {}
    for key, limit in {
        "sources": 10,
        "documents": 6,
        "evidences": 8,
        "key_facts": 8,
        "evidence_pack": 10,
        "gaps": 6,
    }.items():
        items = _coerce_list(retrieval.get(key))[:limit]
        trimmed[key] = [_trim_item(item) for item in items]
    return trimmed


def _trim_item(item: Any) -> Any:
    if isinstance(item, dict):
        return {str(key): _trim_text(value) for key, value in item.items() if key in {
            "title",
            "url",
            "snippet",
            "content",
            "text",
            "claim",
            "evidence",
            "evidence_text",
            "fact",
            "source",
            "source_url",
            "source_title",
            "source_urls",
            "sources",
            "summary",
            "confidence",
            "suggested_titles",
            "suggested_sections",
            "id",
            "section_id",
            "question_id",
            "question",
            "material_id",
            "source_type",
            "provider",
            "content_length",
            "content_quality",
            "degraded",
            "excerpt",
            "chunk_count",
            "score",
            "used_by_documents",
        }}
    return _trim_text(item)


def _trim_text(value: Any, limit: int = 1800) -> str:
    text = str(value or "")
    return text[:limit] + ("..." if len(text) > limit else "")


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _source_items(sources: list[Any]) -> list[dict[str, Any]]:
    items = []
    for source in sources:
        if isinstance(source, dict):
            items.append({"label": str(source.get("title") or source.get("url") or "source"), "value": source.get("url")})
        else:
            items.append({"label": str(source), "value": str(source)})
    return items


def _used_sources_from_evidence(
    *,
    candidate_sources: list[dict[str, Any]],
    materials: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the user-facing source list from URLs that support admitted evidence.

    Agentic fetches and Context7 documents can reach materials/evidence without first appearing
    in the web-search candidate list. Filtering candidates alone therefore undercounted real
    citations in result diagnostics and review artifacts.
    """
    candidate_by_url = {
        str(item.get("url") or item.get("source_url") or "").strip(): item
        for item in candidate_sources
        if str(item.get("url") or item.get("source_url") or "").strip()
    }
    material_by_url: dict[str, dict[str, Any]] = {}
    for item in materials:
        url = str(item.get("url") or item.get("source_url") or "").strip()
        if url and url not in material_by_url:
            material_by_url[url] = item

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evidence in evidence_map:
        url = str(evidence.get("source_url") or evidence.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        candidate = candidate_by_url.get(url) or {}
        material = material_by_url.get(url) or {}
        output.append({
            **candidate,
            "title": str(
                evidence.get("source_title")
                or candidate.get("title")
                or material.get("title")
                or url
            ).strip(),
            "url": url,
            "source_type": str(
                evidence.get("source_type")
                or candidate.get("source_type")
                or material.get("source_type")
                or "web_article"
            ).strip(),
            "provider": candidate.get("provider") or material.get("provider"),
        })
    return output


def _document_items(documents: list[Any]) -> list[dict[str, Any]]:
    items = []
    for document in documents:
        if isinstance(document, dict):
            items.append({"label": str(document.get("title") or document.get("url") or "document"), "value": document.get("url")})
        else:
            items.append({"label": "document", "value": _trim_text(document, 120)})
    return items


def _material_items(materials: list[Any]) -> list[dict[str, Any]]:
    items = []
    for material in materials:
        if isinstance(material, dict):
            items.append({
                "label": str(material.get("title") or material.get("url") or material.get("id") or "material"),
                "value": f"{material.get('content_length') or 0} chars · {material.get('content_quality') or 'unknown'}",
            })
        else:
            items.append({"label": "material", "value": _trim_text(material, 120)})
    return items


def _chunk_items(chunks: list[Any]) -> list[dict[str, Any]]:
    items = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            items.append({
                "label": str(chunk.get("question") or chunk.get("section_id") or chunk.get("id") or "chunk"),
                "value": f"{chunk.get('score') or 0} · {chunk.get('source_title') or chunk.get('source_url') or ''}",
            })
        else:
            items.append({"label": "chunk", "value": _trim_text(chunk, 120)})
    return items


def _diagnostic_items(diagnostics: dict[str, Any], key: str) -> list[dict[str, Any]]:
    data = diagnostics.get(key) if isinstance(diagnostics, dict) else None
    if not isinstance(data, dict):
        return []
    items = []
    for field in (
        "mode",
        "elapsed_ms",
        "model_input_chars",
        "raw_model_output_chars",
        "material_count",
        "selected_chunk_count",
        "selected_chunk_chars",
        "section_count",
        "evidence_count",
    ):
        if field not in data:
            continue
        value = data.get(field)
        if field == "elapsed_ms" and isinstance(value, int):
            value = f"{round(value / 1000, 1)}s"
        items.append({"label": f"{key}.{field}", "value": value})
    return items


def _evidence_items(items: list[Any]) -> list[dict[str, Any]]:
    output = []
    for item in items[:8]:
        if isinstance(item, dict):
            output.append({
                "label": str(item.get("claim") or item.get("fact") or item.get("title") or item.get("evidence_text") or "evidence"),
                "value": item.get("source_url") or item.get("source") or item.get("sources") or item.get("source_title"),
            })
        else:
            output.append({"label": _trim_text(item, 80), "value": ""})
    return output


def _validate_synthesis_grounding(documents: list[dict[str, Any]], retrieval: dict[str, Any]) -> dict[str, Any]:
    urls = set(_source_strings(_coerce_list(retrieval.get("sources"))))
    urls.update(_source_strings(_coerce_list(retrieval.get("documents"))))
    evidence_urls: set[str] = set()
    evidence_claims: list[str] = []
    evidence_by_id = _evidence_by_id([
        *_coerce_list(retrieval.get("evidence_map")),
        *_coerce_list(retrieval.get("evidence_pack")),
        *_coerce_list(retrieval.get("selected_chunks")),
    ])
    for item in [
        *_coerce_list(retrieval.get("evidence_pack")),
        *_coerce_list(retrieval.get("evidences")),
        *_coerce_list(retrieval.get("key_facts")),
    ]:
        if not isinstance(item, dict):
            continue
        claim_text = str(item.get("claim") or item.get("evidence_text") or item.get("evidence") or "").strip()
        if claim_text:
            evidence_claims.append(claim_text)
        for key in ("source_url", "url", "source"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("http"):
                urls.add(value)
                evidence_urls.add(value)
        source_urls = item.get("source_urls")
        if isinstance(source_urls, list):
            urls.update(str(url) for url in source_urls if str(url).startswith("http"))
            evidence_urls.update(str(url) for url in source_urls if str(url).startswith("http"))

    per_document: list[dict[str, Any]] = []

    for document in documents:
        content = document.get("content", "")
        if "## 来源" not in content and "### 来源" not in content:
            raise ValueError(f"draft synthesis must include a 来源 section for title: {document.get('title')}")
        if (
            not any(url in content for url in urls)
            and "证据不足" not in content
            and "待核验" not in content
            and "暂无可绑定" not in content
            and "未形成可审计证据表" not in content
        ):
            raise ValueError(
                f"draft synthesis is not grounded in retrieved sources for title: {document.get('title')}"
            )
        refs = [ref for ref in document.get("evidence_refs") or [] if isinstance(ref, dict)]
        matched_refs = []
        for ref in refs:
            evidence_id = str(ref.get("evidence_id") or ref.get("id") or "").strip()
            chunk_id = str(ref.get("chunk_id") or "").strip()
            url = str(ref.get("source_url") or "").strip()
            claim = str(ref.get("claim") or "").strip()
            if (
                (evidence_id and evidence_id in evidence_by_id)
                or (chunk_id and chunk_id in evidence_by_id)
                or (url in evidence_urls and _claim_overlaps_evidence(claim, evidence_claims))
            ):
                matched_refs.append({"evidence_id": evidence_id, "chunk_id": chunk_id, "claim": claim, "source_url": url})
        per_document.append({
            "title": document.get("title"),
            "evidence_ref_count": len(refs),
            "matched_evidence_ref_count": len(matched_refs),
            "matched_refs": matched_refs[:6],
        })
    return {
        "evidence_url_count": len(evidence_urls),
        "per_document": per_document,
    }


def _claim_overlaps_evidence(claim: str, evidence_claims: list[str]) -> bool:
    if not claim:
        return False
    claim_tokens = set(_text_tokens(claim))
    if not claim_tokens:
        return False
    for evidence in evidence_claims:
        evidence_tokens = set(_text_tokens(evidence))
        if len(claim_tokens & evidence_tokens) >= max(2, min(5, len(claim_tokens) // 3)):
            return True
    return False


def _text_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    tokens = [token for token in normalized.split() if len(token) >= 2]
    if any("\u4e00" <= ch <= "\u9fff" for ch in normalized):
        tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
    return tokens


def _validate_documents_against_evidence(documents: list[dict[str, Any]], evidence_items: list[Any]) -> dict[str, Any]:
    evidence_urls: set[str] = set()
    evidence_claims: list[str] = []
    evidence_by_id = _evidence_by_id(evidence_items)
    evidence_rows = [item for item in evidence_items if isinstance(item, dict)]
    evidence_by_section: dict[str, dict[str, Any]] = {}
    for item in evidence_rows:
        section_id = str(item.get("section_id") or "").strip() or "unknown"
        section_bucket = evidence_by_section.setdefault(
            section_id,
            {"section_id": section_id, "evidence_count": 0, "source_urls": set(), "source_domains": set(), "question_ids": set(), "used_ref_count": 0},
        )
        section_bucket["evidence_count"] += 1
        url = str(item.get("source_url") or item.get("url") or "").strip()
        if url:
            evidence_urls.add(url)
            section_bucket["source_urls"].add(url)
            domain = _url_domain(url)
            if domain:
                section_bucket["source_domains"].add(domain)
        question_id = str(item.get("question_id") or "").strip()
        if question_id:
            section_bucket["question_ids"].add(question_id)
        claim = str(item.get("claim") or item.get("evidence_text") or item.get("evidence") or "").strip()
        if claim:
            evidence_claims.append(claim)

    per_document = []
    reasons: list[str] = []
    for doc in documents:
        refs = [ref for ref in doc.get("evidence_refs") or [] if isinstance(ref, dict)]
        content = str(doc.get("content") or "")
        matched = 0
        matched_by_id = 0
        matched_by_fallback = 0
        unsupported_refs: list[dict[str, str]] = []
        for ref in refs:
            evidence_id = str(ref.get("evidence_id") or ref.get("id") or "").strip()
            chunk_id = str(ref.get("chunk_id") or "").strip()
            url = str(ref.get("source_url") or "").strip()
            claim = str(ref.get("claim") or "").strip()
            evidence_by_evidence_id = evidence_by_id.get(evidence_id) if evidence_id else None
            evidence_by_chunk_id = evidence_by_id.get(chunk_id) if chunk_id else None
            if evidence_by_evidence_id and _ref_id_matches(ref, evidence_by_evidence_id):
                matched += 1
                matched_by_id += 1
                _mark_section_ref_used(evidence_by_section, evidence_by_evidence_id)
            elif evidence_by_chunk_id and _ref_id_matches(ref, evidence_by_chunk_id):
                matched += 1
                matched_by_id += 1
                _mark_section_ref_used(evidence_by_section, evidence_by_chunk_id)
            elif url in evidence_urls and (
                _claim_overlaps_evidence(claim, evidence_claims)
                or _claim_has_source_support(claim, url, evidence_items)
            ):
                matched += 1
                matched_by_fallback += 1
            else:
                unsupported_refs.append({
                    "evidence_id": evidence_id,
                    "chunk_id": chunk_id,
                    "claim": claim[:120],
                    "source_url": url,
                })
        ratio = matched / len(refs) if refs else 0.0
        if ratio < 0.8:
            reasons.append(f"{doc.get('title')} 证据引用匹配率低于 80%")
        unsupported_claims = _unsupported_strong_claims(
            content,
            evidence_items=evidence_rows,
            document_title=str(doc.get("title") or ""),
        )
        blocking_unsupported_claims = _blocking_unsupported_claims(unsupported_claims)
        if blocking_unsupported_claims:
            reasons.append(f"{doc.get('title')} 存在未引用证据的强事实结论")
        missing_inline_refs = _missing_inline_evidence_refs(content, refs)
        if missing_inline_refs:
            reasons.append(f"{doc.get('title')} evidence_refs 未在正文中标注")
        per_document.append({
            "title": doc.get("title"),
            "evidence_ref_count": len(refs),
            "matched_evidence_ref_count": matched,
            "matched_by_id_count": matched_by_id,
            "matched_by_fallback_count": matched_by_fallback,
            "match_ratio": round(ratio, 3),
            "unsupported_refs": unsupported_refs[:6],
            "unsupported_claims": unsupported_claims[:6],
            "blocking_unsupported_claims": blocking_unsupported_claims[:6],
            "missing_inline_refs": missing_inline_refs[:6],
        })

    return {
        "status": "passed" if not reasons else "failed",
        "reason": "证据引用验收通过。" if not reasons else "；".join(reasons),
        "per_document": per_document,
        "per_section": _validation_section_rows(evidence_by_section),
        "evidence_url_count": len(evidence_urls),
        "evidence_claim_count": len(evidence_claims),
        "evidence_id_count": len(evidence_by_id),
    }


def _blocking_unsupported_claims(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocking = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("initial_category") or "") == "validation_false_positive":
            continue
        if str(item.get("initial_category") or "") == "citation_inheritance_supported":
            continue
        if str(item.get("block_type") or "") in {"sources", "evidence_index", "boundary"}:
            continue
        blocking.append(item)
    return blocking


def _evidence_by_id(evidence_items: list[Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        for key in ("id", "evidence_id", "chunk_id"):
            value = str(item.get(key) or "").strip()
            if value:
                indexed[value] = item
    return indexed


def _unsupported_strong_claims(
    content: str,
    *,
    evidence_items: list[dict[str, Any]] | None = None,
    document_title: str = "",
) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    sentence_rows = _content_sentence_rows(content)
    for index, row in enumerate(sentence_rows):
        sentence = row["text"]
        trigger_patterns = _strong_claim_trigger_patterns(sentence)
        if not trigger_patterns:
            continue
        if _has_inline_evidence_marker(sentence):
            continue
        audit_item = _validation_audit_item(
            row=row,
            index=index,
            sentence_rows=sentence_rows,
            trigger_patterns=trigger_patterns,
            evidence_items=evidence_items or [],
            document_title=document_title,
        )
        unsupported.append(audit_item)
    return unsupported


def _missing_inline_evidence_refs(content: str, refs: list[dict[str, Any]]) -> list[dict[str, str]]:
    text = str(content or "")
    missing: list[dict[str, str]] = []
    for ref in refs:
        evidence_id = str(ref.get("evidence_id") or ref.get("id") or "").strip()
        if not evidence_id:
            continue
        if f"[{evidence_id}]" in text or evidence_id in text:
            continue
        missing.append({
            "evidence_id": evidence_id,
            "claim": str(ref.get("claim") or "")[:120],
            "source_url": str(ref.get("source_url") or "")[:300],
        })
    return missing


def _content_sentences(content: str) -> list[str]:
    return [row["text"] for row in _content_sentence_rows(content)]


def _content_sentence_rows(content: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    current_section = ""
    current_block = "body"
    line_index = 0
    for raw_line in str(content or "").splitlines():
        line_index += 1
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = line.strip("# ").strip()
            current_block = _validation_block_type(current_section)
            continue
        if line.startswith("|") or line.startswith("```"):
            continue
        if re.match(r"^[-*]\s*(https?://|来源|参考)", line, re.I):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            continue
        if "证据不足" in line or "待核验" in line or "不确定" in line:
            continue
        parts = re.split(r"(?<=[。！？.!?])\s+", line)
        for part in parts:
            text = part.strip("-* 　")
            if not text:
                continue
            sentences.append({
                "text": text,
                "line_index": line_index,
                "section": current_section,
                "block_type": current_block,
                "line": line[:500],
            })
    return sentences


def _is_strong_claim(sentence: str) -> bool:
    return bool(_strong_claim_trigger_patterns(sentence))


def _strong_claim_trigger_patterns(sentence: str) -> list[str]:
    text = str(sentence)
    if len(text) < 12:
        return []
    patterns = {
        "year": r"\b\d{4}\b",
        "number": r"\d+(?:\.\d+)?\s*(?:%|倍|个|项|次|万|亿|ms|s|秒|分钟|小时|天|GB|MB|KB)",
        "version": r"\bv?\d+\.\d+(?:\.\d+)?\b",
        "capability_or_change": r"(?:发布|推出|支持|实现|提供|导致|提升|降低|增长|减少|超过|达到|成为|依赖|采用|内置|默认)",
        "authority_or_product": r"(?:官方|论文|报告|研究|数据显示|基准|benchmark|API|SDK|版本|模型|公司|产品)",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]


def _validation_block_type(section: str) -> str:
    label = str(section or "")
    if "来源" in label:
        return "sources"
    if "证据索引" in label or "证据与参考" in label:
        return "evidence_index"
    if "边界" in label:
        return "boundary"
    return "body"


def _validation_audit_item(
    *,
    row: dict[str, Any],
    index: int,
    sentence_rows: list[dict[str, Any]],
    trigger_patterns: list[str],
    evidence_items: list[dict[str, Any]],
    document_title: str,
) -> dict[str, Any]:
    previous_rows = sentence_rows[max(0, index - 2):index]
    next_rows = sentence_rows[index + 1:index + 3]
    previous_texts = [item["text"] for item in previous_rows]
    next_texts = [item["text"] for item in next_rows]
    nearby_text = "\n".join([*previous_texts, row["text"], *next_texts])
    nearby_ids = sorted(set(re.findall(r"\[(ev_\d{3,4})\]", nearby_text, re.I)))
    inherited_ids = _inherited_evidence_ids(sentence_rows, index)
    candidate_evidence = _candidate_evidence_for_ids(inherited_ids or nearby_ids, evidence_items)
    initial_category = _initial_validation_category(
        row=row,
        inherited_ids=inherited_ids,
        nearby_ids=nearby_ids,
        candidate_evidence=candidate_evidence,
    )
    audit: dict[str, Any] = {
        "text": str(row.get("text") or "")[:240],
        "reason": "strong_claim_without_evidence_ref",
        "document_title": document_title,
        "section": row.get("section"),
        "block_type": row.get("block_type"),
        "line_index": row.get("line_index"),
        "previous_sentences": previous_texts,
        "next_sentences": next_texts,
        "nearby_evidence_ids": nearby_ids,
        "inherited_evidence_ids": inherited_ids,
        "trigger_patterns": trigger_patterns,
        "initial_category": initial_category,
        "llm_review": None,
    }
    if candidate_evidence:
        audit["candidate_evidence"] = [
            {
                "id": item.get("id") or item.get("evidence_id"),
                "claim": str(item.get("claim") or "")[:180],
                "evidence_text": str(item.get("evidence_text") or item.get("quote") or "")[:240],
                "source_url": str(item.get("source_url") or item.get("url") or "")[:300],
            }
            for item in candidate_evidence[:3]
        ]
    if os.getenv("RESEARCH_VALIDATION_LLM_REVIEW", "0") == "1":
        audit["llm_review"] = _review_validation_claim_with_llm(audit)
    return audit


def _inherited_evidence_ids(sentence_rows: list[dict[str, Any]], index: int) -> list[str]:
    window = sentence_rows[index + 1:index + 3]
    ids: list[str] = []
    for item in window:
        text = str(item.get("text") or "")
        found = re.findall(r"\[(ev_\d{3,4})\]", text, re.I)
        if found:
            ids.extend(found)
            break
    return sorted(set(ids))


def _candidate_evidence_for_ids(evidence_ids: list[str], evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not evidence_ids:
        return []
    wanted = {item.lower() for item in evidence_ids}
    output = []
    for item in evidence_items:
        keys = {
            str(item.get("id") or "").lower(),
            str(item.get("evidence_id") or "").lower(),
            str(item.get("chunk_id") or "").lower(),
        }
        if wanted & keys:
            output.append(item)
    return output


def _initial_validation_category(
    *,
    row: dict[str, Any],
    inherited_ids: list[str],
    nearby_ids: list[str],
    candidate_evidence: list[dict[str, Any]],
) -> str:
    if _is_validation_meta_sentence(str(row.get("text") or "")):
        return "validation_false_positive"
    if str(row.get("block_type") or "") in {"sources", "evidence_index", "boundary"}:
        return "validation_false_positive"
    if inherited_ids and candidate_evidence:
        return "citation_inheritance_supported"
    if nearby_ids:
        return "needs_review"
    return "real_writer_issue"


def _is_validation_meta_sentence(sentence: str) -> bool:
    text = re.sub(r"\s+", "", str(sentence or ""))
    if not text:
        return False
    if text.startswith("本文围绕") and ("可追溯网页证据" in text or "证据表" in text):
        return True
    if "内容仅依据下列证据表生成" in text and "引用标记对应文末证据索引" in text:
        return True
    return False


def _review_validation_claim_with_llm(audit: dict[str, Any]) -> dict[str, Any]:
    prompt = {
        "task": "复核 validation 拦截的 unsupported claim。只返回 JSON。",
        "rules": [
            "不要补充外部知识，只根据 claim、上下文和 candidate_evidence 判断。",
            "分类只能是 real_writer_issue、validation_false_positive、evidence_noise、sentence_split_issue、citation_inheritance_supported。",
            "supported=true 只表示建议人工复核可降级，不表示自动放行。",
        ],
        "claim": audit.get("text"),
        "previous_sentences": audit.get("previous_sentences"),
        "next_sentences": audit.get("next_sentences"),
        "nearby_evidence_ids": audit.get("nearby_evidence_ids"),
        "inherited_evidence_ids": audit.get("inherited_evidence_ids"),
        "candidate_evidence": audit.get("candidate_evidence") or [],
        "initial_category": audit.get("initial_category"),
    }
    try:
        response = _invoke_legacy_research_model(prompt, call_kind="validation_review")
        data = _json_from_text(str(getattr(response, "content", response) or ""))
        category = str(data.get("category") or data.get("classification") or "").strip()
        allowed = {
            "real_writer_issue",
            "validation_false_positive",
            "evidence_noise",
            "sentence_split_issue",
            "citation_inheritance_supported",
        }
        if category not in allowed:
            category = "real_writer_issue"
        return {
            "category": category,
            "supported": bool(data.get("supported")),
            "evidence_ids": data.get("evidence_ids") if isinstance(data.get("evidence_ids"), list) else [],
            "reason": str(data.get("reason") or "")[:300],
        }
    except Exception as exc:  # pragma: no cover - model/network dependent
        return {"category": "review_failed", "supported": False, "evidence_ids": [], "reason": str(exc)[:300]}


def _has_inline_evidence_marker(sentence: str) -> bool:
    text = str(sentence)
    return bool(
        re.search(r"\[(?:ev|chk)_\d{3,4}\]", text, re.I)
        or re.search(r"\b(?:ev|chk)_\d{3,4}\b", text, re.I)
        or "http://" in text
        or "https://" in text
    )


def _ref_id_matches(ref: dict[str, Any], evidence: dict[str, Any]) -> bool:
    ref_url = str(ref.get("source_url") or "").strip()
    evidence_url = str(evidence.get("source_url") or evidence.get("url") or "").strip()
    if ref_url and evidence_url and ref_url != evidence_url:
        return False
    return True


def _mark_section_ref_used(evidence_by_section: dict[str, dict[str, Any]], evidence: dict[str, Any]) -> None:
    section_id = str(evidence.get("section_id") or "").strip() or "unknown"
    bucket = evidence_by_section.get(section_id)
    if bucket is not None:
        bucket["used_ref_count"] += 1


def _validation_section_rows(evidence_by_section: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for section_id, bucket in sorted(evidence_by_section.items()):
        source_urls = bucket.get("source_urls") or set()
        source_domains = bucket.get("source_domains") or set()
        question_ids = bucket.get("question_ids") or set()
        rows.append({
            "section_id": section_id,
            "evidence_count": int(bucket.get("evidence_count") or 0),
            "source_count": len(source_urls),
            "domain_count": len(source_domains),
            "question_count": len(question_ids),
            "used_ref_count": int(bucket.get("used_ref_count") or 0),
        })
    return rows


def _claim_has_source_support(claim: str, url: str, evidence_items: list[Any]) -> bool:
    if not claim or not url:
        return False
    claim_tokens = set(_text_tokens(claim))
    if not claim_tokens:
        return False
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        item_url = str(item.get("source_url") or item.get("url") or "").strip()
        if item_url != url:
            continue
        evidence_text = " ".join([
            str(item.get("claim") or ""),
            str(item.get("evidence_text") or ""),
            str(item.get("evidence") or ""),
        ])
        evidence_tokens = set(_text_tokens(evidence_text))
        if not evidence_tokens:
            continue
        overlap = len(claim_tokens & evidence_tokens)
        if overlap >= max(1, min(3, len(claim_tokens) // 4)):
            return True
    return False


def _quality_gate(
    documents: list[dict[str, Any]],
    coverage: dict[str, Any],
    grounding: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    document_metrics = _quality_document_metrics(documents)
    evidence_pack_count = int(coverage.get("evidence_pack_count") or 0)
    evidence_count = max(int(coverage.get("evidence_count") or 0), document_metrics["evidence_count"])
    source_count = max(int(coverage.get("source_count") or 0), document_metrics["source_count"])
    document_count = max(int(coverage.get("document_count") or 0), document_metrics["document_count"])
    original_weak_sections = [
        item for item in coverage.get("weak_sections") or []
        if isinstance(item, dict)
    ]
    weak_sections = _filter_resolved_weak_sections(original_weak_sections, documents)
    resolved_weak_sections = [
        item for item in original_weak_sections
        if item not in weak_sections
    ]
    final_depth_ready = _final_documents_depth_ready(documents, weak_sections)
    depth_ready = bool(coverage.get("depth_ready")) or final_depth_ready

    if source_count < 2:
        reasons.append("来源少于 2 个")
    if document_count < 1:
        reasons.append("未抓取到可用正文")
    if max(evidence_pack_count, evidence_count) < 6:
        reasons.append("证据少于 6 条")
    if not depth_ready:
        reasons.append("文档级深度不足")
    if weak_sections:
        labels = [
            f"{item.get('title') or item.get('section_id')}({item.get('reason') or '证据不足'})"
            for item in weak_sections[:3]
        ]
        reasons.append("章节证据不足: " + "、".join(labels))
    if validation and validation.get("status") != "passed":
        reasons.append(str(validation.get("reason") or "证据验收未通过"))

    uncertain_docs = []
    non_blocking_uncertain_docs = []
    weak_grounding_docs = []
    for document in documents:
        content = document.get("content", "")
        if "## 来源" not in content and "### 来源" not in content:
            reasons.append(f"{document.get('title')} 缺少来源小节")
        if "证据不足" in content or "待核验" in content:
            title = str(document.get("title") or "untitled")
            if _document_uncertainty_blocks_quality(document):
                uncertain_docs.append(title)
            else:
                non_blocking_uncertain_docs.append(title)
    for item in grounding.get("per_document") or []:
        if int(item.get("matched_evidence_ref_count") or 0) < 2:
            weak_grounding_docs.append(str(item.get("title") or "untitled"))
    if weak_grounding_docs:
        reasons.append("部分文档缺少足够的证据引用: " + "、".join(weak_grounding_docs[:3]))
    if uncertain_docs:
        reasons.append("部分文档明确标注证据不足/待核验: " + "、".join(uncertain_docs[:3]))

    ready = not reasons
    return {
        "status": "passed" if ready else "needs_more_evidence",
        "approval_ready": ready,
        "reason": "证据质量达到落库门槛。" if ready else "；".join(reasons),
        "checks": {
            "source_count": source_count,
            "document_count": document_count,
            "evidence_count": evidence_count,
            "evidence_pack_count": evidence_pack_count,
            "depth_ready": depth_ready,
            "coverage_depth_ready": bool(coverage.get("depth_ready")),
            "final_depth_ready": final_depth_ready,
            "weak_sections": weak_sections,
            "resolved_weak_sections": resolved_weak_sections,
            "document_metrics": document_metrics,
            "uncertain_documents": uncertain_docs,
            "non_blocking_uncertain_documents": non_blocking_uncertain_docs,
            "weak_grounding_documents": weak_grounding_docs,
            "validation": validation or {},
        },
    }


def _quality_document_metrics(documents: list[dict[str, Any]]) -> dict[str, int]:
    source_urls: set[str] = set()
    domains: set[str] = set()
    evidence_ids: set[str] = set()
    evidence_total = 0
    for document in documents:
        if not isinstance(document, dict):
            continue
        table = document.get("evidence_table") or []
        table_count = 0
        if isinstance(table, list):
            table_count = len([row for row in table if isinstance(row, dict)])
            evidence_total += table_count
            for row in table:
                if not isinstance(row, dict):
                    continue
                if row.get("id"):
                    evidence_ids.add(str(row.get("id")))
                if row.get("source_url"):
                    source_urls.add(str(row.get("source_url")))
                if row.get("source_domain"):
                    domains.add(str(row.get("source_domain")))
        metrics = document.get("evidence_metrics") or {}
        if not table_count:
            evidence_total += int(metrics.get("evidence_count") or 0)
        source_count = int(metrics.get("source_count") or 0)
        domain_count = int(metrics.get("domain_count") or 0)
        for index in range(source_count):
            source_urls.add(f"metric-source:{document.get('title')}:{index}")
        for index in range(domain_count):
            domains.add(f"metric-domain:{document.get('title')}:{index}")
    return {
        "document_count": len([doc for doc in documents if isinstance(doc, dict)]),
        "evidence_count": max(evidence_total, len(evidence_ids)),
        "source_count": len(source_urls),
        "domain_count": len(domains),
    }


def _filter_resolved_weak_sections(
    weak_sections: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_title = {
        _normalize_title_key(str(document.get("title") or "")): document
        for document in documents
        if isinstance(document, dict)
    }
    unresolved = []
    for item in weak_sections:
        title = str(item.get("title") or item.get("section_title") or item.get("section_id") or "")
        document = by_title.get(_normalize_title_key(title))
        if document and _document_resolves_weak_section(document, item):
            continue
        unresolved.append(item)
    return unresolved


def _final_documents_depth_ready(documents: list[dict[str, Any]], weak_sections: list[dict[str, Any]]) -> bool:
    if weak_sections:
        return False
    substantive_docs = [
        document for document in documents
        if isinstance(document, dict) and not _is_reference_or_overview_title(str(document.get("title") or ""))
    ]
    if not substantive_docs:
        return False
    for document in substantive_docs:
        metrics = document.get("evidence_metrics") or {}
        target_evidence = _target_section_evidence_count(str(document.get("title") or ""))
        target_sources = _target_section_source_count(str(document.get("title") or ""))
        evidence_count = int(metrics.get("evidence_count") or 0)
        source_count = int(metrics.get("source_count") or 0)
        domain_count = int(metrics.get("domain_count") or 0)
        weak_question_count = int(metrics.get("weak_question_count") or 0)
        weak_matrix_count = int(metrics.get("weak_matrix_cell_count") or 0)
        if evidence_count < target_evidence or source_count < target_sources or domain_count < min(2, target_sources):
            return False
        if weak_question_count or weak_matrix_count:
            return False
    return True


def _document_resolves_weak_section(document: dict[str, Any], weak_section: dict[str, Any]) -> bool:
    metrics = document.get("evidence_metrics") or {}
    target_evidence = int(weak_section.get("target_evidence_count") or _target_section_evidence_count(str(document.get("title") or "")))
    target_sources = int(weak_section.get("target_source_count") or _target_section_source_count(str(document.get("title") or "")))
    evidence_count = int(metrics.get("evidence_count") or 0)
    source_count = int(metrics.get("source_count") or 0)
    domain_count = int(metrics.get("domain_count") or 0)
    weak_question_count = int(metrics.get("weak_question_count") or 0)
    weak_matrix_count = int(metrics.get("weak_matrix_cell_count") or 0)
    matrix_count = int(metrics.get("matrix_cell_count") or 0)
    if evidence_count < target_evidence or source_count < target_sources or domain_count < min(2, target_sources):
        return False
    reason = str(weak_section.get("reason") or "")
    if "question_evidence" in reason and weak_question_count:
        return False
    if "matrix_cell" in reason and matrix_count and weak_matrix_count:
        return False
    if weak_question_count:
        return False
    return True


def _normalize_title_key(title: str) -> str:
    return re.sub(r"^\s*\d+[_\-.、\s]*", "", title).strip().lower()


def _document_uncertainty_blocks_quality(document: dict[str, Any]) -> bool:
    title = str(document.get("title") or "")
    metrics = document.get("evidence_metrics") or {}
    evidence_count = int(metrics.get("evidence_count") or 0)
    source_count = int(metrics.get("source_count") or 0)
    weak_matrix_count = int(metrics.get("weak_matrix_cell_count") or 0)
    matrix_count = int(metrics.get("matrix_cell_count") or 0)
    if matrix_count and weak_matrix_count == 0 and source_count >= 2:
        return False
    if _is_reference_or_overview_title(title) and evidence_count >= 2 and source_count >= 2:
        return False
    return True


def _coverage_items(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    answer_counts = coverage.get("answer_status_counts") if isinstance(coverage.get("answer_status_counts"), dict) else {}
    return [
        {"label": "ready_to_write", "value": coverage.get("ready_to_write")},
        {"label": "完整覆盖", "value": answer_counts.get("complete")},
        {"label": "部分覆盖", "value": answer_counts.get("partial")},
        {"label": "未覆盖", "value": answer_counts.get("uncovered")},
        {"label": "covered_titles", "value": ", ".join(coverage.get("covered_titles") or [])},
        {"label": "partial_titles", "value": ", ".join(coverage.get("partial_titles") or [])},
        {"label": "missing_titles", "value": ", ".join(coverage.get("missing_titles") or [])},
        {"label": "sources", "value": coverage.get("source_count")},
        {"label": "evidence", "value": coverage.get("evidence_count")},
    ]


def _validation_items(validation: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        {"label": "status", "value": validation.get("status")},
        {"label": "evidence_urls", "value": validation.get("evidence_url_count")},
        {"label": "evidence_claims", "value": validation.get("evidence_claim_count")},
    ]
    for item in (validation.get("per_document") or [])[:6]:
        items.append({
            "label": str(item.get("title") or "document"),
            "value": f"{item.get('matched_evidence_ref_count')}/{item.get('evidence_ref_count')} refs",
        })
    return items


def _source_strings(sources: list[Any]) -> list[str]:
    strings = []
    for source in sources:
        if isinstance(source, dict):
            value = source.get("url") or source.get("title")
        else:
            value = source
        if value:
            strings.append(str(value))
    return strings
