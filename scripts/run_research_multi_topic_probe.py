"""Run multiple real Research Studio E2E probes and write aggregate reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas.research import ResearchJobCreateRequest  # noqa: E402
from backend.services import eval_case_store  # noqa: E402
from scripts.run_research_e2e_probe import run_probe, summary_failed  # noqa: E402


_PROD_MAX_ROUNDS = int(ResearchJobCreateRequest.model_fields["max_rounds"].default)


def _default_cases() -> list[dict[str, Any]]:
    return eval_case_store.load_cases("research_probe")


def _case_args(case: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    payload = {
        "user_request": case["user_request"],
        # v2 cases intentionally omit titles so the real Planner must decide the report shape.
        # Older/versioned cases can still provide titles when a fixed-section regression is wanted.
        "titles": str(case.get("titles") or ""),
        "strategy": case.get("strategy", args.strategy),
        "depth": case.get("depth", args.depth),
        "max_rounds": int(case.get("max_rounds", args.max_rounds)),
        "timeout_seconds": args.timeout_seconds,
        "poll_interval": args.poll_interval,
        "via_api": args.via_api,
        "verbose": args.verbose,
    }
    for optional_name in ("parent_job_id", "on_child_created"):
        if hasattr(args, optional_name):
            payload[optional_name] = getattr(args, optional_name)
    if args.force_depth:
        payload["depth"] = args.force_depth
    if args.force_max_rounds is not None:
        payload["max_rounds"] = args.force_max_rounds
    return argparse.Namespace(**payload)


def classify_summary(summary: dict[str, Any]) -> dict[str, Any]:
    job_status = str(summary.get("job_status") or "")
    if job_status == "cancelled":
        return {
            "outcome": "cancelled",
            "execution_status": "cancelled",
            "production_gate_status": "not_evaluated",
        }
    if job_status not in {"waiting_approval", "succeeded"} or summary.get("result_status") != "waiting_approval":
        return {
            "outcome": "execution_failed",
            "execution_status": "failed",
            "production_gate_status": "not_evaluated",
        }
    production_passed = not summary_failed(summary)
    return {
        "outcome": "completed" if production_passed else "pipeline_gate_failed",
        "execution_status": "succeeded",
        "production_gate_status": "passed" if production_passed else "failed",
    }


def _flatten_case_result(case: dict[str, Any], summary: dict[str, Any], failed: bool) -> dict[str, Any]:
    section_rows = ((summary.get("diagnostics") or {}).get("section_evidence") or {}).get("sections") or []
    question_loop_metrics = _question_loop_metrics(summary) if not section_rows else {}
    rerank = ((summary.get("diagnostics") or {}).get("chunk_rerank") or {})
    docs = summary.get("documents") or []
    unsupported_claims = sum((doc.get("validation") or {}).get("unsupported_claims") or 0 for doc in docs)
    blocking_unsupported_claims = sum((doc.get("validation") or {}).get("blocking_unsupported_claims") or 0 for doc in docs)
    unsupported_categories = _merge_category_counts(
        (doc.get("validation") or {}).get("unsupported_claim_categories") or {} for doc in docs
    )
    missing_inline_refs = sum((doc.get("validation") or {}).get("missing_inline_refs") or 0 for doc in docs)
    paragraph_failed = sum(1 for doc in docs if (doc.get("paragraph_validation") or {}).get("status") == "failed")
    classification = classify_summary(summary)
    hybrid_scored_chunks = int(rerank.get("hybrid_scored_count") or 0)
    entailment = (summary.get("validation") or {}).get("entailment") or {}
    claim_count = int(entailment.get("claim_count") or 0)
    entailed_count = int(entailment.get("entailed_count") or 0)
    inference_count = int(entailment.get("inference_count") or 0)
    supported_claim_count = entailed_count + inference_count
    validation_stage = (summary.get("pipeline_diagnostics") or {}).get("validation_stage") or {}
    marker_count = int(validation_stage.get("unsupported_marker_count") or 0)
    model_execution = summary.get("model_execution") or {}
    report_output = summary.get("report_output") or {}
    row = {
        "case_id": case["id"],
        "case_version": int(case.get("case_version") or 1),
        "kind": case["kind"],
        "title": case["title"],
        "failed": failed,
        **classification,
        "job_id": summary.get("job_id"),
        "writing_architecture": summary.get("writing_architecture") or "unknown",
        "writing_architecture_version": summary.get("writing_architecture_version"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "quality_gate_status": (summary.get("quality_gate") or {}).get("status"),
        "approval_ready": (summary.get("quality_gate") or {}).get("approval_ready"),
        "validation_status": (summary.get("validation") or {}).get("status"),
        "sources": (summary.get("counts") or {}).get("sources"),
        "materials": (summary.get("counts") or {}).get("materials"),
        "selected_chunks": (summary.get("counts") or {}).get("selected_chunks"),
        "evidence_map": (summary.get("counts") or {}).get("evidence_map"),
        "documents": (summary.get("counts") or {}).get("documents"),
        # Compatibility field retained for old report readers. The current RRF path does not
        # use the historical literal rerank_mode="hybrid_embedding"; the reliable signal is
        # whether a chunk actually received a hybrid_score.
        "hybrid_embedding_chunks": hybrid_scored_chunks,
        "hybrid_scored_chunks": hybrid_scored_chunks,
        "rerank_modes": rerank.get("modes") or {},
        "semantic_scored_count": rerank.get("semantic_scored_count"),
        "semantic_score_range": rerank.get("semantic_score_range"),
        "hybrid_score_range": rerank.get("hybrid_score_range"),
        "section_count": len(section_rows) or int(question_loop_metrics.get("section_count") or 0),
        "weak_question_count": (
            sum(int(row.get("weak_question_count") or 0) for row in section_rows if isinstance(row, dict))
            if section_rows
            else int(question_loop_metrics.get("weak_question_count") or 0)
        ),
        "risk_single_source_count": sum(
            int(row.get("risk_single_source_count") or 0) for row in section_rows if isinstance(row, dict)
        ) if section_rows else int(question_loop_metrics.get("risk_single_source_count") or 0),
        "min_section_source_count": (
            _min_int(row.get("source_count") for row in section_rows if isinstance(row, dict))
            if section_rows
            else question_loop_metrics.get("min_section_source_count")
        ),
        "min_section_domain_count": (
            _min_int(row.get("domain_count") for row in section_rows if isinstance(row, dict))
            if section_rows
            else question_loop_metrics.get("min_section_domain_count")
        ),
        "unsupported_claims": unsupported_claims,
        "blocking_unsupported_claims": blocking_unsupported_claims,
        "unsupported_claim_categories": unsupported_categories,
        "missing_inline_refs": missing_inline_refs,
        "paragraph_failed_documents": paragraph_failed,
        "entailment_payload_profile": entailment.get("payload_profile"),
        "entailment_prompt_version": entailment.get("prompt_version"),
        "cited_claim_count": claim_count,
        "supported_cited_claim_count": supported_claim_count,
        "cited_claim_support_rate": round(supported_claim_count / claim_count, 4) if claim_count else None,
        "unsupported_cited_claim_count": int(entailment.get("unsupported_count") or 0),
        "label_mismatch_cited_claim_count": int(entailment.get("label_mismatch_count") or 0),
        "non_atomic_cited_claim_count": int(entailment.get("non_atomic_count") or 0),
        "semantic_review_unavailable_count": int(entailment.get("semantic_review_unavailable_count") or 0),
        "model_reviewed_cited_claim_count": int(entailment.get("model_reviewed_count") or 0),
        "core_unsupported_cited_claim_count": int(entailment.get("core_unsupported_count") or 0),
        "unsupported_marker_count": marker_count,
        "unsupported_marker_rate": round(marker_count / claim_count, 4) if claim_count else None,
        "unsupported_marker_policy": entailment.get("unsupported_marker_policy"),
        "preferred_model": model_execution.get("preferred_model"),
        "preferred_model_call_count": int(model_execution.get("preferred_call_count") or 0),
        "preferred_model_failure_count": int(model_execution.get("preferred_failure_count") or 0),
        "provider_fallback_triggered_count": int(model_execution.get("fallback_triggered_count") or 0),
        "provider_fallback_success_count": int(model_execution.get("fallback_success_count") or 0),
        "provider_fallback_failure_count": int(model_execution.get("fallback_failure_count") or 0),
        "unresolved_internal_claim_ref_count": int(
            report_output.get("unresolved_internal_claim_ref_count") or 0
        ),
        "unresolved_internal_claim_group_count": int(
            report_output.get("unresolved_internal_claim_group_count") or 0
        ),
        "quality_reason": (summary.get("quality_gate") or {}).get("reason"),
        "validation_reason": (summary.get("validation") or {}).get("reason"),
    }
    return row


def _question_loop_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    question_states = [row for row in (summary.get("question_states") or []) if isinstance(row, dict)]
    coverage_rows = [row.get("coverage") for row in question_states if isinstance(row.get("coverage"), dict)]
    writer = ((summary.get("pipeline_diagnostics") or {}).get("writer") or {})
    weak_count = sum(
        1
        for coverage in coverage_rows
        if str(coverage.get("answer_status") or "").strip().lower() != "complete"
    )
    source_counts = [coverage.get("source_count") for coverage in coverage_rows]
    domain_counts = [coverage.get("domain_count") for coverage in coverage_rows]
    return {
        "section_count": int(writer.get("section_count") or len(summary.get("documents") or [])),
        "weak_question_count": weak_count,
        "risk_single_source_count": sum(
            1 for value in source_counts if value is not None and int(value) <= 1
        ),
        "min_section_source_count": _min_int(source_counts),
        "min_section_domain_count": _min_int(domain_counts),
    }


def _min_int(values: Any) -> int | None:
    nums = [int(value) for value in values if value is not None]
    return min(nums) if nums else None


def _merge_category_counts(items: Any) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            output[str(key)] = output.get(str(key), 0) + int(value or 0)
    return output


def _aggregate(rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    total = len(rows)
    execution_succeeded = sum(1 for row in rows if row.get("execution_status") == "succeeded")
    execution_failed = sum(1 for row in rows if row.get("execution_status") == "failed")
    cancelled = sum(1 for row in rows if row.get("execution_status") == "cancelled")
    production_passed = sum(1 for row in rows if row.get("production_gate_status") == "passed")
    production_failed = sum(1 for row in rows if row.get("production_gate_status") == "failed")
    cited_claim_total = sum(int(row.get("cited_claim_count") or 0) for row in rows)
    supported_cited_claim_total = sum(int(row.get("supported_cited_claim_count") or 0) for row in rows)
    return {
        "total_cases": total,
        # Compatibility fields: these continue to describe the production quality gate.
        "passed_cases": production_passed,
        "failed_cases": total - production_passed,
        "pass_rate": round(production_passed / total, 3) if total else 0,
        "execution_succeeded_cases": execution_succeeded,
        "execution_failed_cases": execution_failed,
        "cancelled_cases": cancelled,
        "production_gate_passed_cases": production_passed,
        "production_gate_failed_cases": production_failed,
        "production_gate_not_evaluated_cases": total - production_passed - production_failed,
        "production_gate_pass_rate": round(production_passed / execution_succeeded, 3)
        if execution_succeeded
        else 0,
        "elapsed_seconds": round(time.time() - started, 1),
        "avg_case_seconds": round(sum(float(row.get("elapsed_seconds") or 0) for row in rows) / total, 1)
        if total
        else 0,
        "weak_question_total": sum(int(row.get("weak_question_count") or 0) for row in rows),
        "unsupported_claim_total": sum(int(row.get("unsupported_claims") or 0) for row in rows),
        "blocking_unsupported_claim_total": sum(int(row.get("blocking_unsupported_claims") or 0) for row in rows),
        "unsupported_claim_categories": _merge_category_counts(
            row.get("unsupported_claim_categories") or {} for row in rows
        ),
        "missing_inline_ref_total": sum(int(row.get("missing_inline_refs") or 0) for row in rows),
        "cited_claim_total": cited_claim_total,
        "supported_cited_claim_total": supported_cited_claim_total,
        "cited_claim_support_rate": round(supported_cited_claim_total / cited_claim_total, 4)
        if cited_claim_total
        else None,
        "unsupported_cited_claim_total": sum(int(row.get("unsupported_cited_claim_count") or 0) for row in rows),
        "core_unsupported_cited_claim_total": sum(int(row.get("core_unsupported_cited_claim_count") or 0) for row in rows),
        "unsupported_marker_total": sum(int(row.get("unsupported_marker_count") or 0) for row in rows),
        "semantic_review_unavailable_total": sum(
            int(row.get("semantic_review_unavailable_count") or 0) for row in rows
        ),
        "atomic_boundary_case_count": sum(
            str(row.get("entailment_payload_profile") or "") == "atomic_boundary_v2"
            for row in rows
        ),
        "provider_fallback_case_count": sum(
            int(row.get("provider_fallback_triggered_count") or 0) > 0 for row in rows
        ),
        "unresolved_internal_claim_ref_total": sum(
            int(row.get("unresolved_internal_claim_ref_count") or 0) for row in rows
        ),
        "unresolved_internal_claim_ref_case_count": sum(
            int(row.get("unresolved_internal_claim_ref_count") or 0) > 0 for row in rows
        ),
        "hybrid_embedding_total": sum(int(row.get("hybrid_scored_chunks") or 0) for row in rows),
        "hybrid_scored_total": sum(int(row.get("hybrid_scored_chunks") or 0) for row in rows),
        "content_review_completed_cases": sum(
            1 for row in rows if (row.get("content_review") or {}).get("status") in {"completed", "completed_with_advice"}
        ),
        "content_review_advisory_cases": sum(
            1 for row in rows if (row.get("content_review") or {}).get("status") == "completed_with_advice"
        ),
        "content_review_unavailable_cases": sum(
            1 for row in rows if (row.get("content_review") or {}).get("status") == "content_review_unavailable"
        ),
    }


def _write_reports(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    json_path = output_dir / f"research-multi-topic-e2e-{stamp}.json"
    md_path = output_dir / f"research-multi-topic-e2e-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _markdown_report(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    rows = payload["rows"]
    lines = [
        "# 调研全流程多主题 E2E 评测报告",
        "",
        f"- started_at: `{payload['started_at']}`",
        f"- 评测集版本：`{(payload.get('configuration') or {}).get('case_set_version') or '历史版本'}`",
        f"- 用例数：`{aggregate['total_cases']}`",
        f"- 执行成功：`{aggregate.get('execution_succeeded_cases', 0)}`",
        f"- 执行失败：`{aggregate.get('execution_failed_cases', 0)}`",
        f"- 用户取消：`{aggregate.get('cancelled_cases', 0)}`",
        f"- 生产质量门通过：`{aggregate.get('production_gate_passed_cases', 0)}`",
        f"- 生产质量门未通过：`{aggregate.get('production_gate_failed_cases', 0)}`",
        f"- 内容建议已完成：`{aggregate.get('content_review_completed_cases', 0)}`",
        f"- Cited claims：`{aggregate.get('cited_claim_total', 0)}`",
        f"- Cited claim 支持率：`{aggregate.get('cited_claim_support_rate')}`",
        f"- `[待核实]` marker：`{aggregate.get('unsupported_marker_total', 0)}`",
        f"- 语义核验不可用：`{aggregate.get('semantic_review_unavailable_total', 0)}`",
        f"- Atomic-boundary 用例：`{aggregate.get('atomic_boundary_case_count', 0)}`",
        f"- Provider fallback 用例：`{aggregate.get('provider_fallback_case_count', 0)}`",
        f"- 内部 claim 引用泄漏用例：`{aggregate.get('unresolved_internal_claim_ref_case_count', 0)}`",
        f"- 内部 claim 引用泄漏总数：`{aggregate.get('unresolved_internal_claim_ref_total', 0)}`",
        f"- 总耗时（秒）：`{aggregate['elapsed_seconds']}`",
        "",
        "| 用例 | 架构/核验 | 执行 | 生产门 | 内容建议 | Cited 支持 | Marker | 内部引用 | 语义不可用 | 片段/证据 | 弱问题 | 来源/域名 | Provider | 耗时 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        source_domain = f"{row.get('min_section_source_count')}/{row.get('min_section_domain_count')}"
        lines.append(
            "| {title} | {architecture}/{profile} | {execution} | {production} | {content_review} | {support} | {markers} | {internal_refs} | {unavailable} | {chunks}/{evidence} | {weak_q} | {source_domain} | {provider} | {seconds} |".format(
                title=_md_cell(row["title"]),
                architecture=row.get("writing_architecture"),
                profile=row.get("entailment_payload_profile"),
                execution=row.get("execution_status"),
                production=row.get("production_gate_status"),
                content_review=(row.get("content_review") or {}).get("status") or "not_run",
                support=(
                    f"{row.get('supported_cited_claim_count')}/{row.get('cited_claim_count')} "
                    f"({row.get('cited_claim_support_rate')})"
                ),
                markers=row.get("unsupported_marker_count"),
                internal_refs=row.get("unresolved_internal_claim_ref_count"),
                unavailable=row.get("semantic_review_unavailable_count"),
                chunks=row.get("selected_chunks"),
                evidence=row.get("evidence_map"),
                weak_q=row.get("weak_question_count"),
                source_domain=source_domain,
                provider=(
                    f"{row.get('preferred_model') or 'unknown'}"
                    f"/fallback:{row.get('provider_fallback_triggered_count') or 0}"
                ),
                seconds=row.get("elapsed_seconds"),
            )
        )
    lines.extend(["", "## 需要查看的用例", ""])
    failed_rows = [row for row in rows if row["failed"]]
    if not failed_rows:
        lines.append("无。")
    for row in failed_rows:
        lines.extend(
            [
                f"### {row['title']}",
                "",
                f"- case_id: `{row['case_id']}`",
                f"- outcome: `{row.get('outcome')}`",
                f"- quality_reason: {row.get('quality_reason') or ''}",
                f"- validation_reason: {row.get('validation_reason') or ''}",
                f"- weak_question_count: `{row.get('weak_question_count')}`",
                f"- unsupported_claims: `{row.get('unsupported_claims')}`",
                f"- unsupported_claim_categories: `{row.get('unsupported_claim_categories') or {}}`",
                f"- writing_architecture: `{row.get('writing_architecture')}`",
                f"- entailment_profile: `{row.get('entailment_payload_profile')}`",
                f"- cited_claim_support: `{row.get('supported_cited_claim_count')}/{row.get('cited_claim_count')}`",
                f"- unsupported_marker_count: `{row.get('unsupported_marker_count')}`",
                f"- semantic_review_unavailable_count: `{row.get('semantic_review_unavailable_count')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


def _category_text(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = deepcopy(_default_cases())
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted or case["kind"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    return cases


def _case_set_version(cases: list[dict[str, Any]]) -> str:
    versions = sorted({int(case.get("case_version") or 1) for case in cases})
    return ",".join(f"v{version}" for version in versions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--via-api", action="store_true", default=True)
    parser.add_argument("--service", dest="via_api", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default="")
    parser.add_argument("--strategy", default="hv_analysis", choices=["standard", "hv_analysis"])
    parser.add_argument("--depth", default="normal", choices=["normal", "deep"])
    parser.add_argument("--force-depth", choices=["normal", "deep"])
    # Historically this had to track ResearchRequest.max_rounds (backend/schemas/research.py)
    # because research_service.py used to clamp state.budget.max_rounds down to whatever
    # max_rounds value the caller sent (via min()), so passing the schema's own default of 2
    # was silently shrinking QuestionBudget.for_depth()'s real per-question budget (normal=3,
    # deep=4) to 2 rounds for every case this probe ran -- see the 2026-08-13 fix in
    # research_service.py::_apply_requested_max_rounds (min() -> max()). That fix means an
    # explicit max_rounds now only ever RAISES the depth default, never lowers it, so this
    # value is effectively a floor, not a ceiling: leaving it at the schema default is now a
    # harmless no-op (for_depth's 3/4 already exceeds it), and a case can still opt into more
    # rounds via its own "max_rounds" key in config/eval_cases/research_probe.json.
    parser.add_argument("--max-rounds", type=int, default=_PROD_MAX_ROUNDS)
    parser.add_argument("--force-max-rounds", type=int)
    # Single question_loop case historically ran ~500-600s, but validation+revision can push
    # a real job to ~1000-1100s (retrieval ~4min + writing ~2min + entailment/critic/revision
    # ~11min). 900s therefore reports finished jobs as `quality=error validation=error
    # elapsed=None` — the job actually completed, the probe just stopped polling before the
    # process returned `succeeded`. See docs/evaluations/e2e-eval-sweep-20260809.md §3.
    # 1800 -> 3600 (2026-08-12, same reasoning as run_research_e2e_probe.py's matching default):
    # RESEARCH_VALIDATION_BUDGET_SECONDS rose to 2400s, so 1800s could no longer cover validation
    # alone -- case-probe-running-timeout was misreported as failed while still genuinely running.
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument(
        "--case-interval-seconds",
        type=float,
        default=30.0,
        help=(
            "Sleep this long between cases. All cases share one in-process model credential "
            "pool (backend/research/model_runtime.py) with no per-job reset, so running cases "
            "back-to-back with no gap compresses what production treats as independent, spaced-out "
            "research jobs into one continuous burst against the upstream gateway's rate-limit "
            "window, which previously caused avoidable rate-limit failures."
        ),
    )
    parser.add_argument("--output-dir", default="logs/research_e2e")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-content-review", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("RESEARCH_PARAGRAPH_COMPOSER", "0")
    os.environ.setdefault("RESEARCH_HYBRID_RERANK", "1")
    os.environ.setdefault("RESEARCH_RERANK_EMBEDDING_BATCH_SIZE", "32")

    cases = _selected_cases(args)
    started = time.time()
    started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
    rows: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[multi-topic] {index}/{len(cases)} {case['id']} - {case['title']}", flush=True)
        case_args = _case_args(case, args)
        try:
            summary = run_probe(case_args)
            failed = summary_failed(summary)
        except Exception as exc:
            summary = {"error": str(exc)}
            failed = True
        row = _flatten_case_result(case, summary, failed) if "error" not in summary else _error_row(case, summary)
        if row.get("execution_status") == "succeeded" and not args.skip_content_review:
            from backend.services.evaluation_service import _review_research_content

            row["content_review"] = _review_research_content(case, summary)
        else:
            row["content_review"] = {"status": "not_run", "blocking": False}
        rows.append(row)
        full_results.append({
            "case": case,
            "summary": summary,
            "failed": failed,
            "content_review": row.get("content_review"),
        })
        print(
            f"[multi-topic] result {case['id']}: {'failed' if failed else 'passed'} "
            f"quality={row.get('quality_gate_status')} validation={row.get('validation_status')} "
            f"weak_q={row.get('weak_question_count')} elapsed={row.get('elapsed_seconds')}",
            flush=True,
        )
        if failed and args.fail_fast:
            break
        if index < len(cases) and args.case_interval_seconds > 0:
            time.sleep(args.case_interval_seconds)

    payload = {
        "schema_version": 2,
        "started_at": started_text,
        "configuration": {
            "case_set_version": _case_set_version(cases),
            "strategy": args.strategy,
            "depth": args.depth,
            "content_review_enabled": not args.skip_content_review,
        },
        "env": {
            "RESEARCH_HYBRID_RERANK": os.getenv("RESEARCH_HYBRID_RERANK"),
            "RESEARCH_RERANK_EMBEDDING_BATCH_SIZE": os.getenv("RESEARCH_RERANK_EMBEDDING_BATCH_SIZE"),
            "RESEARCH_HYBRID_LEXICAL_WEIGHT": os.getenv("RESEARCH_HYBRID_LEXICAL_WEIGHT", "0.42"),
            "RESEARCH_HYBRID_SEMANTIC_WEIGHT": os.getenv("RESEARCH_HYBRID_SEMANTIC_WEIGHT", "0.48"),
            "RESEARCH_HYBRID_QUALITY_WEIGHT": os.getenv("RESEARCH_HYBRID_QUALITY_WEIGHT", "0.10"),
        },
        "aggregate": _aggregate(rows, started),
        "rows": rows,
        "results": full_results,
    }
    paths = _write_reports(Path(args.output_dir), payload)
    payload["report_paths"] = paths
    print(json.dumps({"aggregate": payload["aggregate"], "report_paths": paths, "rows": rows}, ensure_ascii=False, indent=2))
    return 1 if any(row["failed"] for row in rows) else 0


def _error_row(case: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "case_version": int(case.get("case_version") or 1),
        "kind": case["kind"],
        "title": case["title"],
        "failed": True,
        "outcome": "execution_failed",
        "execution_status": "failed",
        "production_gate_status": "not_evaluated",
        "job_id": None,
        "writing_architecture": "unknown",
        "writing_architecture_version": None,
        "elapsed_seconds": None,
        "quality_gate_status": "error",
        "approval_ready": False,
        "validation_status": "error",
        "sources": 0,
        "materials": 0,
        "selected_chunks": 0,
        "evidence_map": 0,
        "documents": 0,
        "hybrid_embedding_chunks": 0,
        "hybrid_scored_chunks": 0,
        "rerank_modes": {},
        "semantic_scored_count": 0,
        "semantic_score_range": None,
        "hybrid_score_range": None,
        "section_count": 0,
        "weak_question_count": 0,
        "risk_single_source_count": 0,
        "min_section_source_count": None,
        "min_section_domain_count": None,
        "unsupported_claims": 0,
        "unsupported_claim_categories": {},
        "missing_inline_refs": 0,
        "paragraph_failed_documents": 0,
        "entailment_payload_profile": None,
        "entailment_prompt_version": None,
        "cited_claim_count": 0,
        "supported_cited_claim_count": 0,
        "cited_claim_support_rate": None,
        "unsupported_cited_claim_count": 0,
        "label_mismatch_cited_claim_count": 0,
        "non_atomic_cited_claim_count": 0,
        "semantic_review_unavailable_count": 0,
        "model_reviewed_cited_claim_count": 0,
        "core_unsupported_cited_claim_count": 0,
        "unsupported_marker_count": 0,
        "unsupported_marker_rate": None,
        "unsupported_marker_policy": None,
        "preferred_model": None,
        "preferred_model_call_count": 0,
        "preferred_model_failure_count": 0,
        "provider_fallback_triggered_count": 0,
        "provider_fallback_success_count": 0,
        "provider_fallback_failure_count": 0,
        "unresolved_internal_claim_ref_count": 0,
        "unresolved_internal_claim_group_count": 0,
        "quality_reason": summary.get("error"),
        "validation_reason": summary.get("error"),
        "content_review": {"status": "not_run", "blocking": False},
    }


if __name__ == "__main__":
    raise SystemExit(main())
