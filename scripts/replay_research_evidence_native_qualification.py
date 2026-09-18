#!/usr/bin/env python3
"""Qualify the production Evidence-native writer against frozen Development artifacts.

This runner never searches, fetches, extracts Evidence, or mutates a Research Job. It reuses
the controlled-evaluation artifact loader and the production-neutral Evidence-native writing
module, while redirecting every model trace into the requested experiment directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.research.audit import code_version  # noqa: E402
from backend.research import evidence_native as evidence_native_module  # noqa: E402
from backend.research.evidence_native import (  # noqa: E402
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    CONTEXT_VERSION,
    PROMPT_VERSION,
    EvidenceContextManager,
    EvidenceNativeCheckpointStore,
    build_evidence_native_blueprint,
    validate_evidence_traceability,
    write_evidence_native_report,
)
from backend.research.finding_architecture_eval import (  # noqa: E402
    CheckpointArtifactLoader,
    DeadlineJsonModel,
    EvaluationContractError,
    FrozenJob,
    IsolatedProductionModelInvoker,
    RecordingJsonModel,
    atomic_write_json,
    atomic_write_text,
    build_reference_inventory,
    inventory_metrics,
    load_dataset,
    normalized_request_key,
    preflight_job,
    select_dataset_rows,
    stable_hash,
    utc_now,
    validate_dataset_isolation,
    validate_frozen_inventory,
)
from backend.research.finding_architecture_resilience import (  # noqa: E402
    DEFAULT_504_BACKOFF_SECONDS,
    Backoff504JsonInvoker,
)
from backend.research.quality import (  # noqa: E402
    ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION,
    ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
    combined_quality_gate,
    evaluate_report_quality,
    validate_claim_entailment,
    validate_report_citations,
)
from backend.research.writer import _splice_unsupported_markers  # noqa: E402
from backend.services.research_service import (  # noqa: E402
    _annotate_entailment_core_scope,
    _documents_with_editorial_pseudo_document,
)


SCHEMA_VERSION = "evidence-native-production-qualification/v1"
ATOMIC_BOUNDARY_PROFILE = "atomic_boundary_v1"
ATOMIC_BOUNDARY_PROMPT_VERSION = ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION


def _quality_payload_profile(entailment_profile: str) -> str:
    return (
        ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2
        if entailment_profile == ATOMIC_BOUNDARY_PROFILE
        else ENTAILMENT_PAYLOAD_PROFILE_DEFAULT
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the production Evidence-native writer on frozen Development inputs."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--job-id", action="append", dest="job_ids")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "data" / "research_artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--writer-user-agent", default="curl/8.0")
    parser.add_argument("--expected-model", default="mimo-v2.5")
    parser.add_argument("--retry-504", action="store_true")
    parser.add_argument("--validation-budget-seconds", type=float, default=2400.0)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument(
        "--entailment-profile",
        choices=("current", ATOMIC_BOUNDARY_PROFILE),
        default="current",
    )
    parser.add_argument(
        "--reuse-section-drafts-from",
        type=Path,
        help="Freeze section documents from a prior qualification output while rerunning downstream stages.",
    )
    return parser.parse_args()


def _selected_rows(dataset: dict[str, Any], job_ids: list[str] | None) -> list[dict[str, Any]]:
    requested = {str(item) for item in job_ids or []}
    rows = select_dataset_rows(dataset, "development")
    selected = [row for row in rows if not requested or str(row["job_id"]) in requested]
    missing = sorted(requested - {str(row["job_id"]) for row in selected})
    if missing:
        raise EvaluationContractError(
            f"qualification jobs must exist in Development: {', '.join(missing)}"
        )
    if not selected:
        raise EvaluationContractError("qualification selected no Development jobs")
    return selected


def _user_request(job: FrozenJob) -> str:
    value = job.normalized_request
    for key in ("user_request", "request", "query", "topic"):
        text = str(value.get(key) or "").strip()
        if text:
            return text
    raise EvaluationContractError(f"normalized request has no user request text: {job.job_id}")


def _qualification_blueprint(job: FrozenJob, section_ids: set[str]) -> dict[str, Any]:
    blueprint = build_evidence_native_blueprint(
        normalized_request=job.normalized_request,
        research_plan=job.research_plan,
        question_states=job.question_states,
        evidence_map=job.evidence_map,
        selected_chunks=job.selected_chunks,
    )
    kept_sections = []
    for section in blueprint.get("sections") or []:
        section_id = str(section.get("section_id") or "")
        title = str(section.get("title") or "")
        if section_id in section_ids or section_id == "sec_00" or "总览" in title or "来源" in title or "参考" in title:
            kept_sections.append(copy.deepcopy(section))
    if not any(str(item.get("section_id") or "") in section_ids for item in kept_sections):
        raise EvaluationContractError(
            f"selected section missing from Evidence-native blueprint: {job.job_id}/{sorted(section_ids)}"
        )
    return {**blueprint, "sections": kept_sections}


def _load_inventory(dataset_path: Path, row: dict[str, Any], job: FrozenJob) -> dict[str, Any]:
    raw_path = str(row.get("reference_inventory") or "").strip()
    if not raw_path:
        return build_reference_inventory(job, set(row.get("section_ids") or []))
    path = Path(raw_path)
    if not path.is_absolute():
        path = dataset_path.resolve().parent / path
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"reference inventory unreadable: {path}: {exc}") from exc
    validate_frozen_inventory(inventory, job)
    return inventory


def _preflight_eplus(job: FrozenJob, row: dict[str, Any], blueprint: dict[str, Any]) -> dict[str, Any]:
    validate_evidence_traceability(job.evidence_map, job.selected_chunks)
    manager = EvidenceContextManager()
    packets = []
    for section in blueprint.get("sections") or []:
        section_id = str(section.get("section_id") or "")
        if section_id not in {str(item) for item in row.get("section_ids") or []}:
            continue
        packet, diagnostics = manager.build(
            section=section,
            question_states=job.question_states,
            evidence_map=job.evidence_map,
        )
        packets.append({
            "section_id": section_id,
            "packet_hash": packet.get("packet_hash"),
            "selected_evidence_ids": diagnostics.get("selected_evidence_ids") or [],
            "cropped_evidence_ids": diagnostics.get("cropped_evidence_ids") or [],
            "estimated_tokens": diagnostics.get("estimated_tokens"),
        })
    return {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "blueprint_input_hash": blueprint.get("input_hash"),
        "context_manager_version": CONTEXT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "packets": packets,
    }


def _model(
    *,
    output_dir: Path,
    run_id: str,
    role: str,
    user_agent: str,
    retry_504: bool,
    budget_seconds: float | None = None,
) -> RecordingJsonModel:
    invoke: Any = IsolatedProductionModelInvoker(
        trace_root=output_dir / "model_traces",
        run_id=run_id,
        role=role,
        max_input_tokens=None,
        max_output_tokens=None,
        user_agent=user_agent,
    )
    if retry_504:
        invoke = Backoff504JsonInvoker(invoke)
    if budget_seconds is not None:
        invoke = DeadlineJsonModel(invoke, budget_seconds)
    return RecordingJsonModel(invoke)


def _load_frozen_section_documents(
    source_dir: Path,
    *,
    run_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_path = source_dir.resolve() / "runs" / f"{run_id}.json"
    try:
        source_run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"frozen section source unreadable: {run_path}: {exc}") from exc
    expected = str(source_run.get("run_sha256") or "")
    unhashed = {key: value for key, value in source_run.items() if key != "run_sha256"}
    if not expected or stable_hash(unhashed) != expected:
        raise EvaluationContractError(f"frozen section source hash mismatch: {run_path}")
    documents = {
        str(item.get("section_id") or ""): copy.deepcopy(item)
        for item in ((source_run.get("candidate") or {}).get("documents") or [])
        if isinstance(item, dict) and str(item.get("section_id") or "")
    }
    if not documents:
        raise EvaluationContractError(f"frozen section source has no documents: {run_path}")
    return documents, {
        "source_path": str(run_path),
        "source_run_sha256": expected,
        "document_hashes": {
            section_id: stable_hash(document) for section_id, document in sorted(documents.items())
        },
    }


def _call_metadata(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for call in calls:
        response = call.get("response") if isinstance(call.get("response"), dict) else {}
        metadata = response.get("_experiment_model_metadata") if isinstance(response, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        retry = response.get("_experiment_504_retry") if isinstance(response, dict) else {}
        result.append({
            "status": call.get("status"),
            "prompt_hash": call.get("prompt_hash"),
            "response_hash": call.get("response_hash"),
            "duration_ms": call.get("duration_ms"),
            "provider": metadata.get("provider"),
            "model": metadata.get("model"),
            "fallback": bool(metadata.get("fallback")),
            "usage": metadata.get("usage"),
            "estimated_input_tokens": metadata.get("estimated_input_tokens"),
            "user_agent_override": metadata.get("user_agent_override"),
            "retry_504_count": (retry or {}).get("retry_count"),
        })
    return result


def _provider_qualification(
    calls: list[dict[str, Any]], *, expected_model: str
) -> dict[str, Any]:
    metadata = _call_metadata(calls)
    failed = [item for item in metadata if item.get("status") != "success"]
    model_mismatches = [
        item
        for item in metadata
        if str(item.get("model") or "").strip() != str(expected_model or "").strip()
    ]
    fallbacks = [item for item in metadata if item.get("fallback")]
    return {
        "status": "passed" if not failed and not model_mismatches and not fallbacks else "provider_mismatch",
        "expected_model": expected_model,
        "call_count": len(metadata),
        "failed_call_count": len(failed),
        "model_mismatch_count": len(model_mismatches),
        "fallback_call_count": len(fallbacks),
        "actual_providers": sorted({str(item.get("provider") or "") for item in metadata}),
        "actual_models": sorted({str(item.get("model") or "") for item in metadata}),
    }


def _generation_failures(candidate: dict[str, Any]) -> list[dict[str, str]]:
    failures = []
    for document in candidate.get("documents") or []:
        if not isinstance(document, dict):
            continue
        diagnostics = document.get("synthesis_diagnostics") or {}
        mode = str(diagnostics.get("mode") or "")
        content = str(document.get("content") or "").strip()
        if mode == "model_failed" or not content:
            failures.append({
                "section_id": str(document.get("section_id") or ""),
                "mode": mode or "empty_content",
                "failure_reason": str(
                    diagnostics.get("failure_reason")
                    or diagnostics.get("upstream_failure_reason")
                    or "writer_returned_no_deliverable_content"
                ),
            })
    if not candidate.get("documents"):
        failures.append({
            "section_id": "",
            "mode": "missing_documents",
            "failure_reason": "writer_returned_no_documents",
        })
    return failures


def _live_validation(
    *,
    candidate: dict[str, Any],
    job: FrozenJob,
    model: RecordingJsonModel,
    parallelism: int,
    entailment_profile: str,
) -> dict[str, Any]:
    documents = candidate["documents"]
    report_document = candidate["report_document"]
    blueprint = candidate["report_blueprint"]
    evidence = candidate.get("validation_evidence_map") or job.evidence_map
    citation = validate_report_citations(
        documents=documents,
        report_document=report_document,
        evidence_map=job.evidence_map,
    )
    entailment = validate_claim_entailment(
        report_document=report_document,
        evidence_map=evidence,
        model=model,
        parallelism=max(1, parallelism),
        payload_profile=_quality_payload_profile(entailment_profile),
    )
    entailment = _annotate_entailment_core_scope(
        entailment,
        blueprint,
        unsupported_blocks_quality_gate=False,
    )
    critic_documents = _documents_with_editorial_pseudo_document(documents, candidate.get("editorial"))
    quality = evaluate_report_quality(
        report_document=report_document,
        question_states=job.question_states,
        findings=[],
        rejected_evidence=job.rejected_evidence,
        documents=critic_documents,
        model=model,
        evidence_map=evidence,
        section_critic_parallelism=max(1, parallelism),
        support_units=candidate.get("editorial_claim_bank") or [],
        support_unit_kind="validated_section_claim",
    )
    grounding = {
        "status": "passed" if all(bool(item.get("grounded_exactly")) for item in job.evidence_map) else "failed",
        "evidence_url_count": len({str(item.get("source_url") or "") for item in job.evidence_map}),
        "grounded_evidence_count": sum(bool(item.get("grounded_exactly")) for item in job.evidence_map),
        "evidence_count": len(job.evidence_map),
        "per_document": [],
    }
    return {
        "mode": "production_eplus_validators_live",
        "citation": citation,
        "entailment": entailment,
        "critic": quality,
        "grounding": grounding,
        "quality_gate": combined_quality_gate(
            grounding=grounding,
            validation=citation,
            report_quality=quality,
            entailment=entailment,
        ),
    }


def _resume_run(path: Path, contract_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = str(document.get("run_sha256") or "")
    unhashed = {key: value for key, value in document.items() if key != "run_sha256"}
    if not expected or stable_hash(unhashed) != expected:
        raise EvaluationContractError(f"qualification run hash mismatch: {path}")
    if document.get("contract_sha256") != contract_hash:
        raise EvaluationContractError(f"qualification run contract mismatch: {path}")
    return document


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    rows = _selected_rows(dataset, args.job_ids)
    loader = CheckpointArtifactLoader(args.artifact_root)

    all_rows = [*select_dataset_rows(dataset, "development"), *select_dataset_rows(dataset, "holdout")]
    jobs = {str(row["job_id"]): loader.load_job(str(row["job_id"])) for row in all_rows}
    isolation = validate_dataset_isolation(dataset, jobs)
    selected_jobs = {str(row["job_id"]): jobs[str(row["job_id"])] for row in rows}

    preflight_rows = []
    inventories = {}
    blueprints = {}
    for row in rows:
        job = selected_jobs[str(row["job_id"])]
        artifact_preflight = preflight_job(job, row)
        section_ids = {str(item) for item in row.get("section_ids") or []}
        blueprint = _qualification_blueprint(job, section_ids)
        eplus_preflight = _preflight_eplus(job, row, blueprint)
        inventory = _load_inventory(args.dataset, row, job)
        inventories[job.job_id] = inventory
        blueprints[job.job_id] = blueprint
        preflight_rows.append({"job_id": job.job_id, "artifact": artifact_preflight, "eplus": eplus_preflight})

    frozen_section_sources: dict[str, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}
    if args.reuse_section_drafts_from:
        for row in rows:
            job_id = str(row["job_id"])
            for section_id in [str(item) for item in row.get("section_ids") or []]:
                run_id = f"{job_id}.{section_id}"
                frozen_section_sources[run_id] = _load_frozen_section_documents(
                    args.reuse_section_drafts_from,
                    run_id=run_id,
                )

    contract = {
        "schema_version": SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "architecture_version": ARCHITECTURE_VERSION,
        "context_manager_version": CONTEXT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset_sha256": stable_hash(dataset),
        "job_ids": sorted(selected_jobs),
        "expected_model": args.expected_model,
        "configured_preferred_base_url": os.getenv("RESEARCH_PREFERRED_BASE_URL"),
        "writer_user_agent": args.writer_user_agent,
        "retry_504_schedule_seconds": list(DEFAULT_504_BACKOFF_SECONDS) if args.retry_504 else [],
        "validation_budget_seconds": float(args.validation_budget_seconds),
        "model_tokens": "provider_defaults_no_experiment_cap",
        "entailment_profile": args.entailment_profile,
        "entailment_prompt_version": (
            ATOMIC_BOUNDARY_PROMPT_VERSION
            if args.entailment_profile == ATOMIC_BOUNDARY_PROFILE
            else "claim-entailment/v2"
        ),
        "frozen_section_sources": {
            run_id: metadata
            for run_id, (_documents, metadata) in sorted(frozen_section_sources.items())
        },
    }
    contract_hash = stable_hash(contract)
    runs = []

    if not args.dry_run:
        if not (os.getenv("RESEARCH_PREFERRED_API_KEY") or os.getenv("MIMO_API_KEY")):
            raise EvaluationContractError("Mimo credential is not present in the process environment")
        for row in rows:
            job = selected_jobs[str(row["job_id"])]
            for section_id in [str(item) for item in row.get("section_ids") or []]:
                run_id = f"{job.job_id}.{section_id}"
                run_path = output_dir / "runs" / f"{run_id}.json"
                resumed = _resume_run(run_path, contract_hash) if args.resume else None
                if resumed is not None:
                    runs.append(resumed)
                    continue
                started = time.perf_counter()
                writer = _model(
                    output_dir=output_dir,
                    run_id=f"{run_id}.writer",
                    role="writer",
                    user_agent=args.writer_user_agent,
                    retry_504=args.retry_504,
                )
                prevalidation = _model(
                    output_dir=output_dir,
                    run_id=f"{run_id}.prevalidation",
                    role="validator",
                    user_agent=args.writer_user_agent,
                    retry_504=args.retry_504,
                    budget_seconds=args.validation_budget_seconds,
                )
                checkpoint_store = EvidenceNativeCheckpointStore(
                    output_dir / "checkpoints" / job.job_id / section_id,
                    contract={
                        **contract,
                        "job_id": job.job_id,
                        "section_id": section_id,
                        "normalized_request_hash": normalized_request_key(job.normalized_request),
                        "evidence_artifact_sha256": job.artifacts["evidence_map"].sha256,
                        "selected_chunks_artifact_sha256": job.artifacts["selected_chunks"].sha256,
                        "blueprint_input_hash": blueprints[job.job_id].get("input_hash"),
                    },
                )
                frozen_documents, frozen_metadata = frozen_section_sources.get(run_id, ({}, {}))
                original_write_section = evidence_native_module._write_section
                if frozen_documents:
                    def frozen_write_section(*, section: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
                        frozen = frozen_documents.get(str(section.get("section_id") or ""))
                        if not frozen:
                            raise EvaluationContractError(
                                f"frozen section document missing for {run_id}/{section.get('section_id')}"
                            )
                        return copy.deepcopy(frozen)

                    evidence_native_module._write_section = frozen_write_section
                try:
                    candidate = write_evidence_native_report(
                        user_request=_user_request(job),
                        blueprint={
                            **blueprints[job.job_id],
                            "sections": [
                                item
                                for item in blueprints[job.job_id].get("sections") or []
                                if str(item.get("section_id") or "") in {section_id, "sec_00"}
                                or "总览" in str(item.get("title") or "")
                                or "来源" in str(item.get("title") or "")
                                or "参考" in str(item.get("title") or "")
                            ],
                        },
                        question_states=job.question_states,
                        evidence_map=job.evidence_map,
                        selected_chunks=job.selected_chunks,
                        writer_model=writer,
                        prevalidation_model=prevalidation,
                        parallelism=max(1, args.parallelism),
                        checkpoint_store=checkpoint_store,
                        entailment_payload_profile=_quality_payload_profile(args.entailment_profile),
                    )
                finally:
                    evidence_native_module._write_section = original_write_section
                if frozen_metadata:
                    candidate.setdefault("diagnostics", {})["frozen_section_drafts"] = frozen_metadata
                candidate["arm"] = "evidence_native_eplus_production"
                candidate["model_calls"] = [*writer.calls, *prevalidation.calls]

                generation_failures = _generation_failures(candidate)
                if generation_failures:
                    provider_qualification = _provider_qualification(
                        candidate["model_calls"], expected_model=args.expected_model
                    )
                    run = {
                        "schema_version": SCHEMA_VERSION,
                        "experiment_id": args.experiment_id,
                        "job_id": job.job_id,
                        "section_id": section_id,
                        "split": "development",
                        "status": "generation_failed",
                        "started_at": utc_now(),
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        "contract_sha256": contract_hash,
                        "candidate": candidate,
                        "validation": {
                            "mode": "skipped_generation_failed",
                            "generation_failures": generation_failures,
                        },
                        "provider_qualification": provider_qualification,
                        "model_call_summary": _call_metadata(candidate["model_calls"]),
                    }
                    run["run_sha256"] = stable_hash(run)
                    atomic_write_json(run_path, run)
                    atomic_write_text(
                        output_dir / "reports" / f"{run_id}.md",
                        str(candidate.get("report_document", {}).get("content") or ""),
                    )
                    runs.append(run)
                    continue

                prevalidation_seconds = sum(
                    max(0.0, float(item.get("duration_ms") or 0))
                    for item in prevalidation.calls
                ) / 1000.0
                remaining = max(1.0, float(args.validation_budget_seconds) - prevalidation_seconds)
                final_validator = _model(
                    output_dir=output_dir,
                    run_id=f"{run_id}.final_validation",
                    role="validator",
                    user_agent=args.writer_user_agent,
                    retry_504=args.retry_504,
                    budget_seconds=remaining,
                )
                validation = _live_validation(
                    candidate=candidate,
                    job=job,
                    model=final_validator,
                    parallelism=max(1, args.parallelism),
                    entailment_profile=args.entailment_profile,
                )
                documents, editorial, content, marker_count = _splice_unsupported_markers(
                    documents=candidate["documents"],
                    editorial=candidate.get("editorial") or {},
                    entailment=validation["entailment"],
                    report_quality=validation["critic"],
                    user_request=_user_request(job),
                    evidence_map=job.evidence_map,
                    completeness=candidate["report_document"].get("completeness") or {},
                    research_mode=candidate["report_document"].get("research_mode"),
                )
                candidate["documents"] = documents
                candidate["editorial"] = editorial
                candidate["report_document"]["content"] = content
                candidate["model_calls"] = [*candidate["model_calls"], *final_validator.calls]
                validation["unsupported_marker_count"] = marker_count
                validation["budget_accounting"] = {
                    "configured_seconds": float(args.validation_budget_seconds),
                    "prevalidation_model_seconds": round(prevalidation_seconds, 3),
                    "final_validation_budget_seconds": round(remaining, 3),
                }
                validation["inventory_metrics"] = inventory_metrics(
                    candidate, inventories[job.job_id], job
                )
                provider_qualification = _provider_qualification(
                    candidate["model_calls"], expected_model=args.expected_model
                )
                run = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": args.experiment_id,
                    "job_id": job.job_id,
                    "section_id": section_id,
                    "split": "development",
                    "status": (
                        "succeeded"
                        if provider_qualification["status"] == "passed"
                        else "provider_mismatch"
                    ),
                    "started_at": utc_now(),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "contract_sha256": contract_hash,
                    "candidate": candidate,
                    "validation": validation,
                    "provider_qualification": provider_qualification,
                    "model_call_summary": _call_metadata(candidate["model_calls"]),
                }
                run["run_sha256"] = stable_hash(run)
                atomic_write_json(run_path, run)
                atomic_write_text(output_dir / "reports" / f"{run_id}.md", content)
                runs.append(run)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "created_at": utc_now(),
        "dry_run": args.dry_run,
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": stable_hash(dataset),
        "dataset_split": "development",
        "dataset_isolation": isolation,
        "job_filter": sorted(selected_jobs),
        "preflight": preflight_rows,
        "contract": contract,
        "contract_sha256": contract_hash,
        "code_version": code_version(),
        "known_limits": [
            "This qualification replays frozen Development artifacts and does not run retrieval or Evidence extraction.",
            "Only the selected sections are generated; this is not a Holdout or default-switch decision.",
            "The final quality conclusion still requires manual cited-span and readability review.",
        ],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "dry_run": args.dry_run,
        "run_count": len(runs),
        "preflight_count": len(preflight_rows),
        "runs": [
            {
                "job_id": run.get("job_id"),
                "section_id": run.get("section_id"),
                "status": run.get("status"),
                "quality_gate": ((run.get("validation") or {}).get("quality_gate") or {}).get("status"),
                "entailment_status": ((run.get("validation") or {}).get("entailment") or {}).get("status"),
                "unsupported_count": ((run.get("validation") or {}).get("entailment") or {}).get("unsupported_count"),
                "unsupported_marker_count": (run.get("validation") or {}).get("unsupported_marker_count"),
                "provider_qualification": (run.get("provider_qualification") or {}).get("status"),
                "model_calls": len((run.get("candidate") or {}).get("model_calls") or []),
            }
            for run in runs
        ],
        "conclusion": None if args.dry_run else "Requires manual final-output qualification review.",
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "status": "succeeded",
        "dry_run": args.dry_run,
        "output_dir": str(output_dir),
        "preflight_count": len(preflight_rows),
        "run_count": len(runs),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationContractError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
