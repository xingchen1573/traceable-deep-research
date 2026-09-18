"""Controlled Finding versus Direct Evidence evaluation infrastructure.

This module is intentionally disconnected from the Research Job lifecycle. It only reads
content-addressed historical audit artifacts and writes isolated experiment outputs chosen by
the caller. Nothing here is imported by the production research pipeline.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import random
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import audit
from .audit import sanitized_json_safe, stable_hash
from .claims import split_atomic_units
from .quality import (
    combined_quality_gate,
    evaluate_report_quality,
    validate_claim_entailment,
    validate_report_citations,
)
from .text_utils import normalize_for_substring
from .writer import (
    _call_writer_model,
    _annotate_claim_ledger,
    _assemble_report,
    _citation_map,
    _delivery_completeness,
    _edit_report,
    _evidence_metrics,
    _evidence_refs,
    _evidence_table_row,
    _excluded_from_assembled_body,
    _is_reference_title,
    _normalize_internal_citation_markers,
    _reference_title_from_blueprint,
    _render_internal_citations,
    _section_is_underdeveloped,
    _target_length_chars,
    _write_section,
)


SCHEMA_VERSION = "finding-architecture-ab/v1"
DATASET_SCHEMA_VERSION = "finding-architecture-dataset/v1"
INVENTORY_SCHEMA_VERSION = "reference-evidence-inventory/v1"
CLAIM_AUDIT_SCHEMA_VERSION = "finding-architecture-claim-evidence-audit/v2"
CONTEXT_MANAGER_VERSION = "direct-evidence-context/v1"
PROMPT_VERSION = "direct-evidence-writer-zh/v2"
CLUSTER_GUARDED_CONTEXT_MANAGER_VERSION = "direct-evidence-context-cluster-guarded/v1"
CLUSTER_GUARDED_PROMPT_VERSION = "direct-evidence-writer-zh/v4-cited-span-claim-bank"
DIRECT_CONTEXT_PROFILE_BASELINE = "baseline"
DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED = "cluster_guarded"
SUPPORTED_DIRECT_CONTEXT_PROFILES = {
    DIRECT_CONTEXT_PROFILE_BASELINE,
    DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED,
}
DIRECT_BLUEPRINT_PROFILE_HISTORICAL_PROJECTION = "historical_projection"
DIRECT_BLUEPRINT_PROFILE_EVIDENCE_NATIVE = "evidence_native"
SUPPORTED_DIRECT_BLUEPRINT_PROFILES = {
    DIRECT_BLUEPRINT_PROFILE_HISTORICAL_PROJECTION,
    DIRECT_BLUEPRINT_PROFILE_EVIDENCE_NATIVE,
}
ARM_FINDING = "finding"
ARM_DIRECT = "direct_evidence"
SUPPORTED_ARMS = {ARM_FINDING, ARM_DIRECT}
SUPPORTED_STAGES = {"section", "report"}
SUPPORTED_VALIDATION_MODES = {"deterministic", "live"}
REQUIRED_CHECKPOINT_STAGES = (
    "request_normalization",
    "planner",
    "fetch",
    "chunk_generation",
    "evidence_admission",
    "finding_synthesis",
    "draft",
    "validation",
)

WRITER_SYSTEM_PROMPT = (
    "你是资深行业分析师和技术编辑，负责把结构化的研究证据改写成连贯、可读的分析性报告正文。"
    "撰写时要有清晰的段落论点、自然的论证递进和恰当的过渡，避免逐条罗列孤立事实；"
    "同时必须严格遵守 payload 中的 rules 和引用要求，不能编造材料之外的内容。"
    "写作前必须分别核对证据主体、当前问题或比较维度、以及来源角色：主体不同不能互换，"
    "相邻指标不能替代目标指标，厂商或项目方自评必须明确归因且不能冒充独立评测。"
)


class EvaluationContractError(RuntimeError):
    """Raised when frozen inputs do not satisfy the controlled-evaluation contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(sanitized_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def estimate_tokens(value: Any) -> int:
    """Deterministic tokenizer-independent estimate used for equal-arm input accounting.

    Han characters are counted individually; ASCII-like runs use the conventional four
    characters per token approximation. Provider usage remains authoritative after a live call.
    """
    text = value if isinstance(value, str) else canonical_json(value)
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    other = max(0, len(text) - han)
    return han + (other + 3) // 4


def opaque_id(seed: int, *parts: str, prefix: str = "run") -> str:
    digest = hashlib.sha256((str(seed) + "\x1f" + "\x1f".join(parts)).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def normalized_request_key(value: Any) -> str:
    if isinstance(value, dict):
        request = value.get("user_request") or value.get("normalized_request") or value
    else:
        request = value
    text = re.sub(r"\s+", " ", str(request or "")).strip().casefold()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(sanitized_json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    stage: str
    side: str
    sha256: str
    path: str
    value: Any


@dataclass
class FrozenJob:
    job_id: str
    checkpoints: dict[str, dict[str, Any]]
    artifacts: dict[str, ArtifactRecord]
    normalized_request: dict[str, Any]
    research_plan: dict[str, Any]
    question_states: list[dict[str, Any]]
    materials: list[dict[str, Any]]
    selected_chunks: list[dict[str, Any]]
    evidence_map: list[dict[str, Any]]
    rejected_evidence: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    report_blueprint: dict[str, Any]
    historical_draft: dict[str, Any]
    historical_validation: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def direct_inputs(
        self,
        *,
        blueprint_profile: str = DIRECT_BLUEPRINT_PROFILE_HISTORICAL_PROJECTION,
    ) -> dict[str, Any]:
        """Return the Finding-free view allowed to Arm E."""
        if blueprint_profile not in SUPPORTED_DIRECT_BLUEPRINT_PROFILES:
            raise EvaluationContractError(
                f"unsupported Direct Evidence blueprint profile: {blueprint_profile}; "
                f"expected one of {sorted(SUPPORTED_DIRECT_BLUEPRINT_PROFILES)}"
            )
        if blueprint_profile == DIRECT_BLUEPRINT_PROFILE_EVIDENCE_NATIVE:
            report_blueprint = evidence_native_blueprint(
                normalized_request=self.normalized_request,
                research_plan=self.research_plan,
                question_states=self.question_states,
                evidence_map=self.evidence_map,
            )
        else:
            report_blueprint = direct_blueprint_view(self.report_blueprint, self.evidence_map)
        return {
            "job_id": self.job_id,
            "normalized_request": copy.deepcopy(self.normalized_request),
            "research_plan": copy.deepcopy(self.research_plan),
            "question_states": copy.deepcopy(self.question_states),
            "materials": copy.deepcopy(self.materials),
            "selected_chunks": copy.deepcopy(self.selected_chunks),
            "evidence_map": copy.deepcopy(self.evidence_map),
            # Rejected Evidence is retained by the experiment audit, never exposed to a Writer.
            "report_blueprint": report_blueprint,
        }


def direct_blueprint_view(
    blueprint: dict[str, Any], evidence_map: list[dict[str, Any]]
) -> dict[str, Any]:
    """Project the frozen Blueprint to structure without leaking Finding-derived content.

    The historical thesis, finding/evidence bindings, dossier, and Finding-count reportability
    fields are removed. Experiment 1 still freezes the section layout, questions, criteria,
    research mode, known gaps, and core/supplemental role.
    """
    sections = []
    evidence_section_ids = {str(item.get("section_id") or "") for item in evidence_map}
    core_section_ids: list[str] = []
    for section in blueprint.get("sections") or []:
        if not isinstance(section, dict):
            continue
        reportability = section.get("reportability") if isinstance(section.get("reportability"), dict) else {}
        core_section = bool(reportability.get("core_section"))
        section_id = str(section.get("section_id") or "")
        if core_section and section_id:
            core_section_ids.append(section_id)
        sections.append({
            "section_id": section.get("section_id"),
            "title": section.get("title"),
            "purpose": section.get("purpose"),
            "render_as": section.get("render_as"),
            "writing_strategy": section.get("writing_strategy"),
            "research_questions": copy.deepcopy(section.get("research_questions") or []),
            "known_gaps": copy.deepcopy(section.get("known_gaps") or []),
            "reportability": {"core_section": core_section},
        })
    empty_core = [section_id for section_id in core_section_ids if section_id not in evidence_section_ids]
    completeness = {
        "status": "complete_report" if core_section_ids and not empty_core else "partial_report",
        "required_section_count": len(core_section_ids),
        "ready_section_count": len(core_section_ids) - len(empty_core),
        "core_empty_section_ids": empty_core,
        "limited_section_ids": [],
        "supplemental_empty_section_ids": [],
        "reason": "evidence-native completeness projection for Direct Evidence experiment",
    }
    return {
        "audience": copy.deepcopy(blueprint.get("audience")),
        "completeness": completeness,
        "evidence_landscape": copy.deepcopy(blueprint.get("evidence_landscape") or {}),
        "report_type": blueprint.get("report_type"),
        "research_mode": blueprint.get("research_mode"),
        "style": copy.deepcopy(blueprint.get("style")),
        "sections": sections,
    }


def evidence_native_blueprint(
    *,
    normalized_request: dict[str, Any],
    research_plan: dict[str, Any],
    question_states: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic report Blueprint without reading historical Finding output."""
    state_by_question = {
        str(item.get("question_id") or ""): item
        for item in question_states
        if isinstance(item, dict) and item.get("question_id")
    }
    evidence_by_section: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_map:
        if not isinstance(item, dict):
            continue
        evidence_by_section.setdefault(str(item.get("section_id") or ""), []).append(item)

    sections: list[dict[str, Any]] = []
    core_section_ids: list[str] = []
    ready_core_section_ids: list[str] = []
    for index, plan_section in enumerate(research_plan.get("sections") or []):
        if not isinstance(plan_section, dict):
            continue
        section_id = str(plan_section.get("section_id") or plan_section.get("id") or "")
        if not section_id:
            continue
        plan_questions = [
            item for item in plan_section.get("questions") or plan_section.get("research_questions") or []
            if isinstance(item, dict)
        ]
        research_questions = []
        section_known_gaps: list[str] = []
        hard_question = False
        comparison_requested = False
        for question_index, plan_question in enumerate(plan_questions):
            question_id = str(
                plan_question.get("question_id")
                or plan_question.get("id")
                or f"{section_id}_q{question_index + 1}"
            )
            state = state_by_question.get(question_id, {})
            criteria = [
                {
                    "criterion_id": criterion.get("criterion_id"),
                    "description": criterion.get("description"),
                    "level": _criterion_level(criterion),
                    "status": criterion.get("status"),
                    "preferred_source_types": copy.deepcopy(
                        criterion.get("preferred_source_types") or []
                    ),
                }
                for criterion in state.get("criteria") or []
                if isinstance(criterion, dict)
            ]
            answer_level = str(plan_question.get("answer_contract_level") or "").lower()
            if answer_level == "hard" or any(item.get("level") == "hard" for item in criteria):
                hard_question = True
            comparison_mode = str(plan_question.get("comparison_axis_mode") or "")
            if comparison_mode not in {"", "not_applicable"}:
                comparison_requested = True
            coverage = state.get("coverage") if isinstance(state.get("coverage"), dict) else {}
            for value in [
                *(coverage.get("missing_question_points") or []),
                *(coverage.get("enrichment_gaps") or []),
            ]:
                if isinstance(value, dict):
                    gap = str(
                        value.get("missing_fact")
                        or value.get("description")
                        or value.get("gap_type")
                        or ""
                    ).strip()
                else:
                    gap = str(value or "").strip()
                if gap and gap not in section_known_gaps:
                    section_known_gaps.append(gap)
            research_questions.append({
                "question_id": question_id,
                "question": state.get("question") or plan_question.get("text") or plan_question.get("question"),
                "purpose": state.get("purpose") or plan_question.get("purpose"),
                "criteria": criteria,
            })

        rows = evidence_by_section.get(section_id, [])
        source_count = len({_evidence_source_key(item) for item in rows})
        overview = section_id == "sec_00" or "总览" in str(plan_section.get("title") or "")
        core_section = hard_question and not overview
        if core_section:
            core_section_ids.append(section_id)
            if rows:
                ready_core_section_ids.append(section_id)
        render_as = "executive_summary" if overview else "narrative"
        if (
            not overview
            and comparison_requested
            and research_plan.get("subjects")
            and research_plan.get("comparison_axes")
        ):
            render_as = "comparison_matrix"
        purpose = str(plan_section.get("purpose") or plan_section.get("intent") or "").strip()
        if not purpose and research_questions:
            purpose = str(research_questions[0].get("purpose") or "").strip()
        sections.append({
            "section_id": section_id,
            "title": plan_section.get("title") or f"{index:02d}_{section_id}",
            "purpose": purpose,
            "render_as": render_as,
            "writing_strategy": "evidence_native_cluster_guarded",
            "research_questions": research_questions,
            "known_gaps": section_known_gaps,
            "reportability": {
                "status": "ready" if rows else "limited",
                "evidence_count": len(rows),
                "source_count": source_count,
                "core_section": core_section,
                "reason": (
                    "accepted Evidence is available for this planned section"
                    if rows
                    else "no accepted Evidence is available for this planned section"
                ),
            },
        })

    core_empty = sorted(set(core_section_ids) - set(ready_core_section_ids))
    evidence_sources = {_evidence_source_key(item) for item in evidence_map if isinstance(item, dict)}
    return {
        "blueprint_profile": DIRECT_BLUEPRINT_PROFILE_EVIDENCE_NATIVE,
        "blueprint_inputs": ["research_plan", "question_states", "accepted_evidence"],
        "audience": {"language": "zh-CN", "reader": "research_user"},
        "completeness": {
            "status": "complete_report" if core_section_ids and not core_empty else "partial_report",
            "required_section_count": len(core_section_ids),
            "ready_section_count": len(ready_core_section_ids),
            "core_empty_section_ids": core_empty,
            "limited_section_ids": [
                section["section_id"]
                for section in sections
                if section["reportability"]["status"] != "ready"
            ],
            "supplemental_empty_section_ids": [
                section["section_id"]
                for section in sections
                if not section["reportability"]["core_section"]
                and section["reportability"]["status"] != "ready"
            ],
            "reason": "deterministic Evidence-native completeness from planned sections and accepted Evidence",
        },
        "evidence_landscape": {
            "accepted_evidence_count": len(evidence_map),
            "source_count": len(evidence_sources),
            "conflict_or_limitation_evidence_count": sum(
                _is_conflict_evidence(item) for item in evidence_map if isinstance(item, dict)
            ),
            "per_section_evidence_count": {
                section_id: len(rows) for section_id, rows in sorted(evidence_by_section.items())
            },
        },
        "report_type": research_plan.get("research_type") or "research_report",
        "research_mode": (
            normalized_request.get("research_mode")
            or research_plan.get("research_mode")
            or "domain_practice_review"
        ),
        "style": {"language": "zh-CN", "format": "multi_section_research_report"},
        "sections": sections,
    }


class CheckpointArtifactLoader:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root).resolve()

    def _job_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(job_id or "")):
            raise EvaluationContractError(f"unsafe job_id: {job_id!r}")
        path = (self.artifact_root / job_id).resolve()
        if self.artifact_root not in path.parents:
            raise EvaluationContractError(f"job path escapes artifact root: {job_id}")
        return path

    def latest_checkpoint(self, job_id: str, stage: str) -> tuple[Path, dict[str, Any]]:
        paths = sorted((self._job_dir(job_id) / "audit" / "checkpoints").glob(f"*_{stage}.json"))
        if not paths:
            raise EvaluationContractError(f"checkpoint missing: {job_id}/{stage}")
        path = paths[-1]
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationContractError(f"checkpoint unreadable: {path}: {exc}") from exc
        if str(document.get("job_id") or "") != job_id or str(document.get("stage") or "") != stage:
            raise EvaluationContractError(f"checkpoint identity mismatch: {path}")
        expected = str(document.get("checkpoint_sha256") or "")
        unhashed = {key: value for key, value in document.items() if key != "checkpoint_sha256"}
        actual = stable_hash(unhashed)
        if not expected or actual != expected:
            raise EvaluationContractError(
                f"checkpoint hash mismatch: {path} expected={expected or 'missing'} actual={actual}"
            )
        return path, document

    def artifact(self, checkpoint: dict[str, Any], side: str, name: str) -> ArtifactRecord:
        ref = ((checkpoint.get(side) or {}).get(name) or {})
        expected = str(ref.get("sha256") or "")
        raw_path = str(ref.get("path") or "")
        if not expected or not raw_path:
            raise EvaluationContractError(
                f"artifact reference missing: {checkpoint.get('job_id')}/{checkpoint.get('stage')}/{side}/{name}"
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.artifact_root.parent.parent / path
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise EvaluationContractError(f"artifact unreadable: {path}: {exc}") from exc
        job_dir = self._job_dir(str(checkpoint.get("job_id") or ""))
        if job_dir not in resolved.parents:
            raise EvaluationContractError(f"artifact path outside frozen job: {resolved}")
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationContractError(f"artifact JSON invalid: {resolved}: {exc}") from exc
        actual = stable_hash(value)
        if actual != expected:
            raise EvaluationContractError(
                f"artifact hash mismatch: {resolved} expected={expected} actual={actual}"
            )
        if resolved.name != f"sha256_{expected}.json":
            raise EvaluationContractError(f"artifact filename is not content-addressed: {resolved}")
        return ArtifactRecord(
            name=name,
            stage=str(checkpoint.get("stage") or ""),
            side=side,
            sha256=expected,
            path=str(resolved),
            value=value,
        )

    def load_job(self, job_id: str) -> FrozenJob:
        checkpoints: dict[str, dict[str, Any]] = {}
        checkpoint_paths: dict[str, str] = {}
        for stage in REQUIRED_CHECKPOINT_STAGES:
            path, document = self.latest_checkpoint(job_id, stage)
            checkpoints[stage] = document
            checkpoint_paths[stage] = str(path)

        wanted = {
            "normalized_request": ("request_normalization", "output_artifacts", "normalized_request"),
            "research_plan": ("planner", "output_artifacts", "research_plan"),
            "question_states": ("finding_synthesis", "input_artifacts", "question_states"),
            "materials": ("fetch", "output_artifacts", "materials"),
            "selected_chunks": ("chunk_generation", "output_artifacts", "selected_chunks"),
            "evidence_map": ("evidence_admission", "output_artifacts", "evidence_map"),
            "rejected_evidence": ("evidence_admission", "output_artifacts", "rejected_evidence"),
            "findings": ("finding_synthesis", "output_artifacts", "findings"),
            "report_blueprint": ("finding_synthesis", "output_artifacts", "report_blueprint"),
            "historical_draft": ("draft", "output_artifacts", "question_loop_snapshot"),
            "historical_validation": ("validation", "output_artifacts", "question_loop_snapshot"),
        }
        artifacts: dict[str, ArtifactRecord] = {}
        for key, (stage, side, name) in wanted.items():
            artifacts[key] = self.artifact(checkpoints[stage], side, name)

        trace_path = self._job_dir(job_id) / "audit" / "model_calls.jsonl"
        trace_events = []
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    trace_events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return FrozenJob(
            job_id=job_id,
            checkpoints=checkpoints,
            artifacts=artifacts,
            normalized_request=_expect_dict(artifacts["normalized_request"].value, "normalized_request"),
            research_plan=_expect_dict(artifacts["research_plan"].value, "research_plan"),
            question_states=_expect_list(artifacts["question_states"].value, "question_states"),
            materials=_expect_list(artifacts["materials"].value, "materials"),
            selected_chunks=_expect_list(artifacts["selected_chunks"].value, "selected_chunks"),
            evidence_map=_expect_list(artifacts["evidence_map"].value, "evidence_map"),
            rejected_evidence=_expect_list(artifacts["rejected_evidence"].value, "rejected_evidence"),
            findings=_expect_list(artifacts["findings"].value, "findings"),
            report_blueprint=_expect_dict(artifacts["report_blueprint"].value, "report_blueprint"),
            historical_draft=_expect_dict(artifacts["historical_draft"].value, "historical_draft"),
            historical_validation=_expect_dict(artifacts["historical_validation"].value, "historical_validation"),
            diagnostics={"checkpoint_paths": checkpoint_paths, "model_trace_events": trace_events},
        )


def _expect_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{name} must be an object")
    return value


def _expect_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise EvaluationContractError(f"{name} must be an array of objects")
    return value


def load_dataset(path: Path) -> dict[str, Any]:
    try:
        dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"dataset unreadable: {path}: {exc}") from exc
    if not isinstance(dataset, dict) or dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise EvaluationContractError(f"dataset schema must be {DATASET_SCHEMA_VERSION}")
    for split in ("development", "holdout"):
        rows = dataset.get(split)
        if not isinstance(rows, list):
            raise EvaluationContractError(f"dataset.{split} must be an array")
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("job_id") or ""):
                raise EvaluationContractError(f"dataset.{split} contains an invalid row")
    job_ids = [str(row["job_id"]) for split in ("development", "holdout") for row in dataset[split]]
    if len(job_ids) != len(set(job_ids)):
        raise EvaluationContractError("development and holdout job_id values must be disjoint")
    return dataset


def select_dataset_rows(dataset: dict[str, Any], selection: str) -> list[dict[str, Any]]:
    selection = str(selection or "development").strip().lower()
    if selection == "development":
        return [{**row, "split": "development"} for row in dataset["development"]]
    if selection == "holdout":
        return [{**row, "split": "holdout"} for row in dataset["holdout"]]
    if selection == "all":
        return [
            *[{**row, "split": "development"} for row in dataset["development"]],
            *[{**row, "split": "holdout"} for row in dataset["holdout"]],
        ]
    raise EvaluationContractError("dataset selection must be development, holdout, or all")


def validate_dataset_isolation(dataset: dict[str, Any], jobs: dict[str, FrozenJob]) -> dict[str, Any]:
    keys: dict[str, list[tuple[str, str]]] = {}
    for split in ("development", "holdout"):
        for row in dataset[split]:
            job_id = str(row["job_id"])
            job = jobs[job_id]
            key = normalized_request_key(job.normalized_request)
            keys.setdefault(key, []).append((split, job_id))
    overlaps = [values for values in keys.values() if len({split for split, _ in values}) > 1]
    if overlaps:
        rendered = "; ".join(",".join(f"{split}:{job_id}" for split, job_id in values) for values in overlaps)
        raise EvaluationContractError(f"normalized_request leakage across dataset splits: {rendered}")
    return {
        "status": "passed",
        "development_count": len(dataset["development"]),
        "holdout_count": len(dataset["holdout"]),
        "unique_normalized_request_count": len(keys),
    }


def _criterion_index(job: FrozenJob) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for state in job.question_states:
        question_id = str(state.get("question_id") or "")
        for criterion in state.get("criteria") or []:
            if not isinstance(criterion, dict):
                continue
            criterion_id = str(criterion.get("criterion_id") or "")
            if question_id and criterion_id:
                result[(question_id, criterion_id)] = criterion
    return result


def _chunk_quote_matches(evidence: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> bool:
    chunk = chunks.get(str(evidence.get("chunk_id") or ""))
    if not chunk:
        return False
    quote = str(evidence.get("cleaned_quote") or evidence.get("raw_quote") or evidence.get("evidence_text") or "")
    text = str(chunk.get("text") or "")
    if not quote or not text:
        return False
    def comparable(value: str) -> str:
        normalized = normalize_for_substring(unicodedata.normalize("NFKC", value))
        return "".join(character for character in normalized if unicodedata.category(character) != "Cf")

    return comparable(quote) in comparable(text)


def preflight_job(job: FrozenJob, row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = row or {}
    errors: list[str] = []
    warnings: list[str] = []
    evidence_ids = [str(item.get("id") or "") for item in job.evidence_map]
    evidence_id_set = {item for item in evidence_ids if item}
    if not evidence_ids or "" in evidence_ids or len(evidence_ids) != len(evidence_id_set):
        errors.append("accepted evidence ids must be non-empty and unique")
    # Historical accepted evidence maps may include ``inference_only`` rows. They passed
    # admission but are restricted to inference/recommendation use; quarantined/rejected rows
    # live in the separate rejected_evidence artifact and remain forbidden to both arms.
    not_accepted = [
        item for item in job.evidence_map
        if str(item.get("admission_status") or "accepted") not in {"accepted", "inference_only"}
    ]
    if not_accepted:
        errors.append("evidence_map contains non-accepted evidence")
    rejected_ids = {str(item.get("id") or "") for item in job.rejected_evidence if item.get("id")}
    if evidence_id_set & rejected_ids:
        errors.append("accepted and rejected evidence ids overlap")
    invalidated = {str(item) for item in row.get("invalidated_evidence_ids") or []}
    if evidence_id_set & invalidated:
        errors.append("dataset marks accepted evidence as later invalidated")

    states = {str(item.get("question_id") or ""): item for item in job.question_states}
    criterion_index = _criterion_index(job)
    chunks = {str(item.get("id") or ""): item for item in job.selected_chunks}
    missing_quote_trace: list[str] = []
    missing_mapping: list[str] = []
    for evidence in job.evidence_map:
        evidence_id = str(evidence.get("id") or "")
        question_id = str(evidence.get("question_id") or "")
        if question_id not in states:
            missing_mapping.append(evidence_id)
        for criterion_id in evidence.get("criterion_ids") or []:
            if (question_id, str(criterion_id)) not in criterion_index:
                missing_mapping.append(f"{evidence_id}:{criterion_id}")
        if not _chunk_quote_matches(evidence, chunks):
            missing_quote_trace.append(evidence_id)
    if missing_mapping:
        errors.append(f"evidence question/criterion mapping missing: {missing_mapping[:12]}")
    if missing_quote_trace:
        errors.append(f"evidence quote not traceable to frozen selected chunk: {missing_quote_trace[:12]}")

    finding_ids = [str(item.get("finding_id") or "") for item in job.findings]
    if not finding_ids or "" in finding_ids or len(finding_ids) != len(set(finding_ids)):
        errors.append("accepted finding ids must be non-empty and unique")
    finding_evidence_missing = sorted({
        str(evidence_id)
        for finding in job.findings
        for evidence_id in [*(finding.get("evidence_ids") or []), *(finding.get("counter_evidence_ids") or [])]
        if str(evidence_id) not in evidence_id_set
    })
    if finding_evidence_missing:
        errors.append(f"finding references unknown evidence: {finding_evidence_missing[:12]}")
    blueprint_finding_missing = sorted({
        str(finding_id)
        for section in job.report_blueprint.get("sections") or []
        for finding_id in section.get("finding_ids") or []
        if str(finding_id) not in set(finding_ids)
    })
    if blueprint_finding_missing:
        errors.append(f"blueprint references unknown finding: {blueprint_finding_missing[:12]}")
    blueprint_evidence_missing = sorted({
        str(evidence_id)
        for section in job.report_blueprint.get("sections") or []
        for evidence_id in section.get("evidence_ids") or []
        if str(evidence_id) not in evidence_id_set
    })
    if blueprint_evidence_missing:
        errors.append(f"blueprint references unknown evidence: {blueprint_evidence_missing[:12]}")
    if not job.materials:
        warnings.append("materials artifact is empty")

    result = {
        "job_id": job.job_id,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "evidence": len(job.evidence_map),
            "rejected_evidence": len(job.rejected_evidence),
            "findings": len(job.findings),
            "question_states": len(job.question_states),
            "materials": len(job.materials),
            "selected_chunks": len(job.selected_chunks),
            "blueprint_sections": len(job.report_blueprint.get("sections") or []),
        },
        "checkpoint_ids": {stage: checkpoint.get("checkpoint_id") for stage, checkpoint in job.checkpoints.items()},
        "checkpoint_hashes": {stage: checkpoint.get("checkpoint_sha256") for stage, checkpoint in job.checkpoints.items()},
        "artifact_hashes": {name: record.sha256 for name, record in job.artifacts.items()},
        "code_versions": {
            stage: checkpoint.get("code_version") for stage, checkpoint in job.checkpoints.items()
        },
        "configurations": {
            stage: checkpoint.get("configuration") for stage, checkpoint in job.checkpoints.items()
        },
    }
    if errors:
        raise EvaluationContractError(f"preflight failed for {job.job_id}: {'; '.join(errors)}")
    return result


def _is_conflict_evidence(evidence: dict[str, Any]) -> bool:
    text = " ".join(str(evidence.get(key) or "") for key in ("stance", "evidence_type", "claim_type"))
    flags = " ".join(str(item) for item in evidence.get("quality_flags") or [])
    value = f"{text} {flags}".casefold()
    return any(token in value for token in ("counter", "conflict", "limitation", "contradict", "negative"))


def _evidence_source_key(evidence: dict[str, Any]) -> str:
    return str(evidence.get("source_url") or evidence.get("canonical_project") or evidence.get("source_title") or "")


def _evidence_sort_key(evidence: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if str(evidence.get("admission_status") or "accepted") == "accepted" else 1,
        -float(evidence.get("directness_score") or 0),
        -float(evidence.get("evidence_score") or 0),
        -float(evidence.get("authority_score") or 0),
        str(evidence.get("id") or ""),
    )


def _criterion_level(criterion: dict[str, Any]) -> str:
    value = str(criterion.get("level") or "soft").lower()
    return value if value in {"hard", "soft", "optional"} else "soft"


def _guarded_cluster_key(evidence: dict[str, Any]) -> tuple[str, str, str, str]:
    """Keep nominally similar claims separate when subject or comparison axis differs."""
    evidence_id = str(evidence.get("id") or "")
    return (
        str(evidence.get("question_id") or ""),
        str(evidence.get("subject") or "__unspecified_subject__"),
        str(
            evidence.get("comparison_axis")
            or evidence.get("expected_comparison_axis")
            or "__unspecified_axis__"
        ),
        str(evidence.get("claim_cluster_id") or evidence_id),
    )


def _synthesis_boundary_id(evidence: dict[str, Any]) -> str:
    digest = hashlib.sha256("\x1f".join(_guarded_cluster_key(evidence)).encode("utf-8")).hexdigest()
    return f"boundary_{digest[:12]}"


def _guarded_comparison_cell_key(evidence: dict[str, Any]) -> tuple[str, str, str]:
    """Prefer semantic Subject x Axis cells over extractor-local matrix row ids."""
    subject = str(evidence.get("subject") or "")
    axis = str(evidence.get("comparison_axis") or evidence.get("expected_comparison_axis") or "")
    if subject or axis:
        return ("subject_axis", subject, axis)
    return ("matrix_cell", str(evidence.get("matrix_cell_id") or ""), "")


def _guarded_cluster_keys_by_breadth(
    items: Iterable[dict[str, Any]],
    *,
    max_clusters: int,
) -> list[tuple[str, str, str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(_guarded_cluster_key(item), []).append(item)
    cells: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = {}
    for cluster_key, rows in grouped.items():
        cell = _guarded_comparison_cell_key(sorted(rows, key=_evidence_sort_key)[0])
        cells.setdefault(cell, []).append(cluster_key)
    for cell in cells:
        cells[cell].sort(
            key=lambda key: (_evidence_sort_key(sorted(grouped[key], key=_evidence_sort_key)[0]), key)
        )
    selected: list[tuple[str, str, str, str]] = []
    offset = 0
    while len(selected) < max_clusters and any(offset < len(keys) for keys in cells.values()):
        round_keys = [keys[offset] for keys in cells.values() if offset < len(keys)]
        round_keys.sort(
            key=lambda key: (_evidence_sort_key(sorted(grouped[key], key=_evidence_sort_key)[0]), key)
        )
        for cluster_key in round_keys:
            selected.append(cluster_key)
            if len(selected) >= max_clusters:
                break
        offset += 1
    return selected


class DirectEvidenceContextManager:
    def __init__(
        self,
        *,
        max_input_tokens: int = 48000,
        prompt_reserve_tokens: int = 6500,
        profile: str = DIRECT_CONTEXT_PROFILE_BASELINE,
    ) -> None:
        self.max_input_tokens = max(2000, int(max_input_tokens))
        self.prompt_reserve_tokens = max(500, min(int(prompt_reserve_tokens), self.max_input_tokens - 500))
        if profile not in SUPPORTED_DIRECT_CONTEXT_PROFILES:
            raise EvaluationContractError(
                f"unsupported Direct Evidence context profile: {profile}; "
                f"expected one of {sorted(SUPPORTED_DIRECT_CONTEXT_PROFILES)}"
            )
        self.profile = profile

    @property
    def version(self) -> str:
        if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED:
            return CLUSTER_GUARDED_CONTEXT_MANAGER_VERSION
        return CONTEXT_MANAGER_VERSION

    def build(
        self,
        direct_inputs: dict[str, Any],
        section: dict[str, Any],
        *,
        overview: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if any(key in direct_inputs for key in ("findings", "finding_ids", "finding_admission")):
            raise EvaluationContractError("Arm E input contains forbidden Finding data")
        evidence_map = _expect_list(direct_inputs.get("evidence_map"), "direct evidence_map")
        question_states = _expect_list(direct_inputs.get("question_states"), "direct question_states")
        section_id = str(section.get("section_id") or "")
        section_questions = {
            str(item.get("question_id") or "")
            for item in section.get("research_questions") or []
            if isinstance(item, dict) and item.get("question_id")
        }
        if overview:
            candidates = list(evidence_map)
        else:
            candidates = [
                item for item in evidence_map
                if str(item.get("section_id") or "") == section_id
                or str(item.get("question_id") or "") in section_questions
            ]
        states = [
            state for state in question_states
            if overview or str(state.get("section_id") or "") == section_id
            or str(state.get("question_id") or "") in section_questions
        ]
        if not states:
            states = [
                {
                    "question_id": str(item.get("question_id") or ""),
                    "section_id": section_id,
                    "question": str(item.get("question") or ""),
                    "criteria": list(item.get("criteria") or []),
                }
                for item in section.get("research_questions") or []
                if isinstance(item, dict)
            ]

        criterion_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for state in states:
            question_id = str(state.get("question_id") or "")
            for criterion in state.get("criteria") or []:
                if isinstance(criterion, dict) and criterion.get("criterion_id"):
                    criterion_by_key[(question_id, str(criterion["criterion_id"]))] = criterion

        by_id = {str(item.get("id") or ""): item for item in candidates if item.get("id")}
        selected: list[str] = []
        selected_set: set[str] = set()
        selected_sources: set[str] = set()
        budget = self.max_input_tokens - self.prompt_reserve_tokens
        state_budget_view = [
            {
                "question_id": state.get("question_id"),
                "section_id": state.get("section_id"),
                "question": state.get("question"),
                "purpose": state.get("purpose"),
                "criteria": [
                    {
                        "criterion_id": criterion.get("criterion_id"),
                        "description": criterion.get("description"),
                        "level": criterion.get("level"),
                        "status": criterion.get("status"),
                    }
                    for criterion in state.get("criteria") or []
                    if isinstance(criterion, dict)
                ],
            }
            for state in states
        ]
        consumed = estimate_tokens({
            "section": {key: value for key, value in section.items() if key != "dossier"},
            "question_states": state_budget_view,
        })

        def add(evidence: dict[str, Any]) -> bool:
            nonlocal consumed
            evidence_id = str(evidence.get("id") or "")
            if not evidence_id or evidence_id in selected_set:
                return False
            cost = estimate_tokens(self._packet_row(evidence))
            if consumed + cost > budget:
                return False
            selected.append(evidence_id)
            selected_set.add(evidence_id)
            selected_sources.add(_evidence_source_key(evidence))
            consumed += cost
            return True

        def ranked(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(items, key=lambda item: (_evidence_sort_key(item), _evidence_source_key(item)))

        if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED:
            selection_diagnostics = self._select_cluster_guarded(
                candidates=candidates,
                criterion_by_key=criterion_by_key,
                comparison_context=str(section.get("render_as") or "") == "comparison_matrix",
                add=add,
                ranked=ranked,
                selected=selected,
                selected_set=selected_set,
                selected_sources=selected_sources,
            )
        else:
            selection_diagnostics = self._select_baseline(
                candidates=candidates,
                criterion_by_key=criterion_by_key,
                by_id=by_id,
                add=add,
                ranked=ranked,
                selected=selected,
                selected_set=selected_set,
                selected_sources=selected_sources,
            )

        packet = self._organize(section, states, [by_id[item] for item in selected if item in by_id])
        selected_clusters = {
            str(by_id[item].get("claim_cluster_id") or "") for item in selected if item in by_id
        }
        all_clusters = {str(item.get("claim_cluster_id") or "") for item in candidates if item.get("claim_cluster_id")}
        selected_cells = {
            (
                str(by_id[item].get("matrix_cell_id") or ""),
                str(by_id[item].get("subject") or ""),
                str(by_id[item].get("comparison_axis") or by_id[item].get("expected_comparison_axis") or ""),
            )
            for item in selected if item in by_id
            if any((by_id[item].get("matrix_cell_id"), by_id[item].get("subject"), by_id[item].get("comparison_axis")))
        }
        diagnostics = {
            "context_manager_version": self.version,
            "context_profile": self.profile,
            "max_input_tokens": self.max_input_tokens,
            "prompt_reserve_tokens": self.prompt_reserve_tokens,
            "material_budget_tokens": budget,
            "estimated_material_tokens": estimate_tokens(packet),
            "candidate_evidence_count": len(candidates),
            "selected_evidence_count": len(selected),
            "selected_evidence_ids": selected,
            "cropped_evidence_ids": sorted(set(by_id) - selected_set),
            "candidate_claim_cluster_count": len(all_clusters),
            "selected_claim_cluster_count": len(selected_clusters),
            "cropped_claim_cluster_ids": sorted(all_clusters - selected_clusters),
            "selected_source_count": len({_evidence_source_key(by_id[item]) for item in selected if item in by_id}),
            "selected_comparison_cells": [list(item) for item in sorted(selected_cells)],
            "criterion_coverage": self._criterion_coverage(states, [by_id[item] for item in selected if item in by_id]),
            **selection_diagnostics,
        }
        return packet, diagnostics

    def _select_baseline(
        self,
        *,
        candidates: list[dict[str, Any]],
        criterion_by_key: dict[tuple[str, str], dict[str, Any]],
        by_id: dict[str, dict[str, Any]],
        add: Callable[[dict[str, Any]], bool],
        ranked: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]],
        selected: list[str],
        selected_set: set[str],
        selected_sources: set[str],
    ) -> dict[str, Any]:
        # 1. Each hard criterion gets a representative cluster before depth is added anywhere.
        for (question_id, criterion_id), criterion in sorted(criterion_by_key.items()):
            if _criterion_level(criterion) != "hard":
                continue
            pool = [item for item in candidates if str(item.get("question_id") or "") == question_id]
            explicit = [
                item for item in pool
                if criterion_id in {str(value) for value in item.get("criterion_ids") or []}
            ]
            add((ranked(explicit or pool) or [{}])[0])

        # 2. Comparison cell breadth: cover each matrix cell or Subject x Axis before adding depth.
        cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for item in candidates:
            cell = (
                str(item.get("matrix_cell_id") or ""),
                str(item.get("subject") or ""),
                str(item.get("comparison_axis") or item.get("expected_comparison_axis") or ""),
            )
            if any(cell):
                cells.setdefault(cell, []).append(item)
        for cell in sorted(cells):
            add((ranked(cells[cell]) or [{}])[0])

        # 3. Counter, limitation, and conflict evidence is always attempted explicitly.
        for item in ranked(item for item in candidates if _is_conflict_evidence(item)):
            add(item)

        # 4. Add one distinct-source corroborator for already selected claim clusters.
        selected_clusters = {
            str(by_id[evidence_id].get("claim_cluster_id") or "")
            for evidence_id in selected if evidence_id in by_id
        }
        for cluster_id in sorted(item for item in selected_clusters if item):
            cluster_rows = ranked(
                item for item in candidates if str(item.get("claim_cluster_id") or "") == cluster_id
            )
            for item in cluster_rows:
                if _evidence_source_key(item) not in selected_sources and add(item):
                    break

        # 5-6. Round-robin criterion coverage, soft before optional.
        for level in ("soft", "optional"):
            groups: list[list[dict[str, Any]]] = []
            for (question_id, criterion_id), criterion in sorted(criterion_by_key.items()):
                if _criterion_level(criterion) != level:
                    continue
                pool = [
                    item for item in candidates
                    if str(item.get("question_id") or "") == question_id
                    and criterion_id in {str(value) for value in item.get("criterion_ids") or []}
                ]
                groups.append(ranked(pool))
            offset = 0
            while any(offset < len(group) for group in groups):
                for group in groups:
                    if offset < len(group):
                        add(group[offset])
                offset += 1

        # 7-8. Fill remaining capacity with source-diverse rows; duplicates naturally come last.
        remainder = ranked(item for item in candidates if str(item.get("id") or "") not in selected_set)
        for item in sorted(
            remainder,
            key=lambda evidence: (
                1 if _evidence_source_key(evidence) in selected_sources else 0,
                1 if not evidence.get("criterion_ids") else 0,
                _evidence_sort_key(evidence),
            ),
        ):
            add(item)
        return {"selection_policy": "baseline_fill_to_budget"}

    def _select_cluster_guarded(
        self,
        *,
        candidates: list[dict[str, Any]],
        criterion_by_key: dict[tuple[str, str], dict[str, Any]],
        comparison_context: bool,
        add: Callable[[dict[str, Any]], bool],
        ranked: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]],
        selected: list[str],
        selected_set: set[str],
        selected_sources: set[str],
    ) -> dict[str, Any]:
        clusters: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for item in candidates:
            clusters.setdefault(_guarded_cluster_key(item), []).append(item)

        selected_cluster_keys: set[tuple[str, str, str, str]] = set()
        primary_by_cluster: dict[tuple[str, str, str, str], str] = {}

        def ordered_cluster_keys(items: Iterable[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
            pool: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
            for item in items:
                pool.setdefault(_guarded_cluster_key(item), []).append(item)
            return sorted(
                pool,
                key=lambda key: (_evidence_sort_key(ranked(pool[key])[0]), key),
            )

        def add_primary(cluster_key: tuple[str, str, str, str]) -> bool:
            if cluster_key in selected_cluster_keys:
                return True
            rows = ranked(clusters.get(cluster_key) or [])
            rows = sorted(
                rows,
                key=lambda item: (
                    1 if _evidence_source_key(item) in selected_sources else 0,
                    _evidence_sort_key(item),
                ),
            )
            for item in rows:
                if add(item):
                    selected_cluster_keys.add(cluster_key)
                    primary_by_cluster.setdefault(cluster_key, str(item.get("id") or ""))
                    return True
                if str(item.get("id") or "") in selected_set:
                    selected_cluster_keys.add(cluster_key)
                    primary_by_cluster.setdefault(cluster_key, str(item.get("id") or ""))
                    return True
            return False

        # Hard criteria get bounded breadth across distinct semantic clusters.
        for (question_id, criterion_id), criterion in sorted(criterion_by_key.items()):
            if _criterion_level(criterion) != "hard":
                continue
            question_pool = [
                item for item in candidates
                if str(item.get("question_id") or "") == question_id
            ]
            explicit = [
                item for item in question_pool
                if criterion_id in {str(value) for value in item.get("criterion_ids") or []}
            ]
            declared = [
                item for item in question_pool
                if criterion_id in {
                    str(value) for value in item.get("declared_criterion_ids") or []
                }
            ]
            for cluster_key in _guarded_cluster_keys_by_breadth(
                explicit or declared or question_pool,
                max_clusters=8,
            ):
                add_primary(cluster_key)

        # Preserve comparison breadth before adding depth to any one cluster.
        if comparison_context:
            cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            for item in candidates:
                cell = _guarded_comparison_cell_key(item)
                if cell[1] or cell[2]:
                    cells.setdefault(cell, []).append(item)
            for cell in sorted(cells):
                keys = ordered_cluster_keys(cells[cell])
                if keys:
                    add_primary(keys[0])

        # Every conflict/limitation cluster gets an explicit attempt, with at most one such row.
        conflict_cluster_keys = ordered_cluster_keys(
            item for item in candidates if _is_conflict_evidence(item)
        )
        for cluster_key in conflict_cluster_keys:
            add_primary(cluster_key)
            conflict_rows = ranked(
                item for item in clusters.get(cluster_key) or [] if _is_conflict_evidence(item)
            )
            if not any(str(item.get("id") or "") in selected_set for item in conflict_rows):
                for item in conflict_rows:
                    if add(item):
                        break

        # Add one different-source corroborator per selected cluster, never unrestricted depth.
        for cluster_key in sorted(selected_cluster_keys):
            primary = primary_by_cluster.get(cluster_key, "")
            primary_source = _evidence_source_key(next(
                (item for item in clusters.get(cluster_key) or [] if str(item.get("id") or "") == primary),
                {},
            ))
            corroborators = ranked(
                item for item in clusters.get(cluster_key) or []
                if str(item.get("id") or "") != primary
                and _evidence_source_key(item) != primary_source
            )
            for item in corroborators:
                if add(item):
                    break

        # Soft and optional criteria receive bounded cluster breadth after the protected material.
        for level, limit in (("soft", 2), ("optional", 1)):
            for (question_id, criterion_id), criterion in sorted(criterion_by_key.items()):
                if _criterion_level(criterion) != level:
                    continue
                explicit = [
                    item for item in candidates
                    if str(item.get("question_id") or "") == question_id
                    and criterion_id in {str(value) for value in item.get("criterion_ids") or []}
                ]
                for cluster_key in ordered_cluster_keys(explicit)[:limit]:
                    add_primary(cluster_key)

        # Imperfect historical criterion binding may leave useful context unbound. Keep only a
        # small, subject/axis-diverse sample instead of filling all remaining token capacity.
        unbound = [item for item in candidates if not item.get("criterion_ids")]
        unbound_by_question: dict[str, list[dict[str, Any]]] = {}
        for item in unbound:
            unbound_by_question.setdefault(str(item.get("question_id") or ""), []).append(item)
        for question_id in sorted(unbound_by_question):
            added = 0
            seen_cells: set[tuple[str, str]] = set()
            for cluster_key in ordered_cluster_keys(unbound_by_question[question_id]):
                cell = (cluster_key[1], cluster_key[2])
                if cell in seen_cells:
                    continue
                if add_primary(cluster_key):
                    seen_cells.add(cell)
                    added += 1
                if added >= 2:
                    break

        selected_cluster_counts: dict[str, int] = {}
        candidate_by_id = {
            str(item.get("id") or ""): item for item in candidates if str(item.get("id") or "")
        }
        for evidence_id in selected:
            evidence = candidate_by_id.get(evidence_id)
            if evidence is None:
                continue
            boundary_id = _synthesis_boundary_id(evidence)
            selected_cluster_counts[boundary_id] = selected_cluster_counts.get(boundary_id, 0) + 1
        candidate_conflict_ids = {
            evidence_id for evidence_id, item in candidate_by_id.items() if _is_conflict_evidence(item)
        }
        selected_conflict_ids = candidate_conflict_ids & selected_set
        candidate_comparison_cells = (
            {_guarded_comparison_cell_key(item) for item in candidates}
            if comparison_context else set()
        )
        selected_comparison_cells = (
            {
                _guarded_comparison_cell_key(candidate_by_id[evidence_id])
                for evidence_id in selected
                if evidence_id in candidate_by_id
            }
            if comparison_context else set()
        )
        return {
            "selection_policy": "cluster_guarded_no_budget_fill",
            "guarded_cluster_count": len(clusters),
            "selected_synthesis_boundary_count": len(selected_cluster_counts),
            "selected_rows_per_synthesis_boundary": dict(sorted(selected_cluster_counts.items())),
            "max_selected_rows_per_synthesis_boundary": max(selected_cluster_counts.values(), default=0),
            "unrestricted_remainder_fill": False,
            "comparison_breadth_applied": comparison_context,
            "max_hard_criterion_clusters": 8,
            "candidate_conflict_evidence_count": len(candidate_conflict_ids),
            "selected_conflict_evidence_count": len(selected_conflict_ids),
            "cropped_conflict_evidence_ids": sorted(candidate_conflict_ids - selected_conflict_ids),
            "candidate_guarded_comparison_cell_count": len(candidate_comparison_cells),
            "selected_guarded_comparison_cell_count": len(selected_comparison_cells),
        }

    def _packet_row(self, evidence: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": evidence.get("id"),
            "atomic_claim": evidence.get("atomic_claim") or evidence.get("claim"),
            "quote": evidence.get("cleaned_quote") or evidence.get("raw_quote") or evidence.get("evidence_text"),
            "source_url": evidence.get("source_url"),
            "source_title": evidence.get("source_title"),
            "source_type": evidence.get("source_type"),
            "publisher_relation": evidence.get("publisher_relation"),
            "subject": evidence.get("subject"),
            "subject_relation": evidence.get("subject_relation"),
            "comparison_axis": evidence.get("comparison_axis"),
            "comparison_metric": evidence.get("comparison_metric"),
            "expected_comparison_axis": evidence.get("expected_comparison_axis"),
            "matrix_cell_id": evidence.get("matrix_cell_id"),
            "scope": evidence.get("scope"),
            "scope_preservation": {
                "required": True,
                "basis": "quote_plus_structured_metadata",
                "subject": evidence.get("subject"),
                "scope": evidence.get("scope"),
                "target_fit": evidence.get("target_fit"),
                "instruction": (
                    "带引用事实必须保留 Quote 中的对象、领域、数据集、条件、时间和强度限定；"
                    "跨范围推广只能放在引用标记之后，作为未引用分析。"
                ),
            },
            "stance": evidence.get("stance"),
            "support_relation": evidence.get("support_relation"),
            "target_fit": evidence.get("target_fit"),
            "criterion_ids": [str(item) for item in evidence.get("criterion_ids") or []],
            "declared_criterion_ids": [
                str(item) for item in evidence.get("declared_criterion_ids") or []
            ],
            "criterion_binding_status": evidence.get("criterion_binding_status"),
            "comparison_axis_binding_status": evidence.get("comparison_axis_binding_status"),
            "directness_score": evidence.get("directness_score"),
            "evidence_score": evidence.get("evidence_score"),
            "claim_cluster_id": evidence.get("claim_cluster_id"),
            "conflict_or_limitation": _is_conflict_evidence(evidence),
        }
        if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED:
            row.update({
                "synthesis_boundary_id": _synthesis_boundary_id(evidence),
                "atomic_claim_role": "navigation_only",
                "quote_role": "factual_authority",
            })
        return row

    def _organize(
        self,
        section: dict[str, Any],
        states: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence_by_question: dict[str, list[dict[str, Any]]] = {}
        for item in evidence:
            evidence_by_question.setdefault(str(item.get("question_id") or ""), []).append(item)
        threads = []
        seen_evidence: set[str] = set()

        def bounded_guarded_rows(
            items: list[dict[str, Any]],
            *,
            max_clusters: int,
        ) -> list[dict[str, Any]]:
            allowed = set(_guarded_cluster_keys_by_breadth(items, max_clusters=max_clusters))
            return [item for item in items if _guarded_cluster_key(item) in allowed]

        for state in states:
            question_id = str(state.get("question_id") or "")
            rows = evidence_by_question.get(question_id, [])
            criteria = []
            for criterion in state.get("criteria") or []:
                if not isinstance(criterion, dict):
                    continue
                criterion_id = str(criterion.get("criterion_id") or "")
                directly_bound = [
                    item for item in rows
                    if criterion_id in {str(value) for value in item.get("criterion_ids") or []}
                ]
                declared_candidates = [
                    item for item in rows
                    if criterion_id in {
                        str(value) for value in item.get("declared_criterion_ids") or []
                    }
                ]
                if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED:
                    if directly_bound:
                        matched = directly_bound
                        binding_basis = "direct_criterion_binding"
                    elif declared_candidates:
                        matched = bounded_guarded_rows(declared_candidates, max_clusters=8)
                        binding_basis = "declared_criterion_candidates"
                    elif _criterion_level(criterion) == "hard":
                        matched = bounded_guarded_rows(rows, max_clusters=8)
                        binding_basis = "question_scope_candidates"
                    else:
                        matched = []
                        binding_basis = "no_selected_candidates"
                else:
                    matched = directly_bound
                    binding_basis = "direct_criterion_binding" if matched else "no_selected_candidates"
                clusters: dict[str, list[dict[str, Any]]] = {}
                for item in matched:
                    evidence_id = str(item.get("id") or "")
                    seen_evidence.add(evidence_id)
                    cluster_id = (
                        _synthesis_boundary_id(item)
                        if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
                        else str(item.get("claim_cluster_id") or evidence_id)
                    )
                    clusters.setdefault(cluster_id, []).append(item)
                criteria.append({
                    "criterion_id": criterion_id,
                    "description": criterion.get("description"),
                    "level": _criterion_level(criterion),
                    "coverage_status": criterion.get("status"),
                    **({
                        "evidence_binding_basis": binding_basis,
                        "candidate_material_only": binding_basis in {
                            "declared_criterion_candidates",
                            "question_scope_candidates",
                        },
                    } if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED else {}),
                    "claim_clusters": [
                        {
                            "claim_cluster_id": str(items[0].get("claim_cluster_id") or ""),
                            **({
                                "synthesis_boundary_id": cluster_id,
                                "subject": str(items[0].get("subject") or ""),
                                "comparison_axis": str(
                                    items[0].get("comparison_axis")
                                    or items[0].get("expected_comparison_axis")
                                    or ""
                                ),
                                "cross_boundary_synthesis": "forbidden_without_relational_quote",
                            } if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED else {}),
                            "evidence": [self._packet_row(item) for item in sorted(items, key=_evidence_sort_key)],
                        }
                        for cluster_id, items in sorted(clusters.items())
                    ],
                })
            unbound = [
                self._packet_row(item) for item in rows if str(item.get("id") or "") not in seen_evidence
            ]
            seen_evidence.update(str(item.get("id") or "") for item in rows)
            threads.append({
                "question_id": question_id,
                "question": state.get("question"),
                "purpose": state.get("purpose"),
                "criteria": criteria,
                "unbound_context": unbound,
            })
        global_unbound = [
            self._packet_row(item) for item in evidence if str(item.get("id") or "") not in seen_evidence
        ]
        return {
            "schema_version": (
                "direct-evidence-packet/v2-cluster-guarded"
                if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
                else "direct-evidence-packet/v1"
            ),
            "context_profile": self.profile,
            "section_id": section.get("section_id"),
            "section_title": section.get("title"),
            "section_purpose": section.get("purpose"),
            "question_threads": threads,
            "unbound_context": global_unbound,
            "known_gaps": [str(item) for item in section.get("known_gaps") or []],
        }

    def _criterion_coverage(
        self, states: list[dict[str, Any]], evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        result = []
        for state in states:
            question_id = str(state.get("question_id") or "")
            for criterion in state.get("criteria") or []:
                if not isinstance(criterion, dict):
                    continue
                criterion_id = str(criterion.get("criterion_id") or "")
                directly_bound = [
                    str(item.get("id") or "") for item in evidence
                    if str(item.get("question_id") or "") == question_id
                    and criterion_id in {str(value) for value in item.get("criterion_ids") or []}
                ]
                declared_rows = [
                    item for item in evidence
                    if str(item.get("question_id") or "") == question_id
                    and criterion_id in {
                        str(value) for value in item.get("declared_criterion_ids") or []
                    }
                ]
                question_scope_rows = [
                    item for item in evidence
                    if str(item.get("question_id") or "") == question_id
                ]
                declared_keys = set(_guarded_cluster_keys_by_breadth(
                    declared_rows,
                    max_clusters=8,
                ))
                question_scope_keys = set(_guarded_cluster_keys_by_breadth(
                    question_scope_rows,
                    max_clusters=8,
                ))
                declared_candidates = [
                    str(item.get("id") or "") for item in declared_rows
                    if _guarded_cluster_key(item) in declared_keys
                ]
                question_scope_candidates = [
                    str(item.get("id") or "") for item in question_scope_rows
                    if _guarded_cluster_key(item) in question_scope_keys
                ]
                if directly_bound:
                    matched = directly_bound
                    coverage_basis = "direct_criterion_binding"
                elif self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED and declared_candidates:
                    matched = declared_candidates
                    coverage_basis = "declared_criterion_candidates"
                elif (
                    self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
                    and _criterion_level(criterion) == "hard"
                    and question_scope_candidates
                ):
                    matched = question_scope_candidates
                    coverage_basis = "question_scope_candidates"
                else:
                    matched = []
                    coverage_basis = "none"
                row = {
                    "question_id": question_id,
                    "criterion_id": criterion_id,
                    "level": _criterion_level(criterion),
                    "selected_evidence_ids": matched,
                    "covered_in_context": bool(matched),
                }
                if self.profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED:
                    row.update({
                        "coverage_basis": coverage_basis,
                        "candidate_material_only": coverage_basis in {
                            "declared_criterion_candidates",
                            "question_scope_candidates",
                        },
                    })
                result.append(row)
        return result


def build_reference_inventory(job: FrozenJob, section_ids: set[str] | None = None) -> dict[str, Any]:
    """Export a Finding-independent inventory candidate for later human freezing."""
    evidence = [
        item for item in job.evidence_map
        if section_ids is None or str(item.get("section_id") or "") in section_ids
    ]
    by_question: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        by_question.setdefault(str(item.get("question_id") or ""), []).append(item)
    sections: dict[str, dict[str, Any]] = {}
    for state in job.question_states:
        section_id = str(state.get("section_id") or "")
        if section_ids is not None and section_id not in section_ids:
            continue
        section = sections.setdefault(section_id, {"section_id": section_id, "criteria": []})
        question_id = str(state.get("question_id") or "")
        rows = by_question.get(question_id, [])
        criteria = [item for item in state.get("criteria") or [] if isinstance(item, dict)]
        for criterion in criteria:
            criterion_id = str(criterion.get("criterion_id") or "")
            directly_bound = [
                item for item in rows
                if criterion_id in {str(value) for value in item.get("criterion_ids") or []}
            ]
            declared_candidates = [
                item for item in rows
                if criterion_id in {str(value) for value in item.get("declared_criterion_ids") or []}
                and str(item.get("id") or "") not in {
                    str(bound.get("id") or "") for bound in directly_bound
                }
            ]
            primary_candidates = [*directly_bound, *declared_candidates]
            if primary_candidates:
                fallback_candidates = []
            else:
                question_unbound = [
                    item for item in rows
                    if not item.get("criterion_ids") and not item.get("declared_criterion_ids")
                ]
                include_question_scope = _criterion_level(criterion) == "hard" or len(criteria) == 1
                fallback_candidates = rows if include_question_scope else question_unbound
            candidates_by_id = {
                str(item.get("id") or ""): item
                for item in [*primary_candidates, *fallback_candidates]
                if item.get("id")
            }
            candidates = list(candidates_by_id.values())
            direct_ids = {str(item.get("id") or "") for item in directly_bound if item.get("id")}
            declared_ids = {str(item.get("id") or "") for item in declared_candidates if item.get("id")}
            question_scope_ids = sorted(set(candidates_by_id) - direct_ids - declared_ids)
            clusters = sorted({
                str(item.get("claim_cluster_id") or "")
                for item in candidates
                if item.get("claim_cluster_id")
            })
            direct_clusters = sorted({
                str(item.get("claim_cluster_id") or "")
                for item in directly_bound
                if item.get("claim_cluster_id")
            })
            candidate_counter = sorted({
                str(item.get("claim_cluster_id") or "") for item in candidates
                if item.get("claim_cluster_id") and _is_conflict_evidence(item)
            })
            direct_counter = sorted({
                str(item.get("claim_cluster_id") or "") for item in directly_bound
                if item.get("claim_cluster_id") and _is_conflict_evidence(item)
            })
            subjects = sorted({str(item.get("subject") or "") for item in candidates if item.get("subject")})
            axes = sorted({str(item.get("comparison_axis") or "") for item in candidates if item.get("comparison_axis")})
            if directly_bound:
                answerability = "conflicted" if direct_counter else "supported"
                answerability_basis = "direct_criterion_binding"
            elif declared_candidates:
                answerability = "partial"
                answerability_basis = "declared_criterion_candidates_require_human_review"
            elif candidates:
                answerability = "partial"
                answerability_basis = "question_scope_candidates_require_human_review"
            else:
                answerability = "unsupported"
                answerability_basis = "no_question_scope_candidates"
            section["criteria"].append({
                "question_id": question_id,
                "criterion_id": criterion_id,
                "description": criterion.get("description"),
                "level": _criterion_level(criterion),
                "answerability": answerability,
                "answerability_basis": answerability_basis,
                "candidate_claim_cluster_ids": clusters,
                "required_claim_cluster_ids": [],
                "optional_claim_cluster_ids": direct_clusters,
                "counter_claim_cluster_ids": direct_counter,
                "candidate_counter_claim_cluster_ids": candidate_counter,
                "subjects": subjects,
                "comparison_axes": axes,
                "evidence_ids": [str(item.get("id") or "") for item in sorted(candidates, key=_evidence_sort_key)],
                "direct_criterion_evidence_ids": sorted(direct_ids),
                "declared_criterion_candidate_evidence_ids": sorted(declared_ids),
                "question_scope_candidate_evidence_ids": question_scope_ids,
                "review_note": "",
                "human_review_status": "pending",
            })
    evidence_catalog = []
    for item in sorted(evidence, key=_evidence_sort_key):
        evidence_catalog.append({
            "evidence_id": str(item.get("id") or ""),
            "section_id": str(item.get("section_id") or ""),
            "question_id": str(item.get("question_id") or ""),
            "criterion_ids": [str(value) for value in item.get("criterion_ids") or []],
            "claim_cluster_id": str(item.get("claim_cluster_id") or ""),
            "atomic_claim": str(item.get("atomic_claim") or item.get("claim") or ""),
            "quote": str(item.get("cleaned_quote") or item.get("raw_quote") or item.get("evidence_text") or ""),
            "source_url": str(item.get("source_url") or ""),
            "source_title": str(item.get("source_title") or ""),
            "source_type": str(item.get("source_type") or ""),
            "chunk_id": str(item.get("chunk_id") or ""),
            "subject": str(item.get("subject") or ""),
            "comparison_axis": str(item.get("comparison_axis") or ""),
            "stance": str(item.get("stance") or ""),
            "scope": str(item.get("scope") or ""),
            "directness_score": item.get("directness_score"),
            "evidence_score": item.get("evidence_score"),
            "admission_status": str(item.get("admission_status") or ""),
        })
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "job_id": job.job_id,
        "normalized_request_hash": normalized_request_key(job.normalized_request),
        "source_evidence_artifact_sha256": job.artifacts["evidence_map"].sha256,
        "generated_at": utc_now(),
        "status": "candidate_unfrozen",
        "review_contract": {
            "instructions": [
                "Review atomic_claim and quote before changing human_review_status to reviewed.",
                "Question-scope candidates are discovery aids, not proof that the criterion is supported.",
                "For supported, partial, or conflicted hard/soft criteria, select at least one required claim cluster.",
                "Explain why the evidence is sufficient, partial, conflicted, or unsupported in review_note.",
            ],
            "finding_independent": True,
        },
        "sections": [sections[key] for key in sorted(sections)],
        "evidence_catalog": evidence_catalog,
    }
    inventory["candidate_sha256"] = stable_hash(inventory)
    return inventory


def inventory_metrics(candidate: dict[str, Any], inventory: dict[str, Any], job: FrozenJob) -> dict[str, Any]:
    used_evidence_ids = {
        str(evidence_id)
        for row in candidate.get("report_document", {}).get("claim_ledger") or []
        for evidence_id in row.get("evidence_ids") or []
    }
    evidence_by_id = {str(item.get("id") or ""): item for item in job.evidence_map}
    used_clusters = {
        str(evidence_by_id[evidence_id].get("claim_cluster_id") or "")
        for evidence_id in used_evidence_ids
        if evidence_id in evidence_by_id and evidence_by_id[evidence_id].get("claim_cluster_id")
    }
    eligible = []
    required_clusters: set[str] = set()
    per_criterion = []
    for section in inventory.get("sections") or []:
        for criterion in section.get("criteria") or []:
            if str(criterion.get("level") or "") not in {"hard", "soft"}:
                continue
            answerability = str(criterion.get("answerability") or "")
            if answerability not in {"supported", "partial", "conflicted"}:
                continue
            eligible.append(criterion)
            required = {str(item) for item in criterion.get("required_claim_cluster_ids") or []}
            required_clusters.update(required)
            cited = sorted(required & used_clusters)
            per_criterion.append({
                "section_id": section.get("section_id"),
                "question_id": criterion.get("question_id"),
                "criterion_id": criterion.get("criterion_id"),
                "level": criterion.get("level"),
                "required_claim_cluster_ids": sorted(required),
                "cited_required_claim_cluster_ids": cited,
                "coverage_proxy": bool(cited) if required else None,
            })
    countable = [item for item in per_criterion if item["coverage_proxy"] is not None]
    covered = sum(bool(item["coverage_proxy"]) for item in countable)
    used_sources = {
        _evidence_source_key(evidence_by_id[item]) for item in used_evidence_ids if item in evidence_by_id
    }
    return {
        "mode": "citation_cluster_proxy_requires_human_confirmation",
        "eligible_criterion_count": len(eligible),
        "countable_criterion_count": len(countable),
        "covered_criterion_count": covered,
        "supported_criterion_coverage_proxy": round(covered / len(countable), 3) if countable else None,
        "required_claim_cluster_count": len(required_clusters),
        "used_required_claim_cluster_count": len(required_clusters & used_clusters),
        "material_retention_proxy": (
            round(len(required_clusters & used_clusters) / len(required_clusters), 3)
            if required_clusters else None
        ),
        "used_evidence_count": len(used_evidence_ids),
        "used_source_count": len(used_sources),
        "per_criterion": per_criterion,
    }


def claim_evidence_fidelity_audit(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    job: FrozenJob,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Build a Finding-independent, quote-level review packet for every final claim."""
    report = candidate.get("report_document") if isinstance(candidate.get("report_document"), dict) else {}
    claims = [item for item in report.get("claim_ledger") or [] if isinstance(item, dict)]
    evidence_by_id = {str(item.get("id") or ""): item for item in job.evidence_map if item.get("id")}
    entailment = validation.get("entailment") if isinstance(validation.get("entailment"), dict) else {}
    judgments = {
        int(item["claim_index"]): item
        for item in entailment.get("judgments") or []
        if isinstance(item, dict) and str(item.get("claim_index", "")).lstrip("-").isdigit()
    }
    rows = []
    used_evidence_ids: set[str] = set()
    for index, claim in enumerate(claims):
        evidence_ids = [str(item) for item in claim.get("evidence_ids") or [] if str(item)]
        used_evidence_ids.update(evidence_ids)
        judgment = judgments.get(index) or {}
        status = str(judgment.get("status") or "not_reviewed")
        automatic_error_tags = []
        if status == "unsupported":
            automatic_error_tags.append("unsupported_or_contradicted")
        elif status == "label_mismatch":
            automatic_error_tags.append("scope_or_support_mismatch")
        elif status == "non_atomic":
            automatic_error_tags.append("non_atomic_claim")
        elif status == "validation_unavailable":
            automatic_error_tags.append("semantic_validation_unavailable")
        rows.append({
            "review_unit_origin": "claim_ledger",
            "claim_index": index,
            "section_id": claim.get("section_id"),
            "claim": claim.get("cited_span") or claim.get("text") or claim.get("claim"),
            "cited_span": claim.get("cited_span") or claim.get("text") or claim.get("claim"),
            "uncited_analysis_span": claim.get("uncited_analysis_span") or "",
            "full_atomic_unit": claim.get("full_atomic_unit") or claim.get("text") or claim.get("claim"),
            "claim_context": claim.get("claim_context"),
            "claim_type": claim.get("claim_type"),
            "support_scope": claim.get("support_scope"),
            "cited_evidence_ids": evidence_ids,
            "original_evidence": [
                {
                    "evidence_id": evidence_id,
                    "atomic_claim": evidence_by_id[evidence_id].get("atomic_claim")
                    or evidence_by_id[evidence_id].get("claim"),
                    "quote": evidence_by_id[evidence_id].get("cleaned_quote")
                    or evidence_by_id[evidence_id].get("raw_quote")
                    or evidence_by_id[evidence_id].get("evidence_text"),
                    "source_title": evidence_by_id[evidence_id].get("source_title"),
                    "source_url": evidence_by_id[evidence_id].get("source_url"),
                    "source_type": evidence_by_id[evidence_id].get("source_type"),
                    "publisher_relation": evidence_by_id[evidence_id].get("publisher_relation"),
                    "subject": evidence_by_id[evidence_id].get("subject"),
                    "subject_relation": evidence_by_id[evidence_id].get("subject_relation"),
                    "comparison_axis": evidence_by_id[evidence_id].get("comparison_axis"),
                    "expected_comparison_axis": evidence_by_id[evidence_id].get("expected_comparison_axis"),
                    "scope": evidence_by_id[evidence_id].get("scope"),
                    "stance": evidence_by_id[evidence_id].get("stance"),
                    "support_relation": evidence_by_id[evidence_id].get("support_relation"),
                    "target_fit": evidence_by_id[evidence_id].get("target_fit"),
                    "claim_cluster_id": evidence_by_id[evidence_id].get("claim_cluster_id"),
                    "conflict_or_limitation": _is_conflict_evidence(evidence_by_id[evidence_id]),
                }
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            ],
            "automatic_entailment": {
                "status": status,
                "reason": judgment.get("reason"),
                "mode": judgment.get("mode"),
                "claim_scope": judgment.get("claim_scope"),
                "scope_relation": _scope_relation_diagnostic(judgment),
                "scope_relation_source": "existing_entailment_status_and_reason",
                "confirmed": judgment.get("confirmed"),
            },
            "automatic_error_tags": automatic_error_tags,
            "human_review": {
                "status": "pending",
                "verdict": None,
                "severity": None,
                "error_types": [],
                "reason": "",
            },
        })

    ledger_texts = {
        normalize_for_substring(str(claim.get("cited_span") or claim.get("text") or claim.get("claim") or ""))
        for claim in claims
        if str(claim.get("cited_span") or claim.get("text") or claim.get("claim") or "").strip()
    }
    uncited_analysis_texts = {
        normalize_for_substring(str(claim.get("uncited_analysis_span") or ""))
        for claim in claims
        if str(claim.get("uncited_analysis_span") or "").strip()
    }
    covered_texts = ledger_texts | uncited_analysis_texts
    ordered_ledger_texts = sorted((item for item in covered_texts if len(item) >= 3), key=len, reverse=True)
    seen_uncited_texts: set[str] = set()
    uncited_units = []
    for document in candidate.get("documents") or []:
        if not isinstance(document, dict):
            continue
        section_id = str(document.get("section_id") or "")
        for line in str(document.get("content") or "").splitlines():
            if not line.strip() or line.lstrip().startswith(("#", "|", "- ", "* ")):
                continue
            normalized_line = normalize_for_substring(line)
            line_claims = [
                claim
                for claim in claims
                if normalize_for_substring(str(claim.get("text") or claim.get("claim") or ""))
                and normalize_for_substring(str(claim.get("text") or claim.get("claim") or ""))
                in normalized_line
            ]
            adjacent_evidence_ids = sorted({
                str(evidence_id)
                for claim in line_claims
                for evidence_id in claim.get("evidence_ids") or []
                if str(evidence_id)
            })
            for unit in split_atomic_units(line):
                claim_text = re.sub(r"\[\d+\]", "", str(unit or "")).strip()
                normalized = normalize_for_substring(claim_text)
                if not normalized:
                    continue
                residual = normalized
                for ledger_text in ordered_ledger_texts:
                    residual = residual.replace(ledger_text, "")
                if not residual or len(residual) <= 2 or normalized in seen_uncited_texts:
                    continue
                if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", claim_text):
                    continue
                uncited_units.append({
                    "section_id": section_id,
                    "claim": claim_text,
                    "line_context": re.sub(r"\[\d+\]", "", line).strip(),
                    "adjacent_cited_evidence_ids": adjacent_evidence_ids,
                })
                seen_uncited_texts.add(normalized)
    for unit in uncited_units:
        section_id = unit["section_id"]
        claim_text = unit["claim"]
        adjacent_evidence_ids = unit["adjacent_cited_evidence_ids"]
        rows.append({
            "review_unit_origin": "uncited_analysis_unit",
            "claim_index": None,
            "section_id": section_id,
            "claim": claim_text,
            "claim_context": unit["line_context"],
            "claim_type": None,
            "support_scope": None,
            "cited_evidence_ids": [],
            "original_evidence": [],
            "adjacent_cited_evidence_ids": adjacent_evidence_ids,
            "adjacent_original_evidence": [
                {
                    "evidence_id": evidence_id,
                    "atomic_claim": evidence_by_id[evidence_id].get("atomic_claim")
                    or evidence_by_id[evidence_id].get("claim"),
                    "quote": evidence_by_id[evidence_id].get("cleaned_quote")
                    or evidence_by_id[evidence_id].get("raw_quote")
                    or evidence_by_id[evidence_id].get("evidence_text"),
                    "subject": evidence_by_id[evidence_id].get("subject"),
                    "scope": evidence_by_id[evidence_id].get("scope"),
                }
                for evidence_id in adjacent_evidence_ids
                if evidence_id in evidence_by_id
            ],
            "automatic_entailment": {
                "status": "not_citation_evaluated",
                "reason": (
                    "the text carries no citation marker and is therefore recorded as Writer "
                    "analysis rather than evaluated as a citation-backed claim"
                ),
                "mode": "structural_audit",
                "claim_scope": None,
                "confirmed": False,
            },
            "automatic_error_tags": [],
            "human_review": {
                "status": "pending",
                "verdict": None,
                "severity": None,
                "error_types": [],
                "reason": "",
            },
        })

    statuses = [str(row["automatic_entailment"]["status"] or "") for row in rows]
    fidelity_error_count = sum(item in {"unsupported", "label_mismatch", "non_atomic"} for item in statuses)
    cited_statuses = statuses[:len(claims)]
    unavailable_count = sum(item in {"validation_unavailable", "not_reviewed"} for item in cited_statuses)
    supported_count = sum(item in {"entailed", "reasonable_inference"} for item in statuses)
    required_counter_clusters = {
        str(cluster_id)
        for section in inventory.get("sections") or []
        for criterion in section.get("criteria") or []
        for cluster_id in criterion.get("counter_claim_cluster_ids") or []
        if str(cluster_id)
    }
    used_clusters = {
        str(evidence_by_id[evidence_id].get("claim_cluster_id") or "")
        for evidence_id in used_evidence_ids
        if evidence_id in evidence_by_id and evidence_by_id[evidence_id].get("claim_cluster_id")
    }
    claim_section_ids = {
        str(claim.get("section_id") or "") for claim in claims if str(claim.get("section_id") or "")
    }
    available_conflict_ids = {
        str(item.get("id") or "")
        for item in job.evidence_map
        if item.get("id")
        and _is_conflict_evidence(item)
        and (not claim_section_ids or str(item.get("section_id") or "") in claim_section_ids)
    }
    critic = validation.get("critic") if isinstance(validation.get("critic"), dict) else {}
    quality_gate = validation.get("quality_gate") if isinstance(validation.get("quality_gate"), dict) else {}
    report_quality = (
        (quality_gate.get("checks") or {}).get("report_quality")
        if isinstance(quality_gate.get("checks"), dict)
        else None
    )
    if not isinstance(report_quality, dict):
        report_quality = critic
    semantic_live = str(validation.get("mode") or "") == "production_validators_live"
    automatic_status = (
        "inconclusive"
        if not semantic_live or unavailable_count
        else "failed"
        if fidelity_error_count
        else "passed"
    )
    return {
        "schema_version": CLAIM_AUDIT_SCHEMA_VERSION,
        "arm": candidate.get("arm"),
        "review_basis": "final claim versus cited original Evidence atomic claim and Quote",
        "finding_text_used_as_ground_truth": False,
        "citation_id_validity_status": (validation.get("citation") or {}).get("status"),
        "automatic_fidelity_status": automatic_status,
        "human_review_status": "pending",
        "human_verdict_options": [
            "entailed",
            "reasonable_synthesis",
            "partial_support",
            "missing_citation",
            "unsupported_addition",
            "scope_mismatch",
            "subject_mismatch",
            "comparison_axis_mismatch",
            "source_role_mismatch",
            "contradicted_by_citation",
            "conflict_or_limitation_omission",
            "non_atomic",
            "non_claim_narration",
        ],
        "summary": {
            "claim_count": len(claims),
            "uncited_analysis_span_count": len(uncited_analysis_texts),
            "claim_ledger_gap_count": 0,
            "uncited_text_unit_count": len(uncited_units),
            "review_unit_count": len(rows),
            "cited_claim_review_count": len(claims),
            "automatically_supported_count": supported_count,
            "automatic_fidelity_error_count": fidelity_error_count,
            "semantic_validation_unavailable_count": unavailable_count,
            "automatic_support_rate": round(supported_count / len(claims), 3) if claims else None,
            "unsupported_count": statuses.count("unsupported"),
            "label_mismatch_count": statuses.count("label_mismatch"),
            "non_atomic_count": statuses.count("non_atomic"),
            "required_counter_claim_cluster_count": len(required_counter_clusters),
            "used_required_counter_claim_cluster_count": len(required_counter_clusters & used_clusters),
            "counter_material_retention_proxy": (
                round(len(required_counter_clusters & used_clusters) / len(required_counter_clusters), 3)
                if required_counter_clusters else None
            ),
            "available_conflict_or_limitation_evidence_count": len(available_conflict_ids),
            "used_conflict_or_limitation_evidence_count": len(available_conflict_ids & used_evidence_ids),
            "overall_quality_status": report_quality.get("status") if semantic_live else "inconclusive",
            "overall_quality_average_score": report_quality.get("average_score") if semantic_live else None,
            "overall_quality_scores": report_quality.get("scores") or {} if semantic_live else {},
            "raw_validator_quality_status": report_quality.get("status"),
        },
        "rows": rows,
    }


def _quality_first_metrics(run: dict[str, Any]) -> dict[str, Any]:
    validation = run.get("validation") if isinstance(run.get("validation"), dict) else {}
    audit_summary = validation.get("claim_evidence_audit_summary") or {}
    return {
        "automatic_fidelity_status": audit_summary.get("automatic_fidelity_status"),
        "claim_count": audit_summary.get("claim_count"),
        "uncited_text_unit_count": audit_summary.get("uncited_text_unit_count"),
        "review_unit_count": audit_summary.get("review_unit_count"),
        "automatic_fidelity_error_count": audit_summary.get("automatic_fidelity_error_count"),
        "unsupported_count": audit_summary.get("unsupported_count"),
        "label_mismatch_count": audit_summary.get("label_mismatch_count"),
        "semantic_validation_unavailable_count": audit_summary.get("semantic_validation_unavailable_count"),
        "automatic_support_rate": audit_summary.get("automatic_support_rate"),
        "counter_material_retention_proxy": audit_summary.get("counter_material_retention_proxy"),
        "citation_id_validity_status": (validation.get("citation") or {}).get("status"),
        "overall_quality_status": audit_summary.get("overall_quality_status"),
        "overall_quality_average_score": audit_summary.get("overall_quality_average_score"),
        "human_review_status": "pending",
    }


def historical_finding_trace_summary(job: FrozenJob) -> dict[str, Any]:
    events = [
        item for item in job.diagnostics.get("model_trace_events") or []
        if item.get("stage") == "finding_synthesis"
    ]
    parsed = [
        item for item in events
        if item.get("event") == "model_parse" and item.get("status") == "success"
    ]
    generated = sum(
        len((item.get("parsed_result") or {}).get("findings") or [])
        for item in parsed
        if isinstance(item.get("parsed_result"), dict)
    )
    calls = [item for item in events if item.get("event") == "model_call"]
    return {
        "input_evidence_count": len(job.evidence_map),
        "input_claim_cluster_count": len({
            str(item.get("claim_cluster_id") or "") for item in job.evidence_map if item.get("claim_cluster_id")
        }),
        "generated_candidate_finding_count_from_trace": generated,
        "accepted_finding_count": len(job.findings),
        "rejected_finding_count": None,
        "rejection_reasons": [],
        "rejection_details_available": False,
        "model_call_count": len(calls),
        "model_failure_count": sum(not bool(item.get("success")) for item in calls),
        "model_calls": [
            {
                "call_id": item.get("call_id"),
                "question_id": item.get("question_id"),
                "section_id": item.get("section_id"),
                "status": item.get("status"),
                "actual_provider": item.get("actual_provider"),
                "actual_model": item.get("actual_model"),
                "parameters": item.get("parameters"),
                "duration_ms": item.get("duration_ms"),
            }
            for item in calls
        ],
        "limitation": (
            "Historical finding_synthesis checkpoints persist only admitted findings and the Blueprint; "
            "per-candidate Finding Admission rejections and reasons were not frozen."
        ),
    }


def validate_frozen_inventory(inventory: dict[str, Any], job: FrozenJob) -> dict[str, Any]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise EvaluationContractError("reference inventory schema mismatch")
    if str(inventory.get("job_id") or "") != job.job_id:
        raise EvaluationContractError("reference inventory job mismatch")
    if inventory.get("source_evidence_artifact_sha256") != job.artifacts["evidence_map"].sha256:
        raise EvaluationContractError("reference inventory evidence hash mismatch")
    if inventory.get("status") != "frozen" or not inventory.get("frozen_sha256"):
        raise EvaluationContractError("reference inventory is not frozen")
    evidence_ids = {str(item.get("id") or "") for item in job.evidence_map}
    catalog_ids = [str(item.get("evidence_id") or "") for item in inventory.get("evidence_catalog") or []]
    if len(catalog_ids) != len(set(catalog_ids)):
        raise EvaluationContractError("reference inventory evidence catalog contains duplicate ids")
    if not set(catalog_ids).issubset(evidence_ids):
        raise EvaluationContractError("reference inventory evidence catalog references unknown evidence")
    for section in inventory.get("sections") or []:
        for criterion in section.get("criteria") or []:
            answerability = str(criterion.get("answerability") or "")
            if answerability not in {"supported", "partial", "unsupported", "conflicted"}:
                raise EvaluationContractError("reference inventory has invalid answerability")
            candidate_clusters = {str(item) for item in criterion.get("candidate_claim_cluster_ids") or []}
            classified_clusters = {
                str(item)
                for key in ("required_claim_cluster_ids", "optional_claim_cluster_ids", "counter_claim_cluster_ids")
                for item in criterion.get(key) or []
            }
            if not classified_clusters.issubset(candidate_clusters):
                raise EvaluationContractError("reference inventory classifies an unknown claim cluster")
            if not {str(item) for item in criterion.get("evidence_ids") or []}.issubset(evidence_ids):
                raise EvaluationContractError("reference inventory references unknown evidence")
            if criterion.get("human_review_status") != "reviewed":
                raise EvaluationContractError("reference inventory contains an unreviewed criterion")
            if not str(criterion.get("review_note") or "").strip():
                raise EvaluationContractError("reference inventory reviewed criterion is missing review_note")
            if (
                answerability in {"supported", "partial", "conflicted"}
                and str(criterion.get("level") or "") in {"hard", "soft"}
                and not criterion.get("required_claim_cluster_ids")
            ):
                raise EvaluationContractError(
                    "reference inventory answerable hard/soft criterion is missing required claim clusters"
                )
            if answerability == "conflicted" and not criterion.get("counter_claim_cluster_ids"):
                raise EvaluationContractError("reference inventory conflicted criterion is missing counter claim clusters")
    unhashed = {key: value for key, value in inventory.items() if key != "frozen_sha256"}
    actual = stable_hash(unhashed)
    if actual != inventory["frozen_sha256"]:
        raise EvaluationContractError("reference inventory frozen hash mismatch")
    return {"status": "passed", "frozen_sha256": actual}


def freeze_inventory(inventory: dict[str, Any], *, reviewer: str) -> dict[str, Any]:
    result = copy.deepcopy(inventory)
    pending = [
        f"{section.get('section_id')}:{criterion.get('criterion_id')}"
        for section in result.get("sections") or []
        for criterion in section.get("criteria") or []
        if criterion.get("human_review_status") != "reviewed"
    ]
    if pending:
        raise EvaluationContractError(f"inventory has unreviewed criteria: {pending[:12]}")
    result["status"] = "frozen"
    result["frozen_at"] = utc_now()
    result["reviewer"] = str(reviewer or "").strip()
    if not result["reviewer"]:
        raise EvaluationContractError("inventory reviewer is required")
    result.pop("frozen_sha256", None)
    result["frozen_sha256"] = stable_hash(result)
    return result


def direct_writer_rules(
    target_length: int,
    *,
    context_profile: str = DIRECT_CONTEXT_PROFILE_BASELINE,
) -> list[str]:
    if context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED:
        return [
            "<hard_constraints>",
            "1. 只用 payload 里的 evidence_packet，不使用外部知识，不补写 Quote 没有表达的事实、因果、目的、性能、排序、适用场景或评价。",
            "2. Quote 是事实权威；atomic_claim 只用于定位材料。两者范围或强度不一致时，只能采用 Quote 能直接支持的较窄表述。",
            "3. 每个陈述性子句都带自己的 [[ev_XXX]] 引用；引用必须支持该子句的全部主体、关系、条件、强度和范围，找不到支持就删除该子句。",
            "3a. [[ev_XXX]] 必须紧跟在它支持的最小事实子句末尾；不得把标记放在下一句开头，也不得让上一句的标记看起来支持下一句。",
            "4. 只能在同一个 synthesis_boundary_id 内综合材料。不同 boundary 的内容默认分别陈述；只有某条 Quote 明确给出二者关系时才能跨 boundary 建立因果、机制、比较或总括，并同时引用该关系证据。",
            "5. 不得把多个分别成立的局部事实自动总结成统一的核心目的、普遍机制、效果保证、最佳实践或推荐结论。",
            "6. 保留 Evidence 的主体、比较维度、适用范围、来源角色和不确定性；来源方或社区文章的判断必须明确归因，不能写成独立验证事实。",
            "6a. 被引用事实不得删除 Quote 中的对象、领域、数据集、条件、时间或强度限定；需要推广时，先在原范围内写完事实并紧跟引用，再另写无引用的限定性分析。",
            "6b. 例如 Quote 只讨论金融报告时，可写“在金融报告场景中……[[ev_XXX]]。这一思路可能适用于其他复杂文档，但仍需验证”；不得写成“适用于所有复杂文档[[ev_XXX]]”。",
            "7. conflict_or_limitation=true 的材料必须与对应支持材料共同保留边界；存在冲突时并列呈现，不强行裁决。",
            "8. Evidence 不足时允许缩短、明确未知或不回答；不得为目标长度补充常识、背景、案例或推断。",
            "9. evidence_binding_basis 为 declared_criterion_candidates 或 question_scope_candidates 时，这些材料只是待核对候选，不证明 Criterion 已有答案；只能写 Quote 自身明确支持的内容。",
            "10. criterion level 只决定覆盖优先级：先 hard，再 soft，最后 optional；内部标签不得出现在正文。",
            "11. 只能逐字复制 allowed_evidence_ids 中的 ID，并写成独立标记 [[ev_001]][[ev_002]]；不要输出 Finding ID、原始 Quote 全文、Sources 章节或审计旁白。",
            "</hard_constraints>",
            "",
            "<writing_style>",
            "12. 用中文写连贯、克制的分析正文，产品名和代码标识保留原文；先直接回答本节能够由证据支持的部分。",
            "13. 可以用自然过渡组织同一 synthesis boundary 内的材料，但过渡词不能承担证据没有提供的逻辑关系。",
            f"14. 目标长度约 {target_length} 字，但证据充分性优先，宁可更短也不扩写。",
            "15. research questions 是覆盖地图，不是固定标题模板；严格遵守当前章节和 report outline 的边界。",
            "16. 只有材料明确支持多维比较或多步流程时才使用 Markdown 表格；表格中的每个事实单元也必须独立引用。",
            "</writing_style>",
        ]
    return [
        "<hard_constraints>",
        "1. 只用 payload 里的 evidence_packet，不使用外部知识，不补写原文没有的事实、因果、目的、比较对象或评价。",
        "2. 每个陈述性子句都带自己的 [[ev_XXX]] 引用；引用必须真实支持该子句，找不到支持就删掉该子句。",
        "2a. [[ev_XXX]] 必须紧跟在它支持的最小事实子句末尾；不得把标记放在下一句开头，也不得让上一句的标记看起来支持下一句。",
        "3. 可以综合多条 Evidence 形成受约束的分析，但综合结论必须同时列出所有桥接 Evidence，且不能把相邻主体或相邻比较维度互相替代。",
        "4. 保留 Evidence 的适用范围、来源角色和不确定性；来源方自述必须归因，不能改写成独立验证结论。",
        "4a. 被引用事实不得删除 Quote 中的对象、领域、数据集、条件、时间或强度限定；跨范围推广只能在引用标记之后另写为未引用分析。",
        "5. conflict_or_limitation=true 的材料不得被支持性材料覆盖掉；存在冲突时并列呈现证据边界，不强行裁决。",
        "6. Evidence 不足时允许缩短、明确未知或不回答；不得为了目标长度添加常识、案例、因果、建议或评价。",
        "7. criterion level 只决定覆盖优先级：先 hard，再 soft，最后 optional；内部标签不得出现在正文。",
        "8. 只能逐字复制 allowed_evidence_ids 中的 ID，并写成独立标记 [[ev_001]][[ev_002]]；不得猜测连续编号或使用列表外 ID。不要输出 Finding ID、原始 quote 全文、Sources 章节或审计旁白。",
        "</hard_constraints>",
        "",
        "<writing_style>",
        "9. 用中文写连贯分析散文，产品名和代码标识保留原文；开头 1-2 句先给本节有证据支持的核心回答。",
        "10. 用自然过渡把相关证据组织成段落，不要逐条转写 Evidence；只有确实适合多维对比或多步序列时才使用 Markdown 表格。",
        f"11. 目标长度约 {target_length} 字，但证据充分性优先，宁可更短也不填充无依据内容。",
        "12. research questions 是覆盖地图，不是固定标题模板；严格遵守当前章节职责和 report outline 的边界。",
        "13. 当材料只表示本次检索未找到某项信息时，只能写成检索范围说明，不能写成现实中不存在。",
        "</writing_style>",
    ]


def build_direct_section_payload(
    *,
    section: dict[str, Any],
    blueprint: dict[str, Any],
    evidence_packet: dict[str, Any],
    target_length: int,
) -> dict[str, Any]:
    context_profile = str(
        evidence_packet.get("context_profile") or DIRECT_CONTEXT_PROFILE_BASELINE
    )
    prompt_version = (
        CLUSTER_GUARDED_PROMPT_VERSION
        if context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
        else PROMPT_VERSION
    )
    return {
        "task": "基于结构化 Evidence Packet 撰写一节中文分析报告。只返回 JSON。",
        "prompt_version": prompt_version,
        "direct_context_profile": context_profile,
        "section_role": (
            "overview_synthesis"
            if str(section.get("section_id") or "") == "sec_00" or "总览" in str(section.get("title") or "")
            else "section_specific_analysis"
        ),
        "section_brief": {key: value for key, value in section.items() if key != "dossier"},
        "report_outline": [
            {"section_id": item.get("section_id"), "title": item.get("title"), "purpose": item.get("purpose")}
            for item in blueprint.get("sections") or []
        ],
        "research_mode": blueprint.get("research_mode"),
        "target_length_chars": target_length,
        "allowed_evidence_ids": sorted(_packet_evidence_ids(evidence_packet)),
        "evidence_packet": evidence_packet,
        "rules": direct_writer_rules(target_length, context_profile=context_profile),
        "schema": {"content": "不含章节标题的实质性 Markdown 正文"},
    }


class RecordingJsonModel:
    """JsonModel adapter that records the exact payload/result for experiment audit."""

    def __init__(self, invoke: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.invoke = invoke
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        payload: dict[str, Any],
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "started_at": utc_now(),
            "prompt": sanitized_json_safe(payload),
            "prompt_hash": stable_hash(payload),
            "estimated_input_tokens": estimate_tokens({"system_prompt": WRITER_SYSTEM_PROMPT, "payload": payload}),
            "timeout_override_seconds": timeout_override,
        }
        try:
            if timeout_override is None:
                result = self.invoke(payload)
            else:
                result = self.invoke(payload, timeout_override=timeout_override)
            record["response"] = sanitized_json_safe(result)
            record["response_hash"] = stable_hash(result)
            record["status"] = "failed" if result.get("_failure_reason") else "success"
            record["failure_reason"] = result.get("_failure_reason")
            return result
        except Exception as exc:
            record["status"] = "exception"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["duration_ms"] = int((time.perf_counter() - started) * 1000)
            self.calls.append(record)


class DeadlineJsonModel:
    """Apply one shared wall-clock budget across all calls made by a validator."""

    def __init__(self, invoke: Callable[..., dict[str, Any]], budget_seconds: float) -> None:
        self.invoke = invoke
        self.budget_seconds = max(1.0, float(budget_seconds))
        self.deadline = time.monotonic() + self.budget_seconds

    def __call__(
        self,
        payload: dict[str, Any],
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 1.0:
            return {
                "_failure_reason": "validation_budget_exhausted",
                "_experiment_model_metadata": {
                    "validation_budget_seconds": self.budget_seconds,
                    "remaining_seconds": max(0.0, remaining),
                },
            }
        effective_timeout = remaining
        if timeout_override is not None:
            effective_timeout = min(effective_timeout, max(1.0, float(timeout_override)))
        return self.invoke(payload, timeout_override=effective_timeout)


_DIRECT_INTERNAL_MARKER_RE = re.compile(
    r"\[\[([^\]]+)\]\]|\[\s*((?:ev_\d{3,4})(?:\s*[,;、]\s*ev_\d{3,4})*)\s*\]"
)


def _clean_direct_span(value: str) -> str:
    rendered = re.sub(r"\[\d+\]", "", str(value or ""))
    rendered = re.sub(r"\b(?:ev|finding)_\d{3,4}\b", "", rendered)
    rendered = re.sub(r"\[\s*\]", "", rendered)
    return rendered.strip().lstrip("，,；;| ").rstrip("| ")


def _has_direct_span_text(value: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", str(value or "")))


def _direct_marker_ids(match: re.Match[str], evidence_by_id: dict[str, dict[str, Any]]) -> list[str]:
    raw = match.group(1) or match.group(2) or ""
    return list(dict.fromkeys(
        token
        for token in re.findall(r"\bev_\d{3,4}\b", raw)
        if token in evidence_by_id
    ))


def _render_direct_citations_with_spans(
    content: str,
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """Render E+ citations while keeping the cited span narrower than the prose sentence."""
    rendered_content, _ = _render_internal_citations(content, evidence_by_id, citation_map)
    normalized = _normalize_internal_citation_markers(str(content or ""))
    ledger: list[dict[str, Any]] = []
    previous_claim_text = ""

    for line in normalized.splitlines():
        is_table_row = line.lstrip().startswith("|")
        units = (
            line.split("|")
            if is_table_row
            else split_atomic_units(line)
            if line.strip() and not line.lstrip().startswith(("#", "- ", "* "))
            else [line]
        )
        for unit in units:
            full_rendered, _ = _render_internal_citations(unit, evidence_by_id, citation_map)
            full_atomic_unit = _clean_direct_span(full_rendered)
            cursor = 0
            last_row: dict[str, Any] | None = None
            for match in _DIRECT_INTERNAL_MARKER_RE.finditer(unit):
                evidence_ids = _direct_marker_ids(match, evidence_by_id)
                if not evidence_ids:
                    continue
                raw_span = unit[cursor:match.start()]
                cited_span = _clean_direct_span(raw_span)
                if _has_direct_span_text(cited_span):
                    raw_with_markers = unit[cursor:match.end()]
                    rendered_span, _ = _render_internal_citations(
                        raw_with_markers,
                        evidence_by_id,
                        citation_map,
                    )
                    last_row = {
                        "text": cited_span,
                        "cited_span": cited_span,
                        "uncited_analysis_span": "",
                        "full_atomic_unit": full_atomic_unit,
                        "claim_context": previous_claim_text[:1000] if not is_table_row else "",
                        "evidence_ids": evidence_ids,
                        "rendered_text": rendered_span.strip(),
                        "_raw_with_markers": raw_with_markers,
                    }
                    ledger.append(last_row)
                elif last_row is not None:
                    last_row["evidence_ids"] = list(dict.fromkeys([
                        *last_row.get("evidence_ids", []),
                        *evidence_ids,
                    ]))
                    last_row["_raw_with_markers"] += match.group(0)
                    rendered_span, _ = _render_internal_citations(
                        last_row["_raw_with_markers"],
                        evidence_by_id,
                        citation_map,
                    )
                    last_row["rendered_text"] = rendered_span.strip()
                cursor = match.end()

            trailing = _clean_direct_span(unit[cursor:])
            if last_row is not None and _has_direct_span_text(trailing):
                last_row["uncited_analysis_span"] = trailing
            if not is_table_row and _has_direct_span_text(full_atomic_unit):
                previous_claim_text = full_atomic_unit

    for row in ledger:
        row.pop("_raw_with_markers", None)
    return rendered_content, ledger


def _scope_relation_diagnostic(judgment: dict[str, Any]) -> str:
    status = str(judgment.get("status") or "")
    reason = str(judgment.get("reason") or "").casefold()
    scope_markers = (
        "scope", "范围", "泛化", "扩大", "broader", "generaliz", "universal",
        "single", "narrow", "特定", "单一",
    )
    if status == "label_mismatch" or (
        status == "unsupported" and any(marker in reason for marker in scope_markers)
    ):
        return "broader_scope"
    if status in {"entailed", "reasonable_inference"}:
        return "same_or_narrower_scope"
    return "unclear"


def dry_run_json_model(payload: dict[str, Any]) -> dict[str, Any]:
    serialized = canonical_json(payload)
    evidence_ids = re.findall(r'"id":"(ev_[^"]+)"', serialized)
    claim_ids = re.findall(r'"claim_id":"([^"]+)"', serialized)
    marker_id = claim_ids[0] if claim_ids else evidence_ids[0] if evidence_ids else ""
    marker = f" [[{marker_id}]]" if marker_id else ""
    task = str(payload.get("task") or "")
    if "executive summary" in task or "执行摘要" in task:
        return {
            "report_title": "受控评测预检",
            "executive_summary": f"dry-run 不调用模型，仅验证输入合同。{marker}",
            "conclusion": f"dry-run 不产生质量结论。{marker}",
        }
    return {"content": f"dry-run 不调用模型，仅验证输入合同。{marker}"}


def _direct_section_document(
    *,
    section: dict[str, Any],
    blueprint: dict[str, Any],
    evidence_map: list[dict[str, Any]],
    packet: dict[str, Any],
    model: RecordingJsonModel,
) -> dict[str, Any]:
    selected_ids = _packet_evidence_ids(packet)
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map if str(item.get("id") or "") in selected_ids}
    target_length = _target_length_chars(section, [{} for _ in selected_ids])
    payload = build_direct_section_payload(
        section=section,
        blueprint=blueprint,
        evidence_packet=packet,
        target_length=target_length,
    )
    content, meta = _call_writer_model(model, payload, evidence_by_id, allowed_finding_ids=set())
    max_retries = max(0, int(os.getenv("RESEARCH_WRITER_MAX_RETRIES", "3")))
    retries_used = 0
    active_payload = payload
    marker_repair_attempted = False
    while not content and retries_used < max_retries:
        if meta.get("failure_reason") == "evidence_validation_failed":
            if marker_repair_attempted:
                break
            marker_repair_attempted = True
            active_payload = {
                **payload,
                "task": "修复一节引用标记无效的中文分析正文。只返回 JSON。",
                "invalid_draft": str(meta.get("invalid_content") or ""),
                "invalid_markers": list(meta.get("invalid_markers") or []),
                "rules": [
                    *payload["rules"],
                    "上一稿使用了 allowed_evidence_ids 中不存在的标记。必须逐字复制 allowlist 内的完整 ID；不要按编号猜测相邻 Evidence。无法找到真实支持时删除对应子句。",
                ],
            }
        content, retry_meta = _call_writer_model(
            model, active_payload, evidence_by_id, allowed_finding_ids=set()
        )
        meta["attempts"] = int(meta.get("attempts") or 1) + 1
        meta["failure_reason"] = None if content else retry_meta.get("failure_reason")
        meta["invalid_content"] = retry_meta.get("invalid_content")
        meta["invalid_markers"] = retry_meta.get("invalid_markers") or []
        retries_used += 1
    if content and _section_is_underdeveloped(content, section, [{} for _ in selected_ids]):
        expanded_payload = {
            **payload,
            "task": "基于同一 Evidence Packet 扩展一节过短的中文分析正文。只返回 JSON。",
            "existing_draft": content,
            "rules": [
                *payload["rules"],
                "保留已有的受支持内容，只从 supplied Evidence 的差异、限制、冲突和不确定性增加分析；不得增加外部事实。",
            ],
        }
        expanded, _ = _call_writer_model(
            model, expanded_payload, evidence_by_id, allowed_finding_ids=set()
        )
        meta["attempts"] = int(meta.get("attempts") or 1) + 1
        if expanded and len(expanded) > len(content):
            content = expanded
    failure_reason = str(meta.get("failure_reason") or "")
    citation_map = _citation_map(evidence_map)
    rendered, ledger = _render_direct_citations_with_spans(content, evidence_by_id, citation_map)
    ledger = _annotate_claim_ledger(ledger, [], evidence_by_id, str(section.get("section_id") or ""))
    used_ids = list(dict.fromkeys(
        evidence_id for row in ledger for evidence_id in row.get("evidence_ids") or []
    ))
    evidence_table = [_evidence_table_row(evidence_by_id[item]) for item in used_ids if item in evidence_by_id]
    return {
        "title": section.get("title"),
        "section_id": section.get("section_id"),
        "content": rendered.strip(),
        "evidence_refs": _evidence_refs([evidence_by_id[item] for item in used_ids if item in evidence_by_id]),
        "evidence_table": evidence_table,
        "claim_ledger": ledger,
        "section_brief": section,
        "paragraph_validation": {"status": "passed", "failed_group_count": 0, "groups": []},
        "evidence_metrics": _evidence_metrics(evidence_table, ledger),
        "synthesis_diagnostics": {
            "mode": "direct_evidence_writer" if content else "model_failed",
            "fallback_reason": failure_reason or None,
            "model_attempts": int(meta.get("attempts") or 1),
            "input_evidence_count": len(selected_ids),
            "used_evidence_count": len(used_ids),
            "evidence_utilization_ratio": round(len(used_ids) / len(selected_ids), 3) if selected_ids else 1.0,
            "reportability": section.get("reportability") or {},
        },
    }


def _packet_evidence_ids(packet: dict[str, Any]) -> set[str]:
    return set(re.findall(r'"id":"(ev_[^"]+)"', canonical_json(packet)))


def _minimal_report_document(document: dict[str, Any], blueprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(document.get("title") or ""),
        "content": str(document.get("content") or ""),
        "claim_ledger": list(document.get("claim_ledger") or []),
        "evidence_refs": list(document.get("evidence_refs") or []),
        "delivery_status": "complete_report" if document.get("content") else "partial_report",
        "completeness": {
            "status": "complete_report" if document.get("content") else "partial_report",
            "required_section_count": 1,
            "ready_section_count": 1 if document.get("content") else 0,
            "core_empty_section_ids": [] if document.get("content") else [document.get("section_id")],
        },
        "research_mode": blueprint.get("research_mode"),
        "sanitized_evidence_id_count": 0,
    }


def _final_report_claim_ledger(
    documents: list[dict[str, Any]], editorial: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return only claims whose text is rendered into the delivered report."""
    return [
        *[
            row
            for document in documents
            if not _excluded_from_assembled_body(str(document.get("title") or ""))
            for row in document.get("claim_ledger") or []
        ],
        *(editorial.get("claim_ledger") or []),
    ]


def run_arm_finding(
    *,
    job: FrozenJob,
    stage: str,
    section_ids: list[str],
    model: RecordingJsonModel,
) -> dict[str, Any]:
    evidence_by_id = {str(item.get("id") or ""): item for item in job.evidence_map}
    finding_by_id = {str(item.get("finding_id") or ""): item for item in job.findings}
    citation_map = _citation_map(job.evidence_map)
    sections = [
        section for section in job.report_blueprint.get("sections") or []
        if not section_ids or str(section.get("section_id") or "") in section_ids
    ]
    documents = [
        _write_section(section, job.report_blueprint, finding_by_id, evidence_by_id, citation_map, model)
        for section in sections
        if not _is_reference_title(str(section.get("title") or ""))
    ]
    if stage == "section":
        if len(documents) != 1:
            raise EvaluationContractError("section stage requires exactly one non-reference section")
        report_document = _minimal_report_document(documents[0], job.report_blueprint)
        editorial = {}
    else:
        editorial = _edit_report(
            str(job.normalized_request.get("user_request") or ""),
            job.report_blueprint,
            job.findings,
            evidence_by_id,
            citation_map,
            model,
        )
        completeness = _delivery_completeness(job.report_blueprint.get("completeness") or {}, documents)
        content, sanitized_count = _assemble_report(
            str(job.normalized_request.get("user_request") or ""),
            documents,
            editorial,
            citation_map,
            job.evidence_map,
            completeness=completeness,
            research_mode=str(job.report_blueprint.get("research_mode") or "domain_practice_review"),
            reference_title=_reference_title_from_blueprint(job.report_blueprint),
        )
        report_document = {
            "title": editorial.get("report_title") or str(job.normalized_request.get("user_request") or ""),
            "content": content,
            "claim_ledger": _final_report_claim_ledger(documents, editorial),
            "delivery_status": completeness.get("status"),
            "completeness": completeness,
            "research_mode": job.report_blueprint.get("research_mode"),
            "sanitized_evidence_id_count": sanitized_count,
        }
    visible_evidence = sorted({
        str(evidence_id)
        for finding in job.findings
        for evidence_id in [*(finding.get("evidence_ids") or []), *(finding.get("counter_evidence_ids") or [])]
    })
    return {
        "arm": ARM_FINDING,
        "documents": documents,
        "editorial": editorial,
        "report_document": report_document,
        "context_diagnostics": {
            "input_finding_count": len(job.findings),
            "input_evidence_count": len(job.evidence_map),
            "writer_visible_evidence_ids": visible_evidence,
            "writer_hidden_accepted_evidence_ids": sorted(
                {str(item.get("id") or "") for item in job.evidence_map} - set(visible_evidence)
            ),
            "historical_finding_generation": historical_finding_trace_summary(job),
        },
        "model_calls": model.calls,
    }


def _build_editorial_claim_bank(
    documents: list[dict[str, Any]],
    section_entailment: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bank: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    section_counts: dict[str, int] = {}
    judgments = {
        int(item.get("claim_index")): item
        for item in (section_entailment or {}).get("judgments") or []
        if isinstance(item, dict) and str(item.get("claim_index", "")).lstrip("-").isdigit()
    }
    semantic_filter_active = section_entailment is not None
    visible_claim_index = 0
    for document in documents:
        if (
            not isinstance(document, dict)
            or _excluded_from_assembled_body(str(document.get("title") or ""))
        ):
            continue
        section_id = str(document.get("section_id") or "section")
        for row in document.get("claim_ledger") or []:
            if not isinstance(row, dict):
                continue
            evidence_ids = [str(item) for item in row.get("evidence_ids") or [] if str(item)]
            cited_span = str(row.get("cited_span") or row.get("text") or "").strip()
            judgment = judgments.get(visible_claim_index) if semantic_filter_active else None
            visible_claim_index += 1
            if not evidence_ids or not cited_span or row.get("atomic") is False:
                continue
            validation_status = str(
                (judgment or {}).get("status")
                or ("validation_unavailable" if semantic_filter_active else "citation_bound")
            )
            if semantic_filter_active and validation_status not in {"entailed", "reasonable_inference"}:
                excluded.append({
                    "section_id": section_id,
                    "cited_span": cited_span,
                    "evidence_ids": evidence_ids,
                    "validation_status": validation_status,
                    "validation_reason": (judgment or {}).get("reason"),
                })
                continue
            section_counts[section_id] = section_counts.get(section_id, 0) + 1
            claim_id = f"{section_id}_claim_{section_counts[section_id]:03d}"
            bank.append({
                "claim_id": claim_id,
                "section_id": section_id,
                "section_title": document.get("title"),
                "cited_span": cited_span,
                "evidence_ids": evidence_ids,
                "claim_type": row.get("claim_type"),
                "support_scope": row.get("support_scope"),
                "validation_status": validation_status,
                "validation_reason": (judgment or {}).get("reason"),
                "scope_contract": (
                    "摘要复用时不得扩大本 claim 的对象、领域、条件、时间或强度；"
                    "需要推广时必须放在 claim marker 之后作为未引用分析。"
                ),
            })
    return bank, excluded


def _mark_prevalidation_rejections(
    documents: list[dict[str, Any]],
    section_entailment: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate rejected cited spans for readers without rewriting section prose."""
    output = copy.deepcopy(documents)
    judgments = {
        int(item.get("claim_index")): item
        for item in (section_entailment or {}).get("judgments") or []
        if isinstance(item, dict) and str(item.get("claim_index", "")).lstrip("-").isdigit()
    }
    marker = "[待核实]"
    visible_claim_index = 0
    rows: list[dict[str, Any]] = []
    for document in output:
        if (
            not isinstance(document, dict)
            or _excluded_from_assembled_body(str(document.get("title") or ""))
        ):
            continue
        content = str(document.get("content") or "")
        for claim in document.get("claim_ledger") or []:
            if not isinstance(claim, dict):
                continue
            judgment = judgments.get(visible_claim_index) or {}
            claim_index = visible_claim_index
            visible_claim_index += 1
            status = str(judgment.get("status") or "")
            if status not in {"unsupported", "label_mismatch", "non_atomic"}:
                continue
            rendered_text = str(claim.get("rendered_text") or "").strip()
            placement = "unresolved"
            if rendered_text and f"{rendered_text} {marker}" in content:
                placement = "already_marked"
            elif rendered_text and rendered_text in content:
                content = content.replace(rendered_text, f"{rendered_text} {marker}", 1)
                placement = "inline_after_citation"
            else:
                cited_span = str(claim.get("cited_span") or claim.get("text") or "").strip()
                content = (
                    content.rstrip()
                    + f"\n\n{marker} 未验证声明：{cited_span[:200]}"
                ).strip()
                placement = "section_end_fallback"
            claim["prevalidation_status"] = status
            claim["reader_warning"] = marker
            rows.append({
                "claim_index": claim_index,
                "section_id": str(document.get("section_id") or ""),
                "cited_span": claim.get("cited_span") or claim.get("text"),
                "evidence_ids": list(claim.get("evidence_ids") or []),
                "status": status,
                "reason": judgment.get("reason"),
                "marker": marker,
                "placement": placement,
            })
        document["content"] = content
    return output, {
        "mode": "prevalidation_reader_warning",
        "marker": marker,
        "marked_claim_count": len(rows),
        "unresolved_marker_count": sum(
            row["placement"] == "section_end_fallback" for row in rows
        ),
        "rows": rows,
    }


def _expand_editorial_claim_markers(
    content: str,
    claim_by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    invalid: list[str] = []
    original = _normalize_internal_citation_markers(str(content or ""))
    invalid.extend(re.findall(r"\b(?:ev|finding)_\d{3,4}\b", original))

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        tokens = [item for item in re.split(r"\s*[,，;；、]\s*", raw) if item]
        if not tokens or any(token not in claim_by_id for token in tokens):
            invalid.extend(token for token in tokens or [raw] if token not in claim_by_id)
            return ""
        evidence_ids = list(dict.fromkeys(
            evidence_id
            for token in tokens
            for evidence_id in claim_by_id[token].get("evidence_ids") or []
        ))
        return "".join(f"[[{evidence_id}]]" for evidence_id in evidence_ids)

    expanded = re.sub(r"\[\[([^\]]+)\]\]", replace, original)
    invalid.extend(re.findall(r"\bclaim_[A-Za-z0-9_-]+\b", expanded))
    return expanded, list(dict.fromkeys(invalid))


def _direct_editorial(
    *,
    user_request: str,
    blueprint: dict[str, Any],
    documents: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    model: RecordingJsonModel,
    context_profile: str,
    section_entailment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    claim_bank, excluded_claims = _build_editorial_claim_bank(documents, section_entailment)
    claim_by_id = {str(item["claim_id"]): item for item in claim_bank}
    selected_ids = {
        str(evidence_id)
        for claim in claim_bank
        for evidence_id in claim.get("evidence_ids") or []
    }
    evidence_by_id = {
        str(item.get("id") or ""): item for item in evidence_map if str(item.get("id") or "") in selected_ids
    }
    payload = {
        "task": "只基于正文中已有的 citation-bound section claims 写执行摘要、综合结论和短标题。只返回 JSON。",
        "prompt_version": (
            CLUSTER_GUARDED_PROMPT_VERSION
            if context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
            else PROMPT_VERSION
        ),
        "direct_context_profile": context_profile,
        "user_request": user_request,
        "research_mode": blueprint.get("research_mode"),
        "evidence_landscape": blueprint.get("evidence_landscape") or {},
        "section_claim_bank": claim_bank,
        "rules": [
            "<hard_constraints>",
            "1. 只使用 section_claim_bank，不读取原始 Evidence Packet，不读取或假设 Finding。",
            "2. 每个事实性子句都用本 payload 中存在的 [[sec_XX_claim_XXX]] 标记；不得直接输出 [[ev_XXX]] 或 Finding ID。",
            "2a. claim marker 必须紧跟它复用的最小事实子句末尾；多个事实合成一句时必须分别列出对应 claim marker，缺一不可。",
            "3. 只能压缩或重述 claim 的 cited_span，不得新增其中没有的主体、关系、因果、效果、比较、推荐或评价。",
            "3a. 必须保留 claim 的对象、领域、条件、时间、强度和 support_scope；跨范围推广只能放在 claim marker 之后作为未引用分析。",
            "4. 冲突和限制 claim 必须与对应支持性 claim 一同保留；材料不足时缩短或明确未知。",
            *(
                [
                    "4a. citation-bound 只表示 claim 已有合法 Evidence 绑定，不授权增强其完整语义；最终仍由共享 entailment 验证。",
                    "4b. 不得把多个分别成立的 claim 自动提升成统一目的、效果保证、最佳实践或普遍推荐。",
                ]
                if context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
                else []
            ),
            "</hard_constraints>",
            "<writing_style>",
            "5. 中文连贯分析，不逐条转写证据；执行摘要概括核心回答和边界，结论综合而不重复首句。",
            "6. report_title 为 8-24 个中文字符的读者标题，不含引用、Markdown、完整句或末尾“报告”。",
            "7. 不添加 Sources 章节，不输出 Evidence ID、Finding ID、quote 全文或审计说明。",
            "</writing_style>",
        ],
        "schema": {"report_title": "", "executive_summary": "", "conclusion": ""},
    }
    result = model(payload)
    missing = [field for field in ("executive_summary", "conclusion") if not str(result.get(field) or "").strip()]
    if missing:
        completion_payload = {
            **payload,
            "task": "只补齐研究报告缺失的执行摘要或综合结论字段。只返回 JSON。",
            "existing_editorial": {
                "executive_summary": result.get("executive_summary") or "",
                "conclusion": result.get("conclusion") or "",
            },
            "missing_fields": missing,
            "schema": {field: "" for field in missing},
        }
        completion = model(completion_payload)
        for field in missing:
            if completion.get(field):
                result[field] = completion[field]
    citation_map = _citation_map(evidence_map)
    output = {}
    ledgers = []
    invalid_claim_markers: dict[str, list[str]] = {}
    for field in ("executive_summary", "conclusion"):
        content = str(result.get(field) or "")
        content, invalid = _expand_editorial_claim_markers(content, claim_by_id)
        if invalid:
            invalid_claim_markers[field] = invalid
            content = ""
        rendered, ledger = _render_direct_citations_with_spans(content, evidence_by_id, citation_map)
        output[field] = rendered
        ledgers.extend(ledger)
    output["report_title"] = str(result.get("report_title") or "").strip()[:80]
    output["claim_ledger"] = _annotate_claim_ledger(ledgers, [], evidence_by_id, "editorial")
    output["diagnostics"] = {
        "mode": "direct_evidence_editorial",
        "input_contract": "citation_bound_section_claim_bank",
        "input_claim_count": len(claim_bank),
        "excluded_section_claim_count": len(excluded_claims),
        "excluded_section_claims": excluded_claims,
        "section_claim_validation_mode": (
            "semantic_entailment" if section_entailment is not None else "citation_bound_preflight"
        ),
        "input_evidence_count": len(selected_ids),
        "invalid_claim_markers": invalid_claim_markers,
        "claim_bank_sha256": stable_hash(claim_bank),
        "upstream_failure_reason": result.get("_failure_reason"),
    }
    return output


def run_arm_direct(
    *,
    job: FrozenJob,
    stage: str,
    section_ids: list[str],
    model: RecordingJsonModel,
    context_manager: DirectEvidenceContextManager,
    blueprint_profile: str = DIRECT_BLUEPRINT_PROFILE_HISTORICAL_PROJECTION,
    editorial_claim_validator: Callable[
        [list[dict[str, Any]], dict[str, Any]], dict[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    direct_inputs = job.direct_inputs(blueprint_profile=blueprint_profile)
    direct_blueprint = _expect_dict(direct_inputs.get("report_blueprint"), "direct report_blueprint")
    sections = [
        section for section in direct_blueprint.get("sections") or []
        if (not section_ids or str(section.get("section_id") or "") in section_ids)
        and not _is_reference_title(str(section.get("title") or ""))
    ]
    documents = []
    packets = []
    context_diagnostics = []
    for section in sections:
        overview = str(section.get("section_id") or "") == "sec_00" or "总览" in str(section.get("title") or "")
        packet, diagnostics = context_manager.build(direct_inputs, section, overview=overview)
        packets.append(packet)
        context_diagnostics.append({"section_id": section.get("section_id"), **diagnostics})
        documents.append(_direct_section_document(
            section=section,
            blueprint=direct_blueprint,
            evidence_map=job.evidence_map,
            packet=packet,
            model=model,
        ))
    if stage == "section":
        if len(documents) != 1:
            raise EvaluationContractError("section stage requires exactly one non-reference section")
        report_document = _minimal_report_document(documents[0], direct_blueprint)
        editorial = {}
        editorial_prevalidation = None
        prevalidation_markers = None
    else:
        editorial_prevalidation = (
            editorial_claim_validator(documents, direct_blueprint)
            if editorial_claim_validator is not None
            else None
        )
        documents, prevalidation_markers = _mark_prevalidation_rejections(
            documents,
            editorial_prevalidation,
        )
        editorial = _direct_editorial(
            user_request=str(job.normalized_request.get("user_request") or ""),
            blueprint=direct_blueprint,
            documents=documents,
            evidence_map=job.evidence_map,
            model=model,
            context_profile=context_manager.profile,
            section_entailment=editorial_prevalidation,
        )
        completeness = _delivery_completeness(direct_blueprint.get("completeness") or {}, documents)
        citation_map = _citation_map(job.evidence_map)
        content, sanitized_count = _assemble_report(
            str(job.normalized_request.get("user_request") or ""),
            documents,
            editorial,
            citation_map,
            job.evidence_map,
            completeness=completeness,
            research_mode=str(direct_blueprint.get("research_mode") or "domain_practice_review"),
            reference_title=_reference_title_from_blueprint(direct_blueprint),
        )
        report_document = {
            "title": editorial.get("report_title") or str(job.normalized_request.get("user_request") or ""),
            "content": content,
            "claim_ledger": _final_report_claim_ledger(documents, editorial),
            "delivery_status": completeness.get("status"),
            "completeness": completeness,
            "research_mode": direct_blueprint.get("research_mode"),
            "sanitized_evidence_id_count": sanitized_count,
        }
    return {
        "arm": ARM_DIRECT,
        "report_blueprint": direct_blueprint,
        "documents": documents,
        "editorial": editorial,
        "editorial_prevalidation": editorial_prevalidation,
        "prevalidation_markers": prevalidation_markers,
        "report_document": report_document,
        "context_diagnostics": {
            "blueprint_profile": blueprint_profile,
            "blueprint_sha256": stable_hash(direct_blueprint),
            "sections": context_diagnostics,
        },
        "model_calls": model.calls,
    }


def deterministic_validation(
    *,
    candidate: dict[str, Any],
    job: FrozenJob,
) -> dict[str, Any]:
    documents = candidate["documents"]
    report_document = candidate["report_document"]
    citation = validate_report_citations(
        documents=documents,
        report_document=report_document,
        evidence_map=job.evidence_map,
    )
    entailment = validate_claim_entailment(
        report_document=report_document,
        evidence_map=job.evidence_map,
        model=None,
    )
    from backend.services.research_service import _annotate_entailment_core_scope

    validation_blueprint = (
        candidate.get("report_blueprint")
        if isinstance(candidate.get("report_blueprint"), dict)
        else job.report_blueprint
    )
    entailment = _annotate_entailment_core_scope(entailment, validation_blueprint)
    report_quality = evaluate_report_quality(
        report_document=report_document,
        question_states=job.question_states,
        # Both arms use the same Finding-neutral validator input. Passing historical Findings
        # only for Arm F would leak forbidden data to Arm E and make the critic asymmetric.
        findings=[],
        rejected_evidence=job.rejected_evidence,
        documents=documents,
        model=None,
        evidence_map=job.evidence_map,
    )
    grounding = {
        "status": "passed" if all(bool(item.get("grounded_exactly")) for item in job.evidence_map) else "failed",
        "evidence_url_count": len({str(item.get("source_url") or "") for item in job.evidence_map}),
        "grounded_evidence_count": sum(bool(item.get("grounded_exactly")) for item in job.evidence_map),
        "evidence_count": len(job.evidence_map),
        "per_document": [],
    }
    quality_gate = combined_quality_gate(
        grounding=grounding,
        validation=citation,
        report_quality=report_quality,
        entailment=entailment,
    )
    return {
        "mode": "deterministic_preflight",
        "citation": citation,
        "entailment": entailment,
        "critic": report_quality,
        "grounding": grounding,
        "quality_gate": quality_gate,
    }


def live_validation(
    *,
    candidate: dict[str, Any],
    job: FrozenJob,
    validation_model: Callable[..., dict[str, Any]],
    parallelism: int = 6,
    prior_entailment_judgments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    documents = candidate["documents"]
    report_document = candidate["report_document"]
    citation = validate_report_citations(
        documents=documents,
        report_document=report_document,
        evidence_map=job.evidence_map,
    )
    # Match production's validation contract: a cited Evidence row may be judged against the
    # section-scoped context from the same source that the Writer was allowed to inspect. The
    # raw frozen Evidence map remains authoritative for citation-ID validation and grounding.
    from backend.services.research_service import (
        _annotate_entailment_core_scope,
        _entailment_evidence_with_writer_context,
    )

    validation_blueprint = (
        candidate.get("report_blueprint")
        if isinstance(candidate.get("report_blueprint"), dict)
        else job.report_blueprint
    )
    entailment_evidence_map = _entailment_evidence_with_writer_context(
        job.evidence_map,
        validation_blueprint,
    )
    entailment = validate_claim_entailment(
        report_document=report_document,
        evidence_map=entailment_evidence_map,
        model=validation_model,
        parallelism=max(1, parallelism),
        prior_judgments=prior_entailment_judgments,
    )
    # Production annotates semantic judgments with the Blueprint's core-section contract before
    # applying the combined gate. The controlled evaluator must do the same; otherwise a core
    # unsupported claim is silently treated as non-blocking in the experiment only.
    entailment = _annotate_entailment_core_scope(entailment, validation_blueprint)
    report_quality = evaluate_report_quality(
        report_document=report_document,
        question_states=job.question_states,
        findings=[],
        rejected_evidence=job.rejected_evidence,
        documents=documents,
        model=validation_model,
        evidence_map=entailment_evidence_map,
        section_critic_parallelism=max(1, parallelism),
    )
    grounding = {
        "status": "passed" if all(bool(item.get("grounded_exactly")) for item in job.evidence_map) else "failed",
        "evidence_url_count": len({str(item.get("source_url") or "") for item in job.evidence_map}),
        "grounded_evidence_count": sum(bool(item.get("grounded_exactly")) for item in job.evidence_map),
        "evidence_count": len(job.evidence_map),
        "per_document": [],
    }
    return {
        "mode": "production_validators_live",
        "citation": citation,
        "entailment": entailment,
        "critic": report_quality,
        "grounding": grounding,
        "quality_gate": combined_quality_gate(
            grounding=grounding,
            validation=citation,
            report_quality=report_quality,
            entailment=entailment,
        ),
    }


def provider_signature(model_calls: list[dict[str, Any]]) -> list[list[str]]:
    signatures = set()
    for call in model_calls:
        response = call.get("response") if isinstance(call.get("response"), dict) else {}
        metadata = response.get("_experiment_model_metadata") if isinstance(response, dict) else {}
        if isinstance(metadata, dict) and metadata.get("provider"):
            signatures.add((str(metadata.get("provider")), str(metadata.get("model") or "")))
    return [list(item) for item in sorted(signatures)]


def _writer_call_contracts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    contracts = []
    for call in candidate.get("model_calls") or []:
        if not isinstance(call, dict):
            continue
        response = call.get("response") if isinstance(call.get("response"), dict) else {}
        metadata = (
            response.get("_experiment_model_metadata")
            if isinstance(response.get("_experiment_model_metadata"), dict)
            else {}
        )
        trace_calls = [
            item
            for item in metadata.get("trace_events") or []
            if isinstance(item, dict) and item.get("event") == "model_call"
        ]
        successful = [item for item in trace_calls if item.get("status") in {"success", "empty_response"}]
        final = successful[-1] if successful else (trace_calls[-1] if trace_calls else {})
        parameters = final.get("parameters") if isinstance(final.get("parameters"), dict) else {}
        contracts.append({
            "status": call.get("status"),
            "provider": metadata.get("provider") or final.get("actual_provider"),
            "model": metadata.get("model") or final.get("actual_model") or final.get("requested_model"),
            "base_url": parameters.get("base_url"),
            "temperature": parameters.get("temperature"),
            "max_input_tokens": metadata.get("max_input_tokens"),
            "max_output_tokens": metadata.get("max_output_tokens") or parameters.get("max_tokens"),
            "configured_timeout_seconds": metadata.get("configured_timeout_seconds"),
            "effective_timeout_seconds": metadata.get("effective_timeout_seconds"),
            "fallback": bool(metadata.get("fallback")),
            "estimated_input_tokens": call.get("estimated_input_tokens") or metadata.get("estimated_input_tokens"),
        })
    return contracts


def _section_prompt_contract(candidate: dict[str, Any]) -> dict[str, Any]:
    calls = [item for item in candidate.get("model_calls") or [] if isinstance(item, dict)]
    prompt = calls[0].get("prompt") if calls and isinstance(calls[0].get("prompt"), dict) else {}
    section = prompt.get("section_brief") if isinstance(prompt.get("section_brief"), dict) else {}
    neutral_section = {
        key: copy.deepcopy(section.get(key))
        for key in (
            "section_id",
            "title",
            "purpose",
            "render_as",
            "writing_strategy",
            "research_questions",
            "known_gaps",
        )
    }
    return {
        "section": neutral_section,
        "report_outline": copy.deepcopy(prompt.get("report_outline") or []),
        "research_mode": prompt.get("research_mode"),
        "target_length_chars": prompt.get("target_length_chars"),
        "schema_fields": sorted((prompt.get("schema") or {}).keys())
        if isinstance(prompt.get("schema"), dict)
        else [],
    }


def _prompt_evidence_ids(candidate: dict[str, Any]) -> set[str]:
    evidence_ids: set[str] = set()
    for call in candidate.get("model_calls") or []:
        prompt = call.get("prompt") if isinstance(call, dict) and isinstance(call.get("prompt"), dict) else {}
        for row in prompt.get("evidence") or []:
            if isinstance(row, dict) and row.get("id"):
                evidence_ids.add(str(row["id"]))

        def collect(value: Any, *, inside_packet: bool = False) -> None:
            if isinstance(value, dict):
                if inside_packet and value.get("id") and (
                    "atomic_claim" in value or "quote" in value or "source_url" in value
                ):
                    evidence_ids.add(str(value["id"]))
                for key, item in value.items():
                    collect(item, inside_packet=inside_packet or key == "evidence_packet")
            elif isinstance(value, list):
                for item in value:
                    collect(item, inside_packet=inside_packet)

        collect(prompt)
    return evidence_ids


def pair_fairness_audit(
    finding_run: dict[str, Any],
    direct_run: dict[str, Any],
    job: FrozenJob,
) -> dict[str, Any]:
    """Audit technical comparability without treating the intended handoff as a defect."""
    finding_candidate = (
        finding_run.get("candidate") if isinstance(finding_run.get("candidate"), dict) else {}
    )
    direct_candidate = (
        direct_run.get("candidate") if isinstance(direct_run.get("candidate"), dict) else {}
    )
    finding_contracts = _writer_call_contracts(finding_candidate)
    direct_contracts = _writer_call_contracts(direct_candidate)
    finding_prompt_contract = _section_prompt_contract(finding_candidate)
    direct_prompt_contract = _section_prompt_contract(direct_candidate)
    finding_visible = _prompt_evidence_ids(finding_candidate)
    direct_visible = _prompt_evidence_ids(direct_candidate)

    direct_prompt_text = canonical_json([
        item.get("prompt")
        for item in direct_candidate.get("model_calls") or []
        if isinstance(item, dict)
    ])
    leaked_finding_ids = sorted({
        str(item.get("finding_id") or "")
        for item in job.findings
        if item.get("finding_id") and str(item.get("finding_id")) in direct_prompt_text
    })
    leaked_finding_text_hashes = sorted({
        stable_hash(str(item.get("finding") or ""))
        for item in job.findings
        if len(str(item.get("finding") or "").strip()) >= 12
        and str(item.get("finding") or "").strip() in direct_prompt_text
    })

    def comparable_resource_view(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                key: item.get(key)
                for key in (
                    "status",
                    "provider",
                    "model",
                    "base_url",
                    "temperature",
                    "max_input_tokens",
                    "max_output_tokens",
                    "configured_timeout_seconds",
                    "effective_timeout_seconds",
                    "fallback",
                )
            }
            for item in contracts
        ]

    checks = {
        "same_job": finding_run.get("job_id") == direct_run.get("job_id") == job.job_id,
        "same_section": finding_run.get("section_id") == direct_run.get("section_id"),
        "same_repetition": finding_run.get("repetition") == direct_run.get("repetition"),
        "same_frozen_evidence_hash": bool(job.artifacts.get("evidence_map")),
        "same_neutral_section_contract": (
            finding_prompt_contract["section"] == direct_prompt_contract["section"]
        ),
        "same_report_outline": (
            finding_prompt_contract["report_outline"] == direct_prompt_contract["report_outline"]
        ),
        "same_research_mode": (
            finding_prompt_contract["research_mode"] == direct_prompt_contract["research_mode"]
        ),
        "same_target_length": (
            finding_prompt_contract["target_length_chars"]
            == direct_prompt_contract["target_length_chars"]
        ),
        "same_output_schema": (
            finding_prompt_contract["schema_fields"] == direct_prompt_contract["schema_fields"]
        ),
        "same_writer_call_count": len(finding_contracts) == len(direct_contracts),
        "same_writer_resource_contract": (
            comparable_resource_view(finding_contracts)
            == comparable_resource_view(direct_contracts)
        ),
        "all_writer_calls_succeeded": all(
            item.get("status") == "success" for item in [*finding_contracts, *direct_contracts]
        ),
        "direct_has_no_finding_id_leakage": not leaked_finding_ids,
        "direct_has_no_finding_text_leakage": not leaked_finding_text_hashes,
    }
    hard_failures = sorted(key for key, passed in checks.items() if not passed)
    finding_history = historical_finding_trace_summary(job)
    return {
        "schema_version": "finding-architecture-pair-fairness-audit/v1",
        "pair_id": finding_run.get("pair_id"),
        "job_id": job.job_id,
        "section_id": finding_run.get("section_id"),
        "assessment_scope": "development_diagnostic_system_effect",
        "technically_comparable": not hard_failures,
        "hard_check_failures": hard_failures,
        "checks": checks,
        "writer_resource_contracts": {
            ARM_FINDING: finding_contracts,
            ARM_DIRECT: direct_contracts,
        },
        "treatment_diagnostics": {
            "finding_writer_visible_evidence_ids": sorted(finding_visible),
            "direct_writer_visible_evidence_ids": sorted(direct_visible),
            "finding_writer_visible_evidence_count": len(finding_visible),
            "direct_writer_visible_evidence_count": len(direct_visible),
            "shared_writer_visible_evidence_count": len(finding_visible & direct_visible),
            "only_finding_writer_visible_evidence_ids": sorted(finding_visible - direct_visible),
            "only_direct_writer_visible_evidence_ids": sorted(direct_visible - finding_visible),
            "interpretation": (
                "Writer-visible Evidence may differ by design: historical Finding bindings are "
                "the Arm F handoff, while deterministic Evidence selection is the Arm E handoff."
            ),
        },
        "finding_history_diagnostics": {
            "accepted_finding_count": finding_history.get("accepted_finding_count"),
            "historical_model_call_count": finding_history.get("model_call_count"),
            "historical_model_failure_count": finding_history.get("model_failure_count"),
            "rejection_details_available": finding_history.get("rejection_details_available"),
        },
        "finding_leakage": {
            "finding_ids": leaked_finding_ids,
            "finding_text_hashes": leaked_finding_text_hashes,
        },
        "known_limits": [
            "The comparison estimates the current frozen historical Finding-mediated handoff, not an intrinsic property of every possible Finding algorithm.",
            "Arm F retains the historical Finding-influenced Blueprint; Arm E uses its Finding-free structural projection.",
            "Historical Finding generation model/provider failures and unavailable rejection details are part of the frozen treatment and can confound algorithm-level attribution.",
            "Prompt contracts are semantically aligned but cannot be text-identical because the supplied material schemas differ.",
        ],
    }


def paired_order(seed: int, units: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], list[str]]:
    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)
    result = {}
    for index, unit in enumerate(shuffled):
        result[unit] = [ARM_FINDING, ARM_DIRECT] if index % 2 == 0 else [ARM_DIRECT, ARM_FINDING]
    return result


def blind_pair(seed: int, pair_id: str, run_ids: dict[str, str]) -> dict[str, Any]:
    order = [ARM_FINDING, ARM_DIRECT]
    random.Random(f"{seed}:{pair_id}").shuffle(order)
    labels = {order[0]: "candidate_A", order[1]: "candidate_B"}
    swapped = {order[0]: "candidate_B", order[1]: "candidate_A"}
    return {
        "pair_id": pair_id,
        "opaque_candidates": {
            labels[arm]: {"opaque_run_id": run_ids[arm]} for arm in order
        },
        "presentation_order": [labels[arm] for arm in order],
        "position_swap": {
            "opaque_candidates": {
                swapped[arm]: {"opaque_run_id": run_ids[arm]} for arm in reversed(order)
            },
            "presentation_order": [swapped[arm] for arm in reversed(order)],
        },
        "arm_mapping_sealed": {
            labels[arm]: arm for arm in order
        },
        "judge_contract": {
            "external_knowledge_allowed": False,
            "evidence_id_level_reasons_required": True,
            "dimensions": [
                "grounding", "completeness", "synthesis", "traceability", "boundary_handling", "readability"
            ],
        },
    }


def code_version_for_experiment(root: Path) -> dict[str, Any]:
    version = audit.code_version()
    if not (root / ".git").exists():
        version = {**version, "git_commit": None, "git_status": "git_commit_unavailable"}
    return version


_ISOLATED_MODEL_TRACE_LOCK = threading.Lock()


def _experiment_model_kwargs_with_user_agent(
    model_kwargs: dict[str, Any],
    user_agent: str | None,
) -> dict[str, Any]:
    output = dict(model_kwargs)
    if not user_agent:
        return output
    headers = dict(output.get("default_headers") or {})
    headers["User-Agent"] = user_agent
    output["default_headers"] = headers
    return output


class IsolatedProductionModelInvoker:
    """Call the current production model adapter while redirecting audit into the experiment.

    The production adapter is reused for routing, temperature, timeout, retries, JSON repair,
    and fallback. Its audit root is temporarily redirected under the experiment output so no
    historical Job directory is modified. Calls are serialized because the audit root is a
    process-global setting.
    """

    def __init__(
        self,
        *,
        trace_root: Path,
        run_id: str,
        role: str,
        max_input_tokens: int | None,
        max_output_tokens: int | None,
        user_agent: str | None = None,
    ) -> None:
        self.trace_root = Path(trace_root)
        self.run_id = run_id
        self.role = role
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.user_agent = str(user_agent or "").strip() or None
        self.call_index = 0

    def __call__(
        self,
        payload: dict[str, Any],
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        estimated = estimate_tokens({"system_prompt": WRITER_SYSTEM_PROMPT, "payload": payload})
        if self.max_input_tokens is not None and estimated > self.max_input_tokens:
            return {
                "_failure_reason": "input_token_budget_exceeded",
                "_experiment_model_metadata": {
                    "estimated_input_tokens": estimated,
                    "max_input_tokens": self.max_input_tokens,
                },
            }
        self.call_index += 1
        trace_job_id = f"{self.run_id}_call_{self.call_index:03d}"
        configured_timeout = float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75"))
        effective_timeout = configured_timeout
        if timeout_override is not None:
            effective_timeout = min(configured_timeout, max(1.0, float(timeout_override)))
        with _ISOLATED_MODEL_TRACE_LOCK:
            previous_root = audit.RESEARCH_ARTIFACT_DIR
            previous_audit = os.environ.get("RESEARCH_AUDIT_ENABLED")
            previous_trace = os.environ.get("RESEARCH_MODEL_TRACE_ENABLED")
            audit.RESEARCH_ARTIFACT_DIR = self.trace_root
            os.environ["RESEARCH_AUDIT_ENABLED"] = "1"
            os.environ["RESEARCH_MODEL_TRACE_ENABLED"] = "1"
            try:
                from backend.research import model_runtime
                from backend.services import research_service

                original_provider_invoke = model_runtime._invoke_provider_chat
                original_chat_openai = model_runtime.ChatOpenAI

                def experiment_chat_openai(*args: Any, **kwargs: Any) -> Any:
                    return original_chat_openai(
                        *args,
                        **_experiment_model_kwargs_with_user_agent(kwargs, self.user_agent),
                    )

                def capped_provider_invoke(*args: Any, **kwargs: Any) -> Any:
                    if self.max_output_tokens is not None:
                        kwargs["max_tokens"] = self.max_output_tokens
                    return original_provider_invoke(*args, **kwargs)

                model_runtime._invoke_provider_chat = capped_provider_invoke
                if self.user_agent:
                    model_runtime.ChatOpenAI = experiment_chat_openai

                try:
                    with audit.audit_run(job_id=trace_job_id, run_id=self.run_id):
                        if self.role == "writer":
                            result = research_service._research_writer_json_model(payload)
                        elif self.role == "validator":
                            result = research_service._invoke_research_json(
                                payload,
                                timeout_seconds=effective_timeout,
                                temperature=0,
                                evidence=True,
                                allow_provider_fallback=True,
                                fallback_stage="finding_architecture_validation",
                            )
                        else:
                            raise EvaluationContractError(f"unsupported experiment model role: {self.role}")
                    events = audit.read_trace_events(trace_job_id)
                finally:
                    model_runtime._invoke_provider_chat = original_provider_invoke
                    model_runtime.ChatOpenAI = original_chat_openai
            finally:
                audit.RESEARCH_ARTIFACT_DIR = previous_root
                if previous_audit is None:
                    os.environ.pop("RESEARCH_AUDIT_ENABLED", None)
                else:
                    os.environ["RESEARCH_AUDIT_ENABLED"] = previous_audit
                if previous_trace is None:
                    os.environ.pop("RESEARCH_MODEL_TRACE_ENABLED", None)
                else:
                    os.environ["RESEARCH_MODEL_TRACE_ENABLED"] = previous_trace
        model_calls = [item for item in events if item.get("event") == "model_call"]
        successful = [item for item in model_calls if item.get("status") in {"success", "empty_response"}]
        final = successful[-1] if successful else (model_calls[-1] if model_calls else {})
        metadata = {
            "provider": final.get("actual_provider"),
            "model": final.get("actual_model") or final.get("requested_model"),
            "requested_provider": final.get("requested_provider"),
            "requested_model": final.get("requested_model"),
            "fallback": any(str(item.get("actual_provider") or "") == "mimo_fallback" for item in model_calls),
            "usage": final.get("usage_metadata") or (final.get("response_metadata") or {}).get("token_usage"),
            "trace_job_id": trace_job_id,
            "trace_events": events,
            "estimated_input_tokens": estimated,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "configured_timeout_seconds": configured_timeout,
            "timeout_override_seconds": timeout_override,
            "effective_timeout_seconds": effective_timeout,
            "user_agent_override": self.user_agent,
        }
        output = dict(result or {})
        output["_experiment_model_metadata"] = sanitized_json_safe(metadata)
        return output


def _section_ids_for_row(job: FrozenJob, row: dict[str, Any], stage: str) -> list[str]:
    if stage == "report":
        return []
    configured = [str(item) for item in row.get("section_ids") or [] if str(item)]
    if configured:
        return configured
    substantive = [
        str(section.get("section_id") or "")
        for section in job.report_blueprint.get("sections") or []
        if str(section.get("section_id") or "")
        and not _is_reference_title(str(section.get("title") or ""))
        and str(section.get("section_id") or "") != "sec_00"
    ]
    if not substantive:
        raise EvaluationContractError(f"no replayable section found for {job.job_id}")
    return [substantive[0]]


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    validation = run.get("validation") or {}
    entailment = validation.get("entailment") or {}
    report = run.get("candidate", {}).get("report_document") or {}
    qualification = run.get("candidate", {}).get("qualification") or {}
    claim_audit = validation.get("claim_evidence_audit_summary") or {}
    return {
        "opaque_run_id": run.get("opaque_run_id"),
        "pair_id": run.get("pair_id"),
        "job_id": run.get("job_id"),
        "section_id": run.get("section_id"),
        "arm": run.get("arm"),
        "status": run.get("status"),
        "candidate_qualification_status": qualification.get("status"),
        "candidate_qualification_reasons": qualification.get("reasons") or [],
        "provider_signature": run.get("provider_signature"),
        "report_chars": len(str(report.get("content") or "")),
        "claim_count": entailment.get("claim_count"),
        "unsupported_count": entailment.get("unsupported_count"),
        "label_mismatch_count": entailment.get("label_mismatch_count"),
        "citation_status": (validation.get("citation") or {}).get("status"),
        "quality_gate_status": (validation.get("quality_gate") or {}).get("status"),
        "claim_evidence_fidelity": claim_audit,
        "input_token_estimates": [call.get("estimated_input_tokens") for call in run.get("candidate", {}).get("model_calls") or []],
    }


def candidate_qualification(candidate: dict[str, Any]) -> dict[str, Any]:
    """Reject vacuous validation passes before a run enters the paired comparison."""
    documents = [item for item in candidate.get("documents") or [] if isinstance(item, dict)]
    report_document = candidate.get("report_document") if isinstance(candidate.get("report_document"), dict) else {}
    report_content = str(report_document.get("content") or "").strip()
    model_calls = [item for item in candidate.get("model_calls") or [] if isinstance(item, dict)]
    failed_calls = [item for item in model_calls if item.get("status") in {"failed", "exception"}]
    successful_calls = [item for item in model_calls if item.get("status") == "success"]
    failed_documents = [
        item for item in documents
        if str((item.get("synthesis_diagnostics") or {}).get("mode") or "") == "model_failed"
    ]
    completeness = report_document.get("completeness") if isinstance(report_document.get("completeness"), dict) else {}
    core_empty = [str(item) for item in completeness.get("core_empty_section_ids") or []]
    reasons = []
    if not report_content:
        reasons.append("empty_report_content")
    if model_calls and not successful_calls:
        reasons.append("no_successful_model_call")
    if failed_documents:
        reasons.append("model_failed_document")
    if core_empty:
        reasons.append("core_section_empty")
    if not report_content or (model_calls and not successful_calls):
        status = "failed"
    elif failed_documents or core_empty:
        status = "partial"
    else:
        status = "passed"
    return {
        "status": status,
        "reasons": reasons,
        "report_chars": len(report_content),
        "document_count": len(documents),
        "model_call_count": len(model_calls),
        "successful_model_call_count": len(successful_calls),
        "failed_model_call_count": len(failed_calls),
        "model_failed_document_count": len(failed_documents),
        "core_empty_section_ids": core_empty,
        "eligible_for_primary_comparison": status == "passed",
    }


def run_experiment(
    *,
    root: Path,
    artifact_root: Path,
    experiment_id: str,
    dataset_path: Path,
    split: str,
    arms: list[str],
    stage: str,
    repetitions: int,
    dry_run: bool,
    output_dir: Path,
    random_seed: int = 20260903,
    max_input_tokens: int = 48000,
    max_output_tokens: int = 8192,
    prompt_reserve_tokens: int = 6500,
    direct_context_profile: str = DIRECT_CONTEXT_PROFILE_BASELINE,
    direct_blueprint_profile: str = DIRECT_BLUEPRINT_PROFILE_HISTORICAL_PROJECTION,
    validation_mode: str = "live",
    validation_budget_seconds: float = 2400,
    job_ids: list[str] | None = None,
    allow_holdout: bool = False,
    writer_user_agent: str | None = None,
    retry_504: bool = False,
    resume: bool = False,
    uncapped_model_tokens: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_id):
        raise EvaluationContractError("experiment_id must contain only letters, numbers, dot, dash, underscore")
    if stage not in SUPPORTED_STAGES:
        raise EvaluationContractError(f"unsupported stage: {stage}")
    if repetitions < 1:
        raise EvaluationContractError("repetitions must be positive")
    if validation_mode not in SUPPORTED_VALIDATION_MODES:
        raise EvaluationContractError(f"unsupported validation mode: {validation_mode}")
    if validation_budget_seconds <= 0:
        raise EvaluationContractError("validation_budget_seconds must be positive")
    if direct_context_profile not in SUPPORTED_DIRECT_CONTEXT_PROFILES:
        raise EvaluationContractError(
            f"unsupported Direct Evidence context profile: {direct_context_profile}; "
            f"expected one of {sorted(SUPPORTED_DIRECT_CONTEXT_PROFILES)}"
        )
    if direct_blueprint_profile not in SUPPORTED_DIRECT_BLUEPRINT_PROFILES:
        raise EvaluationContractError(
            f"unsupported Direct Evidence blueprint profile: {direct_blueprint_profile}; "
            f"expected one of {sorted(SUPPORTED_DIRECT_BLUEPRINT_PROFILES)}"
        )
    if not arms or any(arm not in SUPPORTED_ARMS for arm in arms) or len(arms) != len(set(arms)):
        raise EvaluationContractError(f"arms must be unique values from {sorted(SUPPORTED_ARMS)}")
    if split == "holdout" and not dry_run and not allow_holdout:
        raise EvaluationContractError("live holdout execution requires explicit allow_holdout=True")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise EvaluationContractError(f"experiment output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    loader = CheckpointArtifactLoader(artifact_root)
    all_rows = select_dataset_rows(dataset, "all")
    jobs: dict[str, FrozenJob] = {}
    preflight_rows = []
    for row in all_rows:
        job_id = str(row["job_id"])
        jobs[job_id] = loader.load_job(job_id)
        preflight_rows.append(preflight_job(jobs[job_id], row))
    isolation = validate_dataset_isolation(dataset, jobs)
    selected_rows = select_dataset_rows(dataset, split)
    if job_ids:
        requested_job_ids = {str(item) for item in job_ids if str(item)}
        available_job_ids = {str(row["job_id"]) for row in selected_rows}
        unknown_job_ids = sorted(requested_job_ids - available_job_ids)
        if unknown_job_ids:
            raise EvaluationContractError(
                f"job filter is outside selected dataset split: {unknown_job_ids}"
            )
        selected_rows = [row for row in selected_rows if str(row["job_id"]) in requested_job_ids]
        if not selected_rows:
            raise EvaluationContractError("job filter selected no dataset rows")

    inventories = []
    inventory_by_job: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        job = jobs[str(row["job_id"])]
        section_ids = set(_section_ids_for_row(job, row, stage)) if stage == "section" else None
        inventory = build_reference_inventory(job, section_ids)
        inventory_by_job[job.job_id] = inventory
        inventory_path = output_dir / "inventories" / f"{job.job_id}.candidate.json"
        atomic_write_json(inventory_path, inventory)
        inventories.append({
            "job_id": job.job_id,
            "path": str(inventory_path),
            "candidate_sha256": inventory["candidate_sha256"],
            "status": inventory["status"],
        })
        frozen_path_value = str(row.get("reference_inventory") or "")
        if frozen_path_value:
            frozen_path = Path(frozen_path_value)
            if not frozen_path.is_absolute():
                frozen_path = Path(dataset_path).resolve().parent / frozen_path
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            validate_frozen_inventory(frozen, job)
            inventory_by_job[job.job_id] = frozen
        elif row.get("split") == "holdout" and not dry_run:
            raise EvaluationContractError(f"holdout job lacks frozen reference inventory: {job.job_id}")

    prompt_contracts = {
        ARM_FINDING: {
            "source": "backend.research.writer._model_section/_edit_report",
            "policy": "current production Finding-mediated Writer contract, unchanged",
            "system_prompt_hash": stable_hash(WRITER_SYSTEM_PROMPT),
        },
        ARM_DIRECT: {
            "prompt_version": (
                CLUSTER_GUARDED_PROMPT_VERSION
                if direct_context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
                else PROMPT_VERSION
            ),
            "context_profile": direct_context_profile,
            "blueprint_profile": direct_blueprint_profile,
            "system_prompt": WRITER_SYSTEM_PROMPT,
            "system_prompt_hash": stable_hash(WRITER_SYSTEM_PROMPT),
            "section_rules": direct_writer_rules(
                1000,
                context_profile=direct_context_profile,
            ),
        },
    }
    for arm, contract in prompt_contracts.items():
        atomic_write_json(output_dir / "prompts" / f"arm_{'f' if arm == ARM_FINDING else 'e'}.json", contract)

    units: list[tuple[str, str, int]] = []
    unit_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected_rows:
        job = jobs[str(row["job_id"])]
        section_ids = _section_ids_for_row(job, row, stage)
        unit_section_ids = section_ids if stage == "section" else ["report"]
        for section_id in unit_section_ids:
            unit_rows[(job.job_id, section_id)] = row
            for repetition in range(1, repetitions + 1):
                units.append((job.job_id, section_id, repetition))
    order_by_unit = paired_order(random_seed, units)
    run_summaries = []
    pair_summaries = []
    blind_mapping: dict[str, Any] = {}
    context_manager = DirectEvidenceContextManager(
        max_input_tokens=max_input_tokens,
        prompt_reserve_tokens=prompt_reserve_tokens,
        profile=direct_context_profile,
    )
    model_max_input_tokens = None if uncapped_model_tokens else max_input_tokens
    model_max_output_tokens = None if uncapped_model_tokens else max_output_tokens

    def resilient_invoker(invoker: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        if not retry_504:
            return invoker
        from .finding_architecture_resilience import Backoff504JsonInvoker

        return Backoff504JsonInvoker(invoker)

    for job_id, section_id, repetition in units:
        job = jobs[job_id]
        pair_id = opaque_id(
            random_seed,
            experiment_id,
            job_id,
            section_id,
            str(repetition),
            prefix="pair",
        )
        run_ids = {
            arm: opaque_id(
                random_seed,
                experiment_id,
                job_id,
                section_id,
                str(repetition),
                arm,
                prefix="candidate",
            )
            for arm in arms
        }
        pair_runs: dict[str, dict[str, Any]] = {}
        for arm in order_by_unit[(job_id, section_id, repetition)]:
            if arm not in arms:
                continue
            run_id = run_ids[arm]
            checkpoint_blueprint = (
                job.report_blueprint
                if arm == ARM_FINDING
                else job.direct_inputs(
                    blueprint_profile=direct_blueprint_profile
                )["report_blueprint"]
            )
            checkpoint_source_hash = stable_hash({
                "job_id": job_id,
                "section_id": section_id,
                "normalized_request": job.normalized_request,
                "report_blueprint": checkpoint_blueprint,
                "evidence_map": job.evidence_map,
                "findings": job.findings if arm == ARM_FINDING else None,
            })
            checkpoint_contract = {
                "experiment_id": experiment_id,
                "arm": arm,
                "stage": stage,
                "repetition": repetition,
                "direct_context_profile": direct_context_profile,
                "direct_blueprint_profile": direct_blueprint_profile,
                "validation_mode": validation_mode,
                "validation_budget_seconds": validation_budget_seconds,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
                "prompt_reserve_tokens": prompt_reserve_tokens,
                "uncapped_model_tokens": uncapped_model_tokens,
                "writer_user_agent": writer_user_agent,
                "retry_504": retry_504,
            }
            checkpoint_path = output_dir / "checkpoints" / f"{run_id}.completed.json"
            if resume:
                from .finding_architecture_resilience import checkpoint_contract_hash, load_stage_checkpoint

                completed = load_stage_checkpoint(
                    checkpoint_path,
                    experiment_id=experiment_id,
                    candidate_id=run_id,
                    pair_id=pair_id,
                    arm=arm,
                    stage="completed",
                    source_run_sha256=checkpoint_source_hash,
                    contract_sha256=checkpoint_contract_hash(checkpoint_contract),
                )
                if completed:
                    run_path = output_dir / str(completed.get("run_path") or "")
                    if not run_path.is_file():
                        raise EvaluationContractError(f"completed run checkpoint artifact missing: {run_id}")
                    resumed_run = json.loads(run_path.read_text(encoding="utf-8"))
                    if stable_hash(resumed_run) != str(completed.get("run_sha256") or ""):
                        raise EvaluationContractError(f"completed run checkpoint artifact mismatch: {run_id}")
                    pair_runs[arm] = resumed_run
                    resumed_summary = _run_summary(resumed_run)
                    resumed_summary["resumed_from_checkpoint"] = True
                    run_summaries.append(resumed_summary)
                    continue
            if dry_run:
                invoker: Callable[[dict[str, Any]], dict[str, Any]] = dry_run_json_model
            else:
                invoker = resilient_invoker(IsolatedProductionModelInvoker(
                    trace_root=output_dir / "model_traces",
                    run_id=run_id,
                    role="writer",
                    max_input_tokens=model_max_input_tokens,
                    max_output_tokens=model_max_output_tokens,
                    user_agent=writer_user_agent,
                ))
            model = RecordingJsonModel(invoker)
            prevalidation_state: dict[str, Any] = {}
            editorial_claim_validator = None
            if (
                not dry_run
                and validation_mode == "live"
                and arm == ARM_DIRECT
                and stage == "report"
                and direct_blueprint_profile == DIRECT_BLUEPRINT_PROFILE_EVIDENCE_NATIVE
            ):
                def editorial_claim_validator(
                    documents: list[dict[str, Any]],
                    blueprint: dict[str, Any],
                ) -> dict[str, Any]:
                    from backend.services.research_service import (
                        _entailment_evidence_with_writer_context,
                    )

                    prevalidation_invoker = resilient_invoker(IsolatedProductionModelInvoker(
                        trace_root=output_dir / "model_traces",
                        run_id=f"{run_id}_validation_pre_editorial",
                        role="validator",
                        max_input_tokens=model_max_input_tokens,
                        max_output_tokens=model_max_output_tokens,
                        user_agent=writer_user_agent,
                    ))
                    prevalidation_model = RecordingJsonModel(DeadlineJsonModel(
                        prevalidation_invoker,
                        validation_budget_seconds,
                    ))
                    prevalidation_started = time.perf_counter()
                    try:
                        result = validate_claim_entailment(
                            report_document={
                                "claim_ledger": _final_report_claim_ledger(documents, {}),
                            },
                            evidence_map=_entailment_evidence_with_writer_context(
                                job.evidence_map,
                                blueprint,
                            ),
                            model=prevalidation_model,
                            parallelism=6,
                        )
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "judgments": [],
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    prevalidation_state.update({
                        "result": result,
                        "model_calls": prevalidation_model.calls,
                        "duration_seconds": time.perf_counter() - prevalidation_started,
                    })
                    return result

            selected_sections = [section_id] if stage == "section" else []
            started = time.perf_counter()
            error = None
            try:
                if arm == ARM_FINDING:
                    candidate = run_arm_finding(
                        job=job,
                        stage=stage,
                        section_ids=selected_sections,
                        model=model,
                    )
                else:
                    candidate = run_arm_direct(
                        job=job,
                        stage=stage,
                        section_ids=selected_sections,
                        model=model,
                        context_manager=context_manager,
                        blueprint_profile=direct_blueprint_profile,
                        editorial_claim_validator=editorial_claim_validator,
                    )
                qualification = candidate_qualification(candidate)
                candidate["qualification"] = qualification
                if not dry_run and qualification["status"] == "failed":
                    validation = {
                        "mode": "skipped_candidate_generation_failed",
                        "candidate_qualification": qualification,
                    }
                    status = "failed"
                    error = {
                        "type": "CandidateGenerationFailed",
                        "message": ", ".join(qualification["reasons"]),
                    }
                elif dry_run or validation_mode == "deterministic":
                    validation = deterministic_validation(candidate=candidate, job=job)
                    validation["candidate_qualification"] = qualification
                    status = "succeeded"
                else:
                    prevalidation_duration = float(prevalidation_state.get("duration_seconds") or 0.0)
                    remaining_validation_budget = max(
                        1.0,
                        float(validation_budget_seconds) - prevalidation_duration,
                    )
                    validation_invoker = resilient_invoker(IsolatedProductionModelInvoker(
                        trace_root=output_dir / "model_traces",
                        run_id=f"{run_id}_validation",
                        role="validator",
                        max_input_tokens=model_max_input_tokens,
                        max_output_tokens=model_max_output_tokens,
                        user_agent=writer_user_agent,
                    ))
                    validation_model = RecordingJsonModel(DeadlineJsonModel(
                        validation_invoker,
                        remaining_validation_budget,
                    ))
                    try:
                        prior_judgments = [
                            {**item, "confirmed": True}
                            for item in (
                                (candidate.get("editorial_prevalidation") or {}).get("judgments")
                                if isinstance(candidate.get("editorial_prevalidation"), dict)
                                else []
                            ) or []
                            if isinstance(item, dict)
                            and item.get("mode") == "model"
                            and item.get("status") in {"entailed", "reasonable_inference"}
                        ]
                        validation = live_validation(
                            candidate=candidate,
                            job=job,
                            validation_model=validation_model,
                            prior_entailment_judgments=prior_judgments,
                        )
                        validation["candidate_qualification"] = qualification
                        validation["editorial_prevalidation"] = candidate.get("editorial_prevalidation")
                        validation["model_calls"] = [
                            *(prevalidation_state.get("model_calls") or []),
                            *validation_model.calls,
                        ]
                        validation["validation_budget_accounting"] = {
                            "configured_seconds": float(validation_budget_seconds),
                            "pre_editorial_seconds": round(prevalidation_duration, 3),
                            "remaining_seconds_at_final_validation_start": round(
                                remaining_validation_budget,
                                3,
                            ),
                        }
                        status = "succeeded" if qualification["status"] == "passed" else "partial"
                    except Exception as validation_exc:
                        validation = {
                            "mode": "validation_failed",
                            "candidate_qualification": qualification,
                            "model_calls": [
                                *(prevalidation_state.get("model_calls") or []),
                                *validation_model.calls,
                            ],
                            "error": {
                                "type": type(validation_exc).__name__,
                                "message": str(validation_exc),
                            },
                        }
                        status = "failed"
                        error = {
                            "type": "ValidationFailed",
                            "cause_type": type(validation_exc).__name__,
                            "message": str(validation_exc),
                        }
                if status != "failed":
                    validation["inventory_metrics"] = inventory_metrics(
                        candidate, inventory_by_job[job_id], job
                    )
            except Exception as exc:
                candidate = {
                    "arm": arm,
                    "documents": [],
                    "editorial": {},
                    "report_document": {"content": "", "claim_ledger": []},
                    "model_calls": model.calls,
                }
                validation = {}
                status = "failed"
                error = {"type": type(exc).__name__, "message": str(exc)}
            claim_audit = claim_evidence_fidelity_audit(
                candidate,
                validation,
                job,
                inventory_by_job[job_id],
            )
            validation["claim_evidence_audit_summary"] = {
                "automatic_fidelity_status": claim_audit["automatic_fidelity_status"],
                "human_review_status": claim_audit["human_review_status"],
                **claim_audit["summary"],
            }
            run = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "opaque_run_id": run_id,
                "pair_id": pair_id,
                "job_id": job_id,
                "section_id": section_id,
                "stage": stage,
                "repetition": repetition,
                "arm": arm,
                "execution_position": order_by_unit[(job_id, section_id, repetition)].index(arm) + 1,
                "status": status,
                "dry_run": dry_run,
                "started_at": utc_now(),
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "candidate": candidate,
                "validation": validation,
                "provider_signature": provider_signature(candidate.get("model_calls") or []),
                "error": error,
            }
            atomic_write_json(output_dir / "runs" / f"{run_id}.json", run)
            atomic_write_json(output_dir / "claim_audits" / f"{run_id}.json", claim_audit)
            atomic_write_text(
                output_dir / "reports" / f"{run_id}.md",
                str(candidate.get("report_document", {}).get("content") or ""),
            )
            if status != "failed":
                from .finding_architecture_resilience import checkpoint_contract_hash, save_stage_checkpoint

                save_stage_checkpoint(
                    checkpoint_path,
                    experiment_id=experiment_id,
                    candidate_id=run_id,
                    pair_id=pair_id,
                    arm=arm,
                    stage="completed",
                    source_run_sha256=checkpoint_source_hash,
                    contract_sha256=checkpoint_contract_hash(checkpoint_contract),
                    payload={
                        "run_path": str(Path("runs") / f"{run_id}.json"),
                        "run_sha256": stable_hash(run),
                    },
                )
            pair_runs[arm] = run
            run_summaries.append(_run_summary(run))

        if set(pair_runs) == {ARM_FINDING, ARM_DIRECT}:
            signatures = {arm: pair_runs[arm]["provider_signature"] for arm in pair_runs}
            mismatch = signatures[ARM_FINDING] != signatures[ARM_DIRECT]
            pair_summary = {
                "pair_id": pair_id,
                "job_id": job_id,
                "section_id": section_id,
                "repetition": repetition,
                "execution_order": order_by_unit[(job_id, section_id, repetition)],
                "run_ids": run_ids,
                "provider_signatures": signatures,
                "provider_mismatch": mismatch,
                "quality_first_metrics": {
                    arm: _quality_first_metrics(pair_runs[arm]) for arm in (ARM_FINDING, ARM_DIRECT)
                },
                "included_in_primary_comparison": not dry_run and validation_mode == "live" and not mismatch and all(
                    pair_runs[arm]["status"] == "succeeded" for arm in pair_runs
                ),
            }
            pair_summaries.append(pair_summary)
            blind = blind_pair(random_seed, pair_id, run_ids)
            for presentation in (blind, blind["position_swap"]):
                for item in presentation["opaque_candidates"].values():
                    opaque_run_id = item["opaque_run_id"]
                    item["report_path"] = str(output_dir / "reports" / f"{opaque_run_id}.md")
                    item["claim_evidence_audit_path"] = str(
                        output_dir / "claim_audits" / f"{opaque_run_id}.json"
                    )
            blind_mapping[pair_id] = blind.pop("arm_mapping_sealed")
            atomic_write_json(output_dir / "blind" / "pairs" / f"{pair_id}.json", blind)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "code_version": code_version_for_experiment(root),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": stable_hash(dataset),
        "dataset_split": split,
        "job_filter": sorted({str(item) for item in job_ids or []}),
        "dataset_isolation": isolation,
        "jobs": [
            {
                "job_id": row["job_id"],
                "split": row["split"],
                "normalized_request_hash": normalized_request_key(jobs[str(row["job_id"])].normalized_request),
                "checkpoint_ids": preflight_job(jobs[str(row["job_id"])], row)["checkpoint_ids"],
                "artifact_hashes": preflight_job(jobs[str(row["job_id"])], row)["artifact_hashes"],
            }
            for row in selected_rows
        ],
        "preflight": preflight_rows,
        "inventories": inventories,
        "arms": arms,
        "stage": stage,
        "repetitions": repetitions,
        "dry_run": dry_run,
        "context_manager_version": context_manager.version,
        "direct_context_profile": direct_context_profile,
        "direct_blueprint_profile": direct_blueprint_profile,
        "prompt_version": (
            CLUSTER_GUARDED_PROMPT_VERSION
            if direct_context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
            else PROMPT_VERSION
        ),
        "prompt_hashes": {arm: stable_hash(contract) for arm, contract in prompt_contracts.items()},
        "model_contract": {
            "writer_model": os.getenv("RESEARCH_WRITER_MODEL_NAME") or os.getenv("MODEL_NAME"),
            "writer_temperature": float(os.getenv("RESEARCH_WRITER_TEMPERATURE", "0.3")),
            "timeout_seconds": float(os.getenv("RESEARCH_ROLE_MODEL_TIMEOUT_SECONDS", "75")),
            "max_input_tokens": model_max_input_tokens,
            "max_output_tokens": model_max_output_tokens,
            "uncapped_model_tokens": uncapped_model_tokens,
            "writer_user_agent": writer_user_agent,
            "retry_contract": (
                "current production retries plus audited HTTP 504 backoff"
                if retry_504
                else "current production _invoke_research_json and Writer retries"
            ),
            "retry_504_schedule_seconds": [5, 30, 60, 90, 120] if retry_504 else [],
            "citation_contract": "current production internal marker renderer and citation validator",
            "provider_mismatch_exclusion": True,
            "validation_mode": validation_mode,
            "validation_budget_seconds_per_candidate": validation_budget_seconds,
        },
        "resource_contract": {
            "max_input_tokens_per_call": model_max_input_tokens,
            "max_output_tokens_per_call": model_max_output_tokens,
            "context_selection_max_input_tokens": max_input_tokens,
            "prompt_reserve_tokens": prompt_reserve_tokens,
            "direct_context_profile": direct_context_profile,
            "direct_blueprint_profile": direct_blueprint_profile,
            "direct_context_selection_policy": (
                "cluster_guarded_no_budget_fill"
                if direct_context_profile == DIRECT_CONTEXT_PROFILE_CLUSTER_GUARDED
                else "baseline_fill_to_budget"
            ),
            "usage_source_priority": ["provider_reported", "shared_local_estimate"],
            "finding_historical_generation_cost": "reported_from_frozen_trace_only",
            "writer_incremental_cost": "measured_per_experiment_run",
        },
        "pair_contract": {
            "unit": "job_id + section_id + repetition",
            "execution_order": "deterministic_randomized_balanced",
            "adjacent_arm_execution": True,
            "provider_mismatch_excluded_from_primary_comparison": True,
            "failure_and_retry_contract_symmetric": True,
            "completed_candidate_checkpoint_resume": resume,
        },
        "reference_inventory_contract": {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "finding_independent": True,
            "candidate_sources": [
                "accepted evidence criterion bindings",
                "declared criterion bindings requiring human review",
                "question-scope fallback candidates requiring human review",
            ],
            "human_review_required_before_freeze": True,
            "review_note_required": True,
            "required_cluster_required_for_answerable_hard_soft_criteria": True,
            "frozen_inventory_required_for_live_holdout": True,
        },
        "validation_contract": {
            "shared_between_arms": True,
            "mode": validation_mode,
            "model_budget_seconds_per_candidate": validation_budget_seconds,
            "primary_objective": "final claim fidelity to cited original Evidence Quote and overall report quality",
            "metric_priority": [
                "claim_to_original_evidence_fidelity",
                "contradiction_scope_subject_axis_and_source_role_errors",
                "conflict_and_limitation_handling",
                "blind_overall_report_quality",
                "supported_criterion_coverage",
            "resource_cost_as_secondary_diagnostic",
            ],
            "validators": [
                "validate_report_citations",
                "validate_claim_entailment",
                "evaluate_report_quality",
                "combined_quality_gate",
            ],
            "automatic_winner": False,
            "blind_and_human_review_required_for_conclusion": True,
        },
        "random_seed": random_seed,
        "execution_order": [
            {"job_id": unit[0], "section_id": unit[1], "repetition": unit[2], "arms": order_by_unit[unit]}
            for unit in units
        ],
        "known_limitations": [
            (
                "Arm E rebuilds its Blueprint deterministically from frozen research_plan, question_states, and accepted Evidence; it does not read historical thesis, Finding bindings, dossier, Admission, or Finding-count reportability."
                if direct_blueprint_profile == DIRECT_BLUEPRINT_PROFILE_EVIDENCE_NATIVE
                else "Experiment 1 freezes the historical Blueprint structure, but Arm E receives a sanitized structural projection: thesis, finding/evidence bindings, dossier content, and Finding-count reportability fields are removed to enforce the no-Finding-access boundary."
            ),
            "The experiment enforces max output tokens by wrapping the current provider invocation inside an isolated, serialized model-trace scope; production routing defaults are not modified.",
            "Quality validator receives findings=[] for both arms because Arm E is forbidden from reading historical Finding text and quality.py otherwise embeds Finding-dependent heuristics.",
        ],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dry_run": dry_run,
        "validation_mode": validation_mode,
        "run_count": len(run_summaries),
        "pair_count": len(pair_summaries),
        "provider_mismatch_pair_count": sum(bool(item["provider_mismatch"]) for item in pair_summaries),
        "primary_comparison_pair_count": sum(bool(item["included_in_primary_comparison"]) for item in pair_summaries),
        "primary_objective": "Compare final prose against cited original Evidence, then compare overall report quality.",
        "cost_role": "secondary diagnostic; never substitutes for evidence fidelity or report quality",
        "runs": run_summaries,
        "pairs": pair_summaries,
        "quality_conclusion": (
            None
            if dry_run
            else "Prompt qualification only; deterministic validation cannot support a primary quality comparison."
            if validation_mode == "deterministic"
            else "Requires blind and human review; no automatic winner is computed."
        ),
    }
    atomic_write_json(output_dir / "dataset.json", dataset)
    atomic_write_json(output_dir / "blind" / "arm_mapping.sealed.json", blind_mapping)
    atomic_write_json(output_dir / "human_review.json", {
        "schema_version": "finding-architecture-human-review/v1",
        "experiment_id": experiment_id,
        "status": "pending",
        "preference_options": ["F_better", "E_better", "roughly_equal", "both_unacceptable"],
        "claim_review_verdicts": [
            "entailed",
            "reasonable_synthesis",
            "partial_support",
            "missing_citation",
            "unsupported_addition",
            "scope_mismatch",
            "subject_mismatch",
            "comparison_axis_mismatch",
            "source_role_mismatch",
            "contradicted_by_citation",
            "conflict_or_limitation_omission",
            "non_atomic",
            "non_claim_narration",
        ],
        "dimensions": [
            "claim_evidence_fidelity",
            "contradiction_and_conflict_handling",
            "overall_answer_quality",
            "completeness",
            "synthesis",
            "traceability",
            "boundary_handling",
            "readability",
        ],
        "reviews": [],
    })
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "summary.json", summary)
    return {"manifest": manifest, "summary": summary, "output_dir": str(output_dir)}
