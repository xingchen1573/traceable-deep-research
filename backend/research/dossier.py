"""Section-scoped research dossiers between retrieval and report writing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from urllib.parse import urlparse


def attach_section_dossiers(
    blueprint: dict[str, Any],
    *,
    findings: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
    materials: list[dict[str, Any]],
    max_chunks_per_section: int = 8,
    max_chunk_chars: int = 1600,
) -> dict[str, Any]:
    evidence_by_section = _group(evidence_map, "section_id")
    findings_by_section = _group(findings, "section_id")
    chunks_by_section = _group(selected_chunks, "section_id")
    material_by_url = {
        str(item.get("url") or item.get("source_url") or ""): item
        for item in materials
        if isinstance(item, dict) and str(item.get("url") or item.get("source_url") or "")
    }
    sections = []
    diagnostics = []
    for section in blueprint.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id") or "")
        dossier = build_section_dossier(
            section=section,
            findings=findings_by_section.get(section_id) or [],
            evidence=evidence_by_section.get(section_id) or [],
            chunks=chunks_by_section.get(section_id) or [],
            material_by_url=material_by_url,
            max_chunks=max_chunks_per_section,
            max_chunk_chars=max_chunk_chars,
        )
        sections.append({**section, "dossier": dossier})
        diagnostics.append({
            "section_id": section_id,
            "evidence_count": dossier["evidence_count"],
            "finding_count": dossier["finding_count"],
            "unused_evidence_count": len(dossier["unused_evidence_ids"]),
            "chunk_count": dossier["chunk_count"],
            "source_count": dossier["source_count"],
            "context_chars": dossier["context_chars"],
        })
    return {
        **blueprint,
        "sections": sections,
        "dossier_diagnostics": diagnostics,
    }


def build_section_dossier(
    *,
    section: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    material_by_url: dict[str, dict[str, Any]],
    max_chunks: int,
    max_chunk_chars: int,
) -> dict[str, Any]:
    finding_evidence_ids = {
        str(evidence_id)
        for finding in findings
        for evidence_id in finding.get("evidence_ids") or []
        if str(evidence_id)
    }
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence
        if str(item.get("id") or "")
    }
    ranked_chunks = sorted(
        [item for item in chunks if isinstance(item, dict) and str(item.get("text") or "").strip()],
        key=lambda item: float(item.get("hybrid_score") or item.get("score") or 0),
        reverse=True,
    )[:max_chunks]
    chunk_rows = [{
        "chunk_id": str(item.get("id") or ""),
        "question_id": str(item.get("question_id") or ""),
        "source_url": str(item.get("source_url") or ""),
        "source_title": str(item.get("source_title") or ""),
        "relevance_score": round(float(item.get("hybrid_score") or item.get("score") or 0), 3),
        "excerpt": str(item.get("text") or "").strip()[:max_chunk_chars],
    } for item in ranked_chunks]
    chunks_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in chunk_rows:
        chunks_by_url[item["source_url"]].append(item)
    evidence_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        evidence_by_url[str(item.get("source_url") or "")].append(item)
    source_urls = list(dict.fromkeys([
        *[str(item.get("source_url") or "") for item in evidence],
        *[str(item.get("source_url") or "") for item in chunk_rows],
    ]))
    source_notes = []
    for url in source_urls:
        if not url:
            continue
        source_evidence = evidence_by_url.get(url) or []
        source_chunks = chunks_by_url.get(url) or []
        material = material_by_url.get(url) or {}
        source_notes.append({
            "source_url": url,
            "source_title": str(
                (source_evidence[0] if source_evidence else {}).get("source_title")
                or (source_chunks[0] if source_chunks else {}).get("source_title")
                or material.get("title")
                or url
            ),
            "source_type": str((source_evidence[0] if source_evidence else {}).get("source_type") or material.get("source_type") or ""),
            "evidence_role": _source_role(source_evidence),
            "evidence_ids": [str(item.get("id")) for item in source_evidence if item.get("id")],
            "claims": [str(item.get("atomic_claim") or item.get("claim") or "") for item in source_evidence if str(item.get("atomic_claim") or item.get("claim") or "")],
            "excerpts": [item["excerpt"] for item in source_chunks[:2]] or ([str(material.get("excerpt") or "")[:max_chunk_chars]] if material.get("excerpt") else []),
        })
    question_threads = []
    for question in section.get("research_questions") or []:
        question_id = str(question.get("question_id") or "")
        thread_findings = [item for item in findings if str(item.get("question_id") or "") == question_id]
        question_threads.append({
            **question,
            "finding_ids": [str(item.get("finding_id")) for item in thread_findings if item.get("finding_id")],
            "evidence_ids": list(dict.fromkeys(
                str(evidence_id)
                for item in thread_findings
                for evidence_id in item.get("evidence_ids") or []
                if str(evidence_id)
            )),
        })
    return {
        "section_id": str(section.get("section_id") or ""),
        "purpose": str(section.get("purpose") or ""),
        "research_mode": str(section.get("writing_strategy") or ""),
        "question_threads": question_threads,
        "findings": findings,
        "evidence": [{
            "id": evidence_id,
            "claim": item.get("atomic_claim") or item.get("claim"),
            "quote": item.get("cleaned_quote") or item.get("raw_quote"),
            "source_url": item.get("source_url"),
            "source_title": item.get("source_title"),
            "source_type": item.get("source_type"),
            "publisher_relation": item.get("publisher_relation"),
            "scope": item.get("scope"),
            "claim_type": item.get("claim_type"),
            "stance": item.get("stance"),
            "directness_score": item.get("directness_score"),
            "subject": item.get("subject"),
            "subject_relation": item.get("subject_relation"),
            "criterion_ids": [str(value) for value in item.get("criterion_ids") or []],
            "declared_criterion_ids": [str(value) for value in item.get("declared_criterion_ids") or []],
            "criterion_binding_status": item.get("criterion_binding_status"),
            "comparison_axis": item.get("comparison_axis"),
            "comparison_metric": item.get("comparison_metric"),
            "expected_comparison_axis": item.get("expected_comparison_axis"),
            "comparison_axis_binding_status": item.get("comparison_axis_binding_status"),
        } for evidence_id, item in evidence_by_id.items()],
        "unused_evidence_ids": sorted(set(evidence_by_id) - finding_evidence_ids),
        "source_notes": source_notes,
        "relevant_chunks": chunk_rows,
        "known_gaps": section.get("known_gaps") or [],
        "finding_count": len(findings),
        "evidence_count": len(evidence_by_id),
        "chunk_count": len(chunk_rows),
        "source_count": len({urlparse(url).netloc for url in source_urls if urlparse(url).netloc}),
        "context_chars": sum(len(item["excerpt"]) for item in chunk_rows),
    }


def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if isinstance(item, dict):
            output[str(item.get(key) or "")].append(item)
    return output


def _source_role(evidence: list[dict[str, Any]]) -> str:
    if any(
        str(item.get("scope") or "") == "domain_specific"
        and str(item.get("claim_type") or "") == "direct_fact"
        for item in evidence
    ):
        return "observed_domain_practice"
    if evidence:
        return "transferable_engineering_evidence"
    return "context_only"
