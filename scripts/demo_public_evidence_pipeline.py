#!/usr/bin/env python3
"""Run the public synthetic Evidence trace demo without models or network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.research.evidence_native import (  # noqa: E402
    EvidenceContextManager,
    build_evidence_native_blueprint,
    stable_hash,
    validate_evidence_traceability,
)
from backend.research.quality import validate_report_citations  # noqa: E402


DEFAULT_FIXTURE = ROOT / "examples" / "public_evidence_demo.json"


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "normalized_request",
        "research_plan",
        "question_states",
        "selected_chunks",
        "evidence_map",
        "frozen_writer_output",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(required - set(payload if isinstance(payload, dict) else {}))
        raise ValueError("demo fixture is missing fields: " + ", ".join(missing))
    return payload


def _citation_reader(evidence_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for evidence in evidence_map:
        source_url = str(evidence.get("source_url") or "")
        if not source_url:
            continue
        source = grouped.setdefault(
            source_url,
            {
                "number": len(grouped) + 1,
                "source_title": evidence.get("source_title"),
                "source_url": source_url,
                "evidence": [],
            },
        )
        source["evidence"].append(
            {
                "id": evidence.get("id"),
                "atomic_claim": evidence.get("atomic_claim") or evidence.get("claim"),
                "quote": evidence.get("cleaned_quote") or evidence.get("raw_quote"),
                "stance": evidence.get("stance"),
            }
        )
    return list(grouped.values())


def run_demo(fixture: dict[str, Any]) -> dict[str, Any]:
    evidence_map = list(fixture["evidence_map"])
    selected_chunks = list(fixture["selected_chunks"])
    question_states = list(fixture["question_states"])
    validate_evidence_traceability(evidence_map, selected_chunks)

    blueprint = build_evidence_native_blueprint(
        normalized_request=dict(fixture["normalized_request"]),
        research_plan=dict(fixture["research_plan"]),
        question_states=question_states,
        evidence_map=evidence_map,
        selected_chunks=selected_chunks,
    )
    section = next(
        item
        for item in blueprint["sections"]
        if str(item.get("section_id") or "") == "sec_01"
    )
    packet, context_diagnostics = EvidenceContextManager(
        max_input_tokens=8000,
        prompt_reserve_tokens=1200,
    ).build(
        section=section,
        question_states=question_states,
        evidence_map=evidence_map,
    )

    frozen_output = dict(fixture["frozen_writer_output"])
    documents = list(frozen_output["documents"])
    report_document = dict(frozen_output["report_document"])
    citation_validation = validate_report_citations(
        documents=documents,
        report_document=report_document,
        evidence_map=evidence_map,
    )
    if citation_validation["status"] != "passed":
        raise ValueError(citation_validation["reason"])

    result = {
        "schema_version": "public-evidence-demo/v1",
        "status": "passed",
        "execution_mode": "offline_deterministic_no_model",
        "input_sha256": stable_hash(fixture),
        "section_research_manifest": section,
        "evidence_context_packet": packet,
        "context_diagnostics": context_diagnostics,
        "frozen_writer_output": frozen_output,
        "citation_validation": citation_validation,
        "citation_reader": _citation_reader(evidence_map),
        "semantic_validation": {
            "status": "not_run",
            "reason": (
                "This public demo proves deterministic provenance, context selection, and "
                "citation structure only. Model-based entailment and human content review "
                "remain separate release checks."
            ),
        },
    }
    result["output_sha256"] = stable_hash(result)
    return result


def main(argv: list[str] | None = None) -> int:
    if "backend.app" in sys.modules:
        raise RuntimeError("public Evidence demo must not initialize the Backend application")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    result = run_demo(_load_fixture(args.fixture))
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
