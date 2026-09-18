"""Atomic proposition decomposition and evidence binding."""

from __future__ import annotations

import re
from typing import Any

from .text_utils import to_simplified


def split_atomic_units(text: str, *, max_chars: int = 220) -> list[str]:
    """Split prose into independently verifiable propositions without changing wording."""
    value = str(text or "").strip()
    if not value:
        return []
    split_items = [item.strip() for item in re.split(r"(?<=[。！？!?；;])\s*", value) if item.strip()]
    primary: list[str] = []
    for item in split_items:
        if primary and re.fullmatch(r"(?:\[\[(?:[^\]]+)\]\]\s*)+", item):
            primary[-1] = f"{primary[-1]} {item}"
        else:
            primary.append(item)
    output: list[str] = []
    for item in primary:
        if len(item) <= max_chars:
            output.append(item)
            continue
        secondary = re.split(r"(?<=，)(?=(?:但|而|同时|此外|其中|因此|这|该|其|在|对于|基于|进一步))", item)
        if len(secondary) == 1:
            secondary = re.split(r"(?<=[,:：，])\s*", item)
        buffer = ""
        for clause in secondary:
            clause = clause.strip()
            if not clause:
                continue
            if buffer and len(buffer) + len(clause) > max_chars:
                output.append(buffer)
                buffer = clause
            else:
                buffer += clause
        if buffer:
            output.append(buffer)
    return output or [value]


def atomize_finding(
    finding: dict[str, Any],
    *,
    evidence: list[dict[str, Any]],
    max_units: int = 6,
) -> list[dict[str, Any]]:
    text = str(finding.get("finding") or "").strip()
    units = split_atomic_units(text)[:max_units]
    if not units:
        return []
    candidate_ids = [str(item) for item in finding.get("evidence_ids") or [] if str(item)]
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence}
    rows = []
    for unit in units:
        ids = bind_evidence_ids(unit, candidate_ids, evidence_by_id)
        rows.append({
            **finding,
            "finding": unit,
            "evidence_ids": ids,
            "atomic": True,
            "decomposition": "original" if len(units) == 1 else "deterministic_split",
        })
    return rows


def bind_evidence_ids(
    claim: str,
    candidate_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    candidates = [item for item in candidate_ids if item in evidence_by_id]
    if len(candidates) <= 1:
        return candidates
    scored = [
        (
            _text_similarity(
                claim,
                str((evidence_by_id[item].get("atomic_claim") or evidence_by_id[item].get("claim") or "")),
            ),
            item,
        )
        for item in candidates
    ]
    scored.sort(reverse=True)
    best = scored[0][0]
    if best <= 0:
        return candidates[:1]
    threshold = max(0.08, best * 0.72)
    return [item for score, item in scored if score >= threshold][:3] or [scored[0][1]]


def is_atomic_claim(text: str, *, max_chars: int = 240) -> bool:
    value = str(text or "").strip()
    return bool(value) and len(value) <= max_chars and len(split_atomic_units(value, max_chars=max_chars)) == 1


def _text_similarity(left: str, right: str) -> float:
    left_set = _shingles(left)
    right_set = _shingles(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _shingles(text: str, width: int = 2) -> set[str]:
    # Fold traditional->simplified before shingling so the same claim re-extracted in a
    # different script (or scraped from a traditional-Chinese source) still dedupes against its
    # simplified counterpart instead of being treated as two unrelated claims.
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "", to_simplified(str(text or "").lower()))
    if len(value) <= width:
        return {value} if value else set()
    return {value[index:index + width] for index in range(len(value) - width + 1)}
