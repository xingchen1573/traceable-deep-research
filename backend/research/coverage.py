"""Deterministic coverage gates for question-level research."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .evidence import is_boundary_evidence
from .models import ResearchQuestionState
from .source_policy import (
    canonical_source_identity,
    criterion_requires_source_diversity,
    criterion_source_diversity_target,
    question_requires_boundary,
)


def assess_question_coverage(state: ResearchQuestionState) -> dict[str, Any]:
    evidence = state.accepted_evidence
    clusters = {str(item.get("claim_cluster_id") or item.get("atomic_claim") or "") for item in evidence}
    clusters.discard("")
    source_identity_keys = {
        canonical_source_identity(item.get("source_url") or "")
        for item in evidence
        if canonical_source_identity(item.get("source_url") or "")
    }
    domains = {urlparse(str(item.get("source_url") or "")).netloc.lower() for item in evidence}
    domains.discard("")
    source_floor_mode = _source_floor_mode(state)
    source_diversity_required = _hard_requires_source_diversity(state)
    source_floor_count = len(domains) if source_floor_mode == "domain" else len(source_identity_keys)
    source_floor_target = _source_floor_target(state) if source_diversity_required else state.min_sources
    source_type_counts: dict[str, int] = {}
    for item in evidence:
        source_type = str(item.get("source_type") or "web_article")
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
    evidence_by_source = {}
    for item in evidence:
        identity = canonical_source_identity(item.get("source_url") or "")
        if identity:
            evidence_by_source[identity] = evidence_by_source.get(identity, 0) + 1
    cluster_utilization = {}
    for item in evidence:
        cluster = str(item.get("claim_cluster_id") or item.get("atomic_claim") or "")
        if cluster:
            cluster_utilization[cluster] = cluster_utilization.get(cluster, 0) + 1
    # Recompute from the question so resumed/old checkpoints do not preserve the
    # former question-kind-wide boundary requirement.
    requires_boundary = question_requires_boundary(state.question, state.question_kind)
    boundary_count = sum(1 for item in evidence if is_boundary_evidence(item))
    checks = {
        "evidence_floor": len(clusters) >= state.min_evidence,
        "source_floor": source_floor_count >= source_floor_target,
        "boundary_evidence": (not requires_boundary) or boundary_count >= 1,
        "grounded": all(bool(item.get("grounded_exactly")) for item in evidence),
    }
    semantic = state.diagnostics.get("semantic_coverage") if isinstance(state.diagnostics.get("semantic_coverage"), dict) else {}
    semantic_result = _semantic_assessment(state, semantic)
    if semantic:
        checks["semantic_coverage"] = semantic_result["passes"]
    weak = [name for name, passed in checks.items() if not passed]
    source_gaps = []
    for requirement in state.source_requirements:
        source_type = str(requirement.get("source_type") or "")
        target = max(1, int(requirement.get("min_count") or 1))
        current = source_type_counts.get(source_type, 0)
        if source_type and current < target:
            source_gaps.append({
                "gap_type": "source_type",
                "source_type": source_type,
                "current": current,
                "target": target,
                "blocking": False,
                "fallback_types": requirement.get("fallback_types") or [],
            })
    # Non-blocking by design (see _official_source_check docstring): a required_source_group gap
    # used to force `sufficient=False` unconditionally, with no fallback_types and no way to tell
    # "genuinely no official source exists for this topic" apart from "an official source WAS
    # fetched but normalize_source_type mis-classified it as web_article." case-source-signal-starvation-b
    # hit exactly that: Anthropic's and OpenAI's own engineering blogs were fetched and used, but
    # normalize_source_type only recognizes docs.*/developer.*/`/docs`/`/documentation` URL
    # shapes, so `anthropic.com/engineering/...` came back web_article -- and 3 of 4 questions
    # were then blocked forever (they can never reach "sufficient", only budget_exhausted) despite
    # having good, on-topic, officially-sourced evidence. There's also no real feedback loop here:
    # the gap_action this used to queue (see orchestrator.py's since-removed
    # `required_source_group` branch) was just one more generic source-typed query, no smarter
    # than the ones `source_type` gaps already queue via build_source_requirements's portfolio.
    required_source_gaps = []
    for group in state.coverage_requirements.get("required_source_groups") or []:
        normalized_group = [str(item) for item in group if str(item)]
        if normalized_group and not any(source_type_counts.get(item, 0) > 0 for item in normalized_group):
            required_source_gaps.append({
                "gap_type": "required_source_group",
                "source_types": normalized_group,
                "current": 0,
                "target": 1,
                "blocking": False,
            })
    official_source_check = _official_source_check(state, evidence, required_source_gaps)
    gaps = _hard_gaps(
        state,
        checks,
        len(clusters),
        source_floor_count,
        boundary_count,
        source_diversity_required=source_diversity_required,
        source_floor_target=source_floor_target,
    )
    gaps.extend(source_gaps)
    gaps.extend(required_source_gaps)
    semantic_gaps = semantic.get("gaps") or [
        {"missing_fact": str(aspect)} for aspect in semantic.get("weak_aspects") or []
    ]
    for semantic_gap in semantic_gaps:
        if not isinstance(semantic_gap, dict) or not str(semantic_gap.get("missing_fact") or "").strip():
            continue
        criterion_id = str(semantic_gap.get("criterion_id") or "")
        criterion = next((item for item in state.criteria if item.criterion_id == criterion_id), None)
        legacy_advisory = criterion_id in semantic_result["legacy_advisory_criterion_ids"]
        gaps.append({
            "gap_type": "semantic_coverage",
            "criterion_id": semantic_gap.get("criterion_id"),
            "gap_code": semantic_gap.get("gap_code"),
            "missing_fact": str(semantic_gap.get("missing_fact") or "").strip(),
            "preferred_source_types": semantic_gap.get("preferred_source_types") or [],
            "criterion_level": "soft" if legacy_advisory else criterion.level if criterion else "hard",
            "criterion_level_source": criterion.level_source if criterion else "unknown",
            "blocking": not criterion or (criterion.level == "hard" and not legacy_advisory),
        })
    blocking_gaps = [item for item in gaps if item.get("blocking")]
    source_mix_satisfied = not source_gaps
    safety_blocked = not checks["grounded"]
    core_sufficient = bool(evidence) and semantic_result["passes"] and not blocking_gaps and not safety_blocked
    nonblocking_gaps = [item for item in gaps if not item.get("blocking")]
    coverage_status = (
        "blocked_by_safety"
        if safety_blocked
        else "insufficient"
        if not core_sufficient
        else "sufficient_with_gaps"
        if nonblocking_gaps or weak
        else "sufficient"
    )
    return {
        "sufficient": core_sufficient,
        # Reader-facing content coverage is independent from the legacy sufficient/exhausted
        # control signal. In particular, a question can be partially answered with useful evidence
        # even though the retrieval loop has exhausted its budget, and an unjudged model call must
        # not be presented as a verified complete answer merely because evidence floors passed.
        "answer_status": semantic_result["answer_status"],
        "answered_points": semantic_result["answered_points"],
        "missing_question_points": semantic_result["missing_question_points"],
        "enrichment_gaps": semantic_result["enrichment_gaps"],
        "coverage_status": coverage_status,
        "checks": {**checks, "source_mix": not source_gaps},
        "source_mix_satisfied": source_mix_satisfied,
        "evidence_count": len(evidence),
        "claim_cluster_count": len(clusters),
        # `source_count` remains for older UI/checkpoint readers, but now means distinct
        # canonical documents for implementation/definition questions rather than bare hosts.
        "source_count": source_floor_count,
        "source_identity_count": len(source_identity_keys),
        "domain_count": len(domains),
        "source_floor_mode": source_floor_mode,
        "source_floor_target": source_floor_target,
        "source_diversity_required": source_diversity_required,
        "source_identity_keys": sorted(source_identity_keys),
        "source_domains": sorted(domains),
        "source_document_evidence_counts": dict(sorted(evidence_by_source.items())),
        "claim_cluster_evidence_counts": dict(sorted(cluster_utilization.items())),
        "source_type_counts": source_type_counts,
        "preferred_source_count": sum(source_type_counts.get(item, 0) for item in state.preferred_source_types),
        "boundary_evidence_count": boundary_count,
        "semantic_coverage_status": semantic_result["status"],
        "semantic_coverage_counts": semantic_result["counts"],
        "semantic_coverage_raw_counts": semantic_result["raw_counts"],
        "criterion_level_policy": state.criterion_level_policy,
        "criterion_level_migrations": semantic_result["legacy_level_migrations"],
        "covered_aspects": semantic.get("covered_aspects") or [criterion for criterion in state.success_criteria if _criterion_covered(criterion, evidence)],
        "weak_aspects": list(dict.fromkeys([*weak, *[f"source_type:{item['source_type']}" for item in source_gaps], *[f"required_source_group:{','.join(item['source_types'])}" for item in required_source_gaps], *[str(item) for item in semantic.get("weak_aspects") or []]])),
        "gaps": gaps,
        "official_source_check": official_source_check,
        "reason": (
            "core coverage satisfied"
            if coverage_status == "sufficient"
            else "core coverage satisfied with advisory gaps"
            if coverage_status == "sufficient_with_gaps"
            else "blocked by evidence safety checks"
            if coverage_status == "blocked_by_safety"
            else "core coverage incomplete"
        ),
    }


def _source_floor_mode(state: ResearchQuestionState) -> str:
    return "domain" if _hard_requires_source_diversity(state) else "source_identity"


def _hard_requires_source_diversity(state: ResearchQuestionState) -> bool:
    return any(
        criterion.level == "hard" and criterion_requires_source_diversity(criterion.description)
        for criterion in state.criteria
    )


def _source_floor_target(state: ResearchQuestionState) -> int:
    targets = [
        criterion_source_diversity_target(criterion.description)
        for criterion in state.criteria
        if criterion.level == "hard"
    ]
    return max([target for target in targets if target > 0] or [2])


def _semantic_assessment(state: ResearchQuestionState, semantic: dict[str, Any]) -> dict[str, Any]:
    if not semantic:
        return {
            "passes": True,
            "status": "not_judged",
            # Preserve the old stop behavior when the optional semantic model is unavailable, but
            # expose a conservative reader-facing status. Evidence exists at this point, so
            # "partial" is more honest than either "complete" or "uncovered".
            "answer_status": "partial" if state.accepted_evidence else "uncovered",
            "answered_points": [],
            "missing_question_points": [],
            "enrichment_gaps": [],
            "counts": {},
            "raw_counts": {},
            "legacy_advisory_criterion_ids": set(),
            "legacy_level_migrations": [],
        }

    criteria_by_id = {item.criterion_id: item for item in state.criteria}
    hard_ids = {item.criterion_id for item in state.criteria if item.level == "hard"}
    rows = [item for item in semantic.get("criterion_coverage") or [] if isinstance(item, dict)]
    statuses: dict[str, str] = {}
    raw_statuses: dict[str, str] = {}
    legacy_advisory_criterion_ids: set[str] = set()
    legacy_level_migrations: list[dict[str, str]] = []
    for row in rows:
        criterion_id = str(row.get("criterion_id") or "")
        status = str(row.get("status") or "").lower()
        if criterion_id in hard_ids and status in {"covered", "partial", "uncovered"}:
            raw_statuses[criterion_id] = status
            criterion = criteria_by_id[criterion_id]
            if status == "partial" and criterion.level_source == "legacy_inferred":
                # Unversioned plans had no real hard/soft distinction: every criterion string was
                # defaulted to hard, while the old judge used `partial` for answers whose subject
                # was present but an exhaustive list/example/detail was missing. Reinterpret only
                # that legacy combination as covered + an advisory gap. Explicit/planner-v1 hard
                # criteria remain partial and blocking.
                statuses[criterion_id] = "covered"
                legacy_advisory_criterion_ids.add(criterion_id)
                legacy_level_migrations.append({
                    "criterion_id": criterion_id,
                    "from_level": "legacy_inferred_hard",
                    "to_level": "soft_gap",
                    "semantic_status": "partial",
                    "reason": "legacy_unversioned_partial",
                })
                continue
            statuses[criterion_id] = status

    semantic_gaps = [item for item in semantic.get("gaps") or [] if isinstance(item, dict)]
    unknown_hard_gap = any(not str(item.get("criterion_id") or "") for item in semantic_gaps)
    if rows:
        for criterion_id in hard_ids:
            if criterion_id not in statuses:
                # A top-level model verdict cannot stand in for a missing per-criterion judgment.
                # Reuse only the state machine's existing criterion ledger; unknown rows fail
                # closed so one broad `answer_status=complete` cannot erase a requirement.
                ledger_status = str(criteria_by_id[criterion_id].status or "").lower()
                statuses[criterion_id] = (
                    ledger_status if ledger_status in {"covered", "partial", "uncovered"} else "uncovered"
                )
    else:
        # Historical semantic responses sometimes returned only a gap list. Preserve a concrete
        # state-machine ledger verdict, but never infer coverage from top-level `sufficient`.
        statuses = {}
        for criterion_id in hard_ids:
            ledger_status = str(criteria_by_id[criterion_id].status or "").lower()
            statuses[criterion_id] = (
                ledger_status if ledger_status in {"covered", "partial", "uncovered"} else "uncovered"
            )

    counts = {
        status: sum(value == status for value in statuses.values())
        for status in ("covered", "partial", "uncovered")
    }
    raw_counts = {
        status: sum(value == status for value in raw_statuses.values())
        for status in ("covered", "partial", "uncovered")
    }
    derived_answer_status = _answer_status_from_hard_criteria(
        statuses,
        has_evidence=bool(state.accepted_evidence),
        unknown_hard_gap=unknown_hard_gap,
    )
    explicit_answer_status = str(semantic.get("answer_status") or "").strip().lower()
    # The model-level label is useful diagnostics, but the hard criterion ledger is the actual
    # answer contract. This prevents a broad `complete` verdict from overwriting a concrete hard
    # dimension that the same response marked partial/uncovered. Soft/optional rows never enter
    # `statuses`, so enrichment remains non-blocking.
    answer_status = derived_answer_status
    full = answer_status == "complete"
    enrichment_gaps = [
        item for item in semantic.get("enrichment_gaps") or semantic_gaps
        if isinstance(item, dict)
        and str(item.get("criterion_id") or "") not in hard_ids
    ]
    return {
        "passes": full,
        "status": "full" if full else "insufficient",
        "answer_status": answer_status,
        "model_answer_status": explicit_answer_status,
        "answered_points": [
            str(item).strip()
            for item in semantic.get("answered_points") or semantic.get("covered_aspects") or []
            if str(item).strip()
        ],
        "missing_question_points": [
            str(item).strip()
            for item in semantic.get("missing_question_points") or []
            if str(item).strip()
        ],
        "enrichment_gaps": enrichment_gaps,
        "counts": counts,
        "raw_counts": raw_counts,
        "legacy_advisory_criterion_ids": legacy_advisory_criterion_ids,
        "legacy_level_migrations": legacy_level_migrations,
    }


def _answer_status_from_hard_criteria(
    statuses: dict[str, str],
    *,
    has_evidence: bool,
    unknown_hard_gap: bool,
) -> str:
    """Map the hard-question ledger to reader-facing complete/partial/uncovered.

    This is intentionally content-oriented rather than a delivery gate: any material hard
    dimension that is covered or partial makes the question partially answered; only a ledger
    with no supported hard dimension is uncovered. Fact-safety remains an independent axis.
    """
    values = list(statuses.values())
    if values and not unknown_hard_gap and all(value == "covered" for value in values):
        return "complete"
    if has_evidence and (unknown_hard_gap or any(value in {"covered", "partial"} for value in values)):
        return "partial"
    return "uncovered"


def should_stop(state: ResearchQuestionState, *, elapsed_seconds: float) -> tuple[bool, str]:
    coverage = assess_question_coverage(state)
    state.coverage = coverage
    if coverage["sufficient"]:
        reason = "sufficient_evidence" if coverage.get("coverage_status") == "sufficient" else "sufficient_evidence_with_advisory_gaps"
        return True, reason
    if coverage.get("coverage_status") == "blocked_by_safety":
        return True, "evidence_safety_blocked"
    if state.stagnant_rounds >= 2:
        return True, "no_novel_evidence"
    if state.rounds_used >= state.budget.max_rounds:
        return True, "budget_exhausted"
    # Context7 documents are injected out-of-band via fetch_context7_context/
    # record_context7_documents -- they bypass the per-round HTTP fetch flow
    # (_question_round_fetch_limit/_fetch_source_candidates) entirely and are already bounded
    # by their own caps (RESEARCH_CONTEXT7_MAX_LIBRARIES_PER_ROUND/_PER_QUESTION). The comment
    # at research_service.py's context7 fetch call already establishes this design intent
    # ("these candidates never enter select's ranking competition and don't count against
    # global_cap"), but that only covered the global source pool cap -- max_materials/
    # max_fetch_urls here were never updated to match. Before a multi-key Context7 pool made
    # 429s stop silently eating every call (see docs/evaluations/deep-research-revision-loop-
    # reuse-mismatch-20260811-handoff.md follow-up), context7 rarely contributed enough
    # materials for this gap to matter; a real --via-api run afterward hit it immediately: one
    # library-heavy round injected 14 context7 documents, blowing through max_materials=8/
    # max_fetch_urls=12 before round 2's targeted web-article search (which the sole
    # zero-evidence question in that run actually needed) ever got a chance to run.
    context7_urls = {
        str(item.get("url") or item.get("source_url") or "")
        for item in state.materials
        if str(item.get("provider") or "") == "context7"
    }
    if len({
        str(item.get("url") or item.get("source_url") or "")
        for item in state.materials
    } - context7_urls) >= state.budget.max_materials:
        return True, "budget_exhausted"
    if len([url for url in state.attempted_fetch_urls if url not in context7_urls]) >= state.budget.max_fetch_urls:
        return True, "budget_exhausted"
    if elapsed_seconds >= state.budget.timeout_seconds:
        return True, "budget_exhausted"
    if state.model_calls >= state.budget.max_model_calls:
        return True, "budget_exhausted"
    return False, ""


def _hard_gaps(
    state: ResearchQuestionState,
    checks: dict[str, bool],
    cluster_count: int,
    source_count: int,
    boundary_count: int,
    *,
    source_diversity_required: bool,
    source_floor_target: int,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if not checks["evidence_floor"]:
        gaps.append({
            "gap_type": "evidence_floor",
            "current": cluster_count,
            "target": state.min_evidence,
            "blocking": cluster_count == 0,
        })
    if not checks["source_floor"]:
        gaps.append({
            "gap_type": "source_floor",
            "current": source_count,
            "target": source_floor_target,
            "blocking": source_diversity_required,
        })
    if not checks["boundary_evidence"]:
        gaps.append({"gap_type": "boundary_evidence", "current": boundary_count, "target": 1, "blocking": False})
    if not checks["grounded"]:
        gaps.append({"gap_type": "grounding", "blocking": True})
    return gaps


# Small, explicit allowlist for the "was this actually an official source, just mis-typed"
# heuristic below -- not a general source-type classifier fix (that's normalize_source_type's
# job, deliberately left untouched here; see project notes on case-source-signal-starvation-b).
# TODO: extract to config / generalize once this proves useful beyond the initial vendor set.
_KNOWN_VENDOR_ENGINEERING_HOSTS = (
    "anthropic.com",
    "openai.com",
    "developers.google.com",
    "learn.microsoft.com",
)


def _official_source_check(
    state: ResearchQuestionState,
    evidence: list[dict[str, Any]],
    required_source_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Non-blocking diagnostic for `required_source_groups` -- for troubleshooting only.

    This used to be a hard gate (`required_source_group` gaps always `blocking: True`, forcing
    `sufficient=False` with no fallback_types) until case-source-signal-starvation-b showed it produces
    more false negatives than useful pressure: Anthropic's and OpenAI's own engineering blogs were
    fetched and used as evidence, but normalize_source_type only recognizes docs.*/developer.*/
    `/docs`/`/documentation` URL shapes, so a company-blog URL like `anthropic.com/engineering/...`
    comes back `web_article` -- the gate then reported "no official source" when one had, in fact,
    been recalled and used. There's also no real feedback loop to fix a genuine gap: the query this
    used to trigger was just one more generic source-typed search.

    So this only surfaces what happened, split into "genuinely no official-type evidence" vs.
    "official-looking evidence exists but was classified as something else" -- a human reviewing
    the job can use `likely_classification_artifact` to tell the two apart.
    """
    required_groups = [
        [str(item) for item in group if str(item)]
        for group in state.coverage_requirements.get("required_source_groups") or []
        if [str(item) for item in group if str(item)]
    ]
    unsatisfied_groups = [gap["source_types"] for gap in required_source_gaps]
    satisfied_groups = [group for group in required_groups if group not in unsatisfied_groups]
    likely_classification_artifact = False
    if unsatisfied_groups:
        for item in evidence:
            if str(item.get("source_type") or "") != "web_article":
                continue
            host = urlparse(str(item.get("source_url") or "")).netloc.lower()
            if any(host == domain or host.endswith(f".{domain}") for domain in _KNOWN_VENDOR_ENGINEERING_HOSTS):
                likely_classification_artifact = True
                break
    return {
        "required_groups": required_groups,
        "satisfied_groups": satisfied_groups,
        "unsatisfied_groups": unsatisfied_groups,
        "likely_classification_artifact": likely_classification_artifact,
    }


def _is_evaluative(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("风险", "限制", "优劣", "对比", "评估", "选型", "risk", "limitation", "compare"))


def _criterion_covered(criterion: str, evidence: list[dict[str, Any]]) -> bool:
    terms = [term.lower() for term in str(criterion).replace("：", " ").split() if len(term) >= 2]
    if not terms:
        return bool(evidence)
    corpus = " ".join(str(item.get("atomic_claim") or "").lower() for item in evidence)
    return any(term in corpus for term in terms)
