"""Parallel section writer and serial report editor."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .audit import audit_scope, submit_with_audit_context
from .claims import is_atomic_claim, split_atomic_units
from .evidence import evidence_support_relation, evidence_target_fit, is_direct_exact_evidence
from .synthesis import JsonModel
from .text_utils import normalize_for_substring, tokenize_for_overlap


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _semantic_writer_evidence_fields(evidence: dict[str, Any]) -> dict[str, str]:
    if not evidence.get("support_relation") or not evidence.get("target_fit"):
        return {}
    return {
        "support_relation": evidence_support_relation(evidence),
        "target_fit": evidence_target_fit(evidence),
    }


def _target_length_chars(section: dict[str, Any], findings: list[dict[str, Any]]) -> int:
    """Upfront character-count target the writer prompt should aim for.

    Computed the same way ``_section_is_underdeveloped`` used to compute it
    post-hoc, but now surfaced in the payload *before* generation so the
    model can write to length rather than relying solely on a single
    after-the-fact expansion retry.
    """
    question_count = len(section.get("research_questions") or [])
    dossier = section.get("dossier") if isinstance(section.get("dossier"), dict) else {}
    material_units = max(len(findings), len(dossier.get("evidence") or []))
    min_chars = _env_int("RESEARCH_SECTION_TARGET_CHARS_MIN", 400)
    max_chars = _env_int("RESEARCH_SECTION_TARGET_CHARS_MAX", 2200)
    per_unit = _env_int("RESEARCH_SECTION_TARGET_CHARS_PER_UNIT", 260)
    target = max(min_chars, min(max_chars, material_units * per_unit + min(question_count, 3) * 80))
    if str(section.get("section_id") or "") == "sec_00" or "总览" in str(section.get("title") or ""):
        return min(target, _env_int("RESEARCH_OVERVIEW_TARGET_CHARS_MAX", 650))
    return target



def write_report(
    *,
    user_request: str,
    blueprint: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    model: JsonModel | None = None,
    parallelism: int = 3,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map}
    finding_by_id = {str(item.get("finding_id") or ""): item for item in findings}
    citation_map = _citation_map(evidence_map)
    all_sections = blueprint.get("sections") or []
    reference_title = _reference_title_from_blueprint(blueprint)
    omitted_supplemental_section_ids = [
        str(section.get("section_id") or "")
        for section in all_sections
        if (
            str((section.get("reportability") or {}).get("status") or "") == "empty"
            and not bool((section.get("reportability") or {}).get("core_section"))
        )
        or _is_reference_metadata_section(section)
    ]
    sections = [
        section
        for section in all_sections
        if str(section.get("section_id") or "") not in omitted_supplemental_section_ids
    ]
    documents: list[dict[str, Any]] = []
    # editorial (_edit_report) does not depend on `documents` -- it only reads `findings` and
    # the citation_map/evidence_by_id computed above, both available before any section future
    # starts (see the module docstring and commit history: this used to run serially after a
    # hard sync barrier on every section future, which was a pure implementation-structure
    # choice, not a real data dependency). Submitting it into the same pool as the section
    # futures lets it run concurrently with section writing instead of waiting for the slowest
    # section; task_count includes it so the pool gets an extra worker slot when the configured
    # parallelism allows one, mirroring revise_report_sections's editorial-future handling below.
    task_count = len(sections) + 1
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, task_count or 1))) as executor:
        futures = {
            submit_with_audit_context(
                executor,
                _write_section,
                section,
                blueprint,
                finding_by_id,
                evidence_by_id,
                citation_map,
                model,
            ): index
            for index, section in enumerate(sections)
        }
        editorial_future = submit_with_audit_context(
            executor, _edit_report, user_request, blueprint, findings, evidence_by_id, citation_map, model
        )
        indexed: dict[int, dict[str, Any]] = {}
        for future in as_completed(futures):
            index = futures[future]
            document = future.result()
            indexed[index] = document
            if progress_callback:
                try:
                    progress_callback(
                        {
                            "completed": len(indexed),
                            "total": len(sections),
                            "title": str(document.get("title") or sections[index].get("title") or ""),
                        }
                    )
                except Exception:
                    pass
        documents = [indexed[index] for index in sorted(indexed)]
        editorial = editorial_future.result()
    completeness = _delivery_completeness(
        blueprint.get("completeness") or {"status": "complete_report"}, documents
    )
    research_mode = str(blueprint.get("research_mode") or "domain_practice_review")
    report_content, sanitized_evidence_id_count = _assemble_report(
        user_request,
        documents,
        editorial,
        citation_map,
        evidence_map,
        completeness=completeness,
        research_mode=research_mode,
        reference_title=reference_title,
    )
    editorial = {**editorial, "reference_title": reference_title}
    report_status = str(completeness.get("status") or "partial_report")
    report_label = _report_label(report_status == "complete_report", research_mode)
    return {
        "documents": documents,
        "report_status": report_status,
        "report_document": {
            "title": editorial.get("report_title") or f"{user_request} {report_label}",
            "content": report_content,
            "sanitized_evidence_id_count": sanitized_evidence_id_count,
            "delivery_status": report_status,
            "completeness": completeness,
            "research_mode": research_mode,
            "evidence_refs": _evidence_refs(evidence_map),
            "claim_ledger": [
                *[entry for document in documents for entry in document.get("claim_ledger") or []],
                *(editorial.get("claim_ledger") or []),
            ],
        },
        "editorial": editorial,
        "diagnostics": {
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "section_count": len(documents),
            "parallelism": parallelism,
            "model_mode": "model" if model else "deterministic",
            "report_status": report_status,
            "completeness": completeness,
            "research_mode": research_mode,
            "material_utilization": _material_utilization(documents),
            "omitted_supplemental_section_ids": omitted_supplemental_section_ids,
        },
    }


def revise_report_sections(
    *,
    user_request: str,
    written: dict[str, Any],
    blueprint: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    model: JsonModel | None,
) -> dict[str, Any]:
    """Revise only critic-targeted sections and run one final editorial pass."""
    if not model:
        return written
    targets: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        section_id = str(issue.get("section_id") or "")
        if section_id:
            targets.setdefault(section_id, []).append(issue)
    if not targets:
        return written
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map}
    finding_by_id = {str(item.get("finding_id") or ""): item for item in findings}
    citation_map = _citation_map(evidence_map)
    section_by_title = {str(item.get("title") or ""): item for item in blueprint.get("sections") or []}
    documents = list(written.get("documents") or [])

    def revise_one(index: int, document: dict[str, Any]) -> tuple[int, dict[str, Any], bool | None, list[dict[str, Any]]]:
        section = section_by_title.get(str(document.get("title") or ""))
        section_id = str((section or {}).get("section_id") or "")
        if not section or section_id not in targets:
            return index, document, None, []
        section_issues = targets[section_id]
        revision_brief = {
            **section,
            "revision_issues": section_issues,
            "existing_draft": document.get("content"),
            "existing_claim_ledger": document.get("claim_ledger") or [],
        }
        candidate = _write_section(revision_brief, blueprint, finding_by_id, evidence_by_id, citation_map, model)
        candidate, disputed_issues = _settle_issue_rejections(
            candidate, revision_brief, section_issues, blueprint, finding_by_id, evidence_by_id, citation_map, model
        )
        accepted, rejected_reason = _accept_revision(document, candidate, section_issues)
        retried = False
        if not accepted:
            # One self-correcting retry: tell the model exactly why the first attempt was
            # rejected (too short / fewer citations / fell back to template) and ask it to try
            # again, instead of silently discarding a fixable attempt. See
            # research-manual-section-revision-feature memory for why this was added — a real
            # E2E run showed two legitimate revision instructions get rejected on the first try
            # with no chance to correct course.
            retried = True
            retry_brief = {
                **revision_brief,
                "revision_issues": [
                    *section_issues,
                    {
                        "section_id": section_id,
                        "type": "revision_rejected_retry",
                        "instruction": _retry_feedback_instruction(rejected_reason, document, candidate),
                    },
                ],
            }
            retry_candidate = _write_section(retry_brief, blueprint, finding_by_id, evidence_by_id, citation_map, model)
            retry_candidate, retry_disputed_issues = _settle_issue_rejections(
                retry_candidate, retry_brief, section_issues, blueprint, finding_by_id, evidence_by_id, citation_map, model
            )
            retry_accepted, retry_reason = _accept_revision(document, retry_candidate, section_issues)
            if retry_accepted:
                accepted, rejected_reason, candidate, disputed_issues = True, "", retry_candidate, retry_disputed_issues
            else:
                rejected_reason = retry_reason
        if accepted:
            candidate["revision_diagnostics"] = {"attempted": True, "accepted": True, "retried": retried}
            return index, candidate, True, disputed_issues
        return index, {
                **document,
                "revision_diagnostics": {
                    "attempted": True,
                    "accepted": False,
                    "rejected_reason": rejected_reason,
                    "retried": retried,
                },
            }, False, disputed_issues

    editorial_issues = targets.get("editorial") or []
    parallelism = max(1, int(os.getenv("RESEARCH_REVISION_PARALLELISM", "6")))
    indexed_documents: dict[int, dict[str, Any]] = {}
    outcomes: list[bool | None] = []
    all_disputed_issues: list[dict[str, Any]] = []
    editorial = written.get("editorial") or {}
    task_count = len(documents) + (1 if editorial_issues else 0)
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, task_count or 1))) as executor:
        document_futures = {
            submit_with_audit_context(executor, revise_one, index, document): index
            for index, document in enumerate(documents)
        }
        editorial_future = (
            submit_with_audit_context(
                executor,
                _edit_report,
                user_request,
                blueprint,
                findings,
                evidence_by_id,
                citation_map,
                model,
                revision_issues=editorial_issues,
                existing_editorial=written.get("editorial") or {},
            )
            if editorial_issues
            else None
        )
        for future in as_completed(document_futures):
            index, document, outcome, disputed_issues = future.result()
            indexed_documents[index] = document
            outcomes.append(outcome)
            all_disputed_issues.extend(disputed_issues)
        if editorial_future is not None:
            editorial = editorial_future.result()
    revised_documents = [indexed_documents[index] for index in range(len(documents))]
    revised_count = sum(outcome is True for outcome in outcomes)
    revision_rejected_count = sum(outcome is False for outcome in outcomes)
    completeness = _delivery_completeness(
        blueprint.get("completeness") or {"status": "complete_report"}, revised_documents
    )
    research_mode = str(blueprint.get("research_mode") or "domain_practice_review")
    report_content, sanitized_evidence_id_count = _assemble_report(
        user_request,
        revised_documents,
        editorial,
        citation_map,
        evidence_map,
        completeness=completeness,
        research_mode=research_mode,
        reference_title=(
            str(editorial.get("reference_title") or "").strip()
            or _reference_title_from_blueprint(blueprint)
        ),
    )
    report_status = str(completeness.get("status") or "partial_report")
    report_label = _report_label(report_status == "complete_report", research_mode)
    return {
        "documents": revised_documents,
        "report_status": report_status,
        "report_document": {
            "title": editorial.get("report_title") or f"{user_request} {report_label}",
            "content": report_content,
            "sanitized_evidence_id_count": sanitized_evidence_id_count,
            "delivery_status": report_status,
            "completeness": completeness,
            "research_mode": research_mode,
            "evidence_refs": _evidence_refs(evidence_map),
            "claim_ledger": [
                *[entry for document in revised_documents for entry in document.get("claim_ledger") or []],
                *(editorial.get("claim_ledger") or []),
            ],
        },
        "editorial": editorial,
        "diagnostics": {
            **(written.get("diagnostics") or {}),
            "revision_passes": int((written.get("diagnostics") or {}).get("revision_passes") or 0) + 1,
            "revision_parallelism": parallelism,
            "revised_section_count": revised_count,
            "revision_rejected_count": revision_rejected_count,
            "editorial_revised": bool(editorial_issues),
            "material_utilization": _material_utilization(revised_documents),
            # Issues where the writer explicitly rejected a revision_issue, an independent
            # arbitration overruled that rejection, and a forced follow-up attempt still didn't
            # resolve it (see _settle_issue_rejections). Accumulated across passes (spread from
            # `written["diagnostics"]` above) so a dispute raised on pass 1 isn't lost if pass 2
            # doesn't touch that section again. research_service.py surfaces this as a blocking
            # quality-gate entry rather than silently dropping either side's position.
            "writer_disputed_issues": [
                *((written.get("diagnostics") or {}).get("writer_disputed_issues") or []),
                *all_disputed_issues,
            ],
        },
    }


def _rejected_issue_indices(
    issue_responses: list[Any], section_issues: list[dict[str, Any]]
) -> dict[int, str]:
    """Parse the writer's issue_responses (see _model_section) into {issue_index: reason} for
    every entry declared action="rejected", ignoring malformed/out-of-range entries defensively
    since this is free-form model output."""
    result: dict[int, str] = {}
    for item in issue_responses or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("issue_index"))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(section_issues):
            continue
        if str(item.get("action") or "").strip().lower() == "rejected":
            result[index] = str(item.get("reason") or "").strip()
    return result


def _arbitrate_revision_rejection(
    issue: dict[str, Any],
    rejection_reason: str,
    section_content: str,
    evidence_by_id: dict[str, dict[str, Any]],
    model: JsonModel,
) -> dict[str, Any]:
    """Independently judge whether the writer's rejection of a revision_issue is justified.

    Sees the issue itself, the writer's stated reason, the evidence the issue cited (if any --
    critic issues without evidence_ids get an empty list, since a purely structural issue like
    "table doesn't earn its place" has nothing to arbitrate against evidence-wise), and the
    section's actual current text (2026-08-12 grilling decision: giving only the abstract issue
    description without the real prose is the same blind-judgment gap this whole change starts
    from -- see quality.py's _section_critic_evidence). Returns {} on model failure/timeout,
    which the caller treats as writer_upheld=False-shaped (see _settle_issue_rejections) so an
    unreachable arbitration model degrades toward "surface the dispute for human review" rather
    than toward "silently drop it".
    """
    cited_ids = [str(value) for value in issue.get("evidence_ids") or [] if str(value)]
    payload = {
        "task": (
            "A report writer refused to act on a flagged revision issue. Decide whether the "
            "writer's rejection is justified given the section's actual current text and any "
            "cited evidence, or whether the issue stands and the writer must fix it."
        ),
        "issue": {"type": str(issue.get("type") or ""), "instruction": str(issue.get("instruction") or "")},
        "writer_rejection_reason": rejection_reason[:1000],
        "cited_evidence": [
            {
                "id": evidence_id,
                "claim": evidence_by_id[evidence_id].get("atomic_claim") or evidence_by_id[evidence_id].get("claim"),
                "quote": (
                    evidence_by_id[evidence_id].get("cleaned_quote")
                    or evidence_by_id[evidence_id].get("raw_quote")
                    or evidence_by_id[evidence_id].get("evidence_text")
                ),
                "source_url": evidence_by_id[evidence_id].get("source_url"),
                "source_title": evidence_by_id[evidence_id].get("source_title"),
                "scope": evidence_by_id[evidence_id].get("scope"),
                **_semantic_writer_evidence_fields(evidence_by_id[evidence_id]),
                "stance": evidence_by_id[evidence_id].get("stance"),
            }
            for evidence_id in cited_ids
            if evidence_id in evidence_by_id
        ],
        "section_content": section_content[:12000],
        "schema": {"writer_upheld": True, "reason": "", "required_fix": ""},
    }
    result = model(payload)
    return result if isinstance(result, dict) else {}


def _settle_issue_rejections(
    candidate: dict[str, Any],
    revision_brief: dict[str, Any],
    section_issues: list[dict[str, Any]],
    blueprint: dict[str, Any],
    finding_by_id: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
    model: JsonModel,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Arbitrate every revision_issue the writer explicitly declared "rejected".

    Two possible outcomes per rejected issue, decided by an independent arbitration call
    (`_arbitrate_revision_rejection`) rather than by trusting either side automatically:
    - The writer is upheld: the issue is simply dropped, no further action.
    - The writer is overruled: ALL such overruled issues (across this section, in one batch to
      cap the extra model calls) are sent back in a single forced follow-up `_write_section` call
      whose instruction embeds the arbitration's concrete `required_fix`. This mirrors the
      existing structural self-correcting retry in `_accept_revision`'s caller (same shape:
      explain exactly what was wrong, try once more) but is triggered by a content-substance
      rejection instead of a length/evidence-count floor violation.

    If the forced follow-up still rejects the same issue, OR (for issue types carrying a `claim`
    field, i.e. entailment issues -- see _entailment_revision_issues) the disputed claim text is
    still verbatim present in the new content, no third arbitration round runs: two rounds of
    real model judgment failing to reconcile writer and critic/entailment is itself the signal
    worth surfacing to a human, not something a third automated guess should paper over (2026-08-12
    grilling decision). Returns (final_candidate, disputed_issues) where disputed_issues is empty
    unless this escalation happened.
    """
    rejections = _rejected_issue_indices(candidate.get("issue_responses") or [], section_issues)
    if not rejections:
        return candidate, []
    arbitration_results: dict[int, dict[str, Any]] = {}
    for issue_index, reason in rejections.items():
        arbitration_results[issue_index] = _arbitrate_revision_rejection(
            section_issues[issue_index], reason, str(candidate.get("content") or ""), evidence_by_id, model
        )
    overruled_indices = [
        index for index, verdict in arbitration_results.items()
        if not (isinstance(verdict, dict) and verdict.get("writer_upheld") is True)
    ]
    if not overruled_indices:
        return candidate, []

    forced_issues = list(section_issues)
    for index in overruled_indices:
        original = forced_issues[index]
        required_fix = str((arbitration_results.get(index) or {}).get("required_fix") or "").strip()
        forced_issues[index] = {
            **original,
            "instruction": (
                f"{original.get('instruction') or ''} An independent arbitration reviewed your earlier "
                f"rejection of this issue and overruled it: {required_fix or 'the issue stands, fix it.'} "
                "You must act on this now; rejecting this specific issue again will be escalated for "
                "human review, not resolved automatically."
            ),
            "type": "revision_rejected_arbitration_overruled",
        }
    forced_brief = {
        **revision_brief,
        "revision_issues": forced_issues,
        "existing_draft": candidate.get("content"),
        "existing_claim_ledger": candidate.get("claim_ledger") or [],
    }
    forced_candidate = _write_section(forced_brief, blueprint, finding_by_id, evidence_by_id, citation_map, model)
    if not forced_candidate.get("content"):
        # Forced follow-up itself failed to produce content (model timeout/error) -- keep the
        # pre-forcing candidate rather than losing real content, but every overruled issue is
        # unresolved either way and must be surfaced.
        return candidate, [
            _disputed_issue_entry(section_issues[index], rejections[index], arbitration_results.get(index))
            for index in overruled_indices
        ]
    forced_rejections = _rejected_issue_indices(forced_candidate.get("issue_responses") or [], forced_issues)
    disputed: list[dict[str, Any]] = []
    for index in overruled_indices:
        issue = section_issues[index]
        claim_text = str(issue.get("claim") or "").strip()
        unchanged = bool(claim_text) and claim_text in str(forced_candidate.get("content") or "")
        if index in forced_rejections or unchanged:
            disputed.append(_disputed_issue_entry(issue, rejections[index], arbitration_results.get(index)))
    return forced_candidate, disputed


def _disputed_issue_entry(
    issue: dict[str, Any], writer_reason: str, arbitration_verdict: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "section_id": str(issue.get("section_id") or ""),
        "issue_type": str(issue.get("type") or ""),
        "issue_instruction": str(issue.get("instruction") or "")[:500],
        "writer_reason": str(writer_reason or "")[:500],
        "arbitration_reason": str((arbitration_verdict or {}).get("reason") or "")[:500],
    }


# A revision_issue's instruction may quote an EXAMPLE sentence to illustrate the fix it wants
# (see _CRITIC_RULES in quality.py, which since 2026-08-15 no longer hands such sentences out --
# but a critic regression or a hand-authored issue could). Anything quoted inside `如'…'` /
# `例如「…」` / `'…'` / `「…」` is illustration only: it was never checked against evidence, so
# pasting it verbatim into the returned section smuggles in a new factual/synthesis assertion that
# no cited evidence was validated against. The critic-rule fix is the治本 line of defense; this
# regex extracts the quoted sentences so a machine check (below) can refuse a revision that pasted
# one back. The pattern intentionally matches only quoted spans inside the instruction, never the
# instruction's own direction text.
_INSTRUCTION_QUOTED_SENTENCE_RE = re.compile(
    r"[如例如]+[，,]?\s*['「][^'」]{4,}['」]"
    r"|['「][^'」]{8,}['」]"
)


def _instruction_quoted_examples(instruction: str) -> list[str]:
    """Extract any ready-made example sentences quoted inside a critic/revision instruction.

    Returns the inner sentence text (without the surrounding quotes/leading 如/例如). Empty when
    the instruction only describes a direction -- the case _CRITIC_RULES now enforces -- so the
    check below no-ops on well-formed issues and only arms against the regression case.
    """
    spans: list[str] = []
    for match in _INSTRUCTION_QUOTED_SENTENCE_RE.finditer(str(instruction or "")):
        inner = match.group(0)
        # strip the leading 如/例如，and the quote pair
        inner = re.sub(r"^[如例如]+[，,]?\s*", "", inner)
        inner = re.sub(r"^['「]|['」]$", "", inner).strip()
        if len(inner) >= 4:
            spans.append(inner)
    return spans


def _reintroduces_forbidden_text(
    new: dict[str, Any],
    issues: list[dict[str, Any]] | None,
    *,
    overlap_threshold: float,
    require_evidence_intersect: bool,
) -> list[dict[str, Any]]:
    """Detect a revised section reintroducing text a prior pass already flagged as bad.

    Two independent detection targets, both fed from `issues` (the revision_issues that produced
    `new`), so no new pipeline is needed:

    1. instruction_quote -- an issue's instruction quoted a ready-made example sentence (see
       _INSTRUCTION_QUOTED_EXAMPLES) and the new content pasted it verbatim (normalized substring).
       Verbatim is deliberately strict here: the quoted span was illustration only, never checked
       against evidence, so its verbatim presence is unambiguously a copy, not a paraphrase.
    2. regenerated_claim -- an entailment_* issue carries the prior-rejected `claim` text, and a
       claim in the new claim_ledger overlaps it at >= overlap_threshold of the prior claim's
       tokens (Jaccard-style, denominator = prior claim tokens). This is the a98ba6 dead loop:
       the writer deletes the rejected sentence then rewrites the same proposition reworded, re-
       attaches the citation, marks it fixed -- and entailment rejects it again next pass.

    Returns a list of hit records (empty = clean). Each record names its `source` and the
    matched fragment so the retry feedback and disputed-issue surface can name it concretely.
    Pure / no side effects: callable from unit tests and the isolation replay harness directly.

    The overlap threshold and evidence-intersect gate are caller-supplied (not hard-coded) because
    a98ba6 also has a *legitimate* rewrite (sec_00 S3: rejected "整合为一个闭环...可复用参考设计"
    -> rewritten "紧密耦合的系统工程方案...有效性依赖于协调", which pass_3 entailment accepted). That rewrite
    overlaps the rejected claim on the same token axis as the real regenerations -- the only thing
    separating a real regeneration from a legitimate rewrite is whether entailment re-rejects it,
    which is judgment noise, not a structural gap. Any hard threshold chosen by gut therefore
    misfires on the legitimate-rewrite class; the threshold MUST be calibrated against TP/FP from
    an isolation replay before the check is enabled in production. See the isolation harness in
    scripts/replays/legacy/replay_a98ba6_regeneration.py and the plan that added this function.
    """
    hits: list[dict[str, Any]] = []
    new_content = str(new.get("content") or "")
    new_content_norm = normalize_for_substring(new_content)
    new_ledger = new.get("claim_ledger") or []

    # Target 1: verbatim paste of an instruction-quoted example sentence.
    for issue in issues or []:
        for example in _instruction_quoted_examples(str(issue.get("instruction") or "")):
            if example and normalize_for_substring(example) in new_content_norm:
                hits.append({
                    "source": "instruction_quote",
                    "section_id": str(issue.get("section_id") or ""),
                    "issue_type": str(issue.get("type") or ""),
                    "matched_fragment": example[:300],
                })

    # Target 2: semantically-equivalent regeneration of a prior-rejected entailment claim.
    for issue in issues or []:
        if not str(issue.get("type") or "").startswith("entailment_"):
            continue
        prior_claim = str(issue.get("claim") or "").strip()
        if len(prior_claim) < 4:
            continue
        prior_tokens = tokenize_for_overlap(prior_claim)
        if not prior_tokens:
            continue
        prior_ev = {str(e) for e in issue.get("evidence_ids") or []}
        for index, row in enumerate(new_ledger):
            if not isinstance(row, dict):
                continue
            new_claim = str(row.get("text") or "")
            if len(new_claim) < 4:
                continue
            new_tokens = tokenize_for_overlap(new_claim)
            if not new_tokens:
                continue
            # Denominator is the prior (rejected) claim's tokens: "how much of the bad proposition
            # reappears", not "how much of the new sentence is old". A short new claim that fully
            # restates the rejected proposition must score high even though the new sentence is
            # short; the reverse (a long new sentence that happens to share a few tokens with a
            # short rejected one) should not.
            overlap = len(prior_tokens & new_tokens) / len(prior_tokens)
            if overlap < overlap_threshold:
                continue
            if require_evidence_intersect and prior_ev:
                new_ev = {str(e) for e in row.get("evidence_ids") or []}
                if not (prior_ev & new_ev):
                    continue
            hits.append({
                "source": "regenerated_claim",
                "section_id": str(issue.get("section_id") or ""),
                "issue_type": str(issue.get("type") or ""),
                "matched_claim_index": index,
                "matched_fragment": new_claim[:300],
                "prior_claim": prior_claim[:300],
                "overlap": round(overlap, 3),
            })
    return hits


def _accept_revision(
    old: dict[str, Any], new: dict[str, Any], issues: list[dict[str, Any]] | None = None
) -> tuple[bool, str]:
    """Decide whether a revised section should replace the previously accepted one.

    A revision is a full regeneration (see _write_section), so it can silently come back
    shorter or citing less evidence than what it replaces. Reject those cases and keep the
    old document instead of letting a "fix" quietly make the section worse.

    `issues` is the list of revision_issues that produced `new`. For a manual, user-directed
    revision that explicitly asked for the section to get shorter/more concise (see
    _instruction_requests_shrink), the length-floor check below is skipped — otherwise a
    perfectly correct "make this more concise" edit gets rejected purely for doing what it was
    asked to do. The evidence-count and fallback-mode checks still apply regardless, so a
    shrink request can't be used to smuggle in a genuinely degraded rewrite.
    """
    if not str(new.get("content") or "").strip():
        return False, "empty_revision"
    grounding_correction = _issues_request_grounding_correction(issues or [])
    old_evidence_count = len(old.get("evidence_table") or [])
    new_evidence_count = len(new.get("evidence_table") or [])
    if new_evidence_count < old_evidence_count and not grounding_correction:
        return False, "fewer_evidence_used"
    old_len = len(str(old.get("content") or ""))
    new_len = len(str(new.get("content") or ""))
    if (
        old_len
        and new_len < old_len * 0.9
        and not grounding_correction
        and not _instruction_requests_shrink(issues or [])
    ):
        return False, "content_shrank"
    old_mode = (old.get("synthesis_diagnostics") or {}).get("mode")
    new_mode = (new.get("synthesis_diagnostics") or {}).get("mode")
    if new_mode in {"fallback_template", "model_failed"} and old_mode not in {"fallback_template", "model_failed"}:
        # Distinguish WHY the candidate fell back. Two cases:
        # (a) The original draft had real material (old_mode == "model_writer") but the revision
        #     model call failed/timed out and _write_section degraded to the deterministic
        #     template -- this is an operational failure on the *revision* call, not a content
        #     problem with the section. Falling back to the template here would REPLACE a real
        #     model-written section with a generic one, which is strictly worse than keeping the
        #     pre-revision draft. So in this case we don't accept the fallback, but we also
        #     don't want the caller's retry loop (revise_report_sections -> revise_one) to burn
        #     its single self-correcting retry on a template that will keep falling back as long
        #     as the model call keeps failing. The signal "keep the old draft this pass, the
        #     model wasn't available" is what `revision_model_unavailable` carries -- it lets the
        #     bounded auto-revision loop in research_service.py distinguish "this section needs
        #     another pass once the model is back" (retryable) from "the section genuinely
        #     regressed" (not retryable). See server case-revision-fallback-loop sec_01: three
        #     revision passes all hit revision_fell_back_to_template on the same section because
        #     the revision model call kept failing, and the section's idx5 unsupported claim
        #     never got fixed -- converged=False purely from this, not from the writer
        #     flip-flopping on the claim itself.
        # (b) The original draft was already a fallback_template (old_mode == "fallback_template"
        #     too), meaning the section was never model-written to begin with, so accepting
        #     another fallback is not a regression. Keep the legacy reject behavior for that case.
        if old_mode == "model_writer":
            new_fallback_reason = (new.get("synthesis_diagnostics") or {}).get("fallback_reason") or ""
            # Operational/model-failure reasons (timeout/empty/parse/model_not_configured/
            # no_findings_or_evidence) mean the writer never produced content to replace the old
            # draft with -- the right move is to keep the old draft and let the loop try again
            # next pass, not to accept a template that would erase real content.
            operational_failure = new_fallback_reason in {
                "timeout", "empty_response", "parse_failure", "empty_content",
                "model_not_configured", "no_findings_or_evidence", "unknown",
                "invalid_response_type",
            }
            if operational_failure:
                return False, "revision_model_unavailable"
        return False, "revision_fell_back_to_template"
    # Regeneration guard (2026-08-15, see plan wiggly-snacking-sifakis). A revision can pass every
    # floor above and still be a regression in disguise: the writer deleted an entailment-rejected
    # claim, then rewrote the same proposition reworded, re-attached the citation, and marked the
    # issue fixed -- so the new section is full-length, cites the same evidence, was written by the
    # model (no fallback), yet reintroduces exactly what it was told to remove. entailment catches
    # it again next pass, so the bounded auto-revision loop never converges over it (confirmed on
    # case-rejected-claim-convergence: editorial 4/4 rejected claims regenerated verbatim-equivalently
    # across all 3 passes). The existing verbatim check in _settle_issue_rejections only fires on
    # the forced-arbitration sub-path and is exact-substring, so a paraphrased reintroduction slips
    # past it. This check covers both the paraphrased claim (token overlap vs the issue's `claim`)
    # and a critic instruction's quoted example sentence pasted back verbatim. Gated OFF by default
    # because the overlap threshold is not free to set -- a98ba6 has a legitimate rewrite (sec_00
    # S3) that sits on the same token-overlap axis as the real regenerations; the threshold MUST be
    # calibrated against TP/FP from scripts/replays/legacy/replay_a98ba6_regeneration.py
    # before enabling in prod.
    if os.getenv("RESEARCH_REGENERATION_CHECK", "0") == "1":
        reintroduced = _reintroduces_forbidden_text(
            new,
            issues or [],
            overlap_threshold=float(os.getenv("RESEARCH_REGENERATION_OVERLAP", "0.6")),
            require_evidence_intersect=os.getenv("RESEARCH_REGENERATION_REQUIRE_EV_INTERSECT", "1") == "1",
        )
        if reintroduced:
            return False, "regenerated_overreach"
    return True, ""


_SHRINK_INSTRUCTION_KEYWORDS = (
    "精简", "简洁", "简短", "缩短", "删减", "去掉重复", "去重", "浓缩", "简化", "更简练", "别啰嗦", "不要啰嗦",
    "concise", "shorten", "shorter", "trim", "condense", "less verbose", "remove redundant", "de-duplicate", "dedupe",
)


def _issues_request_grounding_correction(issues: list[dict[str, Any]]) -> bool:
    """Revision issues whose whole point is to *remove* an over-reaching or unsupported claim
    (delete or qualify the offending clause, which can legitimately drop the citation it was
    over-reaching on). For these, the `_accept_revision` evidence-count floor would veto the
    fix: a writer that correctly deletes an unsupported attribution comes back citing fewer
    evidence markers, and without this bypass it gets rejected as fewer_evidence_used -- so the
    bad claim survives unchanged and the quality_gate stays needs_more_evidence.

    See case-grounding-citation-drop: idx=35/38 were label_mismatch (Osmani-cited-Steinberger,
    contrast-with-one-time-goal-as-fact) where the writer's correct fix was to rewrite the
    attribution, dropping a citation. The pre-bypass behavior rejected that fix, so the
    auto-revision loop ran but never converged -- the "stuck in waiting_approval" pattern.
    entailment_unsupported / entailment_non_atomic / entailment_label_mismatch are the three
    grounding-defect types whose fix can legitimately reduce evidence count; other issue types
    (readability, missing_tradeoffs, manual_revision) still go through the floor unchanged.

    2026-08-13 (see docs/evaluations/deep-research-critic-content-direction-20260813.md): also
    recognizes critic issues self-classified `content_direction: "reduce"` (quality.py's
    _CRITIC_RULES) -- e.g. "this table restates the prose, delete it". Before this, a critic
    structural issue got no such exemption: real job data (case-critic-direction-padding sec_03)
    showed the writer, unable to legitimately shrink the section to satisfy exactly that
    complaint, padding in two new unsupported synthesis claims instead just to keep the length/
    evidence-marker count from tripping fewer_evidence_used/content_shrank. The critic self-
    classifies the direction (it has full context on what its own issue implies) rather than
    this function guessing from issue type or instruction keywords, which this project has
    repeatedly found unreliable for semantic-level matching (see the entailment/critic
    cross-pass memory work the same week).
    """
    return any(
        str(issue.get("type") or "")
        in {"entailment_unsupported", "entailment_non_atomic", "entailment_label_mismatch"}
        or str(issue.get("content_direction") or "") == "reduce"
        for issue in issues
    )


def _instruction_requests_shrink(issues: list[dict[str, Any]]) -> bool:
    """True if a manual_revision issue's free-text instruction explicitly asks for a shorter section.

    Only looks at type == "manual_revision" (the user-directed revision entry point) — automated
    quality-gate issues (entailment_label_mismatch, unsupported_claim, etc.) are structured,
    code-generated instructions that never ask for shrinkage, so this never relaxes the guard
    for those.
    """
    for issue in issues:
        if str(issue.get("type") or "") != "manual_revision":
            continue
        instruction = str(issue.get("instruction") or "")
        if any(keyword in instruction for keyword in _SHRINK_INSTRUCTION_KEYWORDS):
            return True
    return False


def _retry_feedback_instruction(reason: str, old: dict[str, Any], candidate: dict[str, Any]) -> str:
    old_len = len(str(old.get("content") or ""))
    new_len = len(str(candidate.get("content") or ""))
    old_evidence_count = len(old.get("evidence_table") or [])
    new_evidence_count = len(candidate.get("evidence_table") or [])
    if reason == "content_shrank":
        return (
            f"Your previous attempt was rejected: at {new_len} characters it was too much shorter than "
            f"the original ({old_len} characters), more than this change justifies. Keep every point from "
            "the original that the requested change did not ask you to remove; rephrase or reorganize "
            "instead of cutting content."
        )
    if reason == "fewer_evidence_used":
        return (
            f"Your previous attempt was rejected: it cited {new_evidence_count} distinct evidence markers "
            f"([[ev_001]]-style) versus {old_evidence_count} in the original. Do not drop an existing "
            "citation marker when merging or rephrasing sentences — every fact and its citation from the "
            "original must still appear somewhere in your revision unless the requested change explicitly "
            "says to remove that specific content."
        )
    if reason == "revision_fell_back_to_template":
        return (
            "Your previous attempt failed to produce real analytical prose and fell back to a template. "
            "Write full connected analytical prose that directly addresses the requested change."
        )
    if reason == "revision_model_unavailable":
        # The previous revision attempt couldn't even reach the model (timeout / empty / parse
        # failure), so there's nothing for the writer to "correct" content-wise -- the next
        # attempt should just try again, not be told it wrote bad prose. Keep the feedback short
        # and neutral so the retry doesn't get steered toward over-correcting non-existent
        # content problems.
        return "Your previous attempt did not return usable content (the model call failed or timed out). Try again, producing full connected analytical prose that addresses the requested change."
    if reason == "regenerated_overreach":
        # The previous attempt reintroduced a proposition a prior pass already rejected (or pasted
        # a sentence quoted inside an issue instruction). Telling the model to "rewrite it
        # differently" backfires for the same reason listing hedge-vocabulary did (08-05 layer F,
        # 08-09 reason variant): it shows the model another way to phrase the same proposition.
        # The only reliable instruction is to DELETE the proposition outright and not replace it
        # with a reworded equivalent -- let the section be shorter if needed (the content_shrank
        # floor below stays bypassed for grounding_correction issues, which these are).
        return (
            "Your previous attempt was rejected: it reintroduced a claim a prior validation pass "
            "already flagged as unsupported/label_mismatch (or pasted a sentence quoted inside a "
            "revision_issue instruction). Rewording the same proposition to dodge the flag does not "
            "fix it -- the proposition itself is what the evidence does not support. Remove that "
            "proposition entirely; do not replace it with a paraphrase, a more qualified version, or "
            "an equivalent synthesis. If the section then lacks a transition, write pure connective "
            "narration carrying no citation marker, or leave the gap -- a shorter section is preferred "
            "over reinserting the rejected point."
        )
    return "Your previous attempt was rejected for returning empty or invalid content. Return non-empty, substantive Markdown content in the `content` field."


def _write_section(
    section: dict[str, Any],
    blueprint: dict[str, Any],
    finding_by_id: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
    model: JsonModel | None,
) -> dict[str, Any]:
    section_findings = [finding_by_id[item] for item in section.get("finding_ids") or [] if item in finding_by_id]
    if not section_findings and "总览" in str(section.get("title") or ""):
        section_findings = list(finding_by_id.values())[:6]
    dossier = section.get("dossier") if isinstance(section.get("dossier"), dict) else {}
    internal = ""
    model_meta: dict[str, Any] = {"attempts": 0, "failure_reason": None}
    evidence_gap = _is_legal_evidence_gap(section, section_findings)
    if model and section_findings:
        with audit_scope(
            stage="section_writer",
            section_id=str(section.get("section_id") or "") or None,
        ):
            internal, model_meta = _model_section(section, blueprint, section_findings, evidence_by_id, model)
    elif evidence_gap:
        internal = _render_evidence_gap(section)
    elif not model:
        model_meta["failure_reason"] = "model_not_configured"
    else:
        model_meta["failure_reason"] = (
            "findings_missing_for_available_evidence"
            if section.get("evidence_ids") or dossier.get("evidence")
            else "no_findings_or_evidence"
        )
    mode = "evidence_gap" if evidence_gap else "model_writer"
    fallback_reason = None
    if not internal:
        fallback_reason = model_meta.get("failure_reason") or "unknown"
        mode = "model_failed"
    rendered, ledger = _render_internal_citations(
        internal,
        evidence_by_id,
        citation_map,
        finding_by_id=finding_by_id,
    )
    ledger = _annotate_claim_ledger(ledger, section_findings, evidence_by_id, str(section.get("section_id") or ""))
    used_ids = list(dict.fromkeys(item for row in ledger for item in row.get("evidence_ids") or []))
    dossier_evidence_ids = [
        str(item.get("id") or "")
        for item in dossier.get("evidence") or []
        if str(item.get("id") or "")
    ]
    finding_ids_used = [
        str(finding.get("finding_id") or "")
        for finding in section_findings
        if set(str(item) for item in finding.get("evidence_ids") or []) & set(used_ids)
        and str(finding.get("finding_id") or "")
    ]
    evidence_table = [_evidence_table_row(evidence_by_id[item]) for item in used_ids if item in evidence_by_id]
    result = {
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
            "mode": mode,
            "fallback_reason": fallback_reason,
            "gap_reason": "no_reportable_findings_or_evidence" if evidence_gap else None,
            "gap_question_ids": _evidence_gap_question_ids(section) if evidence_gap else [],
            "gap_criterion_ids": _evidence_gap_criterion_ids(section) if evidence_gap else [],
            "known_gaps": [str(item) for item in section.get("known_gaps") or []] if evidence_gap else [],
            "model_attempts": model_meta.get("attempts", 0),
            "finding_count": len(section_findings),
            "evidence_count": len(evidence_table),
            "reportability": section.get("reportability") or {},
            "dossier_evidence_count": len(dossier_evidence_ids),
            "dossier_chunk_count": len(dossier.get("relevant_chunks") or []),
            "dossier_context_chars": int(dossier.get("context_chars") or 0),
            "unused_evidence_before_write": len(dossier.get("unused_evidence_ids") or []),
            "used_evidence_count": len(used_ids),
            "evidence_utilization_ratio": round(len(set(used_ids) & set(dossier_evidence_ids)) / len(dossier_evidence_ids), 3) if dossier_evidence_ids else 1.0,
            "input_finding_count": len(section_findings),
            "used_finding_count": len(finding_ids_used),
        },
    }
    if model_meta.get("revision_rationale"):
        # See RESEARCH_DEBUG_REVISION_TRACE in _model_section -- only ever populated when that
        # debug flag requested the field; harmless if a downstream reader ignores it.
        result["revision_rationale"] = model_meta["revision_rationale"]
    if model_meta.get("issue_responses"):
        result["issue_responses"] = model_meta["issue_responses"]
    return result


def _is_legal_evidence_gap(
    section: dict[str, Any], section_findings: list[dict[str, Any]]
) -> bool:
    """True when upstream explicitly established an empty, evidence-free section.

    This is deliberately narrower than ``not section_findings``. Evidence attached without a
    synthesized Finding is an internal pipeline invariant break and must remain a fatal writer
    failure; only the blueprint's explicit ``reportability=empty`` decision may become a
    reader-visible evidence-gap statement.
    """
    reportability = section.get("reportability") if isinstance(section.get("reportability"), dict) else {}
    dossier = section.get("dossier") if isinstance(section.get("dossier"), dict) else {}
    return (
        str(reportability.get("status") or "") == "empty"
        and not section_findings
        and not (section.get("finding_ids") or [])
        and not (section.get("evidence_ids") or [])
        and not (dossier.get("evidence") or [])
        and int(reportability.get("finding_count") or 0) == 0
        and int(reportability.get("evidence_count") or 0) == 0
    )


def _evidence_gap_question_ids(section: dict[str, Any]) -> list[str]:
    return [
        str(question.get("question_id") or "")
        for question in section.get("research_questions") or []
        if isinstance(question, dict) and str(question.get("question_id") or "")
    ]


def _evidence_gap_criterion_ids(section: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(criterion.get("criterion_id") or "")
        for question in section.get("research_questions") or []
        if isinstance(question, dict)
        for criterion in question.get("criteria") or []
        if isinstance(criterion, dict)
        and str(criterion.get("criterion_id") or "")
        and str(criterion.get("status") or "uncovered") != "covered"
    ))


def _render_evidence_gap(section: dict[str, Any]) -> str:
    """Render this run's evidence state without inventing domain facts."""
    questions = [
        str(question.get("question") or "").strip()
        for question in section.get("research_questions") or []
        if isinstance(question, dict) and str(question.get("question") or "").strip()
    ]
    criteria = list(dict.fromkeys(
        str(criterion.get("description") or criterion.get("criterion_id") or "").strip()
        for question in section.get("research_questions") or []
        if isinstance(question, dict)
        for criterion in question.get("criteria") or []
        if isinstance(criterion, dict)
        and str(criterion.get("status") or "uncovered") != "covered"
        and str(criterion.get("description") or criterion.get("criterion_id") or "").strip()
    ))
    lines = [
        "本轮检索未形成能够支持本节事实性结论的合格证据。因此本节仅记录证据缺口，"
        "不据此推断相反结论，也不补写未经证据支持的比较。",
    ]
    missing = criteria or questions
    if missing:
        lines.extend(["", "### 尚未覆盖", ""])
        lines.extend(f"- {item}" for item in missing)
    return "\n".join(lines)


def _model_section(
    section: dict[str, Any],
    blueprint: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    model: JsonModel,
) -> tuple[str, dict[str, Any]]:
    dossier = section.get("dossier") if isinstance(section.get("dossier"), dict) else {}
    allowed_ids = list(dict.fromkeys([
        *[item for finding in findings for item in [*(finding.get("evidence_ids") or []), *(finding.get("counter_evidence_ids") or [])]],
    ]))
    research_mode = str(blueprint.get("research_mode") or section.get("writing_strategy") or "domain_practice_review")
    mode_rules = []
    if research_mode in {"hybrid_practice_and_reference", "evidence_informed_reference_design"}:
        mode_rules = [
            "Treat direct domain cases as observed practice and adjacent sources as transferable engineering evidence.",
            "For transferable evidence, explain the mechanism, the target-domain adaptation, and the implementation boundary.",
            "Use recommendation language for a reference design; never imply that the target-domain deployment already exists.",
            "Consolidate shared evidence limitations into one boundary discussion instead of repeating the same disclaimer after every sentence.",
        ]
    revision_issues = section.get("revision_issues") or []
    grounding_revision = _issues_request_grounding_correction(revision_issues) or any(
        str(issue.get("type") or "") == "entailment_label_mismatch"
        for issue in revision_issues
    )
    revision_rules = []
    if revision_issues:
        revision_rules = [
            "This is a targeted revision, not a fresh draft: section_brief.existing_draft is the previously accepted version and section_brief.revision_issues lists the specific problems to fix.",
            "existing_draft contains delivered numeric citations such as [1], while section_brief.existing_claim_ledger maps each old claim to its internal evidence_ids. Never copy numeric citations into the response; re-emit every preserved claim with [[ev_001]]-style markers from existing_claim_ledger.",
            "Preserve every sentence from existing_draft that revision_issues does not flag as a problem; do not rewrite the whole section from scratch.",
            "Address every item in revision_issues with the minimum necessary change, then return the full revised section body.",
        ]
        # These two branches used to both apply unconditionally and directly contradicted each
        # other: "never end up shorter" vs "only delete, never add new content" for the exact
        # same grounding-fix case. Faced with both, the model reliably resolved it by deleting
        # the flagged clause and padding back in new material to hold the length/citation
        # count -- and the new material was itself frequently a fresh over-reach, so revision
        # never converged (see case-source-metadata-and-revision-growth: claim_count grew 38->46->56 across
        # 3 passes, each pass fixing its flagged claims but introducing 6-13 new unsupported/
        # label_mismatch ones). `grounding_revision` already drives the matching bypass in
        # `_accept_revision`'s length/evidence-count floor below -- keep the prompt consistent
        # with what the acceptance gate will actually allow instead of fighting it.
        if grounding_revision:
            revision_rules.append(
                "Do not add new claims, examples, comparison dimensions, tables, headings, or background while fixing an issue that calls for removal or reduction (an unsupported/over-reaching clause, or a critic issue explicitly marked content_direction=\"reduce\", e.g. redundant content). Only delete or minimally rewrite the rejected material -- a shorter section citing fewer evidence markers is correct and expected once it is removed."
            )
        else:
            revision_rules.append(
                "Do not let a revision end up shorter or cite fewer evidence markers than existing_draft unless a revision_issue specifically requires removing content."
            )
        # 2026-08-15 (case-critic-example-copy sec_00, pass_1, isolated A/B in
        # scripts/replays/legacy/replay_a98ba6_example_copy.py): a critic `structure` issue
        # (content_direction="add")
        # handed the writer a ready-made example sentence in its instruction, and the writer pasted
        # that sentence verbatim as a new claim, attached citations, and marked the issue "fixed"
        # -- adding an unsupported synthesis entailment later flagged. The isolation test showed
        # this was driven by the critic QUOTING a finished sentence (root cause fixed on the critic
        # side in quality.py's _CRITIC_RULES -- critic no longer supplies candidate sentences).
        # This writer-side guard remains as the matching half: any sentence that appears quoted
        # inside a revision_issue instruction is an illustration, never a claim to paste. Write
        # your own wording that states only what cited evidence supports or is pure connective
        # narration carrying no citation marker; do not copy a quoted instruction example into the
        # returned section. (An explicit anti-copy prompt rule is too soft to fully override
        # "the critic handed me a sentence" -- 2/3 stronger-prompt runs still copied -- but with
        # the critic no longer quoting, the trigger is gone. If a future critic regression re-adds
        # quoted examples, this rule is the second line; a machine check rejecting a revision that
        # pastes instruction-quoted text is the third, not yet built.)
        revision_rules.append(
            "A revision_issue's instruction may quote an EXAMPLE sentence to illustrate the kind of fix it wants. Any sentence quoted inside a revision_issue instruction is ILLUSTRATION ONLY -- never paste it into your returned section. Pasting it adds a new factual/synthesis assertion that no cited evidence was checked against. When an issue asks you to add or reword, write your OWN wording that states only what the cited evidence supports, or is pure connective/structural narration carrying no citation marker. If the only way to satisfy the issue would assert content the evidence does not support, return action=\"rejected\" for that issue with a concrete reason."
        )
    # NOTE: a per-issue-type label_mismatch hint was added here on 2026-08-05 (layer F) and
    # rolled back the same day. It told the writer to "downgrade fact to inference using
    # qualifiers like 可推断/表明/这意味着" -- which on real job case-inference-qualifier-overreach
    # correlated with MORE inference-typed over-reaches surviving into the draft (3 unsupported
    # vs 0 on the layer-D run before it). The hint was scoped to revision_issues, but the
    # qualifier vocabulary it pushed ("可推断/表明/这意味着") is exactly the shape that, used
    # loosely, lets a writer rubber-stamp an over-reach as an "inference" and ship it. The
    # problem it was solving -- a fact-typed claim that the writer kept rewriting as a fact --
    # was real but narrow (one idx3 claim on one job); the side effect -- nudging the writer
    # toward qualifier-heavy inference prose across the whole report -- was broader and worse.
    # Keep this rollback point so a future attempt at the same problem can be A/B tested
    # against the bare revision_rules above, not introduced as a blanket rule. See memory
    # project_quality_gate_failure_no_auto_revision_loop_20260805.md (the 855a92 entry).
    #
    # Variant B (2026-08-05 A/B test against the layer-D baseline) was added here as an 8th
    # rule on 2026-08-05 and rolled back the same day. It pushed "a cited claim's content
    # must be findable in the cited evidence; drop the marker or trim the over-reach". On
    # real job case-composite-claim-atomicity it tied the layer-D baseline with no improvement
    # (4 blockers, converged=False, same as 0530f5 baseline). Rolled back to restore the
    # 7-rule layer-D baseline so fixture-based single-point prompt testing can proceed
    # without the dead weight of a variant that didn't help. See same memory (the 6b6146
    # entry) and the replay script scripts/replays/replay_writer_fixture.py.
    target_length = _target_length_chars(section, findings)
    if str(section.get("section_id") or "") == "sec_00" or "总览" in str(section.get("title") or ""):
        mode_rules.append(
            f"This is the concise overview. Keep the returned body at or below about {target_length} Chinese characters, state only the report-level conclusions and boundaries, and leave mechanisms, commands, examples, and component inventories to their assigned body sections."
        )
    payload = {
        "task": "Write one analytical report section in Chinese. Return JSON only.",
        "section_role": (
            "overview_synthesis"
            if str(section.get("section_id") or "") == "sec_00" or "总览" in str(section.get("title") or "")
            else "section_specific_analysis"
        ),
        "section_brief": {key: value for key, value in section.items() if key != "dossier"},
        "section_dossier": {
            key: value
            for key, value in dossier.items()
            if key not in {
                "findings",
                "evidence",
                "unused_evidence_ids",
                "source_notes",
                "relevant_chunks",
            }
        },
        "report_outline": [
            {
                "section_id": item.get("section_id"),
                "title": item.get("title"),
                "purpose": item.get("purpose"),
            }
            for item in blueprint.get("sections") or []
        ],
        "report_thesis": blueprint.get("thesis"),
        "research_mode": research_mode,
        "findings": findings,
        "target_length_chars": target_length,
        "evidence": [{
            "id": item,
            "claim": (evidence_by_id.get(item) or {}).get("atomic_claim"),
            "quote": (evidence_by_id.get(item) or {}).get("cleaned_quote") or (evidence_by_id.get(item) or {}).get("raw_quote"),
            "stance": (evidence_by_id.get(item) or {}).get("stance"),
            "scope": (evidence_by_id.get(item) or {}).get("scope"),
            **_semantic_writer_evidence_fields(evidence_by_id.get(item) or {}),
            "source_type": (evidence_by_id.get(item) or {}).get("source_type"),
            "publisher_relation": (evidence_by_id.get(item) or {}).get("publisher_relation"),
            "source_title": (evidence_by_id.get(item) or {}).get("source_title"),
            "source_url": (evidence_by_id.get(item) or {}).get("source_url"),
            "subject": (evidence_by_id.get(item) or {}).get("subject"),
            "subject_relation": (evidence_by_id.get(item) or {}).get("subject_relation"),
            "criterion_ids": [str(value) for value in (evidence_by_id.get(item) or {}).get("criterion_ids") or []],
            "criterion_binding_status": (evidence_by_id.get(item) or {}).get("criterion_binding_status"),
            "comparison_axis": (evidence_by_id.get(item) or {}).get("comparison_axis"),
            "comparison_metric": (evidence_by_id.get(item) or {}).get("comparison_metric"),
            "expected_comparison_axis": (evidence_by_id.get(item) or {}).get("expected_comparison_axis"),
            "comparison_axis_binding_status": (evidence_by_id.get(item) or {}).get("comparison_axis_binding_status"),
        } for item in allowed_ids],
        "rules": [
            # Condensed from 32 free-form rules to a structured <hard_constraints> +
            # <writing_style> block (2026-08-10, see docs/evaluations/writer-overreach-
            # treatment-20260810.md). The 32-rule baseline let the writer over-reach
            # (cited claims containing content the evidence didn't say) at median 2.0
            # unsupported/section on an isolated 20-run replay of case1 sec_03; this
            # condensed block dropped it to median 1.0, +anchor to 0.0. The cuts:
            #   - removed the #8 "use qualifier words 推断/可能/这意味着" reverse-nudge
            #     (08-05 layer F proved listing hedge vocab INCREASES inference over-reach)
            #   - merged the #5/#6 paragraph-vs-per-clause-citation tension into one rule
            #   - merged the #21-24 dossier "use excerpts" vs "don't invent from excerpts"
            #     self-contradiction into one clear rule
            #   - XML-tagged sections so the model can tell hard checkable constraints
            #     from soft style guidance (Anthropic guidance: structure prompts with XML)
            "<hard_constraints>",
            "1. 只用 payload 里 supplied 的 findings 和 evidence 作为外部可核查事实的来源，不编造证据原文没有的内容（因果、目的、对比对象、方法论、自我审查性判断）。",
            "2. 来源明确陈述的外部事实、数字、主体属性和来源归因必须分别带自己的引用标记 [[ev_XXX]]，标记紧跟 Evidence 真正支持的最小事实子句。可以在引用事实之后另起明确分开的解释、综合或推断，这类未引用分析不强制挂引用，但不得引入新的外部事实，也不得把引用放在分析之后让 Evidence 看似支持整个推断。",
            "3. 段落内的引用标记必须真实支撑对应 cited span 的完整内容；不要为了满足引用数量而给分析挂一条仅部分相关的 Evidence。如果被写成来源事实的子句找不到支持，删掉或收窄该事实，不要硬挂引用。Before returning, audit every inference and recommendation against its finding_type; keep analysis clearly separate from cited facts and do not add facts merely to change its tone.",
            "4. 跨领域/单源证据不能直接写成目标领域的事实；保留证据原本的适用范围（如\"在编译器优化场景中……\"），不要泛化成全系统通用的断言。",
            "5. section_dossier 的 source_notes/excerpts 只能作上下文理解，不能当成可以无引用断言事实的许可；从中来的任何事实细节都要挂对应同源 ev 标记。",
            "6. 不要对证据本身加元评价性判断（如\"值得进一步验证\"\"具有启发性\"），除非 supplied 的 finding/evidence 真的说了这个判断；加个限定词不等于有支撑。",
            "6a. 优先完整回答 section_brief.research_questions 中 level=hard 的标准。soft/optional 缺口不需要用推测性内容补齐；证据不足的非核心细节直接省略，确有阅读价值但仍不确定时只保留为明确待核实信息。",
            "6b. 分别核对主体、问题/比较维度和来源角色：证据写的是谁，就只能支持谁；相邻指标不能替代当前维度；厂商或项目方自评只能归因给该来源，不能写成独立横向结论。",
            "6c. finding.answer_role=context_only 或 evidence.criterion_ids 为空时，只能作为明确区分维度后的背景信息，不能满足 hard/soft criterion，也不能进入排序、优劣或推荐结论。",
            "</hard_constraints>",
            "",
            "<writing_style>",
            "7. 中文写作；仅产品名和代码标识符保留原文。开头 1-2 句先给本节核心结论，再展开支撑分析。",
            "8. 写连贯分析散文：用自然连接词（因此/但/相比之下/进一步/与此同时）把 2-4 个相关命题组织成一段，不要一句一事地孤立罗列；权衡和边界融入论证，不要单贴一段免责。",
            "9. 引用用独立的内部标记，如 [[ev_001]][[ev_002]]；不要写成 [[ev_001], [ev_002]] 或在同一双括号内列出多个 ID。不要包含原始 quote 全文、Sources 章节、证据索引或审计旁白。不要写章节标题。",
            "10. 仅当内容确实适合多维对比/多步序列时才加 Markdown 表格，并给一句导语、表头贴合本节实际讨论、每个事实单元格照正文方式引用；短列表或单一对比已用散文说清的不要加表。",
            f"11. 章节目标长度约 {target_length} 字实质性分析，但 Evidence sufficiency overrides length: return a shorter section rather than padding with plausible but unsupported background, examples, comparisons, or implications.",
            "12. 把 research_questions 当覆盖地图，不当作固定标题模板。仅当能区分两三个真正不同的主题时才加三级小标题。",
            "13. 不要把 hard/soft/optional、coverage、criterion 等内部标签写进正文；它们只用于决定内容优先级，文章仍应自然组织。",
            "14. 严格遵守当前 section_brief 的章节职责，并参考 report_outline 避免越界展开：总览只综合各节结论，不完整复述架构、功能和使用细节；正文直接进入本节独有内容，不用重复报告总论作为开场。一个跨章节核心判断最多在总览概括一次、在最相关正文展开一次。",
            "15. 当材料只说明本次检索没有找到某项信息时，它只能作为检索范围说明，不能写成该信息在现实中不存在；这类补充缺口集中写一次，不要在多个章节反复出现。",
            "</writing_style>",
            *mode_rules,
            *revision_rules,
        ],
        "schema": {"content": "Substantive Markdown section body without the section title"},
    }
    if revision_issues:
        # Production mechanism (2026-08-12, not the debug-only revision_rationale below): every
        # revision_issue must get an explicit fixed/rejected response instead of the writer
        # silently doing whatever it judges best. "rejected" is a legitimate answer, not a
        # failure -- a real critic/entailment issue can itself be wrong (see quality.py's
        # evidence self-check, which exists for the same reason on the critic's side). Rejections
        # go through independent arbitration in revise_report_sections/_settle_issue_rejections
        # rather than being silently honored or silently overridden. See docs/evaluations/
        # deep-research-critic-evidence-grounding-20260812.md.
        payload["schema"]["issue_responses"] = [{"issue_index": 0, "action": "fixed", "reason": ""}]
        payload["rules"].append(
            "For every entry in section_brief.revision_issues (0-indexed in the order given there), return exactly one "
            "issue_responses entry: {issue_index, action: \"fixed\"|\"rejected\", reason}. "
            "action=\"fixed\" means you addressed it in the content you are returning; give a short reason. "
            "action=\"rejected\" means you deliberately left it unaddressed because you have a substantive, specific reason "
            "the issue does not actually apply to this text or evidence -- state that reason concretely (not \"disagree\"); "
            "you may leave the disputed content unchanged when rejecting. Do not mark an issue \"fixed\" if you did not "
            "actually change or verify the specific text it points to. Every issue must get a response; do not omit or add extra ones."
        )
    if revision_issues and os.getenv("RESEARCH_DEBUG_REVISION_TRACE") == "1":
        # Debugging aid only (2026-08-11, see docs/evaluations/deep-research-revision-loop-
        # critic-writer-transparency-20260811-handoff.md): the writer is never normally asked why
        # it changed what it changed, so a persistent "same issue type keeps recurring across
        # passes" pattern (evidence_quality/limitations/structure at 54-73% recurrence even when
        # correctly routed, per a 10-job audit) has no direct evidence of *why*. This optional
        # field never gates acceptance/routing and is dropped entirely when the env var is unset.
        payload["schema"]["revision_rationale"] = (
            "Optional. One short sentence per revision_issue you addressed: what you changed and why."
        )
    allowed_evidence_by_id = {item: evidence_by_id[item] for item in allowed_ids if item in evidence_by_id}
    if any(_semantic_writer_evidence_fields(evidence) for evidence in allowed_evidence_by_id.values()):
        payload["rules"].insert(
            1,
            "Treat support_relation and target_fit as the primary evidence-use contract: direct_support + exact_target may support a target fact; inference_support may support analysis but not be promoted to a source-stated fact; context_only is background only.",
        )
    allowed_finding_ids = {
        str(item.get("finding_id") or "")
        for item in findings
        if str(item.get("finding_id") or "")
    }
    content, meta = _call_writer_model(
        model,
        payload,
        allowed_evidence_by_id,
        allowed_finding_ids=allowed_finding_ids,
    )
    max_retries = _env_int("RESEARCH_WRITER_MAX_RETRIES", 3)
    retries_used = 0
    active_payload = payload
    marker_repair_attempted = False
    while not content and retries_used < max_retries:
        if meta["failure_reason"] == "evidence_validation_failed":
            if marker_repair_attempted:
                break
            marker_repair_attempted = True
            active_payload = {
                **payload,
                "task": "Repair one analytical report section whose citation markers were invalid. Return JSON only.",
                "invalid_draft": str(meta.get("invalid_content") or ""),
                "invalid_markers": list(meta.get("invalid_markers") or []),
                "rules": [
                    *payload["rules"],
                    "The prior draft used markers not present in this payload. Rewrite only as needed so every marker is one of the supplied findings or evidence ids. Do not replace an invalid marker with a merely similar id: remove the unsupported clause or cite an actually supporting supplied item.",
                ],
            }
        content, retry_meta = _call_writer_model(
            model,
            active_payload,
            allowed_evidence_by_id,
            allowed_finding_ids=allowed_finding_ids,
        )
        meta["attempts"] += 1
        meta["failure_reason"] = None if content else retry_meta["failure_reason"]
        meta["invalid_content"] = retry_meta.get("invalid_content")
        meta["invalid_markers"] = retry_meta.get("invalid_markers") or []
        if content and retry_meta.get("revision_rationale"):
            meta["revision_rationale"] = retry_meta["revision_rationale"]
        if content and retry_meta.get("issue_responses"):
            meta["issue_responses"] = retry_meta["issue_responses"]
        retries_used += 1
    meta.pop("invalid_content", None)
    if not content:
        return "", meta
    if not revision_issues and _section_is_underdeveloped(content, section, findings):
        expanded_payload = {
            **payload,
            "task": "Write one analytical report section in Chinese by expanding an underdeveloped draft. Return JSON only.",
            "existing_draft": content,
            "rules": [
                *payload["rules"],
                "Retain every supported and criterion-compatible point from the existing draft while improving depth and transitions; remove points that conflict with the current subject, comparison axis, or source role even if they appeared in the earlier draft.",
                "Cover distinct research questions that have supporting findings; do not discuss unsupported questions as facts.",
                "Expansion must come from supplied findings, their limitations, source differences, and stated uncertainty only.",
            ],
        }
        expanded, expand_meta = _call_writer_model(
            model,
            expanded_payload,
            allowed_evidence_by_id,
            allowed_finding_ids=allowed_finding_ids,
        )
        meta["attempts"] += 1
        if expanded and len(expanded) > len(content):
            content = expanded
    return content, meta


def _call_writer_model(
    model: JsonModel,
    payload: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    *,
    allowed_finding_ids: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Invoke the writer model once and classify why it produced nothing, if it did.

    Returns ``(content, meta)`` where ``meta["failure_reason"]`` is one of the reasons
    surfaced by the model callable (e.g. "timeout", "empty_response", "parse_failure"),
    ``"empty_content"`` when the model returned valid JSON with no content, or
    ``"evidence_validation_failed"`` when the model cited an unknown evidence id.
    """
    result = model(payload)
    meta = {"attempts": 1, "failure_reason": None}
    failure_reason = result.get("_failure_reason") if isinstance(result, dict) else None
    if failure_reason:
        meta["failure_reason"] = str(failure_reason)
        return "", meta
    # Debug-only (RESEARCH_DEBUG_REVISION_TRACE, see _model_section): when the schema requested
    # a `revision_rationale` field, carry whatever the model returned through to the caller so it
    # can be inspected in the revision-pass snapshot. Never required, never affects control flow.
    if isinstance(result, dict) and result.get("revision_rationale"):
        meta["revision_rationale"] = str(result["revision_rationale"])[:2000]
    # Production (see _model_section): when the schema requested issue_responses, carry the raw
    # list through so the caller (revise_report_sections/_settle_issue_rejections) can act on any
    # "rejected" entries. Left unvalidated here -- the caller tolerates malformed entries.
    if isinstance(result, dict) and isinstance(result.get("issue_responses"), list):
        meta["issue_responses"] = result["issue_responses"]
    content = str((result or {}).get("content") or "").strip()
    if not content:
        meta["failure_reason"] = "empty_content"
        return "", meta
    if not _content_uses_only_allowed_markers(
        content,
        set(evidence_by_id),
        allowed_finding_ids or set(),
    ):
        meta["failure_reason"] = "evidence_validation_failed"
        used_evidence = set(re.findall(r"\bev_\d{3,4}\b", content))
        used_findings = set(re.findall(r"\bfinding_\d{3,4}\b", content))
        meta["invalid_markers"] = sorted(
            (used_evidence - set(evidence_by_id))
            | (used_findings - (allowed_finding_ids or set()))
        )
        meta["invalid_content"] = content
        return "", meta
    return content, meta


def _section_content_uses_known_evidence(content: str, evidence_by_id: dict[str, dict[str, Any]]) -> bool:
    return _content_uses_only_allowed_markers(content, set(evidence_by_id), set())


def _content_uses_only_allowed_markers(
    content: str,
    allowed_evidence_ids: set[str],
    allowed_finding_ids: set[str],
) -> bool:
    value = str(content or "")
    used_evidence = set(re.findall(r"\bev_\d{3,4}\b", value))
    used_findings = set(re.findall(r"\bfinding_\d{3,4}\b", value))
    return bool(value.strip()) and not (used_evidence - allowed_evidence_ids) and not (
        used_findings - allowed_finding_ids
    )


def _section_is_underdeveloped(content: str, section: dict[str, Any], findings: list[dict[str, Any]]) -> bool:
    prose = re.sub(r"\[\[(?:ev|finding)_[^\]]+\]\]", "", str(content or ""))
    prose = re.sub(r"^#{1,6}\s+.*$|^\|.*\|$", "", prose, flags=re.M)
    target = _target_length_chars(section, findings)
    return len(re.sub(r"\s+", "", prose)) < target



def _fallback_section(
    section: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> str:
    # Operational diagnostics belong in synthesis_diagnostics/approval UI, not in the delivered
    # report. A generic placeholder or a raw finding dump looks like body text while providing no
    # reader value and can overstate "public evidence is insufficient" when only this run failed.
    return ""


def _delivery_completeness(
    completeness: dict[str, Any], documents: list[dict[str, Any]]
) -> dict[str, Any]:
    result = dict(completeness)
    failed_core_ids = [
        str(document.get("section_id") or document.get("title") or "")
        for document in documents
        if bool(((document.get("section_brief") or {}).get("reportability") or {}).get("core_section"))
        and (
            not str(document.get("content") or "").strip()
            or str((document.get("synthesis_diagnostics") or {}).get("mode") or "")
            in {"fallback_template", "model_failed"}
        )
    ]
    core_empty = list(dict.fromkeys([
        *[str(item) for item in result.get("core_empty_section_ids") or []],
        *failed_core_ids,
    ]))
    result["core_empty_section_ids"] = core_empty
    if core_empty:
        result["status"] = "partial_report"
        result["ready_section_count"] = max(
            0, int(result.get("required_section_count") or 0) - len(core_empty)
        )
        result["reason"] = "some core sections lack evidence-supported findings"
    return result


def _render_internal_citations(
    content: str,
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
    *,
    finding_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    finding_by_id = finding_by_id or {}
    ledger: list[dict[str, Any]] = []
    lines = []
    normalized_content = _normalize_internal_citation_markers(str(content or ""))
    for line in normalized_content.splitlines():
        is_table_row = line.lstrip().startswith("|")
        if is_table_row:
            prose_units = line.split("|")
            unit_joiner = "|"
        else:
            prose_units = (
                split_atomic_units(line)
                if line.strip() and not line.lstrip().startswith(("#", "- ", "* "))
                else [line]
            )
            unit_joiner = ""
        rendered_units = []
        previous_claim_text = ""
        for unit in prose_units:
            valid_ids: list[str] = []

            def render_marker(match: re.Match[str]) -> str:
                marker_ids: list[str] = []
                for token in re.findall(r"\b(?:ev|finding)_\d{3,4}\b", match.group(1)):
                    if token.startswith("ev_"):
                        candidates = [token]
                    else:
                        finding = finding_by_id.get(token) or {}
                        candidates = [
                            str(item)
                            for item in [*(finding.get("evidence_ids") or []), *(finding.get("counter_evidence_ids") or [])]
                        ]
                    for evidence_id in candidates:
                        if evidence_id in evidence_by_id and evidence_id not in marker_ids:
                            marker_ids.append(evidence_id)
                        if evidence_id in evidence_by_id and evidence_id not in valid_ids:
                            valid_ids.append(evidence_id)
                refs: list[int] = []
                for evidence_id in marker_ids:
                    evidence = evidence_by_id[evidence_id]
                    ref = citation_map.get(str(evidence.get("source_url") or ""))
                    if ref and ref not in refs:
                        refs.append(ref)
                return " ".join(f"[{ref}]" for ref in refs)

            rendered = re.sub(r"\[\[([^\]]+)\]\]", render_marker, unit)
            # The model is told to use [[ev_001]] but sometimes writes single-bracket [ev_001]
            # instead. Left unhandled, the sub below (stripping any stray bare ev_/finding_
            # token) deletes the id from inside those single brackets but leaves the brackets
            # themselves, producing a literal empty "[]" in the delivered report. Recognize this
            # single-bracket form too — narrowly, only brackets containing nothing but
            # ev_/finding_ ids — so the citation is still recovered instead of turning into noise.
            rendered = re.sub(
                r"\[\s*((?:ev|finding)_\d{3,4}(?:\s*[,;、]\s*(?:ev|finding)_\d{3,4})*)\s*\]",
                render_marker,
                rendered,
            )
            rendered = re.sub(r"\b(?:ev|finding)_\d{3,4}\b", "", rendered)
            rendered = re.sub(r"\[\s*\]", "", rendered)  # defensive: drop any leftover empty brackets
            rendered = _normalize_malformed_numeric_citations(rendered)
            rendered = _dedupe_line_citations(rendered)
            rendered_units.append(rendered)
            claim_text = re.sub(r"\[\d+\]", "", rendered).strip()
            if valid_ids and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", claim_text):
                # `rendered_text` keeps the citation markers ([N]) intact -- it is the exact
                # substring present in the assembled section content, unlike `text` (citations
                # stripped). _splice_unsupported_markers needs the citation-marker form to do an
                # exact substring match against document["content"] and splice a [\u5f85\u6838\u5b9e] marker
                # in place, right after this claim's sentence.
                ledger.append({
                    "text": claim_text,
                    "claim_context": previous_claim_text[:1000] if not is_table_row else "",
                    "evidence_ids": valid_ids,
                    "rendered_text": rendered,
                })
            if not is_table_row and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", claim_text):
                previous_claim_text = claim_text
        lines.append(unit_joiner.join(rendered_units))
    return "\n".join(lines), ledger


def _normalize_internal_citation_markers(content: str) -> str:
    """Normalize an unambiguous list of internal ids into independent markers.

    Besides Evidence/Finding ids, E+ Editorial uses content-addressed ``claim_*`` ids.
    Models occasionally wrap adjacent ids as ``[[claim_a][claim_b]]``.  That form is
    syntactically unambiguous, so normalize it deterministically before the stricter
    allow-list expansion instead of spending a model repair call on punctuation.
    """
    internal_id = (
        r"(?:ev_\d{3,4}|finding_\d{3,4}|claim_[A-Za-z0-9_-]+|"
        r"[A-Za-z0-9_-]+_claim_[A-Za-z0-9_-]+)"
    )
    pattern = re.compile(
        rf"\[\[\s*({internal_id}(?:\s*\]\s*(?:[,，;；、]\s*)?\[\s*{internal_id})+)\s*\]\]"
    )

    def replace(match: re.Match[str]) -> str:
        ids = re.findall(internal_id, match.group(1))
        return "".join(f"[[{item}]]" for item in ids)

    return pattern.sub(replace, content)


def _dedupe_line_citations(line: str) -> str:
    seen: set[str] = set()
    output = []
    cursor = 0
    for match in re.finditer(r"\[(\d+)\]", line):
        output.append(line[cursor:match.start()])
        ref = match.group(1)
        if ref not in seen:
            output.append(match.group(0))
            seen.add(ref)
        cursor = match.end()
    output.append(line[cursor:])
    result = "".join(output)
    # The model sometimes writes enumeration punctuation (、，,；;) between/after internal
    # markers, e.g. "（[[ev_004]]、[[ev_004]]、[[ev_004]]）". Once markers are rendered to
    # numbered refs and duplicates are collapsed above, that punctuation is orphaned --
    # either stranded between two adjacent refs, or trailing the last ref with nothing
    # after it. Clean both cases so citations never surface as "[4]、、" or "[1]、）".
    result = re.sub(r"\](?:[、，,；;]+)(?=\[)", "] ", result)
    result = re.sub(r"(?<=\])[、，,；;]+(?=[)）]|\Z)", "", result)
    result = re.sub(r"(\[\d+\])(?=[A-Za-z0-9\u4e00-\u9fff])", r"\1 ", result)
    return re.sub(r"\s{2,}", " ", result).strip()


def _normalize_malformed_numeric_citations(line: str) -> str:
    """Repair numeric citation wrappers accidentally emitted in internal-marker syntax."""
    result = str(line or "")
    previous = None
    while result != previous:
        previous = result
        result = re.sub(
            r"\[\[(\d+)\]\s*[,，;；、]\s*\[(\d+)\]\]",
            lambda match: f"[{match.group(1)}] [{match.group(2)}]",
            result,
        )
    result = re.sub(r"\[\[(\d+)\]\s*[,，;；、]?\s*\]", r"[\1]", result)
    result = re.sub(r"\[\[(\d+)\]\]", r"[\1]", result)
    result = re.sub(r"(?<=\])\s*[,，;；、]\s*(?=\[\d+\])", " ", result)
    return result


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Pick `count` items spread across `items`, always including the first and last.

    A plain `items[:count]` biases toward whatever appears earliest in the list. For a
    section's findings, earliest means the first retrieval round -- later rounds (which
    may hold the other side of a comparison, per the bug this guards against) would never
    be sampled. Spanning the full index range keeps later-round evidence in the sample.
    """
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    last = len(items) - 1
    indices = sorted({round(i * last / (count - 1)) for i in range(count)})
    return [items[i] for i in indices]


def _sample_findings_for_editorial(findings: list[dict[str, Any]], cap: int = 12) -> list[dict[str, Any]]:
    """Sample findings for the executive-summary/conclusion prompt, capped at `cap`.

    `findings` is appended in retrieval-round order: every section's round-1 findings,
    then every section's round-2 findings, and so on. A plain `findings[:cap]` can land
    exactly on a round boundary and hand the editor model evidence from only the earliest
    round(s) of only the earliest section(s) -- starving out later sections and later
    rounds entirely. That produced a real bug: the editor concluded "no comparison
    possible" because its truncated sample held only one side of a two-sided comparison,
    while the section writer (which reads findings per-section, not by this global
    truncation) had the full evidence and wrote the comparison into the report body.
    Sampling proportionally per section, and spanning each section's own round order,
    keeps the editor's view representative of what the body actually saw.
    """
    if len(findings) <= cap:
        return list(findings)
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in findings:
        key = str(item.get("section_id") or "")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    section_count = len(order) or 1
    base_quota, remainder = divmod(cap, section_count)
    sampled: list[dict[str, Any]] = []
    for index, key in enumerate(order):
        quota = base_quota + (1 if index < remainder else 0)
        sampled.extend(_evenly_spaced(groups[key], quota))
    return sampled[:cap]


def _edit_report(
    user_request: str,
    blueprint: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    citation_map: dict[str, int],
    model: JsonModel | None,
    *,
    revision_issues: list[dict[str, Any]] | None = None,
    existing_editorial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    retry_result: dict[str, Any] = {}
    summary = ""
    conclusion = ""
    summary_ledger: list[dict[str, Any]] = []
    conclusion_ledger: list[dict[str, Any]] = []
    initial_missing_fields = ["executive_summary", "conclusion"]
    retry_attempted = False
    revision_issues = list(revision_issues or [])
    existing_editorial = existing_editorial or {}
    if model and findings:
        research_mode = str(blueprint.get("research_mode") or "domain_practice_review")
        editorial_findings = _sample_findings_for_editorial(findings)
        editorial_finding_by_id = {
            str(item.get("finding_id") or ""): item
            for item in editorial_findings
            if str(item.get("finding_id") or "")
        }
        editorial_evidence_ids = {
            str(evidence_id)
            for item in editorial_findings
            for evidence_id in [*(item.get("evidence_ids") or []), *(item.get("counter_evidence_ids") or [])]
        }
        editorial_evidence_by_id = {
            evidence_id: evidence_by_id[evidence_id]
            for evidence_id in editorial_evidence_ids
            if evidence_id in evidence_by_id
        }
        payload = {
            "task": "Write an executive summary and conclusion for a research report; also provide a concise report_title. Return JSON only.",
            "user_request": user_request,
            "thesis": blueprint.get("thesis"),
            "research_mode": research_mode,
            "evidence_landscape": blueprint.get("evidence_landscape") or {},
            "findings": editorial_findings,
            "existing_editorial": {
                "executive_summary": existing_editorial.get("executive_summary") or "",
                "conclusion": existing_editorial.get("conclusion") or "",
                "claim_ledger": existing_editorial.get("claim_ledger") or [],
            } if revision_issues else {},
            "revision_issues": revision_issues,
            "rules": [
                "report_title must be a short reader-facing title naming the subject and angle of the report, not a copy of user_request; target 8-24 Chinese characters or 3-10 words.",
                "Do not put citations, Markdown, a full sentence, or a trailing '报告' label in report_title.",
                "Use only supplied findings.",
                "Write connected analytical prose for the executive summary and conclusion: synthesize related findings into paragraphs with a clear argument and natural transitions, not one isolated finding per sentence.",
                "Every factual clause must still cite the specific supplied finding/evidence marker it relies on, such as [[finding_001]] or [[ev_001]].",
                "Cite only finding markers and evidence ids that appear in this payload. A globally valid evidence id that is absent here is forbidden.",
                "Do not invent a new comparison, cost, performance, or architecture conclusion beyond what the supplied findings support.",
                "Write in Chinese.",
                "Use internal evidence markers [[ev_001]].",
                "Respect finding_type: inference and recommendation must remain explicitly qualified.",
                "Never present general or cross-domain support as a domain-specific fact.",
                "Do not attach a meta-evaluative judgment about the evidence or case itself (for example 值得进一步验证/具有启发性/值得关注) unless a supplied finding actually states that judgment.",
                "Do not add a Sources section.",
                "When direct domain evidence is scarce, state that once and present the report as observed practice plus an evidence-informed reference design.",
                "Do not let repeated evidence-gap disclaimers dominate the summary or conclusion.",
                *([
                    "This is a targeted grounding revision. Preserve existing_editorial except for clauses identified by revision_issues.",
                    "Remove or minimally rewrite rejected clauses; do not introduce new conclusions while repairing them.",
                    "existing_editorial prose contains delivered numeric citations such as [1]; rebuild citations from existing_editorial.claim_ledger and emit only [[ev_001]]-style internal markers.",
                ] if revision_issues else []),
                *([
                    "Debugging aid only, does not affect grading: populate editor_notes with one short "
                    "string per revision_issue you addressed, stating what you changed and why."
                ] if revision_issues and os.getenv("RESEARCH_DEBUG_REVISION_TRACE") == "1" else []),
            ],
            "schema": {"report_title": "", "executive_summary": "", "conclusion": "", "editor_notes": []},
        }
        with audit_scope(stage="editorial", section_id="editorial"):
            model_result = model(payload)
        result = model_result if isinstance(model_result, dict) else {"_failure_reason": "invalid_response_type"}
        for field in ("executive_summary", "conclusion"):
            value = str(result.get(field) or "")
            if value and not _content_uses_only_allowed_markers(
                value,
                set(editorial_evidence_by_id),
                set(editorial_finding_by_id),
            ):
                result[field] = ""
                result["_failure_reason"] = "evidence_validation_failed"
        summary, summary_ledger = _render_internal_citations(
            str(result.get("executive_summary") or ""),
            editorial_evidence_by_id,
            citation_map,
            finding_by_id=editorial_finding_by_id,
        )
        conclusion, conclusion_ledger = _render_internal_citations(
            str(result.get("conclusion") or ""),
            editorial_evidence_by_id,
            citation_map,
            finding_by_id=editorial_finding_by_id,
        )
        initial_missing_fields = [
            field
            for field, value in (("executive_summary", summary), ("conclusion", conclusion))
            if not value
        ]
        if initial_missing_fields:
            retry_attempted = True
            completion_payload = {
                "task": "Complete only the missing editorial fields for a research report. Return JSON only.",
                "user_request": user_request,
                "thesis": blueprint.get("thesis"),
                "research_mode": research_mode,
                "findings": editorial_findings,
                "existing_editorial": {
                    "executive_summary": result.get("executive_summary") or "",
                    "conclusion": result.get("conclusion") or "",
                },
                "missing_fields": initial_missing_fields,
                "rules": [
                    "Fill every field listed in missing_fields and return only the fields in the schema.",
                    "Use only supplied findings and write in Chinese.",
                    "Every factual clause must cite a supplied marker such as [[finding_001]] or [[ev_001]].",
                    "The conclusion must synthesize multiple findings and must not repeat the opening executive-summary sentence.",
                    "Keep valid existing editorial text unchanged and do not invent unsupported facts.",
                ],
                "schema": {field: "" for field in initial_missing_fields},
            }
            with audit_scope(stage="editorial", section_id="editorial"):
                completion = model(completion_payload)
            retry_result = completion if isinstance(completion, dict) else {"_failure_reason": "invalid_response_type"}
            for field in initial_missing_fields:
                value = str(retry_result.get(field) or "")
                if value and not _content_uses_only_allowed_markers(
                    value,
                    set(editorial_evidence_by_id),
                    set(editorial_finding_by_id),
                ):
                    retry_result[field] = ""
                    retry_result["_failure_reason"] = "evidence_validation_failed"
            if not summary and str(retry_result.get("executive_summary") or "").strip():
                summary, summary_ledger = _render_internal_citations(
                    str(retry_result.get("executive_summary") or ""),
                    editorial_evidence_by_id,
                    citation_map,
                    finding_by_id=editorial_finding_by_id,
                )
            if not conclusion and str(retry_result.get("conclusion") or "").strip():
                conclusion, conclusion_ledger = _render_internal_citations(
                    str(retry_result.get("conclusion") or ""),
                    editorial_evidence_by_id,
                    citation_map,
                    finding_by_id=editorial_finding_by_id,
                )

    failed_fields: list[str] = []
    report_title = _sanitize_report_title(str(result.get("report_title") or ""))
    if not summary:
        failed_fields.append("executive_summary")
    if not conclusion:
        failed_fields.append("conclusion")

    mode = "model_failed" if failed_fields else "model_completed_missing_fields" if retry_attempted else "model"
    annotated = _annotate_claim_ledger(
        [*summary_ledger, *conclusion_ledger], findings, evidence_by_id, "editorial"
    )
    return {
        "report_title": report_title,
        "executive_summary": summary,
        "conclusion": conclusion,
        "claim_ledger": annotated,
        "editor_notes": [
            *list(result.get("editor_notes") or []),
            *list(retry_result.get("editor_notes") or []),
        ],
        "diagnostics": {
            "mode": mode,
            "initial_missing_fields": initial_missing_fields,
            "failed_fields": failed_fields,
            "fallback_fields": [],
            "retry_attempted": retry_attempted,
            "upstream_failure_reason": str(result.get("_failure_reason") or ""),
            "retry_failure_reason": str(retry_result.get("_failure_reason") or ""),
        },
    }


def _sanitize_report_title(value: str) -> str:
    """Keep model-generated titles short and presentation-safe."""
    title = re.sub(r"\s+", " ", str(value or "")).strip(" #\n\r	:：。.!！")
    title = re.sub(r"\[\[?(?:ev|finding)_\d{3,4}.*?\]?\]", "", title).strip()
    if not title or len(title) > 60 or "\n" in title or any(token in title for token in ("用户问题", "研究问题", "执行摘要")):
        return ""
    return title


def _finding_marker(item: dict[str, Any]) -> str:
    finding_id = str(item.get("finding_id") or "").strip()
    if finding_id:
        return f"[[{finding_id}]]"
    evidence_id = str(next(iter(item.get("evidence_ids") or []), "")).strip()
    return f"[[{evidence_id}]]" if evidence_id else ""


def _fallback_executive_summary(findings: list[dict[str, Any]]) -> str:
    lines = []
    for item in findings[:4]:
        text = _typed_finding_text(item)
        if text:
            lines.append(f"- {text} {_finding_marker(item)}".rstrip())
    return "\n".join(lines)


def _fallback_conclusion(findings: list[dict[str, Any]]) -> str:
    later_findings = [item for item in findings[1:4] if str(item.get("finding") or "").strip()]
    if later_findings:
        takeaways = "；".join(
            f"{_typed_finding_text(item)} {_finding_marker(item)}".rstrip()
            for item in later_findings
        )
        return f"编辑模型未能生成完整的综合结论。基于其余已核验发现，可确认：{takeaways}。具体适用边界以正文为准。"
    if findings:
        return (
            "当前只有一项可引用发现，证据不足以形成独立的跨章节综合结论；"
            f"请以正文发现及其证据边界为准 {_finding_marker(findings[0])}。"
        )
    return "当前证据仍不足以形成稳定的综合结论。"


def _typed_finding_text(finding: dict[str, Any]) -> str:
    text = str(finding.get("finding") or "").strip()
    finding_type = str(finding.get("finding_type") or "inference")
    if finding_type == "inference" and not any(token in text for token in ("推断", "可能", "可映射", "可作为")):
        return f"基于现有证据可以推断：{text}"
    if finding_type == "recommendation" and not any(token in text for token in ("建议", "可考虑", "候选")):
        return f"可考虑将其作为候选设计：{text}"
    return text


def _claim_type_from_text_and_evidence(
    text: str,
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
    fallback: str,
) -> str:
    value = str(text or "").casefold()
    recommendation_markers = ("建议", "可考虑", "候选", "应当", "应该", "recommend", "should", "consider")
    inference_markers = (
        "推断", "可能", "或许", "这意味着", "由此可见", "表明", "说明", "暗示",
        "infer", "suggest", "indicate", "imply", "may", "might", "could", "likely",
    )
    if any(marker in value for marker in recommendation_markers):
        return "recommendation"
    if any(marker in value for marker in inference_markers):
        return "inference"
    evidence = [evidence_by_id[item] for item in evidence_ids if item in evidence_by_id]
    if evidence and all(is_direct_exact_evidence(item) for item in evidence):
        return "fact"
    return fallback


def _annotate_claim_ledger(
    ledger: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    section_id: str,
) -> list[dict[str, Any]]:
    output = []
    priority = {"fact": 0, "limitation": 1, "inference": 2, "recommendation": 3}
    for row in ledger:
        evidence_ids = [str(item) for item in row.get("evidence_ids") or []]
        related = [
            finding for finding in findings
            if set(evidence_ids) & set(str(item) for item in finding.get("evidence_ids") or [])
        ]
        finding_type = max(
            (str(item.get("finding_type") or "inference") for item in related),
            key=lambda item: priority.get(item, 2),
            default="inference",
        )
        finding_type = _claim_type_from_text_and_evidence(
            str(row.get("text") or ""), evidence_ids, evidence_by_id, finding_type
        )
        target_fits = {evidence_target_fit(evidence_by_id.get(item) or {}) for item in evidence_ids}
        support_scope = "domain_specific" if target_fits == {"exact_target"} else "cross_domain" if "exact_target" in target_fits or "related_target" in target_fits else "general"
        output.append({
            **row,
            "section_id": section_id,
            "atomic_claim_ids": [str(item.get("claim_id")) for item in related if item.get("claim_id")],
            "atomic": is_atomic_claim(str(row.get("text") or "")),
            "claim_type": finding_type,
            "support_scope": support_scope,
            "direct_evidence_ids": [item for item in evidence_ids if is_direct_exact_evidence(evidence_by_id.get(item) or {})],
            "inference_evidence_ids": [item for item in evidence_ids if not is_direct_exact_evidence(evidence_by_id.get(item) or {})],
        })
    return output


def _assemble_report(
    user_request: str,
    documents: list[dict[str, Any]],
    editorial: dict[str, Any],
    citation_map: dict[str, int],
    evidence_map: list[dict[str, Any]],
    *,
    completeness: dict[str, Any] | None = None,
    research_mode: str = "domain_practice_review",
    limitations_block: str | None = None,
    reference_title: str | None = None,
) -> tuple[str, int]:
    completeness = completeness or {"status": "complete_report"}
    complete = str(completeness.get("status") or "partial_report") == "complete_report"
    report_label = _report_label(complete, research_mode)
    report_title = _sanitize_report_title(str(editorial.get("report_title") or "")) or f"{user_request} {report_label}"
    lines = [f"# {report_title}", "", f"> 研究问题：{user_request}", ""]
    if not complete:
        limited_ids = {str(item) for item in completeness.get("limited_section_ids") or []}
        document_ids = {
            str(document.get("section_id") or document.get("title") or "") for document in documents
        }
        limited_titles = [
            str(document.get("title") or document.get("section_id") or "").replace("_", " ")
            for document in documents
            if str(document.get("section_id") or document.get("title") or "") in limited_ids
            and "总览" not in str(document.get("title") or "")
        ]
        limited_titles.extend(sorted(limited_ids - document_ids))
        limited = "、".join(item for item in dict.fromkeys(limited_titles) if item)
        empty_ids = {str(item) for item in completeness.get("core_empty_section_ids") or []}
        gap_ids = {
            str(document.get("section_id") or document.get("title") or "")
            for document in documents
            if str((document.get("synthesis_diagnostics") or {}).get("mode") or "") == "evidence_gap"
        } & empty_ids
        gap_titles = [
            str(document.get("title") or document.get("section_id") or "").replace("_", " ")
            for document in documents
            if str(document.get("section_id") or document.get("title") or "") in gap_ids
        ]
        gap_titles.extend(sorted(gap_ids - document_ids))
        gap = "、".join(item for item in dict.fromkeys(gap_titles) if item)
        empty_ids -= gap_ids
        empty_titles = [
            str(document.get("title") or document.get("section_id") or "").replace("_", " ")
            for document in documents
            if str(document.get("section_id") or document.get("title") or "") in empty_ids
        ]
        empty_titles.extend(sorted(empty_ids - document_ids))
        empty = "、".join(item for item in dict.fromkeys(empty_titles) if item)
        notice = "当前报告为阶段性结果。"
        if limited:
            notice += f"证据广度仍可补充的章节：{limited}。"
        if gap:
            notice += f"本轮仅形成证据缺口说明的章节：{gap}。"
        if empty:
            notice += f"本轮未形成可交付正文的章节：{empty}。"
        lines.extend([f"> {notice}", ""])
    lines.extend(["## 执行摘要", "", str(editorial.get("executive_summary") or "").strip(), ""])
    resolved_reference_title = str(reference_title or "来源").strip() or "来源"
    for document in documents:
        title = str(document.get("title") or "").strip()
        if "总览" in title:
            continue
        if _is_reference_title(title):
            resolved_reference_title = title.replace("_", " ")
            continue
        content = _strip_leading_section_heading(str(document.get("content") or ""), title)
        if (
            not content.strip()
            or str((document.get("synthesis_diagnostics") or {}).get("mode") or "")
            in {"fallback_template", "model_failed"}
        ):
            continue
        lines.extend([f"## {title.replace('_', ' ')}", "", content, ""])
    lines.extend(["## 综合结论", "", str(editorial.get("conclusion") or "").strip(), ""])
    if limitations_block:
        # One-shot flow (see docs/plans/writer-critic-loop-redesign, 2026-08-17): the 6-dimension
        # critic no longer drives a revision loop -- its issues, plus the [待核实] marker density,
        # surface here instead, explicitly labeled as unverified AI self-assessment rather than a
        # gate. Placed after the conclusion so it doesn't interrupt the report's own argument.
        lines.extend(["## 局限性声明", "", limitations_block, ""])
    lines.extend([f"## {resolved_reference_title}", ""])
    source_titles = {str(item.get("source_url") or ""): str(item.get("source_title") or item.get("source_url") or "") for item in evidence_map}
    for url, number in sorted(citation_map.items(), key=lambda item: item[1]):
        lines.append(f"[{number}] {source_titles.get(url) or url}: {url}")
    return _sanitize_leaked_evidence_ids("\n".join(lines).strip())


_UNSUPPORTED_ENTAILMENT_STATUSES = {"unsupported", "label_mismatch", "non_atomic"}


def _excluded_from_assembled_body(title: str) -> bool:
    """True for a document title _assemble_report never renders into the delivered body -- a
    "总览" (overview) section folded into the executive summary instead, or the references/
    sources section (rendered separately, from citation_map, not from its own `content`). A
    [待核实] marker spliced into one of these documents' `content` would never reach a reader, so
    _splice_unsupported_markers must redirect flags for these sections elsewhere instead of
    writing into text _assemble_report drops on the floor. See _assemble_report's own
    title-skip loop, which this mirrors."""
    title = str(title or "").strip()
    return "总览" in title or _is_reference_title(title)


def _is_reference_metadata_section(section: dict[str, Any]) -> bool:
    """Return True only for a derived references section, not substantive source analysis."""
    reportability = section.get("reportability") if isinstance(section.get("reportability"), dict) else {}
    return (
        _is_reference_title(str(section.get("title") or ""))
        and not bool(reportability.get("core_section"))
        and not bool(section.get("finding_ids"))
    )


def _reference_title_from_blueprint(blueprint: dict[str, Any]) -> str:
    for section in blueprint.get("sections") or []:
        if isinstance(section, dict) and _is_reference_metadata_section(section):
            return str(section.get("title") or "来源").replace("_", " ").strip() or "来源"
    return "来源"


def _splice_unsupported_markers(
    *,
    documents: list[dict[str, Any]],
    editorial: dict[str, Any],
    entailment: dict[str, Any],
    report_quality: dict[str, Any] | None,
    user_request: str,
    evidence_map: list[dict[str, Any]],
    completeness: dict[str, Any] | None = None,
    research_mode: str = "domain_practice_review",
) -> tuple[list[dict[str, Any]], dict[str, Any], str, int]:
    """One-shot post-processing step (see docs/plans/writer-critic-loop-redesign, 2026-08-17):
    replaces the old multi-pass writer<->critic<->entailment revision loop, which oscillated
    without converging (the a98ba6 dead loop; 54-73% cross-pass issue recurrence per a 10-job
    audit). A claim validate_claim_entailment flags as unsupported is never sent back to the
    writer and never deleted -- it is annotated in place with an inline [待核实] marker so a
    human reader sees the flag exactly where the claim appears, and judges it themselves.

    Matching a judgment back to its rendered position primarily uses `claim_index`, which lines
    up 1:1 with `report_document["claim_ledger"]` (validate_claim_entailment enumerates exactly
    that list -- see quality.py). Falls back to a section_id + claim-text lookup if the index
    doesn't resolve. If neither locates the claim's `rendered_text` as an exact substring of the
    live content (e.g. the ledger snapshot and content diverged), the marker is never silently
    dropped -- it is appended to the end of the section (or, if the section itself can't be
    found, to the editorial conclusion) instead.
    """
    claim_ledger = [
        *[entry for document in documents for entry in document.get("claim_ledger") or []],
        *(editorial.get("claim_ledger") or []),
    ]
    judgments = [
        item for item in entailment.get("judgments") or []
        if str(item.get("status") or "") in _UNSUPPORTED_ENTAILMENT_STATUSES
    ]
    documents = [dict(document) for document in documents]
    documents_by_section = {str(document.get("section_id") or ""): document for document in documents}
    editorial = dict(editorial)
    marker = "[待核实]"
    marker_count = 0
    for judgment in judgments:
        section_id = str(judgment.get("section_id") or "")
        claim_text = str(judgment.get("claim") or "").strip()
        claim_index = judgment.get("claim_index")
        row = (
            claim_ledger[claim_index]
            if isinstance(claim_index, int) and 0 <= claim_index < len(claim_ledger)
            else None
        )
        if row is None or str(row.get("section_id") or "") != section_id:
            row = next(
                (
                    item for item in claim_ledger
                    if str(item.get("section_id") or "") == section_id and str(item.get("text") or "") == claim_text
                ),
                None,
            )
        rendered_text = str((row or {}).get("rendered_text") or "").strip()
        if section_id == "editorial":
            for field in ("executive_summary", "conclusion"):
                content = str(editorial.get(field) or "")
                if rendered_text and rendered_text in content:
                    editorial[field] = content.replace(rendered_text, f"{rendered_text} {marker}", 1)
                    marker_count += 1
                    break
            else:
                editorial["conclusion"] = (
                    str(editorial.get("conclusion") or "").rstrip() + f"\n\n{marker} 未验证声明：{claim_text[:200]}"
                ).strip()
                marker_count += 1
            continue
        document = documents_by_section.get(section_id)
        if document is None or _excluded_from_assembled_body(str(document.get("title") or "")):
            # Either section_id didn't resolve to any known document (should not happen --
            # entailment's section_id always comes from a real document/editorial), or it
            # resolved to a document _assemble_report never renders into the delivered body (a
            # "总览" overview folded into the executive summary, or the references section). In
            # both cases the flag must never be dropped silently: surface it in the editorial
            # conclusion, which _assemble_report always renders, instead of losing it.
            editorial["conclusion"] = (
                str(editorial.get("conclusion") or "").rstrip()
                + f"\n\n{marker} 未验证声明（章节 {section_id or '未知'}）：{claim_text[:200]}"
            ).strip()
            marker_count += 1
            continue
        content = str(document.get("content") or "")
        if rendered_text and rendered_text in content:
            document["content"] = content.replace(rendered_text, f"{rendered_text} {marker}", 1)
        else:
            document["content"] = content.rstrip() + f"\n\n{marker} 未验证声明：{claim_text[:200]}"
        marker_count += 1
    total_claims = len(claim_ledger)
    limitations_block = _build_limitations_disclosure(report_quality, marker_count, total_claims)
    content, _sanitized_count = _assemble_report(
        user_request,
        documents,
        editorial,
        _citation_map(evidence_map),
        evidence_map,
        completeness=completeness,
        research_mode=research_mode,
        limitations_block=limitations_block,
        reference_title=str(editorial.get("reference_title") or "").strip() or None,
    )
    return documents, editorial, content, marker_count


def _build_limitations_disclosure(
    report_quality: dict[str, Any] | None, marker_count: int, total_claims: int
) -> str:
    """Renders the report's appended '局限性声明' -- everything the 6-dimension critic flagged,
    plus the [待核实] marker density, explicitly labeled as unverified AI self-assessment rather
    than a gate (see docs/plans/writer-critic-loop-redesign, 2026-08-17)."""
    lines = ["本节为 AI 自评，未经交叉验证，仅供阅读参考，不作为结论性判断。"]
    issues = (report_quality or {}).get("issues") or []
    if issues:
        lines.extend(["", "**评论模型指出的潜在改进点：**", ""])
        for issue in issues[:8]:
            instruction = str(issue.get("instruction") or issue.get("type") or "").strip()
            if not instruction:
                continue
            section_id = str(issue.get("section_id") or "").strip()
            prefix = f"[{section_id}] " if section_id else ""
            lines.append(f"- {prefix}{_truncate_visible_comment(instruction, 300)}")
    summary = str((report_quality or {}).get("critic_summary") or "").strip()
    if summary:
        lines.extend(["", f"**评论模型总体评价：** {_truncate_visible_comment(summary, 500)}"])
    pct = round(100 * marker_count / total_claims, 1) if total_claims else 0.0
    lines.extend([
        "",
        f"本报告中 {marker_count} / {total_claims} 条声明被标记为待核实"
        f"（约 {pct}%），未设强制拦截阈值，仅供人工复核，标记本身不代表报告整体不可信。",
    ])
    return "\n".join(lines).strip()


def _truncate_visible_comment(text: str, max_chars: int) -> str:
    """Keep advisory critic text within bounds without exposing a dangling half-sentence."""
    normalized = str(text or "").strip()
    # Upstream quality normalization currently stores critic summaries at an exact 500-char
    # ceiling, so equality can already mean a hard-truncated value that needs boundary repair.
    if len(normalized) < max_chars:
        return normalized
    candidate = normalized[:max_chars].rstrip()
    boundaries = list(re.finditer(r"[。！？.!?；;]", candidate))
    if boundaries and boundaries[-1].end() >= int(max_chars * 0.6):
        truncated = candidate[: boundaries[-1].end()].rstrip()
        return re.sub(r"([。！？.!?])[；;]+$", r"\1", truncated)
    return candidate.rstrip("，,：:；;、 ") + "…"


_LEAKED_EVIDENCE_ID_RE = re.compile(r"\[\[[^\]]*\]\]|(?<![\[\w])ev_\d{3,4}(?!\w)")


def _sanitize_leaked_evidence_ids(content: str) -> tuple[str, int]:
    """Defensive final-pass cleanup for internal evidence ids that reached the delivered text.

    `_render_internal_citations` already converts `[[ev_001]]` markers to numbered `[1]`
    references within normal section prose (including model-authored table rows), but a model
    response that leaked a raw marker into a place the renderer didn't walk can still surface a
    bare marker. This strips whatever survives so the delivered report never shows an internal
    id, while returning a count so the caller can still treat the leak as a quality-gate
    blocking issue rather than silently hiding that it happened.
    """
    count = 0

    def _strip(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return ""

    sanitized = _LEAKED_EVIDENCE_ID_RE.sub(_strip, content)
    if count:
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized, count


def _report_label(complete: bool, research_mode: str) -> str:
    if research_mode in {"hybrid_practice_and_reference", "evidence_informed_reference_design"}:
        return "公开实践与参考设计研究报告" if complete else "公开实践与参考设计阶段性报告"
    return "深度研究报告" if complete else "深度研究阶段性报告"


def _material_utilization(documents: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for document in documents:
        diagnostics = document.get("synthesis_diagnostics") if isinstance(document.get("synthesis_diagnostics"), dict) else {}
        rows.append({
            "title": document.get("title"),
            "input_finding_count": int(diagnostics.get("input_finding_count") or 0),
            "used_finding_count": int(diagnostics.get("used_finding_count") or 0),
            "dossier_evidence_count": int(diagnostics.get("dossier_evidence_count") or 0),
            "used_evidence_count": int(diagnostics.get("used_evidence_count") or 0),
            "evidence_utilization_ratio": float(diagnostics.get("evidence_utilization_ratio") or 0),
            "dossier_chunk_count": int(diagnostics.get("dossier_chunk_count") or 0),
            "dossier_context_chars": int(diagnostics.get("dossier_context_chars") or 0),
            "output_chars": len(str(document.get("content") or "")),
        })
    # The overview section has no dossier of its own (_write_section reuses the global first-6
    # findings when a section has no finding_ids and its title contains "总览" -- see the
    # `"总览" in str(section.get("title") or "")` branch above), so its dossier_evidence_count is
    # always 0 while its used_evidence_count reflects those reused findings' evidence. Folding it
    # into the sum inflates the numerator without a matching denominator contribution, which is
    # what let a body-section aggregate genuinely below 1.0 read as a misleading 1.0 overall (see
    # docs/evaluations/deep-research-section-dossier-e2e-20260715.md section 3). Body-section
    # utilization is what this diagnostic is meant to answer, so the overview row is excluded from
    # the aggregate here while still being kept in `sections` for per-section inspection.
    body_rows = [row for row in rows if "总览" not in str(row["title"] or "")]
    input_evidence = sum(row["dossier_evidence_count"] for row in body_rows)
    used_evidence = sum(row["used_evidence_count"] for row in body_rows)
    return {
        "section_count": len(rows),
        "input_evidence_count": input_evidence,
        "used_evidence_count": used_evidence,
        "evidence_utilization_ratio": round(used_evidence / input_evidence, 3) if input_evidence else 1.0,
        "sections": rows,
    }


def _strip_leading_section_heading(content: str, title: str) -> str:
    value = str(content or "").strip()
    match = re.match(r"^##(?!#)\s+([^\n]+)\n+", value)
    if not match:
        return value
    heading = re.sub(r"[\s_]+", "", match.group(1)).lower()
    expected = re.sub(r"[\s_]+", "", title).lower()
    expected_without_number = re.sub(r"^\d+", "", expected)
    if heading in {expected, expected_without_number} or heading.endswith(expected_without_number):
        return value[match.end():].lstrip()
    return value


def _is_reference_title(title: str) -> bool:
    value = str(title or "").lower()
    # "来源" is _assemble_report's own fallback reference_title (used when no document
    # title matches the other tokens) -- it must be recognized here too, or the section
    # it produced fails to self-identify and reader-language-ratio counts its English
    # source titles/URLs as report prose (case-reference-language-ratio: 0.3523 vs 0.35 gate).
    return any(token in value for token in ("证据与参考", "来源索引", "references", "来源"))


def _citation_map(evidence_map: list[dict[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in evidence_map:
        url = str(item.get("source_url") or "")
        if url and url not in output:
            output[url] = len(output) + 1
    return output


def _evidence_refs(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"evidence_id": item.get("id"), "chunk_id": item.get("chunk_id"), "claim": item.get("atomic_claim") or item.get("claim"), "source_url": item.get("source_url")} for item in evidence]


def _evidence_table_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "section_id": item.get("section_id"),
        "question_id": item.get("question_id"),
        "claim": item.get("atomic_claim") or item.get("claim"),
        "quote": item.get("cleaned_quote") or item.get("evidence_text"),
        "source_url": item.get("source_url"),
        "source_title": item.get("source_title"),
        "source_domain": str(item.get("source_url") or "").split("/")[2] if "/" in str(item.get("source_url") or "") else "",
        "confidence": item.get("support_strength") or "medium",
        "chunk_id": item.get("chunk_id"),
        "used_in_document": True,
        "risk_flags": item.get("quality_flags") or [],
    }


def _evidence_metrics(table: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evidence_count": len(table),
        "source_count": len({str(item.get("source_url") or "") for item in table}),
        "domain_count": len({str(item.get("source_domain") or "") for item in table}),
        "question_count": len({str(item.get("question_id") or "") for item in table}),
        "used_evidence_count": len({item for row in ledger for item in row.get("evidence_ids") or []}),
        "risk_flag_count": sum(len(item.get("risk_flags") or []) for item in table),
        "paragraph_validation_status": "passed",
    }
