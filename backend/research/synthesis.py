"""Finding synthesis and report blueprint construction."""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from urllib.parse import urlparse

from .audit import audit_scope, submit_with_audit_context
from .claims import atomize_finding
from .evidence import (
    claim_similarity,
    evidence_support_relation,
    evidence_target_fit,
    is_direct_exact_evidence,
)
from .text_utils import extract_key_terms, extract_terms, is_non_research_section_title

JsonModel = Callable[[dict[str, Any]], dict[str, Any]]


class FindingSynthesisError(RuntimeError):
    """Raised when accepted evidence cannot be converted into model-reviewed findings."""

    def __init__(self, failures: list[dict[str, str]]) -> None:
        self.failures = failures
        summary = ", ".join(
            f"{item.get('question_id') or 'unknown'}:{item.get('failure_reason') or 'empty_findings'}"
            for item in failures
        )
        super().__init__(f"finding synthesis failed: {summary}")

_FINDING_EVIDENCE_CLUSTER_CAP = 24
_FINDING_CLUSTERS_PER_CRITERION = 3
_FINDING_GENERIC_TERMS = {
    "deepseek", "harness", "agent", "问题", "研究", "实质", "回答", "说明", "描述",
    "给出", "补充", "一个", "如何", "什么", "系统", "框架", "内容", "相关",
}

_FINDING_ADMISSION_STATUSES = {
    "admitted",
    "supported",
    "unsupported",
    "irrelevant",
    "overreach",
    "ambiguous",
}


def synthesize_findings(
    question_states: list[dict[str, Any]],
    *,
    model: JsonModel | None = None,
    parallelism: int = 3,
    research_mode: str = "domain_practice_review",
) -> list[dict[str, Any]]:
    def generate(
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str, str]:
        with audit_scope(
            stage="finding_synthesis",
            question_id=str(state.get("question_id") or "") or None,
            section_id=str(state.get("section_id") or "") or None,
        ):
            evidence = [item for item in state.get("accepted_evidence") or [] if isinstance(item, dict)]
            if not evidence:
                return state, evidence, [], "empty", "no_evidence"
            generated, failure_reason = (
                _model_findings(state, evidence, model, research_mode=research_mode)
                if model else ([], "model_not_configured")
            )
            synthesis_mode = "model" if generated else "failed"
            return state, evidence, generated, synthesis_mode, failure_reason

    generated_rows: list[
        tuple[int, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str, str]
    ] = []
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(question_states) or 1))) as executor:
        futures = {
            submit_with_audit_context(executor, generate, state): index
            for index, state in enumerate(question_states)
        }
        for future in as_completed(futures):
            state, evidence, generated, synthesis_mode, failure_reason = future.result()
            generated_rows.append((
                futures[future], state, evidence, generated, synthesis_mode, failure_reason,
            ))
    findings: list[dict[str, Any]] = []
    binding_failures: list[dict[str, str]] = []
    for _index, state, evidence, generated, synthesis_mode, failure_reason in sorted(
        generated_rows, key=lambda item: item[0]
    ):
        if evidence and synthesis_mode == "failed":
            continue
        state_finding_count_before = len(findings)
        atomic_generated = [
            atomic
            for item in generated
            for atomic in atomize_finding({
                **item,
                "_candidate_evidence_ids": [str(value) for value in item.get("evidence_ids") or []],
            }, evidence=evidence)
        ]
        for item in atomic_generated:
            ids = [str(value) for value in item.get("evidence_ids") or []]
            valid_ids = {str(row.get("id") or "") for row in evidence}
            ids = [value for value in ids if value in valid_ids]
            ids = _restore_same_cluster_corroboration(
                ids,
                [str(value) for value in item.get("_candidate_evidence_ids") or []],
                evidence,
            )
            if not ids:
                continue
            deterministic_confidence = _confidence(evidence)
            model_confidence = str(item.get("confidence") or deterministic_confidence)
            requested_type = str(item.get("finding_type") or "fact")
            finding_type, support_scope = _finding_type(requested_type, ids, evidence)
            valid_criterion_ids = {
                str(criterion.get("criterion_id") or "")
                for criterion in _finding_criterion_ledger(state)
                if str(criterion.get("criterion_id") or "")
            }
            criterion_ids, answer_role = _finding_answer_binding(
                item,
                ids=ids,
                evidence=evidence,
                valid_criterion_ids=valid_criterion_ids,
            )
            direct_ids = [
                evidence_id for evidence_id in ids
                if _is_direct_evidence(next((row for row in evidence if str(row.get("id") or "") == evidence_id), {}))
            ]
            findings.append({
                "finding_id": f"finding_{len(findings) + 1:03d}",
                "claim_id": f"claim_{len(findings) + 1:03d}",
                "section_id": state.get("section_id"),
                "question_id": state.get("question_id"),
                "finding": str(item.get("finding") or "").strip(),
                "evidence_ids": ids,
                "synthesis_mode": synthesis_mode,
                "criterion_ids": criterion_ids,
                "answer_role": answer_role,
                "comparison_axis": str(state.get("comparison_axis") or ""),
                "counter_evidence_ids": [str(value) for value in item.get("counter_evidence_ids") or [] if str(value) in valid_ids],
                "limitations": [str(value) for value in item.get("limitations") or [] if str(value).strip()],
                "support_reason": str(item.get("support_reason") or "").strip(),
                "premises": [
                    premise
                    for premise in item.get("premises") or []
                    if isinstance(premise, dict)
                    and str(premise.get("evidence_id") or "") in valid_ids
                    and str(premise.get("premise") or "").strip()
                ],
                "reasoning": str(item.get("reasoning") or "").strip(),
                "uncertainty": str(item.get("uncertainty") or "").strip(),
                "confidence": _lower_confidence(model_confidence, deterministic_confidence),
                "finding_type": finding_type,
                "support_scope": support_scope,
                "direct_evidence_ids": direct_ids,
                "inference_evidence_ids": [evidence_id for evidence_id in ids if evidence_id not in direct_ids],
                "atomic": True,
                "decomposition": item.get("decomposition") or "original",
            })
        if evidence and generated and len(findings) == state_finding_count_before:
            binding_failures.append({
                "section_id": str(state.get("section_id") or ""),
                "question_id": str(state.get("question_id") or ""),
                "failure_reason": "all_findings_unbound",
            })
    failures = [
        {
            "section_id": str(state.get("section_id") or ""),
            "question_id": str(state.get("question_id") or ""),
            "failure_reason": failure_reason or "empty_findings",
        }
        for _index, state, evidence, _generated, synthesis_mode, failure_reason in generated_rows
        if evidence and synthesis_mode == "failed"
    ]
    failures.extend(binding_failures)
    if failures:
        raise FindingSynthesisError(failures)
    return _dedupe_findings(findings)


def _finding_answer_binding(
    item: dict[str, Any],
    *,
    ids: list[str],
    evidence: list[dict[str, Any]],
    valid_criterion_ids: set[str],
) -> tuple[list[str], str]:
    """Keep a finding's answer role within the evidence bindings already admitted upstream.

    The finding model may summarize or reason over evidence, but it must not promote an
    unassigned/background row into evidence for a criterion merely because the wording looks
    related. Historical rows without binding metadata retain their former behavior; current rows
    use the extractor/admission result as the authority.
    """
    rows = [
        row for row in evidence
        if str(row.get("id") or "") in set(ids)
    ]
    requested = list(dict.fromkeys(
        str(value)
        for value in item.get("criterion_ids") or []
        if str(value) in valid_criterion_ids
    ))
    if not valid_criterion_ids:
        return [], "question_support"
    has_current_binding = any("criterion_binding_status" in row for row in rows)
    if not has_current_binding:
        return requested, "legacy_unchecked"
    supported = list(dict.fromkeys(
        str(value)
        for row in rows
        if str(row.get("criterion_binding_status") or "") not in {
            "incompatible_subject",
            "unverified_subject",
        }
        and str(row.get("comparison_axis_binding_status") or "") not in {
            "incompatible_axis",
            "unverified_axis",
        }
        for value in row.get("criterion_ids") or []
        if str(value) in valid_criterion_ids
    ))
    bound = [value for value in requested if value in supported]
    if not requested and len(supported) == 1:
        bound = supported
    return bound, "core_support" if bound else "context_only"


def review_finding_admission(
    question_states: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    *,
    model: JsonModel | None,
    parallelism: int = 3,
) -> dict[str, Any]:
    """Independently admit findings before they can enter the writer.

    Finding generation is intentionally analytical, so its own explanation cannot also serve as
    the admission decision. This review sees only the candidate claim, the question contract, and
    the evidence rows the candidate actually bound. It never rewrites or deterministically repairs
    a rejected finding.
    """
    state_by_question = {
        str(state.get("question_id") or ""): state
        for state in question_states
        if isinstance(state, dict) and str(state.get("question_id") or "")
    }
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        grouped[str(finding.get("question_id") or "")].append((index, finding))

    def review_question(
        question_id: str,
        rows: list[tuple[int, dict[str, Any]]],
    ) -> tuple[str, list[dict[str, Any]]]:
        state = state_by_question.get(question_id) or {}
        section_id = str(state.get("section_id") or rows[0][1].get("section_id") or "")
        evidence_by_id = {
            str(item.get("id") or ""): item
            for item in state.get("accepted_evidence") or []
            if isinstance(item, dict) and str(item.get("id") or "")
        }
        criterion_ledger = _finding_criterion_ledger(state)
        valid_criterion_ids = {
            str(item.get("criterion_id") or "")
            for item in criterion_ledger
            if str(item.get("criterion_id") or "")
        }
        candidate_rows = []
        candidate_evidence_ids_by_index: dict[int, set[str]] = {}
        global_index_by_local_index: dict[int, int] = {}
        used_evidence_ids: set[str] = set()
        for local_index, (global_index, finding) in enumerate(rows):
            evidence_ids = [
                str(value)
                for value in finding.get("evidence_ids") or []
                if str(value) in evidence_by_id
            ]
            global_index_by_local_index[local_index] = global_index
            candidate_evidence_ids_by_index[local_index] = set(evidence_ids)
            used_evidence_ids.update(evidence_ids)
            candidate_rows.append({
                "finding_index": local_index,
                "finding_id": str(finding.get("finding_id") or ""),
                "finding": str(finding.get("finding") or ""),
                "finding_type": str(finding.get("finding_type") or "fact"),
                "criterion_ids": [str(value) for value in finding.get("criterion_ids") or []],
                "answer_role": str(finding.get("answer_role") or ""),
                "evidence_ids": evidence_ids,
            })
        payload = {
            "task": "独立审计候选 finding 是否由其绑定证据支持且回答当前研究问题。只返回 JSON。",
            "question": str(state.get("question") or ""),
            "comparison_axis": str(state.get("comparison_axis") or ""),
            "criterion_ledger": criterion_ledger,
            "candidates": candidate_rows,
            "evidence": [
                {
                    "id": evidence_id,
                    "claim": str(evidence_by_id[evidence_id].get("atomic_claim") or evidence_by_id[evidence_id].get("claim") or ""),
                    "quote": str(
                        evidence_by_id[evidence_id].get("raw_quote")
                        or evidence_by_id[evidence_id].get("cleaned_quote")
                        or evidence_by_id[evidence_id].get("evidence_text")
                        or ""
                    )[:1600],
                    "subject": str(evidence_by_id[evidence_id].get("subject") or ""),
                    "subject_relation": str(evidence_by_id[evidence_id].get("subject_relation") or ""),
                    "criterion_ids": [str(value) for value in evidence_by_id[evidence_id].get("criterion_ids") or []],
                    "criterion_binding_status": str(evidence_by_id[evidence_id].get("criterion_binding_status") or ""),
                    "comparison_axis": str(evidence_by_id[evidence_id].get("comparison_axis") or ""),
                    "comparison_metric": str(evidence_by_id[evidence_id].get("comparison_metric") or ""),
                    "expected_comparison_axis": str(evidence_by_id[evidence_id].get("expected_comparison_axis") or ""),
                    "comparison_axis_binding_status": str(evidence_by_id[evidence_id].get("comparison_axis_binding_status") or ""),
                    "publisher_relation": str(evidence_by_id[evidence_id].get("publisher_relation") or ""),
                    "scope": str(evidence_by_id[evidence_id].get("scope") or ""),
                    "claim_type": str(evidence_by_id[evidence_id].get("claim_type") or evidence_by_id[evidence_id].get("evidence_type") or ""),
                    "source_title": str(evidence_by_id[evidence_id].get("source_title") or ""),
                    "source_url": str(evidence_by_id[evidence_id].get("source_url") or ""),
                }
                for evidence_id in sorted(used_evidence_ids)
            ],
            "rules": [
                "不得使用外部知识；evidence claim 与 quote 冲突时以 quote 为准。",
                "question 和 criterion 只用于判断相关性，不是证据，不能证明其要求的结论为真。",
                "相关性按 criterion 局部判断：一条 finding 不需要回答 question 的全部组成部分。只要其绑定证据直接支持 finding，且 finding 实质回答 criterion_ledger 中至少一个独立 criterion，就不能因为其他 criterion 尚未回答而判 irrelevant。对 supported 判断必须返回 matched_criterion_ids；只能选择该 finding 实质回答且有绑定证据支持的 criterion_id。",
                "主体必须属于 question 明确要求研究的对象，或是 question 明确要求的总体/外部背景。仅被标成 comparison 不足以证明相关；未被 question 纳入的其他产品、框架、人物或案例应判 irrelevant。",
                "若 question 给出 comparison_axis，finding 的核心比较必须由实际讨论同一维度的 evidence 支持。总体评分、有用性、价格、速度或其他维度不能替代当前维度；维度不同的材料最多是背景，应判 irrelevant 或 overreach。",
                "对比问题允许一条 finding 只描述一个已请求主体，只要它直接回答当前 comparison_axis；不要求每条 finding 自身同时覆盖多个主体，跨主体综合由后续写作完成。",
                "保留来源角色：厂商或项目方自评只能作为具名来源的说法，第三方观察不能无条件升级为独立基准或普遍结论。",
                "supported：fact 的全部主体、关系、条件、因果、比较、成本、门槛、风险、限制和范围可由绑定 quote 直接核验；inference/recommendation 的每个前提有绑定 quote 支持，且推理没有加入新的领域关系。",
                "limitation 只有在绑定 quote 明确描述负向结果、代价、约束或失败条件时才能判 supported。功能、选项、接口、文档章节或实现存在，不能单独支持复杂性、成本、性能、风险、学习门槛或用户适用性。",
                "合理后果、常识推断或可能成立的解释，只要 quote 没有给出桥接关系，都判 overreach；它们可以成为后续检索假设，但不能进入 finding。",
                "声称文档、基准或本次研究‘没有提供/尚未验证’某内容，需要明确的研究过程或语料完整性证据；不能由一条引用里没写该内容推出。",
                "irrelevant：结论即使被证据支持，也没有实质回答 question 或任何确有证据的 criterion。",
                "ambiguous：证据冲突、缺失或无法可靠判断。unsupported：绑定证据明确不能支持该结论。",
                "必须逐条审计，不得替生成器补推理；每个 finding_index 返回且只返回一条 judgment。",
            ],
            "schema": {
                "judgments": [{
                    "finding_index": 0,
                    "status": "supported|unsupported|irrelevant|overreach|ambiguous",
                    "unsupported_components": [],
                    "supporting_evidence_ids": [],
                    "matched_criterion_ids": [],
                    "reason": "",
                }],
            },
        }
        if not model:
            result: dict[str, Any] = {}
        else:
            with audit_scope(
                stage="finding_synthesis",
                question_id=question_id or None,
                section_id=section_id or None,
            ):
                result = model(payload)
        if not isinstance(result, dict):
            result = {}
        review_failure_reason = str(result.get("_failure_reason") or "").strip()
        judgments_by_index: dict[int, dict[str, Any]] = {}
        for item in result.get("judgments") or []:
            if not isinstance(item, dict):
                continue
            try:
                local_index = int(item.get("finding_index"))
            except (TypeError, ValueError):
                continue
            if local_index not in global_index_by_local_index:
                continue
            index = global_index_by_local_index[local_index]
            status = str(item.get("status") or "").strip().lower()
            if status not in _FINDING_ADMISSION_STATUSES:
                continue
            if status == "supported":
                status = "admitted"
            unsupported_components = [
                str(value)[:240]
                for value in item.get("unsupported_components") or []
                if str(value).strip()
            ][:8]
            supporting_evidence_ids = [
                str(value)
                for value in item.get("supporting_evidence_ids") or []
                if str(value) in candidate_evidence_ids_by_index.get(local_index, set())
            ]
            matched_criterion_ids = [
                str(value)
                for value in item.get("matched_criterion_ids") or []
                if str(value) in valid_criterion_ids
            ]
            if status == "admitted" and (
                unsupported_components
                or not supporting_evidence_ids
                or (len(valid_criterion_ids) > 1 and not matched_criterion_ids)
            ):
                status = "ambiguous"
            judgments_by_index.setdefault(index, {
                "status": status,
                "unsupported_components": unsupported_components,
                "supporting_evidence_ids": supporting_evidence_ids,
                "matched_criterion_ids": matched_criterion_ids,
                "reason": str(item.get("reason") or "")[:500],
                "mode": "model" if status != "ambiguous" or supporting_evidence_ids else "model_incomplete",
            })
        judgments = []
        for index, finding in rows:
            judgment = judgments_by_index.get(index) or {
                "status": "ambiguous",
                "reason": (
                    f"independent finding admission unavailable: {review_failure_reason}"
                    if review_failure_reason
                    else "independent finding admission model did not return a valid judgment"
                ),
                "mode": "model_unavailable",
            }
            judgments.append({
                "finding_index": index,
                "finding_id": str(finding.get("finding_id") or ""),
                "question_id": question_id,
                "section_id": section_id,
                **judgment,
            })
        return question_id, judgments

    judgment_rows: list[dict[str, Any]] = []
    if grouped:
        with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(grouped)))) as executor:
            futures = {
                submit_with_audit_context(executor, review_question, question_id, rows): question_id
                for question_id, rows in grouped.items()
            }
            for future in as_completed(futures):
                _question_id, judgments = future.result()
                judgment_rows.extend(judgments)
    judgment_rows.sort(key=lambda item: int(item.get("finding_index") or 0))
    judgment_by_index = {
        int(item["finding_index"]): item
        for item in judgment_rows
    }
    admitted = []
    rejected = []
    for index, finding in enumerate(findings):
        judgment = judgment_by_index.get(index) or {
            "finding_index": index,
            "finding_id": str(finding.get("finding_id") or ""),
            "question_id": str(finding.get("question_id") or ""),
            "section_id": str(finding.get("section_id") or ""),
            "status": "ambiguous",
            "reason": "finding was not included in independent admission",
            "mode": "model_unavailable",
        }
        reviewed = {**finding, "admission_review": {
            key: judgment.get(key) for key in ("status", "reason", "mode", "matched_criterion_ids")
        }}
        if judgment.get("status") == "admitted":
            matched_criterion_ids = [
                str(value)
                for value in judgment.get("matched_criterion_ids") or []
                if str(value)
            ]
            if matched_criterion_ids:
                reviewed["criterion_ids"] = list(dict.fromkeys([
                    *[str(value) for value in finding.get("criterion_ids") or [] if str(value)],
                    *matched_criterion_ids,
                ]))
            admitted.append(reviewed)
        else:
            rejected.append(reviewed)

    admitted_question_ids = {
        str(item.get("question_id") or "")
        for item in admitted
    }
    failures = [
        {
            "section_id": str((state_by_question.get(question_id) or {}).get("section_id") or ""),
            "question_id": question_id,
            "failure_reason": "all_findings_rejected_by_independent_admission",
        }
        for question_id in grouped
        if question_id not in admitted_question_ids
    ]
    return {
        "status": "failed" if failures else "passed",
        "admitted_findings": admitted,
        "rejected_findings": rejected,
        "judgments": judgment_rows,
        "failures": failures,
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "review_unavailable_count": sum(item.get("mode") == "model_unavailable" for item in judgment_rows),
    }


def _restore_same_cluster_corroboration(
    bound_ids: list[str],
    candidate_ids: list[str],
    evidence: list[dict[str, Any]],
) -> list[str]:
    """Keep model-selected cross-source corroboration after atomic claim binding.

    ``atomize_finding`` narrows a multi-evidence finding to the rows whose wording best matches
    each atomic unit. That is correct for unrelated claims, but it can discard a second source
    carrying the same ``claim_cluster_id``. Restore only candidates from an already-bound
    cluster; this cannot pull a different angle or unrelated proposition into the finding.
    """
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence
        if str(item.get("id") or "")
    }
    cluster_ids = {
        str((evidence_by_id.get(evidence_id) or {}).get("claim_cluster_id") or "")
        for evidence_id in bound_ids
    }
    cluster_ids.discard("")
    output = list(dict.fromkeys(bound_ids))
    for evidence_id in candidate_ids:
        row = evidence_by_id.get(evidence_id) or {}
        if evidence_id in output or str(row.get("claim_cluster_id") or "") not in cluster_ids:
            continue
        output.append(evidence_id)
        if len(output) >= 3:
            break
    return output


def build_report_blueprint(
    *,
    user_request: str,
    research_plan: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]] | None = None,
    question_states: list[dict[str, Any]] | None = None,
    research_mode: str = "",
    evidence_landscape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_map = evidence_map or []
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence_map
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    evidence_landscape = evidence_landscape or classify_evidence_landscape(evidence_map)
    research_mode = research_mode or str(evidence_landscape.get("mode") or "evidence_informed_reference_design")
    section_findings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        section_findings[str(finding.get("section_id") or "")].append(finding)
    state_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in question_states or []:
        if isinstance(state, dict):
            state_by_section[str(state.get("section_id") or "")].append(state)
    sections = []
    for section in research_plan.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "")
        title = str(section.get("title") or section_id)
        render_as = _render_type(title, str(section.get("axis") or ""))
        current_findings = section_findings.get(section_id) or []
        current_states = state_by_section.get(section_id) or []
        core_section = _section_is_core(title, current_states)
        evidence_ids = list(dict.fromkeys(
            str(evidence_id)
            for finding in current_findings
            for evidence_id in finding.get("evidence_ids") or []
            if str(evidence_id)
        ))
        is_reference_metadata = (
            not core_section
            and "总览" not in title
            and is_non_research_section_title(title, has_questions=bool(current_states))
        )
        if is_reference_metadata:
            evidence_ids = list(evidence_by_id)
        source_domains = {
            urlparse(str((evidence_by_id.get(evidence_id) or {}).get("source_url") or "")).netloc
            for evidence_id in evidence_ids
        }
        source_domains.discard("")
        reportability = _section_reportability(
            title=title,
            finding_count=len(current_findings),
            evidence_count=len(evidence_ids),
            source_count=len(source_domains),
            core_section=core_section,
        )
        # `render_as` only frames `purpose` (a hint for what this section is fundamentally
        # about) for the writer prompt. It used to also force a mechanically generated
        # findings-dump table onto every "table"/"comparison_matrix"/etc-classified section
        # regardless of what the model wrote; that table generation was removed so the writer
        # model decides for itself, per section, whether a table communicates the content
        # better than prose (see writer.py's _model_section rules).
        sections.append({
            "section_id": section_id,
            "title": title,
            "purpose": str(section.get("intent") or _purpose(render_as)),
            "render_as": render_as,
            "writing_strategy": research_mode,
            "research_questions": _section_questions(section, current_states),
            "known_gaps": list(dict.fromkeys(
                str(gap)
                for state in current_states
                for gap in state.get("weak_aspects") or []
                if str(gap).strip()
            ))[:8],
            "finding_ids": [str(item.get("finding_id")) for item in current_findings],
            "evidence_ids": evidence_ids,
            "reportability": reportability,
        })
    completeness = _report_completeness(sections)
    return {
        "report_type": _report_type(user_request),
        "audience": "工程实现与研究决策" if any(token in user_request for token in ("实现", "架构", "方案")) else "研究阅读与审核",
        "thesis": _thesis(user_request, findings),
        "style": "report_first_question_loop",
        "research_mode": research_mode,
        "evidence_landscape": evidence_landscape,
        "sections": sections,
        "completeness": completeness,
    }


def classify_evidence_landscape(evidence_map: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [item for item in evidence_map if isinstance(item, dict)]
    domains = {
        urlparse(str(item.get("source_url") or "")).netloc
        for item in rows
        if str(item.get("source_url") or "")
    }
    domains.discard("")
    domain_specific = [item for item in rows if evidence_target_fit(item) == "exact_target"]
    direct = [
        item for item in rows
        if is_direct_exact_evidence(item)
    ]
    case_studies = [item for item in rows if str(item.get("source_type") or "") == "case_study"]
    ratio = len(domain_specific) / len(rows) if rows else 0.0
    if len(direct) >= 4 and ratio >= 0.35:
        mode = "domain_practice_review"
    elif direct or domain_specific or case_studies:
        mode = "hybrid_practice_and_reference"
    else:
        mode = "evidence_informed_reference_design"
    return {
        "mode": mode,
        "evidence_count": len(rows),
        "source_count": len(domains),
        "domain_specific_count": len(domain_specific),
        "direct_fact_count": len(direct),
        "case_study_count": len(case_studies),
        "domain_specific_ratio": round(ratio, 3),
    }


def _section_questions(section: dict[str, Any], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    state_by_id = {str(state.get("question_id") or ""): state for state in states}
    raw_questions = section.get("questions") or []
    for index, raw in enumerate(raw_questions):
        if isinstance(raw, dict):
            question = str(raw.get("text") or raw.get("question") or raw.get("title") or "").strip()
            purpose = str(raw.get("purpose") or "").strip()
            question_id = str(raw.get("id") or raw.get("question_id") or f"q{index + 1}")
        else:
            question = str(raw or "").strip()
            purpose = ""
            question_id = f"q{index + 1}"
        if question:
            state = state_by_id.get(question_id) or {}
            rows.append({
                "question_id": question_id,
                "question": question,
                "purpose": purpose,
                "criteria": _question_criteria(state),
            })
    if rows:
        return rows
    return [
        {
            "question_id": str(state.get("question_id") or f"q{index + 1}"),
            "question": str(state.get("question") or "").strip(),
            "purpose": str(state.get("purpose") or "").strip(),
            "criteria": _question_criteria(state),
        }
        for index, state in enumerate(states)
        if str(state.get("question") or "").strip()
    ]


def _question_criteria(state: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "criterion_id": str(item.get("criterion_id") or ""),
            "description": str(item.get("description") or ""),
            "level": str(item.get("level") or "hard"),
            "status": str(item.get("status") or "uncovered"),
        }
        for item in state.get("criteria") or []
        if isinstance(item, dict) and str(item.get("description") or "").strip()
    ]


def _section_is_core(title: str, states: list[dict[str, Any]]) -> bool:
    criteria = [
        criterion
        for state in states
        for criterion in state.get("criteria") or []
        if isinstance(criterion, dict)
    ]
    if any(
        str(criterion.get("level") or "hard") == "hard"
        for criterion in criteria
    ):
        return True
    if criteria:
        return False
    # Compatibility for older/hand-built blueprints without serialized criteria. A real question
    # is a stronger structural signal than words such as "证据" or "引用" appearing in its axis.
    return not is_non_research_section_title(title, has_questions=bool(states))


def _section_reportability(
    *,
    title: str,
    finding_count: int,
    evidence_count: int,
    source_count: int,
    core_section: bool,
) -> dict[str, Any]:
    if "总览" in title:
        status = "ready" if finding_count else "limited"
        reason = "overview is derived from report findings" if finding_count else "no findings available for overview"
    elif not core_section and is_non_research_section_title(title, has_questions=False):
        status = "ready" if evidence_count else "empty"
        reason = "source index can be rendered" if evidence_count else "no traceable source is available"
    elif finding_count >= 2 and evidence_count >= 2:
        status = "ready"
        reason = "section has multiple findings and supporting evidence"
    elif finding_count and evidence_count:
        status = "limited"
        reason = "section can be written but lacks breadth"
    else:
        status = "empty"
        reason = "section has no body-worthy finding"
    return {
        "status": status,
        "finding_count": finding_count,
        "evidence_count": evidence_count,
        "source_count": source_count,
        "core_section": core_section,
        "reason": reason,
    }


def _report_completeness(sections: list[dict[str, Any]]) -> dict[str, Any]:
    core_sections = [
        section for section in sections
        if bool((section.get("reportability") or {}).get("core_section"))
    ]
    limited = [
        str(section.get("section_id") or section.get("title") or "")
        for section in sections
        if str((section.get("reportability") or {}).get("status") or "empty") == "limited"
    ]
    core_empty = [
        str(section.get("section_id") or section.get("title") or "")
        for section in core_sections
        if str((section.get("reportability") or {}).get("status") or "empty") == "empty"
    ]
    supplemental_empty = [
        str(section.get("section_id") or section.get("title") or "")
        for section in sections
        if not bool((section.get("reportability") or {}).get("core_section"))
        and str((section.get("reportability") or {}).get("status") or "empty") == "empty"
    ]
    status = "complete_report" if core_sections and not core_empty else "partial_report"
    return {
        "status": status,
        "required_section_count": len(core_sections),
        "ready_section_count": len(core_sections) - len(core_empty),
        "limited_section_ids": limited,
        "core_empty_section_ids": core_empty,
        "supplemental_empty_section_ids": supplemental_empty,
        "reason": "all core sections contain reportable content" if not core_empty else "some core sections are empty",
    }


def _model_findings(
    state: dict[str, Any],
    evidence: list[dict[str, Any]],
    model: JsonModel,
    *,
    research_mode: str,
) -> tuple[list[dict[str, Any]], str]:
    prompt_variant = str(os.getenv("RESEARCH_FINDING_PROMPT_VARIANT", "grounded_reason_v2")).strip().lower()
    if prompt_variant not in {"current", "grounded_reason_v2", "grounded_relevance_v3"}:
        raise ValueError(f"Unsupported finding prompt variant: {prompt_variant}")
    criterion_ledger = _finding_criterion_ledger(state)
    evidence_payload = _select_finding_evidence(
        evidence,
        question=str(state.get("question") or ""),
        criterion_ledger=criterion_ledger,
        include_quotes=prompt_variant in {"grounded_reason_v2", "grounded_relevance_v3"},
    )
    payload = {
        "task": "Synthesize atomic evidence into analytical findings for a research report. Return JSON only.",
        "question": state.get("question"),
        "success_criteria": state.get("success_criteria") or [],
        "criterion_ledger": criterion_ledger,
        "research_mode": research_mode,
        "rules": [
            "Use only the supplied evidence.",
            "Write every finding and limitation in Chinese; keep product names in their original form.",
            "A finding must express a conclusion, not repeat a quote.",
            "Each finding must contain exactly one independently verifiable proposition.",
            "Do not combine architecture, comparison, cost, risk, and recommendation in one finding.",
            "Keep each finding under 180 Chinese characters and do not use semicolons to join conclusions.",
            "Preserve limitations and conflicts.",
            "Every finding must list supporting evidence_ids chosen from an evidence group's evidence_ids. A criterion_id may be used only when it also appears in the bound evidence group's matched_criterion_ids.",
            "Treat multiple sources inside one claim cluster as corroboration for one conclusion, not as a reason to repeat the same finding.",
            "Claims about different mechanisms, scopes, implementation steps, limitations, or exceptions are distinct angles even when they concern the same product; do not collapse those angles merely because some words overlap.",
            "For every criterion that the supplied evidence substantively supports, return at least one finding linked to that criterion_id. Do not invent a finding for an unsupported criterion.",
            "An evidence group with empty matched_criterion_ids is background/context for this question. Do not turn it into a criterion answer or use it to satisfy a comparison axis.",
            "Classify every finding as fact, inference, recommendation, or limitation.",
            "General or cross-domain evidence cannot support a domain-specific fact; use inference or recommendation.",
            "Preserve each evidence subject exactly. Evidence marked comparison, general, unknown, incompatible_subject, or unverified_subject must not be rewritten as a fact about the question target.",
            "When research_mode is hybrid_practice_and_reference or evidence_informed_reference_design, use adjacent evidence rather than discarding it.",
            "In reference-design modes, distinguish evidence observations, transferable implications, and candidate design recommendations.",
            "Do not claim that a transferable pattern has already been deployed in the target domain.",
            "Return the number of non-duplicative findings needed to cover the supported material dimensions. Never return more than 8 findings; do not create filler to reach a count.",
        ],
        "evidence": evidence_payload,
        "schema": {"findings": [{"finding": "", "finding_type": "fact|inference|recommendation|limitation", "criterion_ids": [], "evidence_ids": [], "counter_evidence_ids": [], "limitations": [], "confidence": "high|medium|low"}]},
    }
    if prompt_variant in {"grounded_reason_v2", "grounded_relevance_v3"}:
        payload["rules"][2:3] = [
            "fact 可以忠实压缩 evidence 中已经存在的关系，不要求为了显得分析性而增加新含义。",
            "每条 finding 的全部主体、关系、条件、因果、比较、成本、门槛、风险、限制和适用性都必须能由其 evidence_ids 对应的 claim 与 quote 核验。",
            "能力、功能、配置项、文档章节或扩展接口的存在，不能单独推出学习成本、部署复杂性、维护负担、性能风险或不适合某类用户。",
            "limitation 必须有证据明确描述负向结果、代价、约束或失败条件。inference/recommendation 必须列出 premises、reasoning 和 uncertainty；缺少桥接证据时不要输出该结论。",
            "support_reason 必须具体说明每个 evidence_id 支持 finding 中哪个关系；不得只写‘证据支持该结论’。",
        ]
        payload["schema"] = {
            "findings": [{
                "finding": "",
                "finding_type": "fact|inference|recommendation|limitation",
                "criterion_ids": [],
                "evidence_ids": [],
                "counter_evidence_ids": [],
                "support_reason": "",
                "premises": [{"evidence_id": "ev_001", "premise": ""}],
                "reasoning": "",
                "uncertainty": "",
                "limitations": [],
                "confidence": "high|medium|low",
            }],
        }
        if prompt_variant == "grounded_relevance_v3":
            payload["rules"].insert(
                3,
                "finding 不仅要真实，还必须直接回答当前 question 或一个确实被 evidence 支持的 criterion。与主题相关但不回答当前问题的能力事实不要输出。",
            )
            payload["rules"].insert(
                4,
                "当 question 或 criterion 询问限制、成本、风险、复杂性或适用性，而 evidence 只说明功能、选项、文档或实现存在时，该维度应视为未被支持；不要用一般常识补成答案，也不要为了覆盖 criterion 而返回无关事实。",
            )
            for index, rule in enumerate(payload["rules"]):
                if str(rule).startswith("For every criterion that the supplied evidence"):
                    payload["rules"][index] = (
                        "For each criterion, return a finding only when the supplied evidence directly supports the relation requested by that criterion. "
                        "An evidence row matched by topic words does not count as substantive support."
                    )
                    break
    result = model(payload)
    findings = [
        item
        for item in result.get("findings") or []
        if isinstance(item, dict) and str(item.get("finding") or "").strip()
    ]
    findings = _limit_model_findings(findings, state=state, evidence=evidence)
    failure_reason = str(result.get("_failure_reason") or ("empty_findings" if not findings else ""))
    return findings, failure_reason


def _limit_model_findings(
    findings: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    evidence: list[dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    if len(findings) <= limit:
        return findings
    ledger = _finding_criterion_ledger(state)
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence}
    selected: list[dict[str, Any]] = []

    def add(item: dict[str, Any]) -> bool:
        claim = str(item.get("finding") or "").strip()
        if not claim or any(
            claim_similarity(claim, str(current.get("finding") or "")) >= 0.78
            for current in selected
        ):
            return False
        selected.append(item)
        return True

    ordered_criteria = sorted(
        ledger,
        key=lambda item: str(item.get("description") or "").startswith("实质回答研究问题"),
    )
    for pass_index in range(2):
        for criterion in ordered_criteria:
            criterion_id = str(criterion.get("criterion_id") or "")
            description = str(criterion.get("description") or "")
            candidates = [
                item for item in findings
                if item not in selected
                and (
                    criterion_id in {str(value) for value in item.get("criterion_ids") or []}
                    or _finding_text_overlap(str(item.get("finding") or ""), description) > 0
                )
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: _generated_finding_score(
                    item,
                    target=description,
                    criterion_id=criterion_id,
                    evidence_by_id=evidence_by_id,
                ),
                reverse=True,
            )
            added = 0
            for item in candidates:
                if add(item):
                    added += 1
                    break
            if len(selected) >= limit:
                return selected
            if pass_index == 0 and not added:
                continue

    question = str(state.get("question") or "")
    for item in sorted(
        findings,
        key=lambda candidate: _generated_finding_score(
            candidate,
            target=question,
            criterion_id="",
            evidence_by_id=evidence_by_id,
        ),
        reverse=True,
    ):
        add(item)
        if len(selected) >= limit:
            break
    return selected


def _generated_finding_score(
    item: dict[str, Any],
    *,
    target: str,
    criterion_id: str,
    evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[float, int, float, float]:
    rows = [
        evidence_by_id[evidence_id]
        for evidence_id in [str(value) for value in item.get("evidence_ids") or []]
        if evidence_id in evidence_by_id
    ]
    return (
        _finding_text_overlap(str(item.get("finding") or ""), target),
        1 if criterion_id and criterion_id in {str(value) for value in item.get("criterion_ids") or []} else 0,
        max((float(row.get("directness_score") or 0) for row in rows), default=0.0),
        max((float(row.get("evidence_score") or 0) for row in rows), default=0.0),
    )


def _finding_criterion_ledger(state: dict[str, Any]) -> list[dict[str, str]]:
    ledger = []
    for item in state.get("criteria") or []:
        if not isinstance(item, dict):
            continue
        criterion_id = str(item.get("criterion_id") or "").strip()
        description = str(item.get("description") or "").strip()
        if not criterion_id or not description:
            continue
        ledger.append({
            "criterion_id": criterion_id,
            "description": description,
            "level": str(item.get("level") or "unspecified"),
            "status": str(item.get("status") or "unknown"),
        })
    if ledger:
        return ledger
    for index, description in enumerate(state.get("success_criteria") or [], start=1):
        value = str(description or "").strip()
        if value:
            ledger.append({
                "criterion_id": f"criterion_{index:02d}",
                "description": value,
                "level": "unspecified",
                "status": "unknown",
            })
    return ledger


def _select_finding_evidence(
    evidence: list[dict[str, Any]],
    *,
    question: str,
    criterion_ledger: list[dict[str, str]],
    cap: int = _FINDING_EVIDENCE_CLUSTER_CAP,
    include_quotes: bool = False,
) -> list[dict[str, Any]]:
    """Build a criterion-balanced, cluster-aware finding payload.

    Evidence admission already merges same-source near duplicates and assigns corroborating
    cross-source rows the same ``claim_cluster_id``. Finding synthesis should consume that work
    instead of taking the first N rows by arrival order. Each cluster remains one candidate
    conclusion while all supporting row IDs and source variants stay available to the model.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cluster_order: list[str] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        cluster_id = str(
            item.get("claim_cluster_id")
            or item.get("claim_fingerprint")
            or item.get("id")
            or f"row_{index:04d}"
        )
        if cluster_id not in grouped:
            cluster_order.append(cluster_id)
        grouped[cluster_id].append(item)

    groups = [
        _finding_evidence_group(
            cluster_id,
            grouped[cluster_id],
            criterion_ledger,
            include_quotes=include_quotes,
        )
        for cluster_id in cluster_order
    ]
    groups = [item for item in groups if item.get("claim")]
    if len(groups) <= cap:
        return groups

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(group: dict[str, Any]) -> None:
        cluster_id = str(group.get("claim_cluster_id") or "")
        if not cluster_id or cluster_id in selected_ids or len(selected) >= cap:
            return
        selected_ids.add(cluster_id)
        selected.append(group)

    ordered_criteria = sorted(
        criterion_ledger,
        key=lambda item: str(item.get("description") or "").startswith("实质回答研究问题"),
    )
    for criterion in ordered_criteria:
        description = str(criterion.get("description") or "")
        ranked = sorted(
            groups,
            key=lambda group: _finding_group_score(group, description),
            reverse=True,
        )
        added = 0
        for group in ranked:
            if _finding_group_term_overlap(group, description) <= 0:
                continue
            before = len(selected)
            add(group)
            if len(selected) > before:
                added += 1
            if added >= _FINDING_CLUSTERS_PER_CRITERION or len(selected) >= cap:
                break
        if len(selected) >= cap:
            break

    combined_target = " ".join([
        question,
        *[str(item.get("description") or "") for item in criterion_ledger],
    ])
    source_urls = list(dict.fromkeys(
        str(url)
        for group in groups
        for url in group.get("source_urls") or []
        if str(url)
    ))
    for source_url in source_urls:
        source_groups = [group for group in groups if source_url in (group.get("source_urls") or [])]
        for group in sorted(
            source_groups,
            key=lambda item: (
                _finding_group_term_overlap(item, combined_target),
                *_finding_evidence_group_quality(item),
            ),
            reverse=True,
        )[:2]:
            add(group)
            if len(selected) >= cap:
                break
        if len(selected) >= cap:
            break

    for group in sorted(groups, key=lambda item: _finding_group_score(item, combined_target), reverse=True):
        if _finding_group_term_overlap(group, combined_target) <= 0:
            continue
        add(group)
        if len(selected) >= cap:
            break
    if not selected:
        for group in sorted(groups, key=_finding_evidence_group_quality, reverse=True)[:cap]:
            add(group)
    return selected


def _finding_evidence_group(
    cluster_id: str,
    rows: list[dict[str, Any]],
    criterion_ledger: list[dict[str, str]],
    *,
    include_quotes: bool = False,
) -> dict[str, Any]:
    representative = max(rows, key=_finding_evidence_quality)
    claims = list(dict.fromkeys(
        str(item.get("atomic_claim") or item.get("claim") or "").strip()
        for item in rows
        if str(item.get("atomic_claim") or item.get("claim") or "").strip()
    ))
    source_urls = list(dict.fromkeys(
        str(item.get("source_url") or "")
        for item in rows
        if str(item.get("source_url") or "")
    ))
    claim_text = " ".join(claims)
    valid_criterion_ids = {
        str(item.get("criterion_id") or "")
        for item in criterion_ledger
        if str(item.get("criterion_id") or "")
    }
    explicit_criterion_ids = list(dict.fromkeys(
        str(value)
        for row in rows
        if str(row.get("criterion_binding_status") or "") not in {
            "incompatible_subject",
            "unverified_subject",
        }
        and str(row.get("comparison_axis_binding_status") or "") not in {
            "incompatible_axis",
            "unverified_axis",
        }
        for value in row.get("criterion_ids") or []
        if str(value) in valid_criterion_ids
    ))
    subject_binding_blocked = any(
        str(row.get("criterion_binding_status") or "") in {
            "incompatible_subject",
            "unverified_subject",
        }
        or str(row.get("comparison_axis_binding_status") or "") in {
            "incompatible_axis",
            "unverified_axis",
        }
        for row in rows
    )
    has_current_binding = any("criterion_binding_status" in row for row in rows)
    if explicit_criterion_ids or subject_binding_blocked or has_current_binding:
        matched_criterion_ids = explicit_criterion_ids[:3]
    else:
        matched_criteria = sorted(
            (
                (_finding_text_overlap(claim_text, str(item.get("description") or "")), str(item.get("criterion_id") or ""))
                for item in criterion_ledger
                if str(item.get("criterion_id") or "")
            ),
            reverse=True,
        )
        matched_criterion_ids = [criterion_id for score, criterion_id in matched_criteria[:3] if score > 0]
    group = {
        "id": representative.get("id"),
        "claim_cluster_id": cluster_id,
        "claim": claims[0] if claims else "",
        "claim_variants": claims[1:3],
        "evidence_ids": [str(item.get("id")) for item in rows if item.get("id")],
        "source_urls": source_urls,
        "source_count": len(source_urls),
        "matched_criterion_ids": matched_criterion_ids,
        "subject": representative.get("subject"),
        "subject_relation": representative.get("subject_relation"),
        "criterion_binding_status": representative.get("criterion_binding_status"),
        "comparison_axis": representative.get("comparison_axis"),
        "comparison_metric": representative.get("comparison_metric"),
        "expected_comparison_axis": representative.get("expected_comparison_axis"),
        "comparison_axis_binding_status": representative.get("comparison_axis_binding_status"),
        "stance": representative.get("stance"),
        "source_type": representative.get("source_type"),
        "scope": representative.get("scope"),
        "claim_type": representative.get("claim_type"),
        "directness_score": representative.get("directness_score"),
        "evidence_score": representative.get("evidence_score"),
    }
    if representative.get("support_relation") and representative.get("target_fit"):
        group.update({
            "support_relation": evidence_support_relation(representative),
            "target_fit": evidence_target_fit(representative),
        })
    if include_quotes:
        group["quotes"] = [
            {
                "evidence_id": str(item.get("id") or ""),
                "quote": str(
                    item.get("raw_quote")
                    or item.get("cleaned_quote")
                    or item.get("evidence_text")
                    or ""
                ).strip()[:500],
            }
            for item in rows[:3]
            if str(
                item.get("raw_quote")
                or item.get("cleaned_quote")
                or item.get("evidence_text")
                or ""
            ).strip()
        ]
    return group


def _finding_evidence_quality(item: dict[str, Any]) -> tuple[float, float, float]:
    return (
        1.0 if is_direct_exact_evidence(item) else 0.5 if evidence_support_relation(item) == "inference_support" else 0.0,
        float(item.get("evidence_score") or 0),
        1.0 if item.get("grounded_exactly") else 0.0,
    )


def _finding_group_score(group: dict[str, Any], target: str) -> tuple[float, float, float, int]:
    return (
        _finding_group_term_overlap(group, target),
        float(group.get("directness_score") or 0),
        float(group.get("evidence_score") or 0),
        int(group.get("source_count") or 0),
    )


def _finding_evidence_group_quality(group: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(group.get("directness_score") or 0),
        float(group.get("evidence_score") or 0),
        int(group.get("source_count") or 0),
    )


def _finding_group_term_overlap(group: dict[str, Any], target: str) -> float:
    text = " ".join([
        str(group.get("claim") or ""),
        *[str(item) for item in group.get("claim_variants") or []],
    ])
    return _finding_text_overlap(text, target)


def _finding_text_overlap(left: str, right: str) -> float:
    left_terms = extract_terms(left)
    right_terms = {
        str(term).lower()
        for term in extract_key_terms(right, top_k=12)
        if str(term).lower() not in _FINDING_GENERIC_TERMS
    }
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, len(right_terms))


def _confidence(evidence: list[dict[str, Any]]) -> str:
    domains = {str(item.get("source_url") or "").split("/")[2] for item in evidence if "/" in str(item.get("source_url") or "")}
    clusters = {str(item.get("claim_cluster_id") or item.get("atomic_claim") or "") for item in evidence}
    scores = [float(item.get("evidence_score") or 0.5) for item in evidence]
    average_score = sum(scores) / len(scores) if scores else 0.0
    if len(clusters) >= 4 and len(domains) >= 3 and average_score >= 0.85:
        return "high"
    if len(clusters) >= 2 and len(domains) >= 2 and average_score >= 0.65:
        return "medium"
    return "low"


def _lower_confidence(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    left_value = left if left in order else "low"
    right_value = right if right in order else "low"
    return left_value if order[left_value] <= order[right_value] else right_value


def _finding_type(requested: str, evidence_ids: list[str], evidence: list[dict[str, Any]]) -> tuple[str, str]:
    rows = [item for item in evidence if str(item.get("id") or "") in set(evidence_ids)]
    target_fits = {evidence_target_fit(item) for item in rows}
    support_scope = "domain_specific" if target_fits == {"exact_target"} else "cross_domain" if "related_target" in target_fits or "exact_target" in target_fits else "general"
    value = requested if requested in {"fact", "inference", "recommendation", "limitation"} else "inference"
    if value == "fact" and (not rows or not all(_is_direct_evidence(item) for item in rows)):
        value = "inference"
    return value, support_scope


def _is_direct_evidence(item: dict[str, Any]) -> bool:
    return is_direct_exact_evidence(item)


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for item in findings:
        key = "".join(str(item.get("finding") or "").lower().split())[:160]
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _render_type(title: str, axis: str) -> str:
    if "总览" in title:
        return "executive_summary"
    if axis == "horizontal" or any(token in title for token in ("对比", "选型", "竞品")):
        return "comparison_matrix"
    if any(token in title for token in ("演化", "历史", "时间")):
        return "timeline"
    if any(token in title for token in ("路径", "实施", "落地", "步骤")):
        return "roadmap"
    if any(token in title for token in ("风险", "限制", "挑战", "评估")):
        return "risk_matrix"
    if any(token in title for token in ("架构", "角色", "模块", "机制")):
        return "table"
    return "narrative"


def _purpose(render_as: str) -> str:
    return {
        "comparison_matrix": "集中展示对象或方案之间的关键差异与权衡。",
        "timeline": "展示关键阶段、节点和演化关系。",
        "roadmap": "按阶段组织实施动作、依据和验证点。",
        "risk_matrix": "展示风险、影响、证据和验证方式。",
        "table": "结构化展示模块、职责和证据。",
    }.get(render_as, "围绕研究问题形成递进分析。")


def _report_type(user_request: str) -> str:
    if any(token in user_request for token in ("对比", "比较", "选型", "vs")):
        return "comparison"
    if any(token in user_request for token in ("实现", "架构", "开发", "方案")):
        return "implementation_guide"
    return "research_report"


def _thesis(user_request: str, findings: list[dict[str, Any]]) -> str:
    first = str((findings[0] if findings else {}).get("finding") or "").strip()
    return first or f"本报告围绕“{user_request}”形成证据约束的综合判断。"
