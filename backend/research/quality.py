"""Report-quality rubric and combined approval gate."""

from __future__ import annotations

import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .audit import submit_with_audit_context
from .claims import is_atomic_claim
from .evidence import evidence_support_relation, evidence_target_fit, is_direct_exact_evidence
from .writer import _is_reference_title


ENTAILMENT_PROMPT_VERSION = "claim-entailment/v2"
ENTAILMENT_PAYLOAD_PROFILE_DEFAULT = "default"
ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2 = "atomic_boundary_v2"
ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION = "claim-entailment/atomic-boundary-isolation/v2"


def _semantic_evidence_payload(evidence: dict[str, Any]) -> dict[str, str]:
    if not evidence.get("support_relation") or not evidence.get("target_fit"):
        return {}
    return {
        "support_relation": evidence_support_relation(evidence),
        "target_fit": evidence_target_fit(evidence),
    }


def evaluate_report_quality(
    *,
    report_document: dict[str, Any],
    question_states: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    rejected_evidence: list[dict[str, Any]],
    documents: list[dict[str, Any]] | None = None,
    model: Callable[..., dict[str, Any]] | None = None,
    section_critic_budgets: dict[str, float] | None = None,
    section_critic_parallelism: int = 6,
    evidence_map: list[dict[str, Any]] | None = None,
    prior_report_quality: dict[str, Any] | None = None,
    support_units: list[dict[str, Any]] | None = None,
    support_unit_kind: str = "finding",
) -> dict[str, Any]:
    content = str(report_document.get("content") or "")
    sections = len(re.findall(r"^##\s+", content, re.M))
    citations = len(re.findall(r"\[\d+\]", content))
    noise_hits = _noise_hits(content)
    sufficient = sum(str(item.get("status") or "") == "sufficient" for item in question_states)
    total = len(question_states)
    completeness = report_document.get("completeness") if isinstance(report_document.get("completeness"), dict) else {}
    report_status = str(report_document.get("delivery_status") or completeness.get("status") or "complete_report")
    limitation_disclosed = any(token in content.lower() for token in (
        "现有证据", "证据不足", "证据缺口", "缺乏直接", "不确定性", "无法确认",
        "available evidence", "evidence gap", "lack of direct", "insufficient evidence", "uncertain",
    ))
    transparent_degradation = (
        report_status == "complete_report"
        and citations >= max(3, total)
        and limitation_disclosed
    )
    fallback_sections = [
        str(document.get("title") or "")
        for document in documents or []
        if isinstance(document.get("synthesis_diagnostics"), dict)
        and document["synthesis_diagnostics"].get("mode") == "fallback_template"
        and not _is_harmless_reference_fallback(document)
    ]
    sanitized_evidence_id_count = int(report_document.get("sanitized_evidence_id_count") or 0)
    effective_support_units = findings if support_units is None else support_units
    scores = {
        "structure": 5 if sections >= 4 else 4 if sections >= 3 else 2,
        "synthesis": 5 if len(effective_support_units) >= max(4, total) else 4 if effective_support_units else 1,
        "evidence_quality": 5 if citations >= len(effective_support_units) and not noise_hits else 3 if citations else 1,
        "readability": 5 if "[ev_" not in content and "原始证据表述" not in content else 2,
        # Tables are now the writer model's own judgment call (writer.py no longer forces one
        # per section by title keyword), so a report with zero tables is not inherently worse
        # than one with tables — this deterministic heuristic stays a mild, non-punitive signal
        # and defers the real "was this the right call" judgment to the critic model below.
        "information_design": 5 if "|---" in content else 4,
        "limitations": 5 if any(token in content for token in ("限制", "边界", "风险", "不确定")) else 3,
    }
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map or [] if item.get("id")}
    critic_result = _run_section_critics(
        documents or [], effective_support_units, model, section_critic_budgets, evidence_by_id,
        prior_section_diagnostics=(prior_report_quality or {}).get("section_critic_diagnostics") or [],
        parallelism=section_critic_parallelism,
        support_unit_kind=support_unit_kind,
    )
    critic_scores = critic_result["scores"]
    for key in scores:
        try:
            if key in critic_scores:
                scores[key] = min(scores[key], max(1, min(5, int(critic_scores[key]))))
        except (TypeError, ValueError):
            pass
    average = round(sum(scores.values()) / len(scores), 2)
    blocking = []
    if noise_hits:
        blocking.append("reader_content_contains_noise")
    if "[ev_" in content:
        blocking.append("reader_content_exposes_evidence_ids")
    if fallback_sections:
        # A section that silently degraded to the deterministic template (see writer.py's
        # _fallback_section) is not deliverable prose; never let it pass the quality gate
        # unnoticed just because the report otherwise scores well.
        blocking.append("contains_fallback_template_section")
    if sanitized_evidence_id_count:
        # The report was cleaned of leaked internal evidence ids before delivery, but that leak
        # is still a real defect (see writer._sanitize_leaked_evidence_ids) and should still
        # block approval rather than being silently fixed away.
        blocking.append("evidence_id_leak_sanitized")
    # 2026-08-17: removed reader_language_inconsistent check (pure character ratio can't distinguish
    # legitimate technical terms like "Harness"/"Node.js"/"API token" from machine-translation noise;
    # real readability issues are caught by human review, not a 35% threshold that flags every
    # domain-heavy technical report).
    # Question sufficiency ratio and report completeness remain diagnostics. Approval is decided
    # later from hard criterion gaps, core empty sections, fallback output and fact-safety checks.
    # 2026-08-17 (architecture decision, see docs/plans/writer-critic-loop-redesign):
    # the 6-dimension critic (structure/synthesis/evidence_quality/readability/
    # information_design/limitations) is a subjective model judgment that proved to
    # oscillate across revision passes without converging (a98ba6 dead loop, 54-73%
    # cross-pass recurrence per a 10-job audit). It no longer blocks approval -- its
    # scores/issues/summary still populate below, but purely as material for the
    # report's appended "局限性声明" (limitations disclosure), not as a gate. `average`
    # (which folds in critic scores via the min() merge above) is kept in the return
    # value for that disclosure's stats, but must not itself gate `status`, or a low
    # critic score would still silently block through this second path even with the
    # explicit dimension-threshold bullet removed.
    critic_refuted_issues = critic_result.get("refuted_issues") or []
    critic_summary = critic_result["summary"][:500]
    if critic_refuted_issues:
        critic_summary = (
            f"评论完成：保留 {len(critic_result['issues'])} 条建议；"
            f"{len(critic_refuted_issues)} 条证据类建议经独立复核排除。"
        )
    return {
        "status": "passed" if not blocking else "failed",
        "average_score": average,
        "scores": scores,
        "blocking_issues": blocking,
        "noise_hits": noise_hits,
        "fallback_sections": fallback_sections,
        "sanitized_evidence_id_count": sanitized_evidence_id_count,
        "sufficient_question_count": sufficient,
        "question_count": total,
        "transparent_degradation_accepted": bool(total and sufficient / total < 0.5 and transparent_degradation),
        "limitation_disclosed": limitation_disclosed,
        "report_status": report_status,
        "report_completeness": completeness,
        "rejected_evidence_count": len(rejected_evidence),
        "issues": critic_result["issues"][:8],
        "critic_refuted_issue_count": len(critic_refuted_issues),
        "critic_refuted_issues": critic_refuted_issues[:8],
        "critic_summary": critic_summary,
        "critic_raw_summary": critic_result["summary"][:500],
        "section_critic_diagnostics": critic_result["section_diagnostics"],
        # Purely observational -- see _recheck_low_confidence_issues docstring. Lets a real job's
        # validation_diagnostics show how often low-confidence critic issues actually get
        # rechecked/refuted/left undetermined, to inform whether this needs throttling later.
        "critic_confidence_recheck": critic_result.get("confidence_recheck") or {},
        "critic_evidence_recheck": critic_result.get("evidence_recheck") or {},
    }


_CRITIC_RUBRIC = ["structure", "synthesis", "evidence_quality", "readability", "information_design", "limitations"]
_CRITIC_RULES = [
    "Score every dimension from 1 to 5.",
    "Identify evidence dumping, repetition, weak narrative, unsupported synthesis, and missing tradeoffs.",
    "For information_design, judge whether each table (if any) earns its place: a table used for a genuine multi-option or multi-dimension comparison is good; a table that merely restates rows already covered in nearby prose, or a section that clearly needed a comparison table but stayed prose-only, should lower the score.",
    "Each fixable issue should name a section_id when possible.",
    "Each issue must also include a `confidence` field from 0 to 100: how certain you are that the issue is a real problem in the text, not a plausible-sounding but wrong flag. Reserve 70+ for issues you are confident about; use a lower number when you are guessing or the text is ambiguous.",
    # The 8 rules below are a precision-picked subset of the writer's own ~25 drafting rules
    # (backend/research/writer.py::_model_section payload["rules"]) -- the ones that are
    # actually judgeable as quality-good-or-bad from the rendered text alone, as opposed to pure
    # execution mechanics (citation marker syntax, language switching, target_length_chars,
    # section_dossier field names, mode_rules/revision_rules scaffolding). Giving the critic this
    # subset closes the rubric-alignment gap where the writer was held to ~25 detailed rules but
    # the critic only knew 4 generic ones and couldn't recognize a violation of any of them. See
    # docs/modules/deep-research-step3-writing-and-revision-discussion-v0.1.md, 商榷点 B 风险 B
    # (verbatim from the collapsed decision there -- do not reword or add/remove items here
    # without re-opening that discussion).
    "段首 1-2 句给出本节核心结论，先论点后论据（structure）",
    "相关命题合并成连贯段落，不是逐句孤立罗列（readability/synthesis）",
    "不要仅因解释、综合或推断没有引用就把它判为 Citation Error；未引用分析应从逻辑、范围、不确定性表达、是否引入新的外部可核查事实以及是否与材料矛盾等整体质量角度评价。只有新增外部事实却没有来源，或把分析伪装成来源已经证明的内容时，才作为 evidence_quality issue。",
    "对带引用的内容，关注引用是否被用来替完整 cited span 背书；精确的引用编号映射和逐条语义支持关系由 Citation Validator 与 Claim Entailment Validator 裁决，critic 不得因为正文存在未引用分析而猜测引用错误。",
    "fact/inference/recommendation 要用对应限定语言，不能把推断包装成事实（evidence_quality，与 entailment 的 label_mismatch 判据呼应，critic 可从表层语言先做一道初筛）",
    "证据只覆盖单案例/单来源时，结论范围不能泛化到\"所有系统\"（evidence_quality/scope）",
    "不加没有证据支撑的评价性判断（\"值得关注\"类空判断）（evidence_quality）",
    "表格使用是否合理（有无必要、有无与正文重复）（information_design）",
    "权衡/边界/不确定性应融入论证，不是贴一段免责声明（limitations）",
    # 2026-08-12: the critic previously only saw the rendered prose (`report`) and paraphrased
    # `available_findings`, with no access to the raw evidence (quote/scope/source) the writer
    # itself was given (writer.py's _model_section payload["evidence"]) -- so a rule like the
    # scope-generalization one above ("证据只覆盖单案例/单来源时...") was structurally unverifiable:
    # the critic had no evidence data to check the claim against and could only guess from surface
    # phrasing. `evidence` in this payload is now the section's actual cited material, in the same
    # shape the writer saw it. See docs/evaluations/deep-research-critic-evidence-grounding-
    # 20260812.md.
    "`evidence` 是本节实际引用的原始证据（id/claim/quote/source_url/source_title/scope/stance/source_type），跟写手写作时看到的是同一批材料。判断 evidence_quality 维度（尤其范围泛化、无凭据评价性判断、fact/inference 限定语言、引用是否对题）时必须对照 evidence 里的实际内容核实，不能只凭正文表面语言猜测。",
    "每条 evidence 还带 `writer_context`：这是该证据在同源 dossier 里、本节作用域下的完整摘录（写手写作时也能看到，质量与 entailment 校验方完全一致）。`quote` 经常只是被切分到几百字的片段，未必含全貌；判断\"引用是否真支撑该子句\"时，必须连同 `writer_context` 一起看，不能只看 `quote` 字段就说\"证据里没有\"。只有 quote 和 writer_context 都不包含该内容时，才能判为无凭据。",
    "涉及证据判断的 issue（即上一条能用 evidence 核实的类型）必须在 issue 的 `evidence_ids` 字段里列出你依据的具体证据 id；不能引用者不要归为这类，避免编造出处。",
    "正文里的 `[N]` 是读者可见的来源序号，不是内部 evidence id；内部 evidence id 只使用 `ev_...` 形式。不得因为 `[3]` 与 `ev_003` 或任意 `ev_...` 不同而报告引用错误。当前 payload 没有提供来源序号到 evidence id 的映射，这类映射正确性由独立 citation validator 校验，critic 不得猜测。",
    "不涉及证据、纯粹是文本结构/写法问题的 issue（如段首无结论、表格取舍、段落孤立罗列），`evidence_ids` 留空，但 `instruction` 必须引用你评论所指的原文具体片段或句子，不能只抽象描述问题类型而不定位到原文。",
    # 2026-08-13 (see docs/evaluations/deep-research-critic-content-direction-20260813.md):
    # writer.py's _accept_revision floor rejects a revision that ends up shorter/cites fewer
    # evidence markers than the existing draft, unless the issue is entailment-sourced
    # (unsupported/non_atomic/label_mismatch, where deleting the offending clause is expected).
    # A critic issue like "this table restates the prose, delete it" gets no such exemption --
    # real job data (case-critic-direction-padding sec_03) showed the writer, unable to legitimately
    # shrink the section to satisfy exactly that complaint, padding in two new unsupported
    # synthesis claims instead just to keep the length/evidence-count from tripping the floor.
    # `content_direction` lets the critic -- which has full context on what its own issue implies
    # -- self-classify this instead of writer.py guessing from issue type or instruction
    # keywords (unreliable for semantic-level matching, per this project's repeated experience
    # with claim/issue identity matching).
    "每条 issue 必须包含 `content_direction` 字段，标注修复这条 issue 预期会让本节内容变多还是变少：`reduce`（修复方式是删除/精简，比如证据堆砌、表格与正文重复、无凭据的评价性判断、需要删掉的过度引申）；`add`（修复方式需要实质性新增内容，比如缺权衡论证、叙事单薄需要展开）；`neutral`（修复只是同等篇幅的改写，比如调整措辞、修正引用标记，不涉及增删）。标错方向会导致写手要么被迫编造内容凑数、要么被迫省略真正需要补充的论证，务必如实标注，不要为了保险都填 neutral。",
    # 2026-08-15 (case-critic-example-copy sec_00, isolated A/B:
    # scripts/replays/legacy/replay_a98ba6_example_copy.py):
    # a `structure` issue marked content_direction="add" instructed "添加一句概括性论点句，
    # 如'Agent harness技术围绕部署选项、工程演进和防护机制三个核心维度构建。'" and the writer
    # pasted that quoted example sentence verbatim as a new claim, then justified it ("该句由
    # ev_001/002/003 共同支持") -- a new unsupported synthesis entailment later flagged. Across a
    # 3-run isolation test the writer copied the example sentence in 3/3 baseline runs and 2/3
    # "stronger anti-copy prompt" runs; when the critic was made to NOT quote a ready-made sentence,
    # the writer copied it 0/3 times and wrote its own wording instead. The critic handing the
    # writer a finished sentence to paste is the root cause: an example sentence reads as
    # pre-authorized content the writer may use as-is, and an explicit "do not copy examples" prompt
    # rule is too soft to override that (same lesson as 08-05 layer F / 08-10 anchor: structure >
    # wording). So instruction text must describe the fix DIRECTION and point at the existing
    # prose, never supply a candidate sentence for the writer to adopt.
    "issue 的 instruction 只能描述要修的方向、并定位到原文具体片段，绝对不要给出可供写手直接照搬的成品句子（例如不要写\"如'<某完整论断>'\"、\"例如：<完整句子>\"）。写手会把这种成品句子当成已获授权可直接粘贴的结论照抄，再给它挂引用，导致一条证据未经校验的新综合论断进入正文。要让写手自己组织措辞去满足方向性要求；需要它补内容时，告诉它补什么类型的论证（如\"补一段权衡，讨论成本/延迟\"），而不是把答案递给它。",
]
# Issues below this confidence get an independent re-review (see _recheck_low_confidence_issues)
# instead of going straight into revision_issues unconditionally -- entailment's Layer C already
# has this for claims the model failed to answer on; critic issues had no equivalent for the case
# the model DID answer but the answer itself is a hallucinated/mistaken flag (docs/modules/
# deep-research-step3-writing-and-revision-discussion-v0.1.md, 风险 A). Missing confidence
# (model didn't follow the new schema field) is treated as low-confidence too, matching the
# existing "critic can only lower a score, never raise it" conservative posture elsewhere in
# this module.
_CRITIC_ISSUE_RECHECK_CONFIDENCE_THRESHOLD = 70


def _run_section_critics(
    documents: list[dict[str, Any]],
    support_units: list[dict[str, Any]],
    model: Callable[..., dict[str, Any]] | None,
    section_critic_budgets: dict[str, float] | None,
    evidence_by_id: dict[str, dict[str, Any]],
    *,
    prior_section_diagnostics: list[dict[str, Any]] | None = None,
    parallelism: int = 6,
    support_unit_kind: str = "finding",
) -> dict[str, Any]:
    """Score each section independently instead of one critic call over the joined report.

    A single whole-report critic call either fully succeeds or, on timeout, drops to `{}` --
    turning any budget overrun into every rubric dimension falling back to its deterministic
    default for the *entire* report (docs/evaluations/deep-research-section-dossier-e2e-
    20260715.md section 5.7). Splitting into one call per section means a timeout only degrades
    the sections that didn't finish; sections that did finish still get a real critic score.
    Final scores are the min across sections per dimension, matching the existing "critic can
    only lower, never raise" merge below.

    Sections are scored concurrently (mirroring `validate_claim_entailment`'s established
    `parallelism` pattern in this same module, which already calls the same underlying
    `_validation_model` closure from multiple threads). A real production job
    (case-sequential-validation-budget) showed this loop running sequentially for 6 sections consumed
    267s of the shared 360s validation budget -- ~74% -- which then left 0s for the automatic
    revision pass that the report's own 8 flagged issues should have triggered. Each section
    already gets an even, capped `timeout_override` from `_section_critic_budgets`, so running
    them concurrently only changes wall-clock time (N sections in ~1 section's worth of time
    instead of N), not the per-section budget accounting.

    2026-08-12: `prior_section_diagnostics` (this same function's own return value from the
    previous pass) enables two independent cross-pass mechanisms, mirroring what
    `validate_claim_entailment` already does for individual claims but adapted to critic's
    different problem shape (a whole-section judgment call, not a single fact-check):

    - A-class ("content unchanged, stop re-litigating"): if a section's rendered content is
      byte-identical to what the prior pass scored, AND that prior score was itself already
      `confirmed` (agreed with the pass before it), skip the model call entirely and reuse the
      prior scores/issues. Mirrors validate_claim_entailment's "confirm twice" rule exactly --
      a single independent score is still noisy on boundary cases, so it takes two consecutive
      agreeing looks before a section's score is trusted enough to stop re-asking.
    - B-class ("same content keeps changing, but the critic keeps flip-flopping on the same
      issue"): when content DID change since the prior pass, the critic has no exact-match
      anchor to reuse against (revision is supposed to change content) -- but real job data
      (case-critic-pass-reversal sec_03: critic said "delete this table" pass 0, "add a table
      here" pass 2, then insisted again pass 3, triggering writer arbitration for content that
      then failed entailment) shows the critic can reverse itself on the same rubric dimension
      pass to pass with no self-awareness. So `_model_critic_section` is given the prior pass's
      own issues for this section (never scores -- see the function's own docstring on why) and
      asked to flag it if this pass's judgment reverses direction on the same point.
    """
    section_diagnostics: list[dict[str, Any]] = []
    # The reference/citation-index section's content never reaches the reader (_assemble_report
    # discards it, see _is_harmless_reference_fallback below), so it's not worth a critic call.
    scoreable_documents = []
    for document in documents:
        title = str(document.get("title") or "")
        if _is_reference_title(title):
            continue
        generation_failed = str((document.get("synthesis_diagnostics") or {}).get("mode") or "") in {
            "fallback_template",
            "model_failed",
        }
        if str((document.get("synthesis_diagnostics") or {}).get("mode") or "") == "evidence_gap":
            section_diagnostics.append({
                "section_id": str(document.get("section_id") or ""),
                "title": title,
                "outcome": "evidence_gap",
            })
            continue
        if generation_failed or not str(document.get("content") or "").strip():
            section_diagnostics.append({
                "section_id": str(document.get("section_id") or ""),
                "title": title,
                "outcome": "not_deliverable",
            })
            continue
        scoreable_documents.append(document)
    if not model or not scoreable_documents:
        return {"scores": {}, "issues": [], "summary": "", "section_diagnostics": section_diagnostics}

    budgets = section_critic_budgets or {}
    prior_by_section_id = {
        str(item.get("section_id") or ""): item
        for item in prior_section_diagnostics or []
        if item.get("outcome") == "critic_scored"
    }

    def score_one(document: dict[str, Any]) -> dict[str, Any]:
        section_id = str(document.get("section_id") or "")
        title = str(document.get("title") or "")
        content = str(document.get("content") or "")
        content_hash = hash(content)
        prior = prior_by_section_id.get(section_id)
        content_unchanged = bool(prior) and prior.get("content_hash") == content_hash
        if content_unchanged and prior.get("confirmed") is True:
            # A-class reuse: two consecutive independent looks at this exact same content
            # already agreed, so trust it -- no model call this pass.
            return {
                "section_id": section_id, "title": title, "outcome": "critic_scored",
                "scores": prior.get("scores") or {}, "issues": prior.get("issues") or [],
                "summary": str(prior.get("summary") or ""), "content_hash": content_hash,
                "confirmed": True, "reused": True,
            }
        budget = budgets.get(section_id) if section_critic_budgets is not None else None
        if section_critic_budgets is not None and (budget is None or budget <= 0):
            return {"section_id": section_id, "title": title, "outcome": "budget_exhausted"}
        section_evidence = _section_critic_evidence(document, evidence_by_id)
        # B-class: content changed since the prior pass, so give the critic the prior pass's
        # own issues for this section as a reversal check -- never scores (see
        # _model_critic_section's own docstring for why scores are excluded).
        prior_issues = (prior.get("issues") or []) if (prior and not content_unchanged) else None
        section_support_units = (
            support_units
            if section_id == "editorial"
            else [item for item in support_units if str(item.get("section_id") or "") == section_id]
        )
        result = _model_critic_section(
            content,
            section_support_units,
            model,
            budget,
            evidence=section_evidence,
            prior_issues=prior_issues,
            support_unit_kind=support_unit_kind,
        )
        section_scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
        if not section_scores:
            return {"section_id": section_id, "title": title, "outcome": "model_error_or_timeout"}
        # Confirmed only when content is unchanged from the prior pass AND this fresh,
        # independent score agrees with that prior score on every dimension -- the first of the
        # two consecutive looks the A-class reuse above requires.
        confirmed = bool(
            content_unchanged
            and isinstance(prior.get("scores"), dict)
            and {k: int(v) for k, v in prior["scores"].items()} == {k: int(v) for k, v in section_scores.items()}
        )
        return {
            "section_id": section_id,
            "title": title,
            "outcome": "critic_scored",
            "scores": section_scores,
            "content_hash": content_hash,
            "confirmed": confirmed,
            # The critic is called once per section (see this function's docstring) and never
            # told which section it's looking at (`_model_critic_section`'s payload has no
            # section_id/title, only raw content), so its own `section_id` guess on each issue is
            # unreliable free text -- real jobs show ~42% of issues come back with a section_id
            # that matches no real document section (e.g. "report", "main", "para_1", or even a
            # Chinese section title), which `revise_report_sections`'s exact-match `targets`
            # routing then silently drops before the writer ever sees them (see
            # docs/evaluations/deep-research-revision-loop-critic-writer-transparency-20260811-
            # handoff.md). The caller already knows the true section_id via `document`, so stamp
            # it onto every issue instead of trusting the model's guess.
            "issues": [
                {**item, "section_id": section_id}
                for item in result.get("issues") or []
                if isinstance(item, dict)
            ],
            "summary": str(result.get("summary") or "").strip(),
        }

    results_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(scoreable_documents)))) as executor:
        futures = {
            submit_with_audit_context(executor, score_one, document): index
            for index, document in enumerate(scoreable_documents)
        }
        for future in as_completed(futures):
            results_by_index[futures[future]] = future.result()

    per_dimension_scores: dict[str, list[int]] = {}
    issues: list[dict[str, Any]] = []
    summaries: list[str] = []
    for index in range(len(scoreable_documents)):
        outcome = results_by_index[index]
        title = str(outcome.get("title") or "")
        if outcome["outcome"] != "critic_scored":
            section_diagnostics.append({"section_id": outcome["section_id"], "title": title, "outcome": outcome["outcome"]})
            continue
        section_diagnostics.append({
            "section_id": outcome["section_id"], "title": title, "outcome": "critic_scored", "scores": outcome["scores"],
            "content_hash": outcome.get("content_hash"), "confirmed": outcome.get("confirmed", False),
            "issues": outcome.get("issues") or [], "summary": outcome.get("summary") or "",
        })
        for key in _CRITIC_RUBRIC:
            if key not in outcome["scores"]:
                continue
            try:
                value = max(1, min(5, int(outcome["scores"][key])))
            except (TypeError, ValueError):
                continue
            per_dimension_scores.setdefault(key, []).append(value)
        issues.extend(outcome.get("issues") or [])
        if outcome.get("summary"):
            summaries.append(f"{title}: {outcome['summary']}" if title else outcome["summary"])

    content_by_section_id = {
        str(document.get("section_id") or ""): str(document.get("content") or "")
        for document in scoreable_documents
    }
    issues, confidence_recheck_diagnostics = _recheck_low_confidence_issues(
        issues, content_by_section_id, model, parallelism=parallelism
    )
    issues, evidence_recheck_diagnostics = _recheck_evidence_backed_issues(
        issues, content_by_section_id, evidence_by_id, model, parallelism=parallelism
    )
    refuted_issues = [item for item in issues if item.get("self_check") == "refuted"]
    active_issues = [item for item in issues if item.get("self_check") != "refuted"]

    aggregated_scores = {key: min(values) for key, values in per_dimension_scores.items()}
    return {
        "scores": aggregated_scores,
        "issues": active_issues,
        "refuted_issues": refuted_issues,
        "summary": "; ".join(summaries),
        "section_diagnostics": section_diagnostics,
        "confidence_recheck": confidence_recheck_diagnostics,
        "evidence_recheck": evidence_recheck_diagnostics,
    }


def _section_critic_evidence(
    document: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]], section_id: str | None = None
) -> list[dict[str, Any]]:
    """Resolve a section's `evidence_table` ids into the same evidence shape the writer saw
    (id/claim/quote/source_url/source_title/scope/stance/source_type) for the critic call.

    Real sections already carry `evidence_table` (writer.py's `_write_section`, the precise
    "evidence actually used in this section" subset), but that table's own rows
    (`_evidence_table_row`) don't include scope/stance/source_type -- those only exist on the
    full evidence rows in `evidence_by_id` (sourced from `evidence_map`). The editorial pseudo-
    document (`_documents_with_editorial_pseudo_document`) has no real `evidence_table` of its
    own, so it's given a stub with just `id` entries built from its claim_ledger -- resolved the
    same way here. Real, unbounded quote length: a 6-job sample of real evidence rows showed a
    median of ~120 chars and a max under 500, so no truncation is applied (unlike entailment's
    1600-char cap, which exists for a different, unmeasured worst case).

    `writer_context` (2026-08-15): the same section-scoped, same-source dossier context the
    Writer was allowed to see (writer.py::_model_section's section_dossier.source_notes/
    relevant_chunks excerpts). It is attached by `_entailment_evidence_with_writer_context`
    (research_service.py) onto each evidence row as `validation_context_by_section` and was
    already given to the entailment judge (_entailment_payload). Before this, the critic saw
    the same `quote` (often a <500-char slice of the full source) WITHOUT that fuller context,
    so a claim the Writer correctly grounded in dossier context the `quote` field itself didn't
    spell out. In `case-validator-context-parity`, a short quote omitted a deployment
    recommendation present in the same-source dossier excerpt, so the critic flagged the claim
    as unsupported and the Writer dutifully
    deleted the true-but-undocumented-by-quote claim, and a real grounded assertion was lost to
    a reviewer who saw strictly LESS context than the author it was reviewing. The critic is a
    reviewer; it must see at least as much context as the Writer did.
    """
    resolved_section_id = str(section_id or document.get("section_id") or "")
    evidence_ids = [
        str(row.get("id") or "")
        for row in document.get("evidence_table") or []
        if isinstance(row, dict) and row.get("id")
    ]
    rows = []
    for evidence_id in dict.fromkeys(evidence_ids):
        if evidence_id not in evidence_by_id:
            continue
        evidence = evidence_by_id[evidence_id]
        writer_context = ""
        contexts = evidence.get("validation_context_by_section")
        if isinstance(contexts, dict):
            value = contexts.get(resolved_section_id)
            if value:
                writer_context = str(value)[:3200]
        rows.append({
            "id": evidence_id,
            "claim": evidence.get("atomic_claim") or evidence.get("claim"),
            "quote": (
                evidence.get("cleaned_quote")
                or evidence.get("raw_quote")
                or evidence.get("evidence_text")
            ),
            "writer_context": writer_context,
            "source_url": evidence.get("source_url"),
            "source_title": evidence.get("source_title"),
            "scope": evidence.get("scope"),
            **_semantic_evidence_payload(evidence),
            "stance": evidence.get("stance"),
            "source_type": evidence.get("source_type"),
        })
    return rows


def _recheck_evidence_backed_issues(
    issues: list[dict[str, Any]],
    content_by_section_id: dict[str, str],
    evidence_by_id: dict[str, dict[str, Any]],
    model: Callable[..., dict[str, Any]] | None,
    *,
    parallelism: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently self-check every critic issue that claims evidence backing (non-empty
    `evidence_ids`), i.e. that the cited evidence actually supports the issue being real.

    Unlike `_recheck_low_confidence_issues` (which drops a refuted issue entirely), a refuted
    issue here is kept in the returned list but annotated `self_check: "refuted"` -- the point of
    this mechanism is to stop a critic issue from reaching the writer/revision loop on a
    misreading of the evidence, not to erase the fact that the critic raised it. Callers
    collecting `revision_issues` must skip issues with `self_check == "refuted"`; everything else
    (report_quality.issues, the UI, diagnostics) keeps seeing the full list so a false alarm stays
    visible for review instead of vanishing the way a dropped issue would.
    """
    diagnostics = {
        "evidence_backed_count": 0,
        "rechecked_count": 0,
        "confirmed_count": 0,
        "refuted_count": 0,
        "no_verdict_count": 0,
    }
    evidence_backed_indices = [
        i for i, issue in enumerate(issues) if [str(v) for v in issue.get("evidence_ids") or [] if str(v)]
    ]
    diagnostics["evidence_backed_count"] = len(evidence_backed_indices)
    if not model or not evidence_backed_indices:
        return issues, diagnostics

    def recheck_one(index: int) -> tuple[int, str]:
        issue = issues[index]
        section_id = str(issue.get("section_id") or "")
        cited_ids = [str(v) for v in issue.get("evidence_ids") or [] if str(v)]
        payload = {
            "task": (
                "A report critic flagged a possible evidence-related quality issue below, citing "
                "specific evidence rows as its basis. Decide whether those evidence rows actually "
                "support the issue being a real problem, or whether the critic misread them."
            ),
            "issue": {"type": str(issue.get("type") or ""), "instruction": str(issue.get("instruction") or "")},
            "cited_evidence": [
                {
                    "id": evidence_id,
                    "claim": evidence_by_id[evidence_id].get("atomic_claim") or evidence_by_id[evidence_id].get("claim"),
                    "quote": (
                        evidence_by_id[evidence_id].get("cleaned_quote")
                        or evidence_by_id[evidence_id].get("raw_quote")
                        or evidence_by_id[evidence_id].get("evidence_text")
                    ),
                    "writer_context": str(
                        (evidence_by_id[evidence_id].get("validation_context_by_section") or {}).get(section_id)
                        if isinstance(evidence_by_id[evidence_id].get("validation_context_by_section"), dict)
                        else ""
                    )[:3200],
                    "source_url": evidence_by_id[evidence_id].get("source_url"),
                    "source_title": evidence_by_id[evidence_id].get("source_title"),
                    "scope": evidence_by_id[evidence_id].get("scope"),
                    **_semantic_evidence_payload(evidence_by_id[evidence_id]),
                    "stance": evidence_by_id[evidence_id].get("stance"),
                }
                for evidence_id in cited_ids
                if evidence_id in evidence_by_id
            ],
            "section_content": content_by_section_id.get(section_id, "")[:8000],
            "schema": {"issue_confirmed": True, "reason": ""},
        }
        result = model(payload)
        if not isinstance(result, dict) or "issue_confirmed" not in result:
            return index, "no_verdict"
        return index, "confirmed" if bool(result.get("issue_confirmed")) else "refuted"

    outcomes: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(evidence_backed_indices)))) as executor:
        futures = {
            submit_with_audit_context(executor, recheck_one, index): index
            for index in evidence_backed_indices
        }
        for future in as_completed(futures):
            index, outcome = future.result()
            outcomes[index] = outcome

    diagnostics["rechecked_count"] = len(outcomes)
    diagnostics["confirmed_count"] = sum(1 for outcome in outcomes.values() if outcome == "confirmed")
    diagnostics["refuted_count"] = sum(1 for outcome in outcomes.values() if outcome == "refuted")
    diagnostics["no_verdict_count"] = sum(1 for outcome in outcomes.values() if outcome == "no_verdict")
    annotated = list(issues)
    for index, outcome in outcomes.items():
        if outcome == "refuted":
            annotated[index] = {**annotated[index], "self_check": "refuted"}
    return annotated, diagnostics


def _issue_confidence_below_threshold(issue: dict[str, Any]) -> bool:
    confidence = issue.get("confidence")
    if confidence is None:
        # The model didn't follow the new schema field -- treat as low-confidence rather than
        # skip the recheck, matching this module's existing "when unsure, downgrade" posture.
        return True
    try:
        return float(confidence) < _CRITIC_ISSUE_RECHECK_CONFIDENCE_THRESHOLD
    except (TypeError, ValueError):
        return True


def _recheck_low_confidence_issues(
    issues: list[dict[str, Any]],
    content_by_section_id: dict[str, str],
    model: Callable[..., dict[str, Any]] | None,
    *,
    parallelism: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently re-review each critic issue whose self-reported confidence is below
    threshold (or missing), and drop only the ones the re-review model explicitly calls a false
    alarm.

    Mirrors validate_claim_entailment's Layer C in shape (small, targeted follow-up model calls
    reusing the same deadline-aware `model` closure the caller already passed in -- see
    docs/modules/deep-research-step3-writing-and-revision-discussion-v0.1.md 风险 A) but a
    different trigger: Layer C recovers claims the model never answered; this recovers issues the
    model DID answer but may have hallucinated. No dedicated budget reserve is carved out for
    this -- like Layer C, a recheck call that can't get a model response (budget exhausted /
    timeout) just returns {} and the original issue is kept, so this can never make a report
    worse than not having the recheck at all. `confidence_recheck` diagnostics are returned
    purely for post-launch observation of real call volume/outcomes; they never affect gating.
    """
    diagnostics = {
        "low_confidence_count": 0,
        "rechecked_count": 0,
        "confirmed_count": 0,
        "refuted_count": 0,
        "no_verdict_count": 0,
    }
    if not model or not issues:
        return issues, diagnostics
    low_confidence_indices = [i for i, issue in enumerate(issues) if _issue_confidence_below_threshold(issue)]
    diagnostics["low_confidence_count"] = len(low_confidence_indices)
    if not low_confidence_indices:
        return issues, diagnostics

    def recheck_one(index: int) -> tuple[int, str]:
        issue = issues[index]
        section_content = content_by_section_id.get(str(issue.get("section_id") or ""), "")
        payload = {
            "task": (
                "A report critic flagged a possible quality issue below. Decide whether the "
                "issue genuinely applies to the section text, or is a false alarm."
            ),
            "issue": {"type": str(issue.get("type") or ""), "instruction": str(issue.get("instruction") or "")},
            "section_content": section_content[:8000],
            "schema": {"issue_confirmed": True, "reason": ""},
        }
        result = model(payload)
        if not isinstance(result, dict) or "issue_confirmed" not in result:
            return index, "no_verdict"
        return index, "confirmed" if bool(result.get("issue_confirmed")) else "refuted"

    outcomes: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(low_confidence_indices)))) as executor:
        futures = {
            submit_with_audit_context(executor, recheck_one, index): index
            for index in low_confidence_indices
        }
        for future in as_completed(futures):
            index, outcome = future.result()
            outcomes[index] = outcome

    diagnostics["rechecked_count"] = len(outcomes)
    diagnostics["confirmed_count"] = sum(1 for outcome in outcomes.values() if outcome == "confirmed")
    diagnostics["refuted_count"] = sum(1 for outcome in outcomes.values() if outcome == "refuted")
    diagnostics["no_verdict_count"] = sum(1 for outcome in outcomes.values() if outcome == "no_verdict")
    kept_issues = [issue for i, issue in enumerate(issues) if outcomes.get(i, "confirmed") != "refuted"]
    return kept_issues, diagnostics


def _model_critic_section(
    content: str,
    support_units: list[dict[str, Any]],
    model: Callable[..., dict[str, Any]],
    timeout_override: float | None,
    evidence: list[dict[str, Any]] | None = None,
    prior_issues: list[dict[str, Any]] | None = None,
    support_unit_kind: str = "finding",
) -> dict[str, Any]:
    # `model` must accept an optional `timeout_override` kwarg (backend.services.research_service
    # ._validation_model does); it caps this section's call so one slow section can't consume
    # another section's share of the remaining validation budget.
    payload = {
        "task": "Evaluate one section of a Deep Research report and return JSON only.",
        "report": content[:24000],
        "evidence": evidence or [],
        "rubric": _CRITIC_RUBRIC,
        "rules": _CRITIC_RULES,
        "schema": {
            "scores": {},
            "issues": [{
                "section_id": "sec_01", "type": "", "instruction": "", "confidence": 100,
                "evidence_ids": [], "content_direction": "reduce|add|neutral",
            }],
            "summary": "",
        },
    }
    if support_unit_kind == "finding":
        payload["available_findings"] = support_units[:20]
    else:
        payload["available_support_units"] = support_units[:20]
        payload["support_unit_kind"] = support_unit_kind
    if prior_issues:
        # 2026-08-12 (see docs/evaluations/deep-research-critic-cross-pass-memory-20260812.md):
        # this section's content changed since the previous pass, so there is no exact-match
        # anchor to reuse against (unlike the A-class content-unchanged path in
        # _run_section_critics). Instead give the critic the previous pass's own issues for this
        # section as a self-consistency check -- real data (case-critic-pass-reversal sec_03)
        # showed the critic reverse itself pass to pass on the same rubric dimension with no
        # awareness it had done so ("delete this table" -> "add a table here" -> "add a table
        # here" again, burning a revision pass and producing content entailment then rejected).
        # Deliberately excludes `scores` (only `issues`) -- giving the model its own prior
        # numeric scores risks anchoring THIS pass's score toward the previous one, which is a
        # more damaging and harder-to-guard-against form of anchoring than issue-text influence
        # (scores are the actual quality-gate threshold; issues are informational). Placed last
        # in the payload and phrased as a different reviewer's opinion (not "you last time") to
        # reduce the pull toward continuing a first-person narrative instead of judging the
        # current text independently.
        payload["prior_pass_reviewer_notes"] = {
            "note": (
                "Judge the current report content on its own merits first, independently of the "
                "notes below. The notes below are what a DIFFERENT reviewer flagged about an "
                "earlier version of this same section -- they may no longer apply now that the "
                "content has changed. Only use them to check one thing: if your own judgment on "
                "some point directly reverses what is noted below (e.g. it previously said add "
                "a table and you are now saying remove it, or vice versa), state briefly in your "
                "issue or summary why the situation has changed. If your judgment does not "
                "reverse anything below, ignore this field entirely."
            ),
            "previous_reviewer_issues": [
                {"type": item.get("type"), "instruction": item.get("instruction")}
                for item in prior_issues
                if isinstance(item, dict)
            ],
        }
    result = model(payload, timeout_override=timeout_override)
    return result if isinstance(result, dict) else {}


def carry_forward_stale_critic_issues(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """After a post-revision re-evaluation, fall back to the pre-revision critic's issues if the
    re-evaluation's own section critics never actually ran.

    A real production job (case-shared-validation-budget) showed this failure mode: the shared
    validation budget is spent almost entirely by claim_entailment's first pass + the revision
    it triggers, so by the time the post-revision re-evaluation reaches `_run_section_critics`,
    every section budget has already collapsed to `budget_exhausted` -- even though the *first*
    evaluate_report_quality() call, before revision, genuinely scored sections and produced real
    `issues`. Because research_service.py's caller simply overwrites `report_quality` with the
    second call's return value, those real issues were silently discarded and replaced with an
    empty list -- the same "no per-section problem list" symptom the entailment/critic budget
    reserve fix (docs/evaluations 07-31) was meant to eliminate, just relocated one level up.

    This only ever carries forward `issues`/`critic_summary` (the free-form, informational
    fields) -- never `scores`, `blocking_issues`, or `status`, which must always reflect the
    *current* (possibly revised) report content, not a stale pre-revision snapshot.
    """
    if not previous:
        return current
    current_diagnostics = current.get("section_critic_diagnostics") or []
    if any(item.get("outcome") == "critic_scored" for item in current_diagnostics):
        return current
    previous_diagnostics = previous.get("section_critic_diagnostics") or []
    if not any(item.get("outcome") == "critic_scored" for item in previous_diagnostics):
        return current
    previous_issues = previous.get("issues") or []
    if not previous_issues:
        return current
    merged = dict(current)
    merged["issues"] = previous_issues
    merged["critic_summary"] = previous.get("critic_summary") or current.get("critic_summary") or ""
    merged["issues_stale_pre_revision"] = True
    return merged


def combined_quality_gate(
    *,
    grounding: dict[str, Any],
    validation: dict[str, Any],
    report_quality: dict[str, Any],
    entailment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic_review_unavailable_count = int(
        (entailment or {}).get("semantic_review_unavailable_count")
        or (entailment or {}).get("deterministic_fallback_count")
        or 0
    )
    checks = {
        "grounding": str(grounding.get("status") or "passed") != "failed",
        "citation_validation": validation.get("status") == "passed",
        "report_quality": report_quality.get("status") == "passed",
        "core_claim_support": int((entailment or {}).get("blocking_core_unsupported_count") or 0) == 0,
        "semantic_review_complete": semantic_review_unavailable_count == 0,
    }
    # The whole entailment status is not a gate: label_mismatch/non_atomic and non-core issues are
    # advisory. Only a genuinely unsupported claim inside the report's core answer contract is
    # counted by blocking_core_unsupported_count and blocks approval.
    ready = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "status": "passed" if ready else "needs_more_evidence",
        "approval_ready": ready,
        "reason": "新研究链路质量门通过。" if ready else "未通过：" + "、".join(failed),
        "checks": {**checks, "report_quality": report_quality},
    }
    if ready:
        _annotate_low_confidence_critic_coverage(result, report_quality)
    _annotate_entailment_degradation_transparency(result, entailment)
    _annotate_stale_critic_issues_transparency(result, report_quality)
    return result


def _annotate_low_confidence_critic_coverage(
    quality_gate: dict[str, Any],
    report_quality: dict[str, Any],
) -> None:
    """Flag (never block) a pass reached mostly on heuristic fallback, not real critic scoring.

    A pass is only as trustworthy as the critic coverage behind it: if most sections fell back to
    budget_exhausted / model_error_or_timeout (see _run_section_critics), the scores.get() min-merge
    above never actually ran the critic on them, so "passed" may just mean the lenient deterministic
    heuristic never got checked. This is purely informational for review/eval tooling -- see
    docs/evaluations/deep-research-dossier-utilization-and-evidence-truncation-fix-e2e-20260724.md
    item 5 -- and must never flip status/approval_ready itself.
    """
    diagnostics = report_quality.get("section_critic_diagnostics") or []
    if not diagnostics:
        return
    scored = sum(1 for item in diagnostics if item.get("outcome") == "critic_scored")
    ratio = scored / len(diagnostics)
    if ratio < 0.5:
        quality_gate.setdefault("warnings", []).append(
            f"低置信度通过：仅 {scored}/{len(diagnostics)} 个章节获得真实 critic 打分，"
            "其余因预算耗尽/调用失败退化为启发式分数。"
        )
        quality_gate.setdefault("diagnostics", {})["low_confidence_pass"] = True
        quality_gate["diagnostics"]["critic_coverage_ratio"] = round(ratio, 3)


def _annotate_entailment_degradation_transparency(
    quality_gate: dict[str, Any],
    entailment: dict[str, Any] | None,
) -> None:
    """Expose incomplete semantic review; combined_quality_gate() already blocks approval."""
    if not entailment:
        return
    unavailable = int(
        entailment.get("semantic_review_unavailable_count")
        or entailment.get("deterministic_fallback_count")
        or 0
    )
    if unavailable <= 0:
        return
    reviewed = int(entailment.get("model_reviewed_count") or 0)
    eligible = unavailable + reviewed
    quality_gate.setdefault("warnings", []).append(
        f"语义验证未完成：claim_entailment 中 {unavailable}/{eligible} 条本应交给模型复核的声明"
        "在重试后仍未取得模型判定；系统未使用确定性语义兜底，当前报告不可审批。"
    )
    diagnostics = quality_gate.setdefault("diagnostics", {})
    diagnostics["entailment_semantic_review_unavailable_count"] = unavailable
    diagnostics["entailment_model_reviewed_count"] = reviewed
    if eligible:
        diagnostics["entailment_semantic_review_unavailable_ratio"] = round(unavailable / eligible, 3)


def _annotate_stale_critic_issues_transparency(
    quality_gate: dict[str, Any],
    report_quality: dict[str, Any],
) -> None:
    """Surface carry_forward_stale_critic_issues()'s substitution on quality_gate itself.

    Without this, a reader sees `issues` populated and has no way to tell those issues were
    scored against the pre-revision report content rather than the one actually delivered.
    Purely informational -- never flips status/approval_ready.
    """
    if not report_quality.get("issues_stale_pre_revision"):
        return
    quality_gate.setdefault("warnings", []).append(
        "陈旧问题清单：本次返修后的章节评审未能实际运行（预算耗尽），"
        "以下 issues 来自返修前的评审结果，可能不完全反映返修后的最终正文。"
    )
    quality_gate.setdefault("diagnostics", {})["critic_issues_stale_pre_revision"] = True


def classify_unused_dossier_evidence(
    *,
    documents: list[dict[str, Any]],
    model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    parallelism: int = 3,
) -> dict[str, Any]:
    """Classify, per section, Dossier evidence the Writer had access to but never cited.

    docs/evaluations/deep-research-section-dossier-e2e-20260715.md section 6 item 2: the Dossier
    (backend/research/dossier.py) hands each section's Writer call more admitted evidence than
    what survived into `findings`, but nothing checks afterward whether an uncited item was
    genuinely irrelevant to skip, or actually answered one of the section's research_questions
    and was simply missed. This is diagnostic-only by design (see the revision-guard discussion
    in writer.py's _accept_revision docstring for why an automated "add more content" edit is
    risky to trigger blind): it surfaces `missed_relevant` items in validation diagnostics for a
    human/editor to act on, and never feeds into quality_gate or revision_issues itself.
    """
    def unused_items_for(document: dict[str, Any]) -> list[dict[str, Any]]:
        section_brief = document.get("section_brief") if isinstance(document.get("section_brief"), dict) else {}
        dossier = section_brief.get("dossier") if isinstance(section_brief.get("dossier"), dict) else {}
        used_ids = {
            str(row.get("id") or "")
            for row in document.get("evidence_table") or []
            if isinstance(row, dict) and str(row.get("id") or "")
        }
        return [
            item for item in dossier.get("evidence") or []
            if isinstance(item, dict) and str(item.get("id") or "") and str(item.get("id")) not in used_ids
        ]

    def classify_section(document: dict[str, Any]) -> list[dict[str, Any]]:
        section_id = str(document.get("section_id") or "")
        unused = unused_items_for(document)
        if not unused:
            return []
        if not model:
            return [{
                "section_id": section_id,
                "evidence_id": str(item.get("id") or ""),
                "verdict": "unknown",
                "reason": "no model configured for coverage classification",
                "mode": "no_model",
            } for item in unused]
        section_brief = document.get("section_brief") if isinstance(document.get("section_brief"), dict) else {}
        research_questions = [
            {"question_id": q.get("question_id"), "question": q.get("question")}
            for q in section_brief.get("research_questions") or []
            if isinstance(q, dict)
        ]
        payload = {
            "task": "Classify each unused evidence item for a report section. Return JSON only.",
            "section_title": document.get("title"),
            "section_content": str(document.get("content") or "")[:6000],
            "research_questions": research_questions,
            "unused_evidence": [{
                "evidence_id": str(item.get("id") or ""),
                "claim": item.get("claim"),
                "quote": item.get("quote"),
                "source_title": item.get("source_title"),
            } for item in unused],
            "rules": [
                "For each unused_evidence item, decide irrelevant (fine that the section skipped it) or missed_relevant (it clearly answers a listed research_question or bears directly on the section's core topic, and skipping it left a real coverage gap).",
                "Default to irrelevant when uncertain -- only mark missed_relevant when the evidence clearly and directly bears on a stated research_question or the section's central topic, not merely the same general subject area.",
            ],
            "schema": {"classifications": [{"evidence_id": "", "verdict": "irrelevant|missed_relevant", "reason": ""}]},
        }
        result = model(payload)
        rows = (result.get("classifications") or []) if isinstance(result, dict) else []
        by_id = {str(item.get("evidence_id") or ""): item for item in rows if isinstance(item, dict)}
        output = []
        for item in unused:
            evidence_id = str(item.get("id") or "")
            row = by_id.get(evidence_id)
            verdict = str((row or {}).get("verdict") or "").strip()
            if verdict not in {"irrelevant", "missed_relevant"}:
                verdict = "irrelevant"
                mode = "no_verdict_defaulted"
            else:
                mode = "model"
            output.append({
                "section_id": section_id,
                "evidence_id": evidence_id,
                "verdict": verdict,
                "reason": str((row or {}).get("reason") or "")[:300],
                "mode": mode,
            })
        return output

    classifications: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(documents) or 1))) as executor:
        futures = {
            submit_with_audit_context(executor, classify_section, document): document.get("section_id")
            for document in documents
        }
        for future in as_completed(futures):
            classifications.extend(future.result())
    classifications.sort(key=lambda item: (item.get("section_id") or "", item.get("evidence_id") or ""))
    missed = [item for item in classifications if item.get("verdict") == "missed_relevant"]
    return {
        "unused_evidence_count": len(classifications),
        "missed_relevant_count": len(missed),
        "irrelevant_count": sum(item.get("verdict") == "irrelevant" for item in classifications),
        "classifications": classifications,
    }


def _entailment_evidence_row(
    section_id: str,
    evidence_id: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "claim": evidence.get("atomic_claim") or evidence.get("claim"),
        "quote": str(
            evidence.get("cleaned_quote")
            or evidence.get("raw_quote")
            or evidence.get("evidence_text")
            or ""
        )[:1600],
        "writer_context": str(
            (evidence.get("validation_context_by_section") or {}).get(section_id)
            if isinstance(evidence.get("validation_context_by_section"), dict)
            else ""
        )[:3200],
        "scope": evidence.get("scope"),
        "claim_type": evidence.get("claim_type"),
        "directness_score": evidence.get("directness_score"),
        **_semantic_evidence_payload(evidence),
        "source_title": evidence.get("source_title"),
        "source_url": evidence.get("source_url"),
    }


def _atomic_boundary_entailment_payload(
    section_id: str,
    rows: list[tuple[int, dict[str, Any]]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for index, claim in rows:
        evidence_ids = [str(item) for item in claim.get("evidence_ids") or []]
        cases.append({
            "case_id": str(index),
            "claim_index": index,
            "cited_span": claim.get("text"),
            "claim_context": str(claim.get("claim_context") or "")[:1000],
            "claim_type": claim.get("claim_type"),
            "support_scope": claim.get("support_scope"),
            "evidence_ids": evidence_ids,
            "evidence": [
                _entailment_evidence_row(section_id, evidence_id, evidence_by_id[evidence_id])
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            ],
        })
    return {
        "task": "严格核验每个最小 cited_span 是否被其指定 Evidence 完整支持，只返回 JSON。",
        "prompt_version": ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION,
        "section_id": section_id,
        "atomic_boundary_contract": {
            "target": "cited_span_only",
            "uncited_analysis_is_out_of_scope": True,
            "evidence_is_scoped_per_case": True,
            "all_factual_subclaims_must_be_supported": True,
            "partial_support_status": "unsupported",
            "automatic_rewrite": False,
        },
        "rules": [
            "只核验 cited_span。claim_context 仅用于消解主语或指代，不能作为证据，也不要检查未引用分析。",
            "每个 case 只能使用该 case.evidence 中明确提供的内容；不得借用其他 case 的 Evidence。writer_context、source_title 和 source_url 仅在该 Evidence 自身明确包含相关细节时提供支持。",
            "先把 cited_span 拆成所有可外部核查的事实子主张。对象、机制、关系、范围、条件、程度、因果、目的、比较和并列/或选项都属于独立支持义务。",
            "Evidence 必须支持每一个子主张及其限定词。只支持核心事实但不支持修饰、因果包装、目的、机制、范围扩大或另一并列分支时，必须判 unsupported。",
            "只有全部事实子主张均被 Evidence 支持，才能判 entailed 或 reasonable_inference。不要因为句子整体听起来合理而补足 Evidence 没有表达的内容。",
            "不要因为措辞不同而误杀忠实释义；语义等价即可。",
            "claim_scope 与支持结论分开判断，只能输出 domain_fact、product_limitation、research_process_disclosure 或 unknown。",
        ],
        "few_shot_examples": [
            {
                "name": "atomic_boundary_compound_and_causal_overreach",
                "cited_span": "产品 A 使用分层索引和深度遍历，因此响应更快。",
                "evidence": "产品 A 的响应时间较短。",
                "judgment": {
                    "status": "unsupported",
                    "claim_scope": "domain_fact",
                    "subclaims": ["产品 A 使用分层索引", "产品 A 使用深度遍历", "这些机制导致产品 A 响应更快"],
                    "unsupported_parts": ["使用分层索引", "使用深度遍历", "这些机制导致响应更快"],
                    "reason": "Evidence supports only the shorter response time, not the mechanisms or causal attribution.",
                },
            },
            {
                "name": "atomic_boundary_faithful_paraphrase",
                "cited_span": "产品 A 的响应时间较短。",
                "evidence": "产品 A 的响应时间较短。",
                "judgment": {
                    "status": "entailed",
                    "claim_scope": "domain_fact",
                    "subclaims": ["产品 A 的响应时间较短"],
                    "unsupported_parts": [],
                    "reason": "Every factual subclaim is directly supported.",
                },
            },
        ],
        "cases": cases,
        "output_schema": {
            "judgments": [{
                "case_id": "string",
                "claim_index": 0,
                "status": "entailed|reasonable_inference|unsupported|label_mismatch",
                "claim_scope": "domain_fact|product_limitation|research_process_disclosure|unknown",
                "subclaims": ["string"],
                "unsupported_parts": ["string"],
                "reason": "不超过120字",
            }]
        },
    }


def _entailment_payload(
    section_id: str,
    rows: list[tuple[int, dict[str, Any]]],
    evidence_by_id: dict[str, dict[str, Any]],
    payload_profile: str = ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
) -> dict[str, Any]:
    if payload_profile == ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2:
        return _atomic_boundary_entailment_payload(section_id, rows, evidence_by_id)
    if payload_profile != ENTAILMENT_PAYLOAD_PROFILE_DEFAULT:
        raise ValueError(f"unsupported entailment payload profile: {payload_profile}")
    payload = {
        "task": "Judge whether report claims are supported by supplied evidence. Return JSON only.",
        "prompt_version": ENTAILMENT_PROMPT_VERSION,
        "section_id": section_id,
        "claims": [{
            "claim_index": index,
            "claim": claim.get("text"),
            "claim_context": str(claim.get("claim_context") or "")[:1000],
            "claim_type": claim.get("claim_type"),
            "support_scope": claim.get("support_scope"),
            "evidence_ids": claim.get("evidence_ids") or [],
        } for index, claim in rows],
        "evidence": [
            _entailment_evidence_row(section_id, evidence_id, evidence)
            for evidence_id, evidence in evidence_by_id.items()
            if any(evidence_id in (claim.get("evidence_ids") or []) for _index, claim in rows)
        ],
        "rules": [
            "Use only supplied evidence.",
            "The Writer was allowed to use claim, quote, writer_context, source_title, and source_url. Treat explicit details in any of those fields as supplied evidence.",
            "writer_context is section-scoped same-source context. It may support a claim only when it belongs to the cited evidence row supplied here.",
            "claim_context is the immediately preceding report sentence and is supplied only to resolve pronouns or omitted subjects in claim. Do not treat claim_context as evidence and do not use it to supply factual support.",
            # 2026-08-11 (case-source-metadata-and-revision-growth): before source_title/source_url were added
            # here, the Writer (which does see them, see writer.py's evidence payload) correctly
            # attributed content to its real publication/author ('Cobus Greyling', 'ThoughtWorks
            # 的工程博客', a Bayer case study) and this judge flagged every one as an invented
            # attribution -- it never had the source metadata to check them against. Scope this
            # tightly to the cited row's own source so a claim can't launder an attribution to
            # some OTHER organization/person just because any evidence row has source info.
            "A claim may attribute content to the publication, author, or platform identifiable from a cited evidence row's own source_title/source_url -- treat that attribution as supported. Do not extend this allowance to any name, organization, or system that is not identifiable from that specific evidence row's source_title/source_url.",
            # 2026-08-17 (architecture decision, see docs/plans/writer-critic-loop-redesign):
            # the old 5-label taxonomy let tone/confidence disputes (label_mismatch on an
            # inference stated more confidently than the evidence's own hedging) block
            # approval and drive an unproductive writer<->judge revision loop (a98ba6 dead
            # loop, 54-73% cross-pass recurrence). The recovery action is now a cheap,
            # non-destructive inline [待核实] marker rather than a rewrite, so the bar for
            # blocking narrows to only failure modes a reader could not resolve just by
            # reading the citation.
            "entailed and reasonable_inference are both GROUNDED: the cited evidence justifies the claim's factual content at the stated scope, whether stated as fact or as an explicit inference/recommendation. An inference or recommendation stated more confidently than the evidence's own hedging is still grounded -- tone and confidence-level do not block.",
            "unsupported means one of three things: (a) the cited evidence does not justify the claim at all (missing evidence, evidence about a different subject, or evidence that contradicts the claim); (b) the claim's specific content -- especially a number, date, name, ratio, or quantity -- does not match what the cited evidence actually states; (c) the claim generalizes a single or narrow-scope piece of evidence into a broader or universal conclusion the evidence does not support (scope overstatement).",
            "label_mismatch is now reserved for case (c) above (a fact label overstating indirect or narrow-scope evidence into a broader claim) -- NOT for tone or confidence-level mismatches, which are grounded per the rule above.",
            "claim_type is a conservative provenance hint, not a verdict: if the supplied evidence directly supports the claim's complete factual content at the stated scope, return entailed even when claim_type is inference.",
            "Classify claim_scope independently from support status. Use domain_fact for claims about the subject/domain, product_limitation for claims that the product itself lacks or cannot do something, research_process_disclosure for claims about what this research/search/material set/document set did not find, confirm, or provide, and unknown only when none can be determined.",
            "Document/source completeness takes precedence when classifying scope. A sentence saying documentation, materials, sources, public information, or this search did not provide, mention, confirm, or include something is ALWAYS research_process_disclosure, even when it also says users must consult another source or do additional work. Documentation lacking an installation command is not the product lacking installation support.",
            "Use product_limitation only when the grammatical assertion is about the product/system capability itself, for example 'the product does not support Windows' or 'the system cannot run offline'. Do not use product_limitation for 'the documentation does not describe Windows support' or 'available materials do not provide offline setup steps'; those are research_process_disclosure.",
            "If you genuinely cannot determine whether a claim is grounded, return unsupported rather than guessing grounded -- over-flagging is low-cost here (a reader sees a [待核实] marker), while under-flagging a real fabrication is not.",
        ],
        "few_shot_examples": [
            {
                "name": "compound_claim_only_partially_supported",
                "claim": "对象甲采用层级机制，其索引依赖模型调用且计算开销较高。",
                "evidence": "对象甲的索引阶段依赖模型调用，因此计算开销较高。",
                "judgment": {
                    "status": "unsupported",
                    "claim_scope": "domain_fact",
                    "reason": "Evidence supports the indexing dependency and cost, but is silent about the claimed hierarchical mechanism. A citation after the compound sentence must support every factual component before it.",
                },
            },
            {
                "name": "uncited_analysis_after_marker_is_not_part_of_claim",
                "visible_text": "对象乙支持按子单元检索、再返回父单元 [[ev_example_2]]。这提示实现时可以分别设计匹配粒度与返回粒度。",
                "claim_received_by_validator": "对象乙支持按子单元检索、再返回父单元。",
                "evidence": "对象乙支持按子单元检索，再返回对应父单元。",
                "judgment": {
                    "status": "entailed",
                    "claim_scope": "domain_fact",
                    "reason": "The cited claim is fully supported. The separate sentence after the marker is uncited analysis and is not part of this citation judgment.",
                },
            },
        ],
        "few_shot_policy": "Examples define citation-boundary reasoning only. Judge the supplied claims and evidence independently; never copy example entities or wording into reasons.",
        "schema": {"judgments": [{
            "claim_index": 0,
            "status": "entailed|reasonable_inference|unsupported|label_mismatch",
            "claim_scope": "domain_fact|product_limitation|research_process_disclosure|unknown",
            "reason": "",
        }]},
    }
    if any(_semantic_evidence_payload(evidence) for evidence in evidence_by_id.values()):
        payload["rules"].insert(
            3,
            "support_relation and target_fit are the primary provenance contract. A fact label requires direct_support + exact_target; inference_support can justify an inference but not a target-specific fact; context_only cannot supply the factual core.",
        )
    return payload


def _judge_batch(
    section_id: str,
    rows: list[tuple[int, dict[str, Any]]],
    deterministic: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    model: Callable[[dict[str, Any]], dict[str, Any]] | None,
    payload_profile: str = ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
) -> dict[int, dict[str, Any]]:
    """Run the model on one batch of claims, returning {claim_index: model_row}.

    Model timeouts (a real case-timeout-and-page-chrome sec_01 call hit the 75s role-model
    timeout on a 17-claim payload) collapse to an empty dict here, so callers fall back to
    the deterministic row -- but only for THIS batch, not the whole section.
    """
    if not model or not rows:
        return {}
    payload = _entailment_payload(section_id, rows, evidence_by_id, payload_profile)
    result = model(payload)
    result_rows = (result.get("judgments") or []) if isinstance(result, dict) else []
    normalized: dict[int, dict[str, Any]] = {}
    for raw in result_rows:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        if item.get("claim_index") is None:
            item["claim_index"] = item.get("case_id")
        if str(item.get("status") or "") == "partial_unsupported":
            item["status"] = "unsupported"
        if str(item.get("claim_index", "")).isdigit():
            normalized[int(item["claim_index"])] = item
    return normalized


def _merge_model_rows(target: dict[int, dict[str, Any]], source: dict[int, dict[str, Any]]) -> None:
    for key, value in source.items():
        target.setdefault(key, value)


def _claim_index_of(judgment: dict[str, Any], default: int = -1) -> int:
    """Read a judgment's claim_index without the falsy-zero trap.

    `int(j.get("claim_index") or -1)` maps a real claim_index of 0 to -1 (0 is falsy in `or`),
    silently dropping the claim at index 0 from any re-review/splice that keys on it. That
    dropped 6b6146 sec_00 idx0 (the exact case Layer C was built to fix) and would drop a98ba6
    sec_00 idx0 from Layer H rescue. Use this helper everywhere a claim_index is read for
    indexing/comparison.
    """
    raw = judgment.get("claim_index")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Layer H (2026-08-15, see plan wiggly-snacking-sifakis + memory
# project_regenerated_overreach_guard_20260815): component-wise synthesis rescue. The default
# entailment judge applies a near-verbatim standard ("evidence must contain the word X") to
# synthesis claims, flagging as unsupported the connective glue a research report is supposed to
# provide -- "协同设计"/"持续性依赖"/"逐级评估" where each underlying component IS backed by
# cited evidence but the connective word is not. Calibration
# (scripts/replays/legacy/replay_a98ba6_entailment_standard.py)
# proved this is NOT fixable by relaxing the judge's prompt rules (1/4 flip, idx0 协同设计 4/5 stays
# unsupported) -- the judge's default reasoning mode treats the connective word as a claim requiring
# its own evidence. Per the project's repeatedly-validated pattern (08-09 anchor, 08-05 layer F
# rollback), soft rules do not override a model's default mode; a *structured* output that forces
# decomposition before verdict does.
#
# Layer H re-judges a narrow target set -- claims still unsupported/label_mismatch after Layers
# C+G, judged by the MODEL (not deterministic fallback), whose cited evidence all exists -- under a
# payload that requires the model to split each claim into atomic components, check each component
# against the cited evidence (cross-lingually: components are Chinese, evidence quotes are English,
# so this needs the model's semantic understanding, not code token matching -- confirmed by
# extract_terms producing all-missing across languages), and return per-component verdicts. Code
# then aggregates: every component backed -> upgrade to reasonable_inference; any component
# unbacked -> leave the original verdict untouched. ONLY UPGRADES, never downgrades -- a real
# fabrication (idx5 编译器优化, ev never mentions it; idx33 安装依赖序列, no such sequence) keeps
# an unbacked component and stays unsupported. Safe default: if the model returns no components or a
# malformed shape, leave the original verdict untouched.
_COMPONENT_RESCUE_RULES = [
    "Use only supplied evidence.",
    "The Writer was allowed to use claim, quote, writer_context, source_title, and source_url. Treat explicit details in any of those fields as supplied evidence.",
    "writer_context is section-scoped same-source context. It may support a claim only when it belongs to the cited evidence row supplied here.",
    "claim_context is the immediately preceding report sentence and is supplied only to resolve pronouns or omitted subjects in claim. Do not treat claim_context as evidence and do not use it to supply factual support.",
    "A claim may attribute content to the publication, author, or platform identifiable from a cited evidence row's own source_title/source_url -- treat that attribution as supported. Do not extend this allowance to any name, organization, or system that is not identifiable from that specific evidence row's source_title/source_url.",
    "For EACH claim, first DECOMPOSE it into its atomic components: each concrete entity, fact, relation, step, quantity, or named thing the claim asserts. Do NOT treat connective/relational/synthesis wording (如 协同/共同/整体性/持续性/逐级/形成闭环/依赖/约束) as a component that needs its own evidence -- those words glue backed components together; they are analytical prose, not factual assertions.",
    "Then for EACH component, judge whether ANY cited evidence (quote, claim, or writer_context of any cited evidence row) supports that component. Components are often Chinese while evidence quotes are often English -- judge by semantic equivalence, not verbatim word matching. A component is backed when the cited evidence makes the same concrete point, even in different language or wording.",
    "A component is unbacked ONLY when the cited evidence is entirely silent on that concrete entity/fact/relation/sequence -- the writer introduced something none of the evidence mentions at all.",
    "Return, per claim, the list of components with each one's backed flag and the evidence_ids that back it, plus the overall status: reasonable_inference if EVERY component is backed (this is a legitimate synthesis of individually-backed components), unsupported if ANY component is unbacked (the claim asserts something the evidence does not contain).",
]


def _entailment_component_payload(
    section_id: str,
    rows: list[tuple[int, dict[str, Any]]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Layer H payload: same claim/evidence fields as _entailment_payload, but rules force the
    model to decompose each claim into components and return per-component verdicts.

    The evidence field mapping is duplicated from _entailment_payload rather than shared, because
    the rules/schema differ and the two payloads are deliberately separate judgment regimes (the
    default judge vs the component-rescue re-judge). Keeping them independent avoids a change to
    one accidentally shifting the other's calibration.
    """
    payload = {
        "task": "Decompose each claim into atomic components and judge each against the cited evidence. Return JSON only.",
        "section_id": section_id,
        "claims": [{
            "claim_index": index,
            "claim": claim.get("text"),
            "claim_context": str(claim.get("claim_context") or "")[:1000],
            "claim_type": claim.get("claim_type"),
            "support_scope": claim.get("support_scope"),
            "evidence_ids": claim.get("evidence_ids") or [],
        } for index, claim in rows],
        "evidence": [{
            "id": evidence_id,
            "claim": evidence.get("atomic_claim") or evidence.get("claim"),
            "quote": str(
                evidence.get("cleaned_quote")
                or evidence.get("raw_quote")
                or evidence.get("evidence_text")
                or ""
            )[:1600],
            "writer_context": str(
                (evidence.get("validation_context_by_section") or {}).get(section_id)
                if isinstance(evidence.get("validation_context_by_section"), dict)
                else ""
            )[:3200],
            "scope": evidence.get("scope"),
            "claim_type": evidence.get("claim_type"),
            "directness_score": evidence.get("directness_score"),
            **_semantic_evidence_payload(evidence),
            "source_title": evidence.get("source_title"),
            "source_url": evidence.get("source_url"),
        } for evidence_id, evidence in evidence_by_id.items() if any(evidence_id in (claim.get("evidence_ids") or []) for _index, claim in rows)],
        "rules": list(_COMPONENT_RESCUE_RULES),
        "schema": {
            "judgments": [{
                "claim_index": 0,
                "components": [{"component": "string", "backed": True, "backing_evidence_ids": ["ev_001"], "reason": ""}],
                "status": "reasonable_inference|unsupported",
                "reason": "",
            }],
        },
    }
    if any(_semantic_evidence_payload(evidence) for evidence in evidence_by_id.values()):
        payload["rules"].insert(
            3,
            "Respect support_relation and target_fit: direct_support + exact_target may back a target fact, inference_support may back only a reasoned synthesis, and context_only cannot back a factual component by itself.",
        )
    return payload


def _judge_component_batch(
    section_id: str,
    rows: list[tuple[int, dict[str, Any]]],
    evidence_by_id: dict[str, dict[str, Any]],
    model: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[int, dict[str, Any]]:
    """Layer H model call: return {claim_index: {components, status, reason}} per claim.

    Like _judge_batch, a model timeout/empty collapses to {} so the caller leaves the original
    verdict untouched (the safe default -- no rescue, no change).
    """
    if not model or not rows:
        return {}
    payload = _entailment_component_payload(section_id, rows, evidence_by_id)
    result = model(payload)
    result_rows = (result.get("judgments") or []) if isinstance(result, dict) else []
    return {
        int(item.get("claim_index")): item
        for item in result_rows
        if isinstance(item, dict) and str(item.get("claim_index", "")).isdigit()
    }


def _aggregate_component_verdict(
    original_judgment: dict[str, Any],
    component_result: dict[str, Any] | None,
) -> tuple[str, str, bool]:
    """Decide whether a Layer H component-wise re-judge upgrades the original verdict.

    Returns (new_status, new_reason, rescued). ONLY upgrades unsupported/label_mismatch ->
    reasonable_inference; never downgrades. A claim is rescued (upgraded) only when the model
    decomposed it into >=1 component AND every component is backed by cited evidence. Any unbacked
    component, or a missing/malformed component list, leaves the original status untouched.

    Code aggregates from the per-component `backed` flags rather than trusting the model's
    overall `status` -- the whole point of Layer H is that the model's default overall verdict on
    synthesis claims is unreliable (it treats the connective word as needing evidence). The
    per-component decomposition is the structured signal worth trusting; the model's own roll-up
    status is not.
    """
    original_status = str(original_judgment.get("status") or "")
    if original_status not in {"unsupported", "label_mismatch"}:
        return original_status, str(original_judgment.get("reason") or ""), False
    if not isinstance(component_result, dict):
        return original_status, str(original_judgment.get("reason") or ""), False
    components = component_result.get("components")
    if not isinstance(components, list) or not components:
        return original_status, str(original_judgment.get("reason") or ""), False
    all_backed = all(
        isinstance(c, dict) and bool(c.get("backed"))
        for c in components
    )
    if not all_backed:
        return original_status, str(original_judgment.get("reason") or ""), False
    n = len(components)
    backing = sorted({
        str(eid)
        for c in components if isinstance(c, dict)
        for eid in (c.get("backing_evidence_ids") or [])
        if eid
    })
    backing_clause = f" (backed by {', '.join(backing[:6]) or 'cited evidence'})" if backing else ""
    new_reason = (
        f"all {n} components backed by cited evidence{backing_clause} (component-wise rescue)"
    )
    return "reasonable_inference", new_reason, True


def _normalize_entailment_claim_scope(value: Any) -> str:
    scope = str(value or "").strip().lower()
    if scope in {"domain_fact", "product_limitation", "research_process_disclosure", "unknown"}:
        return scope
    return ""


def validate_claim_entailment(
    *,
    report_document: dict[str, Any],
    evidence_map: list[dict[str, Any]],
    model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    parallelism: int = 3,
    prior_judgments: list[dict[str, Any]] | None = None,
    payload_profile: str = ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
) -> dict[str, Any]:
    if payload_profile not in {
        ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
        ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    }:
        raise ValueError(f"unsupported entailment payload profile: {payload_profile}")
    if payload_profile != ENTAILMENT_PAYLOAD_PROFILE_DEFAULT:
        prior_judgments = None
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map}
    claims = [item for item in report_document.get("claim_ledger") or [] if isinstance(item, dict)]
    by_section: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    prior_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    prior_status_by_signature: dict[tuple[Any, ...], str] = {}
    for item in prior_judgments or []:
        # Historical deterministic fallbacks and current model-unavailable rows never qualify
        # for reuse. Only a real model verdict can become a cached semantic judgment.
        if str(item.get("mode") or "") in {"deterministic_fallback", "model_unavailable"}:
            continue
        signature = (
            str(item.get("section_id") or "unknown"),
            str(item.get("claim") or ""),
            str(item.get("claim_context") or ""),
            tuple(str(value) for value in item.get("evidence_ids") or []),
            str(item.get("claim_type") or "inference"),
        )
        prior_by_signature.setdefault(signature, []).append(item)
        prior_status_by_signature[signature] = str(item.get("status") or "")
    reused_judgments: list[dict[str, Any]] = []
    claim_signature_by_index: dict[int, tuple[Any, ...]] = {}
    # 2026-08-12: reuse now keys purely on a claim's own signature (section_id + exact claim
    # text + evidence_ids + claim_type) instead of also requiring the whole section to have
    # been left untouched (the old `revalidate_section_ids` gate). A revision pass rewrites
    # a section as a whole, but individual claims within it are frequently byte-identical to
    # the prior pass -- e.g. case-critic-pass-reversal sec_00: only 1 of 5 claims actually
    # changed pass-to-pass, yet all 5 were re-judged from scratch every pass purely because
    # the section itself was "touched". Each fresh re-judgment of an unchanged claim, in
    # isolation with unchanged evidence (see judge_section's per-claim payload -- it never
    # includes surrounding section prose, only the claim's own text + its cited evidence rows,
    # so nothing about "what else changed in the section" could legitimately change the correct
    # verdict for an untouched claim), is pure judgment noise: the same claim + same evidence
    # flipped entailed -> entailed -> unsupported across 3 passes with no change to its own
    # input. Reusing whenever the signature matches eliminates this noise outright (no second
    # model call, so no question of whether the model should defer to or override its own past
    # verdict) and, as a side effect, shrinks judge_section's batches to only the claims that
    # actually changed within a partially-revised section.
    #
    # 2026-08-12 (same day, follow-up): reuse now additionally requires `confirmed: True` on the
    # cached entry, not just a signature match on the first encounter. A single independent
    # judgment on an ambiguous/boundary claim is itself noisy (see the same real job: 23 claims
    # whose text never changed still had their status vary pass-to-pass while genuinely
    # unreviewed by anyone -- about 30% of those variances were the SECOND look catching a real,
    # specific defect the first look missed: a misattribution, a fabricated qualifier, a
    # recommendation stated as unqualified fact). Freezing on the very first single judgment
    # would lock in whichever side of that noise happened to land first, forever. A verdict only
    # becomes reuse-eligible once it has been independently reached twice in a row for the exact
    # same signature (this pass's fresh judgment agreeing with the immediately preceding pass's)
    # -- bounded to one extra confirmation look, not unbounded re-litigation, and a claim that
    # keeps flip-flopping keeps getting a fresh look every pass instead of ever being trusted.
    for index, claim in enumerate(claims):
        section_id = str(claim.get("section_id") or "unknown")
        signature = (
            section_id,
            str(claim.get("text") or ""),
            str(claim.get("claim_context") or ""),
            tuple(str(value) for value in claim.get("evidence_ids") or []),
            str(claim.get("claim_type") or "inference"),
        )
        claim_signature_by_index[index] = signature
        cached = prior_by_signature.get(signature) or []
        if cached and cached[0].get("confirmed") is True:
            reused_judgments.append({**cached.pop(0), "claim_index": index, "reused": True})
            continue
        by_section.setdefault(section_id, []).append((index, claim))

    def judge_section(
        section_id: str,
        rows: list[tuple[int, dict[str, Any]]],
        batch_size_override: int | None = None,
    ) -> list[dict[str, Any]]:
        deterministic = [_deterministic_entailment(index, claim, evidence_by_id) for index, claim in rows]
        if not rows:
            return []

        def requires_semantic_review(row: dict[str, Any]) -> bool:
            reviewable_label_mismatch = (
                row["status"] == "label_mismatch"
                and row.get("claim_type") in {"inference", "recommendation"}
            )
            return not (
                row["status"] in {"unsupported", "non_atomic"}
                or (row["status"] == "label_mismatch" and not reviewable_label_mismatch)
            )

        def unavailable_row(row: dict[str, Any]) -> dict[str, Any]:
            return {
                **row,
                "status": "validation_unavailable",
                "reason": "semantic validation model did not return a verdict after retries",
                "mode": "model_unavailable",
            }

        if not model:
            return [unavailable_row(row) if requires_semantic_review(row) else row for row in deterministic]
        # Batch claims so a single timeout only makes that batch unavailable. Batches are capped
        # at RESEARCH_ENTAILMENT_BATCH_SIZE and run concurrently within the section.
        batch_size = batch_size_override or max(1, int(os.getenv("RESEARCH_ENTAILMENT_BATCH_SIZE", "6")))
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)] if batch_size < len(rows) else [rows]
        model_rows: dict[int, dict[str, Any]] = {}
        if len(batches) == 1:
            _merge_model_rows(
                model_rows,
                _judge_batch(
                    section_id, batches[0], deterministic, evidence_by_id, model, payload_profile
                ),
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batches), parallelism)) as executor:
                futures = [
                    submit_with_audit_context(
                        executor,
                        _judge_batch,
                        section_id,
                        batch,
                        deterministic,
                        evidence_by_id,
                        model,
                        payload_profile,
                    )
                    for batch in batches
                ]
                for future in as_completed(futures):
                    _merge_model_rows(model_rows, future.result())
        output = []
        for row in deterministic:
            if not requires_semantic_review(row):
                output.append(row)
                continue
            model_row = model_rows.get(row["claim_index"])
            if not model_row:
                output.append(unavailable_row(row))
                continue
            status = str(model_row.get("status") or "")
            if status not in {"entailed", "reasonable_inference", "unsupported", "label_mismatch"}:
                output.append(unavailable_row(row))
                continue
            output.append({
                **row,
                "status": status,
                "claim_scope": _normalize_entailment_claim_scope(model_row.get("claim_scope")),
                "reason": str(model_row.get("reason") or "")[:300],
                "mode": "model",
            })
        return output

    judgments: list[dict[str, Any]] = list(reused_judgments)
    with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(by_section) or 1))) as executor:
        futures = {
            submit_with_audit_context(executor, judge_section, section_id, rows): section_id
            for section_id, rows in by_section.items()
        }
        for future in as_completed(futures):
            judgments.extend(future.result())
    judgments.sort(key=lambda item: int(item.get("claim_index") or 0))

    # Retry claims whose first semantic review returned no verdict in smaller batches. A second
    # failure remains explicitly unavailable; deterministic checks are never promoted into a
    # semantic entailment verdict.
    if model and judgments:
        # Collect claims whose first-pass model batch returned nothing. Use a list of
        # (index, judgment) pairs rather than a
        # dict comprehension keyed on `int(j.get("claim_index") or -1)` -- that old form
        # maps a real claim_index of 0 to -1 (0 is falsy in `or`), so the claim at index 0
        # is silently dropped from re-review. That dropped 6b6146 sec_00 idx0 on the floor
        # exactly when it was the case we most wanted to fix.
        fallback_pairs: list[tuple[int, dict[str, Any]]] = []
        for j in judgments:
            if j.get("mode") != "model_unavailable":
                continue
            try:
                idx = int(j.get("claim_index"))
            except (TypeError, ValueError):
                continue
            fallback_pairs.append((idx, j))
        if fallback_pairs:
            # Map claim_index -> claim object. claims is the list straight off
            # report_document.claim_ledger; it does NOT carry a claim_index field (the index
            # is the list position), so use enumerate position as the index.
            # version read claim.get("claim_index") first and fell back to idx only when
            # absent -- but since the ledger never sets claim_index, that branch was dead
            # and the `int(claim.get("claim_index") or 0)` (0 for every claim) collapsed
            # the map to {0: last_claim}, dropping every other fallback claim from re-review.
            claim_by_index = {idx: claim for idx, claim in enumerate(claims)}
            # _deterministic_entailment already validated these claims had non-unknown evidence;
            # the fallback happened on the model side, not the structural side. Re-group the
            # fallback claims by section so each re-review batch hits the right evidence scope.
            re_rows_by_section: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for index, j in fallback_pairs:
                claim = claim_by_index.get(index)
                if not claim:
                    continue
                section_id = str(claim.get("section_id") or j.get("section_id") or "unknown")
                re_rows_by_section.setdefault(section_id, []).append((index, claim))
            if re_rows_by_section:
                re_batch_size = max(1, int(os.getenv("RESEARCH_ENTAILMENT_FALLBACK_BATCH_SIZE", "2")))
                re_results: dict[int, dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(re_rows_by_section) or 1))) as executor:
                    futures = {
                        submit_with_audit_context(
                            executor, judge_section, section_id, rows, re_batch_size
                        ): section_id
                        for section_id, rows in re_rows_by_section.items()
                    }
                    for future in as_completed(futures):
                        section_rows = future.result()
                        # Only a real model verdict may replace the unavailable row. A second
                        # timeout leaves the original unavailable result in place.
                        for row in section_rows:
                            if row.get("mode") == "model":
                                re_results[_claim_index_of(row)] = row
                for index, new_row in re_results.items():
                    # Splice the re-reviewed row in, preserving the position of the original
                    # fallback judgment (judgments is sorted by claim_index; re-review doesn't
                    # change claim_index, just replaces the row at that index).
                    judgments = [new_row if _claim_index_of(j) == index else j for j in judgments]
                judgments.sort(key=lambda item: int(item.get("claim_index") or 0))

    # Layer G (2026-08-06): sibling re-review for evidence-group verdict splits. A section's
    # claims are split into RESEARCH_ENTAILMENT_BATCH_SIZE-sized batches (default 6) before
    # each batch is judged in its own model call; two claims citing the exact same evidence_id
    # can land in different batches and get inconsistent verdicts purely from batch composition,
    # not from a real difference in how well-supported they are. In
    # `case-composite-claim-atomicity`, two claims added different unsupported cost and
    # complexity conclusions while citing the same process-only evidence. One landed in a
    # multi-claim batch and was judged unsupported, while the other was judged alone
    # in the next batch and came back entailed. Because _entailment_revision_issues only reads
    # claims with a bad status, idx10 never enters revision_issues, so the writer is never told
    # to touch it -- across three revision passes the same unflagged over-reach survives every
    # round and quality_gate never converges over it (see memory
    # project_layer_c_fallback_re_review_20260805 / project_quality_gate_failure_no_auto_revision_loop_20260805).
    # Detect evidence ids cited by both a model-reviewed flagged claim (unsupported/
    # label_mismatch) and a model-reviewed clean claim (entailed/reasonable_inference) --
    # a genuine model-level disagreement over identical evidence, not a structural artifact --
    # and re-review that whole evidence group together in one batch so the model judges every
    # member with the same context. As with layer C, splice in only rows that come back
    # model-reviewed (mode == "model"); if the re-review itself returns nothing, leave the
    # original verdicts untouched -- this layer only resolves a split it can actually confirm,
    # it never forces a change without a real model verdict.
    if model and judgments:
        judgment_by_index: dict[int, dict[str, Any]] = {}
        for j in judgments:
            # A reused judgment (see the prior_by_signature reuse pass above) belongs to a
            # section this revision pass explicitly decided NOT to revalidate -- its verdict
            # is carried over unchanged from an earlier pass, not "this round's model review".
            # Its `mode` field is copied verbatim from that earlier judgment and is almost
            # always "model", so without this guard it looks eligible to Layer G exactly like
            # a freshly-judged claim: if it happens to share an evidence_id with a genuinely
            # flagged/clean split among THIS round's fresh judgments (common once a report has
            # a few rounds of revision and sections start citing overlapping evidence), Layer G
            # pulls it into re-review and splices in a brand-new row that never carries the
            # `reused` key -- silently discarding the reuse (an extra, unnecessary model call)
            # even when the fresh verdict agrees with the reused one. Confirmed on a live
            # `--via-api` job (case-selective-section-revalidation): revalidate_section_ids excluded
            # sec_00, all 9 of its claims were correctly marked reused at the reuse-decision
            # pass, yet only 3 survived to the final judgments -- Layer G re-reviewed the other
            # 6 because their evidence_ids (ev_001/ev_002/ev_004) were also cited by a
            # flagged/clean split among sec_03's freshly-revised claims. See
            # docs/evaluations/deep-research-revision-loop-reuse-mismatch-20260811-handoff.md.
            if j.get("reused"):
                continue
            raw_index = j.get("claim_index")
            if not str(raw_index).lstrip("-").isdigit():
                continue
            judgment_by_index[int(raw_index)] = j
        evidence_groups: dict[str, list[int]] = {}
        for idx, j in judgment_by_index.items():
            for ev_id in j.get("evidence_ids") or []:
                evidence_groups.setdefault(str(ev_id), []).append(idx)
        split_indices: set[int] = set()
        for indices in evidence_groups.values():
            if len(indices) < 2:
                continue
            flagged = any(
                judgment_by_index[i].get("status") in {"unsupported", "label_mismatch"}
                and judgment_by_index[i].get("mode") == "model"
                for i in indices
            )
            clean = any(
                judgment_by_index[i].get("status") in {"entailed", "reasonable_inference"}
                and judgment_by_index[i].get("mode") == "model"
                for i in indices
            )
            if flagged and clean:
                split_indices.update(i for i in indices if judgment_by_index[i].get("mode") == "model")
        if split_indices:
            claim_by_index = {idx: claim for idx, claim in enumerate(claims)}
            re_rows_by_section: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for index in split_indices:
                claim = claim_by_index.get(index)
                if not claim:
                    continue
                section_id = str(claim.get("section_id") or judgment_by_index[index].get("section_id") or "unknown")
                re_rows_by_section.setdefault(section_id, []).append((index, claim))
            if re_rows_by_section:
                re_results: dict[int, dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(re_rows_by_section) or 1))) as executor:
                    futures = {
                        # Keep every member of a split evidence group in one call (up to 8
                        # rows) so the model sees them together; beyond that, fall back to the
                        # standard small-batch size to stay under the role-model timeout.
                        submit_with_audit_context(
                            executor,
                            judge_section,
                            section_id,
                            rows,
                            len(rows) if len(rows) <= 8 else max(1, int(os.getenv("RESEARCH_ENTAILMENT_FALLBACK_BATCH_SIZE", "2"))),
                        ): section_id
                        for section_id, rows in re_rows_by_section.items()
                    }
                    for future in as_completed(futures):
                        for row in future.result():
                            if row.get("mode") == "model":
                                re_results[_claim_index_of(row)] = row
                for index, new_row in re_results.items():
                    judgments = [new_row if _claim_index_of(j) == index else j for j in judgments]
                judgments.sort(key=lambda item: int(item.get("claim_index") or 0))

    # Layer H (2026-08-15, see _entailment_component_payload docstring + plan
    # wiggly-snacking-sifakis): component-wise synthesis rescue. After Layers C+G, claims still
    # unsupported/label_mismatch that were judged by the MODEL (not deterministic fallback) and
    # whose cited evidence all exists are candidates for "the judge applied a near-verbatim
    # standard to a legitimate synthesis". Re-judge them under a payload that forces the model to
    # decompose each claim into atomic components and return per-component verdicts; code then
    # upgrades to reasonable_inference only when EVERY component is backed. Env-gated OFF by
    # default -- calibration (scripts/replays/legacy/replay_a98ba6_entailment_standard.py)
    # must confirm the rescue
    # flips the 4 legitimate-synthesis a98ba6 claims WITHOUT flipping the 2 real fabrications
    # (idx5 编译器优化, idx33 安装序列) before this is enabled in production. Same shape as
    # Layer G: collect targets -> regroup by section -> re-judge -> splice only upgrades.
    if (
        model
        and payload_profile == ENTAILMENT_PAYLOAD_PROFILE_DEFAULT
        and os.getenv("RESEARCH_ENTAILMENT_COMPONENT_RESCUE", "0") == "1"
        and judgments
    ):
        claim_by_index_h: dict[int, dict[str, Any]] = {}
        for j in judgments:
            if j.get("reused"):
                continue
            raw_index = j.get("claim_index")
            if not str(raw_index).lstrip("-").isdigit():
                continue
            claim_by_index_h[int(raw_index)] = j
        # Targets: model-judged unsupported/label_mismatch whose cited evidence all exists.
        rescue_targets: set[int] = set()
        for idx, j in claim_by_index_h.items():
            if str(j.get("mode") or "") != "model":
                continue
            if str(j.get("status") or "") not in {"unsupported", "label_mismatch"}:
                continue
            eids = [str(e) for e in j.get("evidence_ids") or []]
            if eids and all(eid in evidence_by_id for eid in eids):
                rescue_targets.add(idx)
        if rescue_targets:
            re_rows_by_section_h: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for index in rescue_targets:
                j = claim_by_index_h[index]
                claim = {
                    "text": j.get("claim"),
                    "claim_type": j.get("claim_type") or "inference",
                    "support_scope": j.get("support_scope") or "cross_domain",
                    "evidence_ids": [str(e) for e in j.get("evidence_ids") or []],
                }
                section_id = str(j.get("section_id") or "unknown")
                re_rows_by_section_h.setdefault(section_id, []).append((index, claim))
            if re_rows_by_section_h:
                re_results_h: dict[int, dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(re_rows_by_section_h) or 1))) as executor:
                    futures = {
                        submit_with_audit_context(
                            executor, _judge_component_batch, section_id, rows, evidence_by_id, model
                        ): section_id
                        for section_id, rows in re_rows_by_section_h.items()
                    }
                    for future in as_completed(futures):
                        for index, result in future.result().items():
                            re_results_h[index] = result
                for index, component_result in re_results_h.items():
                    original = next((j for j in judgments if _claim_index_of(j) == index), None)
                    if not original:
                        continue
                    new_status, new_reason, rescued = _aggregate_component_verdict(original, component_result)
                    if not rescued:
                        continue
                    upgraded = {
                        **original,
                        "status": new_status,
                        "reason": new_reason[:300],
                        "mode": "model",
                        "component_rescue": True,
                        # A rescued claim has NOT been confirmed by a second pass yet -- it was
                        # just upgraded THIS pass, so it must not be frozen as confirmed.
                        "confirmed": False,
                    }
                    judgments = [upgraded if _claim_index_of(j) == index else j for j in judgments]
                judgments.sort(key=lambda item: int(item.get("claim_index") or 0))

    # "Confirm twice" tagging (see the reuse-decision comment above): run last, after Layer C
    # and Layer G have both finished mutating `judgments`, so this checks each claim's FINAL
    # status for the pass, not a pre-re-review draft. A judgment is reuse-eligible next pass
    # only once it agrees with what the immediately preceding pass concluded for the exact same
    # signature -- reused judgments already carry `confirmed` forward from their source and are
    # left untouched; every other judgment gets confirmed set fresh here.
    for j in judgments:
        if j.get("reused"):
            continue
        if j.get("mode") != "model":
            j["confirmed"] = False
            continue
        signature = claim_signature_by_index.get(j.get("claim_index"))
        prior_status = prior_status_by_signature.get(signature) if signature is not None else None
        j["confirmed"] = prior_status is not None and prior_status == j.get("status")

    blocking = [item for item in judgments if item.get("status") in {"unsupported", "label_mismatch", "non_atomic"}]
    unavailable = [item for item in judgments if item.get("status") == "validation_unavailable"]
    return {
        "status": "failed" if blocking or unavailable else "passed",
        "payload_profile": payload_profile,
        "prompt_version": (
            ATOMIC_BOUNDARY_ENTAILMENT_PROMPT_VERSION
            if payload_profile == ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2
            else ENTAILMENT_PROMPT_VERSION
        ),
        "claim_count": len(judgments),
        "entailed_count": sum(item.get("status") == "entailed" for item in judgments),
        "inference_count": sum(item.get("status") == "reasonable_inference" for item in judgments),
        "unsupported_count": sum(item.get("status") == "unsupported" for item in judgments),
        "label_mismatch_count": sum(item.get("status") == "label_mismatch" for item in judgments),
        "non_atomic_count": sum(item.get("status") == "non_atomic" for item in judgments),
        "unsupported_inference_count": len(blocking),
        "model_reviewed_count": sum(item.get("mode") == "model" for item in judgments),
        "semantic_review_unavailable_count": len(unavailable),
        # Compatibility for historical diagnostics readers. New runs never emit this mode.
        "deterministic_fallback_count": 0,
        "reused_judgment_count": len(reused_judgments),
        "judgments": judgments,
    }


def _deterministic_entailment(
    index: int,
    claim: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_ids = [str(item) for item in claim.get("evidence_ids") or []]
    evidence = [evidence_by_id[item] for item in evidence_ids if item in evidence_by_id]
    claim_type = str(claim.get("claim_type") or "inference")
    text = str(claim.get("text") or "")
    base = {
        "claim_index": index,
        "section_id": claim.get("section_id"),
        "claim": text,
        "claim_context": str(claim.get("claim_context") or ""),
        "claim_type": claim_type,
        "evidence_ids": evidence_ids,
        "mode": "deterministic",
    }
    if not is_atomic_claim(text):
        return {**base, "status": "non_atomic", "reason": "claim contains multiple propositions or exceeds the atomic claim size limit"}
    if not evidence or len(evidence) != len(evidence_ids):
        return {**base, "status": "unsupported", "reason": "claim has missing or unknown evidence"}
    if claim_type == "fact" and not all(is_direct_exact_evidence(item) for item in evidence):
        return {**base, "status": "label_mismatch", "reason": "fact is supported only by general, cross-domain, or indirect evidence"}
    if claim_type in {"inference", "recommendation"}:
        if not _has_qualified_claim_language(text, claim_type):
            return {**base, "status": "label_mismatch", "reason": "inference or recommendation is written as an unqualified fact"}
        return {**base, "status": "reasonable_inference", "reason": "claim is explicitly qualified"}
    return {**base, "status": "entailed", "reason": "direct evidence and claim type are compatible"}


def _has_qualified_claim_language(text: str, claim_type: str) -> bool:
    normalized = str(text or "").casefold()
    inference_markers = (
        "推断", "可能", "或许", "预计", "可映射", "可作为", "可以认为", "倾向于",
        "基于现有证据", "这意味着", "也就是说", "值得注意的是", "由此可见", "表明", "说明",
        "显示", "暗示",
    )
    inference_english_markers = (
        "infer", "inferred", "suggest", "suggests", "suggested", "indicate", "indicates", "indicated",
        "imply", "implies", "implied", "mean that", "means that", "may", "might", "could", "likely",
        "appears", "seems", "potentially", "based on available evidence",
    )
    recommendation_markers = (
        "建议", "可考虑", "候选", "候选设计", "可以采用", "可以选择", "值得考虑", "优先考虑",
        "应当", "应该", "有必要", "宜采用", "宜选择",
    )
    recommendation_english_markers = (
        "recommend", "recommends", "recommended", "should", "consider", "candidate", "advisable",
        "propose", "proposes", "proposed", "could adopt", "could use", "may want to",
    )
    markers = recommendation_markers if claim_type == "recommendation" else inference_markers
    english_markers = recommendation_english_markers if claim_type == "recommendation" else inference_english_markers
    return any(marker in normalized for marker in markers) or any(
        re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in english_markers
    )


def validate_report_citations(
    *,
    documents: list[dict[str, Any]],
    report_document: dict[str, Any],
    evidence_map: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence_map}
    source_urls = list(dict.fromkeys(str(item.get("source_url") or "") for item in evidence_map if item.get("source_url")))
    per_document = []
    reasons = []
    for document in documents:
        ledger = [item for item in document.get("claim_ledger") or [] if isinstance(item, dict)]
        unknown = sorted({str(evidence_id) for item in ledger for evidence_id in item.get("evidence_ids") or [] if str(evidence_id) not in evidence_by_id})
        refs = document.get("evidence_refs") or []
        if unknown:
            reasons.append(f"{document.get('title')} claim ledger contains unknown evidence")
        if ledger and not refs:
            reasons.append(f"{document.get('title')} has claims without evidence refs")
        per_document.append({
            "title": document.get("title"),
            "evidence_ref_count": len(refs),
            "matched_evidence_ref_count": len(refs) - len(unknown),
            "matched_by_id_count": len(refs) - len(unknown),
            "matched_by_fallback_count": 0,
            "match_ratio": round((len(refs) - len(unknown)) / len(refs), 3) if refs else (1.0 if not ledger else 0.0),
            "unsupported_refs": [{"evidence_id": item} for item in unknown],
            "unsupported_claims": [],
            "blocking_unsupported_claims": [],
            "missing_inline_refs": [],
        })
    content = str(report_document.get("content") or "")
    used_numbers = {int(item) for item in re.findall(r"\[(\d+)\]", content)}
    invalid_numbers = sorted(number for number in used_numbers if number < 1 or number > len(source_urls))
    if invalid_numbers:
        reasons.append("report contains invalid source references")
    if "[ev_" in content:
        reasons.append("report exposes internal evidence ids")
    if re.search(r"\bclaim_[A-Za-z0-9_-]+\b", content):
        reasons.append("report exposes internal claim ids")
    return {
        "status": "passed" if not reasons else "failed",
        "reason": "新写作链路引用验收通过。" if not reasons else "；".join(reasons),
        "per_document": per_document,
        "per_section": [],
        "evidence_url_count": len(source_urls),
        "evidence_claim_count": len(evidence_map),
        "evidence_id_count": len(evidence_by_id),
        "invalid_source_refs": invalid_numbers,
    }
def _is_harmless_reference_fallback(document: dict[str, Any]) -> bool:
    """A reference/citation-index section (e.g. "05_证据与参考") never gets its own research
    questions, so it always has zero findings by design and its writer call is never even
    attempted — see writer.py's _write_section. Its fallback body is also never shown to the
    reader: _assemble_report discards this section's content entirely and renders the real
    source list instead (matched by the same _is_reference_title check). So this specific
    fallback is a structural non-issue, not a degraded section, and should not block approval.
    Any other fallback reason on a reference-titled section (e.g. an actual model failure) still
    counts as blocking below.
    """
    title = str(document.get("title") or "")
    diagnostics = document.get("synthesis_diagnostics") or {}
    return _is_reference_title(title) and diagnostics.get("fallback_reason") == "no_findings_or_evidence"


def _noise_hits(content: str) -> list[str]:
    patterns = {
        "data_uri": r"data:image|<svg",
        "mermaid_fragment": r"mermaid\s+flowchart",
        "truncated_link": r"\[[^\]]+\]\(https?://[^)]*$",
        "promotion": r"商务合作|关注公众号|全站搜索",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, content, re.I | re.M)]
