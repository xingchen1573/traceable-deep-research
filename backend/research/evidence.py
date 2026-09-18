"""Evidence admission, cleaning, grounding, and deduplication."""

from __future__ import annotations

import os
import re
from hashlib import sha1
from typing import Any
from urllib.parse import urlparse

from .source_policy import (
    entity_match_metadata,
    normalize_question_kind,
    normalize_source_type,
    publisher_relation,
    strict_entity_topic,
)
from .text_utils import extract_key_terms, to_simplified


NOISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("data_uri_or_image", re.compile(r"!\[[^\]]*\]\(data:|data:image|<svg", re.I)),
    ("markdown_heading", re.compile(r"^\s{0,3}#{1,6}\s+")),
    ("navigation", re.compile(r"(?:toggle|subscribe|cookie|sign in|login|read more|首页|导航栏|网站导航|主导航|导航菜单|上一篇|下一篇|返回顶部)", re.I)),
    ("page_chrome", re.compile(r"(?:skip to content|navigation menu|appearance settings|跳至主要内容|技术领域\s+close|view all features)", re.I)),
    ("mermaid_or_code", re.compile(r"(?:mermaid\s+(?:flowchart|graph)|(?:flowchart|graph)\s+(?:TD|LR)|```|Handoff\s*\(|\w+\([^)]*target\s*=)", re.I)),
    ("truncated_link", re.compile(r"https?://[^\s)]{0,80}$|\[[^\]]+\]\(https?://[^)]*$", re.I)),
    ("promotion", re.compile(r"(?:商务合作|关注公众号|扫码关注|全站搜索|手把手|纯干货|限时优惠)", re.I)),
    ("table_fragment", re.compile(r"\|\s*:?-{3,}:?\s*\||(?:\|[^|]{0,80}){4,}", re.I)),
    ("page_header_blob", re.compile(r"(?:专刊|前沿技术).{0,80}(?:文\s*[|｜]|摘要|本文)", re.I)),
    ("asset_or_hash_fragment", re.compile(r"(?:^|[\s/])[a-f0-9]{24,}(?:/|$)|(?:image|figure|asset)[-_]?\d+\.(?:png|jpe?g|svg)", re.I)),
)


_QUESTION_EMBEDDING_CACHE: dict[str, list[float]] = {}
_SUBJECT_RELATIONS = {"target", "target_component", "comparison", "general", "unknown"}
SUPPORT_RELATIONS = {"direct_support", "inference_support", "context_only", "unrelated"}
TARGET_FITS = {"exact_target", "related_target", "general"}

_SUPPORT_COMPATIBILITY = {
    "direct_support": 0.90,
    "inference_support": 0.65,
    "context_only": 0.40,
    "unrelated": 0.0,
}
_TARGET_COMPATIBILITY = {
    "exact_target": ("domain_specific", 0.90),
    "related_target": ("cross_domain", 0.60),
    "general": ("general", 0.40),
}


def evidence_semantic_arbitration_mode() -> str:
    value = os.getenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "off").strip().lower()
    return value if value in {"off", "shadow", "live"} else "off"


def evidence_support_relation(item: dict[str, Any]) -> str:
    value = str(item.get("support_relation") or "").strip().lower()
    if value in SUPPORT_RELATIONS:
        return value
    scope = str(item.get("scope") or "general")
    directness = _score_value(item.get("directness_score"), 0.0)
    if scope == "domain_specific" and directness >= 0.75:
        return "direct_support"
    if directness >= 0.55:
        return "inference_support"
    return "context_only"


def evidence_target_fit(item: dict[str, Any]) -> str:
    value = str(item.get("target_fit") or "").strip().lower()
    if value in TARGET_FITS:
        return value
    return {
        "domain_specific": "exact_target",
        "cross_domain": "related_target",
        "general": "general",
    }.get(str(item.get("scope") or "general"), "general")


def is_direct_exact_evidence(item: dict[str, Any]) -> bool:
    explicit_support = str(item.get("support_relation") or "").strip().lower()
    explicit_target = str(item.get("target_fit") or "").strip().lower()
    if explicit_support in SUPPORT_RELATIONS or explicit_target in TARGET_FITS:
        return explicit_support == "direct_support" and explicit_target == "exact_target"
    return (
        str(item.get("scope") or "general") == "domain_specific"
        and str(item.get("claim_type") or item.get("evidence_type") or "") == "direct_fact"
        and _score_value(item.get("directness_score"), 0.0) >= 0.75
    )


def semantic_relation_allows_criterion(item: dict[str, Any]) -> bool:
    return evidence_support_relation(item) in {"direct_support", "inference_support"}


def _normalize_subject_relation(value: Any) -> str:
    relation = str(value or "").strip().lower()
    aliases = {
        "component": "target_component",
        "comparator": "comparison",
        "compare": "comparison",
        "background": "general",
    }
    relation = aliases.get(relation, relation)
    return relation if relation in _SUBJECT_RELATIONS else "unknown"


def _normalize_comparison_axis(value: Any) -> str:
    return "".join(char.lower() for char in str(value or "") if char.isalnum())


def _comparison_axes_compatible(actual: Any, expected: Any) -> bool:
    """Accept exact axes and small label-form extensions, not neighboring dimensions."""
    actual_value = _normalize_comparison_axis(actual)
    expected_value = _normalize_comparison_axis(expected)
    if not actual_value or not expected_value:
        return False
    if actual_value == expected_value:
        return True
    shorter, longer = sorted((actual_value, expected_value), key=len)
    return (
        (longer.startswith(shorter) or longer.endswith(shorter))
        and len(longer) - len(shorter) <= 2
    )


def comparison_subjects_compatible(actual: Any, expected: Any) -> bool:
    """Accept stable product-name extensions without conflating neighboring products."""
    actual_value = _normalize_comparison_axis(to_simplified(actual))
    expected_value = _normalize_comparison_axis(to_simplified(expected))
    if not actual_value or not expected_value:
        return False
    if actual_value == expected_value:
        return True
    shorter, longer = sorted((actual_value, expected_value), key=len)
    return len(shorter) >= 4 and (longer.startswith(shorter) or longer.endswith(shorter))


_COMPARISON_AXIS_MARKERS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("证据引用", "引用", "citation", "references"), (
        "引用", "引文", "出处", "来源", "参考文献", "参考资料", "内联链接", "来源链接",
        "可追溯", "可追踪", "审计轨迹", "citation", "cite", "reference", "source",
        "inline link", "traceable", "auditable",
    )),
    (("报告生成", "报告", "report generation"), (
        "报告", "撰写", "写作", "生成内容", "导出", "输出为", "文档", "摘要", "总结",
        "report", "generate", "generation", "write", "export", "output", "pdf", "document",
    )),
    (("适用场景", "应用场景", "use cases"), (
        "适合", "适用", "用于", "应用", "场景", "用户", "领域", "市场分析", "竞品研究",
        "决策", "工作流", "use case", "suitable", "fits", "fit for", "best for", "ideal for",
        "choose", "when you need", "appropriate for", "designed for", "built for", "workflow",
    )),
    (("架构", "architecture"), (
        "架构", "结构", "图谱", "社区", "索引", "节点", "实体", "关系", "层级", "组件",
        "architecture", "graph", "community", "index", "node", "entity", "relation", "component",
    )),
    (("检索流程", "检索", "retrieval"), (
        "检索", "查询", "搜索", "召回", "遍历", "多跳", "聚合", "排序", "重排", "响应",
        "retrieval", "retrieve", "query", "search", "recall", "ranking", "rerank", "map-reduce",
    )),
    (("局限", "限制", "limitations"), (
        "局限", "限制", "缺点", "不足", "问题", "风险", "成本", "开销", "昂贵", "高昂", "延迟", "失败",
        "不能", "不适合", "难以", "弱于", "瓶颈", "limitation", "drawback", "risk", "cost",
        "expensive", "latency", "fail", "cannot", "not suitable", "challenge", "bottleneck",
    )),
    (("成本", "价格", "cost"), (
        "成本", "价格", "费用", "订阅", "计费", "预算", "token", "api调用", "cost", "price",
        "pricing", "subscription", "billing", "budget",
    )),
)


def _comparison_axis_evidence_supported(expected_axis: Any, quote: Any) -> bool:
    expected_value = _normalize_comparison_axis(expected_axis)
    if not expected_value:
        return True
    markers: tuple[str, ...] | None = None
    for aliases, candidate_markers in _COMPARISON_AXIS_MARKERS:
        if any(_comparison_axes_compatible(expected_value, alias) for alias in aliases):
            markers = candidate_markers
            break
    if markers is None:
        return True
    text = _normalize_comparison_axis(to_simplified(quote))
    return any(_normalize_comparison_axis(marker) in text for marker in markers)


def _semantic_relevance_enabled() -> bool:
    import os

    raw = os.getenv("RESEARCH_EVIDENCE_SEMANTIC_RELEVANCE_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _semantic_relevance_threshold() -> float:
    import os

    try:
        return float(os.getenv("RESEARCH_EVIDENCE_SEMANTIC_RELEVANCE_THRESHOLD", "0.38"))
    except (TypeError, ValueError):
        return 0.38


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(x * y for x, y in zip(left, right))
    norm_left = sum(x * x for x in left) ** 0.5
    norm_right = sum(y * y for y in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def _semantic_rescue_mask(*, question: str, texts: list[str]) -> list[bool]:
    """Batch-check whether each of `texts` is semantically close enough to `question` to
    rescue an item that `topic_relevant`'s pure keyword matching flagged as off_topic.

    `texts` are the CLAIMS (model-generated Chinese assertions of each fact), not the raw
    quotes: the quote is frequently English source code / docstrings that don't contain the
    Chinese question's keywords and have low embedding similarity to the question (real
    job_20260814 data: quote sims 0.24-0.40, the same content as a Chinese claim 0.35-0.61).
    Embedding the claim instead recovers the ~71% of off_topic rejections that are real
    keyword-mismatch false positives without admitting genuinely irrelevant evidence (which
    scores 0.15-0.29, well under the threshold). See the audit in
    docs/evaluations/deep-research-5case-e2e-20260814.md.

    A ranking-stage chunk score was tried first and rejected: real trace data showed
    topic_relevant=True/False groups have near-identical `score` distributions (that score
    measures chunk-to-search-query similarity, not claim-to-question similarity, so it has no
    discriminative power here). A fresh (question, claim) embedding cosine similarity does
    separate the two groups on the same real data -- see
    docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.2.md
    决策7 for why any such threshold must stay env-driven rather than a hardcoded constant.

    Fails open (returns all False, i.e. no rescue -- keeps the original keyword-only verdict)
    on any embedding error, mirroring `_hybrid_rerank_scored_chunks`'s degrade-to-lexical-only
    behavior in workers.py: an embedding outage must never crash or block evidence admission.
    """
    return [score >= _semantic_relevance_threshold() for score in _semantic_rescue_scores(question=question, texts=texts)]


def _semantic_rescue_scores(*, question: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    try:
        from workers import _embed_texts  # lazy: avoid workers.py's app-init import cycle

        question_embedding = _QUESTION_EMBEDDING_CACHE.get(question)
        if question_embedding is None:
            embeddings = _embed_texts([question, *texts])
            question_embedding, text_embeddings = embeddings[0], embeddings[1:]
            _QUESTION_EMBEDDING_CACHE[question] = question_embedding
        else:
            text_embeddings = _embed_texts(texts)
        return [_cosine_similarity(question_embedding, embedding) for embedding in text_embeddings]
    except Exception:
        return [0.0] * len(texts)


def _normalize_support_relation(
    value: Any,
    *,
    scope: str,
    directness: float,
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORT_RELATIONS:
        return normalized
    if scope == "domain_specific" and directness >= 0.75:
        return "direct_support"
    if directness >= 0.55:
        return "inference_support"
    return "context_only"


def _normalize_target_fit(value: Any, *, scope: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in TARGET_FITS:
        return normalized
    return {
        "domain_specific": "exact_target",
        "cross_domain": "related_target",
        "general": "general",
    }.get(scope, "general")


def _lexical_scope_signal(question: str, claim: str, quote: str) -> dict[str, Any]:
    domain_terms = _domain_terms(question)
    corpus = f"{claim} {quote}".lower()
    matched = [term for term in domain_terms if term in corpus]
    overlap = len(matched) / max(1, len(domain_terms))
    if domain_terms and overlap >= 0.35:
        signal = "domain_specific"
    elif matched:
        signal = "cross_domain"
    else:
        signal = "general"
    return {
        "lexical_scope_signal": signal,
        "lexical_overlap_ratio": round(overlap, 4),
        "matched_domain_terms": matched,
    }


def _semantic_conflict_reasons(item: dict[str, Any], *, similarity: float) -> list[str]:
    arbitration = item.get("semantic_arbitration") or {}
    initial = arbitration.get("initial") or {}
    signals = arbitration.get("program_signals") or {}
    support_relation = str(initial.get("support_relation") or "")
    target_fit = str(initial.get("target_fit") or "")
    lexical_scope = str(signals.get("lexical_scope_signal") or "")
    reasons: list[str] = []
    if support_relation == "direct_support" and target_fit == "exact_target" and lexical_scope == "general":
        reasons.append("model_direct_exact_but_lexical_general")
    verified_axis = str(item.get("comparison_axis_binding_status") or "") == "verified_axis"
    verified_subject = str(item.get("criterion_binding_status") or "").startswith("verified_")
    target_phrase = str(item.get("subject") or "").strip().lower()
    corpus = f"{item.get('claim') or ''} {item.get('raw_quote') or ''}".lower()
    if (
        support_relation == "direct_support"
        and similarity < _semantic_relevance_threshold()
        and not (target_phrase and target_phrase in corpus)
        and not verified_axis
    ):
        reasons.append("model_direct_but_semantic_similarity_low")
    if (
        support_relation in {"inference_support", "context_only"}
        and target_fit == "exact_target"
        and verified_subject
        and (not item.get("expected_comparison_axis") or verified_axis)
        and lexical_scope == "domain_specific"
        and similarity >= min(0.95, _semantic_relevance_threshold() + 0.12)
    ):
        reasons.append("model_conservative_but_program_signals_strong")
    raw_directness = _score_value(initial.get("directness_score"), 0.0)
    raw_applicability = _score_value(initial.get("applicability_score"), 0.0)
    raw_scope = str(initial.get("scope") or "")
    if support_relation == "direct_support" and raw_directness < 0.75:
        reasons.append("support_relation_conflicts_with_directness_score")
    if support_relation != "direct_support" and raw_directness >= 0.75:
        reasons.append("indirect_relation_conflicts_with_directness_score")
    if target_fit == "exact_target" and (raw_scope not in {"", "domain_specific"} or raw_applicability < 0.75):
        reasons.append("target_fit_conflicts_with_legacy_applicability")
    if item.get("_semantic_lexical_off_topic") and support_relation == "direct_support":
        reasons.append("lexical_off_topic_only")
    return list(dict.fromkeys(reasons))


def _attach_semantic_diagnostics(items: list[dict[str, Any]], *, question: str) -> None:
    if not items:
        return
    similarities = _semantic_rescue_scores(
        question=question,
        texts=[str(item.get("claim") or "") for item in items],
    ) if question else [0.0] * len(items)
    for item, similarity in zip(items, similarities):
        arbitration = item.get("semantic_arbitration")
        if not isinstance(arbitration, dict):
            continue
        signals = arbitration.setdefault("program_signals", {})
        signals["semantic_similarity"] = round(float(similarity), 4)
        reasons = _semantic_conflict_reasons(item, similarity=float(similarity))
        arbitration["conflict_reasons"] = reasons
        arbitration["status"] = "pending" if reasons else "not_needed"
        item.pop("_semantic_lexical_off_topic", None)


def apply_semantic_arbitration(
    items: list[dict[str, Any]],
    *,
    verdicts: dict[str, dict[str, Any]] | None = None,
    failure_status: str = "",
    failure_reason: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply live verdicts or record shadow verdicts after hard admission has succeeded."""
    mode = evidence_semantic_arbitration_mode()
    if mode == "off":
        return items, []
    verdicts = verdicts or {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        arbitration = item.get("semantic_arbitration")
        if not isinstance(arbitration, dict):
            accepted.append(item)
            continue
        reasons = arbitration.get("conflict_reasons") or []
        initial = arbitration.get("initial") or {}
        evidence_id = str(arbitration.get("evidence_id") or "")
        verdict = verdicts.get(evidence_id) if reasons else None
        if reasons and verdict:
            arbitration["status"] = "shadowed" if mode == "shadow" else "resolved"
            arbitration["verdict"] = verdict
        elif reasons:
            arbitration["status"] = failure_status or "unavailable"
            arbitration["failure_reason"] = str(failure_reason or "arbitration_unavailable")[:300]
        chosen = verdict or initial
        if reasons and not verdict:
            support = str(initial.get("support_relation") or "context_only")
            chosen = {
                **initial,
                "support_relation": "inference_support" if support == "direct_support" else support,
                "reason": str(failure_reason or "unresolved semantic conflict; conservatively downgraded")[:160],
                "confidence": "low",
            }
            arbitration["fallback"] = {
                "support_relation": chosen["support_relation"],
                "target_fit": chosen.get("target_fit"),
                "reason": chosen.get("reason"),
            }
        arbitration["effective"] = {
            "support_relation": chosen.get("support_relation"),
            "target_fit": chosen.get("target_fit"),
        }
        if mode == "shadow":
            arbitration["would_change"] = any(
                chosen.get(key) != initial.get(key)
                for key in ("support_relation", "target_fit")
            )
            accepted.append(item)
            continue
        support_relation = str(chosen.get("support_relation") or "context_only")
        target_fit = str(chosen.get("target_fit") or "general")
        if support_relation not in SUPPORT_RELATIONS:
            support_relation = "inference_support" if reasons else "context_only"
        if target_fit not in TARGET_FITS:
            target_fit = "general"
        scope, applicability = _TARGET_COMPATIBILITY[target_fit]
        item["support_relation"] = support_relation
        item["target_fit"] = target_fit
        item["scope"] = scope
        item["directness_score"] = _SUPPORT_COMPATIBILITY[support_relation]
        item["applicability_score"] = applicability
        item["claim_type"] = normalize_claim_type(
            item.get("claim_type") or item.get("evidence_type"),
            scope=scope,
            stance=str(item.get("stance") or "support"),
        )
        if support_relation != "direct_support" and item["claim_type"] == "direct_fact":
            item["claim_type"] = "design_principle"
        item["evidence_type"] = item["claim_type"]
        item["evidence_score"] = evidence_score(
            claim=str(item.get("claim") or ""),
            quote=str(item.get("raw_quote") or item.get("evidence_text") or ""),
            source_url=str(item.get("source_url") or ""),
            source_type=str(item.get("source_type") or ""),
            grounded=bool(item.get("grounded_exactly")),
            relevant=support_relation != "unrelated",
            directness=float(item["directness_score"]),
            applicability=float(item["applicability_score"]),
        )
        if support_relation == "unrelated":
            item["admission_status"] = "quarantined"
            item["rejection_reason"] = "semantic_unrelated"
            item["quality_flags"] = list(dict.fromkeys([*item.get("quality_flags", []), "semantic_unrelated"]))
            rejected.append(item)
        else:
            item["admission_status"] = (
                "accepted" if support_relation == "direct_support" else "inference_only"
            )
            accepted.append(item)
    return accepted, rejected


def admit_evidence(
    items: list[Any],
    *,
    chunks: list[dict[str, Any]],
    question_id: str,
    section_id: str,
    question: str = "",
    target_subject: str = "",
    comparison_subjects: list[str] | None = None,
    expected_comparison_axis: str = "",
    valid_criterion_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return accepted and quarantined evidence with stable audit metadata."""
    chunk_by_id = {str(item.get("id") or ""): item for item in chunks if isinstance(item, dict)}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    pending_rescue: list[tuple[dict[str, Any], str, tuple[str, str]]] = []  # (normalized, claim, key)
    semantic_enabled = _semantic_relevance_enabled()
    semantic_mode = evidence_semantic_arbitration_mode()
    comparison_question = normalize_question_kind(
        "",
        question=question,
        purpose="",
        success_criteria=[],
    ) == "comparison"
    allowed_comparison_subjects = [
        str(value).strip()
        for value in comparison_subjects or []
        if str(value).strip()
    ]

    def matches_allowed_subject(subject: str) -> bool:
        if not subject or not allowed_comparison_subjects:
            return False
        return any(
            comparison_subjects_compatible(subject, value)
            or float(entity_match_metadata(title=subject, snippet=subject, topic=value).get("entity_match_score") or 0.0) >= 0.75
            for value in allowed_comparison_subjects
        )
    for item_index, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            continue
        semantic_evidence_id = str(raw.get("id") or f"candidate_{item_index:03d}")
        claim = clean_text(raw.get("atomic_claim") or raw.get("claim") or raw.get("fact"))
        quote = clean_text(raw.get("raw_quote") or raw.get("cleaned_quote") or raw.get("evidence_text") or raw.get("evidence"))
        # Strip a trailing cut-URL/markdown-link fragment BEFORE quote_span/evidence_issue: when
        # excerpt slicing landed mid-URL the trailing fragment trips truncated_link and breaks
        # grounding. Prefix-preserving, so the kept quote still resolves against chunk_text.
        quote = _strip_trailing_url_fragment(quote)
        claim = strip_question_prefix(claim, question)
        chunk_id = str(raw.get("chunk_id") or "")
        chunk = chunk_by_id.get(chunk_id)
        source_url = str(raw.get("source_url") or (chunk or {}).get("source_url") or "").strip()
        source_title = str(raw.get("source_title") or (chunk or {}).get("source_title") or "").strip()
        entity = entity_match_metadata(
            url=source_url,
            title=source_title,
            snippet=f"{claim} {quote}",
            topic=question,
        )
        issue = evidence_issue(claim=claim, quote=quote, source_url=source_url)
        span = quote_span(quote, str((chunk or {}).get("text") or "")) if chunk else None
        grounded = span is not None
        if not issue and chunk and not grounded:
            issue = "quote_not_grounded"
        if not issue and not chunk:
            issue = "missing_chunk"
        if not issue and float(entity.get("entity_match_score") or 0.0) < 0.5:
            issue = "entity_mismatch"
        # topic_relevant (pure keyword match) is checked LAST among the structural issues above
        # so that, if it is the only thing that fires, the semantic rescue below knows every
        # other structural check has already had its chance -- rescuing off_topic never papers
        # over a quote that would otherwise have failed grounding/missing_chunk too.
        eligible_for_rescue = False
        lexical_off_topic = False
        if not issue and question and not topic_relevant(quote, question):
            lexical_off_topic = True
            if semantic_mode != "live":
                issue = "off_topic"
                eligible_for_rescue = semantic_enabled
        source_type = normalize_source_type(
            raw.get("source_type") or (chunk or {}).get("source_type"),
            url=source_url,
            title=source_title,
            text=quote,
        )
        scope, directness, applicability = evidence_applicability(
            question=question,
            claim=claim,
            quote=quote,
            model_scope=raw.get("scope"),
            model_directness=raw.get("directness_score"),
            model_applicability=raw.get("applicability_score"),
        )
        initial_support_relation = _normalize_support_relation(
            raw.get("support_relation"),
            scope=str(raw.get("scope") or scope),
            directness=_score_value(raw.get("directness_score"), directness),
        )
        initial_target_fit = _normalize_target_fit(
            raw.get("target_fit"),
            scope=str(raw.get("scope") or scope),
        )
        raw_support_relation = str(raw.get("support_relation") or "").strip().lower()
        raw_target_fit = str(raw.get("target_fit") or "").strip().lower()
        if semantic_mode == "live" and raw_support_relation and raw_support_relation not in SUPPORT_RELATIONS:
            issue = issue or "invalid_support_relation"
        if semantic_mode == "live" and raw_target_fit and raw_target_fit not in TARGET_FITS:
            issue = issue or "invalid_target_fit"
        if (
            not issue
            and strict_entity_topic(question)
            and 0.5 <= float(entity.get("entity_match_score") or 0.0) < 1.0
            and any(marker in f"{claim} {quote}".lower() for marker in ("install", "pip", "model", "config", "参数", "安装", "配置"))
            and (
                (directness < 0.85 or applicability < 0.8)
                if semantic_mode != "live"
                else not (
                    initial_support_relation == "direct_support"
                    and initial_target_fit == "exact_target"
                )
            )
        ):
            issue = "entity_mismatch"
        stance = _normalized_stance(raw, claim)
        claim_type = normalize_claim_type(raw.get("claim_type") or raw.get("evidence_type"), scope=scope, stance=stance)
        raw_criterion_ids = raw.get("criterion_ids")
        if not isinstance(raw_criterion_ids, list):
            raw_criterion_ids = []
        criterion_ids = list(dict.fromkeys(
            str(value).strip()
            for value in raw_criterion_ids
            if str(value).strip()
        ))[:2]
        declared_criterion_ids = list(criterion_ids)
        if semantic_mode == "live" and int(raw.get("_criterion_unknown_filtered_count") or 0) > 0:
            issue = issue or "unknown_criterion_id"
        if semantic_mode == "live" and valid_criterion_ids is not None and any(
            criterion_id not in valid_criterion_ids for criterion_id in declared_criterion_ids
        ):
            issue = issue or "unknown_criterion_id"
        has_subject_relation = "subject_relation" in raw
        subject_relation = _normalize_subject_relation(raw.get("subject_relation"))
        raw_subject_relation = str(raw.get("subject_relation") or "").strip().lower()
        if (
            semantic_mode == "live"
            and has_subject_relation
            and raw_subject_relation
            and raw_subject_relation not in _SUBJECT_RELATIONS
        ):
            issue = issue or "invalid_subject_relation"
        subject = str(raw.get("subject") or (chunk or {}).get("subject") or "").strip()
        criterion_binding_status = "legacy_unchecked"
        if has_subject_relation:
            if subject_relation in {"target", "target_component"}:
                if comparison_question and allowed_comparison_subjects:
                    subject_verified = matches_allowed_subject(subject)
                else:
                    subject_entity = entity_match_metadata(
                        title=subject,
                        snippet=subject,
                        topic=target_subject,
                    ) if subject and target_subject else {"entity_match_score": 1.0}
                    subject_verified = float(subject_entity.get("entity_match_score") or 0.0) >= 0.75
                if subject_verified:
                    criterion_binding_status = f"verified_{subject_relation}"
                else:
                    criterion_ids = []
                    criterion_binding_status = "unverified_subject"
                    if comparison_question and allowed_comparison_subjects:
                        issue = issue or "unrequested_comparison_subject"
            elif subject_relation == "comparison" and comparison_question:
                if not allowed_comparison_subjects or matches_allowed_subject(subject):
                    criterion_binding_status = "verified_comparison"
                else:
                    criterion_ids = []
                    criterion_binding_status = "incompatible_subject"
                    issue = issue or "unrequested_comparison_subject"
            else:
                criterion_ids = []
                criterion_binding_status = (
                    "unverified_subject" if subject_relation == "unknown" else "incompatible_subject"
                )
        comparison_axis = str(raw.get("comparison_axis") or raw.get("axis") or "").strip()
        if "comparison_axis_declared" in raw:
            comparison_axis_declared = bool(raw.get("comparison_axis_declared"))
        else:
            comparison_axis_declared = bool(comparison_axis)
        comparison_axis_binding_status = "not_applicable"
        if expected_comparison_axis:
            if not comparison_axis_declared or not comparison_axis:
                criterion_ids = []
                comparison_axis_binding_status = "unverified_axis"
            elif _comparison_axes_compatible(comparison_axis, expected_comparison_axis):
                if _comparison_axis_evidence_supported(expected_comparison_axis, quote):
                    comparison_axis_binding_status = "verified_axis"
                else:
                    criterion_ids = []
                    comparison_axis_binding_status = "incompatible_axis"
                    if not issue or issue == "off_topic":
                        issue = "comparison_axis_semantic_mismatch"
                        eligible_for_rescue = False
            else:
                criterion_ids = []
                comparison_axis_binding_status = "incompatible_axis"
        if (
            issue == "off_topic"
            and criterion_binding_status in {"verified_target", "verified_target_component"}
            and comparison_axis_binding_status == "verified_axis"
        ):
            issue = ""
            eligible_for_rescue = False
        score = evidence_score(
            claim=claim,
            quote=quote,
            source_url=source_url,
            source_type=source_type,
            grounded=grounded,
            relevant=not bool(issue),
            directness=directness,
            applicability=applicability,
        )
        fallback = str(raw.get("derived_from") or "").startswith("selected_chunk")
        admission_status = "accepted"
        if not issue and fallback and _is_title_only(claim, quote, source_title):
            issue = "fallback_title_only"
        if not issue and fallback and score < 0.72 and semantic_mode != "live":
            issue = "fallback_low_confidence"
        if not issue and fallback and (
            (scope != "domain_specific" or directness < 0.75)
            if semantic_mode != "live"
            else initial_support_relation != "direct_support" or initial_target_fit != "exact_target"
        ):
            admission_status = "inference_only"
        normalized = {
            **raw,
            "question_id": str(raw.get("question_id") or question_id),
            "question_ids": [str(raw.get("question_id") or question_id)],
            "criterion_ids": criterion_ids,
            "declared_criterion_ids": declared_criterion_ids,
            "criterion_binding_status": criterion_binding_status,
            "comparison_axis": comparison_axis,
            "comparison_metric": str(raw.get("comparison_metric") or raw.get("metric") or "").strip(),
            "expected_comparison_axis": str(expected_comparison_axis or "").strip(),
            "comparison_axis_declared": comparison_axis_declared,
            "comparison_axis_binding_status": comparison_axis_binding_status,
            "section_id": str(raw.get("section_id") or section_id),
            "subject": subject,
            "subject_relation": subject_relation,
            "atomic_claim": claim,
            "claim": claim,
            "raw_quote": quote,
            "cleaned_quote": quote,
            "evidence_text": quote,
            "source_url": source_url,
            "source_title": source_title,
            "source_type": source_type,
            "source_type_raw": str(raw.get("source_type") or (chunk or {}).get("source_type") or ""),
            "publisher_relation": publisher_relation(source_type, url=source_url, declared=raw.get("publisher_relation")),
            "entity_match_score": float(entity.get("entity_match_score") or 0.0),
            "canonical_project": str(entity.get("canonical_project") or ""),
            "entity_mismatch_reason": str(entity.get("entity_mismatch_reason") or ""),
            "authority_score": authority_score(source_url, source_type=source_type),
            "evidence_score": score,
            "claim_fingerprint": claim_fingerprint(claim),
            "claim_cluster_id": claim_cluster_id(claim),
            "evidence_type": claim_type,
            "claim_type": claim_type,
            "scope": scope,
            "directness_score": directness,
            "applicability_score": applicability,
            "stance": stance,
            "support_strength": str(raw.get("support_strength") or raw.get("confidence") or _score_label(score)),
            "grounded_exactly": grounded,
            "quote_start": span[0] if span else None,
            "quote_end": span[1] if span else None,
            "quality_flags": [issue] if issue else [],
            "admission_status": "quarantined" if issue else admission_status,
            "admission_tier": "deterministic_fallback" if fallback else "model_extracted",
        }
        if semantic_mode == "shadow":
            normalized.pop("support_relation", None)
            normalized.pop("target_fit", None)
            normalized.pop("semantic_reason", None)
        if semantic_mode != "off":
            quote_context = str((chunk or {}).get("text") or "")
            if span:
                context_start = max(0, span[0] - 240)
                context_end = min(len(quote_context), span[1] + 240)
                quote_context = quote_context[context_start:context_end]
            else:
                quote_context = quote_context[:480]
            normalized["_semantic_lexical_off_topic"] = lexical_off_topic
            normalized["semantic_arbitration"] = {
                "version": "v1",
                "mode": semantic_mode,
                "status": "not_needed",
                "evidence_id": semantic_evidence_id,
                "initial": {
                    "scope": str(raw.get("scope") or scope),
                    "directness_score": _score_value(raw.get("directness_score"), directness),
                    "applicability_score": _score_value(raw.get("applicability_score"), applicability),
                    "support_relation": initial_support_relation,
                    "target_fit": initial_target_fit,
                    "reason": str(raw.get("semantic_reason") or raw.get("reason") or "")[:160],
                },
                "program_signals": _lexical_scope_signal(question, claim, quote),
                "conflict_reasons": [],
                "quote_context": quote_context,
            }
        key = (source_url, _norm(claim)[:180])
        if issue == "off_topic" and eligible_for_rescue:
            normalized["rejection_reason"] = issue
            pending_rescue.append((normalized, claim, key))
            continue
        if issue:
            normalized["rejection_reason"] = issue
            rejected.append(normalized)
        elif key not in seen:
            seen.add(key)
            accepted.append(normalized)
    if pending_rescue:
        claims = [claim for _, claim, _ in pending_rescue]
        try:
            similarities = _semantic_rescue_scores(question=question, texts=claims)
            rescued_mask = [score >= _semantic_relevance_threshold() for score in similarities]
        except Exception:
            similarities = [0.0] * len(pending_rescue)
            rescued_mask = _semantic_rescue_mask(question=question, texts=claims)
        for (normalized, _claim, key), rescued, similarity in zip(pending_rescue, rescued_mask, similarities):
            normalized["semantic_rescue_similarity"] = round(float(similarity), 4)
            normalized["semantic_rescue_trigger"] = "claim_embedding_after_keyword_mismatch"
            if rescued and key not in seen:
                normalized.pop("rejection_reason", None)
                normalized["quality_flags"] = []
                normalized["admission_status"] = "accepted"
                normalized["admission_tier"] = "semantic_rescue"
                normalized["semantic_rescue_entity_gate"] = True
                seen.add(key)
                accepted.append(normalized)
            else:
                rejected.append(normalized)
    if semantic_mode != "off":
        _attach_semantic_diagnostics(accepted, question=question)
    accepted, _diagnostics = merge_evidence_clusters([], accepted)
    return accepted, rejected


def topic_relevant(text: str, question: str) -> bool:
    text_lower = to_simplified(str(text or "").lower())
    question_lower = to_simplified(str(question or "").lower())
    stop = {"如何", "哪些", "什么", "来源", "证据", "实现", "分析", "相关", "关键", "what", "how", "source", "official"}
    terms = [term for term in extract_key_terms(question_lower, top_k=20) if term.lower() not in stop]
    if not terms:
        return True
    return any(term.lower() in text_lower for term in terms)


def strip_question_prefix(claim: str, question: str) -> str:
    question_text = str(question or "").strip()
    value = str(claim or "").strip()
    for separator in (":", "："):
        prefix = f"{question_text}{separator}"
        if question_text and value.startswith(prefix):
            return value[len(prefix):].strip()
    return value


def evidence_issue(*, claim: str, quote: str, source_url: str) -> str:
    if not claim or len(claim) < 12:
        return "claim_too_short"
    if not quote or len(quote) < 18:
        return "quote_too_short"
    if not urlparse(source_url).netloc:
        return "missing_source_domain"
    for name, pattern in NOISE_PATTERNS:
        if pattern.search(f"{claim}\n{quote}"):
            return name
    if claim.endswith(("?", "？")):
        return "question_not_claim"
    if _looks_fragmented(claim) or _looks_fragmented(quote):
        return "fragmented_text"
    return ""


def quote_grounded(quote: str, chunk_text: str) -> bool:
    quote_norm = _norm(quote)
    chunk_norm = _norm(chunk_text)
    if not quote_norm or not chunk_norm:
        return False
    return quote_norm in chunk_norm


def quote_span(quote: str, chunk_text: str) -> tuple[int, int] | None:
    quote_norm = _norm(quote)
    if not quote_norm:
        return None
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(str(chunk_text or "")):
        if char.isspace() or not char.isalnum() and char != "_" and not ("\u4e00" <= char <= "\u9fff"):
            continue
        chars.append(char.lower())
        mapping.append(index)
    chunk_norm = "".join(chars)
    offset = chunk_norm.find(quote_norm)
    if offset < 0 or offset + len(quote_norm) > len(mapping):
        return None
    start = mapping[offset]
    end = mapping[offset + len(quote_norm) - 1] + 1
    leading = re.match(r"[^\w\u4e00-\u9fff]+", str(quote or ""))
    trailing = re.search(r"[^\w\u4e00-\u9fff]+$", str(quote or ""))
    if leading and start >= len(leading.group(0)) and chunk_text[start - len(leading.group(0)):start] == leading.group(0):
        start -= len(leading.group(0))
    if trailing and chunk_text[end:end + len(trailing.group(0))] == trailing.group(0):
        end += len(trailing.group(0))
    return start, end


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n|-_")
    return text


# A trailing cut-URL/markdown-link fragment may exist in historical or replayed evidence created
# by the removed character-budget evidence fallback. When the slice landed mid-URL the quote ends
# in e.g.
# `<https://github.` or `[\\[3\\]](https://zhuanlan.` -- NOISE_PATTERNS' truncated_link entry
# then matches that trailing fragment and rejects an otherwise-good, grounded claim (real false
# positives in case-truncated-url-suffix-a / case-validator-context-parity). The fragment is an
# extraction artifact, not source noise: the quote has substantial on-topic prose before the
# trailing cut. Stripping the suffix before grounding/noise-check lets the prose through; a
# quote that is ITSELF just a bare truncated URL strips to empty and is caught by
# quote_too_short, so this does not weaken truncated_link detection for real noise.
_TRAILING_CUT_URL_FRAGMENT_RE = re.compile(
    r"\s*\\?\[[^\]\\]*(?:\\.[^\]\\]*)*\\?\]\(https?://[^)]*$"  # [text](https://... md-link tail (escaped brackets handled)
    r"|\s*<https?://\S*$"                                     # <https://... angle-bracketed bare URL
    r"|\s*https?://[^\s)]*$",                                 # bare https://...
    re.I,
)


def _strip_trailing_url_fragment(quote: str) -> str:
    """Drop a trailing cut-URL/markdown-link fragment left by excerpt slicing.

    Suffix-preserving: the kept prefix is still a contiguous substring of chunk_text, so
    quote_span grounding still resolves. Also strips a trailing colon/comma/bracket the cut
    left dangling, so the saved cleaned_quote/evidence_text stays clean.
    """
    match = _TRAILING_CUT_URL_FRAGMENT_RE.search(quote)
    if match:
        return quote[: match.start()].rstrip("\uff1a: \uff0c,<> ")
    return quote


def infer_source_type(url: str) -> str:
    return normalize_source_type("", url=url)


def authority_score(url: str, *, source_type: str = "") -> float:
    source_type = normalize_source_type(source_type, url=url)
    return {
        "official_docs": 1.0,
        "paper": 0.95,
        "benchmark": 0.9,
        "github": 0.85,
        "release_notes": 0.85,
        "case_study": 0.8,
        "web_article": 0.65,
        "community": 0.4,
        "video": 0.3,
    }.get(source_type, 0.5)


def evidence_score(
    *,
    claim: str,
    quote: str,
    source_url: str,
    source_type: str,
    grounded: bool,
    relevant: bool,
    directness: float = 1.0,
    applicability: float = 1.0,
) -> float:
    """Return a deterministic, auditable admission score; models cannot override hard failures."""
    score = 0.25 * authority_score(source_url, source_type=source_type)
    score += 0.35 if grounded else 0.0
    score += 0.2 if relevant else 0.0
    score += 0.1 if 18 <= len(claim) <= 280 else 0.03
    score += 0.1 if 18 <= len(quote) <= 800 else 0.03
    score = score * 0.6 + max(0.0, min(1.0, directness)) * 0.2 + max(0.0, min(1.0, applicability)) * 0.2
    return round(min(1.0, score), 3)


def evidence_applicability(
    *,
    question: str,
    claim: str,
    quote: str,
    model_scope: Any = "",
    model_directness: Any = None,
    model_applicability: Any = None,
) -> tuple[str, float, float]:
    domain_terms = _domain_terms(question)
    corpus = f"{claim} {quote}".lower()
    matched = [term for term in domain_terms if term in corpus]
    domain_ratio = len(matched) / max(1, len(domain_terms))
    if domain_terms and domain_ratio >= 0.35:
        deterministic_scope = "domain_specific"
        directness_cap, applicability_cap = 0.95, 0.95
    elif matched:
        deterministic_scope = "cross_domain"
        directness_cap, applicability_cap = 0.7, 0.65
    else:
        deterministic_scope = "general"
        directness_cap, applicability_cap = 0.55, 0.5
    requested_scope = str(model_scope or deterministic_scope)
    allowed_scope = requested_scope if requested_scope in {"domain_specific", "cross_domain", "general"} else deterministic_scope
    # Only override a model-declared domain_specific when the deterministic check found the claim
    # completely off-topic (general) OR when the overlap that pulled it into "domain_specific"
    # territory is a single, possibly-coincidental term. A real GraphRAG/LightRAG case had the
    # model correctly call a claim domain_specific/direct_fact, and the claim matched 3 distinct
    # domain terms ("检索"/"延迟"/"社区") -- just not enough of the (truncated, position-ordered)
    # domain-term list to clear the 0.35 ratio bar. A single coincidental match is a different
    # failure mode though: e.g. a claim explicitly saying "this source doesn't cover travel" only
    # matches the question's "旅游" term once and is genuinely off-topic despite the "hit". Trust
    # the model once there are at least 2 independent matched terms; below that, a thin ratio is
    # too easily coincidental to overrule the deterministic check.
    if deterministic_scope != "domain_specific" and len(matched) < 2 and allowed_scope == "domain_specific":
        allowed_scope = deterministic_scope
    directness = min(_score_value(model_directness, directness_cap), directness_cap)
    applicability = min(_score_value(model_applicability, applicability_cap), applicability_cap)
    return allowed_scope, round(directness, 3), round(applicability, 3)


def normalize_claim_type(value: Any, *, scope: str, stance: str) -> str:
    if stance in {"limitation", "counter"}:
        return "limitation"
    requested = str(value or "").strip().lower()
    aliases = {"fact": "direct_fact", "support": "direct_fact", "principle": "design_principle"}
    requested = aliases.get(requested, requested)
    if scope != "domain_specific":
        return "design_principle"
    return requested if requested in {"direct_fact", "design_principle", "limitation"} else "direct_fact"


def _domain_terms(question: str) -> list[str]:
    # extract_key_terms ranks by TF-IDF importance rather than sentence position, so a
    # distinctive word late in a long question survives the top-K cut instead of being
    # silently truncated away -- see
    # project_evidence_scope_recompute_overrides_correct_model_output_20260729 for the real
    # GraphRAG case where the old positional-bigram-then-truncate version dropped the word for
    # "latency" entirely and misclassified a correct domain_specific/direct_fact claim as
    # cross_domain.
    stop = {
        "agent", "multi", "system", "what", "how", "research", "\u7cfb\u7edf", "\u5982\u4f55", "\u54ea\u4e9b", "\u4ec0\u4e48",
        "\u591a\u667a\u80fd\u4f53", "\u667a\u80fd\u4f53", "\u5b9e\u73b0", "\u5206\u6790", "\u65b9\u6848", "\u8bc4\u4f30", "\u5173\u952e", "\u8fdb\u884c", "\u53ef\u4ee5",
    }
    return [term for term in extract_key_terms(question, top_k=12) if term.lower() not in stop]


def _score_value(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def merge_evidence_clusters(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.82,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collapse same-source near duplicates and cluster independent corroboration."""
    output = [dict(item) for item in existing]
    novel_clusters = 0
    duplicate_count = 0
    replaced_count = 0
    for raw in incoming:
        item = dict(raw)
        claim = str(item.get("atomic_claim") or item.get("claim") or "")
        item.setdefault("claim_fingerprint", claim_fingerprint(claim))
        item.setdefault("claim_cluster_id", claim_cluster_id(claim))
        best_index = -1
        best_similarity = 0.0
        for index, current in enumerate(output):
            similarity = claim_similarity(claim, str(current.get("atomic_claim") or current.get("claim") or ""))
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index >= 0 and best_similarity >= similarity_threshold:
            current = output[best_index]
            item["claim_cluster_id"] = current.get("claim_cluster_id") or item["claim_cluster_id"]
            if str(current.get("source_url") or "") == str(item.get("source_url") or ""):
                duplicate_count += 1
                question_ids = list(dict.fromkeys([
                    *[str(value) for value in current.get("question_ids") or [current.get("question_id")] if value],
                    *[str(value) for value in item.get("question_ids") or [item.get("question_id")] if value],
                ]))
                criterion_ids = list(dict.fromkeys([
                    *[str(value) for value in current.get("criterion_ids") or [] if value],
                    *[str(value) for value in item.get("criterion_ids") or [] if value],
                ]))[:2]
                if float(item.get("evidence_score") or 0) > float(current.get("evidence_score") or 0):
                    item["question_ids"] = question_ids
                    item["criterion_ids"] = criterion_ids
                    output[best_index] = item
                    replaced_count += 1
                else:
                    current["question_ids"] = question_ids
                    current["criterion_ids"] = criterion_ids
                    # Later rounds may extract richer structured audit metadata for the same
                    # grounded claim. Preserve it even when the earlier row keeps the higher
                    # evidence score; deduplication must not erase the axis/metric distinction.
                    for field in (
                        "comparison_axis",
                        "comparison_metric",
                        "expected_comparison_axis",
                        "comparison_axis_binding_status",
                    ):
                        if not current.get(field) and item.get(field):
                            current[field] = item[field]
                continue
        else:
            novel_clusters += 1
        output.append(item)
    return output, {
        "novel_cluster_count": novel_clusters,
        "duplicate_count": duplicate_count,
        "replaced_count": replaced_count,
    }


def claim_fingerprint(text: str) -> str:
    return sha1(_norm(text).encode("utf-8")).hexdigest()[:16]


def claim_cluster_id(text: str) -> str:
    return f"clm_{claim_fingerprint(text)}"


def claim_similarity(left: str, right: str) -> float:
    left_shingles = _shingles(_norm(left))
    right_shingles = _shingles(_norm(right))
    if not left_shingles or not right_shingles:
        return 1.0 if _norm(left) == _norm(right) and _norm(left) else 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)


def _shingles(text: str, width: int = 3) -> set[str]:
    if len(text) <= width:
        return {text} if text else set()
    return {text[index:index + width] for index in range(len(text) - width + 1)}


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


# Bare, single-word limitation markers. These are only safe to use as a one-way fallback (when
# the model declared nothing) -- a word like "\u98ce\u9669" ("risk") shows up constantly in claims that are
# *about* risk management/mitigation design, not claims *admitting* a limitation, so trusting it to
# override an explicit model "support" declaration produces false positives (a real regression:
# "\u5982\u4f55\u5b9e\u73b0\u5e76\u63a7\u5236\u98ce\u9669" ("how to implement and control risk") is a support/design claim
# that happens to contain "\u98ce\u9669").
_BARE_LIMITATION_MARKERS = (
    "\u9650\u5236", "\u98ce\u9669", "\u4e0d\u8db3", "\u5c40\u9650", "\u7f3a\u4e4f", "\u4e22\u5931", "\u4e0b\u964d", "\u526f\u4f5c\u7528", "\u4ee3\u4ef7", "\u65e0\u6cd5",
    "failure", "limitation", "risk", "drawback", "downside", "disadvantage", "trade-off",
    "tradeoff", "lack of", "lacks ", "loss", "degradation", "steep cost", "cannot ",
)

# Compound, directional limitation phrases. High enough confidence (specific magnitude/direction,
# not a bare topic word) to correct a model that explicitly -- and wrongly -- declared "support"
# on a claim that says this. See
# project_evidence_scope_recompute_overrides_correct_model_output_20260729 -- a real GraphRAG case
# with "\u68c0\u7d22\u5ef6\u8fdf\u9ad8" / "\u6d88\u8017\u6781\u9ad8" claims kept falling through the old bare-word list to the
# "support" default because neither "\u5ef6\u8fdf" nor "\u6d88\u8017" alone was in it, and adding them as bare words
# would misclassify improvement claims instead ("\u5ef6\u8fdf\u964d\u4f4e12\u500d" is a support claim for the *faster*
# system).
_DIRECTIONAL_LIMITATION_PHRASES = (
    "\u5ef6\u8fdf\u9ad8", "\u5ef6\u8fdf\u8f83\u9ad8", "\u5ef6\u8fdf\u4e25\u91cd", "\u6d88\u8017\u5927", "\u6d88\u8017\u6781\u9ad8", "\u6d88\u8017\u9ad8", "\u6d88\u8017\u5927\u91cf",
    "\u6210\u672c\u9ad8", "\u6210\u672c\u8f83\u9ad8", "\u5f00\u9500\u5927", "\u5f00\u9500\u8f83\u9ad8", "\u6027\u80fd\u4e0b\u964d", "\u901f\u5ea6\u6162", "\u54cd\u5e94\u6162", "\u8017\u65f6\u957f", "\u8017\u65f6\u8f83\u957f",
    "high latency", "high cost", "costly", "slow response", "resource-intensive",
)


def infer_stance(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in (*_BARE_LIMITATION_MARKERS, *_DIRECTIONAL_LIMITATION_PHRASES)):
        return "limitation"
    if any(token in lowered for token in ("\u53cd\u4f8b", "\u76f8\u53cd", "\u51b2\u7a81", "counter", "contradict")):
        return "counter"
    return "support"


def _high_confidence_limitation_override(text: str) -> bool:
    """Directional-phrase-only check used to correct a wrongly declared model stance.

    Deliberately narrower than infer_stance() -- see _DIRECTIONAL_LIMITATION_PHRASES docstring.
    """
    lowered = text.lower()
    return any(token in lowered for token in _DIRECTIONAL_LIMITATION_PHRASES)


def is_boundary_evidence(item: dict[str, Any]) -> bool:
    stance = str(item.get("stance") or "").strip().lower()
    if stance in {"limitation", "counter"}:
        return True
    if str(item.get("claim_type") or item.get("evidence_type") or "").strip().lower() == "limitation":
        return True
    return infer_stance(str(item.get("atomic_claim") or item.get("claim") or "")) in {"limitation", "counter"}


def _normalized_stance(raw: dict[str, Any], claim: str) -> str:
    declared = str(raw.get("stance") or "").strip().lower()
    # Cross-check even when the model declared a stance, instead of trusting it unconditionally --
    # the old code only consulted infer_stance() as a fallback when the model said nothing, so a
    # model that confidently (and wrongly) declared "support" on an obvious limitation claim
    # sailed through untouched. Deliberately uses the narrower directional-phrase check here, not
    # the full infer_stance() bare-word list: infer_stance's bare markers (e.g. "\u98ce\u9669") are common
    # in claims *about* risk/limitation management design, not claims admitting a limitation, and
    # would false-positive-correct a model's correct "support" declaration on those.
    if declared not in {"limitation", "counter"} and _high_confidence_limitation_override(claim):
        return "limitation"
    if declared in {"support", "limitation", "counter"}:
        return declared
    claim_type = str(raw.get("claim_type") or raw.get("evidence_type") or "").strip().lower()
    if claim_type == "limitation":
        return "limitation"
    return infer_stance(claim)


def _looks_fragmented(text: str) -> bool:
    if re.search(r"(?:\s|^)[1-9]\.$", text):
        return True
    if text.count("[") != text.count("]") or text.count("(") != text.count(")"):
        return True
    # NOTE: this used to also reject anything matching r"(?:https?://\S{0,25}|\w+)\.$" under
    # 80 chars, meant to catch a truncated link trailing off mid-URL. But the `|\w+` alternative
    # matches ANY text ending in an ordinary word followed by a period -- which is how nearly
    # every well-formed short English sentence ends -- so it was rejecting legitimate short
    # English claims/quotes as "fragmented_text" purely for being complete and concise (e.g. "AI
    # agent state machines can include checkpoints for production workflows.", 73 chars, a
    # correct sentence with terminal punctuation). Removed rather than narrowed: a genuinely
    # truncated raw/markdown link is already caught by NOISE_PATTERNS' own "truncated_link" entry
    # (checked earlier in evidence_issue, before _looks_fragmented ever runs), so this branch
    # added false-positive risk without adding real detection coverage.
    return _looks_truncated_mid_word(text)


_TERMINAL_CHARS = set(".!?\u3002\uff01\uff1f\u2026\"'\u2019\u201d\u300d\u300f)]\uff09\u3011\u300b>")

# Short words are where a mid-word cut is most likely to coincide with an unrelated real word, so
# they have to clear a much higher usage bar than long ones before we treat them as complete.
_SHORT_WORD_MAX_LEN = 3
_SHORT_WORD_MIN_ZIPF = 4.5

# Function words are common enough to clear any frequency bar, but prose does not trail off on
# them: "...retry with different age", "...things I research and" are cuts, not endings.
_NON_TERMINAL_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "into", "onto", "as", "than", "that", "which", "who", "whom", "whose",
    "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "not",
    "if", "when", "while", "because", "although", "though", "since", "until", "unless",
    "whether", "about", "over", "under", "between", "among", "through", "during", "without",
    "within", "via", "per", "upon", "toward", "towards", "its", "their", "our", "your", "his",
    "her", "we", "they", "it", "he", "she", "you", "i",
})


def _looks_truncated_mid_word(text: str) -> bool:
    """Flag prose that was hard-cut mid-word with no natural sentence boundary to land on.

    Real extraction failures observed in production (see
    docs/evaluations/deep-research-section-dossier-e2e-20260715.md section 5.4) look like
    "This project shows how to cre" -- a source transcript (e.g. a YouTube caption blob with no
    punctuation to split on) was hard-truncated by the now-removed evidence fallback and landed
    mid-word. Scoped to Latin-script endings only (not CJK) to keep this targeted at the
    observed failure mode instead of a broader, unevidenced completeness heuristic over the
    corpus's overwhelmingly Chinese content.

    Missing terminal punctuation on its own is NOT that signature: plenty of source pages render
    bullet lists with no periods ("Build in: <item> <item> <item>"), so their quotes legitimately
    end on a complete word. Requiring punctuation was rejecting those -- 17 of the 34 real
    triggers of this rule (50%) were that shape, and in one case it ate 7 of a question's 8
    evidence items. So the deciding check is whether the FINAL WORD is itself complete, not
    whether the text stops on punctuation.

    Word completeness is judged by corpus frequency rather than a fixed dictionary because the
    corpus is full of technical vocabulary ("checkpointing", "agentic", "microservices") that
    classic word lists omit -- rejecting those would just relocate the same false positive.
    A truncated remnant that happens to spell a real word ("between" -> "bet") is unfixable at
    the word level and is deliberately let through: admission has several later filters, so a
    missed truncation costs far less than a discarded piece of complete evidence.
    """
    stripped = text.rstrip()
    if not stripped or stripped[-1] in _TERMINAL_CHARS:
        return False
    if not re.search(r"[A-Za-z]$", stripped):
        return False
    if len(stripped) < 24 or " " not in stripped:
        return False
    match = re.search(r"[A-Za-z0-9_']+$", stripped)
    token = match.group(0) if match else ""
    # A compound technical identifier (goto_url / new_tab / Neo4jGraph / _cdp) is an intentional
    # unit, not prose that trailed off. Prose cuts land INSIDE a plain word ("intenti" <-
    # "intentional") and never across an underscore or an interior digit, so an internal "_" or
    # digit anywhere except the very last char marks a complete compound token: job_20260814
    # claims ending in `goto_url`/`Neo4jGraph` were rejected as fragmented_text purely because
    # the tail fragment `url`/`Graph` fails the word-frequency bar for short words.
    if re.search(r"[_\d]", token[:-1]):
        return False
    return not _is_complete_word(token)


def _is_complete_word(word: str) -> bool:
    normalized = word.lower().strip("'")
    if not normalized:
        return False
    # A lone letter is a cut remnant ("...Centralized information m"), never a word ending here.
    if len(normalized) == 1:
        return False
    if normalized in _NON_TERMINAL_FUNCTION_WORDS:
        return False
    frequency = _word_zipf_frequency(normalized)
    if len(normalized) <= _SHORT_WORD_MAX_LEN:
        return frequency >= _SHORT_WORD_MIN_ZIPF
    return frequency > 0


def _word_zipf_frequency(word: str) -> float:
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        # Without the frequency data every word reads as unknown, which would resurrect the
        # false-positive rejections this check exists to prevent. Treat words as complete instead.
        return _SHORT_WORD_MIN_ZIPF
    return zipf_frequency(word, "en")


def _is_title_only(claim: str, quote: str, source_title: str) -> bool:
    title = _norm(source_title)
    if not title:
        return False
    claim_norm = _norm(claim)
    quote_norm = _norm(quote)
    return bool(claim_norm and (claim_norm == title or quote_norm == title))


def _norm(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "").lower())
