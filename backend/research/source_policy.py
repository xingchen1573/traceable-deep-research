"""Source taxonomy, per-question source portfolios, and query hints."""

from __future__ import annotations

import re
from posixpath import normpath
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse


SOURCE_TYPE_ALIASES = {
    "docs": "official_docs",
    "documentation": "official_docs",
    "official": "official_docs",
    "repository": "github",
    "repo": "github",
    "academic": "paper",
    "research_paper": "paper",
    "customer_story": "case_study",
    "success_story": "case_study",
    "postmortem": "case_study",
    "product_page": "web_article",
    "official_product": "web_article",
    "blog": "engineering_blog",
    "technical_blog": "engineering_blog",
    "forum": "community",
    "other": "web_article",
}

SOURCE_TYPES = {
    "official_docs",
    "github",
    "paper",
    "benchmark",
    "case_study",
    "release_notes",
    "engineering_blog",
    "community",
    "web_article",
}

QUESTION_KINDS = {
    "definition",
    "implementation",
    "evaluation",
    "comparison",
    "production_practice",
    "roadmap",
}

SOURCE_QUERY_HINTS = {
    "official_docs": "official documentation architecture API",
    "github": "GitHub README docs issues pulls commits contributors releases architecture implementation",
    "case_study": "production case study engineering lessons",
    "paper": "research paper methodology evaluation",
    "benchmark": "benchmark dataset metrics results",
    "release_notes": "release notes changelog migration",
    "web_article": "technical analysis",
}


_SOURCE_DIVERSITY_CRITERION_PATTERNS = (
    r"多方(?:观点|来源|证据|材料|验证)",
    r"交叉验证",
    r"多个(?:独立)?(?:案例|来源|观点)",
    r"至少\s*[二两三四五六七八九十2-9]\s*个?(?:独立)?(?:案例|来源|观点)",
    r"multiple\s+(?:independent\s+)?(?:cases|sources|perspectives)",
    r"cross[- ](?:check|validation|validate)",
)


def criterion_requires_source_diversity(value: Any) -> bool:
    """Whether a confirmed hard criterion explicitly requires independent corroboration."""
    text = str(value or "").strip().lower()
    return bool(text) and any(re.search(pattern, text, re.I) for pattern in _SOURCE_DIVERSITY_CRITERION_PATTERNS)


def criterion_source_diversity_target(value: Any) -> int:
    """Return the explicit corroboration count, defaulting broad wording to two sources."""
    text = str(value or "").strip().lower()
    if not criterion_requires_source_diversity(text):
        return 0
    numeric = re.search(r"至少\s*([2-9])\s*个?", text)
    if numeric:
        return int(numeric.group(1))
    chinese = re.search(r"至少\s*([二两三四五六七八九十])\s*个?", text)
    if chinese:
        values = {"二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        return values[chinese.group(1)]
    return 2


def canonical_source_identity(url: Any) -> str:
    """Return a stable page/document identity without collapsing a whole host."""
    raw = str(url or "").strip()
    if not raw or raw.startswith("context7://"):
        return raw
    parsed = urlparse(raw)
    host = parsed.netloc.lower().split("@")[-1]
    if not host:
        return raw.lower().split("#", 1)[0].rstrip("/")
    path = normpath(parsed.path or "/")
    if path == ".":
        path = "/"
    path = path.rstrip("/") or "/"
    # Query strings are generally tracking noise; retain only stable documentation selectors.
    stable_query = ""
    if parsed.query:
        pairs = [(key.lower(), value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                 if key.lower() in {"page", "p", "version", "tag"}]
        if pairs:
            stable_query = "?" + urlencode(sorted(pairs))
    return f"{host}{path}{stable_query}"


def canonical_project(url: Any) -> str:
    """Return a project-level identity suitable for same-project source-floor checks."""
    raw = str(url or "").strip()
    if raw.startswith("context7://"):
        return raw.split("#", 1)[0]
    parsed = urlparse(raw)
    host = parsed.netloc.lower().split("@")[-1]
    parts = [part for part in parsed.path.split("/") if part]
    if host == "github.com" and len(parts) >= 2:
        return f"github.com/{parts[0].lower()}/{parts[1].lower()}"
    if host.endswith("github.io") and parts:
        return f"{host}/{parts[0].lower()}"
    if host:
        return host
    return canonical_source_identity(raw)


def strict_entity_topic(topic: Any) -> bool:
    text = str(topic or "").lower()
    return "deepseek" in text and "harness" in text


def entity_match_metadata(*, url: Any = "", title: Any = "", snippet: Any = "", topic: Any = "") -> dict[str, Any]:
    """Score whether a source belongs to a concrete project, when the topic is specific enough."""
    raw_url = str(url or "").strip()
    text = f"{raw_url} {title} {snippet}".lower()
    project = canonical_project(raw_url)
    question_kind = normalize_question_kind(
        "",
        question=str(topic or ""),
        purpose="",
        success_criteria=[],
    )
    if question_kind == "comparison":
        # A comparison question has more than one legitimate entity. Requiring every source to
        # mention the anchor project makes the other subject's primary documentation impossible
        # to fetch or admit and biases the report toward secondary comparison articles. Relevance
        # and grounding still apply downstream, so this only removes the single-entity assumption.
        return {"entity_match_score": 1.0, "canonical_project": project, "entity_mismatch_reason": ""}
    if not strict_entity_topic(topic):
        # Generic topics use a conservative three-way verdict. A missing anchor is uncertain,
        # not a rejection: third-party articles often omit the full product name in every
        # paragraph. Only confirmed matches may receive strong-text protection downstream;
        # uncertain candidates remain available to semantic recall and evidence admission.
        from .text_utils import extract_key_terms, normalize_for_substring, to_simplified

        topic_text = str(topic or "").strip()
        normalized_topic = normalize_for_substring(topic_text)
        normalized_candidate = normalize_for_substring(text)
        if 3 <= len(normalized_topic) <= 80 and normalized_topic in normalized_candidate:
            return {"entity_match_score": 0.75, "canonical_project": project, "entity_mismatch_reason": ""}
        generic = {
            "what", "which", "how", "why", "definition", "architecture", "implementation",
            "example", "examples", "use", "using", "best", "practice", "practices", "system",
            "什么", "哪些", "如何", "为什么", "定义", "架构", "实现", "使用", "最佳", "实践",
            "功能", "场景", "方法", "系统", "调研", "研究", "包括",
        }
        anchor_terms = [
            term for term in extract_key_terms(topic_text, top_k=8)
            if term.lower() not in generic
        ]
        normalized_text = to_simplified(text)
        if anchor_terms and all(to_simplified(term.lower()) in normalized_text for term in anchor_terms):
            return {"entity_match_score": 0.75, "canonical_project": project, "entity_mismatch_reason": ""}
        if (
            len(anchor_terms) == 1
            and len(anchor_terms[0]) >= 4
            and to_simplified(anchor_terms[0].lower()) in to_simplified(f"{raw_url} {title}".lower())
        ):
            return {"entity_match_score": 0.75, "canonical_project": project, "entity_mismatch_reason": ""}
        return {
            "entity_match_score": 0.5,
            "canonical_project": project,
            "entity_mismatch_reason": "entity_uncertain",
        }
    parsed = urlparse(raw_url)
    path = parsed.path.lower()
    unrelated = (
        ("browser-use" in path and "browser-harness" in path)
        or "deepseek-v3" in path
        or ("deepseek-cli" in path and "deepseek-harness" not in text)
    )
    if unrelated:
        return {"entity_match_score": 0.0, "canonical_project": project, "entity_mismatch_reason": "unrelated_project"}
    if "github.com/deepseek-ai/deepseek-harness" in f"{parsed.netloc.lower()}{path}" or "/deepseek-ai/deepseek-harness" in path:
        return {"entity_match_score": 1.0, "canonical_project": "github.com/deepseek-ai/deepseek-harness", "entity_mismatch_reason": ""}
    if "deepseek" in parsed.netloc.lower() and "harness" in path:
        return {"entity_match_score": 1.0, "canonical_project": project, "entity_mismatch_reason": ""}
    if "deepseek harness" in text or "deepseek-harness" in text:
        return {"entity_match_score": 0.75, "canonical_project": project, "entity_mismatch_reason": ""}
    return {"entity_match_score": 0.0, "canonical_project": project, "entity_mismatch_reason": "missing_project_anchor"}


def normalize_source_type(
    value: Any,
    *,
    url: str = "",
    title: str = "",
    text: str = "",
) -> str:
    declared = SOURCE_TYPE_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    context = f"{title} {path} {text[:600]}".lower()
    if declared not in {
        "benchmark", "case_study", "engineering_blog", "github", "paper", "release_notes",
    } and _contains_any(context, (
        "developer community", "community article", "user-contributed",
        "开发者社区", "社区文章", "用户自发贡献",
    )):
        return "community"
    if declared in SOURCE_TYPES and declared not in {"web_article"}:
        return declared
    # github.io/gitlab.io are where projects publish their own docs, so match the host suffix
    # rather than a substring: a blog post merely mentioning "github.io" in its path is not one.
    if any(token in host for token in ("github.com", "gitlab.com", "codeberg.org")) or host.endswith(
        (".github.io", ".gitlab.io")
    ):
        return "github"
    if any(token in host for token in ("arxiv.org", "doi.org", "acm.org", "ieee.org")):
        return "paper"
    if _contains_any(context, ("benchmark", "leaderboard", "基准测试", "评测集", "排行榜")):
        return "benchmark"
    if _contains_any(context, ("release notes", "changelog", "what's new", "版本说明", "更新日志")):
        return "release_notes"
    case_markers = (
        "case study", "customer story", "success story", "postmortem",
        "客户案例", "生产案例", "事故复盘", "落地案例",
    )
    if _contains_any(context, case_markers):
        return "case_study"
    if _contains_any(context, ("best practice", "最佳实践")) and _contains_any(
        context,
        ("production", "customer", "company", "built", "deployed", "构建", "部署", "生产", "客户", "企业"),
    ):
        return "case_study"
    if "/blog" in path and _contains_any(context, ("engineering", "technical", "architecture", "api", "工程", "架构", "开发")):
        return "engineering_blog"
    if (
        any(token in host for token in ("docs.", "developer.", "developers.", "readthedocs", "learn.microsoft.com"))
        or path.startswith("/docs")
        or "/documentation" in path
    ):
        return "official_docs"
    if any(token in host for token in ("zhihu.com", "medium.com", "csdn.net", "reddit.com", "linkedin.com")):
        return "community"
    if any(token in host for token in ("youtube.com", "youtu.be")):
        return "video"
    return "web_article"


def publisher_relation(source_type: str, *, url: str, declared: Any = "") -> str:
    value = str(declared or "").strip().lower()
    if value in {"official", "third_party", "community", "unknown"}:
        return value
    normalized = normalize_source_type(source_type, url=url)
    host = urlparse(url).netloc.lower()
    if normalized in {"official_docs", "release_notes"} or any(token in host for token in ("developers.google.com", "learn.microsoft.com")):
        return "official"
    if normalized == "community":
        return "community"
    if normalized in {"paper", "benchmark", "case_study", "engineering_blog", "web_article"}:
        return "third_party"
    return "unknown"


def normalize_question_kind(value: Any, *, question: str, purpose: str, success_criteria: list[str]) -> str:
    declared = str(value or "").strip().lower()
    if declared in QUESTION_KINDS:
        return declared
    # Purpose and success criteria describe how an answer will be validated. Their
    # common verbs (for example "compare" or "evaluate") must not redefine the
    # intent of the question itself.
    context = str(question or "").lower()
    if _contains_any(context, ("路线", "演进", "版本", "里程碑", "roadmap", "release", "migration")):
        return "roadmap"
    if _contains_any(context, ("生产案例", "生产实践", "事故复盘", "落地案例", "case study", "postmortem", "production practice", "production agent")):
        return "production_practice"
    if _contains_any(context, ("对比", "选型", "比较", "差异", "compare", "comparison", "differ", "versus", " vs ")):
        return "comparison"
    if _contains_any(context, ("评估", "指标", "可靠性", "失败", "风险", "陷阱", "权衡", "平衡", "benchmark", "evaluation", "reliability", "pitfall", "tradeoff", "trade-off")):
        return "evaluation"
    if _contains_any(context, ("实现", "架构", "接口", "机制", "api", "协议", "代码", "implementation", "architecture", "mechanism")):
        return "implementation"
    return "definition"


def question_requires_boundary(question: str, question_kind: str = "") -> bool:
    """Require counter/limitation evidence only when the question asks for it."""
    context = str(question or "").lower()
    comparison_markers = (
        "对比", "比较", "选型", "差异", "优劣", "相比", "compare",
        "comparison", "differ", "versus", " vs ", "advantage", "disadvantage",
    )
    boundary_markers = (
        "风险", "限制", "局限", "失败", "缺点", "副作用", "陷阱", "权衡", "平衡",
        "risk", "limitation", "failure", "drawback", "pitfall", "tradeoff", "trade-off",
    )
    return _contains_any(context, comparison_markers) or _contains_any(context, boundary_markers)


def boundary_query_terms(question: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", str(question or "")):
        return "局限 风险 失败案例 权衡"
    return "limitations risks failure cases tradeoffs"


def coverage_requirements(question_kind: str, *, question: str = "") -> dict[str, Any]:
    requirements = {
        "definition": {"min_claim_clusters": 2, "min_domains": 2, "requires_boundary": False, "required_source_groups": []},
        "implementation": {"min_claim_clusters": 3, "min_domains": 2, "requires_boundary": False, "required_source_groups": [["official_docs", "github"]]},
        "evaluation": {"min_claim_clusters": 3, "min_domains": 3, "requires_boundary": False, "required_source_groups": []},
        "comparison": {"min_claim_clusters": 4, "min_domains": 3, "requires_boundary": False, "required_source_groups": []},
        "production_practice": {"min_claim_clusters": 3, "min_domains": 3, "requires_boundary": False, "required_source_groups": [["case_study"]]},
        "roadmap": {"min_claim_clusters": 3, "min_domains": 3, "requires_boundary": False, "required_source_groups": [["release_notes", "case_study"]]},
    }.get(question_kind, {"min_claim_clusters": 2, "min_domains": 2, "requires_boundary": False, "required_source_groups": []})
    return {**requirements, "requires_boundary": question_requires_boundary(question, question_kind)}


def build_source_requirements(
    *,
    question: str,
    purpose: str,
    success_criteria: list[str],
    preferred_source_types: list[str],
    question_kind: str = "",
) -> list[dict[str, Any]]:
    """Build a topic-agnostic source portfolio from the research intent."""
    kind = normalize_question_kind(
        question_kind,
        question=question,
        purpose=purpose,
        success_criteria=success_criteria,
    )
    defaults = {
        "definition": ["official_docs", "paper"],
        "implementation": ["official_docs", "github"],
        "evaluation": ["benchmark", "case_study"],
        "comparison": ["official_docs", "benchmark"],
        "production_practice": ["case_study", "official_docs"],
        "roadmap": ["release_notes", "case_study"],
    }.get(kind, ["official_docs", "engineering_blog"])
    planner_types = [normalize_source_type(item) for item in preferred_source_types]
    valid_planner_types = [item for item in planner_types if item in SOURCE_TYPES]
    # The planner sees the question's full text (including the concrete "success_criteria" it
    # wrote itself) and can tell "常见陷阱" (wants community/github war-story evidence) apart from
    # "评估基准性能" (wants benchmark) even when normalize_question_kind's keyword table -- which
    # only sees isolated trigger words like "陷阱" -- classifies both as the same `evaluation`
    # kind. In `case-mixed-revision-source-policy`, a framework-practice question about common
    # pitfalls made the planner correctly emit ["community", "github", "official_docs"], but the old
    # `[*defaults, *valid_planner_types][:2]` ordering let the keyword-derived
    # ["benchmark", "case_study"] defaults win the truncation and silently discard the planner's
    # judgment -- so every real war-story source the question actually needed scored against a
    # source-type preference it could never satisfy, and the question fetched zero material.
    # Trust the planner's explicit judgment first; keyword defaults now only fill in when the
    # planner left this unset (matches test_source_portfolio_is_derived_from_question_intent_
    # without_topic_rules, which passes preferred_source_types=[]).
    types = list(dict.fromkeys([*valid_planner_types, *defaults]))[:2]
    return [
        {
            "source_type": source_type,
            "min_count": 1,
            "level": "preferred",
            "fallback_types": _fallback_types(source_type),
        }
        for source_type in types
    ]


def source_query(
    topic_anchor: str,
    source_type: str,
    *,
    excluded_domains: list[str] | None = None,
    content: str = "",
) -> str:
    source_hint = SOURCE_QUERY_HINTS.get(normalize_source_type(source_type), "")
    return compile_research_query(
        topic_anchor=topic_anchor,
        content=" ".join(item for item in (content, source_hint) if item),
        source_type=source_type,
        excluded_domains=excluded_domains,
    )


def targeted_query(
    *,
    topic_anchor: str,
    missing_fact: str = "",
    source_type: str = "",
    excluded_domains: list[str] | None = None,
) -> str:
    return compile_research_query(
        topic_anchor=topic_anchor,
        content=missing_fact,
        source_type=source_type,
        excluded_domains=excluded_domains,
    )


def compile_research_query(
    *,
    topic_anchor: str = "",
    content: str = "",
    source_type: str = "",
    excluded_domains: list[str] | None = None,
    limit: int = 140,
) -> str:
    """Compile a search-engine query from a stable topic anchor plus one round-specific content
    string, without leaking critic prose.

    Two independently-bounded parts, never more: `topic_anchor` (~40 chars, a stable per-job
    subject phrase extracted once and shared across every question/round -- see
    _extract_topic_anchor) and `content` (~100 chars, the one thing that's supposed to differ
    between rounds -- a criterion's query_hint on round 1, a coverage gap's missing_fact on
    gap-fill rounds, or a static source-type/boundary template for the structural gap types).
    Earlier designs joined up to five independently-variable-length signals (question, criterion
    prose, keywords, a source-type hint, exclusions) into one shared character budget and hard-
    truncated left-to-right -- whichever signal landed later in the join order silently lost
    content when the total overflowed. Three real incidents (case-source-signal-starvation-a,
    case-source-signal-starvation-b, case-source-signal-starvation-c) were the same starvation bug
    resurfacing on different fields after each attempted fix. Keeping the two parts here
    independently short and non-competing removes the truncation-order failure mode by
    construction rather than by tuning another length cap.
    """
    if not content and source_type:
        content = SOURCE_QUERY_HINTS.get(normalize_source_type(source_type), "")
    anchor = _clean_query_need(topic_anchor, limit=40)
    clean_content = _clean_query_need(content, limit=100)
    if normalize_source_type(source_type) == "github" and _contains_any(
        clean_content.lower(),
        ("issue", "pull request", "pr", "commit", "contributor", "release", "活跃", "贡献者", "提交", "版本"),
    ):
        clean_content = _clean_query_need(
            f"{clean_content} GitHub issues pulls commits contributors releases",
            limit=100,
        )
    exclusions = " ".join(f"-site:{domain}" for domain in (excluded_domains or [])[:2] if domain)
    parts = [anchor, clean_content, exclusions]
    return _compact_query(" ".join(part for part in parts if part), limit=max(80, min(limit, 160)))


def _fallback_types(source_type: str) -> list[str]:
    return {
        "official_docs": ["github", "web_article"],
        "github": ["official_docs", "web_article"],
        "case_study": ["web_article", "paper"],
        "benchmark": ["paper", "official_docs"],
        "paper": ["benchmark", "official_docs"],
        "release_notes": ["github", "official_docs"],
    }.get(source_type, ["web_article"])


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _compact_query(value: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def _clean_query_need(value: Any, *, limit: int = 90) -> str:
    text = re.sub(r"(.{8,40})\1+", r"\1", str(value or ""))
    text = re.sub(r"(?:当前|仍然)?(?:缺少|不足|未找到|没有找到|尚无|无法确认)[：:，,\s]*", "", text)
    text = re.sub(r"(?:至少|不少于)\s*\d+\s*(?:个|条|项|篇|份)?", "", text)
    text = re.sub(r"(?:需要|请)(?:补充|提供|查找|检索)", "", text)
    text = re.sub(r"可追溯(?:来源|链接)?", "", text)
    return re.sub(r"\s+", " ", text).strip(" ，,。；;：:")[:limit]
