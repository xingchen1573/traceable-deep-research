"""Web retrieval tools and lightweight provider-capability helpers."""

from __future__ import annotations

import html
import io
import json
import os
import re
import threading
import urllib.parse
import urllib.request
from typing import Any

from langchain_core.tools import tool


DEFAULT_SERPER_URL = "https://google.serper.dev/search"
DEFAULT_TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
DEFAULT_FIRECRAWL_SCRAPE_ENDPOINT = "https://api.firecrawl.dev/v1/scrape"

# Defensive circuit-breaker against pathological pages, not a normal-case truncation --
# real long-form articles run far below this. Downstream (single_url paragraph segmentation,
# question_loop chunk selection) already handles arbitrarily long content without dropping it.
WEB_FETCH_MAX_CONTENT_CHARS = int(os.getenv("WEB_FETCH_MAX_CONTENT_CHARS", "150000"))

# Below this length, the CSS-density heuristic in _content_quality is trusted as-is (audited
# false positives were all much longer than this). At or above it, a CSS-density match is
# downgraded to "medium" instead of "low" -- see _content_quality for the audit that motivated
# this gate.
CSS_DENSITY_LENGTH_GATE_CHARS = 5000

# See _clean_tavily_markdown for what these match and why they're this narrow.
_TAVILY_HEAD_NAV_MARKER_RE = re.compile(r"^.*Help Center.*$", re.M)
_TAVILY_TAIL_AI_SUMMARY_RE = re.compile(r"Summarize the content at.*", re.S)
_TAVILY_TAIL_AUTHOR_BIO_RE = re.compile(r"WRITTEN BY.*", re.S)



def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _attempt(
    *,
    capability: str,
    provider: str,
    status: str,
    reason: str = "",
    fallback_used: bool = False,
    degraded: bool = False,
    selected: bool = False,
    content_length: int = 0,
    content_quality: str = "unknown",
    url: str = "",
) -> dict[str, Any]:
    return {
        "capability": capability,
        "provider": provider,
        "status": status,
        "reason": reason,
        "fallback_used": fallback_used,
        "degraded": degraded,
        "selected": selected,
        "content_length": content_length,
        "content_quality": content_quality,
        "url": url,
    }


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 12) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _get_json(url: str, params: dict[str, Any], headers: dict[str, str], timeout: int = 12) -> Any:
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
    separator = "&" if "?" in url else "?"
    request_url = url + (separator + query if query else "")
    request = urllib.request.Request(
        request_url,
        headers={"Accept": "application/json", **headers},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _absolute_endpoint(base_url: str, endpoint: str) -> str:
    text = str(endpoint or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return base_url.rstrip("/") + "/" + text.lstrip("/")


def _normalize_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return text
    path = parsed.path or "/"
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered_query = [
        (key, value)
        for key, value in query
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ]
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            urllib.parse.urlencode(filtered_query),
            "",
        )
    )


def _source_type_from_url(url: str, title: str = "", snippet: str = "") -> str:
    host = urllib.parse.urlsplit(url).netloc.lower()
    path = urllib.parse.urlsplit(url).path.lower()
    context = f"{title} {path} {snippet[:500]}".lower()
    if any(token in context for token in ("benchmark", "leaderboard", "基准测试", "评测集")):
        return "benchmark"
    if any(token in context for token in ("release notes", "changelog", "what's new", "版本说明", "更新日志")):
        return "release_notes"
    if any(token in context for token in ("case study", "customer story", "success story", "postmortem", "客户案例", "生产案例", "事故复盘", "落地案例")):
        return "case_study"
    if any(token in context for token in (
        "developer community", "community article", "user-contributed",
        "开发者社区", "社区文章", "用户自发贡献",
    )):
        return "community"
    if "docs." in host or "readthedocs" in host or "developer." in host or "developers." in host:
        return "official_docs"
    if "github.com" in host or "gitlab.com" in host:
        return "github"
    if "arxiv.org" in host or "doi.org" in host:
        return "paper"
    if any(token in host for token in ["news", "reuters", "apnews", "bloomberg"]):
        return "news"
    if any(token in host for token in ["forum", "reddit", "stackoverflow"]):
        return "community"
    return "web_article"


def _extract_with_trafilatura(raw_html: str) -> str:
    """Run trafilatura's statistical main-content extraction over raw HTML.

    Returns "" (never raises) when trafilatura isn't installed, the input isn't real HTML
    (e.g. Tavily's markdown output -- DOM structure trafilatura needs has already been
    stripped, so it's never called with that), or extraction finds nothing usable. Callers
    treat an empty return as "fall back to the existing extraction path", per the decision to
    keep trafilatura strictly additive rather than a hard dependency.
    """
    if not raw_html or "<" not in raw_html:
        return ""
    try:
        import trafilatura  # type: ignore
    except Exception:
        return ""
    try:
        extracted = trafilatura.extract(
            raw_html,
            output_format="markdown",
            favor_recall=True,
            include_tables=True,
            include_formatting=True,
        )
    except Exception:
        return ""
    return (extracted or "").strip()


def _strip_html(raw: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else ""

    trafilatura_text = _extract_with_trafilatura(raw)
    if trafilatura_text:
        return title, trafilatura_text

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        # Strip not just script/style, but also boilerplate containers (nav/header/footer/
        # aside/form) before get_text -- otherwise navigation and social-link labels get
        # concatenated into the article text, then chunked, then extracted as evidence verbatim.
        # In `case-timeout-and-page-chrome`, a primary-source chunk began with page chrome rather
        # than article prose because none of these containers was decomposed. svg/noscript stay in
        # the list for safety; main/article/section are NOT removed because real article
        # body often lives there. (Fallback path only -- trafilatura above handles this case
        # for any HTML it can parse; this stays as the safety net when it can't.)
        for item in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"]):
            item.decompose()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = soup.get_text("\n")
    except Exception:
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def _content_quality(content: str) -> tuple[str, bool, str]:
    text = (content or "").strip()
    lowered = text.lower()
    if len(text) < 300:
        return "low", True, "content too short"
    error_markers = [
        "403 forbidden",
        "404 not found",
        "access denied",
        "captcha",
        "enable javascript",
        "please enable cookies",
        "login required",
    ]
    if any(marker in lowered for marker in error_markers):
        return "low", True, "content looks like an error, login, or anti-bot page"
    css_markers = [
        "font-family",
        "background-color",
        "border-radius",
        "box-sizing",
        "text-decoration",
        "var(--",
        "@media",
        "display:",
        "padding:",
        "margin:",
    ]
    css_marker_hits = sum(1 for marker in css_markers if marker in lowered)
    brace_count = text.count("{") + text.count("}")
    if css_marker_hits >= 3 and brace_count > 20:
        # Audited 124 real jobs' coverage.json checkpoints: every long-content hit on this
        # heuristic (6897-106801 chars) turned out to be a real article -- MathML in rendered
        # arXiv HTML, embedded widgets in a tech blog -- that happens to brush past the marker
        # count, not an actual stylesheet dump. Short hits (27-250 chars) were true positives
        # (login walls, empty pages). Below the length gate we still trust the heuristic as-is;
        # above it we can't tell true from false positive from this signal alone, so we
        # downgrade to medium (no `degraded` ranking penalty, no `high` either) instead of
        # discarding what's very likely the most substantive content in the pool.
        if len(text) < CSS_DENSITY_LENGTH_GATE_CHARS:
            return "low", True, "content looks like stylesheet or boilerplate instead of article text"
        return (
            "medium",
            False,
            f"length_gated_css_check_uncertain: content is {len(text)} chars (>= {CSS_DENSITY_LENGTH_GATE_CHARS}) "
            "and also matched the stylesheet/boilerplate heuristic; too long to trust the heuristic alone",
        )
    if len(text) < 1200:
        return "medium", False, "content is readable but short"
    return "high", False, "content looks usable"


def _document(
    *,
    url: str,
    title: str,
    content: str,
    provider: str,
    source_type: str = "other",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality, degraded, reason = _content_quality(content)
    inferred_type = _source_type_from_url(url, title, content)
    if source_type in {"", "other", "web", "article", "web_article"} or inferred_type in {"benchmark", "release_notes", "case_study"}:
        source_type = inferred_type
    return {
        "url": url,
        "title": title or url,
        "provider": provider,
        "source_type": source_type,
        "content": content[:WEB_FETCH_MAX_CONTENT_CHARS],
        "content_length": len(content),
        "content_quality": quality,
        "degraded": degraded,
        "reason": reason,
        "metadata": metadata or {},
    }


class _RotatingKeyPool:
    """True round-robin key pool: each call to `ordered_indices` advances a shared cursor so
    load actually spreads across keys, unlike `_get_serper_keys()`'s always-try-index-0-first
    pattern below (appropriate there because Serper's failover is about quota exhaustion, not
    throughput). Used for Tavily/Firecrawl once `global_cap` (research_service.py) stopped
    artificially throttling per-round fetch volume -- with more fetches per round, a single key's
    own rate limit could become the new bottleneck, so requests need to actually distribute
    across multiple keys rather than all landing on the same one.

    `ordered_indices` still returns every index (dead ones last), so a caller that tries each in
    order and only gives up after exhausting the whole list gets the same "keep trying until
    every key has failed" fallback guarantee Serper already has -- rotation only changes which
    key is tried *first*, not whether a temporarily-dead key remains available as a last resort.
    """

    def __init__(self, primary_env: str, extra_env: str) -> None:
        self._primary_env = primary_env
        self._extra_env = extra_env
        self._cursor = 0
        self._lock = threading.Lock()
        self._dead_indices: set[int] = set()

    def keys(self) -> list[str]:
        primary = os.getenv(self._primary_env, "").strip()
        extras = [item.strip() for item in os.getenv(self._extra_env, "").split(",")]
        return list(dict.fromkeys(key for key in [primary, *extras] if key))

    def ordered_indices(self, key_count: int) -> list[int]:
        with self._lock:
            start = self._cursor % key_count
            self._cursor += 1
            dead = set(self._dead_indices)
        rotated = [(start + offset) % key_count for offset in range(key_count)]
        return [index for index in rotated if index not in dead] + [index for index in rotated if index in dead]

    def mark_dead(self, index: int) -> None:
        with self._lock:
            self._dead_indices.add(index)


_TAVILY_POOL = _RotatingKeyPool("TAVILY_API_KEY", "TAVILY_EXTRA_API_KEYS")
_FIRECRAWL_POOL = _RotatingKeyPool("FIRECRAWL_API_KEY", "FIRECRAWL_EXTRA_API_KEYS")
_CONTEXT7_POOL = _RotatingKeyPool("CONTEXT7_API_KEY", "CONTEXT7_EXTRA_API_KEYS")


def _get_serper_keys() -> list[str]:
    """Get the ordered pool of Serper API keys from environment.

    ``SERPER_API_KEY`` stays the primary/legacy single-key variable;
    ``SERPER_EXTRA_API_KEYS`` (comma-separated) adds an optional pool of
    secondary keys so a quota-exhausted key ("Not enough credits") doesn't
    take down every broad_discovery call for the rest of a run -- mirrors
    the RESEARCH_MODEL_EXTRA_API_KEYS rotation pattern used for the research
    model pool (backend/research/model_runtime.py).
    """
    primary = os.getenv("SERPER_API_KEY", "").strip()
    extras = [item.strip() for item in os.getenv("SERPER_EXTRA_API_KEYS", "").split(",")]
    return list(dict.fromkeys(key for key in [primary, *extras] if key))


def _get_serper_config() -> tuple[str | None, str]:
    """Get the primary Serper API key and URL from environment (back-compat helper)."""
    keys = _get_serper_keys()
    base_url = os.getenv("SERPER_URL", DEFAULT_SERPER_URL)
    return (keys[0] if keys else None), base_url


_SERPER_QUOTA_LOCK = threading.Lock()
# Indices into _get_serper_keys() that have been observed to be out of quota.
# Process-lifetime only -- a fresh process re-tries every configured key, which is
# intentional: quota exhaustion is an external account state, not something this
# process can otherwise detect without probing.
_SERPER_DEAD_KEY_INDICES: set[int] = set()


def _is_serper_quota_error(reason: str) -> bool:
    lowered = reason.lower()
    return "not enough credits" in lowered or "http error 400" in lowered


def broad_discovery(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search broad web sources and return normalized Source records plus trace."""
    keys = _get_serper_keys()
    base_url = os.getenv("SERPER_URL", DEFAULT_SERPER_URL)
    if not keys:
        return {
            "sources": [],
            "provider_attempts": [
                _attempt(
                    capability="broad_discovery",
                    provider="serper",
                    status="skipped",
                    reason="SERPER_API_KEY not configured",
                )
            ],
        }

    num_results = min(max(1, num_results), 10)
    attempts: list[dict[str, Any]] = []
    with _SERPER_QUOTA_LOCK:
        dead = set(_SERPER_DEAD_KEY_INDICES)
    ordered_indices = [index for index in range(len(keys)) if index not in dead] + sorted(dead)
    for key_index in ordered_indices:
        api_key = keys[key_index]
        try:
            data = _post_json(
                base_url,
                {"q": query, "num": num_results},
                {"X-API-KEY": api_key},
                timeout=10,
            )
        except Exception as exc:
            reason = str(exc)
            if _is_serper_quota_error(reason):
                with _SERPER_QUOTA_LOCK:
                    _SERPER_DEAD_KEY_INDICES.add(key_index)
            attempts.append(
                _attempt(
                    capability="broad_discovery",
                    provider="serper",
                    status="error",
                    reason=f"key_index={key_index}: {reason}",
                    degraded=True,
                )
            )
            continue
        return _serper_success_result(query, data, num_results, attempts)

    return {"sources": [], "provider_attempts": attempts}


def _serper_success_result(
    query: str, data: dict[str, Any], num_results: int, prior_attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for item in data.get("organic", [])[:num_results]:
        url = str(item.get("link", "")).strip()
        if not url:
            continue
        sources.append(
            {
                "title": item.get("title", "") or url,
                "url": url,
                "normalized_url": _normalize_url(url),
                "snippet": item.get("snippet", ""),
                "source_type": _source_type_from_url(url, str(item.get("title") or ""), str(item.get("snippet") or "")),
                "provider": "serper",
                "query": query,
            }
        )

    answer_box = data.get("answerBox")
    knowledge_graph = data.get("knowledgeGraph")
    return {
        "query": query,
        "sources": sources,
        "answer_box": answer_box,
        "knowledge_graph": knowledge_graph,
        "provider_attempts": [
            *prior_attempts,
            _attempt(
                capability="broad_discovery",
                provider="serper",
                status="success",
                reason=f"returned {len(sources)} sources",
                selected=True,
                content_length=len(json.dumps(data, ensure_ascii=False)),
                content_quality="medium" if sources else "low",
                degraded=not bool(sources),
            ),
        ],
    }


def docs_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search technical docs through Context7-compatible configuration if present.

    Rotates across ``CONTEXT7_API_KEY`` + ``CONTEXT7_EXTRA_API_KEYS`` (see
    ``_RotatingKeyPool``) the same way Tavily/Firecrawl do -- a single-key deploy previously
    had no failover, so one rate-limited key (HTTP 429) took down `docs_search` for every
    question in a round. See docs/evaluations/deep-research-revision-loop-reuse-mismatch-
    20260811-handoff.md follow-up: two real jobs on 2026-08-11 hit 429 on every context7 call
    across 6 questions, starving `official_docs`/`github` source coverage for the whole run.
    """
    base_url = (os.getenv("CONTEXT7_BASE_URL") or "").rstrip("/")
    endpoint = os.getenv("CONTEXT7_SEARCH_ENDPOINT") or ""
    keys = _CONTEXT7_POOL.keys()
    if not keys or not base_url or not endpoint:
        return {
            "sources": [],
            "documents": [],
            "provider_attempts": [
                _attempt(
                    capability="docs_search",
                    provider="context7",
                    status="skipped",
                    reason="Context7 env not fully configured",
                )
            ],
        }

    url = _absolute_endpoint(base_url, endpoint)
    ordered_indices = _CONTEXT7_POOL.ordered_indices(len(keys))
    attempts: list[dict[str, Any]] = []
    for key_index in ordered_indices:
        api_key = keys[key_index]
        try:
            data = _get_json(
                url,
                {"query": query, "limit": min(max(1, num_results), 10)},
                {"Authorization": f"Bearer {api_key}"},
                timeout=12,
            )
        except Exception as exc:
            reason = str(exc)
            if _is_fetch_quota_or_rate_limit_error(reason):
                _CONTEXT7_POOL.mark_dead(key_index)
            attempts.append(
                _attempt(
                    capability="docs_search",
                    provider="context7",
                    status="error",
                    reason=f"key_index={key_index}: {reason}",
                    degraded=True,
                )
            )
            continue

        raw_items = data if isinstance(data, list) else data.get("results") or data.get("documents") or []
        sources: list[dict[str, Any]] = []
        documents: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items[:num_results], 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or f"Context7 result {index}")
            url_value = str(item.get("url") or item.get("source_url") or item.get("id") or f"context7://result/{index}")
            content = str(item.get("content") or item.get("text") or item.get("snippet") or "")
            sources.append(
                {
                    "title": title,
                    "url": url_value,
                    "normalized_url": _normalize_url(url_value),
                    "snippet": content[:500],
                    "source_type": "docs",
                    "provider": "context7",
                    "query": query,
                }
            )
            if content:
                documents.append(
                    _document(
                        url=url_value,
                        title=title,
                        content=content,
                        provider="context7",
                        source_type="docs",
                        metadata={"query": query},
                    )
                )

        return {
            "query": query,
            "sources": sources,
            "documents": documents,
            "provider_attempts": [
                *attempts,
                _attempt(
                    capability="docs_search",
                    provider="context7",
                    status="success",
                    reason=f"returned {len(sources)} sources",
                    selected=True,
                    content_length=len(json.dumps(data, ensure_ascii=False)),
                    content_quality="medium" if sources else "low",
                    degraded=not bool(sources),
                ),
            ],
        }

    return {"sources": [], "documents": [], "provider_attempts": attempts}


def fetch_context7_context(library_id: str, query: str, *, max_snippets: int = 8) -> dict[str, Any]:
    """Fetch documentation context for a Context7 library via /api/v2/context.

    Context7's /api/v2/libs/search (`docs_search`) returns items whose only URL-shaped field
    is a library ID path like `/crewaiinc/crewai` -- not a fetchable URL. To actually read the
    docs, call this function: it hits `/api/v2/context?libraryId=...&query=...` and renders the
    returned `codeSnippets`/`infoSnippets` into `_document` dicts whose `url` is each snippet's
    real `codeId`/`pageId` (e.g. `https://github.com/crewaiinc/crewai/blob/main/docs/...mdx`,
    `https://docs.langchain.com/oss/python/langgraph/use-graph-api`) so downstream
    `normalize_source_type`/`authority_score` still classify them as `official_docs`/`github`.

    See `project_sec03_official_sources_discovered_not_fetched_20260804`: the path-style IDs
    from `docs_search` were being dropped at `source_pool.normalize_url` (which rejects
    non-http(s) URLs) so the official docs they point at never reached the material pool. This
    function is the docs-channel counterpart to `docs_search`: discovery returns the library IDs,
    this fetches their content without going through `url_fetch` (which cannot fetch a
    library ID).
    """
    base_url = (os.getenv("CONTEXT7_BASE_URL") or "").rstrip("/")
    endpoint = os.getenv("CONTEXT7_CONTEXT_ENDPOINT") or ""
    keys = _CONTEXT7_POOL.keys()
    if not keys or not base_url or not endpoint or not library_id:
        return {
            "library_id": library_id,
            "query": query,
            "documents": [],
            "provider_attempts": [
                _attempt(
                    capability="docs_context",
                    provider="context7",
                    status="skipped",
                    reason="Context7 env not fully configured or library_id empty",
                )
            ],
        }

    url = _absolute_endpoint(base_url, endpoint)
    fast_mode = str(os.getenv("CONTEXT7_FAST_MODE", "false")).strip().lower() or "false"
    ordered_indices = _CONTEXT7_POOL.ordered_indices(len(keys))
    attempts: list[dict[str, Any]] = []
    data = None
    for key_index in ordered_indices:
        api_key = keys[key_index]
        try:
            data = _get_json(
                url,
                {
                    "libraryId": library_id,
                    "query": query or library_id,
                    "type": os.getenv("CONTEXT7_RESPONSE_TYPE", "json") or "json",
                    "fast": fast_mode,
                },
                {"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
            break
        except Exception as exc:
            reason = str(exc)
            if _is_fetch_quota_or_rate_limit_error(reason):
                _CONTEXT7_POOL.mark_dead(key_index)
            attempts.append(
                _attempt(
                    capability="docs_context",
                    provider="context7",
                    status="error",
                    reason=f"key_index={key_index}: {reason}",
                    degraded=True,
                )
            )
    if data is None:
        return {
            "library_id": library_id,
            "query": query,
            "documents": [],
            "provider_attempts": attempts,
        }

    code_snippets = data.get("codeSnippets") or [] if isinstance(data, dict) else []
    info_snippets = data.get("infoSnippets") or [] if isinstance(data, dict) else []
    documents: list[dict[str, Any]] = []
    for index, snippet in enumerate((code_snippets or [])[:max_snippets], 1):
        if not isinstance(snippet, dict):
            continue
        snippet_url = str(snippet.get("codeId") or snippet.get("pageId") or f"context7://{library_id}#code_{index}")
        title = str(snippet.get("codeTitle") or snippet.get("pageTitle") or snippet.get("breadcrumb") or snippet_url)
        parts: list[str] = []
        description = str(snippet.get("codeDescription") or "")
        if description:
            parts.append(description)
        for block in snippet.get("codeList") or []:
            if isinstance(block, dict):
                code = str(block.get("code") or "")
                language = str(block.get("language") or "")
                if code:
                    parts.append(f"```{language}\n{code}\n```" if language else code)
        content = "\n\n".join(part for part in parts if part)
        if not content:
            continue
        documents.append(_document(
            url=snippet_url,
            title=title,
            content=content,
            provider="context7",
            source_type="docs",
            metadata={"library_id": library_id, "snippet_kind": "code"},
        ))
    for index, snippet in enumerate((info_snippets or [])[:max_snippets], 1):
        if not isinstance(snippet, dict):
            continue
        snippet_url = str(snippet.get("pageId") or f"context7://{library_id}#info_{index}")
        title = str(snippet.get("breadcrumb") or snippet.get("pageTitle") or snippet_url)
        content = str(snippet.get("content") or "")
        if not content:
            continue
        documents.append(_document(
            url=snippet_url,
            title=title,
            content=content,
            provider="context7",
            source_type="docs",
            metadata={"library_id": library_id, "snippet_kind": "info"},
        ))

    return {
        "library_id": library_id,
        "query": query,
        "documents": documents,
        "provider_attempts": [
            *attempts,
            _attempt(
                capability="docs_context",
                provider="context7",
                status="success",
                reason=f"returned {len(documents)} documents",
                selected=True,
                content_length=sum(len(str(doc.get("content") or "")) for doc in documents),
                content_quality="medium" if documents else "low",
                degraded=not bool(documents),
            ),
        ],
    }


def fetch_url_with_chain(url: str, provider_chain: list[str] | None = None) -> dict[str, Any]:
    """Fetch a URL through Tavily, Firecrawl, then basic fetch unless configured otherwise."""
    if not re.match(r"^https?://", str(url).strip()):
        return {
            "document": None,
            "provider_attempts": [
                _attempt(
                    capability="url_fetch",
                    provider="validator",
                    status="denied",
                    reason="url_fetch requires explicit http(s) URL",
                    degraded=True,
                    url=str(url),
                )
            ],
        }

    adapter_document, adapter_attempt = _fetch_with_content_adapter(url)
    attempts: list[dict[str, Any]] = []
    if adapter_attempt:
        attempts.append(adapter_attempt)
    if adapter_document and not adapter_document.get("degraded"):
        attempts[-1]["selected"] = True
        return {"document": adapter_document, "provider_attempts": attempts}

    chain_text = os.getenv("URL_FETCH_PROVIDER_CHAIN", "tavily,firecrawl,basic")
    chain = provider_chain or [item.strip().lower() for item in chain_text.split(",") if item.strip()]

    for index, provider in enumerate(chain):
        fallback_used = index > 0
        if provider in {"tavily", "tavily_extract"}:
            document, attempt = _fetch_tavily(url, fallback_used=fallback_used)
        elif provider in {"firecrawl", "firecrawl_scrape"}:
            document, attempt = _fetch_firecrawl(url, fallback_used=fallback_used)
        elif provider in {"basic", "urllib", "requests_bs4", "bs4"}:
            document, attempt = _fetch_basic(url, fallback_used=fallback_used)
        else:
            document = None
            attempt = _attempt(
                capability="url_fetch",
                provider=provider,
                status="skipped",
                reason="unknown provider in URL_FETCH_PROVIDER_CHAIN",
                fallback_used=fallback_used,
                url=url,
            )
        attempts.append(attempt)
        if document and not document.get("degraded"):
            attempts[-1]["selected"] = True
            return {"document": document, "provider_attempts": attempts}

    degraded_doc = None
    for provider_attempt in attempts:
        if provider_attempt.get("status") == "success":
            degraded_doc = provider_attempt.get("_document")
            provider_attempt.pop("_document", None)
            provider_attempt["selected"] = True
            break
    for provider_attempt in attempts:
        provider_attempt.pop("_document", None)
    return {"document": degraded_doc, "provider_attempts": attempts}


def _fetch_with_content_adapter(url: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_url = _github_raw_url(url)
    if raw_url:
        return _fetch_github_raw(url, raw_url)
    if urllib.parse.urlsplit(url).path.lower().endswith(".pdf"):
        return _fetch_pdf(url)
    return None, None


def _github_raw_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.netloc.lower() != "github.com":
        return ""
    parts = [urllib.parse.unquote(item) for item in parsed.path.split("/") if item]
    if len(parts) < 5 or parts[2] != "blob":
        return ""
    owner, repo, _blob, ref = parts[:4]
    path = "/".join(urllib.parse.quote(item, safe="") for item in parts[4:])
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{urllib.parse.quote(ref, safe='')}/{path}"


def _fetch_github_raw(original_url: str, raw_url: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        request = urllib.request.Request(
            raw_url,
            headers={"User-Agent": "HarnessAgentMVP/0.2", "Accept": "text/plain"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read(500000).decode("utf-8", errors="replace")
        title = urllib.parse.unquote(urllib.parse.urlsplit(original_url).path.rsplit("/", 1)[-1]) or original_url
        document = _document(
            url=original_url,
            title=title,
            content=raw,
            provider="github_raw",
            source_type="github",
            metadata={"fetched_url": raw_url, "adapter": "github_raw"},
        )
        attempt = _attempt(
            capability="url_fetch",
            provider="github_raw",
            status="success",
            reason=document["reason"],
            degraded=document["degraded"],
            content_length=document["content_length"],
            content_quality=document["content_quality"],
            url=original_url,
        )
        attempt["_document"] = document
        return document, attempt
    except Exception as exc:
        return None, _attempt(
            capability="url_fetch",
            provider="github_raw",
            status="error",
            reason=str(exc),
            degraded=True,
            url=original_url,
        )


def _fetch_pdf(url: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        from pypdf import PdfReader

        request = urllib.request.Request(url, headers={"User-Agent": "HarnessAgentMVP/0.2"})
        with urllib.request.urlopen(request, timeout=18) as response:
            raw = response.read(12_000_000)
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages[:40]).strip()
        title = str((reader.metadata or {}).get("/Title") or urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]))
        document = _document(
            url=url,
            title=title,
            content=text,
            provider="pdf_text",
            source_type="paper",
            metadata={"adapter": "pypdf", "page_count": len(reader.pages)},
        )
        attempt = _attempt(
            capability="url_fetch",
            provider="pdf_text",
            status="success",
            reason=document["reason"],
            degraded=document["degraded"],
            content_length=document["content_length"],
            content_quality=document["content_quality"],
            url=url,
        )
        attempt["_document"] = document
        return document, attempt
    except Exception as exc:
        return None, _attempt(
            capability="url_fetch",
            provider="pdf_text",
            status="error",
            reason=str(exc),
            degraded=True,
            url=url,
        )


def _is_fetch_quota_or_rate_limit_error(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in ("429", "rate limit", "too many requests", "quota", "not enough credits"))


def _clean_tavily_markdown(text: str) -> str:
    """Strip the two noise shapes audited in 480 real Tavily materials from 120 real jobs
    (see docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.1.md,
    "Tavily 具体清洗规则设计"): a head-of-page nav block bundled with a literal "Help Center"
    link (Perplexity/whatagraph blog samples), and a newer tail decoration -- an AI-summary
    share component ("Summarize the content at ... via ChatGPT/Perplexity/...") sometimes
    followed by a truncated "WRITTEN BY ..." author-bio block (whatagraph.com sample).

    Deliberately narrow: only these exact observed patterns, not a generalized "any AI tool
    name + summarize" heuristic -- the audit sample was too small (1 real tail example) to
    generalize safely without risking false positives on legitimate content. Tavily has no
    onlyMainContent-style server-side filter (confirmed via its API docs), so this is the only
    cleaning its output gets.
    """
    if not text:
        return text
    cleaned = text

    # Head nav residue: only treated as noise within a bounded prefix window, not wherever
    # "Help Center" happens to appear -- a mid-article mention (e.g. an article that discusses
    # a product's "Help Center" feature) must not truncate real content.
    head_window = cleaned[:600]
    head_match = _TAVILY_HEAD_NAV_MARKER_RE.search(head_window)
    if head_match:
        cleaned = cleaned[head_match.end():].lstrip()

    tail_match = _TAVILY_TAIL_AI_SUMMARY_RE.search(cleaned)
    if tail_match:
        cleaned = cleaned[: tail_match.start()].rstrip()

    bio_match = _TAVILY_TAIL_AUTHOR_BIO_RE.search(cleaned)
    if bio_match:
        cleaned = cleaned[: bio_match.start()].rstrip()

    return cleaned.strip()


def _fetch_tavily(url: str, *, fallback_used: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    keys = _TAVILY_POOL.keys()
    endpoint = os.getenv("TAVILY_EXTRACT_URL", DEFAULT_TAVILY_EXTRACT_URL)
    if not keys:
        return None, _attempt(
            capability="url_fetch",
            provider="tavily",
            status="skipped",
            reason="TAVILY_API_KEY not configured",
            fallback_used=fallback_used,
            url=url,
        )
    ordered_indices = _TAVILY_POOL.ordered_indices(len(keys))
    last_attempt: dict[str, Any] | None = None
    for key_index in ordered_indices:
        api_key = keys[key_index]
        try:
            data = _post_json(endpoint, {"api_key": api_key, "urls": [url]}, {}, timeout=15)
            results = data.get("results") or []
            first = results[0] if results else {}
            content = _clean_tavily_markdown(str(first.get("raw_content") or first.get("content") or ""))
            title = str(first.get("title") or url)
            document = _document(url=url, title=title, content=content, provider="tavily", source_type=_source_type_from_url(url))
            attempt = _attempt(
                capability="url_fetch",
                provider="tavily",
                status="success",
                reason=document["reason"],
                fallback_used=fallback_used,
                degraded=document["degraded"],
                content_length=document["content_length"],
                content_quality=document["content_quality"],
                url=url,
            )
            attempt["_document"] = document
            return document, attempt
        except Exception as exc:
            reason = str(exc)
            if _is_fetch_quota_or_rate_limit_error(reason):
                _TAVILY_POOL.mark_dead(key_index)
            last_attempt = _attempt(
                capability="url_fetch",
                provider="tavily",
                status="error",
                reason=f"key_index={key_index}: {reason}",
                fallback_used=fallback_used,
                degraded=True,
                url=url,
            )
    return None, last_attempt or _attempt(
        capability="url_fetch",
        provider="tavily",
        status="error",
        reason="all tavily keys failed",
        fallback_used=fallback_used,
        degraded=True,
        url=url,
    )


def _fetch_firecrawl(url: str, *, fallback_used: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    keys = _FIRECRAWL_POOL.keys()
    endpoint = _absolute_endpoint(
        "https://api.firecrawl.dev",
        os.getenv("FIRECRAWL_SCRAPE_ENDPOINT", DEFAULT_FIRECRAWL_SCRAPE_ENDPOINT),
    )
    if not keys:
        return None, _attempt(
            capability="url_fetch",
            provider="firecrawl",
            status="skipped",
            reason="FIRECRAWL_API_KEY not configured",
            fallback_used=fallback_used,
            url=url,
        )
    ordered_indices = _FIRECRAWL_POOL.ordered_indices(len(keys))
    last_attempt: dict[str, Any] | None = None
    for key_index in ordered_indices:
        api_key = keys[key_index]
        try:
            data = _post_json(
                endpoint,
                {"url": url, "formats": ["markdown", "html"], "onlyMainContent": True},
                {"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
            payload = data.get("data", data) if isinstance(data, dict) else {}
            # trafilatura(html) is the primary path -- Firecrawl's own `markdown` field is
            # server-side cleaned but we don't control its rules; running our own statistical
            # extraction over the `html` field (already requested via formats=[...,"html"], but
            # previously fetched-and-never-used) gives consistent, tunable (favor_recall)
            # cleaning across both firecrawl and basic. Falls back to markdown/content/html
            # verbatim if trafilatura isn't available, the html field is missing, or extraction
            # comes back empty.
            html_content = str(payload.get("html") or "")
            content = _extract_with_trafilatura(html_content)
            if not content:
                content = str(payload.get("markdown") or payload.get("content") or html_content)
            title = str((payload.get("metadata") or {}).get("title") or payload.get("title") or url)
            document = _document(url=url, title=title, content=content, provider="firecrawl", source_type=_source_type_from_url(url))
            attempt = _attempt(
                capability="url_fetch",
                provider="firecrawl",
                status="success",
                reason=document["reason"],
                fallback_used=fallback_used,
                degraded=document["degraded"],
                content_length=document["content_length"],
                content_quality=document["content_quality"],
                url=url,
            )
            attempt["_document"] = document
            return document, attempt
        except Exception as exc:
            reason = str(exc)
            if _is_fetch_quota_or_rate_limit_error(reason):
                _FIRECRAWL_POOL.mark_dead(key_index)
            last_attempt = _attempt(
                capability="url_fetch",
                provider="firecrawl",
                status="error",
                reason=f"key_index={key_index}: {reason}",
                fallback_used=fallback_used,
                degraded=True,
                url=url,
            )
    return None, last_attempt or _attempt(
        capability="url_fetch",
        provider="firecrawl",
        status="error",
        reason="all firecrawl keys failed",
        fallback_used=fallback_used,
        degraded=True,
        url=url,
    )


def _fetch_basic(url: str, *, fallback_used: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HarnessAgentMVP/0.2"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_bytes = response.read(80000)
            content_type = response.headers.get("content-type", "")
        raw = raw_bytes.decode("utf-8", errors="replace")
        title, text = _strip_html(raw) if "html" in content_type.lower() or "<html" in raw[:500].lower() else ("", raw)
        document = _document(url=url, title=title or url, content=text, provider="basic", source_type=_source_type_from_url(url))
        attempt = _attempt(
            capability="url_fetch",
            provider="basic",
            status="success",
            reason=document["reason"],
            fallback_used=fallback_used,
            degraded=document["degraded"],
            content_length=document["content_length"],
            content_quality=document["content_quality"],
            url=url,
        )
        attempt["_document"] = document
        return document, attempt
    except Exception as exc:
        return None, _attempt(
            capability="url_fetch",
            provider="basic",
            status="error",
            reason=str(exc),
            fallback_used=fallback_used,
            degraded=True,
            url=url,
        )


@tool
def search_web(query: str, num_results: int = 5) -> str:
    """Search the web using Serper API and return JSON search results."""
    result = broad_discovery(query, num_results=num_results)
    return _json_dumps(
        {
            "query": query,
            "results": [
                {
                    "title": source.get("title", ""),
                    "link": source.get("url", ""),
                    "snippet": source.get("snippet", ""),
                }
                for source in result.get("sources", [])
            ],
            "total_results": len(result.get("sources", [])),
            "answer_box": result.get("answer_box"),
            "knowledge_graph": result.get("knowledge_graph"),
            "provider_attempts": result.get("provider_attempts", []),
        }
    )


@tool
def fetch_url(url: str) -> str:
    """Fetch content from an explicit URL through the configured provider chain."""
    result = fetch_url_with_chain(url)
    document = result.get("document")
    if not document:
        return _json_dumps(
            {
                "url": url,
                "fetch_error": "No provider returned usable content.",
                "provider_attempts": result.get("provider_attempts", []),
            }
        )
    return _json_dumps(
        {
            "url": document.get("url"),
            "title": document.get("title"),
            "provider": document.get("provider"),
            "content_quality": document.get("content_quality"),
            "degraded": document.get("degraded"),
            "content": document.get("content", "")[:WEB_FETCH_MAX_CONTENT_CHARS],
            "provider_attempts": result.get("provider_attempts", []),
        }
    )
