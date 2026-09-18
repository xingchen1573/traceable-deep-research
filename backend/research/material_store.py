"""Job-owned fetched documents shared by question-level research consumers."""

from __future__ import annotations

from threading import Lock
from typing import Any

from .source_pool import SourceCandidate, normalize_url


class JobMaterialStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._documents: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, list[dict[str, Any]]] = {}
        self._failed_urls: set[str] = set()
        # Context7 library-id -> list of sub-keys into self._documents (one library yields
        # multiple snippet docs). `record_fetch` can't store these because its key is
        # `normalize_url(candidate.url)` and the candidate URL is a synthetic
        # `context7://library/<id>` that `normalize_url` rejects.
        self._context7_attempts: dict[str, list[dict[str, Any]]] = {}
        self._context7_library_docs: dict[str, list[str]] = {}
        self._capabilities: dict[str, dict[str, Any]] = {}
        # Job-local chunk vectors used by the independent semantic-recall channel. Keys include
        # the source identity, chunk index and a content digest, so a refetched/changed document
        # naturally gets a new entry without requiring cross-thread invalidation. The cache is
        # intentionally job-scoped: research jobs usually hold only tens of documents, and no
        # persistent vector database is needed for this transient material pool.
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._embedding_cache_hits = 0
        self._embedding_cache_misses = 0

    def disable_capability(self, capability: str, reason: str) -> bool:
        """Disable a flaky job-scoped capability once and retain the first reason."""
        with self._lock:
            if capability in self._capabilities and self._capabilities[capability].get("disabled"):
                return False
            self._capabilities[capability] = {"disabled": True, "reason": str(reason)[:240]}
            return True

    def capability_status(self, capability: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._capabilities.get(capability) or {"disabled": False})

    def record_fetch(self, candidate: SourceCandidate, result: dict[str, Any]) -> None:
        key = normalize_url(candidate.url)
        if not key:
            return
        document = result.get("document") if isinstance(result, dict) else None
        attempts = [
            {key: value for key, value in item.items() if not str(key).startswith("_")}
            for item in (result.get("provider_attempts") or [])
            if isinstance(item, dict)
        ] if isinstance(result, dict) else []
        with self._lock:
            self._attempts[key] = attempts
            if isinstance(document, dict) and str(document.get("content") or "").strip():
                self._documents[key] = {
                    **document,
                    "url": candidate.url,
                    "source_type": candidate.source_type,
                    "source_pool": {
                        "discovered_by_question_ids": sorted(candidate.discovered_by_question_ids),
                        "matched_criterion_ids": sorted(candidate.matched_criterion_ids),
                    },
                }
                self._failed_urls.discard(key)
            else:
                self._failed_urls.add(key)

    def record_context7_documents(
        self,
        candidate: SourceCandidate,
        documents: list[dict[str, Any]],
        provider_attempts: list[dict[str, Any]],
    ) -> int:
        """Store docs fetched via Context7's /api/v2/context for a library candidate.

        `record_fetch` keys on `normalize_url(candidate.url)`, but a Context7 library
        candidate's URL is a synthetic `context7://library/<id>` that `normalize_url` rejects
        (returns ""), so `record_fetch` would no-op on it. This method keys on
        `candidate.library_id` and stores each snippet document under a unique sub-key so a
        single library can yield multiple materials. The snippet's own `url` (its `codeId`/
        `pageId`, a real https URL) is preserved on the stored document so downstream
        `normalize_source_type`/`authority_score` still classify it as `official_docs`/`github`.
        Returns the number of documents actually stored (those with non-empty content).
        """
        if not candidate.library_id:
            return 0
        library_key = candidate.library_id
        attempts = [
            {key: value for key, value in item.items() if not str(key).startswith("_")}
            for item in (provider_attempts or [])
            if isinstance(item, dict)
        ]
        stored = 0
        with self._lock:
            self._context7_attempts.setdefault(library_key, []).extend(attempts)
            for index, doc in enumerate(documents or []):
                if not isinstance(doc, dict):
                    continue
                content = str(doc.get("content") or "").strip()
                if not content:
                    continue
                sub_key = f"{library_key}#snippet_{stored + 1}"
                self._documents[sub_key] = {
                    **doc,
                    "url": str(doc.get("url") or candidate.url),
                    "source_type": candidate.source_type,
                    "source_pool": {
                        "discovered_by_question_ids": sorted(candidate.discovered_by_question_ids),
                        "matched_criterion_ids": sorted(candidate.matched_criterion_ids),
                    },
                }
                stored += 1
            self._context7_library_docs.setdefault(library_key, []).clear()
            self._context7_library_docs[library_key].extend(
                f"{library_key}#snippet_{i + 1}" for i in range(stored)
            )
        return stored

    def all_documents(self) -> list[dict[str, Any]]:
        """Every fetched document currently in the store, regardless of which candidate/question
        discovered it -- backs `search_local`'s "global shared material pool" scope (see
        docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.2.md
        decision 3): a question's search_local call should see materials any question's
        search_web already fetched, not just its own."""
        with self._lock:
            return [dict(item) for item in self._documents.values()]

    def cached_chunk_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        with self._lock:
            output = {key: list(self._chunk_embeddings[key]) for key in keys if key in self._chunk_embeddings}
            self._embedding_cache_hits += len(output)
            self._embedding_cache_misses += max(0, len(keys) - len(output))
            return output

    def store_chunk_embeddings(self, values: dict[str, list[float]]) -> None:
        with self._lock:
            for key, vector in values.items():
                if key and vector:
                    self._chunk_embeddings[key] = list(vector)

    def documents(self, candidates: list[SourceCandidate]) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for candidate in candidates:
                key = normalize_url(candidate.url)
                if key and key in self._documents:
                    out.append(dict(self._documents[key]))
                    continue
                if candidate.library_id:
                    for sub_key in self._context7_library_docs.get(candidate.library_id, []):
                        if sub_key in self._documents:
                            out.append(dict(self._documents[sub_key]))
            return out

    def provider_attempts(self, candidates: list[SourceCandidate]) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for candidate in candidates:
                key = normalize_url(candidate.url)
                if key:
                    out.extend(dict(item) for item in self._attempts.get(key, []))
                if candidate.library_id:
                    out.extend(dict(item) for item in self._context7_attempts.get(candidate.library_id, []))
            return out

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "document_count": len(self._documents),
                "failed_url_count": len(self._failed_urls),
                "provider_attempt_count": sum(len(items) for items in self._attempts.values())
                + sum(len(items) for items in self._context7_attempts.values()),
                "content_chars": sum(len(str(item.get("content") or "")) for item in self._documents.values()),
                "chunk_embedding_count": len(self._chunk_embeddings),
                "embedding_cache_hit_count": self._embedding_cache_hits,
                "embedding_cache_miss_count": self._embedding_cache_misses,
                "capabilities": {key: dict(value) for key, value in self._capabilities.items()},
            }
