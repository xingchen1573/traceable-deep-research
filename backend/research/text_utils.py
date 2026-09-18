"""Shared text normalization and term-extraction primitives for research scoring/matching.

Every module that scores or matches Chinese/English text against a query used to carry its own
copy of the same "convert to simplified + split into character bigrams" logic. That fragmentation
let one bug (missing traditional-to-simplified folding, or naive whole-sentence bigram splitting
that truncates long questions before their most distinctive words) get fixed in one call site and
silently persist in every other one. This module is the single place that logic lives now.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_CJK_RANGE = r"一-鿿"


@lru_cache(maxsize=1)
def _traditional_to_simplified_converter() -> Any:
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    return OpenCC("t2s")


def to_simplified(value: str) -> str:
    """Fold traditional Chinese to simplified before any term extraction or matching.

    Character-level matching (bigrams, substring overlap) treats traditional and simplified script
    as unrelated languages -- "來源" never overlaps "来源". A real ranking replay had the
    most on-topic candidate in a run -- a traditional-Chinese post answering the question almost
    verbatim -- scored below the admission bar purely because of script, while its simplified
    transcription passed.
    """
    converter = _traditional_to_simplified_converter()
    if converter is None:
        return value
    return converter.convert(value)


def extract_terms(value: str, *, min_len: int = 2) -> set[str]:
    """Bigram/token extraction for short-text matching (candidate titles, chunk text).

    Kept for call sites that compare short strings where character bigrams work fine (there's no
    truncation-order problem when the input is a title or a single sentence, only when it's a long
    question -- see extract_key_terms for that case). Behavior is intentionally identical to the
    term extraction this project already relied on and tuned admission thresholds against.
    """
    lowered = to_simplified(str(value or "").lower())
    words = {
        item
        for item in re.findall(rf"[a-z0-9][a-z0-9_.+-]{{1,}}|[{_CJK_RANGE}]{{2,8}}", lowered)
        if len(item) >= min_len
    }
    chinese = "".join(re.findall(rf"[{_CJK_RANGE}]", lowered))
    words.update(chinese[index:index + 2] for index in range(max(0, len(chinese) - 1)))
    return words


@lru_cache(maxsize=1)
def _jieba_analyse():
    try:
        import jieba.analyse
    except ImportError:
        return None
    return jieba.analyse


def extract_key_terms(text: str, *, top_k: int = 16) -> list[str]:
    """Phrase-ranked keyword extraction for long natural-language questions.

    Splitting a whole 20-40 char question into character bigrams and truncating to the first N
    produces a term list ordered by *position*, not *importance* -- a distinctive word near the
    end of the sentence (e.g. "延迟" in "...检索流程（如延迟、Token消耗）上有哪些差异") gets cut off before it's
    ever considered, while low-signal function words near the start survive. TF-IDF keyword
    extraction ranks by importance instead, so truncation drops the least useful terms first.
    Falls back to extract_terms (still simplified-folded) if jieba isn't installed. Lowercased
    like extract_terms -- without this, "Travel" (a source title) and "travel" (a lowercase
    question) tokenize to different tags and never overlap, silently zeroing out relevance for
    any English source whose title happens to capitalize the shared word.
    """
    normalized = to_simplified(str(text or "").lower())
    analyse = _jieba_analyse()
    if analyse is None:
        return list(extract_terms(normalized))[:top_k]
    tags = analyse.extract_tags(normalized, topK=top_k)
    if tags:
        return list(tags)
    return list(extract_terms(normalized))[:top_k]


def normalize_for_substring(value: str) -> str:
    """Collapse a string for verbatim substring matching across cosmetic differences.

    Lowercase, fold to simplified, drop all whitespace and punctuation. Two Chinese sentences
    that differ only in full-width vs half-width commas, quote style, or whitespace then compare
    equal for the purpose of "did this revision paste back a sentence it was told not to". The
    granularity here is *whole-string identity after surface removal* -- deliberately coarse so a
    true verbatim reintroduction is caught without the false-positive risk of token overlap.
    """
    lowered = to_simplified(str(value or "").lower())
    no_space = re.sub(r"[\s　]+", "", lowered)
    return re.sub(r"[，。、；：,.;:!?()（）\[\]【】\"'`\-—_]", "", no_space)


def tokenize_for_overlap(value: str) -> set[str]:
    """Token set for Jaccard-style overlap between two short Chinese/English sentences.

    English words of length >=2 (`[a-z]{2,}`) plus individual CJK characters (one char per token,
    not bigrams -- bigram overlap is brittle to a single character's repositioning during a light
    rewrite, which is exactly the regime we score here). Simplified-folded so traditional/simplified
    pairs overlap. This is the granularity the isolation A/B tests in
    scripts/replays/legacy/replay_case1_ab_variants.py
    .py tuned the 0.5 overlap threshold against; keeping it identical avoids silently changing the
    calibration when the same check moves into the production writer path.
    """
    lowered = to_simplified(str(value or "").lower())
    spaced = re.sub(r"[\s　]+", " ", lowered)
    tokens: set[str] = set()
    for match in re.findall(rf"[a-z]{{2,}}|[{_CJK_RANGE}]", spaced):
        tokens.add(match)
    return tokens


def is_non_research_section_title(value: str, *, has_questions: bool | None = None) -> bool:
    """Return whether a section is synthesized report metadata, not an executable question."""
    text = str(value or "").strip().lower()
    label = re.sub(r"^\d{1,2}[_\-.、\s]+", "", text).strip()
    if any(token in label for token in ("总览", "overview", "summary")):
        return True
    reference_terms = ("证据", "参考", "引用", "来源", "reference", "citation")
    metadata_terms = ("索引", "清单", "资料", "文献", "附录", "index", "references", "appendix")
    matched_reference_terms = {token for token in reference_terms if token in label}
    # An explicitly empty question list is a structural signal supplied by the planner. It lets
    # concise labels such as "证据来源" remain metadata without classifying the substantive axis
    # "证据引用" as metadata when that section carries an actual research question.
    if has_questions is False and matched_reference_terms:
        return True
    if matched_reference_terms and any(token in label for token in metadata_terms):
        return True
    if len(matched_reference_terms) < 2 or not re.search(r"(?:与|和|及|/|&|\+)", label):
        return False
    residue = label
    for token in sorted(matched_reference_terms, key=len, reverse=True):
        residue = residue.replace(token, "")
    residue = re.sub(r"[\s_\-.、，,:：;；与和及/&+()（）]+", "", residue)
    return not residue
