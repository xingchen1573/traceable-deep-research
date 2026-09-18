"""Real E2E evaluation for the single_url brief pipeline.

See docs/VISION.md 第4节"测评体系" -- URL 型用例. Unlike question_loop, the single_url pipeline
(tools/research_jobs.py::start_single_url_brief_tool -> research_service._run_single_url_research)
has no quality_gate/validation payload, so it can't plug into scripts/run_research_multi_topic_probe.py's
runner. This is an independent script with its own four judges, matching VISION.md's spec exactly:

1. citation_authenticity -- every "> {quote}" spliced into brief_markdown is an exact, verbatim
   substring of the real fetched source paragraph it claims to cite (no fabricated locators).
2. content_substantiveness -- summary / key_points_narrative / commentary are real content, not
   the renderer's own fallback placeholder strings, and meet a minimum length.
3. direct_delivery -- the job result never carries status="waiting_approval"; single_url must
   resolve straight to "succeeded".
4. no_auto_save -- save_knowledge / save_knowledge_set are never invoked by the pipeline itself
   (the "confirm after asking" half of this criterion needs a real agent loop and is covered
   separately by tests/test_channel_hitl_resume.py, not by this network-driven script).

Hits real network + real LLM calls, same as run_research_multi_topic_probe.py does for
question_loop -- results will vary run to run with source content and model output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT / ".env", override=False)


_load_dotenv()

import backend.services.research_service as research_service  # noqa: E402
from backend.services import eval_case_store  # noqa: E402
import tools.knowledge as knowledge_tools  # noqa: E402
from backend.services.job_service import job_service  # noqa: E402

OUTPUT_DIR = ROOT / "logs" / "single_url_eval"


def _default_cases() -> list[dict[str, Any]]:
    return eval_case_store.load_cases("single_url_brief")

_SUMMARY_PLACEHOLDER = "（模型未返回摘要）"
_KEY_POINTS_PLACEHOLDER = "（模型未返回核心论点与数据）"
_NO_CITATION_PLACEHOLDER = "（原文未提供可精确定位的引用片段。）"
_COMMENTARY_FIELD_LABELS = {
    "supporting_evidence": "支持证据",
    "counter_evidence": "反驳/制约证据",
    "limitations": "局限性",
    "applicable_scenarios": "适用场景",
    "overall_credibility": "整体可信度",
}
_COMMENTARY_PLACEHOLDER_VALUES = {
    "supporting_evidence": "未提供",
    "counter_evidence": "原文未提供明显的反面信息",
    "limitations": "未提供",
    "applicable_scenarios": "未提供",
    "overall_credibility": "未提供",
}

_CITATION_PATTERN = re.compile(r"> (.*?)\n\n（原文 (P\d+)）", re.DOTALL)
_SECTION_PATTERN = re.compile(r"^## (.+)$", re.MULTILINE)
# Fingerprint of the field-collapse failure mode (docs/evaluations/e2e-eval-sweep-20260726.md
# section 3): the model merges summary/citations/commentary into key_points_narrative, faking
# section breaks with bold markdown labels instead of using the separate schema fields. Catches
# partial variants too -- e.g. summary non-empty but key_points_narrative still contains the
# other fields' content -- not just the "summary ends up a placeholder" case already caught above.
_COLLAPSED_FIELD_MARKER_RE = re.compile(r"\*\*(summary|key_points_narrative|citations|commentary)\*\*")

# Judge blind spot fingerprint (docs/evaluations/single-url-brief-eval-e2e-20260725.md section 4,
# docs/evaluations/e2e-eval-sweep-20260726.md section 3): a real extraction failure can produce a
# structurally complete brief -- summary/key_points/commentary are all non-empty, non-placeholder,
# and pass the length thresholds -- because the model wrote a full, well-formed paragraph *about*
# having nothing to extract, e.g. "本次提取未能获取该来源的具体内容" (wiki_llm, 07-25 run,
# summary_chars=88/key_points_chars=97, both pass their minimums, citation_count=0). Structural
# completeness checks alone can't catch this; these phrases are the model's own admission that
# extraction produced nothing, so they're checked verbatim against every substantive field
# (summary/key_points/each commentary bullet) regardless of length.
_SELF_ADMITTED_EMPTY_EXTRACTION_RE = re.compile(
    "|".join(re.escape(phrase) for phrase in (
        "提取结果为空", "提取未能获取", "无提取到", "未能提取", "提取失败",
        "无法提供具体信息", "未被结构化捕捉", "无法评估其内容", "本次提取未能",
    ))
)


def _extract_section(markdown: str, header: str) -> str:
    """Text between `## {header}` and the next `## ` heading (or end of doc)."""
    matches = list(_SECTION_PATTERN.finditer(markdown))
    for idx, match in enumerate(matches):
        if match.group(1).strip() != header:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        return markdown[start:end].strip()
    return ""


def _extract_commentary_bullets(markdown: str) -> dict[str, str]:
    section = _extract_section(markdown, "点评")
    bullets: dict[str, str] = {}
    for key, label in _COMMENTARY_FIELD_LABELS.items():
        match = re.search(rf"^- {re.escape(label)}：(.*)$", section, re.MULTILINE)
        bullets[key] = match.group(1).strip() if match else ""
    return bullets


def _check_content_substantiveness(case: dict[str, Any], markdown: str, *, citation_count: int) -> dict[str, Any]:
    summary = _extract_section(markdown, "摘要")
    key_points = _extract_section(markdown, "核心论点与数据")
    commentary = _extract_commentary_bullets(markdown)

    problems: list[str] = []
    if not summary or summary == _SUMMARY_PLACEHOLDER:
        problems.append("summary 是占位文案")
    elif len(summary) < case["min_summary_chars"]:
        problems.append(f"summary 长度 {len(summary)} 字 < 最小值 {case['min_summary_chars']}")

    if not key_points or key_points == _KEY_POINTS_PLACEHOLDER:
        problems.append("key_points_narrative 是占位文案")
    elif len(key_points) < case["min_key_points_chars"]:
        problems.append(f"key_points_narrative 长度 {len(key_points)} 字 < 最小值 {case['min_key_points_chars']}")

    placeholder_fields = [
        key
        for key, value in commentary.items()
        if not value or value == _COMMENTARY_PLACEHOLDER_VALUES[key]
    ]
    # counter_evidence's placeholder text is also a legitimate model answer when the source
    # genuinely has no counterpoint -- only flag it alongside at least one other placeholder
    # field, since counter_evidence alone being "no counterpoint" is expected for many sources.
    if placeholder_fields == ["counter_evidence"]:
        placeholder_fields = []
    if placeholder_fields:
        problems.append(f"commentary 占位字段：{', '.join(placeholder_fields)}")

    if _COLLAPSED_FIELD_MARKER_RE.search(key_points):
        problems.append("key_points_narrative 内混入了其他字段的伪标题（字段合并/错位的特征信号）")

    # Judge blind spot fix: a structurally complete, length-passing field can still be the model
    # narrating its own extraction failure rather than real content (see the module-level
    # docstring on _SELF_ADMITTED_EMPTY_EXTRACTION_RE). Scan every substantive field regardless
    # of whether it already tripped a length/placeholder problem above.
    self_admitted_fields = [
        name for name, text in [
            ("summary", summary), ("key_points_narrative", key_points), *commentary.items(),
        ]
        if text and _SELF_ADMITTED_EMPTY_EXTRACTION_RE.search(text)
    ]
    if self_admitted_fields:
        problems.append(f"字段内容是模型自述提取为空/失败，非真实抽取内容：{', '.join(self_admitted_fields)}")

    # citation_count=0 alongside any self-admitted-empty field corroborates a real extraction
    # failure rather than a source that genuinely has few quotable passages -- citations and
    # commentary are independent LLM outputs over the same fetched content, so both reporting
    # "nothing here" is a much stronger signal than either alone.
    if citation_count == 0 and self_admitted_fields:
        problems.append("citation_count=0 且存在自述提取失败字段，两个独立信号共同指向真实抽取失败")

    return {
        "passed": not problems,
        "problems": problems,
        "summary_chars": len(summary),
        "key_points_chars": len(key_points),
        "commentary": commentary,
    }


def _check_citation_authenticity(markdown: str, paragraph_by_id: dict[str, str]) -> dict[str, Any]:
    matches = _CITATION_PATTERN.findall(markdown)
    problems: list[str] = []
    checked = 0
    for quote, locator in matches:
        checked += 1
        source_text = paragraph_by_id.get(locator)
        if source_text is None:
            problems.append(f"{locator} 在原文分段中不存在（虚构定位符）")
            continue
        if quote.strip() != source_text:
            problems.append(f"{locator} 引用文字与原文分段不完全一致（逐字切片校验失败）")
    return {
        "passed": not problems,
        "problems": problems,
        "citations_checked": checked,
    }


def _check_direct_delivery(result: dict[str, Any]) -> dict[str, Any]:
    has_status = "status" in result
    passed = not has_status or result.get("status") != "waiting_approval"
    return {
        "passed": passed,
        "problems": [] if passed else ["job result 携带 status=waiting_approval，未直接交付"],
    }


def _check_html_artifact_generated(result: dict[str, Any]) -> dict[str, Any]:
    """VISION.md 第2节：single_url_brief 现在也要落一份 brief.html（证据标注阅读版，
    backend/services/evidence_reader_service.py 程序化渲染，非 kimi 生成）。判据只检查产物是否
    真的写到了 artifacts 里 -- 内容层面的引用真实性已经由 citation_authenticity 覆盖
    （HTML 和 markdown 共享同一份 synthesis/annotations 数据源，不需要重复校验一遍）。"""
    artifacts = result.get("artifacts") or []
    html_paths = [p for p in artifacts if str(p).endswith("brief.html")]
    passed = len(html_paths) == 1
    return {
        "passed": passed,
        "problems": [] if passed else [f"brief.html 未生成或数量异常（artifacts 中 {len(html_paths)} 个匹配项）"],
    }


def run_case(case: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    started = time.time()
    captured: dict[str, Any] = {}
    save_calls: list[str] = []

    real_fetch = research_service.fetch_url_with_chain

    def _spy_fetch(url: str, provider_chain: list[str] | None = None) -> dict[str, Any]:
        fetch_result = real_fetch(url, provider_chain=provider_chain)
        captured["fetch_result"] = fetch_result
        return fetch_result

    def _poison_save(*args: Any, **kwargs: Any) -> None:
        save_calls.append("save_knowledge")
        raise AssertionError("single_url pipeline must never call save_knowledge directly")

    def _poison_save_set(*args: Any, **kwargs: Any) -> None:
        save_calls.append("save_knowledge_set")
        raise AssertionError("single_url pipeline must never call save_knowledge_set directly")

    original_fetch = research_service.fetch_url_with_chain
    original_save = knowledge_tools.save_knowledge
    original_save_set = knowledge_tools.save_knowledge_set
    research_service.fetch_url_with_chain = _spy_fetch
    knowledge_tools.save_knowledge = _poison_save
    knowledge_tools.save_knowledge_set = _poison_save_set

    row: dict[str, Any] = {"id": case["id"], "url": case["url"]}
    try:
        input_data = {
            "url": case["url"],
            "user_request": case.get("user_request", ""),
            "channel_id": f"eval:single_url:{case['id']}",
        }
        job = job_service.create(job_type="research", input_data=input_data, target=None)
        job_id = str(job["job_id"])
        job_service._run(
            job_id,
            lambda data: research_service._run_single_url_research(job_id, data),
        )
        finished_job = job_service.get(job_id)
        if finished_job.get("status") != "succeeded":
            error = finished_job.get("error") if isinstance(finished_job.get("error"), dict) else {}
            raise RuntimeError(str(error.get("message") or "single_url research job failed"))
        result = finished_job.get("result") if isinstance(finished_job.get("result"), dict) else {}
        row["job_id"] = job_id

        markdown = result.get("brief_markdown", "")
        raw_content = str(((captured.get("fetch_result") or {}).get("document") or {}).get("content") or "")
        paragraphs = research_service._segment_source_paragraphs(raw_content[: research_service._SINGLE_URL_MAX_CONTENT_CHARS])
        paragraph_by_id = {item["id"]: item["text"] for item in paragraphs}

        checks = {
            "direct_delivery": _check_direct_delivery(result),
            "no_auto_save": {"passed": not save_calls, "problems": [f"意外调用了 {name}" for name in save_calls]},
            "content_substantiveness": _check_content_substantiveness(
                case, markdown, citation_count=result.get("citation_count", 0)
            ),
            "citation_authenticity": _check_citation_authenticity(markdown, paragraph_by_id),
            "html_artifact_generated": _check_html_artifact_generated(result),
        }
        row.update(
            {
                "status": "ran",
                "passed": all(c["passed"] for c in checks.values()),
                "checks": checks,
                "citation_count": result.get("citation_count", 0),
                "brief_chars": len(markdown),
                "source_title": result.get("source_title", ""),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- real E2E case, surface any failure as a failed row
        row.update({"status": "error", "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        research_service.fetch_url_with_chain = original_fetch
        knowledge_tools.save_knowledge = original_save
        knowledge_tools.save_knowledge_set = original_save_set

    row["elapsed_seconds"] = round(time.time() - started, 2)
    if verbose:
        print(f"[{case['id']}] passed={row['passed']} elapsed={row['elapsed_seconds']}s")
        if row.get("error"):
            print(f"  error: {row['error']}")
        for name, check in row.get("checks", {}).items():
            if not check["passed"]:
                print(f"  FAIL {name}: {check['problems']}")
    return row


def _markdown_report(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    lines = [
        "# 单 URL 简报评估报告",
        "",
        f"用例总数：{total}，通过：{passed}，通过率：{passed / total:.0%}" if total else "无用例",
        "",
        "| id | 通过 | 引用真实性 | 内容实质性 | 直接交付 | 不自动落库 | HTML产物 | 引用数 | 耗时(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        checks = row.get("checks", {})

        def mark(name: str) -> str:
            if name not in checks:
                return "-"
            return "✅" if checks[name]["passed"] else "❌"

        lines.append(
            f"| {row['id']} | {'✅' if row['passed'] else '❌'} | {mark('citation_authenticity')} | "
            f"{mark('content_substantiveness')} | {mark('direct_delivery')} | {mark('no_auto_save')} | "
            f"{mark('html_artifact_generated')} | {row.get('citation_count', '-')} | {row.get('elapsed_seconds', '-')} |"
        )

    failed_rows = [r for r in rows if not r["passed"]]
    if failed_rows:
        lines += ["", "## 失败用例详情", ""]
        for row in failed_rows:
            lines.append(f"### {row['id']} ({row['url']})")
            if row.get("error"):
                lines.append(f"- 运行异常：{row['error']}")
            for name, check in row.get("checks", {}).items():
                if not check["passed"]:
                    for problem in check["problems"]:
                        lines.append(f"- [{name}] {problem}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=None, help="只跑指定 id 的用例，可重复传入")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    cases = _default_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases matched --only filter")
        return 1

    rows = [run_case(case, verbose=args.verbose) for case in cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "total_cases": len(rows),
        "passed_cases": sum(1 for r in rows if r["passed"]),
        "rows": rows,
    }
    (output_dir / f"single-url-brief-eval-{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"single-url-brief-eval-{stamp}.md").write_text(_markdown_report(rows), encoding="utf-8")

    print(f"\n{payload['passed_cases']}/{payload['total_cases']} passed. Reports written to {output_dir}")
    return 0 if payload["passed_cases"] == payload["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
