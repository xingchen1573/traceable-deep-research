"""Evidence-annotated reading HTML (VISION.md 第2节).

生产实现："固定模板 + 程序化渲染"——不在渲染时调用 LLM。视觉/交互直接照搬
`data/design_experiments/evidence_reader/round3_inline_annotated.html`（暖白学术风 CSS +
点击上标标记跳转/高亮引用卡片的 JS）。两个渲染入口共享页面外壳（`_HEAD_STYLE`/`_SCRIPT`）和
卡片交互，但正文解析/标注逻辑各自独立（两条链路引用体系本质不同，见下方说明）：

- `render_single_url_brief_html()` —— 服务 `single_url_brief` 快链路。`P#` 段落定位式引用，
  正文标注消费 `research_service._bind_narrative_sentences_to_citations()` 产出的真实数据
  （`sentence`/`start`/`end`/`locator`/`score`，见 docs/modules/single-url-brief-citation-binding-v0.3.md）。
  只有"核心论点与数据"区块做正文内联标注（annotations 本身也只对这个字段计算，摘要没有数据可用）；
  四段固定结构（摘要/核心论点与数据/关键原文引用/点评）与 markdown 版本保持一致，不新增/删减章节。
- `render_question_loop_report_html()` —— 服务 `question_loop` 深度调研主链路
  （docs/modules/question-loop-report-evidence-reader-html-v0.1.md）。正文跟随整合报告本身的
  动态章节结构（不套用单URL那套固定四段模板）。写作模型产出的 `[[ev_001]]` 标记在报告组装阶段
  已经被 `backend/research/writer.py::_render_internal_citations()` 转换成了 `[N]` 数字来源引用
  （按 `source_url` 编号，不是按单条证据），所以这里不解析 `ev_xxx`，而是复用同一个
  `_citation_map()` 重算编号、把 `[N]` 原地转可点击上标，卡片粒度是"单个来源"（可能含多条 quote）。
- `render_question_loop_documents_html()` —— 服务层级知识库 ResearchItem，把归档的多篇 Markdown
  按原顺序合成一份连续阅读版。每篇文档仍是独立章节，来源编号沿用完整 evidence map，未被任何
  文档引用的来源不会出现在完整阅读版中。
"""

from __future__ import annotations

import html
import re
from typing import Any

from ..research.writer import _citation_map, _is_reference_title

# 与 research_service.py::_ANNOTATION_FAILURE_NOTICE 措辞一致（有意不做跨模块 import，避免
# evidence_reader_service <-> research_service 的循环依赖；两处各自维护同一句提示文案）。
_ANNOTATION_FAILURE_NOTICE_HTML = "本次未生成正文引用标注。"

_HEAD_STYLE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_escaped}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700;900&family=Noto+Sans+SC:wght@300;400;500;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    --paper:#faf7f2;
    --paper-warm:#f4efe7;
    --ink:#1c1a17;
    --ink-soft:#4a463f;
    --ink-faint:#8a847a;
    --accent:#b3452e;
    --accent-deep:#8f3520;
    --accent-soft:rgba(179,69,46,.08);
    --gold:#a8842c;
    --rule:#ddd5c8;
    --rule-dark:#b8ae9b;
    --card:#fffdf9;
    --quote-bg:#faf6ec;
    --shadow:0 24px 60px -18px rgba(40,32,20,.28), 0 6px 18px -6px rgba(40,32,20,.14);
    --serif-cn:'Noto Serif SC', 'Songti SC', serif;
    --sans-cn:'Noto Sans SC', 'PingFang SC', sans-serif;
    --serif-en:'Source Serif 4', Georgia, serif;
    --mono:'JetBrains Mono', monospace;
  }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  html{{scroll-behavior:smooth}}
  body{{
    background:var(--paper);
    color:var(--ink);
    font-family:var(--serif-cn);
    font-size:17px;
    line-height:1.95;
    -webkit-font-smoothing:antialiased;
    text-rendering:optimizeLegibility;
  }}
  ::selection{{background:var(--accent);color:#fff}}

  #progress{{
    position:fixed;top:0;left:0;height:3px;width:0%;
    background:linear-gradient(90deg,var(--accent),var(--gold));
    z-index:200;transition:width .1s linear;
  }}

  header.topbar{{
    position:fixed;top:0;left:0;right:0;z-index:100;
    display:flex;align-items:center;justify-content:space-between;
    padding:14px 28px;
    background:rgba(250,247,242,.85);
    backdrop-filter:blur(12px);
    border-bottom:1px solid transparent;
    transition:border-color .3s;
  }}
  header.topbar.scrolled{{border-bottom-color:var(--rule)}}
  .brand{{font-family:var(--sans-cn);font-weight:700;font-size:13px;letter-spacing:.35em;color:var(--ink-soft)}}
  .brand em{{color:var(--accent);font-style:normal}}
  .topnav{{display:flex;gap:26px}}
  .topnav a{{
    font-family:var(--sans-cn);font-size:12.5px;letter-spacing:.14em;
    color:var(--ink-faint);text-decoration:none;position:relative;
    transition:color .25s;
  }}
  .topnav a::after{{
    content:'';position:absolute;left:0;bottom:-4px;width:0;height:1px;
    background:var(--accent);transition:width .3s;
  }}
  .topnav a:hover{{color:var(--accent)}}
  .topnav a:hover::after{{width:100%}}
  @media(max-width:720px){{.topnav{{display:none}}}}

  .hero{{
    max-width:880px;margin:0 auto;
    padding:150px 32px 70px;
    text-align:center;
  }}
  .kicker{{
    display:inline-flex;align-items:center;gap:14px;
    font-family:var(--sans-cn);font-size:12px;letter-spacing:.4em;
    color:var(--accent);text-transform:uppercase;margin-bottom:34px;
  }}
  .kicker::before,.kicker::after{{
    content:'';width:44px;height:1px;background:var(--accent);opacity:.5;
  }}
  h1.title{{
    font-weight:900;font-size:clamp(30px,5.2vw,52px);
    line-height:1.32;letter-spacing:.02em;
    margin-bottom:30px;
  }}
  .source-line{{
    font-family:var(--mono);font-size:13px;color:var(--ink-faint);
    display:inline-flex;align-items:center;gap:10px;
    padding:9px 20px;border:1px solid var(--rule);border-radius:999px;
    background:var(--card);transition:border-color .3s, box-shadow .3s;
  }}
  .source-line:hover{{border-color:var(--accent);box-shadow:0 4px 14px -6px rgba(179,69,46,.3)}}
  .source-line a{{color:var(--accent);text-decoration:none;word-break:break-all}}
  .source-line .dot{{width:6px;height:6px;border-radius:50%;background:var(--gold);flex:none}}
  .hero-meta{{
    margin-top:46px;display:flex;justify-content:center;gap:0;
    font-family:var(--sans-cn);font-size:12.5px;color:var(--ink-faint);letter-spacing:.1em;
  }}
  .hero-meta span{{padding:0 22px;border-right:1px solid var(--rule)}}
  .hero-meta span:last-child{{border-right:none}}
  .hero-meta b{{display:block;color:var(--ink);font-size:15px;font-weight:700;margin-bottom:2px;letter-spacing:.05em}}

  .layout{{
    max-width:1180px;margin:0 auto;padding:0 32px;
    display:grid;grid-template-columns:1fr;gap:0;
  }}
  @media(min-width:1000px){{
    .layout{{grid-template-columns:200px minmax(0,720px) 1fr}}
  }}
  aside.toc{{display:none}}
  @media(min-width:1000px){{
    aside.toc{{
      display:block;position:sticky;top:110px;align-self:start;
      padding-right:36px;
    }}
    .toc-label{{
      font-family:var(--sans-cn);font-size:11px;letter-spacing:.35em;
      color:var(--ink-faint);margin-bottom:18px;
    }}
    .toc a{{
      display:block;font-family:var(--sans-cn);font-size:13px;
      color:var(--ink-faint);text-decoration:none;
      padding:7px 0 7px 16px;border-left:2px solid var(--rule);
      transition:all .25s;line-height:1.5;
    }}
    .toc a .no{{color:var(--gold);font-family:var(--mono);font-size:11px;margin-right:8px}}
    .toc a:hover{{color:var(--ink)}}
    .toc a.active{{color:var(--accent);border-left-color:var(--accent);font-weight:500}}
  }}
  main{{min-width:0}}

  section.chapter{{margin-bottom:96px;scroll-margin-top:100px}}
  .chapter-head{{
    display:flex;align-items:baseline;gap:18px;
    margin-bottom:36px;padding-bottom:18px;
    border-bottom:1px solid var(--ink);
  }}
  .chapter-no{{
    font-family:var(--serif-en);font-style:italic;
    font-size:15px;color:var(--gold);
  }}
  .chapter-head h2{{
    font-size:clamp(22px,3vw,30px);font-weight:900;letter-spacing:.06em;
  }}
  .chapter-head .en{{
    margin-left:auto;font-family:var(--serif-en);font-style:italic;
    font-size:13px;color:var(--ink-faint);letter-spacing:.08em;
  }}
  @media(max-width:600px){{.chapter-head .en{{display:none}}}}

  .prose p{{
    margin-bottom:1.6em;text-align:justify;
    hanging-punctuation:first allow-end;
  }}
  .prose p:first-of-type::first-letter{{
    font-size:3.1em;font-weight:900;float:left;
    line-height:.86;padding:6px 12px 0 0;color:var(--accent);
  }}
  .no-dropcap p:first-of-type::first-letter{{
    font-size:inherit;font-weight:inherit;float:none;
    line-height:inherit;padding:0;color:inherit;
  }}
  .prose strong{{color:var(--accent-deep)}}
  .annotation-notice{{
    font-family:var(--sans-cn);font-size:13.5px;color:var(--ink-faint);
    background:var(--accent-soft);border-left:3px solid var(--accent);
    padding:12px 18px;margin-bottom:24px;line-height:1.8;border-radius:0 8px 8px 0;
  }}

  sup.cite{{
    line-height:0;
    vertical-align:super;
    margin:0 2px;
  }}
  .cite-ref{{
    font-family:var(--mono);
    font-size:10px;
    font-weight:500;
    letter-spacing:.02em;
    color:var(--accent);
    text-decoration:none;
    padding:1px 4px;
    border-radius:3px;
    transition:all .2s;
    cursor:pointer;
  }}
  .cite-ref:hover{{
    background:var(--accent);
    color:var(--paper);
  }}
  .cite-ref.active{{
    background:var(--accent);
    color:var(--paper);
  }}

  .sidenote.flash{{
    background:rgba(179,69,46,.05);
  }}
  .sidenote.flash .sn-loc{{
    background:var(--accent);
    color:var(--paper);
    border-color:var(--accent);
    box-shadow:0 0 0 4px rgba(179,69,46,.15), 0 4px 12px -3px rgba(179,69,46,.4);
  }}
  .sidenote.flash .sn-quote{{
    border-left-color:var(--accent);
    box-shadow:0 0 0 1px rgba(179,69,46,.25), 0 8px 24px -10px rgba(179,69,46,.25);
  }}

  .evidence-intro{{
    font-family:var(--sans-cn);font-size:13.5px;color:var(--ink-faint);
    background:var(--accent-soft);border-left:3px solid var(--accent);
    padding:14px 20px;margin-bottom:34px;line-height:1.9;border-radius:0 8px 8px 0;
  }}
  .evidence-intro b{{color:var(--accent-deep)}}

  .sidenote-list{{
    display:flex;flex-direction:column;gap:0;
    border-top:1px solid var(--rule-dark);
  }}
  .sidenote{{
    display:grid;
    grid-template-columns:64px minmax(0,1fr);
    gap:0 28px;
    padding:30px 0 32px;
    border-bottom:1px solid var(--rule);
    position:relative;
    transition:background .35s;
  }}
  .sidenote:hover{{background:rgba(255,253,249,.6)}}

  .sn-aside{{
    display:flex;flex-direction:column;align-items:center;
    padding-top:4px;position:relative;
  }}
  .sn-aside::after{{
    content:'';position:absolute;top:64px;bottom:-32px;left:50%;
    width:1px;background:linear-gradient(180deg,var(--rule) 0%,transparent 100%);
  }}
  .sidenote:last-child .sn-aside::after{{display:none}}
  .sn-num{{
    font-family:var(--serif-en);font-style:italic;
    font-size:12px;color:var(--ink-faint);letter-spacing:.04em;
    margin-bottom:8px;
  }}
  .sn-loc{{
    font-family:var(--mono);font-size:12px;font-weight:500;
    color:var(--accent);background:var(--card);
    border:1px solid rgba(179,69,46,.4);
    border-radius:3px;
    padding:4px 0;width:44px;text-align:center;
    letter-spacing:.04em;
    box-shadow:0 2px 6px -2px rgba(40,32,20,.12);
    transition:all .25s;
  }}
  .sidenote:hover .sn-loc{{
    background:var(--accent);color:var(--paper);border-color:var(--accent);
    box-shadow:0 4px 12px -3px rgba(179,69,46,.4);
  }}

  .sn-body{{min-width:0}}
  .sn-quote{{
    font-family:var(--serif-en);
    font-size:14.5px;line-height:1.82;
    color:#3a352d;
    background:var(--quote-bg);
    border-left:2px solid var(--rule-dark);
    border-radius:0 10px 10px 0;
    padding:18px 22px;
    margin-bottom:14px;
    white-space:pre-wrap;
    overflow-wrap:break-word;
    position:relative;
    transition:border-color .3s, box-shadow .3s;
  }}
  .sidenote:hover .sn-quote{{
    border-left-color:var(--gold);
    box-shadow:0 6px 20px -10px rgba(40,32,20,.18);
  }}
  .sn-quote::before{{
    content:'\\201c';
    position:absolute;top:-14px;left:10px;
    font-family:var(--serif-en);font-size:34px;line-height:1;
    color:var(--gold);opacity:.5;
    background:var(--paper);padding:0 6px;
  }}
  .sn-quote.collapsed{{
    max-height:calc(14.5px * 1.82 * 4 + 36px);
    overflow:hidden;
    cursor:pointer;
    -webkit-mask-image:linear-gradient(180deg,#000 58%,transparent 97%);
            mask-image:linear-gradient(180deg,#000 58%,transparent 97%);
  }}
  .sn-quote.collapsed:hover{{border-left-color:var(--accent)}}
  .sn-meta-row{{
    display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  }}
  .sn-toggle{{
    display:none;
    font-family:var(--sans-cn);font-size:11.5px;letter-spacing:.14em;
    color:var(--accent);background:none;border:none;cursor:pointer;
    padding:2px 0;border-bottom:1px dotted var(--accent);
    transition:color .2s;
  }}
  .sn-toggle:hover{{color:var(--accent-deep)}}
  .quote-block.long .sn-toggle{{display:inline-block}}
  .copy-btn{{
    font-family:var(--sans-cn);font-size:11.5px;letter-spacing:.1em;
    color:var(--ink-faint);background:transparent;
    border:1px solid var(--rule);border-radius:999px;
    padding:4px 14px;cursor:pointer;transition:all .25s;
  }}
  .copy-btn:hover{{color:var(--accent);border-color:var(--accent)}}
  .copy-btn.done{{color:#3d7a4a;border-color:#3d7a4a}}
  .sn-reason{{
    margin-top:12px;
    font-family:var(--sans-cn);font-size:13.5px;line-height:1.85;
    color:var(--ink-soft);
    padding-left:16px;position:relative;
  }}
  .sn-reason::before{{
    content:'';position:absolute;left:0;top:.68em;
    width:5px;height:5px;background:var(--accent);border-radius:50%;
    opacity:.8;
  }}
  .sn-reason .rl{{
    font-family:var(--serif-en);font-style:italic;font-size:11.5px;
    color:var(--gold);letter-spacing:.08em;
    display:block;margin-bottom:2px;
  }}
  @media(max-width:640px){{
    .sidenote{{grid-template-columns:1fr;gap:12px;padding:24px 0 26px}}
    .sn-aside{{flex-direction:row;align-items:center;gap:12px;padding-top:0}}
    .sn-aside::after{{display:none}}
    .sn-num{{margin-bottom:0}}
    .sn-loc{{width:auto;padding:3px 12px}}
  }}

  .review-grid{{display:flex;flex-direction:column;gap:0;border-top:1px solid var(--rule)}}
  .review-row{{
    display:grid;grid-template-columns:170px 1fr;gap:28px;
    padding:26px 4px;border-bottom:1px solid var(--rule);
    transition:background .3s;
  }}
  .review-row:hover{{background:rgba(255,253,249,.7)}}
  .review-label{{
    font-family:var(--sans-cn);font-weight:700;font-size:14px;
    letter-spacing:.12em;display:flex;align-items:flex-start;gap:10px;
    padding-top:3px;
  }}
  .review-label .ico{{
    width:26px;height:26px;flex:none;border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-size:13px;font-family:var(--serif-en);font-style:italic;
  }}
  .r-support .ico{{background:rgba(61,122,74,.12);color:#3d7a4a}}
  .r-against .ico{{background:rgba(179,69,46,.1);color:var(--accent)}}
  .r-limit   .ico{{background:rgba(168,132,44,.14);color:var(--gold)}}
  .r-scene   .ico{{background:rgba(70,96,150,.12);color:#466096}}
  .r-trust   .ico{{background:rgba(28,26,23,.08);color:var(--ink)}}
  .review-body{{font-size:15.5px;line-height:1.9;color:var(--ink-soft);text-align:justify}}
  @media(max-width:640px){{
    .review-row{{grid-template-columns:1fr;gap:10px}}
  }}

  footer{{
    margin-top:60px;border-top:1px solid var(--ink);
    padding:44px 32px 60px;text-align:center;
  }}
  footer .fin{{
    font-family:var(--serif-en);font-style:italic;font-size:15px;
    color:var(--ink-faint);letter-spacing:.2em;margin-bottom:12px;
  }}
  footer .note{{font-family:var(--sans-cn);font-size:12px;color:var(--ink-faint);letter-spacing:.08em}}

  #toTop{{
    position:fixed;right:26px;bottom:26px;z-index:150;
    width:44px;height:44px;border-radius:50%;
    background:var(--ink);color:var(--paper);border:none;cursor:pointer;
    font-size:18px;opacity:0;pointer-events:none;
    transition:opacity .3s, transform .3s;transform:translateY(10px);
    box-shadow:0 8px 20px -6px rgba(0,0,0,.35);
  }}
  #toTop.show{{opacity:1;pointer-events:auto;transform:translateY(0)}}
  #toTop:hover{{background:var(--accent)}}

  /* Report content must never depend on a decorative observer to become readable. */
  .reveal{{opacity:1;transform:none;transition:opacity .7s ease, transform .7s cubic-bezier(.2,.8,.2,1)}}
  .reveal.in{{opacity:1;transform:none}}

  @media print{{
    #progress,#toTop,.topbar,.sn-toggle,.copy-btn{{display:none!important}}
    .sn-quote.collapsed{{max-height:none;-webkit-mask-image:none;mask-image:none}}
    .reveal{{opacity:1;transform:none}}
    body{{background:#fff}}
  }}
</style>
</head>
"""

_SCRIPT = """<script>
(function(){
  'use strict';

  var progress = document.getElementById('progress');
  var topbar = document.getElementById('topbar');
  var toTop = document.getElementById('toTop');

  function onScroll(){
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var pct = max > 0 ? (h.scrollTop / max) * 100 : 0;
    progress.style.width = pct + '%';
    topbar.classList.toggle('scrolled', h.scrollTop > 40);
    toTop.classList.toggle('show', h.scrollTop > 600);
  }
  window.addEventListener('scroll', onScroll, {passive:true});
  onScroll();

  toTop.addEventListener('click', function(){
    window.scrollTo({top:0, behavior:'smooth'});
  });

  var revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window){
    var ro = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if (e.isIntersecting){ e.target.classList.add('in'); ro.unobserve(e.target); }
      });
    }, {threshold:.08});
    revealEls.forEach(function(el){ ro.observe(el); });
  } else {
    revealEls.forEach(function(el){ el.classList.add('in'); });
  }

  var tocLinks = document.querySelectorAll('#tocNav a');
  // Derive the observed sections from the actual toc links rendered (not a hardcoded id list) --
  // single_url_brief always renders the same 4 anchors, question_loop's chapters are dynamic per
  // report, so this needs to work for both without knowing the id set ahead of time.
  var sections = Array.prototype.map.call(tocLinks, function(a){
    var id = (a.getAttribute('href') || '').replace('#', '');
    return id ? document.getElementById(id) : null;
  });
  if ('IntersectionObserver' in window){
    var so = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if (e.isIntersecting){
          tocLinks.forEach(function(a){
            a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id);
          });
        }
      });
    }, {rootMargin:'-30% 0px -55% 0px'});
    sections.forEach(function(s){ if(s) so.observe(s); });
  }

  // Operates on '.quote-block' rather than '.sidenote' -- a single_url_brief sidenote always has
  // exactly one quote-block (no behavior change there), but a question_loop source card can have
  // several quotes stacked in one sidenote, each with its own collapse/copy affordance.
  document.querySelectorAll('.quote-block').forEach(function(qb){
    var q = qb.querySelector('[data-quote]');
    var btn = qb.querySelector('.sn-toggle');
    if (!q || !btn) return;
    var lineH = parseFloat(getComputedStyle(q).lineHeight) || 26;
    if (q.scrollHeight > lineH * 4 + 60){
      qb.classList.add('long');
      q.classList.add('collapsed');
      btn.textContent = '展开全文 ↓';
      var toggle = function(){
        var collapsed = q.classList.toggle('collapsed');
        btn.textContent = collapsed ? '展开全文 ↓' : '收起 ↑';
      };
      btn.addEventListener('click', function(e){ e.stopPropagation(); toggle(); });
      q.addEventListener('click', function(){
        if (q.classList.contains('collapsed')) toggle();
      });
    }
  });

  document.querySelectorAll('.copy-btn').forEach(function(btn){
    btn.addEventListener('click', function(e){
      e.stopPropagation();
      var text = btn.closest('.quote-block').querySelector('[data-quote]').textContent;
      function done(){
        btn.textContent = '✓ 已复制';
        btn.classList.add('done');
        setTimeout(function(){
          btn.textContent = '复制原文';
          btn.classList.remove('done');
        }, 1600);
      }
      function fallback(){
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch(err){}
        document.body.removeChild(ta);
      }
      if (navigator.clipboard && navigator.clipboard.writeText){
        navigator.clipboard.writeText(text).then(done).catch(fallback);
      } else { fallback(); }
    });
  });

  var flashTimer = null;
  document.querySelectorAll('.cite-ref').forEach(function(link){
    link.addEventListener('click', function(e){
      e.preventDefault();
      var id = link.getAttribute('data-note');
      var target = document.getElementById(id);
      if (!target) return;

      var q = target.querySelector('[data-quote]');
      var btn = target.querySelector('.sn-toggle');
      if (q && q.classList.contains('collapsed')){
        q.classList.remove('collapsed');
        if (btn) btn.textContent = '收起 ↑';
      }

      var y = target.getBoundingClientRect().top + window.pageYOffset - 120;
      window.scrollTo({top: y, behavior: 'smooth'});

      if (flashTimer) clearTimeout(flashTimer);
      document.querySelectorAll('.sidenote.flash').forEach(function(el){
        el.classList.remove('flash');
      });
      document.querySelectorAll('.cite-ref.active').forEach(function(el){
        el.classList.remove('active');
      });
      setTimeout(function(){
        target.classList.add('flash');
        link.classList.add('active');
        flashTimer = setTimeout(function(){
          target.classList.remove('flash');
          link.classList.remove('active');
        }, 2400);
      }, 350);
    });
  });
})();
</script>
"""


def _resolve_citations(*, citations: list[dict[str, Any]], paragraph_by_id: dict[str, str]) -> list[dict[str, str]]:
    """Same resolution rule as _render_single_url_brief_markdown: never fabricate a quote, only
    keep citations whose locator resolves to a real paragraph. Preserves input order/duplicates
    so citation_count here matches the markdown renderer's citation_count exactly."""
    resolved: list[dict[str, str]] = []
    for item in citations or []:
        if not isinstance(item, dict):
            continue
        locator = str(item.get("locator") or "").strip()
        paragraph_text = paragraph_by_id.get(locator)
        if not paragraph_text:
            continue
        resolved.append({"locator": locator, "text": paragraph_text, "reason": str(item.get("reason") or "").strip()})
    return resolved


def _annotate_narrative_html(narrative: str, annotations: list[dict[str, Any]], locator_to_note_id: dict[str, str]) -> str:
    """Escape `narrative` and splice a `<sup class="cite">` jump marker right after each annotated
    sentence's end offset. Text between/after annotated spans is emitted untouched (still escaped)
    so the full narrative is always reproduced verbatim, annotated or not (design doc 3.3: below
    threshold sentences are left unannotated, not force-bound)."""
    ordered = sorted(
        (a for a in annotations if isinstance(a, dict)),
        key=lambda a: (a.get("start") or 0, a.get("end") or 0),
    )
    pieces: list[str] = []
    cursor = 0
    length = len(narrative)
    for ann in ordered:
        start = int(ann.get("start") or 0)
        end = int(ann.get("end") or 0)
        if not (0 <= start <= end <= length) or end < cursor:
            continue
        note_id = locator_to_note_id.get(str(ann.get("locator") or "").strip())
        pieces.append(html.escape(narrative[cursor:end]))
        if note_id:
            locator_label = html.escape(str(ann.get("locator") or ""))
            pieces.append(
                f'<sup class="cite"><a class="cite-ref" href="#{note_id}" data-note="{note_id}">{locator_label}</a></sup>'
            )
        cursor = end
    pieces.append(html.escape(narrative[cursor:]))
    return "".join(pieces)


def _render_sidenotes(resolved_citations: list[dict[str, str]]) -> str:
    if not resolved_citations:
        return '<p class="review-body">（原文未提供可精确定位的引用片段。）</p>'
    articles = []
    for idx, item in enumerate(resolved_citations, start=1):
        note_id = f"note-{idx}"
        quote = html.escape(item["text"])
        locator = html.escape(item["locator"])
        reason_html = ""
        if item["reason"]:
            reason_html = (
                f'<div class="sn-reason"><span class="rl">选取理由 · Why this passage</span>'
                f"{html.escape(item['reason'])}</div>"
            )
        articles.append(
            f"""
        <article class="sidenote" id="{note_id}">
          <div class="sn-aside">
            <span class="sn-num">注 {idx}</span>
            <span class="sn-loc">{locator}</span>
          </div>
          <div class="sn-body">
            <div class="quote-block">
              <div class="sn-quote" data-quote>{quote}</div>
              <div class="sn-meta-row">
                <button class="sn-toggle" type="button">展开全文 ↓</button>
                <button class="copy-btn" type="button">复制原文</button>
              </div>
            </div>
            {reason_html}
          </div>
        </article>"""
        )
    return '<div class="sidenote-list" id="sidenoteList">' + "".join(articles) + "</div>"


def render_single_url_brief_html(
    *,
    title: str,
    url: str,
    synthesis: dict[str, Any],
    paragraphs: list[dict[str, str]],
    annotations: list[dict[str, Any]] | None,
    annotation_diagnostics: dict[str, Any] | None,
) -> str:
    """Render the production evidence-annotated reading HTML for a single_url_brief job.

    Pure function, no I/O, no LLM calls -- fixed template + programmatic rendering (VISION.md
    第2节既定架构). Caller (research_service._run_single_url_research) is responsible for writing
    the returned string to disk.
    """
    paragraph_by_id = {item["id"]: item["text"] for item in paragraphs}
    resolved_citations = _resolve_citations(citations=synthesis.get("citations") or [], paragraph_by_id=paragraph_by_id)
    locator_to_note_id: dict[str, str] = {}
    for idx, item in enumerate(resolved_citations, start=1):
        locator_to_note_id.setdefault(item["locator"], f"note-{idx}")

    summary_html = html.escape(str(synthesis.get("summary") or "").strip() or "（模型未返回摘要）")
    narrative = str(synthesis.get("key_points_narrative") or "").strip()
    narrative_html = (
        _annotate_narrative_html(narrative, annotations or [], locator_to_note_id)
        if narrative
        else "（模型未返回核心论点与数据）"
    )
    annotation_notice_html = ""
    if annotation_diagnostics is not None:
        annotation_notice_html = f'<div class="annotation-notice">{html.escape(_ANNOTATION_FAILURE_NOTICE_HTML)}</div>'

    commentary = synthesis.get("commentary") if isinstance(synthesis.get("commentary"), dict) else {}

    def _c(key: str, default: str) -> str:
        return html.escape(str(commentary.get(key) or default))

    title_escaped = html.escape(title)
    url_escaped = html.escape(url)

    body = f"""<body>

<div id="progress"></div>

<header class="topbar" id="topbar">
  <div class="brand">LINK<em>·</em>BRIEF <em>/</em> 链接简报</div>
  <nav class="topnav">
    <a href="#abstract">摘要</a>
    <a href="#arguments">核心论点</a>
    <a href="#evidence">原文核查</a>
    <a href="#review">点评</a>
  </nav>
</header>

<section class="hero">
  <div class="kicker">Research Brief · 调研简报</div>
  <h1 class="title">{title_escaped}</h1>
  <div class="source-line">
    <span class="dot"></span>
    来源&nbsp;<a href="{url_escaped}" target="_blank" rel="noopener">{url_escaped}</a>
  </div>
  <div class="hero-meta">
    <span><b>{len(resolved_citations)} 条</b>原文核查引用</span>
  </div>
</section>

<div class="layout">
  <aside class="toc">
    <div class="toc-label">目录 CONTENTS</div>
    <nav class="toc" id="tocNav">
      <a href="#abstract" class="active"><span class="no">01</span>摘要</a>
      <a href="#arguments"><span class="no">02</span>核心论点与数据</a>
      <a href="#evidence"><span class="no">03</span>关键原文引用</a>
      <a href="#review"><span class="no">04</span>点评</a>
    </nav>
  </aside>

  <main>
    <section class="chapter reveal" id="abstract">
      <div class="chapter-head">
        <span class="chapter-no">No. 01</span>
        <h2>摘要</h2>
        <span class="en">Abstract</span>
      </div>
      <div class="prose">
        <p>{summary_html}</p>
      </div>
    </section>

    <section class="chapter reveal" id="arguments">
      <div class="chapter-head">
        <span class="chapter-no">No. 02</span>
        <h2>核心论点与数据</h2>
        <span class="en">Key Arguments &amp; Data</span>
      </div>
      {annotation_notice_html}
      <div class="prose no-dropcap">
        <p>{narrative_html}</p>
      </div>
    </section>

    <section class="chapter reveal" id="evidence">
      <div class="chapter-head">
        <span class="chapter-no">No. 03</span>
        <h2>关键原文引用</h2>
        <span class="en">Verifiable Citations · Marginalia</span>
      </div>

      <div class="evidence-intro">
        <b>核查指南：</b>以下共 <b>{len(resolved_citations)}</b> 条逐字摘录，每条由「段落定位符 + 原文 + 选取理由」构成，可对照上文论点逐条核验，确认转述无夸大、无编造。较长引文默认折叠，点击即可展开全文。
      </div>

      {_render_sidenotes(resolved_citations)}
    </section>

    <section class="chapter reveal" id="review">
      <div class="chapter-head">
        <span class="chapter-no">No. 04</span>
        <h2>点评</h2>
        <span class="en">Editorial Review</span>
      </div>

      <div class="review-grid">
        <div class="review-row r-support">
          <div class="review-label"><span class="ico">+</span>支持证据</div>
          <div class="review-body">{_c("supporting_evidence", "未提供")}</div>
        </div>
        <div class="review-row r-against">
          <div class="review-label"><span class="ico">−</span>反驳/制约证据</div>
          <div class="review-body">{_c("counter_evidence", "原文未提供明显的反面信息")}</div>
        </div>
        <div class="review-row r-limit">
          <div class="review-label"><span class="ico">!</span>局限性</div>
          <div class="review-body">{_c("limitations", "未提供")}</div>
        </div>
        <div class="review-row r-scene">
          <div class="review-label"><span class="ico">◎</span>适用场景</div>
          <div class="review-body">{_c("applicable_scenarios", "未提供")}</div>
        </div>
        <div class="review-row r-trust">
          <div class="review-label"><span class="ico">★</span>整体可信度</div>
          <div class="review-body">{_c("overall_credibility", "未提供")}</div>
        </div>
      </div>
    </section>
  </main>
  <div></div>
</div>

<footer>
  <div class="fin">— Fin —</div>
  <div class="note">链接简报 · 调研 Pipeline 自动输出 · 原文引用均逐字摘录，未作改写</div>
</footer>

<button id="toTop" aria-label="回到顶部">↑</button>

{_SCRIPT}
</body>
</html>
"""

    return _HEAD_STYLE.format(title_escaped=title_escaped) + body


# ---------------------------------------------------------------------------
# question_loop 深度调研报告渲染
# 见 docs/modules/question-loop-report-evidence-reader-html-v0.1.md
# ---------------------------------------------------------------------------

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def _group_evidence_by_source(evidence_map: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Group `evidence_map` rows by the source-level citation number `writer.py`'s own
    `_citation_map()` assigns. One card per source (not per evidence row) -- by the time the
    report reaches `approve_research()`, the delivered text only carries `[N]` source references,
    not per-evidence `ev_xxx` ids (module docstring / design doc 1.1 节). Reusing the same
    `_citation_map()` here guarantees the numbering matches what's already embedded in the text.
    """
    numbering = _citation_map(evidence_map)  # {source_url: number}, first-seen order
    groups: dict[int, dict[str, Any]] = {}
    for item in evidence_map:
        if not isinstance(item, dict):
            continue
        url = str(item.get("source_url") or "").strip()
        number = numbering.get(url)
        if not number:
            continue
        group = groups.setdefault(
            number,
            {"source_title": str(item.get("source_title") or url or f"来源 {number}"), "source_url": url, "quotes": []},
        )
        quote = str(item.get("quote") or item.get("cleaned_quote") or item.get("evidence_text") or "").strip()
        if quote and quote not in group["quotes"]:
            group["quotes"].append(quote)
    return groups


def build_question_loop_citation_sources(evidence_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the durable citation map stored with a hierarchical ResearchItem.

    The source number uses the writer's original full evidence map, so per-document filtering
    later never renumbers a marker such as ``[7]``. Evidence rows retain their document usage
    metadata so the knowledge reader can show the most relevant quotes for the selected file.
    """

    numbering = _citation_map(evidence_map)
    groups: dict[int, dict[str, Any]] = {}
    seen_evidence: dict[int, set[tuple[str, str]]] = {}
    for item in evidence_map:
        if not isinstance(item, dict):
            continue
        url = str(item.get("source_url") or "").strip()
        number = numbering.get(url)
        if not number:
            continue
        group = groups.setdefault(
            number,
            {
                "number": number,
                "title": str(item.get("source_title") or url or f"来源 {number}"),
                "url": url,
                "evidence": [],
            },
        )
        evidence_id = str(item.get("id") or item.get("evidence_id") or "").strip()
        quote = str(
            item.get("raw_quote")
            or item.get("cleaned_quote")
            or item.get("quote")
            or item.get("evidence_text")
            or ""
        ).strip()
        key = (evidence_id, quote)
        bucket = seen_evidence.setdefault(number, set())
        if key in bucket or (not evidence_id and not quote):
            continue
        bucket.add(key)
        group["evidence"].append(
            {
                "evidence_id": evidence_id,
                "quote": quote,
                "claim": str(item.get("atomic_claim") or item.get("claim") or "").strip(),
                "used_by_documents": [
                    str(value).strip()
                    for value in item.get("used_by_documents") or []
                    if str(value).strip()
                ],
            }
        )
    return [groups[number] for number in sorted(groups)]


def _citation_numbers_in_text(content: str) -> set[int]:
    return {int(value) for value in _CITATION_MARKER_RE.findall(str(content or ""))}


def _is_reference_chapter(heading: str) -> bool:
    """`_assemble_report`'s default reference heading is the bare "来源" (writer.py:977), which
    doesn't match `_is_reference_title`'s substring check (that helper is meant for per-document
    titles, not this default trailing heading). Treat both as "skip -- superseded by source
    cards rendered from evidence_map directly"."""
    value = str(heading or "").strip()
    return value == "来源" or _is_reference_title(value)


def _parse_report_chapters(content: str) -> list[dict[str, Any]]:
    """Parse the constrained markdown subset `writer.py`'s report assembler emits: a leading
    '# title' (redundant with the `title` param already shown in the hero section, skipped),
    '## heading' chapter breaks, blank-line-separated paragraphs, and '> notice' blockquotes
    (the partial-report disclosure -- see `_assemble_report`). This is not a general markdown
    parser: if `report_blueprint` ever emits lists/tables/bold, those render as plain paragraph
    text -- no crash, just lost formatting (known limitation, design doc 第5节).

    Drops the trailing references chapter entirely -- its information is superseded by the
    source cards this renderer builds straight from `evidence_map`.
    """
    chapters: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        text = " ".join(line.strip() for line in paragraph_lines if line.strip())
        paragraph_lines = []
        if text and current is not None:
            current["blocks"].append({"type": "p", "text": text})

    for raw_line in str(content or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:].strip()
            if _is_reference_chapter(heading):
                current = None  # trailing "来源" chapter: stop collecting, drop the rest
                continue
            current = {"heading": heading or "章节", "blocks": []}
            chapters.append(current)
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            continue  # redundant h1 report title, already shown in the hero section
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            if current is not None:
                current["blocks"].append({"type": "notice", "text": stripped[2:].strip()})
            continue
        paragraph_lines.append(line)
    flush_paragraph()
    return chapters


def _annotate_citations_html(text: str, note_ids: set[int]) -> str:
    """Escape `text` and turn each `[N]` marker into a clickable jump-to-source-card superscript,
    provided N resolves to a real source card (`note_ids`). An unresolvable number is left as
    plain escaped text (decision 9: don't fabricate a card, don't let one bad marker break the
    whole render either)."""
    escaped = html.escape(text)

    def _sub(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in note_ids:
            return match.group(0)
        note_id = f"note-{number}"
        return f'<sup class="cite"><a class="cite-ref" href="#{note_id}" data-note="{note_id}">{number}</a></sup>'

    return _CITATION_MARKER_RE.sub(_sub, escaped)


def _render_report_sidenotes(groups: dict[int, dict[str, Any]]) -> str:
    if not groups:
        return '<p class="review-body">（本次报告未产生可核查的来源引用。）</p>'
    articles = []
    for number in sorted(groups):
        group = groups[number]
        note_id = f"note-{number}"
        source_title = html.escape(str(group.get("source_title") or ""))
        source_url = html.escape(str(group.get("source_url") or ""))
        quotes = [q for q in group.get("quotes") or [] if q]
        if quotes:
            quote_blocks = "".join(
                f"""
            <div class="quote-block">
              <div class="sn-quote" data-quote>{html.escape(quote)}</div>
              <div class="sn-meta-row">
                <button class="sn-toggle" type="button">展开全文 ↓</button>
                <button class="copy-btn" type="button">复制原文</button>
              </div>
            </div>"""
                for quote in quotes
            )
        else:
            quote_blocks = '<p class="review-body">（该来源未提供可精确定位的引用片段。）</p>'
        source_link = ""
        if source_url:
            label = source_title or source_url
            source_link = (
                f'<div class="sn-reason"><span class="rl">来源 · Source</span>'
                f'<a href="{source_url}" target="_blank" rel="noopener">{label}</a></div>'
            )
        articles.append(
            f"""
        <article class="sidenote" id="{note_id}">
          <div class="sn-aside">
            <span class="sn-num">注 {number}</span>
            <span class="sn-loc">[{number}]</span>
          </div>
          <div class="sn-body">
            {quote_blocks}
            {source_link}
          </div>
        </article>"""
        )
    return '<div class="sidenote-list" id="sidenoteList">' + "".join(articles) + "</div>"


def _toc_link(chapter_id: str, index: int, heading: str, *, active: bool) -> str:
    active_cls = ' class="active"' if active else ""
    return f'<a href="#{chapter_id}"{active_cls}><span class="no">{index:02d}</span>{html.escape(heading)}</a>'


def render_question_loop_report_html(
    *,
    title: str,
    report_content: str,
    evidence_map: list[dict[str, Any]],
    citation_numbers: set[int] | None = None,
) -> str:
    """Render the evidence-annotated reading HTML for a question_loop job's final integrated
    report. Pure function, no I/O, no LLM calls (same architecture as
    `render_single_url_brief_html`). Caller (`research_service.approve_research`) is responsible
    for writing the returned string to disk and for isolating render failures from the approval
    flow (design doc 决策10 -- this function is allowed to raise on malformed input, the caller
    catches it).
    """
    chapters = _parse_report_chapters(report_content)
    groups = _group_evidence_by_source(evidence_map)
    if citation_numbers is not None:
        groups = {number: group for number, group in groups.items() if number in citation_numbers}
    note_ids = set(groups.keys())
    title_escaped = html.escape(title)

    chapter_ids = [f"chapter-{idx}" for idx in range(1, len(chapters) + 1)]
    sources_id = f"chapter-{len(chapters) + 1}"

    toc_items = "".join(
        _toc_link(cid, idx + 1, chapter["heading"], active=(idx == 0))
        for idx, (cid, chapter) in enumerate(zip(chapter_ids, chapters))
    )
    toc_items += _toc_link(sources_id, len(chapters) + 1, "参考来源", active=(not chapters))

    chapter_sections = ""
    for idx, (cid, chapter) in enumerate(zip(chapter_ids, chapters), start=1):
        blocks_html = []
        for block in chapter["blocks"]:
            if block["type"] == "notice":
                blocks_html.append(f'<div class="annotation-notice">{html.escape(block["text"])}</div>')
            else:
                blocks_html.append(f'<p>{_annotate_citations_html(block["text"], note_ids)}</p>')
        body_html = "".join(blocks_html) or "<p>（本章节暂无正文内容。）</p>"
        chapter_sections += f"""
    <section class="chapter reveal" id="{cid}">
      <div class="chapter-head">
        <span class="chapter-no">No. {idx:02d}</span>
        <h2>{html.escape(chapter["heading"])}</h2>
      </div>
      <div class="prose no-dropcap">
        {body_html}
      </div>
    </section>"""

    chapter_sections += f"""
    <section class="chapter reveal" id="{sources_id}">
      <div class="chapter-head">
        <span class="chapter-no">No. {len(chapters) + 1:02d}</span>
        <h2>参考来源</h2>
        <span class="en">Sources · Marginalia</span>
      </div>
      <div class="evidence-intro">
        <b>核查指南：</b>正文中的 <b>[数字]</b> 标记对应下方来源，点击即可跳转核对；共 <b>{len(groups)}</b> 个来源。
      </div>
      {_render_report_sidenotes(groups)}
    </section>"""

    body = f"""<body>

<div id="progress"></div>

<header class="topbar" id="topbar">
  <div class="brand">DEEP<em>·</em>RESEARCH <em>/</em> 深度调研</div>
</header>

<section class="hero">
  <div class="kicker">Deep Research · 深度调研报告</div>
  <h1 class="title">{title_escaped}</h1>
  <div class="hero-meta">
    <span><b>{len(groups)} 个</b>可核查来源</span>
  </div>
</section>

<div class="layout">
  <aside class="toc">
    <div class="toc-label">目录 CONTENTS</div>
    <nav class="toc" id="tocNav">
      {toc_items}
    </nav>
  </aside>

  <main>
    {chapter_sections}
  </main>
  <div></div>
</div>

<footer>
  <div class="fin">— Fin —</div>
  <div class="note">深度调研 · 调研 Pipeline 自动输出 · 正文引用编号对应下方"参考来源"列表</div>
</footer>

<button id="toTop" aria-label="回到顶部">↑</button>

{_SCRIPT}
</body>
</html>
"""

    return _HEAD_STYLE.format(title_escaped=title_escaped) + body


def render_question_loop_document_html(
    *,
    title: str,
    document_content: str,
    evidence_map: list[dict[str, Any]],
) -> str:
    """Render one archived Markdown document as a standalone evidence-reader HTML file."""

    report_content = f"## {str(title or '文档').strip()}\n\n{str(document_content or '').strip()}\n"
    return render_question_loop_report_html(
        title=title,
        report_content=report_content,
        evidence_map=evidence_map,
        citation_numbers=_citation_numbers_in_text(document_content),
    )


def render_question_loop_documents_html(
    *,
    title: str,
    documents: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
) -> str:
    """Render all Markdown documents in one ResearchItem as one continuous HTML report."""

    report_parts: list[str] = []
    citation_numbers: set[int] = set()
    for index, raw in enumerate(documents, start=1):
        if not isinstance(raw, dict):
            continue
        document_title = str(raw.get("title") or raw.get("filename") or f"章节 {index}").strip()
        document_content = str(raw.get("content") or "").strip()
        if not document_content:
            continue
        report_parts.append(f"## {document_title}\n\n{document_content}")
        citation_numbers.update(_citation_numbers_in_text(document_content))

    if not report_parts:
        raise ValueError("ResearchItem has no readable documents")
    return render_question_loop_report_html(
        title=title,
        report_content="\n\n".join(report_parts),
        evidence_map=evidence_map,
        citation_numbers=citation_numbers,
    )
