import { describe, expect, it } from 'vitest';

import { citationSourceLabel, cleanCitationSnippet, parseMarkdownDocument } from './citationDisplay';

describe('citation display helpers', () => {
  it('separates front matter, source links, and document body', () => {
    const parsed = parseMarkdownDocument(`---
title: Harness overview
sources: ['https://www.example.com/a', 'https://docs.example.org/b']
keywords: ['agent']
---
## 核心结论

Harness Engineering 强调约束与反馈。`);

    expect(parsed.sources).toEqual(['https://www.example.com/a', 'https://docs.example.org/b']);
    expect(parsed.body).toContain('## 核心结论');
    expect(parsed.body).not.toContain('sources:');
  });

  it('turns markdown snippets into compact readable text', () => {
    expect(cleanCitationSnippet('## 核心结论\n\n**Harness Engineering** 强调 `Guides` 与 Sensors。')).toBe(
      '核心结论 Harness Engineering 强调 Guides 与 Sensors。'
    );
  });

  it('hides truncated front matter instead of exposing raw urls', () => {
    const value = cleanCitationSnippet(
      "--- title: Overview topic: harness sources: ['https://example.com/a'] keywords: ['agent']"
    );

    expect(value).toBe('点击查看相关内容');
  });

  it('uses the source hostname as its compact label', () => {
    expect(citationSourceLabel('https://www.example.com/path')).toBe('example.com');
  });
});
