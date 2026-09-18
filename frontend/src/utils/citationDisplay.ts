const FRONTMATTER_BLOCK = /^---\s*\n([\s\S]*?)\n---\s*(?:\n|$)/;
const URL_PATTERN = /https?:\/\/[^\s'"\],)]+/g;
export const CITATION_SNIPPET_FALLBACK = '点击查看相关内容';

export interface ParsedMarkdownDocument {
  body: string;
  sources: string[];
}

export function parseMarkdownDocument(content: string): ParsedMarkdownDocument {
  const value = String(content || '').trim();
  const match = value.match(FRONTMATTER_BLOCK);
  const metadata = match?.[1] || '';
  const body = match ? value.slice(match[0].length).trim() : value;
  const sources = Array.from(new Set((metadata.match(URL_PATTERN) || []).map((url) => url.replace(/[.,;]+$/, ''))));
  return { body, sources };
}

export function citationSourceLabel(source: string): string {
  try {
    return new URL(source).hostname.replace(/^www\./, '');
  } catch {
    return source;
  }
}

export function cleanCitationSnippet(snippet: string): string {
  const parsed = parseMarkdownDocument(snippet);
  let value = parsed.body;

  if (value.startsWith('---')) {
    const headingIndex = value.search(/(?:^|\s)#{1,6}\s+/);
    value = headingIndex >= 0 ? value.slice(headingIndex) : '';
  }

  value = value
    .replace(URL_PATTERN, '')
    .replace(/^\s*(?:title|topic|sources|keywords)\s*:\s*.*$/gim, '')
    .replace(/^\s*[-|: ]{3,}\s*$/gm, '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[`*_>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  return value || CITATION_SNIPPET_FALLBACK;
}
