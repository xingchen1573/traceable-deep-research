import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/utils/cn';

interface MarkdownPreviewProps {
  content: string;
  className?: string;
  evidenceIds?: string[];
  activeEvidenceId?: string;
  onEvidenceClick?: (evidenceId: string) => void;
  citationNumbers?: number[];
  activeCitationNumber?: number;
  onCitationClick?: (citationNumber: number) => void;
}

const EVIDENCE_REF_PATTERN = /(\[ev_\d{3,}\])/g;
const NUMBERED_CITATION_PATTERN = /\[(\d+)\](?!\()/g;

function evidenceLinkedContent(content: string, evidenceIds?: Set<string>) {
  if (!evidenceIds?.size || !content.includes('[ev_')) return content;
  return content.replace(EVIDENCE_REF_PATTERN, (token) => {
    const evidenceId = token.slice(1, -1);
    return evidenceIds.has(evidenceId) ? `[${evidenceId}](#evidence-${evidenceId})` : token;
  });
}

function citationLinkedContent(content: string, citationNumbers?: Set<number>) {
  if (!citationNumbers?.size) return content;
  return content.replace(NUMBERED_CITATION_PATTERN, (token, value: string) => {
    const number = Number(value);
    return citationNumbers.has(number) ? `[${number}](#citation-${number})` : token;
  });
}

export function MarkdownPreview({
  content,
  className,
  evidenceIds,
  activeEvidenceId,
  onEvidenceClick,
  citationNumbers,
  activeCitationNumber,
  onCitationClick,
}: MarkdownPreviewProps) {
  const evidenceSet = evidenceIds?.length ? new Set(evidenceIds) : undefined;
  const citationSet = citationNumbers?.length ? new Set(citationNumbers) : undefined;
  const renderedContent = citationLinkedContent(evidenceLinkedContent(content, evidenceSet), citationSet);
  return (
    <div
      onClick={(event) => {
        const target = event.target as Element | null;
        const citationTarget = target?.closest('[data-inline-citation-number]');
        const citationNumber = Number(citationTarget?.getAttribute('data-inline-citation-number') || 0);
        if (citationNumber && onCitationClick) {
          event.preventDefault();
          onCitationClick(citationNumber);
          return;
        }
        if (!onEvidenceClick) return;
        const evidenceTarget = target?.closest('[data-inline-evidence-id]');
        const evidenceId = evidenceTarget?.getAttribute('data-inline-evidence-id') || '';
        if (!evidenceId) return;
        event.preventDefault();
        onEvidenceClick(evidenceId);
      }}
      className={cn(
        'prose prose-sm max-w-none break-words prose-headings:text-primary prose-p:text-primary prose-a:text-link prose-strong:text-primary prose-code:text-primary prose-pre:overflow-x-auto prose-pre:bg-page prose-pre:text-primary prose-table:my-3 prose-th:whitespace-nowrap prose-th:border prose-th:border-border prose-th:bg-page prose-th:px-2 prose-th:py-1.5 prose-th:text-left prose-td:border prose-td:border-border prose-td:px-2 prose-td:py-1.5 prose-td:align-top',
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table({ children }) {
            return (
              <div className="my-3 max-w-full overflow-x-auto rounded-control border border-border">
                <table className="m-0 min-w-full">{children}</table>
              </div>
            );
          },
          a({ href, children }) {
            const citationNumber = href?.startsWith('#citation-') ? Number(href.replace('#citation-', '')) : 0;
            if (citationNumber && citationSet?.has(citationNumber) && onCitationClick) {
              const active = activeCitationNumber === citationNumber;
              return (
                <sup>
                  <button
                    type="button"
                    data-inline-citation-number={citationNumber}
                    className={cn(
                      'mx-0.5 inline-flex min-w-5 items-center justify-center rounded-control border px-1 py-0.5 align-super text-[10px] font-semibold leading-none no-underline transition-colors',
                      active
                        ? 'border-accent bg-accent text-white'
                        : 'border-accent/25 bg-accent/10 text-accent hover:border-accent/50 hover:bg-white'
                    )}
                    title={`查看来源 ${citationNumber}`}
                  >
                    {citationNumber}
                  </button>
                </sup>
              );
            }
            const evidenceId = href?.startsWith('#evidence-') ? href.replace('#evidence-', '') : '';
            if (evidenceId && evidenceSet?.has(evidenceId) && onEvidenceClick) {
              const active = activeEvidenceId === evidenceId;
              return (
                <button
                  type="button"
                  data-inline-evidence-id={evidenceId}
                  className={cn(
                    'mx-0.5 inline-flex translate-y-[-1px] items-center rounded-control border px-1.5 py-0.5 align-baseline text-[11px] font-medium no-underline transition-colors',
                    active
                      ? 'border-accent bg-accent text-white'
                      : 'border-accent/20 bg-accent/10 text-accent hover:border-accent/40 hover:bg-white'
                  )}
                  title="查看对应证据"
                >
                  {children}
                </button>
              );
            }
            return <a href={href}>{children}</a>;
          },
        }}
      >
        {renderedContent}
      </ReactMarkdown>
    </div>
  );
}
