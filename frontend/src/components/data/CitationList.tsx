import { useEffect, useState } from 'react';
import { Citation } from '@/types';
import { ChevronDown, ChevronRight, ChevronUp, FileText } from 'lucide-react';
import { cn } from '@/utils/cn';
import { cleanCitationSnippet } from '@/utils/citationDisplay';

interface CitationListProps {
  citations: Citation[];
  selectedCitation?: Citation | null;
  onCitationClick?: (citation: Citation) => void;
}

function isSameCitation(a?: Citation | null, b?: Citation | null) {
  return !!a && !!b && a.path === b.path && a.title === b.title;
}

export function CitationList({ citations, selectedCitation, onCitationClick }: CitationListProps) {
  const [expanded, setExpanded] = useState(false);
  useEffect(() => setExpanded(false), [citations]);
  if (!citations.length) return null;
  const uniqueCitations = citations.filter(
    (citation, index) =>
      citations.findIndex((candidate) => candidate.path === citation.path && candidate.title === citation.title) === index
  );
  const visibleCitations = expanded ? uniqueCitations : uniqueCitations.slice(0, 3);
  const remainingCount = uniqueCitations.length - visibleCitations.length;

  return (
    <div className="space-y-2 pt-1">
      <div className="flex items-center gap-2 text-xs text-secondary">
        <FileText className="h-3.5 w-3.5" />
        <h4 className="font-medium text-primary">引用</h4>
        <span>{uniqueCitations.length}</span>
      </div>
      <div className="overflow-hidden rounded-control border border-border bg-white">
        {visibleCitations.map((c, idx) => {
          const active = isSameCitation(c, selectedCitation);
          return (
            <button
              key={`${c.path}-${idx}`}
              type="button"
              aria-pressed={active}
              onClick={() => onCitationClick?.(c)}
              className={cn(
                'flex w-full items-start gap-3 border-b border-border px-3 py-2.5 text-left transition-colors last:border-b-0 hover:bg-page/60',
                active && 'bg-accent/5'
              )}
            >
              <span
                className={cn(
                  'mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-control border text-[11px] font-semibold',
                  active ? 'border-accent bg-accent text-white' : 'border-border bg-page text-secondary'
                )}
              >
                {idx + 1}
              </span>
              <div className="min-w-0 flex-1">
                <p className={cn('truncate text-sm font-medium', active ? 'text-accent' : 'text-primary')}>
                  {c.title}
                </p>
                <p className="mt-0.5 line-clamp-2 text-xs leading-5 text-secondary">
                  {cleanCitationSnippet(c.snippet)}
                </p>
              </div>
              <ChevronRight className={cn('mt-1 h-4 w-4 flex-shrink-0', active ? 'text-accent' : 'text-secondary')} />
            </button>
          );
        })}
        {uniqueCitations.length > 3 && (
          <button
            type="button"
            onClick={() => setExpanded((value) => !value)}
            className="flex w-full items-center justify-center gap-1.5 border-t border-border px-3 py-2 text-xs font-medium text-secondary hover:bg-page hover:text-primary"
          >
            {expanded ? (
              <>
                <ChevronUp className="h-3.5 w-3.5" />
                收起引用
              </>
            ) : (
              <>
                <ChevronDown className="h-3.5 w-3.5" />
                查看其余 {remainingCount} 条
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
