import { useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronRight, Info } from 'lucide-react';
import { cn } from '@/utils/cn';
import { ObservabilityEvent } from '@/types';

interface EventTimelineProps {
  events: ObservabilityEvent[];
  emptyLabel?: string;
}

/** "2026-07-19 10:00:00.123" -> "10:00:00.123" — deliberately not routed through
 * formatDateTime()/toLocaleString(), which only renders down to the minute and
 * would throw away the millisecond precision this view exists to show. */
function formatClock(timestamp: string): string {
  const match = timestamp.match(/(\d{2}:\d{2}:\d{2}(?:\.\d+)?)/);
  return match ? match[1] : timestamp;
}

function levelIcon(level: string) {
  if (level === 'error') return <AlertTriangle className="h-3.5 w-3.5 text-danger" />;
  if (level === 'warning') return <AlertTriangle className="h-3.5 w-3.5 text-warning" />;
  return <Info className="h-3.5 w-3.5 text-link" />;
}

export function EventTimeline({ events, emptyLabel = '暂无事件' }: EventTimelineProps) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  if (!events.length) {
    return <div className="px-3 py-8 text-center text-sm text-secondary">{emptyLabel}</div>;
  }

  const toggle = (idx: number) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  return (
    <ol className="relative space-y-0">
      {events.map((event, idx) => {
        const isOpen = expanded.has(idx);
        const isLast = idx === events.length - 1;
        return (
          <li key={`${event.timestamp}-${idx}`} className="relative pl-6">
            {!isLast && <span className="absolute left-[7px] top-5 h-full w-px bg-border" />}
            <span className="absolute left-0 top-1.5">{levelIcon(event.level)}</span>
            <button
              type="button"
              onClick={() => toggle(idx)}
              className="mb-1 flex w-full items-start justify-between gap-2 rounded-control px-2 py-1.5 text-left hover:bg-page"
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs text-secondary">{formatClock(event.timestamp)}</span>
                  <span className="text-xs font-medium text-accent">{event.event_type}</span>
                  <span className="text-xs text-secondary">{event.component}</span>
                </div>
                <p className="mt-0.5 truncate text-sm text-primary">{event.message}</p>
              </div>
              {isOpen ? (
                <ChevronDown className="mt-0.5 h-4 w-4 flex-shrink-0 text-secondary" />
              ) : (
                <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-secondary" />
              )}
            </button>
            {isOpen && (
              <pre className={cn('mb-2 overflow-x-auto rounded-control bg-page p-3 text-xs text-primary')}>
                {JSON.stringify(event, null, 2)}
              </pre>
            )}
          </li>
        );
      })}
    </ol>
  );
}
