import { Briefcase, Clock, Hash } from 'lucide-react';
import { cn } from '@/utils/cn';
import { formatDateTime } from '@/utils/formatters';
import { RunSummary } from '@/types';
import { StatusBadge } from '@/components/data/StatusBadge';

function formatDuration(ms?: number | null): string | null {
  if (ms == null) return null;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

interface RunListProps {
  runs: RunSummary[];
  selectedId?: string | null;
  onSelect: (run: RunSummary) => void;
}

export function RunList({ runs, selectedId, onSelect }: RunListProps) {
  if (!runs.length) {
    return <div className="px-3 py-8 text-center text-sm text-secondary">暂无匹配的 run</div>;
  }

  return (
    <div className="space-y-2">
      {runs.map((run) => {
        const duration = formatDuration(run.duration_ms);
        return (
          <button
            key={run.id}
            type="button"
            onClick={() => onSelect(run)}
            className={cn(
              'w-full rounded-control border border-border bg-white p-3 text-left hover:border-accent',
              selectedId === run.id && 'border-accent bg-accent/5'
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <p className="min-w-0 flex-1 truncate text-sm font-medium text-primary">{run.title}</p>
              <StatusBadge status={run.status} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-secondary">
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3.5 w-3.5" />
                {formatDateTime(run.started_at)}
                {duration && <span className="ml-1">· 耗时 {duration}</span>}
              </span>
              <span className="inline-flex items-center gap-1">
                <Hash className="h-3.5 w-3.5" />
                {run.event_count} 条事件
                {run.error_count > 0 && <span className="text-danger"> · {run.error_count} 个报错</span>}
              </span>
              {run.job_id && (
                <span className="inline-flex items-center gap-1">
                  <Briefcase className="h-3.5 w-3.5" />
                  {run.job_id}
                </span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
