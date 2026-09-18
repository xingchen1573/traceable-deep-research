import { JobSummary } from '@/types';
import { jobProgressLabel, jobProgressPercent } from './jobDisplay';

export function JobProgress({ job, compact = false }: { job: JobSummary; compact?: boolean }) {
  const percent = jobProgressPercent(job);
  if (percent == null) return <span className="text-xs text-secondary">-</span>;

  return (
    <div className={compact ? 'w-full' : 'min-w-[120px]'}>
      <div className="mb-1 flex items-center justify-between gap-3 text-[11px] text-secondary">
        <span className="truncate">{jobProgressLabel(job)}</span>
        <span className="font-mono">{percent}%</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-border">
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
