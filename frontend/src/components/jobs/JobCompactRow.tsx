import { JobSummary } from '@/types';
import { StatusBadge } from '@/components/data/StatusBadge';
import { TruncatedText } from '@/components/data/TruncatedText';
import { formatDateTime } from '@/utils/formatters';
import { JobActions } from './JobActions';
import { JobProgress } from './JobProgress';
import { jobTitle, jobTypeLabel } from './jobDisplay';

export function JobCompactRow({ job, onNavigate }: { job: JobSummary; onNavigate?: () => void }) {
  return (
    <article className="border-b border-border py-3 last:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="line-clamp-2 text-sm font-semibold leading-5 text-primary" title={jobTitle(job)}>{jobTitle(job)}</p>
          <div className="mt-1 flex min-w-0 items-center gap-2 text-xs text-secondary">
            <span>{jobTypeLabel(job.job_type)}</span>
            <span aria-hidden="true">·</span>
            <TruncatedText text={job.job_id} maxLength={18} middle className="min-w-0 font-mono text-[10px]" />
          </div>
        </div>
        <StatusBadge status={job.health_status === 'stalled' ? 'stalled' : job.status} />
      </div>
      {job.progress && ['queued', 'running', 'waiting_approval'].includes(job.status) && <div className="mt-3"><JobProgress job={job} compact /></div>}
      <div className="mt-2 flex items-center justify-between gap-2">
        <time className="text-[11px] text-secondary">{formatDateTime(job.updated_at)}</time>
        <JobActions job={job} onNavigate={onNavigate} />
      </div>
    </article>
  );
}
