import { Link } from 'react-router-dom';
import { ExternalLink, FileText } from 'lucide-react';
import { JobSummary } from '@/types';
import { StatusBadge } from './StatusBadge';
import { Button } from '@/components/controls/Button';
import { useCancelJob } from '@/hooks';
import { formatDateTime } from '@/utils/formatters';
import { TruncatedText } from './TruncatedText';

interface JobCardProps {
  job: JobSummary;
  onClick?: () => void;
  onNavigate?: () => void;
}

export function JobCard({ job, onClick, onNavigate }: JobCardProps) {
  const cancel = useCancelJob();
  const terminal = ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status);
  const displayStatus = !terminal && job.health_status === 'stalled' ? 'stalled' : job.status;
  const cancelReason = job.error?.code === 'cancelled' ? job.error.message : '';
  const inputSummary = job.input
    ? Object.entries(job.input)
        .slice(0, 3)
        .map(([key, value]) => `${key}: ${String(value)}`)
        .join(' · ')
    : '';

  return (
    <div
      onClick={onClick}
      className="cursor-pointer rounded-panel border border-border bg-white p-4 hover:border-accent"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-primary">{job.job_id}</p>
          <p className="text-xs text-secondary">{job.job_type}</p>
        </div>
        <StatusBadge status={displayStatus} />
      </div>
      {job.progress && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-xs text-secondary">
            <span>{job.progress.message}</span>
            <span>{job.progress.current}/{job.progress.total}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
            <div
              className="h-full rounded-full bg-accent transition-all"
              style={{
                width: `${Math.min(100, (job.progress.current / job.progress.total) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}
      {job.error && (
        <div className="mt-2 rounded-control bg-red-50 p-2 text-xs text-danger">
          <p>{job.error.message}</p>
        </div>
      )}
      {cancelReason && (
        <p className="mt-2 rounded-control bg-gray-50 p-2 text-xs text-secondary">
          取消原因：{cancelReason}
        </p>
      )}
      {inputSummary && (
        <p className="mt-2 text-xs text-secondary">
          <TruncatedText text={inputSummary} maxLength={70} />
        </p>
      )}
      {!!job.artifacts?.length && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {job.artifacts.slice(0, 3).map((artifact) => (
            <span
              key={artifact}
              className="inline-flex min-w-0 items-center gap-1 rounded-control bg-page px-2 py-1 text-xs text-secondary"
              title={artifact}
            >
              <FileText className="h-3.5 w-3.5 flex-shrink-0" />
              <TruncatedText text={artifact} maxLength={30} middle />
            </span>
          ))}
        </div>
      )}
      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-secondary">{formatDateTime(job.updated_at)}</span>
        <div className="flex items-center gap-1">
          <Link
            to={`/observability?job_id=${encodeURIComponent(job.job_id)}`}
            className="inline-flex items-center gap-1 rounded-control px-2 py-1.5 text-xs text-secondary hover:bg-page hover:text-primary"
            onClick={(e) => {
              e.stopPropagation();
              onNavigate?.();
            }}
          >
            <ExternalLink className="h-3.5 w-3.5" />
            观测
          </Link>
          {job.job_type === 'research' && (
            <Link
              to={`/research?job_id=${encodeURIComponent(job.job_id)}`}
              className="rounded-control px-2 py-1.5 text-xs text-accent hover:bg-page"
              onClick={(e) => {
                e.stopPropagation();
                onNavigate?.();
              }}
            >
              查看
            </Link>
          )}
          {!terminal && (
            <Button
              size="sm"
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation();
                cancel.mutate({ jobId: job.job_id });
              }}
            >
              取消
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
