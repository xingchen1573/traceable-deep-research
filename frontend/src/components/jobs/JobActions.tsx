import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Archive, ArchiveRestore, ExternalLink, XCircle } from 'lucide-react';
import { ConfirmDialog } from '@/components/controls/FormControls';
import { useArchiveJob, useCancelJob } from '@/hooks';
import { JobSummary } from '@/types';
import { isTerminalJob, jobTarget } from './jobDisplay';

export function JobActions({
  job,
  allowArchive = false,
  onNavigate,
}: {
  job: JobSummary;
  allowArchive?: boolean;
  onNavigate?: () => void;
}) {
  const [confirmCancel, setConfirmCancel] = useState(false);
  const cancel = useCancelJob();
  const archive = useArchiveJob();
  const terminal = isTerminalJob(job);
  const primaryLabel = job.status === 'waiting_approval' && job.job_type === 'research' ? '处理' : '查看';

  return (
    <>
      <div className="flex flex-wrap items-center justify-end gap-1">
        <Link
          to={jobTarget(job)}
          onClick={(event) => { event.stopPropagation(); onNavigate?.(); }}
          className="rounded-control px-2 py-1.5 text-xs font-medium text-accent hover:bg-page"
        >
          {primaryLabel}
        </Link>
        {job.job_type === 'research' && (
          <Link
            to={`/observability?job_id=${encodeURIComponent(job.job_id)}`}
            onClick={(event) => { event.stopPropagation(); onNavigate?.(); }}
            className="inline-flex items-center gap-1 rounded-control px-2 py-1.5 text-xs text-secondary hover:bg-page hover:text-primary"
          >
            <ExternalLink className="h-3.5 w-3.5" />观测
          </Link>
        )}
        {!terminal && job.status !== 'waiting_approval' && (
          <button
            type="button"
            onClick={(event) => { event.stopPropagation(); setConfirmCancel(true); }}
            className="inline-flex items-center gap-1 rounded-control px-2 py-1.5 text-xs text-secondary hover:bg-page hover:text-danger"
          >
            <XCircle className="h-3.5 w-3.5" />取消
          </button>
        )}
        {allowArchive && terminal && (
          <button
            type="button"
            disabled={archive.isPending}
            onClick={(event) => {
              event.stopPropagation();
              archive.mutate({ jobId: job.job_id, archived: !job.archived });
            }}
            className="inline-flex items-center gap-1 rounded-control px-2 py-1.5 text-xs text-secondary hover:bg-page hover:text-primary disabled:opacity-50"
          >
            {job.archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
            {job.archived ? '恢复' : '归档'}
          </button>
        )}
      </div>
      <ConfirmDialog
        open={confirmCancel}
        title="取消任务"
        message={`确认取消任务 ${job.job_id}？已完成的步骤和产物会保留。`}
        confirmText="确认取消"
        onCancel={() => setConfirmCancel(false)}
        onConfirm={() => cancel.mutate({ jobId: job.job_id }, { onSettled: () => setConfirmCancel(false) })}
      />
    </>
  );
}
