import { useEffect } from 'react';
import { FileText, X } from 'lucide-react';
import { IconButton } from '@/components/controls/Button';
import { EmptyState } from '@/components/data/EmptyState';
import { StatusBadge } from '@/components/data/StatusBadge';
import { TruncatedText } from '@/components/data/TruncatedText';
import { useJobPolling } from '@/hooks';
import { formatDateTime } from '@/utils/formatters';
import { JobActions } from './JobActions';
import { JobProgress } from './JobProgress';
import { jobTitle, jobTypeLabel } from './jobDisplay';

export function JobDetailDrawer({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const job = useJobPolling(jobId);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => event.key === 'Escape' && onClose();
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 bg-black/35" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="ml-auto flex h-full w-full max-w-[520px] flex-col border-l border-border bg-surface shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="job-detail-title">
        <header className="flex min-h-20 flex-none items-center justify-between gap-4 border-b border-border px-5 py-4">
          <div className="min-w-0">
            <p className="font-mono text-[10px] font-semibold uppercase tracking-wide text-accent">TASK DETAIL</p>
            <h2 id="job-detail-title" className="mt-1 truncate font-display text-xl font-semibold text-primary">任务详情</h2>
          </div>
          <IconButton label="关闭任务详情" icon={<X className="h-4 w-4" />} variant="secondary" onClick={onClose} />
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {job.isLoading && <p className="py-8 text-center text-sm text-secondary">正在读取任务...</p>}
          {job.isError && <EmptyState title="任务读取失败" description="任务可能已经被清理，或服务暂时不可用。" />}
          {job.data && (
            <div>
              <div className="flex items-start justify-between gap-4 border-b border-border pb-5">
                <div className="min-w-0">
                  <h3 className="text-base font-semibold leading-6 text-primary">{jobTitle(job.data)}</h3>
                  <p className="mt-1 text-xs text-secondary">{jobTypeLabel(job.data.job_type)}</p>
                </div>
                <StatusBadge status={job.data.health_status === 'stalled' ? 'stalled' : job.data.health_status === 'orphaned' ? 'orphaned' : job.data.status} />
              </div>

              <dl className="grid grid-cols-[108px_minmax(0,1fr)] gap-x-4 border-b border-border py-4 text-xs">
                <dt className="py-2 text-secondary">任务 ID</dt><dd className="min-w-0 py-2 text-primary"><TruncatedText text={job.data.job_id} maxLength={34} middle /></dd>
                <dt className="py-2 text-secondary">创建时间</dt><dd className="py-2 text-primary">{formatDateTime(job.data.created_at)}</dd>
                <dt className="py-2 text-secondary">更新时间</dt><dd className="py-2 text-primary">{formatDateTime(job.data.updated_at)}</dd>
                <dt className="py-2 text-secondary">当前步骤</dt><dd className="py-2 text-primary">{job.data.current_step || '-'}</dd>
                <dt className="py-2 text-secondary">执行环境</dt><dd className="py-2 text-primary">{job.data.environment || '-'}</dd>
              </dl>

              {job.data.progress && <section className="border-b border-border py-5"><h4 className="mb-3 text-xs font-semibold text-primary">任务进度</h4><JobProgress job={job.data} /></section>}

              {!!Object.keys(job.data.input || {}).length && (
                <section className="border-b border-border py-5">
                  <h4 className="mb-3 text-xs font-semibold text-primary">任务输入</h4>
                  <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-control bg-page p-3 text-[11px] leading-5 text-primary">{JSON.stringify(job.data.input, null, 2)}</pre>
                </section>
              )}

              {job.data.error && (
                <section className="border-b border-border py-5">
                  <h4 className="text-xs font-semibold text-danger">执行错误</h4>
                  <p className="mt-2 text-sm leading-6 text-primary">{job.data.error.message}</p>
                  <p className="mt-1 font-mono text-[10px] text-secondary">{job.data.error.code}</p>
                </section>
              )}

              {!!job.data.artifacts?.length && (
                <section className="py-5">
                  <h4 className="mb-3 text-xs font-semibold text-primary">任务产物</h4>
                  <div className="space-y-2">
                    {job.data.artifacts.map((artifact) => <div key={artifact} className="flex min-w-0 items-center gap-2 text-xs text-secondary"><FileText className="h-4 w-4 flex-none" /><span className="min-w-0 break-all">{artifact}</span></div>)}
                  </div>
                </section>
              )}
            </div>
          )}
        </div>

        {job.data && <footer className="flex flex-none justify-end border-t border-border px-5 py-3"><JobActions job={job.data} allowArchive onNavigate={onClose} /></footer>}
      </aside>
    </div>
  );
}
