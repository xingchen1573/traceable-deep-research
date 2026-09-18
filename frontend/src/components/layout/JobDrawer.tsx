import { ArrowRight, RefreshCw, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import { cn } from '@/utils/cn';
import { useUIStore, useJobStore } from '@/stores';
import { useJobStats, useJobs, useJobPolling } from '@/hooks';
import { JobCompactRow, isTerminalJob } from '@/components/jobs';
import { EmptyState } from '@/components/data/EmptyState';
import { IconButton } from '@/components/controls/Button';
import { ROUTES } from '@/router/routes';
import { JobSummary } from '@/types';

export function JobDrawer() {
  const { jobDrawerOpen, setJobDrawerOpen } = useUIStore();
  const { trackedJobs } = useJobStore();
  const attentionJobs = useJobs({ archived: false, status: 'waiting_approval', limit: 8 });
  const activeJobs = useJobs({ archived: false, statuses: ['queued', 'running'], limit: 8 });
  const recentJobs = useJobs({ archived: false, statuses: ['succeeded', 'failed', 'cancelled', 'interrupted'], limit: 6 });
  const stats = useJobStats(false);

  const displayedJobs = [...(attentionJobs.data?.jobs || []), ...(activeJobs.data?.jobs || []), ...(recentJobs.data?.jobs || [])];
  const listedIds = new Set(displayedJobs.map((job) => job.job_id));
  const missingTrackedIds = trackedJobs.filter((jobId) => !listedIds.has(jobId));
  const attention = attentionJobs.data?.jobs || [];
  const active = activeJobs.data?.jobs || [];
  const recent = (recentJobs.data?.jobs || []).filter(isTerminalJob);
  const hasJobs = missingTrackedIds.length + attention.length + active.length + recent.length > 0;
  const isLoading = attentionJobs.isLoading || activeJobs.isLoading || recentJobs.isLoading;
  const isFetching = attentionJobs.isFetching || activeJobs.isFetching || recentJobs.isFetching;
  const refresh = () => {
    attentionJobs.refetch();
    activeJobs.refetch();
    recentJobs.refetch();
    stats.refetch();
  };

  return (
    <>
      {jobDrawerOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/20"
          onClick={() => setJobDrawerOpen(false)}
        />
      )}
      <aside
        className={cn(
          'fixed right-0 top-0 z-[60] flex h-full w-full max-w-[420px] transform flex-col border-l border-border bg-surface shadow-lg transition-transform duration-200',
          jobDrawerOpen ? 'translate-x-0' : 'invisible translate-x-full pointer-events-none'
        )}
        role={jobDrawerOpen ? 'dialog' : undefined}
        aria-modal={jobDrawerOpen || undefined}
        aria-hidden={!jobDrawerOpen}
        aria-labelledby="job-drawer-title"
      >
        <div className="flex min-h-20 flex-none items-center justify-between gap-4 border-b border-border px-5 py-4">
          <div>
            <h2 id="job-drawer-title" className="font-display text-xl font-semibold text-primary">任务</h2>
            <p className="mt-1 text-xs text-secondary">{stats.data?.attention || 0} 待处理 · {stats.data?.active || 0} 进行中</p>
          </div>
          <div className="flex items-center gap-1">
            <IconButton label="刷新任务" icon={<RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />} onClick={refresh} />
            <IconButton label="关闭任务栏" icon={<X className="h-4 w-4" />} onClick={() => setJobDrawerOpen(false)} />
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-2">
          {isLoading && <div className="py-8 text-center text-sm text-secondary">加载中...</div>}
          {!isLoading && !hasJobs && (
            <EmptyState title="暂无任务" description="当前没有长任务记录。" />
          )}
          {!!missingTrackedIds.length && <JobSection title="刚创建">{missingTrackedIds.map((jobId) => <TrackedJobRow key={jobId} jobId={jobId} onNavigate={() => setJobDrawerOpen(false)} />)}</JobSection>}
          {!!attention.length && <JobSection title="待处理" count={stats.data?.attention}>{attention.map((job) => <JobCompactRow key={job.job_id} job={job} onNavigate={() => setJobDrawerOpen(false)} />)}</JobSection>}
          {!!active.length && <JobSection title="进行中" count={stats.data?.active}>{active.map((job) => <JobCompactRow key={job.job_id} job={job} onNavigate={() => setJobDrawerOpen(false)} />)}</JobSection>}
          {!!recent.length && <JobSection title="最近完成">{recent.map((job) => <JobCompactRow key={job.job_id} job={job} onNavigate={() => setJobDrawerOpen(false)} />)}</JobSection>}
        </div>
        <Link to={ROUTES.JOBS} onClick={() => setJobDrawerOpen(false)} className="flex min-h-14 flex-none items-center justify-center gap-2 border-t border-border text-sm font-medium text-primary hover:bg-page">管理全部任务<ArrowRight className="h-4 w-4" /></Link>
      </aside>
    </>
  );
}

function JobSection({ title, count, children }: { title: string; count?: number; children: React.ReactNode }) {
  return <section className="py-3"><div className="flex items-center justify-between border-b border-border pb-2"><h3 className="text-xs font-semibold text-primary">{title}</h3>{count != null && <span className="font-mono text-[10px] text-secondary">{count}</span>}</div>{children}</section>;
}

function TrackedJobRow({ jobId, onNavigate }: { jobId: string; onNavigate: () => void }) {
  const { data: job } = useJobPolling(jobId);
  if (!job) return null;
  return <JobCompactRow job={job as JobSummary} onNavigate={onNavigate} />;
}
