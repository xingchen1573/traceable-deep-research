import { useEffect, useMemo, useState } from 'react';
import { Archive, ArchiveRestore, RefreshCw } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { Button } from '@/components/controls/Button';
import { FilterSelect, SearchInput } from '@/components/controls/FormControls';
import { EmptyState, LoadingState } from '@/components/data/EmptyState';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { StatusBadge } from '@/components/data/StatusBadge';
import { TruncatedText } from '@/components/data/TruncatedText';
import { JobActions, JobCompactRow, JobDetailDrawer, JobProgress, isTerminalJob, jobTitle, jobTypeLabel } from '@/components/jobs';
import { PageHeader } from '@/components/layout/PageHeader';
import { useBulkArchiveJobs, useInfiniteJobs, useJobStats } from '@/hooks';
import { JobStatus } from '@/types';
import { formatDateTime } from '@/utils/formatters';

const STATUS_OPTIONS = [
  { value: 'all', label: '全部状态' },
  { value: 'waiting_approval', label: '待审批' },
  { value: 'running', label: '运行中' },
  { value: 'queued', label: '排队中' },
  { value: 'succeeded', label: '成功' },
  { value: 'failed', label: '失败' },
  { value: 'interrupted', label: '已中断' },
  { value: 'cancelled', label: '已取消' },
];

export function JobsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchInput, setSearchInput] = useState('');
  const [query, setQuery] = useState('');
  const [jobType, setJobType] = useState('all');
  const [status, setStatus] = useState<JobStatus | 'all'>('all');
  const [archived, setArchived] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const stats = useJobStats(false);
  const archivedStats = useJobStats(true);
  const bulkArchive = useBulkArchiveJobs();

  useEffect(() => {
    const timer = window.setTimeout(() => setQuery(searchInput.trim()), 300);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => setSelected(new Set()), [query, jobType, status, archived]);

  const jobsQuery = useInfiniteJobs({
    query: query || undefined,
    jobType: jobType === 'all' ? undefined : jobType,
    status: status === 'all' ? undefined : status,
    archived,
    limit: 40,
  });
  const jobs = jobsQuery.data?.pages.flatMap((page) => page.jobs) || [];
  const total = jobsQuery.data?.pages[0]?.total || 0;
  const selectableJobs = jobs.filter(isTerminalJob);
  const allSelected = selectableJobs.length > 0 && selectableJobs.every((job) => selected.has(job.job_id));
  const typeOptions = useMemo(() => [
    { value: 'all', label: '全部类型' },
    ...Object.keys(stats.data?.by_type || {}).sort().map((type) => ({ value: type, label: jobTypeLabel(type) })),
  ], [stats.data?.by_type]);
  const selectedJobId = searchParams.get('job_id') || '';

  const toggleSelected = (jobId: string) => setSelected((current) => {
    const next = new Set(current);
    next.has(jobId) ? next.delete(jobId) : next.add(jobId);
    return next;
  });
  const toggleAll = () => setSelected(allSelected ? new Set() : new Set(selectableJobs.map((job) => job.job_id)));
  const openDetail = (jobId: string) => setSearchParams({ job_id: jobId });
  const closeDetail = () => setSearchParams({});

  return (
    <div>
      <PageHeader
        title="任务"
        description="查看和处理调研、索引、评估及其他后台任务"
        actions={<Button variant="secondary" icon={<RefreshCw className="mr-2 h-4 w-4" />} onClick={() => jobsQuery.refetch()}>刷新</Button>}
      />

      <section className="mb-5 grid grid-cols-2 border-y border-border bg-surface sm:grid-cols-3 lg:grid-cols-6" aria-label="任务概览">
        <Metric label="未归档" value={stats.data?.total} />
        <Metric label="待处理" value={stats.data?.attention} accent />
        <Metric label="进行中" value={stats.data?.active} />
        <Metric label="失败/中断" value={stats.data?.failed} />
        <Metric label="已完成" value={stats.data?.by_status.succeeded} />
        <Metric label="已归档" value={archivedStats.data?.total} />
      </section>

      <section className="mb-4 flex flex-col gap-3 border-b border-border pb-4 lg:flex-row lg:items-center">
        <SearchInput value={searchInput} onChange={(event) => setSearchInput(event.target.value)} placeholder="搜索主题、URL 或任务 ID" wrapperClassName="min-w-0 flex-1 lg:max-w-md" />
        <div className="flex flex-wrap items-center gap-3">
          <FilterSelect value={jobType} onChange={(event) => setJobType(event.target.value)} options={typeOptions} />
          <FilterSelect value={status} onChange={(event) => setStatus(event.target.value as JobStatus | 'all')} options={STATUS_OPTIONS} />
          <div className="flex items-center rounded-control border border-border bg-surface p-1" aria-label="归档状态">
            <button type="button" onClick={() => setArchived(false)} className={`rounded-control px-3 py-1.5 text-xs ${!archived ? 'bg-primary text-page' : 'text-secondary hover:text-primary'}`}>未归档</button>
            <button type="button" onClick={() => setArchived(true)} className={`rounded-control px-3 py-1.5 text-xs ${archived ? 'bg-primary text-page' : 'text-secondary hover:text-primary'}`}>已归档</button>
          </div>
        </div>
      </section>

      {selected.size > 0 && (
        <div className="mb-3 flex items-center justify-between border-y border-border bg-page px-3 py-2">
          <span className="text-xs text-secondary">已选择 {selected.size} 个已结束任务</span>
          <Button
            size="sm"
            variant="secondary"
            isLoading={bulkArchive.isPending}
            icon={archived ? <ArchiveRestore className="mr-1.5 h-4 w-4" /> : <Archive className="mr-1.5 h-4 w-4" />}
            onClick={() => bulkArchive.mutate({ jobIds: [...selected], archived: !archived }, { onSuccess: () => setSelected(new Set()) })}
          >
            {archived ? '批量恢复' : '批量归档'}
          </Button>
        </div>
      )}

      {jobsQuery.isLoading ? <LoadingState message="正在加载任务..." /> : jobsQuery.isError ? <ErrorPanel error={jobsQuery.error} onRetry={() => jobsQuery.refetch()} /> : !jobs.length ? (
        <EmptyState title="没有匹配任务" description={query || status !== 'all' || jobType !== 'all' ? '调整搜索或筛选条件后再试。' : archived ? '当前没有已归档任务。' : '新任务创建后会出现在这里。'} />
      ) : (
        <>
          <div className="hidden overflow-x-auto rounded-panel border border-border bg-surface md:block">
            <table className="min-w-[980px] w-full text-left text-sm">
              <thead className="border-b border-border bg-page text-xs text-secondary">
                <tr>
                  <th className="w-10 px-3 py-3"><input type="checkbox" aria-label="选择本页已结束任务" checked={allSelected} onChange={toggleAll} className="h-4 w-4 accent-accent" /></th>
                  <th className="px-3 py-3 font-medium">任务</th>
                  <th className="px-3 py-3 font-medium">类型</th>
                  <th className="px-3 py-3 font-medium">状态</th>
                  <th className="px-3 py-3 font-medium">进度</th>
                  <th className="px-3 py-3 font-medium">更新时间</th>
                  <th className="px-3 py-3 text-right font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.job_id} onClick={() => openDetail(job.job_id)} className="cursor-pointer border-b border-border last:border-b-0 hover:bg-page">
                    <td className="px-3 py-3" onClick={(event) => event.stopPropagation()}><input type="checkbox" aria-label={`选择 ${jobTitle(job)}`} disabled={!isTerminalJob(job)} checked={selected.has(job.job_id)} onChange={() => toggleSelected(job.job_id)} className="h-4 w-4 accent-accent disabled:opacity-30" /></td>
                    <td className="max-w-[320px] px-3 py-3"><p className="truncate font-medium text-primary" title={jobTitle(job)}>{jobTitle(job)}</p><TruncatedText text={job.job_id} maxLength={24} middle className="mt-1 block text-[10px] text-secondary" /></td>
                    <td className="whitespace-nowrap px-3 py-3 text-xs text-secondary">{jobTypeLabel(job.job_type)}</td>
                    <td className="whitespace-nowrap px-3 py-3"><StatusBadge status={job.health_status === 'stalled' ? 'stalled' : job.status} /></td>
                    <td className="w-40 px-3 py-3"><JobProgress job={job} /></td>
                    <td className="whitespace-nowrap px-3 py-3 text-xs text-secondary">{formatDateTime(job.updated_at)}</td>
                    <td className="px-3 py-3"><JobActions job={job} allowArchive /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="rounded-panel border border-border bg-surface px-4 md:hidden">
            {jobs.map((job) => <div key={job.job_id} onClick={() => openDetail(job.job_id)}><JobCompactRow job={job} /></div>)}
          </div>

          <div className="mt-4 flex items-center justify-between gap-4 text-xs text-secondary">
            <span>已显示 {jobs.length} / {total}</span>
            {jobsQuery.hasNextPage && <Button variant="secondary" size="sm" isLoading={jobsQuery.isFetchingNextPage} onClick={() => jobsQuery.fetchNextPage()}>加载更多</Button>}
          </div>
        </>
      )}

      {selectedJobId && <JobDetailDrawer jobId={selectedJobId} onClose={closeDetail} />}
    </div>
  );
}

function Metric({ label, value, accent = false }: { label: string; value?: number; accent?: boolean }) {
  return <div className="border-b border-r border-border px-4 py-4 last:border-r-0 sm:border-b-0"><p className="text-[11px] text-secondary">{label}</p><p className={`mt-1 font-display text-2xl font-semibold ${accent ? 'text-accent' : 'text-primary'}`}>{value ?? '-'}</p></div>;
}
