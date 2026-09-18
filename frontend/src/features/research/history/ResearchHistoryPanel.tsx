import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, CircleHelp, LayoutGrid, List } from 'lucide-react';
import { SearchInput, FilterSelect } from '@/components/controls/FormControls';
import { EmptyState, LoadingState } from '@/components/data/EmptyState';
import { useJobs, useArchiveJob } from '@/hooks';
import { JobStatus, ResearchJob } from '@/types';
import { cn } from '@/utils/cn';
import { getPaginationItems } from '@/utils/pagination';
import { ResearchHistoryList } from './ResearchHistoryList';
import { ResearchHistoryCardGrid } from './ResearchHistoryCardGrid';
import { useResearchHistoryViewMode } from './useResearchHistoryViewMode';
import { HISTORY_STATUS_OPTIONS, HISTORY_TYPE_OPTIONS, ResearchHistoryType } from './researchHistory';

const HISTORY_PAGE_SIZE = 10;

interface ResearchHistoryPanelProps {
  onSelectJob: (jobId: string) => void;
}

export function ResearchHistoryPanel({ onSelectJob }: ResearchHistoryPanelProps) {
  const [searchInput, setSearchInput] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<ResearchHistoryType | 'all'>('all');
  const [statusFilter, setStatusFilter] = useState<JobStatus | 'all'>('all');
  const [showArchived, setShowArchived] = useState(false);
  const [page, setPage] = useState(1);
  const [viewMode, setViewMode] = useResearchHistoryViewMode();
  const archiveJob = useArchiveJob();
  const [archivingJobId, setArchivingJobId] = useState<string | undefined>();

  useEffect(() => {
    const timer = window.setTimeout(() => setSearchTerm(searchInput), 300);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => setPage(1), [searchTerm, typeFilter, statusFilter, showArchived]);

  const jobsQuery = useJobs({
    jobType: 'research',
    researchType: typeFilter === 'all' ? undefined : typeFilter,
    status: statusFilter === 'all' ? undefined : statusFilter,
    archived: showArchived,
    query: searchTerm || undefined,
    limit: HISTORY_PAGE_SIZE,
    offset: (page - 1) * HISTORY_PAGE_SIZE,
  });
  const jobs = (jobsQuery.data?.jobs || []) as ResearchJob[];
  const total = jobsQuery.data?.total || 0;
  const totalPages = Math.max(1, Math.ceil(total / HISTORY_PAGE_SIZE));
  const paginationItems = useMemo(() => getPaginationItems(page, totalPages), [page, totalPages]);
  const hasActiveFilters = Boolean(searchTerm) || typeFilter !== 'all' || statusFilter !== 'all';

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const handleArchive = (jobId: string, archived: boolean) => {
    setArchivingJobId(jobId);
    archiveJob.mutate({ jobId, archived }, { onSettled: () => setArchivingJobId(undefined) });
  };

  return (
    <div className="space-y-4">
      <section className="flex flex-wrap items-center gap-3 rounded-panel border border-border bg-white p-4">
        <SearchInput
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="按主题、URL 或任务 ID 搜索"
          wrapperClassName="min-w-[220px] flex-1"
        />
        <FilterSelect
          label="类型"
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value as ResearchHistoryType | 'all')}
          options={HISTORY_TYPE_OPTIONS}
        />
        <FilterSelect
          label="状态"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as JobStatus | 'all')}
          options={HISTORY_STATUS_OPTIONS}
        />
        <div className="flex items-center gap-1 rounded-control border border-border p-1">
          <button
            type="button"
            onClick={() => setShowArchived(false)}
            className={cn('rounded-control px-2.5 py-1 text-xs', !showArchived ? 'bg-accent text-white' : 'text-secondary')}
          >
            未归档
          </button>
          <button
            type="button"
            onClick={() => setShowArchived(true)}
            className={cn('rounded-control px-2.5 py-1 text-xs', showArchived ? 'bg-accent text-white' : 'text-secondary')}
          >
            已归档
          </button>
        </div>
        <span
          className="inline-flex items-center gap-1 text-xs text-secondary"
          title="归档只会把任务移出默认列表，不会删除报告、证据、来源或知识库内容。"
        >
          <CircleHelp className="h-3.5 w-3.5" />
          <span className="hidden xl:inline">归档不删除内容</span>
        </span>
        <div className="flex items-center gap-1 rounded-control border border-border p-1">
          <button
            type="button"
            aria-label="列表视图"
            onClick={() => setViewMode('list')}
            className={cn('rounded-control p-1.5', viewMode === 'list' ? 'bg-accent text-white' : 'text-secondary')}
          >
            <List className="h-4 w-4" />
          </button>
          <button
            type="button"
            aria-label="卡片视图"
            onClick={() => setViewMode('card')}
            className={cn('rounded-control p-1.5', viewMode === 'card' ? 'bg-accent text-white' : 'text-secondary')}
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
        </div>
      </section>

      {jobsQuery.isLoading ? (
        <LoadingState message="正在加载历史调研..." />
      ) : jobsQuery.isError ? (
        <EmptyState title="加载失败" description="历史调研列表加载失败，请稍后重试。" />
      ) : !jobs.length ? (
        <EmptyState
          title={hasActiveFilters ? '没有匹配结果' : showArchived ? '暂无已归档任务' : '暂无历史调研'}
          description={
            hasActiveFilters
              ? '试试调整搜索关键词或筛选条件。'
              : showArchived
                ? '归档任务会出现在这里。'
                : '创建一次调研或单 URL 简报后，历史记录会出现在这里。'
          }
        />
      ) : viewMode === 'list' ? (
        <ResearchHistoryList jobs={jobs} onSelect={onSelectJob} onArchive={handleArchive} archivingJobId={archivingJobId} />
      ) : (
        <ResearchHistoryCardGrid jobs={jobs} onSelect={onSelectJob} onArchive={handleArchive} archivingJobId={archivingJobId} />
      )}

      {!jobsQuery.isLoading && !jobsQuery.isError && total > 0 && (
        <nav className="flex flex-col gap-3 border-t border-border pt-3 sm:flex-row sm:items-center sm:justify-between" aria-label="历史调研分页">
          <p className="text-xs text-secondary">
            第 {(page - 1) * HISTORY_PAGE_SIZE + 1}–{Math.min(page * HISTORY_PAGE_SIZE, total)} 条，共 {total} 条
          </p>
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="上一页"
              disabled={page === 1}
              onClick={() => setPage((current) => Math.max(1, current - 1))}
              className="rounded-control border border-border p-1.5 text-secondary hover:border-accent hover:text-primary disabled:cursor-not-allowed disabled:opacity-35"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            {paginationItems.map((item) => typeof item === 'number' ? (
              <button
                key={item}
                type="button"
                aria-label={`第 ${item} 页`}
                aria-current={item === page ? 'page' : undefined}
                onClick={() => setPage(item)}
                className={cn(
                  'min-w-8 rounded-control border px-2 py-1.5 text-xs tabular-nums',
                  item === page
                    ? 'border-accent bg-accent text-white'
                    : 'border-border text-secondary hover:border-accent hover:text-primary'
                )}
              >
                {item}
              </button>
            ) : (
              <span key={item} className="px-1 text-xs text-secondary" aria-hidden="true">…</span>
            ))}
            <button
              type="button"
              aria-label="下一页"
              disabled={page === totalPages}
              onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
              className="rounded-control border border-border p-1.5 text-secondary hover:border-accent hover:text-primary disabled:cursor-not-allowed disabled:opacity-35"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </nav>
      )}
    </div>
  );
}
