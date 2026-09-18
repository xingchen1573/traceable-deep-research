import { Archive, ArchiveRestore } from 'lucide-react';
import { StatusBadge } from '@/components/data/StatusBadge';
import { TruncatedText } from '@/components/data/TruncatedText';
import { formatDateTime } from '@/utils/formatters';
import { ResearchJob } from '@/types';
import { deriveHistoryDisplay } from './researchHistory';

interface ResearchHistoryCardGridProps {
  jobs: ResearchJob[];
  onSelect: (jobId: string) => void;
  onArchive: (jobId: string, archived: boolean) => void;
  archivingJobId?: string;
}

export function ResearchHistoryCardGrid({ jobs, onSelect, onArchive, archivingJobId }: ResearchHistoryCardGridProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {jobs.map((job) => {
        const display = deriveHistoryDisplay(job);
        return (
          <div
            key={display.jobId}
            onClick={() => onSelect(display.jobId)}
            className="cursor-pointer rounded-panel border border-border bg-white p-4 hover:border-accent"
          >
            <div className="flex items-start justify-between gap-2">
              <p className="min-w-0 truncate text-sm font-medium text-primary" title={display.title}>
                {display.title}
              </p>
              <StatusBadge status={display.status} />
            </div>
            <p className="mt-1 text-xs text-secondary">
              {display.typeLabel}
              {display.archived ? ' · 已归档' : ''}
            </p>
            <p className="mt-2 text-xs text-secondary">
              质量 {display.qualityScore ?? '-'} · 文档 {display.docCount ?? '-'} · 来源 {display.sourceCount ?? '-'}
            </p>
            <div className="mt-3 flex items-center justify-between">
              <div className="min-w-0">
                <TruncatedText text={display.jobId} maxLength={18} middle className="text-xs text-secondary" />
                <p className="mt-1 text-xs text-secondary">{formatDateTime(display.updatedAt)}</p>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(display.jobId);
                  }}
                  className="rounded-control px-2 py-1 text-xs text-accent hover:bg-page"
                >
                  查看
                </button>
                <button
                  type="button"
                  disabled={archivingJobId === display.jobId}
                  onClick={(e) => {
                    e.stopPropagation();
                    onArchive(display.jobId, !display.archived);
                  }}
                  className="inline-flex items-center gap-1 rounded-control px-2 py-1 text-xs text-secondary hover:bg-page hover:text-primary disabled:opacity-50"
                  title={display.archived ? '恢复到默认历史列表' : '从默认列表隐藏；不会删除报告或证据'}
                >
                  {display.archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
                </button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
