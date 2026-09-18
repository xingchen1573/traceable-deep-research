import { Archive, ArchiveRestore } from 'lucide-react';
import { StatusBadge } from '@/components/data/StatusBadge';
import { TruncatedText } from '@/components/data/TruncatedText';
import { formatDateTime } from '@/utils/formatters';
import { ResearchJob } from '@/types';
import { deriveHistoryDisplay } from './researchHistory';

interface ResearchHistoryListProps {
  jobs: ResearchJob[];
  onSelect: (jobId: string) => void;
  onArchive: (jobId: string, archived: boolean) => void;
  archivingJobId?: string;
}

export function ResearchHistoryList({ jobs, onSelect, onArchive, archivingJobId }: ResearchHistoryListProps) {
  return (
    <div className="overflow-hidden rounded-panel border border-border bg-white">
      <table className="min-w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-secondary">
            <th className="px-3 py-2 font-medium">主题</th>
            <th className="px-3 py-2 font-medium">类型</th>
            <th className="px-3 py-2 font-medium">状态</th>
            <th className="px-3 py-2 font-medium">质量/文档/来源</th>
            <th className="px-3 py-2 font-medium">更新时间</th>
            <th className="px-3 py-2 font-medium">任务 ID</th>
            <th className="px-3 py-2 font-medium">操作</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => {
            const display = deriveHistoryDisplay(job);
            return (
              <tr
                key={display.jobId}
                onClick={() => onSelect(display.jobId)}
                className="cursor-pointer border-b border-border last:border-b-0 hover:bg-page"
              >
                <td className="max-w-[280px] px-3 py-2">
                  <p className="truncate font-medium text-primary" title={display.title}>
                    {display.title}
                  </p>
                </td>
                <td className="px-3 py-2 text-secondary">{display.typeLabel}</td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-1.5">
                    <StatusBadge status={display.status} />
                    {display.archived && <span className="text-xs text-secondary">已归档</span>}
                  </div>
                </td>
                <td className="px-3 py-2 text-xs text-secondary">
                  {display.qualityScore ?? '-'} / {display.docCount ?? '-'} / {display.sourceCount ?? '-'}
                </td>
                <td className="px-3 py-2 text-xs text-secondary">{formatDateTime(display.updatedAt)}</td>
                <td className="px-3 py-2">
                  <TruncatedText text={display.jobId} maxLength={16} middle className="text-xs text-secondary" />
                </td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-1">
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
                      title={display.archived ? '恢复到默认历史列表' : '从默认列表隐藏；不会删除报告或证据'}
                      onClick={(e) => {
                        e.stopPropagation();
                        onArchive(display.jobId, !display.archived);
                      }}
                      className="inline-flex items-center gap-1 rounded-control px-2 py-1 text-xs text-secondary hover:bg-page hover:text-primary disabled:opacity-50"
                    >
                      {display.archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
                      {display.archived ? '取消归档' : '归档'}
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
