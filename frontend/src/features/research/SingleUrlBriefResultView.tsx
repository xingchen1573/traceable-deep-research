import { ExternalLink, Save } from 'lucide-react';
import { Button } from '@/components/controls/Button';
import { FormField } from '@/components/controls/FormControls';
import { StatusBadge } from '@/components/data/StatusBadge';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { ResearchJob } from '@/types';

interface SingleUrlBriefResultViewProps {
  job: ResearchJob;
  evidenceReaderUrl: string;
  saveTopic: string;
  onSaveTopicChange: (value: string) => void;
  onSave: () => void;
  isSaving: boolean;
}

export function SingleUrlBriefResultView({
  job,
  evidenceReaderUrl,
  saveTopic,
  onSaveTopicChange,
  onSave,
  isSaving,
}: SingleUrlBriefResultViewProps) {
  const result = job.result;
  const title = result?.documents?.[0]?.title || result?.source_title || job.job_id;
  const markdown = result?.brief_markdown || result?.documents?.[0]?.content || '';
  const sourceUrl = result?.url || '';
  const saveResult = result?.save_result;

  return (
    <section className="space-y-4">
      <div className="rounded-panel border border-border bg-white p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="truncate text-base font-semibold text-primary">{title}</h3>
              <StatusBadge status={job.status} />
            </div>
            {sourceUrl && (
              <a
                href={sourceUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-1 inline-block truncate text-xs text-link hover:underline"
              >
                {sourceUrl}
              </a>
            )}
          </div>
          {evidenceReaderUrl && (
            <a
              href={evidenceReaderUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex shrink-0 items-center gap-1 rounded-control border border-border px-3 py-2 text-sm text-secondary hover:bg-page hover:text-primary"
            >
              <ExternalLink className="h-4 w-4" />
              查看证据阅读版
            </a>
          )}
        </div>
        {job.error && (
          <p className="mt-3 rounded-control bg-red-50 p-3 text-sm text-danger">
            {job.error.message || '生成失败'}
          </p>
        )}
        {markdown && (
          <div className="mt-4 border-t border-border pt-4">
            <MarkdownPreview content={markdown} />
          </div>
        )}
      </div>

      {job.status === 'succeeded' && (
        <div className="rounded-panel border border-border bg-white p-4">
          {saveResult ? (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-primary">
                已保存到知识库（topic：<span className="font-mono">{saveResult.topic}</span>）
              </p>
              <div className="flex items-center gap-2">
                <FormField label="更新到 topic">
                  <input
                    type="text"
                    value={saveTopic}
                    onChange={(e) => onSaveTopicChange(e.target.value)}
                    placeholder={saveResult.topic}
                    className="w-48 rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
                  />
                </FormField>
                <Button variant="secondary" size="sm" onClick={onSave} isLoading={isSaving} disabled={!saveTopic.trim()} className="self-end">
                  更新
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap items-end gap-3">
              <FormField label="存入知识库（topic，必填）">
                <input
                  type="text"
                  value={saveTopic}
                  onChange={(e) => onSaveTopicChange(e.target.value)}
                  placeholder="落库目录名"
                  className="w-64 rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
                />
              </FormField>
              <Button onClick={onSave} isLoading={isSaving} disabled={!saveTopic.trim()} className="self-end">
                <Save className="mr-1.5 h-4 w-4" />
                存入知识库
              </Button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
