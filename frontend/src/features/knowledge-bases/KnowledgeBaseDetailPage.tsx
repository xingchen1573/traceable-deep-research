import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  ExternalLink,
  FileText,
  Play,
  MessageSquare,
  ClipboardCheck,
  Trash2,
} from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { KnowledgePipelineBar } from '@/components/data/KnowledgePipelineBar';
import { StatusBadge } from '@/components/data/StatusBadge';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { EmptyState } from '@/components/data/EmptyState';
import { Button } from '@/components/controls/Button';
import { ConfirmDialog } from '@/components/controls/FormControls';
import {
  useKnowledgeBase,
  useDocuments,
  useDocument,
  useSetActiveKnowledgeBase,
  useCreateIndexJob,
  useDeleteIndex,
} from '@/hooks';
import { useJobStore } from '@/stores';
import { formatBytes, formatNumber } from '@/utils/formatters';
import { ROUTES } from '@/router/routes';
import { getBaseUrl } from '@/api/toggle';
import { deriveKnowledgePipelineStatus } from './knowledgePipelineStatus';

export function KnowledgeBaseDetailPage() {
  const { topic } = useParams<{ topic: string }>();
  const decodedTopic = decodeURIComponent(topic || '');
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const addJob = useJobStore((s) => s.addJob);

  const kb = useKnowledgeBase(decodedTopic);
  const docs = useDocuments(decodedTopic);
  const doc = useDocument(decodedTopic, selectedDoc || '');
  const setActive = useSetActiveKnowledgeBase();
  const createIndex = useCreateIndexJob();
  const deleteIndex = useDeleteIndex();

  if (!decodedTopic) return <ErrorPanel error={new Error('缺少 topic 参数')} />;

  return (
    <div className="space-y-4">
      <PageHeader
        title={kb.data?.metadata?.title || decodedTopic}
        description={
          <Link to={ROUTES.KNOWLEDGE_BASES} className="flex items-center gap-1 text-link hover:underline">
            <ArrowLeft className="h-4 w-4" />
            返回知识库列表
          </Link>
        }
        actions={
          <>
            <Button variant="secondary" onClick={() => setActive.mutate({ topic: decodedTopic })}>
              设为当前
            </Button>
            <Link to={ROUTES.KNOWLEDGE_QA}>
              <Button variant="secondary" icon={undefined}>
                <MessageSquare className="mr-1.5 h-4 w-4" />
                问答
              </Button>
            </Link>
            <Link to={`/evaluations?topic=${encodeURIComponent(decodedTopic)}`}>
              <Button variant="secondary" icon={undefined}>
                <ClipboardCheck className="mr-1.5 h-4 w-4" />
                评估
              </Button>
            </Link>
          </>
        }
      />

      {kb.error && <ErrorPanel error={kb.error} onRetry={() => kb.refetch()} />}

      {kb.data && (
        <KnowledgePipelineBar
          status={deriveKnowledgePipelineStatus(kb.data)}
        />
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="space-y-4 lg:col-span-1">
          <div className="rounded-panel border border-border bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-primary">元数据</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-secondary">知识库标识</span>
                <span className="font-mono text-primary">{decodedTopic}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">元数据状态</span>
                <StatusBadge status={kb.data?.metadata_status || 'missing'} />
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">索引状态</span>
                <StatusBadge status={kb.data?.index_status || 'missing'} />
              </div>
              <div className="flex justify-between">
                <span className="text-secondary">文档数</span>
                <span className="text-primary">{formatNumber(kb.data?.document_count)}</span>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {kb.data?.index_status !== 'ready' ? (
                <Button
                  size="sm"
                  onClick={() => createIndex.mutate({ topic: decodedTopic }, { onSuccess: (res) => addJob(res.job_id) })}
                >
                  <Play className="mr-1.5 h-4 w-4" />
                  构建索引
                </Button>
              ) : (
                <Button size="sm" variant="danger" onClick={() => setShowDeleteConfirm(true)}>
                  <Trash2 className="mr-1.5 h-4 w-4" />
                  删除索引
                </Button>
              )}
            </div>
          </div>

          <div className="rounded-panel border border-border bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-primary">文档列表</h3>
            {docs.data?.documents?.length ? (
              <div className="divide-y divide-border">
                {docs.data.documents.map((d) => (
                  <button
                    key={d.document_id}
                    onClick={() => setSelectedDoc(d.document_id)}
                    className={`flex w-full items-center gap-2 py-2 text-left text-sm ${
                      selectedDoc === d.document_id ? 'text-accent' : 'text-primary hover:text-accent'
                    }`}
                  >
                    <FileText className="h-4 w-4 flex-shrink-0" />
                    <span className="truncate">{d.title}</span>
                    <span className="ml-auto text-xs text-secondary">{formatBytes(d.bytes)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <EmptyState title="无文档" />
            )}
          </div>
        </section>

        <section className="rounded-panel border border-border bg-white p-4 lg:col-span-2">
          {selectedDoc ? (
            doc.isLoading ? (
              <div className="text-secondary">加载中...</div>
            ) : doc.data ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-base font-semibold text-primary">{doc.data.title}</h3>
                  <span className="text-xs text-secondary">{doc.data.path}</span>
                </div>
                {Boolean(doc.data.frontmatter?.source_url || doc.data.frontmatter?.source_job_id) && (
                  <div className="flex flex-wrap items-center gap-3 border-b border-border pb-3 text-sm">
                    {typeof doc.data.frontmatter?.source_url === 'string' && doc.data.frontmatter.source_url && (
                      <a
                        href={doc.data.frontmatter.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="truncate text-link hover:underline"
                      >
                        {doc.data.frontmatter.source_url}
                      </a>
                    )}
                    {typeof doc.data.frontmatter?.source_job_id === 'string' && doc.data.frontmatter.source_job_id && (
                      <a
                        href={`${getBaseUrl()}/api/v1/research/jobs/${encodeURIComponent(
                          doc.data.frontmatter.source_job_id
                        )}/evidence-reader`}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex shrink-0 items-center gap-1 rounded-control border border-border px-2 py-1 text-xs text-secondary hover:bg-page hover:text-primary"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                        查看证据阅读版
                      </a>
                    )}
                  </div>
                )}
                <MarkdownPreview content={doc.data.content} />
              </div>
            ) : (
              <EmptyState title="无法加载文档" />
            )
          ) : (
            <EmptyState title="选择文档预览" description="点击左侧文档查看 Markdown 内容。" />
          )}
        </section>
      </div>

      <ConfirmDialog
        open={showDeleteConfirm}
        title="删除索引"
        message={`确认删除 ${decodedTopic} 的 LightRAG 索引？Markdown 文档不会被删除。`}
        confirmText="删除"
        onConfirm={() => {
          deleteIndex.mutate(decodedTopic);
          setShowDeleteConfirm(false);
        }}
        onCancel={() => setShowDeleteConfirm(false)}
      />
    </div>
  );
}
