import { useEffect, useMemo, useState } from 'react';
import { Network, Play, Trash2, Database } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { buildKnowledgeScopeOptions, KnowledgeScopeSelector } from '@/components/data/KnowledgeScopeSelector';
import { StatusBadge } from '@/components/data/StatusBadge';
import { Button } from '@/components/controls/Button';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { EmptyState, LoadingState } from '@/components/data/EmptyState';
import { KnowledgeGraphView } from '@/components/data/KnowledgeGraphView';
import { ConfirmDialog } from '@/components/controls/FormControls';
import {
  useKnowledgeTree,
  useKnowledgeScopeIndex,
  useKnowledgeScopeGraph,
  useKnowledgeScopeVectors,
  useRebuildKnowledgeScopeIndexes,
  useDeleteResearchIndex,
} from '@/hooks';
import { formatNumber } from '@/utils/formatters';

export function IndexGraphPage() {
  const knowledgeTree = useKnowledgeTree();
  const scopeOptions = useMemo(() => buildKnowledgeScopeOptions(knowledgeTree.data), [knowledgeTree.data]);
  const [topic, setTopic] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const selectedScope = scopeOptions.find((option) => option.key === topic);

  const index = useKnowledgeScopeIndex(selectedScope?.type, selectedScope?.id);
  const graph = useKnowledgeScopeGraph(selectedScope?.type, selectedScope?.id);
  const vectors = useKnowledgeScopeVectors(selectedScope?.type, selectedScope?.id);
  const rebuildIndexes = useRebuildKnowledgeScopeIndexes();
  const deleteIndex = useDeleteResearchIndex();

  useEffect(() => {
    if (!topic && scopeOptions.length) {
      const firstIndexedScope = scopeOptions.find((option) => option.researchIds.length > 0) || scopeOptions[0];
      setTopic(firstIndexedScope.key);
    }
  }, [scopeOptions, topic]);

  return (
    <div className="space-y-4">
      <PageHeader title="索引与图谱" description="管理 LightRAG 索引、向量和图谱状态" />

      <div className="max-w-xl">
        <KnowledgeScopeSelector
          value={topic || null}
          options={scopeOptions}
          onChange={(option) => setTopic(option.key)}
        />
      </div>

      {!selectedScope ? (
        <EmptyState title="选择知识范围" description="先选择知识库、主题或调研条目查看索引状态。" />
      ) : (
        <>
          {index.error && <ErrorPanel error={index.error} onRetry={() => index.refetch()} />}

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              icon={<Network className="h-5 w-5 text-accent" />}
              label="索引状态"
              value={index.data ? <StatusBadge status={index.data.index_status} /> : '-'}
            />
            <StatCard
              icon={<Database className="h-5 w-5 text-link" />}
              label="向量块"
              value={formatNumber(vectors.data?.chunks)}
            />
            <StatCard
              icon={<Network className="h-5 w-5 text-success" />}
              label="图谱节点 / 边"
              value={`${formatNumber(graph.data?.nodes)} / ${formatNumber(graph.data?.edges)}`}
            />
            <StatCard
              icon={<Database className="h-5 w-5 text-warning" />}
              label="向量实体"
              value={formatNumber(vectors.data?.entities)}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="rounded-panel border border-border bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-primary">索引配置</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-secondary">嵌入模型</span>
                  <span className="text-primary">{index.data?.embedding_model || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary">嵌入维度</span>
                  <span className="text-primary">{formatNumber(index.data?.embedding_dim)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary">工作区</span>
                  <span className="font-mono text-xs text-primary">{index.data?.workspace || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary">索引时间</span>
                  <span className="text-primary">{index.data?.indexed_at || '-'}</span>
                </div>
              </div>
            </section>

            <section className="rounded-panel border border-border bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-primary">向量统计</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-secondary">块大小</span>
                  <span className="text-primary">{formatNumber(vectors.data?.chunk_size)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary">块重叠</span>
                  <span className="text-primary">{formatNumber(vectors.data?.chunk_overlap_size)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary">关系</span>
                  <span className="text-primary">{formatNumber(vectors.data?.relationships)}</span>
                </div>
              </div>
            </section>
          </div>

          <section className="rounded-panel border border-border bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-primary">知识图谱</h3>
            {graph.isLoading ? (
              <LoadingState message="加载图谱数据…" />
            ) : graph.error ? (
              <ErrorPanel error={graph.error} onRetry={() => graph.refetch()} />
            ) : graph.data ? (
              <KnowledgeGraphView scopeType={selectedScope.type} scopeId={selectedScope.id} graph={graph.data} />
            ) : (
              <EmptyState title="暂无图谱数据" description="请先构建 LightRAG 索引。" />
            )}
          </section>

          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() => rebuildIndexes.mutate(selectedScope.researchIds)}
              isLoading={rebuildIndexes.isPending}
              disabled={selectedScope.researchIds.length === 0}
            >
              <Play className="mr-1.5 h-4 w-4" />
              {selectedScope.type === 'research' ? '构建 / 刷新索引' : `刷新范围内索引（${selectedScope.researchIds.length}）`}
            </Button>
            {selectedScope.type === 'research' && (
              <Button
                variant="danger"
                onClick={() => setShowDeleteConfirm(true)}
                disabled={index.data?.index_status === 'missing'}
              >
                <Trash2 className="mr-1.5 h-4 w-4" />
                删除索引
              </Button>
            )}
          </div>

          <ConfirmDialog
            open={showDeleteConfirm}
            title="删除索引"
            message={`确认删除“${selectedScope.title}”的 LightRAG 索引？Markdown 文档不会被删除。`}
            confirmText="删除"
            onConfirm={() => {
              deleteIndex.mutate(selectedScope.id);
              setShowDeleteConfirm(false);
            }}
            onCancel={() => setShowDeleteConfirm(false)}
          />
        </>
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="rounded-panel border border-border bg-white p-4">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-secondary">{label}</span>
      </div>
      <div className="mt-2 text-xl font-semibold text-primary">{value}</div>
    </div>
  );
}
