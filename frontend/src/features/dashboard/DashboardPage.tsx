import { Link } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { StatusBadge } from '@/components/data/StatusBadge';
import { EmptyState } from '@/components/data/EmptyState';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import {
  useHealth,
  useKnowledgeTree,
  useJobs,
  useEvaluationReports,
  useModelConfig,
  useObservabilityEvents,
} from '@/hooks';
import { formatDateTime, formatNumber, formatPercent } from '@/utils/formatters';
import { ROUTES } from '@/router/routes';

export function DashboardPage() {
  const health = useHealth();
  const knowledgeTree = useKnowledgeTree();
  const jobs = useJobs();
  const events = useObservabilityEvents({ limit: 5 });
  const evaluations = useEvaluationReports(undefined, 5);
  const modelConfig = useModelConfig();

  const knowledgeBases = knowledgeTree.data?.knowledge_bases || [];
  const researchItems = knowledgeBases.flatMap((knowledgeBase) => [
    ...knowledgeBase.research_items,
    ...knowledgeBase.themes.flatMap((theme) => theme.research_items),
  ]);
  const readyCount = researchItems.filter((item) => item.index_status?.status === 'ready').length;
  const runningJobs = jobs.data?.jobs?.filter((j) => j.status === 'running' || j.status === 'queued') || [];
  const recentErrors = events.data?.events?.filter((e) => e.level === 'error' || e.level === 'warning') || [];

  return (
    <div className="space-y-8">
      <PageHeader title="总览" description="系统总览与下一步动作" />

      {health.isError && <ErrorPanel error={health.error} onRetry={() => health.refetch()} />}

      <div className="grid min-w-0 gap-8 lg:grid-cols-5">
        {/* Left: feature column */}
        <div className="min-w-0 space-y-6 lg:col-span-3">

          {/* Hero stat */}
          <section className="rounded-panel border border-border bg-surface p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-secondary">系统状态</p>
              <span className="h-2 w-2 rounded-full bg-success" />
            </div>
            <p className="mt-3 break-words font-display text-4xl font-bold leading-tight tracking-tight text-accent sm:text-5xl lg:text-6xl">
              {health.data?.service || '-'}
            </p>
            <p className="mt-2 font-mono text-xs text-faint">{health.data?.version || ''}</p>
            <div className="mt-5 grid grid-cols-3 gap-4 border-t border-border pt-4">
              <MiniStat label="知识库" value={formatNumber(knowledgeBases.length)} />
              <MiniStat label="索引就绪" value={formatNumber(readyCount)} />
              <MiniStat label="运行中任务" value={formatNumber(runningJobs.length)} />
            </div>
          </section>

          {/* Knowledge hierarchy feature */}
          <section>
            <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.15em] text-secondary">知识体系</p>
            {knowledgeBases.length ? (
              <div className="space-y-2">
                {knowledgeBases.map((knowledgeBase) => {
                  const scopedResearch = [
                    ...knowledgeBase.research_items,
                    ...knowledgeBase.themes.flatMap((theme) => theme.research_items),
                  ];
                  const allReady = scopedResearch.length > 0
                    && scopedResearch.every((item) => item.index_status?.status === 'ready');
                  return (
                    <Link
                      key={knowledgeBase.id}
                      to={ROUTES.KNOWLEDGE_BASES}
                      className="flex items-center justify-between rounded-panel border border-border bg-surface p-4 hover:border-primary"
                    >
                      <div>
                        <p className="font-display text-lg font-semibold text-primary">{knowledgeBase.title}</p>
                        <p className="text-xs text-faint">
                          {knowledgeBase.themes.length} 个主题 · {scopedResearch.length} 条调研
                        </p>
                      </div>
                      <StatusBadge status={allReady ? 'ready' : scopedResearch.length ? 'incomplete' : 'missing'} />
                    </Link>
                  );
                })}
              </div>
            ) : (
              <EmptyState title="暂无知识库" description="完成一次调研归档后会在这里形成知识体系。" />
            )}
          </section>

          {/* Models */}
          <section>
            <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.15em] text-secondary">模型接入</p>
            {modelConfig.data ? (
              <div className="grid gap-3 sm:grid-cols-3">
                <ModelMini label="对话" model={modelConfig.data.chat_model.model} ready={modelConfig.data.chat_model.base_url_configured} />
                <ModelMini label="复盘" model={modelConfig.data.recall_model.model} ready={modelConfig.data.recall_model.base_url_configured} />
                <ModelMini label="嵌入" model={modelConfig.data.embedding.model} ready={modelConfig.data.embedding.base_url_configured} />
              </div>
            ) : (
              <EmptyState title="模型状态不可用" description="进入模型页检查运行配置。" />
            )}
          </section>
        </div>

        {/* Right: stacked stream */}
        <div className="min-w-0 space-y-6 lg:col-span-2 lg:border-l lg:border-border lg:pl-8">
          <section>
            <div className="mb-2 flex items-center justify-between">
              <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-secondary">最近事件</p>
              <Link to={ROUTES.OBSERVABILITY} className="text-xs text-accent hover:underline">查看全部</Link>
            </div>
            {events.data?.events?.length ? (
              <div className="divide-y divide-border">
                {events.data.events.map((e, idx) => (
                  <div key={idx} className="py-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-primary">{e.event_type}</span>
                      <span className="font-mono text-[11px] text-faint">{formatDateTime(e.timestamp)}</span>
                    </div>
                    <p className="mt-0.5 text-xs text-secondary">{e.message}</p>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState title="暂无事件" />
            )}
          </section>

          <section>
            <div className="mb-2 flex items-center justify-between">
              <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-secondary">最近评估</p>
              <Link to={ROUTES.EVALUATIONS} className="text-xs text-accent hover:underline">查看评估</Link>
            </div>
            {evaluations.data?.reports?.length ? (
              <div className="space-y-3">
                {evaluations.data.reports.slice(0, 3).map((report) => (
                  <Link
                    key={report.report_id}
                    to={report.scope
                      ? `/evaluations?scope_type=${encodeURIComponent(report.scope.type)}&scope_id=${encodeURIComponent(report.scope.id)}`
                      : ROUTES.EVALUATIONS}
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="truncate font-mono text-secondary">{report.report_id}</span>
                      <span className={report.summary.failed ? 'font-display text-base font-semibold text-warning' : 'font-display text-base font-semibold text-success'}>
                        {formatPercent(report.summary.pass_rate)}
                      </span>
                    </div>
                    <div className="mt-1.5 h-[3px] bg-border">
                      <div
                        className={report.summary.failed ? 'h-[3px] bg-warning' : 'h-[3px] bg-success'}
                        style={{ width: `${Math.min(100, report.summary.pass_rate)}%` }}
                      />
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <EmptyState title="暂无评估" description="运行一次评估后会显示最近报告。" />
            )}
          </section>
        </div>
      </div>

      {recentErrors.length > 0 && (
        <section className="rounded-panel border border-danger/30 bg-danger/10 p-4">
          <div className="mb-2 flex items-center gap-2 text-danger">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-sm font-medium">最近异常</span>
          </div>
          <div className="space-y-1">
            {recentErrors.slice(0, 3).map((e, idx) => (
              <p key={idx} className="text-xs text-secondary">
                {e.event_type}: {e.message}
              </p>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <p className="font-display text-2xl font-semibold text-primary">{value}</p>
      <p className="text-[11px] text-secondary">{label}</p>
    </div>
  );
}

function ModelMini({ label, model, ready }: { label: string; model: string; ready: boolean }) {
  return (
    <div className="rounded-panel border border-border bg-surface p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-secondary">{label}</span>
        <StatusBadge status={ready ? 'ready' : 'missing'} />
      </div>
      <p className="truncate font-mono text-xs font-medium text-primary" title={model}>{model}</p>
    </div>
  );
}
