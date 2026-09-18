import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ChevronDown, ChevronUp, Clipboard, ExternalLink, GitCompare, Maximize2, Play, Plus, RotateCcw, SlidersHorizontal, Sparkles, Trash2, X } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { buildKnowledgeScopeOptions, KnowledgeScopeSelector } from '@/components/data/KnowledgeScopeSelector';
import { FilterSelect, ConfirmDialog, FormField } from '@/components/controls/FormControls';
import { Button } from '@/components/controls/Button';
import { DataTable } from '@/components/data/DataTable';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { EmptyState } from '@/components/data/EmptyState';
import {
  useKnowledgeTree,
  useCreateKnowledgeScopeEvaluationJob,
  useKnowledgeScopeEvaluationCases,
  useUpsertKnowledgeScopeEvaluationCase,
  useDeleteKnowledgeScopeEvaluationCase,
  useCreateResearchProbeEvaluationJob,
  useCreateSingleUrlBriefEvaluationJob,
  useEvaluationReports,
  useEvaluationReport,
  useEvaluationCases,
  useUpsertEvaluationCase,
  useDeleteEvaluationCase,
  useEvaluationReportComparison,
  useResearchProbeReports,
  useResearchProbeReport,
} from '@/hooks';
import { useToast } from '@/components/feedback/Toast';
import { formatDateTime, formatPercent } from '@/utils/formatters';
import { EvalCase, EvalCaseListResponse, EvalType, EvaluationCase, EvaluationReportMeta, KnowledgeQaCaseGenerationJob, ResearchProbeEvaluationRow, ResearchProbeReportMeta } from '@/types';
import { QUERY_MODE_OPTIONS, queryBackendLabel, queryModeLabel } from '@/utils/knowledgeQaLabels';

const EVAL_TYPE_OPTIONS: Array<{ value: EvalType; label: string }> = [
  { value: 'knowledge_qa', label: '知识库问答评测' },
  { value: 'research_probe', label: '调研全流程评测（E2E）' },
  { value: 'single_url_brief', label: '单链接简报评测（URL）' },
];

const RESEARCH_KIND_OPTIONS = [
  { value: 'single_explain', label: '单主题解释' },
  { value: 'single_deep', label: '单主题深挖' },
  { value: 'dual_compare', label: '双对象对比' },
  { value: 'multi_compare', label: '多对象对比' },
  { value: 'uncertainty', label: '争议与不确定性' },
];

interface ResearchProbeCaseDraft {
  id: string;
  caseVersion: string;
  kind: string;
  title: string;
  userRequest: string;
  strategy: 'standard' | 'hv_analysis';
  depth: 'normal' | 'deep';
  coreIntents: string;
  optionalTopics: string;
  scopeNotes: string;
}

const EMPTY_RESEARCH_PROBE_DRAFT: ResearchProbeCaseDraft = {
  id: '',
  caseVersion: '2',
  kind: 'single_explain',
  title: '',
  userRequest: '',
  strategy: 'standard',
  depth: 'normal',
  coreIntents: '',
  optionalTopics: '',
  scopeNotes: '',
};

const CASE_INPUT_CLASS = 'w-full rounded-control border border-border bg-surface px-3 py-2 text-sm text-primary placeholder:text-faint focus:border-accent focus:outline-none';

function linesToList(value: string) {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean);
}

function researchCaseToDraft(item: EvalCase): ResearchProbeCaseDraft {
  return {
    id: item.id || '',
    caseVersion: String(item.case_version || 2),
    kind: String(item.kind || 'single_explain'),
    title: String(item.title || ''),
    userRequest: String(item.user_request || ''),
    strategy: item.strategy === 'hv_analysis' ? 'hv_analysis' : 'standard',
    depth: item.depth === 'deep' ? 'deep' : 'normal',
    coreIntents: (item.core_intents || []).join('\n'),
    optionalTopics: (item.optional_topics || []).join('\n'),
    scopeNotes: String(item.scope_notes || ''),
  };
}

function defaultStrategyForKind(kind: string): 'standard' | 'hv_analysis' {
  return kind === 'dual_compare' || kind === 'multi_compare' || kind === 'uncertainty'
    ? 'hv_analysis'
    : 'standard';
}

function SelectionCheckbox({
  checked,
  indeterminate = false,
  label,
  onChange,
}: {
  checked: boolean;
  indeterminate?: boolean;
  label: string;
  onChange: () => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);
  return (
    <input
      ref={ref}
      type="checkbox"
      aria-label={label}
      checked={checked}
      onChange={onChange}
      onClick={(event) => event.stopPropagation()}
      className="h-4 w-4 cursor-pointer accent-accent"
    />
  );
}

const SEMANTIC_METRICS = [
  ['correctness', '答案正确性'],
  ['completeness', '答案完整性'],
  ['faithfulness', '答案忠实度'],
  ['relevance', '答案相关性'],
  ['boundary_awareness', '边界意识'],
  ['citation_support', '引用支撑度'],
] as const;

const RETRIEVAL_METRICS = [
  ['evidence_recall', '标准证据命中率'],
  ['point_coverage', '参考要点覆盖度'],
  ['hit_at_3', '前 3 条关键证据命中率'],
  ['context_relevance', '上下文相关性'],
  ['noise_ratio', '上下文噪声比例'],
  ['boundary_safety', '证据边界安全度'],
] as const;

const RETRIEVAL_LABELS = {
  supporting: '直接支撑',
  related: '背景相关',
  noise: '无关噪声',
  misleading: '可能误导',
} as const;

export function EvaluationsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const knowledgeTree = useKnowledgeTree();
  const scopeOptions = useMemo(() => buildKnowledgeScopeOptions(knowledgeTree.data), [knowledgeTree.data]);
  const [evalType, setEvalType] = useState<EvalType>('knowledge_qa');
  const [topic, setTopic] = useState('');
  const [mode, setMode] = useState('mix');
  const [selectedCaseIds, setSelectedCaseIds] = useState<string[]>([]);
  const [caseFilter, setCaseFilter] = useState('failed');
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  const [baselineReport, setBaselineReport] = useState('');
  const [showReportModal, setShowReportModal] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const { showToast } = useToast();
  const selectedScope = scopeOptions.find((option) => option.key === topic);
  const evaluationScopeKey = selectedScope ? `scope:${selectedScope.type}:${selectedScope.id}` : '';

  const createKnowledgeQaEval = useCreateKnowledgeScopeEvaluationJob();
  const createResearchProbeEval = useCreateResearchProbeEvaluationJob();
  const createSingleUrlBriefEval = useCreateSingleUrlBriefEvaluationJob();
  const reports = useEvaluationReports(evalType === 'knowledge_qa' ? evaluationScopeKey : undefined);
  const report = useEvaluationReport(selectedReport || '');
  const cases = useKnowledgeScopeEvaluationCases(selectedScope?.type, selectedScope?.id);
  const comparison = useEvaluationReportComparison(selectedReport || '', baselineReport);

  useEffect(() => {
    if (!topic && scopeOptions.length) {
      const queryScopeType = searchParams.get('scope_type');
      const queryScopeId = searchParams.get('scope_id');
      const requested = queryScopeType && queryScopeId
        ? scopeOptions.find((option) => option.type === queryScopeType && option.id === queryScopeId)
        : undefined;
      setTopic((requested || scopeOptions.find((option) => option.researchIds.length > 0) || scopeOptions[0]).key);
    }
  }, [scopeOptions, searchParams, topic]);

  useEffect(() => {
    if (selectedScope) setSearchParams({ scope_type: selectedScope.type, scope_id: selectedScope.id });
  }, [selectedScope, setSearchParams]);

  useEffect(() => {
    if (!selectedReport && reports.data?.reports?.length) {
      setSelectedReport(reports.data.reports[0].report_id);
    }
  }, [reports.data?.reports, selectedReport]);

  useEffect(() => {
    const reportRows = reports.data?.reports || [];
    if (!selectedReport || baselineReport === selectedReport || !reportRows.some((item) => item.report_id === baselineReport)) {
      setBaselineReport(reportRows.find((item) => item.report_id !== selectedReport)?.report_id || '');
    }
  }, [baselineReport, reports.data?.reports, selectedReport]);

  useEffect(() => {
    setSelectedCaseIds([]);
  }, [evalType]);

  const reportCases = report.data?.cases || [];
  const visibleCases = reportCases.filter(
    (item) => caseFilter === 'all'
      || !item.passed
      || item.retrieval_evaluation?.quality_status === 'needs_review'
      || item.semantic_evaluation?.quality_status === 'needs_review'
  );
  const hasStructuredCases = Boolean(report.data?.cases);
  const hasReportActions = Boolean(report.data?.recommended_actions?.length);

  const copyCaseQuestion = async (item: EvaluationCase) => {
    if (!item.question) return;
    await navigator.clipboard?.writeText(item.question);
    showToast('已复制失败用例问题', 'success');
  };

  const runEvaluation = () => {
    if (evalType === 'knowledge_qa' && selectedScope) {
      createKnowledgeQaEval.mutate({ scope_type: selectedScope.type, scope_id: selectedScope.id, mode: mode as any });
    } else if (evalType === 'research_probe') {
      createResearchProbeEval.mutate({ case_ids: selectedCaseIds });
    } else {
      createSingleUrlBriefEval.mutate({ case_ids: selectedCaseIds });
    }
  };

  const isRunPending = createKnowledgeQaEval.isPending || createResearchProbeEval.isPending || createSingleUrlBriefEval.isPending;
  const canRun = evalType === 'knowledge_qa' ? Boolean(topic) : selectedCaseIds.length > 0;

  return (
    <div className="space-y-4">
      <PageHeader title="评估" description="用可重复的评测集检查知识库问答、调研全流程和单链接简报质量" />

      <div className="flex flex-col gap-3 rounded-panel border border-border bg-white p-4 sm:flex-row sm:items-center">
        <FilterSelect
          label="评测类型"
          value={evalType}
          onChange={(e) => setEvalType(e.target.value as EvalType)}
          options={EVAL_TYPE_OPTIONS}
        />
        {evalType === 'knowledge_qa' && (
          <>
            <div className="min-w-[18rem] max-w-xl flex-1">
              <KnowledgeScopeSelector
                value={topic || null}
                options={scopeOptions}
                onChange={(option) => setTopic(option.key)}
              />
            </div>
            <button
              type="button"
              onClick={() => setShowAdvanced((value) => !value)}
              className="inline-flex items-center gap-1.5 text-sm text-secondary hover:text-primary"
            >
              <SlidersHorizontal className="h-4 w-4" />
              高级设置
              {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          </>
        )}
        {evalType !== 'knowledge_qa' && (
          <span className="text-xs text-secondary">
            {selectedCaseIds.length ? `本次将运行 ${selectedCaseIds.length} 条用例` : '请在下方选择要运行的用例'}
          </span>
        )}
        <Button onClick={runEvaluation} isLoading={isRunPending} disabled={!canRun}>
          <Play className="mr-1.5 h-4 w-4" />
          运行评估
        </Button>
      </div>

      {evalType === 'knowledge_qa' && showAdvanced && (
        <div className="flex flex-wrap items-center gap-4 border-y border-border bg-page px-4 py-3">
          <FilterSelect
            label="检索模式"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            options={QUERY_MODE_OPTIONS}
          />
          <span className="text-xs text-secondary">分层评测固定运行 KnowledgeScope 聚合检索。</span>
        </div>
      )}

      {evalType !== 'knowledge_qa' && (
        <EvalCaseManager
          evalType={evalType}
          selectable
          selectedCaseIds={selectedCaseIds}
          onSelectCases={setSelectedCaseIds}
          onToggleCase={(id) =>
            setSelectedCaseIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]))
          }
        />
      )}

      {evalType === 'research_probe' && <ResearchProbeReportPanel />}

      {evalType === 'knowledge_qa' && (
        <>
          <KnowledgeQaCaseSetPanel
            cases={cases.data}
            canGenerate={false}
          />
          {reports.error && <ErrorPanel error={reports.error} onRetry={() => reports.refetch()} />}

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="rounded-panel border border-border bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-primary">评估报告</h3>
              <DataTable<EvaluationReportMeta>
                columns={[
                  {
                    key: 'report_id',
                    header: '报告',
                    render: (r) => <span className="font-mono text-xs">{r.report_id}</span>,
                  },
                  {
                    key: 'pass_rate',
                    header: '基础通过率',
                    render: (r) => <span className="text-success">{formatPercent(r.summary.pass_rate)}</span>,
                  },
                  {
                    key: 'failed',
                    header: '失败',
                    render: (r) => <span className={r.summary.failed ? 'text-danger' : 'text-secondary'}>{r.summary.failed || 0}</span>,
                  },
                  {
                    key: 'evidence_recall',
                    header: '证据命中',
                    render: (r) => {
                      const value = r.summary.retrieval_quality?.metrics?.evidence_recall;
                      return <span className="text-xs text-secondary">{value == null ? '未评' : `${Math.round(value * 100)}%`}</span>;
                    },
                  },
                  {
                    key: 'faithfulness',
                    header: '答案忠实度',
                    render: (r) => {
                      const value = r.summary.semantic_quality?.scores?.faithfulness;
                      return <span className="text-xs text-secondary">{value == null ? '未评' : `${Math.round(value * 100)}%`}</span>;
                    },
                  },
                  {
                    key: 'created_at',
                    header: '时间',
                    render: (r) => <span className="text-xs text-secondary">{formatDateTime(r.created_at)}</span>,
                  },
                ]}
                data={reports.data?.reports || []}
                keyExtractor={(r) => r.report_id}
                onRowClick={(r) => setSelectedReport(r.report_id)}
                loading={reports.isLoading}
                emptyMessage="暂无评估报告"
              />
            </section>

            <section className="rounded-panel border border-border bg-white p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-primary">报告详情</h3>
                {report.data?.run_id || report.data?.job_id ? (
                  <Link
                    to={report.data.run_id
                      ? `/observability?run_id=${encodeURIComponent(report.data.run_id)}`
                      : `/observability?job_id=${encodeURIComponent(report.data.job_id || '')}`}
                    className="inline-flex items-center gap-1 text-xs text-link hover:underline"
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    查看运行事件
                  </Link>
                ) : report.data ? (
                  <span className="rounded-control border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">
                    该历史报告未关联观测任务
                  </span>
                ) : null}
              </div>
              {selectedReport ? (
                report.isLoading ? (
                  <div className="text-secondary">加载中...</div>
                ) : report.data ? (
                  <div className="max-h-[68vh] space-y-4 overflow-y-auto pr-1">
                    <div className="border-y border-border py-3 text-xs text-secondary">
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                        <span>评测集版本：{report.data.eval_set_version ? `v${report.data.eval_set_version}` : '历史报告未记录'}</span>
                        <span>
                          问答后端：{queryBackendLabel(report.data.backend || (report.data.scope ? 'knowledge_scope' : undefined))}
                        </span>
                        <span>检索策略：{queryModeLabel(report.data.mode)}</span>
                        {report.data.knowledge_content_stale && <span className="text-amber-900">运行时知识库内容与评测集基准不一致</span>}
                      </div>
                    </div>

                    {(reports.data?.reports.length || 0) > 1 && (
                      <div className="rounded-control border border-border p-3">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="flex items-center gap-1.5 text-sm font-medium text-primary">
                              <GitCompare className="h-4 w-4" />
                              历史报告对比
                            </p>
                            <p className="mt-1 text-xs text-secondary">只比较同一知识库；配置变化会单独提示，避免把不同基准直接当成质量趋势。</p>
                          </div>
                          <FilterSelect
                            label="对比基准"
                            value={baselineReport}
                            onChange={(e) => setBaselineReport(e.target.value)}
                            options={(reports.data?.reports || [])
                              .filter((item) => item.report_id !== selectedReport)
                              .map((item) => ({ value: item.report_id, label: formatDateTime(item.created_at) }))}
                          />
                        </div>
                        {comparison.data && (
                          <div className="mt-3">
                            {comparison.data.configuration_changed && (
                              <p className="mb-2 text-xs text-amber-900">两次运行的评测集、知识库内容或检索配置不同，变化值仅供诊断。</p>
                            )}
                            <div className="grid gap-2 sm:grid-cols-4">
                              {comparison.data.metrics.filter((item) => item.delta != null).map((item) => (
                                <Metric
                                  key={item.label}
                                  label={item.label}
                                  value={`${item.current == null ? '未评' : formatComparisonValue(item.key, item.current)} (${formatDelta(item.key, item.delta)})`}
                                  tone={item.delta != null && item.delta < 0 ? 'danger' : undefined}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {report.data.summary && (
                      <div className="grid gap-2 sm:grid-cols-4">
                        <Metric label="基础规则通过率" value={formatPercent(report.data.summary.pass_rate)} tone="success" />
                        <Metric label="总用例" value={report.data.summary.total} />
                        <Metric label="失败" value={report.data.summary.failed} tone="danger" />
                        <Metric label="平均分" value={report.data.summary.average_score} />
                      </div>
                    )}

                    {report.data.summary?.semantic_quality && (
                      <div className="border-y border-border py-3">
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <p className="text-sm font-semibold text-primary">回答质量语义评测</p>
                            <p className="mt-0.5 text-xs text-secondary">
                              已评 {report.data.summary.semantic_quality.judged_cases} 题
                              {report.data.summary.semantic_quality.unavailable_cases
                                ? `，${report.data.summary.semantic_quality.unavailable_cases} 题未完成`
                                : ''}
                              {`，${report.data.summary.semantic_quality.needs_review_cases} 题需复核`}
                            </p>
                          </div>
                          <span className="text-xs text-secondary">
                            {report.data.summary.semantic_quality.status === 'completed'
                              ? '已完成'
                              : report.data.summary.semantic_quality.status === 'partial'
                                ? '部分完成'
                                : report.data.summary.semantic_quality.status === 'unavailable'
                                  ? '模型不可用'
                                  : '不适用'}
                          </span>
                        </div>
                        <div className="grid gap-2 sm:grid-cols-3">
                          {SEMANTIC_METRICS.map(([key, label]) => (
                            <Metric
                              key={key}
                              label={label}
                              value={`${Math.round((report.data?.summary?.semantic_quality?.scores?.[key] || 0) * 100)}%`}
                            />
                          ))}
                        </div>
                      </div>
                    )}

                    {report.data.summary?.retrieval_quality && (
                      <div className="border-y border-border py-3">
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <p className="text-sm font-semibold text-primary">知识库问答检索质量</p>
                            <p className="mt-0.5 text-xs text-secondary">
                              已评 {report.data.summary.retrieval_quality.evaluated_cases} 题
                              {report.data.summary.retrieval_quality.unavailable_cases
                                ? `，${report.data.summary.retrieval_quality.unavailable_cases} 题没有检索快照`
                                : ''}
                              {report.data.summary.retrieval_quality.semantic_unavailable_cases
                                ? `，${report.data.summary.retrieval_quality.semantic_unavailable_cases} 题只完成确定性检查`
                                : ''}
                              {`，${report.data.summary.retrieval_quality.needs_review_cases} 题需复核`}
                              {report.data.summary.retrieval_quality.advisory_cases
                                ? `，${report.data.summary.retrieval_quality.advisory_cases} 题有排序或效率建议`
                                : ''}
                            </p>
                          </div>
                          <span className="text-xs text-secondary">
                            {report.data.summary.retrieval_quality.status === 'completed'
                              ? '已完成'
                              : report.data.summary.retrieval_quality.status === 'partial'
                                ? '部分完成'
                                : report.data.summary.retrieval_quality.status === 'unavailable'
                                  ? '不可用'
                                  : '不适用'}
                          </span>
                        </div>
                        <div className="grid gap-2 sm:grid-cols-3">
                          {RETRIEVAL_METRICS.map(([key, label]) => {
                            const value = report.data?.summary?.retrieval_quality?.metrics?.[key];
                            return (
                              <Metric
                                key={key}
                                label={label}
                                value={value == null ? '未评' : `${Math.round(Number(value) * 100)}%`}
                              />
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {hasReportActions ? (
                      <div className="rounded-control border border-border bg-page p-3">
                        <p className="mb-2 text-xs font-medium text-secondary">建议动作</p>
                        <div className="space-y-1">
                          {report.data.recommended_actions?.map((action) => (
                            <p key={action} className="text-sm text-primary">{action}</p>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="rounded-control border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                        该历史报告未保留结构化建议动作；可查看下方 Markdown 报告。
                      </div>
                    )}

                    <div className="rounded-control border border-border">
                      <div className="flex items-center justify-between border-b border-border px-3 py-2">
                        <span className="text-sm font-medium text-primary">需复核用例回流</span>
                        <FilterSelect
                          value={caseFilter}
                          onChange={(e) => setCaseFilter(e.target.value)}
                          options={[
                            { value: 'failed', label: '失败或需复核' },
                            { value: 'all', label: '全部用例' },
                          ]}
                        />
                      </div>
                      {!hasStructuredCases ? (
                        <div className="p-3 text-sm text-amber-900">
                          该历史报告未保留结构化用例明细；可查看下方 Markdown 报告。
                        </div>
                      ) : visibleCases.length ? (
                        <div className="divide-y divide-border">
                          {visibleCases.map((item) => (
                            <div key={item.id} className="p-3">
                              <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                  <div className="flex flex-wrap items-center gap-2">
                                    <span className="font-mono text-xs text-secondary">{item.id}</span>
                                    <span className={item.passed && item.retrieval_evaluation?.quality_status !== 'needs_review' && item.semantic_evaluation?.quality_status !== 'needs_review' ? 'text-xs text-success' : 'text-xs text-danger'}>
                                      {!item.passed
                                        ? item.likely_failure_type || '基础规则失败'
                                        : item.retrieval_evaluation?.quality_status === 'needs_review'
                                          ? '检索需复核'
                                        : item.semantic_evaluation?.quality_status === 'needs_review'
                                          ? '语义需复核'
                                          : '通过'}
                                    </span>
                                    <span className="text-xs text-secondary">score {item.score}</span>
                                  </div>
                                  <p className="mt-1 text-sm font-medium text-primary">{item.question || '未提供问题'}</p>
                                  <p className="mt-1 text-xs text-secondary">{item.recommended_action}</p>
                                  {item.retrieval_evaluation?.status && item.retrieval_evaluation.status !== 'unavailable' && (
                                    <div className="mt-2 space-y-1 text-xs text-secondary">
                                      <p>
                                        {RETRIEVAL_METRICS.map(([key, label]) => {
                                          const value = item.retrieval_evaluation?.metrics?.[key];
                                          return `${label} ${value == null ? '未评' : `${Math.round(Number(value) * 100)}%`}`;
                                        }).join(' · ')}
                                      </p>
                                      {item.retrieval_evaluation.reason && <p>{item.retrieval_evaluation.reason}</p>}
                                      {item.retrieval_evaluation.recommended_action && (
                                        <p className="text-primary">{item.retrieval_evaluation.recommended_action}</p>
                                      )}
                                      {item.retrieval_evaluation.chunks?.length ? (
                                        <details className="mt-2 rounded-control border border-border bg-page px-3 py-2">
                                          <summary className="cursor-pointer text-primary">
                                            查看 {item.retrieval_evaluation.chunks.length} 条检索片段
                                          </summary>
                                          <div className="mt-2 space-y-2">
                                            {item.retrieval_evaluation.chunks.map((chunk) => (
                                              <div key={`${item.id}-${chunk.rank}-${chunk.chunk_id || chunk.path}`} className="border-t border-border pt-2 first:border-t-0 first:pt-0">
                                                <p className="text-primary">
                                                  #{chunk.rank} · {chunk.relevance_label ? RETRIEVAL_LABELS[chunk.relevance_label] : '未判定'}
                                                  {chunk.relevance_score == null ? '' : ` · 相关性 ${Math.round(chunk.relevance_score * 100)}%`}
                                                </p>
                                                <p className="truncate font-mono text-[11px]">{chunk.path || '未知文档'}</p>
                                                <p className="mt-1 line-clamp-3">{chunk.content || '未保留片段正文'}</p>
                                              </div>
                                            ))}
                                          </div>
                                        </details>
                                      ) : null}
                                    </div>
                                  )}
                                  {item.semantic_evaluation?.status === 'completed' && (
                                    <div className="mt-2 space-y-1 text-xs text-secondary">
                                      <p>
                                        {SEMANTIC_METRICS.map(([key, label]) => `${label} ${Math.round((item.semantic_evaluation?.scores?.[key] || 0) * 100)}%`).join(' · ')}
                                      </p>
                                      {item.semantic_evaluation.reason && <p>{item.semantic_evaluation.reason}</p>}
                                      {item.semantic_evaluation.recommended_action && (
                                        <p className="text-primary">{item.semantic_evaluation.recommended_action}</p>
                                      )}
                                    </div>
                                  )}
                                </div>
                                <div className="flex flex-shrink-0 gap-2">
                                  <button
                                    type="button"
                                    onClick={() => copyCaseQuestion(item)}
                                    className="inline-flex h-8 w-8 items-center justify-center rounded-control text-secondary hover:bg-page hover:text-primary"
                                    aria-label="复制问题"
                                  >
                                    <Clipboard className="h-4 w-4" />
                                  </button>
                                  {item.question && (
                                    <Link
                                      to={`/qa?scope_type=${encodeURIComponent(report.data.scope?.type || selectedScope?.type || '')}&scope_id=${encodeURIComponent(report.data.scope?.id || selectedScope?.id || '')}&question=${encodeURIComponent(item.question)}&mode=${item.mode}`}
                                      className="inline-flex h-8 w-8 items-center justify-center rounded-control text-secondary hover:bg-page hover:text-primary"
                                      aria-label="复测"
                                    >
                                      <RotateCcw className="h-4 w-4" />
                                    </Link>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <EmptyState title="没有需复核用例" description="当前筛选条件下没有需要回流的用例。" />
                      )}
                    </div>

                    <div className="rounded-control border border-border p-3">
                      <div className="mb-2 flex items-center justify-between">
                        <p className="text-sm font-medium text-primary">Markdown 报告</p>
                        <button
                          type="button"
                          onClick={() => setShowReportModal(true)}
                          className="inline-flex items-center gap-1 text-xs text-link hover:underline"
                        >
                          <Maximize2 className="h-3.5 w-3.5" />
                          查看详情
                        </button>
                      </div>
                      <MarkdownPreview content={report.data.content} />
                    </div>
                  </div>
                ) : (
                  <EmptyState title="无法加载报告" />
                )
              ) : (
                <EmptyState title="选择报告" description="点击左侧报告查看 Markdown 内容。" />
              )}
            </section>
          </div>

          {showReportModal && report.data && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
              <div className="flex h-[90vh] w-full max-w-4xl flex-col rounded-panel bg-surface shadow-lg">
                <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-3">
                  <h3 className="min-w-0 break-all font-mono text-sm font-semibold text-primary">{report.data.report_id}</h3>
                  <button
                    type="button"
                    onClick={() => setShowReportModal(false)}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-control text-secondary hover:bg-page hover:text-primary"
                    aria-label="关闭"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <div className="min-w-0 flex-1 overflow-x-hidden overflow-y-auto px-5 py-4">
                  <MarkdownPreview content={report.data.content} />
                </div>
              </div>
            </div>
          )}

          <EvalCaseManager evalType="knowledge_qa" scopeType={selectedScope?.type} scopeId={selectedScope?.id} />
        </>
      )}
    </div>
  );
}

const RESEARCH_REVIEW_DIMENSIONS: Array<[string, string]> = [
  ['answer_coverage', '核心问题回答度'],
  ['content_richness', '内容充实性'],
  ['readability', '结构与可读性'],
  ['scope_alignment', '范围合理性'],
];

function researchOutcomeLabel(row: ResearchProbeEvaluationRow) {
  if (row.outcome === 'execution_failed') return '执行失败';
  if (row.outcome === 'cancelled') return '已取消';
  if (row.outcome === 'pipeline_gate_failed') return '生产质量门未通过';
  return '执行完成';
}

function researchOutcomeTone(row: ResearchProbeEvaluationRow) {
  if (row.outcome === 'execution_failed' || row.outcome === 'pipeline_gate_failed') return 'text-danger';
  if (row.outcome === 'cancelled') return 'text-secondary';
  return 'text-success';
}

function ResearchProbeReportPanel() {
  const reports = useResearchProbeReports(50, true);
  const [selectedReport, setSelectedReport] = useState('');
  const report = useResearchProbeReport(selectedReport);

  useEffect(() => {
    const rows = reports.data?.reports || [];
    if (!selectedReport && rows.length) setSelectedReport(rows[0].report_id);
    if (selectedReport && rows.length && !rows.some((item) => item.report_id === selectedReport)) {
      setSelectedReport(rows[0].report_id);
    }
  }, [reports.data?.reports, selectedReport]);

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(20rem,0.8fr)_minmax(0,1.7fr)]">
      <section className="rounded-panel border border-border bg-white p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-primary">调研全流程评测报告</h3>
            <p className="mt-1 text-xs text-secondary">生产质量门和内容质量建议分开记录。</p>
          </div>
          <button type="button" onClick={() => reports.refetch()} className="text-xs text-link hover:underline">刷新</button>
        </div>
        {reports.error ? (
          <ErrorPanel error={reports.error} onRetry={() => reports.refetch()} />
        ) : (
          <DataTable<ResearchProbeReportMeta>
            columns={[
              {
                key: 'created_at',
                header: '时间',
                render: (item) => <span className="text-xs text-secondary">{formatDateTime(item.created_at)}</span>,
              },
              {
                key: 'execution',
                header: '执行',
                render: (item) => (
                  <span className="text-xs text-primary">
                    {item.summary.execution_succeeded_cases || 0}/{item.summary.total_cases || 0}
                  </span>
                ),
              },
              {
                key: 'production',
                header: '生产门',
                render: (item) => (
                  <span className="text-xs text-primary">
                    {item.summary.production_gate_passed_cases || 0}/{item.summary.execution_succeeded_cases || 0}
                  </span>
                ),
              },
            ]}
            data={reports.data?.reports || []}
            keyExtractor={(item) => item.report_id}
            onRowClick={(item) => setSelectedReport(item.report_id)}
            loading={reports.isLoading}
            emptyMessage="暂无调研全流程评测报告"
          />
        )}
      </section>

      <section className="rounded-panel border border-border bg-white p-4">
        <h3 className="mb-3 text-sm font-semibold text-primary">报告详情</h3>
        {!selectedReport ? (
          <EmptyState title="选择报告" description="点击左侧报告查看执行、生产质量门和内容建议。" />
        ) : report.isLoading ? (
          <div className="text-sm text-secondary">加载中...</div>
        ) : report.error ? (
          <ErrorPanel error={report.error} onRetry={() => report.refetch()} />
        ) : report.data ? (
          <div className="max-h-[76vh] space-y-5 overflow-y-auto pr-1">
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
              <Metric
                label="执行成功"
                value={`${report.data.summary.execution_succeeded_cases || 0}/${report.data.summary.total_cases || 0}`}
              />
              <Metric
                label="生产质量门通过"
                value={`${report.data.summary.production_gate_passed_cases || 0}/${report.data.summary.execution_succeeded_cases || 0}`}
                tone={report.data.summary.production_gate_failed_cases ? 'danger' : 'success'}
              />
              <Metric
                label="存在内容建议"
                value={report.data.summary.content_review_advisory_cases || 0}
              />
              <Metric
                label="执行异常"
                value={report.data.summary.execution_failed_cases || 0}
                tone={report.data.summary.execution_failed_cases ? 'danger' : 'success'}
              />
            </div>

            <div className="border-y border-border py-3 text-xs text-secondary">
              <span>评测集：{String(report.data.configuration?.case_set_version || '历史版本')}</span>
              <span className="ml-4">开始：{formatDateTime(report.data.started_at)}</span>
              <span className="ml-4">结束：{formatDateTime(report.data.finished_at || report.data.created_at)}</span>
              <span className="ml-4">生产门通过率：{`${Math.round((report.data.summary.production_gate_pass_rate || 0) * 100)}%`}</span>
            </div>

            <div className="divide-y divide-border border-y border-border">
              {report.data.rows.map((row) => (
                <div key={row.case_id} className="py-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-primary">{row.title}</p>
                      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs">
                        <span className={researchOutcomeTone(row)}>{researchOutcomeLabel(row)}</span>
                        <span className="text-secondary">引用校验：{row.validation_status === 'passed' ? '通过' : row.validation_status || '未评估'}</span>
                        <span className="text-secondary">来源 {row.sources || 0} · 证据 {row.evidence_map || 0} · 弱问题 {row.weak_question_count || 0}</span>
                      </div>
                    </div>
                    {row.job_id && (
                      <Link
                        to={`/observability?job_id=${encodeURIComponent(row.job_id)}`}
                        className="inline-flex items-center gap-1 text-xs text-link hover:underline"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                        查看子任务
                      </Link>
                    )}
                  </div>

                  {row.quality_reason && row.production_gate_status === 'failed' && (
                    <p className="mt-2 border-l-2 border-danger pl-3 text-xs text-secondary">{row.quality_reason}</p>
                  )}

                  {row.content_review?.status === 'content_review_unavailable' ? (
                    <p className="mt-3 text-xs text-secondary">内容质量建议未完成：{row.content_review.reason || '模型不可用'}</p>
                  ) : row.content_review?.dimensions ? (
                    <div className="mt-3 border-t border-border pt-3">
                      <div className="grid gap-3 sm:grid-cols-2">
                        {RESEARCH_REVIEW_DIMENSIONS.map(([key, label]) => {
                          const dimension = row.content_review?.dimensions?.[key];
                          if (!dimension) return null;
                          return (
                            <div key={key}>
                              <div className="flex items-center justify-between gap-2">
                                <p className="text-xs font-medium text-primary">{label}</p>
                                <span className="text-xs text-secondary">{dimension.label}</span>
                              </div>
                              <p className="mt-1 text-xs leading-5 text-secondary">{dimension.reason || '未提供判断理由'}</p>
                            </div>
                          );
                        })}
                      </div>
                      {row.content_review.suggestions?.length ? (
                        <div className="mt-3 border-t border-border pt-3">
                          <p className="text-xs font-medium text-secondary">非阻断建议</p>
                          {row.content_review.suggestions.map((suggestion, index) => (
                            <p key={`${row.case_id}-${index}`} className="mt-1 text-xs leading-5 text-primary">
                              {suggestion.section_id ? `${suggestion.section_id}：` : ''}{suggestion.reason}
                            </p>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  <details className="mt-3 text-xs text-secondary">
                    <summary className="cursor-pointer text-link">查看流程诊断</summary>
                    <div className="mt-2 grid gap-1 sm:grid-cols-2">
                      <span>材料：{row.materials || 0}</span>
                      <span>候选片段：{row.selected_chunks || 0}</span>
                      <span>混合评分片段：{row.hybrid_scored_chunks || 0}</span>
                      <span>单一来源风险：{row.risk_single_source_count || 0}</span>
                      <span>阻断性无支撑事实：{row.blocking_unsupported_claims || 0}</span>
                      <span>缺失引用：{row.missing_inline_refs || 0}</span>
                    </div>
                  </details>
                </div>
              ))}
            </div>

            <details className="border-y border-border py-3">
              <summary className="cursor-pointer text-sm font-medium text-primary">查看 Markdown 汇总报告</summary>
              <div className="mt-3"><MarkdownPreview content={report.data.content} /></div>
            </details>
          </div>
        ) : (
          <EmptyState title="无法加载报告" />
        )}
      </section>
    </div>
  );
}

function KnowledgeQaCaseSetPanel({
  cases,
  canGenerate = true,
  generationJob,
  generationError,
  isGenerating,
  isApproving,
  onGenerate,
  onApprove,
  onActivateVersion,
  isActivatingVersion,
}: {
  cases?: EvalCaseListResponse;
  canGenerate?: boolean;
  generationJob?: KnowledgeQaCaseGenerationJob;
  generationError?: unknown;
  isGenerating?: boolean;
  isApproving?: boolean;
  onGenerate?: () => void;
  onApprove?: () => void;
  onActivateVersion?: (version: number) => void;
  isActivatingVersion?: boolean;
}) {
  const draft = generationJob?.result;
  const isApproved = draft?.approval_status === 'approved';
  return (
    <section className="border-y border-border bg-white px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-primary">当前评测集</h2>
          {cases?.scope === 'topic' ? (
            <p className="mt-1 text-sm text-secondary">
              已启用该知识库的专属评测集，共 {cases.cases.length} 题
              {cases.eval_set_version ? `，版本 ${cases.eval_set_version}` : ''}。
            </p>
          ) : (
            <p className="mt-1 text-sm text-amber-900">
              当前使用通用兜底用例，不能充分反映这个范围的真实内容。
              {canGenerate ? '建议先生成专属评测集。' : '可在下方评测集管理中新增这个范围的专属用例。'}
            </p>
          )}
        </div>
        {canGenerate && (
          <Button variant="secondary" onClick={onGenerate} isLoading={isGenerating}>
            <Sparkles className="mr-1.5 h-4 w-4" />
            {draft ? '重新生成草稿' : 'AI 生成专属评测集'}
          </Button>
        )}
      </div>

      {generationError ? (
        <div className="mt-3 border-l-2 border-danger pl-3 text-sm text-danger">{String((generationError as Error).message || generationError)}</div>
      ) : generationJob?.status === 'failed' ? (
        <div className="mt-3 border-l-2 border-danger pl-3 text-sm text-danger">
          生成失败：{generationJob.error?.message || '未返回具体原因'}
        </div>
      ) : draft ? (
        <div className="mt-4 border-t border-border pt-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-primary">
              草稿共 {draft.case_count} 题：可回答 {draft.answerable_count} 题，不可回答 {draft.unanswerable_count} 题；
              依据 {draft.document_count} 篇文档、{draft.evidence_count} 条证据生成。
            </p>
            {!isApproved && (
              <Button onClick={onApprove} isLoading={isApproving}>启用这套评测集</Button>
            )}
            {isApproved && <span className="text-sm font-medium text-success">已启用</span>}
          </div>
          <div className="mt-3 divide-y divide-border border-y border-border">
            {draft.cases.map((item, index) => (
              <div key={item.id} className="grid gap-1 py-3 sm:grid-cols-[2rem_minmax(0,1fr)_8rem] sm:items-start">
                <span className="text-xs text-secondary">{index + 1}</span>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-primary">{item.question}</p>
                  <p className="mt-1 text-xs text-secondary">
                    {item.category || '未分类'} · {item.evidence_refs?.length || 0} 条标准证据
                  </p>
                </div>
                <span className="text-xs text-secondary sm:text-right">
                  {item.answerability === 'unanswerable' ? '应识别证据不足' : '可由知识库回答'}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {cases?.is_stale && (
        <div className="mt-3 border-l-2 border-amber-500 pl-3 text-sm text-amber-900">
          {cases.stale_reason || '知识库内容已变化，当前评测集可能过期。'}
        </div>
      )}
      {cases?.case_set_modified && (
        <div className="mt-3 border-l-2 border-amber-500 pl-3 text-sm text-amber-900">
          当前用例经过手工编辑，已经偏离已启用版本的冻结基准。建议生成并启用新版本后再做正式趋势对比。
        </div>
      )}

      {cases?.versions?.length ? (
        <div className="mt-4 border-t border-border pt-3">
          <p className="mb-2 text-xs font-medium text-secondary">评测集版本</p>
          <div className="divide-y divide-border">
            {cases.versions.map((version) => (
              <div key={version.eval_set_id} className="flex flex-wrap items-center justify-between gap-3 py-2 text-sm">
                <div>
                  <span className="font-medium text-primary">版本 {version.version}</span>
                  <span className="ml-2 text-xs text-secondary">{version.case_count} 题 · {formatDateTime(version.approved_at)}</span>
                </div>
                {version.active ? (
                  <span className="text-xs font-medium text-success">当前使用</span>
                ) : (
                  <Button size="sm" variant="secondary" onClick={() => onActivateVersion?.(version.version)} isLoading={isActivatingVersion}>
                    启用此版本
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function formatComparisonValue(key: string, value: number) {
  if (key === 'pass_rate') return `${value.toFixed(1)}%`;
  return key === 'average_score' ? value.toFixed(1) : `${Math.round(value * 100)}%`;
}

function formatDelta(key: string, value: number | null) {
  if (value == null) return '不可比';
  const rawScale = key === 'average_score' || key === 'pass_rate';
  const scaled = rawScale ? value : value * 100;
  return `${scaled > 0 ? '+' : ''}${scaled.toFixed(1)}${key === 'average_score' ? '' : ' 个百分点'}`;
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  tone?: 'success' | 'danger';
}) {
  return (
    <div className="rounded-control border border-border bg-white p-3">
      <p className="text-xs text-secondary">{label}</p>
      <p className={tone === 'success' ? 'mt-1 text-lg font-semibold text-success' : tone === 'danger' ? 'mt-1 text-lg font-semibold text-danger' : 'mt-1 text-lg font-semibold text-primary'}>
        {value}
      </p>
    </div>
  );
}

function EvalCaseManager({
  evalType,
  topic,
  scopeType,
  scopeId,
  selectable,
  selectedCaseIds,
  onSelectCases,
  onToggleCase,
}: {
  evalType: EvalType;
  topic?: string;
  scopeType?: 'knowledge_base' | 'theme' | 'research';
  scopeId?: string;
  selectable?: boolean;
  selectedCaseIds?: string[];
  onSelectCases?: (ids: string[]) => void;
  onToggleCase?: (id: string) => void;
}) {
  const legacyCases = useEvaluationCases(evalType, topic);
  const legacyUpsert = useUpsertEvaluationCase(evalType, topic);
  const legacyRemove = useDeleteEvaluationCase(evalType, topic);
  const scopeCases = useKnowledgeScopeEvaluationCases(scopeType, scopeId);
  const scopeUpsert = useUpsertKnowledgeScopeEvaluationCase(scopeType, scopeId);
  const scopeRemove = useDeleteKnowledgeScopeEvaluationCase(scopeType, scopeId);
  const usesScope = Boolean(scopeType && scopeId);
  const cases = usesScope ? scopeCases : legacyCases;
  const upsert = usesScope ? scopeUpsert : legacyUpsert;
  const remove = usesScope ? scopeRemove : legacyRemove;
  const { showToast } = useToast();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftJson, setDraftJson] = useState('');
  const [draftBase, setDraftBase] = useState<EvalCase | null>(null);
  const [researchDraft, setResearchDraft] = useState<ResearchProbeCaseDraft>(EMPTY_RESEARCH_PROBE_DRAFT);
  const [draftError, setDraftError] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [showEditor, setShowEditor] = useState(false);
  const caseItems = cases.data?.cases || [];
  const caseIds = caseItems.map((item) => item.id);
  const selectedIds = selectedCaseIds || [];
  const selectedCount = caseIds.filter((id) => selectedIds.includes(id)).length;
  const allSelected = caseIds.length > 0 && selectedCount === caseIds.length;
  const someSelected = selectedCount > 0 && !allSelected;

  useEffect(() => {
    if (!selectable || !onSelectCases || !selectedCaseIds) return;
    const validIds = selectedCaseIds.filter((id) => caseIds.includes(id));
    if (validIds.length !== selectedCaseIds.length) onSelectCases(validIds);
  }, [caseIds.join('|'), onSelectCases, selectable, selectedCaseIds]);

  const openCreate = () => {
    setEditingId(null);
    setDraftBase(null);
    if (evalType === 'research_probe') {
      setResearchDraft(EMPTY_RESEARCH_PROBE_DRAFT);
    } else {
      setDraftJson('{\n  "id": "",\n\n}');
    }
    setDraftError('');
    setShowEditor(true);
  };

  const openEdit = (item: EvalCase) => {
    setEditingId(item.id);
    setDraftBase(item);
    if (evalType === 'research_probe') {
      setResearchDraft(researchCaseToDraft(item));
    } else {
      setDraftJson(JSON.stringify(item, null, 2));
    }
    setDraftError('');
    setShowEditor(true);
  };

  const saveDraft = () => {
    let parsed: EvalCase;
    if (evalType === 'research_probe') {
      const id = researchDraft.id.trim();
      const title = researchDraft.title.trim();
      const userRequest = researchDraft.userRequest.trim();
      const coreIntents = linesToList(researchDraft.coreIntents);
      if (!id || !title || !userRequest) {
        setDraftError('请填写用例 ID、名称和调研请求');
        return;
      }
      if (!coreIntents.length) {
        setDraftError('至少填写一条核心评测目标');
        return;
      }
      parsed = {
        ...(draftBase || {}),
        id,
        case_version: Math.max(1, Number.parseInt(researchDraft.caseVersion, 10) || 1),
        kind: researchDraft.kind,
        title,
        user_request: userRequest,
        strategy: researchDraft.strategy,
        depth: researchDraft.depth,
        core_intents: coreIntents,
        optional_topics: linesToList(researchDraft.optionalTopics),
        scope_notes: researchDraft.scopeNotes.trim(),
      };
    } else {
      try {
        parsed = JSON.parse(draftJson);
      } catch (err) {
        setDraftError(`JSON 解析失败：${(err as Error).message}`);
        return;
      }
    }
    if (!parsed.id) {
      setDraftError('用例必须包含非空的 id 字段');
      return;
    }
    upsert.mutate(parsed, {
      onSuccess: () => {
        showToast(editingId ? '用例已更新' : '用例已新增', 'success');
        setShowEditor(false);
        if (selectable && onSelectCases && !selectedIds.includes(parsed.id)) {
          onSelectCases([...selectedIds, parsed.id]);
        }
      },
      onError: (err) => setDraftError(String((err as Error).message || err)),
    });
  };

  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-sm font-semibold text-primary">评测集管理</h3>
          {selectable && (
            <span className="text-xs text-secondary">
              {selectedCount ? `已选择 ${selectedCount}/${caseIds.length}` : `共 ${caseIds.length} 条，尚未选择`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {selectable && (
            <>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onSelectCases?.(caseIds)}
                disabled={!caseIds.length || allSelected}
              >
                全选
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onSelectCases?.([])}
                disabled={!selectedCount}
              >
                清空
              </Button>
            </>
          )}
          <Button size="sm" variant="secondary" onClick={openCreate}>
            <Plus className="mr-1 h-3.5 w-3.5" />
            新增用例
          </Button>
        </div>
      </div>

      {cases.error && <ErrorPanel error={cases.error} onRetry={() => cases.refetch()} />}

      <DataTable<EvalCase>
        columns={[
          ...(selectable
            ? [
                {
                  key: 'select',
                  header: (
                    <SelectionCheckbox
                      checked={allSelected}
                      indeterminate={someSelected}
                      label="选择全部评测用例"
                      onChange={() => onSelectCases?.(allSelected ? [] : caseIds)}
                    />
                  ),
                  width: '2.5rem',
                  render: (item: EvalCase) => (
                    <SelectionCheckbox
                      checked={selectedCaseIds?.includes(item.id) ?? false}
                      label={`选择用例 ${item.id}`}
                      onChange={() => onToggleCase?.(item.id)}
                    />
                  ),
                },
              ]
            : []),
          {
            key: 'id',
            header: 'id',
            render: (item) => <span className="font-mono text-xs">{item.id}</span>,
          },
          {
            key: 'summary',
            header: '摘要',
            render: (item) => (
              <span className="text-sm text-secondary">
                {String(item.title || item.question || item.user_request || item.url || '')}
              </span>
            ),
          },
          {
            key: 'actions',
            header: '操作',
            width: '6rem',
            render: (item) => (
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    openEdit(item);
                  }}
                  className="text-xs text-link hover:underline"
                >
                  编辑
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteTarget(item.id);
                  }}
                  className="inline-flex items-center text-xs text-danger hover:underline"
                >
                  <Trash2 className="mr-1 h-3 w-3" />
                  删除
                </button>
              </div>
            ),
          },
        ]}
        data={caseItems}
        keyExtractor={(item) => item.id}
        onRowClick={selectable ? (item) => onToggleCase?.(item.id) : undefined}
        rowClassName={selectable
          ? (item) => (selectedIds.includes(item.id) ? 'bg-accent/5 shadow-[inset_3px_0_0_0_#C4531E]' : undefined)
          : undefined}
        loading={cases.isLoading}
        emptyMessage="暂无评测用例"
      />

      {showEditor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-3 sm:p-6">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="eval-case-editor-title"
            className="flex max-h-[calc(100vh-1.5rem)] w-full max-w-3xl flex-col overflow-hidden rounded-panel bg-surface shadow-lg sm:max-h-[calc(100vh-3rem)]"
          >
            <div className="flex items-center justify-between border-b border-border px-5 py-4">
              <div>
                <h3 id="eval-case-editor-title" className="text-lg font-semibold text-primary">
                  {editingId ? '编辑评测用例' : '新增评测用例'}
                </h3>
                <p className="mt-1 text-xs text-secondary">
                  {evalType === 'research_probe' ? '配置调研任务及本次评测真正关心的内容。' : '编辑用例的完整结构化数据。'}
                </p>
              </div>
              <button
                type="button"
                aria-label="关闭编辑器"
                title="关闭"
                onClick={() => setShowEditor(false)}
                className="inline-flex h-9 w-9 items-center justify-center text-secondary hover:text-primary"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="overflow-y-auto px-5 py-4">
              {evalType === 'research_probe' ? (
                <div className="space-y-5">
                  <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_8rem]">
                    <FormField label="用例 ID" hint={editingId ? '已有用例的 ID 不可修改。' : '建议使用简短、稳定的英文标识。'}>
                      <input
                        value={researchDraft.id}
                        disabled={Boolean(editingId)}
                        onChange={(event) => setResearchDraft((draft) => ({ ...draft, id: event.target.value }))}
                        placeholder="例如 single_explain_harness"
                        className={`${CASE_INPUT_CLASS} font-mono disabled:bg-page disabled:text-secondary`}
                      />
                    </FormField>
                    <FormField label="版本">
                      <input
                        type="number"
                        min="1"
                        value={researchDraft.caseVersion}
                        onChange={(event) => setResearchDraft((draft) => ({ ...draft, caseVersion: event.target.value }))}
                        className={CASE_INPUT_CLASS}
                      />
                    </FormField>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-[12rem_minmax(0,1fr)]">
                    <FormField label="用例类型">
                      <select
                        value={researchDraft.kind}
                        onChange={(event) => {
                          const kind = event.target.value;
                          setResearchDraft((draft) => ({ ...draft, kind, strategy: defaultStrategyForKind(kind) }));
                        }}
                        className={CASE_INPUT_CLASS}
                      >
                        {RESEARCH_KIND_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </FormField>
                    <FormField label="用例名称">
                      <input
                        value={researchDraft.title}
                        onChange={(event) => setResearchDraft((draft) => ({ ...draft, title: event.target.value }))}
                        placeholder="例如 Harness Engineering 宽泛解释型"
                        className={CASE_INPUT_CLASS}
                      />
                    </FormField>
                  </div>

                  <FormField label="用户调研请求" hint="这里应像真实用户说话，不要预先塞入报告章节或答案。">
                    <textarea
                      value={researchDraft.userRequest}
                      onChange={(event) => setResearchDraft((draft) => ({ ...draft, userRequest: event.target.value }))}
                      rows={4}
                      placeholder="调研一下……"
                      className={`${CASE_INPUT_CLASS} resize-y leading-6`}
                    />
                  </FormField>

                  <div className="grid gap-4 border-y border-border py-4 sm:grid-cols-2">
                    <FormField
                      label="调研策略"
                      hint={researchDraft.strategy === 'hv_analysis' ? '适合多个对象的横向比较或明确的演化分析。' : '适合单主题解释、机制深挖和一般调研。'}
                    >
                      <div className="grid grid-cols-2 overflow-hidden rounded-control border border-border" role="group" aria-label="调研策略">
                        {([
                          ['standard', '标准调研'],
                          ['hv_analysis', '横纵分析'],
                        ] as const).map(([value, label]) => (
                          <button
                            key={value}
                            type="button"
                            onClick={() => setResearchDraft((draft) => ({ ...draft, strategy: value }))}
                            className={`px-3 py-2 text-sm ${researchDraft.strategy === value ? 'bg-primary text-white' : 'bg-surface text-secondary hover:bg-page'}`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </FormField>
                    <FormField label="调研深度" hint="深度会影响每个问题的检索轮次和材料预算。">
                      <div className="grid grid-cols-2 overflow-hidden rounded-control border border-border" role="group" aria-label="调研深度">
                        {([
                          ['normal', '常规'],
                          ['deep', '深度'],
                        ] as const).map(([value, label]) => (
                          <button
                            key={value}
                            type="button"
                            onClick={() => setResearchDraft((draft) => ({ ...draft, depth: value }))}
                            className={`px-3 py-2 text-sm ${researchDraft.depth === value ? 'bg-primary text-white' : 'bg-surface text-secondary hover:bg-page'}`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </FormField>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <FormField label="核心评测目标" hint="每行一条；报告缺失这些内容时应判为问题。">
                      <textarea
                        value={researchDraft.coreIntents}
                        onChange={(event) => setResearchDraft((draft) => ({ ...draft, coreIntents: event.target.value }))}
                        rows={5}
                        placeholder={'解释概念的定义与定位\n说明关键工作机制'}
                        className={`${CASE_INPUT_CLASS} resize-y leading-6`}
                      />
                    </FormField>
                    <FormField label="可选内容" hint="每行一条；缺失时不应导致整项评测失败。">
                      <textarea
                        value={researchDraft.optionalTopics}
                        onChange={(event) => setResearchDraft((draft) => ({ ...draft, optionalTopics: event.target.value }))}
                        rows={5}
                        placeholder={'具体厂商案例\n量化效果数据'}
                        className={`${CASE_INPUT_CLASS} resize-y leading-6`}
                      />
                    </FormField>
                  </div>

                  <FormField label="范围说明" hint="告诉评审什么应当重点判断，以及哪些内容不要求穷尽。">
                    <textarea
                      value={researchDraft.scopeNotes}
                      onChange={(event) => setResearchDraft((draft) => ({ ...draft, scopeNotes: event.target.value }))}
                      rows={3}
                      className={`${CASE_INPUT_CLASS} resize-y leading-6`}
                    />
                  </FormField>
                </div>
              ) : (
                <FormField label="用例 JSON" hint="必须包含 id 字段；字段结构因评测类型而异，参考同类型其他用例。">
                  <textarea
                    value={draftJson}
                    onChange={(e) => setDraftJson(e.target.value)}
                    rows={18}
                    className={`${CASE_INPUT_CLASS} resize-y font-mono text-xs leading-5`}
                  />
                </FormField>
              )}
              {draftError && <p className="mt-4 border-l-2 border-danger pl-3 text-sm text-danger">{draftError}</p>}
            </div>

            <div className="flex justify-end gap-3 border-t border-border bg-page px-5 py-4">
              <Button type="button" variant="secondary" onClick={() => setShowEditor(false)}>
                取消
              </Button>
              <Button onClick={saveDraft} isLoading={upsert.isPending}>
                保存用例
              </Button>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="删除评测用例"
        message={`确认删除用例 ${deleteTarget} 吗？此操作不可撤销。`}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={() => {
          if (!deleteTarget) return;
          remove.mutate(deleteTarget, {
            onSuccess: () => showToast('用例已删除', 'success'),
          });
          setDeleteTarget(null);
        }}
      />
    </section>
  );
}
