import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ArrowLeft, RefreshCw, RotateCcw } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { Button } from '@/components/controls/Button';
import { FilterSelect, SearchInput } from '@/components/controls/FormControls';
import { StatusBadge } from '@/components/data/StatusBadge';
import {
  useChannels,
  useJobTrace,
  useObservabilityRun,
  useObservabilityRuns,
  useObservabilityStream,
} from '@/hooks';
import { formatDateTime } from '@/utils/formatters';
import { JobTraceResponse, ObservabilityEvent, RunDetailResponse, RunSummary } from '@/types';
import { RunList } from './RunList';
import { EventTimeline } from './EventTimeline';

const DAY_OPTIONS = [
  { value: '1', label: '今天' },
  { value: '3', label: '最近 3 天' },
  { value: '7', label: '最近 7 天' },
  { value: '30', label: '最近 30 天' },
];

const MODE_OPTIONS = [
  { value: 'main', label: '主线对话' },
  { value: 'kb_qa', label: '知识库问答' },
];

function formatDuration(ms?: number | null): string | null {
  if (ms == null) return null;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Live-appends events from the SSE stream onto a base list while a run/job
 * is still "running" — see docs/modules/observability-redesign-v0.1.md 3.9.
 * The run list itself is never live; only an in-progress detail view is. */
function useLiveEvents(
  baseEvents: ObservabilityEvent[],
  streamParams: { runId?: string; jobId?: string } | null
) {
  const { url } = useObservabilityStream(
    streamParams ? { runId: streamParams.runId, jobId: streamParams.jobId, replay: 0 } : undefined
  );
  const [liveEvents, setLiveEvents] = useState<ObservabilityEvent[]>([]);

  useEffect(() => {
    setLiveEvents([]);
    if (!streamParams) return undefined;
    const source = new EventSource(url);
    source.addEventListener('observability', (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as ObservabilityEvent;
        setLiveEvents((prev) => [...prev, payload]);
      } catch {
        // ignore malformed SSE payloads
      }
    });
    return () => source.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamParams?.runId, streamParams?.jobId]);

  return useMemo(() => {
    const seen = new Set(baseEvents.map((e) => `${e.timestamp}-${e.event_type}-${e.message}`));
    const merged = [...baseEvents];
    for (const event of liveEvents) {
      const key = `${event.timestamp}-${event.event_type}-${event.message}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(event);
    }
    return merged;
  }, [baseEvents, liveEvents]);
}

export function ObservabilityPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobId = searchParams.get('job_id') || '';
  const [days, setDays] = useState(7);
  const [threadFilter, setThreadFilter] = useState(searchParams.get('thread_id') || '');
  const [channelFilter, setChannelFilter] = useState(searchParams.get('channel_id') || '');
  const [modeFilter, setModeFilter] = useState<'main' | 'kb_qa'>('main');
  const [selectedRun, setSelectedRun] = useState<RunSummary | null>(null);

  if (jobId) {
    return <JobTraceView jobId={jobId} onBack={() => setSearchParams({})} />;
  }

  return (
    <RunListAndDetail
      days={days}
      setDays={setDays}
      threadFilter={threadFilter}
      setThreadFilter={setThreadFilter}
      channelFilter={channelFilter}
      setChannelFilter={setChannelFilter}
      modeFilter={modeFilter}
      setModeFilter={setModeFilter}
      selectedRun={selectedRun}
      setSelectedRun={setSelectedRun}
    />
  );
}

function RunListAndDetail({
  days,
  setDays,
  threadFilter,
  setThreadFilter,
  channelFilter,
  setChannelFilter,
  modeFilter,
  setModeFilter,
  selectedRun,
  setSelectedRun,
}: {
  days: number;
  setDays: (d: number) => void;
  threadFilter: string;
  setThreadFilter: (v: string) => void;
  channelFilter: string;
  setChannelFilter: (v: string) => void;
  modeFilter: 'main' | 'kb_qa';
  setModeFilter: (v: 'main' | 'kb_qa') => void;
  selectedRun: RunSummary | null;
  setSelectedRun: (r: RunSummary | null) => void;
}) {
  const channels = useChannels();
  const channelOptions = useMemo(
    () => [
      { value: '', label: '全部群' },
      ...(channels.data?.channels || []).map((c) => ({
        value: c.channel_id,
        label: c.title || c.channel_id,
      })),
    ],
    [channels.data?.channels]
  );

  const effectiveThreadFilter = channelFilter
    ? `channel:${channelFilter}:${modeFilter}`
    : threadFilter;

  const { data, error, refetch, isLoading } = useObservabilityRuns({
    thread_id: effectiveThreadFilter || undefined,
    days,
    limit: 50,
  });

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col gap-4 md:h-[calc(100vh-6rem)]">
      <PageHeader title="观测" description="按 run 查看最近发生了什么、排查问题" />

      {error && <ErrorPanel error={error} onRetry={() => refetch()} />}

      <div className="flex flex-wrap items-center gap-2 rounded-panel border border-border bg-white p-3">
        <FilterSelect
          value={String(days)}
          onChange={(e) => setDays(Number(e.target.value))}
          options={DAY_OPTIONS}
        />
        <FilterSelect
          value={channelFilter}
          onChange={(e) => setChannelFilter(e.target.value)}
          options={channelOptions}
        />
        {channelFilter && (
          <FilterSelect
            value={modeFilter}
            onChange={(e) => setModeFilter(e.target.value as 'main' | 'kb_qa')}
            options={MODE_OPTIONS}
          />
        )}
        <SearchInput
          placeholder="按 thread_id 筛选（可选）"
          value={threadFilter}
          onChange={(e) => setThreadFilter(e.target.value)}
          disabled={!!channelFilter}
        />
        {(threadFilter || channelFilter) && (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setThreadFilter('');
              setChannelFilter('');
            }}
          >
            <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
            清除
          </Button>
        )}
        {data && (
          <span className="ml-auto text-xs text-secondary">
            共 {data.total_runs} 个 run{data.truncated ? '（仅显示最近 50 个）' : ''}
          </span>
        )}
      </div>

      <div className="grid flex-1 gap-4 overflow-hidden lg:grid-cols-5">
        <div className="overflow-y-auto rounded-panel border border-border bg-page p-3 lg:col-span-2">
          {isLoading && <div className="px-3 py-8 text-center text-sm text-secondary">加载中...</div>}
          {!isLoading && data && (
            <RunList runs={data.runs} selectedId={selectedRun?.id} onSelect={setSelectedRun} />
          )}
        </div>
        <div className="flex flex-col overflow-hidden rounded-panel border border-border bg-white lg:col-span-3">
          {selectedRun ? (
            <RunDetailPanel runId={selectedRun.id} days={days} />
          ) : (
            <div className="flex h-full items-center justify-center px-4 text-center text-sm text-secondary">
              点击左侧一个 run 查看详细时间轴
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function RunDetailPanel({ runId, days }: { runId: string; days: number }) {
  const { data, error, refetch, isLoading } = useObservabilityRun(runId, days);
  const isRunning = data?.status === 'running';
  const events = useLiveEvents(data?.events || [], isRunning ? { runId } : null);

  return <RunOrJobDetail data={data} error={error} onRetry={refetch} isLoading={isLoading} events={events} idLabel="run_id" idValue={runId} />;
}

function JobTraceView({ jobId, onBack }: { jobId: string; onBack: () => void }) {
  const { data, error, refetch, isLoading, isFetching } = useJobTrace(jobId, 30);
  const isRunning = data ? data.status === 'running' : false;
  const events = useLiveEvents(data?.events || [], isRunning ? { jobId } : null);

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col gap-4 md:h-[calc(100vh-6rem)]">
      <PageHeader
        title="任务观测轨迹"
        description="该任务从触发到结束的完整过程（跨多次 run 拉平为一条时间线）"
        actions={
          <>
            <Button
              variant="secondary"
              onClick={() => void refetch()}
              disabled={isFetching}
              title="刷新任务观测轨迹"
            >
              <RefreshCw className={`mr-1.5 h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
              刷新
            </Button>
            <Button variant="secondary" onClick={onBack}>
              <ArrowLeft className="mr-1.5 h-4 w-4" />
              返回 run 列表
            </Button>
          </>
        }
      />
      <div className="flex-1 overflow-hidden rounded-panel border border-border bg-white">
        <RunOrJobDetail data={data} error={error} onRetry={refetch} isLoading={isLoading} events={events} idLabel="job_id" idValue={jobId} />
      </div>
    </div>
  );
}

function RunOrJobDetail({
  data,
  error,
  onRetry,
  isLoading,
  events,
  idLabel,
  idValue,
}: {
  data: RunDetailResponse | JobTraceResponse | undefined;
  error: unknown;
  onRetry: () => void;
  isLoading: boolean;
  events: ObservabilityEvent[];
  idLabel: string;
  idValue: string;
}) {
  if (error) return <ErrorPanel error={error} onRetry={onRetry} />;
  if (isLoading || !data) {
    return <div className="px-4 py-8 text-center text-sm text-secondary">加载中...</div>;
  }
  const duration = formatDuration(data.duration_ms);
  const runIds = 'run_ids' in data ? data.run_ids : undefined;
  const displayStatus = 'job_status' in data && data.job_status ? data.job_status : data.status;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-border p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-primary">{data.title}</p>
            <p className="mt-1 text-xs text-secondary">
              {idLabel}: {idValue}
            </p>
          </div>
          <StatusBadge status={displayStatus} />
        </div>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-secondary">
          <span>开始：{formatDateTime(data.started_at)}</span>
          <span>结束：{formatDateTime(data.finished_at)}</span>
          {duration && <span>耗时：{duration}</span>}
          <span>{events.length} 条事件</span>
          {data.error_count > 0 && <span className="text-danger">{data.error_count} 个报错</span>}
          {runIds && runIds.length > 0 && <span>涉及 {runIds.length} 个 run</span>}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        <EventTimeline events={events} />
      </div>
    </div>
  );
}
