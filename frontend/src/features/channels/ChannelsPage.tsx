import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { BellRing, FileSearch, Hash, Link2, RefreshCw, Search, Send } from 'lucide-react';
import { Button } from '@/components/controls/Button';
import { FilterSelect } from '@/components/controls/FormControls';
import { useToast } from '@/components/feedback/Toast';
import { Switch } from '@/components/controls/Switch';
import { EmptyState } from '@/components/data/EmptyState';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { JobCard } from '@/components/data/JobCard';
import { InlineJobCard, extractJobId } from '@/components/data/InlineJobCard';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { StatusBadge } from '@/components/data/StatusBadge';
import {
  buildKnowledgeScopeOptions,
  KnowledgeScopeSelector,
  knowledgeScopeKey,
} from '@/components/data/KnowledgeScopeSelector';
import { PageHeader } from '@/components/layout/PageHeader';
import {
  useChannel,
  useChannelJobs,
  useChannelMessages,
  useChannelPushConfig,
  useChannels,
  useCreateResearchJob,
  useSyncFeishuChannelTitles,
  useKnowledgeTree,
  useChannelScopeBinding,
  useBindChannelScope,
  useUnbindChannelScope,
  useSendChannelMessage,
  useUpdateChannel,
  useUpdateChannelPushConfig,
} from '@/hooks';
import { Channel, ChannelMode, ChannelPushContentStrategy, ChannelPushFrequency } from '@/types';
import { cn } from '@/utils/cn';
import { formatDateTime } from '@/utils/formatters';


export function ChannelsPage() {
  const channels = useChannels();
  const syncTitles = useSyncFeishuChannelTitles();
  const { showToast } = useToast();
  const [selectedId, setSelectedId] = useState<string>('');
  const [channelSearch, setChannelSearch] = useState('');

  const feishuChannels = useMemo(
    () => (channels.data?.channels || []).filter((item) => item.platform === 'feishu'),
    [channels.data?.channels]
  );
  const selectedChannelId = selectedId || feishuChannels[0]?.channel_id || '';
  const filteredChannels = useMemo(() => {
    const query = channelSearch.trim().toLowerCase();
    return feishuChannels.filter((item) => {
      if (!query) return true;
      return item.channel_id.toLowerCase().includes(query) || (item.title || '').toLowerCase().includes(query);
    });
  }, [channelSearch, feishuChannels]);

  return (
    <div className="space-y-5 xl:flex xl:h-[calc(100vh-6rem)] xl:flex-col xl:space-y-0">
      <PageHeader
        title="飞书学习群"
        description="管理已接入的飞书群、知识范围、群聊模式和调研任务。"
        className="xl:shrink-0"
        actions={
          <Button
            variant="secondary"
            size="sm"
            icon={<RefreshCw className="mr-2 h-4 w-4" />}
            onClick={() => syncTitles.mutate(undefined, {
              onSuccess: (result) => {
                if (result.attempted === 0) {
                  showToast('目前没有已接入的飞书群。', 'info');
                } else if (result.failed > 0) {
                  showToast(`已更新 ${result.updated} 个群名，${result.failed} 个未获取到；请检查飞书群信息读取权限。`, 'warning');
                } else if (result.updated > 0) {
                  showToast(`已更新 ${result.updated} 个飞书群名。`, 'success');
                } else {
                  showToast('飞书群名已是最新。', 'success');
                }
              },
              onError: () => showToast('同步群名失败，请稍后重试。', 'error'),
            })}
            isLoading={syncTitles.isPending}
          >
            同步群名
          </Button>
        }
      />

      <div className="grid gap-4 xl:min-h-0 xl:flex-1 xl:grid-cols-[280px_minmax(0,1fr)_360px]">
        <section className="rounded-panel border border-border bg-white p-4 xl:flex xl:min-h-0 xl:flex-col xl:overflow-hidden">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-primary">已接入的飞书群</h2>
            <StatusBadge status="active">{feishuChannels.length} 个</StatusBadge>
          </div>
          <div className="space-y-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-secondary" />
              <input
                className="w-full rounded-control border border-border py-1.5 pl-8 pr-2 text-xs focus:border-accent focus:outline-none"
                value={channelSearch}
                onChange={(event) => setChannelSearch(event.target.value)}
                placeholder="搜索群名称"
              />
            </div>
            <p className="text-[11px] text-secondary">
              {filteredChannels.length}/{feishuChannels.length} 个
            </p>
          </div>

          <div className="mt-2 max-h-[420px] space-y-2 overflow-y-auto pr-1 xl:min-h-0 xl:max-h-none xl:flex-1">
            {channels.isLoading && <p className="text-sm text-secondary">正在加载群...</p>}
            {channels.error && <ErrorPanel error={channels.error} onRetry={() => channels.refetch()} />}
            {filteredChannels.map((channel) => (
              <ChannelListItem
                key={channel.channel_id}
                channel={channel}
                active={channel.channel_id === selectedChannelId}
                onClick={() => setSelectedId(channel.channel_id)}
              />
            ))}
            {channels.data && feishuChannels.length === 0 && (
              <EmptyState title="暂无飞书群" description="机器人在飞书群收到第一条消息后，该群会自动出现在这里。" />
            )}
            {feishuChannels.length > 0 && filteredChannels.length === 0 && (
              <p className="py-4 text-center text-xs text-secondary">没有匹配的群，换个关键词试试。</p>
            )}
          </div>
        </section>

        <ChannelWorkspace channelId={selectedChannelId} />
        <ChannelJobs channelId={selectedChannelId} />
      </div>
    </div>
  );
}

function ChannelListItem({ channel, active, onClick }: { channel: Channel; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full rounded-control border px-3 py-2 text-left transition-colors',
        active ? 'border-accent bg-teal-50/60' : 'border-border hover:bg-page'
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm font-medium text-primary">{channel.title || channel.channel_id}</span>
        <StatusBadge status={channel.mode === 'kb_qa' ? 'answered' : 'ready'}>
          {channel.mode === 'kb_qa' ? '知识问答' : '学习助理'}
        </StatusBadge>
      </div>
      {channel.active_kb && <p className="mt-1 truncate text-xs text-accent">知识范围：{channel.active_kb}</p>}
    </button>
  );
}

function ChannelWorkspace({ channelId }: { channelId: string }) {
  const channel = useChannel(channelId);
  const jobs = useChannelJobs(channelId);
  const updateChannel = useUpdateChannel();
  const sendMessage = useSendChannelMessage();
  const activeMode = (channel.data?.mode || 'main') as ChannelMode;
  const messages = useChannelMessages(channelId, activeMode);
  const knowledgeTree = useKnowledgeTree();
  const scopeBinding = useChannelScopeBinding(channelId);
  const bindScope = useBindChannelScope();
  const unbindScope = useUnbindChannelScope();
  const [message, setMessage] = useState('');
  const [workspaceTab, setWorkspaceTab] = useState<'chat' | 'settings'>('chat');
  const messageListRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const activeConversationRef = useRef('');

  const mergedMessages = useMemo(() => messages.data?.messages || [], [messages.data]);
  const latestMessageId = mergedMessages[mergedMessages.length - 1]?.id;
  const scopeOptions = useMemo(
    () => buildKnowledgeScopeOptions(knowledgeTree.data).filter((option) => option.type !== 'research'),
    [knowledgeTree.data],
  );
  const boundScopeKey = scopeBinding.data && !scopeBinding.data.legacy
    ? knowledgeScopeKey(scopeBinding.data.scope_type as 'knowledge_base' | 'theme', scopeBinding.data.scope_id)
    : null;
  const boundScope = scopeOptions.find((option) => option.key === boundScopeKey);

  useLayoutEffect(() => {
    const list = messageListRef.current;
    if (!list) return;

    const conversationKey = `${channelId}:${activeMode}`;
    if (activeConversationRef.current !== conversationKey) {
      activeConversationRef.current = conversationKey;
      stickToBottomRef.current = true;
    }

    if (stickToBottomRef.current) {
      list.scrollTop = list.scrollHeight;
    }
  }, [activeMode, channelId, latestMessageId, mergedMessages.length, sendMessage.status]);

  useEffect(() => {
    const list = messageListRef.current;
    if (!list || typeof ResizeObserver === 'undefined') return;

    const observer = new ResizeObserver(() => {
      if (stickToBottomRef.current) {
        list.scrollTop = list.scrollHeight;
      }
    });
    observer.observe(list);
    return () => observer.disconnect();
  }, [activeMode, channelId]);

  if (!channelId) {
    return (
      <section className="rounded-panel border border-border bg-white p-4">
        <EmptyState title="选择一个飞书群" description="从左侧选择已接入的群后，在这里查看对话和管理群聊模式。" />
      </section>
    );
  }

  if (channel.error) {
    return <ErrorPanel error={channel.error} onRetry={() => channel.refetch()} />;
  }

  const submitMessage = () => {
    const clean = message.trim();
    if (!clean || !channelId || sendMessage.isPending) return;
    sendMessage.reset();
    setMessage('');
    sendMessage.mutate({ channelId, message: clean, mode: activeMode });
  };

  const setMode = (mode: ChannelMode) => {
    updateChannel.mutate({ channelId, patch: { mode } }, { onSuccess: () => channel.refetch() });
  };

  return (
    <section className="min-w-0 rounded-panel border border-border bg-white xl:flex xl:min-h-0 xl:flex-col xl:overflow-hidden">
      <div className="shrink-0 border-b border-border p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Hash className="h-5 w-5 text-accent" />
              <h2 className="truncate text-base font-semibold text-primary">
                {channel.data?.title || channelId}
              </h2>
              <StatusBadge status={activeMode === 'kb_qa' ? 'answered' : 'ready'}>
                {activeMode === 'kb_qa' ? '知识问答' : '学习助理'}
              </StatusBadge>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant={activeMode === 'main' ? 'primary' : 'secondary'} onClick={() => setMode('main')}>
              学习助理
            </Button>
            <Button size="sm" variant={activeMode === 'kb_qa' ? 'primary' : 'secondary'} onClick={() => setMode('kb_qa')}>
              知识问答
            </Button>
          </div>
        </div>
        <p className="mt-2 text-xs text-secondary">
          当前问答范围：{boundScope?.breadcrumb || (scopeBinding.data?.legacy ? `旧 topic：${scopeBinding.data.scope_id}` : '未绑定')}
          {' · '}更新：{channel.data?.updated_at ? formatDateTime(channel.data.updated_at) : '-'}
        </p>
        <div className="mt-3 inline-flex rounded-control border border-border bg-page/60 p-0.5">
          <button
            type="button"
            onClick={() => setWorkspaceTab('chat')}
            className={cn(
              'rounded-[8px] px-3 py-1.5 text-xs font-medium transition-colors',
              workspaceTab === 'chat' ? 'bg-accent text-white' : 'text-secondary hover:text-primary'
            )}
          >
            对话
          </button>
          <button
            type="button"
            onClick={() => setWorkspaceTab('settings')}
            className={cn(
              'rounded-[8px] px-3 py-1.5 text-xs font-medium transition-colors',
              workspaceTab === 'settings' ? 'bg-accent text-white' : 'text-secondary hover:text-primary'
            )}
          >
            设置
          </button>
        </div>
      </div>

      {workspaceTab === 'settings' ? (
        <div className="space-y-4 overflow-y-auto bg-page/30 p-4 xl:min-h-0 xl:flex-1">
          <div className="grid gap-2 md:grid-cols-3">
            <RuntimeMetric label="Main thread" value={channel.data?.threads?.main || `channel:${channelId}:main`} />
            <RuntimeMetric label="QA thread" value={channel.data?.threads?.kb_qa || `channel:${channelId}:kb_qa`} />
            <RuntimeMetric
              label="Running research"
              value={String((jobs.data?.jobs || []).filter((job) => job.job_type === 'research' && ['queued', 'running'].includes(job.status)).length)}
            />
          </div>

          <div>
            <div className="flex items-center gap-2 text-sm font-medium text-primary">
              <Link2 className="h-4 w-4 text-accent" />
              知识库绑定（持久，推送目标）
            </div>
            <p className="mt-1 text-xs text-secondary">
              绑定知识库或主题后，群内问答与主动推送都会默认限定在这个稳定范围。
            </p>
            <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto]">
              <KnowledgeScopeSelector
                value={boundScopeKey}
                options={scopeOptions}
                onChange={(option) => bindScope.mutate({
                  channelId,
                  scopeType: option.type as 'knowledge_base' | 'theme',
                  scopeId: option.id,
                })}
                placeholder="选择要绑定的知识范围"
              />
              <Button
                size="sm"
                variant="secondary"
                disabled={!scopeBinding.data || unbindScope.isPending}
                onClick={() => unbindScope.mutate(channelId)}
              >
                解绑
              </Button>
            </div>
          </div>

          <PushConfigPanel channelId={channelId} topicBound={Boolean(scopeBinding.data)} />
        </div>
      ) : (
        <>
          <div
            ref={messageListRef}
            className="h-[560px] space-y-3 overflow-y-auto bg-page/40 p-4 xl:min-h-0 xl:flex-1"
            onScroll={(event) => {
              const list = event.currentTarget;
              stickToBottomRef.current = list.scrollHeight - list.clientHeight - list.scrollTop <= 80;
            }}
          >
            {messages.isLoading && <p className="text-sm text-secondary">正在加载消息...</p>}
            {mergedMessages.map((item) => {
              // Only assistant replies carry job_id references worth rendering a card for -- a user
              // typing something that happens to look like a job_id shouldn't spawn a fake card.
              const jobId = item.role === 'assistant' ? extractJobId(item.content) : null;
              return (
                <div key={item.id} className={cn('flex flex-col', item.role === 'user' ? 'items-end' : 'items-start')}>
                  <div
                    className={cn(
                      'max-w-[78%] rounded-panel border px-3 py-2 text-sm',
                      item.role === 'user'
                        ? 'border-accent bg-accent text-white'
                        : 'border-border bg-white text-primary'
                    )}
                  >
                    {item.role === 'assistant' ? (
                      <MarkdownPreview
                        content={item.content}
                        className="text-sm leading-6 prose-headings:my-2 prose-p:my-2 prose-li:my-0.5 prose-ul:my-2 prose-ol:my-2 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                      />
                    ) : (
                      <p className="whitespace-pre-wrap break-words">{item.content}</p>
                    )}
                    <p className={cn('mt-1 text-[11px]', item.role === 'user' ? 'text-white/70' : 'text-secondary')}>
                      {formatDateTime(item.created_at)}
                    </p>
                  </div>
                  {jobId && <InlineJobCard jobId={jobId} />}
                </div>
              );
            })}
            {sendMessage.isPending
              && sendMessage.variables?.channelId === channelId
              && sendMessage.variables.mode === activeMode && (
              <div className="flex justify-start" role="status" aria-live="polite">
                <div className="flex items-center gap-2 rounded-panel border border-border bg-white px-3 py-2 text-sm text-secondary">
                  <span className="flex gap-1" aria-hidden="true">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-secondary" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-secondary [animation-delay:150ms]" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-secondary [animation-delay:300ms]" />
                  </span>
                  正在回复
                </div>
              </div>
            )}
            {sendMessage.isError
              && sendMessage.variables?.channelId === channelId
              && sendMessage.variables.mode === activeMode && (
              <div className="flex justify-start" role="alert">
                <div className="max-w-[78%] rounded-panel border border-red-200 bg-red-50 px-3 py-2 text-sm">
                  <p className="font-medium text-danger">回复失败</p>
                  <p className="mt-1 text-primary">
                    {sendMessage.error instanceof Error && sendMessage.error.message.trim()
                      ? sendMessage.error.message
                      : '暂时无法生成回答，请稍后再试。'}
                  </p>
                  {sendMessage.error instanceof Error && sendMessage.error.message.includes('模型不可用') && (
                    <Link className="mt-2 inline-block text-xs font-medium text-link hover:underline" to="/models">
                      检查模型设置
                    </Link>
                  )}
                </div>
              </div>
            )}
            {mergedMessages.length === 0 && !sendMessage.isPending && (
              <EmptyState title="还没有消息" description="发送一条消息，验证这个群当前由哪个 Agent 接管。" />
            )}
          </div>

          <div className="shrink-0 border-t border-border p-4">
            <div className="flex gap-2">
              <textarea
                className="min-h-[72px] flex-1 resize-none rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
                placeholder={activeMode === 'kb_qa' ? '向当前知识库提问...' : '和主管理 Agent 对话...'}
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    submitMessage();
                  }
                }}
              />
              <Button
                className="self-end"
                onClick={submitMessage}
                disabled={!message.trim()}
                isLoading={sendMessage.isPending}
                icon={<Send className="mr-2 h-4 w-4" />}
              >
                发送
              </Button>
            </div>
            <p className="mt-2 text-xs text-secondary">Enter 发送 · Shift + Enter 换行</p>
          </div>
        </>
      )}
    </section>
  );
}

function RuntimeMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-control border border-border bg-page/60 px-3 py-2">
      <p className="text-[11px] uppercase text-secondary">{label}</p>
      <p className="mt-1 truncate text-xs font-medium text-primary" title={value}>
        {value}
      </p>
    </div>
  );
}

const PUSH_FREQUENCY_OPTIONS: Array<{ value: ChannelPushFrequency; label: string }> = [
  { value: 'daily', label: '每天' },
  { value: 'weekly_3x', label: '每周三次' },
  { value: 'weekly_1x', label: '每周一次' },
];

const PUSH_CONTENT_STRATEGY_OPTIONS: Array<{ value: ChannelPushContentStrategy; label: string }> = [
  { value: 'rotation', label: '轮询复习+新增插队' },
  { value: 'new_only', label: '仅推新增' },
  { value: 'random', label: '随机' },
];

const PUSH_TIMEZONE_OPTIONS = [
  { value: 'Asia/Shanghai', label: '北京时间（UTC+8）' },
  { value: 'Asia/Tokyo', label: '东京时间（UTC+9）' },
  { value: 'Europe/London', label: '伦敦时间' },
  { value: 'America/New_York', label: '纽约时间' },
  { value: 'America/Los_Angeles', label: '洛杉矶时间' },
];

function pushTimezoneLabel(timezone: string) {
  return PUSH_TIMEZONE_OPTIONS.find((option) => option.value === timezone)?.label || timezone;
}

function PushConfigPanel({ channelId, topicBound }: { channelId: string; topicBound: boolean }) {
  const pushConfig = useChannelPushConfig(channelId);
  const updatePushConfig = useUpdateChannelPushConfig();
  const config = pushConfig.data;
  const controlsDisabled = !topicBound || pushConfig.isLoading;

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-primary">
          <BellRing className="h-4 w-4 text-accent" />
          知识主动推送
        </div>
        <Switch
          checked={Boolean(config?.enabled)}
          disabled={controlsDisabled}
          onChange={(checked) => updatePushConfig.mutate({ channelId, patch: { enabled: checked } })}
          label="推送开关"
        />
      </div>
      {!topicBound && <p className="mt-1 text-xs text-secondary">先绑定知识库后才能开启推送。</p>}
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <FilterSelect
          label="频率"
          value={config?.frequency || 'daily'}
          disabled={controlsDisabled}
          onChange={(event) =>
            updatePushConfig.mutate({
              channelId,
              patch: { frequency: event.target.value as ChannelPushFrequency },
            })
          }
          options={PUSH_FREQUENCY_OPTIONS}
        />
        <label className="flex items-center gap-2 text-sm text-secondary">
          <span>推送时间</span>
          <input
            type="time"
            value={config?.push_time || '09:00'}
            disabled={controlsDisabled}
            onChange={(event) =>
              updatePushConfig.mutate({ channelId, patch: { push_time: event.target.value } })
            }
            className="rounded-control border border-border bg-surface px-3 py-2 text-sm text-primary focus:border-accent focus:outline-none"
          />
        </label>
        <FilterSelect
          label="时区"
          value={config?.timezone || 'Asia/Shanghai'}
          disabled={controlsDisabled}
          onChange={(event) =>
            updatePushConfig.mutate({ channelId, patch: { timezone: event.target.value } })
          }
          options={PUSH_TIMEZONE_OPTIONS}
        />
        <FilterSelect
          label="内容策略"
          value={config?.content_strategy || 'rotation'}
          disabled={controlsDisabled}
          onChange={(event) =>
            updatePushConfig.mutate({
              channelId,
              patch: { content_strategy: event.target.value as ChannelPushContentStrategy },
            })
          }
          options={PUSH_CONTENT_STRATEGY_OPTIONS}
        />
      </div>
      {config?.enabled && (
        <div className="mt-3 rounded-control border border-accent/20 bg-accent/5 px-3 py-2.5">
          <p className="text-[11px] uppercase tracking-wide text-secondary">下次推送</p>
          <p className="mt-1 text-sm font-semibold text-primary">
            {config.next_push_at || '正在计算下一次时间'}
          </p>
          <p className="mt-0.5 text-xs text-secondary">{pushTimezoneLabel(config.timezone)}</p>
        </div>
      )}
      {config?.enabled && !config.scheduler_running && (
        <p className="mt-2 rounded-control border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          {config.scheduler_enabled
            ? '调度服务尚未运行，配置会保留，但暂时不会自动发送。'
            : '调度服务未启用，配置会保留，但暂时不会自动发送。'}
        </p>
      )}
      <p className="mt-2 text-xs text-secondary">
        上次推送：{config?.last_pushed_at ? formatDateTime(config.last_pushed_at) : '从未推送'}
        {config?.last_pushed_doc ? ` · ${config.last_pushed_doc}` : ''}
      </p>
    </div>
  );
}

function ChannelJobs({ channelId }: { channelId: string }) {
  const jobs = useChannelJobs(channelId);
  const createResearch = useCreateResearchJob();
  const [researchRequest, setResearchRequest] = useState('');
  const [researchDepth, setResearchDepth] = useState<'normal' | 'deep'>('normal');
  const researchJobs = jobs.data?.jobs.filter((job) => job.job_type === 'research') || [];
  const runningResearch = researchJobs.filter((job) => job.status === 'queued' || job.status === 'running');

  const submitResearch = () => {
    const clean = researchRequest.trim();
    if (!clean || !channelId) return;
    createResearch.mutate(
      {
        user_request: clean,
        source_mode: 'web',
        research_strategy: 'hv_analysis',
        depth: researchDepth,
        channel_id: channelId,
        delivery_target: 'web',
        output_mode: 'document_set',
        auto_save: false,
      },
      {
        onSuccess: () => {
          setResearchRequest('');
          jobs.refetch();
        },
      }
    );
  };

  return (
    <section className="rounded-panel border border-border bg-white p-4 xl:flex xl:min-h-0 xl:flex-col xl:overflow-hidden">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Search className="h-4 w-4 text-accent" />
          <h2 className="text-sm font-semibold text-primary">群内调研任务</h2>
        </div>
        <StatusBadge status={runningResearch.length ? 'running' : 'ready'}>{researchJobs.length}</StatusBadge>
      </div>
      <div className="mb-4 shrink-0 rounded-panel border border-border bg-page/40 p-3">
        <div className="flex items-center gap-2 text-sm font-medium text-primary">
          <FileSearch className="h-4 w-4 text-accent" />
          发起群内调研
        </div>
        <textarea
          className="mt-3 min-h-[84px] w-full resize-none rounded-control border border-border bg-white px-3 py-2 text-sm focus:border-accent focus:outline-none"
          placeholder="例如：深度调研企业级 Agent skill 书写指南"
          value={researchRequest}
          onChange={(event) => setResearchRequest(event.target.value)}
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <select
            className="rounded-control border border-border bg-white px-2 py-1.5 text-xs text-primary focus:border-accent focus:outline-none"
            value={researchDepth}
            onChange={(event) => setResearchDepth(event.target.value as 'normal' | 'deep')}
          >
            <option value="normal">normal</option>
            <option value="deep">deep</option>
          </select>
          <Button
            size="sm"
            onClick={submitResearch}
            disabled={!channelId || !researchRequest.trim()}
            isLoading={createResearch.isPending}
          >
            创建调研任务
          </Button>
        </div>
      </div>
      <div className="min-h-0 space-y-3 overflow-y-auto pr-1 xl:flex-1">
        {!channelId && <EmptyState title="请选择飞书群" description="选择群后，这里会显示与该群关联的调研任务。" />}
        {jobs.error && <ErrorPanel error={jobs.error} onRetry={() => jobs.refetch()} />}
        {researchJobs.map((job) => (
          <div key={job.job_id} className="space-y-2">
            <JobCard job={job} />
            <Link
              to={`/research?job_id=${encodeURIComponent(job.job_id)}`}
              className="inline-flex w-full items-center justify-center rounded-control border border-border bg-white px-3 py-2 text-xs font-medium text-primary hover:bg-page"
            >
              查看调研 / 发起补查
            </Link>
          </div>
        ))}
        {channelId && !jobs.isLoading && researchJobs.length === 0 && (
          <EmptyState title="暂无调研任务" description="后续从主 Agent 发起调研时，会绑定到当前群。" />
        )}
      </div>
    </section>
  );
}
