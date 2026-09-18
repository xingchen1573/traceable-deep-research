import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ChevronDown, ChevronUp, ExternalLink, Send, Loader2, X, History, Trash2, Plus, SlidersHorizontal } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import {
  buildKnowledgeScopeOptions,
  KnowledgeScopeSelector,
  type KnowledgeScopeOption,
} from '@/components/data/KnowledgeScopeSelector';
import { FilterSelect } from '@/components/controls/FormControls';
import { Button, IconButton } from '@/components/controls/Button';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { CitationList } from '@/components/data/CitationList';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { useKnowledgeTree, useResearchDocument } from '@/hooks';
import { Citation, QueryResponse } from '@/types';
import { formatRelativeTime } from '@/utils/formatters';
import {
  CITATION_SNIPPET_FALLBACK,
  citationSourceLabel,
  cleanCitationSnippet,
  parseMarkdownDocument,
} from '@/utils/citationDisplay';
import { api } from '@/api/client';
import { QUERY_MODE_OPTIONS, confidenceLabel, queryBackendLabel, queryModeLabel } from '@/utils/knowledgeQaLabels';
import {
  buildKnowledgeQueryHistory,
  knowledgeQaFailureAnswer,
  sanitizeKnowledgeQaAnswer,
} from './knowledgeQaConversation';

const QA_HISTORY_KEY = 'harness-agent:qa-history';
const QA_HISTORY_CACHE_LIMIT = 50;
const QA_HISTORY_PAGE_SIZE = 20;

interface QAHistoryEntry {
  id: string;
  topic: string;
  topicTitle: string;
  backend: string;
  mode: string;
  title: string;
  messages: QAConversationMessage[];
  createdAt: string;
  updatedAt: string;
}

interface QAConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  question?: string;
  answer?: QueryResponse;
  createdAt: string;
}

function loadQAHistory(): QAHistoryEntry[] {
  if (typeof window === 'undefined') return [];
  try {
    const value = window.localStorage.getItem(QA_HISTORY_KEY);
    if (!value) return [];
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    return parsed.map(normalizeHistoryEntry).filter(Boolean).slice(0, QA_HISTORY_CACHE_LIMIT) as QAHistoryEntry[];
  } catch {
    return [];
  }
}

function normalizeHistoryEntry(value: any): QAHistoryEntry | null {
  if (!value || typeof value !== 'object') return null;
  if (Array.isArray(value.messages)) {
    const title = String(value.title || value.messages.find((message: any) => message?.question)?.question || '未命名对话');
    return {
      id: String(value.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`),
      topic: String(value.topic || ''),
      topicTitle: String(value.topicTitle || value.topic_title || value.topic || ''),
      backend: String(value.backend || 'auto'),
      mode: String(value.mode || 'mix'),
      title,
      messages: value.messages.filter((message: any) => message?.role === 'user' || message?.role === 'assistant'),
      createdAt: String(value.createdAt || value.created_at || new Date().toISOString()),
      updatedAt: String(value.updatedAt || value.updated_at || value.createdAt || value.created_at || new Date().toISOString()),
    };
  }
  if (value.question && value.answer) {
    const createdAt = String(value.createdAt || new Date().toISOString());
    const question = String(value.question);
    return {
      id: String(value.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`),
      topic: String(value.topic || value.answer?.topic || ''),
      topicTitle: String(value.topicTitle || value.topic || value.answer?.topic || ''),
      backend: String(value.backend || value.answer?.backend || 'auto'),
      mode: String(value.mode || value.answer?.mode || 'mix'),
      title: question,
      messages: [
        {
          id: `${value.id || Date.now()}-user`,
          role: 'user',
          question,
          createdAt,
        },
        {
          id: `${value.id || Date.now()}-assistant`,
          role: 'assistant',
          question,
          answer: value.answer,
          createdAt,
        },
      ],
      createdAt,
      updatedAt: createdAt,
    };
  }
  return null;
}

function saveQAHistory(items: QAHistoryEntry[]) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(QA_HISTORY_KEY, JSON.stringify(items.slice(0, QA_HISTORY_CACHE_LIMIT)));
  } catch {
    // Storage can be unavailable in private/restricted browser contexts.
  }
}

export function KnowledgeQAPage() {
  const [searchParams] = useSearchParams();
  const knowledgeTree = useKnowledgeTree();
  const scopeOptions = useMemo(() => buildKnowledgeScopeOptions(knowledgeTree.data), [knowledgeTree.data]);
  const [topic, setTopic] = useState('');
  const backend = 'lightrag';
  const [mode, setMode] = useState('mix');
  const [question, setQuestion] = useState('');
  const [activeSession, setActiveSession] = useState<QAHistoryEntry | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [showFullCitationDocument, setShowFullCitationDocument] = useState(false);
  const [qaHistory, setQaHistory] = useState<QAHistoryEntry[]>(() => loadQAHistory());
  const [historyOffset, setHistoryOffset] = useState(0);
  const [hasMoreHistory, setHasMoreHistory] = useState(false);
  const [isLoadingMoreHistory, setIsLoadingMoreHistory] = useState(false);
  const [historyLoadError, setHistoryLoadError] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<Error | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const selectedScope = scopeOptions.find((option) => option.key === topic);
  const selectedCitationFilename = selectedCitation?.path?.split('/').pop() || '';
  const doc = useResearchDocument(selectedCitation?.research_id, selectedCitationFilename);
  const citationDocument = parseMarkdownDocument(doc.data?.content || '');
  const selectedCitationSnippet = cleanCitationSnippet(selectedCitation?.snippet || '');
  const citationDetailSnippet =
    selectedCitationSnippet === CITATION_SNIPPET_FALLBACK && citationDocument.body
      ? cleanCitationSnippet(citationDocument.body).slice(0, 420)
      : selectedCitationSnippet;

  useEffect(() => {
    if (!topic && scopeOptions.length) {
      const firstQueryable = scopeOptions.find((option) => option.researchIds.length > 0) || scopeOptions[0];
      setTopic(firstQueryable.key);
    }
  }, [scopeOptions, topic]);

  useEffect(() => {
    const nextTopic = searchParams.get('topic');
    const nextScopeType = searchParams.get('scope_type');
    const nextScopeId = searchParams.get('scope_id');
    const nextQuestion = searchParams.get('question');
    const nextMode = searchParams.get('mode');
    const directScope = nextScopeType && nextScopeId
      ? scopeOptions.find((option) => option.type === nextScopeType && option.id === nextScopeId)
      : undefined;
    const legacyScope = nextTopic
      ? scopeOptions.find((option) =>
          option.key === nextTopic || option.breadcrumb === nextTopic || option.title === nextTopic
        )
      : undefined;
    if (directScope || legacyScope) setTopic((directScope || legacyScope)!.key);
    if (nextQuestion) setQuestion(nextQuestion);
    if (nextMode) setMode(nextMode);
  }, [scopeOptions, searchParams]);

  const persistQAHistory = (entry: QAHistoryEntry) => {
    setQaHistory((current) => {
      const next = [entry, ...current.filter((item) => item.id !== entry.id)];
      saveQAHistory(next);
      return next;
    });
    void api.knowledgeBases.saveQaHistory(entry).catch(() => {
      // Keep the local fallback if the API is temporarily unavailable.
    });
  };

  useEffect(() => {
    let cancelled = false;
    api.knowledgeBases.listQaHistory(QA_HISTORY_PAGE_SIZE, 0).then(({ items, has_more }) => {
      if (cancelled) return;
      const remote = (items || []).map(normalizeHistoryEntry).filter(Boolean) as QAHistoryEntry[];
      if (remote.length) {
        setQaHistory(remote);
        saveQAHistory(remote);
      }
      setHistoryOffset(remote.length);
      setHasMoreHistory(has_more);
    }).catch(() => {
      // Existing local history remains available as a fallback.
    });
    return () => { cancelled = true; };
  }, []);

  const loadMoreHistory = async () => {
    if (isLoadingMoreHistory || !hasMoreHistory) return;
    setIsLoadingMoreHistory(true);
    setHistoryLoadError('');
    try {
      const response = await api.knowledgeBases.listQaHistory(QA_HISTORY_PAGE_SIZE, historyOffset);
      const remote = (response.items || []).map(normalizeHistoryEntry).filter(Boolean) as QAHistoryEntry[];
      setQaHistory((current) => {
        const existingIds = new Set(current.map((item) => item.id));
        const next = [...current, ...remote.filter((item) => !existingIds.has(item.id))];
        saveQAHistory(next);
        return next;
      });
      setHistoryOffset(response.offset + remote.length);
      setHasMoreHistory(response.has_more);
    } catch {
      setHistoryLoadError('加载失败，请重试');
    } finally {
      setIsLoadingMoreHistory(false);
    }
  };

  const createSession = (
    currentTopic: string,
    currentBackend: string,
    currentMode: string,
    firstQuestion: string,
    scope: KnowledgeScopeOption,
  ): QAHistoryEntry => {
    const now = new Date().toISOString();
    return {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      topic: currentTopic,
      topicTitle: scope.breadcrumb,
      backend: currentBackend,
      mode: currentMode,
      title: firstQuestion,
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
  };

  const submitQuestion = async () => {
    const currentQuestion = question.trim();
    if (!selectedScope || !currentQuestion) return;
    const currentTopic = topic;
    const currentBackend = backend;
    const currentMode = mode;
    const baseSession =
      activeSession &&
      activeSession.topic === currentTopic &&
      activeSession.backend === currentBackend &&
      activeSession.mode === currentMode
        ? activeSession
        : createSession(currentTopic, currentBackend, currentMode, currentQuestion, selectedScope);
    const userMessage: QAConversationMessage = {
      id: `${Date.now()}-user-${Math.random().toString(16).slice(2)}`,
      role: 'user',
      question: currentQuestion,
      createdAt: new Date().toISOString(),
    };
    const pendingSession = {
      ...baseSession,
      title: baseSession.title || currentQuestion,
      messages: [...baseSession.messages, userMessage],
      updatedAt: userMessage.createdAt,
    };
    setActiveSession(pendingSession);
    setSelectedCitation(null);
    setQuestion('');
    const assistantId = `${Date.now()}-assistant-${Math.random().toString(16).slice(2)}`;
    const assistantMessage: QAConversationMessage = {
      id: assistantId,
      role: 'assistant',
      question: currentQuestion,
      answer: {
        status: 'streaming',
        topic: currentTopic,
        backend: currentBackend as any,
        mode: currentMode as any,
        answer: '',
        citations: [],
        confidence: 'low',
        gaps: [],
      },
      createdAt: new Date().toISOString(),
    };
    const sessionWithPlaceholder = {
      ...pendingSession,
      messages: [...pendingSession.messages, assistantMessage],
      updatedAt: assistantMessage.createdAt,
    };
    setActiveSession(sessionWithPlaceholder);
    // Persist the conversation as soon as it starts, so a refresh or an
    // interrupted stream does not make the user's question disappear.
    persistQAHistory(sessionWithPlaceholder);
    setIsStreaming(true);
    setStreamError(null);
    try {
      const scopeResult = await api.knowledgeTree.queryScope(selectedScope.type, selectedScope.id, {
        question: currentQuestion,
        mode: currentMode as any,
        history: buildKnowledgeQueryHistory(baseSession.messages),
      });
      const data: QueryResponse = {
        status: scopeResult.status || 'answered',
        topic: currentTopic,
        backend: 'lightrag',
        requested_backend: 'lightrag',
        actual_backend: 'lightrag',
        mode: (scopeResult.mode || currentMode) as any,
        answer: scopeResult.answer || '',
        citations: (scopeResult.citations || []).map((citation) => ({
          title: citation.title || citation.research_title || citation.path?.split('/').pop() || '知识库引用',
          path: citation.path || '',
          snippet: citation.snippet || citation.content || citation.text || '',
          research_id: citation.research_id,
          research_title: citation.research_title,
        })),
        confidence: scopeResult.citations?.length >= 3 ? 'high' : scopeResult.citations?.length ? 'medium' : 'low',
        gaps: [...(scopeResult.gaps || []), ...(scopeResult.warnings || [])],
      };
      const finalAnswer = sanitizeKnowledgeQaAnswer(String(data.answer || '')).trim();
      if (!finalAnswer) throw new Error('问答服务未返回回答内容');
      const completedAnswer = { ...data, answer: finalAnswer };
      const nextSession = {
        ...sessionWithPlaceholder,
        messages: sessionWithPlaceholder.messages.map((message) =>
          message.id === assistantId ? { ...message, answer: completedAnswer } : message
        ),
        updatedAt: new Date().toISOString(),
      };
      setActiveSession(nextSession);
      persistQAHistory(nextSession);
    } catch (error) {
      const failure = error instanceof Error ? error : new Error('回答生成失败');
      const failedSession = {
        ...sessionWithPlaceholder,
        messages: sessionWithPlaceholder.messages.map((message) =>
          message.id === assistantId && message.answer
            ? {
                ...message,
                answer: {
                  ...message.answer,
                  status: 'error',
                  answer: knowledgeQaFailureAnswer(failure),
                  gaps: [failure.message],
                },
              }
            : message
        ),
        updatedAt: new Date().toISOString(),
      };
      setStreamError(failure);
      setActiveSession(failedSession);
      persistQAHistory(failedSession);
      setQuestion(currentQuestion);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitQuestion();
  };

  const restoreHistory = (entry: QAHistoryEntry) => {
    const restoredScope = scopeOptions.find((option) =>
      option.key === entry.topic || option.breadcrumb === entry.topicTitle || option.title === entry.topicTitle
    );
    if (restoredScope) setTopic(restoredScope.key);
    setMode(entry.mode);
    setQuestion('');
    setActiveSession(entry);
    setSelectedCitation(null);
  };

  const deleteHistory = (id: string) => {
    setQaHistory((current) => {
      const next = current.filter((item) => item.id !== id);
      saveQAHistory(next);
      return next;
    });
    void api.knowledgeBases.deleteQaHistory(id).then(({ deleted }) => {
      if (deleted) setHistoryOffset((current) => Math.max(0, current - 1));
    }).catch(() => {
      // The local list is already updated; retry is safe on a later refresh.
    });
    if (activeSession?.id === id) {
      setActiveSession(null);
      setSelectedCitation(null);
    }
  };

  const startNewSession = () => {
    setActiveSession(null);
    setQuestion('');
    setSelectedCitation(null);
    setShowFullCitationDocument(false);
  };

  const handleCitationClick = (citation: Citation) => {
    setShowFullCitationDocument(false);
    setSelectedCitation((current) =>
      current?.path === citation.path && current.title === citation.title ? null : citation
    );
  };

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col gap-3 md:h-[calc(100vh-6rem)]">
      <PageHeader title="知识库问答" description="基于知识库进行可信问答" />

      {!topic && <ErrorPanel error={{ code: 'need_topic', message: '请先选择知识范围', retryable: false }} />}

      <form
        onSubmit={handleSubmit}
        className="rounded-panel border border-border bg-white p-3"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0 max-w-xl flex-1 sm:min-w-[18rem]">
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
        </div>
        {showAdvanced && (
          <div className="mt-3 flex flex-wrap items-center gap-4 border-t border-border pt-3">
            <FilterSelect
              label="检索模式"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              options={QUERY_MODE_OPTIONS}
            />
            <span className="text-xs text-secondary">分层范围统一使用独立 ResearchItem 的 LightRAG 索引。</span>
          </div>
        )}
      </form>

      <div className="grid flex-1 gap-4 overflow-hidden lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="flex flex-1 flex-col overflow-hidden rounded-panel border border-border bg-white">
          <div className="flex-1 overflow-y-auto px-5 py-4">
            {streamError && (
              <ErrorPanel
                error={streamError}
                onRetry={() => {
                  setStreamError(null);
                  void submitQuestion();
                }}
              />
            )}
            {activeSession?.messages.length ? (
              <div className="mx-auto flex max-w-5xl flex-col gap-5">
                {activeSession.messages.map((message) => {
                  if (message.role === 'user') {
                    return (
                      <div key={message.id} className="flex justify-end">
                        <div className="max-w-[78%] rounded-control border border-accent/30 bg-accent px-4 py-3 text-white shadow-sm">
                          <p className="text-xs font-medium text-white/75">你</p>
                          <p className="mt-1 whitespace-pre-wrap text-sm font-medium leading-6">{message.question}</p>
                        </div>
                      </div>
                    );
                  }
                  const answerRequestId = message.answer?.metadata?.request_id
                    ? String(message.answer.metadata.request_id)
                    : '';
                  const isCurrentStreamingMessage =
                    isStreaming && message.id === activeSession.messages[activeSession.messages.length - 1]?.id;
                  const hasCompletedAnswer =
                    Boolean(message.answer?.answer?.trim()) && message.answer?.status !== 'streaming';
                  const answerContent = message.answer?.answer
                    ? sanitizeKnowledgeQaAnswer(message.answer.answer)
                    : (isCurrentStreamingMessage ? '正在生成回答...' : '上次回答中断，请重新提问。');
                  return (
                    <div key={message.id} className="flex justify-start">
                      <div className="max-w-[88%] space-y-3">
                        <div className="rounded-control border border-border bg-white px-4 py-3 shadow-sm">
                          <p className="mb-2 text-xs font-medium text-secondary">知识库</p>
                          <MarkdownPreview
                            content={answerContent}
                            className="text-[15px] leading-7 prose-p:my-2 prose-li:my-1 prose-headings:my-3"
                          />
                        </div>
                        {hasCompletedAnswer && (
                          <div className="flex flex-wrap items-center gap-2 pl-1 text-xs text-secondary">
                            <span>实际后端：{queryBackendLabel(message.answer?.actual_backend || message.answer?.backend)}</span>
                            <span>检索模式：{queryModeLabel(message.answer?.mode)}</span>
                            <span title="根据召回的相关知识片段数量与匹配度计算">
                              证据充分度：{confidenceLabel(message.answer?.confidence)}
                            </span>
                            {message.answer?.fallback_occurred && (
                              <span className="text-amber-800" title={message.answer.fallback_reason || ''}>
                                已自动回退{message.answer.fallback_reason ? `：${message.answer.fallback_reason}` : ''}
                              </span>
                            )}
                            {answerRequestId && (
                              <Link
                                to={`/observability?request_id=${encodeURIComponent(answerRequestId)}`}
                                className="text-link hover:underline"
                              >
                                {answerRequestId}
                              </Link>
                            )}
                          </div>
                        )}
                        <CitationList
                          citations={message.answer?.citations || []}
                          selectedCitation={selectedCitation}
                          onCitationClick={handleCitationClick}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-secondary">输入问题并提交</div>
            )}
          </div>
          <div className="border-t border-border p-3">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="输入问题..."
                className="flex-1 rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    submitQuestion();
                  }
                }}
              />
              <Button type="button" disabled={!topic || !question.trim() || isStreaming} onClick={submitQuestion}>
                {isStreaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </div>

        <aside className="hidden overflow-hidden rounded-panel border border-border bg-white lg:flex lg:flex-col">
          {selectedCitation ? (
            <>
            <div className="border-b border-border bg-page/60 px-4 py-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h4 className="truncate text-sm font-semibold text-primary">{selectedCitation.title}</h4>
                  <p className="mt-0.5 truncate font-mono text-xs text-secondary">{selectedCitation.path}</p>
                </div>
                <IconButton
                  icon={<X className="h-4 w-4" />}
                  label="关闭引用预览"
                  onClick={() => {
                    setSelectedCitation(null);
                    setShowFullCitationDocument(false);
                  }}
                  className="h-8 w-8 flex-shrink-0"
                />
              </div>
            </div>
            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              <section>
                <p className="mb-1.5 text-xs font-medium text-secondary">相关片段</p>
                <p className="line-clamp-6 text-sm leading-6 text-primary" title={citationDetailSnippet}>
                  {citationDetailSnippet}
                </p>
              </section>

              {doc.isLoading ? (
                <div className="text-sm text-secondary">正在加载文档信息...</div>
              ) : doc.data ? (
                <>
                  {citationDocument.sources.length > 0 && (
                    <section>
                      <p className="mb-2 text-xs font-medium text-secondary">
                        原始来源 · {citationDocument.sources.length}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {citationDocument.sources.map((source) => (
                          <a
                            key={source}
                            href={source}
                            target="_blank"
                            rel="noreferrer"
                            title={source}
                            className="inline-flex max-w-full items-center gap-1 rounded-control border border-border bg-page px-2 py-1 text-xs text-link hover:border-accent"
                          >
                            <span className="truncate">{citationSourceLabel(source)}</span>
                            <ExternalLink className="h-3 w-3 flex-shrink-0" />
                          </a>
                        ))}
                      </div>
                    </section>
                  )}

                  <section className="border-t border-border pt-3">
                    <button
                      type="button"
                      onClick={() => setShowFullCitationDocument((value) => !value)}
                      className="flex w-full items-center justify-between text-left text-sm font-medium text-primary hover:text-accent"
                    >
                      完整文档
                      {showFullCitationDocument ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )}
                    </button>
                    {showFullCitationDocument && (
                      <MarkdownPreview content={citationDocument.body} className="mt-3" />
                    )}
                  </section>
                </>
              ) : (
                <div className="text-sm text-secondary">无法加载完整文档</div>
              )}
            </div>
            </>
          ) : (
            <>
              <div className="flex items-center justify-between border-b border-border bg-page/60 px-4 py-3">
                <div className="flex items-center gap-2">
                  <History className="h-4 w-4 text-accent" />
                  <h4 className="text-sm font-semibold text-primary">对话历史</h4>
                </div>
                <IconButton
                  icon={<Plus className="h-4 w-4" />}
                  label="新建对话"
                  onClick={startNewSession}
                  className="h-8 w-8"
                />
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {qaHistory.length ? (
                  <div className="space-y-2">
                    {qaHistory.map((item) => (
                      <div
                        key={item.id}
                        className="group relative rounded-control border border-border hover:bg-page focus-within:border-accent"
                      >
                        <button
                          type="button"
                          onClick={() => restoreHistory(item)}
                          className="block w-full p-3 pr-9 text-left focus:outline-none"
                        >
                          <p className="max-h-10 overflow-hidden text-sm font-medium text-primary">{item.title}</p>
                          <p className="mt-1 text-xs text-secondary">
                            {item.topicTitle} · {Math.ceil(item.messages.length / 2)} 轮 · {formatRelativeTime(item.updatedAt)}
                          </p>
                        </button>
                        <button
                          type="button"
                          onClick={() => deleteHistory(item.id)}
                          className="absolute right-2 top-2 hidden rounded-control p-1 text-secondary hover:bg-red-50 hover:text-danger group-hover:block"
                          title="删除"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                    {hasMoreHistory && (
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        className="w-full"
                        isLoading={isLoadingMoreHistory}
                        onClick={loadMoreHistory}
                      >
                        加载更多
                      </Button>
                    )}
                    {historyLoadError && (
                      <p className="px-2 text-center text-xs text-danger">{historyLoadError}</p>
                    )}
                  </div>
                ) : (
                  <div className="px-3 py-8 text-center text-sm text-secondary">
                    暂无对话记录
                    <p className="mt-1 text-xs">提交知识库问答后自动保存</p>
                  </div>
                )}
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
