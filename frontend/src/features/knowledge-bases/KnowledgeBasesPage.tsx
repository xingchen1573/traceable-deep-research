import { useEffect, useMemo, useState } from 'react';
import {
  ArchiveRestore,
  BookOpen,
  Boxes,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  ExternalLink,
  FileText,
  FolderPlus,
  History,
  Loader2,
  MoreHorizontal,
  MoveRight,
  Plus,
  Quote,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  Tag,
  Trash2,
  X,
} from 'lucide-react';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { useToast } from '@/components/feedback/Toast';
import {
  useConfirmKnowledgePreview,
  useKnowledgeOverview,
  useKnowledgePreview,
  useKnowledgeScopeQuery,
  useKnowledgeTrash,
  useKnowledgeTree,
  useResearchDocument,
  useRetryResearchIndex,
} from '@/hooks';
import type {
  HierarchicalKnowledgeBase,
  KnowledgePreview,
  KnowledgeResearchItem,
  KnowledgeTheme,
  KnowledgeTreeAction,
  ResearchTimeStatus,
} from '@/types';
import { cn } from '@/utils/cn';
import './knowledge-workspace.css';

type Selection =
  | { type: 'knowledge_base'; id: string }
  | { type: 'theme'; id: string }
  | { type: 'research'; id: string };

type ActionKind = 'create_kb' | 'create_theme' | 'rename' | 'move' | 'trash' | 'status';

interface ActionDraft {
  kind: ActionKind;
  title: string;
  description: string;
  destinationKbId: string;
  destinationThemeId: string;
  status: ResearchTimeStatus;
  preview?: KnowledgePreview;
}

export function KnowledgeBasesPage() {
  const tree = useKnowledgeTree();
  const [selection, setSelection] = useState<Selection>();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState('');
  const [trashOpen, setTrashOpen] = useState(false);
  const [action, setAction] = useState<ActionDraft>();
  const { showToast } = useToast();

  useEffect(() => {
    const first = tree.data?.knowledge_bases[0];
    const currentStillExists = locateSelection(tree.data?.knowledge_bases || [], selection).knowledgeBase;
    if (first && (!selection || !currentStillExists)) {
      setSelection({ type: 'knowledge_base', id: first.id });
      setExpanded(new Set([first.id]));
    }
  }, [selection, tree.data?.knowledge_bases]);

  const context = useMemo(() => locateSelection(tree.data?.knowledge_bases || [], selection), [selection, tree.data]);
  const selectedNode = context.research || context.theme || context.knowledgeBase;
  const selectedOverviewType = context.research ? undefined : context.theme ? 'theme' : context.knowledgeBase ? 'knowledge_base' : undefined;
  const overview = useKnowledgeOverview(selectedOverviewType, selectedNode?.id);
  const [selectedFilename, setSelectedFilename] = useState<string>();

  useEffect(() => {
    setSelectedFilename(context.research?.documents[0]?.filename);
  }, [context.research?.id]);

  const document = useResearchDocument(context.research?.id, selectedFilename);
  const visibleKnowledgeBases = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    if (!query) return tree.data?.knowledge_bases || [];
    return (tree.data?.knowledge_bases || []).filter((kb) =>
      [kb.title, kb.description, ...kb.themes.map((theme) => theme.title), ...allResearch(kb).map((item) => item.title)]
        .join(' ')
        .toLocaleLowerCase()
        .includes(query),
    );
  }, [search, tree.data]);

  const openAction = (kind: ActionKind) => {
    setAction({
      kind,
      title: kind === 'rename' ? selectedNode?.title || '' : '',
      description: '',
      destinationKbId: context.knowledgeBase?.id || tree.data?.knowledge_bases[0]?.id || '',
      destinationThemeId: context.theme?.id || '',
      status: context.research?.status || 'active',
    });
  };

  return (
    <div className="knowledge-workspace -mx-4 -mt-4 min-h-[calc(100vh-7rem)] md:-mx-6 md:-mt-6">
      <header className="kw-header">
        <div>
          <p className="kw-kicker">PERSONAL KNOWLEDGE SYSTEM</p>
          <div className="mt-1 flex items-center gap-3">
            <h1>知识库</h1>
            <span className="kw-count">{tree.data?.knowledge_bases.length || 0} 个知识库</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <IconAction label="刷新知识树" onClick={() => tree.refetch()} icon={<RefreshCw className={cn('h-4 w-4', tree.isFetching && 'animate-spin')} />} />
          <button className="kw-command" type="button" onClick={() => openAction('create_kb')}>
            <Plus className="h-4 w-4" /> 新建知识库
          </button>
        </div>
      </header>

      {tree.error && <div className="p-5"><ErrorPanel error={tree.error} onRetry={() => tree.refetch()} /></div>}

      <div className="kw-layout">
        <aside className="kw-tree-panel">
          <div className="kw-search">
            <Search className="h-4 w-4" />
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索层级和调研" />
          </div>
          <div className="kw-tree-scroll">
            {tree.isLoading ? (
              <LoadingLine label="读取知识结构" />
            ) : visibleKnowledgeBases.length ? (
              visibleKnowledgeBases.map((kb) => (
                <KnowledgeTreeBranch
                  key={kb.id}
                  knowledgeBase={kb}
                  selection={selection}
                  expanded={expanded}
                  onSelect={setSelection}
                  onToggle={(id) => setExpanded((current) => toggleSet(current, id))}
                />
              ))
            ) : (
              <EmptyTree onCreate={() => openAction('create_kb')} />
            )}
          </div>
          <button className="kw-trash-link" type="button" onClick={() => setTrashOpen(true)}>
            <Trash2 className="h-4 w-4" />
            回收站
            <span>{tree.data?.trash_count || 0}</span>
          </button>
        </aside>

        <main className="kw-content-panel">
          {selectedNode ? (
            <>
              <NodeHeader
                type={selection?.type || 'knowledge_base'}
                title={selectedNode.title}
                description={'description' in selectedNode ? selectedNode.description : context.research?.summary}
                path={selectedNode.path}
                count={context.research ? context.research.document_count : (context.theme || context.knowledgeBase)?.research_count || 0}
                onRename={() => openAction('rename')}
                fullHtmlUrl={context.research?.html_filename ? `/api/v1/knowledge-tree/research/${encodeURIComponent(context.research.id)}/reader` : undefined}
              />
              <div className="kw-content-scroll">
                {context.research ? (
                  <ResearchReader
                    research={context.research}
                    selectedFilename={selectedFilename}
                    onSelectFilename={setSelectedFilename}
                    document={document.data}
                    loading={document.isLoading}
                  />
                ) : (
                  <ScopeOverview
                    content={overview.data?.content}
                    loading={overview.isLoading}
                    knowledgeBase={context.knowledgeBase}
                    theme={context.theme}
                    onSelectResearch={(id) => setSelection({ type: 'research', id })}
                    onSelectTheme={(id) => setSelection({ type: 'theme', id })}
                  />
                )}
              </div>
            </>
          ) : (
            <EmptyTree onCreate={() => openAction('create_kb')} />
          )}
        </main>

        <Inspector
          selection={selection}
          knowledgeBase={context.knowledgeBase}
          theme={context.theme}
          research={context.research}
          onCreateTheme={() => openAction('create_theme')}
          onMove={() => openAction('move')}
          onStatus={() => openAction('status')}
          onTrash={() => openAction('trash')}
          onTreeRefresh={() => tree.refetch()}
        />
      </div>

      {action && (
        <ActionModal
          draft={action}
          selection={selection}
          knowledgeBases={tree.data?.knowledge_bases || []}
          currentKb={context.knowledgeBase}
          onChange={setAction}
          onClose={() => setAction(undefined)}
          onCommitted={(result) => {
            setAction(undefined);
            showToast('变更已确认并写入知识库', 'success');
            const nodeId = String(result?.result?.node_id || '');
            const nodeType = String(result?.result?.node_type || '');
            if (nodeId && ['knowledge_base', 'theme', 'research'].includes(nodeType)) {
              setSelection({ type: nodeType as Selection['type'], id: nodeId });
            }
          }}
        />
      )}
      {trashOpen && <TrashDrawer onClose={() => setTrashOpen(false)} />}
    </div>
  );
}

function KnowledgeTreeBranch({ knowledgeBase, selection, expanded, onSelect, onToggle }: {
  knowledgeBase: HierarchicalKnowledgeBase;
  selection?: Selection;
  expanded: Set<string>;
  onSelect: (selection: Selection) => void;
  onToggle: (id: string) => void;
}) {
  const isOpen = expanded.has(knowledgeBase.id);
  return (
    <div className="kw-tree-branch">
      <div className="kw-tree-row kw-tree-root">
        <button type="button" className="kw-chevron" onClick={() => onToggle(knowledgeBase.id)} aria-label={isOpen ? '收起' : '展开'}>
          {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
        <button type="button" onClick={() => onSelect({ type: 'knowledge_base', id: knowledgeBase.id })} className={cn('kw-node-button', selection?.id === knowledgeBase.id && 'is-active')}>
          <Boxes className="h-4 w-4" />
          <span>{knowledgeBase.title}</span>
          <small>{knowledgeBase.research_count}</small>
        </button>
      </div>
      {isOpen && (
        <div className="kw-tree-children">
          {knowledgeBase.research_items.map((research) => (
            <ResearchTreeRow key={research.id} research={research} selection={selection} onSelect={onSelect} />
          ))}
          {knowledgeBase.themes.map((theme) => {
            const themeOpen = expanded.has(theme.id);
            return (
              <div key={theme.id}>
                <div className="kw-tree-row">
                  <button type="button" className="kw-chevron" onClick={() => onToggle(theme.id)} aria-label={themeOpen ? '收起' : '展开'}>
                    {themeOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                  </button>
                  <button type="button" onClick={() => onSelect({ type: 'theme', id: theme.id })} className={cn('kw-node-button', selection?.id === theme.id && 'is-active')}>
                    <BookOpen className="h-4 w-4" />
                    <span>{theme.title}</span>
                    <small>{theme.research_count}</small>
                  </button>
                </div>
                {themeOpen && <div className="kw-tree-leaves">{theme.research_items.map((research) => <ResearchTreeRow key={research.id} research={research} selection={selection} onSelect={onSelect} />)}</div>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ResearchTreeRow({ research, selection, onSelect }: { research: KnowledgeResearchItem; selection?: Selection; onSelect: (selection: Selection) => void }) {
  return (
    <button type="button" onClick={() => onSelect({ type: 'research', id: research.id })} className={cn('kw-research-row', selection?.id === research.id && 'is-active')}>
      <FileText className="h-3.5 w-3.5" />
      <span>{research.title}</span>
      <i className={cn('kw-status-dot', `is-${research.status}`)} />
    </button>
  );
}

function NodeHeader({ type, title, description, path, count, onRename, fullHtmlUrl }: { type: Selection['type']; title: string; description?: string; path: string; count: number; onRename: () => void; fullHtmlUrl?: string }) {
  return (
    <div className="kw-node-header">
      <div className="min-w-0">
        <p className="kw-kicker">{nodeTypeLabel(type)}</p>
        <h2>{title}</h2>
        <p className="kw-node-description">{description || (type === 'research' ? '一次独立、可追溯的调研记录' : '持续更新的学习范围')}</p>
        <p className="kw-path">{path}</p>
      </div>
      <div className="kw-node-actions">
        {fullHtmlUrl && <a className="kw-report-html" href={fullHtmlUrl} target="_blank" rel="noreferrer"><ExternalLink className="h-4 w-4" />完整 HTML 阅读版</a>}
        <span className="kw-count">{count} {type === 'research' ? '份文档' : '次调研'}</span>
        <IconAction label="重命名" icon={<MoreHorizontal className="h-4 w-4" />} onClick={onRename} />
      </div>
    </div>
  );
}

function ScopeOverview({ content, loading, knowledgeBase, theme, onSelectResearch, onSelectTheme }: {
  content?: string;
  loading: boolean;
  knowledgeBase?: HierarchicalKnowledgeBase;
  theme?: KnowledgeTheme;
  onSelectResearch: (id: string) => void;
  onSelectTheme: (id: string) => void;
}) {
  const directResearch = theme?.research_items || knowledgeBase?.research_items || [];
  return (
    <div className="space-y-8">
      <section>
        <SectionLabel label="范围总览" />
        {loading ? <LoadingLine label="读取总览" /> : content ? <MarkdownPreview content={cleanOverview(content)} className="kw-markdown" /> : <p className="kw-muted">暂无总览。</p>}
      </section>
      {!!directResearch.length && (
        <section>
          <SectionLabel label={theme ? 'Theme 内调研' : '直属调研'} />
          <div className="kw-research-list">{directResearch.map((item) => <ResearchListItem key={item.id} item={item} onClick={() => onSelectResearch(item.id)} />)}</div>
        </section>
      )}
      {!theme && knowledgeBase?.themes.map((item) => (
        <section key={item.id} className="kw-theme-band">
          <button type="button" onClick={() => onSelectTheme(item.id)} className="kw-theme-heading">
            <span><BookOpen className="h-4 w-4" /> {item.title}</span>
            <small>{item.research_count} 次调研 <ChevronRight className="h-4 w-4" /></small>
          </button>
          {item.research_items.length ? <div className="kw-research-list">{item.research_items.map((research) => <ResearchListItem key={research.id} item={research} onClick={() => onSelectResearch(research.id)} />)}</div> : <p className="kw-muted py-4">这个 Theme 还没有调研。</p>}
        </section>
      ))}
    </div>
  );
}

function ResearchListItem({ item, onClick }: { item: KnowledgeResearchItem; onClick: () => void }) {
  return (
    <button className="kw-research-item" type="button" onClick={onClick}>
      <div className="kw-date-block"><strong>{datePart(item.researched_at, 'day')}</strong><span>{datePart(item.researched_at, 'month')}</span></div>
      <div className="min-w-0 flex-1"><h3>{item.title}</h3><p>{item.summary || '暂无摘要'}</p><div className="kw-meta-line"><span>{item.document_count} 份文档</span><span>{item.source_count || 0} 个来源</span><span>{statusLabel(item.status)}</span></div></div>
      <ChevronRight className="h-4 w-4 text-faint" />
    </button>
  );
}

function ResearchReader({ research, selectedFilename, onSelectFilename, document: documentData, loading }: { research: KnowledgeResearchItem; selectedFilename?: string; onSelectFilename: (filename: string) => void; document?: import('@/types').ResearchDocumentResponse; loading: boolean }) {
  const [activeCitation, setActiveCitation] = useState<number>();
  useEffect(() => setActiveCitation(undefined), [selectedFilename]);
  const openCitation = (number: number) => {
    setActiveCitation(number);
    requestAnimationFrame(() => window.document.getElementById(`citation-${number}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
  };
  const htmlUrl = selectedFilename
    ? `/api/v1/knowledge-tree/research/${encodeURIComponent(research.id)}/documents/${encodeURIComponent(selectedFilename)}/html`
    : '';
  return (
    <div>
      <div className="kw-document-tabs" role="tablist">
        {research.documents.map((item) => <button key={item.filename} type="button" role="tab" aria-selected={selectedFilename === item.filename} onClick={() => onSelectFilename(item.filename)} className={cn(selectedFilename === item.filename && 'is-active')}><FileText className="h-3.5 w-3.5" />{item.title || item.filename}</button>)}
      </div>
      <div className="kw-reader-toolbar">
        <span>{documentData?.citations.length || 0} 个引用来源</span>
        {documentData?.has_html && <a href={htmlUrl} target="_blank" rel="noreferrer"><ExternalLink className="h-3.5 w-3.5" />本章 HTML</a>}
      </div>
      <div className="kw-document-body">
        {loading ? <LoadingLine label="读取文档" /> : documentData?.content ? (
          <>
            <MarkdownPreview
              content={documentData.content}
              className="kw-markdown"
              citationNumbers={documentData.citations.map((item) => item.number)}
              activeCitationNumber={activeCitation}
              onCitationClick={openCitation}
            />
            <CitationList citations={documentData.citations} activeCitation={activeCitation} />
          </>
        ) : <p className="kw-muted">请选择一份文档。</p>}
      </div>
    </div>
  );
}

function CitationList({ citations, activeCitation }: { citations: import('@/types').ResearchDocumentResponse['citations']; activeCitation?: number }) {
  if (!citations.length) return null;
  return (
    <section className="kw-citations" aria-label="参考来源">
      <SectionLabel label={`参考来源 · ${citations.length}`} />
      <div className="kw-citation-list">
        {citations.map((citation) => <CitationCard key={citation.number} citation={citation} active={activeCitation === citation.number} />)}
      </div>
    </section>
  );
}

function CitationCard({ citation, active }: { citation: import('@/types').ResearchDocumentResponse['citations'][number]; active: boolean }) {
  const primaryEvidence = citation.evidence.slice(0, 3);
  const additionalEvidence = citation.evidence.slice(3);
  const renderEvidence = (evidence: typeof citation.evidence[number], index: number) => (
    <blockquote key={evidence.evidence_id || `${citation.number}-${index}`}><Quote className="h-3.5 w-3.5" /><span>{evidence.quote || evidence.claim}</span></blockquote>
  );
  return (
    <article id={`citation-${citation.number}`} className={cn('kw-citation-card', active && 'is-active')}>
      <div className="kw-citation-number">[{citation.number}]</div>
      <div className="min-w-0">
        <h4>{citation.url ? <a href={citation.url} target="_blank" rel="noreferrer">{citation.title || citation.url}<ExternalLink className="h-3 w-3" /></a> : citation.title || `来源 ${citation.number}`}</h4>
        {primaryEvidence.length ? primaryEvidence.map(renderEvidence) : <p>该来源没有保留可逐字展示的证据片段。</p>}
        {!!additionalEvidence.length && <details><summary>查看另外 {additionalEvidence.length} 条证据</summary>{additionalEvidence.map((item, index) => renderEvidence(item, index + 3))}</details>}
      </div>
    </article>
  );
}

function Inspector({ selection, knowledgeBase, theme, research, onCreateTheme, onMove, onStatus, onTrash, onTreeRefresh }: {
  selection?: Selection;
  knowledgeBase?: HierarchicalKnowledgeBase;
  theme?: KnowledgeTheme;
  research?: KnowledgeResearchItem;
  onCreateTheme: () => void;
  onMove: () => void;
  onStatus: () => void;
  onTrash: () => void;
  onTreeRefresh: () => void;
}) {
  const query = useKnowledgeScopeQuery();
  const retryIndex = useRetryResearchIndex();
  const [question, setQuestion] = useState('');
  if (!selection || !knowledgeBase) return <aside className="kw-inspector" />;
  const scopeTitle = research?.title || theme?.title || knowledgeBase.title;
  return (
    <aside className="kw-inspector">
      <section>
        <SectionLabel label="当前范围" />
        <h3 className="kw-inspector-title">{scopeTitle}</h3>
        <dl className="kw-facts">
          <div><dt>类型</dt><dd>{nodeTypeLabel(selection.type)}</dd></div>
          <div><dt>调研</dt><dd>{research ? 1 : theme?.research_count ?? knowledgeBase.research_count}</dd></div>
          {research && <><div><dt>时间状态</dt><dd>{statusLabel(research.status)}</dd></div><div><dt>索引</dt><dd>{research.index_status?.status || 'pending'}</dd></div></>}
        </dl>
        <div className="kw-action-grid">
          {selection.type === 'knowledge_base' && <button type="button" onClick={onCreateTheme}><FolderPlus className="h-4 w-4" />新建 Theme</button>}
          {selection.type === 'research' && <button type="button" onClick={onMove}><MoveRight className="h-4 w-4" />移动</button>}
          {selection.type === 'research' && <button type="button" onClick={onStatus}><History className="h-4 w-4" />时间状态</button>}
          {selection.type === 'research' && research && research.index_status?.status !== 'ready' && <button type="button" onClick={() => retryIndex.mutate(research.id, { onSuccess: onTreeRefresh })} disabled={retryIndex.isPending}><RotateCcw className={cn('h-4 w-4', retryIndex.isPending && 'animate-spin')} />重建索引</button>}
          <button type="button" className="is-danger" onClick={onTrash}><Trash2 className="h-4 w-4" />移入回收站</button>
        </div>
      </section>
      {research && !!research.tags?.length && <section><SectionLabel label="标签" /><div className="kw-tags">{research.tags.map((item) => <span key={item}><Tag className="h-3 w-3" />{item}</span>)}</div></section>}
      <section className="kw-ask">
        <SectionLabel label="询问这个范围" />
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={`向“${scopeTitle}”提问`} rows={4} />
        <button type="button" disabled={!question.trim() || query.isPending} onClick={() => query.mutate({ scopeType: selection.type, scopeId: selection.id, request: { question, mode: 'mix' } })}>{query.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}提问</button>
        {query.data && <div className="kw-answer"><MarkdownPreview content={query.data.answer} />{!!query.data.warnings?.length && <p className="kw-warning"><CircleAlert className="h-4 w-4" />{query.data.warnings.join(' ')}</p>}</div>}
      </section>
    </aside>
  );
}

function ActionModal({ draft, selection, knowledgeBases, currentKb, onChange, onClose, onCommitted }: {
  draft: ActionDraft;
  selection?: Selection;
  knowledgeBases: HierarchicalKnowledgeBase[];
  currentKb?: HierarchicalKnowledgeBase;
  onChange: (draft: ActionDraft) => void;
  onClose: () => void;
  onCommitted: (preview: KnowledgePreview) => void;
}) {
  const previewMutation = useKnowledgePreview();
  const confirmMutation = useConfirmKnowledgePreview();
  const change = (values: Partial<ActionDraft>) => {
    const explicitlySetsPreview = Object.prototype.hasOwnProperty.call(values, 'preview');
    const editsForm = Object.keys(values).some((key) => key !== 'preview');
    onChange({ ...draft, ...values, preview: explicitlySetsPreview ? values.preview : editsForm ? undefined : draft.preview });
  };
  const createPreview = () => {
    const operation = actionPayload(draft, selection, currentKb);
    previewMutation.mutate(operation, { onSuccess: (preview) => change({ preview }) });
  };
  const targetKb = knowledgeBases.find((kb) => kb.id === draft.destinationKbId);
  return (
    <div className="kw-modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="kw-modal" role="dialog" aria-modal="true" aria-labelledby="kw-modal-title">
        <div className="kw-modal-header"><div><p className="kw-kicker">PREVIEW / CONFIRM</p><h2 id="kw-modal-title">{actionTitle(draft.kind)}</h2></div><IconAction label="关闭" icon={<X className="h-4 w-4" />} onClick={onClose} /></div>
        <div className="kw-modal-body">
          {!draft.preview ? (
            <div className="space-y-4">
              {['create_kb', 'create_theme', 'rename'].includes(draft.kind) && <Field label="名称"><input value={draft.title} onChange={(event) => change({ title: event.target.value })} autoFocus /></Field>}
              {['create_kb', 'create_theme'].includes(draft.kind) && <Field label="描述（可选）"><textarea value={draft.description} onChange={(event) => change({ description: event.target.value })} rows={3} /></Field>}
              {draft.kind === 'move' && <><Field label="目标知识库"><select value={draft.destinationKbId} onChange={(event) => change({ destinationKbId: event.target.value, destinationThemeId: '' })}>{knowledgeBases.map((kb) => <option key={kb.id} value={kb.id}>{kb.title}</option>)}</select></Field><Field label="目标位置"><select value={draft.destinationThemeId} onChange={(event) => change({ destinationThemeId: event.target.value })}><option value="">知识库直属</option>{targetKb?.themes.map((theme) => <option key={theme.id} value={theme.id}>{theme.title}</option>)}</select></Field></>}
              {draft.kind === 'status' && <Field label="时间状态"><select value={draft.status} onChange={(event) => change({ status: event.target.value as ResearchTimeStatus })}><option value="active">当前有效</option><option value="possibly_stale">可能过时</option><option value="superseded">已被替代</option></select></Field>}
              {draft.kind === 'trash' && <div className="kw-risk-note"><Trash2 className="h-5 w-5" /><div><strong>节点会移入回收站</strong><p>默认问答和推送将不再使用它，之后仍可恢复。</p></div></div>}
            </div>
          ) : (
            <div>
              <p className="kw-preview-summary">确认后将执行 {draft.preview.changes.length} 项文件变更。系统会在执行前再次校验原始内容。</p>
              <div className="kw-change-list">{draft.preview.changes.map((item, index) => <div key={`${item.op}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{changeLabel(item.op)}</strong><p>{String(item.to || item.path || item.from || item.title || '')}</p></div></div>)}</div>
            </div>
          )}
          {(previewMutation.error || confirmMutation.error) && <ErrorPanel error={previewMutation.error || confirmMutation.error} />}
        </div>
        <div className="kw-modal-footer"><button type="button" className="kw-secondary-command" onClick={draft.preview ? () => change({ preview: undefined }) : onClose}>{draft.preview ? '返回修改' : '取消'}</button>{draft.preview ? <button type="button" className="kw-command" onClick={() => confirmMutation.mutate(draft.preview!.preview_id, { onSuccess: onCommitted })} disabled={confirmMutation.isPending}>{confirmMutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}确认执行</button> : <button type="button" className="kw-command" onClick={createPreview} disabled={previewMutation.isPending || (['create_kb', 'create_theme', 'rename'].includes(draft.kind) && !draft.title.trim())}>{previewMutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}生成变更预览</button>}</div>
      </div>
    </div>
  );
}

function TrashDrawer({ onClose }: { onClose: () => void }) {
  const trash = useKnowledgeTrash();
  const preview = useKnowledgePreview();
  const confirm = useConfirmKnowledgePreview();
  const { showToast } = useToast();
  const restore = (nodeId: string) => preview.mutate({ action: 'restore_node', payload: { node_id: nodeId } }, { onSuccess: (result) => confirm.mutate(result.preview_id, { onSuccess: () => { showToast('已从回收站恢复', 'success'); trash.refetch(); } }) });
  return (
    <div className="kw-drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="kw-drawer" role="dialog" aria-modal="true" aria-labelledby="kw-trash-title">
        <div className="kw-modal-header">
          <div><p className="kw-kicker">RECYCLE BIN</p><h2 id="kw-trash-title">回收站</h2></div>
          <IconAction label="关闭" icon={<X className="h-4 w-4" />} onClick={onClose} />
        </div>
        <div className="kw-drawer-body">
          {trash.isLoading ? <LoadingLine label="读取回收站" /> : trash.data?.items.length ? trash.data.items.map((item) => (
            <div className="kw-trash-item" key={item.id}>
              <div className="kw-trash-item-copy"><strong>{item.title}</strong><p>{nodeTypeLabel(item.type)} · {item.original_path}</p></div>
              <button type="button" title="恢复" aria-label={`恢复 ${item.title}`} onClick={() => restore(item.id)} disabled={preview.isPending || confirm.isPending}><ArchiveRestore className="h-4 w-4" />恢复</button>
            </div>
          )) : <div className="kw-empty"><Trash2 className="h-7 w-7" /><p>回收站是空的</p></div>}
        </div>
      </aside>
    </div>
  );
}

function actionPayload(draft: ActionDraft, selection?: Selection, currentKb?: HierarchicalKnowledgeBase): { action: KnowledgeTreeAction; payload: Record<string, unknown> } {
  if (draft.kind === 'create_kb') return { action: 'create_knowledge_base', payload: { title: draft.title, description: draft.description } };
  if (draft.kind === 'create_theme') return { action: 'create_theme', payload: { knowledge_base_id: currentKb?.id, title: draft.title, description: draft.description } };
  if (draft.kind === 'rename') return { action: 'rename_node', payload: { node_id: selection?.id, title: draft.title } };
  if (draft.kind === 'move') return { action: 'move_research', payload: { research_id: selection?.id, knowledge_base_id: draft.destinationKbId, theme_id: draft.destinationThemeId || null } };
  if (draft.kind === 'status') return { action: 'update_research_status', payload: { research_id: selection?.id, status: draft.status } };
  return { action: 'trash_node', payload: { node_id: selection?.id } };
}

function locateSelection(knowledgeBases: HierarchicalKnowledgeBase[], selection?: Selection) {
  for (const knowledgeBase of knowledgeBases) {
    if (selection?.type === 'knowledge_base' && selection.id === knowledgeBase.id) return { knowledgeBase };
    for (const research of knowledgeBase.research_items) if (selection?.type === 'research' && selection.id === research.id) return { knowledgeBase, research };
    for (const theme of knowledgeBase.themes) {
      if (selection?.type === 'theme' && selection.id === theme.id) return { knowledgeBase, theme };
      for (const research of theme.research_items) if (selection?.type === 'research' && selection.id === research.id) return { knowledgeBase, theme, research };
    }
  }
  return {};
}

function allResearch(kb: HierarchicalKnowledgeBase) { return [...kb.research_items, ...kb.themes.flatMap((theme) => theme.research_items)]; }
function toggleSet(current: Set<string>, value: string) { const next = new Set(current); next.has(value) ? next.delete(value) : next.add(value); return next; }
function nodeTypeLabel(type: string) { return ({ knowledge_base: '知识库', theme: 'Theme', research: 'ResearchItem', legacy: '旧测试知识库' } as Record<string, string>)[type] || type; }
function statusLabel(status: ResearchTimeStatus) { return ({ active: '当前有效', possibly_stale: '可能过时', superseded: '已被替代' } as const)[status]; }
function actionTitle(kind: ActionKind) { return ({ create_kb: '新建知识库', create_theme: '新建 Theme', rename: '重命名节点', move: '移动调研', trash: '移入回收站', status: '更新时间状态' } as const)[kind]; }
function changeLabel(op: string) { return ({ create: '创建节点', rename: '重命名目录', move: '移动调研', trash: '移入回收站', restore: '恢复节点', update_status: '更新时间状态', update_overview: '刷新总览' } as Record<string, string>)[op] || op; }
function datePart(value: string | undefined, part: 'day' | 'month') { if (!value) return part === 'day' ? '--' : '未知'; const date = new Date(value); return part === 'day' ? String(date.getDate()).padStart(2, '0') : `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}`; }
function cleanOverview(content: string) { return content.replace(/<!--\s*harness-managed:(?:start|end)\s*-->/g, '').replace(/\n{3,}/g, '\n\n'); }
function SectionLabel({ label }: { label: string }) { return <div className="kw-section-label"><span>{label}</span><i /></div>; }
function LoadingLine({ label }: { label: string }) { return <div className="kw-loading"><Loader2 className="h-4 w-4 animate-spin" />{label}</div>; }
function Field({ label, children }: { label: string; children: React.ReactNode }) { return <label className="kw-field"><span>{label}</span>{children}</label>; }
function IconAction({ label, icon, onClick }: { label: string; icon: React.ReactNode; onClick: () => void }) { return <button className="kw-icon-action" type="button" title={label} aria-label={label} onClick={onClick}>{icon}</button>; }
function EmptyTree({ onCreate }: { onCreate: () => void }) { return <div className="kw-empty"><Boxes className="h-8 w-8" /><p>还没有分层知识库</p><button type="button" onClick={onCreate}><Plus className="h-4 w-4" />创建第一个</button></div>; }
