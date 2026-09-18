import { useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods, NodeObject } from 'react-force-graph-2d';
import { ArrowLeft, Info, X } from 'lucide-react';
import { SearchInput, FilterSelect } from '@/components/controls/FormControls';
import { EmptyState, LoadingState } from './EmptyState';
import { useKnowledgeScopeGraphNode } from '@/hooks';
import { GraphStatsResponse, GraphNodeLite } from '@/types';
import { cn } from '@/utils/cn';

interface KnowledgeGraphViewProps {
  scopeType: 'knowledge_base' | 'theme' | 'research';
  scopeId: string;
  graph: GraphStatsResponse;
}

type GraphNode = NodeObject<GraphNodeLite>;

// Read a CSS custom property from :root (tokens.css) so we never hardcode the
// palette or fall back to the graph library's default theme colors.
function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function formatGraphDescription(description: string): string {
  return description.split('<SEP>').join('\n\n').trim();
}

// Deterministic type -> token color mapping so filtering/legend stay consistent.
function useTokenPalette() {
  return useMemo(
    () => ({
      accent: cssVar('--color-accent', '#C4531E'),
      link: cssVar('--color-link', '#C4531E'),
      success: cssVar('--color-success', '#4E7A4A'),
      warning: cssVar('--color-warning', '#B8801F'),
      danger: cssVar('--color-danger', '#B23A2C'),
      secondary: cssVar('--color-secondary', '#8A7D6C'),
      border: cssVar('--color-border', '#E4D9C6'),
      surface: cssVar('--color-surface', '#FFFDF8'),
      primary: cssVar('--color-primary', '#241C15'),
    }),
    []
  );
}

export function KnowledgeGraphView({ scopeType, scopeId, graph }: KnowledgeGraphViewProps) {
  const palette = useTokenPalette();
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<GraphNode> | undefined>(undefined);
  const [size, setSize] = useState({ width: 0, height: 480 });
  const [typeFilter, setTypeFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectionHistory, setSelectionHistory] = useState<string[]>([]);
  const [isCoarsePointer, setIsCoarsePointer] = useState(false);

  const detail = useKnowledgeScopeGraphNode(scopeType, scopeId, selectedId);

  // Distinct entity types, in stable order, for the filter dropdown + legend.
  const entityTypes = useMemo(() => {
    const set = new Set<string>();
    graph.graph_nodes.forEach((n) => set.add(n.type || 'unknown'));
    return Array.from(set).sort();
  }, [graph.graph_nodes]);

  // Map each entity type to a token color (cycled through the accent palette).
  const typeColors = useMemo(() => {
    const cycle = [palette.accent, palette.link, palette.success, palette.warning, palette.danger, palette.secondary];
    const map: Record<string, string> = {};
    entityTypes.forEach((t, i) => {
      map[t] = cycle[i % cycle.length];
    });
    return map;
  }, [entityTypes, palette]);

  const searchLower = search.trim().toLowerCase();

  const filteredNodeIds = useMemo(() => {
    const ids = new Set<string>();
    graph.graph_nodes.forEach((n) => {
      const typeOk = typeFilter === 'all' || (n.type || 'unknown') === typeFilter;
      const searchOk = !searchLower || n.id.toLowerCase().includes(searchLower);
      if (typeOk && searchOk) ids.add(n.id);
    });
    return ids;
  }, [graph.graph_nodes, typeFilter, searchLower]);

  const graphData = useMemo(
    () => ({
      nodes: graph.graph_nodes.map((n) => ({ ...n })),
      links: graph.graph_edges
        .filter((e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target))
        .map((e) => ({ ...e })),
    }),
    [graph.graph_nodes, graph.graph_edges, filteredNodeIds]
  );

  // Detect touch / small-screen environments to offer the list fallback.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(pointer: coarse)');
    const update = () => setIsCoarsePointer(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);

  // Track container width so the canvas stays responsive.
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width;
        setSize({ width, height: width < 640 ? 360 : 480 });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setSelectedId(null);
    setSelectionHistory([]);
  }, [scopeType, scopeId]);

  const openEntity = (entityId: string) => {
    setSelectionHistory([]);
    setSelectedId(entityId);
  };

  const openRelatedEntity = (entityId: string) => {
    if (!selectedId || entityId === selectedId) return;
    setSelectionHistory((history) => [...history, selectedId]);
    setSelectedId(entityId);
  };

  const returnToPreviousEntity = () => {
    const previousId = selectionHistory[selectionHistory.length - 1];
    if (!previousId) return;
    setSelectionHistory((history) => history.slice(0, -1));
    setSelectedId(previousId);
  };

  const closeDetail = () => {
    setSelectedId(null);
    setSelectionHistory([]);
  };

  const nodeCount = graph.graph_nodes.length;

  if (!nodeCount) {
    return (
      <EmptyState
        title="暂无图谱数据"
        description="该知识库尚未生成实体关系图谱，请先构建 LightRAG 索引。"
      />
    );
  }

  const listView = (
    <NodeListView
      nodes={graph.graph_nodes}
      typeColors={typeColors}
      filteredIds={filteredNodeIds}
      selectedId={selectedId}
      onSelect={openEntity}
    />
  );

  return (
    <div className="space-y-3">
      {graph.truncated && (
        <div className="flex items-start gap-2 rounded-panel border border-warning/40 bg-warning/5 p-3 text-sm text-primary">
          <Info className="mt-0.5 h-4 w-4 flex-shrink-0 text-warning" />
          <span>
            图谱较大，已显示连接度最高的核心部分（{nodeCount} / {graph.total_nodes} 个节点）。
          </span>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <SearchInput
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索实体名称…"
          wrapperClassName="min-w-[200px] flex-1"
        />
        <FilterSelect
          label="类型"
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          options={[
            { value: 'all', label: `全部 (${nodeCount})` },
            ...entityTypes.map((t) => ({ value: t, label: t })),
          ]}
        />
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-secondary">
        {entityTypes.map((t) => (
          <span key={t} className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: typeColors[t] }} />
            {t}
          </span>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div
          ref={containerRef}
          className="relative overflow-hidden rounded-panel border border-border bg-white"
        >
          {isCoarsePointer ? (
            <div className="p-3">
              <p className="mb-2 flex items-center gap-1.5 text-xs text-secondary">
                <Info className="h-3.5 w-3.5" />
                移动端已切换为列表视图，建议在桌面端查看完整力导向图谱。
              </p>
              {listView}
            </div>
          ) : (
            <div style={{ height: size.height }}>
              {size.width > 0 && (
                <ForceGraph2D
                  ref={fgRef}
                  graphData={graphData}
                  width={size.width}
                  height={size.height}
                  backgroundColor={palette.surface}
                  nodeRelSize={4}
                  nodeVal={(n) => 1 + Math.sqrt((n as GraphNode).degree || 1)}
                  linkColor={() => palette.border}
                  linkWidth={(l) => 0.5 + Math.min(3, (l as { weight?: number }).weight || 1) * 0.4}
                  cooldownTicks={100}
                  onNodeClick={(n) => openEntity(String((n as GraphNode).id))}
                  nodeCanvasObject={(node, ctx, globalScale) => {
                    const n = node as GraphNode;
                    const isSelected = String(n.id) === selectedId;
                    const radius = (1 + Math.sqrt(n.degree || 1)) * 1.4;
                    ctx.beginPath();
                    ctx.arc(n.x || 0, n.y || 0, radius, 0, 2 * Math.PI, false);
                    ctx.fillStyle = typeColors[n.type || 'unknown'] || palette.accent;
                    ctx.fill();
                    if (isSelected) {
                      ctx.lineWidth = 2 / globalScale;
                      ctx.strokeStyle = palette.primary;
                      ctx.stroke();
                    }
                    // Only draw labels when zoomed in enough or for hub nodes.
                    if (globalScale > 1.4 || (n.degree || 0) >= 8) {
                      const label = String(n.id);
                      const fontSize = Math.max(3, 12 / globalScale);
                      ctx.font = `${fontSize}px sans-serif`;
                      ctx.fillStyle = palette.primary;
                      ctx.textAlign = 'center';
                      ctx.textBaseline = 'top';
                      ctx.fillText(label, n.x || 0, (n.y || 0) + radius + 1);
                    }
                  }}
                  nodePointerAreaPaint={(node, color, ctx) => {
                    const n = node as GraphNode;
                    const radius = (1 + Math.sqrt(n.degree || 1)) * 1.4 + 2;
                    ctx.beginPath();
                    ctx.arc(n.x || 0, n.y || 0, radius, 0, 2 * Math.PI, false);
                    ctx.fillStyle = color;
                    ctx.fill();
                  }}
                />
              )}
            </div>
          )}
        </div>

        <NodeDetailPanel
          open={!!selectedId}
          isLoading={detail.isLoading}
          error={detail.error}
          detail={detail.data}
          typeColors={typeColors}
          canGoBack={selectionHistory.length > 0}
          onBack={returnToPreviousEntity}
          onSelectEntity={openRelatedEntity}
          onClose={closeDetail}
        />
      </div>
    </div>
  );
}

function NodeListView({
  nodes,
  typeColors,
  filteredIds,
  selectedId,
  onSelect,
}: {
  nodes: GraphNodeLite[];
  typeColors: Record<string, string>;
  filteredIds: Set<string>;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const visible = nodes
    .filter((n) => filteredIds.has(n.id))
    .sort((a, b) => b.degree - a.degree);
  if (!visible.length) {
    return <p className="py-6 text-center text-sm text-secondary">没有匹配的实体。</p>;
  }
  return (
    <ul className="max-h-[420px] space-y-1 overflow-y-auto">
      {visible.map((n) => (
        <li key={n.id}>
          <button
            type="button"
            onClick={() => onSelect(n.id)}
            className={cn(
              'flex w-full items-center justify-between gap-2 rounded-control border px-3 py-2 text-left text-sm transition-colors hover:border-accent',
              n.id === selectedId ? 'border-accent bg-accent/5' : 'border-border bg-white'
            )}
          >
            <span className="flex items-center gap-2 truncate">
              <span
                className="h-2.5 w-2.5 flex-shrink-0 rounded-full"
                style={{ backgroundColor: typeColors[n.type || 'unknown'] }}
              />
              <span className="truncate text-primary">{n.id}</span>
            </span>
            <span className="flex-shrink-0 text-xs text-secondary">度 {n.degree}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function NodeDetailPanel({
  open,
  isLoading,
  error,
  detail,
  typeColors,
  canGoBack,
  onBack,
  onSelectEntity,
  onClose,
}: {
  open: boolean;
  isLoading: boolean;
  error: unknown;
  detail?: import('@/types').GraphNodeDetailResponse;
  typeColors: Record<string, string>;
  canGoBack: boolean;
  onBack: () => void;
  onSelectEntity: (entityId: string) => void;
  onClose: () => void;
}) {
  if (!open) {
    return (
      <div className="hidden rounded-panel border border-dashed border-border bg-white p-4 text-sm text-secondary lg:flex lg:items-center lg:justify-center">
        点击图谱中的节点查看详情
      </div>
    );
  }
  return (
    <div className="self-start rounded-panel border border-border bg-white p-4">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {canGoBack && (
            <button
              type="button"
              onClick={onBack}
              className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-control text-secondary transition-colors hover:bg-page hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
              aria-label="返回上一实体"
              title="返回上一实体"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
          )}
          {detail && (
            <span
              className="h-3 w-3 flex-shrink-0 rounded-full"
              style={{ backgroundColor: typeColors[detail.entity_type || 'unknown'] }}
            />
          )}
          <h4 className="truncate text-sm font-semibold text-primary">{detail?.entity_id || '加载中'}</h4>
        </div>
        <button type="button" onClick={onClose} className="text-secondary hover:text-primary" aria-label="关闭">
          <X className="h-4 w-4" />
        </button>
      </div>

      {isLoading && <LoadingState message="加载节点详情…" />}
      {!isLoading && error != null && (
        <p className="text-sm text-danger">加载节点详情失败。</p>
      )}
      {!isLoading && detail && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2 text-xs text-secondary">
            <span className="rounded-control bg-page px-2 py-0.5">类型：{detail.entity_type || '-'}</span>
            <span className="rounded-control bg-page px-2 py-0.5">连接度：{detail.degree}</span>
          </div>
          <p className="max-h-48 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-primary">
            {detail.description ? formatGraphDescription(detail.description) : '（无描述）'}
          </p>
          {detail.edges.length > 0 && (
            <div>
              <div className="mb-1.5 flex items-baseline justify-between gap-2">
                <h5 className="text-xs font-semibold text-primary">
                  直接关联实体（{detail.edges.length}）
                </h5>
                <span className="flex-shrink-0 text-[11px] text-secondary">当前实体的一跳关系</span>
              </div>
              <ul className="flex max-h-48 flex-col gap-1.5 overflow-y-auto pr-1">
                {detail.edges.map((e, idx) => (
                  <li key={`${e.entity_id}-${idx}`} className="flex-none">
                    <button
                      type="button"
                      onClick={() => onSelectEntity(e.entity_id)}
                      className="m-0 block h-auto min-h-0 w-full appearance-none rounded-control border border-border bg-white p-2 text-left text-xs leading-normal transition-colors hover:border-accent hover:bg-accent/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
                      aria-label={`查看 ${e.entity_id} 完整详情`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="flex min-w-0 items-center gap-1.5">
                          <span
                            className="h-2 w-2 flex-shrink-0 rounded-full"
                            style={{ backgroundColor: typeColors[e.entity_type || 'unknown'] }}
                          />
                          <span className="truncate text-primary">{e.entity_id}</span>
                        </span>
                        <span className="flex-shrink-0 text-secondary">权重 {e.weight}</span>
                      </div>
                      {e.neighbor_preview && (
                        <p className="mt-1 line-clamp-2 text-secondary">{e.neighbor_preview}</p>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
