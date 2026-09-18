import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  MessagesSquare,
  Search,
  Library,
  MessageSquare,
  Network,
  ClipboardCheck,
  Activity,
  Cpu,
  Database,
  Briefcase,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { useUIStore } from '@/stores';
import { useJobStats, useKnowledgeTree, useModelConfig } from '@/hooks';
import { ROUTES } from '@/router/routes';

const navItems = [
  { to: ROUTES.DASHBOARD, label: '总览', icon: LayoutDashboard },
  { to: ROUTES.CHANNELS, label: '学习群', icon: MessagesSquare },
  { to: ROUTES.RESEARCH, label: '调研', icon: Search },
  { to: ROUTES.KNOWLEDGE_BASES, label: '知识库', icon: Library },
  { to: ROUTES.KNOWLEDGE_QA, label: '问答', icon: MessageSquare },
  { to: ROUTES.INDEX_GRAPH, label: '索引与图谱', icon: Network },
  { to: ROUTES.EVALUATIONS, label: '评估', icon: ClipboardCheck },
  { to: ROUTES.OBSERVABILITY, label: '观测', icon: Activity },
  { to: ROUTES.MODELS, label: '模型', icon: Cpu },
];

export function TopNav() {
  const { pathname } = useLocation();
  const { setJobDrawerOpen } = useUIStore();
  const jobStats = useJobStats(false);
  const knowledgeTree = useKnowledgeTree();
  const models = useModelConfig();

  const knowledgeBases = knowledgeTree.data?.knowledge_bases || [];
  const knowledgeLabel = knowledgeBases.length === 1
    ? knowledgeBases[0].title
    : knowledgeBases.length
      ? `${knowledgeBases.length} 个知识库`
      : '暂无知识库';
  const modelReady = !!models.data?.chat_model.base_url_configured && !!models.data?.embedding.base_url_configured;
  const taskCount = (jobStats.data?.active || 0) + (jobStats.data?.attention || 0);

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-surface/95 backdrop-blur">
      <div className="mx-auto max-w-[1240px] px-6">
        <div className="flex h-16 items-center justify-between">
          <Link to={ROUTES.DASHBOARD} className="font-display text-xl font-semibold tracking-tight text-primary">
            Deep Research
          </Link>
          <div className="flex items-center gap-2">
            <Link
              to={ROUTES.KNOWLEDGE_BASES}
              className="hidden items-center gap-1.5 rounded-control border border-border px-3 py-1.5 text-sm text-secondary hover:border-primary hover:text-primary md:flex"
            >
              <Database className="h-4 w-4" />
              <span className="max-w-48 truncate">{knowledgeLabel}</span>
            </Link>
            <Link
              to={ROUTES.MODELS}
              className="hidden items-center gap-1.5 rounded-control border border-border px-3 py-1.5 text-sm text-secondary hover:border-primary hover:text-primary lg:flex"
            >
              <Cpu className={cn('h-4 w-4', modelReady ? 'text-success' : 'text-warning')} />
              {modelReady ? '模型就绪' : '模型待检查'}
            </Link>
            <button
              onClick={() => setJobDrawerOpen(true)}
              className={cn(
                'relative flex items-center gap-1.5 rounded-control border px-3 py-1.5 text-sm',
                pathname === ROUTES.JOBS ? 'border-primary bg-primary text-page' : 'border-border text-secondary hover:border-primary hover:text-primary'
              )}
            >
              <Briefcase className="h-4 w-4" />
              <span className="hidden sm:inline">任务</span>
              {taskCount > 0 && (
                <span className="ml-1 rounded-full bg-accent px-1.5 py-0.5 text-xs text-white">
                  {taskCount > 99 ? '99+' : taskCount}
                </span>
              )}
            </button>
          </div>
        </div>

        <nav className="flex items-center gap-1 overflow-x-auto border-t border-border py-2.5">
          {navItems.map((item) => {
            const active = pathname === item.to || pathname.startsWith(item.to + '/');
            const Icon = item.icon;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  'flex flex-shrink-0 items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[13px] font-medium transition-colors',
                  active ? 'bg-primary text-page' : 'text-secondary hover:bg-page hover:text-primary'
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
