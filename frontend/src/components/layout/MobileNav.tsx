import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Library,
  MessageSquare,
  Briefcase,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ROUTES } from '@/router/routes';

const mobileItems = [
  { to: ROUTES.DASHBOARD, label: '总览', icon: LayoutDashboard },
  { to: ROUTES.KNOWLEDGE_BASES, label: '知识库', icon: Library },
  { to: ROUTES.KNOWLEDGE_QA, label: '问答', icon: MessageSquare },
  { to: ROUTES.INDEX_GRAPH, label: '任务', icon: Briefcase },
];

export function MobileNav() {
  const { pathname } = useLocation();

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-40 border-t border-border bg-surface md:hidden">
      <div className="flex items-center justify-around">
        {mobileItems.map((item) => {
          const active = pathname === item.to || pathname.startsWith(item.to + '/');
          const Icon = item.icon;
          return (
            <Link
              key={item.to}
              to={item.to}
              className={cn(
                'flex flex-1 flex-col items-center gap-0.5 py-2 text-xs',
                active ? 'text-accent' : 'text-secondary'
              )}
            >
              <Icon className="h-5 w-5" />
              {item.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
