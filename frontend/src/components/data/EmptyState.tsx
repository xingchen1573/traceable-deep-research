import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  action?: React.ReactNode;
}

export function EmptyState({
  title = '暂无数据',
  description = '当前没有可展示的内容。',
  action,
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-panel border border-border bg-surface p-10 text-center">
      <Inbox className="h-10 w-10 text-secondary" />
      <h3 className="mt-4 text-base font-medium text-primary">{title}</h3>
      <p className="mt-1 text-sm text-secondary">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function LoadingState({ message = '加载中...' }: { message?: string }) {
  return (
    <div className="flex items-center justify-center rounded-panel border border-border bg-surface p-10 text-secondary">
      <span className="mr-3 inline-block h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
      {message}
    </div>
  );
}
