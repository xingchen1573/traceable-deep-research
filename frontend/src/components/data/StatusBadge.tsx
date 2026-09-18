import { cn } from '@/utils/cn';

const STATUS_COLORS: Record<
  string,
  { bg: string; text: string; dot: string }
> = {
  ready: { bg: 'bg-success/10', text: 'text-success', dot: 'bg-success' },
  indexed: { bg: 'bg-success/10', text: 'text-success', dot: 'bg-success' },
  active: { bg: 'bg-success/10', text: 'text-success', dot: 'bg-success' },
  stale: { bg: 'bg-warning/10', text: 'text-warning', dot: 'bg-warning' },
  missing: { bg: 'bg-secondary/10', text: 'text-secondary', dot: 'bg-faint' },
  invalid: { bg: 'bg-danger/10', text: 'text-danger', dot: 'bg-danger' },
  failed: { bg: 'bg-danger/10', text: 'text-danger', dot: 'bg-danger' },
  error: { bg: 'bg-danger/10', text: 'text-danger', dot: 'bg-danger' },
  running: { bg: 'bg-link/10', text: 'text-link', dot: 'bg-link' },
  stalled: { bg: 'bg-warning/10', text: 'text-warning', dot: 'bg-warning' },
  orphaned: { bg: 'bg-danger/10', text: 'text-danger', dot: 'bg-danger' },
  queued: { bg: 'bg-secondary/10', text: 'text-secondary', dot: 'bg-faint' },
  waiting_approval: { bg: 'bg-accent/10', text: 'text-accent', dot: 'bg-accent' },
  succeeded: { bg: 'bg-success/10', text: 'text-success', dot: 'bg-success' },
  completed: { bg: 'bg-success/10', text: 'text-success', dot: 'bg-success' },
  cancelled: { bg: 'bg-secondary/10', text: 'text-secondary', dot: 'bg-faint' },
  interrupted: { bg: 'bg-warning/10', text: 'text-warning', dot: 'bg-warning' },
  waiting_user: { bg: 'bg-accent/10', text: 'text-accent', dot: 'bg-accent' },
  answered: { bg: 'bg-success/10', text: 'text-success', dot: 'bg-success' },
  unknown: { bg: 'bg-secondary/10', text: 'text-secondary', dot: 'bg-faint' },
};

interface StatusBadgeProps {
  status: string;
  children?: React.ReactNode;
}

const STATUS_LABELS: Record<string, string> = {
  ready: '就绪',
  indexed: '已索引',
  active: '活跃',
  stale: '过期',
  missing: '缺失',
  invalid: '无效',
  failed: '失败',
  error: '异常',
  running: '运行中',
  stalled: '进度停滞',
  orphaned: '执行器失联',
  queued: '排队中',
  waiting_approval: '待审批',
  succeeded: '成功',
  completed: '完成',
  cancelled: '已取消',
  interrupted: '已中断',
  waiting_user: '等你处理',
  answered: '已回答',
  unknown: '未知',
};

export function StatusBadge({ status, children }: StatusBadgeProps) {
  const colors = STATUS_COLORS[status] || STATUS_COLORS.missing;
  const label = children || STATUS_LABELS[status] || status;
  return (
    <span
      className={cn(
        'inline-flex flex-none items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium',
        colors.bg,
        colors.text
      )}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', colors.dot)} />
      {label}
    </span>
  );
}
