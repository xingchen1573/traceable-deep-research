import { cn } from '@/utils/cn';

interface PageHeaderProps {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({ title, description, actions, className }: PageHeaderProps) {
  return (
    <div
      className={cn(
        'mb-6 flex flex-col gap-3 border-b-2 border-primary pb-4 sm:flex-row sm:items-end sm:justify-between',
        className
      )}
    >
      <div>
        <h1 className="font-display text-2xl font-semibold tracking-tight text-primary">{title}</h1>
        {description && <p className="mt-1 text-sm text-secondary">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
