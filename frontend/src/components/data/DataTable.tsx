import { cn } from '@/utils/cn';

interface DataTableProps<T> {
  columns: Array<{
    key: string;
    header: React.ReactNode;
    width?: string;
    render?: (row: T) => React.ReactNode;
  }>;
  data: T[];
  keyExtractor: (row: T) => string;
  onRowClick?: (row: T) => void;
  rowClassName?: (row: T) => string | undefined;
  emptyMessage?: string;
  loading?: boolean;
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  onRowClick,
  rowClassName,
  emptyMessage = '暂无数据',
  loading,
}: DataTableProps<T>) {
  if (loading) {
    return (
      <div className="rounded-panel border border-border bg-surface p-8 text-center text-secondary">
        加载中...
      </div>
    );
  }

  if (!data.length) {
    return (
      <div className="rounded-panel border border-border bg-surface p-8 text-center text-secondary">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-panel border border-border bg-surface">
      <table className="w-full text-left text-sm">
        <thead className="bg-page font-mono text-[11px] uppercase tracking-wide text-secondary">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className="whitespace-nowrap px-4 py-3"
                style={{ width: col.width }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.map((row) => (
            <tr
              key={keyExtractor(row)}
              onClick={() => onRowClick?.(row)}
              className={cn(
                'bg-surface',
                onRowClick && 'cursor-pointer hover:bg-page',
                rowClassName?.(row)
              )}
            >
              {columns.map((col) => (
                <td key={col.key} className="whitespace-nowrap px-4 py-3 text-primary">
                  {col.render ? col.render(row) : (row as Record<string, unknown>)[col.key] as React.ReactNode}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
