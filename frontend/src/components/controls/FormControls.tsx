import { cn } from '@/utils/cn';

interface SearchInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  wrapperClassName?: string;
}

export function SearchInput({ className, wrapperClassName, ...props }: SearchInputProps) {
  return (
    <div className={cn('relative', wrapperClassName)}>
      <input
        type="text"
        className={cn(
          'w-full rounded-control border border-border bg-surface px-3 py-2 pl-9 text-sm text-primary placeholder:text-secondary focus:border-accent focus:outline-none',
          className
        )}
        {...props}
      />
      <svg
        className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-secondary"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <circle cx="11" cy="11" r="8" strokeWidth="2" />
        <path d="M21 21l-4.35-4.35" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </div>
  );
}

interface FilterSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: Array<{ value: string; label: string }>;
}

export function FilterSelect({ label, options, className, ...props }: FilterSelectProps) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      {label && <span className="text-sm text-secondary">{label}</span>}
      <select
        className="rounded-control border border-border bg-surface px-3 py-2 text-sm text-primary focus:border-accent focus:outline-none"
        {...props}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmText = '确认',
  cancelText = '取消',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-sm rounded-panel bg-surface p-5 shadow-lg">
        <h3 className="text-lg font-semibold text-primary">{title}</h3>
        <p className="mt-2 text-sm text-secondary">{message}</p>
        <div className="mt-5 flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="rounded-full border border-border bg-surface px-4 py-2 text-sm text-primary hover:border-primary"
          >
            {cancelText}
          </button>
          <button
            onClick={onConfirm}
            className="rounded-full bg-danger px-4 py-2 text-sm text-white hover:opacity-90"
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

interface FormFieldProps {
  label: string;
  children: React.ReactNode;
  error?: string;
  hint?: string;
}

export function FormField({ label, children, error, hint }: FormFieldProps) {
  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-primary">{label}</label>
      {children}
      {error && <p className="text-xs text-danger">{error}</p>}
      {hint && !error && <p className="text-xs text-secondary">{hint}</p>}
    </div>
  );
}
