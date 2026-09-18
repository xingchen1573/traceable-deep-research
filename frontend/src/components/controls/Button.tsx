import { cn } from '@/utils/cn';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  icon?: React.ReactNode;
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  isLoading,
  disabled,
  className,
  icon,
  ...props
}: ButtonProps) {
  const variants = {
    primary: 'bg-accent text-white hover:opacity-90',
    secondary: 'bg-surface text-primary border border-border hover:border-primary',
    ghost: 'bg-transparent text-secondary hover:bg-page',
    danger: 'bg-danger text-white hover:opacity-90',
  };
  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base',
  };
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center rounded-full font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading && (
        <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
      )}
      {!isLoading && icon}
      {children}
    </button>
  );
}

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon: React.ReactNode;
  label: string;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
}

export function IconButton({ icon, label, variant = 'ghost', className, ...props }: IconButtonProps) {
  const variants = {
    primary: 'bg-accent text-white hover:opacity-90',
    secondary: 'bg-surface text-primary border border-border hover:border-primary',
    ghost: 'bg-transparent text-secondary hover:bg-page hover:text-primary',
    danger: 'bg-danger text-white hover:opacity-90',
  };
  return (
    <button
      type="button"
      aria-label={label}
      className={cn(
        'inline-flex h-9 w-9 items-center justify-center rounded-full transition-colors disabled:opacity-50',
        variants[variant],
        className
      )}
      {...props}
    >
      {icon}
    </button>
  );
}
