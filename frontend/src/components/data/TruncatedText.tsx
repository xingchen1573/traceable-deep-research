import { truncateMiddle } from '@/utils/text';
import { cn } from '@/utils/cn';

interface TruncatedTextProps {
  text: string;
  maxLength?: number;
  middle?: boolean;
  className?: string;
  title?: string;
}

export function TruncatedText({ text, maxLength = 40, middle = false, className, title }: TruncatedTextProps) {
  const display = middle ? truncateMiddle(text, maxLength) : text.length > maxLength ? text.slice(0, maxLength - 3) + '...' : text;
  return (
    <span className={cn('font-mono', className)} title={title || text}>
      {display}
    </span>
  );
}
