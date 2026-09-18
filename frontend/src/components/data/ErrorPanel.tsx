import { AlertTriangle, RefreshCw } from 'lucide-react';
import { ApiError } from '@/api/adapters';
import { isAppError, getErrorActions, getUserFriendlyMessage } from '@/utils/errors';
import { Button } from '@/components/controls/Button';
import { TruncatedText } from './TruncatedText';
import { Link } from 'react-router-dom';

interface ErrorPanelProps {
  error: unknown;
  onRetry?: () => void;
  className?: string;
}

export function ErrorPanel({ error, onRetry, className }: ErrorPanelProps) {
  let code = 'internal_error';
  let message = '请求失败';
  let requestId: string | undefined;

  if (isAppError(error)) {
    code = error.code;
    message = error.message;
    requestId = error.requestId;
  } else if (error instanceof ApiError) {
    code = error.code;
    message = error.message;
    requestId = error.requestId;
  } else if (error instanceof Error) {
    message = error.message;
  }

  const userMessage = getUserFriendlyMessage(code, message);
  const actions = getErrorActions(code);

  return (
    <div className={`rounded-panel border border-danger/30 bg-danger/10 p-4 ${className || ''}`}>
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 text-danger" />
        <div className="flex-1">
          <h4 className="text-sm font-semibold text-danger">{userMessage}</h4>
          {requestId && (
            <p className="mt-1 text-xs text-secondary">
              request_id: <TruncatedText text={requestId} maxLength={32} />
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            {onRetry && (
              <Button size="sm" variant="secondary" onClick={onRetry}>
                <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
                重试
              </Button>
            )}
            {actions.map((action) =>
              action.to ? (
                <Link key={action.label} to={action.to}>
                  <Button size="sm" variant="secondary">{action.label}</Button>
                </Link>
              ) : (
                <Button key={action.label} size="sm" variant="secondary" onClick={action.onClick}>
                  {action.label}
                </Button>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
