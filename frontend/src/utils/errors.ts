export interface AppError {
  code: string;
  message: string;
  retryable: boolean;
  requestId?: string;
  details?: Record<string, unknown>;
}

export function isAppError(err: unknown): err is AppError {
  return (
    typeof err === 'object' &&
    err !== null &&
    'code' in err &&
    typeof (err as AppError).code === 'string'
  );
}

export function getErrorMessage(err: unknown): string {
  if (isAppError(err)) return err.message;
  if (err instanceof Error) return err.message;
  return '发生未知错误';
}

export interface ErrorAction {
  label: string;
  to?: string;
  onClick?: () => void;
}

export function getErrorActions(code: string): ErrorAction[] {
  switch (code) {
    case 'need_topic':
    case 'topic_not_found':
      return [{ label: '选择知识库', to: '/knowledge-bases' }];
    case 'index_missing':
    case 'index_stale':
      return [
        { label: '构建索引', to: '/index-graph' },
        { label: '切换 local fallback', onClick: () => {} },
      ];
    case 'upstream_auth_error':
      return [{ label: '检查模型配置', to: '/models' }];
    case 'upstream_rate_limited':
    case 'upstream_unavailable':
      return [{ label: '稍后重试', onClick: () => window.location.reload() }];
    case 'document_not_found':
      return [{ label: '刷新文档列表', onClick: () => window.location.reload() }];
    default:
      return [];
  }
}

export function getUserFriendlyMessage(code: string, fallback?: string): string {
  const map: Record<string, string> = {
    validation_error: '请求字段不合法，请检查输入。',
    topic_not_found: '知识库 topic 不存在，请重新选择。',
    document_not_found: '文档不存在，可能已被删除。',
    job_not_found: '任务不存在或已清理。',
    pending_hitl_required: '需要人工审批后才能继续。',
    index_missing: '索引缺失，请先构建索引或切换 local backend。',
    index_stale: '索引已过期，建议重建索引。',
    upstream_auth_error: '模型或 embedding 凭据失败，请检查配置。',
    upstream_rate_limited: '上游服务限流，请稍后重试。',
    upstream_unavailable: '上游服务不可用，请检查网络或配置。',
    internal_error: '系统内部错误，请稍后重试或查看日志。',
    need_topic: '请先选择一个知识库。',
    no_match: '未在知识库中找到相关片段，可补充文档后重试。',
  };
  return map[code] ?? fallback ?? '请求失败';
}
