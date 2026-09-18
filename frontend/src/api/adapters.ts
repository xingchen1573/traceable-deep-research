import { createRealAdapter } from './realAdapter';
import { createMockAdapter } from './mockAdapter';
import { shouldUseMock } from './toggle';
import { ApiEnvelope } from '@/types';

export interface ApiAdapter {
  get: <T>(path: string, params?: Record<string, unknown>) => Promise<T>;
  post: <T>(path: string, body?: unknown) => Promise<T>;
  put: <T>(path: string, body?: unknown) => Promise<T>;
  delete: <T>(path: string) => Promise<T>;
  stream: <T>(path: string, body: unknown, onDelta: (text: string) => void) => Promise<T>;
}

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public retryable: boolean,
    public requestId?: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function parseEnvelope<T>(res: Response): Promise<T> {
  let envelope: ApiEnvelope<T>;
  try {
    envelope = (await res.json()) as ApiEnvelope<T>;
  } catch {
    throw new ApiError('internal_error', 'Invalid JSON response', false);
  }

  if (envelope.status === 'error' && envelope.error) {
    throw new ApiError(
      envelope.error.code,
      envelope.error.message,
      envelope.error.retryable,
      envelope.request_id,
      envelope.error.details
    );
  }

  if (envelope.status === 'ok' || envelope.status === 'accepted') {
    return envelope.data as T;
  }

  throw new ApiError('internal_error', 'Unexpected response status', false, envelope.request_id);
}

export function createAdapter(): ApiAdapter {
  return shouldUseMock() ? createMockAdapter() : createRealAdapter();
}

export const adapter = createAdapter();
