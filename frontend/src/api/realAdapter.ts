import { ApiAdapter, parseEnvelope } from './adapters';
import { getBaseUrl, shouldUseMock } from './toggle';
import { createMockAdapter } from './mockAdapter';

export function createRealAdapter(): ApiAdapter {
  const baseUrl = getBaseUrl();
  const mock = createMockAdapter();

  async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
    if (shouldUseMock()) {
      if (method === 'POST') return mock.post<T>(path, body);
      if (method === 'PUT') return mock.put<T>(path, body);
      if (method === 'DELETE') return mock.delete<T>(path);
      return mock.get<T>(path, body as Record<string, unknown> | undefined);
    }
    const url = path.startsWith('http') ? path : baseUrl ? `${baseUrl}${path}` : path;
    const init: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    };
    if (body !== undefined) {
      init.body = JSON.stringify(body);
    }
    const res = await fetch(url, init);
    return parseEnvelope<T>(res);
  }

  return {
    get: async <T>(path: string, params?: Record<string, unknown>) => {
      if (shouldUseMock()) {
        return mock.get<T>(path, params);
      }
      const url = new URL(
        path,
        baseUrl || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost')
      );
      if (params) {
        Object.entries(params).forEach(([k, v]) => {
          if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
        });
      }
      const res = await fetch(url.toString(), {
        headers: { Accept: 'application/json' },
      });
      return parseEnvelope<T>(res);
    },
    post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
    put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
    delete: <T>(path: string) => request<T>('DELETE', path),
    stream: async <T>(path: string, body: unknown, onDelta: (text: string) => void) => {
      if (shouldUseMock()) return mock.stream<T>(path, body, onDelta);
      const url = path.startsWith('http') ? path : baseUrl ? `${baseUrl}${path}` : path;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(body),
      });
      if (!res.ok || !res.body) throw new Error(`Stream request failed (${res.status})`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalPayload: T | undefined;
      const consume = (block: string) => {
        let event = 'message';
        let data = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          if (line.startsWith('data:')) data += line.slice(5).trim();
        }
        if (!data) return;
        const payload = JSON.parse(data);
        if (event === 'delta') onDelta(String(payload.text || ''));
        if (event === 'done') finalPayload = payload as T;
        if (event === 'error') throw new Error(String(payload.message || 'Stream request failed'));
      };
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';
        blocks.forEach(consume);
        if (done) break;
      }
      if (buffer.trim()) consume(buffer);
      if (!finalPayload) throw new Error('Stream ended without a final response');
      return finalPayload;
    },
  };
}
