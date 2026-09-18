import { ApiAdapter } from '../adapters';

export interface ConversationSummary {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationMessage {
  id: string;
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  total: number;
}

export interface ConversationMessagesResponse {
  thread_id: string;
  conversation: ConversationSummary | null;
  messages: ConversationMessage[];
}

export function conversationEndpoints(adapter: ApiAdapter) {
  return {
    list: (limit = 50, offset = 0) =>
      adapter.get<ConversationListResponse>('/api/v1/conversations', { limit, offset }),

    get: (threadId: string) =>
      adapter.get<ConversationSummary>(`/api/v1/conversations/${encodeURIComponent(threadId)}`),

    getMessages: (threadId: string, limit = 200, offset = 0) =>
      adapter.get<ConversationMessagesResponse>(
        `/api/v1/conversations/${encodeURIComponent(threadId)}/messages`,
        { limit, offset }
      ),

    delete: (threadId: string) =>
      adapter.delete<{ thread_id: string; deleted: boolean }>(
        `/api/v1/conversations/${encodeURIComponent(threadId)}`
      ),
  };
}
