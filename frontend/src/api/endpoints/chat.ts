import { ApiAdapter } from '../adapters';
import { ChannelChatResponse } from '@/types';

export function chatEndpoints(adapter: ApiAdapter) {
  return {
    send: (message: string, threadId?: string) => adapter.post<{
      status: string;
      thread_id: string;
      run_id: string;
      assistant_text: string;
    }>('/api/v1/chat', { message, thread_id: threadId }),

    sendChannel: (message: string, channelId: string, userId?: string) =>
      adapter.post<ChannelChatResponse>('/api/v1/chat', {
        message,
        channel_id: channelId,
        user_id: userId,
      }),

    getPending: (threadId: string) => adapter.get<Record<string, unknown>>(`/api/v1/chat/pending/${encodeURIComponent(threadId)}`),

    resume: (threadId: string, decision: string, message?: string) => adapter.post<{
      status: string;
      thread_id: string;
      run_id: string;
      assistant_text: string;
    }>('/api/v1/chat/resume', { thread_id: threadId, decision, message }),
  };
}
