import { MutationObserver, QueryClient } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '@/api/client';
import { ChannelMessagesResponse } from '@/types';
import { createSendChannelMessageMutationOptions } from './useChannels';

describe('send channel message mutation', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('removes a failed optimistic message and refetches the authoritative server history', async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        mutations: { retry: false },
        queries: { retry: false },
      },
    });
    const channelId = 'local:test-channel';
    const queryKey = ['channels', channelId, 'messages', 'main'] as const;
    const authoritative: ChannelMessagesResponse = {
      channel_id: channelId,
      mode: 'main',
      thread_id: `channel:${channelId}:main`,
      conversation: null,
      messages: [{
        id: 'server-message-1',
        thread_id: `channel:${channelId}:main`,
        role: 'assistant',
        content: '已有服务端消息',
        created_at: '2026-09-17T09:00:00.000Z',
      }],
    };
    const fetchMessages = vi.fn().mockResolvedValue(authoritative);
    await queryClient.fetchQuery({ queryKey, queryFn: fetchMessages });

    vi.spyOn(api.chat, 'sendChannel').mockImplementation(async () => {
      const optimistic = queryClient.getQueryData<ChannelMessagesResponse>(queryKey);
      expect(optimistic?.messages.at(-1)).toMatchObject({ role: 'user', content: '这条会失败' });
      throw new Error('network unavailable');
    });

    const observer = new MutationObserver(
      queryClient,
      createSendChannelMessageMutationOptions(queryClient),
    );
    await expect(observer.mutate({
      channelId,
      message: '这条会失败',
      mode: 'main',
    })).rejects.toThrow('network unavailable');

    expect(fetchMessages).toHaveBeenCalledTimes(2);
    expect(queryClient.getQueryData(queryKey)).toEqual(authoritative);
    queryClient.clear();
  });
});
