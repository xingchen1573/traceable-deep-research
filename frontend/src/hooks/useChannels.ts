import { MutationOptions, QueryClient, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { ChannelChatResponse, ChannelMessage, ChannelMessagesResponse, ChannelMode } from '@/types';

const CHANNELS_KEY = 'channels';

export function useChannels() {
  return useQuery({
    queryKey: [CHANNELS_KEY, 'list'],
    queryFn: () => api.channels.list(),
  });
}

export function useChannel(channelId?: string) {
  return useQuery({
    queryKey: [CHANNELS_KEY, channelId],
    queryFn: () => api.channels.get(channelId || ''),
    enabled: Boolean(channelId),
  });
}

export function useChannelScopeBinding(channelId?: string) {
  return useQuery({
    queryKey: [CHANNELS_KEY, channelId, 'scope-binding'],
    queryFn: () => api.channels.getScopeBinding(channelId || ''),
    enabled: Boolean(channelId),
  });
}

export function useBindChannelScope() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      channelId,
      scopeType,
      scopeId,
    }: {
      channelId: string;
      scopeType: 'knowledge_base' | 'theme';
      scopeId: string;
    }) => api.channels.bindScope(channelId, scopeType, scopeId),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY] });
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId] });
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId, 'scope-binding'] });
    },
  });
}

export function useUnbindChannelScope() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (channelId: string) => api.channels.unbindScope(channelId),
    onSuccess: (_data, channelId) => {
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY] });
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, channelId] });
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, channelId, 'scope-binding'] });
    },
  });
}

export function useChannelMessages(channelId?: string, mode: ChannelMode = 'main') {
  return useQuery({
    queryKey: [CHANNELS_KEY, channelId, 'messages', mode],
    queryFn: () => api.channels.messages(channelId || '', mode),
    enabled: Boolean(channelId),
  });
}

export function useChannelJobs(channelId?: string) {
  return useQuery({
    queryKey: [CHANNELS_KEY, channelId, 'jobs'],
    queryFn: () => api.channels.jobs(channelId || ''),
    enabled: Boolean(channelId),
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs || [];
      return jobs.some((job) => job.status === 'queued' || job.status === 'running') ? 1500 : false;
    },
  });
}

export function useCreateChannel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ externalId, title }: { externalId: string; title?: string }) =>
      api.channels.create(externalId, title),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY] }),
  });
}

export function useSyncFeishuChannelTitles() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.channels.syncFeishuTitles(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY] }),
  });
}

export function useUpdateChannel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ channelId, patch }: { channelId: string; patch: Parameters<typeof api.channels.update>[1] }) =>
      api.channels.update(channelId, patch),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY] });
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId] });
    },
  });
}

export function useChannelPushConfig(channelId?: string) {
  return useQuery({
    queryKey: [CHANNELS_KEY, channelId, 'push-config'],
    queryFn: () => api.channels.getPushConfig(channelId || ''),
    enabled: Boolean(channelId),
  });
}

export function useUpdateChannelPushConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      channelId,
      patch,
    }: {
      channelId: string;
      patch: Parameters<typeof api.channels.updatePushConfig>[1];
    }) => api.channels.updatePushConfig(channelId, patch),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId, 'push-config'] });
      queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId] });
    },
  });
}


export interface SendChannelMessageVariables {
  channelId: string;
  message: string;
  mode: ChannelMode;
}

interface SendChannelMessageMutationContext {
  optimisticMessageId: string;
  queryKey: readonly [typeof CHANNELS_KEY, string, 'messages', ChannelMode];
}

export function createSendChannelMessageMutationOptions(
  queryClient: QueryClient,
): MutationOptions<ChannelChatResponse, Error, SendChannelMessageVariables, SendChannelMessageMutationContext> {
  return {
    mutationFn: ({ channelId, message }: { channelId: string; message: string; mode: ChannelMode }) =>
      api.chat.sendChannel(message, channelId),
    onMutate: async (variables) => {
      const queryKey = [CHANNELS_KEY, variables.channelId, 'messages', variables.mode] as const;
      await queryClient.cancelQueries({ queryKey, exact: true });
      const previous = queryClient.getQueryData<ChannelMessagesResponse>(queryKey);
      const optimisticMessage: ChannelMessage = {
        id: `optimistic-channel-message-${Date.now()}`,
        thread_id: previous?.thread_id || `channel:${variables.channelId}:${variables.mode}`,
        role: 'user',
        content: variables.message,
        created_at: new Date().toISOString(),
      };

      queryClient.setQueryData<ChannelMessagesResponse>(queryKey, {
        channel_id: variables.channelId,
        mode: variables.mode,
        thread_id: optimisticMessage.thread_id,
        conversation: previous?.conversation ?? null,
        ...previous,
        messages: [...(previous?.messages || []), optimisticMessage],
      });
      return { optimisticMessageId: optimisticMessage.id, queryKey };
    },
    onError: (_error, _variables, context) => {
      if (!context) return;
      queryClient.setQueryData<ChannelMessagesResponse>(context.queryKey, (current) => {
        if (!current) return current;
        return {
          ...current,
          messages: current.messages.filter((message) => message.id !== context.optimisticMessageId),
        };
      });
    },
    onSuccess: async (_data, variables) => {
      const refreshes = [
        queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, 'list'], exact: true }),
        queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId], exact: true }),
        queryClient.invalidateQueries({ queryKey: [CHANNELS_KEY, variables.channelId, 'jobs'], exact: true }),
      ];
      await Promise.all(refreshes);
    },
    onSettled: async (data, _error, variables) => {
      const modes = new Set<ChannelMode>([variables.mode]);
      if (data?.mode) modes.add(data.mode);
      await Promise.all([...modes].map((mode) => queryClient.invalidateQueries({
        queryKey: [CHANNELS_KEY, variables.channelId, 'messages', mode],
        exact: true,
        refetchType: 'all',
      })));
    },
  };
}

export function useSendChannelMessage() {
  const queryClient = useQueryClient();
  return useMutation(createSendChannelMessageMutationOptions(queryClient));
}
