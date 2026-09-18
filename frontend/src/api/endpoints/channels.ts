import { ApiAdapter } from '../adapters';
import {
  Channel,
  ChannelJobsResponse,
  ChannelListResponse,
  ChannelMessagesResponse,
  ChannelMode,
  ChannelPushConfig,
  ChannelTitleSyncResponse,
} from '@/types';

export function channelEndpoints(adapter: ApiAdapter) {
  return {
    list: (limit = 100, offset = 0) =>
      adapter.get<ChannelListResponse>('/api/v1/channels', { limit, offset }),

    create: (externalId: string, title?: string, platform = 'local') =>
      adapter.post<Channel>('/api/v1/channels', {
        platform,
        external_id: externalId,
        title,
      }),

    syncFeishuTitles: () =>
      adapter.post<ChannelTitleSyncResponse>('/api/v1/channels/sync-feishu-titles', {}),

    get: (channelId: string) =>
      adapter.get<Channel>(`/api/v1/channels/${encodeURIComponent(channelId)}`),

    update: (
      channelId: string,
      patch: Partial<Pick<Channel, 'title' | 'active_kb' | 'topic_binding' | 'memory_scope'>> & { mode?: ChannelMode }
    ) => adapter.put<Channel>(`/api/v1/channels/${encodeURIComponent(channelId)}`, patch),

    getScopeBinding: (channelId: string) =>
      adapter.get<import('@/types').ChannelScopeBinding | null>(
        `/api/v1/channels/${encodeURIComponent(channelId)}/scope-binding`,
      ),

    bindScope: (channelId: string, scopeType: 'knowledge_base' | 'theme', scopeId: string) =>
      adapter.put<Channel & { scope_binding?: import('@/types').ChannelScopeBinding }>(
        `/api/v1/channels/${encodeURIComponent(channelId)}/scope-binding`,
        { scope_type: scopeType, scope_id: scopeId },
      ),

    unbindScope: (channelId: string) =>
      adapter.delete<Channel>(`/api/v1/channels/${encodeURIComponent(channelId)}/scope-binding`),

    messages: (channelId: string, mode: ChannelMode = 'main', limit = 200) =>
      adapter.get<ChannelMessagesResponse>(
        `/api/v1/channels/${encodeURIComponent(channelId)}/messages`,
        { mode, limit }
      ),

    jobs: (channelId: string, jobType?: string, status?: string, limit = 50) =>
      adapter.get<ChannelJobsResponse>(
        `/api/v1/channels/${encodeURIComponent(channelId)}/jobs`,
        { job_type: jobType, status, limit }
      ),

    getPushConfig: (channelId: string) =>
      adapter.get<ChannelPushConfig>(`/api/v1/channels/${encodeURIComponent(channelId)}/push-config`),

    updatePushConfig: (
      channelId: string,
      patch: Partial<Pick<ChannelPushConfig, 'enabled' | 'frequency' | 'content_strategy' | 'push_time' | 'timezone'>>
    ) => adapter.put<ChannelPushConfig>(`/api/v1/channels/${encodeURIComponent(channelId)}/push-config`, patch),
  };
}
