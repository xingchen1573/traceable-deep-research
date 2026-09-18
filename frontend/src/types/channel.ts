import { JobSummary } from './api';

export type ChannelMode = 'main' | 'kb_qa';

export interface Channel {
  channel_id: string;
  platform: string;
  external_id: string;
  title: string;
  mode: ChannelMode;
  active_kb?: string | null;
  topic_binding?: string | null;
  scope_type?: 'knowledge_base' | 'theme' | null;
  scope_id?: string | null;
  memory_scope: 'channel';
  created_at: string;
  updated_at: string;
  threads?: {
    main: string;
    kb_qa: string;
  };
}

export interface ChannelScopeBinding {
  scope_type: 'knowledge_base' | 'theme' | 'legacy_topic';
  scope_id: string;
  legacy: boolean;
}

export interface ChannelListResponse {
  channels: Channel[];
  total: number;
}

export interface ChannelTitleSyncResponse {
  attempted: number;
  resolved: number;
  updated: number;
  failed: number;
}

export type ChannelPushFrequency = 'daily' | 'weekly_3x' | 'weekly_1x';
export type ChannelPushContentStrategy = 'rotation' | 'new_only' | 'random';

export interface ChannelPushConfig {
  enabled: boolean;
  frequency: ChannelPushFrequency;
  content_strategy: ChannelPushContentStrategy;
  push_time: string;
  timezone: string;
  next_push_at: string | null;
  last_pushed_at: string | null;
  last_pushed_doc: string | null;
  scheduler_enabled: boolean;
  scheduler_running: boolean;
  scheduler_tick_interval_seconds: number;
}

export interface ChannelMessage {
  id: string;
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

export interface ChannelMessagesResponse {
  channel_id: string;
  mode: ChannelMode;
  thread_id: string;
  conversation: unknown;
  messages: ChannelMessage[];
}

export interface ChannelJobsResponse {
  channel_id: string;
  jobs: JobSummary[];
}

export interface ChannelChatResponse {
  status: string;
  state?: string;
  channel_id: string;
  mode: ChannelMode;
  active_kb?: string | null;
  thread_id: string;
  run_id?: string | null;
  assistant_text: string;
  channel?: Channel;
  qa?: {
    channel_id: string;
    mode: 'kb_qa';
    active_kb?: string | null;
    answer: string;
    citations: Array<Record<string, unknown>>;
    gaps: string[];
    raw?: Record<string, unknown>;
  };
}
