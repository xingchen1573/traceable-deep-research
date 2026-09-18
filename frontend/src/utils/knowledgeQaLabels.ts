import { QueryBackend, QueryMode } from '@/types';

export const QUERY_BACKEND_OPTIONS: Array<{ value: QueryBackend; label: string }> = [
  { value: 'auto', label: '自动选择（推荐）' },
  { value: 'lightrag', label: '知识图谱检索（LightRAG）' },
  { value: 'local', label: 'Markdown 基础检索' },
];

export const QUERY_MODE_OPTIONS: Array<{ value: QueryMode; label: string }> = [
  { value: 'mix', label: '综合检索（Mix）' },
  { value: 'local', label: '局部事实（Local）' },
  { value: 'global', label: '全局主题（Global）' },
  { value: 'hybrid', label: '局部与全局（Hybrid）' },
  { value: 'naive', label: '纯向量（Naive）' },
];

const BACKEND_LABELS: Record<string, string> = {
  auto: '自动选择',
  lightrag: '知识图谱检索（LightRAG）',
  local: 'Markdown 基础检索',
  mixed: '混合后端',
  knowledge_scope: '分层知识范围检索',
};

const MODE_LABELS: Record<string, string> = {
  mix: '综合检索（Mix）',
  local: '局部事实（Local）',
  global: '全局主题（Global）',
  hybrid: '局部与全局（Hybrid）',
  naive: '纯向量（Naive）',
  markdown: 'Markdown 关键词检索',
};

const CONFIDENCE_LABELS: Record<string, string> = {
  high: '高',
  medium: '中',
  low: '低',
};

export const queryBackendLabel = (value?: string | null) => BACKEND_LABELS[value || ''] || value || '未知';
export const queryModeLabel = (value?: string | null) => MODE_LABELS[value || ''] || value || '未知';
export const confidenceLabel = (value?: string | null) => CONFIDENCE_LABELS[value || ''] || value || '未知';
