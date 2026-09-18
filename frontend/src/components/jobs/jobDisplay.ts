import { JobStatus, JobSummary } from '@/types';

export const TERMINAL_JOB_STATUSES = new Set<JobStatus>(['succeeded', 'failed', 'cancelled', 'interrupted']);

const JOB_TYPE_LABELS: Record<string, string> = {
  research: '调研',
  knowledge_index: '知识库索引',
  knowledge_qa_evaluation: '问答评估',
  knowledge_qa_case_generation: '问答用例生成',
  knowledge_scope_evaluation: '知识范围评估',
  research_probe_evaluation: '调研评估',
  research_index: '调研索引',
  single_url_brief_evaluation: '单链接评估',
  test_run: '测试',
};

export function jobTypeLabel(jobType: string): string {
  return JOB_TYPE_LABELS[jobType] || jobType.replace(/_/g, ' ');
}

export function jobTitle(job: JobSummary): string {
  const input = job.input || {};
  const result = (job.result || {}) as Record<string, unknown>;
  const candidates = [
    input.user_request,
    input.title,
    input.topic,
    input.url,
    input.query,
    result.suggested_topic,
    result.source_title,
  ];
  const title = candidates.find((value) => typeof value === 'string' && value.trim());
  return typeof title === 'string' ? title.trim() : jobTypeLabel(job.job_type);
}

export function jobTarget(job: JobSummary): string {
  if (job.job_type === 'research') return `/research?job_id=${encodeURIComponent(job.job_id)}`;
  return `/observability?job_id=${encodeURIComponent(job.job_id)}`;
}

export function isTerminalJob(job: JobSummary): boolean {
  return TERMINAL_JOB_STATUSES.has(job.status);
}

export function jobProgressPercent(job: JobSummary): number | null {
  if (!job.progress?.total) return null;
  return Math.min(100, Math.max(0, Math.round((job.progress.current / job.progress.total) * 100)));
}

export function jobProgressLabel(job: JobSummary): string {
  const labels: Record<string, string> = {
    queued: '等待执行',
    running: '正在执行',
    waiting_approval: '等待审批',
    succeeded: '已完成',
    failed: '执行失败',
    cancelled: '已取消',
    interrupted: '已中断',
  };
  const message = job.progress?.message || '';
  return labels[message] || labels[job.status] || message || '处理中';
}
