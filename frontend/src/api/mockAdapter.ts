import { ApiAdapter, ApiError } from './adapters';
import {
  ActiveKnowledgeBaseResponse,
  CapabilitiesResponse,
  DeleteIndexResponse,
  DocumentListResponse,
  DocumentResponse,
  EvaluationCase,
  EvaluationJob,
  EvaluationJobRequest,
  EvaluationReportListResponse,
  EvaluationReportResponse,
  GraphStatsResponse,
  GraphNodeDetailResponse,
  GraphNodeLite,
  GraphEdgeLite,
  HealthResponse,
  IndexStatusResponse,
  JobCreatedResponse,
  JobSummary,
  KnowledgeBase,
  KnowledgeBaseDetail,
  KnowledgeBaseListResponse,
  KnowledgeBaseSummaryResponse,
  ModelCheckResponse,
  ModelCheckKeyResult,
  ModelConfigResponse,
  ObservabilityEvent,
  ProvidersResponse,
  ResearchRole,
  ResearchRoleKey,
  ResearchRoleKeyMutationResponse,
  ResearchRoleModelConfig,
  QueryResponse,
  Channel,
  ChannelChatResponse,
  ChannelJobsResponse,
  ChannelListResponse,
  ChannelMessagesResponse,
  ChannelPushConfig,
  ConfirmedResearchPlan,
  ResearchApproveRequest,
  ResearchApproveResponse,
  ResearchArtifactsResponse,
  ResearchDocument,
  ResearchFollowupRequest,
  ResearchFollowupResponse,
  ResearchJobResult,
  ResearchJob,
  ResearchJobCreateRequest,
  ResearchPlanSection,
  ResearchPlanningClarifyRequest,
  ResearchPlanningClarifyResponse,
  ResearchPlanningDraftRequest,
  ResearchPlanningDraftResponse,
  ResearchSectionRevisionRequest,
  ResearchSectionRevisionResponse,
  RuntimeConfigResponse,
  VectorStatsResponse,
} from '@/types';

const MOCK_TOPIC = 'harness-agent-v03-e2e-1781143504';
const MOCK_TOPIC_2 = 'langgraph-state-management';

function nowText(): string {
  const d = new Date();
  return d.toISOString().replace('T', ' ').slice(0, 19);
}

function mockNextPushAt(pushTime: string, timezone: string): string {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(Date.now() + 24 * 60 * 60 * 1000));
  const part = (type: Intl.DateTimeFormatPartTypes) => parts.find((item) => item.type === type)?.value || '';
  return `${part('year')}-${part('month')}-${part('day')} ${pushTime}`;
}

function newRunId(): string {
  return `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

function newJobId(type: string): string {
  return `job_${Date.now().toString(36)}_${type}_${Math.random().toString(36).slice(2, 6)}`;
}

function okEnvelope<T>(data: T) {
  return {
    status: 'ok' as const,
    request_id: newRunId(),
    data,
  };
}

function acceptedEnvelope<T>(data: T) {
  return {
    status: 'accepted' as const,
    request_id: newRunId(),
    data,
  };
}

function sleep(ms = 300): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const knowledgeBases: KnowledgeBase[] = [
  {
    topic: MOCK_TOPIC,
    title: 'Harness Engineering 调研知识库',
    document_count: 3,
    entry_document: '00_总览.md',
    updated_at: '2026-06-12 09:00:00',
    index_status: 'ready',
    metadata_status: 'explicit',
    source_mode: 'web',
    version: 'v0.1',
    graph_stats: { nodes: 111, edges: 159 },
    vector_stats: { chunks: 13, entities: 111, relationships: 159 },
  },
  {
    topic: MOCK_TOPIC_2,
    title: 'LangGraph 状态管理',
    document_count: 3,
    entry_document: '00_总览.md',
    updated_at: '2026-06-17 10:30:00',
    index_status: 'missing',
    metadata_status: 'inferred',
    source_mode: 'web',
    version: 'v0.1',
  },
];

const documents: Record<
  string,
  { title: string; content: string; frontmatter: Record<string, unknown> }
> = {
  [MOCK_TOPIC]: {
    title: '00_总览',
    content: `# 00_总览

Deep Research 是一个本地单用户知识工作台，用于承载知识闭环：调研、落库、索引、问答、评估与观测。

## 核心能力

- **调研**：支持 web / repo / mixed 三种 source_mode。
- **知识库管理**：统一查看 topic、文档、metadata、索引状态。
- **LightRAG 索引**：构建向量和图谱索引。
- **知识库问答**：基于 active knowledge base 进行可信问答。
- **质量评估**：运行真实 QA 链路并生成 Markdown 报告。
- **观测**：通过 SSE 事件流追踪任务和接口状态。

## 架构

前端采用 React + TypeScript + Vite，后端采用 FastAPI。`,
    frontmatter: { title: '00_总览', topic: MOCK_TOPIC },
  },
  [`${MOCK_TOPIC}_01`]: {
    title: '01_核心概念',
    content: `# 01_核心概念

## Controls

Controls 是 Deep Research 中用于封装工具调用和状态流转的基础结构。

## Sensors

Sensors 负责观测 Agent 运行状态并生成事件。`,
    frontmatter: { title: '01_核心概念', topic: MOCK_TOPIC },
  },
  [`${MOCK_TOPIC}_02`]: {
    title: '02_实践指南',
    content: `# 02_实践指南

## 快速开始

1. 选择一个 active knowledge base。
2. 进入 Knowledge QA 页面提问。
3. 查看 citations 与 confidence。`,
    frontmatter: { title: '02_实践指南', topic: MOCK_TOPIC },
  },
  [MOCK_TOPIC_2]: {
    title: '00_总览',
    content: `# LangGraph 状态管理

LangGraph 是 LangChain 生态中用于构建有状态、多角色 Agent 应用的框架。`,
    frontmatter: { title: '00_总览', topic: MOCK_TOPIC_2 },
  },
};

const metadataByTopic: Record<string, KnowledgeBase['metadata']> = {
  [MOCK_TOPIC]: {
    topic: MOCK_TOPIC,
    title: 'Harness Engineering 调研知识库',
    version: 'v0.1',
    source_mode: 'web',
    description: 'Deep Research 项目核心知识库',
  },
  [MOCK_TOPIC_2]: {
    topic: MOCK_TOPIC_2,
    title: 'LangGraph 状态管理',
    version: 'v0.1',
    source_mode: 'web',
    description: 'LangGraph 状态管理调研',
  },
};

const jobs: Record<string, any> = {};
const channels: Record<string, Channel> = {
  'feishu:oc_agent_learning_demo': {
    channel_id: 'feishu:oc_agent_learning_demo',
    platform: 'feishu',
    external_id: 'oc_agent_learning_demo',
    title: 'Agent 学习群',
    mode: 'main',
    active_kb: MOCK_TOPIC,
    topic_binding: MOCK_TOPIC,
    memory_scope: 'channel',
    created_at: nowText(),
    updated_at: nowText(),
  },
  'feishu:oc_vibecoding_demo': {
    channel_id: 'feishu:oc_vibecoding_demo',
    platform: 'feishu',
    external_id: 'oc_vibecoding_demo',
    title: 'Vibecoding 学习群',
    mode: 'kb_qa',
    active_kb: MOCK_TOPIC_2,
    topic_binding: null,
    memory_scope: 'channel',
    created_at: nowText(),
    updated_at: nowText(),
  },
};
const channelMessages: Record<string, any[]> = {};
const channelPushConfigs: Record<string, ChannelPushConfig> = {};

function getChannelPushConfig(channelId: string): ChannelPushConfig {
  return (
    channelPushConfigs[channelId] || {
      enabled: false,
      frequency: 'daily',
      content_strategy: 'rotation',
      push_time: '09:00',
      timezone: 'Asia/Shanghai',
      next_push_at: null,
      last_pushed_at: null,
      last_pushed_doc: null,
      scheduler_enabled: true,
      scheduler_running: true,
      scheduler_tick_interval_seconds: 600,
    }
  );
}


const mockMatrixSubjects = ['Elicit', 'Perplexity', 'Gemini Deep Research'];
const mockMatrixAxes = ['证据引用', '报告生成', '适用场景'];

function urlPart(value: string) {
  return value.toLowerCase().split(' ').join('-');
}

function buildMockResearchResult(input: ResearchJobCreateRequest, options?: { resolved?: boolean }): ResearchJobResult {
  const problemMode = !options?.resolved && /风险|问题|失败|weak|unsupported|missing/i.test(input.user_request || '');
  const matrixCells = mockMatrixSubjects.flatMap((subject, subjectIndex) =>
    mockMatrixAxes.map((axis, axisIndex) => ({
      id: `cell_${subjectIndex + 1}_${axisIndex + 1}`,
      subject,
      axis,
      question: `${subject} 在“${axis}”上的公开证据是什么？`,
    }))
  );
  const evidenceTable = matrixCells.map((cell, index) => ({
    id: `ev_${String(index + 1).padStart(3, '0')}`,
    section_id: 'horizontal_compare',
    section_title: '02_横向对比',
    question_id: cell.id,
    question: cell.question,
    subject: cell.subject,
    comparison_axis: cell.axis,
    matrix_cell_id: cell.id,
    claim: `${cell.subject} 在${cell.axis}维度有可审计的公开资料支撑。`,
    quote: `${cell.subject} mock quote for ${cell.axis} evidence.`,
    source_url: `https://example.com/${urlPart(cell.subject)}/${index + 1}`,
    source_title: `${cell.subject} ${cell.axis} reference`,
    source_domain: 'example.com',
    confidence: 'medium',
    chunk_id: `chunk_${String(index + 1).padStart(3, '0')}`,
    used_in_document: problemMode && index === 4 ? false : true,
    risk_flags: problemMode && index === 2 ? ['single_source', 'needs_confirmation'] : [],
  }));
  const evidenceMap = evidenceTable.map((row) => ({
    id: row.id,
    section_id: row.section_id,
    question_id: row.question_id,
    subject: row.subject,
    comparison_axis: row.comparison_axis,
    matrix_cell_id: row.matrix_cell_id,
    claim: row.claim,
    evidence_text: row.quote,
    source_url: row.source_url,
    source_title: row.source_title,
    confidence: row.confidence,
    used_by_documents: row.used_in_document ? ['02_横向对比'] : [],
  }));
  const selectedChunks = evidenceTable.map((row, index) => ({
    id: row.chunk_id,
    material_id: `mat_${(index % 3) + 1}`,
    section_id: row.section_id,
    question_id: row.question_id,
    question: row.question,
    subject: row.subject,
    comparison_axis: row.comparison_axis,
    matrix_cell_id: row.matrix_cell_id,
    text: row.quote,
    score: 0.72 + index / 100,
    source_url: row.source_url,
    source_title: row.source_title,
  }));
  const weakCellId = matrixCells[matrixCells.length - 1]?.id;
  const matrixRows = matrixCells.map((cell) => ({
    matrix_cell_id: cell.id,
    subject: cell.subject,
    comparison_axis: cell.axis,
    evidence_count: problemMode && cell.id === weakCellId ? 0 : 1,
    source_count: problemMode && cell.id === weakCellId ? 0 : 1,
    domain_count: 1,
    used_evidence_count: problemMode && cell.id === 'cell_2_2' ? 0 : 1,
    risk_flag_count: problemMode && cell.id === 'cell_1_3' ? 2 : 0,
    risk_flags: problemMode && cell.id === 'cell_1_3' ? ['single_source', 'needs_confirmation'] : [],
  }));
  const weakMatrixRows = problemMode ? matrixRows.filter((row) => row.evidence_count < 1 || row.source_count < 1) : [];
  const documentRiskFlags: Record<string, number> = problemMode ? { single_source: 1, needs_confirmation: 1 } : {};
  const questionStates: NonNullable<ResearchJobResult['question_states']> = matrixCells.map((cell, index) => {
    const isUncovered = problemMode && cell.id === weakCellId;
    const isPartial = problemMode && index === 4;
    const answerStatus = isUncovered ? 'uncovered' : isPartial ? 'partial' : 'complete';
    const hardStatus = isUncovered ? 'uncovered' : isPartial ? 'partial' : 'covered';
    const evidenceId = `ev_${String(index + 1).padStart(3, '0')}`;
    return {
      question_id: cell.id,
      section_id: 'horizontal_compare',
      section_title: '02_横向对比',
      question: cell.question,
      success_criteria: [
        `说明 ${cell.subject} 在“${cell.axis}”上的公开事实与机制`,
        '补充限制条件、反例或第三方评价',
      ],
      question_kind: 'comparison',
      status: answerStatus === 'complete' ? 'sufficient' : 'exhausted',
      stop_reason: answerStatus === 'complete' ? 'sufficient_evidence' : 'max_rounds_reached',
      rounds_used: answerStatus === 'complete' ? 1 : 2,
      criteria: [
        {
          criterion_id: `${cell.id}_criterion_01`,
          description: `说明 ${cell.subject} 在“${cell.axis}”上的公开事实与机制`,
          level: 'hard',
          status: hardStatus,
          evidence_ids: isUncovered ? [] : [evidenceId],
        },
        {
          criterion_id: `${cell.id}_criterion_02`,
          description: '补充限制条件、反例或第三方评价',
          level: 'soft',
          status: answerStatus === 'complete' ? 'covered' : 'partial',
          evidence_ids: isUncovered ? [] : [evidenceId],
        },
      ],
      accepted_evidence: isUncovered ? [] : [evidenceMap[index]],
      rejected_evidence: [],
      gap_actions: isUncovered
        ? [{ action: 'search_more', gap_type: 'criterion_uncovered', query: `${cell.subject} ${cell.axis} limitations` }]
        : [],
      gap_execution_log: [],
      access_issues: [],
      coverage: {
        sufficient: answerStatus === 'complete',
        answer_status: answerStatus,
        coverage_status: answerStatus === 'complete' ? 'sufficient' : 'insufficient',
        evidence_count: isUncovered ? 0 : 1,
        claim_cluster_count: isUncovered ? 0 : 1,
        source_count: isUncovered ? 0 : 1,
        reason: answerStatus === 'complete'
          ? '必需覆盖标准均有证据支持。'
          : answerStatus === 'partial'
            ? '核心事实已有证据，但部分覆盖标准仍需补强。'
            : '必需覆盖标准尚未获得可准入证据。',
      },
    };
  });
  const reportBlueprintSections = [
    {
      section_id: 'overview',
      title: '00_总览',
      reportability: {
        status: 'ready',
        core_section: true,
        evidence_count: evidenceTable.length,
        source_count: evidenceTable.length,
      },
    },
    {
      section_id: 'horizontal_compare',
      title: '02_横向对比',
      reportability: {
        status: problemMode ? 'limited' : 'ready',
        core_section: true,
        evidence_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
        source_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
        reason: problemMode ? 'mock 问题态：存在一个未覆盖的比较矩阵格。' : undefined,
      },
    },
    {
      section_id: 'references',
      title: '05_证据与参考',
      reportability: {
        status: 'ready',
        core_section: false,
        evidence_count: evidenceTable.length,
        source_count: evidenceTable.length,
      },
    },
  ];
  const reportBlueprint = {
    title: input.user_request,
    sections: reportBlueprintSections,
  };
  const documents = [
    {
      title: '00_总览',
      section_id: 'overview',
      content: `# ${input.user_request}\n\n本 mock 调研展示多主题对比任务的证据矩阵、覆盖度和审批状态。`,
      section_brief: reportBlueprintSections[0],
    },
    {
      title: '02_横向对比',
      section_id: 'horizontal_compare',
      content: `# 02_横向对比\n\n${mockMatrixAxes
        .map((axis, axisIndex) => `## ${axis}\n\n${mockMatrixSubjects
          .map((subject, subjectIndex) => {
            const evidenceIndex = subjectIndex * mockMatrixAxes.length + axisIndex + 1;
            if (problemMode && evidenceIndex === 5) {
              return `- ${subject}: 该维度有证据但正文暂未绑定引用。`;
            }
            return `- ${subject}: 该维度已有绑定证据。[ev_${String(evidenceIndex).padStart(3, '0')}]`;
          })
          .join('\n')}`)
        .join('\n\n')}\n\n## 待复核结论\n\n- 这个 mock 问题态包含一条未支撑的强事实结论，用于验证前端审查栏。`,
      evidence_table: evidenceTable,
      evidence_metrics: {
        evidence_count: evidenceTable.length,
        source_count: evidenceTable.length,
        domain_count: 1,
        question_count: matrixCells.length,
        used_evidence_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
        risk_flag_count: problemMode ? 2 : 0,
        risk_flags: documentRiskFlags,
        weak_question_count: problemMode ? 1 : 0,
        weak_questions: problemMode
          ? [
              {
                question_id: weakCellId,
                question: 'Gemini Deep Research 在“适用场景”上的公开证据是什么？',
                evidence_count: 0,
                source_count: 0,
                domain_count: 0,
                reason: 'mock 问题态：该格子未补足证据。',
              },
            ]
          : [],
        matrix_cell_count: matrixRows.length,
        covered_matrix_cell_count: matrixRows.length - weakMatrixRows.length,
        weak_matrix_cell_count: weakMatrixRows.length,
        subject_count: mockMatrixSubjects.length,
        axis_count: mockMatrixAxes.length,
        matrix_rows: matrixRows,
        weak_matrix_cells: weakMatrixRows.map((row) => ({
          matrix_cell_id: row.matrix_cell_id,
          subject: row.subject,
          comparison_axis: row.comparison_axis,
          evidence_count: row.evidence_count,
          source_count: row.source_count,
          domain_count: row.domain_count,
          reason: 'mock 问题态：矩阵格证据不足。',
        })),
        paragraph_validation_status: problemMode ? 'needs_review' : 'passed',
      },
      paragraph_validation: {
        status: problemMode ? 'needs_review' : 'passed',
        failed_group_count: problemMode ? 1 : 0,
        composer: {
          group_count: matrixRows.length,
          modes: { matrix_deterministic: matrixRows.length },
          fallback_count: problemMode ? 1 : 0,
          model_composed_count: 0,
          elapsed_ms_total: problemMode ? 98000 : 74000,
          elapsed_ms_max: problemMode ? 24000 : 16000,
          slow_groups: [
            {
              title: 'Gemini Deep Research · 适用场景',
              mode: problemMode ? 'fallback_after_validation' : 'matrix_deterministic',
              elapsed_ms: problemMode ? 24000 : 16000,
              row_count: 1,
              validation_status: problemMode ? 'failed' : 'passed',
              validation_reason: problemMode ? 'missing_evidence_ids' : 'ok',
            },
          ],
        },
        groups: problemMode
          ? [
              {
                title: '待复核结论',
                status: 'failed',
                reason: '段落包含未支撑强事实结论。',
                used_ids: ['ev_003'],
                missing_ids: ['ev_005'],
                unknown_ids: ['ev_999'],
              },
            ]
          : [],
      },
      synthesis_diagnostics: {
        title: '02_横向对比',
        elapsed_ms: problemMode ? 102000 : 78000,
        evidence_count: evidenceTable.length,
        source_count: evidenceTable.length,
        group_count: matrixRows.length,
        composer: {
          group_count: matrixRows.length,
          modes: { matrix_deterministic: matrixRows.length },
          fallback_count: problemMode ? 1 : 0,
          model_composed_count: 0,
          elapsed_ms_total: problemMode ? 98000 : 74000,
          elapsed_ms_max: problemMode ? 24000 : 16000,
          slow_groups: [
            {
              title: 'Gemini Deep Research · 适用场景',
              mode: problemMode ? 'fallback_after_validation' : 'matrix_deterministic',
              elapsed_ms: problemMode ? 24000 : 16000,
              row_count: 1,
              validation_status: problemMode ? 'failed' : 'passed',
              validation_reason: problemMode ? 'missing_evidence_ids' : 'ok',
            },
          ],
        },
        groups: mockMatrixSubjects.map((subject, index) => ({
          title: `${subject} · ${mockMatrixAxes[index % mockMatrixAxes.length]}`,
          mode: index === 2 && problemMode ? 'fallback_after_validation' : 'matrix_deterministic',
          row_count: 1,
          elapsed_ms: index === 2 && problemMode ? 24000 : 9000 + index * 1200,
          validation_status: index === 2 && problemMode ? 'failed' : 'passed',
          validation_reason: index === 2 && problemMode ? 'missing_evidence_ids' : 'ok',
          fallback_reason: index === 2 && problemMode ? 'missing_evidence_ids' : undefined,
        })),
      },
      section_brief: reportBlueprintSections[1],
    },
    {
      title: '05_证据与参考',
      section_id: 'references',
      content: '# 05_证据与参考\n\n所有矩阵格子均绑定到 evidence table。',
      section_brief: reportBlueprintSections[2],
    },
  ];

  return {
    status: 'waiting_approval',
    message: '调研产物已生成，等待审批。',
    research_strategy: input.research_strategy,
    depth: input.depth,
    evidence_native_blueprint: reportBlueprint,
    report_blueprint: reportBlueprint,
    research_plan: {
      strategy: input.research_strategy || 'hv_analysis',
      strategy_profile: 'multi_subject_compare',
      topic: input.user_request,
      subjects: mockMatrixSubjects,
      comparison_axes: mockMatrixAxes,
      matrix_cells: matrixCells,
      judgement_questions: ['这些工具在证据引用、报告生成和适用场景上的主要差异是什么？'],
      sections: [
        { id: 'overview', title: '00_总览', axis: 'vertical', questions: ['调研对象是什么？'] },
        { id: 'horizontal_compare', title: '02_横向对比', axis: 'horizontal', questions: matrixCells.map((cell) => cell.question) },
      ],
      vertical: [],
      horizontal: [],
      intersections: [],
    },
    query_plan: {
      strategy: input.research_strategy || 'hv_analysis',
      depth: input.depth || 'normal',
      query_limit: 12,
      queries: matrixCells.map((cell) => ({
        section_id: 'horizontal_compare',
        section_title: '02_横向对比',
        question_id: cell.id,
        question: cell.question,
        query: `${cell.subject} ${cell.axis} evidence official documentation`,
        subject: cell.subject,
        comparison_axis: cell.axis,
        matrix_cell_id: cell.id,
      })),
    },
    question_states: questionStates,
    coverage: {
      ready_to_write: true,
      round_index: 1,
      max_rounds: input.max_rounds || 2,
      source_count: 9,
      document_count: 3,
      evidence_count: evidenceTable.length,
      key_fact_count: evidenceTable.length,
      evidence_pack_count: evidenceTable.length,
      depth_ready: !problemMode,
      answer_status_counts: {
        complete: questionStates.filter((state) => state.coverage?.answer_status === 'complete').length,
        partial: questionStates.filter((state) => state.coverage?.answer_status === 'partial').length,
        uncovered: questionStates.filter((state) => state.coverage?.answer_status === 'uncovered').length,
      },
      per_title_status: [],
      weak_sections: problemMode
        ? [
            {
              section_id: 'horizontal_compare',
              title: '02_横向对比',
              evidence_count: evidenceTable.length - 1,
              source_count: evidenceTable.length - 1,
              domain_count: 1,
              question_count: matrixCells.length,
              covered_question_count: matrixCells.length - 1,
              weak_matrix_cell_count: weakMatrixRows.length,
              weak_matrix_cells: weakMatrixRows.map((row) => ({
                matrix_cell_id: row.matrix_cell_id,
                subject: row.subject,
                comparison_axis: row.comparison_axis,
                evidence_count: row.evidence_count,
                source_count: row.source_count,
                domain_count: row.domain_count,
                reason: 'mock 问题态：矩阵格证据不足。',
              })),
              reason: 'mock 问题态：存在薄弱矩阵格和正文引用缺口。',
            },
          ]
        : [],
      covered_titles: input.required_titles || ['00_总览', '02_横向对比', '05_证据与参考'],
      missing_titles: [],
      fatal_gaps: [],
      followup_queries: problemMode ? ['Gemini Deep Research 适用场景 evidence official documentation'] : [],
      reason: problemMode ? 'mock 问题态：存在薄弱矩阵格、未引用证据和未支撑结论。' : '多主题矩阵覆盖完整，可以写入。',
    },
    quality_gate: {
      status: problemMode ? 'needs_more_evidence' : 'passed',
      approval_ready: !problemMode,
      reason: problemMode ? 'mock 问题态：需补充薄弱矩阵格并处理正文缺失引用。' : 'mock 多主题矩阵证据覆盖通过。',
      checks: {
        grounding: true,
        citation_validation: !problemMode,
        report_quality: {
          status: problemMode ? 'failed' : 'passed',
          blocking_issues: problemMode ? ['unsupported_core_claim', 'missing_inline_reference'] : [],
        },
        core_claim_support: !problemMode,
        semantic_review_complete: true,
        report_completeness: true,
        fallback_sections_absent: !problemMode,
      },
    },
    grounding: {
      evidence_url_count: evidenceTable.length,
      per_document: [
        { title: '02_横向对比', evidence_ref_count: evidenceTable.length, matched_evidence_ref_count: evidenceTable.length },
      ],
    },
    documents,
    sources: evidenceTable.map((row) => ({ title: row.source_title, url: row.source_url })),
    evidence_pack: evidenceTable.map((row, index) => ({ fact: row.claim, sources: [index] })),
    materials: mockMatrixSubjects.map((subject, index) => ({
      id: `mat_${index + 1}`,
      title: `${subject} public materials`,
      url: `https://example.com/${urlPart(subject)}`,
      provider: 'mock',
      source_type: 'web',
      content_length: 2400,
      content_quality: 'medium',
      degraded: false,
      excerpt: `${subject} mock material excerpt.`,
      chunk_count: 3,
    })),
    selected_chunks: selectedChunks,
    evidence_map: evidenceMap,
    validation: {
      status: problemMode ? 'failed' : 'passed',
      reason: problemMode ? 'mock 问题态：存在未支撑结论、缺失正文证据标记和未引用证据。' : 'mock 证据引用全部匹配。',
      evidence_url_count: evidenceTable.length,
      evidence_claim_count: evidenceTable.length,
      per_document: [
        {
          title: '02_横向对比',
          evidence_ref_count: evidenceTable.length,
          matched_evidence_ref_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
          matched_by_id_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
          matched_by_fallback_count: 0,
          match_ratio: problemMode ? 0.89 : 1,
          unsupported_refs: [],
          unsupported_claims: problemMode
            ? [
                {
                  text: '这个 mock 问题态包含一条未支撑的强事实结论。',
                  reason: '该句没有匹配到 evidence_table 中的证据。',
                },
              ]
            : [],
          missing_inline_refs: problemMode
            ? [
                {
                  evidence_id: 'ev_005',
                  claim: 'Perplexity 在报告生成维度有可审计的公开资料支撑。',
                  source_url: 'https://example.com/perplexity/5',
                },
              ]
            : [],
        },
      ],
      per_section: [
        {
          section_id: 'horizontal_compare',
          evidence_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
          source_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
          domain_count: 1,
          question_count: matrixCells.length,
          used_ref_count: problemMode ? evidenceTable.length - 1 : evidenceTable.length,
        },
      ],
    },
    retrieval: {
      evidence_diagnostics: {
        extraction_mode: 'mock_matrix',
        section_count: 1,
        selected_chunk_count: selectedChunks.length,
        evidence_count: evidenceTable.length,
        model_evidence_count: evidenceTable.length,
        fallback_evidence_count: problemMode ? 1 : 0,
        timeout_count: 0,
        error_count: 0,
        rejected_count: problemMode ? 1 : 0,
        extraction_sections: [
          {
            section_id: 'horizontal_compare',
            title: '02_横向对比',
            status: problemMode ? 'fallback_after_validation' : 'complete',
            selected_chunk_count: selectedChunks.length,
            evidence_count: evidenceTable.length,
            source_count: evidenceTable.length,
            domain_count: 1,
            model_evidence_count: evidenceTable.length,
            fallback_evidence_count: problemMode ? 1 : 0,
            rejected_count: problemMode ? 1 : 0,
            weak_question_count: weakMatrixRows.length,
            fallback_used: problemMode,
            elapsed_ms: 1200,
          },
        ],
      },
      diagnostics: {
        section_evidence_extraction: {
          mode: 'mock_matrix',
          elapsed_ms: 1200,
          model_evidence_count: evidenceTable.length,
          fallback_evidence_count: 0,
          timeout_count: 0,
          error_count: 0,
          rejected_count: 0,
          sections: [
            {
              section_id: 'horizontal_compare',
              title: '02_横向对比',
              status: 'complete',
              chunk_count: selectedChunks.length,
              model_evidence_count: evidenceTable.length,
              fallback_evidence_count: 0,
              rejected_count: 0,
              fallback_used: false,
              elapsed_ms: 1200,
            },
          ],
        },
      },
    },
    suggested_topic: input.topic || slugify(input.user_request),
    artifacts: ['coverage_report', 'document_draft'],
  };
}

function mockFollowupQuery(request: ResearchFollowupRequest, input?: Record<string, unknown>) {
  if (request.query?.trim()) return request.query.trim();
  const parts = [
    request.subject,
    request.comparison_axis,
    request.reason || input?.user_request,
    '权威资料',
    '官方文档',
    'evidence',
  ];
  return parts.map((item) => String(item || '').trim()).filter(Boolean).join(' ');
}

function resolveMockResearchFollowup(job: ResearchJob, request: ResearchFollowupRequest): ResearchFollowupResponse {
  const input = (job.input || {}) as unknown as ResearchJobCreateRequest;
  const query = mockFollowupQuery(request, job.input);
  const action = {
    id: `followup_${Date.now()}`,
    ...request,
    query,
    status: 'running',
    requested_at: nowText(),
    complete_at: Date.now() + 900,
  };
  const currentResult = (job.result as ResearchJobResult | undefined) || buildMockResearchResult(input);
  const runningResult = {
    ...currentResult,
    message: '补查已启动，正在后台执行 targeted follow-up。',
    coverage: {
      ...currentResult.coverage!,
      followup_queries: [query],
      reason: `已启动补查：${query}`,
    },
    followup_actions: [
      ...(((job.result as ResearchJobResult | undefined)?.followup_actions as unknown[]) || []),
      action,
    ] as unknown as ResearchJobResult['followup_actions'],
  };
  job.result = runningResult;
  job.status = 'running';
  job.updated_at = nowText();
  job.progress = { current: 0, total: 1, message: 'followup running' };
  window.setTimeout(() => maybeCompleteMockResearchFollowup(job), 900);
  return {
    job_id: job.job_id,
    followup_job_id: action.id,
    status: 'followup_started',
    message: 'mock targeted follow-up 已启动。',
    followup_queries: [query],
  };
}

function maybeCompleteMockResearchFollowup(job: ResearchJob) {
  const actions = ((job.result as ResearchJobResult | undefined)?.followup_actions as Array<Record<string, unknown>>) || [];
  const running = actions.find((action) => action.status === 'running');
  if (!running) return;
  resolveMockResearchFollowupResult(
    job,
    running as unknown as ResearchFollowupRequest,
    String(running.query || ''),
    String(running.id || '')
  );
}

function resolveMockResearchFollowupResult(job: ResearchJob, request: ResearchFollowupRequest, query: string, actionId: string) {
  const input = (job.input || {}) as unknown as ResearchJobCreateRequest;
  const nextResult = buildMockResearchResult(input, { resolved: true });
  nextResult.coverage = {
    ...nextResult.coverage!,
    followup_queries: [query],
    reason: `已完成补查：${query}`,
  };
  nextResult.message = '补查完成，质量门已重新通过。';
  nextResult.quality_gate = {
    ...nextResult.quality_gate!,
    reason: 'mock targeted follow-up 已补齐证据并通过验收。',
    checks: {
      ...nextResult.quality_gate?.checks,
      followup_resolved: true,
    },
  };
  nextResult.followup_actions = (((job.result as ResearchJobResult | undefined)?.followup_actions as Array<Record<string, unknown>>) || []).map(
    (action) =>
      action.id === actionId
        ? {
            ...action,
            ...request,
            query,
            status: 'resolved',
            resolved_at: nowText(),
          }
        : action
  ) as unknown as ResearchJobResult['followup_actions'];
  job.result = nextResult;
  job.status = 'waiting_approval';
  job.updated_at = nowText();
  job.progress = { current: 1, total: 1, message: 'followup_resolved' };
}

function resolveMockResearchSectionRevision(
  job: ResearchJob,
  request: ResearchSectionRevisionRequest
): ResearchSectionRevisionResponse {
  const sectionIds = request.section_ids || [];
  const instruction = request.instruction || '';
  const revisionId = `revision_${Date.now()}`;
  const currentResult = (job.result as ResearchJobResult | undefined) || buildMockResearchResult(job.input as unknown as ResearchJobCreateRequest);
  const action = {
    id: revisionId,
    section_ids: sectionIds,
    instruction,
    status: 'running',
    requested_at: nowText(),
  };
  job.result = {
    ...currentResult,
    message: `章节修改已启动，正在后台执行（${sectionIds.length} 节）。`,
    revision_actions: [...(currentResult.revision_actions || []), action],
  };
  job.status = 'running';
  job.updated_at = nowText();
  job.progress = { current: 0, total: 1, message: 'section revision running' };
  window.setTimeout(() => {
    const resultBeforeRevision = (job.result as ResearchJobResult | undefined) || currentResult;
    const documents = ((resultBeforeRevision.documents as ResearchDocument[] | undefined) || []).map((doc) => {
      if (!sectionIds.includes(doc.section_id || '')) return doc;
      return {
        ...doc,
        content: `${doc.content}\n\n（已按要求修改：${instruction}）`,
        revision_diagnostics: { attempted: true, accepted: true },
      };
    });
    const resolvedAt = nowText();
    const resolvedAction = { ...action, status: 'resolved', resolved_at: resolvedAt };
    job.result = {
      ...resultBeforeRevision,
      documents,
      message: '章节修改已完成，请重新审核。',
      revision_actions: (resultBeforeRevision.revision_actions || []).map((item) =>
        item.id === revisionId ? resolvedAction : item
      ),
      revision_history: [
        ...(resultBeforeRevision.revision_history || []),
        {
          revision_id: revisionId,
          section_ids: sectionIds,
          instruction,
          requested_at: action.requested_at,
          resolved_at: resolvedAt,
        },
      ],
    };
    job.status = 'waiting_approval';
    job.updated_at = nowText();
    job.progress = { current: 1, total: 1, message: 'section revision applied' };
  }, 900);
  return {
    job_id: job.job_id,
    revision_id: revisionId,
    status: 'revision_started',
    section_ids: sectionIds,
  };
}

const mockEvalCases: {
  knowledge_qa: Record<string, Array<Record<string, unknown> & { id: string }>>;
  research_probe: Array<Record<string, unknown> & { id: string }>;
  single_url_brief: Array<Record<string, unknown> & { id: string }>;
} = {
  knowledge_qa: {
    default: [
      { id: 'overview', category: '通用概览', question: '请概括这个知识库的核心主题和主要结论。' },
      { id: 'sources', category: '通用来源', question: '这个知识库主要依据哪些来源或文档？' },
    ],
    [MOCK_TOPIC]: [
      { id: 'definition', category: '基础定义', question: 'Harness Engineering 是什么？' },
      { id: 'context_comparison', category: '对比推理', question: 'Harness Engineering 和 Context Engineering 有什么区别与互补关系？' },
    ],
  },
  research_probe: [
    { id: 'single_explain_harness', kind: 'single_explain', title: 'Harness Engineering 解释型', user_request: '调研 Harness Engineering 循环工程的定义、关键机制和实操路径' },
    { id: 'dual_compare_lightrag_graphrag', kind: 'dual_compare', title: 'LightRAG vs GraphRAG 对比', user_request: '对比调研 LightRAG 与 GraphRAG 的架构差异' },
  ],
  single_url_brief: [
    { id: 'wiki_rag', url: 'https://en.wikipedia.org/wiki/Retrieval-augmented_generation', user_request: '总结这篇文章的核心内容' },
    { id: 'wiki_llm', url: 'https://en.wikipedia.org/wiki/Large_language_model', user_request: '总结一下大语言模型的定义和发展' },
  ],
};

const evaluationCases: EvaluationCase[] = [
  {
    id: 'case-001',
    category: 'retrieval',
    passed: true,
    score: 0.96,
    failed_checks: [],
    likely_failure_type: '',
    recommended_action: '保持当前索引配置。',
    backend: 'lightrag',
    mode: 'mix',
    citation_count: 3,
    question: 'Deep Research 的知识闭环包含哪些步骤？',
    answer: '知识闭环包含调研、落库、索引、问答、评估与观测。',
  },
  {
    id: 'case-002',
    category: 'citation',
    passed: false,
    score: 0.48,
    failed_checks: ['citation_count', 'evidence_alignment'],
    likely_failure_type: 'citation_weak',
    recommended_action: '补充来源文档后在 QA 中复测，并检查引用是否指向具体段落。',
    backend: 'lightrag',
    mode: 'mix',
    citation_count: 1,
    question: '失败任务如何通过 request_id 定位？',
    answer: '可以在观测页按 request_id 过滤事件，再回到对应 job。',
  },
  {
    id: 'case-003',
    category: 'answer_quality',
    passed: false,
    score: 0.54,
    failed_checks: ['coverage_gap'],
    likely_failure_type: 'knowledge_gap',
    recommended_action: '从调研工作台补充日志关联说明，然后重建索引。',
    backend: 'lightrag',
    mode: 'mix',
    citation_count: 2,
    question: '评估失败用例应该如何回流到 QA？',
    answer: '失败用例应复制到 QA 页面，用户调整 backend 与 mode 后复测。',
  },
];

const reports: EvaluationReportResponse[] = [
  {
    report_id: `knowledge-qa-${MOCK_TOPIC}-v0.1-20260612-090724`,
    topic: MOCK_TOPIC,
    created_at: '2026-06-12 09:07:24',
    run_id: 'run_eval_demo_090724',
    content: `# 知识库问答质量评估报告

- **Topic**: ${MOCK_TOPIC}
- **Total**: 3
- **Passed**: 1
- **Failed**: 2
- **Pass Rate**: 33.3%

## 结论

当前知识库可回答核心问题，但引用强度和失败用例回流仍需要维护。`,
    summary: {
      total: 3,
      passed: 1,
      failed: 2,
      pass_rate: 33.3,
      average_score: 66,
      failed_checks: { citation_count: 1, evidence_alignment: 1, coverage_gap: 1 },
      categories: { retrieval: 1, citation: 1, answer_quality: 1 },
      failure_types: { citation_weak: 1, knowledge_gap: 1 },
    },
    cases: evaluationCases,
    recommended_actions: [
      '将失败用例回流到 QA 页面复测。',
      '补充日志关联文档后重建索引。',
      '在观测页按 run_id 检查评估链路。',
    ],
  },
];

function getKbDocumentList(topic: string): DocumentListResponse {
  if (topic === MOCK_TOPIC) {
    return {
      topic,
      documents: [
        {
          document_id: '00_%E6%80%BB%E8%A7%88',
          title: '00_总览',
          filename: '00_总览.md',
          path: `knowledge_base/${topic}/00_总览.md`,
          bytes: 1200,
          sha256: 'abc123',
          frontmatter: { title: '00_总览' },
        },
        {
          document_id: '01_%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5',
          title: '01_核心概念',
          filename: '01_核心概念.md',
          path: `knowledge_base/${topic}/01_核心概念.md`,
          bytes: 950,
          sha256: 'def456',
          frontmatter: { title: '01_核心概念' },
        },
        {
          document_id: '02_%E5%AE%9E%E8%B7%B5%E6%8C%87%E5%8D%97',
          title: '02_实践指南',
          filename: '02_实践指南.md',
          path: `knowledge_base/${topic}/02_实践指南.md`,
          bytes: 800,
          sha256: 'ghi789',
          frontmatter: { title: '02_实践指南' },
        },
      ],
    };
  }
  if (topic === MOCK_TOPIC_2) {
    return {
      topic,
      documents: [
        {
          document_id: '00_%E6%80%BB%E8%A7%88',
          title: '00_总览',
          filename: '00_总览.md',
          path: `knowledge_base/${topic}/00_总览.md`,
          bytes: 600,
          sha256: 'zzz111',
          frontmatter: { title: '00_总览' },
        },
      ],
    };
  }
  throw new ApiError('topic_not_found', `知识库 ${topic} 不存在`, false);
}

function getDocument(topic: string, documentId: string): DocumentResponse {
  const list = getKbDocumentList(topic);
  const meta = list.documents.find((d) => d.document_id === documentId);
  if (!meta) throw new ApiError('document_not_found', '文档不存在', false);

  const key = documentId.includes('%E6%80%BB%E8%A7%88')
    ? topic
    : `${topic}_${documentId.split('_')[0].replace('%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5', '01').replace('%E5%AE%9E%E8%B7%B5%E6%8C%87%E5%8D%97', '02')}`;
  const doc = documents[key] || documents[topic] || { title: meta.title, content: '# 文档内容\n\n（mock 占位）', frontmatter: {} };

  return {
    topic,
    document_id: documentId,
    title: doc.title,
    content: doc.content,
    frontmatter: doc.frontmatter,
    path: meta.path,
  };
}

function getKbDetail(topic: string): KnowledgeBaseDetail {
  const kb = knowledgeBases.find((k) => k.topic === topic);
  if (!kb) throw new ApiError('topic_not_found', `知识库 ${topic} 不存在`, false);
  const list = getKbDocumentList(topic);
  return {
    topic,
    metadata: metadataByTopic[topic] || {},
    metadata_status: kb.metadata_status || 'inferred',
    document_count: list.documents.length,
    documents: list.documents,
    index_status: kb.index_status,
    index_diagnosis: { reason: kb.index_status === 'ready' ? 'ready' : 'missing_storage_files' },
    embedding_model: kb.index_status === 'ready' ? 'BAAI/bge-large-zh-v1.5' : undefined,
    embedding_dim: kb.index_status === 'ready' ? 1024 : undefined,
    graph_stats: kb.graph_stats,
    vector_stats: kb.vector_stats,
  };
}

const MOCK_ENTITY_TYPES = ['concept', 'artifact', 'method', 'tool', 'organization'];

// Build a deterministic mock graph. If the KB reports > NODE_LIMIT nodes, we
// still only emit NODE_LIMIT of them and flag truncated, mirroring the backend.
const MOCK_NODE_LIMIT = 600;

function buildMockGraph(topic: string): {
  nodes: GraphNodeLite[];
  edges: GraphEdgeLite[];
  total: number;
} {
  const kb = getKbDetail(topic);
  const total = kb.graph_stats?.nodes || 0;
  const emit = Math.min(total, MOCK_NODE_LIMIT);
  const nodes: GraphNodeLite[] = [];
  for (let i = 0; i < emit; i += 1) {
    nodes.push({
      id: `${i === 0 ? 'Harness Engineering' : `Entity ${i}`}`,
      type: MOCK_ENTITY_TYPES[i % MOCK_ENTITY_TYPES.length],
      degree: Math.max(1, ((emit - i) % 12) + (i === 0 ? 40 : 0)),
    });
  }
  const edges: GraphEdgeLite[] = [];
  for (let i = 1; i < nodes.length; i += 1) {
    edges.push({ source: nodes[0].id, target: nodes[i].id, weight: 1 + (i % 5) });
    if (i > 1) edges.push({ source: nodes[i - 1].id, target: nodes[i].id, weight: 1 + (i % 3) });
  }
  return { nodes, edges, total };
}

function channelThreadId(channelId: string, mode: string) {
  return `channel:${channelId}:${mode}`;
}

function getChannelOrThrow(channelId: string): Channel {
  const channel = channels[channelId];
  if (!channel) throw new ApiError('channel_not_found', `channel not found: ${channelId}`, false);
  return channel;
}

function pushChannelMessage(channelId: string, mode: string, role: 'user' | 'assistant', content: string) {
  const threadId = channelThreadId(channelId, mode);
  const key = `${channelId}:${mode}`;
  channelMessages[key] ||= [];
  channelMessages[key].push({
    id: `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`,
    thread_id: threadId,
    role,
    content,
    created_at: nowText(),
  });
}

const mockEvidenceKeys: ResearchRoleKey[] = [
  { id: 'ev-key-1', masked: 'sk-ZU3A...4DVX', source: 'dedicated', deletable: true, shared: false },
  { id: 'ev-key-2', masked: 'sk-dzTU...Y6AE', source: 'dedicated', deletable: true, shared: false },
  { id: 'ev-key-3', masked: 'sk-UOuT...DN7q', source: 'dedicated', deletable: true, shared: false },
  { id: 'ev-key-4', masked: 'sk-kJ6l...ykKk', source: 'dedicated', deletable: true, shared: false },
  { id: 'ev-key-5', masked: 'sk-Cd2p...nRGP', source: 'legacy_shared', deletable: true, shared: true },
  { id: 'ev-key-6', masked: 'sk-W4TQ...lqH2', source: 'legacy_shared', deletable: true, shared: true },
];
let mockWriterKeys: ResearchRoleKey[] = [];

function mockRoleConfig(role: ResearchRole): ResearchRoleModelConfig {
  const inherited = role === 'writer' && mockWriterKeys.length === 0;
  const keys = inherited
    ? mockEvidenceKeys.map((key) => ({ ...key, source: 'inherited_evidence' as const, deletable: false, shared: true }))
    : role === 'writer'
      ? mockWriterKeys
      : mockEvidenceKeys;
  return {
    model: 'deepseek-v4-flash',
    base_url_configured: true,
    base_url: 'https://api.example.com/v1',
    base_url_source: role === 'writer' ? 'inherited_evidence' : 'dedicated',
    key_count: keys.length,
    keys,
    inherits_keys_from: inherited ? 'research_evidence_model' : null,
  };
}

function mockMaskKey(key: string): string {
  const trimmed = key.trim();
  return trimmed.length <= 8 ? `${trimmed.slice(0, 2)}...${trimmed.slice(-2)}` : `${trimmed.slice(0, 7)}...${trimmed.slice(-4)}`;
}

const mockHandlers: Array<{
  match: (path: string, method: string) => boolean;
  handler: (path: string, body?: unknown, params?: Record<string, unknown>) => unknown;
}> = [
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/channels',
    handler: () =>
      okEnvelope<ChannelListResponse>({
        channels: Object.values(channels).sort((a, b) => b.updated_at.localeCompare(a.updated_at)),
        total: Object.keys(channels).length,
      }),
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/channels',
    handler: (_p, body) => {
      const payload = (body || {}) as { platform?: string; external_id?: string; title?: string };
      const platform = payload.platform || 'local';
      const externalId = payload.external_id || '';
      const channelId = `${platform}:${externalId}`;
      channels[channelId] = {
        channel_id: channelId,
        platform,
        external_id: externalId,
        title: payload.title || externalId,
        mode: 'main',
        active_kb: null,
        topic_binding: null,
        memory_scope: 'channel',
        created_at: nowText(),
        updated_at: nowText(),
      };
      return okEnvelope<Channel>(channels[channelId]);
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/channels/sync-feishu-titles',
    handler: () => okEnvelope({
      attempted: Object.values(channels).filter((channel) => channel.platform === 'feishu').length,
      resolved: Object.values(channels).filter((channel) => channel.platform === 'feishu').length,
      updated: 0,
      failed: 0,
    }),
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/channels\/[^/]+$/.test(p),
    handler: (p) => {
      const channelId = decodeURIComponent(p.split('/')[4]);
      const channel = getChannelOrThrow(channelId);
      return okEnvelope<Channel>({
        ...channel,
        threads: {
          main: channelThreadId(channelId, 'main'),
          kb_qa: channelThreadId(channelId, 'kb_qa'),
        },
      });
    },
  },
  {
    match: (p, m) => m === 'PUT' && /^\/api\/v1\/channels\/[^/]+$/.test(p),
    handler: (p, body) => {
      const channelId = decodeURIComponent(p.split('/')[4]);
      const channel = getChannelOrThrow(channelId);
      const payload = (body || {}) as Partial<Channel>;
      channels[channelId] = {
        ...channel,
        title: payload.title ?? channel.title,
        mode: payload.mode ?? channel.mode,
        active_kb: payload.active_kb ?? channel.active_kb,
        topic_binding: payload.topic_binding !== undefined ? payload.topic_binding || null : channel.topic_binding,
        updated_at: nowText(),
      };
      return okEnvelope<Channel>(channels[channelId]);
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/channels\/[^/]+\/push-config$/.test(p),
    handler: (p) => {
      const channelId = decodeURIComponent(p.split('/')[4]);
      getChannelOrThrow(channelId);
      return okEnvelope<ChannelPushConfig>(getChannelPushConfig(channelId));
    },
  },
  {
    match: (p, m) => m === 'PUT' && /^\/api\/v1\/channels\/[^/]+\/push-config$/.test(p),
    handler: (p, body) => {
      const channelId = decodeURIComponent(p.split('/')[4]);
      getChannelOrThrow(channelId);
      const current = getChannelPushConfig(channelId);
      const payload = (body || {}) as Partial<ChannelPushConfig>;
      const enabled = payload.enabled ?? current.enabled;
      const pushTime = payload.push_time ?? current.push_time;
      const timezone = payload.timezone ?? current.timezone;
      channelPushConfigs[channelId] = {
        ...current,
        enabled,
        frequency: payload.frequency ?? current.frequency,
        content_strategy: payload.content_strategy ?? current.content_strategy,
        push_time: pushTime,
        timezone,
        next_push_at: enabled ? mockNextPushAt(pushTime, timezone) : null,
      };
      return okEnvelope<ChannelPushConfig>(channelPushConfigs[channelId]);
    },
  },

  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/channels\/[^/]+\/messages$/.test(p),
    handler: (p, _body, params) => {
      const channelId = decodeURIComponent(p.split('/')[4]);
      const mode = String(params?.mode || 'main');
      getChannelOrThrow(channelId);
      const threadId = channelThreadId(channelId, mode);
      return okEnvelope<ChannelMessagesResponse>({
        channel_id: channelId,
        mode: mode as any,
        thread_id: threadId,
        conversation: null,
        messages: channelMessages[`${channelId}:${mode}`] || [],
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/channels\/[^/]+\/jobs$/.test(p),
    handler: (p) => {
      const channelId = decodeURIComponent(p.split('/')[4]);
      return okEnvelope<ChannelJobsResponse>({
        channel_id: channelId,
        jobs: Object.values(jobs).filter((job) => job.input?.channel_id === channelId),
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/health',
    handler: () =>
      okEnvelope<HealthResponse>({
        service: 'Deep Research API',
        version: '0.1.0',
        time: nowText(),
      }),
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/capabilities',
    handler: () =>
      okEnvelope<CapabilitiesResponse>({
        chat: true,
        research_jobs: true,
        knowledge_base_management: true,
        knowledge_qa: true,
        lightrag_index: true,
        knowledge_evaluation: true,
        observability: true,
        model_check: true,
      }),
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/config/runtime',
    handler: () =>
      okEnvelope<RuntimeConfigResponse>({
        model_name: 'gpt-4.1-mini',
        base_url_configured: true,
        embedding_model: 'BAAI/bge-large-zh-v1.5',
        embedding_dim: 1024,
        lightrag_enabled: true,
        knowledge_base_root: 'knowledge_base',
        log_enabled: true,
      }),
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/knowledge-bases',
    handler: () =>
      okEnvelope<KnowledgeBaseListResponse>({
        active_topic: MOCK_TOPIC,
        topic_count: knowledgeBases.length,
        topics: knowledgeBases,
      }),
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/knowledge-bases/active',
    handler: () => okEnvelope<ActiveKnowledgeBaseResponse>({ active_topic: MOCK_TOPIC }),
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/knowledge-bases/active',
    handler: (_p, body) => {
      const { topic } = (body || {}) as { topic?: string };
      if (topic && !knowledgeBases.some((k) => k.topic === topic)) {
        throw new ApiError('topic_not_found', `知识库 ${topic} 不存在`, false);
      }
      return okEnvelope<ActiveKnowledgeBaseResponse>({ active_topic: topic || null });
    },
  },
  {
    match: (p, m) => {
      const m2 = /^\/api\/v1\/knowledge-bases\/([^/]+)$/.exec(p);
      return m === 'GET' && !!m2 && !p.includes('/documents') && !p.includes('/summary') && !p.includes('/index') && !p.includes('/graph') && !p.includes('/vectors');
    },
    handler: (p) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      return okEnvelope(getKbDetail(topic));
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/summary$/.test(p),
    handler: (p) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      return okEnvelope<KnowledgeBaseSummaryResponse>({
        topic,
        summary: `${topic} 是一个调研知识库，包含 ${getKbDocumentList(topic).documents.length} 篇文档。当前索引状态：${getKbDetail(topic).index_status}。`,
        status: { ready_for_qa: getKbDetail(topic).index_status === 'ready' },
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/documents$/.test(p),
    handler: (p) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      return okEnvelope(getKbDocumentList(topic));
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/documents\/[^/]+$/.test(p),
    handler: (p) => {
      const parts = p.split('/');
      const topic = decodeURIComponent(parts[4]);
      const documentId = parts[6];
      return okEnvelope(getDocument(topic, documentId));
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/knowledge-bases\/[^/]+\/query$/.test(p),
    handler: (p, body) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      const kb = getKbDetail(topic);
      const { question, backend, mode } = (body || {}) as {
        question?: string;
        backend?: string;
        mode?: string;
      };
      if (!question) throw new ApiError('validation_error', '问题不能为空', false);
      if (backend === 'lightrag' && kb.index_status !== 'ready') {
        throw new ApiError('index_missing', 'LightRAG 索引不可用', false);
      }
      const useLightRag = (backend === 'lightrag' || (backend === 'auto' && kb.index_status === 'ready'));
      return okEnvelope<QueryResponse>({
        status: 'answered',
        topic,
        backend: useLightRag ? 'lightrag' : 'local',
        mode: useLightRag ? (mode as QueryResponse['mode']) || 'mix' : 'markdown',
        answer: `关于「${question}」的回答（mock）：\n\nDeep Research 使用 LightRAG 构建知识图谱和向量索引，支持 mix、local、global、hybrid、naive 等多种检索模式。`,
        citations: [
          {
            title: '00_总览',
            path: '00_总览.md',
            snippet: 'Deep Research 是一个本地单用户知识工作台，用于承载知识闭环。',
          },
          {
            title: '01_核心概念',
            path: '01_核心概念.md',
            snippet: 'Controls 是 Deep Research 中用于封装工具调用和状态流转的基础结构。',
          },
        ],
        confidence: 'medium',
        gaps: [],
        metadata: {
          embedding_model: 'BAAI/bge-large-zh-v1.5',
          embedding_dim: 1024,
          indexed_at: '2026-06-12 09:00:00',
          request_id: 'req_qa_demo_001',
        },
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/index$/.test(p),
    handler: (p) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      const kb = getKbDetail(topic);
      return okEnvelope<IndexStatusResponse>({
        topic,
        index_status: kb.index_status,
        index_reason: kb.index_status,
        diagnosis: kb.index_diagnosis,
        embedding_model: kb.embedding_model,
        embedding_dim: kb.embedding_dim,
        indexed_at: '2026-06-12 09:00:00',
        workspace: `lightrag_workspaces/${topic}`,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/graph$/.test(p),
    handler: (p, _body, params) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      const kb = getKbDetail(topic);
      const limit = Math.max(1, Number(params?.max_nodes) || MOCK_NODE_LIMIT);
      const { nodes, edges, total } = buildMockGraph(topic);
      const ranked = [...nodes].sort((a, b) => b.degree - a.degree);
      const truncated = ranked.length > limit;
      const kept = truncated ? ranked.slice(0, limit) : ranked;
      const keptIds = new Set(kept.map((n) => n.id));
      const keptEdges = edges.filter((e) => keptIds.has(e.source) && keptIds.has(e.target));
      return okEnvelope<GraphStatsResponse>({
        topic,
        nodes: kb.graph_stats?.nodes || 0,
        edges: kb.graph_stats?.edges || 0,
        top_entities: ranked.slice(0, 10).map((n) => ({ entity: n.id, type: n.type, count: n.degree })),
        workspace: `lightrag_workspaces/${topic}`,
        graph_file: 'graph_chunk_entity_relation.graphml',
        graph_nodes: kept,
        graph_edges: keptEdges,
        total_nodes: total,
        total_edges: kb.graph_stats?.edges || edges.length,
        truncated,
        node_limit: limit,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/graph\/nodes\/[^/]+$/.test(p),
    handler: (p) => {
      const parts = p.split('/');
      const topic = decodeURIComponent(parts[4]);
      const entityId = decodeURIComponent(parts[7]);
      const { nodes, edges } = buildMockGraph(topic);
      const node = nodes.find((n) => n.id === entityId);
      if (!node) throw new ApiError('validation_error', `节点 ${entityId} 不存在`, false);
      const neighbors = edges
        .filter((e) => e.source === entityId || e.target === entityId)
        .map((e) => {
          const otherId = e.source === entityId ? e.target : e.source;
          const other = nodes.find((n) => n.id === otherId);
          return {
            entity_id: otherId,
            entity_type: other?.type || '',
            weight: e.weight,
            keywords: 'relates_to, part_of',
            description: `${entityId} 与 ${otherId} 之间的关联说明（mock）。`,
            neighbor_preview: `${otherId} 是一个 ${other?.type || 'concept'} 类型实体（mock 摘要）。`,
          };
        });
      return okEnvelope<GraphNodeDetailResponse>({
        topic,
        entity_id: node.id,
        entity_type: node.type,
        degree: node.degree,
        description: `${node.id} 的完整描述段落（mock 数据）。这是一段用于演示懒加载详情面板的较长文本，实际环境下来自 GraphML 的 description 字段。`,
        source_id: 'mock-chunk-000',
        file_path: '00_总览.md',
        edges: neighbors,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/knowledge-bases\/[^/]+\/vectors$/.test(p),
    handler: (p) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      const kb = getKbDetail(topic);
      return okEnvelope<VectorStatsResponse>({
        topic,
        chunks: kb.vector_stats?.chunks || 0,
        entities: kb.vector_stats?.entities || 0,
        relationships: kb.vector_stats?.relationships || 0,
        embedding_model: kb.embedding_model,
        embedding_dim: kb.embedding_dim,
        chunk_size: 1200,
        chunk_overlap_size: 100,
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/knowledge-bases\/[^/]+\/index\/jobs$/.test(p),
    handler: (p, body) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      const jobId = newJobId('knowledge_index');
      const job: JobSummary = {
        job_id: jobId,
        job_type: 'knowledge_index',
        status: 'running',
        created_at: nowText(),
        updated_at: nowText(),
        progress: { current: 0, total: 3, message: '准备构建索引' },
        input: { topic, ...(body || {}) },
        result: null,
        error: null,
        artifacts: [`lightrag_workspaces/${topic}/index_metadata.json`],
      };
      jobs[jobId] = job;
      setTimeout(() => {
        job.progress = { current: 1, total: 3, message: '生成向量 chunks' };
        job.updated_at = nowText();
      }, 1500);
      setTimeout(() => {
        job.progress = { current: 2, total: 3, message: '构建实体关系图谱' };
        job.updated_at = nowText();
      }, 3000);
      setTimeout(() => {
        job.status = 'succeeded';
        job.progress = { current: 3, total: 3, message: '索引构建完成' };
        job.updated_at = nowText();
        job.result = {
          status: 'indexed',
          topic,
          backend: 'lightrag',
          workspace: `lightrag_workspaces/${topic}`,
          indexed_documents: getKbDocumentList(topic).documents.length,
          metadata: {},
        };
        const kb = knowledgeBases.find((k) => k.topic === topic);
        if (kb) {
          kb.index_status = 'ready';
          kb.graph_stats = { nodes: 111, edges: 159 };
          kb.vector_stats = { chunks: 13, entities: 111, relationships: 159 };
        }
      }, 4500);
      return acceptedEnvelope<JobCreatedResponse>({ job_id: jobId, job_type: 'knowledge_index', status: 'running' });
    },
  },
  {
    match: (p, m) => m === 'DELETE' && /^\/api\/v1\/knowledge-bases\/[^/]+\/index$/.test(p),
    handler: (p) => {
      const topic = decodeURIComponent(p.split('/')[4]);
      const kb = knowledgeBases.find((k) => k.topic === topic);
      if (kb) {
        kb.index_status = 'missing';
        kb.graph_stats = undefined;
        kb.vector_stats = undefined;
      }
      return okEnvelope<DeleteIndexResponse>({
        status: 'deleted',
        topic,
        deleted_path: `lightrag_workspaces/${topic}`,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/jobs',
    handler: (_p, _b, params) => {
      let list = Object.values(jobs);
      if (params?.job_type) list = list.filter((j) => j.job_type === params.job_type);
      if (params?.research_type === 'single_url') {
        list = list.filter((job) => job.result?.pipeline_mode === 'single_url');
      } else if (params?.research_type === 'deep_research') {
        list = list.filter((job) => job.result?.pipeline_mode !== 'single_url');
      }
      if (params?.status) list = list.filter((j) => j.status === params.status);
      if (!params?.status && params?.statuses) {
        const statuses = String(params.statuses).split(',');
        list = list.filter((job) => statuses.includes(job.status));
      }
      if (params?.query) {
        const term = String(params.query).toLowerCase();
        list = list.filter((job) => JSON.stringify(job).toLowerCase().includes(term));
      }
      if (params?.archived !== undefined) {
        list = list.filter((j) => Boolean(j.archived) === Boolean(params.archived));
      }
      list = [...list].sort((a, b) => {
        const aWaiting = a.status === 'waiting_approval';
        const bWaiting = b.status === 'waiting_approval';
        if (aWaiting !== bWaiting) return aWaiting ? -1 : 1;
        return String(b.updated_at).localeCompare(String(a.updated_at));
      });
      const limit = Number(params?.limit || 50);
      const offset = Number(params?.cursor || params?.offset || 0);
      const page = list.slice(offset, offset + limit);
      return okEnvelope({
        jobs: page,
        next_cursor: offset + limit < list.length ? String(offset + limit) : null,
        total: list.length,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/jobs/stats',
    handler: (_p, _b, params) => {
      let list = Object.values(jobs);
      if (params?.archived !== undefined) {
        list = list.filter((job) => Boolean(job.archived) === Boolean(params.archived));
      }
      const byStatus = list.reduce<Record<string, number>>((counts, job) => {
        counts[job.status] = (counts[job.status] || 0) + 1;
        return counts;
      }, {});
      const byType = list.reduce<Record<string, number>>((counts, job) => {
        counts[job.job_type] = (counts[job.job_type] || 0) + 1;
        return counts;
      }, {});
      return okEnvelope({
        total: list.length,
        attention: byStatus.waiting_approval || 0,
        active: (byStatus.queued || 0) + (byStatus.running || 0),
        failed: (byStatus.failed || 0) + (byStatus.interrupted || 0),
        by_status: byStatus,
        by_type: byType,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/jobs\/[^/]+$/.test(p),
    handler: (p) => {
      const jobId = p.split('/')[4];
      const job = jobs[jobId];
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      return okEnvelope(job);
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/jobs\/[^/]+\/cancel$/.test(p),
    handler: (p) => {
      const jobId = p.split('/')[4];
      const job = jobs[jobId];
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      job.status = 'cancelled';
      job.error = { code: 'cancelled', message: '用户取消', retryable: false };
      job.updated_at = nowText();
      return okEnvelope({ job_id: jobId, status: 'cancelled' });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/jobs\/[^/]+\/archive$/.test(p),
    handler: (p, body) => {
      const jobId = p.split('/')[4];
      const job = jobs[jobId];
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      const { archived } = (body || {}) as { archived?: boolean };
      job.archived = Boolean(archived);
      job.updated_at = nowText();
      return okEnvelope({ job_id: jobId, archived: job.archived });
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/evaluations/knowledge-qa/jobs',
    handler: (_p, body) => {
      const { topic } = (body || {}) as EvaluationJobRequest;
      const jobId = newJobId('knowledge_qa_evaluation');
      const runId = newRunId();
      const reportId = `knowledge-qa-${topic || MOCK_TOPIC}-v0.1-${nowText().replace(/[-: ]/g, '')}`;
      const job: EvaluationJob = {
        job_id: jobId,
        job_type: 'knowledge_qa_evaluation',
        status: 'running',
        created_at: nowText(),
        updated_at: nowText(),
        progress: { current: 0, total: 20, message: '开始评估' },
        input: body as Record<string, unknown>,
        result: null,
        error: null,
        artifacts: [`docs/evaluations/${reportId}.md`],
      };
      jobs[jobId] = job;
      let current = 0;
      const interval = setInterval(() => {
        current += 4;
        if (current >= 20) {
          clearInterval(interval);
          job.status = 'succeeded';
          job.progress = { current: 20, total: 20, message: '评估完成' };
          job.updated_at = nowText();
          job.result = {
            topic: topic || MOCK_TOPIC,
            summary: {
              total: 3,
              passed: 1,
              failed: 2,
              pass_rate: 33.3,
              average_score: 66,
              failed_checks: { citation_count: 1, evidence_alignment: 1, coverage_gap: 1 },
              categories: { retrieval: 1, citation: 1, answer_quality: 1 },
              failure_types: { citation_weak: 1, knowledge_gap: 1 },
            },
            report_path: `docs/evaluations/${reportId}.md`,
            results: evaluationCases,
          };
          reports.unshift({
            report_id: reportId,
            topic: topic || MOCK_TOPIC,
            created_at: nowText(),
            run_id: runId,
            content: `# 知识库问答质量评估报告\n\n- **Topic**: ${topic || MOCK_TOPIC}\n- **Total**: 3\n- **Passed**: 1\n- **Failed**: 2\n- **Pass Rate**: 33.3%\n\n## 建议\n\n将失败用例回流到 QA 页面复测。`,
            summary: job.result.summary,
            cases: evaluationCases,
            recommended_actions: ['将失败用例回流到 QA 页面复测。', '补充来源文档后重建索引。'],
          });
        } else {
          job.progress = { current, total: 20, message: `已评估 ${current}/20` };
          job.updated_at = nowText();
        }
      }, 600);
      return acceptedEnvelope<JobCreatedResponse>({
        job_id: jobId,
        job_type: 'knowledge_qa_evaluation',
        status: 'running',
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/evaluations\/jobs\/[^/]+$/.test(p),
    handler: (p) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId];
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      return okEnvelope(job as EvaluationJob);
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/evaluations/reports',
    handler: (_p, _b, params) => {
      const topic = params?.topic ? String(params.topic) : '';
      const limit = Number(params?.limit || 50);
      const filtered = reports.filter((r) => !topic || r.topic === topic).slice(0, limit);
      return okEnvelope<EvaluationReportListResponse>({
        reports: filtered.map((r) => ({
          report_id: r.report_id,
          topic: r.topic,
          path: `docs/evaluations/${r.report_id}.md`,
          created_at: r.created_at || '2026-06-12 09:07:24',
          run_id: r.run_id,
          summary: {
            pass_rate: r.summary?.pass_rate || 0,
            total: r.summary?.total,
            failed: r.summary?.failed,
          },
        })),
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/evaluations\/reports\/[^/]+$/.test(p),
    handler: (p) => {
      const reportId = p.split('/')[5];
      const report = reports.find((r) => r.report_id === reportId);
      if (!report) throw new ApiError('not_found', '报告不存在', false);
      return okEnvelope<EvaluationReportResponse>(report);
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/evaluations/research-probe/jobs',
    handler: (_p, body) => {
      const { case_ids } = (body || {}) as { case_ids?: string[] | null };
      const jobId = newJobId('research_probe_evaluation');
      const caseIds = case_ids?.length ? case_ids : mockEvalCases.research_probe.map((c) => c.id as string);
      const job: JobSummary = {
        job_id: jobId,
        job_type: 'research_probe_evaluation',
        status: 'running',
        created_at: nowText(),
        updated_at: nowText(),
        progress: { current: 0, total: caseIds.length, message: '开始调研评估' },
        steps: caseIds.map((id) => ({ id, title: id, status: 'pending' })),
        input: body as Record<string, unknown>,
        result: null,
        error: null,
        artifacts: [],
      };
      jobs[jobId] = job;
      setTimeout(() => {
        job.status = 'succeeded';
        job.progress = { current: caseIds.length, total: caseIds.length, message: '评估完成' };
        job.steps = caseIds.map((id) => ({ id, title: id, status: 'succeeded' }));
        job.updated_at = nowText();
        job.result = {
          aggregate: { total_cases: caseIds.length, passed_cases: caseIds.length, failed_cases: 0, pass_rate: 1 },
          rows: caseIds.map((id) => ({ case_id: id, failed: false })),
          report_paths: {},
        };
      }, 2000);
      return acceptedEnvelope<JobCreatedResponse>({ job_id: jobId, job_type: 'research_probe_evaluation', status: 'running' });
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/evaluations/single-url-brief/jobs',
    handler: (_p, body) => {
      const { case_ids } = (body || {}) as { case_ids?: string[] | null };
      const jobId = newJobId('single_url_brief_evaluation');
      const caseIds = case_ids?.length ? case_ids : mockEvalCases.single_url_brief.map((c) => c.id as string);
      const job: JobSummary = {
        job_id: jobId,
        job_type: 'single_url_brief_evaluation',
        status: 'running',
        created_at: nowText(),
        updated_at: nowText(),
        progress: { current: 0, total: caseIds.length, message: '开始单URL简报评估' },
        steps: caseIds.map((id) => ({ id, title: id, status: 'pending' })),
        input: body as Record<string, unknown>,
        result: null,
        error: null,
        artifacts: [],
      };
      jobs[jobId] = job;
      setTimeout(() => {
        job.status = 'succeeded';
        job.progress = { current: caseIds.length, total: caseIds.length, message: '评估完成' };
        job.steps = caseIds.map((id) => ({ id, title: id, status: 'succeeded' }));
        job.updated_at = nowText();
        job.result = {
          total_cases: caseIds.length,
          passed_cases: caseIds.length,
          rows: caseIds.map((id) => ({ id, passed: true })),
          report_paths: {},
        };
      }, 2000);
      return acceptedEnvelope<JobCreatedResponse>({ job_id: jobId, job_type: 'single_url_brief_evaluation', status: 'running' });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/evaluations/cases',
    handler: (_p, _b, params) => {
      const evalType = String(params?.eval_type || 'knowledge_qa') as keyof typeof mockEvalCases;
      const topic = params?.topic ? String(params.topic) : undefined;
      const cases = evalType === 'knowledge_qa' ? mockEvalCases.knowledge_qa[topic || 'default'] || [] : mockEvalCases[evalType] || [];
      return okEnvelope({
        eval_type: evalType,
        topic: topic || null,
        scopes: evalType === 'knowledge_qa' ? Object.keys(mockEvalCases.knowledge_qa).filter((k) => k !== 'default') : [],
        cases,
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/evaluations/cases',
    handler: (_p, body) => {
      const { eval_type, topic, case: evalCase } = (body || {}) as {
        eval_type: keyof typeof mockEvalCases;
        topic?: string;
        case: Record<string, unknown> & { id: string };
      };
      if (!evalCase?.id) throw new ApiError('validation_error', 'case must include a non-empty id field', false);
      if (eval_type === 'knowledge_qa') {
        const key = topic || 'default';
        const bucket = mockEvalCases.knowledge_qa[key] || [];
        mockEvalCases.knowledge_qa[key] = [...bucket.filter((c) => c.id !== evalCase.id), evalCase];
      } else {
        const bucket = mockEvalCases[eval_type] || [];
        mockEvalCases[eval_type] = [...bucket.filter((c) => c.id !== evalCase.id), evalCase];
      }
      return okEnvelope(evalCase);
    },
  },
  {
    match: (p, m) => m === 'DELETE' && /^\/api\/v1\/evaluations\/cases\/[^/?]+/.test(p),
    handler: (p, _b, params) => {
      const caseId = decodeURIComponent(p.split('/')[5].split('?')[0]);
      const evalType = String(params?.eval_type || 'knowledge_qa') as keyof typeof mockEvalCases;
      const topic = params?.topic ? String(params.topic) : undefined;
      if (evalType === 'knowledge_qa') {
        const key = topic || 'default';
        mockEvalCases.knowledge_qa[key] = (mockEvalCases.knowledge_qa[key] || []).filter((c) => c.id !== caseId);
      } else {
        mockEvalCases[evalType] = (mockEvalCases[evalType] || []).filter((c) => c.id !== caseId);
      }
      return okEnvelope({ deleted: caseId });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/observability/events',
    handler: (_p, _b, params) => {
      const events: ObservabilityEvent[] = [
        {
          timestamp: nowText(),
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          run_id: 'run_qa_demo_001',
          request_id: 'req_qa_demo_001',
          event_type: 'knowledge_qa.answered',
          level: 'info',
          component: 'knowledge_qa',
          message: 'Knowledge QA answered with LightRAG citations.',
          data: { topic: MOCK_TOPIC, backend: 'lightrag', mode: 'mix' },
        },
        {
          timestamp: nowText(),
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          run_id: 'run_eval_demo_090724',
          report_id: reports[0]?.report_id,
          event_type: 'evaluation.completed',
          level: 'warning',
          component: 'evaluation',
          message: 'Evaluation completed with failed cases.',
          data: { topic: MOCK_TOPIC, failed: reports[0]?.summary?.failed || 0 },
        },
        {
          timestamp: nowText(),
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          run_id: 'run_index_demo_001',
          job_id: Object.values(jobs).find((j) => j.job_type === 'knowledge_index')?.job_id,
          event_type: 'index.build.completed',
          level: 'info',
          component: 'index',
          message: 'LightRAG index build completed.',
          data: { topic: MOCK_TOPIC },
        },
        {
          timestamp: nowText(),
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          run_id: 'run_model_demo_001',
          request_id: 'req_model_demo_001',
          event_type: 'model.check.completed',
          level: 'info',
          component: 'models',
          message: 'Model and embedding connectivity check completed.',
          data: { chat_model: 'gpt-4.1-mini', embedding: 'BAAI/bge-large-zh-v1.5' },
        },
      ];
      const filtered = events.filter((event) => {
        if (params?.session_id && event.session_id !== params.session_id) return false;
        if (params?.thread_id && event.thread_id !== params.thread_id) return false;
        if (params?.run_id && event.run_id !== params.run_id) return false;
        if (params?.job_id && event.job_id !== params.job_id) return false;
        if (params?.request_id && event.request_id !== params.request_id) return false;
        if (params?.event_type && event.event_type !== params.event_type) return false;
        if (params?.level && event.level !== params.level) return false;
        return true;
      });
      const limit = Number(params?.limit || 100);
      return okEnvelope({
        events: filtered.slice(0, limit),
        next_cursor: null,
        has_more: false,
        limit,
        scanned: events.length,
        days: Number(params?.days || 7),
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/observability/runs',
    handler: () => {
      const runs = [
        {
          id: 'run_qa_demo_001',
          title: `对话轮次 · thread_de`,
          status: 'completed',
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          job_id: null,
          started_at: nowText(),
          finished_at: nowText(),
          duration_ms: 820,
          event_count: 1,
          error_count: 0,
          components: ['knowledge_qa'],
        },
        {
          id: 'run_eval_demo_090724',
          title: '评估任务执行',
          status: 'error',
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          job_id: null,
          started_at: nowText(),
          finished_at: nowText(),
          duration_ms: 4200,
          event_count: 1,
          error_count: 0,
          components: ['evaluation'],
        },
        {
          id: 'run_index_demo_001',
          title: '调研任务执行 · thread_de',
          status: 'completed',
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          job_id: Object.values(jobs).find((j) => j.job_type === 'knowledge_index')?.job_id || null,
          started_at: nowText(),
          finished_at: nowText(),
          duration_ms: 15400,
          event_count: 1,
          error_count: 0,
          components: ['index'],
        },
        {
          id: 'run_model_demo_001',
          title: '模型连通性检查',
          status: 'completed',
          session_id: 'thread_demo',
          thread_id: 'thread_demo',
          job_id: null,
          started_at: nowText(),
          finished_at: nowText(),
          duration_ms: 300,
          event_count: 1,
          error_count: 0,
          components: ['models'],
        },
      ];
      return okEnvelope({ runs, days: 7, total_runs: runs.length, truncated: false });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/observability\/runs\/[^/]+$/.test(p),
    handler: (p) => {
      const runId = p.split('/')[5];
      return okEnvelope({
        id: runId,
        run_id: runId,
        title: `run ${runId}`,
        status: 'completed',
        session_id: 'thread_demo',
        thread_id: 'thread_demo',
        job_id: null,
        started_at: nowText(),
        finished_at: nowText(),
        duration_ms: 820,
        event_count: 0,
        error_count: 0,
        components: [],
        events: [],
        tool_calls: [],
        artifacts: [],
        errors: [],
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/observability\/jobs\/[^/]+\/trace$/.test(p),
    handler: (p) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId];
      const topic = (job?.input as { topic?: string })?.topic;
      return okEnvelope({
        id: jobId,
        job_id: jobId,
        title: topic ? `调研任务 · ${topic}` : `任务 ${jobId}`,
        status: job?.status === 'failed' ? 'error' : job?.status === 'succeeded' ? 'completed' : 'running',
        session_id: null,
        thread_id: null,
        started_at: job?.created_at || nowText(),
        finished_at: job?.updated_at || nowText(),
        duration_ms: null,
        event_count: 0,
        error_count: 0,
        components: [],
        run_ids: [],
        events: [],
        errors: [],
        job_type: job?.job_type || null,
        job_status: job?.status || null,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/models/config',
    handler: () =>
      okEnvelope<ModelConfigResponse>({
        chat_model: { model: 'gpt-4.1-mini', base_url_configured: true, api_key_configured: true },
        recall_model: { model: 'gpt-4.1-mini', base_url_configured: true, api_key_configured: true },
        embedding: {
          model: 'BAAI/bge-large-zh-v1.5',
          dim: 1024,
          base_url_configured: true,
          api_key_configured: true,
        },
        dream_model: { model: 'gpt-4.1-mini', base_url_configured: true, api_key_configured: true },
        research_evidence_model: mockRoleConfig('evidence'),
        research_writer_model: mockRoleConfig('writer'),
      }),
  },
  {
    match: (p, m) => m === 'PUT' && p === '/api/v1/models/config',
    handler: () =>
      okEnvelope({
        status: 'saved',
        restart_required: false,
        updated_fields: ['MODEL_NAME', 'RECALL_MODEL_NAME', 'EMBEDDING_MODEL', 'EMBEDDING_DIM'],
        config_path: '.env',
        message: '配置已保存并生效。',
      }),
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/models/check',
    handler: (_p, body) => {
      const { targets } = (body || {}) as { targets?: string[] };
      const allTargets = targets || ['chat_model', 'embedding'];
      const multiKeyTargets = new Set(['research_evidence_model', 'research_writer_model']);
      return okEnvelope<ModelCheckResponse>({
        results: allTargets.map((t) =>
          multiKeyTargets.has(t)
            ? {
                target: t,
                status: 'partial',
                latency_ms: 700 + Math.floor(Math.random() * 500),
                message: `${mockRoleConfig(t === 'research_writer_model' ? 'writer' : 'evidence').key_count - 1}/${mockRoleConfig(t === 'research_writer_model' ? 'writer' : 'evidence').key_count} keys ok`,
                keys: mockRoleConfig(t === 'research_writer_model' ? 'writer' : 'evidence').keys.map((key, index) => ({
                  key_index: index,
                  key_id: key.id,
                  masked: key.masked,
                  status: index === 4 ? 'rate_limited' : 'ok',
                  latency_ms: 650 + index * 35,
                  message: index === 4 ? 'rate limit reached' : 'real completion request succeeded',
                })),
              }
            : {
                target: t,
                status: 'ok',
                latency_ms: 700 + Math.floor(Math.random() * 500),
                message: 'real completion request succeeded',
              },
        ),
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/models\/research-roles\/(evidence|writer)\/keys$/.test(p),
    handler: (p, body) => {
      const role = p.split('/')[5] as ResearchRole;
      const apiKey = String((body as { api_key?: string })?.api_key || '').trim();
      const key: ResearchRoleKey = {
        id: `mock-key-${Date.now().toString(36)}`,
        masked: mockMaskKey(apiKey),
        source: 'dedicated',
        deletable: true,
        shared: false,
      };
      if (role === 'writer') mockWriterKeys = [...mockWriterKeys, key];
      else mockEvidenceKeys.push(key);
      return okEnvelope<ResearchRoleKeyMutationResponse>({ status: 'added', key, config: mockRoleConfig(role) });
    },
  },
  {
    match: (p, m) => m === 'DELETE' && /^\/api\/v1\/models\/research-roles\/(evidence|writer)\/keys\/[^/]+$/.test(p),
    handler: (p) => {
      const parts = p.split('/');
      const role = parts[5] as ResearchRole;
      const keyId = parts[7];
      if (role === 'writer') mockWriterKeys = mockWriterKeys.filter((key) => key.id !== keyId);
      else {
        const index = mockEvidenceKeys.findIndex((key) => key.id === keyId);
        if (index >= 0) mockEvidenceKeys.splice(index, 1);
      }
      return okEnvelope<ResearchRoleKeyMutationResponse>({ status: 'deleted', deleted_key_id: keyId, config: mockRoleConfig(role) });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/models\/research-roles\/(evidence|writer)\/keys\/check$/.test(p),
    handler: (_p, body) => {
      const apiKey = String((body as { api_key?: string })?.api_key || '').trim();
      return okEnvelope<ModelCheckKeyResult>({
        key_id: `candidate-${Date.now().toString(36)}`,
        masked: mockMaskKey(apiKey),
        status: 'ok',
        latency_ms: 680,
        message: 'real completion request succeeded',
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/models\/research-roles\/(evidence|writer)\/keys\/[^/]+\/check$/.test(p),
    handler: (p) => {
      const parts = p.split('/');
      const role = parts[5] as ResearchRole;
      const keyId = parts[7];
      const key = mockRoleConfig(role).keys.find((item) => item.id === keyId);
      return okEnvelope<ModelCheckKeyResult>({
        key_id: keyId,
        masked: key?.masked || 'unknown',
        status: 'ok',
        latency_ms: 720,
        message: 'real completion request succeeded',
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/providers',
    handler: () =>
      okEnvelope<ProvidersResponse>({
        providers: [
          { provider: 'openai_compatible', enabled: true },
          { provider: 'siliconflow_embedding', enabled: true },
          { provider: 'lightrag_local', enabled: true },
        ],
      }),
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/research/planning/clarify',
    handler: (_p, body) => {
      const input = (body || {}) as ResearchPlanningClarifyRequest;
      const ambiguous = /同名|歧义|具体指的是哪个/.test(input.user_request)
        && !input.user_request.includes('用户补充说明：');
      const response: ResearchPlanningClarifyResponse = ambiguous
        ? {
            status: 'ambiguous',
            ambiguities: ['当前主题可能对应多个同名概念或产品'],
            suggested_question: `“${input.user_request}”具体指的是哪个概念或产品？`,
            recon_token: `mock_recon_${Date.now()}`,
            recommended_strategy: 'standard',
          }
        : {
            status: 'clear',
            entity_type: 'topic',
            recon_token: `mock_recon_${Date.now()}`,
            disambiguation_terms: [],
            recommended_strategy: /对比|比较|横向|纵向|演化/.test(input.user_request) ? 'hv_analysis' : 'standard',
          };
      return okEnvelope<ResearchPlanningClarifyResponse>(response);
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/research/planning/draft',
    handler: (_p, body) => {
      const input = (body || {}) as ResearchPlanningDraftRequest;
      const makeSection = (title: string, index: number): ResearchPlanSection => ({
        id: `sec_${String(index + 1).padStart(2, '0')}`,
        title,
        intent: `围绕“${input.user_request}”回答这一部分的核心问题。`,
        questions: title.includes('总览')
          ? []
          : [{
              id: `sec_${String(index + 1).padStart(2, '0')}_q1`,
              text: `${title.replace(/^\d+_/, '')}需要回答哪些核心问题？`,
              purpose: '形成可执行、可核验的研究范围。',
              success_criteria: ['说明核心事实与工作机制', '提供可追溯来源'],
            }],
      });
      let sections: ResearchPlanSection[];
      if (input.adjustment) {
        const keep = input.adjustment.keep || [];
        const generated = [...(input.adjustment.redo || []), ...(input.adjustment.add || [])]
          .map((section, index) => makeSection(section.title, keep.length + index));
        sections = [...keep, ...generated];
      } else {
        sections = [
          makeSection('00_总览', 0),
          makeSection('01_核心机制', 1),
          makeSection('02_实践与边界', 2),
        ];
      }
      const questionCount = sections.reduce((total, section) => total + (section.questions?.length || 0), 0);
      const strategy = input.strategy === 'hv_analysis' ? 'hv_analysis' : 'standard';
      const confirmedPlan: ConfirmedResearchPlan = {
        version: 'v1',
        user_request: input.user_request,
        topic: input.topic || '',
        strategy,
        research_type: strategy === 'hv_analysis' ? 'comparison' : 'concept_explainer',
        subjects: [],
        comparison_axes: [],
        sections,
      };
      return okEnvelope<ResearchPlanningDraftResponse>({
        status: 'ready',
        planned_sections: sections,
        confirmed_plan: confirmedPlan,
        plan_id: `mock_plan_${Date.now()}`,
        plan_token: `mock_token_${Date.now()}`,
        strategy,
        plan_markdown: '',
        section_count: sections.length,
        question_count: questionCount,
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/research/jobs',
    handler: (_p, body) => {
      const input = (body || {}) as ResearchJobCreateRequest;
      const jobId = newJobId('research');
      const result = buildMockResearchResult(input);
      const job: ResearchJob = {
        job_id: jobId,
        job_type: 'research',
        status: 'waiting_approval',
        created_at: nowText(),
        updated_at: nowText(),
        progress: { current: 1, total: 1, message: 'waiting_approval' },
        current_step: 'approval',
        steps: [
          { id: 'plan', title: '规划调研', status: 'succeeded', summary: '研究框架已生成' },
          { id: 'query_plan', title: '生成查询计划', status: 'succeeded', summary: '生成 9 个问题级查询' },
          { id: 'search', title: '检索来源', status: 'succeeded', summary: '发现 9 个候选来源' },
          { id: 'fetch', title: '抓取材料', status: 'succeeded', summary: '抓取并保留 3 份网页材料' },
          { id: 'material_pool', title: '建立素材池', status: 'succeeded', summary: '建立 3 份清洗后材料' },
          { id: 'chunk_select', title: '筛选关键片段', status: 'succeeded', summary: '选中 9 个问题相关片段' },
          { id: 'evidence', title: '整理证据', status: 'succeeded', summary: '整理 9 条可引用证据' },
          { id: 'coverage', title: '覆盖度检查', status: 'succeeded', summary: '多主题矩阵覆盖完整，可以写入' },
          { id: 'draft', title: '生成文档', status: 'succeeded', summary: '生成 3 篇待审批文档' },
          { id: 'validation', title: '证据验收', status: 'succeeded', summary: '证据引用全部匹配' },
          { id: 'approval', title: '等待审批', status: 'waiting_approval', summary: '调研产物已生成，等待审批' },
        ],
        input: input as unknown as Record<string, unknown>,
        result,
        error: null,
        artifacts: ['coverage_report', 'document_draft'],
      };
      jobs[jobId] = job;
      return acceptedEnvelope<JobCreatedResponse>({ job_id: jobId, job_type: 'research', status: 'waiting_approval' });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/research\/jobs\/[^/]+$/.test(p),
    handler: (p) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId] as ResearchJob | undefined;
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      maybeCompleteMockResearchFollowup(job);
      return okEnvelope(job);
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/research\/jobs\/[^/]+\/artifacts$/.test(p),
    handler: (p) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId] as ResearchJob | undefined;
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      maybeCompleteMockResearchFollowup(job);
      const sources = (job.result?.sources || []).map((source) => {
        if (typeof source === 'string') return source;
        return { title: String(source.title || 'source'), url: String(source.url || '') };
      });
      const evidencePack = (job.result?.evidence_pack || []).map((item) => ({
        fact: String(item.fact || ''),
        sources: Array.isArray(item.sources) ? item.sources : [],
      }));
      return okEnvelope<ResearchArtifactsResponse>({
        sources,
        evidence_pack: evidencePack,
        materials: job.result?.materials,
        selected_chunks: job.result?.selected_chunks,
        evidence_map: job.result?.evidence_map,
        validation: job.result?.validation,
        research_plan: job.result?.research_plan,
        query_plan: job.result?.query_plan,
        coverage: job.result?.coverage!,
        documents: job.result?.documents || [],
        suggested_topic: job.result?.suggested_topic || '',
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/research\/jobs\/[^/]+\/approve$/.test(p),
    handler: (p, body) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId] as ResearchJob | undefined;
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      const payload = (body || {}) as ResearchApproveRequest;
      const documents = payload.documents || job.result?.documents || [];
      if (!documents.length) throw new ApiError('validation_error', 'research job has no documents to approve', false);
      job.status = 'succeeded';
      job.updated_at = nowText();
      return okEnvelope<ResearchApproveResponse>({
        topic: payload.topic,
        document_count: documents.length,
        paths: documents.map((d) => `knowledge_base/${payload.topic}/${d.title}.md`),
      });
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/research\/jobs\/[^/]+\/followup$/.test(p),
    handler: (p, body) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId] as ResearchJob | undefined;
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      return okEnvelope<ResearchFollowupResponse>(
        resolveMockResearchFollowup(job, (body || {}) as ResearchFollowupRequest)
      );
    },
  },
  {
    match: (p, m) => m === 'POST' && /^\/api\/v1\/research\/jobs\/[^/]+\/revise-sections$/.test(p),
    handler: (p, body) => {
      const jobId = p.split('/')[5];
      const job = jobs[jobId] as ResearchJob | undefined;
      if (!job) throw new ApiError('job_not_found', '任务不存在或已清理', false);
      return okEnvelope<ResearchSectionRevisionResponse>(
        resolveMockResearchSectionRevision(job, (body || {}) as ResearchSectionRevisionRequest)
      );
    },
  },
  {
    match: (p, m) => m === 'GET' && p === '/api/v1/conversations',
    handler: () =>
      okEnvelope({
        conversations: [
          {
            thread_id: 'thread_demo_001',
            title: 'Deep Research 功能讨论',
            created_at: '2026-06-24 10:00:00',
            updated_at: '2026-06-24 10:15:00',
            message_count: 4,
          },
          {
            thread_id: 'thread_demo_002',
            title: '知识库问答测试',
            created_at: '2026-06-23 14:00:00',
            updated_at: '2026-06-23 14:30:00',
            message_count: 6,
          },
        ],
        total: 2,
      }),
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/conversations\/[^/]+$/.test(p) && !p.endsWith('/messages'),
    handler: (p) => {
      const threadId = p.split('/').pop();
      return okEnvelope({
        thread_id: threadId,
        title: 'Mock 对话',
        created_at: '2026-06-24 10:00:00',
        updated_at: '2026-06-24 10:15:00',
        message_count: 2,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/conversations\/[^/]+\/messages$/.test(p),
    handler: (p) => {
      const threadId = p.split('/')[5];
      return okEnvelope({
        thread_id: threadId,
        conversation: { thread_id: threadId, title: 'Mock 对话', created_at: '2026-06-24 10:00:00', updated_at: '2026-06-24 10:15:00', message_count: 4 },
        messages: [
          { id: 'msg_001', thread_id: threadId, role: 'user', content: '你好，请介绍一下 Deep Research。', created_at: '2026-06-24 10:00:00' },
          { id: 'msg_002', thread_id: threadId, role: 'assistant', content: 'Deep Research 是一个本地 AI Agent 助手，支持知识库管理、调研、问答等功能。', created_at: '2026-06-24 10:00:30' },
          { id: 'msg_003', thread_id: threadId, role: 'user', content: '它的知识库是怎么工作的？', created_at: '2026-06-24 10:10:00' },
          { id: 'msg_004', thread_id: threadId, role: 'assistant', content: '知识库使用 LightRAG 进行图谱与向量混合检索，支持 auto/lightrag/local 三种后端。', created_at: '2026-06-24 10:10:30' },
        ],
      });
    },
  },
  {
    match: (p, m) => m === 'DELETE' && /^\/api\/v1\/conversations\/[^/]+$/.test(p),
    handler: (p) => {
      const threadId = p.split('/').pop();
      return okEnvelope({ thread_id: threadId, deleted: true });
    },
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/chat',
    handler: (_p, body) => {
      const { message, channel_id } = (body || {}) as { message?: string; channel_id?: string };
      if (channel_id) {
        const channel = getChannelOrThrow(channel_id);
        const mode = channel.mode || 'main';
        pushChannelMessage(channel_id, mode, 'user', message || '');
        let assistantText = `（mock 主管理）已收到「${message || ''}」。`;
        if (mode === 'kb_qa') {
          assistantText = `（mock QA）基于 ${channel.active_kb || '未选择知识库'} 回答：${message || ''}`;
        }
        if ((message || '').includes('退出知识库')) {
          channels[channel_id] = { ...channel, mode: 'main', updated_at: nowText() };
          assistantText = '已回到日常主管理模式。';
        }
        if ((message || '').includes('知识库问答模式')) {
          channels[channel_id] = { ...channel, mode: 'kb_qa', updated_at: nowText() };
          assistantText = `已切换到知识库问答模式，当前知识库：${channel.active_kb || '未选择'}。`;
        }
        pushChannelMessage(channel_id, channels[channel_id].mode, 'assistant', assistantText);
        return okEnvelope<ChannelChatResponse>({
          status: 'completed',
          state: 'completed',
          channel_id,
          mode: channels[channel_id].mode,
          active_kb: channels[channel_id].active_kb,
          thread_id: channelThreadId(channel_id, channels[channel_id].mode),
          run_id: newRunId(),
          assistant_text: assistantText,
          channel: channels[channel_id],
          qa: channels[channel_id].mode === 'kb_qa' ? {
            channel_id,
            mode: 'kb_qa',
            active_kb: channels[channel_id].active_kb,
            answer: assistantText,
            citations: [],
            gaps: [],
          } : undefined,
        });
      }
      return okEnvelope({
        status: 'completed',
        thread_id: 'thread_demo',
        run_id: newRunId(),
        assistant_text: `（mock 回答）关于「${message || ''}」的问题，请使用 Knowledge QA 页面基于知识库进行可信问答。`,
      });
    },
  },
  {
    match: (p, m) => m === 'GET' && /^\/api\/v1\/chat\/pending\/[^/]+$/.test(p),
    handler: () =>
      okEnvelope({
        status: 'not_found',
        thread_id: 'thread_demo',
        resume_supported: false,
      }),
  },
  {
    match: (p, m) => m === 'POST' && p === '/api/v1/chat/resume',
    handler: () =>
      okEnvelope({
        status: 'completed',
        thread_id: 'thread_demo',
        run_id: newRunId(),
        assistant_text: '（mock 恢复回答）',
      }),
  },
];

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .slice(0, 80);
}

export function createMockAdapter(): ApiAdapter {
  function unwrapEnvelope<T>(result: unknown): T {
    const envelope = result as {
      status?: string;
      request_id?: string;
      data?: unknown;
      error?: {
        code: string;
        message: string;
        retryable: boolean;
        details?: Record<string, unknown>;
      };
    };

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

    return result as T;
  }

  return {
    get: async <T>(path: string, params?: Record<string, unknown>) => {
      await sleep();
      const handler = mockHandlers.find((h) => h.match(path, 'GET'));
      if (!handler) throw new ApiError('not_found', `Mock GET handler not found: ${path}`, false);
      return unwrapEnvelope<T>(handler.handler(path, undefined, params));
    },
    post: async <T>(path: string, body?: unknown) => {
      await sleep();
      const handler = mockHandlers.find((h) => h.match(path, 'POST'));
      if (!handler) throw new ApiError('not_found', `Mock POST handler not found: ${path}`, false);
      return unwrapEnvelope<T>(handler.handler(path, body));
    },
    put: async <T>(path: string, body?: unknown) => {
      await sleep();
      const handler = mockHandlers.find((h) => h.match(path, 'PUT'));
      if (!handler) throw new ApiError('not_found', `Mock PUT handler not found: ${path}`, false);
      return unwrapEnvelope<T>(handler.handler(path, body));
    },
    delete: async <T>(path: string) => {
      await sleep();
      const handler = mockHandlers.find((h) => h.match(path, 'DELETE'));
      if (!handler) throw new ApiError('not_found', `Mock DELETE handler not found: ${path}`, false);
      return unwrapEnvelope<T>(handler.handler(path));
    },
    stream: async <T>(path: string, body: unknown, onDelta: (text: string) => void) => {
      const result = await (async () => {
        const handler = mockHandlers.find((h) => h.match(path.replace('/query/stream', '/query'), 'POST'));
        if (!handler) throw new ApiError('not_found', `Mock POST handler not found: ${path}`, false);
        return unwrapEnvelope<T>(handler.handler(path.replace('/query/stream', '/query'), body));
      })();
      const answer = String((result as any).answer || '');
      for (let i = 0; i < answer.length; i += 48) {
        onDelta(answer.slice(i, i + 48));
        await sleep(20);
      }
      return result;
    },
  };
}
