import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Circle,
  Clock3,
  ExternalLink,
  FileText,
  Filter,
  Globe2,
  ListChecks,
  Loader2,
  Play,
  Save,
  Search,
  ShieldAlert,
  ShieldCheck,
  Table2,
  X,
} from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { Button } from '@/components/controls/Button';
import { FilterSelect, FormField } from '@/components/controls/FormControls';
import { StatusBadge } from '@/components/data/StatusBadge';
import { MarkdownPreview } from '@/components/data/MarkdownPreview';
import { useClarifyResearchPlan, useConfirmResearchArchive, useCreateResearchJob, useDraftResearchPlan, useKnowledgeTree, useResearchArchivePreview, useResearchJob, useResearchArtifacts, useApproveResearch, useResearchFollowup, useResearchSectionRevision } from '@/hooks';
import { useJobStore, useUIStore } from '@/stores';
import { ConfirmedResearchPlan, HierarchicalKnowledgeBase, KnowledgePreview, ResearchArchivePreviewRequest, ResearchDepth, ResearchJobResult, ResearchPlanSection, ResearchPlanningAdjustment, ResearchStep, ResearchStepStatus, ResearchStrategy } from '@/types';
import { cn } from '@/utils/cn';
import { getBaseUrl } from '@/api/toggle';
import { SingleUrlBriefResultView } from './SingleUrlBriefResultView';
import { ResearchPlanReview } from './ResearchPlanReview';
import { buildClarifiedResearchRequest, buildConfirmedResearchJobRequest, clearResearchPlanningDraft, loadResearchPlanningDraft, saveResearchPlanningDraft } from './researchPlanning';
import { deriveResearchDetailTitle, deriveResearchSectionMetadata, findResearchSectionMeta, type ResearchSectionMeta } from './researchDetail';
import { buildQualitySummary, type QualitySummary } from './researchQualitySummary';
import { getResearchStepTargetId, RESEARCH_STEP_FALLBACK_TARGET } from './researchStepNavigation';
import { buildChunkLifecycleMap, CHUNK_LIFECYCLE_STAGES, type ChunkLifecycle } from './researchChunkLifecycle';
import { ResearchStudioTab, ResearchStudioTabs } from './history/ResearchStudioTabs';
import { ResearchHistoryPanel } from './history/ResearchHistoryPanel';
import { getResearchStudioSearchParams } from './history/researchHistory';
import '../knowledge-bases/knowledge-workspace.css';

type EvidenceFilterState = {
  subject: string;
  axis: string;
  usage: 'all' | 'used' | 'unused' | 'risk';
};

type PlanningStage = 'idle' | 'clarifying' | 'needs_clarification' | 'drafting' | 'reviewing' | 'revising' | 'error';

export function ResearchStudioPage() {
  const restoredPlanningDraft = useRef(loadResearchPlanningDraft()).current;
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState<ResearchStudioTab>(() => searchParams.get('job_id') ? 'detail' : 'create');
  const lastSyncedJobId = useRef(searchParams.get('job_id') || '');
  const [studioMode, setStudioMode] = useState<'question_loop' | 'single_url'>('question_loop');
  const [request, setRequest] = useState(restoredPlanningDraft?.request || '');
  const [url, setUrl] = useState('');
  const [depth, setDepth] = useState<ResearchDepth>(restoredPlanningDraft?.depth || 'normal');
  const [topic, setTopic] = useState(restoredPlanningDraft?.topic || '');
  const planningRequestVersion = useRef(0);
  const [planningStage, setPlanningStage] = useState<PlanningStage>(restoredPlanningDraft ? 'reviewing' : 'idle');
  const [planningBaseRequest, setPlanningBaseRequest] = useState('');
  const [resolvedRequest, setResolvedRequest] = useState(restoredPlanningDraft?.resolvedRequest || '');
  const [clarificationQuestion, setClarificationQuestion] = useState('');
  const [clarificationAnswer, setClarificationAnswer] = useState('');
  const [reconToken, setReconToken] = useState(restoredPlanningDraft?.reconToken || '');
  const [disambiguationTerms, setDisambiguationTerms] = useState<string[]>(restoredPlanningDraft?.disambiguationTerms || []);
  const [plannedSections, setPlannedSections] = useState<ResearchPlanSection[]>(restoredPlanningDraft?.plannedSections || []);
  const [confirmedPlan, setConfirmedPlan] = useState<ConfirmedResearchPlan | undefined>(restoredPlanningDraft?.confirmedPlan);
  const [planId, setPlanId] = useState(restoredPlanningDraft?.planId || '');
  const [planToken, setPlanToken] = useState(restoredPlanningDraft?.planToken || '');
  const [resolvedStrategy, setResolvedStrategy] = useState<ResearchStrategy>(restoredPlanningDraft?.strategy || 'standard');
  const [planningError, setPlanningError] = useState('');
  const [jobId, setJobId] = useState(searchParams.get('job_id') || '');
  const [selectedStepId, setSelectedStepId] = useState('');
  const manualStepSelectionRef = useRef(false);
  const [stepRailExpanded, setStepRailExpanded] = useState(false);
  const [evidenceFilters, setEvidenceFilters] = useState<EvidenceFilterState>({
    subject: 'all',
    axis: 'all',
    usage: 'all',
  });
  const [activeEvidenceId, setActiveEvidenceId] = useState('');
  const [activeIssueId, setActiveIssueId] = useState('');
  const [reviewDrawerOpen, setReviewDrawerOpen] = useState(false);
  const [selectedSectionIds, setSelectedSectionIds] = useState<string[]>([]);
  const [singleUrlSaveTopic, setSingleUrlSaveTopic] = useState('');
  const [revisionInstruction, setRevisionInstruction] = useState('');
  const [archiveDialogOpen, setArchiveDialogOpen] = useState(false);
  const [matrixGapsAcknowledged, setMatrixGapsAcknowledged] = useState(false);
  const addJob = useJobStore((s) => s.addJob);
  const setJobDrawerOpen = useUIStore((s) => s.setJobDrawerOpen);

  const create = useCreateResearchJob();
  const clarifyPlan = useClarifyResearchPlan();
  const draftPlan = useDraftResearchPlan();
  const job = useResearchJob(jobId);
  const artifacts = useResearchArtifacts(jobId);
  const approve = useApproveResearch();
  const archivePreview = useResearchArchivePreview();
  const archiveConfirm = useConfirmResearchArchive();
  const knowledgeTree = useKnowledgeTree();
  const followup = useResearchFollowup();
  const sectionRevision = useResearchSectionRevision();

  useEffect(() => {
    const nextJobId = searchParams.get('job_id');
    if (nextJobId) setJobId(nextJobId);
    else if (lastSyncedJobId.current) setJobId('');
    // A job_id identifies the dedicated detail view. Do not place a historical job
    // underneath the new-research form: direct links and global task links should
    // restore the same detail state as the history list.
    if (nextJobId && nextJobId !== lastSyncedJobId.current) {
      setActiveTab('detail');
    } else if (!nextJobId && lastSyncedJobId.current) {
      setActiveTab((current) => current === 'detail' ? 'create' : current);
    }
    lastSyncedJobId.current = nextJobId || '';
  }, [searchParams]);

  useEffect(() => {
    manualStepSelectionRef.current = false;
    setSelectedStepId('');
    setMatrixGapsAcknowledged(false);
  }, [jobId]);

  useEffect(() => {
    const current = job.data?.current_step;
    if (!manualStepSelectionRef.current && current) {
      setSelectedStepId(current);
    } else if (!manualStepSelectionRef.current && !selectedStepId && job.data?.steps?.length) {
      const active =
        job.data.steps.find((step) => step.status === 'running') ||
        [...job.data.steps].reverse().find((step) => step.status !== 'pending') ||
        job.data.steps[0];
      setSelectedStepId(active.id);
    }
    if (job.data?.status === 'running' || job.data?.status === 'queued') {
      setStepRailExpanded(true);
    }
  }, [job.data?.current_step, job.data?.steps, job.data?.status, selectedStepId]);

  const researchDocuments = job.data?.result?.documents || artifacts.data?.documents || [];
  const sources = job.data?.result?.sources || artifacts.data?.sources || [];
  const evidencePack = job.data?.result?.evidence_pack || artifacts.data?.evidence_pack || [];
  const researchPlan = job.data?.result?.research_plan;
  const queryPlan = job.data?.result?.query_plan;

  useEffect(() => {
    if (job.data?.status === 'waiting_approval' || researchDocuments.length > 0) {
      setJobDrawerOpen(false);
    }
  }, [job.data?.status, researchDocuments.length, setJobDrawerOpen]);

  const sectionRefs = useRef(new Map<string, HTMLDivElement | null>());
  const [autoExpandedSectionKey, setAutoExpandedSectionKey] = useState<string | null>(null);
  const [manuallyToggledSectionKeys, setManuallyToggledSectionKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!autoExpandedSectionKey && researchDocuments.length) {
      const first = researchDocuments[0];
      setAutoExpandedSectionKey(first.section_id || first.title);
    }
  }, [researchDocuments, autoExpandedSectionKey]);

  useEffect(() => {
    if (!researchDocuments.length) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting);
        if (!visible.length) return;
        const topMost = visible.reduce((a, b) => (a.boundingClientRect.top < b.boundingClientRect.top ? a : b));
        const key = (topMost.target as HTMLElement).dataset.sectionKey;
        if (key) setAutoExpandedSectionKey(key);
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    );
    sectionRefs.current.forEach((el) => {
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [researchDocuments]);

  const handleToggleEvidenceAudit = (sectionKey: string) => {
    setManuallyToggledSectionKeys((prev) => {
      const next = new Set(prev);
      if (next.has(sectionKey)) next.delete(sectionKey);
      else next.add(sectionKey);
      return next;
    });
  };
  const materials = job.data?.result?.materials || [];
  const selectedChunks = job.data?.result?.selected_chunks || [];
  const evidenceMap = job.data?.result?.evidence_map || [];
  const validation = job.data?.result?.validation;
  const coverage = job.data?.result?.coverage || artifacts.data?.coverage;
  const qualityGate = job.data?.result?.quality_gate;
  const matrixCoverage = job.data?.result?.matrix_coverage || coverage?.matrix_coverage;
  const matrixGaps = matrixCoverage?.gaps || [];
  const requiresMatrixGapAcknowledgement = Boolean(
    matrixCoverage?.enabled && matrixCoverage.requires_gap_acknowledgement,
  );
  const questionStates = job.data?.result?.question_states || [];
  const questionAnswerCounts = useMemo(() => {
    const values = questionStates.map(questionAnswerStatus);
    return {
      complete: values.filter((value) => value === 'complete').length,
      partial: values.filter((value) => value === 'partial').length,
      uncovered: values.filter((value) => value === 'uncovered').length,
    };
  }, [questionStates]);
  const displayedAnswerCounts = coverage?.answer_status_counts || questionAnswerCounts;
  const coverageReason = questionStates.length
    ? `${displayedAnswerCounts.complete ?? 0} 个研究问题完整覆盖，${displayedAnswerCounts.partial ?? 0} 个部分覆盖，${displayedAnswerCounts.uncovered ?? 0} 个未覆盖。`
    : coverage?.reason;
  const reportQuality = job.data?.result?.report_quality;
  const rejectedEvidence = job.data?.result?.rejected_evidence || [];
  const grounding = job.data?.result?.grounding;
  const sectionEvidenceDiagnostics =
    job.data?.result?.retrieval?.evidence_diagnostics ||
    job.data?.result?.retrieval?.diagnostics?.section_evidence_extraction;
  const suggestedTopic = topic || job.data?.result?.suggested_topic || artifacts.data?.suggested_topic || '';
  const researchTitle = job.data ? deriveResearchDetailTitle(job.data) : '';
  const sectionMetadata = useMemo(
    () => deriveResearchSectionMetadata(job.data?.result),
    [job.data?.result]
  );
  const coreSections = sectionMetadata.filter((section) => section.role === 'core');
  const emptyCoreSections = coreSections.filter(
    (section) => section.role === 'core' && section.status === 'empty'
  );
  const coreSectionCount = coreSections.length;
  const coreEmptySectionCount = emptyCoreSections.length;
  const readableQualityReason = emptyCoreSections.length
    ? `未通过：核心章节为空：${emptyCoreSections
      .map((section) => `${section.title || section.sectionId}${section.title && section.sectionId ? `（${section.sectionId}）` : ''}`)
      .join('；')}`
    : qualityGate?.reason;
  const hasDocuments = researchDocuments.length > 0;
  const qualityReady = qualityGate ? qualityGate.approval_ready : true;
  const canApprove = job.data?.status === 'waiting_approval'
    && hasDocuments
    && qualityReady
    && (!requiresMatrixGapAcknowledgement || matrixGapsAcknowledged);
  const steps = job.data?.steps || [];
  const matrixEvidence = useMemo(
    () => buildMatrixEvidence({ documents: researchDocuments, evidenceMap, selectedChunks, plan: researchPlan }),
    [researchDocuments, evidenceMap, selectedChunks, researchPlan]
  );
  const evidenceLedger = useMemo(
    () => buildEvidenceLedger({ documents: researchDocuments, evidenceMap }),
    [researchDocuments, evidenceMap]
  );
  const qualitySummary = useMemo(
    () =>
      buildQualitySummary({
        qualityGate,
        validation,
        coverage,
        matrixCoverage,
        matrix: matrixEvidence,
        documents: researchDocuments,
        sources,
        materials,
        evidenceMap,
        evidencePack,
        qualityReason: readableQualityReason,
      }),
    [qualityGate, validation, coverage, matrixCoverage, matrixEvidence, researchDocuments, sources, materials, evidenceMap, evidencePack, readableQualityReason]
  );
  const selectedStep = useMemo(
    () => steps.find((step) => step.id === selectedStepId) || steps.find((step) => step.status === 'running') || steps[0],
    [steps, selectedStepId]
  );
  const selectedStepTargetId = selectedStep ? getResearchStepTargetId(selectedStep.id) : '';
  const activeEvidence = useMemo(
    () => evidenceLedger.find((item) => item.id === activeEvidenceId),
    [activeEvidenceId, evidenceLedger]
  );
  const reviewModel = useMemo(
    () =>
      buildEvidenceReviewModel({
        quality: qualitySummary,
        validation,
        matrix: matrixEvidence,
        ledger: evidenceLedger,
        documents: researchDocuments,
      }),
    [qualitySummary, validation, matrixEvidence, evidenceLedger, researchDocuments]
  );
  const runningFollowupActions = job.data?.result?.followup_actions?.filter((action) => action.status === 'running') || [];
  const runningFollowupIssueIds = useMemo(
    () => new Set(runningFollowupActions.map((action) => String(action.issue_id || '')).filter(Boolean)),
    [runningFollowupActions]
  );
  const hasRunningFollowup = followup.isPending || runningFollowupActions.length > 0 || job.data?.progress?.message === 'followup running';

  const runningRevisionActions = job.data?.result?.revision_actions?.filter((action) => action.status === 'running') || [];
  const hasRunningRevision =
    sectionRevision.isPending || runningRevisionActions.length > 0 || job.data?.progress?.message === 'section revision running';
  // Manual section revision reuses revise_report_sections (backend/research/writer.py), which only
  // exists for question_loop jobs — legacy-pipeline reports don't carry report_blueprint/findings/
  // evidence_map in a shape it can consume, so the entry point is hidden rather than silently
  // producing a wrong result (see feishu-channel-router-legacy-pipeline-bug lesson).
  const canReviseSections =
    job.data?.result?.pipeline_mode === 'question_loop' &&
    job.data?.status === 'waiting_approval' &&
    job.data?.result?.capabilities?.section_revision !== false;
  const canRunTargetedFollowup = job.data?.result?.capabilities?.targeted_followup !== false;
  // 证据标注阅读版（VISION.md 第2节）：程序化渲染的静态 HTML，落在
  // data/research_artifacts/<job_id>/brief.html，后端直接按原始 HTML 返回（不是 JSON 包裹），
  // 所以这里只需要一个可以在新标签页打开的链接，不需要在前端另起一个页面渲染它。single_url_brief
  // 和 question_loop（审批通过后才生成，见 approve_research()/
  // question-loop-report-evidence-reader-html-v0.1.md）都可能有这份产物；两条链路共用同一个
  // 后端路由和文件名判断逻辑，前端这里只需要放开 pipeline_mode 的判断条件即可，不需要区分处理。
  const evidenceReaderUrl =
    (job.data?.result?.pipeline_mode === 'single_url' || job.data?.result?.pipeline_mode === 'question_loop') &&
    job.data?.status === 'succeeded'
      ? `${getBaseUrl()}/api/v1/research/jobs/${encodeURIComponent(jobId)}/evidence-reader`
      : '';

  const handleSelectStep = (stepId: string) => {
    manualStepSelectionRef.current = true;
    setSelectedStepId(stepId);
    window.setTimeout(() => {
      const target = document.getElementById(getResearchStepTargetId(stepId))
        || document.getElementById(RESEARCH_STEP_FALLBACK_TARGET);
      if (!target) return;
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      target.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth', block: 'start' });
    }, 0);
  };

  const handleSelectEvidence = (evidenceId: string, hint?: { subject?: string; axis?: string }) => {
    const row = evidenceLedger.find((item) => item.id === evidenceId);
    const subject = hint?.subject || row?.subject || 'all';
    const axis = hint?.axis || row?.axis || 'all';
    if (subject !== 'all' || axis !== 'all') {
      setEvidenceFilters({
        subject,
        axis,
        usage: 'all',
      });
    }
    setActiveEvidenceId(evidenceId);
    window.setTimeout(() => {
      document.getElementById('research-evidence-ledger')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
  };

  const handleInlineEvidenceClick = (evidenceId: string) => {
    const row = evidenceLedger.find((item) => item.id === evidenceId);
    handleSelectEvidence(evidenceId, { subject: row?.subject, axis: row?.axis });
  };

  const handleReviewIssueSelect = (issue: ReviewIssue) => {
    setActiveIssueId(issue.id);
    if (issue.evidenceId) {
      handleSelectEvidence(issue.evidenceId, { subject: issue.subject, axis: issue.axis });
      return;
    }
    if (issue.subject || issue.axis) {
      setEvidenceFilters({
        subject: issue.subject || 'all',
        axis: issue.axis || 'all',
        usage: issue.type === 'risk' ? 'risk' : 'all',
      });
      window.setTimeout(() => {
        document.getElementById('research-evidence-ledger')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 0);
      return;
    }
    window.setTimeout(() => {
      document.getElementById('research-validation-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
  };

  const resetPlanningState = () => {
    planningRequestVersion.current += 1;
    setPlanningStage('idle');
    setPlanningBaseRequest('');
    setResolvedRequest('');
    setClarificationQuestion('');
    setClarificationAnswer('');
    setReconToken('');
    setDisambiguationTerms([]);
    setPlannedSections([]);
    setConfirmedPlan(undefined);
    setPlanId('');
    setPlanToken('');
    setResolvedStrategy('standard');
    setPlanningError('');
    clearResearchPlanningDraft();
  };

  useEffect(() => {
    if (planningStage !== 'reviewing' || !confirmedPlan || !planId || !planToken || !plannedSections.length) return;
    saveResearchPlanningDraft({
      request,
      resolvedRequest,
      topic,
      depth,
      reconToken,
      disambiguationTerms,
      plannedSections,
      confirmedPlan,
      planId,
      planToken,
      strategy: resolvedStrategy,
    });
  }, [planningStage, request, resolvedRequest, topic, depth, reconToken, disambiguationTerms, plannedSections, confirmedPlan, planId, planToken, resolvedStrategy]);

  const handleStudioModeChange = (mode: 'question_loop' | 'single_url') => {
    if (mode === studioMode) return;
    resetPlanningState();
    setStudioMode(mode);
  };

  const handleRequestChange = (value: string) => {
    if (studioMode === 'question_loop' && planningStage !== 'idle') resetPlanningState();
    setRequest(value);
  };

  const handleTopicChange = (value: string) => {
    if (planningStage !== 'idle') resetPlanningState();
    setTopic(value);
  };

  const applyDraftResult = (result: Awaited<ReturnType<typeof draftPlan.mutateAsync>>, requestId: number) => {
    if (requestId !== planningRequestVersion.current) return false;
    if (result.status !== 'ready' || !result.planned_sections?.length) {
      setPlanningError(formatPlanningFailure(result.failure_reason || result.reason || result.message));
      return false;
    }
    if (!result.confirmed_plan || !result.plan_id || !result.plan_token || !result.strategy) {
      setPlanningError('规划服务未返回完整的确认计划凭证，请重试。');
      return false;
    }
    setPlannedSections(result.planned_sections);
    setConfirmedPlan(result.confirmed_plan);
    setPlanId(result.plan_id);
    setPlanToken(result.plan_token);
    setResolvedStrategy(result.strategy);
    setPlanningError('');
    return true;
  };

  const runPlanning = async (candidateRequest: string) => {
    const requestId = ++planningRequestVersion.current;
    setPlanningStage('clarifying');
    setPlanningError('');
    try {
      const clarification = await clarifyPlan.mutateAsync({ user_request: candidateRequest });
      if (requestId !== planningRequestVersion.current) return;
      if (clarification.status === 'ambiguous') {
        setPlanningBaseRequest(candidateRequest);
        setClarificationQuestion(clarification.suggested_question || '请补充说明你具体想调研的对象和范围。');
        setClarificationAnswer('');
        setPlanningStage('needs_clarification');
        return;
      }
      if (clarification.status === 'unavailable' || !clarification.recon_token) {
        setPlanningError(formatPlanningFailure(clarification.message || clarification.failure_reason));
        setPlanningStage('error');
        return;
      }

      setResolvedRequest(candidateRequest);
      setReconToken(clarification.recon_token);
      setDisambiguationTerms(clarification.disambiguation_terms || []);
      const strategy = clarification.recommended_strategy || 'standard';
      setResolvedStrategy(strategy);
      setPlanningStage('drafting');
      const draft = await draftPlan.mutateAsync({
        user_request: candidateRequest,
        topic: topic.trim() || undefined,
        recon_token: clarification.recon_token,
        strategy,
      });
      if (applyDraftResult(draft, requestId)) setPlanningStage('reviewing');
      else if (requestId === planningRequestVersion.current) setPlanningStage('error');
    } catch (error) {
      if (requestId !== planningRequestVersion.current) return;
      setPlanningError(formatPlanningFailure(error));
      setPlanningStage('error');
    }
  };

  const handlePreparePlan = () => {
    const candidateRequest = request.trim();
    if (!candidateRequest) return;
    void runPlanning(candidateRequest);
  };

  const handleClarificationContinue = () => {
    if (!clarificationAnswer.trim()) return;
    const candidateRequest = buildClarifiedResearchRequest(planningBaseRequest, clarificationAnswer);
    void runPlanning(candidateRequest);
  };

  const revisePlan = async (adjustment: ResearchPlanningAdjustment) => {
    if (!resolvedRequest || !reconToken) return;
    const requestId = ++planningRequestVersion.current;
    setPlanningStage('revising');
    setPlanningError('');
    try {
      const draft = await draftPlan.mutateAsync({
        user_request: resolvedRequest,
        topic: topic.trim() || undefined,
        recon_token: reconToken,
        strategy: resolvedStrategy,
        adjustment,
      });
      applyDraftResult(draft, requestId);
      if (requestId === planningRequestVersion.current) setPlanningStage('reviewing');
    } catch (error) {
      if (requestId !== planningRequestVersion.current) return;
      setPlanningError(formatPlanningFailure(error));
      setPlanningStage('reviewing');
    }
  };

  const handleRedoPlanSection = (index: number) => {
    const section = plannedSections[index];
    if (!section) return;
    void revisePlan({
      keep: plannedSections.filter((_, itemIndex) => itemIndex !== index),
      redo: [{ title: section.title, axis: section.axis, intent: section.intent }],
    });
  };

  const handleDeletePlanSection = (index: number) => {
    const keep = plannedSections.filter((_, itemIndex) => itemIndex !== index);
    if (!keep.length) return;
    void revisePlan({ keep });
  };

  const handleAddPlanSection = (title: string) => {
    void revisePlan({ keep: plannedSections, add: [{ title }] });
  };

  const handleStart = () => {
    if (studioMode === 'single_url') {
      if (!url.trim()) return;
      create.mutate(
        {
          user_request: request,
          source_mode: 'web',
          pipeline_mode: 'single_url',
          url: url.trim(),
        },
        {
          onSuccess: (res) => {
            setJobId(res.job_id);
            setSearchParams({ job_id: res.job_id });
            lastSyncedJobId.current = res.job_id;
            setActiveTab('detail');
            addJob(res.job_id);
          },
        }
      );
      return;
    }
    if (planningStage !== 'reviewing' || !plannedSections.length) {
      handlePreparePlan();
      return;
    }
    if (!confirmedPlan || !planId || !planToken) {
      setPlanningError('确认计划凭证缺失，请重新生成计划。');
      setPlanningStage('error');
      return;
    }
    create.mutate(
      buildConfirmedResearchJobRequest({
        request,
        resolvedRequest,
        topic,
        depth,
        strategy: resolvedStrategy,
        plannedSections,
        confirmedPlan,
        planId,
        planToken,
        disambiguationTerms,
      }),
      {
        onSuccess: (res) => {
          setJobId(res.job_id);
          setSearchParams({ job_id: res.job_id });
          lastSyncedJobId.current = res.job_id;
          setActiveTab('detail');
          addJob(res.job_id);
          resetPlanningState();
        },
      }
    );
  };

  const handleApprove = () => {
    if (!jobId) return;
    archivePreview.mutate(
      { jobId, request: { acknowledge_matrix_gaps: matrixGapsAcknowledged } },
      { onSuccess: () => setArchiveDialogOpen(true) },
    );
  };

  const handleSaveSingleUrlBrief = () => {
    if (!jobId || !singleUrlSaveTopic.trim()) return;
    approve.mutate({
      jobId,
      request: {
        topic: singleUrlSaveTopic.trim(),
      },
    });
  };

  const handleFollowupIssue = (issue: ReviewIssue) => {
    if (!jobId) return;
    setActiveIssueId(issue.id);
    followup.mutate({
      jobId,
      request: {
        issue_id: issue.id,
        issue_type: issue.type,
        evidence_id: issue.evidenceId,
        subject: issue.subject,
        comparison_axis: issue.axis,
        query: buildFollowupQuery(issue, request || job.data?.input?.user_request),
        reason: issue.detail,
      },
    });
  };

  const handleToggleSection = (sectionId: string) => {
    setSelectedSectionIds((prev) =>
      prev.includes(sectionId) ? prev.filter((id) => id !== sectionId) : [...prev, sectionId]
    );
  };

  const handleSubmitSectionRevision = () => {
    if (!jobId || !selectedSectionIds.length || !revisionInstruction.trim()) return;
    sectionRevision.mutate(
      { jobId, request: { section_ids: selectedSectionIds, instruction: revisionInstruction.trim() } },
      {
        onSuccess: () => {
          setSelectedSectionIds([]);
          setRevisionInstruction('');
        },
      }
    );
  };

  const handleSelectHistoryJob = (id: string) => {
    setJobId(id);
    setSearchParams({ job_id: id });
    lastSyncedJobId.current = id;
    setActiveTab('detail');
    window.setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 0);
  };

  const handleStudioTabChange = (nextTab: ResearchStudioTab) => {
    if (nextTab === 'detail') return;
    if (searchParams.has('job_id')) {
      setSearchParams(getResearchStudioSearchParams(searchParams, nextTab));
    }
    lastSyncedJobId.current = '';
    setJobId('');
    setSelectedStepId('');
    setActiveTab(nextTab);
  };

  const planningBusy = ['clarifying', 'drafting', 'revising'].includes(planningStage);
  const planningActionLabel = planningStage === 'clarifying'
    ? '正在核实主题'
    : planningStage === 'drafting'
      ? '正在生成计划'
      : planningStage === 'reviewing'
        ? '确认并开始调研'
        : planningStage === 'error'
          ? '重新生成计划'
          : '生成调研计划';

  return (
    <div className="space-y-4">
      <PageHeader title="调研工作台" description="从问题拆解、证据检索到文档审阅和知识库归档" />

      <ResearchStudioTabs active={activeTab} onChange={handleStudioTabChange} />

      <div className={cn('space-y-4', activeTab === 'history' ? 'hidden' : '')}>
      {activeTab === 'create' && (
      <section className="rounded-panel border border-border bg-white p-4">
        <div className="mb-5">
          <p className="mb-2 text-sm font-medium text-primary">调研类型</p>
          <div
            className="inline-grid w-full max-w-md grid-cols-2 rounded-control border border-border bg-page p-1"
            role="tablist"
            aria-label="调研类型"
          >
            <button
              type="button"
              role="tab"
              aria-selected={studioMode === 'question_loop'}
              onClick={() => handleStudioModeChange('question_loop')}
              className={cn(
                'flex min-h-12 items-center justify-center gap-2 rounded-[5px] px-3 text-sm font-medium transition-colors',
                studioMode === 'question_loop' ? 'bg-white text-primary shadow-sm' : 'text-secondary hover:text-primary'
              )}
            >
              <Search className="h-4 w-4" />
              主题调研
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={studioMode === 'single_url'}
              onClick={() => handleStudioModeChange('single_url')}
              className={cn(
                'flex min-h-12 items-center justify-center gap-2 rounded-[5px] px-3 text-sm font-medium transition-colors',
                studioMode === 'single_url' ? 'bg-white text-primary shadow-sm' : 'text-secondary hover:text-primary'
              )}
            >
              <ExternalLink className="h-4 w-4" />
              链接简报
            </button>
          </div>
        </div>

        {studioMode === 'single_url' ? (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(240px,320px)]">
            <FormField label="网页链接">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/article"
                className="w-full rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
              />
            </FormField>
            <FormField label="关注内容（可选）">
              <input
                type="text"
                value={request}
                onChange={(e) => handleRequestChange(e.target.value)}
                placeholder="想重点看哪方面"
                className="w-full rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
              />
            </FormField>
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(240px,320px)]">
            <FormField label="调研主题">
              <input
                type="text"
                value={request}
                onChange={(e) => handleRequestChange(e.target.value)}
                placeholder="例如：调研 LangGraph 状态管理"
                className="w-full rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
              />
            </FormField>
            <FormField label="保存名称（可选）">
              <input
                type="text"
                value={topic}
                onChange={(e) => handleTopicChange(e.target.value)}
                placeholder="默认根据主题自动生成"
                className="w-full rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
              />
            </FormField>
          </div>
        )}
        {studioMode === 'question_loop' && (
          <div className="mt-5 border-t border-border pt-4">
            <div className="flex flex-wrap items-end gap-5">
              <fieldset>
                <legend className="mb-2 text-sm font-medium text-primary">调研范围</legend>
                <div className="inline-grid grid-cols-2 rounded-control border border-border bg-page p-1">
                  {([
                    { value: 'normal', label: '标准', note: '日常学习' },
                    { value: 'deep', label: '深入', note: '更广覆盖' },
                  ] as const).map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      aria-pressed={depth === option.value}
                      onClick={() => setDepth(option.value)}
                      className={cn(
                        'min-w-28 rounded-[5px] px-4 py-2 text-left transition-colors',
                        depth === option.value ? 'bg-white text-primary shadow-sm' : 'text-secondary hover:text-primary'
                      )}
                    >
                      <span className="block text-sm font-medium">{option.label}</span>
                      <span className="block text-xs text-secondary">{option.note}</span>
                    </button>
                  ))}
                </div>
              </fieldset>
              <div>
                <p className="mb-2 text-sm font-medium text-primary">资料来源</p>
                <div className="flex min-h-12 items-center gap-2 text-sm text-secondary">
                  <Globe2 className="h-4 w-4 text-accent" />
                  <span>公开网页，优先官方与权威资料</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {studioMode === 'question_loop' && planningStage === 'needs_clarification' && (
          <section className="mt-5 border-t border-border pt-4" aria-labelledby="clarification-title">
            <h3 id="clarification-title" className="text-base font-semibold text-primary">确认调研对象</h3>
            <p className="mt-2 text-sm text-primary">{clarificationQuestion}</p>
            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
              <input
                type="text"
                value={clarificationAnswer}
                onChange={(event) => setClarificationAnswer(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    handleClarificationContinue();
                  }
                }}
                placeholder="补充具体对象、产品或范围"
                aria-label="调研对象补充说明"
                className="min-w-0 flex-1 rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
              />
              <Button
                type="button"
                disabled={!clarificationAnswer.trim()}
                onClick={handleClarificationContinue}
              >
                继续生成计划
              </Button>
            </div>
          </section>
        )}

        {studioMode === 'question_loop' && planningStage === 'error' && planningError && (
          <p className="mt-4 text-sm text-danger">{planningError}</p>
        )}

        {studioMode === 'question_loop' && plannedSections.length > 0 && ['reviewing', 'revising'].includes(planningStage) && (
          <ResearchPlanReview
            sections={plannedSections}
            isRevising={planningStage === 'revising'}
            error={planningError}
            onRedo={handleRedoPlanSection}
            onDelete={handleDeletePlanSection}
            onAdd={handleAddPlanSection}
          />
        )}

        <div className="mt-5 flex justify-end border-t border-border pt-4">
          <Button
            onClick={handleStart}
            isLoading={create.isPending || planningBusy}
            disabled={
              studioMode === 'single_url'
                ? !url.trim()
                : !request.trim() || planningStage === 'needs_clarification' || planningStage === 'revising'
            }
            className="w-full self-end sm:w-auto"
          >
            <Play className="mr-1.5 h-4 w-4" />
            {studioMode === 'single_url' ? '生成简报' : planningActionLabel}
          </Button>
        </div>
      </section>
      )}

      {activeTab === 'detail' && job.data && (
        <section className="overflow-hidden rounded-panel border border-border bg-white">
          <div className="h-1 bg-accent" />
          <div className="flex flex-col gap-5 p-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0 max-w-5xl">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-accent">
                <span>调研详情</span>
                <span className="h-1 w-1 rounded-full bg-border" />
                <span className="normal-case tracking-normal text-secondary">
                  {job.data.result?.pipeline_mode === 'single_url' ? '链接简报' : '主题调研'}
                </span>
              </div>
              <h2 className="font-display text-2xl font-semibold leading-tight text-primary sm:text-3xl">
                {researchTitle}
              </h2>
              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-secondary">
                <span className="font-mono">{job.data.job_id}</span>
                <span>更新于 {job.data.updated_at}</span>
                {coreSectionCount > 0 && (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/20 bg-accent/10 px-2.5 py-1 font-medium text-accent">
                    {coreSectionCount} 个核心章节
                    {coreEmptySectionCount > 0 ? ` · ${coreEmptySectionCount} 个空缺` : ''}
                  </span>
                )}
              </div>
              {!!coreSections.length && (
                <div className="mt-4 border-t border-border pt-4">
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                    <p className="text-xs font-semibold text-primary">核心章节</p>
                    <p className="text-[11px] text-secondary">以下章节决定报告是否完整、可交付</p>
                  </div>
                  <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                    {coreSections.map((section) => (
                      <div
                        key={section.sectionId || section.title}
                        className="flex min-w-0 items-center justify-between gap-2 rounded-control border border-accent/20 bg-accent/5 px-3 py-2"
                      >
                        <span className="min-w-0 truncate text-xs font-medium text-primary" title={section.title}>
                          {section.title || section.sectionId}
                        </span>
                        <SectionStatusBadge section={section} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="flex shrink-0 flex-wrap gap-2">
              <Button variant="secondary" size="sm" onClick={() => handleStudioTabChange('history')}>
                返回历史调研
              </Button>
              <Button variant="secondary" size="sm" onClick={() => handleStudioTabChange('create')}>
                新建调研
              </Button>
            </div>
          </div>
        </section>
      )}

      {activeTab === 'detail' && (
      job.isLoading ? (
        <section className="flex items-center justify-center gap-2 rounded-panel border border-border bg-white p-10 text-sm text-secondary">
          <Loader2 className="h-4 w-4 animate-spin text-accent" />
          正在加载调研详情...
        </section>
      ) : job.isError ? (
        <section className="rounded-panel border border-red-100 bg-red-50 p-5 text-sm text-danger">
          <p className="font-medium">调研详情加载失败</p>
          <p className="mt-1">{(job.error as Error)?.message || '请稍后重试。'}</p>
          <button type="button" className="mt-3 underline" onClick={() => job.refetch()}>重新加载</button>
        </section>
      ) : job.data && job.data.result?.pipeline_mode === 'single_url' ? (
        <SingleUrlBriefResultView
          job={job.data}
          evidenceReaderUrl={evidenceReaderUrl}
          saveTopic={singleUrlSaveTopic}
          onSaveTopicChange={setSingleUrlSaveTopic}
          onSave={handleSaveSingleUrlBrief}
          isSaving={approve.isPending}
        />
      ) : job.data ? (
        <section className="grid gap-4 xl:grid-cols-[390px_minmax(0,1fr)]">
          <div className="min-w-0 self-start rounded-panel border border-border bg-white xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:overflow-y-auto">
            <div className="border-b border-border p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-primary">{job.data.job_id}</p>
                  <p className="mt-1 text-xs text-secondary">{job.data.progress?.message}</p>
                  {!!steps.length && <p className="mt-1 text-[11px] text-faint">点击步骤定位到右侧产物</p>}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <StatusBadge status={job.data.status} />
                  <button
                    type="button"
                    onClick={() => setStepRailExpanded((prev) => !prev)}
                    className="rounded-control border border-border p-1 text-secondary hover:border-primary hover:text-primary"
                    aria-label={stepRailExpanded ? '收起步骤列表' : '展开步骤列表'}
                  >
                    {stepRailExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </button>
                </div>
              </div>
            </div>
            {stepRailExpanded ? (
              <div className="space-y-2 p-3">
                {steps.map((step, index) => (
                  <button
                    key={step.id}
                    type="button"
                    onClick={() => handleSelectStep(step.id)}
                    aria-controls={getResearchStepTargetId(step.id)}
                    aria-current={selectedStep?.id === step.id ? 'step' : undefined}
                    className={cn(
                      'grid w-full grid-cols-[28px_minmax(0,1fr)] gap-2 rounded-control p-2 text-left transition-colors',
                      selectedStep?.id === step.id ? 'bg-teal-50' : 'hover:bg-page'
                    )}
                  >
                    <div className="flex flex-col items-center">
                      <StepIcon status={step.status} />
                      {index < steps.length - 1 && <span className="mt-1 h-8 w-px bg-border" />}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <p className="truncate text-sm font-medium text-primary">{step.title}</p>
                        <span className="shrink-0 text-[11px] uppercase text-secondary">
                          {formatStepMeta(step)}
                        </span>
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs text-secondary">
                        {displayStepSummary(step, coverageReason) || '等待执行'}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setStepRailExpanded(true)}
                className="flex w-full flex-wrap items-center gap-1.5 p-3 text-left hover:bg-page"
              >
                {steps.map((step) => (
                  <span key={step.id} title={step.title}>
                    <StepIcon status={step.status} />
                  </span>
                ))}
                {selectedStep && (
                  <span className="ml-1 truncate text-xs text-secondary">{selectedStep.title}</span>
                )}
              </button>
            )}
          </div>

          <div className="min-w-0 space-y-4">
            <div id={RESEARCH_STEP_FALLBACK_TARGET} className="scroll-mt-24">
              <ResearchStepTarget
                id="research-approval-panel"
                active={selectedStepTargetId === 'research-approval-panel'}
              >
              <div className="rounded-panel border border-border bg-white p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-semibold text-primary">{selectedStep?.title || '执行详情'}</h3>
                  <p className="mt-1 text-sm text-secondary">
                    {(selectedStep && displayStepSummary(selectedStep, coverageReason)) || '创建任务后会在这里展示执行细节。'}
                  </p>
                  {selectedStep && (
                    <p className="mt-1 text-xs text-secondary">
                      {selectedStep.status}
                      {selectedStep.elapsed_ms != null ? ` · ${formatDuration(selectedStep.elapsed_ms)}` : ''}
                      {selectedStep.started_at ? ` · ${selectedStep.started_at}` : ''}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Link
                    to={`/observability?job_id=${encodeURIComponent(jobId)}`}
                    className="inline-flex items-center gap-1 rounded-control border border-border px-3 py-2 text-sm text-secondary hover:bg-page hover:text-primary"
                  >
                    <ExternalLink className="h-4 w-4" />
                    查看原始事件
                  </Link>
                  {evidenceReaderUrl && (
                    <a
                      href={evidenceReaderUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 rounded-control border border-border px-3 py-2 text-sm text-secondary hover:bg-page hover:text-primary"
                    >
                      <ExternalLink className="h-4 w-4" />
                      查看证据阅读版
                    </a>
                  )}
                  <Button
                    onClick={handleApprove}
                    isLoading={archivePreview.isPending}
                    disabled={!canApprove}
                    title={canApprove
                      ? '审批并写入知识库'
                      : requiresMatrixGapAcknowledgement && !matrixGapsAcknowledged
                        ? '请先查看并确认 Matrix Gap'
                        : qualityGate?.reason || '等待调研生成 documents 后才能审批'}
                  >
                    <Save className="mr-1.5 h-4 w-4" />
                    审批落库
                  </Button>
                </div>
              </div>
              {!!selectedStep?.items?.length && (
                <div className="mt-4 grid gap-2 md:grid-cols-2">
                  {selectedStep.items.map((item, index) => (
                    <StepItem
                      key={index}
                      item={item}
                      sectionMeta={findResearchSectionMeta(
                        sectionMetadata,
                        item !== null && typeof item === 'object' ? item.label : undefined,
                      )}
                    />
                  ))}
                </div>
              )}
              {job.data.error && (
                <div className="mt-4 rounded-control border border-red-100 bg-red-50 p-3 text-sm text-danger">
                  {job.data.error.message}
                </div>
              )}
              {qualityGate && !qualityGate.approval_ready && (
                <div className="mt-4 rounded-control border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                  <p className="font-medium">证据质量不足，暂不建议落库</p>
                  <p className="mt-1">
                    {emptyCoreSections.length ? '未通过：以下核心章节为空' : qualityGate.reason}
                  </p>
                  {!!emptyCoreSections.length && (
                    <ul className="mt-2 flex flex-wrap gap-2">
                      {emptyCoreSections.map((section) => (
                        <li
                          key={section.sectionId || section.title}
                          className="rounded-full border border-amber-200 bg-white/70 px-2.5 py-1 text-xs font-medium"
                        >
                          {section.title || section.sectionId}
                          {section.title && section.sectionId ? `（${section.sectionId}）` : ''}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              {requiresMatrixGapAcknowledgement && (
                <div className="mt-4 border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
                  <p className="font-medium">比较矩阵存在 {matrixGaps.length} 个缺口，报告已标记为 partial</p>
                  <ul className="mt-2 space-y-1 text-xs">
                    {matrixGaps.slice(0, 8).map((cell) => (
                      <li key={cell.cell_id}>
                        {cell.subject} × {cell.axis}：{cell.status}
                      </li>
                    ))}
                  </ul>
                  <label className="mt-3 flex cursor-pointer items-start gap-2">
                    <input
                      type="checkbox"
                      className="mt-0.5 h-4 w-4 accent-[#176B55]"
                      checked={matrixGapsAcknowledged}
                      onChange={(event) => setMatrixGapsAcknowledged(event.target.checked)}
                    />
                    <span>我已查看 Matrix Gap，确认以不完整比较报告知情归档。</span>
                  </label>
                </div>
              )}
              </div>
              </ResearchStepTarget>
            </div>

            <div className="grid gap-4 lg:grid-cols-3">
              <Metric label="来源" value={coverage?.source_count ?? sources.length} />
              <Metric label="素材" value={materials.length || coverage?.document_count || 0} />
              <Metric label="证据" value={evidenceMap.length || coverage?.evidence_pack_count || evidencePack.length} />
              <Metric label="文档" value={researchDocuments.length} />
            </div>

            <QualityOverviewPanel
              summary={qualitySummary}
              jobStatus={job.data.status}
              canApprove={canApprove}
              requiresMatrixGapAcknowledgement={requiresMatrixGapAcknowledgement}
              matrixGapsAcknowledged={matrixGapsAcknowledged}
            />
            <ResearchDiagnosticsPanel
              qualityGate={qualityGate}
              documents={researchDocuments}
              evidenceDiagnostics={sectionEvidenceDiagnostics}
              sectionMetadata={sectionMetadata}
            />

            <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <ResearchStepTarget
                id="research-plan-panel"
                active={selectedStepTargetId === 'research-plan-panel'}
              >
                <ResearchPlanPanel plan={researchPlan} queryPlan={queryPlan} />
              </ResearchStepTarget>
              <ResearchStepTarget
                id="research-validation-panel"
                active={selectedStepTargetId === 'research-validation-panel'}
              >
                <ValidationPanel validation={validation} qualityReason={qualityGate?.reason} />
              </ResearchStepTarget>
            </section>

            <EvidenceExtractionPanel diagnostics={sectionEvidenceDiagnostics} />

            <QuestionLoopPanel
              states={questionStates}
              reportQuality={reportQuality}
              rejectedEvidenceCount={rejectedEvidence.length}
            />

            {coverage ? (
              <ResearchStepTarget
                id="research-coverage-panel"
                active={selectedStepTargetId === 'research-coverage-panel'}
              >
              <section className="rounded-panel border border-border bg-white p-4">
                <h3 className="text-sm font-semibold text-primary">覆盖度</h3>
                <p className="mt-2 text-sm text-secondary">{coverageReason}</p>
                {!!questionStates.length && (
                  <div className="mt-3 flex flex-wrap gap-2 text-xs">
                    <span className="rounded-control bg-green-50 px-2 py-1 text-success">
                      完整 {displayedAnswerCounts.complete ?? 0}
                    </span>
                    <span className="rounded-control bg-amber-50 px-2 py-1 text-amber-800">
                      部分 {displayedAnswerCounts.partial ?? 0}
                    </span>
                    <span className="rounded-control bg-page px-2 py-1 text-secondary">
                      未覆盖 {displayedAnswerCounts.uncovered ?? 0}
                    </span>
                    {!!coverage.safety_blocked_count && (
                      <span className="rounded-control bg-red-50 px-2 py-1 text-red-700">
                        事实安全 {coverage.safety_blocked_count}
                      </span>
                    )}
                  </div>
                )}
                {!!coverage.weak_sections?.length && (
                  <div className="mt-3 grid gap-2 md:grid-cols-2">
                    {coverage.weak_sections.map((section) => (
                      <div key={section.section_id || section.title} className="rounded-control border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
                        <p className="font-medium">{section.title || section.section_id}</p>
                        <p className="mt-1">
                          证据 {section.evidence_count ?? 0} · 来源 {section.source_count ?? 0} · 域名 {section.domain_count ?? 0} · 问题 {section.covered_question_count ?? 0}/{section.question_count ?? 0}
                        </p>
                        {section.reason && <p className="mt-1">{section.reason}</p>}
                      </div>
                    ))}
                  </div>
                )}
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  {(coverage.covered_titles || []).map((title) => (
                    <span key={title} className="rounded-control bg-green-50 px-2 py-1 text-success">
                      {title}
                    </span>
                  ))}
                  {(coverage.partial_titles || []).map((title) => (
                    <span key={`partial-${title}`} className="rounded-control bg-amber-50 px-2 py-1 text-amber-800">
                      部分：{title}
                    </span>
                  ))}
                  {(coverage.missing_titles || []).map((title) => (
                    <span key={title} className="rounded-control bg-amber-50 px-2 py-1 text-amber-800">
                      缺口：{title}
                    </span>
                  ))}
                </div>
              </section>
              </ResearchStepTarget>
            ) : (
              <ResearchStepTarget
                id="research-coverage-panel"
                active={selectedStepTargetId === 'research-coverage-panel'}
              >
                <section className="rounded-panel border border-border bg-white p-4">
                  <h3 className="text-sm font-semibold text-primary">覆盖度</h3>
                  <p className="mt-2 text-sm text-secondary">覆盖度检查完成后会在这里显示完整、部分和未覆盖的问题。</p>
                </section>
              </ResearchStepTarget>
            )}

            <MatrixEvidencePanel
              matrix={matrixEvidence}
              onSelectCell={(subject, axis) => setEvidenceFilters({ subject, axis, usage: 'all' })}
            />

            <ResearchStepTarget
              id="research-evidence-ledger"
              active={selectedStepTargetId === 'research-evidence-ledger'}
            >
              <EvidenceLedgerPanel
                ledger={evidenceLedger}
                matrix={matrixEvidence}
                filters={evidenceFilters}
                activeEvidenceId={activeEvidenceId}
                onFiltersChange={setEvidenceFilters}
                onSelectEvidence={setActiveEvidenceId}
              />
            </ResearchStepTarget>

            <section className="grid min-w-0 gap-4 lg:grid-cols-2">
              <ResearchStepTarget
                id="research-source-panel"
                active={selectedStepTargetId === 'research-source-panel'}
              >
                <EvidencePanel
                  title="已采用来源"
                  description="为已准入证据提供支持的出处；这里记录来源身份和原始链接。"
                  items={sources}
                />
              </ResearchStepTarget>
              <ResearchStepTarget
                id="research-material-panel"
                active={selectedStepTargetId === 'research-material-panel'}
              >
                <EvidencePanel
                  title="已抓取素材"
                  description="从来源抓取并清洗后的正文；用于筛选片段和提取证据。"
                  items={materials}
                />
              </ResearchStepTarget>
              <div className="min-w-0 lg:col-span-2">
                <EvidencePanel title="证据地图" items={evidenceMap.length ? evidenceMap : evidencePack} />
              </div>
            </section>

            <ResearchStepTarget
              id="research-chunks-panel"
              active={selectedStepTargetId === 'research-chunks-panel'}
            >
              <SelectedChunksPanel
                chunks={selectedChunks}
                evidenceMap={evidenceMap}
                documents={researchDocuments}
                contextDiagnostics={job.data?.result?.evidence_context_diagnostics}
              />
            </ResearchStepTarget>

            <ResearchStepTarget
              id="research-document-preview"
              active={selectedStepTargetId === 'research-document-preview'}
            >
            <section className="rounded-panel border border-border bg-white p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-primary">文档预览</h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-secondary">{suggestedTopic || '待生成 topic'}</span>
                  {reviewModel.issues.filter((issue) => issue.type !== 'clear').length > 0 && (
                    <button
                      type="button"
                      onClick={() => setReviewDrawerOpen(true)}
                      className="relative flex items-center gap-1.5 rounded-control border border-border px-3 py-1.5 text-xs text-secondary hover:border-primary hover:text-primary"
                    >
                      <ListChecks className="h-3.5 w-3.5" />
                      复核
                      <span className="ml-1 rounded-full bg-accent px-1.5 py-0.5 text-[11px] text-white">
                        {reviewModel.issues.filter((issue) => issue.type !== 'clear').length}
                      </span>
                    </button>
                  )}
                </div>
              </div>
              {hasDocuments ? (
                <div className="min-w-0 space-y-3">
                    {researchDocuments.map((doc) => {
                      const sectionKey = doc.section_id || doc.title;
                      const sectionMeta = findResearchSectionMeta(sectionMetadata, doc.section_id)
                        || findResearchSectionMeta(sectionMetadata, doc.title);
                      const isSelectedForRevision = selectedSectionIds.includes(sectionKey);
                      const isAutoExpanded = sectionKey === autoExpandedSectionKey;
                      const auditExpanded = manuallyToggledSectionKeys.has(sectionKey) ? !isAutoExpanded : isAutoExpanded;
                      return (
                        <div
                          key={sectionKey}
                          ref={(el) => {
                            sectionRefs.current.set(sectionKey, el);
                          }}
                          data-section-key={sectionKey}
                          className="rounded-control border border-border p-3"
                        >
                          <div className="mb-2 flex items-center justify-between gap-2">
                            <div className="flex min-w-0 flex-wrap items-center gap-2">
                              <p className="flex min-w-0 items-center gap-2 text-sm font-medium text-accent">
                                <FileText className="h-4 w-4 shrink-0" />
                                <span>{doc.title}</span>
                              </p>
                              {sectionMeta && <SectionRoleBadges section={sectionMeta} />}
                            </div>
                            {canReviseSections && doc.section_id && (
                              <label className="flex shrink-0 items-center gap-1.5 text-xs text-secondary">
                                <input
                                  type="checkbox"
                                  checked={isSelectedForRevision}
                                  disabled={hasRunningRevision}
                                  onChange={() => handleToggleSection(doc.section_id!)}
                                />
                                选中此节修改
                              </label>
                            )}
                          </div>
                          {doc.revision_diagnostics && (
                            <p
                              className={cn(
                                'mb-2 text-xs',
                                doc.revision_diagnostics.accepted ? 'text-success' : 'text-amber-700'
                              )}
                            >
                              {doc.revision_diagnostics.accepted
                                ? '✅ 已按要求修改'
                                : `⚠️ 修改被拒绝：${revisionRejectReasonLabel(doc.revision_diagnostics.rejected_reason)}`}
                            </p>
                          )}
                          <GroundingSummary title={doc.title} grounding={grounding} />
                          <DocumentEvidenceIndex
                            document={doc}
                            activeEvidenceId={activeEvidenceId}
                            onSelectEvidence={handleSelectEvidence}
                          />
                          <div className="max-h-72 overflow-y-auto">
                            <MarkdownPreview
                              content={doc.content}
                              evidenceIds={documentEvidenceIds(doc)}
                              activeEvidenceId={activeEvidenceId}
                              onEvidenceClick={handleInlineEvidenceClick}
                            />
                          </div>
                          <EvidenceAuditPanel
                            document={doc}
                            expanded={auditExpanded}
                            onToggle={() => handleToggleEvidenceAudit(sectionKey)}
                          />
                        </div>
                      );
                    })}
                    {canReviseSections && (
                      <div className="rounded-control border border-dashed border-border bg-page p-3">
                        <p className="text-xs font-medium text-primary">章节修改（可跨节选中，只重写选中章节）</p>
                        <p className="mt-1 text-xs text-secondary">
                          勾选上方要调整的章节，填写修改意见（如"这两节风格对不上，统一一下"），提交后只有选中章节会被重写，其余章节保持不变。不会触发新的网络检索——如果是材料缺失问题，请先用右侧的"补查"入口。
                        </p>
                        <textarea
                          value={revisionInstruction}
                          onChange={(e) => setRevisionInstruction(e.target.value)}
                          disabled={hasRunningRevision}
                          placeholder="描述想要的修改，例如：补充更多实操细节 / 精简重复表述 / 让第 2 节和第 3 节的结论呼应"
                          rows={3}
                          className="mt-2 w-full rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
                        />
                        <div className="mt-2 flex items-center justify-between">
                          <span className="text-xs text-secondary">已选中 {selectedSectionIds.length} 节</span>
                          <Button
                            onClick={handleSubmitSectionRevision}
                            isLoading={sectionRevision.isPending || hasRunningRevision}
                            disabled={hasRunningRevision || !selectedSectionIds.length || !revisionInstruction.trim()}
                          >
                            提交修改
                          </Button>
                        </div>
                        {sectionRevision.isError && (
                          <p className="mt-2 text-xs text-danger">
                            {(sectionRevision.error as Error)?.message || '提交失败'}
                          </p>
                        )}
                      </div>
                    )}
                </div>
              ) : (
                <div className="rounded-control bg-page p-4 text-sm text-secondary">
                  调研完成 draft 阶段后会在这里出现可审批文档。
                </div>
              )}
            </section>
            </ResearchStepTarget>
          </div>
        </section>
      ) : (
        <section className="rounded-panel border border-dashed border-border bg-white p-8 text-center text-sm text-secondary">
          没有找到这条调研任务，请返回历史调研后重新选择。
        </section>
      ))}

      {(approve.isSuccess || archiveConfirm.isSuccess) && (
        <div className="rounded-panel border border-success/30 bg-green-50 p-4 text-success">
          调研产物已成功落库，可在知识库列表中查看。
        </div>
      )}
      {reviewDrawerOpen && (
        <ReviewQueueDrawer
          activeEvidence={activeEvidence}
          activeIssueId={activeIssueId}
          followupPending={hasRunningFollowup}
          runningFollowupIssueIds={runningFollowupIssueIds}
          canRunTargetedFollowup={canRunTargetedFollowup}
          review={reviewModel}
          onSelectIssue={handleReviewIssueSelect}
          onFollowupIssue={handleFollowupIssue}
          onClose={() => setReviewDrawerOpen(false)}
        />
      )}
      {archiveDialogOpen && archivePreview.data && (
        <ResearchArchiveDialog
          jobId={jobId}
          preview={archivePreview.data}
          knowledgeBases={knowledgeTree.data?.knowledge_bases || []}
          isPreviewing={archivePreview.isPending}
          isConfirming={archiveConfirm.isPending}
          error={archivePreview.error || archiveConfirm.error}
          onClose={() => setArchiveDialogOpen(false)}
          onRepreview={(request) => archivePreview.mutate({
            jobId,
            request: { ...request, acknowledge_matrix_gaps: matrixGapsAcknowledged },
          })}
          onConfirm={() => archiveConfirm.mutate(
            { jobId, previewId: archivePreview.data!.preview_id },
            { onSuccess: () => setArchiveDialogOpen(false) },
          )}
        />
      )}
      </div>

      <div className={activeTab === 'history' ? '' : 'hidden'}>
        <ResearchHistoryPanel onSelectJob={handleSelectHistoryJob} />
      </div>
    </div>
  );
}

function ResearchArchiveDialog({
  jobId,
  preview,
  knowledgeBases,
  isPreviewing,
  isConfirming,
  error,
  onClose,
  onRepreview,
  onConfirm,
}: {
  jobId: string;
  preview: KnowledgePreview;
  knowledgeBases: HierarchicalKnowledgeBase[];
  isPreviewing: boolean;
  isConfirming: boolean;
  error: unknown;
  onClose: () => void;
  onRepreview: (request: ResearchArchivePreviewRequest) => void;
  onConfirm: () => void;
}) {
  const recommendedKb = preview.archive?.knowledge_base;
  const recommendedTheme = preview.archive?.theme;
  const [kbMode, setKbMode] = useState<'existing' | 'new'>(recommendedKb?.created ? 'new' : 'existing');
  const [knowledgeBaseId, setKnowledgeBaseId] = useState(recommendedKb?.created ? knowledgeBases[0]?.id || '' : recommendedKb?.id || knowledgeBases[0]?.id || '');
  const [newKnowledgeBaseTitle, setNewKnowledgeBaseTitle] = useState(recommendedKb?.created ? recommendedKb.title || '' : '');
  const [themeMode, setThemeMode] = useState<'none' | 'existing' | 'new'>(recommendedTheme ? (recommendedTheme.created ? 'new' : 'existing') : 'none');
  const [themeId, setThemeId] = useState(recommendedTheme?.created ? '' : recommendedTheme?.id || '');
  const [newThemeTitle, setNewThemeTitle] = useState(recommendedTheme?.created ? recommendedTheme.title || '' : '');
  const [title, setTitle] = useState(preview.archive?.title || '');
  const [tags, setTags] = useState((preview.archive?.tags || []).join('，'));
  const selectedKb = knowledgeBases.find((item) => item.id === knowledgeBaseId);

  const requestPreview = () => {
    const request: ResearchArchivePreviewRequest = {
      title: title.trim() || undefined,
      tags: tags.split(/[，,]/).map((item) => item.trim()).filter(Boolean),
    };
    if (kbMode === 'new') {
      request.new_knowledge_base = { title: newKnowledgeBaseTitle.trim() };
      if (themeMode === 'new' && newThemeTitle.trim()) request.new_theme = { title: newThemeTitle.trim() };
    } else {
      request.knowledge_base_id = knowledgeBaseId;
      if (themeMode === 'existing') request.theme_id = themeId;
      if (themeMode === 'new' && newThemeTitle.trim()) request.new_theme = { title: newThemeTitle.trim() };
    }
    onRepreview(request);
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 p-4" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-border bg-white shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="archive-dialog-title">
        <div className="flex items-start justify-between gap-4 border-b border-border bg-[#F4F7F5] p-5">
          <div>
            <p className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[#176B55]">Archive Preview · {jobId}</p>
            <h2 id="archive-dialog-title" className="mt-1 font-display text-2xl font-semibold text-primary">确认调研归档位置</h2>
            <p className="mt-1 text-sm text-secondary">一次调研将作为一个独立 ResearchItem 保存，多篇 Markdown 保持在同一目录内。</p>
          </div>
          <button type="button" onClick={onClose} title="关闭" aria-label="关闭" className="flex h-9 w-9 items-center justify-center rounded-md border border-border bg-white text-secondary hover:text-primary"><X className="h-4 w-4" /></button>
        </div>

        <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_280px]">
          <div className="space-y-5 p-5">
            <FormField label="ResearchItem 名称"><input className="w-full rounded-md border border-border px-3 py-2 text-sm outline-none focus:border-[#176B55]" value={title} onChange={(event) => setTitle(event.target.value)} /></FormField>
            <div>
              <p className="mb-2 text-sm font-medium text-primary">知识库</p>
              <div className="grid grid-cols-2 gap-2">
                <button type="button" onClick={() => setKbMode('existing')} className={cn('rounded-md border px-3 py-2 text-sm', kbMode === 'existing' ? 'border-[#176B55] bg-[#E5F1EC] text-[#176B55]' : 'border-border text-secondary')}>已有知识库</button>
                <button type="button" onClick={() => { setKbMode('new'); setThemeMode(recommendedTheme ? 'new' : 'none'); }} className={cn('rounded-md border px-3 py-2 text-sm', kbMode === 'new' ? 'border-[#176B55] bg-[#E5F1EC] text-[#176B55]' : 'border-border text-secondary')}>新建知识库</button>
              </div>
              {kbMode === 'existing' ? <select className="mt-2 w-full rounded-md border border-border px-3 py-2 text-sm" value={knowledgeBaseId} onChange={(event) => { setKnowledgeBaseId(event.target.value); setThemeId(''); }}><option value="">请选择</option>{knowledgeBases.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select> : <input className="mt-2 w-full rounded-md border border-border px-3 py-2 text-sm" value={newKnowledgeBaseTitle} onChange={(event) => setNewKnowledgeBaseTitle(event.target.value)} placeholder="知识库名称" />}
            </div>
            <div>
              <p className="mb-2 text-sm font-medium text-primary">Theme（可选）</p>
              <div className="grid grid-cols-3 gap-2">
                {(['none', 'existing', 'new'] as const).map((mode) => <button key={mode} type="button" disabled={kbMode === 'new' && mode === 'existing'} onClick={() => setThemeMode(mode)} className={cn('rounded-md border px-2 py-2 text-xs disabled:opacity-40', themeMode === mode ? 'border-[#176B55] bg-[#E5F1EC] text-[#176B55]' : 'border-border text-secondary')}>{mode === 'none' ? '知识库直属' : mode === 'existing' ? '已有 Theme' : '新建 Theme'}</button>)}
              </div>
              {themeMode === 'existing' && <select className="mt-2 w-full rounded-md border border-border px-3 py-2 text-sm" value={themeId} onChange={(event) => setThemeId(event.target.value)}><option value="">请选择</option>{selectedKb?.themes.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select>}
              {themeMode === 'new' && <input className="mt-2 w-full rounded-md border border-border px-3 py-2 text-sm" value={newThemeTitle} onChange={(event) => setNewThemeTitle(event.target.value)} placeholder="Theme 名称" />}
            </div>
            <FormField label="标签"><input className="w-full rounded-md border border-border px-3 py-2 text-sm" value={tags} onChange={(event) => setTags(event.target.value)} placeholder="Harness，Agent，工程实践" /></FormField>
            {Boolean(error) && <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-danger">{error instanceof Error ? error.message : String(error)}</div>}
          </div>

          <aside className="border-t border-border bg-[#F8FAF9] p-5 lg:border-l lg:border-t-0">
            <p className="font-mono text-[10px] font-semibold uppercase tracking-wider text-[#176B55]">即将写入</p>
            <h3 className="mt-2 text-base font-semibold text-primary">{preview.archive?.knowledge_base?.title}</h3>
            <p className="mt-1 text-sm text-secondary">{preview.archive?.theme?.title ? `Theme / ${preview.archive.theme.title}` : '知识库直属'}</p>
            <dl className="mt-4 space-y-2 border-y border-border py-3 text-xs">
              <div className="flex justify-between"><dt className="text-secondary">Markdown</dt><dd className="font-mono text-primary">{preview.archive?.documents?.length || 0}</dd></div>
              <div className="flex justify-between"><dt className="text-secondary">来源</dt><dd className="font-mono text-primary">{preview.archive?.source_count || 0}</dd></div>
              <div className="flex justify-between"><dt className="text-secondary">证据</dt><dd className="font-mono text-primary">{preview.archive?.evidence_count || 0}</dd></div>
              <div className="flex justify-between"><dt className="text-secondary">总览更新</dt><dd className="font-mono text-primary">{preview.changes.filter((item) => item.op === 'update_overview').length}</dd></div>
            </dl>
            <div className="mt-4 space-y-2">{preview.changes.map((item, index) => <div key={`${item.op}-${index}`} className="grid grid-cols-[24px_1fr] gap-2 text-xs"><span className="font-mono text-faint">{String(index + 1).padStart(2, '0')}</span><div><strong className="text-primary">{item.op}</strong><p className="mt-0.5 break-all text-[10px] text-secondary">{String(item.path || item.to || item.title || '')}</p></div></div>)}</div>
          </aside>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border p-4">
          <p className="text-xs text-secondary">修改目标后先重新生成预览；最终确认前后端会再次校验目录版本。</p>
          <div className="flex gap-2">
            <button type="button" onClick={requestPreview} disabled={isPreviewing || (kbMode === 'new' ? !newKnowledgeBaseTitle.trim() : !knowledgeBaseId)} className="inline-flex min-h-9 items-center gap-2 rounded-md border border-border bg-white px-3 text-sm text-primary disabled:opacity-50">{isPreviewing && <Loader2 className="h-4 w-4 animate-spin" />}更新预览</button>
            <button type="button" onClick={onConfirm} disabled={isConfirming} className="inline-flex min-h-9 items-center gap-2 rounded-md bg-[#176B55] px-4 text-sm font-semibold text-white disabled:opacity-50">{isConfirming && <Loader2 className="h-4 w-4 animate-spin" />}确认归档</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StepIcon({ status }: { status: ResearchStepStatus }) {
  if (status === 'running') return <Loader2 className="h-4 w-4 animate-spin text-accent" />;
  if (status === 'succeeded') return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (status === 'failed') return <AlertCircle className="h-4 w-4 text-danger" />;
  if (status === 'waiting_approval') return <Clock3 className="h-4 w-4 text-amber-600" />;
  return <Circle className="h-4 w-4 text-secondary" />;
}

function formatStepMeta(step: ResearchStep) {
  if (step.status === 'running') return 'running';
  if (step.elapsed_ms != null) return formatDuration(step.elapsed_ms);
  return step.status;
}

function displayStepSummary(step: ResearchStep, coverageReason?: string) {
  if (step.id === 'coverage' && coverageReason) return coverageReason;
  return step.summary;
}

function formatDuration(ms: number) {
  if (ms < 1000) return `${ms}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

function formatPlanningFailure(value: unknown) {
  const raw = value instanceof Error ? value.message : String(value || '');
  const normalized = raw.toLowerCase();
  if (normalized.includes('timeout')) return '生成调研计划超时，请稍后重试。';
  if (normalized.includes('rate') || normalized.includes('429')) return '规划模型当前繁忙，请稍后重试。';
  if (normalized.includes('invalid_recon_token')) return '调研主题已经变化，请重新生成计划。';
  if (normalized.includes('missing_recon_context')) return '请先重新核实调研主题，再生成计划。';
  return '暂时无法生成调研计划，请稍后重试。';
}

function ResearchStepTarget({
  id,
  active,
  children,
}: {
  id: string;
  active: boolean;
  children: ReactNode;
}) {
  return (
    <div
      id={id}
      data-research-step-target={id}
      aria-current={active ? 'location' : undefined}
      className={cn(
        'min-w-0 scroll-mt-24 rounded-panel transition-[box-shadow] duration-300',
        active && 'ring-2 ring-inset ring-success/40'
      )}
    >
      {children}
    </div>
  );
}

function SectionStatusBadge({ section }: { section: ResearchSectionMeta }) {
  const statusLabel = section.status === 'empty'
    ? '空缺'
    : section.status === 'limited'
      ? '证据有限'
      : section.status === 'ready'
        ? '可成文'
        : '';
  if (!statusLabel) return null;
  return (
    <span
      className={cn(
        'shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium',
        section.status === 'empty'
          ? 'bg-red-50 text-danger'
          : section.status === 'limited'
            ? 'bg-amber-50 text-amber-800'
            : 'bg-green-50 text-success',
      )}
    >
      {statusLabel}
    </span>
  );
}

function SectionRoleBadges({ section }: { section: ResearchSectionMeta }) {
  return (
    <span className="inline-flex shrink-0 flex-wrap items-center gap-1.5">
      <span
        className={cn(
          'rounded-full border px-2 py-0.5 text-[11px] font-medium',
          section.role === 'core'
            ? 'border-accent/20 bg-accent/10 text-accent'
            : 'border-border bg-white text-secondary',
        )}
      >
        {section.role === 'core' ? '核心章节' : '补充章节'}
      </span>
      <SectionStatusBadge section={section} />
    </span>
  );
}

function StepItem({
  item,
  sectionMeta,
}: {
  item: NonNullable<ResearchStep['items']>[number];
  sectionMeta?: ResearchSectionMeta;
}) {
  if (item === null || typeof item !== 'object') {
    return <div className="rounded-control bg-page p-2 text-xs text-secondary">{String(item)}</div>;
  }
  return (
    <div className="min-w-0 rounded-control bg-page p-3">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-2">
        <p className="min-w-0 flex-1 text-xs font-medium leading-5 text-primary">{String(item.label || 'item')}</p>
        {sectionMeta && <SectionRoleBadges section={sectionMeta} />}
      </div>
      <p className="mt-1 line-clamp-2 text-xs leading-5 text-secondary">{String(item.value ?? '')}</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-panel border border-border bg-white p-4">
      <p className="text-xs text-secondary">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-primary">{String(value ?? 0)}</p>
    </div>
  );
}

function QualityOverviewPanel({
  summary,
  jobStatus,
  canApprove,
  requiresMatrixGapAcknowledgement,
  matrixGapsAcknowledged,
}: {
  summary: QualitySummary;
  jobStatus?: string;
  canApprove: boolean;
  requiresMatrixGapAcknowledgement: boolean;
  matrixGapsAcknowledged: boolean;
}) {
  const toneClass =
    summary.tone === 'pass'
      ? 'border-success/30 bg-green-50 text-success'
      : summary.tone === 'fail'
        ? 'border-danger/30 bg-red-50 text-danger'
        : summary.tone === 'warn'
          ? 'border-warning/30 bg-amber-50 text-amber-900'
          : 'border-border bg-white text-secondary';
  const Icon = summary.tone === 'pass' ? ShieldCheck : ShieldAlert;
  const decisionTitle = summary.tone === 'pass'
    ? canApprove
      ? '质量校验通过，可审批落库'
      : requiresMatrixGapAcknowledgement && !matrixGapsAcknowledged
        ? '质量校验通过，需确认覆盖缺口'
        : '质量校验通过，等待审批条件'
    : summary.title;
  const validationLabel = summary.validationStatus === 'passed'
    ? '引用校验通过'
    : summary.validationStatus === 'failed'
      ? '引用校验失败'
      : '引用校验等待中';
  const gateLabel = summary.gateAvailableCount
    ? `硬门槛 ${summary.gatePassedCount}/${summary.gateAvailableCount}`
    : summary.status === 'waiting' || summary.status === 'pending'
      ? '硬门槛等待中'
      : summary.approvalReady
        ? '硬门槛已通过'
        : '硬门槛未通过';
  const additionalWarnings = summary.warnings.filter(
    (warning) => !(summary.weakMatrixCount && warning.includes('矩阵')),
  );
  const advisories = [
    ...(summary.weakMatrixCount
      ? [{
          id: 'matrix',
          label: `比较矩阵有 ${summary.weakMatrixCount} 个覆盖缺口`,
          detail: '不改变硬质量门结果，但落库前必须查看并确认。',
        }]
      : []),
    ...(summary.questionGapCount
      ? [{
          id: 'coverage',
          label: `${summary.questionGapCount} 个研究问题未完整覆盖`,
          detail: '报告可以交付，但覆盖范围不是完整状态。',
        }]
      : []),
    ...(summary.unsupportedClaimCount
      ? [{
          id: 'unsupported',
          label: `${summary.unsupportedClaimCount} 条结论需要复核`,
          detail: '请在证据台账中核对结论与原始材料。',
        }]
      : []),
    ...(summary.missingInlineRefCount
      ? [{
          id: 'inline-ref',
          label: `${summary.missingInlineRefCount} 处正文引用待补`,
          detail: '证据存在，但正文引用标记尚不完整。',
        }]
      : []),
    ...additionalWarnings.map((warning, index) => ({
      id: `warning-${index}`,
      label: warning,
      detail: '这是透明度提醒，不单独改变硬质量门结果。',
    })),
  ];
  const approvalConditions = [
    { label: '硬性质量检查通过', status: summary.approvalReady },
    { label: '已生成可审阅文档', status: summary.documentCount > 0 },
    { label: '任务已进入审批阶段', status: jobStatus === 'waiting_approval' },
    ...(requiresMatrixGapAcknowledgement
      ? [{ label: '已确认比较矩阵缺口', status: matrixGapsAcknowledged }]
      : []),
  ];
  return (
    <section className={cn('rounded-panel border p-4', toneClass)}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Icon className="h-5 w-5 shrink-0" />
            <h3 className="text-base font-semibold text-primary">{decisionTitle}</h3>
          </div>
          <p className="mt-2 max-w-3xl text-sm text-secondary">{summary.reason}</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-control bg-white/80 px-2 py-1 text-primary">
            {gateLabel}
          </span>
          <span className="rounded-control bg-white/80 px-2 py-1 text-primary">{validationLabel}</span>
          <span className={cn(
            'rounded-control px-2 py-1',
            canApprove ? 'bg-white/80 text-success' : summary.approvalReady ? 'bg-amber-100 text-amber-900' : 'bg-white/80 text-primary',
          )}>
            {canApprove
              ? '可落库'
              : summary.approvalReady
                ? requiresMatrixGapAcknowledgement && !matrixGapsAcknowledged
                  ? '确认缺口后可落库'
                  : '等待审批条件'
                : '暂不审批'}
          </span>
        </div>
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
        <QualityMetric label="证据" value={summary.evidenceCount} />
        <QualityMetric label="来源" value={summary.sourceCount} />
        <QualityMetric label="素材" value={summary.materialCount} />
        <QualityMetric label="文档" value={summary.documentCount} />
        <QualityMetric label="矩阵覆盖" value={summary.matrixTotal ? `${summary.matrixCovered}/${summary.matrixTotal}` : '-'} />
        <QualityMetric label="待复核" value={summary.reviewItemCount} />
      </div>
      <details className="group mt-3 rounded-control border border-white/80 bg-white/55 px-3 py-2 text-primary">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium marker:content-none">
          <span>为什么是这个结果</span>
          <span className="flex items-center gap-2 text-xs font-normal text-secondary">
            硬门槛决定通过，覆盖缺口单独提醒
            <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
          </span>
        </summary>
        <div className="mt-3 grid gap-3 border-t border-border/70 pt-3 lg:grid-cols-3">
          <QualityDecisionColumn
            title="硬门槛"
            description={summary.gateAvailableCount
              ? '任一项失败都会阻止自动审批。'
              : '当前任务未返回逐项判定，此处仅展示总体结果。'}
          >
            {summary.gateChecks.map((check) => (
              <QualityDecisionRow
                key={check.id}
                label={check.label}
                status={check.status}
                blocking
              />
            ))}
          </QualityDecisionColumn>
          <QualityDecisionColumn
            title="非阻断提醒"
            description={summary.matrixSource === 'backend' ? '矩阵采用后端权威口径。' : '矩阵暂由页面证据绑定结果估算。'}
          >
            {advisories.length ? advisories.map((item) => (
              <div key={item.id} className="rounded-control border border-amber-200 bg-amber-50 px-2.5 py-2">
                <p className="flex items-start gap-1.5 text-xs font-medium text-amber-950">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  {item.label}
                </p>
                <p className="mt-1 pl-5 text-[11px] leading-4 text-amber-800">{item.detail}</p>
              </div>
            )) : (
              <QualityDecisionRow label="没有非阻断提醒" status />
            )}
          </QualityDecisionColumn>
          <QualityDecisionColumn
            title="实际审批条件"
            description="全部满足后，审批落库按钮才会启用。"
          >
            {approvalConditions.map((condition) => (
              <QualityDecisionRow
                key={condition.label}
                label={condition.label}
                status={condition.status}
              />
            ))}
          </QualityDecisionColumn>
        </div>
      </details>
    </section>
  );
}

function QualityDecisionColumn({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-control border border-border/70 bg-white/70 p-3">
      <p className="text-xs font-semibold text-primary">{title}</p>
      <p className="mt-1 text-[11px] leading-4 text-secondary">{description}</p>
      <div className="mt-2 space-y-1.5">{children}</div>
    </div>
  );
}

function QualityDecisionRow({
  label,
  status,
  blocking = false,
}: {
  label: string;
  status: boolean | null;
  blocking?: boolean;
}) {
  const RowIcon = status === true ? CheckCircle2 : status === false ? AlertCircle : Circle;
  return (
    <div className={cn(
      'flex items-start gap-1.5 rounded-control px-2 py-1.5 text-xs',
      status === true
        ? 'bg-green-50 text-success'
        : status === false && blocking
          ? 'bg-red-50 text-danger'
          : status === false
            ? 'bg-amber-50 text-amber-900'
            : 'bg-page text-secondary',
    )}>
      <RowIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>{label}</span>
    </div>
  );
}

function QualityMetric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-control border border-white/70 bg-white/75 px-3 py-2">
      <p className="text-[11px] text-secondary">{label}</p>
      <p className="mt-1 text-lg font-semibold text-primary">{String(value ?? 0)}</p>
    </div>
  );
}

function QuestionLoopPanel({
  states,
  reportQuality,
  rejectedEvidenceCount,
}: {
  states: NonNullable<ResearchJobResult['question_states']>;
  reportQuality?: ResearchJobResult['report_quality'];
  rejectedEvidenceCount: number;
}) {
  if (!states.length && !reportQuality) return null;
  const answerStatuses = states.map(questionAnswerStatus);
  const complete = answerStatuses.filter((status) => status === 'complete').length;
  const partial = answerStatuses.filter((status) => status === 'partial').length;
  const uncovered = answerStatuses.filter((status) => status === 'uncovered').length;
  const requiredCriteria = states.flatMap((state) => {
    const criteria = state.criteria || [];
    const hard = criteria.filter((criterion) => !criterion.level || criterion.level === 'hard');
    return hard.length ? hard : criteria;
  });
  const coveredRequiredCriteria = requiredCriteria.filter((criterion) => criterion.status === 'covered').length;
  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-primary">问题闭环</h3>
          <p className="mt-1 text-xs text-secondary">按研究问题追踪检索、证据准入和停止条件</p>
        </div>
        <div className="flex flex-wrap gap-1.5 text-[11px]">
          <span className="rounded-control bg-green-50 px-2 py-1 text-success">完整 {complete}</span>
          <span className="rounded-control bg-amber-50 px-2 py-1 text-amber-800">部分 {partial}</span>
          <span className="rounded-control bg-page px-2 py-1 text-secondary">未覆盖 {uncovered}</span>
          {!!requiredCriteria.length && (
            <span className="rounded-control border border-slate-200 bg-white px-2 py-1 text-primary">
              必需标准 {coveredRequiredCriteria}/{requiredCriteria.length}
            </span>
          )}
          <span className="rounded-control bg-page px-2 py-1 text-secondary">隔离 {rejectedEvidenceCount}</span>
          {reportQuality?.average_score != null && (
            <span className={cn('rounded-control px-2 py-1', reportQuality.status === 'passed' ? 'bg-green-50 text-success' : 'bg-amber-50 text-amber-800')}>
              报告 {reportQuality.average_score}/5
            </span>
          )}
        </div>
      </div>
      {!!states.length && (
        <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {states.map((state) => {
            const answerStatus = questionAnswerStatus(state);
            const statusLabel = answerStatus === 'complete' ? '完整' : answerStatus === 'partial' ? '部分' : '未覆盖';
            return (
              <div key={state.question_id} className="rounded-control border border-border bg-page p-3 text-xs">
                <div className="flex items-start justify-between gap-2">
                  <p className="line-clamp-2 font-medium text-primary">{state.question}</p>
                  <span className={cn('shrink-0', answerStatus === 'complete' ? 'text-success' : answerStatus === 'partial' ? 'text-amber-800' : 'text-secondary')}>
                    {statusLabel}
                  </span>
                </div>
                <p className="mt-2 text-secondary">
                  {state.question_kind || 'research'} · 轮次 {state.rounds_used ?? 0} · 证据 {state.accepted_evidence?.length ?? 0} · 拒绝 {state.rejected_evidence?.length ?? 0}
                </p>
                <p className="mt-1 text-secondary">
                  主张簇 {state.coverage?.claim_cluster_count ?? 0} · 来源 {state.coverage?.source_count ?? 0} · 补检 {state.gap_execution_log?.length ?? 0} · 访问异常 {state.access_issues?.length ?? 0}
                </p>
                <QuestionCoverageCriteria state={state} answerStatus={answerStatus} />
                {(state.coverage?.reason || state.stop_reason) && (
                  <p className="mt-1 line-clamp-2 text-secondary">{state.coverage?.reason || state.stop_reason}</p>
                )}
                {!!state.gap_actions?.length && (
                  <p className="mt-1 line-clamp-2 text-secondary">
                    下一步：{state.gap_actions[0].source_type || state.gap_actions[0].gap_type || state.gap_actions[0].action}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function QuestionCoverageCriteria({
  state,
  answerStatus,
}: {
  state: NonNullable<ResearchJobResult['question_states']>[number];
  answerStatus: 'complete' | 'partial' | 'uncovered';
}) {
  const [expanded, setExpanded] = useState(answerStatus !== 'complete');
  const criteria = state.criteria?.length
    ? state.criteria
    : (state.success_criteria || []).map((description, index) => ({
        criterion_id: `${state.question_id}_criterion_${index + 1}`,
        description,
        level: 'hard',
        status: answerStatus === 'complete' ? 'covered' : 'uncovered',
        evidence_ids: [],
      }));

  if (!criteria.length) return null;

  const required = criteria.filter((criterion) => !criterion.level || criterion.level === 'hard');
  const requiredForSummary = required.length ? required : criteria;
  const covered = requiredForSummary.filter((criterion) => criterion.status === 'covered').length;
  const hasGap = covered < requiredForSummary.length;

  return (
    <div className={cn(
      'mt-3 overflow-hidden rounded-control border bg-white',
      hasGap ? 'border-amber-200' : 'border-emerald-100',
    )}>
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 px-2.5 py-2 text-left hover:bg-page/70 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className={cn(
            'h-2 w-2 shrink-0 rounded-full',
            hasGap ? 'bg-amber-400' : 'bg-emerald-500',
          )} />
          <span className="font-medium text-primary">覆盖标准</span>
          <span className="text-secondary">必需项 {covered}/{requiredForSummary.length}</span>
        </span>
        <ChevronDown className={cn('h-3.5 w-3.5 shrink-0 text-secondary transition-transform', expanded && 'rotate-180')} />
      </button>

      {expanded && (
        <div className="border-t border-border/70 px-2.5 py-1.5">
          {criteria.map((criterion, index) => {
            const status = criterion.status === 'covered'
              ? 'covered'
              : criterion.status === 'partial' || criterion.status === 'weak'
                ? 'partial'
                : 'uncovered';
            const statusLabel = status === 'covered' ? '已覆盖' : status === 'partial' ? '部分覆盖' : '未覆盖';
            const levelLabel = criterion.level === 'soft' ? '补充' : criterion.level === 'optional' ? '可选' : '必需';
            const evidenceCount = criterion.evidence_ids?.length || 0;
            const StatusIcon = status === 'covered' ? CheckCircle2 : status === 'partial' ? AlertCircle : Circle;
            return (
              <div
                key={criterion.criterion_id || `${state.question_id}-${index}`}
                className={cn('flex gap-2 py-2', index > 0 && 'border-t border-border/60')}
              >
                <StatusIcon className={cn(
                  'mt-0.5 h-3.5 w-3.5 shrink-0',
                  status === 'covered' ? 'text-success' : status === 'partial' ? 'text-amber-600' : 'text-slate-400',
                )} />
                <div className="min-w-0 flex-1">
                  <p className="leading-5 text-primary">{criterion.description}</p>
                  <div className="mt-1 flex flex-wrap gap-x-2 text-[10px] text-secondary">
                    <span className={cn(
                      status === 'covered' ? 'text-success' : status === 'partial' ? 'text-amber-700' : 'text-secondary',
                    )}>
                      {statusLabel}
                    </span>
                    <span>{levelLabel}</span>
                    <span>证据 {evidenceCount}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function questionAnswerStatus(
  state: NonNullable<ResearchJobResult['question_states']>[number],
): 'complete' | 'partial' | 'uncovered' {
  const answerStatus = state.coverage?.answer_status;
  if (answerStatus === 'complete' || answerStatus === 'partial' || answerStatus === 'uncovered') {
    return answerStatus;
  }
  if (state.coverage?.coverage_status === 'sufficient' || state.coverage?.coverage_status === 'sufficient_with_gaps' || state.status === 'sufficient') {
    return 'complete';
  }
  const evidenceCount = state.coverage?.evidence_count ?? state.accepted_evidence?.length ?? 0;
  return evidenceCount > 0 ? 'partial' : 'uncovered';
}

function ResearchDiagnosticsPanel({
  documents,
  evidenceDiagnostics,
  qualityGate,
  sectionMetadata,
}: {
  documents: NonNullable<ResearchJobResult['documents']>;
  evidenceDiagnostics?: ResearchJobResult['retrieval'] extends infer R
    ? R extends { evidence_diagnostics?: infer E }
      ? E
      : never
      : never;
  qualityGate?: ResearchJobResult['quality_gate'];
  sectionMetadata: ResearchSectionMeta[];
}) {
  const synthesis = buildSynthesisDiagnostics(documents);
  const evidenceSections = evidenceDiagnostics?.extraction_sections || evidenceDiagnostics?.sections || [];
  const hasDiagnostics = Boolean(
    evidenceDiagnostics ||
      synthesis.documentCount ||
      qualityGate?.checks
  );
  if (!hasDiagnostics) return null;
  const checks = qualityGate?.checks || {};
  const weakSections = arrayFromUnknown(checks.weak_sections);
  const resolvedWeakSections = arrayFromUnknown(checks.resolved_weak_sections);
  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-primary">可观测诊断</h3>
          <p className="mt-1 text-xs text-secondary">
            质量门、证据抽取和文档合成的结构化记录
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5 text-[11px]">
          <span className={cn('rounded-control px-2 py-1', qualityGate?.approval_ready ? 'bg-green-50 text-success' : 'bg-amber-50 text-amber-800')}>
            {qualityGate?.approval_ready ? '可落库' : '待复核'}
          </span>
          <span className="rounded-control bg-page px-2 py-1 text-secondary">
            合成 {formatDuration(synthesis.elapsedMs)}
          </span>
          <span className="rounded-control bg-page px-2 py-1 text-secondary">
            组 {synthesis.groupCount}
          </span>
        </div>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-[1.05fr_1.25fr]">
        <div className="rounded-control border border-border bg-page p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold text-primary">证据抽取</p>
            <span className="text-[11px] text-secondary">
              {evidenceDiagnostics?.extraction_mode || evidenceDiagnostics?.mode || 'diagnostic'}
            </span>
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2">
            <DiagnosticMetric label="模型" value={evidenceDiagnostics?.model_evidence_count ?? 0} />
            <DiagnosticMetric label="Fallback" value={evidenceDiagnostics?.fallback_evidence_count ?? 0} />
            <DiagnosticMetric label="超时" value={evidenceDiagnostics?.timeout_count ?? 0} />
            <DiagnosticMetric label="拒绝" value={evidenceDiagnostics?.rejected_count ?? 0} />
            <DiagnosticMetric label="证据" value={evidenceDiagnostics?.evidence_count ?? 0} />
            <DiagnosticMetric label="章节" value={evidenceDiagnostics?.section_count ?? evidenceSections.length} />
          </div>
          {!!evidenceSections.length && (
            <div className="mt-3 space-y-2">
              {evidenceSections.slice(0, 4).map((section, index) => (
                <div key={section.section_id || section.title || index} className="rounded-control bg-white p-2 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <p className="min-w-0 truncate font-medium text-primary">{section.title || section.section_id}</p>
                    <span className={cn('shrink-0', section.status === 'complete' ? 'text-success' : 'text-secondary')}>
                      {section.status || formatDuration(section.elapsed_ms || 0)}
                    </span>
                  </div>
                  <p className="mt-1 text-secondary">
                    evidence {section.evidence_count ?? section.model_evidence_count ?? 0} · source {section.source_count ?? '-'} · fallback {section.fallback_evidence_count ?? 0}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-control border border-border bg-page p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold text-primary">文档合成</p>
            <span className="text-[11px] text-secondary">
              {synthesis.modelComposedCount} model · {synthesis.fallbackCount} fallback
            </span>
          </div>
          <div className="mt-3 grid grid-cols-4 gap-2">
            <DiagnosticMetric label="文档" value={synthesis.documentCount} />
            <DiagnosticMetric label="证据" value={synthesis.evidenceCount} />
            <DiagnosticMetric label="组" value={synthesis.groupCount} />
            <DiagnosticMetric label="最慢" value={formatDuration(synthesis.maxDocumentElapsedMs)} />
          </div>
          <div className="mt-3 space-y-2">
            {synthesis.slowDocuments.slice(0, 4).map((doc) => {
              const sectionMeta = findResearchSectionMeta(sectionMetadata, doc.sectionId || doc.title);
              return (
              <div key={doc.sectionId || doc.title} className="rounded-control bg-white p-2 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <p className="min-w-0 truncate font-medium text-primary">{doc.title}</p>
                    {sectionMeta && <SectionRoleBadges section={sectionMeta} />}
                  </div>
                  <span className="shrink-0 text-secondary">{formatDuration(doc.elapsedMs)}</span>
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-accent"
                    style={{ width: `${Math.max(8, Math.min(100, doc.elapsedMs / Math.max(synthesis.maxDocumentElapsedMs, 1) * 100))}%` }}
                  />
                </div>
                <p className="mt-1 text-secondary">
                  group {doc.groupCount} · evidence {doc.evidenceCount} · fallback {doc.fallbackCount}
                </p>
                {!!doc.slowGroups.length && (
                  <p className="mt-1 line-clamp-1 text-secondary">
                    慢组：{doc.slowGroups.map((group) => `${group.title || group.mode} ${formatDuration(group.elapsedMs || 0)}`).join(' / ')}
                  </p>
                )}
              </div>
              );
            })}
          </div>
        </div>
      </div>

      {(weakSections.length || resolvedWeakSections.length) && (
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          <DiagnosticList title="仍需补强" items={weakSections} tone="warn" />
          <DiagnosticList title="已由最终文档修复" items={resolvedWeakSections} tone="pass" />
        </div>
      )}
    </section>
  );
}

function DiagnosticMetric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-control bg-white px-2 py-2">
      <p className="text-[11px] text-secondary">{label}</p>
      <p className="mt-1 truncate text-sm font-semibold text-primary">{String(value ?? 0)}</p>
    </div>
  );
}

function DiagnosticList({
  items,
  title,
  tone,
}: {
  title: string;
  items: Array<Record<string, unknown>>;
  tone: 'pass' | 'warn';
}) {
  if (!items.length) return null;
  return (
    <div className={cn('rounded-control border p-3 text-xs', tone === 'pass' ? 'border-green-100 bg-green-50 text-success' : 'border-amber-200 bg-amber-50 text-amber-900')}>
      <p className="font-semibold">{title}</p>
      <div className="mt-2 space-y-1.5">
        {items.slice(0, 4).map((item, index) => (
          <p key={index} className="line-clamp-2">
            {String(item.title || item.section_id || 'section')} · {String(item.reason || 'ok')}
          </p>
        ))}
      </div>
    </div>
  );
}

function arrayFromUnknown(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object')) : [];
}

function GroundingSummary({
  title,
  grounding,
}: {
  title: string;
  grounding?: NonNullable<ResearchJobResult['grounding']>;
}) {
  const item = grounding?.per_document?.find((entry) => entry.title === title);
  if (!item) return null;
  return (
    <div className="mb-2 rounded-control bg-page px-2 py-1 text-xs text-secondary">
      证据引用：{item.matched_evidence_ref_count}/{item.evidence_ref_count} 条匹配 retrieval
    </div>
  );
}

function documentEvidenceIds(document: NonNullable<ResearchJobResult['documents']>[number]) {
  const ids = new Set<string>();
  (document.evidence_table || []).forEach((row) => {
    if (row.id) ids.add(row.id);
  });
  (document.evidence_refs || []).forEach((ref) => {
    if (ref.evidence_id) ids.add(ref.evidence_id);
  });
  return Array.from(ids);
}

function DocumentEvidenceIndex({
  document,
  activeEvidenceId,
  onSelectEvidence,
}: {
  document: NonNullable<ResearchJobResult['documents']>[number];
  activeEvidenceId: string;
  onSelectEvidence: (evidenceId: string, hint?: { subject?: string; axis?: string }) => void;
}) {
  const refs = document.evidence_table?.length
    ? document.evidence_table.map((row) => ({
        id: row.id,
        label: row.id,
        subject: row.subject,
        axis: row.comparison_axis,
      }))
    : (document.evidence_refs || []).map((ref) => ({
        id: ref.evidence_id,
        label: ref.evidence_id,
        subject: '',
        axis: '',
      }));
  const usableRefs = refs.filter((ref) => ref.id);
  if (!usableRefs.length) return null;
  return (
    <div className="mb-2 rounded-control border border-border bg-page p-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] font-medium text-secondary">本文证据</span>
        {usableRefs.slice(0, 18).map((ref) => (
          <button
            key={ref.id}
            type="button"
            data-evidence-id={ref.id}
            onClick={() => onSelectEvidence(ref.id || '', { subject: ref.subject, axis: ref.axis })}
            className={cn(
              'rounded-control border px-1.5 py-0.5 text-[11px] transition-colors',
              activeEvidenceId === ref.id
                ? 'border-accent bg-accent text-white'
                : 'border-border bg-white text-secondary hover:border-accent/40 hover:text-primary'
            )}
            title={[ref.subject, ref.axis].filter(Boolean).join(' · ') || '定位证据'}
          >
            {ref.label}
          </button>
        ))}
        {usableRefs.length > 18 && <span className="text-[11px] text-secondary">+{usableRefs.length - 18}</span>}
      </div>
    </div>
  );
}

function EvidenceAuditPanel({
  document,
  expanded,
  onToggle,
}: {
  document: NonNullable<ResearchJobResult['documents']>[number];
  expanded: boolean;
  onToggle: () => void;
}) {
  const table = document.evidence_table || [];
  const metrics = document.evidence_metrics;
  const paragraphStatus = document.paragraph_validation?.status || metrics?.paragraph_validation_status;
  if (!table.length && !metrics) return null;
  const grouped = groupEvidenceRows(table);
  return (
    <div className="mt-3 border-t border-border pt-3">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full flex-wrap items-center justify-between gap-2 text-left"
      >
        <div>
          <p className="flex items-center gap-1 text-xs font-semibold text-primary">
            证据审计
            {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          </p>
          <p className="mt-1 text-xs text-secondary">
            {metrics?.used_evidence_count ?? table.length}/{metrics?.evidence_count ?? table.length} 已引用 · 来源 {metrics?.source_count ?? '-'} · 域名 {metrics?.domain_count ?? '-'}
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5 text-[11px]">
          <span className={cn('rounded-control px-2 py-1', paragraphStatus === 'passed' ? 'bg-green-50 text-success' : 'bg-amber-50 text-amber-800')}>
            段落 {paragraphStatus || 'pending'}
          </span>
          {!!metrics?.risk_flag_count && (
            <span className="rounded-control bg-amber-50 px-2 py-1 text-amber-800">
              风险 {metrics.risk_flag_count}
            </span>
          )}
          {!!metrics?.weak_question_count && (
            <span className="rounded-control bg-amber-50 px-2 py-1 text-amber-800">
              薄弱问题 {metrics.weak_question_count}
            </span>
          )}
          {!!metrics?.composer?.fallback_count && (
            <span className="rounded-control bg-slate-100 px-2 py-1 text-secondary">
              润色回退 {metrics.composer.fallback_count}
            </span>
          )}
        </div>
      </button>
      {expanded && (
        <div className="mt-3 space-y-2">
          {grouped.slice(0, 6).map((group) => (
            <div key={group.key} className="rounded-control bg-page p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="min-w-0 truncate text-xs font-medium text-primary">{group.title}</p>
                <span className="shrink-0 text-[11px] text-secondary">{group.rows.length} 条</span>
              </div>
              <div className="mt-2 space-y-2">
                {group.rows.slice(0, 4).map((row) => (
                  <div key={row.id || `${row.source_url}-${row.claim}`} className="rounded-control border border-border bg-white p-2 text-xs">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded-control bg-accent/10 px-1.5 py-0.5 font-medium text-accent">{row.id || 'ev'}</span>
                      {row.confidence && <span className="text-secondary">{row.confidence}</span>}
                      {row.used_in_document !== undefined && (
                        <span className={cn(row.used_in_document ? 'text-success' : 'text-amber-700')}>
                          {row.used_in_document ? '已引用' : '未引用'}
                        </span>
                      )}
                      {(row.risk_flags || []).slice(0, 3).map((flag) => (
                        <span key={flag} className="rounded-control bg-amber-50 px-1.5 py-0.5 text-amber-800">{flag}</span>
                      ))}
                    </div>
                    <p className="mt-2 font-medium text-primary">{row.claim}</p>
                    {row.quote && <blockquote className="mt-1 border-l-2 border-border pl-2 text-secondary">{row.quote}</blockquote>}
                    <div className="mt-2 flex min-w-0 items-center gap-1 text-secondary">
                      <ExternalLink className="h-3 w-3 shrink-0" />
                      <span className="truncate">{row.source_title || row.source_domain || row.source_url}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function groupEvidenceRows(rows: NonNullable<NonNullable<ResearchJobResult['documents']>[number]['evidence_table']>) {
  const groups = new Map<string, { key: string; title: string; rows: typeof rows }>();
  rows.forEach((row) => {
    const key = row.matrix_cell_id || matrixKey(row.subject, row.comparison_axis) || row.question_id || row.question || row.section_id || 'evidence';
    const title = row.subject && row.comparison_axis
      ? `${row.subject} · ${row.comparison_axis}`
      : row.question || row.question_id || row.section_title || row.section_id || '关键证据';
    const group = groups.get(key) || { key, title, rows: [] };
    group.rows.push(row);
    groups.set(key, group);
  });
  return Array.from(groups.values());
}

type MatrixEvidenceRow = {
  subject: string;
  axis: string;
  matrixCellId?: string;
  evidenceCount: number;
  sourceCount: number;
  documentCount: number;
  chunkCount: number;
  claims: Array<{
    id?: string;
    claim?: string;
    quote?: string;
    sourceTitle?: string;
    sourceUrl?: string;
    confidence?: string;
    used?: boolean;
  }>;
};

type EvidenceLedgerRow = {
  id?: string;
  subject?: string;
  axis?: string;
  matrixCellId?: string;
  question?: string;
  claim?: string;
  quote?: string;
  sourceTitle?: string;
  sourceUrl?: string;
  confidence?: string;
  used?: boolean;
  riskFlags: string[];
  documentTitle?: string;
};

type EvidenceReviewItem = {
  id: string;
  tone: 'pass' | 'warn' | 'fail';
  label: string;
  detail: string;
};

type ReviewIssue = {
  id: string;
  type: 'unsupported' | 'missing_ref' | 'weak_cell' | 'unused' | 'risk' | 'clear';
  title: string;
  detail: string;
  evidenceId?: string;
  subject?: string;
  axis?: string;
};

type EvidenceReviewModel = {
  checks: EvidenceReviewItem[];
  issues: ReviewIssue[];
};

type SynthesisDiagnosticsModel = {
  documentCount: number;
  elapsedMs: number;
  maxDocumentElapsedMs: number;
  evidenceCount: number;
  groupCount: number;
  fallbackCount: number;
  modelComposedCount: number;
  slowDocuments: Array<{
    sectionId: string;
    title: string;
    elapsedMs: number;
    evidenceCount: number;
    groupCount: number;
    fallbackCount: number;
    slowGroups: Array<{
      title?: string;
      mode?: string;
      elapsedMs?: number;
    }>;
  }>;
};

function buildSynthesisDiagnostics(documents: NonNullable<ResearchJobResult['documents']>): SynthesisDiagnosticsModel {
  const slowDocuments = documents.map((document) => {
    const diagnostics = document.synthesis_diagnostics;
    const composer = diagnostics?.composer || document.evidence_metrics?.composer || document.paragraph_validation?.composer || {};
    const elapsedMs = Number(diagnostics?.elapsed_ms || composer.elapsed_ms_total || 0);
    const slowGroups = (diagnostics?.groups || composer.slow_groups || [])
      .map((group) => ({
        title: group.title,
        mode: group.mode,
        elapsedMs: Number(group.elapsed_ms || 0),
      }))
      .sort((a, b) => b.elapsedMs - a.elapsedMs);
    return {
      sectionId: document.section_id || '',
      title: document.title,
      elapsedMs,
      evidenceCount: Number(diagnostics?.evidence_count || document.evidence_metrics?.evidence_count || document.evidence_table?.length || 0),
      groupCount: Number(diagnostics?.group_count || composer.group_count || 0),
      fallbackCount: Number(composer.fallback_count || 0),
      modelComposedCount: Number(composer.model_composed_count || 0),
      slowGroups,
    };
  });
  slowDocuments.sort((a, b) => b.elapsedMs - a.elapsedMs);
  return {
    documentCount: documents.length,
    elapsedMs: slowDocuments.reduce((sum, doc) => sum + doc.elapsedMs, 0),
    maxDocumentElapsedMs: slowDocuments.reduce((max, doc) => Math.max(max, doc.elapsedMs), 0),
    evidenceCount: slowDocuments.reduce((sum, doc) => sum + doc.evidenceCount, 0),
    groupCount: slowDocuments.reduce((sum, doc) => sum + doc.groupCount, 0),
    fallbackCount: slowDocuments.reduce((sum, doc) => sum + doc.fallbackCount, 0),
    modelComposedCount: slowDocuments.reduce((sum, doc) => sum + doc.modelComposedCount, 0),
    slowDocuments,
  };
}

type MatrixEvidenceModel = {
  subjects: string[];
  axes: string[];
  rows: MatrixEvidenceRow[];
  cells: Record<string, MatrixEvidenceRow>;
  totalEvidence: number;
  weakCellCount: number;
};

function buildMatrixEvidence({
  documents,
  evidenceMap,
  selectedChunks,
  plan,
}: {
  documents: NonNullable<ResearchJobResult['documents']>;
  evidenceMap: NonNullable<ResearchJobResult['evidence_map']>;
  selectedChunks: NonNullable<ResearchJobResult['selected_chunks']>;
  plan?: ResearchJobResult['research_plan'];
}): MatrixEvidenceModel {
  const subjects = new Set<string>((plan?.subjects || []).filter(Boolean));
  const axes = new Set<string>((plan?.comparison_axes || []).filter(Boolean));
  const cells: Record<string, MatrixEvidenceRow> = {};

  const ensureCell = (subject?: string, axis?: string, matrixCellId?: string) => {
    const cleanSubject = String(subject || '').trim();
    const cleanAxis = String(axis || '').trim();
    if (!cleanSubject || !cleanAxis) return null;
    subjects.add(cleanSubject);
    axes.add(cleanAxis);
    const key = matrixKey(cleanSubject, cleanAxis);
    if (!cells[key]) {
      cells[key] = {
        subject: cleanSubject,
        axis: cleanAxis,
        matrixCellId,
        evidenceCount: 0,
        sourceCount: 0,
        documentCount: 0,
        chunkCount: 0,
        claims: [],
      };
    }
    if (matrixCellId && !cells[key].matrixCellId) cells[key].matrixCellId = matrixCellId;
    return cells[key];
  };

  (plan?.matrix_cells || []).forEach((cell) => ensureCell(cell.subject, cell.axis, cell.id));

  documents.forEach((document) => {
    const seenInDocument = new Set<string>();
    (document.evidence_table || []).forEach((row) => {
      const cell = ensureCell(row.subject, row.comparison_axis, row.matrix_cell_id);
      if (!cell) return;
      cell.evidenceCount += 1;
      if (!seenInDocument.has(matrixKey(cell.subject, cell.axis))) {
        cell.documentCount += 1;
        seenInDocument.add(matrixKey(cell.subject, cell.axis));
      }
      cell.claims.push({
        id: row.id,
        claim: row.claim,
        quote: row.quote,
        sourceTitle: row.source_title || row.source_domain,
        sourceUrl: row.source_url,
        confidence: row.confidence,
        used: row.used_in_document,
      });
    });
  });

  evidenceMap.forEach((item) => {
    const cell = ensureCell(item.subject, item.comparison_axis, item.matrix_cell_id);
    if (!cell) return;
    if (!cell.claims.some((claim) => claim.id && claim.id === item.id)) {
      cell.claims.push({
        id: item.id,
        claim: item.claim,
        quote: item.evidence_text,
        sourceTitle: item.source_title,
        sourceUrl: item.source_url,
        confidence: item.confidence,
      });
    }
  });

  selectedChunks.forEach((chunk) => {
    const cell = ensureCell(chunk.subject, chunk.comparison_axis, chunk.matrix_cell_id);
    if (!cell) return;
    cell.chunkCount += 1;
  });

  Object.values(cells).forEach((cell) => {
    const sourceUrls = new Set(cell.claims.map((claim) => claim.sourceUrl).filter(Boolean));
    cell.sourceCount = sourceUrls.size;
  });

  const rows = Object.values(cells).sort((a, b) => {
    const axisCompare = a.axis.localeCompare(b.axis, 'zh-Hans-CN');
    return axisCompare || a.subject.localeCompare(b.subject, 'zh-Hans-CN');
  });

  return {
    subjects: Array.from(subjects),
    axes: Array.from(axes),
    rows,
    cells,
    totalEvidence: rows.reduce((sum, row) => sum + row.evidenceCount, 0),
    weakCellCount: rows.filter((row) => row.evidenceCount < 1 || row.sourceCount < 1).length,
  };
}

function matrixKey(subject?: string, axis?: string) {
  const cleanSubject = String(subject || '').trim();
  const cleanAxis = String(axis || '').trim();
  return cleanSubject && cleanAxis ? `${cleanSubject}::${cleanAxis}` : '';
}

function buildEvidenceLedger({
  documents,
  evidenceMap,
}: {
  documents: NonNullable<ResearchJobResult['documents']>;
  evidenceMap: NonNullable<ResearchJobResult['evidence_map']>;
}): EvidenceLedgerRow[] {
  const rows: EvidenceLedgerRow[] = [];
  const seenIds = new Set<string>();
  documents.forEach((document) => {
    (document.evidence_table || []).forEach((row) => {
      const rowId = row.id || `${document.title}-${row.source_url}-${row.claim}`;
      seenIds.add(rowId);
      rows.push({
        id: row.id,
        subject: row.subject,
        axis: row.comparison_axis,
        matrixCellId: row.matrix_cell_id,
        question: row.question,
        claim: row.claim,
        quote: row.quote,
        sourceTitle: row.source_title || row.source_domain,
        sourceUrl: row.source_url,
        confidence: row.confidence,
        used: row.used_in_document,
        riskFlags: row.risk_flags || [],
        documentTitle: document.title,
      });
    });
  });
  evidenceMap.forEach((item) => {
    const rowId = item.id || `${item.source_url}-${item.claim}`;
    if (seenIds.has(rowId)) return;
    rows.push({
      id: item.id,
      subject: item.subject,
      axis: item.comparison_axis,
      matrixCellId: item.matrix_cell_id,
      claim: item.claim,
      quote: item.evidence_text,
      sourceTitle: item.source_title,
      sourceUrl: item.source_url,
      confidence: item.confidence,
      used: (item.used_by_documents || []).length > 0,
      riskFlags: [],
      documentTitle: item.used_by_documents?.[0],
    });
  });
  return rows;
}

function filterEvidenceLedger(rows: EvidenceLedgerRow[], filters: EvidenceFilterState) {
  return rows.filter((row) => {
    if (filters.subject !== 'all' && row.subject !== filters.subject) return false;
    if (filters.axis !== 'all' && row.axis !== filters.axis) return false;
    if (filters.usage === 'used' && !row.used) return false;
    if (filters.usage === 'unused' && row.used) return false;
    if (filters.usage === 'risk' && !row.riskFlags.length) return false;
    return true;
  });
}

function buildEvidenceReviewModel({
  quality,
  validation,
  matrix,
  ledger,
  documents,
}: {
  quality: QualitySummary;
  validation?: ResearchJobResult['validation'];
  matrix: MatrixEvidenceModel;
  ledger: EvidenceLedgerRow[];
  documents: NonNullable<ResearchJobResult['documents']>;
}): EvidenceReviewModel {
  const unusedRows = ledger.filter((row) => row.id && row.used === false);
  const riskRows = ledger.filter((row) => row.riskFlags.length > 0);
  const metricWeakCells = documents.flatMap((document) => document.evidence_metrics?.weak_matrix_cells || []);
  const weakCellCount = Math.max(matrix.weakCellCount, metricWeakCells.length);
  const checks: EvidenceReviewItem[] = [
    {
      id: 'quality',
      tone: quality.approvalReady ? 'pass' : quality.tone === 'fail' ? 'fail' : 'warn',
      label: '质量门',
      detail: quality.approvalReady ? '通过，可进入审批复核。' : quality.reason,
    },
    {
      id: 'validation',
      tone: validation?.status === 'passed' ? 'pass' : validation?.status === 'failed' ? 'fail' : 'warn',
      label: '引用验收',
      detail: validation?.reason || '等待文档证据引用验收。',
    },
    {
      id: 'matrix',
      tone: weakCellCount ? 'warn' : 'pass',
      label: '矩阵覆盖',
      detail: matrix.rows.length
        ? `${matrix.rows.length - weakCellCount}/${matrix.rows.length} 个对象维度已覆盖。`
        : '当前任务未形成矩阵结构。',
    },
    {
      id: 'inline_refs',
      tone: quality.missingInlineRefCount ? 'warn' : 'pass',
      label: '正文标记',
      detail: quality.missingInlineRefCount
        ? `${quality.missingInlineRefCount} 条证据缺少正文标记。`
        : '正文证据标记已匹配。',
    },
    {
      id: 'unused',
      tone: unusedRows.length ? 'warn' : 'pass',
      label: '未引用证据',
      detail: unusedRows.length ? `${unusedRows.length} 条证据未进入正文。` : '证据均已被正文使用。',
    },
  ];

  const issues: ReviewIssue[] = [];
  (validation?.per_document || []).forEach((document, docIndex) => {
    (document.unsupported_claims || []).forEach((claim, index) => {
      issues.push({
        id: `unsupported-${docIndex}-${index}`,
        type: 'unsupported',
        title: `${document.title || '文档'} · 未支撑结论`,
        detail: [claim.text, claim.reason].filter(Boolean).join(' · '),
      });
    });
    (document.missing_inline_refs || []).forEach((ref, index) => {
      issues.push({
        id: `missing-ref-${docIndex}-${index}`,
        type: 'missing_ref',
        title: `${document.title || '文档'} · 缺少正文标记`,
        detail: ref.claim || ref.source_url || ref.evidence_id || '证据未在正文中标记。',
        evidenceId: ref.evidence_id,
      });
    });
  });
  matrix.rows
    .filter((row) => row.evidenceCount < 1 || row.sourceCount < 1)
    .forEach((row) => {
      issues.push({
        id: `weak-cell-${matrixKey(row.subject, row.axis)}`,
        type: 'weak_cell',
        title: `${row.subject} · ${row.axis}`,
        detail: `证据 ${row.evidenceCount} · 来源 ${row.sourceCount}`,
        subject: row.subject,
        axis: row.axis,
      });
    });
  metricWeakCells.forEach((cell, index) => {
    const issueId = `weak-metric-cell-${cell.matrix_cell_id || index}`;
    if (issues.some((issue) => issue.id === issueId)) return;
    issues.push({
      id: issueId,
      type: 'weak_cell',
      title: `${cell.subject || '对象'} · ${cell.comparison_axis || '维度'}`,
      detail: cell.reason || `证据 ${cell.evidence_count ?? 0} · 来源 ${cell.source_count ?? 0}`,
      subject: cell.subject,
      axis: cell.comparison_axis,
    });
  });
  riskRows.forEach((row) => {
    issues.push({
      id: `risk-${row.id || row.sourceUrl || row.claim}`,
      type: 'risk',
      title: `${row.id || '证据'} · 风险标记`,
      detail: row.riskFlags.join('、'),
      evidenceId: row.id,
      subject: row.subject,
      axis: row.axis,
    });
  });
  unusedRows.slice(0, 6).forEach((row) => {
    issues.push({
      id: `unused-${row.id || row.sourceUrl || row.claim}`,
      type: 'unused',
      title: `${row.id || '证据'} · 未引用`,
      detail: row.claim || row.quote || row.sourceTitle || '证据尚未进入正文。',
      evidenceId: row.id,
      subject: row.subject,
      axis: row.axis,
    });
  });

  if (!issues.length && documents.length) {
    issues.push({
      id: 'clear',
      type: 'clear',
      title: '暂无阻断项',
      detail: '当前文档没有未支撑结论、缺失正文标记或薄弱矩阵格。',
    });
  }

  return { checks, issues };
}

function MatrixEvidencePanel({
  matrix,
  onSelectCell,
}: {
  matrix: MatrixEvidenceModel;
  onSelectCell?: (subject: string, axis: string) => void;
}) {
  if (!matrix.rows.length) return null;
  return (
    <section className="min-w-0 overflow-hidden rounded-panel border border-border bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-primary">对比证据矩阵</h3>
          <p className="mt-1 text-xs text-secondary">
            {matrix.subjects.length} 个对象 · {matrix.axes.length} 个维度 · {matrix.totalEvidence} 条文档证据
          </p>
        </div>
        <span className={cn('rounded-control px-2 py-1 text-xs', matrix.weakCellCount ? 'bg-amber-50 text-amber-800' : 'bg-green-50 text-success')}>
          {matrix.weakCellCount ? `薄弱格子 ${matrix.weakCellCount}` : '矩阵覆盖完整'}
        </span>
      </div>

      <div className="mt-3 max-w-full overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-0 text-left text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 border-b border-border bg-white px-2 py-2 font-medium text-secondary">对象</th>
              {matrix.axes.map((axis) => (
                <th key={axis} className="min-w-[180px] border-b border-border px-2 py-2 font-medium text-secondary">
                  {axis}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.subjects.map((subject) => (
              <tr key={subject}>
                <td className="sticky left-0 z-10 border-b border-border bg-white px-2 py-2 font-medium text-primary">
                  {subject}
                </td>
                {matrix.axes.map((axis) => {
                  const cell = matrix.cells[matrixKey(subject, axis)];
                  return (
                    <td key={`${subject}-${axis}`} className="align-top border-b border-border px-2 py-2">
                      {cell ? (
                        <MatrixCellSummary cell={cell} onSelect={() => onSelectCell?.(subject, axis)} />
                      ) : (
                        <span className="text-secondary">无矩阵数据</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MatrixCellSummary({ cell, onSelect }: { cell: MatrixEvidenceRow; onSelect?: () => void }) {
  const isWeak = cell.evidenceCount < 1 || cell.sourceCount < 1;
  const lead = cell.claims[0];
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full rounded-control border p-2 text-left transition-colors hover:border-accent/40 hover:bg-white',
        isWeak ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-border bg-page text-secondary'
      )}
      title="筛选这个对象与维度的证据"
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className={cn('font-medium', isWeak ? 'text-amber-900' : 'text-primary')}>
          证据 {cell.evidenceCount}
        </span>
        <span>来源 {cell.sourceCount}</span>
        {!!cell.chunkCount && <span>片段 {cell.chunkCount}</span>}
      </div>
      {lead ? (
        <>
          <p className="mt-2 line-clamp-2 font-medium text-primary">{lead.claim || lead.quote}</p>
          <p className="mt-1 truncate">{lead.sourceTitle || lead.sourceUrl}</p>
          {lead.quote && <p className="mt-1 line-clamp-2 border-l-2 border-border pl-2">{lead.quote}</p>}
        </>
      ) : (
        <p className="mt-2">当前没有可展示证据。</p>
      )}
    </button>
  );
}

function ReviewQueueDrawer({
  activeEvidence,
  activeIssueId,
  followupPending,
  runningFollowupIssueIds,
  canRunTargetedFollowup,
  review,
  onSelectIssue,
  onFollowupIssue,
  onClose,
}: {
  activeEvidence?: EvidenceLedgerRow;
  activeIssueId: string;
  followupPending: boolean;
  runningFollowupIssueIds: Set<string>;
  canRunTargetedFollowup: boolean;
  review: EvidenceReviewModel;
  onSelectIssue: (issue: ReviewIssue) => void;
  onFollowupIssue: (issue: ReviewIssue) => void;
  onClose: () => void;
}) {
  const issues = useMemo(() => review.issues.filter((issue) => issue.type !== 'clear'), [review.issues]);
  const initialIndex = Math.max(
    0,
    issues.findIndex((issue) => issue.id === activeIssueId)
  );
  const [queueIndex, setQueueIndex] = useState(initialIndex);
  const currentIssue = issues[queueIndex];

  useEffect(() => {
    if (currentIssue) onSelectIssue(currentIssue);
    // Only re-run when the queue position changes, not on every parent re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queueIndex]);

  return (
    <div className="kw-drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="kw-drawer" role="dialog" aria-modal="true" aria-labelledby="review-drawer-title">
        <div className="kw-modal-header">
          <div>
            <p className="text-xs font-medium text-secondary">复核队列</p>
            <h2 id="review-drawer-title" className="mt-1 text-base font-semibold text-primary">
              {issues.length ? `${queueIndex + 1} / ${issues.length}` : '没有待复核项'}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-control border border-border p-1.5 text-secondary hover:border-primary hover:text-primary"
            aria-label="关闭"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="kw-drawer-body">
          <div className="rounded-control border border-border bg-white p-3">
            <p className="text-xs font-semibold text-primary">审批检查</p>
            <div className="mt-2 space-y-2">
              {review.checks.map((check) => (
                <div key={check.id} className="flex items-start gap-2 text-xs">
                  <span
                    className={cn(
                      'mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full',
                      check.tone === 'pass' ? 'bg-success' : check.tone === 'fail' ? 'bg-danger' : 'bg-warning'
                    )}
                  />
                  <div className="min-w-0">
                    <p className="font-medium text-primary">{check.label}</p>
                    <p className="mt-0.5 line-clamp-2 text-secondary">{check.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {currentIssue ? (
            <div className="mt-3 rounded-control border border-border bg-white p-3">
              <div
                className={cn(
                  'rounded-control border p-3 text-sm',
                  currentIssue.type === 'unsupported'
                    ? 'border-red-100 bg-red-50 text-danger'
                    : currentIssue.type === 'missing_ref' || currentIssue.type === 'weak_cell'
                      ? 'border-amber-200 bg-amber-50 text-amber-900'
                      : 'border-border bg-page text-secondary'
                )}
              >
                <p className="font-medium text-primary">{currentIssue.title}</p>
                <p className="mt-1">{currentIssue.detail}</p>
                <p className="mt-2 text-xs text-secondary">{reviewIssueAction(currentIssue)}</p>
              </div>

              {activeEvidence && activeEvidence.id === currentIssue.evidenceId && (
                <div className="mt-3 rounded-control border border-border bg-page p-3 text-xs">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="rounded-control bg-accent px-1.5 py-0.5 font-medium text-white">
                      {activeEvidence.id || 'ev'}
                    </span>
                    {activeEvidence.subject && <span className="text-primary">{activeEvidence.subject}</span>}
                    {activeEvidence.axis && <span className="text-secondary">{activeEvidence.axis}</span>}
                    <span className={cn(activeEvidence.used ? 'text-success' : 'text-amber-700')}>
                      {activeEvidence.used ? '已引用' : '未引用'}
                    </span>
                  </div>
                  <p className="mt-2 font-medium text-primary">{activeEvidence.claim || activeEvidence.question || '证据结论'}</p>
                  {activeEvidence.quote && (
                    <blockquote className="mt-2 border-l-2 border-border pl-2 text-secondary">{activeEvidence.quote}</blockquote>
                  )}
                  <div className="mt-2 flex min-w-0 items-center gap-1 text-secondary">
                    <ExternalLink className="h-3 w-3 shrink-0" />
                    <span className="truncate">
                      {activeEvidence.sourceTitle || activeEvidence.sourceUrl || activeEvidence.documentTitle || 'source'}
                    </span>
                  </div>
                  {!!activeEvidence.riskFlags.length && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {activeEvidence.riskFlags.map((flag) => (
                        <span key={flag} className="rounded-control bg-amber-50 px-1.5 py-0.5 text-amber-800">
                          {flag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <p className="mt-3 text-sm text-secondary">当前没有需要复核的项目。</p>
          )}
        </div>

        {issues.length > 0 && (
          <div className="kw-modal-footer">
            <button
              type="button"
              onClick={() => setQueueIndex((i) => Math.max(0, i - 1))}
              disabled={queueIndex === 0}
              className="rounded-control border border-border px-3 py-1.5 text-xs font-medium text-primary hover:border-accent/40 hover:text-accent disabled:opacity-40"
            >
              上一条
            </button>
            <button
              type="button"
              onClick={() => setQueueIndex((i) => Math.min(issues.length - 1, i + 1))}
              disabled={queueIndex === issues.length - 1}
              className="rounded-control border border-border px-3 py-1.5 text-xs font-medium text-primary hover:border-accent/40 hover:text-accent disabled:opacity-40"
            >
              下一条
            </button>
            {canRunTargetedFollowup && currentIssue && currentIssue.type !== 'clear' && (
              <button
                type="button"
                onClick={() => onFollowupIssue(currentIssue)}
                disabled={followupPending}
                className="inline-flex items-center gap-1 rounded-control border border-border bg-white px-3 py-1.5 text-xs font-medium text-primary hover:border-accent/40 hover:text-accent disabled:opacity-60"
              >
                {runningFollowupIssueIds.has(currentIssue.id) ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Search className="h-3 w-3" />
                )}
                {runningFollowupIssueIds.has(currentIssue.id) ? '补查中' : '补查'}
              </button>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

function reviewIssueAction(issue: ReviewIssue) {
  if (issue.type === 'unsupported') return '点击定位到证据验收，补证或删改这条强事实。';
  if (issue.type === 'missing_ref') return '点击定位到对应证据，给正文补上 ev 标记。';
  if (issue.type === 'weak_cell') return '点击筛选这个对象/维度，补查来源或增加证据。';
  if (issue.type === 'risk') return '点击查看风险证据，确认来源多样性和引用边界。';
  if (issue.type === 'unused') return '点击查看未引用证据，决定写入正文或移出证据表。';
  return '点击查看详情。';
}

function buildFollowupQuery(issue: ReviewIssue, request?: unknown) {
  const base = String(request || '').trim();
  if (issue.subject && issue.axis) {
    return [issue.subject, issue.axis, issue.detail, '权威资料', '官方文档', 'evidence']
      .filter(Boolean)
      .join(' ');
  }
  if (issue.evidenceId) {
    return [issue.evidenceId, issue.detail, '补充来源', '官方文档', 'evidence'].filter(Boolean).join(' ');
  }
  return [base, issue.title, issue.detail, '补查证据', '权威资料'].filter(Boolean).join(' ');
}

function revisionRejectReasonLabel(reason?: string) {
  switch (reason) {
    case 'fewer_evidence_used':
      return '修改后引用证据变少了';
    case 'content_shrank':
      return '修改后正文变短了';
    case 'revision_fell_back_to_template':
      return '模型改写失败，回退到了模板占位内容';
    case 'empty_revision':
      return '模型返回了空内容';
    default:
      return reason || '未达到质量护栏要求';
  }
}

function EvidenceLedgerPanel({
  ledger,
  matrix,
  filters,
  activeEvidenceId,
  onFiltersChange,
  onSelectEvidence,
}: {
  ledger: EvidenceLedgerRow[];
  matrix: MatrixEvidenceModel;
  filters: EvidenceFilterState;
  activeEvidenceId: string;
  onFiltersChange: (filters: EvidenceFilterState) => void;
  onSelectEvidence: (evidenceId: string) => void;
}) {
  // Deliberately not re-sorting the active row to the top on selection: with the dense
  // single-column list layout (see EvidenceLedgerPanel row rendering below), moving a row
  // would jump the reader away from where they were scanning, which defeats the point of an
  // audit-friendly list -- clicking a row should only expand it in place.
  const filteredRows = filterEvidenceLedger(ledger, filters);
  const updateFilter = (patch: Partial<EvidenceFilterState>) => onFiltersChange({ ...filters, ...patch });
  return (
    <section className="min-w-0 rounded-panel border border-border bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Table2 className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-semibold text-primary">证据台账</h3>
          </div>
          <p className="mt-1 text-xs text-secondary">
            {filteredRows.length}/{ledger.length} 条证据 · 可按对象、维度和引用状态筛选
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <FilterSelect
            label="对象"
            value={filters.subject}
            onChange={(e) => updateFilter({ subject: e.target.value })}
            options={[
              { value: 'all', label: '全部对象' },
              ...matrix.subjects.map((subject) => ({ value: subject, label: subject })),
            ]}
          />
          <FilterSelect
            label="维度"
            value={filters.axis}
            onChange={(e) => updateFilter({ axis: e.target.value })}
            options={[
              { value: 'all', label: '全部维度' },
              ...matrix.axes.map((axis) => ({ value: axis, label: axis })),
            ]}
          />
          <FilterSelect
            label="状态"
            value={filters.usage}
            onChange={(e) => updateFilter({ usage: e.target.value as EvidenceFilterState['usage'] })}
            options={[
              { value: 'all', label: '全部证据' },
              { value: 'used', label: '已引用' },
              { value: 'unused', label: '未引用' },
              { value: 'risk', label: '有风险' },
            ]}
          />
          <button
            type="button"
            onClick={() => onFiltersChange({ subject: 'all', axis: 'all', usage: 'all' })}
            className="inline-flex h-9 items-center gap-1 rounded-control border border-border px-3 text-xs text-secondary hover:bg-page hover:text-primary"
          >
            <Filter className="h-3.5 w-3.5" />
            重置
          </button>
        </div>
      </div>
      {filteredRows.length ? (
        <div className="mt-3 divide-y divide-border overflow-hidden rounded-control border border-border">
          {filteredRows.slice(0, 30).map((row, index) => {
          const active = Boolean(activeEvidenceId && row.id === activeEvidenceId);
          const showQuoteLine = row.quote && !isSameEvidenceText(row.quote, row.claim);
          return (
            <div key={row.id || `${row.sourceUrl}-${index}`} className={cn('transition-colors', active ? 'bg-teal-50' : index % 2 === 1 ? 'bg-page/60' : 'bg-white')}>
              <button
                type="button"
                data-evidence-id={row.id}
                onClick={() => row.id && onSelectEvidence(active ? '' : row.id)}
                className="grid w-full min-w-0 grid-cols-[auto_1fr_auto] items-center gap-3 px-3 py-2.5 text-left text-xs hover:bg-teal-50/60"
              >
                <span className={cn('shrink-0 rounded-control px-1.5 py-0.5 font-medium', active ? 'bg-accent text-white' : 'bg-accent/10 text-accent')}>
                  {row.id || 'ev'}
                </span>
                <span className="min-w-0 truncate font-medium text-primary">
                  {row.claim || row.quote || row.question}
                </span>
                <span className="flex shrink-0 items-center gap-1.5">
                  {row.subject && <span className="hidden text-secondary sm:inline">{row.subject}</span>}
                  <span className={cn('rounded-control px-1.5 py-0.5', row.used ? 'bg-green-50 text-success' : 'bg-amber-50 text-amber-700')}>
                    {row.used ? '已引用' : '未引用'}
                  </span>
                  {!!row.riskFlags.length && (
                    <span className="rounded-control bg-amber-50 px-1.5 py-0.5 text-amber-800">风险 {row.riskFlags.length}</span>
                  )}
                </span>
              </button>
              {active && (
                <div className="border-t border-border/70 bg-white px-3 py-3 text-xs">
                  {showQuoteLine && (
                    <blockquote className="border-l-2 border-border pl-2 text-secondary">{row.quote}</blockquote>
                  )}
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-secondary">
                    <ExternalLink className="h-3 w-3 shrink-0" />
                    {row.sourceUrl ? (
                      <a href={row.sourceUrl} target="_blank" rel="noreferrer" className="truncate text-link hover:underline">
                        {row.sourceTitle || row.sourceUrl}
                      </a>
                    ) : (
                      <span className="truncate">{row.sourceTitle || row.documentTitle || 'source'}</span>
                    )}
                    {row.axis && <span className="rounded-control bg-page px-1.5 py-0.5">{row.axis}</span>}
                    {row.confidence && <span className="rounded-control bg-page px-1.5 py-0.5">置信度 {row.confidence}</span>}
                  </div>
                  {!!row.riskFlags.length && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {row.riskFlags.map((flag) => (
                        <span key={flag} className="rounded-control bg-amber-50 px-1.5 py-0.5 text-amber-800">{flag}</span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
          })}
        </div>
      ) : (
        <p className="mt-3 rounded-control bg-page p-4 text-sm text-secondary">
          {ledger.length ? '当前筛选条件下没有证据。' : '整理证据完成后会在这里显示可筛选的证据台账。'}
        </p>
      )}
      {filteredRows.length > 30 && (
        <p className="mt-3 text-xs text-secondary">已显示前 30 条，继续收窄筛选可查看更具体证据。</p>
      )}
    </section>
  );
}

function isSameEvidenceText(a?: string, b?: string): boolean {
  const normalize = (value?: string) => (value || '').trim().replace(/[。.!！?？\s]+$/g, '').toLowerCase();
  const left = normalize(a);
  const right = normalize(b);
  return Boolean(left) && left === right;
}

function ResearchPlanPanel({
  plan,
  queryPlan,
}: {
  plan?: ResearchJobResult['research_plan'];
  queryPlan?: ResearchJobResult['query_plan'];
}) {
  const sections = plan?.sections || [];
  const queries = queryPlan?.queries || [];
  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-primary">研究框架</h3>
        <span className="text-xs text-secondary">{plan?.strategy || '待生成'}</span>
      </div>
      {sections.length ? (
        <div className="mt-3 space-y-2">
          {sections.slice(0, 6).map((section) => (
            <div key={section.id || section.title} className="rounded-control bg-page p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="truncate text-sm font-medium text-primary">{section.title}</p>
                <span className="text-[11px] uppercase text-secondary">{section.axis}</span>
              </div>
              {!!section.questions?.length && (
                <p className="mt-1 line-clamp-2 text-xs text-secondary">
                  {section.questions.map((question) => typeof question === 'string' ? question : question.text || question.question || '').filter(Boolean).join(' / ')}
                </p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-sm text-secondary">创建任务后会显示横纵研究框架。</p>
      )}
      {!!queries.length && (
        <div className="mt-3 rounded-control border border-border p-3">
          <p className="text-xs font-medium text-primary">查询计划</p>
          <p className="mt-1 text-xs text-secondary">{queries.length} 个查询 · 上限 {queryPlan?.query_limit ?? '-'}</p>
        </div>
      )}
    </section>
  );
}

function ValidationPanel({
  validation,
  qualityReason,
}: {
  validation?: ResearchJobResult['validation'];
  qualityReason?: string;
}) {
  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-primary">证据验收</h3>
        <StatusBadge status={validation?.status || 'pending'} />
      </div>
      <p className="mt-2 text-sm text-secondary">{validation?.reason || qualityReason || '生成文档后会检查引用是否匹配证据地图。'}</p>
      {!!validation?.per_document?.length && (
        <div className="mt-3 grid gap-2">
          {validation.per_document.map((item) => (
            <div key={item.title} className="rounded-control bg-page px-3 py-2 text-xs">
              <div className="flex items-center justify-between gap-3">
                <span className="truncate text-primary">{item.title}</span>
                <span className="text-secondary">
                  {item.matched_evidence_ref_count}/{item.evidence_ref_count} · {Math.round((item.match_ratio || 0) * 100)}%
                </span>
              </div>
              {!!item.unsupported_claims?.length && (
                <div className="mt-2 rounded-control bg-amber-50 p-2 text-amber-900">
                  <p className="font-medium">未引用证据的强事实结论：{item.unsupported_claims.length}</p>
                  {item.unsupported_claims.slice(0, 2).map((claim, index) => (
                    <p key={index} className="mt-1 line-clamp-2">{claim.text}</p>
                  ))}
                </div>
              )}
              {!!item.missing_inline_refs?.length && (
                <div className="mt-2 rounded-control bg-amber-50 p-2 text-amber-900">
                  <p className="font-medium">正文缺少证据标记：{item.missing_inline_refs.length}</p>
                  <p className="mt-1">
                    {item.missing_inline_refs.slice(0, 3).map((ref) => ref.evidence_id).filter(Boolean).join('、')}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      {!!validation?.per_section?.length && (
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {validation.per_section.map((item) => (
            <div key={item.section_id} className="rounded-control bg-page px-3 py-2 text-xs text-secondary">
              <p className="font-medium text-primary">{item.section_id}</p>
              <p className="mt-1">
                证据 {item.evidence_count ?? 0} · 来源 {item.source_count ?? 0} · 域名 {item.domain_count ?? 0} · 问题 {item.question_count ?? 0} · 已引用 {item.used_ref_count ?? 0}
              </p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function EvidenceExtractionPanel({
  diagnostics,
}: {
  diagnostics?: NonNullable<ResearchJobResult['retrieval']>['evidence_diagnostics'];
}) {
  if (!diagnostics) return null;
  const sections = diagnostics.extraction_sections || diagnostics.sections || [];
  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-primary">证据抽取诊断</h3>
          <p className="mt-1 text-xs text-secondary">
            {diagnostics.extraction_mode || diagnostics.mode || 'unknown'} · {diagnostics.elapsed_ms != null ? formatDuration(diagnostics.elapsed_ms) : '-'}
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs text-secondary">
          <span>模型 {diagnostics.model_evidence_count ?? 0}</span>
          <span>Fallback {diagnostics.fallback_evidence_count ?? 0}</span>
          <span>超时 {diagnostics.timeout_count ?? 0}</span>
          <span>拒绝 {diagnostics.rejected_count ?? 0}</span>
        </div>
      </div>
      {!!sections.length && (
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {sections.map((section) => (
            <div key={section.section_id || section.title} className="rounded-control bg-page p-3 text-xs text-secondary">
              <div className="flex items-center justify-between gap-2">
                <p className="truncate font-medium text-primary">{section.title || section.section_id}</p>
                <span className={cn('shrink-0', section.status === 'complete' ? 'text-success' : 'text-amber-700')}>
                  {section.status}
                </span>
              </div>
              <p className="mt-1">
                chunks {section.chunk_count ?? 0} · model {section.model_evidence_count ?? 0} · fallback {section.fallback_evidence_count ?? 0} · rejected {section.rejected_count ?? 0}
              </p>
              {section.elapsed_ms != null && <p className="mt-1">{formatDuration(section.elapsed_ms)}</p>}
              {section.error && <p className="mt-1 line-clamp-2 text-danger">{section.error}</p>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function EvidencePanel({
  title,
  description,
  items,
}: {
  title: string;
  description?: string;
  items: unknown[];
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const visibleItems = items.slice(0, 20);
  return (
    <section className="min-w-0 overflow-hidden rounded-panel border border-border bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-primary">{title}</h3>
          {description && <p className="mt-1 text-xs leading-5 text-secondary">{description}</p>}
        </div>
        <span className="shrink-0 rounded-control bg-page px-2 py-1 text-xs tabular-nums text-secondary">
          {items.length} 条
        </span>
      </div>
      {visibleItems.length ? (
        <div className="mt-3 space-y-2">
          {visibleItems.map((item, index) => {
            const expanded = expandedIndex === index;
            return (
              <button
                key={index}
                type="button"
                onClick={() => setExpandedIndex(expanded ? null : index)}
                className={cn(
                  'block w-full min-w-0 rounded-control p-2 text-left text-xs transition-colors',
                  expanded ? 'bg-teal-50' : 'bg-page hover:bg-teal-50/50'
                )}
              >
                {renderEvidence(item, expanded)}
              </button>
            );
          })}
        </div>
      ) : (
        <p className="mt-3 text-sm text-secondary">暂无数据。</p>
      )}
      {items.length > visibleItems.length && (
        <p className="mt-2 text-xs text-secondary">已显示前 {visibleItems.length} 条，共 {items.length} 条。</p>
      )}
    </section>
  );
}

function renderEvidence(item: unknown, expanded = false) {
  if (item && typeof item === 'object') {
    const data = item as Record<string, unknown>;
    const title = String(data.title || data.fact || data.claim || data.evidence_text || data.url || data.source_url || data.summary || '证据');
    const url = data.url || data.source_url ? String(data.url || data.source_url) : '';
    const evidence = data.evidence_text || data.evidence ? String(data.evidence_text || data.evidence) : '';
    const usedBy = Array.isArray(data.used_by_documents) ? data.used_by_documents.map(String).filter(Boolean) : [];
    const meta = data.content_length
      ? `${String(data.content_length)} chars · ${String(data.content_quality || 'unknown')}`
      : data.section_id
        ? `${String(data.section_id)} · ${String(data.confidence || 'medium')}`
        : '';
    return (
      <div className="min-w-0">
        <p className={cn('font-medium text-primary', expanded ? 'whitespace-pre-wrap break-words' : 'truncate')}>{title}</p>
        {evidence && (
          <p className={cn('mt-1 text-secondary', expanded ? 'whitespace-pre-wrap break-words' : 'line-clamp-2')}>{evidence}</p>
        )}
        {meta && <p className="mt-1 text-secondary">{meta}</p>}
        {'used_by_documents' in data && (
          <p className={cn('mt-1', usedBy.length ? 'text-success' : 'text-amber-700')}>
            {usedBy.length ? `用于：${expanded ? usedBy.join('、') : usedBy.slice(0, 3).join('、')}` : '未被文档引用'}
          </p>
        )}
        {url && (
          expanded ? (
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              onClick={(event) => event.stopPropagation()}
              className="mt-1 block break-words text-link hover:underline"
            >
              {url}
            </a>
          ) : (
            <p className="mt-1 truncate text-secondary">{url}</p>
          )
        )}
      </div>
    );
  }
  return <span>{String(item)}</span>;
}

const CHUNK_STAGE_TONES: Record<ChunkLifecycle['stage'], string> = {
  candidate: 'border-border bg-slate-100 text-secondary',
  evidence: 'border-amber-200 bg-amber-50 text-amber-800',
  writer: 'border-sky-200 bg-sky-50 text-sky-700',
  cited: 'border-green-200 bg-green-50 text-success',
};

function SelectedChunksPanel({
  chunks,
  evidenceMap,
  documents,
  contextDiagnostics,
}: {
  chunks: NonNullable<ResearchJobResult['selected_chunks']>;
  evidenceMap: NonNullable<ResearchJobResult['evidence_map']>;
  documents: NonNullable<ResearchJobResult['documents']>;
  contextDiagnostics?: Array<Record<string, unknown>>;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const visible = chunks.slice(0, 20);
  const lifecycleByChunk = useMemo(
    () => buildChunkLifecycleMap({ chunks, evidenceMap, documents, contextDiagnostics }),
    [chunks, evidenceMap, documents, contextDiagnostics],
  );
  const stageCounts = CHUNK_LIFECYCLE_STAGES.map((stage) => ({
    ...stage,
    count: chunks.filter((chunk, index) => (
      lifecycleByChunk.get(String(chunk.id || `chunk_${index}`))?.stage === stage.id
    )).length,
  }));
  return (
    <section className="rounded-panel border border-border bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-primary">候选网页片段</h3>
          <p className="mt-1 text-xs text-secondary">从抓取正文中筛选的候选内容；状态按它在证据与写作链路中到达的最远步骤显示。</p>
        </div>
        {!!chunks.length && (
          <div className="flex flex-wrap gap-1.5 text-[11px]">
            {stageCounts.filter((stage) => stage.count > 0).map((stage) => (
              <span
                key={stage.id}
                className={cn('rounded-full border px-2 py-1 font-medium', CHUNK_STAGE_TONES[stage.id])}
              >
                {stage.label} {stage.count}
              </span>
            ))}
          </div>
        )}
      </div>
      {visible.length ? (
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {visible.map((chunk, index) => {
            const expanded = expandedIndex === index;
            const lifecycle = lifecycleByChunk.get(String(chunk.id || `chunk_${index}`)) || {
              stage: 'candidate' as const,
              stageIndex: 0,
              evidenceIds: [],
              writerEvidenceIds: [],
              citedEvidenceIds: [],
            };
            const stage = CHUNK_LIFECYCLE_STAGES[lifecycle.stageIndex];
            return (
              <article
                key={chunk.id || index}
                className={cn(
                  'relative min-w-0 rounded-control p-3 text-xs transition-colors',
                  expanded ? 'bg-teal-50' : 'bg-page'
                )}
              >
                <div className="min-w-0 pr-8">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <p className="min-w-0 flex-1 font-medium text-primary">{chunk.question || chunk.section_id}</p>
                    <span className={cn('shrink-0 rounded-full border px-2 py-1 text-[11px] font-medium', CHUNK_STAGE_TONES[lifecycle.stage])}>
                      {stage.label}
                    </span>
                  </div>
                  <div
                    className="mt-2 grid grid-cols-4 gap-1"
                    aria-label={`片段进度：${stage.label}`}
                    title="候选 → 证据 → 写作上下文 → 正文引用"
                  >
                    {CHUNK_LIFECYCLE_STAGES.map((item, stageIndex) => (
                      <span
                        key={item.id}
                        className={cn(
                          'h-1.5 rounded-full',
                          stageIndex <= lifecycle.stageIndex
                            ? lifecycle.stage === 'cited' ? 'bg-green-500' : 'bg-accent'
                            : 'bg-border',
                        )}
                      />
                    ))}
                  </div>
                  <div className={cn('mt-1', !expanded && 'max-h-24 overflow-hidden')}>
                    <MarkdownPreview
                      content={chunk.text || ''}
                      className="text-xs text-secondary prose-headings:my-1 prose-headings:text-sm prose-p:my-1 prose-p:text-secondary prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:my-2 prose-pre:max-h-80 prose-pre:text-xs"
                    />
                  </div>
                  {chunk.source_url ? (
                    <a
                      href={chunk.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className={cn('mt-2 block text-link hover:underline', expanded ? 'break-words' : 'truncate')}
                    >
                      {chunk.source_title || chunk.source_url}
                    </a>
                  ) : (
                    chunk.source_title && <p className="mt-2 truncate text-secondary">{chunk.source_title}</p>
                  )}
                  {expanded && lifecycle.evidenceIds.length > 0 && (
                    <p className="mt-2 break-words text-[11px] text-secondary">
                      关联证据：{lifecycle.evidenceIds.join('、')}
                      {lifecycle.citedEvidenceIds.length > 0
                        ? ` · 正文引用 ${lifecycle.citedEvidenceIds.join('、')}`
                        : lifecycle.writerEvidenceIds.length > 0
                          ? ' · 已提供给 Writer，但正文未引用'
                          : ' · 未进入 Writer 上下文'}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setExpandedIndex(expanded ? null : index)}
                  className="absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded-full text-secondary transition-colors hover:bg-white hover:text-primary"
                  title={expanded ? '收起片段' : '展开片段'}
                  aria-label={expanded ? '收起片段' : '展开片段'}
                >
                  {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>
              </article>
            );
          })}
        </div>
      ) : (
        <p className="mt-3 text-sm text-secondary">调研完成后会显示按研究问题筛选出的候选网页片段。</p>
      )}
      {chunks.length > visible.length && (
        <p className="mt-2 text-xs text-secondary">已显示前 {visible.length} 条，共 {chunks.length} 条。</p>
      )}
    </section>
  );
}
