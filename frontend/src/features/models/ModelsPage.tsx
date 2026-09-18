import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Eye, EyeOff, Plus, RefreshCw, Save, Trash2, XCircle } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { StatusBadge } from '@/components/data/StatusBadge';
import { Button } from '@/components/controls/Button';
import { ConfirmDialog, FormField } from '@/components/controls/FormControls';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { EmptyState } from '@/components/data/EmptyState';
import {
  useAddResearchRoleKey,
  useCheckResearchRoleCandidate,
  useCheckResearchRoleKey,
  useDeleteResearchRoleKey,
  useModelConfig,
  useModelCheck,
  useProviders,
  useUpdateModelConfig,
} from '@/hooks';
import { useToast } from '@/components/feedback/Toast';
import {
  ModelCheckKeyResult,
  ModelCheckResult,
  ModelConfigUpdateRequest,
  ResearchKeySource,
  ResearchRole,
  ResearchRoleKey,
  ResearchRoleModelConfig,
} from '@/types';

const ALL_CHECK_TARGETS = [
  'chat_model',
  'recall_model',
  'embedding',
  'dream_model',
  'research_evidence_model',
  'research_writer_model',
];

interface ModelConfigDraft {
  chatModel: string;
  chatBaseUrl: string;
  chatApiKey: string;
  recallFollowsChat: boolean;
  recallModel: string;
  recallBaseUrl: string;
  recallApiKey: string;
  embeddingModel: string;
  embeddingBaseUrl: string;
  embeddingApiKey: string;
  embeddingDim: string;
  dreamModel: string;
  dreamBaseUrl: string;
  dreamApiKey: string;
  evidenceModel: string;
  evidenceBaseUrl: string;
  writerModel: string;
  writerBaseUrl: string;
}

const emptyDraft: ModelConfigDraft = {
  chatModel: '',
  chatBaseUrl: '',
  chatApiKey: '',
  recallFollowsChat: true,
  recallModel: '',
  recallBaseUrl: '',
  recallApiKey: '',
  embeddingModel: '',
  embeddingBaseUrl: '',
  embeddingApiKey: '',
  embeddingDim: '',
  dreamModel: '',
  dreamBaseUrl: '',
  dreamApiKey: '',
  evidenceModel: '',
  evidenceBaseUrl: '',
  writerModel: '',
  writerBaseUrl: '',
};

export function ModelsPage() {
  const config = useModelConfig();
  const providers = useProviders();
  const check = useModelCheck();
  const updateConfig = useUpdateModelConfig();
  const { showToast } = useToast();
  const [draft, setDraft] = useState<ModelConfigDraft>(emptyDraft);
  const [checkResults, setCheckResults] = useState<Record<string, ModelCheckResult>>({});
  const [checkingTargets, setCheckingTargets] = useState<Set<string>>(new Set());

  const runCheck = (targets: string[]) => {
    setCheckingTargets((current) => new Set([...current, ...targets]));
    check.mutate(targets, {
      onSuccess: (res) => {
        setCheckResults((current) => {
          const next = { ...current };
          for (const result of res.results || []) next[result.target] = result;
          return next;
        });
      },
      onSettled: () => {
        setCheckingTargets((current) => {
          const next = new Set(current);
          for (const target of targets) next.delete(target);
          return next;
        });
      },
    });
  };

  useEffect(() => {
    if (!config.data) return;
    setDraft((current) => ({
      ...current,
      chatModel: config.data.chat_model.model,
      recallModel: config.data.recall_model.model,
      recallFollowsChat: config.data.recall_model.model === config.data.chat_model.model,
      embeddingModel: config.data.embedding.model,
      embeddingDim: String(config.data.embedding.dim),
      dreamModel: config.data.dream_model.model,
      evidenceModel: config.data.research_evidence_model.model,
      writerModel: config.data.research_writer_model.model,
    }));
  }, [config.data]);

  const updateDraft = (patch: Partial<ModelConfigDraft>) => {
    setDraft((current) => ({ ...current, ...patch }));
  };

  const buildPayload = (): ModelConfigUpdateRequest | null => {
    const embeddingDim = Number.parseInt(draft.embeddingDim, 10);
    if (!draft.chatModel.trim()) {
      showToast('请填写对话模型名称', 'warning');
      return null;
    }
    if (!draft.embeddingModel.trim()) {
      showToast('请填写嵌入模型名称', 'warning');
      return null;
    }
    if (!Number.isFinite(embeddingDim) || embeddingDim <= 0) {
      showToast('请填写有效的嵌入维度', 'warning');
      return null;
    }
    if (!draft.recallFollowsChat && !draft.recallModel.trim()) {
      showToast('请填写复盘模型名称，或选择跟随对话模型', 'warning');
      return null;
    }

    const payload: ModelConfigUpdateRequest = {
      chat_model: {
        model: draft.chatModel.trim(),
        ...(draft.chatBaseUrl.trim() ? { base_url: draft.chatBaseUrl.trim() } : {}),
        ...(draft.chatApiKey.trim() ? { api_key: draft.chatApiKey.trim() } : {}),
      },
      recall_model: {
        follow_chat_model: draft.recallFollowsChat,
        ...(draft.recallFollowsChat ? {} : { model: draft.recallModel.trim() }),
        ...(draft.recallBaseUrl.trim() ? { base_url: draft.recallBaseUrl.trim() } : {}),
        ...(draft.recallApiKey.trim() ? { api_key: draft.recallApiKey.trim() } : {}),
      },
      embedding: {
        model: draft.embeddingModel.trim(),
        dim: embeddingDim,
        ...(draft.embeddingBaseUrl.trim() ? { base_url: draft.embeddingBaseUrl.trim() } : {}),
        ...(draft.embeddingApiKey.trim() ? { api_key: draft.embeddingApiKey.trim() } : {}),
      },
    };

    // dream_model / research_evidence_model / research_writer_model are optional role
    // blocks: leaving every one of their fields blank omits the block entirely, which
    // keeps whatever is already saved in .env untouched (same "blank = keep current"
    // convention as the base_url/api_key fields above).
    if (draft.dreamModel.trim() || draft.dreamBaseUrl.trim() || draft.dreamApiKey.trim()) {
      if (!draft.dreamModel.trim()) {
        showToast('修改 AutoDream 模型时请填写模型名称', 'warning');
        return null;
      }
      payload.dream_model = {
        model: draft.dreamModel.trim(),
        ...(draft.dreamBaseUrl.trim() ? { base_url: draft.dreamBaseUrl.trim() } : {}),
        ...(draft.dreamApiKey.trim() ? { api_key: draft.dreamApiKey.trim() } : {}),
      };
    }

    if (draft.evidenceModel.trim() || draft.evidenceBaseUrl.trim()) {
      if (!draft.evidenceModel.trim()) {
        showToast('修改证据抽取模型时，请填写模型名称', 'warning');
        return null;
      }
      payload.research_evidence_model = {
        model: draft.evidenceModel.trim(),
        ...(draft.evidenceBaseUrl.trim() ? { base_url: draft.evidenceBaseUrl.trim() } : {}),
      };
    }

    if (draft.writerModel.trim() || draft.writerBaseUrl.trim()) {
      if (!draft.writerModel.trim()) {
        showToast('修改写作模型时，请填写模型名称', 'warning');
        return null;
      }
      payload.research_writer_model = {
        model: draft.writerModel.trim(),
        ...(draft.writerBaseUrl.trim() ? { base_url: draft.writerBaseUrl.trim() } : {}),
      };
    }

    return payload;
  };

  const handleSave = () => {
    const payload = buildPayload();
    if (!payload) return;
    updateConfig.mutate(payload, {
      onSuccess: () => {
        updateDraft({
          chatApiKey: '',
          recallApiKey: '',
          embeddingApiKey: '',
          dreamApiKey: '',
        });
        showToast('配置已保存并生效', 'success');
      },
      onError: (error) => {
        showToast(error instanceof Error ? error.message : '保存配置失败', 'error');
      },
    });
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="模型与接入"
        description="查看模型状态，并保存后端模型接入配置"
        actions={
          <Button
            variant="secondary"
            onClick={() => runCheck(ALL_CHECK_TARGETS)}
            isLoading={checkingTargets.size > 0}
          >
            <RefreshCw className="mr-1.5 h-4 w-4" />
            检查全部
          </Button>
        }
      />

      {config.error && <ErrorPanel error={config.error} onRetry={() => config.refetch()} />}

      <div className="grid gap-4 md:grid-cols-3">
        {config.data && (
          <>
            <ModelCard
              title="对话模型"
              target="chat_model"
              model={config.data.chat_model.model}
              configured={config.data.chat_model.base_url_configured}
              checking={checkingTargets.has('chat_model')}
              result={checkResults.chat_model}
              onCheck={() => runCheck(['chat_model'])}
            />
            <ModelCard
              title="复盘模型"
              target="recall_model"
              model={config.data.recall_model.model}
              configured={config.data.recall_model.base_url_configured}
              checking={checkingTargets.has('recall_model')}
              result={checkResults.recall_model}
              onCheck={() => runCheck(['recall_model'])}
            />
            <ModelCard
              title="嵌入模型"
              target="embedding"
              model={config.data.embedding.model}
              configured={config.data.embedding.base_url_configured}
              dim={config.data.embedding.dim}
              checking={checkingTargets.has('embedding')}
              result={checkResults.embedding}
              onCheck={() => runCheck(['embedding'])}
            />
            <ModelCard
              title="AutoDream 复盘"
              target="dream_model"
              model={config.data.dream_model.model}
              configured={config.data.dream_model.base_url_configured}
              checking={checkingTargets.has('dream_model')}
              result={checkResults.dream_model}
              onCheck={() => runCheck(['dream_model'])}
            />
            <ModelCard
              title="调研·证据抽取"
              target="research_evidence_model"
              model={config.data.research_evidence_model.model}
              configured={config.data.research_evidence_model.base_url_configured && config.data.research_evidence_model.key_count > 0}
              keyCount={config.data.research_evidence_model.key_count}
              checking={checkingTargets.has('research_evidence_model')}
              result={checkResults.research_evidence_model}
              onCheck={() => runCheck(['research_evidence_model'])}
            />
            <ModelCard
              title="调研·写作"
              target="research_writer_model"
              model={config.data.research_writer_model.model}
              configured={config.data.research_writer_model.base_url_configured && config.data.research_writer_model.key_count > 0}
              keyCount={config.data.research_writer_model.key_count}
              checking={checkingTargets.has('research_writer_model')}
              result={checkResults.research_writer_model}
              onCheck={() => runCheck(['research_writer_model'])}
            />
          </>
        )}
      </div>

      <section className="rounded-panel border border-border bg-white p-4">
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-primary">常规模型</h3>
          <p className="mt-1 text-xs text-secondary">
            对话 / 复盘 / 嵌入 / AutoDream 复盘：单条串行调用，各自最多一个 API key。保存后立即生效，无需重启后端。
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
          <ConfigPanel title="对话模型" ready={!!config.data?.chat_model.base_url_configured && !!config.data?.chat_model.api_key_configured}>
            <FormField label="模型名称">
              <ConfigInput value={draft.chatModel} onChange={(value) => updateDraft({ chatModel: value })} placeholder="MODEL_NAME" />
            </FormField>
            <FormField label="Base URL" hint={config.data?.chat_model.base_url_configured ? '已配置；后端不回显具体地址。' : '未配置。'}>
              <ConfigInput
                value={draft.chatBaseUrl}
                onChange={(value) => updateDraft({ chatBaseUrl: value })}
                placeholder={config.data?.chat_model.base_url_configured ? '保持当前地址' : 'https://api.example.com/v1'}
              />
            </FormField>
            <FormField label="API key" hint={config.data?.chat_model.api_key_configured ? '已配置；密钥不会回显。' : '未配置。'}>
              <ConfigInput type="password" value={draft.chatApiKey} onChange={(value) => updateDraft({ chatApiKey: value })} placeholder="输入新密钥" />
            </FormField>
          </ConfigPanel>

          <ConfigPanel title="复盘模型" ready={!!config.data?.recall_model.base_url_configured && !!config.data?.recall_model.api_key_configured}>
            <label className="flex items-center gap-2 rounded-control border border-border px-3 py-2 text-sm text-primary">
              <input
                type="checkbox"
                checked={draft.recallFollowsChat}
                onChange={(e) => updateDraft({ recallFollowsChat: e.target.checked })}
                className="h-4 w-4 accent-teal-700"
              />
              跟随对话模型
            </label>
            <FormField label="模型名称" hint={draft.recallFollowsChat ? '保存时将写入与对话模型一致的复盘模型名称。' : undefined}>
              <ConfigInput
                value={draft.recallFollowsChat ? draft.chatModel : draft.recallModel}
                onChange={(value) => updateDraft({ recallModel: value })}
                disabled={draft.recallFollowsChat}
                placeholder="RECALL_MODEL_NAME"
              />
            </FormField>
            <FormField label="Base URL" hint="留空则复用对话模型的 base_url。">
              <ConfigInput value={draft.recallBaseUrl} onChange={(value) => updateDraft({ recallBaseUrl: value })} placeholder="留空跟随对话模型" />
            </FormField>
            <FormField label="API key" hint="留空则复用对话模型的密钥。">
              <ConfigInput type="password" value={draft.recallApiKey} onChange={(value) => updateDraft({ recallApiKey: value })} placeholder="留空跟随对话模型" />
            </FormField>
          </ConfigPanel>

          <ConfigPanel title="嵌入模型" ready={!!config.data?.embedding.base_url_configured && !!config.data?.embedding.api_key_configured}>
            <FormField label="模型名称">
              <ConfigInput value={draft.embeddingModel} onChange={(value) => updateDraft({ embeddingModel: value })} placeholder="EMBEDDING_MODEL" />
            </FormField>
            <FormField label="维度">
              <ConfigInput value={draft.embeddingDim} onChange={(value) => updateDraft({ embeddingDim: value.replace(/[^\d]/g, '') })} placeholder="1024" />
            </FormField>
            <FormField label="Base URL" hint={config.data?.embedding.base_url_configured ? '已配置；后端不回显具体地址。' : '未配置。'}>
              <ConfigInput
                value={draft.embeddingBaseUrl}
                onChange={(value) => updateDraft({ embeddingBaseUrl: value })}
                placeholder={config.data?.embedding.base_url_configured ? '保持当前地址' : 'https://api.example.com/v1'}
              />
            </FormField>
            <FormField label="API key" hint={config.data?.embedding.api_key_configured ? '已配置；密钥不会回显。' : '未配置。'}>
              <ConfigInput type="password" value={draft.embeddingApiKey} onChange={(value) => updateDraft({ embeddingApiKey: value })} placeholder="输入新密钥" />
            </FormField>
          </ConfigPanel>

          <ConfigPanel title="AutoDream 复盘" ready={!!config.data?.dream_model.base_url_configured && !!config.data?.dream_model.api_key_configured}>
            <FormField label="模型名称" hint="留空整组不修改，沿用当前 DREAM_* 配置（未配置时跟随对话模型）。">
              <ConfigInput value={draft.dreamModel} onChange={(value) => updateDraft({ dreamModel: value })} placeholder="DREAM_MODEL_NAME" />
            </FormField>
            <FormField label="Base URL" hint="留空则复用对话模型的 base_url。">
              <ConfigInput value={draft.dreamBaseUrl} onChange={(value) => updateDraft({ dreamBaseUrl: value })} placeholder="留空跟随对话模型" />
            </FormField>
            <FormField label="API key" hint="留空则复用对话模型的密钥。">
              <ConfigInput type="password" value={draft.dreamApiKey} onChange={(value) => updateDraft({ dreamApiKey: value })} placeholder="留空跟随对话模型" />
            </FormField>
          </ConfigPanel>
        </div>
      </section>

      <section className="rounded-panel border border-border bg-white p-4">
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-primary">调研高频模型</h3>
          <p className="mt-1 text-xs text-secondary">
            证据抽取和写作使用独立的密钥池轮询。已保存的密钥只显示脱敏摘要，可以逐个测试、添加和删除。
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          {config.data && (
            <>
              <ResearchRolePanel
                title="证据抽取模型"
                role="evidence"
                config={config.data.research_evidence_model}
                model={draft.evidenceModel}
                baseUrl={draft.evidenceBaseUrl}
                onModelChange={(value) => updateDraft({ evidenceModel: value })}
                onBaseUrlChange={(value) => updateDraft({ evidenceBaseUrl: value })}
                checkingAll={checkingTargets.has('research_evidence_model')}
                checkResult={checkResults.research_evidence_model}
                onCheckAll={() => runCheck(['research_evidence_model'])}
              />
              <ResearchRolePanel
                title="写作模型"
                role="writer"
                config={config.data.research_writer_model}
                model={draft.writerModel}
                baseUrl={draft.writerBaseUrl}
                onModelChange={(value) => updateDraft({ writerModel: value })}
                onBaseUrlChange={(value) => updateDraft({ writerBaseUrl: value })}
                checkingAll={checkingTargets.has('research_writer_model')}
                checkResult={checkResults.research_writer_model}
                onCheckAll={() => runCheck(['research_writer_model'])}
              />
            </>
          )}
        </div>
      </section>

      <section className="rounded-panel border border-border bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs text-secondary">
            此按钮保存模型名称、Base URL 和常规模型密钥；调研密钥池的添加与删除会即时生效。
          </p>
          <Button variant="secondary" onClick={handleSave} isLoading={updateConfig.isPending} disabled={config.isLoading || updateConfig.isPending}>
            <Save className="mr-1.5 h-4 w-4" />
            保存配置
          </Button>
        </div>
      </section>

      <section className="rounded-panel border border-border bg-white p-4">
        <h3 className="mb-3 text-sm font-semibold text-primary">接入方</h3>
        {providers.data ? (
          <div className="grid gap-2 sm:grid-cols-3">
            {providers.data.providers.map(({ provider, enabled }) => (
              <div key={provider} className="flex items-center justify-between rounded-control border border-border p-3">
                <span className="text-sm text-primary">{provider}</span>
                <StatusBadge status={enabled ? 'ready' : 'missing'} />
              </div>
            ))}
          </div>
        ) : (
          <EmptyState title="加载中" />
        )}
      </section>

      {Object.keys(checkResults).length > 0 && (
        <section className="rounded-panel border border-border bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-primary">连通性检查结果</h3>
          <div className="space-y-2">
            {Object.values(checkResults).map((r) => <CheckResultRow key={r.target} result={r} />)}
          </div>
        </section>
      )}
    </div>
  );
}

function CheckResultRow({ result }: { result: ModelCheckResult }) {
  return (
    <div className="rounded-control border border-border p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <StatusIcon status={result.status} />
          <span className="text-sm text-primary">{result.target}</span>
        </div>
        <span className="text-xs text-secondary">{checkStatusLabel(result.status, result.message)} · {result.latency_ms}ms</span>
      </div>
      {result.keys && result.keys.length > 0 && (
        <div className="mt-2 space-y-1 border-t border-border pt-2 pl-6">
          {result.keys.map((keyResult) => (
            <div key={keyResult.key_id} className="flex items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 text-secondary">
                <StatusIcon status={keyResult.status} />
                <span className="font-mono">{keyResult.masked}</span>
              </div>
              <span className="text-secondary">{checkStatusLabel(keyResult.status)} · {keyResult.latency_ms}ms</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'ok') return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (status === 'partial' || status === 'rate_limited' || status === 'timeout') {
    return <AlertTriangle className="h-4 w-4 text-warning" />;
  }
  return <XCircle className="h-4 w-4 text-danger" />;
}

function checkStatusLabel(status: string, fallback?: string): string {
  const labels: Record<string, string> = {
    ok: '连接正常',
    partial: '部分可用',
    auth_error: '认证失败',
    rate_limited: '触发限流',
    timeout: '连接超时',
    missing_config: '配置缺失',
    unknown: '检查失败',
    error: '检查失败',
  };
  return labels[status] || fallback || status;
}

const KEY_SOURCE_LABELS: Record<ResearchKeySource, string> = {
  dedicated: '独立配置',
  inherited_evidence: '继承证据抽取',
  inherited_chat: '继承对话模型',
  legacy_shared: '旧版共享配置',
};

const BASE_URL_SOURCE_LABELS: Partial<Record<ResearchKeySource, string>> = {
  dedicated: '独立配置',
  inherited_evidence: '继承证据抽取',
  inherited_chat: '继承对话模型',
};

function ResearchRolePanel({
  title,
  role,
  config,
  model,
  baseUrl,
  onModelChange,
  onBaseUrlChange,
  checkingAll,
  checkResult,
  onCheckAll,
}: {
  title: string;
  role: ResearchRole;
  config: ResearchRoleModelConfig;
  model: string;
  baseUrl: string;
  onModelChange: (value: string) => void;
  onBaseUrlChange: (value: string) => void;
  checkingAll: boolean;
  checkResult?: ModelCheckResult;
  onCheckAll: () => void;
}) {
  const { showToast } = useToast();
  const addKey = useAddResearchRoleKey();
  const deleteKey = useDeleteResearchRoleKey();
  const checkCandidate = useCheckResearchRoleCandidate();
  const checkStoredKey = useCheckResearchRoleKey();
  const [adding, setAdding] = useState(false);
  const [showNewKey, setShowNewKey] = useState(false);
  const [newKey, setNewKey] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<ResearchRoleKey | null>(null);
  const [checkingKeyId, setCheckingKeyId] = useState<string | null>(null);
  const [localResults, setLocalResults] = useState<Record<string, ModelCheckKeyResult>>({});

  const checkResultsById = {
    ...(checkResult?.keys || []).reduce<Record<string, ModelCheckKeyResult>>((current, item) => {
      current[item.key_id] = item;
      return current;
    }, {}),
    ...localResults,
  };
  const configured = config.base_url_configured && config.key_count > 0;
  const okCount = checkResult?.keys?.filter((item) => item.status === 'ok').length;
  const status = checkingAll
    ? 'running'
    : !checkResult
      ? configured ? 'active' : 'missing'
      : checkResult.status === 'ok'
        ? 'ready'
        : checkResult.status === 'partial'
          ? 'stale'
          : 'invalid';
  const statusText = checkingAll
    ? '检查中'
    : checkResult && okCount !== undefined
      ? `${okCount}/${config.key_count} 可用`
      : checkResult
        ? checkStatusLabel(checkResult.status, checkResult.message)
      : configured
        ? '已配置'
        : '配置缺失';

  const testPayload = {
    ...(model.trim() ? { model: model.trim() } : {}),
    ...(baseUrl.trim() ? { base_url: baseUrl.trim() } : {}),
  };

  const handleTestAndAdd = () => {
    const apiKey = newKey.trim();
    if (!apiKey) {
      showToast('请输入 API key', 'warning');
      return;
    }
    checkCandidate.mutate(
      { role, payload: { ...testPayload, api_key: apiKey } },
      {
        onSuccess: (result) => {
          if (result.status !== 'ok') {
            showToast(`密钥未添加：${checkStatusLabel(result.status, result.message)}`, 'warning');
            return;
          }
          addKey.mutate(
            { role, apiKey },
            {
              onSuccess: (response) => {
                if (response.key) setLocalResults((current) => ({ ...current, [response.key!.id]: { ...result, key_id: response.key!.id, masked: response.key!.masked } }));
                setNewKey('');
                setAdding(false);
                setShowNewKey(false);
                showToast('密钥已验证并添加', 'success');
              },
              onError: (error) => showToast(error instanceof Error ? error.message : '添加密钥失败', 'error'),
            }
          );
        },
        onError: (error) => showToast(error instanceof Error ? error.message : '测试密钥失败', 'error'),
      }
    );
  };

  const handleCheckKey = (key: ResearchRoleKey) => {
    setCheckingKeyId(key.id);
    checkStoredKey.mutate(
      { role, keyId: key.id, payload: testPayload },
      {
        onSuccess: (result) => setLocalResults((current) => ({ ...current, [key.id]: result })),
        onError: (error) => showToast(error instanceof Error ? error.message : '测试密钥失败', 'error'),
        onSettled: () => setCheckingKeyId(null),
      }
    );
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteKey.mutate(
      { role, keyId: deleteTarget.id },
      {
        onSuccess: () => {
          setLocalResults((current) => {
            const next = { ...current };
            delete next[deleteTarget.id];
            return next;
          });
          setDeleteTarget(null);
          showToast('密钥已删除', 'success');
        },
        onError: (error) => showToast(error instanceof Error ? error.message : '删除密钥失败', 'error'),
      }
    );
  };

  return (
    <div className="rounded-control border border-border bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-primary">{title}</h4>
          <p className="mt-1 text-xs text-secondary">{config.key_count} 个密钥参与轮询</p>
        </div>
        <StatusBadge status={status}>{statusText}</StatusBadge>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <FormField label="模型名称">
          <ConfigInput value={model} onChange={onModelChange} placeholder="MODEL_NAME" />
        </FormField>
        <FormField
          label="Base URL"
          hint={config.base_url
            ? `当前：${config.base_url}${config.base_url_source ? ` · ${BASE_URL_SOURCE_LABELS[config.base_url_source] || ''}` : ''}`
            : '尚未配置'}
        >
          <ConfigInput value={baseUrl} onChange={onBaseUrlChange} placeholder={config.base_url || 'https://api.example.com/v1'} />
        </FormField>
      </div>

      {config.inherits_keys_from && (
        <div className="mt-4 flex gap-2 rounded-control border border-accent/20 bg-accent/5 px-3 py-2.5 text-xs text-primary">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
          <span>当前沿用证据抽取模型的密钥池。添加首个写作密钥后，将切换为独立密钥池。</span>
        </div>
      )}

      <div className="mt-4 border-t border-border pt-4">
        <div className="mb-2 flex items-center justify-between gap-3">
          <div>
            <h5 className="text-sm font-medium text-primary">API 密钥池</h5>
            <p className="mt-0.5 text-xs text-secondary">仅显示前缀和末尾字符，完整密钥不会回显。</p>
          </div>
          <Button variant="secondary" size="sm" onClick={onCheckAll} isLoading={checkingAll} disabled={config.key_count === 0}>
            <RefreshCw className="mr-1 h-3.5 w-3.5" />
            测试全部
          </Button>
        </div>

        <div className="divide-y divide-border rounded-control border border-border">
          {config.keys.length === 0 && <p className="px-3 py-4 text-sm text-secondary">还没有可用密钥。</p>}
          {config.keys.map((key) => {
            const result = checkResultsById[key.id];
            const checking = checkingKeyId === key.id;
            return (
              <div key={key.id} className="grid min-h-14 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 py-2.5">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="font-mono text-sm font-medium text-primary">{key.masked}</span>
                    <span className="text-xs text-secondary">{KEY_SOURCE_LABELS[key.source]}</span>
                    {key.shared && key.source === 'legacy_shared' && <span className="text-xs text-warning">删除会影响旧版共享池</span>}
                  </div>
                  <div className="mt-1 flex items-center gap-1.5 text-xs text-secondary">
                    {checking ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : result ? <StatusIcon status={result.status} /> : <span className="h-1.5 w-1.5 rounded-full bg-faint" />}
                    <span>{checking ? '正在测试' : result ? `${checkStatusLabel(result.status, result.message)} · ${result.latency_ms}ms` : '尚未测试'}</span>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => handleCheckKey(key)}
                    disabled={checking || checkStoredKey.isPending}
                    title="测试此密钥"
                    aria-label={`测试 ${key.masked}`}
                    className="flex h-8 w-8 items-center justify-center rounded-control text-secondary hover:bg-page hover:text-primary disabled:opacity-50"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setDeleteTarget(key)}
                    disabled={!key.deletable || deleteKey.isPending}
                    title={key.deletable ? '删除此密钥' : '请从密钥来源处管理'}
                    aria-label={`删除 ${key.masked}`}
                    className="flex h-8 w-8 items-center justify-center rounded-control text-secondary hover:bg-danger/10 hover:text-danger disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {adding ? (
          <div className="mt-3 rounded-control border border-accent/30 bg-page p-3">
            <label className="text-sm font-medium text-primary">添加 API key</label>
            <div className="relative mt-2">
              <input
                type={showNewKey ? 'text' : 'password'}
                value={newKey}
                onChange={(event) => setNewKey(event.target.value)}
                placeholder="输入完整密钥"
                autoComplete="off"
                className="w-full rounded-control border border-border bg-white px-3 py-2 pr-10 text-sm text-primary placeholder:text-secondary focus:border-accent focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setShowNewKey((current) => !current)}
                title={showNewKey ? '隐藏密钥' : '显示密钥'}
                aria-label={showNewKey ? '隐藏密钥' : '显示密钥'}
                className="absolute right-1 top-1 flex h-8 w-8 items-center justify-center text-secondary hover:text-primary"
              >
                {showNewKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            {config.inherits_keys_from && <p className="mt-2 text-xs text-warning">添加后将停止继承证据抽取密钥池，只使用写作角色的独立密钥。</p>}
            <div className="mt-3 flex justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => { setAdding(false); setNewKey(''); setShowNewKey(false); }}>取消</Button>
              <Button size="sm" onClick={handleTestAndAdd} isLoading={checkCandidate.isPending || addKey.isPending}>测试并添加</Button>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setAdding(true)}
            className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-primary"
          >
            <Plus className="h-4 w-4" />
            添加密钥
          </button>
        )}
      </div>

      <ConfirmDialog
        open={!!deleteTarget}
        title="删除 API 密钥"
        message={deleteTarget?.shared
          ? `确定删除 ${deleteTarget.masked}？这是旧版共享密钥，删除可能同时影响仍在复用该共享池的模型。`
          : `确定删除 ${deleteTarget?.masked || ''}？删除后立即生效；若这是最后一个独立密钥，该角色会恢复到继承配置。`}
        confirmText={deleteKey.isPending ? '删除中' : '删除'}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}

function ConfigPanel({ title, ready, children }: { title: string; ready: boolean; children: React.ReactNode }) {
  return (
    <div className="rounded-control border border-border p-3">
      <div className="mb-3 flex items-center justify-between">
        <h4 className="text-sm font-semibold text-primary">{title}</h4>
        <StatusBadge status={ready ? 'active' : 'missing'}>{ready ? '已配置' : '配置缺失'}</StatusBadge>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function ConfigInput({
  value,
  onChange,
  placeholder,
  type = 'text',
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'password';
  disabled?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className="w-full rounded-control border border-border bg-white px-3 py-2 text-sm text-primary placeholder:text-secondary focus:border-accent focus:outline-none disabled:bg-page disabled:text-secondary"
    />
  );
}

function ModelCard({
  title,
  model,
  configured,
  dim,
  keyCount,
  checking,
  result,
  onCheck,
}: {
  title: string;
  target: string;
  model: string;
  configured: boolean;
  dim?: number;
  keyCount?: number;
  checking?: boolean;
  result?: ModelCheckResult;
  onCheck?: () => void;
}) {
  return (
    <div className="rounded-panel border border-border bg-white p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-primary">{title}</h3>
        <StatusBadge status={configured ? 'active' : 'missing'}>{configured ? '已配置' : '配置缺失'}</StatusBadge>
      </div>
      <p className="mt-3 text-base font-medium text-primary">{model}</p>
      {dim !== undefined && <p className="text-xs text-secondary">维度：{dim}</p>}
      {keyCount !== undefined && <p className="text-xs text-secondary">已配置 {keyCount} 个 key</p>}
      <div className="mt-3 flex items-center justify-between gap-2">
        <Button variant="secondary" size="sm" onClick={onCheck} isLoading={checking}>
          <RefreshCw className="mr-1 h-3.5 w-3.5" />
          检查此项
        </Button>
        {result && !checking && (
          <span className="flex items-center gap-1 text-xs text-secondary">
            <StatusIcon status={result.status} />
            {checkStatusLabel(result.status, result.message)}
          </span>
        )}
      </div>
    </div>
  );
}
