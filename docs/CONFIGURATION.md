# Deep Research 配置与秘密管理合同

> 状态：`current`
>
> 创建日期：2026-09-07
>
> 最后核对日期：2026-09-14
>
> 当前事实来源：`main.py::load_env_files`、`backend/services/model_service.py`、`backend/research/model_runtime.py`、`scripts/run_feishu_ws_worker.py`、Frontend Vite 配置和 Docker/Compose 配置
>
> 替代文档或归档去向：无

## 1. 文档职责

本文定义当前配置来源、变量职责、继承关系、进程生效边界和秘密处理规则。它不记录任何真实值，也不证明本地或生产环境已经正确配置。

安全模板：

- Backend/CLI/Worker：根目录 `.env.example`
- Frontend：`frontend/.env.example`

## 2. 配置优先级和加载

### 2.1 Backend 与 CLI

`main.py::load_env_files()` 使用 `override=False`，因此优先级是：

1. 启动进程已经存在的环境变量。
2. 找到的本地 `.env` 文件。
3. 代码中的默认值。

当前代码还检查若干历史父目录候选路径。新的本地开发和部署合同统一以仓库根 `.env` 为准，不应继续依赖父目录中的隐式配置。

### 2.2 Feishu Worker

`scripts/run_feishu_ws_worker.py` 独立加载仓库根 `.env`，同样使用 `override=False`。它是独立进程，不共享 Backend 的 `os.environ` 或缓存 Agent。

### 2.3 Frontend

Frontend 使用 Vite `VITE_*` 配置。浏览器可见的任何 `VITE_*` 都不能存放秘密。

| 变量 | 用途 | 安全默认 |
| --- | --- | --- |
| `VITE_USE_MOCK` | 选择 mock adapter | `false` |
| `VITE_DEV_PORT` | Vite 本地端口 | `24681` |
| `VITE_API_PROXY_TARGET` | real 模式开发代理目标 | `http://127.0.0.1:24680` |
| `VITE_API_BASE_URL` | 可选 API 前缀 | 空，优先使用同源 `/api` |

## 3. 配置分组

本文只列稳定、面向使用者的配置组。大量 `RESEARCH_*` 超时、并发、预算和实验开关仍以对应代码与模块文档为准，不把它们全部提升为公共配置合同。

### 3.1 模型角色

| 角色 | 主要变量 | 当前继承边界 |
| --- | --- | --- |
| 主聊天 / KB QA | 系统统一/Preferred provider 优先；未启用时使用 `MODEL_NAME`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_EXTRA_API_KEYS` | 模型名、Base URL 和 Key 池按同一路由原子选择，避免 GLM 模型名落到旧网关 |
| Recall | `RECALL_MODEL_NAME`、`RECALL_BASE_URL`、`RECALL_API_KEY` | 空值可继承主聊天 |
| Dream/Review | `DREAM_MODEL_NAME`、`DREAM_BASE_URL`、`DREAM_API_KEY` | 空值可继承主聊天 |
| Worker | Repo Worker 使用 `WORKER_MODEL_NAME`、`WORKER_BASE_URL`、`WORKER_API_KEY`；Knowledge Manager 优先使用系统统一/Preferred provider | Knowledge Manager 复用系统 GLM 路由和多 Key 轮换，未配置统一通道时回退主聊天；Repo Worker 的独立配置仍未纳入 Models 页面 |
| Evidence | `RESEARCH_EVIDENCE_*` | 专用值优先，未配置时可落到主聊天配置 |
| Writer | `RESEARCH_WRITER_*` | 专用值优先，key/base 可继承 Evidence，再继承主聊天 |
| Ranker | `RESEARCH_RANKER_*` | 独立可选角色，未纳入当前 Models 页面完整合同 |
| Preferred provider | `RESEARCH_UNIFIED_MODEL_CHANNEL_ENABLED`、`RESEARCH_UNIFIED_MODEL_{NAME,BASE_URL,API_KEY,EXTRA_API_KEYS}`；兼容旧 `RESEARCH_PREFERRED_*` | Unified 启用且有 Key 时优先；否则使用显式启用的旧 Preferred。调研模型及用户侧主 Agent、KB QA、Knowledge Manager 共用同一原子路由 |
| MiMo fallback | `MIMO_*` | 显式 fallback 能力，不代表每个角色和错误都会切换 |

Extra key 变量使用逗号分隔。API 只应返回数量、来源、稳定指纹和脱敏摘要，不得返回原始 key。

当 Preferred provider 前面是 gpt-load 一类“由网关管理上游 key 池”的入口时，应设置
`RESEARCH_PREFERRED_GATEWAY_MANAGED=true`。此时本地 key 只是网关凭据，不代表单个上游
账号：运行时不会因为某个上游 key 的额度错误禁用整条网关凭据，失败后的新请求会继续发往
同一网关，由网关选择下一个上游 key。该模式默认允许每个本地网关凭据 64 并发、最长等待
300 秒获取本地凭据、不设置本地限流冷却，并最多执行 2 次
网关级瞬时错误重试；其中包括网关上游路由偶发返回的精确
`404 model is not found`，但不包括普通 404、鉴权失败或参数错误。可分别用
`RESEARCH_PREFERRED_PER_KEY_CONCURRENCY` 和
`RESEARCH_PREFERRED_KEY_ACQUIRE_TIMEOUT_SECONDS`、
`RESEARCH_PREFERRED_GATEWAY_RETRIES` 覆盖。这三个执行策略变量同时作用于 Unified 和旧
Preferred 路由，主 Agent 与调研运行时读取同一组值。直接连接模型供应商时保持 managed
开关关闭；其动态默认依次为 1 并发、30 秒和 0 次网关重试，并通过
`RESEARCH_PREFERRED_RATE_LIMIT_COOLDOWN_SECONDS` 使用默认 20 秒的本地限流冷却。Managed
模式始终跳过本地冷却，让下一请求交给网关重新选上游 Key。`.env.example` 故意将这些覆盖值
留空，避免把直连默认意外固定到 managed 路由。

Evidence 抽取区分单次模型 HTTP 超时和外层编排阈值。`RESEARCH_SECTION_EVIDENCE_MODEL_TIMEOUT_SECONDS`
及 `RESEARCH_QUESTION_EVIDENCE_MODEL_TIMEOUT_SECONDS` 限制单次模型请求；对应的不带 `MODEL` 的
变量只用于记录慢调用。编排阈值到达后会等待已在执行且仍受 HTTP 超时约束的请求结束，不再先判失败、
再让同一请求留在后台继续运行。模板默认分别为 300 秒和 1200 秒。

### 3.2 Embedding 与 LightRAG

面向 Models UI 的变量：

- `EMBEDDING_MODEL`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_DIM`

当前 LightRAG 建索引路径仍可能读取：

- `EMBEDDING_BINDING_HOST`
- `EMBEDDING_BINDING_API_KEY`

这是现行命名不一致，不得宣称 Models 页面测试成功就证明索引进程使用相同凭据。`.env.example` 暂时同时列出两组；后续应统一读取和迁移语义。

### 3.3 搜索与抓取

| 能力 | 主要变量 |
| --- | --- |
| Serper 搜索 | `SERPER_API_KEY`、`SERPER_EXTRA_API_KEYS`、`SERPER_URL` |
| Tavily 抽取 | `TAVILY_EXTRACT_URL` 及调用方所需凭据 |
| Firecrawl | `FIRECRAWL_SCRAPE_ENDPOINT` 及调用方所需凭据 |
| Context7 | `CONTEXT7_BASE_URL` 和 Context7 相关 endpoint/模式变量 |
| Provider chain | `URL_FETCH_PROVIDER_CHAIN`、`SINGLE_URL_FETCH_PROVIDER_CHAIN` |

Provider chain 的 fallback 是抓取策略，不等同于模型供应商 fallback。

### 3.4 飞书

秘密或敏感标识：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_VERIFICATION_TOKEN`
- `FEISHU_ENCRYPT_KEY`
- `FEISHU_DOC_FOLDER_TOKEN`

行为配置：

- `FEISHU_OPEN_API_BASE`
- `FEISHU_ENABLE_REAL_SEND`
- `FEISHU_REPLY_FORMAT`
- `FEISHU_DOC_DELIVERY_ENABLED`
- `FEISHU_DOC_DELIVERY_FOR_FEISHU_CHANNELS`
- `FEISHU_RESEARCH_PROGRESS_PUSH`
- `FEISHU_HTTP_TIMEOUT` 及 progress/revision/watchdog 超时变量

当前代码中飞书文档交付的默认值偏向开启，而 IM 真实发送由另一开关控制。安全模板显式把两类真实写入都设为 `false`，避免出现“不发群消息但仍创建真实飞书文档”的混合状态。

### 3.5 主动推送与运行状态

| 变量 | 当前含义 |
| --- | --- |
| `PUSH_SCHEDULER_ENABLED` | 是否在 Backend 内启动推送线程，代码默认关闭 |
| `PUSH_TICK_INTERVAL_SECONDS` | Scheduler 扫描间隔 |
| `OBSERVABILITY_DIR` | JSONL 运行事件目录；空值使用仓库本地 `logs/observability` |
| `FEISHU_WS_HEARTBEAT_PATH` | 飞书长连接健康心跳文件；空值使用仓库本地 `data/feishu_ws_heartbeat.txt` |
| `RUNTIME_STATE_MONITOR_ENABLED` | 是否运行进程/Job 状态监控 |
| `HARNESS_ENV` | 记录运行环境标签，不自动实施网络隔离 |

每个频道在 `channel_push_config` 中独立保存 `push_time`（24 小时制 `HH:MM`）、IANA `timezone` 和持久化的 `next_push_at`；默认是 `09:00`、`Asia/Shanghai`。调度线程按 `next_push_at` 判断到期，因此短暂重启后会在下一轮扫描补跑已到期项目。当前仍不是任意 cron，也没有可靠 outbox 或分布式调度保证。

`OBSERVABILITY_DIR` 在事件读写时解析，因此 Backend 加载本地 `.env` 后即可生效。飞书 Worker 与 healthcheck 必须接收相同的 `FEISHU_WS_HEARTBEAT_PATH`；当前 Compose 已显式把同一值注入容器环境，避免一端写入自定义路径、另一端仍检查默认路径。

### 3.6 调研、审计与评测

Research Plan 确认 token 使用 `RECON_TOKEN_SECRET` 签名。未配置时，Backend 会在进程启动时生成随机的进程内密钥，适合单进程本地开发，但进程重启后旧 token 会失效，不同 worker 之间也不能互认。生产或多 worker 部署必须为所有 worker 注入同一个高熵值；模板不提供固定回退值，避免公开默认密钥变成可伪造凭据。

- `RESEARCH_CHECKPOINTS_ENABLED`：旧阶段 checkpoint 开关。
- `RESEARCH_DEBUG_DUMP`、`RESEARCH_DEBUG_AGENT_TRACE`、`RESEARCH_DEBUG_REVISION_TRACE`：可能增加本地诊断产物，只在受控环境开启。
- `RESEARCH_TRACE_DIR`、`RESEARCH_TRACE_JSONL`：调研 trace 输出位置或形式。
- `RUN_LIVE_MODEL_TESTS`：真实模型测试总开关之一；当前仍有个别历史测试未正确使用该门。

调试和评测 artifact 可能包含原始问题、来源材料和模型响应，不能因为已做模式化脱敏就视为可公开内容。

## 4. 秘密处理规则

1. `.env*` 默认忽略；只允许 `.env.example` 和 `.env.*.example` 进入 Git。
2. 不读取、展示、复制或提交真实 key、token、密码、Cookie 和 `.env` 备份。
3. 不把秘密放入 `VITE_*`、URL query、Frontend bundle、Markdown、测试 fixture、截图或日志。
4. 健康检查和配置 API 只报告是否配置、角色来源、数量和脱敏指纹。
5. 部署比较配置时只比较变量名或摘要，不打印完整 `.env` diff。
6. 真实模型、飞书和部署操作必须使用显式开关与专用环境，默认保持关闭。

## 5. 当前热更新限制

Models API 当前会：

1. 尝试写项目根 `.env`。
2. 更新当前 Backend 进程的 `os.environ`。
3. 清理受影响的 Backend `app.state` Agent cache。

但当前 Compose 把 `.env` 以只读方式挂载给 Backend 和 Worker。因此：

- 容器内写入可能失败。
- 即使本地 Backend 写入成功，Worker 不会自动更新。
- Ranker、Worker、preferred provider、MiMo fallback 和部分研究角色不在页面的完整管理范围。
- “保存成功”不能解释为“所有进程立即生效”。

当前可靠操作合同是：在宿主机通过受控方式更新根 `.env`，不打印值，然后重启所有依赖该配置的进程并分别执行脱敏检查。跨进程热更新仍属于后续改造范围。

## 6. 安全模板原则

`.env.example`：

- 可以进入 Git，但不能包含真实值或看似真实的 key。
- 只提供安全默认和主要角色，不复制全部实验参数。
- 默认不发送飞书、不创建飞书文档、不启动主动推送、不运行 live tests。
- 新增秘密变量时，必须同时更新 ignore、本文和模板，并确认 API/日志不回显。

Frontend 示例只包含公开、浏览器可见的 `VITE_*` 配置。

## 7. 配置变更验收

每次修改配置系统至少验证：

1. 环境变量优先级和继承路径。
2. Backend、Worker 和 Scheduler 的生效时机。
3. 缺失配置是否安全失败。
4. fallback 是否只在允许的错误类别发生。
5. API、日志和错误消息是否只含脱敏诊断。
6. Docker 只读挂载场景。
7. 示例配置是否仍保持所有真实副作用关闭。

配置事实以当前代码和当次运行环境为准；本文只承担公开合同和已知限制。
