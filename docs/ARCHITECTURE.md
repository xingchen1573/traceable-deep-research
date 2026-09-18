# Deep Research 当前架构

> 状态：`current`
>
> 创建日期：2026-09-07
>
> 最后核对日期：2026-09-18
>
> 当前事实来源：`main.py`、`backend/`、`frontend/`、`tools/`、`scripts/run_feishu_ws_worker.py`、Docker/Compose 配置与相关测试
>
> 替代文档或归档去向：无

## 1. 文档职责

本文说明当前代码实际采用的运行组件、依赖方向、主要数据流和状态边界。它不复制部署步骤，也不把历史设计文档当作现行接口规范。

调研阶段的判断与数据流见[研究管线](./RESEARCH_PIPELINE.md)，归档、索引与范围问答见[知识生命周期](./KNOWLEDGE_LIFECYCLE.md)，运行事件、checkpoint 和隔离 replay 见[可观测性与回放](./OBSERVABILITY_AND_REPLAY.md)。

## 2. 运行组件

| 组件 | 当前入口 | 职责 | 主要状态 |
| --- | --- | --- | --- |
| CLI | `python main.py` | 构建主 Agent，提供终端对话和 HITL | 进程内 Agent、LangGraph `InMemorySaver`、本地运行目录 |
| Backend API | `python -m uvicorn backend.app:app --host 127.0.0.1 --port 24680` | FastAPI 路由、业务服务、Job、知识库、评测、模型和频道 API | `app.state` Agent cache、SQLite、文件工作区、后台线程 |
| Frontend | `frontend/` 下 `npm run dev:real` 或生产 nginx | React/Vite 工作台，通过 `/api` 访问 Backend | 浏览器状态、React Query cache、局部 `localStorage` |
| Feishu Worker | `python scripts/run_feishu_ws_worker.py` | 飞书长连接、消息事件处理、频道 Agent 调度 | 独立进程 Agent cache、共享 SQLite/文件、心跳和进程锁 |
| Push Scheduler | 由 `backend.app` lifespan 调用 | 扫描已启用频道并生成、发送知识卡 | 当前仍是 Backend 内 daemon thread，默认关闭 |
| LightRAG | `LightRAG/` vendored dependency | ResearchItem 级索引和查询 | `lightrag_workspaces/` |

当前 `docker-compose.yml` 运行 `backend`、`feishu-worker`、`frontend` 三个服务。Scheduler 尚未拆成独立服务，因此不能把当前部署描述为四个可靠、独立的工作进程。

## 3. 代码层次

```text
Frontend
  -> frontend/src/api
  -> FastAPI routes
  -> schemas / dependencies
  -> domain services
  -> research core / tools / adapters
  -> local files, SQLite, LightRAG, model and external APIs
```

### 3.1 Frontend

- `frontend/src/app/App.tsx`：页面路由装配。
- `frontend/src/api/`：HTTP adapter 和 endpoint 封装。
- `frontend/src/features/`：按业务页面组织 Research、Knowledge、QA、Channels、Models、Evaluations、Observability 和 Jobs。
- `frontend/src/components/`：共享控件、布局、数据和反馈组件。
- `frontend/src/hooks/`、`stores/`、`types/`：共享状态、轮询和类型合同。

Frontend 不直接访问知识文件、SQLite 或模型供应商。

### 3.2 Backend

- `backend/app.py`：应用装配、环境加载、共享状态和 lifespan。
- `backend/routes/`：HTTP 边界，不应承载长业务流程。
- `backend/schemas/`：请求和响应结构。
- `backend/services/`：Job、知识树、频道、飞书、模型、评测、索引和运行状态等领域服务。
- `backend/research/`：Question-loop 调研、Evidence、Coverage、Writer、验证、质量门、审计和模型运行时。

正式 API 路由由 `backend/routes/__init__.py::register_routes()` 统一注册。

### 3.3 Agent 与工具

`main.py` 目前同时承担 CLI、环境加载、模型构建和三类 Agent 工厂：

- 主 Agent：调研立项、计划确认、Job 调度、审批、归档和通用协作。
- Review Agent：记忆整理和复盘。
- KB QA Agent：面向频道的知识库问答。

`tools/` 提供 Agent 可调用工具。高风险写入通常通过 proposal/preview 和 HITL 收口，但当前 HITL 与部分 Agent 状态仍是进程内状态。

## 4. 主要业务流

### 4.1 多源调研

```text
Web/CLI/Feishu 请求
  -> 澄清和 Plan First
  -> SQLite Job
  -> 后台执行线程
  -> Question Orchestrator
  -> 来源发现与抓取
  -> Evidence 抽取和 Coverage
  -> Evidence-native E+ Writer
  -> 验证与质量门
  -> 用户审批
  -> ResearchItem 原子归档
  -> 独立 LightRAG 索引
  -> Web 或飞书交付
```

公开多源落库调研使用 `question_loop`。`single_url` 是独立快速简报路径；`agentic_retrieval` 只保留为内部可识别的检索变体，不是第二条公开调研管线。

Job 持久化不等于执行可自动续跑。实际后台工作仍由进程内线程承担，进程退出时 Job 可以被标记为中断，但不会自动从任意 checkpoint 恢复。

### 4.2 知识归档和查询

```text
KnowledgeBase
  -> Theme（可选）
  -> ResearchItem
  -> Markdown / HTML / sources / evidence / index status
```

- Markdown 是知识正文正本。
- `_kb.json`、`_theme.json`、`_research.json` 和目录是身份、层级和调研血缘正本。
- 归档使用 preview/confirm，确认时暂存整棵 KnowledgeBase 并交换目录。
- 每个 ResearchItem 使用稳定 `research_id` 和独立 LightRAG workspace。
- KnowledgeScope 可以指向 KnowledgeBase、Theme 或 ResearchItem；上层 Scope 会选择并查询多个 ResearchItem，再聚合回答和引用。
- legacy topic API 与旧式写入仍为兼容路径，不代表新内容应继续使用该模型。

### 4.3 飞书消息

```text
Feishu WS 或 HTTP Webhook
  -> feishu_service.handle_event
  -> channel/session
  -> chat_runtime_service
  -> 主 Agent 或 KB QA Agent
  -> Feishu IM / 文档 / 进度消息
```

HTTP 与 WS 共用业务处理逻辑。当前去重只保存 seen message ID，真实发送、业务成功与去重终态没有 durable inbox/outbox 事务；安全与可靠性改造计划见实施路线图。

### 4.4 主动推送

当前 `push_service`：

1. Scheduler 周期扫描已启用且已绑定知识范围的频道；每个频道保存自己的 wall-clock 时间、IANA timezone 和 `next_push_at`。
2. 按 KnowledgeScope 或 legacy topic 选择 Markdown。
3. 在发送时调用 Writer 模型生成短卡片。
4. 调用飞书发送。
5. 成功发送后记录上次文件、时间并推进持久化的下一次到期时间；短暂重启后可在后续扫描中补跑已到期项目。

它是“按频道计划、按范围轮播摘要”，还不是包含 learner、review event、mastery 和个体复习算法的学习调度系统，也没有可靠 outbox 或分布式调度保证。

## 5. 当前数据正本

| 数据 | 当前位置 | 边界 |
| --- | --- | --- |
| 知识正文和层级 manifest | `knowledge_base/` | 个人本地知识，不进入公共源码仓库 |
| ResearchItem 索引 | `lightrag_workspaces/` | 可重建但成本可能较高，不作为源码 |
| 会话、频道、Job、计划草稿、飞书去重 | `data/conversations.db` | 多服务共享 SQLite，属于运行状态 |
| 调研 checkpoint、模型 trace 和评测 artifact | `data/research_artifacts/` 等 | 只对对应 run 有效，可能包含敏感上下文 |
| 通用观测日志 | `logs/` | 不作为当前代码事实，不进入 Git |
| 记忆与复盘状态 | `memory/` | 个人运行数据，不进入 Git 或镜像 |
| 评测用例合同 | `config/eval_cases/` | 可进入 Git，结果解释需结合版本和哈希 |
| 评测报告 | 开发仓库的 `docs/evaluations/` | 原始记录按样本和日期保存；作品快照提供[评测方法](./EVALUATION.md)与[脱敏验证摘要](./VALIDATION_EVIDENCE.md) |

## 6. 进程与一致性边界

- Backend 和 Feishu Worker 是不同 Python 进程，只通过共享文件、SQLite 和外部 API 交换持久状态。
- `app.state` Agent、LangGraph `InMemorySaver`、pending HITL、progress bridge、线程锁和后台线程不跨进程共享。
- `channel_service`、`job_service`、conversation、plan store 和 Feishu 去重分别管理同一个 `conversations.db` 的表和连接。
- SQLite 已使用 WAL 的局部实现，但当前没有一份覆盖所有服务的统一 migration、`busy_timeout`、lease 和跨进程事务合同。
- Backend 模型配置热更新只影响当前 Backend 进程；独立 Worker 不会自动获得新的 `os.environ` 或 Agent cache。
- Worker 心跳表示 watchdog 最近运行，不证明消息、模型、文档和权限端到端成功。

## 7. 外部依赖边界

- 模型、Embedding、搜索和页面抓取均可能访问外部服务并产生费用或限流。
- 飞书 IM、文档和长连接依赖真实应用权限、租户配置和网络。
- `LightRAG/` 当前是本地 editable vendored 依赖，版本和许可证见根目录 `THIRD_PARTY_NOTICES.md`。
- 默认测试并未通过统一 socket deny 保证完全离线，运行入口和评测层次见 [`EVALUATION.md`](./EVALUATION.md)。

## 8. 不得从架构图推断的事项

- 有代码入口不等于当前环境已配置或在线。
- 有 Job/checkpoint 不等于进程退出后会自动续跑。
- 有 healthcheck 不等于真实业务链路成功。
- 有 fallback 代码不等于所有角色、错误类别和进程都采用同一策略。
- 自动 Gate 通过不等于人工可交付或已完成 Holdout。
- 三服务 Compose 不等于具备分布式队列、事务消息或高可用能力。

上述边界发生改变后，应先以代码、测试和真实运行证据复核，再更新本文日期和结论。
