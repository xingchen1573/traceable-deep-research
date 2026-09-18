# Deep Research 可观测性、审计与隔离重放

> 状态：`current`
>
> 创建日期：2026-09-18
>
> 最后核对日期：2026-09-18
>
> 当前事实来源：`observability.py`、`backend/services/observability_service.py`、`backend/routes/observability.py`、`backend/routes/observability_v1.py`、`backend/research/audit.py`、`backend/research/replay.py`、`backend/research/evidence_native.py` 及相关测试
>
> 替代文档或归档去向：无

## 1. 文档职责

本文说明当前代码如何回答四个不同的问题：一次运行发生了什么、某个 Research Job 经过了哪些阶段、一次模型调用使用了什么输入并得到什么输出，以及如何在不重新执行整个 Job 的前提下隔离重放该调用。

这些能力共享关联标识，但不是一个统一的分布式 Trace 系统：

| 能力 | 主要用途 | 数据粒度 | 当前存储 |
| --- | --- | --- | --- |
| 通用 Observability（可观测性） | 查看对话、工具和后台任务的运行时间线 | event、run、job | 按日 JSONL |
| Research Audit（调研审计） | 追踪调研阶段、模型调用、解析和输入输出血缘 | checkpoint、artifact、model call | 每个 Job 的审计目录 |
| E+ Writing Checkpoint（写作检查点） | 在合同和输入完全一致时复用已完成的写作阶段 | section draft、prevalidation、editorial | 每个 Job 的 E+ checkpoint 目录 |
| Isolated Replay（隔离重放） | 复现或替换一个已记录的模型调用 | 单个 model call | 独立 replay 目录 |

系统总览、进程边界和状态正本见 [ARCHITECTURE.md](./ARCHITECTURE.md)；测试、冻结 replay 和 Live E2E 的证据等级见 [EVALUATION.md](./EVALUATION.md)。

## 2. 数据流

```text
Web / CLI / Feishu / background job
                 |
                 +--> emit_event(...)
                 |      -> logs/observability/<date>.jsonl
                 |      -> run 列表 / run 时间线 / job 聚合轨迹 / 进程内 SSE
                 |
                 +--> Research pipeline
                        -> audit_run(job, run, attempt, trace)
                        -> audit_scope(stage, question, section, round)
                        |
                        +--> stage checkpoint
                        |      -> content-addressed input/output artifacts
                        |      -> previous checkpoint + related model calls
                        |
                        +--> model call
                        |      -> request / response / provider / usage / hashes
                        |      -> model_parse event
                        |
                        +--> E+ writing checkpoint
                               -> verify contract + input + output + document hashes
                               -> reuse valid stage or recompute rejected stage

Saved model call
  -> isolated replay
  -> current provider credential + optional controlled overrides
  -> data/research_replays/<replay_id>/{request.json,result.json}
```

通用 Observability 与 Research Audit 不互相替代。前者适合回答“这次运行在哪一步失败”，后者适合回答“这个调研阶段实际调用了哪个模型、输入输出和解析结果如何关联”。Research model trace 当前也不会自动出现在 Observability 页面中。

## 3. 通用 Observability

[observability.py](../observability.py) 的 `emit_event()` 将业务事件写入按日 JSONL，并把新事件发布给当前进程内的 SSE 订阅者。`session_id`、`run_id` 和 `job_id` 用来关联不同层级：

- `session_id` 或 `thread_id` 表示跨多轮的对话或频道范围；
- `run_id` 表示一次具体执行，默认列表按它聚合；
- `job_id` 表示一个持续时间更长的后台任务，同一 Job 可以跨多个 run。

[Observability Service](../backend/services/observability_service.py) 在查询时扫描时间窗口并现场聚合，不维护第二份持久化索引。当前公开 API 由 [v1 routes](../backend/routes/observability_v1.py) 提供：

| API | 返回内容 |
| --- | --- |
| `GET /api/v1/observability/events` | 按精确字段过滤的事件页 |
| `GET /api/v1/observability/runs` | run 的状态、标题、耗时和事件数摘要 |
| `GET /api/v1/observability/runs/{run_id}` | 一个 run 的顺序时间线 |
| `GET /api/v1/observability/jobs/{job_id}/trace` | 跨 run 拉平后的 Job 时间线 |
| `GET /api/v1/observability/stream` | 当前进程中新事件的 SSE 流 |

查询窗口上限为 90 天，但这是扫描上限，不是自动保留期。SSE 队列只存在于当前进程，断线回放只读取当天的有限事件，也没有持久游标或 `Last-Event-ID` 保证。前端时间线因此是排障视图，不是完整的分布式 span 树，也不是可靠消息队列。

Job 轨迹会优先采用 SQLite Job 注册表的生命周期状态；如果旧 Job 没有通用观测事件，服务会生成一条明确标记为 derived snapshot（派生快照）的状态事件。这能展示已知状态，但不会补造没有发生或没有记录的中间步骤。

## 4. Research Audit

[backend/research/audit.py](../backend/research/audit.py) 为 question-loop 调研提供独立审计域。它由两个开关控制：

```bash
RESEARCH_AUDIT_ENABLED=1
RESEARCH_MODEL_TRACE_ENABLED=1
```

`RESEARCH_AUDIT_ENABLED` 默认关闭；模型 trace 只有在 audit 开启后才可能生效，并在未另行配置时默认随 audit 开启。关闭 audit 时不生成这套审计产物，也不改变模型输入或业务结果。

启用后的主要布局为：

```text
data/research_artifacts/<job_id>/audit/
  checkpoint_head.json
  checkpoints/<sequence>_<stage>.json
  artifacts/sha256_<content_hash>.json
  model_calls.jsonl
```

阶段 checkpoint 记录 Job、run、trace、attempt、question、section、round、转换原因、错误或中断、代码版本、非秘密配置摘要和相关模型调用。较大的阶段输入输出单独写入 artifact，checkpoint 只保存引用，避免在多个阶段重复复制同一对象。

checkpoint、head 和 artifact 使用临时文件加 `os.replace()` 写入；序号和 checkpoint head 更新在本机 POSIX 文件锁下串行化。模型 trace 以 `0600` 模式追加写入 JSONL，并在每次追加后 `fsync`。这些保证面向单机文件系统，不能外推为多主机共享目录或对象存储的一致性协议。

## 5. 内容寻址与 hash 的实际语义

Research artifact 会先转成稳定 JSON、执行模式化脱敏，再计算 SHA-256；摘要同时进入文件名和 artifact 引用。相同的脱敏内容因此得到相同地址。阶段 checkpoint 也保存自身摘要，并通过 `previous_checkpoint_id` 和 `previous_checkpoint_sha256` 形成顺序血缘。

当前不同读取路径的校验强度并不相同：

| 对象 | 写入时记录 | 读取时当前行为 |
| --- | --- | --- |
| Research audit artifact | 内容 SHA-256、内容寻址文件名 | 通用 audit/replay 不会自动重算并拒绝不匹配内容 |
| Research stage checkpoint | checkpoint SHA-256、前序 checkpoint 摘要 | head 和汇总读取不会自动验证整条链 |
| Model call trace | prompt、input、output 和 parsed output hash | JSONL reader 跳过损坏行，但不重新验证各字段 hash |
| E+ writing checkpoint | contract、input、output、完整 document hash | `load()` 会全部重算；任一缺失或不匹配即拒绝复用并重新执行该阶段 |

因此，通用 audit 中的 hash 当前是可比较的内容指纹和受控评测输入，不应描述成签名、不可篡改日志或全局完整性 Gate。真正用于生产阶段复用的是 [EvidenceNativeCheckpointStore](../backend/research/evidence_native.py)：它同时绑定架构版本、Context Manager 版本、Prompt 版本、模型与配置合同、Blueprint/Evidence/Chunk 输入摘要，并对载荷和整个 checkpoint 做读取校验。

相关回归分别覆盖 Research checkpoint 血缘、fail-open、单调用 replay，以及 E+ checkpoint 的篡改拒绝和有效复用，见 [Research audit tests](../tests/test_research_audit.py) 与 [E+ tests](../tests/test_research_evidence_native.py)。

## 6. Model Call Lineage（模型调用血缘）

`audit_run()` 建立 Job 级上下文，`audit_scope()` 再附加 stage、question、section 和 round。并行章节或校验任务必须通过 `submit_with_audit_context()` 显式复制 `ContextVar`；普通 `ThreadPoolExecutor.submit()` 不会自动获得该血缘。

每次被统一 wrapper 覆盖的 provider attempt 都可记录为独立 `model_call`，包含：

- 请求和实际 provider/model、调用参数、耗时、usage、finish reason 和状态；
- 脱敏后的 system prompt、model input、raw response 和异常摘要；
- prompt/input/output hash；
- 调用前 checkpoint 引用，以及 parent call、Job、run、trace 和 scope。

结构化解析另写 `model_parse` 事件。它先从进程内有界索引恢复原调用上下文，未命中时反向扫描当前 Job 的 JSONL，最后才退回解析时的当前 scope。这样 JSON repair 或延迟解析仍能与原调用关联，但这不是跨服务的全局 Trace。

阶段 checkpoint 反向列出该阶段的模型调用，模型调用也指向调用前的 checkpoint，形成可审计的双向链接。覆盖范围仍取决于调用是否经过 Research wrapper；主聊天模型、普通 Worker、replay 自身和未来直接调用 SDK 的路径不自动获得这套记录。

## 7. 单调用隔离 Replay

[backend/research/replay.py](../backend/research/replay.py) 从一个 Job 的 `model_calls.jsonl` 中按 `call_id` 选择调用，或选择指定 stage 中 sequence 最大的一次调用。它重建已保存的 system/model messages，并允许显式替换 model、system prompt、user prompt、temperature 或 timeout。

Replay 使用当前运行环境中的凭据，不从 trace 恢复 secret；模型和 provider 行为也以重放时环境为准。SDK 内部 retry 被关闭，避免与业务层重试叠加。每次 replay 创建独立目录，并保存：

- `request.json`：原调用引用、实验变量、实际重放 messages、input hash 和副作用策略；
- `result.json`：状态、raw response、当前生产 parser 的解析结果、usage、耗时和 output hash。

默认 `json_repair_mode=none`，只运行当前生产 JSON parser。显式选择 `production` 时，首次解析失败后可能再调用一次 repair 模型，因此会增加费用与非确定性。

隔离边界是明确的：replay 只允许模型 provider 调用，不启动搜索、抓取、Chunk、Evidence Extraction 或整个 Job，也不修改数据库、知识库、审批状态、飞书对象、原 trace 或原 checkpoint。它仍会产生外部模型调用，并在本地保存新的 raw response，因此不是零副作用 dry-run。

Replay 也不是逐位复现保证。模型服务、模型别名、当前 parser、当前代码、凭据路由和被替换的实验变量都可能发生漂移；保存输入经过脱敏，也可能与 provider 当时实际收到的字节不完全相同。当前单调用 replay 在读取源 JSONL 时不会主动验证 prompt/input hash，做冻结对照实验时还应采用 [EVALUATION.md](./EVALUATION.md) 所定义的 dataset、artifact 和运行合同。

## 8. Fail-open（失败开放）边界

可观测性不能改变业务调用，这是当前实现的明确取舍：

- 通用 `emit_event()` 写入或发布失败时直接返回；
- artifact 和 stage checkpoint 写入失败时返回空引用；
- Research trace 调用前准备失败时仍执行原模型调用；
- 模型成功后的 trace 写入失败时仍返回原响应；
- 模型本身抛出的异常仍原样向上传播，审计异常不能覆盖它；
- trace reader 遇到单行损坏 JSON 时跳过该行。

代价是业务成功不代表审计完整。当前没有独立的 audit completeness Gate、缺口补写或审计存储告警，因此“没有 trace”既可能表示未启用或未覆盖，也可能表示记录失败，不能直接推断“没有发生模型调用”。

E+ writing checkpoint 是执行路径中的缓存复用机制，不属于上述 audit fail-open 包装。其 `load()` 会把损坏或合同不匹配视为 cache miss（缓存未命中）并重新计算；其 `save()` 文件错误当前可以向上传播。两类 checkpoint 不能用同一个失败策略概括。

## 9. 脱敏与本地数据边界

两条记录链的处理强度不同：

- 通用 Observability 的 `_sanitize()` 只限制字符串长度、集合条数和嵌套深度，不按 secret 键或环境变量值脱敏；调用方不得把凭据、Cookie 或完整敏感正文写入事件。
- Research Audit 会处理敏感键、Bearer 文本、常见 `key=value`、URL secret query 和当前环境中的已知 secret 值，并移除内部 trace 字段。
- Research Audit 的规则不是自然语言 DLP。用户请求、网页材料、prompt 和模型 response 仍可能含私人或机密内容。
- Replay request 继承已保存的脱敏输入，但 replay response 仍是新生成的原文。

`logs/`、`data/research_artifacts/` 和 `data/research_replays/` 都是本地运行数据，不属于源码或公开评测样本。当前没有统一加密、应用级访问控制、保留期、容量配额、自动清理或删除 API；除模型 trace 的 `0600` 文件模式外，其他文件权限依赖宿主系统和 umask。部署者应限制目录访问，并按自己的数据政策清理。

## 10. 恢复能力与非目标

必须区分“有 checkpoint”“能复用局部结果”和“整个 Job 能自动续跑”：

- Research audit checkpoint 用于审计、诊断和构造受控复验输入，本身不驱动执行恢复。
- question-loop 的业务阶段快照可用于检查和定向重试输入，但没有解释 Python 调用栈或自动选择恢复点的执行器。
- E+ writing checkpoint 只在合同和输入 hash 完全一致时复用 section draft、section prevalidation 或 editorial；它不恢复检索、Evidence Extraction、Job owner、线程或审批流程。
- SQLite 中保存 Job 状态和 retry lineage，不等于后台线程跨进程存活。进程退出后，Job 可以被识别为 interrupted，但不会从最近 audit checkpoint 自动继续。
- SSE 重连和 Observability 时间线只恢复视图，不恢复任务。
- 单调用 replay 创建的是新实验结果，不会把结果写回原 Job 或让原 Job 继续。

因此，当前系统提供的是“可观察、可追溯、可隔离复验，以及 E+ 写作阶段的受合同约束复用”，不是任意节点的 durable workflow resume（持久化工作流续跑）。这一边界与 [ARCHITECTURE.md](./ARCHITECTURE.md) 中的进程和状态说明一致。
