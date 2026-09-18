# Deep Research 知识生命周期

> 状态：`current`
>
> 创建日期：2026-09-18
>
> 最后核对日期：2026-09-18
>
> 当前事实来源：[知识树服务](../backend/services/knowledge_tree_service.py)、[legacy knowledge-base 服务](../backend/services/knowledge_base_service.py)、[ResearchItem 索引服务](../backend/services/index_service.py)、[KnowledgeScope 服务](../backend/services/knowledge_scope_service.py)、[Evidence Reader 服务](../backend/services/evidence_reader_service.py)、[主动推送服务](../backend/services/push_service.py)、对应 API schema/routes 与自动化测试
>
> 替代文档或归档去向：无；本文是公开仓库中的当前实现说明

## 1. 文档职责

本文说明一次研究成果在 Deep Research 中如何从“等待人工确认的报告”变成可移植、可检索、可核查和可回顾的个人知识资产。重点是当前代码已经形成的合同，而不是未来愿景。

完整运行组件和状态边界见 [当前架构](./ARCHITECTURE.md)，测试与真实运行的证据口径见 [评测方法与发布门](./EVALUATION.md)。

## 2. 生命周期主线

```text
已完成的 Research Job
  │  报告、Evidence、来源、质量门结果
  ▼
Archive Preview（归档预览）
  │  建议或接受 KnowledgeBase / Theme 位置
  │  暂存 Markdown、manifest、总览与派生 HTML
  │  不修改可见知识树
  ▼
用户确认 + 预览版本校验
  │
  ▼
ResearchItem 原子归档
  │  Markdown / manifest 成为知识正本
  ├──────────────► Evidence Reader（可重建阅读视图）
  │
  └──────────────► 独立 LightRAG 索引 Job
                       │
                       ▼
                 KnowledgeScope 问答
                       │
                       └────────► 飞书知识卡回顾
```

这里有三类不同状态：

- **知识正本**：Markdown、可读目录和 manifest。归档一旦确认，即使后续索引失败，内容仍然存在。
- **派生状态**：HTML 阅读版、LightRAG 向量/图谱、索引元数据。它们可以失败、删除和重建，不能反过来替代知识正本。
- **运行状态**：Research Job、频道绑定、推送游标等。它们用于执行和消费知识，不定义知识正文或层级。

## 3. 知识模型与文件正本

当前知识树只允许三种领域对象：

```text
KnowledgeBase
├── ResearchItem
└── Theme
    └── ResearchItem
```

`KnowledgeBase` 是顶层学习空间；`Theme` 是可选的一层分类；`ResearchItem` 表示一次完整调研，可以直属 KnowledgeBase，也可以归入 Theme。Theme 不能继续嵌套，ResearchItem 只有一个主归属。一次相同主题的新调研会创建新的 ResearchItem，不会覆盖旧版本。

一个典型目录如下：

```text
knowledge_base/
└── 系统工程/
    ├── _kb.json
    ├── 00_总览.md
    ├── 2026-09-18_可靠性调研/
    │   ├── _research.json
    │   ├── 00_总览.md
    │   ├── 01_关键机制.md
    │   └── 完整报告.html
    └── 分布式系统/
        ├── _theme.json
        ├── 00_总览.md
        └── 2026-09-18_一致性模型调研/
            ├── _research.json
            └── *.md
```

正本职责刻意拆开：

| 内容 | 正本 | 说明 |
| --- | --- | --- |
| 报告正文 | ResearchItem 内的 Markdown | 不依赖数据库或索引即可阅读、复制和迁移 |
| 身份与归属 | `_kb.json`、`_theme.json`、`_research.json` | 稳定 ID 才是身份；标题和路径可以变更 |
| KnowledgeBase / Theme 总览 | `00_总览.md` | 人工内容保留，程序只替换受管区块 |
| 来源与证据血缘 | `_research.json` 中的 sources、citation sources 与质量摘要 | 支持阅读时从引用回到来源和 Quote |
| 检索索引 | ResearchItem 独立 LightRAG workspace | 可重建，不是正文或层级正本 |

`_research.json` 还记录 Research Job 血缘、文档哈希、研究时间、标签、来源数量、Evidence 数量、比较覆盖、质量确认和索引状态。目录用于人类阅读，稳定 ID 用于移动后的绑定、索引和推送连续性。

KnowledgeBase 与 Theme 的总览用受管标记区分人工内容和程序内容。归档刷新“当前覆盖、调研目录、知识缺口”等受管段落时，标记之外的人工笔记保持不变。

## 4. Preview / Confirm 原子归档

归档是需要用户确认的结构写操作，不会在 Research Job 完成时静默发生。

`archive-preview` 首先选择位置。用户可以显式选择已有 KnowledgeBase / Theme 或新建位置；没有显式选择时，服务依次参考已确认的 topic path、现有节点的标题/描述/标签匹配和默认分类规则，生成一个**建议**。建议只决定预览内容，不等于已经写入。

随后服务在内部 staging 目录构建目标 KnowledgeBase 的完整候选树：

- 写入本次 ResearchItem 的 Markdown 和 `_research.json`；
- 保存来源、Evidence 引用映射和质量审计摘要；
- 生成每篇文档及整份 ResearchItem 的可选 HTML 阅读版；
- 刷新直接父 Theme 和 KnowledgeBase 的受管总览；
- 记录目标目录在预览时的存在状态和内容哈希。

此时可见知识树没有变化。用户确认时，服务再次校验预览所依据的目录状态和哈希；如果期间有人修改了目标，预览进入 `stale`，不会覆盖新内容。

校验通过后，归档在文件锁内通过目录交换提交。已有 KnowledgeBase 会先移动到备份位置，再由 staging 候选树替换；交换失败时恢复备份。进程若在 `applying` 状态中断，后续进入知识树 `list_tree()`、`create_preview()` 或 `confirm_preview()` 时会检查持久化预览并完成提交、恢复备份或明确标记失败。因此这些主入口不会把“文档已保存、manifest 或总览尚未保存”的半成品当作成功归档；其他只读辅助入口不承诺主动触发恢复。

归档还有两道显式门：

- 自动质量门未通过时，默认拒绝生成归档预览；只有仍处于 `waiting_approval` 的 Job 接受用户明确的质量警告确认，才会继续，并把原始自动结论与人工确认时间写入 manifest。
- 比较研究存在未覆盖 Matrix Gap 时，需要单独确认缺口；缺口和确认结果随 ResearchItem 保存，不会被“归档成功”掩盖。

同一套 preview/confirm 协议也用于创建、重命名、移动、状态变更、移入回收站和恢复。ResearchItem 预览还会校验 `source_job_id`，避免一个 Job 确认另一个 Job 的归档计划。

实现与合同测试入口：

- [knowledge_tree_service.py](../backend/services/knowledge_tree_service.py)
- [research_service.py](../backend/services/research_service.py)
- [test_knowledge_tree_service.py](../tests/test_knowledge_tree_service.py)
- [test_research_archive_api.py](../tests/test_research_archive_api.py)

## 5. 独立 LightRAG 索引

归档确认后，系统为新的 ResearchItem 创建异步索引 Job。每个 ResearchItem 使用稳定 `research_id` 作为 workspace 身份，而不是使用会因重命名或移动而变化的文件路径。

索引状态独立保存在 `_research.json` 中：

```text
pending -> queued/running -> ready
                         └-> failed
```

还允许 `stale` 和 `disabled` 表达已过期或停用的派生状态。构建过程会校验 ResearchItem、抽取 Markdown、写入向量与图谱，并在完成后验证必需存储和元数据。

这条链路采用“知识先落盘、索引后完成”的故障语义：

- 无法启动索引 Job 或 embedding/LightRAG 构建失败，只把 `index_status` 记为 `failed`；
- Markdown、manifest 和层级不会回滚；
- 可以按稳定 ID 强制重试；
- 删除索引只删除可重建 workspace，并将状态重置为 `pending`，不会删除知识正文。

这项隔离避免外部模型、embedding 或索引存储故障破坏已经人工确认的知识资产。对应实现和回归见 [index_service.py](../backend/services/index_service.py) 与 [test_research_item_index.py](../tests/test_research_item_index.py)。

## 6. KnowledgeScope 聚合问答

`KnowledgeScope` 把知识树节点转成稳定的查询范围：

| Scope | 成员 |
| --- | --- |
| `knowledge_base` | 直属 ResearchItem 加所有 Theme 下的 ResearchItem |
| `theme` | 该 Theme 下的 ResearchItem |
| `research` | 单个 ResearchItem |

默认查询包含 `active` 和 `possibly_stale`，排除 `superseded`、`archived` 与已移入回收站的内容。显式 `include_history=true` 时才允许已被替代的 ResearchItem 参与；`possibly_stale` 和历史内容都会降低候选权重，前者还会产生显式时效警告，合并引用则保留每条材料的 ResearchItem 状态。

一次聚合查询的实际流程是：

```text
解析 Scope 成员
  -> 按标题、摘要、标签、时间和状态确定性预筛（最多 12 个）
  -> 跳过未 ready 的索引
  -> 最多 4 个 ResearchItem 并发调用各自的 LightRAG 查询
  -> 保留成功结果，记录失败或跳过原因
  -> 合并答案、引用、时效警告和 gaps
```

上层 Scope 不把全部 Markdown 拼成一个超长 prompt。它查询各自隔离的 ResearchItem 索引，再按 ResearchItem 汇总答案。引用按路径、URL 和内容去重，并补上 `research_id`、标题和状态，保留“这条材料来自哪一次调研”的血缘。

失败不会伪装成空答案：全部已就绪索引查询失败时返回 `query_failed`；没有可用索引时返回 `insufficient_index`；部分失败时保留成功答案，同时把未参与的 ResearchItem 和原因放入 `gaps`。Scope 的 index、graph 和 vector 视图同样聚合独立 workspace，而不另造一份共享索引。

代码和测试入口：

- [knowledge_scope_service.py](../backend/services/knowledge_scope_service.py)
- [knowledge_tree.py routes](../backend/routes/knowledge_tree.py)
- [knowledge_tree.py schemas](../backend/schemas/knowledge_tree.py)
- [test_knowledge_scope_service.py](../tests/test_knowledge_scope_service.py)

## 7. Evidence Reader

Evidence Reader 是知识正本之上的确定性阅读视图。归档预览会把完整 Evidence map 转成耐久的 source-level citation map：来源编号保持 Writer 使用的全局编号，每个来源下保存可核查 Quote、原子 Claim、Evidence ID，以及该 Evidence 被哪些文档使用。

阅读版的处理规则是：

- 把 Markdown 正文中的 `[N]` 转成可点击的引用标记；
- 按来源聚合多条 Quote，正文和来源卡使用同一套编号；
- 单篇文档只展示它实际引用的来源，完整阅读版保留跨文档的全局编号；
- 无法解析的编号保持普通文本，不伪造来源卡；
- 标题、正文与 Quote 在插入 HTML 前转义，避免文档内容直接执行为页面标记；
- 渲染不调用模型，给定相同报告和 Evidence map 会得到相同页面。

Markdown 与 manifest 始终优先于 HTML。HTML 渲染失败会记录 warning，但不会阻断归档；阅读版可以稍后从正本重新生成。当前 renderer 只保证标题、段落、局部告警和数字引用等受约束 Markdown 子集，复杂表格或通用 Markdown 语法不属于其完整渲染承诺。

实现和注入/编号回归见 [evidence_reader_service.py](../backend/services/evidence_reader_service.py) 与 [test_question_loop_evidence_reader_html.py](../tests/test_question_loop_evidence_reader_html.py)。

## 8. 频道问答与飞书回顾

频道可以绑定一个稳定的 KnowledgeBase 或 Theme Scope。绑定保存 `scope_type + scope_id`，因此节点重命名不会使频道失联；新式 Scope 绑定会清除旧 `topic_binding`。ResearchItem 可以在 Web 中单独查询，但当前不作为频道持久绑定类型。

主动回顾默认关闭，只有频道同时满足以下条件才会进入调度：

- 已显式绑定 KnowledgeScope 或 legacy topic；
- 已显式启用 push；
- 平台为飞书；
- 持久化的 `next_push_at` 已到期。

对于 KnowledgeScope，系统从当前可查询的 ResearchItem 中读取 manifest 声明的 Markdown，排除总览、来源、Evidence、Knowledge QA 和入库建议等辅助文档。轮换游标使用 `research_id/filename`，不会因 ResearchItem 移动而失效。内容策略支持 `rotation`、`new_only` 和 `random`。

发送前，提示词要求 Writer 模型把单篇文档压缩为“一条核心结论 + 1 至 2 条文内依据”的知识卡，再通过现有飞书发送通道投递。只有发送成功才记录游标和推进下一次时间；单个频道失败不会终止其他频道，本轮没有合适内容则只推进计划时间。调度周期用进程锁避免重叠执行，启动和停止可以重复调用。

知识卡是复习摘要，不是新的研究结论，也没有继承 Evidence Reader 的逐条引用核验。它的可信边界仍是被选中的已归档文档；需要核查时应回到 ResearchItem 正文和 Evidence Reader。

相关实现和合同：

- [push_service.py](../backend/services/push_service.py)
- [channel_service.py](../backend/services/channel_service.py)
- [channels.py routes](../backend/routes/channels.py)
- [test_knowledge_scope_channel_push.py](../tests/test_knowledge_scope_channel_push.py)
- [test_push_service.py](../tests/test_push_service.py)

## 9. 状态与失败语义

| 对象 | 主要状态 | 当前语义 |
| --- | --- | --- |
| Preview | `pending / applying / committed / stale / failed` | 只有 `committed` 表示结构写入成功；`stale` 必须重新预览 |
| ResearchItem | `active / possibly_stale / superseded / archived` | 状态变更也需要 preview/confirm；历史不会被自动覆盖 |
| Trash | 独立 `.trash` 目录 | 从正常树和默认 Scope 隐藏，保留稳定 ID 与原路径供恢复 |
| Index | `pending / queued / running / ready / failed / stale / disabled` | 只有 `ready` 参与问答；失败不影响 Markdown 正本 |
| Scope Query | `answered / query_failed / insufficient_index` | 明确区分有答案、查询故障和没有可用索引 |
| Push | `not_due / no_content / pushed / error` | 只有 `pushed` 记录成功投递游标 |

这套状态设计的原则是：**不让派生能力的失败冒充知识丢失，也不让部分成功冒充完整成功。**

## 10. 安全与一致性边界

当前实现包含以下确定性约束：

- 节点标题有长度和保留名检查，目录名会去除路径分隔符与控制字符；ResearchItem 文档读取拒绝路径片段，并在 manifest 已声明文档清单时执行 Markdown 文件白名单。
- Theme 只能创建在 KnowledgeBase 下；ResearchItem 移动时会重新校验目标 Theme 的父 KnowledgeBase。
- Preview ID、节点类型、Scope 类型和频道绑定类型均有白名单；HTTP schema 进一步限制 action、mode 和候选上限。
- 结构写操作在进程锁和支持的平台文件锁内执行；`list_tree()`、`create_preview()` 与 `confirm_preview()` 会先尝试恢复中断操作。
- Evidence Reader 对可见文本做 HTML 转义；外部来源仍应视为不可信内容，不能因为出现在来源卡中就默认安全或正确。
- Push Scheduler 和真实飞书发送默认关闭，只有显式配置后才可能产生外部副作用。

仍需保留的边界：

- 当前适合本地单用户或受控环境；知识树服务本身不提供多租户授权或全局 HTTP 鉴权。
- 归档的目录交换依赖同一文件系统的原子重命名语义；Windows 无 `fcntl` 时只有进程内锁，不等价于 Unix 下的跨进程文件锁。
- `possibly_stale`、`superseded` 等知识状态由用户确认管理，系统不会自动裁决哪个历史结论真实。
- LightRAG 回答与知识卡仍经过模型；引用存在不等于整句话已被完整支持，质量结论必须按 [评测方法](./EVALUATION.md) 的层次解释。
- [legacy topic API](../backend/services/knowledge_base_service.py) 仍用于读取旧式单目录知识库，并保留旧问答、文档和索引接口；它不会把旧目录自动提升为新的层级对象。新知识归档、Scope 问答和稳定频道绑定应使用 manifest-backed 模型。

## 11. 可复核入口

知识生命周期的主要公开代码与回归证据如下：

| 能力 | 实现 | 测试 |
| --- | --- | --- |
| 层级、manifest、preview/confirm、回收与恢复 | [knowledge_tree_service.py](../backend/services/knowledge_tree_service.py) | [test_knowledge_tree_service.py](../tests/test_knowledge_tree_service.py)、[test_knowledge_tree_api.py](../tests/test_knowledge_tree_api.py) |
| Research Job 归档接线与质量确认 | [research_service.py](../backend/services/research_service.py) | [test_research_archive_api.py](../tests/test_research_archive_api.py) |
| ResearchItem 独立索引 | [index_service.py](../backend/services/index_service.py) | [test_research_item_index.py](../tests/test_research_item_index.py) |
| Scope 成员、聚合问答、图谱和向量 | [knowledge_scope_service.py](../backend/services/knowledge_scope_service.py) | [test_knowledge_scope_service.py](../tests/test_knowledge_scope_service.py) |
| 引用阅读版 | [evidence_reader_service.py](../backend/services/evidence_reader_service.py) | [test_question_loop_evidence_reader_html.py](../tests/test_question_loop_evidence_reader_html.py) |
| 频道绑定与主动回顾 | [channel_service.py](../backend/services/channel_service.py)、[push_service.py](../backend/services/push_service.py) | [test_knowledge_scope_channel_push.py](../tests/test_knowledge_scope_channel_push.py)、[test_push_service.py](../tests/test_push_service.py) |

这些测试证明对应确定性合同在隔离 fixture 中成立；它们不单独证明真实模型质量、长期调度可靠性或真实飞书权限配置已经通过。运行和发布结论应继续按 [EVALUATION.md](./EVALUATION.md) 区分单元测试、服务合同、Live E2E 与人工验收。
