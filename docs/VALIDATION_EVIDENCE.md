# Deep Research 公开验证证据摘要

> 状态：`evaluation`
>
> 创建日期：2026-09-18
>
> 最后核对日期：2026-09-18
>
> 当前事实来源：冻结五类全链路评测、最小完整系统受控运行、调研审计验收与 Public 候选资格回执
>
> 替代文档或归档去向：无

## 1. 这份摘要证明什么

本文只汇总能够公开且可以复核口径的验证结果。它用于回答三个问题：证据写作链是否跑通、调研成果能否进入知识复用闭环、公开快照能否在无私人数据和凭证的环境中被检查。

每组结果都限定到记录的日期、样本和环境。自动 Gate 通过不等于全文绝对正确；模型式语义判断也不等于人工确认。原始检索正文、模型输入输出、运行标识、Provider 诊断和个人知识库保留在开发环境，不随作品快照分发。

## 2. 五类 Deep Research 全链路

2026-09-05 的资格验证覆盖解释型深挖、技术深挖、双对象比较、多对象比较和争议分析五类请求。第一次执行有两例在 Planner 合同校验处被阻断；修复通用的重复 obligation 映射后，五类样本各自完成一次从真实检索、Evidence 抽取、章节写作、Editorial 到最终引用核验的全链路。

| 判据 | 结果 | 解释边界 |
| --- | ---: | --- |
| 修复后完成全链路 | 5 / 5 | 是组合后的资格结果，不是首次单批 5 / 5 |
| 自动生产 Gate | 5 / 5 | 只说明自动合同通过 |
| Validator 判定 cited claim 获支持 | 231 / 247（93.52%） | 模型 judgment，不是人工确认的真实正确率 |
| 原位显示的 `[待核实]` | 16 | 提示人工复核，不自动删除正文 |
| 内部 `claim_*` 泄漏、嵌套残缺引用、相邻重复编号 | 0 / 0 / 0 | 限本轮五个样本 |

人工复核 16 个 marker 后，11 个属于正文扩大了 Evidence 未覆盖的范围、因果、评价、并列分支或比较结论，5 个属于偏严格判断或分析边界；小样本 marker precision 为 11 / 16（约 68.8%）。最终交付口径为：2 篇可直接交付，2 篇可用但应保留来源质量提醒，1 篇只能作为明确披露证据缺口的阶段性报告。

这组结果支持 Evidence-native E+ 作为当前写作主链；它没有提供重新引入 Finding 中间层的证据。尚未运行正式 Holdout，因此不能从这五例外推总体质量。

## 3. 调研到知识复用闭环

2026-09-10 使用一个受控主题完成最小完整系统验证。调研生成 4 份 Markdown、11 个来源和 83 条 Evidence，Quality Gate 与 Validation 通过；人工确认后归档为 ResearchItem，独立 LightRAG 索引进入 `ready`。同一 Theme 的范围问答返回答案和 4 条文档引用；两次受控飞书知识卡发送验证了轮换选材和成功后的游标推进。

同日的只读知识树快照中，44 个 active ResearchItem 的独立索引均为 `ready`。这只是当日状态，不是长期可用性承诺。

该 smoke 证明的是“调研、审批、归档、索引、范围问答、受控推送”可以在一个样本中闭环。它没有验证长期无人值守调度、多用户隔离、多实例并发或逐句回到网页原文的知识库问答引用。

对应实现边界见[知识生命周期](./KNOWLEDGE_LIFECYCLE.md)。

## 4. 审计与隔离回放

2026-08-28 的本地确定性验收没有访问外部 Provider，也没有创建调研 Job、写数据库或知识库。验收证明：

- model-call 记录可以定位调用前 checkpoint；调用后 checkpoint 再记录相关 call 与 prompt hash，从而支持交叉定位；
- fenced JSON 回放复用生产 parser，默认不触发 repair；
- 审计准备或写入抛出 `OSError` 时，业务模型返回仍按 fail-open 语义交付；
- 当次聚焦审计测试为 18 passed，相关调研回归为 398 passed。

这些数字是历史验收记录，不是当前完整测试套件规模。审计 lineage 用于复盘和归因，不等同于 Research Job 跨进程自动续跑；具体边界见[可观测性与回放](./OBSERVABILITY_AND_REPLAY.md)。

## 5. Public 候选资格

2026-09-18 的首个合格本地 Public 候选采用 tracked-only 白名单导出，并在全新目录验证。该候选的结果为：

| 检查 | 结果 |
| --- | --- |
| Private 非 live Python 回归 | 1396 passed，6 skipped |
| Public Python | 329 passed |
| Frontend | 17 / 17 files、62 / 62 tests；typecheck 与 production build 通过 |
| 容器 | Compose 配置通过，Backend、Feishu Worker、Frontend 镜像构建通过 |
| 公开秘密扫描 | unresolved finding、unexpected binary、baseline error 均为 0 |
| 合成 Evidence demo | 两次输出逐字节一致，引用结构校验通过 |

合成 demo 的输入哈希为 `70844dce75930175757bd4fb685ec166e4e7dc555634f926bd7028605d3f2956`，输出哈希为 `2b2a4c8783bae21c73e4fb58c3973724eed97459079789f521f6334cdcfa2bf0`。它不读取凭证、不访问网络、不调用模型，也不写正式状态；模型语义验证明确为 `not_run`，因此只证明冻结输入、Evidence 血缘与引用结构的确定性。

可以用下列命令复核这条公开演示：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/demo_public_evidence_pipeline.py
```

评测层次、真实服务副作用和发布判据见[评测方法](./EVALUATION.md)。

## 6. 仍未由这些结果证明

- 没有正式 Holdout，五类样本不能代表未知主题上的稳定质量。
- cited claim 的自动支持率不能替代人工阅读原文，也不能证明未引用分析一定正确。
- 单次完整系统 smoke 不能证明长期调度、故障接管或生产级可靠性。
- Public 测试通过只证明所导出 revision 的源码、依赖和演示合同；内容或 manifest 变化后需要重新生成候选并复验。
