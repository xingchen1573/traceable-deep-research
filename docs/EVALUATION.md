# Deep Research 评测方法与发布门

> 状态：`current`
>
> 创建日期：2026-09-07
>
> 最后核对日期：2026-09-07
>
> 当前事实来源：`pytest.ini`、`tests/`、`frontend/package.json`、当前评测 API 与 Private 开发仓库中的完整评测账本
>
> 替代文档或归档去向：无

## 1. 文档职责

本文定义评测层次、证据记录、基线选择和发布门。Public 候选只保留方法、选定测试和脱敏摘要；历史报告、原始产物、冻结 Job 清单和当前推荐基线保留在 Private 开发仓库。

## 2. 评测原则

1. 代码通过、自动 Gate、人工可交付、真实渠道成功和产品发布资格是不同结论。
2. Development 与 Holdout 必须隔离，不能把调参样本当作最终泛化证据。
3. 不同日期、模型、配置、case set 或代码版本的局部成功不能相加成一个“总体通过率”。
4. 测试和评测默认不应调用真实模型、发送飞书或修改正式知识库；当前尚未完全达到该默认隔离目标，必须按本文边界运行。
5. 每个结论必须能定位命令、输入、环境摘要和原始产物。

## 3. 测试与评测层次

| 层次 | 目的 | 当前入口 | 是否允许外部副作用 |
| --- | --- | --- | --- |
| Python unit/component | 验证纯逻辑和局部组件 | `pytest` 单文件或模块 | 默认不允许，外部调用应 mock |
| Service contract | 验证 Knowledge、Channel、Push、Runtime、Feishu adapter 合同 | `tests/test_*service.py` 等 | 默认不允许真实发送 |
| API integration | 验证 FastAPI route/schema/service 接线 | `tests/test_backend_api.py` 等 | 应使用临时状态，当前隔离仍不完整 |
| Frontend | 类型、组件逻辑和生产构建 | `npm run typecheck`、`npm test`、`npm run build` | 不允许真实后端副作用 |
| Deterministic evaluation | 验证评测器、Gate 和报告 schema | research/single-url/finding 测试 | 不允许真实模型 |
| Frozen replay/A-B | 对冻结 checkpoint 或 artifact 复算 | `scripts/replay_*`、audit/reaudit 脚本 | 按脚本分类，部分会调用模型 |
| Live model E2E | 验证真实供应商下的完整调研质量 | `scripts/run_*`、部分 `verify_*` | 允许付费模型、网页和 artifact 写入 |
| Live Feishu smoke | 验证真实群消息、文档、权限和重启恢复 | 专用脚本与运行手册 | 只允许专用测试租户和群 |
| Human review | 判断正确性、完整性、可读性和可交付性 | 评测报告人工结论 | 无自动替代 |

## 4. 默认本地验证

快速确定性 Backend 回归：

```bash
RUN_LIVE_MODEL_TESTS=0 FEISHU_ENABLE_REAL_SEND=false \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
.venv/bin/python -m pytest -q \
  tests/test_knowledge_tree_service.py \
  tests/test_knowledge_scope_service.py \
  tests/test_push_service.py \
  tests/test_channel_push_config.py \
  tests/test_feishu_ws_watchdog.py
```

Frontend：

```bash
cd frontend
npm run typecheck
npm test
npm run build
```

Private 开发仓库中的 `test_agent_dispatcher_revision_regression.py` 已使用 `live_model` marker 和 `RUN_LIVE_MODEL_TESTS=1` 显式门禁；该文件不随 Public 候选分发。对当前 checkout 的完整 Python 测试可以通过 marker 排除真实模型用例：

```bash
RUN_LIVE_MODEL_TESTS=0 FEISHU_ENABLE_REAL_SEND=false \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
.venv/bin/python -m pytest -q tests \
  -m "not live_model"
```

这仍不能宣称严格无网络、无工作区写入。当前 `pytest.ini` 只额外注册了 `live_model`，还没有完整的 marker 分层、timeout 或 socket deny；`tests/conftest.py` 对共享 SQLite 的隔离也不代表知识库、LightRAG、logs 和 artifacts 全部被隔离。

## 5. Live 运行门槛

真实模型或网页 E2E 必须：

- 使用显式开关和独立 case 选择。
- 写入独立输出目录，不覆盖推荐基线。
- 记录预计调用次数、时间上限和中止语义。
- 不输出真实 key、Cookie 或完整 `.env`。
- 禁止同时修改正式知识库，除非该用例就是受控归档验收。

真实飞书 smoke 必须额外满足：

- 使用专用测试应用、测试群和允许的测试用户。
- 明确开启 IM、文档或推送中的哪一种真实写入。
- 记录 message/document ID 的脱敏摘要，不复制凭据。
- 覆盖失败重试、限流、权限、重复消息和进程重启。
- 执行后能定位和清理由测试创建的外部对象。

## 6. 多维度评测

发布判断至少分开记录：

| 维度 | 主要信号 |
| --- | --- |
| 代码与合同 | 单元、集成、迁移、并发和恢复测试 |
| 调研执行 | 计划完成度、来源发现、证据抽取、Coverage、停止原因 |
| 报告质量 | 正确性、完整性、忠实度、可读性、引用支撑、边界意识 |
| 知识体系 | 课程覆盖、来源新鲜度、冲突、重复、缺口收敛 |
| 检索问答 | Hit@k、首条关键证据、要点覆盖、噪声、引用一致性 |
| 学习卡 | 忠实度、可回忆性、答案泄露、重复度、来源可访问性 |
| 学习效果 | 完成率、首次正确率、再次遗忘率、到期积压 |
| 飞书可靠性 | 收发率、重复/丢失、限流恢复、权限和重启恢复 |
| 安全与配置 | 失败关闭、秘密不泄露、跨进程配置一致性 |

单一 `passed` 数量不能代替上述维度。

## 7. 基线和发布资格账本

每次要成为推荐基线或发布证据的运行，应保存：

- 日期和环境标签。
- Git commit；在尚未初始化 Git 时记录工作区指纹和明确限制。
- case set ID、版本、内容哈希和 Development/Holdout 身份。
- 模型、provider、prompt/config revision、fallback 和关键 feature flags。
- 完整命令、选择范围、超时和退出码。
- 自动 Gate、人工结论、环境失败和豁免理由。
- 原始产物路径和报告路径。
- 是否产生知识库、飞书、数据库或其他外部副作用。

Private 开发仓库中的新增报告遵循内部 `DOCUMENTATION_STANDARD.md` 的 evaluation 命名和元数据规范，并登记到 Private 评测二级索引；该内部规范和原始评测账本不随 Public 候选分发。

Public 只保留不绑定个人知识库的 `research_probe` 与 `single_url_brief` 用例配置。`knowledge_qa` 用例与维护者的本地 KnowledgeScope 和索引配套，默认不导出；新 clone 会把缺失的 `knowledge_qa.json` 视为空用例集，使用者需要为自己的知识范围生成或录入用例，不能把 Private 样本当作开箱即跑的公共基线。

## 8. Writer 规则专项门

修改 `revision_rules` 中要求 Writer 使用特定措辞或关键词的规则，上线前，必须在 Private 开发仓库用冻结 checkpoint 重放，并确认 `unsupported` 或 `label_mismatch` 类 issue 数不高于修改前 baseline。冻结输入和运行脚本可能包含内部 Job 标识及付费模型配置，因此不随 Public 候选分发；专项通过也不替代完整 Writer、E+ 和人工内容评测。

## 9. 当前缺口

- 已注册 `live_model`，尚未注册统一的 `unit`、`integration`、`slow`、`live_feishu`、`writes_workspace` markers。
- 默认测试没有 socket deny。
- 部分测试和 probe 会读取本地 `.env` 或写真实工作区。
- `/api/v1/tests/runs` 的 `api_smoke` 当前只是 compile Backend Python，不等于发起 HTTP API smoke。

这些缺口完成前，README 和 GitHub 不能宣称“默认测试严格离线”或“全量评测全部通过”。
