# Deep Research 本地开发指南

> 状态：`current`
>
> 创建日期：2026-09-07
>
> 最后核对日期：2026-09-07
>
> 当前事实来源：`requirements.txt`、`Dockerfile`、`frontend/package.json`、`frontend/vite.config.ts`、当前运行入口和测试配置
>
> 替代文档或归档去向：无

## 1. 文档职责

本文只说明从源码进行本地开发、启动和基础验证。配置变量和安全语义见 [`CONFIGURATION.md`](./CONFIGURATION.md)；生产服务器、备份和发布操作不属于公开候选快照。

## 2. 已核对工具链

- Python：Docker 使用 `3.12`；2026-09-07 本地 `.venv` 为 Python `3.12.13`。
- Node.js：Frontend Docker builder 使用 Node `20`；仓库没有 `.nvmrc` 强制本地版本。
- 包管理：Python 使用 `requirements.txt`，Frontend 使用 npm 和 `package-lock.json`。
- Docker：可选，仅在需要验证 Compose 或镜像时使用。

仓库尚无顶层 `pyproject.toml`、`.python-version` 或统一 Python lockfile。开发时优先采用 Python 3.12，避免与当前镜像产生无意义差异。

## 3. 首次安装

在仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 包含 `-e ./LightRAG`，因此必须从仓库根目录安装，并保留 `LightRAG/`。安装依赖需要网络访问。

安装 Frontend：

```bash
cd frontend
npm ci
```

`npm ci` 使用锁文件重建 `node_modules/`，不要提交该目录。

## 4. 本地配置

新工作区可以从无秘密值模板创建本地配置：

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

随后只在本地 `.env` 中填写实际需要的凭据。不要把真实值写入文档、命令历史、测试 fixture 或 Git。

模板默认关闭：

- 真实飞书消息发送。
- 自动飞书文档交付。
- 主动推送 Scheduler。
- live model tests。

已有本地 `.env` 时不要用示例文件覆盖。当前配置加载和跨进程生效限制见 [`CONFIGURATION.md`](./CONFIGURATION.md)。

## 5. 启动方式

### 5.1 Backend API

```bash
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 24680
```

健康入口：`http://127.0.0.1:24680/api/v1/health`。

Backend 启动会初始化应用状态和 runtime monitor，并可能创建本地 SQLite/运行目录。Push Scheduler 只有在 `PUSH_SCHEDULER_ENABLED=true` 时才启动。

### 5.2 Frontend real 模式

```bash
cd frontend
npm run dev:real
```

当前 Vite 默认只监听 `127.0.0.1:24681`，并把 `/api` 代理到 `http://127.0.0.1:24680`。Backend 当前没有通用 CORS middleware，本地开发优先使用该同源代理，不要随意把 API 改成跨域直连。确需从其他设备访问时，必须显式覆盖监听地址并自行控制网络边界。

### 5.3 Frontend mock 模式

```bash
cd frontend
npm run dev:mock
```

Mock 模式用于页面开发，不证明 Backend、模型、知识库或飞书链路可用。

### 5.4 CLI

```bash
.venv/bin/python main.py
```

CLI 会构建 Agent，并可能在收到请求后调用真实模型、创建知识或运行产物。使用 `--debug` 会显示更多内部消息，但不应依赖 debug 输出处理秘密。

### 5.5 飞书长连接 Worker

```bash
.venv/bin/python scripts/run_feishu_ws_worker.py
```

该命令会尝试连接飞书并写 heartbeat、进程状态、频道和消息数据。只有在专用测试应用、权限和发送开关均明确后才运行；它不是普通的无副作用开发命令。

### 5.6 Docker

仓库提供的 [`docker-compose.yml`](../docker-compose.yml) 会同时启动 Backend、Frontend 和飞书 Worker。Worker 需要真实飞书配置，因此普通本地开发优先使用前述宿主机命令；只有在专用测试应用和持久化目录边界都已确认后才运行 Compose。不要把 `docker compose down -v` 当作普通清理命令。

## 6. 基础验证

无需配置、模型、网络或正式状态的合成 Evidence 演示：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/demo_public_evidence_pipeline.py --compact
```

Backend 快速确定性回归：

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

完整测试分层、当前非严格离线边界和真实评测入口见 [`EVALUATION.md`](./EVALUATION.md)。

## 7. 本地状态和清理边界

以下路径是个人数据、运行状态或可重建产物，不是源码：

- `.env*`
- `data/`
- `knowledge_base/`
- `lightrag_workspaces/`
- `memory/`
- `logs/`
- `backups/`
- `demo_workspace/`
- `.venv/`
- `frontend/node_modules/`
- `frontend/dist/`

不要因为测试失败而直接删除数据库、知识库、LightRAG workspace、memory 或 Docker Volume。需要隔离测试时，应通过临时目录和 monkeypatch 改写路径，而不是操作真实工作区。

## 8. 常见问题

### Backend 显示模型未配置

先确认启动进程是否能读取仓库根 `.env`，以及变量是否被进程已有环境覆盖。不要打印 key 验证；使用 Models API 返回的 configured 布尔值或脱敏标识。

### Frontend 无法访问 API

确认 Backend 监听 `24680`，Frontend 使用 `dev:real`，且 `VITE_API_PROXY_TARGET` 没有指向错误地址。优先修复代理，不要先开放宽泛 CORS。

### 修改 Models 页面后 Worker 未变化

这是当前已知边界。页面更新只影响 Backend 进程，Docker 中 `.env` 还是只读挂载，Worker 也不会自动重载。按 [`CONFIGURATION.md`](./CONFIGURATION.md) 的进程重启合同处理。

### Pytest 意外访问真实环境

当前测试还没有完整 marker 和 socket deny。先使用本文快速回归命令；扩大运行范围前检查测试涉及的模型、网络和写入边界，不要仅凭文件名判断是否无副作用。
