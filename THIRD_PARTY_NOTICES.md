# Third-Party Notices

> 状态：`current`
>
> 创建日期：2026-09-07
>
> 最后核对日期：2026-09-17
>
> 当前事实来源：`requirements.txt`、`Dockerfile`、`LightRAG/pyproject.toml`、`LightRAG/lightrag/_version.py`、`LightRAG/LICENSE`、Public export manifest，以及 2026-09-07 从官方 `HKUDS/LightRAG` GitHub 下载的指定提交源码包比对和 2026-09-17 的隔离构建验证
>
> 替代文档或归档去向：无

## LightRAG

- 项目：LightRAG
- 上游：`HKUDS/LightRAG`
- 仓库内路径：`LightRAG/`
- 当前声明版本：`1.5.2`
- 当前 API 版本：`0308`
- 安装方式：`requirements.txt` 通过 `-e ./LightRAG` 使用本地 vendored 快照
- 许可证：MIT，完整文本见 `LightRAG/LICENSE`
- 官方 `v1.5.2` tag：[`2935d0ced5f5c27dcdc0b808d76fc2d052856d4b`](https://github.com/HKUDS/LightRAG/commit/2935d0ced5f5c27dcdc0b808d76fc2d052856d4b)
- 本地源码内容基点：[`3fa73ecb9e19f19e184ab6e6c5608472d9b5371b`](https://github.com/HKUDS/LightRAG/commit/3fa73ecb9e19f19e184ab6e6c5608472d9b5371b)
- 初次导入日期：尚未追溯
- 本地内容补丁：在本次可审计范围内未发现；已恢复上游执行位，运行生成物差异见下文

Public 候选为保持本项目现有 editable 安装方式，分发经过审核的包级运行子集：`LICENSE`、`MANIFEST.in`、`README.md`、`pyproject.toml` 与 `lightrag/` 包目录；上游工作流、测试、示例和独立产品文档不随快照分发。项目未调用的独立 API Swagger UI 与 LightRAG visualizer 整体排除，后者同时避免分发本项目不需要的第三方字体。当前 vendored Swagger bundle 声明的独立许可证文件并不存在于已审计上游快照中，不能靠猜测补齐。在已有构建依赖的隔离环境中，该子集已通过 `--no-deps --no-build-isolation` 普通 wheel 构建、离线 wheel 安装和 editable 安装验证，面向本项目的 `-e ./LightRAG` 使用，不宣称是 LightRAG 的独立完整源码发行版。由于保留上游 `README.md` 供包元数据使用，其中指向未分发上游文档和图片的相对链接会由导出器明确报告为告警。

## 来源审计结论

本地 `LightRAG/lightrag/_version.py` 与官方 `v1.5.2` 都声明包版本 `1.5.2` 和 API 版本 `0308`，但版本字符串不等于源码 tag。官方 `v1.5.2` 指向 `2935d0c…`，而本地可审计源码内容与更早的官方提交 `3fa73ec…` 一致。该提交是 `v1.5.2` tag 的直系祖先，官方之后又依次合入：

1. `ae3a8d1d123681f0ecefce2c58f4576008091f6c`：为 parse-stage 失败增加状态与 UI 字段映射。
2. `1c121ace61e3c72f57f813e0f7dbd75a924bd996`：移除已弃用的 `docs_format="lightrag"` enqueue 入口。
3. `2935d0ced5f5c27dcdc0b808d76fc2d052856d4b`：将默认 LLM timeout 从 `180` 秒调整为 `240` 秒，并打上 `v1.5.2` tag。

因此，当前目录应描述为“包版本声明为 `1.5.2`、源码基于官方提交 `3fa73ec…` 的 vendored 快照”，不能描述为“未修改的官方 `v1.5.2` 发布包”。

## Private 导出源的 2026-09-07 全量 vendor 导入审计

- 官方来源：[`v1.5.2` release](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.2) 及上述官方 commit archive。
- 内容比对：对 `3fa73ec…` 官方 archive 与本地目录中 `666` 个非 `.env*` 普通文件计算 SHA-256，结果为 `666/666` 匹配。
- 许可证比对：官方基点与本地 `LICENSE` 的 SHA-256 均为 `0c16924564059e493c86854574660a47ed1dcc2a476ba7b5f949d3f2f73a1984`。
- 版本文件比对：官方基点与本地 `lightrag/_version.py` 的 SHA-256 均为 `008227288073aecbf864457a46e4b3ec77198468dc1a320bd3f1681e0d3fff6f`。
- 安全边界：本次未读取或比较任何 `.env*` 文件，因此“内容一致”结论仅适用于上述 `666` 个非敏感文件。
- 非源码差异：本地存在可重建的 `__pycache__`、`*.pyc`、`lightrag_hku.egg-info/` 和 `.DS_Store`，不属于上游源码，不应进入 Git。
- 文件模式：本次审计发现 `18` 个上游 `100755` 文件在本地丢失执行位；已于 2026-09-07 按官方基点恢复为 `755`，首次 Git 基线不再保留这项模式差异。

短期继续保留 vendored 目录，避免破坏现有构建。若后续升级到官方 `v1.5.2` tag 或更新版本，需对上述三组上游变更运行 LightRAG 回归，不应只修改版本字符串。

上述 `666/666` 结论针对 Private 导出源中的完整 vendor 导入，不表示 Public 包级运行子集仍包含 666 个文件；Public 选择集由 manifest 和导出回执单独审计。

LightRAG 的 MIT 许可证只适用于对应第三方代码，不能代替 Deep Research 项目本身的授权。Deep Research 的公开候选是 source-visible 作品展示；除 GitHub 服务条款允许的服务内查看、使用、展示和 fork，以及第三方许可证授予的权利外，在另行提供项目级许可证前，不表示另行授予修改或再分发项目代码的许可。
