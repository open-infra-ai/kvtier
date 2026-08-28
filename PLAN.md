# ai-infra 后续开发计划

更新时间：2026-08-28

## 目标与边界

主线保持为：SGLang HiCache / vLLM KV offloading 学习，聚焦 CPU 侧 KV 分层存储，
用可复现实验支撑上游贡献和面试材料。开发顺序固定为：

```text
可复现环境 → 正确性与路径证据 → 性能结果审计 → 数据驱动优化 → 上游贡献
```

单卡 L2 基准只能说明指定模型、指定设备和指定配置下的 host 回载表现；它不能代表
多卡、L3、EAGLE 或问题 #31600 的完整复现。没有真实 GPU 结果包和审计记录时，不发布
性能结论。

## 当前已完成

- [x] 初始化根级 Git 边界；未暂存、未提交、未推送。
- [x] 增加 `.gitignore`，忽略本地环境、模型材料、实验结果、Python 缓存和嵌套上游仓库。
- [x] 增加 `make test` 与 `make audit RESULT=...` 入口。
- [x] 固定基准客户端 `requests` 依赖。
- [x] 增加结果包审计器，校验服务配置、W/E/R/P 路径、命中条件、TTFT 比值和计数器增量。
- [x] 正式基准要求传入 40 位 SGLang commit SHA。
- [x] 离线测试 14 项通过；真实 GPU 基准尚未运行。
- [x] 2026-08-28（P0-1）：审阅全部未跟踪文件；`.gitignore` 覆盖 `.venv/`、`models/`、
  `results/`、`third_party/`、`__pycache__` 已确认完整；完成根级初始本地提交，
  未建远程、未推送（远程归属待用户决定）。
- [x] 2026-08-28（P0-2）：确定 SGLang 构建基线 `6afee75f918f3d59bdae7d619ba26b0ca22b1354`
  （third_party 浅克隆），Python 3.12.3 与安装方式（uv + editable）记录于
  [`kv-tiering/bench/repro-31600/ENVIRONMENT.md`](kv-tiering/bench/repro-31600/ENVIRONMENT.md)。
- [x] 2026-08-28（P0-3）：用 uv 创建隔离客户端环境并安装 `requests==2.31.0`；
  `Makefile` 改为自动优先使用 `.venv/bin/python`；`make test` 14 项在隔离环境通过，
  零 Python 字节码写入。

## 后续任务

### P0：版本与环境可追溯性

- [x] 审阅当前未跟踪文件，确认根级 Git 初始内容和远程仓库归属。（2026-08-28 完成）
  - 验收：明确保留的文件、忽略的上游目录和结果目录；只在用户要求时暂存、提交或推送。
- [x] 选择一个干净且明确的 SGLang commit，记录源码路径、commit、Python 版本、镜像或
  安装方式。（2026-08-28 完成，见 `kv-tiering/bench/repro-31600/ENVIRONMENT.md`）
  - 验收：启动命令中的服务来源与基准参数中的 `--sglang-commit` 一致，禁止只记录
    `latest` 或未标记的 `main`。
- [x] 创建隔离的基准客户端环境并按 `kv-tiering/bench/repro-31600/requirements.txt`
  安装依赖。（2026-08-28 完成；uv venv + `requests==2.31.0`；`make test` 已改为
  自动优先使用 `.venv`）
  - 验收：`make test` 在隔离环境中通过，且不写入 Python 字节码。

### P1：本机服务与正确性门禁

> 阻塞说明（2026-08-28）：本机仅有 GGUF 格式模型（SGLang 不支持），需下载 HF 格式
> 小模型；服务端 sglang 尚未安装。用户已授权"安装 sglang 服务端 + 下载
> Qwen2.5-1.5B-Instruct"。两项完成后按下列任务继续。

- [ ] 选择适配 RTX 3060 Laptop 6GB 的模型并记录绝对路径、权重 dtype 和上下文长度。
  - 当前选定：`Qwen2.5-1.5B-Instruct`（HF 格式，约 3 GB，目标路径 `~/models/`）。
  - 验收：模型路径、GPU 型号、驱动和启动参数可复核；不把模型下载或云 GPU 费用默认纳入本地任务。
- [ ] 用 `run_server.sh` 启动 HiCache 单卡服务，检查 `/health`、`/server_info`、
  `/model_info` 和 `/metrics`。
  - 验收：TP/DP/PP/attention CP 均为 1，启用 hierarchical cache、cache report、
    metrics 和 `write_through`，未启用 speculative decoding 或 L3 storage。
- [ ] 先运行 `workload.py --dry-run`，再用小规模参数做服务 smoke test。
  - 验收：SSE 有正常结束事件和输出 token，token 数、cache namespace 和服务配置均通过门禁。

### P2：单卡 L2 证据闭环

- [ ] 使用固定的 SGLang commit 和模型运行单卡 L2 微基准，保存完整 JSON 结果包。
  - 建议命令：`python3 workload.py --sglang-commit <40-char-commit> --output-dir results`。
- [ ] 用 `make audit RESULT=results/results-<UTC>.json` 审计结果。
  - 验收：结果为 `status=complete`；R 的 host 命中率、P 的零命中、backup/eviction/load-back
    counter 增量和配对 TTFT 数据全部通过。
- [ ] 编写一次实验报告，包含硬件、软件 commit、模型、完整启动命令、参数、原始结果、
      统计方法、结论和限制。
  - 验收：报告只描述本机本次运行；不把 R/P 比值写成 #31600 复现或通用性能结论。

### P3：根据数据选择优化方向

- [ ] 先检查结果中的 L2 命中率、驱逐量、load-back counter、TTFT 分布和失败原因。
- [ ] 优先候选：L3 prefetch 失败原因的有限枚举与结构化可观测性；先核对上游 `main`
      和重复 issue/PR。
- [ ] 只有在 correctness 与带宽基线建立后，才评估 NUMA 感知、局部性优化或 KV 换出压缩。
- [ ] 只有获得匹配的多卡、EAGLE、L3、并行配置和预算后，才单独搭建 #31600 复现实验。

### P4：上游与作品集交付

- [ ] 将已验证的最小问题、复现命令、原始日志、预期/实际行为和有限修复方案整理为上游
  issue/PR 材料。
- [ ] 为每个完成的实验保留一张架构图、一条调用链、核心数据结构、可运行命令和面试问题。
- [ ] 上游状态、重复工作和当前 `main` 在公开评论或 PR 前重新核对。
- [ ] 提交和推送必须得到用户明确指示；共享工作区只暂存本任务文件。

## 常用验收命令

```bash
make test
cd kv-tiering/bench/repro-31600
python3 workload.py --dry-run
python3 validate_results.py results/results-<UTC>.json
```

## 暂不执行

- 不从零开发完整 mini 推理引擎，不为改名或搬目录消耗主线时间。
- 不在没有真实 GPU 结果和审计包时宣称性能提升。
- 不把本机单卡 L2 微基准写成 #31600 完整复现。
- 不默认下载大模型、租用云 GPU、修改 `third_party` 上游源码或发布 PR。
