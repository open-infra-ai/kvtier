# 环境记录（P0：版本与环境可追溯性）

> 更新时间：2026-08-28。本文件记录构建与运行本基准所用的所有可追溯信息。
> 正式 GPU 运行时，`workload.py --sglang-commit` 必须与下文"服务端"一节的 commit 一致。

## 硬件

- GPU：NVIDIA GeForce RTX 3060 Laptop，6144 MiB
- 驱动：610.57.01（KMD 610.88），CUDA UMD 13.3

## 操作系统与 Python

- OS：Linux（Debian/Ubuntu 系）
- Python：3.12.3（`/usr/bin/python3`）
- 环境管理工具：uv 0.9.13（系统缺 `python3.12-venv`，改用 uv 创建 venv，无需 sudo）

## 上游源码（third_party，只读浅克隆）

| 仓库 | commit | 说明 |
|---|---|---|
| sglang | `6afee75f918f3d59bdae7d619ba26b0ca22b1354` | 基准服务端构建基线；docs 01/02 的核验基线为 `67853c580482dbc9b69aad5022a0cdd3a1606570`，两者不同，正式运行以实际服务构建来源为准 |
| nano-vllm | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` | 仅用于架构对照阅读 |

## 基准客户端（已就绪）

- 位置：`kv-tiering/bench/repro-31600/.venv`（uv venv，CPython 3.12.3）
- 依赖：`requirements.txt` → `requests==2.31.0`
- 验证：`make test`（根目录）自动优先使用该 venv；PYTHONDONTWRITEBYTECODE=1 下
  14 项离线测试通过，且不写入任何 `__pycache__`/`.pyc`。

## 服务端（待构建）

- 计划方式：`uv venv ~/.venvs/sglang-server`，从本地浅克隆以 editable 方式安装
  `uv pip install -e third_party/sglang/python`，使安装内容与上表 commit 严格对应；
- 该 commit 的 `pyproject.toml` 要求 `torch==2.13.0`、CUDA 13 系列 wheel
  （`flashinfer_python[cu13]` 等），与本机 CUDA UMD 13.3 匹配；
- ⚠️ 2026-08-28 首次尝试失败，**失败点在 Rust 构建而非 Python 依赖**：editable
  安装会触发 `rust/sglang-server` cdylib 编译，`rustpython-ruff 0.15.8` 系列依赖
  要求 rustc ≥ 1.92，本机 rustc 为 1.90.0（日志：/tmp/sglang-install.log）。
  修复路径：`rustup update stable`（或安装 ≥ 1.92 的 rustc）后重跑
  `uv pip install -e third_party/sglang/python`；
- 状态：**尚未安装成功**。安装完成前，任何真实 GPU 运行都不可开始。

## 模型（已就绪，仅限本机）

- 已下载：`/home/shane/models/Qwen2.5-1.5B-Instruct`（HF 格式，2.9 GB，
  含 `model.safetensors`；2026-08-28 由 `hf download` 获取）；
- 注意：模型是机器本地产物，不随仓库走；新环境需重新下载；
- P1 门禁启动服务时使用 `MODEL_PATH=/home/shane/models/Qwen2.5-1.5B-Instruct`。

## 验证入口

```bash
make test          # 离线 14 项测试（隔离环境、零字节码）
cd kv-tiering/bench/repro-31600 && python3 workload.py --dry-run
```
