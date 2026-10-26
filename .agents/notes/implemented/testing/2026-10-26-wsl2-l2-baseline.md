# Agent Note: 完成 WSL2 单卡 HiCache L2 基线并修复 host 指针问题

Status: implemented

## Problem

单卡 HiCache L2 微基准此前只有客户端、审计器和离线测试，没有真实 GPU 结果包；
公开门槛第 1 条因此标注“未达成”。在 WSL2 / RTX 3060 Laptop 上首次运行时：

- 原定 Qwen2.5-1.5B-Instruct BF16（2.9 GB）在 Windows 桌面占用约 1.97 GB 显存后，
  SGLang 报告 KV cache 最低预算 `mem-fraction-static > 0.827`，物理空闲显存不足；
- 改用 Qwen2.5-0.5B-Instruct 后，M 阶段第一个 R（L2 回载）请求稳定触发
  `cudaErrorIllegalAccess`，服务在 flashinfer prefill 崩溃；W/E/P 阶段本身可完成；
- 无 HiCache 的基础服务在同样 4224-token 请求下稳定，排除基础推理栈问题。

## Decision

1. **模型替换**：本次运行使用同家族 Qwen2.5-0.5B-Instruct BF16；在证据中明确
   绝对 TTFT 不能外推到 1.5B/7B。
2. **小内存配置**：SGLang 的 host 预算是 `psutil.available - 10 GiB`，本机约
   1.18 GB。选择 `mem-fraction-static=0.28`（设备池 33,744 tokens）、
   `hicache-ratio=2.8`（host pool 94,484 tokens / 1.16 GB）、6 轮而不是默认 8 轮，
   使 host pool 容纳 W + E + P；禁用 CUDA graph 以释放 host 预留并缩短启动。
3. **WSL2 host 指针修复**：指针诊断确认 `cudaHostRegister` 的
   `cudaHostGetDevicePointer` 结果不等于 host 地址，而 torch pinned 内存是恒等的。
   采用上游未合入 PR #43375 的 host 分配部分作为本地补丁（保存为
   `bench/repro-31600/evidence/2026-10-26-wsl2-l2-baseline/sglang-43375-local.patch`，
   sha256 记录在证据 README），基础 commit 仍为 `6afee75f9...`。
4. **失败留档**：修复前的两个失败结果包（不同设备池大小）保留在 `failed/`，
   不删除失败样本。

## Alternatives considered

- **继续使用 1.5B**：需要释放 Windows 桌面显存或换更大显存的机器；当前无法在
  自动化流程内保证，且 0.5B 足以验证 L2 回载机制。绝对性能数值不是本次目标。
- **压缩 host pool 到 ratio 4 + 更少轮次**：轮次太少会削弱配对统计；ratio 2.8 +
  6 轮在 1.18 GB 预算内保留了 W/E/P 容量。
- **不应用未合入补丁，仅记录阻塞**：可作为负结果交付，但无法回答“L2 回载在本机
  表现如何”；补丁来自直接相关的上游 PR，且保存了完整差异和哈希，可追溯性足够。
- **修改 `third_party` 并提交父仓**：`third_party/sglang` 是 gitignored 的嵌套
  仓库；把补丁作为证据文件归档，父仓不跟踪上游源码，避免伪版本。
- **用 `direct` IO 或 `layer_first` 规避**：两者分别在本机启动预热阶段崩溃，
  不能作为稳定替代。

## Verification

- `make test`：离线 14 项通过；
- `validate_results.py`：最终结果包 `status=complete`，6 轮，
  `paired_ttft_ratio_median=0.214677`；
- R 请求 host 命中 4223/4224，P 请求零命中；
- counter 增量：backup 88,625 tokens、eviction 80,304 tokens、load-back
  25,338 tokens / 311,353,344 bytes；
- 指针诊断：`cudaHostRegister` identity=False，torch pin_memory identity=True；
- 无 HiCache 基础服务在 4224-token 请求下健康。

## Consequences

- 公开门槛第 1 条（真实 GPU 实验 + 结果审计）已达成，但范围仅限该机、该模型、
  该配置；第 2 条（固定 commit/模型/硬件/参数/原始日志）已就绪，证据中明确标注
  本地补丁；
- 该结果不能用于声称 #31600 已复现，也不能外推到多卡/L3/EAGLE；
- PR #43375 合入后应以新 commit 重跑并移除本地补丁；
- 新增 `run_server.sh` 的 `DISABLE_CUDA_GRAPH` 与 `HICACHE_LAYOUT` 环境开关，
  默认行为不变（graph 开启、page_first）；
- 失败运行与根因证据保留，后续若再遇 WSL2 HiCache 崩溃可直接复用诊断脚本。
