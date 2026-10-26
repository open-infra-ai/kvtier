# 2026-10-26 单卡 HiCache L2 回载基线（WSL2 / RTX 3060 Laptop）

## 结论

在 WSL2 + RTX 3060 Laptop 上完成并审计了一组单卡 HiCache L2 微基准：

- 运行 ID：`20261026T091835Z`；`num_trials=6`；前缀 4096 + tail 128 tokens；
- L2 回载 TTFT 中位数 **72.3 ms**，冷 prefill **338.0 ms**，
  配对 R/P 中位数 **0.215**（本次配置下 L2 回载约为冷 prefill 的 1/4.7）；
- 6/6 轮 R 请求的 `cached_tokens_details.host` 均为 **4223/4224**，
  冷 P 请求命中为 0；
- Prometheus 增量：backup 88,625 tokens、device eviction 80,304 tokens、
  load-back 25,338 tokens / 311,353,344 bytes；
- `validate_results.py` 通过（`trials=6`，
  `paired_ttft_ratio_median=0.214677`）。

**范围**：只描述本机、本模型、本配置下“host→device 回载 vs 等长冷 prefill”的
TTFT；**不是** SGLang #31600 的复现，也不代表多卡、L3、EAGLE 或 decode 吞吐。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6144 MiB，sm_86 |
| 驱动 | 610.88（nvidia-smi 610.57.01），CUDA UMD 13.3 |
| OS | WSL2，Linux 6.18.54.1-microsoft-standard-WSL2 |
| Python | 3.12.3（`~/.venvs/sglang-server`） |
| torch | 2.13.0+cu130 |
| SGLang | `6afee75f918f3d59bdae7d619ba26b0ca22b1354`（`0.0.0.dev2+g6afee75f9`） |
| 本地补丁 | PR [#43375](https://github.com/sgl-project/sglang/pull/43375) 的 host 分配部分，见 `sglang-43375-local.patch`（sha256 `d9e99574df7ae920619a292059f2ce6cb486b12f8bb7342b92c1e93e7530df68`） |
| 模型 | `Qwen2.5-0.5B-Instruct`（HF 原版 BF16，`model.safetensors` 943 MB） |

模型替换原因：原定 Qwen2.5-1.5B-Instruct BF16 权重约 2.9 GB，启动时 Windows
桌面侧已占用约 1.97 GB 显存，SGLang 报告 KV cache 最低预算需
`mem-fraction-static > 0.827`，物理空闲显存不足；0.5B 在同家族、同 tokenizer
接口下可稳定运行。0.5B 的绝对 TTFT 不代表 1.5B/7B 的数值。

## 服务配置与偏离默认项

| 项 | 值 | 原因 |
|---|---|---|
| `mem-fraction-static` | 0.28 | 设备 KV 33,744 tokens；给 4096 前缀和驱逐留余量 |
| `hicache-ratio` | 2.8 | host pool 94,484 tokens / 1.16 GB |
| `hicache-io-backend` | kernel | 默认值；修复后可用 |
| `hicache-mem-layout` | page_first | 默认值 |
| `disable-cuda-graph` | 1 | 小显存/小内存：省 GPU 与 host 预留，启动从 ~85 s 降到秒级；TTFT 两侧同等受影响 |
| `num-trials` | 6（默认 8） | host pool 需容纳 W+E+P：6×2×4224 + 37,888 = 88,576 tokens ≤ 94,484；8 轮需约 105,472 tokens，超过 1.18 GB host 预算 |
| L3 storage | 未启用 | 基准默认边界 |
| TP/DP/PP/attn CP | 1/1/1/1 | 基准校验要求 |

host 预算来自 SGLang 的 `host_memory_budget_bytes()`：`psutil.available - 10 GiB`；
本机启动时为 1.18 GB。该函数与 WSL2 的 10 GiB 保守预留是配置上限的直接原因。

## 根因与修复

修复前，M 阶段的第一个 R（L2 回载）请求稳定触发
`cudaErrorIllegalAddress`，服务在 flashinfer prefill 的 `begin_forward` 处崩溃；
W/E/P 阶段本身可完成。无 HiCache 的基础服务在同样 4224-token 请求下稳定。

指针诊断（`wsl2-host-pointer-diagnostic.out`）显示：

```text
cudaHostRegister: rc_register=0 host=0x7deea800c040 dev=0x204c00040 identity=False
torch pin_memory: host=0x204c00000 dev=0x204c00000 identity=True
```

SGLang 的 `kernel` IO 后端把 host 地址直接当作 device 地址使用；在 WSL2 上
`cudaHostRegister` 的映射不是恒等的，于是回载内核访问了非法地址。上游
[PR #43375](https://github.com/sgl-project/sglang/pull/43375)（截至本次运行仍为
OPEN）改为在 WSL2 自动使用 torch pinned memory。本仓将该 PR 的
`pool_host/common.py` 改动作为本地补丁，记录在 `sglang-43375-local.patch`；
补丁只影响 host 分配器选择，不修改传输语义。

修复前的失败结果包保留在 `failed/`：

- `results-n8-small-pool.json`：`mem-fraction-static=0.22`、ratio 11、
  设备池 7,642 tokens，N=8；
- `results-n1-large-pool.json`：`mem-fraction-static=0.28`、ratio 2.5、
  设备池 33,744 tokens，N=1。

两次均在首个 R 请求失败，说明不是单纯的容量问题。另观察到
`--hicache-io-backend direct` 在启动预热阶段崩溃、`--hicache-mem-layout
layer_first` 在预热 decode 阶段崩溃；本次成功路径只使用 `kernel + page_first +
torch pinned`。

## 复现

```bash
# 1) 应用本地补丁（基线 commit 必须为 6afee75f9）
git -C third_party/sglang apply \
  bench/repro-31600/evidence/2026-10-26-wsl2-l2-baseline/sglang-43375-local.patch

# 2) 启动服务（需先安装 editable SGLang 与模型）
MODEL_PATH=~/models/Qwen2.5-0.5B-Instruct \
  PORT=30000 MEM_FRACTION=0.28 HICACHE_RATIO=2.8 \
  HICACHE_IO=kernel HICACHE_LAYOUT=page_first DISABLE_CUDA_GRAPH=1 \
  bash bench/repro-31600/run_server.sh

# 3) 运行并审计
python3 bench/repro-31600/workload.py \
  --port 30000 --sglang-commit 6afee75f918f3d59bdae7d619ba26b0ca22b1354 \
  --num-trials 6 --prefix-tokens 4096 --tail-tokens 128 \
  --output-dir bench/repro-31600/results
python3 bench/repro-31600/validate_results.py \
  bench/repro-31600/evidence/2026-10-26-wsl2-l2-baseline/results.json
```

## 原始结果

- `results.json`：最终运行（已把个人绝对路径替换为 `<home>`、主机名替换为
  `<host>`；原始文件在 gitignored 的 `bench/repro-31600/results/`）；
- `results-pilot-n1.json`：单轮预跑，host 命中 4223/4224，用于先验证回载路径；
- `server-startup.log`：权重加载、设备池 33,744、host pool 94,484 / 1.16 GB、
  分配器告警；
- `wsl2-host-pointer-diagnostic.py/.out`：根因证据。

## 限制

- 单机、单卡、单模型（0.5B 替代 1.5B）、单次运行；6 轮配对，统计量有限；
- WSL2 与 RTX 3060 的 host 指针行为、驱动版本和 CUDA 栈都可能影响结果；
- 本地补丁来自未合入的上游 PR；上游合入后应以新 commit 重跑；
- 禁用 CUDA graph，绝对值与默认配置不可直接横向比较；
- 未做 nsys/ncu 归因，不解释 TTFT 组成；
- 不能用于声明 #31600 已复现或多卡/L3 性能。
