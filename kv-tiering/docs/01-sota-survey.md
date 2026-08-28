# 调研笔记 01：SGLang 与 vLLM 的 KV 分层存储现状

> 校订日期：2026-08-25。
> SGLang 核验基线：`main@67853c580482dbc9b69aad5022a0cdd3a1606570`。
> 结论来自当前源码、官方文档和 GitHub issue/PR；不再把第三方代码问答当一手证据。

## 一、SGLang HiCache

### 1. 当前默认架构

[PR #35081](https://github.com/sgl-project/sglang/pull/35081) 于 2026-08-21 合并后，
`UnifiedRadixCache` 成为默认前缀缓存实现。启用 HiCache 时，构建流程在这棵统一树
上附着 host pool、`HiCacheController` 和可选的 storage backend。

三层职责如下：

| 层级 | 数据位置 | 当前主要实现 |
|---|---|---|
| L1 | GPU HBM | `KVCache` 的各类 device pool；前缀索引由 `UnifiedRadixCache` 管理 |
| L2 | host DRAM | `mem_cache/pool_host/` 下的 MHA/MLA/混合模型 host pool |
| L3 | 本地或远端存储 | file、Mooncake、HF3FS、NIXL、AIBrix 等 storage backend |

device KV pool 继承 `KVCache`，并不继承 `RadixCache`。前者保存 KV 张量，后者维护
token 前缀到物理槽位的索引和生命周期；旧笔记把这两个层次混在了一起。

### 2. 代码入口

- `mem_cache/registry.py`：选择默认树实现；普通路径最终创建 `UnifiedRadixCache`；
- `mem_cache/unified_radix_cache.py`：match、insert、evict、L2 load-back、L3 prefetch
  及 scheduler 入口；
- `mem_cache/unified_cache/`：FULL/SWA/MAMBA 等组件和 storage attachment；
- `managers/cache_controller.py`：L1↔L2 操作队列以及 L2↔L3 后台 I/O；
- `mem_cache/l2_transfer.py`：D2H/H2D 专用 stream 和逐层 H2D 提交；
- `mem_cache/pool_host/`：host 内存布局、分配与实际传输分支；
- `observability/metrics_collector.py`：已有 HiCache 指标。

`hiradix_cache.py` 仍保留用于兼容或特定路径，但它不是当前默认阅读主线。

### 3. 写入和预取策略

`--hicache-write-policy` 当前默认是 `write_through`：新插入或尚未备份且达到命中
阈值的节点会进入备份流程。已经具备完整 host 副本、没有剩余 transfer 的节点会被
跳过，因此不能概括成“每次命中都把整段 KV 再拷一遍”。

- `write_through`：阈值 1，优先保证 L2 副本；
- `write_through_selective`：阈值 2，只备份再次命中的节点；
- `write_back`：设备驱逐时再备份，驱逐路径需要等待 D2H 完成。

L3 prefetch 的默认 token 门槛仍为 256；停止策略默认是 `timeout`，不是
`wait_complete`。三种策略是 `best_effort`、`wait_complete` 和 `timeout`。

### 4. 已有可观测性

[PR #32388](https://github.com/sgl-project/sglang/pull/32388) 已在 2026-08-05 合入
主要 HiCache telemetry。当前源码至少包含：

- `sglang:cached_tokens_total{cache_source=...}`：device/host/storage 命中；
- `sglang:evicted_tokens_total`：device 驱逐；
- `sglang:hicache_backup_tokens_total`、`hicache_backup_bytes_total`、
  `hicache_backup_duration_seconds`：L1→L2；
- `sglang:load_back_tokens_total`、`load_back_bytes_total`、
  `load_back_duration_seconds`：L2→L1；
- `sglang:prefetch_bandwidth`、`backup_bandwidth`：支持统计的 L3 backend 带宽；
- host pool 占用、drop、aux allocation failure 等补充指标。

因此“HiCache 完全没有传输指标”已经是错误结论。当前更可信的缺口是：部分
prefetch outcome 只在 `UnifiedRadixCache` 内部累计，默认 collector 没有完整导出；
`batch_exists` 命中后真实读取失败也缺少统一、低基数的原因分类。详见 #35808 的
本地纠正记录。

### 5. 可研究但尚未证实的方向

- host pool 的 CPU/GPU/NUMA 局部性和跨 socket 带宽；SGLang 已有 worker NUMA
  binding 配置，是否仍有 HiCache 特有问题必须实测；
- 多卡下 scheduler 每步 HiCache 状态同步的成本；源码存在 CPU tensor collective，
  但它是否构成瓶颈需要 profile；
- prefetch threshold、timeout 与实时带宽/队列深度的自适应；
- 不同 `hicache-io-backend`、host layout 和 KV dtype 的真实带宽。

这些都是实验问题，不应先写成已确认的性能缺陷。

## 二、vLLM KV offloading

vLLM 当前同时存在两类名字相近但语义不同的 CPU offload：

| 参数 | 对象 | 含义 |
|---|---|---|
| `--cpu-offload-gb` | 模型权重 | 每 GPU 把一部分参数放在 CPU，在 forward 时通过 UVA 访问或搬运 |
| `--kv-offloading-size` | KV cache | 配置 KV offloading buffer；设置后由 `kv_offloading_backend` 选择实现 |

因此旧笔记中“`--cpu-offload-gb` 已被 `kv_offloading_size` 取代”是错误的：两者面向
不同数据。除此之外，vLLM 还用 KVConnector 支持跨实例 KV 传输或外部缓存系统；
不能把 connector、单实例 native KV offload 和模型权重 offload 合并成同一机制。

当前项目只把 vLLM 用作设计对照。若要做横向性能比较，必须先统一模型、KV dtype、
block/page size、CPU 内存上限、命中率和 workload，不能仅比较两个默认命令。

## 三、与本项目直接相关的上游事项

| 上游事项 | 截至 2026-08-25 的核验 | 对本项目的含义 |
|---|---|---|
| [#31600](https://github.com/sgl-project/sglang/issues/31600) | open；store hit TTFT 4.2s，明显快于 cold 31.5s；真正退化是 EAGLE accept rate、decode throughput 和总时长 | 原问题主要不是“L2 TTFT 比 prefill 慢”；需要匹配 EAGLE/L3/多卡环境 |
| [#30393](https://github.com/sgl-project/sglang/pull/30393) | 已合并；为 MTP/EAGLE/DSpark 增加 packed/sidecar draft state 支持 | 原报告之后实现已变化，不能用简化单卡实验宣称复现 |
| [#32418](https://github.com/sgl-project/sglang/pull/32418) | 仍 open；修改 legacy draft host pool 的 MLA stride 传播 | 当前默认 unified builder 已按 pool 的 `kv_cache_dim` 构建 sidecar；仍需原配置回归，不能据 PR 状态推断默认路径必然有同一 bug |
| [#35808](https://github.com/sgl-project/sglang/issues/35808) | open；RFC 关注全生命周期诊断 | 大部分 transfer telemetry 已存在，候选贡献应聚焦尚未导出的 outcome 和真实失败分类 |

## 四、行动结论

1. 先以 `UnifiedRadixCache` 为主线重建理解，旧 `HiRadixCache` 笔记只作历史对照；
2. 用单卡 L2 微基准验证脚本和观测方法，但严格限制结论范围；
3. 若要处理 #31600，另建尽量匹配原报告的 EAGLE + L3 + TP/CP 实验，重点记录
   accept length/rate、decode throughput、TTFT 和总时长；
4. 只有 profile 证据出现后，才在 NUMA、collective、传输 kernel 或策略中选择优化点。
