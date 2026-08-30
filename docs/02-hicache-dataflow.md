# 精读笔记 02：当前 HiCache 数据流与 #31600 边界

> 校订日期：2026-08-25。
> SGLang 核验基线：`main@67853c580482dbc9b69aad5022a0cdd3a1606570`。
> 旧版逐行笔记基于 legacy `HiRadixCache`，且把若干性能假设写成了结论；本版改以
> 当前默认 `UnifiedRadixCache` 为主线。源码行号变化快，以下以类和方法名定位。

## 一、当前组件边界

| 组件 | 路径 | 责任 |
|---|---|---|
| `UnifiedRadixCache` | `mem_cache/unified_radix_cache.py` | 统一前缀树入口；协调 FULL/SWA/MAMBA 组件、L1/L2/L3 生命周期及 scheduler 事件 |
| `TreeCore` 与 components | `mem_cache/unified_cache/` | 树结构、LRU/引用、各类缓存状态及 backup/load-back action |
| `HiCacheController` | `managers/cache_controller.py` | D2H/H2D 队列、operation merge、L3 query/get/set 和后台线程 |
| `L2TransferEngine` | `mem_cache/l2_transfer.py` | 在两条专用 device stream 上提交 L1↔L2 传输 |
| host pools | `mem_cache/pool_host/` | host 内存分配、布局和 MHA/MLA/混合模型传输实现 |
| `StorageAttachment` | `mem_cache/unified_cache/storage_attachment.py` | L3 backend 的运行时 attach/detach 与配置 |
| `Scheduler` / `PrefillAdder` | `managers/scheduler.py`、`managers/schedule_policy.py` | 每步轮询事件；构建 prefill batch 时发起必要的 L2 load-back |

默认构建链在 `mem_cache/registry.py`。除 ChunkCache、实验性 C++ tree、LMCache、
FlexKV 等显式分支外，普通场景创建 `UnifiedRadixCache`；启用 hierarchical cache
后再执行 `init_hicache()`。因此下面的关系不是“device pool 继承 radix tree”，而是：

```text
token / prefix key
        │
        ▼
UnifiedRadixCache + TreeCore          负责索引与生命周期
        │
        ├── L1 device KV pools       GPU KV 张量和物理槽位
        ├── L2 host pool group       pinned host memory
        └── L3 storage backend       hash/page 寻址的持久或远端数据
```

## 二、L1 写入与 L2 备份

以默认 `write_through` 为例：

1. prefill/decode 生成的新 KV 被插入统一前缀树；
2. TreeCore 产生 backup action；`_execute_and_commit_kv_backup()` 从父到子处理；
3. `build_backup_spec()` 只返回尚需传输的 KV/sidecar。节点已有完整 host 副本时，
   没有剩余 transfer，会被跳过；
4. host 空间不足时先尝试 `evict_host()`；分配成功后，controller 将 operation 入队；
5. `start_writing()` 合并本轮 operation，并通过 `L2TransferEngine` 的
   `device_to_host_stream` 提交 D2H；
6. scheduler 后续轮询完成事件，TreeCore 才把节点标记为已完成 host backup；启用
   L3 时再把相应页面交给 storage backup 队列。

三种策略的核心差别：

- `write_through`：命中阈值为 1，新节点尽早保留 host 副本；
- `write_through_selective`：阈值为 2，减少只访问一次的数据写入；
- `write_back`：设备需要驱逐时才执行 D2H，并在释放 device 槽位前等待完成。

旧版“write_through 每次命中都重写整段 KV”不准确。它会覆盖 decode 新增段等尚未
备份的数据，但不会无条件重传已有完整 host 副本的节点。

## 三、L1 驱逐与 L2 load-back

### 1. 驱逐

对已经备份的节点，device eviction 可以释放 L1 槽位而保留 L2 副本。未备份数据在
不同 write policy 下可能被备份或丢弃；当前指标会分别记录 device eviction、host
backup 和 dropped tokens。

### 2. 回载

请求匹配到 host 前缀后，主要路径是：

```text
match_prefix
  → PrefillAdder / init_load_back
  → UnifiedRadixCache.load_back
  → CacheController.load（分配 device 槽并入队）
  → ready_to_load_host_cache / start_loading
  → L2TransferEngine.submit_host_to_device
  → 每层 H2D event 与计算侧 wait
  → check_hicache_events / loading_check 收尾和记指标
```

关键约束：

- FULL KV 少于 `load_back_threshold=10` 且没有其他 component transfer 时会跳过回载；
- device 空间不足时会先尝试驱逐，仍不足则退回重新 prefill；
- H2D 在独立 stream 上按 layer 提交；forward 在消费对应 layer 前等待 event，后续层
  传输可以与前面层的计算重叠；
- 完成后 host 副本仍可作为可回收 duplicate 保留，不等于每次 load-back 都删除 L2；
- `cached_tokens_details.host` 和 `load_back_*` counter 可用于证明请求确实走了 L2。

`LayerDoneCounter` 当前使用 3 个事件槽循环复用。它提供有限的在途流水和复用前检查，
但仅凭这个结构不能推出“它防止 host pool 爆炸”或“它就是性能瓶颈”。

## 四、L3 prefetch

L1/L2 未覆盖足够前缀且配置了 storage backend 时，系统可以发起 L3 prefetch：

1. storage query 按页面 hash 检查连续存在的前缀；首次缺页后停止继续扩展；
2. 多 rank 场景对可用前缀做一致性同步；
3. scheduler drain 命中结果后才分配 host 页面。分配不足可能缩短结果或 revoke；
4. 后台 I/O 调用 `batch_get*` 把页面取回 L2；
5. scheduler 根据停止策略决定何时让请求继续，并把实际完成的连续前缀挂入树中。

停止策略默认值是 `timeout`：

| 策略 | 语义 |
|---|---|
| `best_effort` | 使用当前已完成部分，不等待全部页面 |
| `wait_complete` | 等待目标页面全部完成 |
| `timeout` | 完成或达到按请求长度计算的超时后结束 |

`best_effort`、timeout、host 分配不足、低于 threshold 和真实 `batch_get*` 失败不是
同一种故障。旧版把它们全部映射为“present but read failed”，会污染可观测性设计。

## 五、传输后端与 JIT fallback

`MHATokenToKVPoolHost` 根据 IO backend、host layout、方向和 kernel 能力选择不同
分支：

- H2D 是逐层路径；D2H 是 all-layer 路径；
- `layer_first`、`page_first`、`page_head`、direct 等布局并不共用同一实现；
- page-first D2H 还有独立的 write-back JIT 能力判断；
- JIT 不可用时会落到对应的预编译 kernel/direct 路径，而不是一个统一的“慢 3 倍”
  分支。

设计文档的“up to 3x”是 GPU-assisted kernel 相对基线的上限描述，不是任意模型、
方向和布局的固定损失。#31600 日志中的 `element_size=656` 确实不满足 JIT 条件，
但 issue 本身把它列为次要因素，并指出它主要影响约 4 秒的 TTFT。

旧实验计划用 Qwen2.5-7B 的 BF16/FP8 去验证同一 fallback 也不成立。该模型配置为
4 个 KV heads、head dim 128：MHA host pool 的 element size 分别是 BF16 1024 bytes
和 FP8 512 bytes，均能被 128 整除，不能复现 `656 % 128 != 0` 的条件。

## 六、scheduler 同步：事实与假设

`check_hicache_events()` 每个 scheduler step 都会执行。当前 unified 路径会把 write
ack、load ack、storage queue size 和一致性摘要打包到 CPU tensor；在 TP/CP/PP
world size 大于 1 时执行 collective/点对点同步，单 rank 则不会真的 all-reduce。

可以确认的事实：

- 这是 scheduler 主线程上的固定事件处理入口；
- 完成 ack 在处理前会同步相应 finish event；
- 多 rank 必须对 ready count 和若干决策保持一致。

不能由源码直接确认的结论：

- 这些 CPU collective 已经是用户可见延迟的主要来源；
- `finish_event.synchronize()` 经常阻塞很久；
- D2H/H2D 一定争抢到使 decode throughput 明显下降。

这些都应通过 CPU profile、NCCL/Gloo trace、CUDA timeline 和已有 duration/byte
counter 验证，再决定是否优化。

## 七、#31600 的真实问题

原 issue 的对比数据是：

| 指标 | cold prefill + decode | store hit |
|---|---:|---:|
| TTFT | 31.5 s | 4.2 s |
| EAGLE accept rate | 约 0.98 | 约 0.10 |
| accept length | 5.94 | 2.87，稳定后约 1.4 |
| decode throughput | 约 770 tok/s | 约 130–320 tok/s |
| 总时长 | 98.99 s | 152.62 s |

因此 store hit 的 TTFT 实际更快；“命中更慢”指 decode 和总时长。issue 作者的主要
解释是：命中后 target KV 恢复了，但 EAGLE draft KV/hidden state 没有完整恢复，
导致低接受率；JIT fallback 是次要因素，RDMA batch get 也不是报告中的主瓶颈。

原报告之后，代码发生了两项重要变化：

1. [PR #30393](https://github.com/sgl-project/sglang/pull/30393) 已合并，为多类
   MTP/EAGLE/DSpark topology 加入 packed 或 sidecar draft cache 传输；
2. [PR #35081](https://github.com/sgl-project/sglang/pull/35081) 让 unified tree 成为
   默认。当前 unified sidecar builder 用 draft pool 自身 `kv_cache_dim` 构造 MLA
   host pool；仍 open 的 [PR #32418](https://github.com/sgl-project/sglang/pull/32418)
   修改的是 legacy draft registration。

这些事实说明原始路径已显著变化，但不能反向证明 #31600 已彻底修复。正确结论是：
需要在当前 main 上尽量匹配 GLM-5.2、FP8、EAGLE、Mooncake、TP=8、attention CP=8、
64K×10 输入与长 decode，再比较 cold/replay 的 draft state 和接受率。

## 八、两类实验必须分开

### A. 单卡 L2 微基准

`bench/repro-31600/` 因历史原因保留旧目录名，现已明确改为单卡 L2 微基准。它验证：

- L1→L2 backup、device eviction、L2→L1 load-back 是否都真实发生；
- 同 token 数、独立 cache namespace 下，L2 reload 与 cold prefill 的 TTFT 比值；
- 不同 IO backend/layout/dtype 在单一配置中的表现。

它不启用 EAGLE/L3/TP/CP，因此无论结果如何，都不能声称复现或否定 #31600。

### B. #31600 回归

真正回归至少应记录：

- target 与 draft/sidecar 的 L2/L3 命中和传输量；
- EAGLE accept rate、accept length、decode throughput；
- TTFT、总时长、输出正确性；
- SGLang commit、模型 revision、GPU/网络拓扑、全部启动参数和 raw results。

若要做 target-only 对照，应启动一个不带任何 speculative 参数的独立服务。不要用
`--speculative-eagle-topk 0` 冒充“关闭 EAGLE”，因为该参数只是 EAGLE 配置的一部分，
不是通用禁用开关。

## 九、当前最可信的下一步

1. 先在真实 GPU 上运行并审计单卡 L2 结果包，确认测试工具本身可靠；
2. 有条件匹配原环境时再做 #31600 回归，先看 draft state 和接受率；
3. 对多卡同步、NUMA、D2H/H2D 竞争只提出可证伪假设，并以 profile 排序；
4. 可观测性贡献避免重复 #32388，优先考虑 prefetch outcome 导出和有限失败分类。
