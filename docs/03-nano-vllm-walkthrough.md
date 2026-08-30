# 精读笔记 03：nano-vLLM 调度、块管理与 SGLang 对照

> 校订日期：2026-08-25。
> nano-vLLM 核验基线：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`。
> SGLang 对照基线：`main@67853c580482dbc9b69aad5022a0cdd3a1606570`。

nano-vLLM 是约 1,200 行 Python 的离线推理教学实现，包含连续批处理、paged KV、
prefix caching、Tensor Parallel 和 decode CUDA graph。上游 README 给出的
Qwen3-0.6B/RTX 4070 Laptop 结果是 1434.13 output tok/s；本项目没有在同一硬件与
环境复测，因此只能标成“上游报告值”，不能当作本地性能证据。

## 一、文件地图

```text
nanovllm/
├── config.py                    参数与模型配置
├── engine/
│   ├── llm_engine.py            generate/step 主循环与 TP 子进程
│   ├── scheduler.py             prefill/decode 调度与抢占
│   ├── block_manager.py         物理块、prefix hash、引用计数
│   ├── sequence.py              请求状态和 token 容器
│   └── model_runner.py          输入组装、KV 分配、forward、CUDA graph
├── layers/                      attention、TP linear、sampler 等
├── models/qwen3.py              dense Qwen3 实现
└── utils/loader.py              safetensors 权重加载
```

最短阅读顺序：

1. `LLMEngine.step()`：schedule → run → postprocess；
2. `Scheduler.schedule()`：一轮选择 prefill 或 decode；
3. `BlockManager`：块分配、前缀复用和回收；
4. `ModelRunner.prepare_*()`：逻辑 block table 如何变成 attention 输入；
5. `Attention`：slot mapping 写 KV，再调用 flash-attn。

## 二、请求生命周期

```text
prompt
  → tokenizer.encode
  → Sequence(WAITING)
  → Scheduler.schedule
       ├── prefill：分配/复用 block，允许队首 chunked prefill
       └── decode：每个 running sequence 安排 1 token
  → ModelRunner.run
       ├── prepare_prefill / prepare_decode
       ├── eager forward 或 decode CUDA graph replay
       └── rank 0 sampling
  → Scheduler.postprocess
       ├── 登记完整 prefix block 的 hash
       ├── append token
       └── 完成后 deallocate
```

`LLMEngine.generate()` 是同步循环；TP>1 时 rank 0 通过固定大小 shared memory 和
pickle 把方法及参数广播给本机子进程。它没有跨节点控制面、DP/PP/EP 或 serving
协议，这也是教学代码保持紧凑的关键。

## 三、Scheduler

### 1. prefill

`Scheduler.schedule()` 先扫描 `waiting`：

- 新请求先由 `BlockManager.can_allocate()` 计算完整 prefix block 命中数和所需新块；
- block 不足时停止本轮 prefill，而不是继续扫描后面的请求；
- batch token budget 不足时，只允许本轮第一个请求做 chunked prefill；
- prompt 全部调度完成后，请求转为 `RUNNING`。

只要本轮选中了 prefill 请求，函数立即返回，不再混入 decode。因此 nano-vLLM 是
连续批处理，但不是同一 forward 内的 prefill/decode 混批。

### 2. decode 与抢占

decode 每个 running request 安排一个 token。跨块而没有空闲块时，scheduler 从
队尾抢占请求：释放其整张 block table、状态退回 WAITING，之后重新 prefill。没有
host swap、部分恢复或多级抢占状态。

如果所有候选都无法被调度，代码最终触发 `assert scheduled_seqs`。这是简化实现，
不能把它描述成生产级请求失败处理。

## 四、BlockManager 与 prefix caching

### 1. 块与引用

`BlockManager` 维护：

- `free_block_ids`：可重新分配的物理块；
- `used_block_ids`：当前被请求引用的块；
- `ref_count`：多个请求共享同一物理 prefix block；
- `hash_to_block_id`：链式 prefix hash 到一个物理块的映射。

默认 `kvcache_block_size=256`，但配置断言是 `block_size % 256 == 0`，所以允许
256、512 等倍数，并非硬编码“只能等于 256”。`Sequence.block_size` 在 engine
初始化时同步为配置值。

### 2. 链式 hash

第 i 个完整块的 hash 由“上一块 hash + 当前块 token ids”计算。查询仍会比较
`Block.token_ids`，避免只凭 hash 接受错误内容。只有完整块会登记，因此 prefix
复用粒度是 block，不支持 radix tree 的任意 token 边界。

一个 hash 只指向一个 block id；物理块被重新分配时，旧 hash 映射才被清理。这里
没有主动 LRU。更准确的描述是“空闲块仍可保留 prefix 身份，直到复用该物理块时
失效”，而不是表永远只增或存在某种隐藏驱逐算法。

### 3. Python 切片语义

`Sequence.block(i)` 使用 `self.token_ids[start:end]`。Python list 切片会创建新 list，
不是零拷贝 view；`prompt_token_ids` 和 `completion_token_ids` 也一样。旧笔记写成
“就地切片、不复制”是错误的。

## 五、ModelRunner

### 1. KV cache

`allocate_kv_cache()` 根据 warmup 峰值内存、GPU utilization 和模型配置计算 block
数量，然后一次性创建：

```text
[K/V, layer, block, block_size, local_kv_head, head_dim]
```

每个 attention layer 的 `k_cache`/`v_cache` 指向大 tensor 的相应 view。QKV 的 KV
head 按 TP rank 切分；它并不是“没有 GQA”，而是只实现了这类常规 MHA/GQA dense
Qwen3 pool，没有 SGLang 的 MLA、SWA、Mamba、量化 KV 和 sidecar 组合。

### 2. prefill 输入

`prepare_prefill()`：

- 只收集未缓存 token 的 `input_ids` 和 positions；
- 构造 `cu_seqlens_q`/`cu_seqlens_k`；有 prefix hit 时两者不同；
- 根据逻辑 block table 生成写 KV 的 `slot_mapping`；
- 有 prefix hit 时传 `block_tables` 给 paged attention；
- CPU tensor 用 pinned memory，并通过 non-blocking copy 送入 GPU。

这里已经会为物理 block table 重建 slot mapping。旧版“没有 block-table rewrite，
生产引擎靠 PagedAttention 解决碎片”的表述把逻辑映射与物理 compaction 混淆了。
准确差异是：nano-vLLM 有逻辑→物理映射，但没有在线迁移、压缩或分层重映射。

### 3. decode 输入

每个 sequence 只提交最后一个 token、当前位置、context length、写入 slot 和完整
block table；attention 调用 `flash_attn_with_kvcache`。

### 4. CUDA graph

nano-vLLM 只为 decode 捕获 CUDA graph。`run_model()` 遇到 prefill 会直接 eager，
所以不存在旧笔记所说的“prefill 静态宽度浪费”。

decode graph 按 batch size 档位捕获：1、2、4、8，之后每 16 一个档位直到 512。
回放选择第一个能容纳真实 batch 的档位，并用 `slot_mapping=-1` 让 padding 行不写 KV。
所有档位共享按 `max_model_len` 宽度分配的 block-table buffer；这会带来静态 buffer
容量开销，但不是按真实 context length 单独捕获 graph。

## 六、Qwen3 与权重加载

`models/qwen3.py` 接收 `transformers.Qwen3Config`，层数、hidden size、head 数等都
来自配置，因此实现并非 Qwen3-0.6B 专用。上游示例和 benchmark 使用 0.6B，但其他
满足该 dense Qwen3 实现及 TP divisibility 假设的尺寸也可构建；是否能运行还取决于
显存和 kernel 支持。

模型包含 Q/K norm、RoPE、QKV/MLP packed mapping 和 TP linear。loader 遍历
safetensors；遇到 `packed_modules_mapping` 时把 q/k/v 或 gate/up 分发到合并参数。
它没有生产引擎的量化 loader、远端格式、并行加载与完整模型注册系统。

## 七、与当前 SGLang 的对应关系

SGLang 路径相对 `python/sglang/srt/`：

| nano-vLLM | SGLang 当前对应 | 主要差异 |
|---|---|---|
| `LLMEngine` | `managers/tokenizer_manager.py`、`scheduler.py`、`detokenizer_manager.py` | 同步单机循环 vs serving 管线 |
| `Scheduler` | `managers/scheduler.py`、`schedule_policy.py` | SGLang 有更多 batching、优先级、分布式与 overlap 分支 |
| `Sequence` | `managers/schedule_batch.py` 等 request/batch 状态 | SGLang 请求元数据和生命周期更完整 |
| `BlockManager` | `mem_cache/allocator/` + `unified_radix_cache.py` + `unified_cache/` | block hash vs 统一 radix tree/components |
| KV 大 tensor | `mem_cache/kv_cache_configurator.py`、`memory_pool.py` | 多 pool、dtype、模型拓扑；VMM 工具位于 `utils/cuda_vmm_utils.py`，并非旧路径 `model_executor/cuda_vmm_utils.py` |
| `ModelRunner` | `model_executor/model_runner.py`、`model_executor/runner/`、`input_buffers.py` | eager/CUDA graph/backend 组件化 |
| decode CUDA graph | `model_executor/runner/decode_cuda_graph_runner.py` | SGLang `ShapeKey.size` 在 decode 表示 batch size，并可附加 stream/variant/DSA key |
| 无 prefill graph | `model_executor/runner/prefill_cuda_graph_runner.py` | SGLang `ShapeKey.size` 在 prefill 表示 num_tokens，并支持执行 variant |
| TP linear | `layers/linear.py`、`distributed/` | 更多 topology、backend 和加载 hook |
| Qwen3 packed mapping | `models/qwen3.py`；CLI 参数分组在 `arg_groups/` | 旧笔记的 `args_groups/` 路径拼写错误 |
| 无分层 KV | `unified_radix_cache.py`、`managers/cache_controller.py`、`pool_host/` | SGLang 可把 KV 扩展到 L2/L3 |

SGLang 当前 `ShapeKey` 不是 `(batch size, max sequence length)` 二元组。统一定义是
“当前 phase 的 capture size”：prefill 为 `num_tokens`，decode 为 `bs`，并可组合
stream index、variant label 和 DSA variant。

## 八、nano-vLLM 刻意省略的生产问题

### KV 与内存

1. prefix cache 只有完整 block hash，没有任意 token 粒度 radix tree 和主动 LRU；
2. KV 只能留在 GPU 或被物理块复用，没有 L2/L3 backup、prefetch 与失败恢复；
3. 没有 KV dtype/布局矩阵、hybrid pool、draft sidecar、在线迁移或压缩；
4. 每个完整块保留一份 Python token list 用于碰撞校验，hash 也在 Python/NumPy 路径；
5. 不能由这份 toy 实现反推“生产系统必定在 C++/Rust 并行哈希”，不同引擎实现不同。

### 调度与运行时

1. 每轮只做 prefill 或 decode，chunked prefill 只允许队首一个请求；
2. 抢占会丢弃整段 KV 并重算，没有 swap/retraction 的多态状态；
3. TP 只支持本机 spawn、固定端口和 pickle/shared-memory 控制；
4. 无 serving API、DP/PP/EP/PD、分布式容错和生产级 admission control；
5. OOM 路径依赖断言，没有请求级错误结果。

### 模型与运维

1. 模型家族局限于 dense Qwen3，attention 锁定 flash-attn；
2. 无 LoRA、多模态、量化权重、投机解码、结构化输出等；
3. CUDA graph 只覆盖 decode batch 档位；
4. 无 Prometheus metrics、trace、系统化 profile hook 和结果包规范。

## 九、适合的练习

1. 为 `BlockManager` 写纯 CPU 单元测试：共享、碰撞校验、ref count、物理块复用失效；
2. 加一个明确的 LRU，将“空闲但保留 hash 的块”与“待复用块”分开，测命中率变化；
3. 把完整 block hash 替换为简化 radix tree，比较不同 prefix 分布的可复用 token 数；
4. 设计一个 host tier 状态机，先验证 correctness，再加入异步 copy 与回载；
5. 将每次实验保存 workload、seed、配置和原始事件，而不是只保留 tok/s 汇总。

这些练习能建立调度与 KV 生命周期直觉，但完成 toy 扩展不等于完成 SGLang 上游优化；
后者还需要在当前统一树、模型 pool 组合和真实 GPU workload 上重新验证。
