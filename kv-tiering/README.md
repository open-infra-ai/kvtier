# kv-tiering：推理引擎 KV 分层存储学习与实验

本项目用于学习 SGLang HiCache 和 vLLM KV offloading，并把源码结论、可复现实验和
上游贡献连成证据链。当前主线是 SGLang；vLLM 作为架构对照。

## 完成边界

- 源码阅读只能支持“实现如何工作”或“值得测量的假设”，不能替代性能数据；
- 单卡 L2 微基准只能说明该模型和配置下的 host 回载表现，不能自动复现多卡、L3、
  EAGLE 场景；
- 只有真实 GPU 运行、缓存路径核验和结果包审计完成后，才可以对外给出性能结论；
- 上游 issue/PR 的状态会变化，公开评论前必须再次核对当前 `main` 和重复工作。

## 当前技术基线

SGLang 自 [PR #35081](https://github.com/sgl-project/sglang/pull/35081) 起默认使用
`UnifiedRadixCache`。启用 `--enable-hierarchical-cache` 后，HiCache 作为 L2/L3
能力附着在这棵统一前缀树上。`HiRadixCache` 仍在代码中，但不应再写成默认架构。

建议按以下顺序阅读：

1. `mem_cache/unified_radix_cache.py`：前缀树与 HiCache 的统一入口；
2. `mem_cache/unified_cache/`：树组件、FULL/SWA/MAMBA 状态及 storage attachment；
3. `managers/cache_controller.py` 与 `mem_cache/l2_transfer.py`：L2/L3 I/O 编排；
4. `mem_cache/pool_host/`：不同 KV 布局的 host pool 与传输分支；
5. `observability/metrics_collector.py`：已有指标，避免重复提案。

## 可执行任务

| 优先级 | 任务 | 当前边界 |
|---|---|---|
| 1 | 单卡 HiCache L2 回载微基准 | 脚本已完成离线测试，尚无真实 GPU 数据；不是 #31600 复现 |
| 2 | 按原配置复现 #31600 | 需要 GLM-5.2、FP8、EAGLE、L2+Mooncake、TP/CP，并比较 accept rate、decode throughput 和总时长 |
| 3 | HiCache 多卡同步与 host 内存局部性 profiling | 目前只是测量方向；不能先验宣称 all-reduce 或 NUMA 是瓶颈 |
| 4 | L3 prefetch 失败分类可观测性 | 已有大量传输指标；剩余候选是有限枚举的失败原因和结构化日志 |
| 5 | KV 换出压缩 | 需先建立 correctness 与带宽基线，再判断是否值得实现 |

## 文档

- [`docs/01-sota-survey.md`](docs/01-sota-survey.md)：SGLang/vLLM 当前架构与 issue
  边界；
- [`docs/02-hicache-dataflow.md`](docs/02-hicache-dataflow.md)：当前
  `UnifiedRadixCache + HiCache` 数据流及 #31600 分析；
- [`docs/03-nano-vllm-walkthrough.md`](docs/03-nano-vllm-walkthrough.md)：nano-vLLM
  调度、块管理与 SGLang 对照；
- [`bench/repro-31600/README.md`](bench/repro-31600/README.md)：历史目录名下的单卡
  L2 微基准说明。

## 状态

- [x] 2026-08-25：重新核对 SGLang `main`，修正默认缓存架构、#31600 根因范围、
  HiCache 指标和 nano-vLLM 映射；
- [x] 2026-08-25：重构单卡 L2 微基准，加入精确 token、cache namespace 隔离、
  L2 命中验证、指标交叉验证和 JSON 结果包；
- [x] 2026-08-28：建立根级 Git 边界、客户端依赖与 `make test` 入口；
- [x] 2026-08-28：加入结果包审计器，正式运行要求记录 40 位 SGLang commit；
- [ ] 在真实 GPU 上运行并审计单卡 L2 微基准；
- [ ] 若要回应 #31600，另行搭建尽量匹配原报告的多卡 EAGLE/L3 实验；
- [x] 已提交 [PR #36115](https://github.com/sgl-project/sglang/pull/36115)；本轮未重复审核；
- [x] 已提交 [PR #36116](https://github.com/sgl-project/sglang/pull/36116)；本轮未重复审核；
- [x] 已发布并更正
  [#35808 评论](https://github.com/sgl-project/sglang/issues/35808#issuecomment-5396206747)，
  本地证据见 `docs/issue-35808-comment-draft.md`。
