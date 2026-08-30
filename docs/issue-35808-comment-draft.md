# issue #35808 讨论评论（纠正版）

> 状态：已于 2026-08-24 更新原评论，不是新增评论。
> 线上地址：https://github.com/sgl-project/sglang/issues/35808#issuecomment-5396206747
> 最新复核基线：`sgl-project/sglang` main `67853c580482dbc9b69aad5022a0cdd3a1606570`（2026-08-25）。

## 更正原因

Qwen 初稿漏查了已经合入的 HiCache 可观测能力，并把不同的预取结束状态都归为“数据存在但读取失败”。原评论已收缩为当前源码能够直接支持的结论。

| 原断言 | 复核证据 | 结论 |
|---|---|---|
| 分层传输没有指标 | PR #32388 已加入 L2 H2D/D2H token、byte 和 duration 指标；当前 `metrics_collector.py` 同时定义了 L2 transfer 指标和 L3 prefetch/backup 带宽直方图 | 错误，不能保留 |
| `kv_transfer_*` 是唯一传输指标 | 这组指标只用于 PD disaggregation，但不是 HiCache 的全部指标 | 错误，检索范围不完整 |
| 所有预取结束状态都对应 “present but read failed” | `best_effort`、timeout、低于 threshold 的 revoke 和真实读取失败语义不同 | 错误，只有 existence hit 后 `batch_get*` 失败符合该分类 |
| JIT 条件不满足会让所有 host-device copy 回退 | H2D/D2H、layout 和 IO backend 走不同分支，page-first write-back 还有独立的 JIT 能力判断 | 过度概括；“up to 3x”也不能作为该配置的实测损失 |
| 已有 per-layer hit counters | 当前实际指标是 `cached_tokens_total{cache_source=...}`，不是草稿所说的 per-layer counter | 术语错误，已更正为按 cache source/tier 统计 |
| #31600 的慢命中主要需要传输带宽诊断 | #31600 报告把 EAGLE draft state 未保存列为主要原因，把 JIT fallback 列为次要因素 | 原关联容易误导，已删除 |

## 当前可确认的缺口

- `UnifiedRadixCache` 内部维护了一部分 prefetch outcome 统计，但默认 `StorageMetricsCollector` 没有导出这些字段。
- `batch_exists` 命中后 `batch_get*` 失败只产生 warning，没有可按有限原因聚合的 Prometheus counter。
- `rid` 适合放在日志或 trace 中，不应作为 Prometheus label。

## 评论正文

见 `issue-35808-comment-body.txt`。该文件应与线上评论逐字一致。

## 更新既有评论

先确认 `gh api user --jq .login` 返回 `holtwood`，再执行：

```bash
gh api --method PATCH \
  repos/sgl-project/sglang/issues/comments/5396206747 \
  -F body=@docs/issue-35808-comment-body.txt
```

不要使用 `gh issue comment`，该命令会新增重复评论。
