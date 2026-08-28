# #35438 分析与上游修复确认(2026-08-24)

## 一、结论:上游已存在完整修复,无需另起炉灶

Issue #35438(bench_multiturn 忽略 reasoning 输出、不报 TPOT)的**作者本人
(wes-lyu)**在 2026-08-19 提交了 PR **#35443**
"Fix reasoning metrics and add TPOT to bench_multiturn",当前 OPEN:
- https://github.com/sgl-project/sglang/pull/35443
- 3 个文件,+118/-4:bench_multiturn.py、cache_hit_kit.py、新增 CPU 单测
  (test/registered/unit/test_cache_hit_kit_metrics.py,4 个测试)
- 修改内容与 issue 的 Expected behavior 逐条对应:reasoning parsing、TPOT
  公式、NumPy 聚合、Average/P90/P99/Median/Max 输出顺序
- CI:pr-test 尚无门禁结果、AMD ROCm job 红(与本 PR 无关,AMD 基建问题概率大)

同名 PR #35439 为同一分支的重复提交,已 CLOSED(作者自己关掉)。

**建议:跟随 #35443,而不是从零提交一个重复 PR。** 审阅意见见下,若采纳
其中一两条,可以 review comment 的形式提出,或在其基础上提交补充 PR。

## 二、对 #35443 的代码审查

### 已确认正确的部分

1. delta 解析:`get_openai_chat_output_delta`(cache_hit_kit.py:16-21)把
   `reasoning_content`、`reasoning` 与 `content` 统一视为生成输出,
   ttft/itl 逻辑只依赖"是否有文本"这一条件——**修复了 TTFT/ITL 全 0 的根因**
   (原代码 cache_hit_kit.py:133 只读 `delta.content`)。
2. TPOT 公式与 hicache/bench_serving.py:317-321 一致:
   `(latency - ttft) / (completion_tokens - 1)`,空输入/非法输入返回 None,
   上层用 `getattr(response, "tpot", None) is not None` 过滤——对老对象兼容。
3. `output.generated_len = len(all_output_ids)`(sglang /generate 路径)替代
   `len(output.itl) + 1`:server 每个 chunk 都带完整 output_ids,该值更准确
   (例如 ignore_eos 截断时不再依赖 token 间隔计数)。
4. 单测覆盖 delta 三种字段、tpot 三个非法输入分支、空列表聚合——质量合格。
5. `raw_delta.get("content") or ""` + `reasoning or ""` 的兜底顺序正确:
   任一字段缺失不会中断解析。

### 发现的可改进点(按重要性)

**A. assistant history 不回灌 reasoning → 后续轮次 cache hit 率系统性失真
(对"缓存命中评测工具"是实质缺陷)**

- bench_multiturn.py:449-453(openai 模式)把 `response.generated_text` 作为
  assistant 消息追加进下一轮 history;而 #35443 中
  `generated_text += content`(cache_hit_kit.py:171)——**reasoning 文本被丢弃**。
- 结果:第二轮及以后的 payload 与 server 端真实 context(含 reasoning,由
  `--reasoning-parser` 按 template 编码)不一致 → 即使 server 缓存完好,
  round>=1 的 `cache_hit_rate` 也会偏低甚至为 0。对一个**专门测缓存命中率**
  的 benchmark,这会直接污染核心指标。
- 单轮(round=0)场景不受影响;issue 的 repro 是 num-rounds 2,同样中招。

**B. `len(all_output_ids)` 的防御性回归**

- 若 server 因某种参数组合(如 return_logprob 关闭时的变体路径)不返回
  `output_ids`,`generated_len` 从 `len(itl)+1` 退化为 0,T POT 静默消失
  (calculate_tpot 对 completion_tokens<=1 返回 None)。当前 server 行为总是
  返回,风险低;但改为
  `generated_len = len(all_output_ids) or len(output.itl) + 1` 更稳。

**C. 小问题**

- 若 delta 同时含 reasoning 与 content(罕见),`get_openai_chat_output_delta`
  拼接两段会合并成一个时间戳样本——与逐 token chunk 的假设不符,但对该
  场景无害,可不改。
- `calculate_tpot_statistics` 的空列表返回全 0.0——bench_serving 的空数组
  上 np.mean 会告警,这里显式清 0 更好,已做。

### 补充修复草案(仅当要提交补充 PR 时)

```python
# cache_hit_kit.py,async_request_openai_chat_completions 内
if output_delta:
    content = raw_delta.get("content") or ""
    reasoning = (
        raw_delta.get("reasoning_content") or raw_delta.get("reasoning") or ""
    )
    # Keep history faithful to the served context so round>=1 cache hit
    # ratios measure the cache, not the prompt mismatch.
    generated_text += reasoning + content
    ...
```

## 三、若决定自行提交 PR 的 checklist

> 前提:先确认 #35443 的状态。若已被 merge,本 checklist 作废,改提 review。

1. **fork + 分支**:`git checkout -b fix/hicache-multiturn-tpot`(基于最新 main)。
   注意本仓库 third_party/sglang 是只读浅克隆,须在个人 fork 上操作。
2. **改动范围**(沿用 #35443 的三个文件 + 上述补丁 A/B):
   - `benchmark/hicache/bench_multiturn.py`(tpot 收集/聚合/打印)
   - `python/sglang/test/kits/cache_hit_kit.py`(delta 解析 + tpot 计算 +
     reasoning 回灌)
   - `test/registered/unit/test_cache_hit_kit_metrics.py`(补齐
     reasoning 回灌的断言)
3. **风格**:SGLang 仓库有 ruff + 类型检查(CI lint),新增函数需遵循
   同文件行文(unittest + CustomTestCase + register_cpu_ci 与 #35443 相同);大
   类/benchmark 辅助函数无 arg 文档要求,但函数名需自解释。
4. **单元测试**:`python -m pytest test/registered/unit/test_cache_hit_kit_metrics.py`
   或文件直接 `python test/...py`(本就带 unittest.main)。
5. **提交信息**与 checklist 照抄 #35443 的格式,引用 `Fix #35438`。
6. **不推送**:等用户确认后再 push + gh pr create。

## 四、存档

- `patches/35438-fix-upstream-PR35443.diff` — 上游 PR #35443 的完整 diff
  (2026-08-24 抓取),供离线审阅。