# PR #35443 Review Draft(PENDING — 未发布)

> 状态:草稿,仅供审核。发布前请用户确认,然后执行文末命令。
> 对象:`sgl-project/sglang` PR #35443 "Fix reasoning metrics and add TPOT to bench_multiturn"
> 位置:benchmark/hicache/bench_multiturn.py、python/sglang/test/kits/cache_hit_kit.py、test/registered/unit/test_cache_hit_kit_metrics.py
> 2026-08-24 更新:按独立审查(见 kv-tiering/docs/05-issue-35438-analysis.md)并入第一重点
> —— reasoning 丢弃导致多轮 cache_hit_rate 失真;顺序:① reasoning 丢弃 ② getattr ③ TPOT 分母。

---

## Review comment body(英文,直接可发布)

Thanks for the fix — the root cause (TTFT/ITL collapsing to zero because
reasoning chunks never reach the `delta.content` branch) was real, the TPOT
formula was independently verified against the reported number, and the helper
functions are tested separately from the client code. Three comments, in order
of importance — the first one touches the benchmark's core metric.

### 1. `generated_text` drops reasoning, which biases multi-turn cache-hit rates

`get_openai_chat_output_delta` counts `reasoning_content` / `reasoning` chunks
for TTFT/ITL timing, but `generated_text` still only appends `content`:

```python
output_delta = get_openai_chat_output_delta(raw_delta)
if output_delta:
    content = raw_delta.get("content") or ""
    generated_text += content
```

`bench_multiturn` rebuilds the next round's history from the returned text
(`history.append({"role": "assistant", "content": response.generated_text})`,
bench_multiturn.py:449-453). With `--reasoning-parser qwen3`, the served
context includes the assistant's reasoning tokens, but the next round's
payload does not — so from round 1 on, the client prompt no longer matches the
server's cached prefix and `cache_hit_rate` is systematically underestimated
even when the cache is perfectly healthy. For a benchmark whose headline
metric is the cache-hit rate, that measures the prompt mismatch, not the
cache. Round 0 is unaffected; the issue's own repro with `--num-rounds 2` is
affected.

Suggested minimal change — keep the replayed history faithful to the served
context:

```python
if output_delta:
    content = raw_delta.get("content") or ""
    reasoning = (
        raw_delta.get("reasoning_content") or raw_delta.get("reasoning") or ""
    )
    # Keep history faithful to the served context so round>=1 cache hit
    # ratios measure the cache, not the prompt mismatch.
    generated_text += reasoning + content
```

Side benefit: for reasoning models, pasting only `content` back into the next
turn is also _conversationally_ wrong — the model never gets to condition on
its own past reasoning tokens — so this aligns the client with the served
context on both axes.

### 2. `getattr(response, "tpot", ...)` — defensive access hides a missing field

In `benchmark/hicache/bench_multiturn.py`:

```python
if getattr(response, "tpot", None) is not None:
    self.performance_metrics["tpot"].append(response.tpot)
```

`tpot` is not a declared field of `RequestFuncOutput` — the request funcs set
it dynamically, so `getattr` with a default silently tolerates a future
request func that forgets to compute it, and benchmarks would then report a
TPOT of 0 with no error. SGLang's code rules reject defensive
`getattr`/`hasattr` for fields that should always exist (see
`.claude/rules/no-getattr-defensive.md`). Suggestion: make the contract
explicit by declaring the field on the dataclass, then access it directly:

```python
# python/sglang/benchmark/serving.py
@dataclass
class RequestFuncOutput:
    ...
    tpot: Optional[float] = None  # Time per output token; None when not measurable
```

```python
# benchmark/hicache/bench_multiturn.py
if response.tpot is not None:
    self.performance_metrics["tpot"].append(response.tpot)
```

With the field declared, any request func that forgets to set it fails loudly
(None samples just never make it into the stats), and type checkers see the
attribute.

### 3. TPOT denominator: the two request funcs use different sources

In `cache_hit_kit.py`, the sglang path uses `generated_len` as the denominator,
but the OpenAI path passes the raw `completion_tokens`:

```python
# async_request_sglang_generate
output.generated_len = len(all_output_ids)
output.tpot = calculate_tpot(output.latency, output.ttft, output.generated_len)

# async_request_openai_chat_completions
output.generated_len = (
    completion_tokens if completion_tokens else len(output.itl) + 1
)
output.tpot = calculate_tpot(output.latency, output.ttft, completion_tokens)
```

Two issues: (a) when the `usage` block is missing, `generated_len` falls back
to `len(itl) + 1` but the TPOT denominator still uses `completion_tokens` (0),
so `calculate_tpot` returns None and the sample silently drops; (b) the two
paths no longer agree on what "completion tokens" means. Suggest passing the
value that was already stored:

```python
output.tpot = calculate_tpot(output.latency, output.ttft, output.generated_len)
```

And a one-line comment documenting the intent would help future readers — e.g.
that OpenAI's `usage.completion_tokens` counts reasoning + content tokens
together, which is exactly the semantic the issue asked for (reported output
length is 32 including reasoning in the reproduction).

Minor, optional: the new unit test file sits at
`test/registered/unit/test_cache_hit_kit_metrics.py`. The mirrored-tree
convention (see `test/registered/unit/README.md`) maps `python/sglang/...` one
level at a time, so a path like `test/registered/unit/test_kits/` (or at least
a comment in the file naming the source) would keep the mirror unambiguous for
someone grepping "where is `cache_hit_kit.py` tested". Purely cosmetic — the
tests themselves are in the right place protocol-wise.

Thanks again — the fix itself is solid, and the reasoning-content handling
(`get_openai_chat_output_delta`) is exactly the right shape.

---

## 发布命令(未执行)

```bash
# 一遍确认正文无误后,用 --comment 模式发布(不会 approve / request changes)
gh pr review 35443 --repo sgl-project/sglang --comment \
  --body "$(cat /home/shane/github/ai-infra/kv-tiering/docs/pr-35443-review-draft.md  | sed -n '/^## Review comment body/,/^## 发布命令/p' | sed '1d;$d')"

# 若想先看一眼渲染效果,可先输出到终端:
# sed -n '/^## Review comment body/,/^## 发布命令/p' pr-35443-review-draft.md | sed '1d;$d'
```

> 红线:未执行任何 gh 写操作。发布=对外可见操作,需用户明确授权。
> 备注:第一点修复建议与 05-issue-35438-analysis.md 的补充草案一致(该文档
> 存有上游 PR 完整 diff 存档 patches/35438-fix-upstream-PR35443.diff,供离线审阅)。