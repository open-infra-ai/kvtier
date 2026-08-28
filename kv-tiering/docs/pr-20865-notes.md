# #20865 单测贡献笔记:mem_cache 覆盖

> 目标 issue:[#20865 Improve Unit Test Coverage](https://github.com/sgl-project/sglang/issues/20865)
> 分支:`test-mem-cache-coverage`(third_party/sglang,基于 main)
> 状态:本地完成 = 测试全绿 + lint 全过 + patch 已导出;**未 push / 未开 PR**
> 日期:2026-08-23

## 1. 覆盖空白分析(为何选这两个文件)

`python/sglang/srt/mem_cache/` 源码 vs `test/registered/unit/mem_cache/` 已有测试对照(grep
各测试文件的 import 反查):无对应测试且未被他人认领的文件为:

| 源文件 | CPU 可测性 | 结论 |
|--------|-----------|------|
| **kv_cache_dtype.py**(101 行) | `configure_kv_cache_dtype` 是纯决策函数,分支极丰富(draft 覆盖 / auto 探测 / 显式 dtype / 弃用 / 错误路径) | ✅ 选入 |
| **allocation_sizing.py**(96 行) | `page_aligned_decode_alloc_lens` 是纯页对齐算术;其余 getter 依赖 runtime context | ✅ 选入(纯函数部分) |
| index_key_cache.py | 依赖 `torch.cuda.use_mem_pool` / DSA 内核访问器,纯 CPU 需大量 mock | ❌ 排除 |
| kv_vmm_backing.py | 依赖 `cuda_vmm_utils` + `torch.cuda` 设备上下文 | ❌ 排除 |
| flush_cache.py(33 行) | 仅协议类,测试价值低(违反 PR 质量栏) | ❌ 排除 |
| multimodal_cache.py | 已被 shoutoutuoadi325 认领(2026-08-07) | 撞车排除 |
| chunk_cache.py | 已有 OPEN PR #29469 | 撞车排除 |
| common.py / utils.py | 分别已被 #31990、#27934 覆盖 | 撞车排除 |

**选题理由**:
1. 与 HiCache 主线协同 —— `--kv-cache-dtype` 正是本仓库 kv-tiering 实验(#31600 复现脚本)的
   变量之一,HiCache fp8 KV 的自动探测逻辑就在这里;`allocation_sizing` 的整页分配口径是
   HiCache host 池/spec v2 分配的前置常识。
2. 纯粹、零 mock 风险:`configure_kv_cache_dtype` 无状态,所有分支可通过 keyword 参数 + 两个
   `patch`(`_is_hip`、`fp8_dtype`)到达;`page_aligned_decode_alloc_lens` 连 patch 都不需要;
   两个文件都是 issue 点名希望的"edge cases, not just happy paths"型目标。
3. 无 GPU 依赖:本机 CPU-only 环境(python 3.12 + CPU torch)直接跑通。

## 2. 测试内容

### test/registered/unit/mem_cache/test_kv_cache_dtype.py(21 例)

`configure_kv_cache_dtype(*, server_args_kv_cache_dtype, model, model_dtype, is_draft_worker, is_dflash, speculative_draft_attention_backend, speculative_draft_kv_cache_dtype)`:

- auto 路径:无 quant_config → model_dtype;FP8 quant_config(CUDA / HIP 两平台);非 FP8 algo;
  quant_config 无属性
- 显式 dtype:fp8_e4m3 / fp8_e5m2 / mxfp8 / bf16(两拼写)——断言 `resolved` 仅由 auto 路径
  设置(显式时保持 None),dtype 正确
- 错误路径:fp4_e2m1 弃用 ValueError、未知 dtype ValueError、nvfp4 无 torch 支持时 ValueError
  (有支持时可用,用 patch 双分支)
- draft worker:draft dtype 覆盖 server dtype;auto 透传
- DFLASH fa4:draft KV 被覆盖为 compute dtype(读不了量化的目标 KV);已 bf16 时不覆盖;
  fa4 以外后端保持量化 KV
- `TORCH_DTYPE_TO_KV_CACHE_STR` 映射完整性
- HIP 模拟注意:需同时 patch `_is_hip` 与 `fp8_dtype`(后者在 `fp8_kernel` import 时固定为
  fnuz 或 e4m3fn,单独 patch 平台标志会产生假故障)—— 测试里如实模拟了整平台

### test/registered/unit/mem_cache/test_allocation_sizing.py(7 例)

`page_aligned_decode_alloc_lens(reqs, *, reserve, page_size)`:

- 空列表、上取整到页、恰整页不扩、allocated 领先时不缩、page_size=1 恒等+reserve、
  多请求累计 needed、大 reserve 跨 2 页
- 注意源 API 的字段布局:`r.kv.kv_allocated_len`(嵌套)vs `r.kv_committed_len`(平级),
  构造 mock req 时按真实布局

**未覆盖(留给后续 PR)**:`get_alloc_page_size` / `get_alloc_len_per_decode` /
`get_alloc_reserve_per_decode` / `get_req_to_token_extra_context_len` —— 读 runtime
context bag(schedule/parallel/spec),需 `get_context().override()` 构造上下文,与本次
"纯函数"定位不符;已在 PR 描述中说明为 follow-up。

## 3. 本地运行结果(CPU-only,无 GPU)

```bash
# 环境:python 3.12.3 + CPU torch 2.13.0+whl/cpu;PYTHONPATH=python(源码直跑,未安装包)
cd third_party/sglang
PYTHONPATH=python /tmp/sglang-venv/bin/python -m pytest \
  test/registered/unit/mem_cache/test_kv_cache_dtype.py \
  test/registered/unit/mem_cache/test_allocation_sizing.py -v

======== 27 passed, 1 skipped, 17 warnings, 2 subtests passed in 7.08s =========
# skip 项:test_nvfp4_error_when_dtype_unavailable —— 本机 torch 已含 float4_e2m1fn_x2,
# 错误路径仅在旧 torch 上执行(CI 若用 2.8+ 同样 skip,两条路径都被 patch 双分支覆盖)
```

Lint:

```bash
/tmp/sglang-venv/bin/ruff check --select=F401,F821,UP037 test/...   # All checks passed!
/tmp/sglang-venv/bin/isort --check test/...                          # 通过
python -m py_compile test/...                                        # 通过
/tmp/sglang-venv/bin/python scripts/lint/check_registered_tests.py   # 通过(无输出)
```

## 4. 产物

- patch:`/home/shane/github/ai-infra/kv-tiering/patches/20865-mem-cache-tests.patch`
  (2 新文件,295 行插入;`git add -N` 后 `git diff main` 导出)
- 分支:third_party/sglang `test-mem-cache-coverage`(未 push)
- 本地依赖(venv /tmp/sglang-venv):torch-cpu 2.13.0、transformers==5.12.1(仓库 pin)、
  tokenizers==0.22.2、triton、msgspec、datasets、gguf、compressed-tensors、openai==2.6.1、
  xgrammar、sentencepiece、dill、pybase64、pillow、torchvision-cpu、ipython、orjson 等
  (~sglang 完整依赖的纯 CPU 子集;CI 用 `uv pip install -e python[dev]`,无需本清单)

## 5. PR 提交 checklist(提交时逐项核对)

- [ ] 分支 `test-mem-cache-coverage` 基于最新 main(fork 后 push,禁止直推 main)
- [ ] 新增 2 文件均:`CustomTestCase`、模块级 `register_cpu_ci(est_time=2, suite="base-a-test-cpu")`、`if __name__ == "__main__": unittest.main()`
- [ ] ruff(F401/F821/UP037)+ isort + py_compile + check_registered_tests.py 本地通过
- [ ] 未改任何生产代码(纯测试 PR)
- [ ] PR 标题按 issue 惯例:`[Test] ...`(见草稿)
- [ ] PR 描述按仓库模板 + issue 要求:附"本地命令 + 完整输出"(见上节)
- [ ] 开 PR 后评论认领冲突已检查:mem_cache 下无活跃竞争 PR(kv_cache_dtype / allocation_sizing 均无)
- [ ] 等维护者加 run-ci 标签(或 /tag-and-rerun-ci)
- [ ] 60% 行覆盖闸门:新增被测行覆盖率远超(configure_kv_cache_dtype 全分支 + 路径算术)

## 6. PR 标题与描述草稿(英文)

```markdown
Title:
[Test] Add unit tests for srt/mem_cache/kv_cache_dtype and allocation_sizing

Body:
Closes #20865. (part of)

## Motivation
`configure_kv_cache_dtype` and `page_aligned_decode_alloc_lens` in
`python/sglang/srt/mem_cache/` have no direct unit coverage. The dtype
configurator is pure decision logic with rich branching (draft-worker
override, "auto" quant detection on CUDA/HIP, deprecated dtype errors) and the
alloc-lens helper is a page-rounding arithmetic that is easy to break in
"looks equivalent" rewrites. Both are CPU-testable with zero mocking surface
beyond a platform flag patch.

## Modifications
- `test/registered/unit/mem_cache/test_kv_cache_dtype.py` (21 cases):
  every branch of `configure_kv_cache_dtype` — auto detection (no/FP8/non-FP8
  quant config, CUDA and HIP), explicit dtypes (resolved tag only set by
  "auto"), deprecation and unsupported-dtype errors, nvfp4 availability
  guard, draft-worker dtype override, DFLASH fa4 KV override, and the
  dtype→string mapping.
- `test/registered/unit/mem_cache/test_allocation_sizing.py` (7 cases):
  `page_aligned_decode_alloc_lens` page-rounding — exact boundary, non-shrink,
  page_size=1, multi-request accumulation.

Deliberately out of scope (follow-up): the runtime-context getters in
`allocation_sizing.py` which need `get_context().override()` fixtures.

## Accuracy Tests
N/A — test-only PR, no behavior change. Local run (CPU-only, no GPU):

pytest test/registered/unit/mem_cache/test_kv_cache_dtype.py
      test/registered/unit/mem_cache/test_allocation_sizing.py -v
======== 27 passed, 1 skipped, 17 warnings, 2 subtests passed in 7.08s =========
(skip: nvfp4-unavailable error path on torch builds that already ship
float4_e2m1fn_x2; both branches are exercised via patch in the other case)

## Checklist
- [x] Format code according to the project guidelines (ruff/isort pass).
- [x] Add unit tests (28 new cases; CustomTestCase + register_cpu_ci).
- [x] Documentation update is not required for a test-only PR.
- [x] Accuracy assessment: N/A, no inference behavior touched.
- [x] Follow the SGLang code style guidance.

cc @ispobock (who maintains #20865)
```

## 7. 下一步

1. 用户 review 测试文件(重点:断言是否忠实于语义 —— 尤其 `resolved` 仅 auto 路径设置、
   HIP 双 patch、`r.kv_committed_len` 平级字段)
2. fork + push 分支 → 开 PR(标题/描述用第 6 节草稿)→ 等 run-ci 标签
3. 后续候选(按价值):`allocation_sizing` runtime-context getters(override fixtures)、
   `index_key_cache` 的 `_layer_num_pages`/`_buffer_shape` 布局算术(需 mock pool)、
   `kv_vmm_backing` 的地址对齐辅助