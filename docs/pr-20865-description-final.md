# PR #20865 最终描述(平实风格,替换此前 AI 味草稿)

> 标题:`Add CPU unit tests for mem_cache kv_cache_dtype and allocation_sizing`
> 风格说明:无形容词、无客套、只陈述事实,符合仓库 PR 模板四段结构。

---

## Motivation

Part of #20865. `srt/mem_cache/kv_cache_dtype.py` and the page-alignment
arithmetic in `srt/mem_cache/allocation_sizing.py` currently have no unit
tests. Both are pure logic (no server, model, or GPU needed), so they are
cheap to cover with CPU-only tests.

## Modifications

Two new test files under `test/registered/unit/mem_cache/` (28 cases total,
CPU only):

- `test_kv_cache_dtype.py` — covers `configure_kv_cache_dtype`: auto detection
  with and without an FP8 quant config (CUDA and HIP paths), explicit dtype
  spellings, the deprecated `fp4_e2m1` error, both `nvfp4` availability
  branches, draft-worker dtype override, and the DFLASH fa4 override
  (K.dtype == Q.dtype constraint).
- `test_allocation_sizing.py` — covers `page_aligned_decode_alloc_lens`:
  round-up to page, exact page boundary, never-shrink-below-allocated,
  `page_size=1`, multi-request accumulation, and reserve larger than a page.

No production code changes.

## Accuracy Tests

Local run (CPU-only venv, torch 2.13):

```
27 passed, 1 skipped in 7.08s
```

The skipped case is the `nvfp4`-unavailable error path, which cannot trigger
on a torch build that already ships `float4_e2m1fn_x2`; the supported-path
variant of the same branch is covered with a patched attribute.

## Checklist

- [x] `pre-commit run --all-files` passes
- [x] Tests registered via `register_cpu_ci`; `scripts/lint/check_registered_tests.py` passes
- [x] No production code modified
