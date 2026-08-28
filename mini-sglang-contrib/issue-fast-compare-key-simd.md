# [Perf] Vectorize fast_compare_key with SIMD

`fast_compare_key` (`python/minisgl/kernel/csrc/src/radix.cpp`) performs element-wise comparison of two 1-D CPU int tensors and returns the position of the first mismatch. It is called from `RadixTreeNode.get_match_len`, which runs on every node visit inside `_tree_walk` — i.e. on every `match_prefix` / `insert_prefix` in the request scheduling path. This is one of the few CPU-side hot spots owned by the scheduler thread.

## Current implementation

The kernel dispatches to `std::mismatch` over `int32_t` / `int64_t`:

```cpp
if (a.dtype().bits == 64) {
    const auto diff_pos = std::mismatch(a_ptr_64, a_ptr_64 + common_len, b_ptr_64);
    return diff_pos.first - a_ptr_64;
} else {
    const auto diff_pos = std::mismatch(a_ptr_32, a_ptr_32 + common_len, b_ptr_32);
    return diff_pos.first - a_ptr_32;
}
```

libstdc++ implements `std::mismatch` as a scalar element-by-element loop. Throughput is roughly one element compared per cycle regardless of the host's vector width.

## Proposal

Vectorize the comparison loop:

- **AVX2**: compare 8 × int32 (4 × int64) per iteration with `_mm256_cmpeq_epi32` / `_mm256_cmpeq_epi64`; locate the first differing lane with `_mm256_movemask_epi8` + `__builtin_ctzll`.
- **AVX-512** where available: 16 × int32 per iteration via mask registers.
- **NEON** on aarch64 Linux (README lists aarch64 as supported).
- Scalar tail loop for the remainder.
- Dispatch: runtime via `__builtin_cpu_supports`, or separate translation units compiled with `-mavx2` / `-mavx512f`, whichever fits the tvm-ffi AOT build best.

## Expected impact

For long keys: roughly 8× (AVX2) to 16× (AVX-512) on the comparison itself. In absolute terms each call is small (a few microseconds for a 32k-token key), but this function runs per request on the scheduler thread, and overlap scheduling depends on keeping that thread's CPU work minimal, so shaving per-request cost matters. It helps most for workloads with deep radix trees (long shared prefixes, `page_size=1`).

## Benchmark plan

- Microbenchmark sweeping key length {128, 1k, 4k, 32k} for int32 and int64, comparing scalar vs SIMD. Can live under `tests/kernel/` or `benchmark/` as a small standalone C++ script.
- End-to-end sanity check with `benchmark/offline` (WildChat prompts exercise prefix reuse) to confirm no regression.

## Notes

- Both int32 and int64 paths need to stay supported (the kernel currently checks dtype equality before dispatching).
- #150 adds a pure-torch fallback for when the AOT module is unavailable; this proposal only touches the kernel path.

Would a PR in this direction be welcome? Also, is there a preferred way to pass arch-specific compile flags through the tvm-ffi AOT build system?