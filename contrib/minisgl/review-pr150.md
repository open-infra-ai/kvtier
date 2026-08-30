# PR #150 评审评论草稿

> 发布前请逐条验证。「验证笔记」不发布。

---

## 拟发布评论

Read through the diff and traced the accounting against `insert_prefix`, `lock_handle`, and `split_at`. The walk-based counter check, the duplicate-index detection, and the tree-vs-free_slots overlap check together cover what #149 asked for. Details that look right to me:

- `split_at` preserves `ref_count` on both halves and conserves total length, so the walk's evictable/protected sum is consistent with how the counters are maintained.
- Root is excluded from `_iter_non_root_nodes`, which matters because root has `ref_count=1` but no `_length`.
- The `indices[::page_size]` stride in the CacheManager check is safe because node values are always page-aligned (`insert_prefix` aligns down, `split_at` splits at aligned positions).
- The manual free_slots update in `test_evict_tail_then_ok` matches what `_allocate` does with evicted pages (cache.py:109), so it is faithful to production.

A few suggestions:

1. **No negative tests.** All current cases are positive paths — nothing proves the check actually fires. Consider adding, e.g.:
   - Manually corrupt `self.evictable_size` → expect `RuntimeError` from the size walk.
   - Make two nodes own the same page index → expect `RuntimeError` from the duplicate check.
   Without these, a future refactor could silently break the checker while tests stay green.

2. **Silent fallback in `get_match_len`.** If the AOT module fails to build/load, radix matching now silently switches to the pure-torch path instead of erroring out. For a repo people use as a reference implementation, a broken kernel build probably should stay visible — maybe log a warning once (module-level flag) on first fallback.

3. Minor: `except (FileNotFoundError, ModuleNotFoundError, OSError, ImportError)` can be shortened to `except (OSError, ImportError)` — the first two are subclasses of the latter two.

4. Minor: in production, `node.value` comes from page-table rows (GPU), so the per-node `node.value.tolist()` inside `check_integrity` issues one device sync per node. Could cat first and call `tolist()` once. Since this only runs when idle, it is optional.

---

## 验证笔记(不发布)

1. **evictable/protected 计数追踪**:`insert_prefix` 中 `self.evictable_size += new_node.length`(ref_count=0 的新节点);`lock_handle` 中 ref_count 0→1 时 `evictable_size -= node.length` 且 `protected_size += node.length`,1→0 时反向;`split_at` 中 `new_node.ref_count = self.ref_count` 且总长度守恒。遍历结果与存储计数器一致。✓
2. **root 节点排除**:`root_node` 在 `__init__` 中仅创建,未调用 `set_key_value`,所以 `_length` 属性不存在。`_iter_non_root_nodes` 不将 root 加入 out 列表,因此不会访问 `node.length`。`root_node.ref_count = 1`(总是 protected),但遍历排除 root 后,walk 统计的 protected 不包含 root 的贡献——这需要确认:root 在 `lock_handle` 操作中是否被计入保护大小?`lock_handle` 中 `while not node.is_root()` 循环,root 本身不被修改 ref_count。`evictable_size` 和 `protected_size` 的维护中 root 不影响——因为 root 的 ref_count 永远是 1 且无 _length。所以 walk 统计与存储计数器一致。✓
3. **indices[::page_size] 正确性**:`insert_prefix` 中 `insert_len = align_down(len(input_ids), page_size)`,`split_at` 中 `match_len = align_down(match_len, page_size)`,所以每个 node.value 的长度都是 page_size 的倍数。串联后每 page_size 步进恰好取到每页的起始索引。`free_slots` 已有 `% page_size == 0` 断言。✓
4. **node.value 的 device**:`cache_req` 中 `page_indices = self.page_table[req.table_idx, : req.cached_len]` —— page_table 来自 `CacheManager.__init__` 的 `page_table.device`,在引擎中为 GPU tensor。`insert_prefix` 中 `indices[prefix_len:].clone()` 创建 GPU clone。所以 tree 中的 value 是 GPU tensor。`.tolist()` 触发 device→host sync。`cached_token_indices` 中 `torch.cat` 的 concatenation 也在 GPU 上。✓
5. **CacheManager 重叠检查**:`tree_starts = set(indices[::page_size].tolist())` — page-start indices;`free_starts = set(free_slots.tolist())` — 也是 page-start indices。重叠 = 同一页同时在 tree 和 free 中 → 双重分配 corrupt。覆盖 #149 的"size counters drift"之外的检测。✓
6. **get_match_len 的 fallback 范围**:`_cpu_compare_key` 使用 `torch.nonzero` 实现纯 Python 等价逻辑。`except` 捕获了 AOT 加载失败的所有情况。建议的 one-time warning 不会影响热路径性能(import 在 sys.modules 中缓存后 try/except 开销可忽略)。✓