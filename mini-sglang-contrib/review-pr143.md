# PR #143 评审评论草稿

> 发布前请逐条验证。下方是拟发布的评论正文,「验证笔记」不发布。

---

## 拟发布评论

Tried this branch locally. The branch does not parse — `append_host` in `python/minisgl/core.py` has mixed indentation:

```
$ python3 -m py_compile python/minisgl/core.py
  File "python/minisgl/core.py", line 65
    new_len = len(self.input_ids) + 1
IndentationError: unindent does not match any outer indentation level
```

The `if self._host_buf is None:` body is indented 13 spaces, and the following statements sit at 9 spaces, which matches neither the method body (8) nor the if body (13):

```python
        if self._host_buf is None:
             self._host_buf = torch.empty(...)      # 13 spaces
             self._host_buf[: len(self.input_ids)] = self.input_ids
         new_len = len(self.input_ids) + 1          # 9 spaces
```

Running pre-commit should catch this (the repo has black with `line-length=100` and ruff configured in `.pre-commit-config.yaml`).

Once the indentation is fixed, the approach looks sound to me:

- Writes into `_host_buf` are strictly append-only, so existing radix keys (slices of `input_ids` stored by `insert_prefix`) remain stable across later appends. `test_append_host_never_mutates_written_positions` covers exactly this hazard, which is the right test to have.
- Capacity: `max_device_len` bounds the final length (`device_len` cannot exceed it), and the overflow test confirms scalar assignment raises `IndexError` rather than silently growing.
- There is trailing whitespace at the end of the first comment line ("...instead of ").

A related but separate note: `CacheManager._free` / `_allocate` also grow `free_slots` via `torch.cat` (scheduler/cache.py). That is per-batch rather than per-token, so much lower frequency — probably worth its own PR if anyone wants it.

---

## 验证笔记(不发布)

1. **缩进错误已用 `py_compile` 确认**:从 PR head 分支下载了 `core.py`,`python3 -m py_compile` 确实报 IndentationError。错误发生位置:if body 为 13 空格,后续语句为 9 空格,与任何外层缩进级别都不匹配。
2. **pre-commit 配置确认**:`.pre-commit-config.yaml` 存在,包含 black 和 ruff hooks。black 会纠正此问题。
3. **append_host 调用方确认**:唯一的调用方是 `scheduler.py:151` (`req.append_host(next_token.unsqueeze(0))`),其中 `next_token` 来自 `next_tokens_cpu`(CPU tensor)。ChunkedReq 重写了 append_host 为 NotImplementedError。
4. **radix key 稳定性**:`insert_prefix` 存储 `input_ids[prefix_len:]` 的切片。在旧实现中 torch.cat 创建新 tensor,切片独立;在新实现中切片是 _host_buf 的 view,但后续写入仅发生在 `new_len-1` 位置(严格超出所有已有切片范围),所以旧切片数据不变。PR 的测试 `test_append_host_never_mutates_written_positions` 覆盖了这一点。
5. **cache.py:89 的 torch.cat**:`_free` 和 `_allocate` 中使用 `torch.cat([self.free_slots, ...])` 追加 free_slots。注释中提及此为"per-batch"频率,与 #143 的"per-token"频率不同。