# Agent Note: 目录摊平——kv-tiering 上提至根，单一 README

Status: implemented

## Problem

仓内曾嵌套一层 `kv-tiering/` 目录（docs/bench/patches 在其下），根 README
与子目录 README 两本并行。单线研究仓加一层包裹目录只带来导航噪音：
进仓先多下一层目录，两本 README 还要互相同步。

## Decision

`kv-tiering/` 摊平至仓根：`docs/`、`bench/`、`patches/` 上提一级，
贡献草稿归入 `contrib/minisgl/` 并补状态 README；根 README 与原
kv-tiering README 合并为一本（commit `f1a3ff9`，2026-08-30）。
`third_party/` 保持只读研读的上游浅克隆，不入库。

## Alternatives considered

- **保留嵌套、等仓变大再摊平** — 避免一次移动；但嵌套层从不产生价值，
  越早摊平外部引用（patches 相对路径、文档链接）越少受影响。
- **拆成多仓（研究 + 基准分离）** — 边界更清；但孵化器阶段最忌分散，
  单仓单 README 让「研究到哪一步」一眼可见。

## Consequences

- **收益**：根即研究主线，`docs/` 调查、`bench/` 实验、`patches/` 上游
  绑定、`contrib/` 草案各就其位；README 单源。
- **代价**：摊平前的历史路径引用在 git 历史里仍需追溯一层旧目录名。

## Verification

根目录直接挂 `docs/ bench/ patches/ contrib/ third_party/`；`f1a3ff9`
提交 diff 记录移动范围。
