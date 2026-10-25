# Agent Note: 提前公开决策（所有者裁定）

Status: implemented

## Problem

2026-09-04 的公开门槛约定「四条全达成才转公开」。截至 2026-10-25：门槛
1–2 未达成（无真实 GPU 运行与结果包审计）；门槛 3 部分达成（PR #36116
OPEN、#36115 未合入关闭、#35808 评论已发布，无已合入 PR）；门槛 4 本批
清理完成。私有状态使分支保护等平台能力在 free plan 下不可用，孵化进展
对外不可见，而已提交的上游 PR 与评论本就是公开事实。

## Decision

所有者裁定 2026-10-25 提前转公开并登记入组织状态注册表（`active`）。
证据纪律通过注册表与 README 的诚实口径保持：明确标注「尚无完成审计的
GPU 结果包、亦无已合入上游 PR」，完成审计结果包之前不作为简历主项目。
个人绝对路径随公开批次清理；README「公开门槛」改写为逐条标注现状的
「公开门槛与现状」；分支保护以 `verify` 为必需检查。

## Alternatives considered

- **维持私有至四条全达成** — 证据驱动最纯粹；但私有仓在 free plan 下无
  分支保护，且公开的上游参与记录与私有仓库状态互相矛盾。
- **公开但不入注册表** — 避免作品集叙事负担；但公开仓游离在注册表外
  会让治理再次出现盲区（radar 曾长期零登记的先例）。

## Consequences

- **收益**：分支保护可用（required: verify）；上游贡献记录与孵化过程
  对外可见；组织注册表覆盖全部公开仓。
- **代价**：公开面出现一个尚无审计结果的 active 仓，需要以诚实标注防止
  被当成完成作品引用；上游 PR 状态会随时间漂移，引用前须按 README
  约定重新核对。

## Verification

README「公开门槛与现状」逐条标注；组织侧决策记录见 open-infra-ai 仓
`.agents/notes/implemented/process/2026-10-25-kvtier-radar-publication.md`；
`npm run verify-notes` 通过。

[历史快照：公开门槛——达成前不进入 open-infra-ai 状态注册表](../../archived/process/2026-09-04-publication-gate.md)
