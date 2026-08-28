# ai-infra 转型工作区

生信 HPC(C++/CPU 并行优化)→ AI-infra 推理优化方向的全职转型工作区。
目标岗位:推理加速/部署优化(vLLM/SGLang 引擎组件层),2026 年内投递国内大厂/头部创业公司。

## 目录结构

- `kv-tiering/` — **主项目**:推理引擎 CPU 侧 KV 分层存储优化(详见其 README)
- `third_party/` — 只读研读的上游源码浅克隆(sglang 等)
- 学习笔记与练习代码后续按需新增目录

## 战略主线

1. **上游贡献**:主攻 SGLang(切入点 HiCache 子系统),vLLM 作为第二熟悉对象
2. **差异化项目**:CPU 侧 KV 分层存储优化 —— HPC 内存层级优化功力的直接迁移
3. **知识体系**:推理原理 + 引擎源码 + 适量 CUDA(会读、会 profile、会写中等 kernel)

完整 24 周路线图见 `~/.claude/plans/bwa-gatk-ai-infra-c-kernel-vllm-sglang-glowing-bee.md`。

## 持续开发入口

主线按“可复现实验 → 结果审计 → 数据驱动优化”推进。单卡 HiCache 基准的客户端依赖、
测试和结果包审计说明位于 `kv-tiering/bench/repro-31600/`；本地离线检查使用：

完整的阶段计划、任务清单和验收条件见 [`PLAN.md`](PLAN.md)。

```bash
make test
```

真实 GPU 运行前，必须固定 SGLang commit、模型、GPU 和启动参数；真实结果通过结果审计
后，才进入下一项 CPU 侧 KV 分层优化或上游贡献评估。
