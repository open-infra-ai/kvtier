# PR 草稿:#3244 Slurm 示例(改 `docs/docs/references/multi_node_deployment/multi_node.mdx`)

## 撞车与定位分析

- issue 下已有 4 人表态;PR #31544(加 examples/slurm/ 脚本目录)被关未合并(无明确原因,
  推测与"文档应放 docs/ 而非 examples/"或质量有关);PR #29991(open、已冲突、停滞 6 周)
  在做 PD 分离 + Apptainer 示例
- **我们的差异化**:不碰 #29991 的范围,专做 ① 修复现有多节点示例的真实缺陷
  ② 补单节点服务 ③ 补离线批量推理 ④ 共享集群注意事项。PR 描述中说明与 #29991 互补
- 只改一个现有文件,不动 docs.json 导航,冲突面最小

## 现有示例的缺陷(HPC 视角)

1. `echo "... on node $SLURM_PROCID"`:批处理脚本里 `SLURM_PROCID` 未定义(仅 srun 任务内有值),打印为空且误导
2. `while ! nc -z` 无超时:服务起不来就空转到作业时限,白烧机时
3. 端口 8000/30000 硬编码:共享集群必撞,无任何提示
4. 缺 HPC 最常见场景:单节点服务、离线批量推理

## 改动一:修复现有多节点示例(最小改动)

原句:
```bash
echo "[INFO] Activating environment on node $SLURM_PROCID"
```
改为:
```bash
echo "[INFO] Activating environment on $(hostname)"
```

原等待循环:
```bash
# Wait for the NCCL server to be ready on port 30000
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done
```
改为:
```bash
# Wait for the server to be ready on port 30000 (give up after 10 minutes)
waited=0
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 600 ]; then
        echo "[ERROR] $HEAD_NODE:30000 did not become ready in time" >&2
        exit 1
    fi
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done
```

并在 `NCCL_INIT_ADDR` 处加一行注释:
```bash
# Set NCCL initialization address using the hostname of the head node.
# If port 8000 is already used by another job, pick a different one.
```

## 改动二:新增章节(追加在现有多节点章节之后)

```markdown
## Single-Node Serving on SLURM

For a single node, submit the following job. It picks a free port automatically,
which avoids collisions with other jobs on shared clusters.

```bash Command
#!/bin/bash

#SBATCH -o SLURM_Logs/%x_%j.out
#SBATCH -e SLURM_Logs/%x_%j.err
#SBATCH -J sglang-single-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source ENV_FOLDER/bin/activate

model=MODEL_PATH

# Pick a free port to avoid collisions with other jobs
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python3 -m sglang.launch_server \
    --model-path "$model" \
    --host 0.0.0.0 \
    --port "$port" &
server_pid=$!

# Wait until the server is healthy (give up after 10 minutes)
ready=0
for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
done
if [ "$ready" -ne 1 ]; then
    echo "[ERROR] Server did not become healthy in time" >&2
    kill "$server_pid"
    exit 1
fi

echo "[INFO] Server ready on $(hostname):${port}"
echo "[INFO] From a login node, forward the port with:"
echo "[INFO]   ssh -L ${port}:$(hostname):${port} <login-node>"

# Keep the job alive while the server runs
wait "$server_pid"
```

## Offline Batch Inference on SLURM

Batch jobs that process a fixed dataset do not need a long-running server; the
job exits when inference finishes. The offline engine is shown in
[offline_batch_inference.py](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference.py).

```bash Command
#!/bin/bash

#SBATCH -o SLURM_Logs/%x_%j.out
#SBATCH -e SLURM_Logs/%x_%j.err
#SBATCH -J sglang-batch-inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

source ENV_FOLDER/bin/activate

python3 examples/runtime/engine/offline_batch_inference.py \
    --model MODEL_PATH
```

## Notes for Shared Clusters

- **Ports**: the server listens on port 30000 by default. On shared clusters,
  pass `--port` with a port unique to your job (or pick a free one as above),
  and choose a free port for `--dist-init-addr` in multi-node runs.
- **Network interfaces**: on nodes with several interfaces (InfiniBand plus
  Ethernet), NCCL and Gloo may pick the wrong one. Pin them with
  `export NCCL_SOCKET_IFNAME=<iface>` and `export GLOO_SOCKET_IFNAME=<iface>`
  before launching.
- **Model weights**: loading large weights from a shared filesystem can be slow
  when many jobs start at once. Stage the weights to node-local storage, or set
  `--download-dir` to a fast cache.
```

## 请你(用户)审核的要点

1. 单节点示例的"选空闲端口 + health 轮询 + ssh 隧道提示"流程是否符合你在生信集群的经验?
2. `--gres=gpu:1` 的写法(部分集群用 `--gpus-per-task` 或 `--gres=gpu:a100:1`)——
   示例保持最通用写法还是加注释说明变体?
3. `#!/bin/bash -l`(原多节点脚本用的登录 shell)要不要保留?新脚本我没加 `-l`,
   因为 venv 激活已显式 source
4. Notes 里的 NCCL/GLOO 接口、download-dir 建议是否准确?
5. 离线示例引用的是仓库内真实文件(已核实存在,参数为 `--model`)

## PR 标题与描述(平实风格)

标题:`[docs] Slurm: fix multi-node example, add single-node and offline batch examples`

描述:
```
## Motivation

Part of #3244. The existing SLURM section only covers multi-node serving and
has two issues: `$SLURM_PROCID` is printed in the batch script where it is
unset, and the port-wait loop has no timeout, so a failed server keeps the
job spinning until its time limit.

Complements #29991 (PD disaggregation + Apptainer); this PR does not touch
those topics.

## Modifications

In `docs/docs/references/multi_node_deployment/multi_node.mdx`:

- Fix the two issues above in the multi-node example.
- Add a single-node serving example (free-port selection, health check with
  timeout, port-forward hint).
- Add an offline batch inference example using the existing
  `examples/runtime/engine/offline_batch_inference.py`.
- Add short notes on ports, NCCL/GLOO interface selection, and weight staging
  for shared clusters.

## Checklist

- [x] Docs only; no code changes
- [x] Referenced example file and `--download-dir` flag verified to exist
```
