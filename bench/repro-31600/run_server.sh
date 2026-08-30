#!/usr/bin/env bash
# 启动用于 HiCache L2 单卡微基准的 SGLang 服务。
# 用法：MODEL_PATH=/path/to/model bash run_server.sh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?需要设置 MODEL_PATH，例如 /root/models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.78}"
HICACHE_RATIO="${HICACHE_RATIO:-4.0}"
HICACHE_IO="${HICACHE_IO:-kernel}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enable-hierarchical-cache \
  --hicache-host-memory-mode cache \
  --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_through \
  --hicache-io-backend "$HICACHE_IO" \
  --enable-metrics \
  --enable-cache-report \
  --log-level info
