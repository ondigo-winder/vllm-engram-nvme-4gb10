#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on the 4x GB10 ring cluster, Engram tables served from NVMe.
#   node-v41.sh start <rank>   (0 = head with the API on :8000, 1-3 = workers)
#   node-v41.sh stop | status
# Container name gb10-v41; serves the production port 8000 (the V4-Flash gb10-llm service, node.sh,
# is the fallback and must be stopped first: both do not fit in memory).
set -euo pipefail
MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash-NVFP4"
SERVED_NAME="deepseek-v4.1-flash"
IMAGE="${IMAGE:-vllm-engram-nvme:latest}"
NCCL_DIR="$HOME/nccl-switchless-4hca"
HEAD_IP="192.168.1.101"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.80}"
CONTAINER="gb10-v41"
mgmt_ip() { ip -4 -br addr show enP7s7 | awk '{print $3}' | cut -d/ -f1; }
case "${1:-}" in
start)
  R="${2:?usage: node-v41.sh start <rank 0-3>}"
  [ -f "$MODEL_DIR/model.safetensors.index.json" ] || { echo "geen index in $MODEL_DIR"; exit 1; }
  [ -f "$NCCL_DIR/libnccl.so.2" ] || { echo "patched NCCL ontbreekt: $NCCL_DIR"; exit 1; }
  [ -f "$HOME/vllm-cache/no-merge.jinja" ] || printf '%s\n' "{% for m in messages %}{{ m['content'] }}{% endfor %}" > "$HOME/vllm-cache/no-merge.jinja"
  if docker ps --format '{{.Names}}' | grep -qx gb10-llm; then echo "gb10-llm draait nog: eerst node.sh stop"; exit 1; fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if [ "$R" = 0 ]; then TAIL=(--host 0.0.0.0 --port "$PORT"); else TAIL=(--headless); fi
  docker run -d --name "$CONTAINER" --network host --ipc host --shm-size 32g --gpus all \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
    -v "$NCCL_DIR":/opt/patched-nccl:ro -v "$MODEL_DIR":/model:ro -v "$HOME/vllm-cache":/root/.cache \
    -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2 \
    -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_SKIP_TREE_CONNECT=1 \
    -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e VLLM_HOST_IP="$(mgmt_ip)" \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1 \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
    -e NCCL_ALGO=Ring -e NCCL_PROTO=LL,LL128,Simple -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
    -e VLLM_ONE_GPU_PER_NODE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_SAFETENSORS_CLONE_ON_LOAD=1 -e VLLM_ENGRAM_TIMING="${ENGRAM_TIMING:-1}" \
    "$IMAGE" /model --served-model-name "$SERVED_NAME" deepseek-v4-flash \
    --tensor-parallel-size 4 --nnodes 4 --node-rank "$R" \
    --master-addr "$HEAD_IP" --master-port 29521 --distributed-executor-backend mp \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" --max-num-batched-tokens "${BATCH_TOKENS:-8192}" --max-num-seqs "${MAX_SEQS:-32}" \
    --trust-remote-code --language-model-only \
    --kernel-config '{"enable_flashinfer_autotune":false}' \
    --chat-template /root/.cache/no-merge.jinja \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --engram-config '{"disk_offload": true, "disk_offload_threads": 64}' \
    --reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
    "${TAIL[@]}" >/dev/null
  echo "$(hostname): V4.1 rank $R gestart (poort $PORT)"
  ;;
stop)   docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; echo "$(hostname): gestopt" ;;
status) echo "$(hostname): $(docker inspect -f '{{.State.Status}} sinds {{.State.StartedAt}}' "$CONTAINER" 2>/dev/null || echo 'niet actief')" ;;
*) echo "gebruik: node-v41.sh start <rank> | stop | status"; exit 1 ;;
esac
