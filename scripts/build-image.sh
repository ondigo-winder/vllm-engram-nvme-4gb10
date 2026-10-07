#!/usr/bin/env bash
# Build the patched vLLM image on this node. Run from anywhere; needs the base image locally.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:-vllm-engram-nvme:latest}"
docker build -t "$TAG" .
docker run --rm --entrypoint python3 "$TAG" -c \
  "from vllm.config.engram import EngramConfig; c = EngramConfig(disk_offload=True); print('ok:', c.disk_offload, c.disk_offload_threads)"
echo "built $TAG"
