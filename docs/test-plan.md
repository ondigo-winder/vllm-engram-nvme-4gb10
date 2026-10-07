# End-to-end test plan (not yet executed)

Prerequisites: the four nodes hold the full checkpoint (today: shards split over
nodes 01–04 after the download; every node needs all 84 files, or at least the
index, config/tokenizer files, its 46 backbone shards and both Engram shards on
local NVMe). The V4-Flash service (`gb10-llm`) must be stopped first: both
models cannot share the memory.

1. `scripts/build-image.sh` on every node (uses the local nightly base image).
2. `scripts/verify_disk_mapping.py` on one node (CPU only) — already passes.
3. Start rank 3, 2, 1, then 0 with `scripts/node-v41.sh start <rank>`; follow
   `docker logs -f gb10-v41` on node 01. Expect per node:
   `Engram table mapped from local storage (model-000{47,48}-of-00048.safetensors): N rows x 256, ~23.6 GiB, 64 gather threads`
   twice (layers 1 and 14), then the usual weight loading (~11 min), then
   `Available KV cache memory` / `GPU KV cache size`.
4. Failure points to watch:
   - `disk_offload requires cpu_offload=True`: pass `"cpu_offload": true` too.
   - A hang or error inside CUDA graph capture: run once with
     `--enforce-eager` to separate graph issues from lookup issues.
   - OOM during the profiler run: lower `--gpu-memory-utilization` to 0.75.
   - Garbage output but no error: compare one request with `--engram-config
     '{"disk_offload": false}'` on a single node with a short context (needs
     47 GiB per node, only possible with `--max-model-len 4096` and 0.9 util).
5. `curl :8001/v1/models`, then a chat completion, a `/v1/messages` call,
   and a tool call. Then `scripts/engram_disk_bench.py`-style numbers from
   `/metrics` (`time_to_first_token`, `inter_token_latency`).
6. Add DSpark: `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`
   and re-measure.
