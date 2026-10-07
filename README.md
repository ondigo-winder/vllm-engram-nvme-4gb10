# vllm-engram-nvme

**Run DeepSeek-V4.1-Flash on four DGX Spark / GB10 boxes by serving its Engram tables from NVMe instead of RAM.**

> Status: **draft, partially validated.** The storage layer (memory-mapping the
> checkpoint, per-rank row gather, thread pool) is tested bit-exact against the
> real `nvidia/DeepSeek-V4.1-Flash-NVFP4` checkpoint. The vLLM integration
> (lookup kernel on gathered rows, side-stream prefetch, CUDA-graph behaviour)
> has **not** been run end-to-end yet, because the cluster this was written
> for is busy serving DeepSeek-V4-Flash. See [Roadmap](#roadmap).

## The problem

DeepSeek-V4.1-Flash is a 552B-parameter MoE with a 196B-parameter **Engram**
conditional memory. The NVFP4 checkpoint is 491 GiB. With tensor parallelism
over four GB10 nodes that is:

| Component | Total | Per node (TP=4) |
|---|---:|---:|
| Backbone weights (experts, attention, MTP/DSpark, vision) | 302 GiB | 75.5 GiB |
| Engram tables, FP8 + E8M0 scales (`layers.{1,14}.engram.embed`) | 189 GiB | 47.3 GiB |
| **Total** | **491 GiB** | **122.8 GiB** |

A GB10 has 121.6 GiB of unified memory. vLLM already offloads the Engram
tables to host memory (`EngramConfig.cpu_offload`, default on), but on a
unified-memory machine "host memory" and "GPU memory" are the same pool, so the
offload buys nothing: the tables still need 47 GiB per node, before the KV
cache, activations and the OS.

## The idea

Engram is a hashed n-gram **gather**, not a matrix multiply. The rows a token
needs are determined by the token ids alone (`NgramHashState` computes them
from `input_ids` before the decoder runs), each position reads 24 rows of
264 bytes per Engram layer, and the two tables are the only big thing left.
So keep them on the NVMe:

* The weight loader does not copy anything. This rank's row range of
  `engram.embed.weight` / `.scale` is memory-mapped straight from the
  safetensors shard (`MAP_PRIVATE`, `MADV_RANDOM`).
* Per forward step, the rows the batch needs are gathered on the CPU through
  the mapping — Linux's page cache in front of the NVMe — into pinned staging
  buffers, copied to the device, and dequantized by the existing Triton lookup
  kernel, now run against the dense gathered rows instead of the full table.
* Memory per node drops from 122.8 GiB to ~76 GiB of weights plus activations,
  leaving room for a KV cache. The page cache uses whatever RAM is free and
  gives it back under pressure.

Nothing changes for machines with real host memory: the option is off by
default and the existing pinned / shared-memory paths are untouched.

## Measurements (one GB10, KIOXIA BG7 1 TB NVMe, `scripts/engram_disk_bench.py`)

Random row gathers from the 91.6 GiB `layers.1.engram.embed.weight` tensor,
2048-token prefill chunk, 6 rows/token (one TP=4 rank's share of 24 heads):

| Setting | Cold rows/s | ms per token | Decode step (32 tokens, cold) |
|---|---:|---:|---:|
| default read-around | 20 900 | 0.29 | 9.2 ms |
| `MADV_RANDOM`, 16 threads | 73 000 | 0.08 | 3.6 ms |
| `MADV_RANDOM`, 32 threads | 120 000 | 0.05 | 4–5 ms |
| `MADV_RANDOM`, 64 threads | 190 000 | 0.03 | 6 ms |
| warm (page cache hit) | 3 900 000 | 0.001 | <1 ms |

`MADV_RANDOM` matters: without it every 264-byte row drags a 128 KiB
read-around window from the disk and the NVMe saturates on bandwidth, not
IOPS. With it the device does ~190k random reads/s.

What that means per node (two Engram layers, this rank's heads):

* **Decode** (4 users × 8 DSpark draft tokens): ≈ 8–12 ms of gather per step
  on cold rows, ~1 ms warm, against a ~70 ms decode step today. Common n-grams
  (code tokens, English) stay in the page cache.
* **Prefill** of a 2048-token chunk: ≈ 2 × 65 ms cold, on top of the
  1–2 s the chunk takes on this cluster. A 200k-token prompt adds roughly
  10–15 s of NVMe reads.

## What is in the repo

```
0001-engram-disk-offload.patch   # against vLLM 0cbac6cd1 (nightly 0.30.1rc1.dev558)
Dockerfile                       # vllm/vllm-openai nightly + the patch
scripts/
  build-image.sh                 # build the patched image on a node
  engram_disk_bench.py           # the NVMe gather benchmark above (no vLLM, no GPU)
  verify_disk_mapping.py         # bit-exact check of the storage layer vs. pread()
  node-v41.sh                    # per-node launcher used on the 4x GB10 ring cluster
docs/
  test-plan.md                   # the end-to-end validation still to be done
```

### The patch

Two files in vLLM:

* `vllm/config/engram.py` — `EngramConfig` gains `disk_offload: bool = False`
  and `disk_offload_threads: int = 64`, validated against `dp_shared_memory`
  and `use_thp`.
* `vllm/models/deepseek_v41/common/engram.py` — a `DiskEngramStorage` class
  (safetensors header parsing, per-rank mmap, pooled gather into pinned
  buffers, no-op weight loader) and a `_lookup_from_disk` path in
  `ParallelEngramEmbedding` that feeds the gathered rows to the unchanged
  `_engram_lookup_kernel` in its sorted mode with identity destinations.

Scope: TP sharding only (`engram_dp_size == 1`). The multi-DP table
sharing/sharding paths raise a clear error with `disk_offload` on.

## Usage

```bash
# build the image on each node (base image must be present locally)
scripts/build-image.sh            # -> vllm-engram-nvme:latest

vllm serve /models/DeepSeek-V4.1-Flash-NVFP4 \
  --tensor-parallel-size 4 --nnodes 4 --node-rank 0 ... \
  --language-model-only \
  --engram-config '{"disk_offload": true, "disk_offload_threads": 64}' \
  ...
```

Requirements and caveats:

* The model must be a **local checkpoint directory** with
  `model.safetensors.index.json`; the Engram shards (for this checkpoint
  `model-00047/00048-of-00048.safetensors`) must sit on a fast local NVMe on
  every node. The other shards can live anywhere vLLM can read them.
* Do **not** use `--model-loader-extra-config '{"enable_multithread_load": true}'`
  with this option: that loader reads whole tensors into memory, which is the
  thing we are avoiding. The default safetensors loader hands the weight loader
  a lazily mapped tensor, and the loader only checks its shape.
* `cpu_offload` must stay `true` (the default); `dp_shared_memory` and
  `use_thp` must be off.
* First-token latency on a cold page cache is dominated by random reads; the
  numbers above are the realistic cost.

## Memory budget on a GB10 node (121.6 GiB)

| | Today (V4-Flash) | V4.1-Flash, Engram on NVMe |
|---|---:|---:|
| Weights | 42.7 GiB | ~76 GiB |
| Peak activation + CUDA graphs | ~7 GiB | ~7 GiB (to be measured) |
| KV cache (`--gpu-memory-utilization 0.8`) | 47 GiB | ~14 GiB |
| OS + page cache for Engram | ~24 GiB | ~24 GiB |

14 GiB of KV cache is enough for a handful of 131k–262k requests; the exact
number depends on V4.1's KV layout (FP4 main KV, CSA2), which vLLM reports at
startup.

## Roadmap

1. End-to-end smoke test on the 4-node cluster: load, `/v1/models`, a chat
   completion, reasoning + tool-call parsing. Watch for: the eager-break
   lookup path under CUDA graph capture (the gather is a host sync), the
   side-stream copy/kernel ordering, and the memory profiler's dummy-weight
   run with no real table in memory.
2. Measure decode and prefill against the in-memory numbers NVIDIA publishes
   for GB300, and against V4-Flash on the same cluster.
3. Prefetch: the hash ids are known before the decoder runs, so the gather for
   step *t+1* can start while step *t* computes (and for DSpark drafts).
4. Engram DP sharding support (`engram_dp_size > 1`).
5. Upstream discussion: whether this belongs in vLLM as `cpu_offload="disk"`.

## Credits

* DeepSeek for V4.1-Flash and the Engram design; NVIDIA for the NVFP4
  checkpoint; the vLLM contributors for the DeepSeek-V4.1 port this patches.
* The observation that everything but the two Engram tables must stay
  resident, and that Engram is "a gather, not a GEMM", is from the README of
  `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4`.
* Built on a four-node Dell Pro Max GB10 ring cluster; the NCCL side of that
  story is at `ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10`.

## License

Apache-2.0, same as vLLM. The patch is a derivative of vLLM's
`vllm/models/deepseek_v41/common/engram.py` and `vllm/config/engram.py`.
