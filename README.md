# vllm-engram-nvme-4gb10

**Run DeepSeek-V4.1-Flash on four DGX Spark / GB10 boxes by serving its Engram tables from NVMe instead of RAM.**

> Status: **working on the 4x GB10 cluster (2026-10-07).** DeepSeek-V4.1-Flash
> loads in 74.9 GiB per node with both Engram tables served from NVMe, vLLM
> reports a 4.0M-token KV cache, and the model answers correctly (short
> prompts, a 12k-token needle-in-a-haystack, reasoning). Getting there also
> needed `0002-sm120-deepseek-v41-page-sizes.patch`: V4.1 was not runnable
> on sm_120 (GB10 / DGX Spark) with this vLLM nightly at all, for reasons
> unrelated to Engram — see [docs/test-results.md](docs/test-results.md).
> Not yet done: DSpark, tool calls through the parsers, throughput numbers,
> long-running stability.

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
0002-sm120-deepseek-v41-page-sizes.patch  # DeepSeek-V4.1 on sm_120 (GB10 / DGX Spark), see below
Dockerfile                       # vllm/vllm-openai nightly + the patches
scripts/
  build-image.sh                 # build the patched image on a node
  engram_disk_bench.py           # the NVMe gather benchmark above (no vLLM, no GPU)
  verify_disk_mapping.py         # bit-exact check of the storage layer vs. pread()
  node-v41.sh                    # per-node launcher used on the 4x GB10 ring cluster
  dev/                           # the scripts that generated the patch hunks
docs/
  test-plan.md                   # the end-to-end validation plan
  test-results.md                # what happened on the 4x GB10 cluster (2026-10-07)
```

### The patches

`0001-engram-disk-offload.patch` — the actual feature, four files in vLLM:

* `vllm/config/engram.py` — `EngramConfig` gains `disk_offload: bool = False`
  and `disk_offload_threads: int = 64`, validated against `dp_shared_memory`
  and `use_thp`.
* `vllm/models/deepseek_v41/common/engram.py` — a `DiskEngramStorage` class
  (safetensors header parsing, per-rank mmap, pooled gather into pinned
  buffers, no-op weight loader) and a `_lookup_from_disk` path in
  `ParallelEngramEmbedding` that feeds the gathered rows to the unchanged
  `_engram_lookup_kernel` in its sorted mode with identity destinations. A
  plain (non-breakable) CUDA graph capture — vLLM's memory-sizing capture —
  skips the host gather; the breakable capture used for replay runs it
  eagerly every step.
* `vllm/model_executor/model_loader/weight_utils.py` — the safetensors
  iterators hand the weight loader an empty placeholder for the registered
  Engram tensors instead of reading them, and with
  `VLLM_SAFETENSORS_CLONE_ON_LOAD=1` clone every other tensor off the file
  mapping before a loader touches it (a device copy straight out of the
  `MAP_PRIVATE` mapping pins its pages into anonymous memory that is only
  freed when the mapping closes).
* `vllm/models/deepseek_v41/nvidia/vl_model.py` — the V4.1 wrapper sorted a
  *materialized* list of every mapped weight before loading, which keeps all
  49 shard mappings (and the pinned pages above) alive for the whole load and
  does not fit on a 121.6 GiB node even with the Engram tables gone. It now
  streams the `language_model.model.` group, with the LM head last in the
  same group, so each shard is released as soon as it is consumed.

`0002-sm120-deepseek-v41-page-sizes.patch` — makes DeepSeek-V4.1 run on
sm_120 at all (five files, nothing to do with Engram; needed on GB10 / DGX
Spark, harmless elsewhere):

* 64-token sliding-window pages for the SM120 FlashInfer attention class
  (its sparse-MLA decode kernels only exist for 64-token pages; V4.1 used 32).
* Per-layer KV block sizes on sm_120: 128 tokens for the compress-ratio-2
  layers, 64 for the ratio-1 layers, so every compressed and indexer page
  holds 64 rows — the only page size FlashInfer's sm_120 sparse prefill
  (`extra_page_block_size`) and DeepGEMM's sm_120 FP8 paged MQA logits
  accept. vLLM's hybrid KV cache manager packs the two groups with different
  block sizes, as it already does for the sliding-window cache.
* The null block (block 0) is zeroed after warmup and CUDA graph capture.
  Warmup runs store their dummy KV into slot 0, and near-zero dummy
  activations quantize to NaN bytes there; the sm_120 sparse-MLA kernels read
  block 0 for every masked (-1) index and multiply by a zero probability,
  which turned every partially filled attention row into NaN. Finding this
  took most of the day; the kernel-level fix belongs in FlashInfer.
* `indexer_kv_dtype=mxfp4` is allowed on sm_120 (DeepGEMM 2.8 ships the
  kernels); not needed with the per-layer block sizes, FP8 is the default.

Scope: TP sharding only (`engram_dp_size == 1`). The multi-DP table
sharing/sharding paths raise a clear error with `disk_offload` on.

## Usage

```bash
# build the image on each node (base image must be present locally)
scripts/build-image.sh            # -> vllm-engram-nvme:latest

VLLM_SAFETENSORS_CLONE_ON_LOAD=1 vllm serve /models/DeepSeek-V4.1-Flash-NVFP4 \
  --tensor-parallel-size 4 --nnodes 4 --node-rank 0 ... \
  --language-model-only \
  --engram-config '{"disk_offload": true, "disk_offload_threads": 64}' \
  ...
```

`scripts/node-v41.sh` is the complete per-node command used on the cluster.

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

## Memory budget on a GB10 node (121.6 GiB), measured

| | V4-Flash (production) | V4.1-Flash, Engram on NVMe |
|---|---:|---:|
| Weights | 42.7 GiB | **74.9 GiB** (vLLM "Model loading took") |
| KV cache (`--gpu-memory-utilization 0.8`) | 47 GiB | 3.98M tokens, 30 x 131k requests |
| Page cache for Engram | – | whatever is left (~20 GiB) |

## Roadmap

1. Measure: decode and prefill against V4-Flash on the same cluster and the
   in-memory numbers NVIDIA publishes for GB300; page-cache hit rates on
   real coding traffic. First numbers: ~24 tok/s single-stream decode,
   a 12k-token prompt prefills in ~10 s.
2. DSpark speculative decoding (`--speculative-config` with `method: dspark`),
   tool-call and `/v1/messages` parsing, multi-request stability.
3. Prefetch: the hash ids are known before the decoder runs, so the gather for
   step *t+1* can start while step *t* computes (and for DSpark drafts).
4. Engram DP sharding support (`engram_dp_size > 1`).
5. Upstream: the `vl_model.py` streaming load and the null-block fix are bugs
   on their own; the per-layer block sizes and `cpu_offload="disk"` are
   proposals.

## Credits

* DeepSeek for V4.1-Flash and the Engram design; NVIDIA for the NVFP4
  checkpoint; the vLLM contributors for the DeepSeek-V4.1 port this patches.
* The observation that everything but the two Engram tables must stay
  resident, and that Engram is "a gather, not a GEMM", is from the README of
  `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4`.
* Built on a four-node Dell Pro Max GB10 ring cluster; the NCCL side of that
  story is at `ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10`.

## License

Apache-2.0, same as vLLM. The patches are derivatives of the vLLM files they
modify.
