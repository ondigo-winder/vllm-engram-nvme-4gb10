# End-to-end test log — 4x GB10 (DGX Spark class), 2026-10-07

Cluster: four Dell Pro Max GB10 (121.6 GiB unified memory each, sm_120),
tensor parallel 4 over a QSFP ring, vLLM nightly 0.30.1rc1.dev558
(`0cbac6cd1`, FlashInfer 0.7.0.post1, vendored DeepGEMM 2.8.0), checkpoint
`nvidia/DeepSeek-V4.1-Flash-NVFP4` on local NVMe on every node.
Launcher: `scripts/node-v41.sh` (`--language-model-only`, 131072 context,
`--gpu-memory-utilization 0.80`, `--max-num-batched-tokens 2048`).

## What works

| Step | Result |
|---|---|
| Storage layer (`scripts/verify_disk_mapping.py`) | bit-exact vs `pread()` on all shards tested |
| Engram tables mapped at startup | `96,000,564 rows x 256, 23.60 GiB` per table per rank, 64 threads |
| Weight loading on a 121.6 GiB node | **74.92 GiB, 200 s** per node (nightly without the patch: OOM-killed at shard 4–20) |
| Gather + `_engram_lookup_kernel` on real rows (`ktest`, one GPU) | max abs diff 0.0 vs numpy dequant, masked rows zero |
| Memory profiler, piecewise and full CUDA graph capture | pass (the plain capture skips the host gather) |
| KV cache | `GPU KV cache size: 6,278,654 tokens`, 47.9 concurrent 131k requests |
| `/v1/models`, server startup | `Application startup complete` on all four nodes |
| Decode throughput (400 tokens, one request) | ~23 tok/s wall clock including the NVMe gathers |

So the thing this repo is about — keeping 189 GiB of Engram tables off the
node memory — works: the model loads, the tables are served from NVMe, and
there is room for a large KV cache.

## What does not work yet: DeepSeek-V4.1 attention on sm_120

Every request returns NaN logits (the API reports `Out of range float
values: nan`; a `/v1/chat/completions` call produces byte garbage). With
`VLLM_DEBUG_NAN=1` (a few `torch.isnan` prints in the decoder layer, not in
the repo) the NaN first appears in **layer 0's attention output**, a
sliding-window-only layer that has no Engram, no compressed cache and no
indexer. Setting the Engram rows to zero (`VLLM_ENGRAM_DEBUG_ZERO=1`,
debug-only) does not change the output, so the Engram path is not the cause.

The cause is that this vLLM nightly's DeepSeek-V4.1 path is only exercised
on sm_100 (B200/GB300) and sm_90, and on sm_120 three kernel page-size
constraints collide. All of them were hit one after the other and are what
`0002-sm120-deepseek-v41-page-sizes.patch` works around:

1. **SWA pages.** `deepseek_v41/attention.py` allocates the sliding-window
   cache with 32-token pages; FlashInfer's sm_120 sparse-MLA decode kernels
   are only instantiated for 64-token pages (`page_block_size=32 is
   unsupported`). The patch makes the SM120 attention class use 64 (the
   DeepSeek-V4 default). After that the layer-0 output is NaN on real input
   while the 2048-token profiling run is finite — not understood yet.
2. **Indexer pages.** DeepGEMM's sm_120 paged MQA logits need 64-row pages
   for FP8 (`arch_major == 12 and not is_fp4 and block_kv == 64`), 32 or 64
   for MXFP4. V4.1 has `compress_ratios` 2 (layers 2–19) and 1 (20–39), and
   vLLM puts all layers in one KV cache group with one block size, so FP8
   cannot satisfy both (64-token blocks → 32 rows for ratio 2; 128-token
   blocks → 128 rows for ratio 1). The patch lets the indexer run on 64-token
   blocks and allows `indexer_kv_dtype=mxfp4` on sm_120 (the kernels exist in
   DeepGEMM 2.8; vLLM only gates them to sm_100) — `node-v41.sh` passes
   `--attention-config '{"indexer_kv_dtype": "mxfp4"}'`.
3. **Sparse prefill extra pages.** With 64-token blocks the compressed cache
   of the ratio-2 layers has 32-row pages, and FlashInfer's sm_120 sparse
   prefill rejects them: `Unsupported sparse-MLA prefill configuration:
   model=DSV4 num_heads=16 topk=128 page_block_size=64 topk_extra=512
   extra_page_block_size=32`. The kernel takes extra pages of 2 or 64 rows
   (DeepSeek-V4 with 256-token blocks: ratios 4 and 128 → 64 and 2). With
   128-token blocks the ratio-1 layers would need 128-row pages instead. Same
   collision as (2), no block size satisfies both ratios.

Conclusion: serving DeepSeek-V4.1-Flash on sm_120 needs either per-layer-group
KV block sizes in vLLM (the ratio-1 and ratio-2 layers want different page
geometries) or sm_120 kernel instantiations for 32- and 128-row pages in
FlashInfer and DeepGEMM, plus the layer-0 NaN fixed. That is attention-stack
work independent of this repo; the Engram-from-NVMe part is ready for it.

## Not run

Chat/`/v1/messages`/tool-call checks, DSpark, latency measurements: blocked by
the above. The production DeepSeek-V4-Flash service was restored afterwards.
