# Tuning log — DeepSeek-V4.1-Flash on 4x GB10, Engram from NVMe

Everything here was measured on the production cluster with
`scripts/bench_serving.py` (prefill time by prompt length with `max_tokens=1`
on a fresh random prompt, decode throughput with `ignore_eos`, page cache and
NVMe counters from the head node). Numbers are single runs, not averages; the
decode numbers move with DSpark acceptance, which depends on the text.

## Baseline: the "it works" configuration (2026-10-08 00:29)

Launcher as first put in service: `--max-num-batched-tokens 2048
--max-num-seqs 16 --gpu-memory-utilization 0.80`, 262144 context, DSpark 7.

| Prompt | Prefill | Rate |
|---:|---:|---:|
| 2,037 tokens | 1.15 s | 1,770 tok/s |
| 8,211 tokens | 2.68 s | 3,060 tok/s |
| 32,775 tokens | 9.36 s | 3,500 tok/s |

| Decode | Aggregate | Per stream |
|---|---:|---:|
| 1 stream × 256 | 52 tok/s | 52 tok/s |
| 4 streams × 256 | 121 tok/s | 30 tok/s |

Memory on the head node: 12.6 GiB available, 11.5 GiB page cache (that is
all the room the two 94 GiB Engram tables get). NVMe idle after the run:
~1,200 IOPS, 5 MiB/s.

Earlier, less favourable samples with the same configuration: 23.7 tok/s
single-stream on a first run (cold page cache), 27.6 tok/s on essays and ~48
tok/s on Python with DSpark; 74 tok/s aggregate at 4 streams. V4-Flash on the
same cluster: ~31 tok/s accepted throughput single-stream with DSpark.

## What is not optimized yet, and why

1. **Prefill chunk size.** `--max-num-batched-tokens 2048` was chosen during
   the OOM hunt and never revisited; vLLM's default is larger. Each chunk is a
   full pass over 40 layers plus a host-side Engram gather of 2048 × 6 rows
   per table, so bigger chunks amortize launch and gather overhead.
2. **Engram gather cost is not measured.** The gather runs eagerly on the
   CPU each step (64 threads, `np.take` over the page cache), with a device
   sync, and nothing prefetches: the hash ids are known before the decoder
   runs, so the gather for step *t+1* could overlap step *t*.
   `VLLM_ENGRAM_TIMING=1` (added to 0001) logs the per-lookup cost.
3. **Memory split.** `gpu-memory-utilization 0.80` gives vLLM 97 GiB: 75 GiB
   weights, ~3 GiB CUDA graphs, the rest KV cache (3.15M tokens, 12 × 262k
   concurrent). The page cache for Engram rows gets what is left of the
   121.6 GiB. A lower utilization trades KV cache you will not use for cache
   hits on hot n-gram rows; worth it only if (2) shows cold gathers dominate.
4. **DSpark.** `num_speculative_tokens` 7 and greedy draft sampling are the
   V4-Flash settings; acceptance length is 3–3.5 here versus 4.2 for
   V4-Flash. Not tuned.
5. **Not touched:** torch.compile is off for this model (`CompilationMode.NONE`,
   same as V4-Flash), FlashInfer autotune is off, `VLLM_SAFETENSORS_CLONE_ON_LOAD`
   costs load time only, and the masked-index guard adds two small
   element-wise ops per attention layer per step (unmeasured).

## Round 1: prefill chunks 8192, max-num-seqs 32, timing on (00:41–00:44)

Same image plus `VLLM_ENGRAM_TIMING=1`, `--max-num-batched-tokens 8192
--max-num-seqs 32`. The first bench after a restart is slow everywhere (JIT
and cold page cache); the second is the one to compare:

| Prompt | Prefill | Rate | vs baseline |
|---:|---:|---:|---:|
| 2,070 tokens | 0.65 s | 3,170 tok/s | 1.8× |
| 8,226 tokens | 2.19 s | 3,760 tok/s | 1.2× |
| 32,892 tokens | 8.29 s | 3,970 tok/s | 1.1× |
| 65,503 tokens | 16.63 s | 3,940 tok/s | – |

Decode 28 tok/s single, 136 tok/s aggregate at 8 streams (17 per stream).
NVMe during the run: one burst of 37 MiB/s at the first prefill, then
<1 MiB/s — with these synthetic prompts nearly every Engram row is a page
cache hit.

Cost: the profiler reserves activation memory per chunk, so the KV cache
dropped from 3.15M to 1.59M tokens and the page cache from 11.5 to 5.9 GiB.
A 2048-token chunk is the better trade on a 121.6 GiB node; 4096 is the
compromise still to test.

## Finding: Engram was switched off during decode

The timing log never fired during 1,500 generated tokens. Reason: vLLM runs
decode steps as FULL CUDA graphs, and `eager_break_during_capture` does not
apply in FULL mode (`breakable_cudagraph.py`: `if mode == CUDAGraphMode.FULL:
return fn(...)`). The disk lookup was therefore *captured*: the graph held
the lookup kernel over whatever was in the staging buffers at capture time
(the masked, all-zero rows), and every replay reproduced that. Decode ran
without Engram; prefill (eager) had it. The model still produced sensible
text, which is why nobody noticed — Engram is a gated residual addition.

Fix (now in 0001): the host-side gather moved out of the forward into
`DeepseekV41ModelState.prepare_inputs`, which the runner calls eagerly
before every step. It hashes the padded batch with the model's own
`NgramHashState` kernel (same inputs the forward uses: `input_ids`,
`positions`, `query_start_loc`, the lookback window), gathers this rank's
rows for every Engram layer into pinned memory, and copies them into
*static* device buffers. The lookup kernel inside the forward only reads
those buffers, so it is safe to capture: each replay sees the rows staged
for that step. One device-to-host copy per step for all tables; the tables'
gathers run concurrently.

Effect on output: "The capital of France is" now completes to " Paris." and
stops, where the Engram-less decode looped "Paris. The capital of France is
Paris. ...".

First per-step numbers (decode, 8 query tokens with DSpark → ~56 rows per
table): 5.4 ms per table of host time, i.e. ~11 ms of a ~36 ms step. 56
rows should take well under a millisecond, so this is overhead, not NVMe —
see round 2.

## Round 2: the gather itself (offline, one node, cold rows, `scripts/dev/gather_bench.py`)

With the staging on the critical path of every step, how the rows are read
matters. Measured against the real 94 GiB table, random rows (so cold):

| Rows | thread pool, 7 threads | 56 threads | `MADV_WILLNEED` all rows, then one `np.take` |
|---:|---:|---:|---:|
| 56 (one decode step) | 2.1 ms | 3.2 ms | **0.7 ms** |
| 224 (4 streams) | 8.7 ms | 6.1 ms | **1.3 ms** |
| 2,048 | 82 ms | 20 ms (32 thr.) | **10 ms** |

Faulting rows one by one from Python threads leaves the NVMe idle between
reads and pays ~40 µs of thread-pool overhead per task. Telling the kernel
about every page first (`madvise(MADV_WILLNEED)` per 4 KiB page, for the
weight and the scale mapping) issues all reads at once; the copy that follows
mostly hits pages already in flight. 2,048 rows in 10 ms is ~200k rows/s —
the device's random-read limit, as measured in the first benchmark. The
storage class now does exactly this (bit-exact against `pread()`, 56 rows
0.8 ms, 12,288 rows — one 2048-token prefill chunk — 50 ms cold).

## Round 3: staging out of the graph + WILLNEED gather, 2048-token chunks (01:30)

Back to `--max-num-batched-tokens 2048 --max-num-seqs 16` (KV cache 3.14M
tokens, page cache ~9–11 GiB), with the two Engram changes above.

| Prompt | Prefill | Rate |
|---:|---:|---:|
| 2,035 tokens | 0.94 s | 2,170 tok/s |
| 8,176 tokens | 2.48 s | 3,300 tok/s |
| 32,715 tokens | 9.09 s | 3,600 tok/s |

| Decode | Aggregate | Per stream |
|---|---:|---:|
| 1 stream × 256 | 31 tok/s | 31 tok/s |
| 4 streams × 256 | 84 tok/s | 21 tok/s |
| 8 streams × 256 | 125 tok/s | 16 tok/s |

Engram host time per decode step: 1.5–1.7 ms per table for ~170 rows (the
two tables are staged concurrently), down from 4–5 ms for 56 rows — and this
time it runs on every step. The 12k-token needle test and a Python coding
request pass; DSpark mean acceptance length on the coding request is 4.2
(was 3.5 while decode ran without Engram).

Honest comparison with the baseline table above: the baseline's 52 tok/s
single-stream was measured with Engram silently absent from decode and with
warm, repetitive prompts; the 31 tok/s here is with the full model. The
remaining per-step overhead is the device-to-host copy of the hash ids
(one sync per step) plus ~2 ms of staging; a true prefetch would need the
next step's token ids before the current step finishes, which only exists
for prefill chunks and DSpark verification batches.

## Still open

* `--max-num-batched-tokens 4096` as the middle ground (2048 vs 8192 cost
  1.6M KV-cache tokens and 5 GiB of page cache for 1.1–1.8× prefill).
* `--gpu-memory-utilization 0.75` to give the page cache ~6 GiB more; only
  worth it if real traffic shows cold gathers (watch the `gather` ms in the
  `VLLM_ENGRAM_TIMING` log and NVMe IOPS during decode).
* Overlap the DSpark verification batch's staging with the drafter.
* DSpark `num_speculative_tokens`; torch.compile; FlashInfer autotune.
