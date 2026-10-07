#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: random Engram row gathers straight from a safetensors shard on NVMe.

Simulates what the disk-offload patch does per forward step: for N tokens, gather
``rows_per_token`` random rows (fp8 row + ue8m0 scales) from the memory-mapped
checkpoint tensor, using a thread pool so page faults overlap. No vLLM, no GPU.

Usage:
  engram_disk_bench.py <shard.safetensors> <tensor name> [--tokens N] [--rows-per-token R]
                       [--threads T] [--repeat K]
Example (DeepSeek-V4.1-Flash, layer 1 table, 2048-token chunk, TP=4 rank share):
  engram_disk_bench.py model-00047-of-00048.safetensors layers.1.engram.embed.weight \
      --tokens 2048 --rows-per-token 6
"""
import argparse, json, mmap, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def open_tensor(path: str, name: str, random_access: bool = True):
    """Memory-map one tensor of a safetensors file: returns (np.ndarray view, mmap, info)."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    info = header[name]
    start, end = info["data_offsets"]
    base = 8 + header_len
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    if random_access:
        # Rows are 264 B and scattered: without this the kernel reads a 128 KiB
        # read-around window per fault and the NVMe saturates on bandwidth.
        mm.madvise(mmap.MADV_RANDOM)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    rows, dim = info["shape"][0], int(np.prod(info["shape"][1:]))
    itemsize = {"F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4}[info["dtype"]]
    arr = np.frombuffer(mm, dtype=np.uint8, count=end - start, offset=base + start)
    arr = arr.reshape(rows, dim * itemsize)
    return arr, mm, info, fd, base + start


def gather_pread(fd: int, data_start: int, width: int, idx: np.ndarray, out: np.ndarray, threads: int) -> None:
    """Same gather through pread(2): one syscall per row, no page-fault path."""
    bounds = np.linspace(0, len(idx), threads + 1, dtype=np.int64)

    def work(part):
        for i in range(int(bounds[part]), int(bounds[part + 1])):
            out[i] = np.frombuffer(os.pread(fd, width, data_start + int(idx[i]) * width), dtype=np.uint8)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(work, range(threads)))


def gather(table: np.ndarray, idx: np.ndarray, out: np.ndarray, threads: int) -> None:
    """out[i] = table[idx[i]] with page faults overlapped across threads."""
    bounds = np.linspace(0, len(idx), threads + 1, dtype=np.int64)

    def work(part):
        a, b = int(bounds[part]), int(bounds[part + 1])
        if b > a:  # slices, so out= really lands in `out`
            np.take(table, idx[a:b], axis=0, out=out[a:b])

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(work, range(threads)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("tensor")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--rows-per-token", type=int, default=6)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["take", "pread"], default="take")
    ap.add_argument("--no-random-advice", action="store_true", help="keep kernel read-around (slow)")
    args = ap.parse_args()

    table, mm, info, fd, data_start = open_tensor(args.shard, args.tensor, not args.no_random_advice)
    if args.method == "pread":
        gather_ = lambda t, i, o, th: gather_pread(fd, data_start, t.shape[1], i, o, th)
    else:
        gather_ = gather
    print(f"method={args.method} threads={args.threads} random_advice={not args.no_random_advice}")
    rows, width = table.shape
    print(f"{args.tensor}: {rows:,} rows x {width} B ({rows * width / 2**30:.1f} GiB), "
          f"dtype {info['dtype']}")
    n = args.tokens * args.rows_per_token
    rng = np.random.default_rng(args.seed)
    out = np.empty((n, width), dtype=np.uint8)
    # Cold: fresh random rows each repeat (mostly page-cache misses on a 98 GiB table).
    for k in range(args.repeat):
        idx = rng.integers(0, rows, size=n, dtype=np.int64)
        t = time.perf_counter(); gather_(table, idx, out, args.threads); dt = time.perf_counter() - t
        print(f"cold  #{k}: {n:,} rows in {dt*1e3:8.1f} ms  = {n/dt:10,.0f} rows/s "
              f"({n*width/dt/2**20:7.1f} MiB/s)  [{dt*1e3/args.tokens:.3f} ms/token]")
    # Warm: same rows again (page cache hits).
    t = time.perf_counter(); gather_(table, idx, out, args.threads); dt = time.perf_counter() - t
    print(f"warm    : {n:,} rows in {dt*1e3:8.1f} ms  = {n/dt:10,.0f} rows/s")
    # Decode-like step: 32 tokens.
    for k in range(2):
        m = 32 * args.rows_per_token
        idx = rng.integers(0, rows, size=m, dtype=np.int64)
        t = time.perf_counter(); gather_(table, idx, out[:m], min(args.threads, m)); dt = time.perf_counter() - t
        print(f"decode #{k}: {m} rows (32 tokens) in {dt*1e3:.2f} ms")
    del table; mm.close(); os.close(fd)


if __name__ == "__main__":
    main()
