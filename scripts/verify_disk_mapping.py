#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check of the disk-offload storage against a real checkpoint.

Pulls ``DiskEngramStorage`` and its helpers out of the patched ``engram.py``
(no vLLM import needed: they only use json/struct/mmap/numpy/torch) and
verifies, for a few random rank shards:

1. the memory-mapped rows equal the bytes read with pread() at the offsets
   the safetensors header gives (weight and scale tensors);
2. ``gather()`` returns exactly those rows for random local indices, with the
   pinned-buffer and CUDA-event parts stubbed so this runs without a GPU.

Usage: verify_disk_mapping.py <patched engram.py> <model dir> [--layer 1] [--rows 4096]
"""
import argparse, ast, os, struct, json, sys, types
import numpy as np
import torch


def extract(src: str, names: set[str]) -> str:
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            name = node.name if hasattr(node, "name") else getattr(node.targets[0], "id", None)
            if name in names:
                out.append("".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n".join(out)


def pread_rows(path, data_start, width, rows):
    fd = os.open(path, os.O_RDONLY)
    try:
        return np.stack([np.frombuffer(os.pread(fd, width, data_start + int(r) * width), np.uint8) for r in rows])
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engram_py")
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = open(args.engram_py).read()
    ns = {"json": json, "mmap": __import__("mmap"), "os": os, "struct": struct, "np": np,
          "torch": torch, "ThreadPoolExecutor": __import__("concurrent.futures").futures.ThreadPoolExecutor,
          "logger": types.SimpleNamespace(info=print, warning=print)}
    exec(extract(src, {"_SAFETENSORS_ITEMSIZE", "_safetensors_tensor_span",
                       "_resolve_checkpoint_tensor", "DiskEngramStorage"}), ns)
    Storage = ns["DiskEngramStorage"]

    # Stub the device-side pieces: plain CPU buffers, no CUDA events.
    def _buffers(self, rows):
        if rows > self._capacity:
            self._capacity = max(rows, 2 * self._capacity)
            self._weight_buf = torch.empty((self._capacity, self.dim), dtype=torch.uint8)
            self._scales_buf = torch.empty((self._capacity, self.scale_dim), dtype=torch.uint8)
        return self._weight_buf[:rows], self._scales_buf[:rows]
    Storage._buffers = _buffers
    class _Ev:
        def record(self, *_): pass
        def synchronize(self): pass
    torch.cuda.Event = _Ev
    torch.cuda.current_stream = lambda *a, **k: None

    wname = f"layers.{args.layer}.engram.embed.weight"
    sname = f"layers.{args.layer}.engram.embed.scale"
    wpath = ns["_resolve_checkpoint_tensor"](args.model_dir, wname)
    wstart, header = ns["_safetensors_tensor_span"](wpath)
    total_rows = header[wname]["shape"][0]
    dim = int(np.prod(header[wname]["shape"][1:]))
    sdim = int(np.prod(header[sname]["shape"][1:]))
    print(f"{wname}: {total_rows:,} rows, dim {dim}, scale dim {sdim}, shard {os.path.basename(wpath)}")

    rng = np.random.default_rng(args.seed)
    ok = True
    for trial in range(3):
        num_rows = int(rng.integers(1_000_000, 120_000_000))
        vocab_start = int(rng.integers(0, total_rows - num_rows))
        st = Storage(args.model_dir, wname, sname, vocab_start, num_rows, dim, dim // sdim, threads=16)
        local = rng.integers(0, num_rows, size=args.rows, dtype=np.int64)
        # 1. mapping vs pread
        w_ref = pread_rows(wpath, wstart + header[wname]["data_offsets"][0], dim, local + vocab_start)
        s_ref = pread_rows(wpath, wstart + header[sname]["data_offsets"][0], sdim, local + vocab_start)
        m1 = np.array_equal(st.weight_np[local], w_ref) and np.array_equal(st.scales_np[local], s_ref)
        # 2. gather() on CPU
        w_g, s_g = st.gather(local, torch.device("cpu"))
        m2 = np.array_equal(w_g.view(torch.uint8).numpy(), w_ref) and np.array_equal(s_g.numpy(), s_ref)
        # 3. the parameter views alias the same rows
        pw, ps = st.parameters()
        m3 = tuple(pw.shape) == (num_rows, dim) and np.array_equal(pw[local].view(torch.uint8).numpy(), w_ref)
        print(f"shard {trial}: start {vocab_start:,} rows {num_rows:,}: mapping {'OK' if m1 else 'FAIL'}, "
              f"gather {'OK' if m2 else 'FAIL'}, params {'OK' if m3 else 'FAIL'}, "
              f"nonzero bytes {int(np.count_nonzero(w_ref))}/{w_ref.size}")
        ok &= m1 and m2 and m3
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
