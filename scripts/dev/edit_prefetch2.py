import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/common/engram.py"; s = open(p).read()
# stage(): take precomputed local rows + record sub-timings; split into prepare (sync) and gather parts
old = '''        t0 = time.perf_counter() if _ENGRAM_TIMING else 0.0
        rows = int(local_rows.shape[0])
        self._ensure_static(rows, device)
        owned = (local_rows >= 0) & (local_rows < self.num_rows)
        weight_buf, scales_buf = self._buffers(rows)
        if self._ids_buf is None or self._ids_buf.shape[0] < self._capacity:
            self._ids_buf = torch.empty(
                (self._capacity,), dtype=torch.int64, pin_memory=True
            )
        ids_buf = self._ids_buf[:rows]
        self._gather_into(local_rows, weight_buf.numpy(), scales_buf.numpy())
        np.copyto(ids_buf.numpy(), np.where(owned, np.arange(rows), -1))
'''
new = '''        t0 = time.perf_counter() if _ENGRAM_TIMING else 0.0
        rows = int(local_rows.shape[0])
        self._ensure_static(rows, device)
        owned = (local_rows >= 0) & (local_rows < self.num_rows)
        weight_buf, scales_buf = self._buffers(rows)
        if self._ids_buf is None or self._ids_buf.shape[0] < self._capacity:
            self._ids_buf = torch.empty(
                (self._capacity,), dtype=torch.int64, pin_memory=True
            )
        ids_buf = self._ids_buf[:rows]
        t1 = time.perf_counter() if _ENGRAM_TIMING else 0.0
        self._gather_into(local_rows, weight_buf.numpy(), scales_buf.numpy())
        t2 = time.perf_counter() if _ENGRAM_TIMING else 0.0
        np.copyto(ids_buf.numpy(), np.where(owned, np.arange(rows), -1))
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        self._copy_done.record(torch.cuda.current_stream())
        if _ENGRAM_TIMING:
            self.record_timing(rows, time.perf_counter() - t0)

    def record_timing(self, rows: int, seconds: float) -> None:
'''
new = '''        self._copy_done.record(torch.cuda.current_stream())
        if _ENGRAM_TIMING:
            self.record_timing(rows, time.perf_counter() - t0, t1 - t0, t2 - t1)

    def record_timing(
        self, rows: int, seconds: float, wait: float = 0.0, gather: float = 0.0
    ) -> None:
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        bucket = "decode" if rows <= 4096 else "prefill"
        n, t, r = self._timing.get(bucket, (0, 0.0, 0))
        self._timing[bucket] = (n + 1, t + seconds, r + rows)
        self._timing_calls += 1
        if self._timing_calls % 200 == 0:
            parts = []
            for b, (n, t, r) in sorted(self._timing.items()):
                if n:
                    parts.append(
                        f"{b}: {n} lookups, {1e3 * t / n:.2f} ms avg, "
                        f"{r / n:.0f} rows avg, {r / max(t, 1e-9) / 1e3:.0f}k rows/s"
                    )
'''
new = '''        bucket = "decode" if rows <= 4096 else "prefill"
        n, t, r, w, g = self._timing.get(bucket, (0, 0.0, 0, 0.0, 0.0))
        self._timing[bucket] = (n + 1, t + seconds, r + rows, w + wait, g + gather)
        self._timing_calls += 1
        if self._timing_calls % 200 == 0:
            parts = []
            for b, (n, t, r, w, g) in sorted(self._timing.items()):
                if n:
                    parts.append(
                        f"{b}: {n} lookups, {1e3 * t / n:.2f} ms avg "
                        f"(wait {1e3 * w / n:.2f}, gather {1e3 * g / n:.2f}), "
                        f"{r / n:.0f} rows avg, {r / max(g, 1e-9) / 1e3:.0f}k rows/s"
                    )
'''
assert s.count(old) == 1; s = s.replace(old, new)
s = s.replace("        self._timing: dict[str, tuple[int, float, int]] = {}\n", "        self._timing: dict[str, tuple[int, float, int, float, float]] = {}\n")
# prefetch_rows accepts precomputed local rows
old = '''        assert self._disk is not None
        if indices.shape[0] == 0:
            return
        self._disk.stage(self._local_rows(indices), indices.device)
'''
new = '''        assert self._disk is not None
        if indices.shape[0] == 0:
            return
        self._disk.stage(self._local_rows(indices), indices.device)

    def local_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """This rank's rows for `indices` ([num_tokens, n_hash_cols]) as a
        flat device tensor of local indices (-1 for other ranks' heads)."""
        head_end = self.head_start + self.part_n_hash_cols
        local = indices[:, self.head_start : head_end] - self.vocab_start_idx
        if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
            local = nn.functional.pad(local, (0, pad), value=-1)
        return local.reshape(-1)

    def stage_local_rows(self, local_np: np.ndarray, device: torch.device) -> None:
        assert self._disk is not None
        if local_np.shape[0]:
            self._disk.stage(local_np, device)
'''
assert s.count(old) == 1; s = s.replace(old, new)
open(p, "w").write(s)

p = root + "/vllm/models/deepseek_v41/nvidia/model_state.py"; s = open(p).read()
old = '''        hashes = gather_engram_hashes(hashes)
        for engram in layers:
            engram.embed_tokens.prefetch_rows(hashes[:, engram.layer_hash_index])
'''
new = '''        hashes = gather_engram_hashes(hashes)
        # One device-to-host copy for all tables, then the gathers run side by
        # side in the tables' thread pools.
        locals_dev = [
            engram.embed_tokens.local_rows(hashes[:, engram.layer_hash_index])
            for engram in layers
        ]
        locals_np = torch.stack(locals_dev).to("cpu", dtype=torch.int64).numpy()
        if len(layers) == 1:
            layers[0].embed_tokens.stage_local_rows(locals_np[0], hashes.device)
            return
        import concurrent.futures

        if self._engram_pool is None:
            self._engram_pool = concurrent.futures.ThreadPoolExecutor(len(layers))
        futures = [
            self._engram_pool.submit(
                engram.embed_tokens.stage_local_rows, locals_np[i], hashes.device
            )
            for i, engram in enumerate(layers)
        ]
        for f in futures:
            f.result()
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        self._engram_disk: tuple | None = None
'''
new = '''        self._engram_disk: tuple | None = None
        self._engram_pool = None
'''
assert s.count(old) == 1; s = s.replace(old, new)
open(p, "w").write(s); print("ok")
