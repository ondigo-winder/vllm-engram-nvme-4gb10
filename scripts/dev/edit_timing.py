import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/common/engram.py"; s = open(p).read()
old = '''        else:
            head_end = self.head_start + self.part_n_hash_cols
            local = indices[:, self.head_start : head_end] - self.vocab_start_idx
            if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
                local = nn.functional.pad(local, (0, pad), value=-1)
            local_np = local.reshape(-1).to("cpu", dtype=torch.int64).numpy()
            owned = (local_np >= 0) & (local_np < self.part_num_embeddings)
            weight, scales = self._disk.gather(local_np, indices.device)
'''
new = '''        else:
            t0 = time.perf_counter() if _ENGRAM_TIMING else 0.0
            head_end = self.head_start + self.part_n_hash_cols
            local = indices[:, self.head_start : head_end] - self.vocab_start_idx
            if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
                local = nn.functional.pad(local, (0, pad), value=-1)
            local_np = local.reshape(-1).to("cpu", dtype=torch.int64).numpy()
            owned = (local_np >= 0) & (local_np < self.part_num_embeddings)
            weight, scales = self._disk.gather(local_np, indices.device)
            if _ENGRAM_TIMING:
                self._disk.record_timing(rows, time.perf_counter() - t0)
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''    def device_buffers(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
'''
new = '''    def record_timing(self, rows: int, seconds: float) -> None:
        """VLLM_ENGRAM_TIMING=1: log the host-side gather cost every 200 lookups,
        split by batch size (decode steps gather a few hundred rows, prefill
        chunks tens of thousands)."""
        bucket = "decode" if rows <= 4096 else "prefill"
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
            logger.info("Engram disk gather [%s]: %s", self.weight_name, "; ".join(parts))
            self._timing = {}

    def device_buffers(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
'''
assert s.count(old) == 1; s = s.replace(old, new)
# init fields: find DiskEngramStorage __init__ marker '_copy_done: torch.cuda.Event | None = None'
old = '''        self._copy_done: torch.cuda.Event | None = None
'''
new = '''        self._copy_done: torch.cuda.Event | None = None
        self._timing: dict[str, tuple[int, float, int]] = {}
        self._timing_calls = 0
'''
assert s.count(old) == 1; s = s.replace(old, new)
# module-level flag + imports
old = '''from vllm.compilation.breakable_cudagraph import eager_break_during_capture
'''
new = '''from vllm.compilation.breakable_cudagraph import eager_break_during_capture
'''
assert "import time" in s or True
if "\nimport time\n" not in s:
    s = s.replace("\nimport os\n", "\nimport os\nimport time\n", 1)
assert "\nimport time\n" in s
marker = "\nlogger = init_logger(__name__)\n"
assert marker in s
s = s.replace(marker, marker + '''
# VLLM_ENGRAM_TIMING=1 logs the host-side cost of the NVMe gathers.
_ENGRAM_TIMING = os.environ.get("VLLM_ENGRAM_TIMING", "0") == "1"
''', 1)
open(p, "w").write(s); print("ok")
