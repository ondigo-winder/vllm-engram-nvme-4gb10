import sys
p = sys.argv[1] + "/vllm/models/deepseek_v41/common/engram.py"; s = open(p).read()
old = '''        assert self._disk is not None
        head_end = self.head_start + self.part_n_hash_cols
        local = indices[:, self.head_start : head_end] - self.vocab_start_idx
        if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
            local = nn.functional.pad(local, (0, pad), value=-1)
        local_np = local.reshape(-1).to("cpu", dtype=torch.int64).numpy()
        owned = (local_np >= 0) & (local_np < self.part_num_embeddings)
        weight, scales = self._disk.gather(local_np, indices.device)
        # The gathered rows form a dense table of `rows` entries in output
        # order, so the sorted-lookup path is run against it with identity
        # destinations; masked rows (DEAD_ID, other ranks' heads) read as -1 and
        # are written as zeros, exactly as in the in-memory path.
        ids = torch.from_numpy(np.where(owned, np.arange(rows), -1)).to(
            indices.device, non_blocking=True
        )
'''
new = '''        assert self._disk is not None
        if torch.cuda.is_current_stream_capturing():
            # A plain (non-breakable) capture, e.g. the one that sizes CUDA
            # graph memory at startup, cannot carry host work or syncs. Stage
            # nothing and mask every row; the breakable capture used for
            # replay runs this lookup eagerly on every step.
            weight, scales = self._disk.device_buffers(rows, indices.device)
            ids = torch.full((rows,), -1, dtype=torch.int64, device=indices.device)
        else:
            head_end = self.head_start + self.part_n_hash_cols
            local = indices[:, self.head_start : head_end] - self.vocab_start_idx
            if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
                local = nn.functional.pad(local, (0, pad), value=-1)
            local_np = local.reshape(-1).to("cpu", dtype=torch.int64).numpy()
            owned = (local_np >= 0) & (local_np < self.part_num_embeddings)
            weight, scales = self._disk.gather(local_np, indices.device)
            # The gathered rows form a dense table of `rows` entries in output
            # order, so the sorted-lookup path is run against it with identity
            # destinations; masked rows (DEAD_ID, other ranks' heads) read as
            # -1 and are written as zeros, exactly as in the in-memory path.
            ids = torch.from_numpy(np.where(owned, np.arange(rows), -1)).to(
                indices.device, non_blocking=True
            )
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''    def gather(
        self, local_rows: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
'''
new = '''    def device_buffers(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Device tensors of the shape `gather` returns, contents unspecified."""
        return (
            torch.empty((rows, self.dim), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((rows, self.scale_dim), dtype=torch.uint8, device=device),
        )

    def gather(
        self, local_rows: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s); print("ok")
