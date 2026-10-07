import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/common/engram.py"; s = open(p).read()

# 1. DiskEngramStorage: static device buffers + stage()
old = '''        self._copy_done: torch.cuda.Event | None = None
        self._timing: dict[str, tuple[int, float, int]] = {}
        self._timing_calls = 0
'''
new = '''        self._copy_done: torch.cuda.Event | None = None
        self._timing: dict[str, tuple[int, float, int]] = {}
        self._timing_calls = 0
        self._ids_buf: torch.Tensor | None = None
        # Static device buffers the lookup kernel reads. CUDA graphs capture
        # the kernel on them, and `stage()` refills them before every step
        # (from the model runner's prepare_inputs, outside the graph).
        self._dev_capacity = 0
        self._dev_weight: torch.Tensor | None = None
        self._dev_scales: torch.Tensor | None = None
        self._dev_ids: torch.Tensor | None = None
        self._dev_dst: torch.Tensor | None = None
'''
assert s.count(old) == 1; s = s.replace(old, new)

old = '''    def record_timing(self, rows: int, seconds: float) -> None:
'''
new = '''    def _ensure_static(self, rows: int, device: torch.device) -> None:
        if self._dev_weight is not None and rows <= self._dev_capacity:
            return
        assert not torch.cuda.is_current_stream_capturing(), (
            "engram staging buffers must be sized before CUDA graph capture "
            f"(need {rows} rows, have {self._dev_capacity})"
        )
        cap = max(rows, 2 * self._dev_capacity)
        self._dev_weight = torch.empty(
            (cap, self.dim), dtype=torch.float8_e4m3fn, device=device
        )
        self._dev_scales = torch.empty(
            (cap, self.scale_dim), dtype=torch.uint8, device=device
        )
        self._dev_ids = torch.full((cap,), -1, dtype=torch.int64, device=device)
        self._dev_dst = torch.arange(cap, dtype=torch.int32, device=device)
        self._dev_capacity = cap

    def staged(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(rows, scales, ids, dst) views of the static buffers for `rows`."""
        self._ensure_static(rows, device)
        assert self._dev_weight is not None and self._dev_scales is not None
        assert self._dev_ids is not None and self._dev_dst is not None
        return (
            self._dev_weight[:rows],
            self._dev_scales[:rows],
            self._dev_ids[:rows],
            self._dev_dst[:rows],
        )

    def stage(self, local_rows: np.ndarray, device: torch.device) -> None:
        """Gather `local_rows` into the static device buffers for this step.

        Rows outside this rank's range (other ranks' heads, DEAD_ID) get id -1
        and are written as zeros by the lookup kernel.
        """
        t0 = time.perf_counter() if _ENGRAM_TIMING else 0.0
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
        assert self._dev_weight is not None and self._dev_scales is not None
        assert self._dev_ids is not None
        self._dev_weight[:rows].copy_(
            weight_buf.view(torch.float8_e4m3fn), non_blocking=True
        )
        self._dev_scales[:rows].copy_(scales_buf, non_blocking=True)
        self._dev_ids[:rows].copy_(ids_buf, non_blocking=True)
        if self._copy_done is None:
            self._copy_done = torch.cuda.Event()
        self._copy_done.record(torch.cuda.current_stream())
        if _ENGRAM_TIMING:
            self.record_timing(rows, time.perf_counter() - t0)

    def record_timing(self, rows: int, seconds: float) -> None:
'''
assert s.count(old) == 1; s = s.replace(old, new)

# 2. factor the thread-pooled gather out of gather()
old = '''        rows = int(local_rows.shape[0])
        weight_buf, scales_buf = self._buffers(rows)
        weight_out, scales_out = weight_buf.numpy(), scales_buf.numpy()
        local_rows = np.clip(local_rows, 0, self.num_rows - 1)
'''
new = '''        rows = int(local_rows.shape[0])
        weight_buf, scales_buf = self._buffers(rows)
        self._gather_into(local_rows, weight_buf.numpy(), scales_buf.numpy())
        weight_dev = weight_buf.to(device, non_blocking=True).view(torch.float8_e4m3fn)
        scales_dev = scales_buf.to(device, non_blocking=True)
        if self._copy_done is None:
            self._copy_done = torch.cuda.Event()
        self._copy_done.record(torch.cuda.current_stream())
        return weight_dev, scales_dev

    def _gather_into(
        self, local_rows: np.ndarray, weight_out: np.ndarray, scales_out: np.ndarray
    ) -> None:
        rows = int(local_rows.shape[0])
        local_rows = np.clip(local_rows, 0, self.num_rows - 1)
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        if splits == 1:
            work(0)
        else:
            list(self._pool.map(work, range(splits)))
        weight_dev = weight_buf.to(device, non_blocking=True).view(torch.float8_e4m3fn)
        scales_dev = scales_buf.to(device, non_blocking=True)
        if self._copy_done is None:
            self._copy_done = torch.cuda.Event()
        self._copy_done.record(torch.cuda.current_stream())
        return weight_dev, scales_dev
'''
new = '''        if splits == 1:
            work(0)
        else:
            list(self._pool.map(work, range(splits)))
'''
assert s.count(old) == 1; s = s.replace(old, new)

# 3. ParallelEngramEmbedding: prefetch_rows + lookup from static buffers
start = s.index("    def _lookup_from_disk(")
end = s.index("    def forward(self, indices: torch.Tensor) -> torch.Tensor:", start)
new_block = '''    def _local_rows(self, indices: torch.Tensor) -> np.ndarray:
        head_end = self.head_start + self.part_n_hash_cols
        local = indices[:, self.head_start : head_end] - self.vocab_start_idx
        if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
            local = nn.functional.pad(local, (0, pad), value=-1)
        return local.reshape(-1).to("cpu", dtype=torch.int64).numpy()

    def prefetch_rows(self, indices: torch.Tensor) -> None:
        """Stage this step's rows from storage (called by the model runner
        before the forward, outside any CUDA graph).

        `indices` is [num_tokens, n_hash_cols] for the padded batch the
        forward will see, so the staged rows line up with the kernel's.
        """
        assert self._disk is not None
        if indices.shape[0] == 0:
            return
        self._disk.stage(self._local_rows(indices), indices.device)

    def _lookup_from_disk(
        self, indices: torch.Tensor, out: torch.Tensor, rows: int, background: bool
    ) -> None:
        """Dequantize the staged rows into `out`.

        The rows were staged by `prefetch_rows` for this step; the kernel only
        reads the static buffers, so it can be captured in a CUDA graph and
        replayed after every restage. Masked rows (id -1) come out as zeros.
        """
        assert self._disk is not None
        weight, scales, ids, dst = self._disk.staged(rows, indices.device)
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            weight,
            scales,
            ids,
            dst,
            out,
            0,
            rows,
            rows,
            0,
            0,
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
            SORTED=True,
        )

'''
s = s[:start] + new_block + s[end:]
# device_buffers no longer needed; keep (harmless). Remove the capture-only path's helper usage check:
assert "device_buffers(rows, indices.device)" not in s
open(p, "w").write(s)

# 4. model_state hook
p = root + "/vllm/models/deepseek_v41/nvidia/model_state.py"; s = open(p).read()
old = '''        model_inputs["lookback_token_ids"] = window
        return model_inputs
'''
new = '''        model_inputs["lookback_token_ids"] = window
        self._stage_engram_rows(input_batch, window)
        return model_inputs

    def _stage_engram_rows(
        self, input_batch: InputBatch, lookback_token_ids: torch.Tensor
    ) -> None:
        """Engram disk offload: hash this step's tokens and stage their rows
        from storage before the forward runs (CUDA graphs replay the lookup
        kernel on the staged buffers, so the host work has to happen here)."""
        if self._engram_disk is None:
            from vllm.models.deepseek_v41.common.engram import (
                Engram,
                NgramHashState,
            )

            hash_state = next(
                (m for m in self.model.modules() if isinstance(m, NgramHashState)),
                None,
            )
            layers = [
                m
                for m in self.model.modules()
                if isinstance(m, Engram)
                and getattr(m.embed_tokens, "_disk", None) is not None
            ]
            self._engram_disk = (hash_state, layers) if hash_state and layers else ()
        if not self._engram_disk:
            return
        hash_state, layers = self._engram_disk
        from vllm.models.deepseek_v41.common.engram import gather_engram_hashes
        from vllm.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask

        num_tokens = input_batch.num_tokens_after_padding
        input_ids = input_batch.input_ids[:num_tokens]
        positions = input_batch.positions[:num_tokens]
        query_start_loc = input_batch.query_start_loc[: input_batch.num_reqs + 1]
        hashes = hash_state(
            input_ids,
            positions,
            query_start_loc,
            image_sentinel_mask(input_ids),
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        hashes = gather_engram_hashes(hashes)
        for engram in layers:
            engram.embed_tokens.prefetch_rows(hashes[:, engram.layer_hash_index])
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        self.decoder_replay_layers: DecoderReplayLayers | None = getattr(
            model, "decoder_replay_layers", None
        )
'''
new = '''        self.decoder_replay_layers: DecoderReplayLayers | None = getattr(
            model, "decoder_replay_layers", None
        )
        # (hash state, Engram layers with disk-offloaded tables), resolved on
        # first use; () when the model has none.
        self._engram_disk: tuple | None = None
'''
assert s.count(old) == 1; s = s.replace(old, new)
open(p, "w").write(s); print("ok")
