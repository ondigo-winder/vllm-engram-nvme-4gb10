import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/common/engram.py"; s = open(p).read()
old = '''        self._mappings: list[mmap.mmap] = []
        self._paths: list[str] = []
'''
new = '''        self._mappings: list[mmap.mmap] = []
        self._paths: list[str] = []
        self._view_offsets: list[int] = []
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        mapping.madvise(mmap.MADV_RANDOM)
        self._mappings.append(mapping)
        self._paths.append(path)
'''
new = '''        mapping.madvise(mmap.MADV_RANDOM)
        self._mappings.append(mapping)
        self._paths.append(path)
        self._view_offsets.append(offset - page_offset)
'''
assert s.count(old) == 1; s = s.replace(old, new)
start = s.index("    def _gather_into(")
end = s.index("\n\nclass ParallelEngramEmbedding(nn.Module):")
new_fn = '''    def _gather_into(
        self, local_rows: np.ndarray, weight_out: np.ndarray, scales_out: np.ndarray
    ) -> None:
        rows = int(local_rows.shape[0])
        local_rows = np.clip(local_rows, 0, self.num_rows - 1)
        # Tell the kernel about every page first (MADV_WILLNEED issues the
        # reads asynchronously, so the NVMe sees them all at once), then copy.
        # Faulting the rows one by one from a thread pool instead leaves the
        # device idle between reads: 56 cold rows took 2-3 ms that way and
        # ~0.7 ms this way; 2048 rows 20 ms vs 10 ms.
        uniq = np.unique(local_rows)
        for mapping, base, width in (
            (self._mappings[0], self._view_offsets[0], self.dim),
            (self._mappings[1], self._view_offsets[1], self.scale_dim),
        ):
            starts = base + uniq * width
            first = starts & ~4095
            last = (starts + width - 1) & ~4095
            pages = np.unique(np.concatenate([first, last]))
            for page in pages.tolist():
                mapping.madvise(mmap.MADV_WILLNEED, page, 4096)
        splits = 1 if rows <= 1024 else min(8, rows // 512)
        bounds = np.linspace(0, rows, splits + 1, dtype=np.int64)

        def work(part: int) -> None:
            # Slices (not index arrays) so `out=` writes into the staging buffer.
            a, b = int(bounds[part]), int(bounds[part + 1])
            if b > a:
                np.take(self.weight_np, local_rows[a:b], axis=0, out=weight_out[a:b])
                np.take(self.scales_np, local_rows[a:b], axis=0, out=scales_out[a:b])

        if splits == 1:
            work(0)
        else:
            list(self._pool.map(work, range(splits)))
'''
s = s[:start] + new_fn + s[end:]
open(p, "w").write(s); print("ok")
