import sys
root = sys.argv[1]
note = '''        # Block 0 is the null block: never written, but the sm_120 sparse-MLA
        # kernels read it for every masked (-1) index and multiply by a zero
        # probability, so any NaN byte left in it poisons the output. Zero it.
        if kv_cache.numel() and kv_cache.shape[0] > 0:
            kv_cache[0].zero_()
'''
p = root + "/vllm/models/deepseek_v41/attention.py"; s = open(p).read()
old = '''    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)
'''
assert s.count(old) == 2
s = s.replace(old, old + note); open(p, "w").write(s)
p = root + "/vllm/v1/attention/backends/mla/sparse_swa.py"; s = open(p).read()
assert s.count(old) == 1
s = s.replace(old, old + note); open(p, "w").write(s); print("ok")
