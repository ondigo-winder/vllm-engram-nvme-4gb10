import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/attention.py"; s = open(p).read()
old = '''    # Provided by the platform subclass.
'''
new = '''    # Sliding-window KV cache page size in tokens. Any multiple of 32 for the
    # FlashMLA kernels; FlashInfer's SM120 sparse-MLA decode kernels are only
    # instantiated for 64-token pages, so that subclass overrides it.
    swa_cache_block_size: ClassVar[int] = 32

    # Provided by the platform subclass.
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''            backend_cls=self.swa_backend_cls,
            block_size=32,
'''
new = '''            backend_cls=self.swa_backend_cls,
            block_size=self.swa_cache_block_size,
'''
assert s.count(old) == 1; s = s.replace(old, new)
assert "ClassVar" in s.split("class DeepseekV4Attention")[0], "ClassVar not imported"
open(p, "w").write(s)
p = root + "/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py"; s = open(p).read()
old = '''class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
    """DeepSeek V4 sparse MLA attention through FlashInfer's SM120 kernels."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
'''
new = old + '''    # The SM120 sparse-MLA decode kernels (flashinfer 0.7.0) exist for
    # page_block_size=64 only; a 32-token SWA page fails at the first forward.
    swa_cache_block_size: ClassVar[int] = 64
'''
assert s.count(old) == 1; s = s.replace(old, new)
assert "ClassVar" in s.split("class DeepseekV4FlashInferSM120Attention")[0], "ClassVar not imported in flashinfer_sparse"
open(p, "w").write(s); print("ok")
