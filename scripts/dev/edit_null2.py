import sys
root = sys.argv[1]
p = root + "/vllm/v1/worker/gpu_worker.py"; s = open(p).read()
old = '''        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            with self._get_cudagraph_capture_context():
                cuda_graph_memory_bytes = self.model_runner.capture_model()
'''
new = '''        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            with self._get_cudagraph_capture_context():
                cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Warmup and capture runs store their dummy KV into slot 0 of the null
        # block, and the rows they leave there can hold NaN bytes (near-zero
        # dummy activations quantized with an underflowing scale). Masked
        # (-1) sparse indices read that block on sm_120 and multiply by a zero
        # probability, which turns NaN; keep the null block zero.
        for cache in getattr(self.model_runner, "kv_caches", []):
            if isinstance(cache, torch.Tensor) and cache.ndim >= 1 and cache.shape[0] > 0:
                cache[0].zero_()
'''
assert s.count(old) == 1; s = s.replace(old, new)
assert "\nimport torch\n" in s
open(p, "w").write(s); print("ok")
