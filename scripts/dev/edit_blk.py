import sys
root = sys.argv[1]
p = root + "/vllm/v1/attention/backends/mla/indexer.py"; s = open(p).read()
old = '''    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [64 if current_platform.is_device_capability_family(90) else 128]
'''
new = '''    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # The indexer page of a compress_ratio=1 layer holds block_size rows,
        # and DeepGEMM's paged MQA logits take 32- or 64-row pages everywhere
        # but on SM100 (the sparse-indexer kernels). 64 keeps the ratio-1
        # layers at 64 rows and the ratio-2 layers at 32.
        return [128 if current_platform.is_device_capability_family(100) else 64]
'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s)
p = root + "/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py"; s = open(p).read()
old = '''    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41"
'''
new = '''    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # The SM120 sparse-MLA decode kernels take the compressed pages at any
        # size (DeepSeek-V4 serves 32-row pages through them); 64 is what the
        # V4.1 indexer needs on SM120, see DeepseekV41IndexerBackend.
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41"
'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s); print("ok")
