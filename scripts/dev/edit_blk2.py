import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/attention.py"; s = open(p).read()
old = '''def _use_v41_mxfp8_kv_record() -> bool:
    return current_platform.is_device_capability_family(100)
'''
new = '''def _use_v41_mxfp8_kv_record() -> bool:
    return current_platform.is_device_capability_family(100)


def _kv_block_size_for_layer(compress_ratio: int, default_block_size: int) -> int:
    """KV block size (tokens) of a compressed-KV source layer.

    On sm_120 every kernel that reads a compressed or indexer page (FlashInfer
    sparse-MLA decode/prefill, DeepGEMM paged MQA logits) is instantiated for
    64-row pages only, while V4.1 mixes compress ratios 2 and 1. One block size
    cannot give both 64 rows, so each ratio gets its own: 128 tokens for the
    ratio-2 layers, 64 for the ratio-1 layers. The groups then pack with
    different block sizes, as the sliding-window cache already does.
    """
    if current_platform.is_device_capability_family(120) and compress_ratio in (1, 2):
        return 64 * compress_ratio
    return default_block_size
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
'''
new = '''        return MLAAttentionSpec(
            block_size=_kv_block_size_for_layer(
                self.compress_ratio, vllm_config.cache_config.block_size
            ),
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
'''
new = '''        return MLAAttentionSpec(
            block_size=_kv_block_size_for_layer(
                self.compress_ratio, self.cache_config.block_size
            ),
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
'''
assert s.count(old) == 1; s = s.replace(old, new)
open(p, "w").write(s)
p = root + "/vllm/v1/attention/backends/mla/indexer.py"; s = open(p).read()
old = '''        return [128 if current_platform.is_device_capability_family(100) else 64]
'''
new = '''        if current_platform.is_device_capability_family(100):
            return [128]
        if current_platform.is_device_capability_family(120):
            # Per-layer block sizes (see _kv_block_size_for_layer): 128 for
            # the ratio-2 layers, 64 for the ratio-1 layers, 64 rows each.
            return [64, 128]
        return [64]
'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s); print("ok")
