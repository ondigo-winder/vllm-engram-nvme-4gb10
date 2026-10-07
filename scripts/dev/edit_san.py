import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py"; s = open(p).read()
helper = '''

def _sm120_fill_masked_indices(indices: torch.Tensor) -> torch.Tensor:
    """Replace -1 (masked) sparse indices by a slot the row already reads.

    The sm_120 sparse-MLA kernels still fetch a KV row for a masked index --
    block 0 -- and multiply it by a zero probability, so any NaN byte in the
    null block poisons the row (warmup and padded rows leave such bytes
    there). Pointing the masked entries at the row's first valid slot keeps
    every read inside rows that hold finite data; `topk_lens` masks them.
    """
    first = indices[..., :1]
    safe = torch.where(first >= 0, first, torch.zeros_like(first))
    return torch.where(indices >= 0, indices, safe.expand_as(indices))


class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
'''
old = '''

class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
'''
assert s.count(old) == 1; s = s.replace(old, helper)
old = '''        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
'''
new = '''        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        swa_indices = _sm120_fill_masked_indices(swa_indices[:num_decode_tokens])
        if extra_sparse_indices is not None:
            extra_sparse_indices = _sm120_fill_masked_indices(extra_sparse_indices)
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''            q_chunk = q[query_start:query_end]
            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
'''
new = '''            q_chunk = q[query_start:query_end]
            swa_indices_chunk = _sm120_fill_masked_indices(
                swa_metadata.prefill_swa_indices[query_start:query_end]
            )
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_sparse_indices_chunk is not None:
                extra_sparse_indices_chunk = _sm120_fill_masked_indices(
                    extra_sparse_indices_chunk
                )
'''
assert s.count(old) == 1; s = s.replace(old, new)
open(p, "w").write(s); print("ok")
