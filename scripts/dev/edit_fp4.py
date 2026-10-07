import sys
p = sys.argv[1] + "/vllm/v1/attention/backends/mla/indexer.py"; s = open(p).read()
old = '''    if use_fp4 and not current_platform.is_device_capability_family(100):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
'''
new = '''    if use_fp4 and not (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' requires Blackwell GPUs (sm_10x or "
            "sm_120); earlier architectures are not supported."
        )
'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s); print("ok")
