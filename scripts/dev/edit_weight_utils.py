import sys
p = sys.argv[1]
s = open(p).read()

old_default = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''
new_default = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    if name in _PLACEHOLDER_WEIGHTS:
                        yield name, _placeholder_weight(f, name)
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''
assert old_default in s
s = s.replace(old_default, new_default)

old_eager = '''        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
'''
new_eager = '''        if safetensors_load_strategy == "eager" and _has_placeholder_weights(st_file):
            # The file holds tensors a module maps from storage itself: never
            # read those into memory; the rest of the file is read lazily.
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    if name in _PLACEHOLDER_WEIGHTS:
                        yield name, _placeholder_weight(f, name)
                        continue
                    yield name, f.get_tensor(name)
        elif safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
'''
assert old_eager in s
s = s.replace(old_eager, new_eager)

old_fn = '''def multi_thread_safetensors_weights_iterator(
'''
new_fn = '''# Checkpoint tensors that a module maps from storage itself (see
# DiskEngramStorage): the iterators above hand the weight loader an empty
# placeholder of the right dtype and trailing shape instead of reading them,
# which keeps a 100 GiB table out of host memory.
_PLACEHOLDER_WEIGHTS: set[str] = set()

_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", torch.uint8),
    "U8": torch.uint8,
    "I8": torch.int8,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "I64": torch.int64,
}


def register_placeholder_weight(name: str) -> None:
    """Yield an empty placeholder for checkpoint tensor `name` instead of
    reading it; the owning module's weight loader must accept it."""
    _PLACEHOLDER_WEIGHTS.add(name)


def _placeholder_weight(f, name: str) -> torch.Tensor:
    view = f.get_slice(name)
    shape = view.get_shape()
    dtype = _SAFETENSORS_DTYPES.get(view.get_dtype(), torch.uint8)
    return torch.empty((0, *shape[1:]), dtype=dtype)


def _has_placeholder_weights(st_file: str) -> bool:
    if not _PLACEHOLDER_WEIGHTS:
        return False
    with safe_open(st_file, framework="pt") as f:
        return any(name in _PLACEHOLDER_WEIGHTS for name in f.keys())  # noqa: SIM118


def multi_thread_safetensors_weights_iterator(
'''
assert s.count(old_fn) == 1
s = s.replace(old_fn, new_fn)
open(p, "w").write(s)
print("ok")
