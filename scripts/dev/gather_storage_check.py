import sys, time, os, json, struct, types, ast, numpy as np, torch, mmap
from concurrent.futures import ThreadPoolExecutor
src = open("/tmp/engram_new.py").read()
def extract(src, names):
    tree = ast.parse(src); lines = src.splitlines(keepends=True); out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            name = node.name if hasattr(node, "name") else getattr(node.targets[0], "id", None)
            if name in names: out.append("".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n".join(out)
ns = {"json": json, "mmap": mmap, "os": os, "struct": struct, "np": np, "torch": torch, "time": time,
      "ThreadPoolExecutor": ThreadPoolExecutor, "logger": types.SimpleNamespace(info=print, warning=print),
      "_ENGRAM_TIMING": False, "register_placeholder_weight": lambda n: None}
exec(extract(src, {"_SAFETENSORS_ITEMSIZE", "_safetensors_tensor_span", "_resolve_checkpoint_tensor", "DiskEngramStorage"}), ns)
St = ns["DiskEngramStorage"]
st = St("/model", "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", 96000564, 96000564, 256, 32, 64)
rng = np.random.default_rng(int(time.time()))
def pread_rows(path, start, width, rows):
    fd = os.open(path, os.O_RDONLY)
    try: return np.stack([np.frombuffer(os.pread(fd, width, start + int(r) * width), np.uint8) for r in rows])
    finally: os.close(fd)
wpath = ns["_resolve_checkpoint_tensor"]("/model", "layers.1.engram.embed.weight"); wstart, hdr = ns["_safetensors_tensor_span"](wpath)
for rows in [56, 224, 2048, 12288]:
    ts = []
    for rep in range(4):
        sel = rng.integers(0, st.num_rows, size=rows, dtype=np.int64)
        w = np.empty((rows, 256), np.uint8); sc = np.empty((rows, 8), np.uint8)
        t = time.perf_counter(); st._gather_into(sel, w, sc); ts.append(time.perf_counter() - t)
    ref = pread_rows(wpath, wstart + hdr["layers.1.engram.embed.weight"]["data_offsets"][0], 256, sel + 96000564)
    refs = pread_rows(wpath, wstart + hdr["layers.1.engram.embed.scale"]["data_offsets"][0], 8, sel + 96000564)
    print(f"rows={rows:6d}: {1e3*np.median(ts):7.2f} ms (median of 4, cold)  exact={np.array_equal(w, ref) and np.array_equal(sc, refs)}")
