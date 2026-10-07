import sys, time, os, numpy as np, torch, mmap, struct, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
# minimal: map the weight tensor of layer 1 like DiskEngramStorage does
model = "/model"; name = "layers.1.engram.embed.weight"
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]; path = f"{model}/{idx[name]}"
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); data_start = 8 + n
off0, off1 = hdr[name]["data_offsets"]; rows_total = hdr[name]["shape"][0]; dim = 256
fd = os.open(path, os.O_RDONLY); size = os.fstat(fd).st_size
mm = mmap.mmap(fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE); mm.madvise(mmap.MADV_RANDOM)
arr = np.frombuffer(mm, dtype=np.uint8, count=rows_total * dim, offset=data_start + off0).reshape(rows_total, dim)
rng = np.random.default_rng(int(time.time()))
def bench(rows, threads, mode, reps=5):
    pool = ThreadPoolExecutor(threads); ts = []
    for _ in range(reps):
        sel = rng.integers(0, rows_total, size=rows, dtype=np.int64)  # cold rows
        out = np.empty((rows, dim), np.uint8)
        t = time.perf_counter()
        if mode == "take":
            splits = min(threads, rows); b = np.linspace(0, rows, splits + 1, dtype=np.int64)
            def work(i):
                a, e = int(b[i]), int(b[i+1])
                if e > a: np.take(arr, sel[a:e], axis=0, out=out[a:e])
            list(pool.map(work, range(splits)))
        elif mode == "pread":
            def work(i):
                out[i] = np.frombuffer(os.pread(fd, dim, data_start + off0 + int(sel[i]) * dim), np.uint8)
            list(pool.map(work, range(rows)))
        ts.append(time.perf_counter() - t)
    pool.shutdown()
    return 1e3 * np.median(ts)
for rows in [56, 224]:
    for threads in [7, 56, 128]:
        print(f"rows={rows:4d} threads={threads:3d} mmap/take: {bench(rows, threads, 'take'):6.2f} ms   pread: {bench(rows, threads, 'pread'):6.2f} ms", flush=True)
print("--- with readahead hints")
def bench2(rows, threads, mode, reps=5):
    pool = ThreadPoolExecutor(threads); ts = []
    for _ in range(reps):
        sel = rng.integers(0, rows_total, size=rows, dtype=np.int64); out = np.empty((rows, dim), np.uint8)
        t = time.perf_counter()
        base = data_start + off0
        if mode == "fadvise+take":
            for r in sel.tolist(): os.posix_fadvise(fd, base + r * dim, dim, os.POSIX_FADV_WILLNEED)
        elif mode == "madvise+take":
            for r in sel.tolist():
                o = base + r * dim; pg = o & ~4095; mm.madvise(mmap.MADV_WILLNEED, pg, (o - pg) + dim)
        splits = min(threads, rows); b = np.linspace(0, rows, splits + 1, dtype=np.int64)
        def work(i):
            a, e = int(b[i]), int(b[i+1])
            if e > a: np.take(arr, sel[a:e], axis=0, out=out[a:e])
        list(pool.map(work, range(splits)))
        ts.append(time.perf_counter() - t)
    pool.shutdown(); return 1e3 * np.median(ts)
for rows in [56, 224, 2048]:
    for threads in [1, 7, 32]:
        print(f"rows={rows:4d} threads={threads:3d} fadvise+take: {bench2(rows, threads, 'fadvise+take'):6.2f} ms  madvise+take: {bench2(rows, threads, 'madvise+take'):6.2f} ms  plain take: {bench(rows, threads, 'take'):6.2f} ms", flush=True)
