import torch, math
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
d = torch.load("/root/.cache/dbg_layer0.pt")
dev = torch.device("cuda"); bs = d["block_size"]
blocks = sorted(d["pages"]); remap = {b: i for i, b in enumerate(blocks)}
cache = torch.stack([d["pages"][b] for b in blocks]).to(dev)  # [n,64,1,584]
idx = d["swa_indices"].clone(); valid = idx >= 0
blk = torch.where(valid, idx // bs, 0); off = torch.where(valid, idx % bs, 0)
new_blk = torch.tensor([remap.get(int(b), 0) for b in blk.flatten().tolist()]).view_as(blk)
idx_r = torch.where(valid, new_blk * bs + off, torch.full_like(idx, -1)).to(dev).int()
q = d["q"].to(dev); lens = d["swa_lens"].to(dev); sinks = d["sinks"].to(dev); scale = d["scale"]
ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
# reference: dequantized K/V rows per slot
def dequant_block(b):
    flat = cache[b].reshape(-1)
    data = flat[:64*576].view(64, 576); sc = flat[64*576:64*576+64*8].view(64, 8)
    fp8 = data[:, :448].view(torch.float8_e4m3fn).float().view(64, 7, 64)
    s = torch.ldexp(torch.ones(64, 7, device=dev), (sc[:, :7].int() - 127))
    nope = (fp8 * s[:, :, None]).reshape(64, 448)
    rope = data[:, 448:576].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, rope], 1)  # [64, 512]
K = torch.cat([dequant_block(b) for b in range(cache.shape[0])], 0)  # [slots, 512]
def reference(q, idx, lens, sinks):
    T, H, D = q.shape; out = torch.zeros(T, H, D, device=dev)
    for t in range(T):
        L = int(lens[t]); sl = idx[t, 0, :L].long(); k = K[sl]  # [L,512]
        logits = q[t].float() @ k.T * scale  # [H, L]
        if sinks is not None:
            logits = torch.cat([logits, sinks.float()[:, None]], 1)
        p = torch.softmax(logits, -1)[:, :L]
        out[t] = p @ k
    return out
ref = reference(q, idx_r, lens, sinks)
def run(q, cache, idx, lens, sinks, label, ref=None):
    out = torch.empty_like(q)
    flashinfer_trtllm_batch_decode_sparse_mla_dsv4(query=q, swa_kv_cache=cache, workspace_buffer=ws, sparse_indices=idx, compressed_kv_cache=None, out=out, bmm1_scale=scale, sinks=sinks, kv_layout="NHD", swa_topk_lens=lens, extra_sparse_indices=None, extra_sparse_topk_lens=None)
    torch.cuda.synchronize(); o = out.float()
    nanrows = torch.isnan(o).flatten(1).any(1)
    msg = f"{label}: nan rows {int(nanrows.sum())}/{o.shape[0]}"
    if ref is not None and (~nanrows).any():
        err = (o[~nanrows] - ref[~nanrows]).abs().max().item(); msg += f" maxerr(good rows)={err:.4f} refabsmax={ref.abs().max().item():.3f}"
    print(msg, flush=True); return out
run(q, cache, idx_r, lens, sinks, "A as dumped", ref)
run(q, cache, idx_r, lens, None, "B no sinks", reference(q, idx_r, lens, None))
run(q[:8].contiguous(), cache, idx_r[:8].contiguous(), lens[:8].contiguous(), sinks, "C first 8 (decode form)", ref[:8])
run(q[:64].contiguous(), cache, idx_r[:64].contiguous(), lens[:64].contiguous(), sinks, "D first 64 (decode form)", ref[:64])
run(q[:65].contiguous(), cache, idx_r[:65].contiguous(), lens[:65].contiguous(), sinks, "E first 65 (prefill form)", ref[:65])
# F: fill padding with a valid slot (no -1) keeping lens
idx_f = idx_r.clone(); idx_f[idx_f < 0] = idx_r[:, :, :1].expand_as(idx_f)[idx_f < 0]
run(q, cache, idx_f, lens, sinks, "F no -1 padding", ref)
# G: topk lens all = 128 with -1 padding (kernel masks -1?)
run(q, cache, idx_r, torch.full_like(lens, 128), sinks, "G lens=128 (-1 as mask)", ref)
# H: q scaled down
run(q * 0.1, cache, idx_r, lens, sinks, "H q*0.1", reference(q*0.1, idx_r, lens, sinks))
for stride_bytes in [138240, 64*584, 2*64*584, 138240 - 416]:
    big = torch.zeros((cache.shape[0], stride_bytes), dtype=torch.uint8, device=dev)
    big[:, :64*584] = cache.reshape(cache.shape[0], -1)
    padded = big[:, :64*584].view(cache.shape[0], 64, 1, 584)
    run(q, padded, idx_r, lens, sinks, f"P block stride {stride_bytes} (strides {padded.stride()})", ref)
# place pages at an offset inside the block (like a layer page not at block start)
big = torch.zeros((cache.shape[0], 138240), dtype=torch.uint8, device=dev)
big[:, 4096:4096+64*584] = cache.reshape(cache.shape[0], -1)
padded = big[:, 4096:4096+64*584].view(cache.shape[0], 64, 1, 584)
run(q, padded, idx_r, lens, sinks, f"Q offset page in 138240 block", ref)
import gc; del big; gc.collect(); torch.cuda.empty_cache()
nb = d["cache_shape"][0]
huge = torch.zeros((nb, 138240), dtype=torch.uint8, device=dev)
for b in blocks: huge[b, :64*584] = d["pages"][b].reshape(-1).to(dev)
hview = huge[:, :64*584].view(nb, 64, 1, 584)
idx_orig = d["swa_indices"].to(dev).int()
run(q, hview, idx_orig, lens, sinks, f"R full-size cache {tuple(hview.shape)} strides {hview.stride()}", ref)
for n in [1000, 10000, 15000, 20000, 40000, 80000]:
    v = huge[:n, :64*584].view(n, 64, 1, 584)
    run(q, v, idx_orig, lens, sinks, f"R{n} blocks", ref)
del huge; gc.collect(); torch.cuda.empty_cache()
for mb, fill in [(128, 0), (128, 255), (256, 255), (128, 127)]:
    ws = torch.full((mb << 20,), fill, dtype=torch.uint8, device=dev)
    run(q, cache, idx_r, lens, sinks, f"W workspace {mb}MB filled {fill}", ref)
# random garbage workspace
ws = torch.randint(0, 256, (128 << 20,), dtype=torch.uint8, device=dev)
run(q, cache, idx_r, lens, sinks, "W workspace random", ref)
ws = torch.randint(0, 256, (128 << 20,), dtype=torch.uint8, device=dev)
run(q[:8].contiguous(), cache, idx_r[:8].contiguous(), lens[:8].contiguous(), sinks, "W random, decode form 8", ref[:8])
