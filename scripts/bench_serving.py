#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serving benchmark for the GB10 cluster: prefill time by prompt length, decode
throughput by concurrency, and NVMe / page-cache activity during decode.

Usage: bench_serving.py [--url http://localhost:8000] [--model deepseek-v4.1-flash]
       [--prefill 2048,8192,32768,65536] [--decode 1,4,8] [--tokens 256] [--iostat nvme0n1]
"""
import argparse, json, random, subprocess, sys, time, urllib.request, concurrent.futures

def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def prompt_of(n_tokens, seed=0):
    # ~1.35 tokens per word for this tokenizer on random greek words; the server reports the real count.
    rng = random.Random(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
    return " ".join(rng.choice(words) for _ in range(int(n_tokens / 1.35))) + " The capital of France is"

def read_iostat(dev):
    try:
        out = subprocess.run(["iostat", "-dk", dev, "1", "2"], capture_output=True, text=True, timeout=10).stdout
        line = [l for l in out.splitlines() if l.startswith(dev)][-1].split()
        return float(line[1]), float(line[2])  # tps, kB_read/s
    except Exception:
        return None

def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":"); d[k] = int(v.split()[0]) // 1024
    return d["MemAvailable"], d["Cached"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prefill", default="2048,8192,32768"); ap.add_argument("--decode", default="1,4,8")
    ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--iostat", default="nvme0n1")
    a = ap.parse_args()
    print(f"# {time.strftime('%Y-%m-%d %H:%M')} model={a.model} url={a.url}")
    print("## prefill (max_tokens=1, no prefix-cache hits: fresh random prompt each)")
    for n in [int(x) for x in a.prefill.split(",")]:
        body = {"model": a.model, "prompt": prompt_of(n, seed=n + int(time.time())), "max_tokens": 1, "temperature": 0}
        t = time.time(); r = post(a.url, body); dt = time.time() - t
        pt = r["usage"]["prompt_tokens"]
        print(f"prompt {pt:6d} tokens: {dt:6.2f} s  -> {pt/dt:7.0f} tok/s")
    print("## decode (ignore_eos, essay prompts)")
    for n in [int(x) for x in a.decode.split(",")]:
        def one(i):
            body = {"model": a.model, "prompt": f"Explain in detail how a {i} stage rocket reaches orbit. Essay:", "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True}
            t = time.time(); r = post(a.url, body); return time.time() - t, r["usage"]["completion_tokens"]
        io0 = read_iostat(a.iostat); t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(one, range(n)))
        wall = time.time() - t0; tot = sum(c for _, c in res)
        print(f"{n} streams x {a.tokens}: wall {wall:5.1f} s, aggregate {tot/wall:6.1f} tok/s, per stream {tot/wall/n:5.1f} tok/s")
    avail, cached = meminfo()
    print(f"## memory: MemAvailable {avail} MiB, page cache {cached} MiB")
    io = read_iostat(a.iostat)
    if io: print(f"## {a.iostat} now: {io[0]:.0f} IOPS, {io[1]/1024:.1f} MiB/s read")

if __name__ == "__main__":
    main()
