#!/usr/bin/env python3
"""Markdown table of the GEMM peaks / streaming bandwidths of the third pass (newest row per
device, backend, toolchain, BLAS and dtype).   python scripts/peaks-table.py results/ws-amd/2026-09-05 results/ws-nvidia/2026-09-05"""
import glob, json, os, sys
TOOL = {"gcc14mkl": "gcc-14 + oneMKL", "gcc14amd": "gcc-14 amdgcn", "gcc14nv": "gcc-14 nvptx", "gcc14": "gcc-14",
        "clang18nv": "clang-18 nvptx", "clang18": "clang-18", "clang22": "clang-22", "amdclang": "amdclang-22", "acpp": "AdaptiveCpp"}
def tool(sweep, r):
    for k, v in TOOL.items():
        if sweep.endswith("-" + k): return v
    bi = r.get("build_info") or {}
    if "AdaptiveCpp" in (bi.get("sycl_implementation") or ""): return "AdaptiveCpp"
    if r["backend"] == "syclnn": return "DPC++"
    if r["backend"] == "cudann": return "hipcc" if "hip" in (bi.get("compiler") or "").lower() else "nvcc"
    return "?"
def dev(r):
    n = r["device"]["name"].strip()
    if n == "AdaptiveCpp OpenMP host device":
        n = ((r.get("sysinfo") or {}).get("cpu") or {}).get("model", "host") + " (AdaptiveCpp host)"
    return n.replace("AMD Ryzen Threadripper 2950X 16-Core Processor", "TR 2950X").replace("Intel(R) Xeon(R) CPU E5-2643 v2 @ 3.50GHz", "Xeon E5-2643 v2").replace("NVIDIA GeForce GTX 1080 Ti", "GTX 1080 Ti").replace("AMD Radeon RX 7900 XTX", "RX 7900 XTX")
best = {}
for d in sys.argv[1:] or ["results/ws-amd/2026-09-05", "results/ws-nvidia/2026-09-05"]:
    for f in glob.glob(os.path.join(d, "peaks*", "peak.jsonl")):
        sweep = os.path.basename(os.path.dirname(f))
        for line in open(f):
            if not line.strip(): continue
            r = json.loads(line)
            blas = (r.get("options") or {}).get("blas") or r.get("blas")
            key = (d.split("/")[-2], dev(r), r["backend"], tool(sweep, r), blas, r["dtype"])
            if key not in best or r["timestamp"] > best[key]["timestamp"]:
                best[key] = r
print("| machine | device | backend / toolchain | BLAS | dtype | n | GEMM peak (GFLOP/s) | element-wise streaming (GB/s) | host clock (MHz) |")
print("|---|---|---|---|---|---|---|---|---|")
for k in sorted(best, key=lambda k: (k[0], "GPU" not in k[1] and "XTX" not in k[1] and "Ti" not in k[1], k[1], k[2], k[3], k[4], k[5])):
    r = best[k]; cm = (r.get("results") or {}).get("cpu_monitor") or r.get("cpu_monitor") or {}
    mhz = f"{cm['mhz_mean']:.0f}" if cm.get("mhz_mean") else ("2800 (cap)" if str((r.get('sysinfo') or {}).get('cpu', {}).get('scaling_max_khz')) == "2800000" else "-")
    print(f"| {k[0]} | {k[1]} | {k[2]} / {k[3]} | {k[4]} | {k[5]} | {r.get('n') or r.get('size')} | {r['gemm_gflops']:.0f} | {r['act_bandwidth_gbs']:.0f} | {mhz} |")
