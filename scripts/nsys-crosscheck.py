#!/usr/bin/env python3
"""Compare the in-library profiler's per-phase device time with nsys' kernel sums.

    scripts/nsys-crosscheck.py results/ws-nvidia/2026-09-04 [workload] [batch] [dtype]

Reads nsys-<backend>-mnist-kernels_cuda_gpu_kern_sum.csv (from `nsys stats
--report cuda_gpu_kern_sum`) for cudann and syclnn, classifies the kernel names
into the profiler's phases, and prints both shares side by side together with the
E3 row of the same configuration (profile_per_epoch_ms). Writes
nsys-crosscheck.md in the results directory."""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PHASES = ("gemm", "act", "delta", "biasgrad", "update", "loss", "reg", "other")


def phase_of(name: str) -> str:
    n = name.lower()
    if "gemm" in n or "gemv" in n or "sgemv" in n or "dgemv" in n:
        return "gemm"
    if "activ" in n or "act_k" in n or "forward" in n:
        return "act"
    if "output_delta" in n or "hidden_delta" in n or "delta" in n:
        return "delta"
    if "bias" in n:
        return "biasgrad"
    if "update" in n or "step" in n:
        return "update"
    if "loss" in n or "reduce" in n or "reduction" in n:
        return "loss"
    if "penalty" in n or "asum" in n or "nrm2" in n:
        return "reg"
    return "other"


def nsys_shares(path: Path) -> tuple[dict[str, float], float, int]:
    tot = defaultdict(float)
    total = 0.0
    launches = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["Total Time (ns)"])
            tot[phase_of(row["Name"])] += t
            total += t
            launches += int(row["Instances"])
    return {p: tot[p] / total * 100 for p in PHASES}, total, launches


def profiler_shares(root: Path, backend: str, workload: str, batch: int, dtype: str) -> tuple[dict[str, float], float, dict] | None:
    for f in root.glob("gpu-cuda-vs-sycl/*.jsonl"):
        if f.name == "superseded.jsonl":
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "results" not in r or r["backend"] != backend or r["workload"] != workload or r["batch"] != batch or r["dtype"] != dtype:
                continue
            if {k: v for k, v in (r.get("options") or {}).items() if k != "profile"}:
                continue
            per = r["results"].get("profile_per_epoch_ms") or {}
            dev = {p: per.get(f"{p}_ns", 0.0) for p in PHASES}
            if "h2d_ns" in per:
                dev["other"] += per.get("h2d_ns", 0.0) + per.get("d2h_ns", 0.0)
            total = sum(dev.values())
            return ({p: (dev[p] / total * 100 if total else 0.0) for p in PHASES}, total, r)
    return None


def main() -> None:
    root = Path(sys.argv[1])
    workload = sys.argv[2] if len(sys.argv) > 2 else "mnist-512-256"
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    dtype = sys.argv[4] if len(sys.argv) > 4 else "float"
    out = [f"# Profiler vs nsys, {workload} b={batch} {dtype}\n",
           "Shares of device time per phase: the in-library profiler (event pairs around every launch, "
           "per epoch, from the E3 row) against `nsys stats --report cuda_gpu_kern_sum` of a separate "
           "one-epoch run (kernel time only, no copies). Kernel names classified by substring.\n"]
    for backend in ("cudann", "syclnn"):
        csvs = list(root.glob(f"nsys-{backend}-*kernels*_cuda_gpu_kern_sum.csv"))
        if not csvs:
            out.append(f"\n## {backend}: no nsys kernel summary\n")
            continue
        ns, ns_total, ns_launches = nsys_shares(csvs[0])
        pr = profiler_shares(root, backend, workload, batch, dtype)
        out.append(f"\n## {backend}\n")
        out.append(f"nsys: {ns_total/1e6:.1f} ms of kernel time in the traced run, {ns_launches} kernel instances\n")
        if pr:
            shares, total, r = pr
            per = r["results"]["profile_per_epoch_ms"]
            out.append(f"profiler: {total/1e6:.1f} ms device time per epoch ({r['results'].get('steady_epoch_s', 0)*1e3:.1f} ms steady epoch), "
                       f"{(r['results'].get('profile') or {}).get('launches', 0)} launches over the timed calls\n")
        out.append("\n| phase | profiler % | nsys % |\n|---|---|---|\n")
        for p in PHASES:
            out.append(f"| {p} | {shares[p]:.1f} | {ns[p]:.1f} |\n" if pr else f"| {p} | - | {ns[p]:.1f} |\n")
    text = "".join(out)
    (root / "nsys-crosscheck.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
