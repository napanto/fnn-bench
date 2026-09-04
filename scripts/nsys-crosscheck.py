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
    """Kernel name -> profiler phase (cudann's kernels and syclnn's mangled lambda names)."""
    n = name.lower()
    if "output_delta" in n or "ndrangereduction" in n or "reduction" in n or "loss" in n:
        return "loss"  # the fused output-delta + loss kernel (a sycl::reduction in syclnn)
    if "hidden_delta" in n or "delta" in n:
        return "delta"
    if "gemv" in n:
        return "biasgrad"  # the only GEMV is the bias-gradient (ones-vector) call
    if "gemm" in n:
        return "gemm"
    if "activ" in n or "forward_layer" in n:
        return "act"
    if "update" in n:
        return "update"
    if "penalty" in n or "asum" in n or "nrm2" in n:
        return "reg"
    return "other"


def nsys_shares(path: Path) -> tuple[dict[str, float], float, int, int]:
    tot = defaultdict(float)
    total = 0.0
    launches = 0
    batches = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["Total Time (ns)"])
            ph = phase_of(row["Name"])
            tot[ph] += t
            total += t
            launches += int(row["Instances"])
            if ph == "loss":  # one output-delta/loss kernel per batch
                batches += int(row["Instances"])
    return {p: tot[p] / total * 100 for p in PHASES}, total, launches, max(batches, 1)


def profiler_shares(root: Path, backend: str, workload: str, batch: int, dtype: str):
    """Kernel-only shares from the E3 row (profile_per_epoch_ms is in ms; H2D/D2H excluded)."""
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
            total = sum(dev.values())
            batches = (r["n_samples"] + batch - 1) // batch
            return {p: (dev[p] / total * 100 if total else 0.0) for p in PHASES}, total, batches, per, r
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
        ns, ns_total, ns_launches, ns_batches = nsys_shares(csvs[0])
        pr = profiler_shares(root, backend, workload, batch, dtype)
        out.append(f"\n## {backend}\n\n")
        out.append(f"- nsys: {ns_total/1e6:.1f} ms of kernel time over {ns_batches} batches "
                   f"({ns_total/1e6/ns_batches:.3f} ms per batch), {ns_launches} kernel instances "
                   f"({ns_launches/ns_batches:.1f} per batch)\n")
        if pr:
            shares, total, batches, per, r = pr
            out.append(f"- profiler (E3 row): {total:.1f} ms of kernel phases per epoch over {batches} batches "
                       f"({total/batches:.3f} ms per batch), plus H2D {per.get('h2d_ns', 0):.1f} ms and D2H {per.get('d2h_ns', 0):.2f} ms "
                       f"per epoch; steady epoch {r['results'].get('steady_epoch_s', 0)*1e3:.1f} ms\n")
            out.append(f"- ratio profiler/nsys per batch: {(total/batches)/(ns_total/1e6/ns_batches):.2f}x "
                       "(event pairs bracket the launch, nsys measures the kernel)\n")
        out.append("\n| phase | profiler % | nsys % |\n|---|---|---|\n")
        for p in PHASES:
            out.append(f"| {p} | {shares[p]:.1f} | {ns[p]:.1f} |\n" if pr else f"| {p} | - | {ns[p]:.1f} |\n")
    text = "".join(out)
    (root / "nsys-crosscheck.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
