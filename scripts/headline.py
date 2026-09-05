#!/usr/bin/env python3
"""Headline tables from a results directory (pass-3 layout, one sweep per subdirectory).

    PYTHONPATH=bench python scripts/headline.py results/ws-amd/2026-09-05 [results/ws-nvidia/2026-09-05 ...]

Prints, as Markdown:
  1. the default-configuration steady epoch per (machine, device, backend, toolchain) for the
     reference workloads (monk b40, cup b40, mnist-512-256 b256 / b1024, float);
  2. the tiled-BLAS (E7) epoch next to the vendor-BLAS one, same grouping;
  3. the run-to-run spread: identical visible configurations that were measured in more than one
     sweep (E3 default row vs E7 `blas=auto` row vs E6 ablation baseline ...).

The toolchain comes from the sweep directory name (the rows themselves only carry `build_info`),
so the layout must be `<results>/<sweep>/<backend>.jsonl`.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from fnnbench import results as R  # noqa: E402

REF = [("monk", 40), ("cup", 40), ("mnist-512-256", 256), ("mnist-512-256", 1024)]

TOOLCHAIN = {  # sweep-directory suffix -> label
    "gcc14mkl": "gcc-14 + oneMKL", "gcc14amd": "gcc-14 amdgcn", "gcc14nv": "gcc-14 nvptx", "gcc14": "gcc-14",
    "clang18nv": "clang-18 nvptx", "clang18": "clang-18", "clang22": "clang-22", "amdclang": "amdclang-22",
    "acpp": "AdaptiveCpp", "dpcpp": "DPC++",
}


def toolchain(sweep: str, row: dict) -> str:
    for suf, lab in TOOLCHAIN.items():
        if sweep.endswith("-" + suf) or sweep == suf:
            return lab
    impl = (row.get("build_info") or {}).get("sycl_implementation") or ""
    if "AdaptiveCpp" in impl:
        return "AdaptiveCpp"
    if row["backend"] == "syclnn":
        return "DPC++"
    if row["backend"] == "cudann":
        comp = (row.get("build_info") or {}).get("compiler") or ""
        return "hipcc" if "hip" in comp.lower() or "clang" in comp.lower() and "rocm" in comp.lower() else "nvcc"
    return "?"


def load(dirs):
    out = []
    for d in dirs:
        machine = d.rstrip("/").split("/")[-2] if "/" in d.rstrip("/") else d
        for f in glob.glob(os.path.join(d, "*", "*.jsonl")):
            base = os.path.basename(f)
            if base in ("superseded.jsonl", "peak.jsonl"):
                continue
            sweep = os.path.basename(os.path.dirname(f))
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("error") or not (r.get("results") or {}).get("steady_epoch_s"):
                    continue
                r["_sweep"], r["_machine"] = sweep, machine
                out.append(r)
    return out


def dev(r):
    n = r["device"]["name"].strip()
    return n.replace("AMD Ryzen Threadripper 2950X 16-Core Processor", "TR 2950X").replace(
        "Intel(R) Xeon(R) CPU E5-2643 v2 @ 3.50GHz", "Xeon E5-2643v2").replace("NVIDIA GeForce GTX 1080 Ti", "1080 Ti").replace(
        "AMD Radeon RX 7900 XTX", "7900 XTX").replace("AdaptiveCpp OpenMP host device", "TR 2950X (acpp host)")


def ms(v):
    return f"{st.median(v):.1f}"


def main(dirs):
    raw = [r for r in load(dirs) if r.get("mode") == "train" and r["dtype"] == "float"]
    rows = R.dedupe(raw)
    # 1. defaults
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        o = {k: v for k, v in (r.get("options") or {}).items()}
        if o.get("blas") in ("auto", "mklcpu", "openblas", "mkl", "rocblas", "cublas", "hipblas", "netlib"):
            o.pop("blas")
        if o or r.get("threads"):
            continue
        if (r["workload"], r["batch"]) not in REF:
            continue
        tab[(r["_machine"], dev(r), r["backend"], toolchain(r["_sweep"], r), str(r.get("blas")))][(r["workload"], r["batch"])].append(
            r["results"]["steady_epoch_s"] * 1e3)
    print("### Default configuration, steady epoch (ms, median over repeats and duplicate sweeps), float\n")
    print("| machine | device | backend | toolchain | BLAS | " + " | ".join(f"{w} b{b}" for w, b in REF) + " |")
    print("|---|---|---|---|---|" + "---|" * len(REF))
    for k in sorted(tab):
        cells = [ms(tab[k][wb]) if tab[k].get(wb) else "-" for wb in REF]
        print("| " + " | ".join(k) + " | " + " | ".join(cells) + " |")
    # 2. tiled vs vendor
    tt = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        o = r.get("options") or {}
        if r.get("threads") or set(o) - {"blas"}:
            continue
        if (r["workload"], r["batch"]) not in REF:
            continue
        kind = "tiled" if o.get("blas") in ("tiled", "omp") else "vendor"
        if o.get("blas") == "omp":
            kind = "omp-loops"
        tt[(r["_machine"], dev(r), r["backend"], toolchain(r["_sweep"], r))][(kind, r["workload"], r["batch"])].append(r["results"]["steady_epoch_s"] * 1e3)
    print("\n### Hand-written tiled BLAS (E7) vs vendor BLAS, steady epoch ms and ratio tiled/vendor\n")
    print("| machine | device | backend | toolchain | " + " | ".join(f"{w} b{b}" for w, b in REF) + " |")
    print("|---|---|---|---|" + "---|" * len(REF))
    for k in sorted(tt):
        cells = []
        for w, b in REF:
            v, t = tt[k].get(("vendor", w, b)), tt[k].get(("tiled", w, b))
            if v and t:
                cells.append(f"{ms(v)} / {ms(t)} ({st.median(t) / st.median(v):.2f}x)")
            elif t:
                cells.append(f"- / {ms(t)}")
            elif v:
                cells.append(f"{ms(v)} / -")
            else:
                cells.append("-")
        print("| " + " | ".join(k) + " | " + " | ".join(cells) + " |")
    # 3. spread across sweeps for identical visible configurations
    sp = collections.defaultdict(dict)
    for r in raw:  # every row, before de-duplication: the spread between sweeps is the point
        if r.get("threads"):
            continue
        o = dict(r.get("options") or {})
        if o.get("blas") == "auto":
            o.pop("blas")
        key = (r["_machine"], dev(r), r["backend"], toolchain(r["_sweep"], r), r["workload"], r["batch"], json.dumps(o, sort_keys=True))
        sp[key].setdefault(r["_sweep"], []).append(r["results"]["steady_epoch_s"] * 1e3)
    print("\n### Same visible configuration measured in more than one sweep (run-to-run spread)\n")
    print("Every configuration with rows in two or more sweeps, worst 25 shown (medians per sweep):\n")
    print("| machine | device | backend | toolchain | workload | options | sweeps | min ms | max ms | spread |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    worst = []
    for k, d in sp.items():
        if len(d) < 2:
            continue
        med = {sw: st.median(v) for sw, v in d.items()}
        lo, hi = min(med.values()), max(med.values())
        worst.append((hi / lo, k, d, lo, hi))
    for ratio, k, d, lo, hi in sorted(worst, key=lambda t: -t[0])[:25]:
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} b{k[5]} | {k[6] if k[6] != '{}' else 'default'} | {', '.join(sorted(d))} | {lo:.1f} | {hi:.1f} | {ratio - 1:+.1%} |")
    if worst:
        spreads = sorted(t[0] - 1 for t in worst)
        print(f"\n{len(worst)} configurations measured in more than one sweep; median spread {st.median(spreads):.1%}, 90th percentile {spreads[int(0.9 * (len(spreads) - 1))]:.1%}, max {spreads[-1]:.1%}.")
        big = [t for t in worst if t[1][4] in ("monk", "cup")]
        small = [t for t in worst if t[1][4] not in ("monk", "cup")]
        for name, group in (("monk/cup (launch-bound, sub-10 ms epochs)", big), ("other workloads", small)):
            if group:
                g = sorted(t[0] - 1 for t in group)
                print(f"- {name}: {len(g)} configurations, median {st.median(g):.1%}, max {g[-1]:.1%}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/ws-amd/2026-09-05"])
