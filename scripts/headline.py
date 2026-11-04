#!/usr/bin/env python3
"""Headline medians of independent live rows, with configuration and sweep counts.

    PYTHONPATH=bench python scripts/headline.py results/hpc-a30/2026-09-17

Superseded rows are archived explicitly; timestamps never silently discard a
valid repeated sweep. Exact duplicate records are counted once. Different
protocols, builds, loaded libraries, environments or effective options are kept
separate. The configuration digest links each entry to its full conditions.
"""
from __future__ import annotations

import collections
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from fnnbench import results as R  # noqa: E402

REF = {("monk", 40), ("cup", 40), ("mnist-512-256", 256), ("mnist-512-256", 1024)}


def load(dirs):
    rows = []
    for directory in dirs:
        root = Path(directory)
        for row in R.read([root]):
            if not R.valid(row):
                continue
            row["_machine"] = root.parent.name
            row["_sweep"] = Path(row["_source"]).parent.name
            rows.append(row)
    return R.dedupe(rows)


def toolchain(row):
    build = row.get("build_info") or {}
    return build.get("sycl_implementation") or build.get("compiler") or "not recorded"


def dev(row):
    name = (row.get("device") or {}).get("name", "?")
    if name == "AdaptiveCpp OpenMP host device":
        name = ((row.get("sysinfo") or {}).get("cpu") or {}).get("model", "host CPU") + " (AdaptiveCpp host)"
    for old, new in (("AMD Ryzen Threadripper 2950X 16-Core Processor", "TR 2950X"),
                     ("Intel(R) Xeon(R) CPU E5-2643 v2 @ 3.50GHz", "Xeon E5-2643v2"),
                     ("NVIDIA GeForce GTX 1080 Ti", "GTX 1080 Ti"), ("AMD Radeon RX 7900 XTX", "RX 7900 XTX")):
        name = name.replace(old, new)
    return name.strip()


def tag(row):
    return hashlib.sha256(R.configuration_key(row).encode()).hexdigest()[:10]


def groups(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[R.configuration_key(row)].append(row)
    return sorted(grouped.values(), key=lambda rs: (rs[0]["_machine"], dev(rs[0]), rs[0]["backend"],
                                                   toolchain(rs[0]), rs[0]["workload"], rs[0]["batch"], tag(rs[0])))


def default(row):
    opts = {k: v for k, v in (row.get("options") or {}).items() if k != "blas" and not (k == "profile" and not v)}
    return not opts and not row.get("threads")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in rows:
        print("| " + " | ".join(cell(v) for v in row) + " |")


def median_ms(values):
    values = [v for v in values if v is not None]
    return f"{st.median(values) * 1e3:.3f}" if values else "not recorded"


def timing_table(selected, infer=False):
    output = []
    for rs in groups(selected):
        r = rs[0]
        times = [R.epoch_s(x) for x in rs]
        within = [x["results"].get("iqr_epoch_s" if infer else "steady_iqr_s") for x in rs]
        internal = [x["results"].get("steady_epoch_s") for x in rs]
        output.append([tag(r), r["_machine"], dev(r), r["backend"], toolchain(r), r.get("blas"),
                       (r.get("build_info") or {}).get("version", "?"), f"{r['workload']} b{r['batch']} {r['dtype']}",
                       r["n_samples"], f"{r.get('epochs')} / {r.get('repeat')}", len(rs),
                       len({x["_sweep"] for x in rs}), median_ms(times),
                       f"{min(times)*1e3:.3f}–{max(times)*1e3:.3f}", median_ms(within)]
                      + ([median_ms(internal)] if infer else []))
    table(["config", "machine", "device", "backend", "toolchain", "BLAS", "version", "workload", "N",
           "epochs / repeats", "runs", "sweeps", "whole predict ms" if infer else "steady epoch ms",
           "process range ms", "median within-process call IQR ms" if infer else "median within-process steady IQR ms"]
          + (["library internal predict ms (secondary)"] if infer else []), output)


def main(dirs):
    rows = load(dirs)
    rows = [r for r in rows if not (r.get("effective_options") or r.get("options") or {}).get("profile")]
    train = [r for r in rows if r.get("mode", "train") == "train" and r["dtype"] == "float"
             and (r["workload"], r["batch"]) in REF and default(r)]
    infer = [r for r in rows if r.get("mode") == "infer" and not r.get("threads")]
    print("# Headline timings\n")
    print("Each entry is the median of the per-process medians of valid live rows; runs and distinct sweep "
          "counts are shown. Identical copies count once. Re-measurements replace rows only through explicit "
          "supersession (the archived superseded.jsonl is excluded), never by choosing the latest timestamp. "
          "Configuration groups retain effective options, library versions, compiler/flags, loaded libraries, "
          "thread/NUMA conditions and the epochs/repeats/warm-up protocol. check.ok=false, nonfinite losses "
          "and invalid timing rows use the same rejection rule as plots.\n")
    print("Training uses the median of steady_epochs_s when available, then steady_epoch_s, with call-level "
          "time as a labelled fallback for old rows. Within-process IQR is not between-process uncertainty; "
          "old rows without steady_iqr_s are marked not recorded. Inference uses the whole predict call.\n")
    print("## Default / vendor training, float\n")
    timing_table([r for r in train if str(r.get("blas")) not in ("tiled", "handwritten", "omp")])
    print("\n## Hand-written BLAS training, float\n")
    timing_table([r for r in train if str(r.get("blas")) in ("tiled", "handwritten", "omp")])
    print("\n## Inference (whole-call time; internal clock shown separately)\n")
    timing_table(infer, infer=True)
    print("\n## Repeated sweeps of the same complete configuration\n")
    spread = []
    for rs in groups([r for r in rows if r.get("mode", "train") == "train"]):
        sweeps = collections.defaultdict(list)
        for r in rs:
            sweeps[r["_sweep"]].append(R.epoch_s(r))
        if len(sweeps) < 2:
            continue
        medians = {s: st.median(v) for s, v in sweeps.items()}
        lo, hi = min(medians.values()), max(medians.values())
        spread.append((hi / lo - 1, [tag(rs[0]), len(rs), len(sweeps), f"{lo*1e3:.3f}", f"{hi*1e3:.3f}",
                                        f"{hi/lo-1:.1%}", ", ".join(sorted(sweeps))]))
    table(["config", "runs", "sweeps", "min sweep median ms", "max sweep median ms", "spread", "directories"],
          [values for _, values in sorted(spread, reverse=True)[:25]])
    print("\n## Configuration provenance\n")
    table(["config", "time source", "effective options", "compiler flags", "source rows"],
          [[tag(rs[0]), "whole predict call" if rs[0].get("mode") == "infer" else
            ("steady" if all(x["results"].get("steady_epoch_s") or x["results"].get("steady_epochs_s") for x in rs)
             else "call-level fallback present"),
            json.dumps(rs[0].get("effective_options") or rs[0].get("options") or {}, sort_keys=True),
            (rs[0].get("build_info") or {}).get("flags", "not recorded"),
            ", ".join(f"{x['_source']}:{x.get('_line', '?')}" for x in rs)] for rs in groups(train + infer)])


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/ws-amd/2026-09-05"])
