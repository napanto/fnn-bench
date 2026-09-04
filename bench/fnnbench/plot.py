"""`fnnbench plot`: the standard figures of the report from the JSONL results.

    fnnbench plot results/ws-amd/2026-09-03 -o analysis/figures

Figures (one PNG + one PDF each, only when the data for them exists):
* throughput   samples/s vs batch, one line per backend/device, per workload
* breakdown    stacked per-phase device time per epoch (profile) per configuration
* roofline     achieved GEMM GFLOP/s vs arithmetic intensity against the measured peaks
* ablation     syclnn/cudann ablation switches vs the default (relative epoch time)
* compilers    ompnn compiler x workload heat map (epoch time)
* precision    float vs double epoch time per backend/device
* scaling      CPU thread scaling (samples/s vs threads)
* sweep        W4 width/depth/batch sweep (GFLOP/s vs batch, one line per width/depth)

Colours follow a fixed backend palette so that every figure reads the same way.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import results

BACKEND_COLOR = {"syclnn": "#1f77b4", "cudann": "#2ca02c", "ompnn": "#d62728", "numpy": "#7f7f7f"}
PHASES = ("gemm", "act", "delta", "biasgrad", "update", "loss", "reg", "h2d", "d2h", "other")
PHASE_COLOR = {"gemm": "#4c72b0", "act": "#dd8452", "delta": "#55a868", "biasgrad": "#c44e52", "update": "#8172b3",
               "loss": "#937860", "reg": "#da8bc3", "h2d": "#8c8c8c", "d2h": "#ccb974", "other": "#64b5cd"}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "legend.fontsize": 8, "axes.titlesize": 10})
    return plt


METRIC = "steady"  # "steady": steady_epoch_s when the row has it, else median_epoch_s; "median": always the call-level number


def _epoch_s(row: dict[str, Any]) -> float:
    r = row["results"]
    if METRIC == "steady" and r.get("steady_epoch_s"):
        return float(r["steady_epoch_s"])
    return float(r["median_epoch_s"])



def _samples_per_s(row: dict[str, Any]) -> float:
    return row["n_samples"] / _epoch_s(row)


def _gflops_gemm(row: dict[str, Any]) -> float:
    return row["results"]["flops_per_epoch"]["gemm"] / _epoch_s(row) / 1e9


def _label(row: dict[str, Any]) -> str:
    dev = (row.get("device") or {}).get("name", "?")
    short = dev.split("(")[0].strip()
    short = short.replace("AMD Ryzen Threadripper 2950X 16-Core Processor", "TR 2950X").replace("AMD Radeon RX 7900 XTX", "RX 7900 XTX")
    comp = (row.get("build_info") or {}).get("compiler", "")
    for k in ("AdaptiveCpp", "Clang 22", "GNU 14", "Clang 18", "nvcc", "hipcc"):
        if k in comp:
            comp = {"AdaptiveCpp": "acpp", "Clang 22": "clang-22", "GNU 14": "gcc-14", "Clang 18": "clang-18", "nvcc": "nvcc", "hipcc": "hipcc"}[k]
            break
    else:
        comp = comp.split(" ")[0] if comp else ""
    blas = row.get("blas", "")
    return f"{row['backend']}/{comp}/{blas} @ {short}"


def _opt_label(opts_json: str) -> str:
    """'{"memory": "shared", "streams": 4}' -> 'memory=shared, streams=4'"""
    d = json.loads(opts_json)
    if len(d) >= 6:
        return "all switches off (0.1 behaviour)"
    return ", ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in d.items()) or "default"


def _save(plt, fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out / name}.png")


def _good(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if "results" in r and (r.get("check") or {}).get("ok", True) and r["results"].get("losses_finite", True)]


def plot_throughput(plt, rows, out):
    by_wl = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("mode", "train") != "train" or r.get("threads"):
            continue
        by_wl[(r["workload"], r["dtype"])][_label(r)].append((r["batch"], _samples_per_s(r)))
    for (wl, dt), series in by_wl.items():
        if all(len(v) < 2 for v in series.values()):
            continue
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        markers = {"acpp": "o", "clang-22": "s", "clang-18": "^", "gcc-14": "v", "nvcc": "D", "hipcc": "P", "icpx": "X"}
        styles = {"tiled": "--", "omp": ":", "handwritten": "--"}
        for label, pts in sorted(series.items()):
            pts.sort()
            parts = label.split("@")[0].strip().split("/")
            comp = parts[1] if len(parts) > 1 else ""
            blas = parts[2] if len(parts) > 2 else ""
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=markers.get(comp, "o"), ms=4,
                    linestyle=styles.get(blas, "-"), label=label, color=BACKEND_COLOR.get(parts[0]))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("samples / s")
        ax.set_title(f"training throughput - {wl} ({dt})")
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        _save(plt, fig, out, f"throughput-{wl}-{dt}")


def plot_breakdown(plt, rows, out):
    groups = defaultdict(list)
    for r in rows:
        per = r["results"].get("profile_per_epoch_ms")
        if not per or r.get("threads"):
            continue
        groups[(r["workload"], r["dtype"], r.get("mode", "train"))].append(r)
    for (wl, dt, mode), rs in groups.items():
        rs = sorted(rs, key=lambda r: (_label(r), r["batch"]))
        labels = [f"{_label(r)}\nb={r['batch']} {json.dumps({k: v for k, v in r.get('options', {}).items() if k != 'profile'}) if r.get('options') else ''}".strip() for r in rs]
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rs) + 2), 4.2))
        bottom = [0.0] * len(rs)
        for ph in PHASES:
            vals = [r["results"]["profile_per_epoch_ms"].get(f"{ph}_ns", 0.0) for r in rs]
            if not any(vals):
                continue
            ax.bar(range(len(rs)), vals, bottom=bottom, label=ph, color=PHASE_COLOR[ph])
            bottom = [b + v for b, v in zip(bottom, vals)]
        walls = [_epoch_s(r) * 1e3 for r in rs]
        ax.plot(range(len(rs)), walls, "k_", markersize=14, label="wall / epoch")
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("ms per epoch")
        ax.set_title(f"per-phase device time - {wl} ({dt}, {mode})")
        ax.legend(ncol=2)
        _save(plt, fig, out, f"breakdown-{wl}-{dt}-{mode}")


def plot_roofline(plt, rows, peaks, out):
    if not peaks:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    dev_peaks = defaultdict(dict)
    for p in peaks:
        name = (p.get("device") or {}).get("name", "?")
        dev_peaks[name][p["dtype"]] = (p.get("gemm_gflops"), p.get("act_bandwidth_gbs"))
    xs = [0.1, 1000.0]
    for name, byd in dev_peaks.items():
        for dt, (gf, bw) in byd.items():
            if not gf or not bw:
                continue
            ax.plot([x for x in xs], [min(gf, x * bw) for x in xs], "--", lw=1, label=f"peak {name.split('(')[0].strip()[:24]} {dt}")
    for r in rows:
        if r.get("mode", "train") != "train" or r.get("threads"):
            continue
        ax.scatter(r["results"]["intensity_flop_per_byte"], _gflops_gemm(r), s=14,
                   color=BACKEND_COLOR.get(r["backend"]), marker="o" if r["dtype"] == "float" else "s", alpha=0.7)
    for be, c in BACKEND_COLOR.items():
        ax.scatter([], [], color=c, label=be)
    ax.scatter([], [], color="k", marker="o", label="float")
    ax.scatter([], [], color="k", marker="s", label="double")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity of the step's GEMMs [FLOP/byte]")
    ax.set_ylabel("achieved GEMM GFLOP/s (per epoch wall)")
    ax.set_title("roofline placement of every training row")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _save(plt, fig, out, "roofline")


def plot_ablation(plt, rows, out):
    for backend in ("syclnn", "cudann"):
        groups = defaultdict(dict)
        for r in rows:
            if r["backend"] != backend or r.get("threads"):
                continue
            key = (_label(r), r["workload"], r["dtype"], r["batch"])
            opts = {k: v for k, v in r.get("options", {}).items() if k != "profile"}
            groups[key][json.dumps(opts, sort_keys=True)] = _epoch_s(r)
        for key, variants in groups.items():
            base = variants.get("{}")
            if not base or len(variants) < 3:
                continue
            names = [k for k in variants if k != "{}"]
            rel = [variants[k] / base for k in names]
            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(names) + 2), 3.8))
            ax.bar(range(len(names)), rel, color=BACKEND_COLOR[backend])
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([_opt_label(n) for n in names], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("epoch time / default")
            ax.set_title(f"{backend} ablations - {key[1]} ({key[2]}, b={key[3]}) on {key[0].split('@')[1].strip()}")
            _save(plt, fig, out, f"ablation-{backend}-{key[1]}-{key[2]}-{key[0].split('@')[1].strip().replace(' ', '')}")


def plot_compilers(plt, rows, out):
    grid = defaultdict(dict)
    for r in rows:
        if r["backend"] != "ompnn" or r.get("threads") or r.get("mode", "train") != "train":
            continue
        opts = {k: v for k, v in r.get("options", {}).items() if k != "profile"}
        if opts:
            continue
        col = f"{r['workload']}\n{r['dtype']} b={r['batch']}"
        grid[_label(r)][col] = _epoch_s(r) * 1e3
    if not grid:
        return
    cols = sorted({c for v in grid.values() for c in v})
    labels = sorted(grid)
    import numpy as np

    data = np.full((len(labels), len(cols)), np.nan)
    for i, l in enumerate(labels):
        for j, c in enumerate(cols):
            data[i, j] = grid[l].get(c, np.nan)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(cols)), max(3, 0.45 * len(labels) + 1.5)))
    im = ax.imshow(np.log10(data), cmap="viridis", aspect="auto")
    for i in range(len(labels)):
        for j in range(len(cols)):
            if not math.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6, color="w")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="log10(ms per epoch)")
    ax.set_title("ompnn: compiler x workload (ms per epoch)")
    _save(plt, fig, out, "compilers-ompnn")


def plot_precision(plt, rows, out):
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("threads") or r.get("mode", "train") != "train":
            continue
        opts = {k: v for k, v in r.get("options", {}).items() if k != "profile"}
        pairs[(_label(r), r["workload"], r["batch"], json.dumps(opts, sort_keys=True))][r["dtype"]] = _epoch_s(r)
    items = [(k, v) for k, v in pairs.items() if "float" in v and "double" in v]
    if not items:
        return
    items.sort(key=lambda kv: (kv[0][1], kv[0][0]))
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(items) + 2), 3.8))
    ax.bar(range(len(items)), [v["double"] / v["float"] for _, v in items],
           color=[BACKEND_COLOR.get(k[0].split("/")[0]) for k, _ in items])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{k[0]} {k[1]} b={k[2]}" for k, _ in items], rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("double / float epoch time")
    ax.set_yscale("log")
    ax.set_title("cost of double precision")
    _save(plt, fig, out, "precision")


def plot_scaling(plt, rows, out):
    series = defaultdict(list)
    for r in rows:
        if not r.get("threads"):
            continue
        series[(_label(r), r["workload"], r["dtype"], r["batch"])].append((r["threads"], _samples_per_s(r)))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, pts in sorted(series.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{key[0]} {key[1]} {key[2]}",
                color=BACKEND_COLOR.get(key[0].split("/")[0]))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads")
    ax.set_ylabel("samples / s")
    ax.set_title("CPU thread scaling")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _save(plt, fig, out, "scaling-threads")



def plot_tiled(plt, rows, out):
    """E7: hand-written tiled GEMM vs the vendor / oneMath BLAS, per backend+compiler and device."""
    pairs = defaultdict(dict)  # (device, workload, dtype, batch) -> {(series, tiled|library): epoch_s}
    for r in rows:
        if r.get("mode", "train") != "train" or r.get("threads"):
            continue
        opts = {k: v for k, v in r.get("options", {}).items() if k != "profile"}
        blas = opts.pop("blas", None)
        if opts:  # ablation rows are not part of E7
            continue
        lab = _label(r)
        dev = lab.split("@")[1].strip()
        series = "/".join(lab.split("@")[0].strip().split("/")[:2])  # backend/compiler
        pairs[(dev, r["workload"], r["dtype"], r["batch"])][(series, "tiled" if blas in ("tiled", "handwritten") else "library")] = _epoch_s(r)
    by_dev = defaultdict(list)
    for key, variants in pairs.items():
        for series in sorted({k[0] for k in variants}):
            if (series, "tiled") in variants and (series, "library") in variants:
                by_dev[key[0]].append((f"{key[1]} {key[2]} b{key[3]}", series, variants[(series, "tiled")] / variants[(series, "library")]))
    hatches = ["", "//", "..", "xx"]
    for dev, items in by_dev.items():
        labels = sorted({it[0] for it in items})
        series_all = sorted({it[1] for it in items}, key=lambda x: (["syclnn", "cudann", "ompnn"].index(x.split("/")[0]), x))
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(labels) + 3), 4.0))
        width = 0.8 / max(1, len(series_all))
        per_backend = defaultdict(int)
        for si, series in enumerate(series_all):
            be = series.split("/")[0]
            vals = {it[0]: it[2] for it in items if it[1] == series}
            xs = [i + (si - (len(series_all) - 1) / 2) * width for i, l in enumerate(labels) if l in vals]
            ax.bar(xs, [vals[l] for l in labels if l in vals], width, label=series, color=BACKEND_COLOR[be],
                   hatch=hatches[per_backend[be] % len(hatches)], edgecolor="white", linewidth=0.5)
            per_backend[be] += 1
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("epoch time: tiled / library BLAS")
        ax.set_yscale("log")
        ax.set_title(f"hand-written 16x16 tiled GEMM vs library on {dev}")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        _save(plt, fig, out, f"tiled-{dev.replace(' ', '')}")


def plot_sweep(plt, rows, out):
    series = defaultdict(list)
    for r in rows:
        if not r["workload"].startswith("sweep-") or r.get("mode", "train") != "train":
            continue
        _, w, d, _ = r["workload"].split("-")
        series[(_label(r), r["dtype"], w, d)].append((r["batch"], _gflops_gemm(r)))
    if not series:
        return
    by_dev = defaultdict(dict)
    for k, v in series.items():
        by_dev[(k[0], k[1])][(k[2], k[3])] = sorted(v)
    for (label, dt), sub in by_dev.items():
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for (w, d), pts in sorted(sub.items()):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{w} x {d}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("GEMM GFLOP/s")
        ax.set_title(f"W4 sweep - {label} ({dt})")
        ax.legend(fontsize=6, ncol=3, title="width x depth")
        _save(plt, fig, out, f"sweep-{label.split('/')[0]}-{label.split('@')[1].strip().replace(' ', '')}-{dt}")


def plot_all(paths: list[str], out: str) -> int:
    plt = _mpl()
    rows = _good(results.read(paths))
    peaks = [r for r in results.read(paths) if r.get("kind") == "peak"]
    outp = Path(out)
    print(f"{len(rows)} usable rows, {len(peaks)} peak rows")
    if not rows:
        return 1
    plot_throughput(plt, rows, outp)
    plot_breakdown(plt, rows, outp)
    plot_roofline(plt, rows, peaks, outp)
    plot_ablation(plt, rows, outp)
    plot_compilers(plt, rows, outp)
    plot_precision(plt, rows, outp)
    plot_scaling(plt, rows, outp)
    plot_sweep(plt, rows, outp)
    plot_tiled(plt, rows, outp)
    return 0
