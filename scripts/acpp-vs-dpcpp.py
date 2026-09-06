#!/usr/bin/env python3
"""DPC++ vs AdaptiveCpp table (docs/optional.md, item 2) from the third-pass rows.

    python scripts/acpp-vs-dpcpp.py results/ws-amd/2026-09-05 results/ws-nvidia/2026-09-05

One line per (device, SYCL implementation): steady epoch in ms for the reference float
training rows with the vendor BLAS (default options) and with the hand-written tiled BLAS,
plus the ratio between the two implementations on the same device. The implementation is
read from build_info.sycl_implementation; the compute unit / thread count from the row.
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


def impl(r):
    s = (r.get("build_info") or {}).get("sycl_implementation") or ""
    return "AdaptiveCpp" if "AdaptiveCpp" in s else "DPC++"


def _cpu(model):
    return model.replace("AMD Ryzen Threadripper 2950X 16-Core Processor", "TR 2950X").replace("Intel(R) Xeon(R) CPU E5-2643 v2 @ 3.50GHz", "Xeon E5-2643 v2").strip()


def devname(r):
    n = r["device"]["name"].strip()
    if n == "AdaptiveCpp OpenMP host device":  # the host CPU of whichever machine ran it
        return _cpu(((r.get("sysinfo") or {}).get("cpu") or {}).get("model") or "host") + ", OpenMP host"
    if r["device"]["type"] == "cpu":
        return _cpu(n) + ", OpenCL CPU"
    for a, b in (("NVIDIA GeForce GTX 1080 Ti", "GTX 1080 Ti"), ("AMD Radeon RX 7900 XTX", "RX 7900 XTX")):
        n = n.replace(a, b)
    return n


def main(dirs):
    rows = []
    for d in dirs:  # de-duplicate per machine: the AdaptiveCpp host device carries the same name on every host
        per = []
        for f in glob.glob(os.path.join(d, "*", "syclnn.jsonl")):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error") and (r.get("results") or {}).get("steady_epoch_s"):
                        per.append(r)
        rows += R.dedupe(per)
    rows = [r for r in rows if r["backend"] == "syclnn" and r.get("mode") == "train" and r["dtype"] == "float" and not r.get("threads")]
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    units = {}
    for r in rows:
        o = r.get("options") or {}
        if set(o) - {"blas"}:
            continue
        kind = "tiled" if o.get("blas") == "tiled" else ("vendor" if o.get("blas") in (None, "auto") else None)
        if kind is None or (r["workload"], r["batch"]) not in REF:
            continue
        key = (devname(r), impl(r))
        tab[key][(kind, r["workload"], r["batch"])].append(r["results"]["steady_epoch_s"] * 1e3)
        cu = r["device"].get("compute_units")
        if r["device"]["type"] == "cpu":
            env = (r.get("sysinfo") or {}).get("env") or {}
            cu = f"{r.get('threads_effective') or cu} threads" if "OpenMP" in key[0] else f"{env.get('DPCPP_CPU_NUM_CUS') or cu} CUs" + (" (device reports %s)" % cu if env.get("DPCPP_CPU_NUM_CUS") else "")
        units[key] = cu
    cols = " | ".join(f"{w} b{b}" for w, b in REF)
    print("| device | SYCL implementation | units | " + cols + " |")
    print("|---|---|---|" + "---|" * len(REF))
    for key in sorted(tab):
        cells = []
        for w, b in REF:
            v, t = tab[key].get(("vendor", w, b)), tab[key].get(("tiled", w, b))
            cells.append((f"{st.median(v):.1f}" if v else "-") + " / " + (f"{st.median(t):.1f}" if t else "-"))
        print(f"| {key[0]} | {key[1]} | {units.get(key, '')} | " + " | ".join(cells) + " |")
    # ratios per device where both implementations exist
    devs = collections.defaultdict(dict)
    for (dev, im), d in tab.items():
        devs[dev.split(",")[0]][im] = d
    lines = []
    for dev, ims in devs.items():
        if len(ims) == 2:
            parts = []
            for w, b in REF:
                for kind in ("vendor", "tiled"):
                    a, d_ = ims["AdaptiveCpp"].get((kind, w, b)), ims["DPC++"].get((kind, w, b))
                    if a and d_:
                        parts.append(f"{w} b{b} {kind}: {st.median(a) / st.median(d_):.2f}x")
            if parts:
                lines.append(f"- {dev}: AdaptiveCpp / DPC++ epoch ratio, " + "; ".join(parts))
    if lines:
        print("\nvendor / tiled cells are steady epochs in ms; ratios above 1 mean AdaptiveCpp is slower:\n")
        print("\n".join(lines))


if __name__ == "__main__":
    args = sys.argv[1:]
    target = None
    if "--update" in args:  # rewrite the table block of a Markdown file between the markers
        i = args.index("--update"); target = args[i + 1]; args = args[:i] + args[i + 2:]
    if target:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(args or ["results/ws-amd/2026-09-05", "results/ws-nvidia/2026-09-05"])
        doc = open(target).read()
        start, end = "<!-- DPCPP-VS-ACPP-TABLE -->", "<!-- /DPCPP-VS-ACPP-TABLE -->"
        body = doc[doc.index(start) + len(start):doc.index(end)]
        a, b = body.index("| device |"), body.index("Same-GPU findings")
        doc = doc[:doc.index(start) + len(start)] + body[:a] + buf.getvalue().strip() + "\n\n" + body[b:] + doc[doc.index(end):]
        open(target, "w").write(doc)
        print(f"updated {target}")
    else:
        main(args or ["results/ws-amd/2026-09-05", "results/ws-nvidia/2026-09-05"])
