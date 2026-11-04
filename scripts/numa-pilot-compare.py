#!/usr/bin/env python3
"""Compare the rows of a NUMA pilot (results/<machine>/numa-pilot-<date>/<policy>-r<k>/<env>/*.jsonl).

For each row configuration: the median steady epoch of every policy and round, the policy ratio
against the baseline policy (mean over rounds), and the round-to-round difference inside each policy
(the noise the ratio has to exceed). Markdown on stdout.

    scripts/numa-pilot-compare.py results/ws-amd/numa-pilot-2026-11-03 [--baseline first-touch]
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def key(r):
    o = r.get("options") or {}
    return (r["backend"], r["workload"], r["batch"], r.get("mode", "train"), r["dtype"],
            ",".join(f"{k}={v}" for k, v in sorted(o.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot")
    ap.add_argument("--baseline", default=None, help="policy the others are compared with (default: first-touch or default)")
    a = ap.parse_args()
    root = Path(a.pilot)
    t = defaultdict(lambda: defaultdict(dict))   # (env, key) -> policy -> round -> median
    policies = set()
    for f in sorted(root.glob("*-r*/*/*.jsonl")):
        if f.name == "superseded.jsonl":
            continue
        m = re.fullmatch(r"(.+)-r(\d+)", f.parent.parent.name)
        pol, rnd, env = m.group(1), int(m.group(2)), f.parent.name
        policies.add(pol)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "results" not in r:
                continue
            t[(env, key(r))][pol][rnd] = r["results"]["median_epoch_s"]
    base = a.baseline or ("first-touch" if "first-touch" in policies else "default")
    others = sorted(p for p in policies if p != base)
    print(f"# NUMA pilot {root}\n\nbaseline policy: `{base}`; ratio = policy / baseline (mean of the rounds); "
          "noise = largest round-to-round difference inside one policy\n")
    hdr = ["env", "backend", "workload", "batch", "mode", "options", f"{base} [s]"] + [f"{p} [s]" for p in others] \
        + [f"{p}/{base}" for p in others] + ["noise"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for (env, k), byp in sorted(t.items()):
        def mean(p):
            v = list(byp.get(p, {}).values())
            return statistics.mean(v) if v else None
        def spread(p):
            v = list(byp.get(p, {}).values())
            return (max(v) / min(v) - 1) if len(v) > 1 else None
        b = mean(base)
        cells = [env, *[str(x) for x in k[:4]], k[5] or "-", f"{b:.4f}" if b else "-"]
        cells += [f"{mean(p):.4f}" if mean(p) else "-" for p in others]
        cells += [f"{mean(p) / b:.3f}" if (mean(p) and b) else "-" for p in others]
        noise = [s for s in (spread(p) for p in [base, *others]) if s is not None]
        cells.append(f"{100 * max(noise):.1f} %" if noise else "-")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
