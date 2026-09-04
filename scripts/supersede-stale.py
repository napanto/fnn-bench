#!/usr/bin/env python3
"""Move result rows measured with an older harness to superseded.jsonl so that the
next `fnnbench sweep` re-measures them (rows already present are skipped).

A row is stale when: its numerical check failed; it has an error; it is a
training row with epochs > 1 and no `steady_epoch_s`; it is a GPU row without
`gpu_monitor`; or its backend is listed in --backend (library changed).

    scripts/supersede-stale.py results/ws-amd/2026-09-03 [--backend cudann] [--dry-run]
"""
import argparse
import json
from pathlib import Path


def stale(r: dict, backends: set[str]) -> str | None:
    if "results" not in r:
        return "error row"
    chk = r.get("check") or {}
    if chk.get("enabled") and chk.get("ok") is False:
        return "check failed"
    if r.get("backend") in backends:
        return "backend re-measured"
    res = r["results"]
    if r.get("mode", "train") == "train" and (r.get("epochs") or 1) > 1 and not res.get("steady_epoch_s"):
        return "no steady_epoch_s"
    dev = (r.get("device") or {}).get("type") or r.get("device_selector")
    timed = (res.get("median_epoch_s") or 0) * (r.get("epochs") or 1) * (r.get("repeat") or 1)
    if dev and dev != "cpu" and r.get("backend") != "numpy" and not res.get("gpu_monitor") and timed > 2.0:
        return "no gpu_monitor"  # rows shorter than the 200 ms sampling interval legitimately have none
    if r.get("mode") == "infer" and r.get("batch") == 1 and (r.get("n_samples") or 0) > 2048:
        return "batch-1 inference on the full set (plans now use 2048 samples)"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--backend", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    total = 0
    for f in sorted(Path(a.root).rglob("*.jsonl")):
        if f.name in ("superseded.jsonl", "peak.jsonl"):  # peak probes have their own row schema
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        keep, drop = [], []
        for r in rows:
            why = stale(r, set(a.backend))
            (drop if why else keep).append((r, why))
        if not drop:
            continue
        reasons = {}
        for _, why in drop:
            reasons[why] = reasons.get(why, 0) + 1
        print(f"{f}: keep {len(keep)}, supersede {len(drop)} {reasons}")
        total += len(drop)
        if a.dry_run:
            continue
        with (f.parent / "superseded.jsonl").open("a") as fh:
            for r, _ in drop:
                fh.write(json.dumps(r) + "\n")
        f.write_text("".join(json.dumps(r) + "\n" for r, _ in keep))
    print(f"{'would supersede' if a.dry_run else 'superseded'} {total} rows")


if __name__ == "__main__":
    main()
