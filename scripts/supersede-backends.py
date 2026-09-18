#!/usr/bin/env python3
"""Move every row of the given backends to superseded.jsonl (a library default changed and the
backend's rows are being re-measured).
    scripts/supersede-backends.py results/<machine>/<date> syclnn cudann [--skip portable-acpp ...] [--dry-run]
Peak probes (peak.jsonl) and the directories named with --skip are left alone."""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("backends", nargs="+")
ap.add_argument("--skip", action="append", default=[], help="sweep directory to leave untouched (repeatable)")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
total = 0
for f in sorted(Path(a.root).rglob("*.jsonl")):
    if f.name in ("superseded.jsonl", "peak.jsonl") or any(part in a.skip for part in f.parts):
        continue
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    drop = [r for r in rows if r.get("backend") in a.backends]
    keep = [r for r in rows if r.get("backend") not in a.backends]
    if not drop:
        continue
    total += len(drop)
    print(f"{f}: keep {len(keep)}, supersede {len(drop)}")
    if a.dry_run:
        continue
    with (f.parent / "superseded.jsonl").open("a") as fh:
        for r in drop:
            fh.write(json.dumps(r) + "\n")
    f.write_text("".join(json.dumps(r) + "\n" for r in keep))
print(f"{'would supersede' if a.dry_run else 'superseded'} {total} rows")
