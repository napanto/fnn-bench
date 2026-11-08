#!/usr/bin/env python3
"""Summarise scripts/sched/wake3.bt for the Radeon processes of scripts/ws-amd-radeon-wake.sh: per process, its
steady epoch and, for the thread with the most wake-ups, the wake-up count and mean latency (us).
    python3 scripts/sched/wake3-table.py <result-dir>   (reads wake3.txt, writes wake-summary.json)"""
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1]); txt = (d/'wake3.txt').read_text(); out = {}
lat = {}
for m in re.finditer(r'@lat_us\[v-mn-r(\d+), (\d+)\]: \{ .count = (\d+), .average = (\d+), .total = (\d+) \}', txt):
    lat.setdefault(int(m[1]), []).append((int(m[3]), int(m[4]), int(m[2])))
for k, threads in sorted(lat.items()):
    n, avg, tid = max(threads)
    row = json.loads(next((d/f'round-{k}').glob('*.jsonl')).read_text().splitlines()[0])
    out[f'round-{k}'] = {'steady_epoch_ms': row['results']['steady_epoch_s']*1e3, 'top_thread_wakeups': n, 'top_thread_mean_latency_us': avg}
(d/'wake-summary.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out)[:400])
