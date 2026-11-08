#!/usr/bin/env python3
"""Nsight Systems GPU-metrics capture of one train call (scripts/nsys_epoch.py, NVTX range fnn.train.capture):
means over the compute part, from the first sample with GR Active above 50 % (after the upload) to the end
of the range. Writes gpu-metrics-summary.json next to the databases.
    python3 scripts/gpu-metrics-summary.py results/hpc-a30/stream-explain-2026-11-08"""
import json, sqlite3, statistics as st, sys
from pathlib import Path
d = Path(sys.argv[1]); out = {}
for db in sorted(d.glob('gm-*.sqlite')):
    c = sqlite3.connect(db)
    ids = {v: k for k, v in c.execute('select metricId, metricName from TARGET_INFO_GPU_METRICS')}
    s, e = c.execute("select start, end from NVTX_EVENTS where text like 'fnn.train.capture%'").fetchone()
    def series(name):
        return c.execute('select timestamp, value from GPU_METRICS where metricId=? and timestamp between ? and ? order by timestamp', (ids[name], s, e)).fetchall()
    gr = series('GR Active [Throughput %]')
    t0 = next(t for t, v in gr if v > 50)
    def mean(name): return st.mean(v for t, v in series(name) if t >= t0)
    out[db.stem[3:]] = {'capture_ms': (e - s) / 1e6, 'compute_ms': (e - t0) / 1e6, 'samples': sum(1 for t, _ in gr if t >= t0),
                        'gr_active_pct': mean('GR Active [Throughput %]'), 'sms_active_pct': mean('SMs Active [Throughput %]'),
                        'sm_issue_pct': mean('SM Issue [Throughput %]'),
                        'compute_warps_per_cycle': mean('Compute Warps in Flight [Avg Warps per Cycle]')}
(d/'gpu-metrics-summary.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
