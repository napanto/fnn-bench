#!/usr/bin/env python3
"""Summarise the launch-cost probes: median microseconds per operation over process rounds, first repeat dropped."""
import csv, glob, collections, json, statistics as st, sys
from pathlib import Path
d = Path(sys.argv[1]); g = collections.defaultdict(list)
for f in sorted(d.glob('**/*.csv')):
    size = next((p for p in f.stem.split('-') if p in ('m64', 'm256')), None)
    for row in csv.reader(f.open()):
        if len(row) == 6:
            m, c, r, n, l, t = row; g[((size+'/') if size else '')+m, c].append((int(r), int(n), int(l), int(t)))
out = {}
for (m, c), v in g.items():
    h = [x[2]/x[1]/1e3 for x in v if x[0] >= 1]; t = [x[3]/x[1]/1e3 for x in v if x[0] >= 1]
    out[f'{m}/{c}'] = dict(repeats=len(h), host_us=st.median(h), total_us=st.median(t), min_us=min(t), max_us=max(t))
    print(f"{m:8s} {c:26s} {len(h):3d} host {st.median(h):9.2f} total {st.median(t):9.2f}  [{min(t):.2f}, {max(t):.2f}]")
(d/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
