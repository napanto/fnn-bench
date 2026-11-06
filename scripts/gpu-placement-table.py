#!/usr/bin/env python3
"""Summarise the A30 placement rows: steady-state epoch by NUMA node of the submitting thread."""
import json, sys, statistics as st, collections
from pathlib import Path
d = Path(sys.argv[1]); g = collections.defaultdict(dict); psr = collections.defaultdict(list)
for f in sorted(d.glob('*/*/round-*/*.jsonl')):
    place, cfg, rnd = f.parts[-4], f.parts[-3], f.parts[-2]
    for line in f.read_text().splitlines():
        r = json.loads(line); res = r.get('results') or {}
        if (r.get('check') or {}).get('ok') is True and res.get('steady_epoch_s'):
            g[cfg].setdefault(place, []).append(res['steady_epoch_s']*1e3)
    p = f.parent/'psr.txt'
    if place.startswith('unbound') and p.exists():
        vals = [int(x) for x in p.read_text().split()]
        if vals:
            psr[cfg].append(collections.Counter(v % 8 for v in vals).most_common(1)[0][0])
order = ['unbound', 'node4', 'node5', 'node6', 'node7', 'node0', 'node1', 'node2', 'node3', 'unbound2', 'core12', 'core12,20', 'core8']
print('config        ' + ' '.join(f'{p:>16s}' for p in order))
out = {}
for cfg, by in sorted(g.items()):
    print(f'{cfg:13s} ' + ' '.join((f"{st.median(by[p]):6.2f} [{min(by[p]):.2f},{max(by[p]):.2f}]" if p in by else '-').rjust(16) for p in order), 'nodes', psr.get(cfg))
    out[cfg] = {p: dict(median=st.median(v), rows=[round(x, 3) for x in v]) for p, v in by.items()}
(d/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
