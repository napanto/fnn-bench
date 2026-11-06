#!/usr/bin/env python3
"""Summarise scripts/ws-amd-cup-interference.sh: steady-state epoch of every fresh process, by backend and variant."""
import json, sys, statistics as st, collections
from pathlib import Path
d = Path(sys.argv[1]); g = collections.defaultdict(list); bad = []
for f in sorted(d.glob('*/*/round-*/*.jsonl')):
    v, be, rnd = f.parts[-4], f.parts[-3], f.parts[-2]
    for line in f.read_text().splitlines():
        r = json.loads(line); res = r.get('results') or {}
        if (r.get('check') or {}).get('ok') is True and res.get('steady_epoch_s'):
            g[(be, v)].append((int(rnd.split('-')[1]), res['steady_epoch_s']*1e3, res.get('iqr_epoch_s', 0)*1e3, r['timestamp']))
        else:
            bad.append(str(f))
out = {}
for (be, v), xs in sorted(g.items()):
    t = [x[1] for x in xs]; med = st.median(t)
    out[f'{be}/{v}'] = dict(n=len(t), median_ms=med, min_ms=min(t), max_ms=max(t), spread_pct=100*(max(t)/min(t)-1),
                            within_iqr_ms=st.median(x[2] for x in xs), rows=[round(x, 3) for x in t])
    print(f"{be:6s} {v:14s} n={len(t)} med {med:7.2f} min {min(t):7.2f} max {max(t):7.2f} spread {100*(max(t)/min(t)-1):5.1f} % within-row IQR {st.median(x[2] for x in xs):5.2f}  {[round(x,2) for x in t]}")
(d/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
print('invalid rows', len(bad), bad[:3])
