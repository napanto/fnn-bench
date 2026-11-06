#!/usr/bin/env python3
"""Align the per-call epoch times of a long A30 run with the logged GPU clock."""
import datetime, json, statistics as st, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = []
for line in (d/'clocks.csv').read_text().splitlines():
    p = [x.strip() for x in line.split(',')]
    try:
        rows.append((datetime.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f').timestamp(), int(p[1].split()[0]), int(p[2].split()[0]), int(p[3].split()[0]), float(p[4].split()[0])))
    except (ValueError, IndexError):
        pass
out = {}
for tag in ('cudann', 'syclnn', 'nvcpp'):
    f = d/f'ramp-{tag}.json'
    if not f.exists():
        continue
    calls = json.loads(f.read_text())['calls']; t0 = calls[0][0]; series = []
    for a, b in calls:
        clk = [r[1] for r in rows if a <= r[0] <= b]
        series.append(dict(t=round(a-t0, 2), epoch_ms=round((b-a)/3*1e3, 2), clock_mhz=max(clk) if clk else None))
    out[tag] = series
    print(tag, len(series), 'calls; first', series[:6], '... last', series[-2:])
    late = [s['epoch_ms'] for s in series if s['t'] > 6]
    print('  median epoch after 6 s', round(st.median(late), 1) if late else None)
(d/'ramp-summary.json').write_text(json.dumps(out)+'\n')
print('clock histogram', {c: sum(1 for r in rows if r[1] == c) for c in sorted({r[1] for r in rows})})
