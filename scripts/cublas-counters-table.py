#!/usr/bin/env python3
"""Per-library hardware counters of scripts/ws-nvidia-cublas-counters.sh: for each run, cycles per instruction,
iTLB and L1 instruction-cache misses per thousand instructions inside libcublas and libcublasLt.
    python3 scripts/cublas-counters-table.py <result-dir>   (writes counters-summary.json)"""
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1]); out = {}
for f in sorted(d.glob('*-r?.dso.txt')):
    ev, cur = {}, None
    for line in f.read_text().splitlines():
        m = re.match(r"# Samples: .* of event '([^']+)'", line)
        if m: cur = m[1]; ev[cur] = 0; continue
        m = re.match(r'\s+[\d.]+%\s+(\d+)\s+(\S+)', line)
        if m and cur and m[2].startswith('libcublas'): ev[cur] += int(m[1])
    get = lambda key: next(v for k, v in ev.items() if key in k.lower())
    ins = get('instructions')
    out[f.name.replace('.dso.txt', '')] = {'cpi': get('cycles')/ins, 'itlb_mpki': 1e3*get('itlb')/ins, 'l1i_mpki': 1e3*get('icache')/ins}
(d/'counters-summary.json').write_text(json.dumps(out, indent=1) + '\n'); print(len(out), 'runs')
