#!/usr/bin/env python3
"""Split the samples that pass through cuBLAS by the outermost cuBLAS entry point on the stack, by its
caller library, and by whether the sample is inside the CUDA driver below it or in cuBLAS's own code.
Scaled to ms per epoch with each path's warm-call time over its sampled epochs, as wallprof_layers.py
does for the inclusive figures. Writes cublas-entries.json into the result directory.
    python3 scripts/wallprof/cublas_entries.py results/hpc-a30/explain-e-2026-11-08"""
import collections, json, re, statistics as st, sys
from pathlib import Path
d = Path(sys.argv[1]); out = {}
tags = sorted({re.sub(r'-r\d+\.stacks\.json$', '', p.name) for p in d.glob('*-r*.stacks.json')})
for tag in tags:
    C = collections.Counter(); N = 0; ep = []
    for f in sorted(d.glob(f'{tag}-r*.stacks.json')):
        o = json.loads(f.read_text()); info = json.loads(f.with_name(f.name.replace('.stacks.json', '.json')).read_text())
        N += o['n']; ep.append(500*(info['before_s'] + info['after_s'])/info['epochs'])
        for n, stk in o['stacks']:
            idx = [i for i, fr in enumerate(stk) if fr.startswith('libcublas')]
            if not idx:
                continue
            entry = stk[max(idx)].split(';', 1)[1].split('@@')[0]
            caller = stk[max(idx) + 1].split(';', 1)[0] if max(idx) + 1 < len(stk) else '?'
            caller = 'oneMath' if 'onemath' in caller else 'cudann' if 'cudann' in caller else caller
            part = 'driver' if any(fr.startswith('libcuda.so') for fr in stk[:min(idx)]) else 'cublas'
            C[f'{entry}|{caller}|{part}'] += n
    e = st.mean(ep)
    out[tag] = {'epoch_ms': e, 'samples': N, 'ms_per_epoch': {k: e*v/N for k, v in sorted(C.items())}}
(d/'cublas-entries.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps({t: {k: round(v, 2) for k, v in o['ms_per_epoch'].items() if v > .5} for t, o in out.items()}, indent=1))
