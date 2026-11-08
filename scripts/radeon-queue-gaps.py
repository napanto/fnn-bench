#!/usr/bin/env python3
"""rocprofv3 kernel traces of HIP MNIST processes (scripts/ws-amd-radeon-modes.sh, variant trace): for the
second half of the dispatches (the steady calls) the summed kernel time, the busy union, and every gap
between the end of the work so far and the next kernel start, split by whether the next kernel runs on the
same HSA queue as the kernel that ended last or on another one. Gaps above 2 ms (host pauses between calls)
are left out. Writes trace-summary.json into the result directory; the raw traces stay outside the repository.
    python3 scripts/radeon-queue-gaps.py <trace-dir with round-*/run_kernel_trace.csv> <result-dir>"""
import csv, collections, json, statistics as st, sys
from pathlib import Path
traces, res = Path(sys.argv[1]), Path(sys.argv[2]); out = {}
for rd in sorted(traces.glob('round-*')):
    row = json.loads(next((res/'trace'/rd.name).glob('*.jsonl')).read_text().splitlines()[0])
    K = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp']), int(r['Queue_Id']), int(r['Stream_Id']))
               for r in csv.DictReader(open(rd/'run_kernel_trace.csv')))
    S = K[len(K)//2:]
    busy, (cs, ce) = 0, S[0][:2]
    for s, e, *_ in S[1:]:
        if s > ce: busy += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs
    gaps = collections.defaultdict(list); end, q0 = S[0][1], S[0][2]
    for s, e, q, _ in S[1:]:
        g = (s - end) / 1e3
        if g < 2000: gaps['same' if q == q0 else 'switch'].append(max(g, 0.0))
        if e > end: end, q0 = e, q
    out[rd.name] = {'steady_epoch_ms': row['results']['steady_epoch_s']*1e3, 'kernels': len(K), 'steady_kernels': len(S),
                    'kernel_ms': sum(e - s for s, e, *_ in S)/1e6, 'busy_ms': busy/1e6, 'span_ms': (S[-1][1] - S[0][0])/1e6,
                    'stream_queue': sorted({(st_, q) for _, _, q, st_ in K}),
                    **{f'{k}_{m}': f(v) for k, v in gaps.items() for m, f in (('n', len), ('median_us', st.median),
                       ('p90_us', lambda x: sorted(x)[int(.9*len(x))]), ('total_ms', lambda x: sum(x)/1e3))}}
(res/'trace-summary.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1)[:1500])
