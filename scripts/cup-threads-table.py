#!/usr/bin/env python3
"""Analyse scripts/cup_threads.py: which per-call record separates fast calls from slow ones."""
import collections
import json
import statistics as st
import sys
from pathlib import Path

d = Path(sys.argv[1])
SMT = 16  # CPU n and n+16 share a core; CPUs 0-7 and 16-23 on node 0, 8-15 and 24-31 on node 1


def core(c):
    return c % SMT


def node(c):
    return 0 if core(c) < 8 else 1


def describe(call, master):
    ths = call['threads']
    active = {t: v for t, v in ths.items() if v[1] > 100_000}  # at least 0.1 ms on CPU in the call
    cpus = [v[0] for v in active.values()]
    m = ths.get(str(master))
    return dict(wall=call['wall_ns']/5e6, active=len(active), cores=len({core(c) for c in cpus}),
                node1=sum(node(c) for c in cpus), shared=len(cpus)-len({core(c) for c in cpus}),
                wait=sum(v[2] for v in ths.values())/1e6, run=sum(v[1] for v in ths.values())/1e6,
                vcs=sum(v[4] for v in ths.values()), nvcs=sum(v[5] for v in ths.values()),
                m_cpu=m[0] if m else None, m_run=m[1]/1e6 if m else None, m_wait=m[2]/1e6 if m else None, m_nvcs=m[5] if m else None,
                m_irq=call['irq'][m[0]] if m else None, m_softirq=call['softirq'][m[0]] if m else None,
                irq_total=sum(call['irq']), irq_node0=sum(call['irq'][:8])+sum(call['irq'][16:24]), irq_node1=sum(call['irq'][8:16])+sum(call['irq'][24:32]))


groups = collections.defaultdict(list)
for f in sorted(d.glob('*.json')):
    if f.name == 'summary.json':
        continue
    data = json.loads(f.read_text())
    be, rest = f.stem.split('-', 1)
    variant, rnd = rest.rsplit('-r', 1)
    calls = [describe(c, data['master']) for c in data['calls'][2:]]
    groups[(be, variant)].append((int(rnd), calls))

out = {}
keys = ('wall', 'active', 'cores', 'node1', 'shared', 'wait', 'run', 'vcs', 'nvcs', 'm_run', 'm_wait', 'm_nvcs', 'm_irq', 'm_softirq', 'irq_node0', 'irq_node1')
for (be, v), procs in sorted(groups.items()):
    meds = [st.median(c['wall'] for c in calls) for _, calls in procs]
    every = [c for _, calls in procs for c in calls]
    print(f"\n{be}/{v}: {len(procs)} processes, {len(every)} calls; process medians {[round(x, 2) for x in meds]}; spread {100*(max(meds)/min(meds)-1):.1f} %; call-level p5/p50/p95 "
          f"{sorted(c['wall'] for c in every)[len(every)//20]:.2f}/{st.median(c['wall'] for c in every):.2f}/{sorted(c['wall'] for c in every)[-len(every)//20]:.2f}")
    print('   master CPUs', collections.Counter(c['m_cpu'] for c in every).most_common(4))
    ordered = sorted(every, key=lambda c: c['wall'])
    third = len(ordered)//3
    fast, slow = ordered[:third], ordered[-third:]
    print('   metric     fast-third  slow-third')
    for k in keys:
        a, b = st.mean(c[k] for c in fast), st.mean(c[k] for c in slow)
        print(f'   {k:10s} {a:10.2f} {b:10.2f}')
    out[f'{be}/{v}'] = dict(process_medians=meds, spread_pct=100*(max(meds)/min(meds)-1),
                            fast={k: st.mean(c[k] for c in fast) for k in keys}, slow={k: st.mean(c[k] for c in slow) for k in keys})
(d/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
