#!/usr/bin/env python3
"""Attribute the steady epoch of a delimited Nsight capture to GPU work, device latency and host time.

Input: a directory of <tag>.sqlite exports of a two-epoch capture (scripts/nsys_epoch.py). The steady
epoch is the second one: it starts at the host launch of kernel n/2 and ends at the train-call boundary.

The epoch wall time W is split twice, and each split adds up to W.

GPU view: W = busy + idle. Busy is the union of the process's kernel, copy and memset intervals. Each
idle interval before operation j is split at the end of j's host submission call: the part up to that
end is host-limited (the GPU had nothing to run yet); the part after it is device latency (the command
was submitted and had not started).

Host view, on the submitting thread: W = time inside CUDA API calls, by class, + time outside them.
The time outside is the language runtime and the library's own code between two CUDA calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics as st


def union(intervals):
    total, end = 0, None
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if end is None or lo >= end:
            total += hi-lo
            end = hi
        elif hi > end:
            total += hi-end
            end = hi
    return total


def api_class(name):
    n = name.lower()
    if 'launch' in n or 'graphlaunch' in n:
        return 'launch'
    if 'event' in n and 'sync' not in n and 'wait' not in n and 'query' not in n:
        return 'event-mgmt'
    if 'synchronize' in n or 'streamwaitevent' in n or 'eventquery' in n or 'streamquery' in n:
        return 'sync'
    if 'memalloc' in n or 'memfree' in n or 'malloc' in n or 'free' in n:
        return 'allocation'
    if 'memcpy' in n or 'memset' in n:
        return 'copy-set'
    return 'other-api'


def gpu_class(name):
    n = name.lower()
    if 'gemm' in n:
        return 'GEMM'
    if 'gemv' in n:
        return 'GEMV'
    if any(k in n for k in ('asum', 'nrm2', 'cublas', 'dot', 'copy_k')):
        return 'other BLAS'
    return 'hand-written'


def load(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings = dict(db.execute('select id,value from StringIds'))
    capture = None
    if 'NVTX_EVENTS' in tables:
        cols = [r[1] for r in db.execute('pragma table_info(NVTX_EVENTS)')]
        for row in db.execute('select * from NVTX_EVENTS'):
            r = dict(zip(cols, row))
            if (r.get('text') or strings.get(r.get('textId'))) == 'fnn.train.capture' and r.get('end'):
                capture = (r['start'], r['end'])
    apis = []
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            apis += [dict(start=s, end=e, tid=t, cid=c, name=strings.get(n, ''))
                     for s, e, t, c, n in db.execute(f'select start,end,globalTid,correlationId,nameId from {table}')]
    gpu = []
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
        for s, e, c, n, st_id in db.execute('select start,end,correlationId,shortName,streamId from CUPTI_ACTIVITY_KIND_KERNEL'):
            name = strings.get(n, '')
            gpu.append(dict(start=s, end=e, cid=c, name=name, kind=gpu_class(name), stream=st_id, kernel=True))
    for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
        if table in tables:
            for s, e, c, st_id in db.execute(f'select start,end,correlationId,streamId from {table}'):
                gpu.append(dict(start=s, end=e, cid=c, name=kind, kind=kind, stream=st_id, kernel=False))
    osrt = []
    if 'OSRT_API' in tables:
        osrt = [(s, e, strings.get(n, '')) for s, e, n in db.execute('select start,end,nameId from OSRT_API')]
    db.close()
    return capture, apis, gpu, osrt


def analyze(path):
    capture, apis, gpu, osrt = load(path)
    if capture is None:
        raise ValueError('no capture marker')
    lo, hi = capture
    apis = [a for a in apis if a['end'] > lo and a['start'] < hi]
    gpu = sorted((g for g in gpu if g['end'] > lo and g['start'] < hi), key=lambda g: g['start'])
    kernels = [g for g in gpu if g['kernel']]
    if len(kernels) < 2:
        raise ValueError('no kernels in the capture')
    by_cid = {a['cid']: a for a in apis if api_class(a['name']) in ('launch', 'copy-set') and a['cid']}
    mid = kernels[len(kernels)//2]
    begin = by_cid[mid['cid']]['start'] if mid['cid'] in by_cid else mid['start']
    window = [g for g in gpu if g['start'] >= begin]
    wall = hi-begin
    busy = union((g['start'], min(g['end'], hi)) for g in window)
    work = Counter()
    for g in window:
        work[g['kind']] += g['end']-g['start']
    # idle intervals, each split at the end of the host call that submitted the next operation
    host_limited = device = 0
    edge = begin
    latency = []
    for g in window:
        sub = by_cid.get(g['cid'])
        if sub:
            latency.append(g['start']-sub['end'])
        if g['start'] > edge:
            gap = g['start']-edge
            submitted = sub['end'] if sub else edge
            part = max(0, min(gap, submitted-edge))
            host_limited += part
            device += gap-part
        edge = max(edge, g['end'])
    tail = max(0, hi-edge)
    host_limited += tail
    # host view: the thread that issues the most API calls in the window
    in_window = [a for a in apis if a['end'] > begin]
    tids = Counter(a['tid'] for a in in_window)
    main = tids.most_common(1)[0][0]
    per = defaultdict(list)
    for a in in_window:
        if a['tid'] == main:
            per[api_class(a['name'])].append((max(a['start'], begin), min(a['end'], hi)))
    host = {k: union(v) for k, v in per.items()}
    inside = union([x for v in per.values() for x in v])
    calls = Counter(a['name'] for a in in_window if a['tid'] == main)
    time_by = Counter()
    for a in in_window:
        if a['tid'] == main:
            time_by[a['name']] += min(a['end'], hi)-max(a['start'], begin)
    launches = sum(n for name, n in calls.items() if api_class(name) == 'launch')
    return dict(wall_ms=wall/1e6, busy_ms=busy/1e6, idle_ms=(wall-busy)/1e6, idle_host_ms=host_limited/1e6,
                idle_device_ms=device/1e6, gpu_ms={k: v/1e6 for k, v in work.items()},
                host_api_ms=inside/1e6, host_outside_ms=(wall-inside)/1e6, host_class_ms={k: v/1e6 for k, v in host.items()},
                kernels=len([g for g in window if g['kernel']]), launches=launches, api_calls=sum(calls.values()),
                outside_us_per_kernel=(wall-inside)/1e3/max(1, len([g for g in window if g['kernel']])),
                median_submit_to_start_us=st.median(latency)/1e3 if latency else None,
                streams=len({g['stream'] for g in window}), threads=len(tids),
                top_calls=[(n, c, round(time_by[n]/1e6, 3)) for n, c in calls.most_common(10)],
                osrt_top=Counter(n for s, e, n in osrt if s >= begin).most_common(8) if osrt else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--glob', default='*.sqlite')
    a = ap.parse_args()
    rows = {}
    for p in sorted(a.directory.glob(a.glob)):
        try:
            rows[p.stem] = analyze(p)
        except (ValueError, KeyError, sqlite3.Error) as exc:
            print(f'{p.stem}: {exc}')
    (a.directory/'gap-summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    head = ('capture', 'wall', 'busy', 'idle', 'host-lim', 'device', 'GEMM', 'GEMV', 'hand', 'copy', 'memset',
            'API', 'outside', 'launch', 'event', 'sync', 'alloc', 'kernels', 'out/k us', 'sub→start us', 'streams', 'threads')
    print(' | '.join(head))
    for k, r in rows.items():
        g, h = r['gpu_ms'], r['host_class_ms']
        v = [k]+[f'{x:.2f}' for x in (r['wall_ms'], r['busy_ms'], r['idle_ms'], r['idle_host_ms'], r['idle_device_ms'],
                                     g.get('GEMM', 0), g.get('GEMV', 0), g.get('hand-written', 0), g.get('copy', 0), g.get('memset', 0),
                                     r['host_api_ms'], r['host_outside_ms'], h.get('launch', 0), h.get('event-mgmt', 0), h.get('sync', 0), h.get('allocation', 0))]
        v += [str(r['kernels']), f"{r['outside_us_per_kernel']:.1f}", f"{r['median_submit_to_start_us']:.1f}" if r['median_submit_to_start_us'] is not None else '-', str(r['streams']), str(r['threads'])]
        print(' | '.join(v))
    for k, r in rows.items():
        print(k, r['top_calls'])
        if r['osrt_top']:
            print('   osrt', r['osrt_top'])


if __name__ == '__main__':
    main()
