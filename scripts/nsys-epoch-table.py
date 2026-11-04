#!/usr/bin/env python3
"""Analyze delimited Nsight Systems GPU intervals, including overlaps and gaps."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import runpy
import sqlite3
from extra_conditions import write_report

PHASE = runpy.run_path(str(Path(__file__).with_name('nsys-crosscheck.py')))['phase_of']


def union_ns(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        total += hi-lo if end is None or lo >= end else max(0,hi-end)
        end = hi if end is None else max(end,hi)
    return total


def csv_rows(path, header):
    lines = path.read_text().splitlines()
    start = next((i for i,s in enumerate(lines) if header in s), None)
    if start is None:
        raise ValueError(f'{path}: CSV header not found')
    return list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))


def gpu_class(name):
    n=name.lower()
    if 'memset' in n: return 'memset'
    if 'memcpy' in n:
        if any(s in n for s in ('htod','host-to-device','h2d')): return 'H2D'
        if any(s in n for s in ('dtoh','device-to-host','d2h')): return 'D2H'
        return 'other copy'
    if PHASE(name)=='gemm': return 'GEMM'
    if any(s in n for s in ('cudann','syclnn','ompnn')): return 'hand-written'
    if any(s in n for s in ('gemv','asum','nrm2','cublas')): return 'BLAS vector/reduction'
    return 'other kernels'


def api_class(name):
    n=name.lower()
    if 'launch' in n: return 'launch'
    if 'synchronize' in n or 'streamwaitevent' in n or 'eventquery' in n: return 'synchronization'
    return None


def ranges(path):
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings=dict(db.execute('select id,value from StringIds')) if 'StringIds' in tables else {}
    nv=[]
    if 'NVTX_EVENTS' in tables:
        for raw in db.execute('select * from NVTX_EVENTS'):
            r=dict(raw); name=r.get('text') or strings.get(r.get('textId'))
            if name=='fnn.train.capture' and r.get('end') is not None:
                nv.append((r['start'],r['end']))
    apis=[]
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            for raw in db.execute(f'select * from {table}'):
                r=dict(raw); r['name']=strings.get(r.get('nameId'),'')
                apis.append(r)
    db.close()
    starts=[r['end'] for r in apis if r['name'] in ('cudaProfilerStart','cuProfilerStart')]
    stops=[r['start'] for r in apis if r['name'] in ('cudaProfilerStop','cuProfilerStop')]
    # Capturing profiler API calls depends on the nsys version. NVTX is a precise,
    # independently recorded fallback; never infer wall time from first/last GPU work.
    if starts and stops and max(stops)>min(starts):
        return (min(starts),max(stops)), 'CUDA profiler API boundaries', apis, nv
    if len(nv)!=1:
        raise ValueError(f'{path}: expected one NVTX train interval, found {len(nv)}')
    return nv[0], 'NVTX train interval inside CUDA capture', apis, nv


def analyze(prefix):
    driver=json.loads(Path(str(prefix)+'.json').read_text())
    (lo,hi),source,apis,nv=ranges(Path(str(prefix)+'.sqlite'))
    gpu=csv_rows(Path(str(prefix)+'-gpu_cuda_gpu_trace.csv'),'Start (ns)')
    totals=defaultdict(float); intervals=[]; kernels=0
    for r in gpu:
        start=float(r['Start (ns)'].replace(',','')); duration=float(r['Duration (ns)'].replace(',',''))
        left,right=max(lo,start),min(hi,start+duration)
        if right<=left: continue
        kind=gpu_class(r['Name']); totals[kind]+=right-left; intervals.append((left,right))
        kernels += kind in ('GEMM','hand-written','BLAS vector/reduction','other kernels')
    hosts=defaultdict(list); counts=defaultdict(int)
    for r in apis:
        kind=api_class(r['name'])
        left,right=max(lo,r['start']),min(hi,r['end'])
        if kind and right>left:
            hosts[(kind,r.get('globalTid'))].append((left,right)); counts[kind]+=1
    host_union=defaultdict(float)
    for (kind,tid),ints in hosts.items(): host_union[kind]+=union_ns(ints)
    busy=union_ns(intervals); summed=sum(totals.values())
    if not intervals or not kernels:
        raise ValueError(f'{prefix}: no GPU kernels inside the delimited interval')
    return dict(name=prefix.name,backend=driver['backend'],profile=driver['profile_enabled'],range_source=source,
                range_ns=hi-lo,train_call_ns=driver['train_call_ns'],gpu_ns=dict(totals),gpu_busy_ns=busy,
                gpu_overlap_ns=summed-busy,gpu_idle_ns=hi-lo-busy,kernel_launches=kernels,
                host_api_calls=dict(counts),host_api_thread_union_ns=dict(host_union),
                internal_profile=driver['internal_profile'],epochs=driver['epochs'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    a=ap.parse_args(); rows=[]; failures=[]
    for path in sorted(a.directory.glob('*.json')):
        if path.name=='summary.json': continue
        try:
            d=json.loads(path.read_text())
            if 'train_call_ns' not in d: continue
            rows.append(analyze(path.with_suffix('')))
        except (ValueError,KeyError,FileNotFoundError,sqlite3.Error) as exc:
            failures.append(f'{path.name}: {exc}')
    expected={f'{be}-{state}' for be in ('cudann','syclnn','ompnn') for state in ('off','on')}
    missing=expected-{r['name'] for r in rows}
    failures += [f'{name}: complete trace/metadata not available' for name in sorted(missing)]
    (a.directory/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    out=['# Delimited training profiles\n\n',
         'Exact nsys and driver commands '
         'are in commands.txt and each driver JSON. MNIST 784–512–256–10, float, B=256, N=8192; '
         'backends cudann/nvcc, syclnn/DPC++, and ompnn/clang18nv. '
         'warm-up excluded, one captured train(epochs=2): upload, first epoch, one steady epoch. '
         'The internal phase counters cover this entire call, not the steady epoch alone.\n\n',
         'Range wall comes from CUDA profiler API boundaries when present, otherwise the explicitly named NVTX '
         'train interval inside that capture (boundary source is shown). No first-to-last-GPU-event wall estimate. '
         'Busy is the union of this process’s GPU intervals clipped to that range; overlap is sum minus union; '
         'idle is range minus union. Host API time is a union per thread, then summed across threads; nested '
         'runtime/driver API calls are not double-counted in that time. API call counts include both layers. '
         'The target-work gaps are not evidence that the physical GPU was idle: background services may use it.\n\n',
         '| capture | range boundary | range ms | whole train ms | GEMM ms | BLAS vector/reduction ms | hand-written ms | other kernels ms | H2D ms | D2H ms | memset ms | busy union ms | overlap ms | idle ms | kernel launches |\n',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n']
    for r in rows:
        g=r['gpu_ns']; vals=[r['range_ns'],r['train_call_ns'],g.get('GEMM',0),g.get('BLAS vector/reduction',0),g.get('hand-written',0),g.get('other kernels',0),
                            g.get('H2D',0),g.get('D2H',0),g.get('memset',0),r['gpu_busy_ns'],r['gpu_overlap_ns'],r['gpu_idle_ns']]
        out.append(f"| {r['name']} | {r['range_source']} | "+' | '.join(f'{v/1e6:.4f}' for v in vals)+f" | {r['kernel_launches']} |\n")
    out += ['\n| capture | launch API calls | launch host thread-ms | sync API calls | sync host thread-ms | first / steady epoch internal wall ms |\n|---|---|---|---|---|---|\n']
    for r in rows:
        h=r['host_api_thread_union_ns']; c=r['host_api_calls']; ew=r['internal_profile'].get('epoch_wall_ns',[])
        out.append(f"| {r['name']} | {c.get('launch',0)} | {h.get('launch',0)/1e6:.4f} | {c.get('synchronization',0)} | {h.get('synchronization',0)/1e6:.4f} | "+' / '.join(f'{v/1e6:.4f}' for v in ew)+' |\n')
    phases=('gemm','act','delta','biasgrad','update','loss','reg','h2d','d2h','other','wall')
    out += ['\nInternal profiler attribution, ms over the same two-epoch call (including upload where attributed). '
            'CUDA stream-event intervals, SYCL command timestamps, OpenMP synchronous host intervals. '
            'These are not an additive partition of wall time. OpenMP loss includes its implicit D2H readback.\n\n',
            '| capture | '+' | '.join(phases)+' |\n|---|'+'---|'*len(phases)+'\n']
    for r in rows:
        out.append(f"| {r['name']} | "+' | '.join(f"{r['internal_profile'].get(p+'_ns',0)/1e6:.4f}" if r['profile'] else 'disabled' for p in phases)+' |\n')
    for r in rows:
        missing=r['internal_profile'].get('unprofiled',0)
        if r['profile'] and missing:
            out.append(f"\n{r['name']}: {missing} launches lacked internal profiling information; zero/missing phase attribution does not establish absence of GPU work.\n")
    for name in ('versions.txt','NUMA-POLICY'):
        p=a.directory/name
        if p.exists(): out += [f'\n{name}:\n\n```text\n',p.read_text(),'\n```\n']
    if failures: out += ['\nIncomplete captures:\n\n']+['- '+s+'\n' for s in failures]
    write_report(a.directory, ''.join(out))
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
