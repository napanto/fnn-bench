#!/usr/bin/env python3
"""Independent fnnbench run processes using the reported GPU-row protocols."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import subprocess
from experiment_utils import run_command
from extra_conditions import write_report
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'bench'))
from fnnbench import results as R

CONFIGS={f'{be}-mnist-b{batch}':(be,'mnist-512-256',batch,{}) for be in ('cudann','syclnn') for batch in (256,1024)}
CONFIGS.update({f'cudann-cup-{name}':('cudann','cup',40,opts) for name,opts in
                [('default',{}),('in-order',{'queue':'in_order'}),('graph',{'queue':'graph'})]})


def protocol(label,source):
    be,wl,batch,opts=CONFIGS[label]
    matches=[r for r in R.read([source/f'{be}.jsonl']) if R.valid(r) and r.get('backend')==be
             and r.get('mode')=='train' and r.get('workload')==wl and r.get('batch')==batch
             and r.get('dtype')=='float' and r.get('options',{})==opts]
    if not matches: raise ValueError(f'no pass row for {label}')
    protocols={tuple(r.get(k) for k in ('epochs','repeat','warmup','n_samples','seed')) for r in matches}
    if len(protocols)!=1: raise ValueError(f'ambiguous pass protocols for {label}: {protocols}')
    return matches[0]


def measure(a):
    r=protocol(a.run,a.source)
    target=a.out/f'round-{a.round}'/a.run
    target.mkdir(parents=True,exist_ok=True)
    if any(R.valid(x) for x in R.read([target])): return
    cmd=[sys.executable,'-m','fnnbench.cli','run','--backend',r['backend'],'--device','gpu','--dtype','float',
         '--workload',r['workload'],'--batch',str(r['batch']),'--epochs',str(r['epochs']),
         '--repeat',str(r['repeat']),'--warmup',str(r['warmup']),'--seed',str(r.get('seed',1)),
         '--samples',str(r['n_samples']),'--out',str(target)]
    for k,v in (r.get('options') or {}).items(): cmd += ['--option',f'{k}={v}']
    (target/'command.json').write_text(json.dumps({'command':cmd,'protocol_source':r['_source'],
         'source_line':r['_line'],'source_timestamp':r['timestamp'],'round':a.round},indent=2)+'\n')
    with (target/'run.log').open('w') as fh:
        rc=run_command(cmd,fh,300)

    (target/'exit.txt').write_text(str(rc)+'\n')
    if rc: raise SystemExit(rc)


def report(out):
    groups=defaultdict(list)
    for r in R.dedupe(R.read([out])):
        if R.valid(r): groups[(Path(r['_source']).parent.name,R.configuration_key(r))].append(r)
    text=['# Independent-process statistical confirmation\n\n',
          'Three rounds, alternating '
          'CUDA/DPC++ order for each MNIST batch; CUP option order reverses in round 2. Each entry is one '
          'fresh `fnnbench run` process. Exact commands and original protocol source rows are in '
          'round-*/<configuration>/command.json. Epochs/repeats follow the pass: MNIST 3/3; '
          'CUP default 5/5, in-order and graph 5/3. Different conditions stay in separate rows.\n\n',
          'Per-process median = median of steady_epochs_s (first epoch of every call excluded). '
          'Across-process spread is (max−min)/median. Within-process steady IQR is reported separately; '
          'these epochs are not independent process replicates.\n\n',
          '| configuration | processes | medians ms | median ms | min–max ms | spread | within-process steady IQR ms | epochs / repeats |\n',
          '|---|---|---|---|---|---|---|---|\n']
    summary=[]
    for (label,key),rs in sorted(groups.items()):
        rs.sort(key=lambda r:r['_source'])
        times=[R.epoch_s(r)*1e3 for r in rs]; iqrs=[r['results'].get('steady_iqr_s') for r in rs]
        text.append(f"| {label} | {len(rs)} | "+', '.join(f'{t:.4f}' for t in times)+
                    f" | {st.median(times):.4f} | {min(times):.4f}–{max(times):.4f} | {(max(times)-min(times))/st.median(times):.1%} | "+
                    ', '.join(f'{v*1e3:.4f}' if v is not None else 'not recorded' for v in iqrs)+
                    f" | {rs[0]['epochs']} / {rs[0]['repeat']} |\n")
        summary.append(dict(configuration=label,conditions=json.loads(key),process_medians_ms=times,
                            within_steady_iqr_s=iqrs,sources=[r['_source'] for r in rs]))
    missing={label:3-sum(len(rs) for (name,key),rs in groups.items() if name==label) for label in CONFIGS}
    text.append('\nRemaining independent processes: '+json.dumps({k:v for k,v in missing.items() if v})+'\n')
    for name in ('versions.txt','NUMA-POLICY'):
        p=out/name
        if p.exists(): text += [f'\n{name}:\n\n```text\n',p.read_text(),'\n```\n']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out, ''.join(text))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=Path('results/hpc-a30/2026-09-17/gpu-cuda-vs-sycl'))
    ap.add_argument('--run',choices=CONFIGS)
    ap.add_argument('--round',type=int,choices=(1,2,3),default=1)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.run: measure(a)
    report(a.out)


if __name__=='__main__': main()
