#!/usr/bin/env python3
"""Bounded Pascal counter compatibility/access probes; no driver setting changes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from experiment_utils import run_command
from extra_conditions import write_report
from gemm_shapes import driver_args


def denied(text):
    low=text.lower()
    return any(s in low for s in ('err_nvgpuctrperm','insufficient_privileges','does not have permission','permission to access gpu performance counters'))


def occupancy_data(path):
    if not path.exists(): return False
    index=None
    with path.open() as fh:
        for row in csv.reader(fh):
            if 'achieved_occupancy' in row:
                index=row.index('achieved_occupancy');continue
            if index is not None and len(row)>index:
                try:
                    if 0 < float(row[index].strip('%')) <= 100: return True
                except ValueError: pass
    return False


def ncu_counter_data(path):
    """A report with only static launch metadata does not prove counter access."""
    if not path.exists(): return False
    header=None
    with path.open() as fh:
        for row in csv.reader(fh):
            if 'Metric Name' in row and 'Metric Value' in row:
                header=row; continue
            if not header or len(row)!=len(header): continue
            record=dict(zip(header,row));name=record['Metric Name'].lower()
            if not any(s in name for s in ('achieved_occupancy','warps_active','sm_efficiency','sm__throughput','sm__cycles')):
                continue
            try:
                if float(record['Metric Value'].replace(',','').rstrip('%'))>0: return True
            except ValueError: pass
    return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--tools',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); a.tools.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic(); results=[]
    for tool in ('ncu-legacy','nvprof'):
        left=3600-(time.monotonic()-begin)
        if left<120:
            results.append(dict(tool=tool,status='one-hour acquisition/probe budget exhausted'));break
        with (a.out/f'{tool}-setup.log').open('w') as log:
            cmd=[sys.executable,'scripts/fetch-nvidia-profiler.py',tool,str(a.tools),'--budget-seconds',str(min(1200,int(left)-90))]
            code=run_command(cmd,log,min(1300,left))
        if code:
            results.append(dict(tool=tool,status='tool acquisition failed',exit=code));continue
        binary=(a.tools/f'{tool}-path.txt').read_text().strip()
        version=(a.tools/f'{tool}-version.txt').read_text()
        (a.out/f'{tool}-version.txt').write_text(version)
        prefix=a.out/f'{tool}-access'
        driver=driver_args('cudann','tiled','mnist',Path(str(prefix)+'.driver.json'),'cuda')
        if tool=='ncu-legacy':
            help_text=subprocess.run([binary,'--help'],capture_output=True,text=True,timeout=20)
            (a.out/'ncu-legacy-help.txt').write_text(help_text.stdout+help_text.stderr)
            clock=['--clock-control','none'] if '--clock-control' in help_text.stdout+help_text.stderr else []
            cmd=[binary,*clock,'--profile-from-start','off','--launch-count','1','--force-overwrite','--export',str(prefix),*driver]
        else:
            cmd=[binary,'--profile-from-start','off','--kernels','.*gemm.*','--metrics','achieved_occupancy',
                 '--print-gpu-trace','--csv','--log-file',str(prefix)+'.csv',*driver]
        with (a.out/'commands.txt').open('a') as log: log.write(shlex.join(cmd)+'\n')
        with Path(str(prefix)+'.log').open('w') as log:
            code=run_command(cmd,log,180)
        text=Path(str(prefix)+'.log').read_text()
        csvfile=Path(str(prefix)+'.csv')
        if csvfile.exists(): text+='\n'+csvfile.read_text()
        item=dict(tool=tool,executable=binary,version=version,command=cmd,exit=code)
        if denied(text): item['status']='counter access denied; host policy left unchanged'
        elif code: item['status']='runtime/compatibility probe failed; see log'
        elif tool=='nvprof' and occupancy_data(csvfile): item['status']='counter access confirmed'
        elif tool=='ncu-legacy':
            reports=[p for p in a.out.glob(prefix.name+'*') if p.suffix in ('.ncu-rep','.nv-nsight-cu-rep','.nsight-cuprof')]
            if reports:
                export=[binary,'--import',str(reports[0]),'--page','raw','--csv']
                with csvfile.open('w') as log: ec=run_command(export,log,60)
                item['status']='counter access confirmed' if ec==0 and ncu_counter_data(csvfile) else 'no verified hardware counter in probe report'
            else: item['status']='no counter report produced'
        else: item['status']='no numeric counter data produced'
        results.append(item)
        (a.out/'access-checks.json').write_text(json.dumps(results,indent=2)+'\n')
        print(tool,item['status'],flush=True)
    (a.out/'access-checks.json').write_text(json.dumps(results,indent=2)+'\n')
    text=['# GTX 1080 Ti counter-tool support and access\n\n',
          'The current Nsight Compute support matrix excludes Pascal. The archived 2019.5 branch lists Pascal GP10x, '
          'and the official NVIDIA package repository provides 2019.5.0.14-1. That older tool must also work with the '
          'current CUDA 12.9/driver stack. CUDA 12.9 nvprof officially supports Pascal and is the matched fallback. '
          'Tools were extracted privately; no driver permission, service or host setting was changed.\n\n',
          '| tool | result | exit |\n|---|---|---|\n']
    for r in results: text.append(f"| {r['tool']} | {r['status']} | {r.get('exit','not run')} |\n")
    text+=['\nOfficial sources (downloaded with provenance in results/extra-sources-2026-11-03): '
           '[current Nsight Compute GPU matrix](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#gpu-support), '
           '[legacy 2019.5 GPU matrix](https://archive.docs.nvidia.com/nsight-compute/2019.5.1/ReleaseNotes/index.html), '
           '[CUDA 12.9 profiler support matrix](https://docs.nvidia.com/cuda/archive/12.9.1/profiler-users-guide/index.html#migrating-to-nsight-tools-from-visual-profiler-and-nvprof).\n']
    for p in a.out.glob('*-version.txt'): text += [f'\n{p.name}:\n\n```text\n',p.read_text(),'\n```\n']
    if not any(r['status']=='counter access confirmed' for r in results):
        text+=['\nNo full three-model counter matrix or counter-based kernel roofline can be claimed from these probes. '
               'STREAM and Nsight Systems remain separate experiments; no modeled byte count is substituted for missing counters.\n']
    else:
        text+=['\nAccess succeeded; the full matrix is a separate subsequent step. A probe is not the requested three-model result.\n']
    write_report(a.out,''.join(text))


if __name__=='__main__': main()
