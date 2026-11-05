#!/usr/bin/env python3
"""Gate ws-nvidia experiments and bracket them with the runtime bench mode.

This controller never accesses the department cluster and never manages services.
Run only after the launch gate clears. Every transition is logged.
NVIDIA_HOST and BENCH_HOST select local SSH aliases; NVIDIA_DIR selects the
remote repository parent and NVIDIA_BENCH_DIR the VM runtime-script directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from extra_conditions import public_bench_status

ROOT=Path(__file__).resolve().parent.parent
VM_HOST=os.environ.get("NVIDIA_HOST", "ws-nvidia")
HOST_HOST=os.environ.get("BENCH_HOST", "root@ws-nvidia-host")
REMOTE_ROOT=os.environ.get("NVIDIA_DIR", str(Path.home()/"fnn"))+"/fnn-bench"
VM_BENCH_DIR=os.environ.get("NVIDIA_BENCH_DIR", str(Path.home()))
FILES=('ws-nvidia-extras.sh','nsys_epoch.py','nsys-captures.py','nsys-epoch-table.py','nsys-crosscheck.py',
       'nvtx-range.cpp','stream-probe.cu','stream-table.py','stat-confirm.py','sanitizer-cases.py',
       'fetch-sanitizer.py','fetch-nvidia-profiler.py','nvidia-counter-check.py','experiment_utils.py',
       'extra_conditions.py','extra-plots.py','gemm_shapes.py','cuda-access-check.py','cublas-race-probe.cu')


def ssh(host, args, **kwargs):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shlex.join(args)],**kwargs)


def gate():
    if subprocess.run(['pgrep','-f',os.environ.get("NVIDIA_LAUNCH_PATTERN", "[l]aunch-.*-om")],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('ws-nvidia benchmark chain is still active')
    state=ssh(VM_HOST,['podman','ps','--format','{{.Names}} {{.Image}}'],capture_output=True,text=True)
    if state.returncode: raise RuntimeError('cannot verify ws-nvidia container state')
    if any('fnn-' in line for line in state.stdout.splitlines()):
        raise RuntimeError('a fnn image/container is still running on ws-nvidia')


def mode(on, directory):
    name='on' if on else 'off'
    calls=[('the host',HOST_HOST,[f'/root/bench-mode-{name}.sh']),
           ('the VM',VM_HOST,['sudo','-n',f'{VM_BENCH_DIR}/bench-mode-{name}.sh'])]
    failures=[]
    for role,host,args in calls:
        result=ssh(host,args,capture_output=True,text=True)
        with (directory/'bench-mode-transitions.txt').open('a') as fh:
            fh.write(f'{datetime.now().isoformat()} {role} bench-mode-{name}: exit {result.returncode}\n')
        if result.returncode: failures.append(role)
    status=ssh(HOST_HOST,['/root/bench-mode/status.sh'],capture_output=True,text=True)
    summary=public_bench_status(status.stdout)
    (directory/f'bench-mode-{name}-status.txt').write_text(summary)
    if status.returncode: failures.append('host status')
    if failures: raise RuntimeError('bench-mode transition failed on '+', '.join(failures))
    return summary


def cancel_owned(date, remote):
    # The marker prevents the remote loop from starting another container after
    # an interrupted SSH call; only our labelled fnn-extra containers are stopped.
    ssh(VM_HOST, ['mkdir', '-p', remote], check=True)
    ssh(VM_HOST, ['touch', remote+'/CANCELLED'], check=True)
    state=ssh(VM_HOST,['podman','ps','--filter','label=fnn.extra.owner=workstation-extras',
                       '--filter','label=fnn.extra.date='+date,'--format','{{.ID}} {{.Names}}'],capture_output=True,text=True)
    if state.returncode: raise RuntimeError('cannot verify owned container cleanup')
    for line in state.stdout.splitlines():
        ident,name=line.split(maxsplit=1)
        if name.startswith('fnn-extra-'):
            ssh(VM_HOST,['podman','stop','--time','5',ident],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date',default=datetime.now().strftime('%Y-%m-%d'))
    ap.add_argument('--phases',nargs='+',default=['sanitizer','stream','nsys-epoch','counter-check','stat-confirm'],
                    choices=['sanitizer','stream','nsys-epoch','counter-check','stat-confirm','diagnostics','counter-api','counter-memory'])
    a=ap.parse_args(); gate()
    # Only changed code, after the chain is over. Never sync a pass or a wheel.
    files=['./scripts/'+name for name in FILES]+['./bench/fnnbench/results.py','./bench/fnnbench/sysinfo.py']
    subprocess.run(['rsync','-a','--relative',*files,VM_HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
    directory=ROOT/'results/ws-nvidia'/f'extra-tools-{a.date}'; directory.mkdir(parents=True,exist_ok=True)
    remote=f'{REMOTE_ROOT}/results/ws-nvidia/extra-tools-{a.date}'
    on_attempted=False; finished=False
    try:
        on_attempted=True; status=mode(True,directory)
        ssh(VM_HOST,['mkdir','-p',remote],check=True)
        ssh(VM_HOST,['rm','-f',remote+'/CANCELLED'],check=True)
        local=directory/'bench-mode-status.txt';local.write_text(status)
        subprocess.run(['rsync','-a',str(local),VM_HOST+':'+remote+'/bench-mode-status.txt'],check=True)
        for phase in a.phases:
            cmd=['bash',REMOTE_ROOT+'/scripts/ws-nvidia-extras.sh',phase,a.date]
            with (directory/f'{phase}-controller.log').open('w') as fh:
                result=ssh(VM_HOST,cmd,stdout=fh,stderr=subprocess.STDOUT)
            with (directory/'phase-status.jsonl').open('a') as fh:
                fh.write(json.dumps(dict(phase=phase,exit=result.returncode,command=cmd,timestamp=datetime.now().isoformat()))+'\n')
            print(phase,'exit',result.returncode,flush=True)
        finished=True
    finally:
        # Try both restores even if one mode-on call or an experiment failed.
        if on_attempted:
            try:
                if not finished: cancel_owned(a.date,remote)
            finally:
                mode(False,directory)


if __name__=='__main__': main()
