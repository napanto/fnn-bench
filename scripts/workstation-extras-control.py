#!/usr/bin/env python3
"""Gate ws-nvidia experiments and bracket them with the authorised runtime bench mode.

This controller never accesses the department cluster and never manages services.
Run only after the local launch gate clears. Every transition is logged.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
FILES=('ws-nvidia-extras.sh','nsys_epoch.py','nsys-captures.py','nsys-epoch-table.py','nsys-crosscheck.py',
       'nvtx-range.cpp','stream-probe.cu','stream-table.py','stat-confirm.py','sanitizer-cases.py',
       'fetch-sanitizer.py','fetch-nvidia-profiler.py','nvidia-counter-check.py','experiment_utils.py',
       'extra_conditions.py','extra-plots.py','gemm_shapes.py','cuda-access-check.py','cublas-race-probe.cu')


def ssh(host, args, **kwargs):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shlex.join(args)],**kwargs)


def gate():
    if subprocess.run(['pgrep','-f','[l]aunch-ws-nvidia-om'],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('ws-nvidia measurement chain is still active')
    state=ssh('ws-nvidia',['podman','ps','--format','{{.Names}} {{.Image}}'],capture_output=True,text=True)
    if state.returncode: raise RuntimeError('cannot verify ws-nvidia container state: '+state.stderr)
    if any('fnn-' in line for line in state.stdout.splitlines()):
        raise RuntimeError('a fnn image/container is still running on ws-nvidia')


def mode(on, directory):
    name='on' if on else 'off'
    calls=[('root@ws-nvidia-host',[f'/root/bench-mode-{name}.sh']),
           ('ws-nvidia',['sudo','-n',f'/var/home/antonio/bench-mode-{name}.sh'])]
    failures=[]
    for host,args in calls:
        result=ssh(host,args,capture_output=True,text=True)
        with (directory/'bench-mode-transitions.txt').open('a') as fh:
            fh.write(datetime.now().isoformat()+' '+host+' '+shlex.join(args)+'\n'+result.stdout+result.stderr+'\n')
        if result.returncode: failures.append(host)
    status=ssh('root@ws-nvidia-host',['/root/bench-mode/status.sh'],capture_output=True,text=True)
    (directory/f'bench-mode-{name}-status.txt').write_text(status.stdout+status.stderr)
    if failures: raise RuntimeError('bench-mode transition failed on '+', '.join(failures))
    return status.stdout+status.stderr


def cancel_owned(date, remote):
    # The marker prevents the remote loop from starting another container after
    # an interrupted SSH call; only our labelled fnn-extra containers are stopped.
    ssh('ws-nvidia', ['mkdir', '-p', remote], check=True)
    ssh('ws-nvidia', ['touch', remote+'/CANCELLED'], check=True)
    state=ssh('ws-nvidia',['podman','ps','--filter','label=fnn.extra.owner=workstation-extras',
                       '--filter','label=fnn.extra.date='+date,'--format','{{.ID}} {{.Names}}'],capture_output=True,text=True)
    if state.returncode: raise RuntimeError('cannot verify owned container cleanup')
    for line in state.stdout.splitlines():
        ident,name=line.split(maxsplit=1)
        if name.startswith('fnn-extra-'):
            ssh('ws-nvidia',['podman','stop','--time','5',ident],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date',default=datetime.now().strftime('%Y-%m-%d'))
    ap.add_argument('--phases',nargs='+',default=['sanitizer','stream','nsys-epoch','counter-check','stat-confirm'],
                    choices=['sanitizer','stream','nsys-epoch','counter-check','stat-confirm','diagnostics','counter-api','counter-memory'])
    a=ap.parse_args(); gate()
    # Only changed code, after the chain is over. Never sync a pass or a wheel.
    files=['./scripts/'+name for name in FILES]+['./bench/fnnbench/results.py']
    subprocess.run(['rsync','-a','--relative',*files,'ws-nvidia:/var/home/antonio/fnn/fnn-bench/'],cwd=ROOT,check=True)
    directory=ROOT/'results/ws-nvidia'/f'extra-tools-{a.date}'; directory.mkdir(parents=True,exist_ok=True)
    remote=f'/var/home/antonio/fnn/fnn-bench/results/ws-nvidia/extra-tools-{a.date}'
    on_attempted=False; finished=False
    try:
        on_attempted=True; status=mode(True,directory)
        ssh('ws-nvidia',['mkdir','-p',remote],check=True)
        ssh('ws-nvidia',['rm','-f',remote+'/CANCELLED'],check=True)
        local=directory/'bench-mode-status.txt';local.write_text(status)
        subprocess.run(['rsync','-a',str(local),'ws-nvidia:'+remote+'/bench-mode-status.txt'],check=True)
        for phase in a.phases:
            cmd=['bash','/var/home/antonio/fnn/fnn-bench/scripts/ws-nvidia-extras.sh',phase,a.date]
            with (directory/f'{phase}-controller.log').open('w') as fh:
                result=ssh('ws-nvidia',cmd,stdout=fh,stderr=subprocess.STDOUT)
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
