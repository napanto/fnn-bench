#!/usr/bin/env python3
"""Capture one workstation backend/profile pair, then export matching SQLite and CSV."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from experiment_utils import run_command


def nsys_binary():
    candidates=sorted(Path('/opt/nvidia').glob('nsight-systems/*/target-linux-x64/nsys'),reverse=True)
    found=shutil.which('nsys')
    if found: candidates.append(Path(found))
    for binary in candidates:
        result=subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=20)
        if result.returncode==0:
            return str(binary),result.stdout+result.stderr
    raise RuntimeError('No working Nsight Systems CLI in image')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',required=True)
    ap.add_argument('--state',choices=('off','on'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); binary,version=nsys_binary()
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/f'{a.backend}-version.txt').write_text(version)
    tag=f'{a.backend}-{a.state}'; prefix=a.out/tag
    expected=[Path(str(prefix)+suffix) for suffix in ('.json','.sqlite','-gpu_cuda_gpu_trace.csv','-api_cuda_api_sum.csv')]
    status=Path(str(prefix)+'.exit')
    if status.exists() and status.read_text().strip()=='0' and all(p.exists() for p in expected): return
    existing=[p for p in a.out.glob(tag+'.*') if p.is_file()]+list(a.out.glob(tag+'-gpu_*.csv'))+list(a.out.glob(tag+'-api_*.csv'))
    if existing:
        attempt=a.out/'attempts'/f'{tag}-{len(list((a.out/"attempts").glob(tag+"-*")))+1}'
        attempt.mkdir(parents=True,exist_ok=True)
        for p in existing: shutil.move(str(p),attempt/p.name)
    profile=[binary,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite','true',
             '-o',str(prefix),'python','scripts/nsys_epoch.py','--backend',a.backend,
             '--nvtx-lib',str((a.out/'nvtx-range.so').resolve()),'--out',str(prefix)+'.json']
    if a.state=='on': profile.append('--profile')
    commands=[profile,[binary,'export','--type','sqlite','--force-overwrite','true','-o',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep']]
    for report,kind in [('cuda_gpu_trace','gpu'),('cuda_api_sum','api')]:
        commands.append([binary,'stats','--report',report,'--format','csv','-o',str(prefix)+'-'+kind,str(prefix)+'.sqlite'])
    with Path(str(prefix)+'.log').open('w') as log:
        for cmd in commands:
            with (a.out/'commands.txt').open('a') as fh: fh.write(shlex.join(cmd)+'\n')
            log.write('$ '+shlex.join(cmd)+'\n');log.flush()
            code=run_command(cmd,log,300)
            if code: break
    status.write_text(str(code)+'\n')
    print(tag,'exit',code,flush=True)
    raise SystemExit(code)


if __name__=='__main__': main()
