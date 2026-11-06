#!/usr/bin/env python3
"""Hold an ordinary DRM profiling context in standard power state, then release it.

The public libdrm API owns the temporary state through this process's context.
No root command, persistent sysfs write, CPU clock change or service action is used.
"""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import re
import signal
import subprocess


def check(code, operation):
    if code: raise OSError(abs(code),operation+': '+os.strerror(abs(code)))


def stop(signum, frame):
    raise KeyboardInterrupt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date',required=True);ap.add_argument('--phase',choices=('probe','full','cache','peak','lds'),default='probe')
    a=ap.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',a.date): raise ValueError('invalid date')
    for command in (['pgrep','-f','[w]s-amd-cpu-numa'],['pgrep','-x','fnnbench']):
        if subprocess.run(command,stdout=subprocess.DEVNULL).returncode==0: raise SystemExit('measurement gate active')
    if Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq').read_text().strip()!='2800000' or Path('/sys/devices/system/cpu/cpufreq/boost').read_text().strip()!='0':
        raise SystemExit('CPU cap/boost no longer match the recorded conditions')
    cards=[p for p in Path('/sys/class/drm').glob('card*/device') if (p/'power_dpm_force_performance_level').exists()
           and (p/'vendor').exists() and (p/'vendor').read_text().strip()=='0x1002']
    card=max(cards,key=lambda p:int((p/'mem_info_vram_total').read_text()))
    control=card/'power_dpm_force_performance_level';previous=control.read_text().strip()
    if previous!='auto': raise SystemExit('GPU already has a non-auto policy; refusing to override it')
    render=next((card/'drm').glob('renderD*')).name
    lib=c.CDLL('libdrm_amdgpu.so.1')
    lib.amdgpu_device_initialize.argtypes=[c.c_int,c.POINTER(c.c_uint32),c.POINTER(c.c_uint32),c.POINTER(c.c_void_p)]
    lib.amdgpu_cs_ctx_create.argtypes=[c.c_void_p,c.POINTER(c.c_void_p)]
    lib.amdgpu_cs_ctx_stable_pstate.argtypes=[c.c_void_p,c.c_uint32,c.c_uint32,c.POINTER(c.c_uint32)]
    lib.amdgpu_cs_ctx_free.argtypes=[c.c_void_p];lib.amdgpu_device_deinitialize.argtypes=[c.c_void_p]
    fd=os.open('/dev/dri/'+render,os.O_RDWR|os.O_CLOEXEC)
    dev=c.c_void_p();ctx=c.c_void_p();major=c.c_uint32();minor=c.c_uint32();state=c.c_uint32()
    proc=None;selected=False;code=1
    signal.signal(signal.SIGTERM,stop)
    try:
        check(lib.amdgpu_device_initialize(fd,c.byref(major),c.byref(minor),c.byref(dev)),'DRM device init')
        check(lib.amdgpu_cs_ctx_create(dev,c.byref(ctx)),'DRM context create')
        check(lib.amdgpu_cs_ctx_stable_pstate(ctx,5,0,c.byref(state)),'query stable pstate')
        if state.value: raise RuntimeError('another profiling state is active')
        check(lib.amdgpu_cs_ctx_stable_pstate(ctx,6,1,None),'select standard profiling pstate');selected=True
        print(json.dumps(dict(api='amdgpu_cs_ctx_stable_pstate',uid=os.getuid(),previous=previous,
                              selected=control.read_text().strip(),drm_version=[major.value,minor.value])),flush=True)
        root=Path(__file__).resolve().parent.parent
        cmd=['bash',str(root/'scripts/ws-amd-stable-extras.sh'),a.phase,a.date]
        proc=subprocess.Popen(cmd,cwd=root,start_new_session=True);code=proc.wait()
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        code=130
    finally:
        if selected: check(lib.amdgpu_cs_ctx_stable_pstate(ctx,6,0,None),'release profiling pstate')
        if ctx.value:lib.amdgpu_cs_ctx_free(ctx)
        if dev.value:lib.amdgpu_device_deinitialize(dev)
        os.close(fd)
        restored=control.read_text().strip()
        print(json.dumps(dict(restored=restored,expected=previous)),flush=True)
        if restored!=previous: raise RuntimeError('GPU policy restoration failed')
    raise SystemExit(code)


if __name__=='__main__':main()
