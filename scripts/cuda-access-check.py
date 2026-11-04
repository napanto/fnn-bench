#!/usr/bin/env python3
"""Record CUDA/CUPTI loader diagnostics without changing the host driver."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import sys

out={"environment":{k:os.environ.get(k) for k in ("LD_LIBRARY_PATH","CUDA_VISIBLE_DEVICES","CUDA_INJECTION64_PATH")}}
for name in ("libcuda.so.1","libcuda.so"):
    try:
        lib=ctypes.CDLL(name);lib.cuInit.argtypes=[ctypes.c_uint];lib.cuInit.restype=ctypes.c_int
        code=lib.cuInit(0);n=ctypes.c_int(-1)
        get=lib.cuDeviceGetCount(ctypes.byref(n))
        error=ctypes.c_char_p();lib.cuGetErrorName(code,ctypes.byref(error))
        out[name]={"init_code":code,"init_name":error.value.decode() if error.value else None,"count_code":get,"devices":n.value}
    except OSError as exc: out[name]={"load_error":str(exc)}
try:
    lib=ctypes.CDLL('libcudart.so.12');n=ctypes.c_int(-1)
    code=lib.cudaGetDeviceCount(ctypes.byref(n));lib.cudaGetErrorString.restype=ctypes.c_char_p
    out['runtime']={"count_code":code,"devices":n.value,"error":lib.cudaGetErrorString(code).decode()}
except OSError as exc: out['runtime']={"load_error":str(exc)}
try:
    lib=ctypes.CDLL('libcupti.so.12');version=ctypes.c_uint32();code=lib.cuptiGetVersion(ctypes.byref(version))
    out['cupti']={"code":code,"api_version":version.value}
    if '--counter' in sys.argv:
        driver=ctypes.CDLL('libcuda.so.1');runtime=ctypes.CDLL('libcudart.so.12')
        runtime.cudaFree(ctypes.c_void_p())
        context=ctypes.c_void_p();driver.cuCtxGetCurrent(ctypes.byref(context))
        def status(code):
            name=ctypes.c_char_p();lib.cuptiGetResultString(code,ctypes.byref(name))
            return {'code':code,'name':name.value.decode() if name.value else None}
        event=ctypes.c_uint32();group=ctypes.c_void_p()
        calls={}
        code=lib.cuptiEventGetIdFromName(0,b'active_warps',ctypes.byref(event));calls['event_id']=status(code)
        if code==0:
            code=lib.cuptiEventGroupCreate(context,ctypes.byref(group),0);calls['group_create']=status(code)
            if code==0:
                try:
                    code=lib.cuptiEventGroupAddEvent(group,event);calls['group_add']=status(code)
                    if code==0:
                        code=lib.cuptiEventGroupEnable(group);calls['group_enable']=status(code)
                        if code==0: calls['group_disable']=status(lib.cuptiEventGroupDisable(group))
                finally: calls['group_destroy']=status(lib.cuptiEventGroupDestroy(group))
        out['legacy_counter_api']=calls
except OSError as exc:out['cupti']={"load_error":str(exc)}
out['mapped_libraries']=sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
    if '/' in s and any(k in s.lower() for k in ('libcuda','cupti','nvperf','injection'))})
print(json.dumps(out,indent=2))
