#!/usr/bin/env python3
"""Warm up, then capture exactly one train call with the CUDA profiler API.

The default interval is upload + first epoch + one steady epoch. Internal
phase counters cover this whole call; the API exposes no per-epoch phase split.
Also used for small Nsight Compute captures (--samples 256 --epochs 1).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import sys
import time


def cudart():
    # Use the runtime already loaded by the wheel, avoiding a second CUDA major.
    candidates = [s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                  if '/libcudart.so' in s and 'stubs' not in s]
    candidates += [ctypes.util.find_library('cudart'), 'libcudart.so.12', 'libcudart.so']
    for name in dict.fromkeys(candidates):
        if name:
            try:
                lib = ctypes.CDLL(name)
                for symbol in ('cudaProfilerStart', 'cudaProfilerStop', 'cudaDeviceSynchronize'):
                    getattr(lib, symbol).restype = ctypes.c_int
                    getattr(lib, symbol).argtypes = []
                return lib
            except OSError:
                pass
    raise RuntimeError('libcudart not found')


def checked(call):
    status = call()
    if status:
        raise RuntimeError(f'{call.__name__}: CUDA status {status}')


def nvtx(path):
    if not path:
        return None
    lib = ctypes.CDLL(path)
    lib.fnn_range_push.argtypes = [ctypes.c_char_p]
    lib.fnn_range_push.restype = ctypes.c_int
    lib.fnn_range_pop.argtypes = []
    lib.fnn_range_pop.restype = ctypes.c_int
    return lib


def hip_control(pause=True):
    """ROCTx scopes all submission threads, including AdaptiveCpp's workers."""
    rt = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    rt.hipDeviceSynchronize.restype = ctypes.c_int
    rt.hipDeviceSynchronize.argtypes = []
    marker = ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
    for name in ('roctxProfilerPause', 'roctxProfilerResume'):
        getattr(marker, name).restype = ctypes.c_int
        getattr(marker, name).argtypes = [ctypes.c_uint64]
    marker.roctxRangePushA.argtypes = [ctypes.c_char_p]
    marker.roctxRangePushA.restype = ctypes.c_int
    marker.roctxRangePop.argtypes = []
    marker.roctxRangePop.restype = ctypes.c_int
    if pause and marker.roctxProfilerPause(0):
        raise RuntimeError('ROCTx global profiler pause failed')
    return rt, marker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['cudann', 'syclnn', 'ompnn'], required=True)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--blas', default='auto')
    ap.add_argument('--workload', default='mnist-512-256')
    ap.add_argument('--samples', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--warm-epochs', type=int, default=1, help='epochs of the warm-up call before the capture')
    ap.add_argument('--option', action='append', default=[], help='key=value network option, e.g. queue=in_order')
    ap.add_argument('--nvtx-lib')
    ap.add_argument('--runtime', choices=('cuda', 'hip'), default='cuda')
    ap.add_argument('--hip-marker-only', action='store_true', help='record the ROCTx range without pausing SDK contexts')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    hip_rt, hip_marker = hip_control(not a.hip_marker_only) if a.runtime == 'hip' else (None, None)
    import numpy as np
    from fnn_testkit import backends as be, workloads
    from fnnbench import sysinfo

    w = workloads.get(a.workload)
    x, y = w.data()
    x, y = np.ascontiguousarray(x[:a.samples], dtype=np.float32), np.ascontiguousarray(y[:a.samples], dtype=np.float32)
    def value(text):
        return {'true': True, 'false': False}.get(text.lower(), int(text) if text.isdigit() else text)
    options = {k: value(v) for k, v in (o.split('=', 1) for o in a.option)}
    net = be.NetworkAdapter(a.backend, w.spec, dtype='float', device='gpu', seed=1, profile=a.profile, blas=a.blas, **options)
    if hip_marker and not a.hip_marker_only:
        # HSA creates the counter contexts lazily during network construction.
        # Refresh the paused set now so the warm-up itself is unprofiled.
        hip_marker.roctxProfilerResume(0)
        if hip_marker.roctxProfilerPause(0):
            raise RuntimeError('ROCTx pause after HSA initialisation failed')
    net.train(x, y, a.batch, a.warm_epochs)
    net.reset_profile()
    if a.runtime == 'cuda':
        rt, marker = cudart(), nvtx(a.nvtx_lib)
        checked(rt.cudaDeviceSynchronize)
        if marker:
            # Initialise NVTX's injection dispatch outside the measured range.
            marker.fnn_range_push(b'fnn.marker.warmup')
            marker.fnn_range_pop()
        checked(rt.cudaProfilerStart)
        if marker:
            marker.fnn_range_push(b'fnn.train.capture')
    else:
        rt, marker = hip_rt, hip_marker
        checked(rt.hipDeviceSynchronize)
        if not a.hip_marker_only and marker.roctxProfilerResume(0):
            raise RuntimeError('ROCTx global profiler resume failed')
        marker.roctxRangePushA(b'fnn.train.capture')
    begin = time.perf_counter_ns()
    try:
        losses = net.train(x, y, a.batch, a.epochs)
        end = time.perf_counter_ns()
    finally:
        if a.runtime == 'cuda':
            if marker:
                marker.fnn_range_pop()
            checked(rt.cudaProfilerStop)
        else:
            marker.roctxRangePop()
            if not a.hip_marker_only and marker.roctxProfilerPause(0):
                raise RuntimeError('ROCTx global profiler pause failed')
    profile = net.profile or {}
    effective = {k: getattr(net.net.options, k) for k in dir(net.net.options) if not k.startswith('_')}
    effective = {k: v.name if hasattr(v, 'name') else v for k, v in effective.items()
                 if isinstance(v, (str, int, float, bool)) or hasattr(v, 'name')}
    result = {'backend': a.backend, 'runtime': a.runtime, 'hip_marker_only': a.hip_marker_only,
              'build_info': be.build_info(a.backend), 'device': net.net.device,
              'workload': a.workload, 'samples': len(x), 'layers': list(w.spec.sizes), 'batch': a.batch,
              'epochs': a.epochs, 'profile_enabled': a.profile, 'blas': a.blas, 'options': options, 'effective_options': effective,
              'train_call_ns': end-begin, 'capture_marker': 'fnn.train.capture' if marker else None,
              'nvtx_range': 'fnn.train.capture' if marker and a.runtime == 'cuda' else None,
              'internal_profile': profile, 'losses': losses.tolist(), 'losses_finite': bool(np.isfinite(losses).all()),
              'sysinfo': sysinfo.collect(), 'pid': os.getpid(), 'command': sys.argv}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{a.backend}: {a.epochs} epochs, {len(x)} samples, {end-begin} ns; profile={a.profile}')
    if not result['losses_finite']:
        raise RuntimeError('nonfinite loss')


if __name__ == '__main__':
    main()
