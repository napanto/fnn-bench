#!/usr/bin/env python3
"""Per-call epoch times of a long training on the GPU, with absolute timestamps to match a clock log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['cudann', 'syclnn', 'ompnn'], required=True)
    ap.add_argument('--workload', default='mnist-512-256')
    ap.add_argument('--samples', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--seconds', type=float, default=12.0)
    ap.add_argument('--option', action='append', default=[], help='key=value network option')
    ap.add_argument('--spin', action='store_true', help='make the primary CUDA context spin before the library starts')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.spin:
        import ctypes
        rt = ctypes.CDLL('libcudart.so.12')
        rt.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
        status = rt.cudaSetDeviceFlags(1)  # cudaDeviceScheduleSpin
        p = ctypes.c_void_p()
        status = status or rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(16))
        status = status or rt.cudaFree(p)
        if status:
            raise SystemExit(f'cuda status {status}')
    import numpy as np
    from fnn_testkit import backends as be, workloads
    w = workloads.get(a.workload)
    x, y = w.data()
    x, y = np.ascontiguousarray(x[:a.samples], dtype=np.float32), np.ascontiguousarray(y[:a.samples], dtype=np.float32)
    options = dict(o.split('=', 1) for o in a.option)
    net = be.NetworkAdapter(a.backend, w.spec, dtype='float', device='gpu', seed=1, **options)
    calls = []
    begin = time.time(); cpu0 = time.process_time()
    while time.time() - begin < a.seconds:
        t0 = time.time()
        net.train(x, y, a.batch, 3)  # three epochs per call, the first of a call carries the upload
        calls.append((t0, time.time()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'backend': a.backend, 'workload': a.workload, 'samples': a.samples, 'batch': a.batch,
                                 'epochs_per_call': 3, 'calls': calls, 'spin': a.spin,
                                 'cpu_s': time.process_time()-cpu0, 'wall_s': time.time()-begin}) + '\n')
    print(a.backend, len(calls), 'calls')


if __name__ == '__main__':
    main()
