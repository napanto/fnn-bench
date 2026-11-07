#!/usr/bin/env python3
"""One training process sampled by wall clock: warm-up, then unsampled, sampled, unsampled calls of equal length."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['cudann', 'syclnn', 'ompnn'], required=True)
    ap.add_argument('--workload', default='mnist-512-256')
    ap.add_argument('--samples', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--warm', type=int, default=12, help='epochs before any timing, past the GPU clock ramp')
    ap.add_argument('--epochs', type=int, required=True, help='epochs of each timed call')
    ap.add_argument('--period-us', type=int, default=2000)
    ap.add_argument('--delay-ms', type=int, default=1000, help='samples start after this, past the upload of the call')
    ap.add_argument('--max-samples', type=int, default=40000)
    ap.add_argument('--discard', action='store_true', help='keep only the timing: the sampler acts as a measured host delay')
    ap.add_argument('--option', action='append', default=[], help='key=value network option')
    ap.add_argument('--lib', type=Path, required=True, help='wallprof.so')
    ap.add_argument('--out', type=Path, required=True, help='sample dump; the summary goes next to it as .json')
    a = ap.parse_args()
    import numpy as np
    from fnn_testkit import backends as be, workloads
    prof = ctypes.CDLL(str(a.lib.resolve()))
    prof.wallprof_stop.restype = ctypes.c_long
    prof.wallprof_stop.argtypes = [ctypes.c_char_p]
    prof.wallprof_start.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_long]
    prof.wallprof_handler_ns.restype = ctypes.c_uint64
    w = workloads.get(a.workload)
    x, y = w.data()
    x, y = np.ascontiguousarray(x[:a.samples], dtype=np.float32), np.ascontiguousarray(y[:a.samples], dtype=np.float32)
    def value(text):
        return {'true': True, 'false': False}.get(text, int(text) if text.isdigit() else text)
    options = {k: value(v) for k, v in (o.split('=', 1) for o in a.option)}
    net = be.NetworkAdapter(a.backend, w.spec, dtype='float', device='gpu', seed=1, **options)
    net.train(x, y, a.batch, a.warm)

    def timed():
        t0 = time.perf_counter()
        net.train(x, y, a.batch, a.epochs)
        return time.perf_counter() - t0

    before = timed()
    status = prof.wallprof_start(a.delay_ms, a.period_us, a.max_samples)
    if status:
        raise SystemExit(f'wallprof_start {status}')
    t0 = time.perf_counter(); cpu0 = time.thread_time()
    net.train(x, y, a.batch, a.epochs)
    sampled = time.perf_counter() - t0; cpu = time.thread_time() - cpu0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = prof.wallprof_stop(None if a.discard else str(a.out).encode())
    handler_s = prof.wallprof_handler_ns()/1e9
    after = timed()
    summary = {'backend': a.backend, 'workload': a.workload, 'samples': a.samples, 'batch': a.batch, 'epochs': a.epochs,
               'options': options, 'period_us': a.period_us, 'delay_ms': a.delay_ms, 'n_samples': n, 'handler_s': handler_s,
               'before_s': before, 'sampled_s': sampled, 'after_s': after, 'sampled_thread_cpu_s': cpu}
    a.out.with_suffix('.json').write_text(json.dumps(summary) + '\n')
    print(a.backend, a.epochs, 'epochs', f'{1e3*before/a.epochs:.2f}', f'{1e3*sampled/a.epochs:.2f}', f'{1e3*after/a.epochs:.2f}', 'ms/epoch', n, 'samples')


if __name__ == '__main__':
    main()
