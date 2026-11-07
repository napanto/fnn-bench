#!/usr/bin/env python3
"""Split the wall time of an epoch into layers, from the .stacks.json files of one result directory.

Each sample goes to exactly one layer. A sample inside the driver goes to the outermost driver entry
point on its stack; any other sample goes to the first frame, walking out from the leaf, that is not
in the C or C++ runtime, so memcpy and malloc count for the library that called them. A layer in ms
is its share of the pooled samples of a path times the median unsampled epoch of its processes.
Inclusive figures count every sample whose stack passes through a frame, so they overlap the layers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import median

WAIT = {'cuStreamSynchronize', 'cuEventSynchronize', 'cuCtxSynchronize', 'cuCtxSynchronize_v2'}
MEMORY = ('cuMemAlloc', 'cuMemFree', 'cuMemset')
EVENT = {'cuEventRecord', 'cuEventCreate', 'cuEventDestroy_v2', 'cuEventQuery', 'cuStreamWaitEvent'}
RUNTIME = (r'libc[.-]', r'libm[.-]', r'ld-linux', r'libstdc\+\+', r'libgcc', r'\[vdso\]', r'libpthread', r'^\?$', r'^anon$')
OWNERS = [('cuBLAS, host side', r'libcublas'), ('oneMath', r'libonemath'), ('SYCL runtime', r'libsycl'),
          ('UR adapter', r'libur_|libumf'), ('CUDA runtime library', r'libcudart'),
          ('OpenMP runtime', r'libomp|libacc|libnvomp|libnvhpc|libnvcpumath|libnvc|libpgm'),
          ('network library', r'^_(syclnn|cudann|ompnn)'), ('Python', r'python|libpython')]


INCLUSIVE = {'through oneMath': r'^libonemath', 'through cuBLAS': r'^libcublas\.so', 'SYCL graph scheduler': r'Scheduler::addCG',
             'SYCL scheduler bypass': r'submit_kernel_scheduler_bypass', 'buffer release': r'removeMemoryObject'}


def split(frame):
    return frame.split(';', 1)


def layer(stack):
    entry = None
    for frame in stack:
        dso, sym = split(frame)
        if dso.startswith('libcuda.so'):
            entry = sym.split('@')[0]
    if entry is not None:
        blas = any(split(f)[0].startswith('libcublas') for f in stack)
        if entry in WAIT:
            return 'wait for the GPU'
        if entry.startswith('cuLaunchKernel'):
            return 'driver: launch, cuBLAS' if blas else 'driver: launch, own kernels'
        if entry in EVENT:
            return 'driver: events'
        if entry.startswith('cuOccupancy'):
            return 'driver: occupancy query'
        if entry.startswith('cuMemcpy'):
            return 'driver: copies'
        if entry.startswith(MEMORY):
            return 'driver: allocate and fill'
        return 'driver: other'
    for frame in stack:
        dso = split(frame)[0]
        if any(re.search(p, dso) for p in RUNTIME):
            continue
        for name, pattern in OWNERS:
            if re.search(pattern, dso):
                return name
        return 'other'
    return 'other'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dir', type=Path)
    a = ap.parse_args()
    runs = defaultdict(list)
    for path in sorted(a.dir.glob('*-r?.stacks.json')):
        tag = path.name.rsplit('-r', 1)[0]
        stacks = json.loads(path.read_text())
        info = json.loads(path.with_name(path.name.replace('.stacks.json', '.json')).read_text())
        epoch_ms = 500*(info['before_s']+info['after_s'])/info['epochs']
        counts = defaultdict(int)
        through = defaultdict(int)
        for n, stack in stacks['stacks']:
            counts[layer(stack)] += n
            for name, pattern in INCLUSIVE.items():
                if any(re.search(pattern, frame) for frame in stack):
                    through[name] += n
        total = sum(counts.values())
        runs[tag].append({'round': path.name.rsplit('-r', 1)[1][0], 'epoch_ms': epoch_ms,
                          'sampled_ms': 1e3*info['sampled_s']/info['epochs'], 'n': total,
                          'handler_us': stacks['handler_us_mean'],
                          'cpu_wall': info['sampled_thread_cpu_s']/info['sampled_s'],
                          'counts': dict(counts), 'through': dict(through),
                          'ms': {k: epoch_ms*v/total for k, v in counts.items()}})
    summary = {}
    for tag, rows in runs.items():
        names = sorted({k for r in rows for k in r['ms']}, key=lambda k: -median(r['ms'].get(k, 0) for r in rows))
        summary[tag] = {'processes': rows, 'epoch_ms': median(r['epoch_ms'] for r in rows),
                        'layers_ms': {k: median(r['ms'].get(k, 0) for r in rows) for k in names},
                        'range_ms': {k: [min(r['ms'].get(k, 0) for r in rows), max(r['ms'].get(k, 0) for r in rows)] for k in names}}
        s = summary[tag]
        total = sum(r['n'] for r in rows)
        s['samples'] = total
        s['pooled_ms'] = {k: s['epoch_ms']*sum(r['counts'].get(k, 0) for r in rows)/total for k in names}
        s['inclusive_ms'] = {k: s['epoch_ms']*sum(r['through'].get(k, 0) for r in rows)/total for k in INCLUSIVE}
        s['overhead_pct'] = 100*(median(r['sampled_ms'] for r in rows)/s['epoch_ms']-1)
        over = s['overhead_pct']
        print(f"{tag}: epoch {s['epoch_ms']:.2f} ms, {sum(r['n'] for r in rows)} samples in {len(rows)} processes, sampled epoch {over:+.1f} %")
        for k in names:
            lo, hi = s['range_ms'][k]
            print(f"  {s['layers_ms'][k]:7.2f}  ({lo:6.2f} {hi:6.2f})  {k}")
    (a.dir/'layers.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
