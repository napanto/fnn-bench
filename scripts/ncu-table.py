#!/usr/bin/env python3
"""Extract per-kernel counters; never substitute modeled bytes for missing DRAM counters."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path


def raw_csv(path):
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if '"ID"' in line and '"Metric Name"' in line), None)
    if start is None:
        raise ValueError(f'no NCU raw CSV header: {path}')
    kernels = defaultdict(dict)
    names = {}
    for r in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        if not str(r.get('ID', '')).isdigit() or not r.get('Metric Name'):
            continue
        key = int(r['ID'])
        names[key] = r['Kernel Name']
        try:
            value = float(r['Metric Value'].replace(',', ''))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        unit = r.get('Metric Unit', '')
        if r['Metric Name'] == 'gpu__time_duration.sum':
            scale = {'second': 1e9, 's': 1e9, 'msecond': 1e6, 'ms': 1e6,
                     'usecond': 1e3, 'us': 1e3, 'nsecond': 1, 'ns': 1}.get(unit)
            if scale is None:
                continue
            value *= scale
        kernels[key][r['Metric Name']] = value
    return [(k, names[k], kernels[k]) for k in sorted(kernels)]


def ratio(metrics, numerator, denominator):
    a, b = metrics.get(numerator), metrics.get(denominator)
    return a/b if a is not None and b else None


def summarize(path):
    meta = json.loads(path.with_suffix('.command.json').read_text())
    kernels = raw_csv(path)
    main = [(k, n, m) for k, n, m in kernels if 'gemm' in n.lower()
            and not any(s in n.lower() for s in ('reduce', 'epilogue'))]
    shapes = meta['shapes_in_submission_order']
    verified = len(main) == len(shapes)
    mapping = {k: shape for (k, _, _), shape in zip(main, shapes)} if verified else {}
    rows = []
    for key, name, m in kernels:
        shape = mapping.get(key)
        fma, add, mul = [m.get('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum') for op in ('ffma','fadd','fmul')]
        flops = 2*fma+add+mul if None not in (fma, add, mul) else None
        dram = m.get('dram__bytes_read.sum', math.nan)+m.get('dram__bytes_write.sum', math.nan)
        duration = m.get('gpu__time_duration.sum')
        rows.append(dict(source=path.name, backend=meta['backend'], label=meta['label'], blas=meta['blas'],
                         kernel_id=key, kernel=name, shape=shape, shape_verified=verified,
                         duration_ns=duration, ffma=fma, fadd=add, fmul=mul, measured_scalar_flops=flops,
                         dram_bytes=dram if math.isfinite(dram) else None,
                         intensity=flops/dram if flops and dram > 0 and math.isfinite(dram) else None,
                         gflops=flops/duration if flops and duration else None,
                         registers=m.get('launch__registers_per_thread'),
                         occupancy_pct=m.get('sm__warps_active.avg.pct_of_peak_sustained_active'),
                         load_sectors_per_request=ratio(m, 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'),
                         store_sectors_per_request=ratio(m, 'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'),
                         shared_load_conflicts=m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
                         shared_store_conflicts=m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum')))
    return rows


def fmt(x):
    return f'{x:.3g}' if x is not None and math.isfinite(x) else 'not available'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    a = ap.parse_args()
    rows = []
    for path in sorted(a.directory.glob('*.csv')):
        if path.with_suffix('.command.json').exists():
            rows += summarize(path)
    (a.directory/'kernels.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    out = ['# A30 GEMM access patterns, counters and time\n\n',
           'Commands: `DATE=<date> bash scripts/hpc-blocks.sh ncu-tiled <cuda|sycl|omp>`; exact invocations and '
           'compiler/build details are in each .command.json / .driver.json. Actual wheel kernels, float, '
           'MNIST 784–512–256–10, one batch of 256 after warm-up; large case is W4 4096-wide NN. '
           'NCU kernel replay flushes caches between passes. Timings describe isolated replay, not application wall time.\n\n',
           'Shape labels follow the source submission order (three forward, two hidden delta, three gradients); '
           'a capture with an unexpected number of main GEMM kernels is left unlabelled. Helpers remain separate. '
           'Scalar FLOPs = 2×FFMA + FADD + FMUL thread instructions; this does not count tensor instructions. '
           'Intensity uses measured DRAM read+write bytes of that same kernel, never the application traffic model.\n\n',
           'In the tiled source, NN loads of A/B are contiguous along the fast lane; TN makes A strided, NT makes B strided. '
           'Bs[li][lj] stores have stride 16 words across li. These source patterns are hypotheses to compare with counters, '
           'not claims that a compiler produced a particular instruction or bank-conflict rate.\n\n',
           '| implementation | shape / access pattern | registers/thread | occupancy % | global sectors/request ld / st | shared conflicts ld / st | DRAM bytes | scalar FLOPs | time us | intensity FLOP/B | GFLOP/s |\n',
           '|---|---|---|---|---|---|---|---|---|---|---|\n']
    for r in rows:
        shape = ' '.join(map(str,r['shape'])) if r['shape'] else f"helper or unverified (ID {r['kernel_id']})"
        out.append(f"| {r['label']} {r['blas']} | {shape} | {fmt(r['registers'])} | {fmt(r['occupancy_pct'])} | "
                   f"{fmt(r['load_sectors_per_request'])} / {fmt(r['store_sectors_per_request'])} | "
                   f"{fmt(r['shared_load_conflicts'])} / {fmt(r['shared_store_conflicts'])} | {fmt(r['dram_bytes'])} | "
                   f"{fmt(r['measured_scalar_flops'])} | {fmt(r['duration_ns']/1e3 if r['duration_ns'] else None)} | {fmt(r['intensity'])} | {fmt(r['gflops'])} |\n")
    for name in ('ncu-version.txt','access-status.txt'):
        p=a.directory/name
        if p.exists(): out += [f'\n{name}:\n\n```text\n',p.read_text(),'\n```\n']
    for path in sorted(a.directory.glob('*.exit')):
        if path.read_text().strip() != '0': out.append(f'\nIncomplete capture: {path.stem}, exit {path.read_text().strip()}; see corresponding log.\n')
    expected={f'{label}-{kind}-{size}' for label,kind in [('cudann-nvcc','tiled'),('cudann-nvcc','auto'),
              ('syclnn-dpcpp','tiled'),('ompnn-clang18nv','tiled')] for size in ('mnist','large')}
    missing=expected-{p.stem for p in a.directory.glob('*.csv')}
    if missing:
        out.append('\nPending or incomplete captures: '+', '.join(sorted(missing))+'.\n')
    (a.directory/'README.md').write_text(''.join(out))


if __name__ == '__main__':
    main()
