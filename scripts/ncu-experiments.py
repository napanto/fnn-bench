#!/usr/bin/env python3
"""Profile the actual wheel's GEMMs in one MNIST batch and a large W4 batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from experiment_utils import run_command

SECTIONS = ('SpeedOfLight', 'MemoryWorkloadAnalysis', 'Occupancy', 'LaunchStats', 'SourceCounters')
METRICS = ('gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
           'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
           'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
           'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
           'sm__warps_active.avg.pct_of_peak_sustained_active', 'launch__registers_per_thread',
           'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
           'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum')
# network.hpp: forward layers 0,1,2; hidden delta 1,0; gradients 0,1,2.
MNIST = [('NN', 512, 256, 784), ('NN', 256, 256, 512), ('NN', 10, 256, 256),
         ('TN', 256, 256, 10), ('TN', 512, 256, 256),
         ('NT', 512, 784, 256), ('NT', 256, 512, 256), ('NT', 10, 256, 256)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ncu', required=True)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--blas', choices=('auto', 'tiled'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for size in ('mnist', 'large'):
        tag = f'{a.label}-{a.blas}-{size}'
        prefix = a.out / tag
        if prefix.with_suffix('.csv').exists():
            continue
        cmd = [a.ncu, '--profile-from-start', 'off', '--target-processes', 'all', '--replay-mode', 'kernel',
               '--cache-control', 'all', '--kernel-name-base', 'demangled', '--kernel-name', 'regex:.*(gemm|cublasLt|splitK).*',
               '--force-overwrite', '-o', str(prefix)]
        for section in SECTIONS:
            cmd += ['--section', section]
        cmd += ['--metrics', ','.join(METRICS)]
        if size == 'large':
            cmd += ['--launch-skip', '1', '--launch-count', '1']
        cmd += ['python', 'scripts/nsys_epoch.py', '--backend', a.backend, '--blas', a.blas, '--epochs', '1',
                '--out', str(prefix)+'.driver.json']
        cmd += ['--samples', '256', '--batch', '256'] if size == 'mnist' else [
            '--workload', 'sweep-w4096-d2-b4096', '--samples', '4096', '--batch', '4096']
        manifest = {'command': cmd, 'backend': a.backend, 'label': a.label, 'blas': a.blas, 'size': size,
                    'shapes_in_submission_order': MNIST if size == 'mnist' else [('NN', 4096, 4096, 4096)],
                    'shape_source': 'network.hpp forward_layer, hidden_delta, gradients and train batch submission order',
                    'cache_control': 'all; cold caches between kernel replay passes'}
        Path(str(prefix)+'.command.json').write_text(json.dumps(manifest, indent=2)+'\n')
        with Path(str(prefix)+'.log').open('w') as fh:
            fh.write('$ '+shlex.join(cmd)+'\n'); fh.flush()
            rc = run_command(cmd, fh, 700)

        Path(str(prefix)+'.exit').write_text(str(rc)+'\n')
        if rc:
            print(tag, 'failed', rc, flush=True)
            continue
        with prefix.with_suffix('.csv').open('w') as fh:
            export = [a.ncu, '--import', str(prefix)+'.ncu-rep', '--csv', '--page', 'raw', '--units', 'base']
            subprocess.run(export, stdout=fh, check=True)
        print(tag, 'collected', flush=True)


if __name__ == '__main__':
    main()
