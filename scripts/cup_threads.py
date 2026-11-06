#!/usr/bin/env python3
"""Per-call times of a short CPU training with what each thread did during the call.

Between two calls, outside the timed interval, it reads for every thread the CPU it last ran on, its
schedstat (time on CPU, time waiting in a run queue, time slices) and its context switches, and per CPU
the interrupt and softirq counters. --master-fifo moves only the main thread to SCHED_FIFO after the warm-up.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time


def threads():
    out = {}
    for tid in os.listdir('/proc/self/task'):
        base = f'/proc/self/task/{tid}/'
        try:
            stat = open(base+'stat').read()
            cpu = int(stat[stat.rindex(')')+2:].split()[36])
            run, wait, slices = (int(x) for x in open(base+'schedstat').read().split())
            status = open(base+'status').read()
            vcs = int(re.search(r'^voluntary_ctxt_switches:\s+(\d+)', status, re.M).group(1))
            nvcs = int(re.search(r'^nonvoluntary_ctxt_switches:\s+(\d+)', status, re.M).group(1))
        except (OSError, ValueError, AttributeError):
            continue
        out[int(tid)] = (cpu, run, wait, slices, vcs, nvcs)
    return out


def per_cpu(path):
    lines = open(path).read().splitlines()
    if not lines:
        return []
    n = len(lines[0].split())
    totals = [0]*n
    for line in lines[1:]:
        fields = line.split()[1:n+1]
        for i, f in enumerate(fields):
            if f.isdigit():
                totals[i] += int(f)
    return totals


def census():
    """Every task of the machine: name, last CPU, scheduler policy priority, time on CPU and slices (needs the host pid namespace)."""
    out = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            tids = os.listdir(f'/proc/{pid}/task')
        except OSError:
            continue
        for tid in tids:
            base = f'/proc/{pid}/task/{tid}/'
            try:
                stat = open(base+'stat').read()
                rest = stat[stat.rindex(')')+2:].split()
                run, wait, slices = (int(x) for x in open(base+'schedstat').read().split())
                out[tid] = (stat[stat.index('(')+1:stat.rindex(')')], int(rest[36]), int(rest[37]), run, slices)
            except (OSError, ValueError, IndexError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--calls', type=int, default=60)
    ap.add_argument('--master-fifo', type=int, default=0)
    ap.add_argument('--master-other', action='store_true', help='return the main thread to SCHED_OTHER after the warm-up')
    ap.add_argument('--census', action='store_true')
    ap.add_argument('--marker', type=Path, help='file created when the warm-up has ended')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    from fnn_testkit import backends as be, workloads
    w = workloads.get('cup')
    x, y = w.data()
    x, y = np.ascontiguousarray(x[:1000], dtype=np.float32), np.ascontiguousarray(y[:1000], dtype=np.float32)
    net = be.NetworkAdapter(a.backend, w.spec, dtype='float', device='cpu', seed=1)
    net.train(x, y, 40, 5)
    if a.master_fifo:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(a.master_fifo))
    if a.master_other:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    master = os.getpid(); calls = []
    before = census() if a.census else None
    if a.marker:
        a.marker.write_text(str(master))
    started = time.perf_counter_ns()
    for _ in range(a.calls):
        t1 = threads(); i1 = per_cpu('/proc/interrupts'); s1 = per_cpu('/proc/softirqs')
        begin = time.perf_counter_ns()
        net.train(x, y, 40, 5)
        wall = time.perf_counter_ns()-begin
        t2 = threads(); i2 = per_cpu('/proc/interrupts'); s2 = per_cpu('/proc/softirqs')
        rows = {tid: (t2[tid][0], t2[tid][1]-t1[tid][1], t2[tid][2]-t1[tid][2], t2[tid][3]-t1[tid][3], t2[tid][4]-t1[tid][4], t2[tid][5]-t1[tid][5])
                for tid in t2 if tid in t1}
        calls.append({'wall_ns': wall, 'threads': rows, 'irq': [b-a_ for a_, b in zip(i1, i2)], 'softirq': [b-a_ for a_, b in zip(s1, s2)]})
        time.sleep(0.01)
    seconds = (time.perf_counter_ns()-started)/1e9
    tasks = []
    if a.census:
        after = census()
        for tid, (name, cpu, prio, run, slices) in after.items():
            if tid in before and slices > before[tid][4]:
                tasks.append((name, cpu, prio, run-before[tid][3], slices-before[tid][4]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'backend': a.backend, 'master': master, 'calls': calls, 'seconds': seconds, 'tasks': tasks})+'\n')


if __name__ == '__main__':
    main()
