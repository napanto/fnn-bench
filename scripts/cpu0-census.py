#!/usr/bin/env python3
"""Who uses a given CPU while a run is in progress: per-thread CPU time of every task last seen on it,
from /proc/*/task/*/stat, sampled every 50 ms, plus the CPU's busy share from /proc/stat. Kernel threads
keep their names; user threads are reported only by process name class, so nothing private is recorded."""
import argparse
import json
import os
import re
import time

def tasks():
    out = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            for tid in os.listdir(f'/proc/{pid}/task'):
                with open(f'/proc/{pid}/task/{tid}/stat') as f:
                    s = f.read()
                comm = s[s.index('(')+1:s.rindex(')')]
                rest = s[s.rindex(')')+2:].split()
                out[(int(pid), int(tid))] = (comm, int(rest[11])+int(rest[12]), int(rest[36]), rest[0] == 'R', pid)
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
    return out

def cpu_busy(cpu):
    with open('/proc/stat') as f:
        for line in f:
            if line.startswith(f'cpu{cpu} '):
                v = list(map(int, line.split()[1:]))
                return sum(v) - v[3] - v[4], sum(v)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--cpu', type=int, required=True)
ap.add_argument('--seconds', type=float, default=10)
ap.add_argument('--out', required=True)
a = ap.parse_args()
kernel_pids = set()
for pid in os.listdir('/proc'):
    if pid.isdigit():
        try:
            if os.readlink(f'/proc/{pid}/exe'):
                pass
        except OSError:
            kernel_pids.add(pid)
b0, t0 = cpu_busy(a.cpu)
prev = tasks(); usage = {}
end = time.time() + a.seconds
while time.time() < end:
    time.sleep(0.05)
    cur = tasks()
    for k, (comm, ticks, cpu, running, pid) in cur.items():
        if cpu != a.cpu or k not in prev:
            continue
        d = ticks - prev[k][1]
        if d > 0:
            kernel = pid in kernel_pids and re.match(r'(ksoftirqd|ktimers|kworker|rcu|migration|irq/|cpuhp|watchdog|kcompactd|khugepaged|kswapd|kthreadd)', comm)
            name = comm if kernel else ('benchmark' if comm.startswith('python') else 'other process')
            usage[name] = usage.get(name, 0) + d
    prev = cur
b1, t1 = cpu_busy(a.cpu)
hz = os.sysconf('SC_CLK_TCK')
json.dump({'cpu': a.cpu, 'seconds': a.seconds, 'busy_share': (b1-b0)/max(t1-t0, 1),
           'ms_by_task': {k: 1e3*v/hz for k, v in sorted(usage.items(), key=lambda kv: -kv[1])}}, open(a.out, 'w'), indent=1)
