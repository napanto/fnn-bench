#!/usr/bin/env python3
"""Symbolise wallprof dumps, inside the image that ran them, into one JSON table per dump."""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


def symbols(path):
    table = {}
    for flags in (['-C'], ['-C', '-D']):
        r = subprocess.run(['nm', '--defined-only', *flags, path], capture_output=True, text=True, errors='replace')
        for line in r.stdout.splitlines():
            parts = line.split(' ', 2)
            if len(parts) == 3 and parts[1] in 'tTwWiV':
                table.setdefault(int(parts[0], 16), parts[2])
    addrs = sorted(table)
    return addrs, [table[a] for a in addrs]


class Resolver:
    def __init__(self, maps):
        self.maps = sorted(maps)
        self.starts = [m[0] for m in self.maps]
        self.base = {}
        for start, _end, _offset, path in self.maps:
            self.base.setdefault(path, start)
        self.tables = {}
        self.memo = {}

    def __call__(self, addr):
        hit = self.memo.get(addr)
        if hit is None:
            hit = self.memo[addr] = self.resolve(addr)
        return hit

    def resolve(self, addr):
        i = bisect.bisect_right(self.starts, addr) - 1
        if i < 0 or addr >= self.maps[i][1]:
            return '?', '?'
        path = self.maps[i][3]
        if not path.startswith('/'):
            return path or 'anon', '?'
        name = Path(path).name
        if path not in self.tables:
            self.tables[path] = symbols(path)
        addrs, names = self.tables[path]
        j = bisect.bisect_right(addrs, addr - self.base[path]) - 1
        return name, names[j] if j >= 0 else '?'


def read(path):
    maps, samples = [], []
    for line in open(path, errors='replace'):
        if line.startswith('M '):
            f = line.split(None, 6)
            lo, hi = f[1].split('-')
            maps.append((int(lo, 16), int(hi, 16), int(f[3], 16), f[6].strip() if len(f) > 6 else ''))
        else:
            f = line.split()
            samples.append((int(f[1]), int(f[2]), int(f[3]), int(f[4], 16), [int(v, 16) for v in f[5:]]))
    return maps, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dumps', nargs='+', type=Path)
    a = ap.parse_args()
    for dump in a.dumps:
        maps, samples = read(dump)
        res = Resolver(maps)
        stacks = Counter()
        for _wall, _cpu, _handler, rip, frames in samples:
            # frames: handler, signal trampoline, interrupted function (often repeated as rip), callers
            for i, pc in enumerate(frames):
                if res(pc)[1] == '__restore_rt' or (res(pc)[0].startswith('libc') and 'restore_rt' in res(pc)[1]):
                    frames = frames[i+1:]
                    break
            else:
                frames = frames[2:]
            if frames and frames[0] == rip:
                frames = frames[1:]
            names = [res(rip)] + [res(pc) for pc in frames]
            stacks[tuple(f'{d};{s[:90]}' for d, s in names)] += 1
        walls = [s[0] for s in samples]
        cpus = [s[1] for s in samples]
        out = {'dump': dump.name, 'n': len(samples),
               'wall_s': (walls[-1]-walls[0])/1e9 if walls else 0, 'thread_cpu_s': (cpus[-1]-cpus[0])/1e9 if cpus else 0,
               'handler_us_mean': sum(s[2] for s in samples)/max(len(samples), 1)/1e3,
               'stacks': [[n, list(st)] for st, n in stacks.most_common()]}
        dump.with_suffix('.stacks.json').write_text(json.dumps(out) + '\n')
        print(dump.name, out['n'], 'samples', f"handler {out['handler_us_mean']:.1f} us", len(stacks), 'stacks')


if __name__ == '__main__':
    main()
