#!/usr/bin/env python3
"""Turn a stream-probe CSV into the experiment README and a machine-readable summary."""
import argparse
import csv
import io
import json
from pathlib import Path
from extra_conditions import write_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    a = ap.parse_args()
    raw = (a.directory / 'stream.csv').read_text()
    rows = list(csv.DictReader(io.StringIO('\n'.join(l for l in raw.splitlines() if not l.startswith('#')))))
    if {r['operation'] for r in rows} != {'copy', 'scale', 'add', 'triad'}:
        raise ValueError('incomplete STREAM probe')
    data = [{k: v if k == 'operation' else float(v) for k, v in r.items()} for r in rows]
    (a.directory / 'summary.json').write_text(json.dumps(data, indent=2)+'\n')
    out = ['# Device-resident STREAM-style bandwidth\n\n',
           'See commands.txt for the exact '
           'NUMA prefix and compiler invocation. Three device-resident float arrays (size in the table), five warm-ups per operation, '
           'and repeated device-event timings. GB/s is decimal and counts algorithmic read/write bytes; '
           'it is an effective bandwidth, not a DRAM counter. These are new workstation experiments, separate from the pass rows.\n\n',
           '| operation | total array MiB | best ms | median ms | best GB/s | median GB/s |\n|---|---|---|---|---|---|\n']
    for r in data:
        out.append(f"| {r['operation']} | {r['arrays_bytes']/2**20:.0f} | {r['best_ms']:.4f} | {r['median_ms']:.4f} | {r['best_GB_s']:.2f} | {r['median_GB_s']:.2f} |\n")
    for name in ('versions.txt', 'commands.txt'):
        p = a.directory / name
        if p.exists(): out += [f'\n{name}:\n\n```text\n', p.read_text(), '\n```\n']
    write_report(a.directory, ''.join(out))


if __name__ == '__main__':
    main()
