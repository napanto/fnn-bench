#!/usr/bin/env python3
"""Hardware placement of each HIP process's compute queues from the KFD debugfs snapshots taken by
scripts/ws-amd-radeon-queues.sh: every queue descriptor in kfd/mqds (creation order) is matched by its ring
base to the pipe and queue slot in kfd/hqds that holds it. Writes queue-slots.json with the slots and the
steady epoch of each process; the raw snapshots stay outside the repository.
    python3 scripts/radeon-queue-slots.py <snapshot-dir> <result-dir>"""
import json, re, sys
from pathlib import Path
snap, res = Path(sys.argv[1]), Path(sys.argv[2]); out = {}
for h in sorted(snap.glob('hqds-*.txt'), key=lambda p: int(p.stem.split('-')[1])):
    k = h.stem.split('-')[1]; slots = {}; cur = None
    for line in h.read_text().splitlines():
        m = re.match(r'\s*Inst 0,\s+CP Pipe (\d), Queue (\d)', line)
        if m: cur = f'pipe {m[1]} queue {m[2]}'; continue
        if re.match(r'\s*(SDMA|Inst 0, HIQ)', line): cur = None; continue
        if cur and line.strip().startswith('0000c844:'): slots[line.split()[1]] = cur
    order, comp = [], False
    for line in (snap/f'mqds-{k}.txt').read_text().splitlines():
        if 'Compute queue' in line: comp = True; continue
        if 'SDMA queue' in line: comp = False; continue
        if comp and line.strip().startswith('00000220:'): order.append(slots.get(line.split()[1], 'not found'))
    row = json.loads(next((res/f'round-{k}').glob('*.jsonl')).read_text().splitlines()[0])
    out[f'round-{k}'] = {'steady_epoch_ms': row['results']['steady_epoch_s']*1e3, 'queue_slots_in_creation_order': order}
(res/'queue-slots.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out)[:600])
