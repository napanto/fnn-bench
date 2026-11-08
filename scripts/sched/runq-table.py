#!/usr/bin/env python3
"""Turn the output of scripts/sched/runq.bt into runq-summary.json: per tag (v-<variant>-r<round>) and role
(main = the thread that exec'd, other = every thread it created), run-queue delay count and total, delays of
1 ms or more, migrations, and switches away while still runnable by what ran next (0 same run, 1 another
real-time task, 2 another normal task).
    python3 scripts/sched/runq-table.py <runq.txt> <out.json>"""
import collections, json, re, sys
txt = open(sys.argv[1]).read(); out = collections.defaultdict(dict)
def key(v, k, role): return f'{v}-r{k}', 'main' if role == '1' else 'other'
for m in re.finditer(r'@delay_us\[v-([\w]+)-r(\d), (\d)\]: \{ .count = (\d+), .average = \d+, .total = (\d+) \}', txt):
    t, r = key(*m.groups()[:3]); out[t].setdefault(r, {}).update(wakeups=int(m[4]), delay_ms=int(m[5])/1e3)
for m in re.finditer(r'@migrations\[v-([\w]+)-r(\d), (\d)\]: (\d+)', txt):
    t, r = key(*m.groups()[:3]); out[t].setdefault(r, {})['migrations'] = int(m[4])
for m in re.finditer(r'@preempted\[v-([\w]+)-r(\d), (\d), (\d)\]: (\d+)', txt):
    t, r = key(*m.groups()[:3]); out[t].setdefault(r, {}).setdefault('runnable_switches', {})[('same', 'rt', 'normal')[int(m[4])]] = int(m[5])
cur = None
for line in txt.splitlines():
    h = re.match(r'@delay_hist_us\[v-([\w]+)-r(\d), (\d)\]:', line)
    if h: cur = key(*h.groups()); continue
    b = re.match(r'^\[(\d+)([KM]?)(?:, *\d+[KM]?\)|\])\s+(\d+)', line)
    if cur and b:
        lo = int(b[1]) * (1000 if b[2] == 'K' else 1000000 if b[2] == 'M' else 1)
        d = out[cur[0]].setdefault(cur[1], {}); d['delays_ge_1ms'] = d.get('delays_ge_1ms', 0) + (int(b[3]) if lo >= 1000 else 0)
    elif not line.strip(): cur = None
open(sys.argv[2], 'w').write(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(len(out), 'runs')
