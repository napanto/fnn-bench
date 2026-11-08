#!/usr/bin/env python3
"""Turn the output of scripts/sched/runq2.bt into runq2-summary.json: per tag (v-<variant>-r<round>) and role
(main = the thread that exec'd, other = every thread it created): run time, run time of slices that began
while the SMT sibling CPU ran another tagged thread, run-queue waits (from wakeup, preemption or a switch
while runnable) with their count, switches away while runnable, migrations; and the reported prev_state values.
    python3 scripts/sched/runq2-table.py <runq2.txt> <out.json>"""
import collections, json, re, sys
txt = open(sys.argv[1]).read(); out = collections.defaultdict(dict)
def role(r): return 'main' if r == '1' else 'other'
for name, field in (('run_us', 'run_ms'), ('run_sib_us', 'run_sibling_busy_ms'), ('involuntary', 'runnable_switches'), ('migrations', 'migrations')):
    for m in re.finditer(r'@%s\[v-(\w+)-r(\d), (\d)\]: (\d+)' % name, txt):
        out[f'{m[1]}-r{m[2]}'].setdefault(role(m[3]), {})[field] = int(m[4]) / (1e3 if field.endswith('_ms') else 1)
for m in re.finditer(r'@wait_us\[v-(\w+)-r(\d), (\d)\]: \{ .count = (\d+), .average = \d+, .total = (\d+) \}', txt):
    out[f'{m[1]}-r{m[2]}'].setdefault(role(m[3]), {}).update(waits=int(m[4]), wait_ms=int(m[5]) / 1e3)
states = {m[1]: int(m[2]) for m in re.finditer(r'@states\[(\d+)\]: (\d+)', txt)}
open(sys.argv[2], 'w').write(json.dumps({'runs': out, 'prev_state_counts': states}, indent=1, sort_keys=True) + '\n'); print(len(out), 'runs')
