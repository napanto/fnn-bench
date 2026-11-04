#!/usr/bin/env python3
"""Run small cudann parity cases under one CUDA Compute Sanitizer tool."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from experiment_utils import run_command
from extra_conditions import write_report
import time

CASES = ('test_parity_with_reference[cup-like]', 'test_remainder_batch')
OPTIONS = {'default': [], 'graph': ['--option', 'queue=graph'], 'tiled': ['--option', 'blas=tiled']}


def outcome(text, rc):
    counts = re.findall(r'ERROR SUMMARY:\s*(\d+) errors', text)
    race = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards displayed \((\d+) errors?, (\d+) warnings?\)', text)
    hazards = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards', text)
    if rc == 124:
        return 'timeout', None
    if counts or hazards:
        errors = sum(int(r[1]) for r in race) if race else sum(map(int, counts + hazards))
        if errors:
            return 'errors', errors
        if race and any(int(r[2]) for r in race):
            return 'warnings', 0
        if rc == 0 and re.search(r'\b\d+ passed\b', text):
            return 'pass', 0
    return 'failed / incomplete', None


def report(out):
    attempts = []
    for p in sorted(out.glob('cases-*.jsonl')):
        attempts += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    for r in attempts:
        log = out / r.get('log', r['key']+'.log')
        if log.exists():
            text = log.read_text()
            r['result'], r['errors'] = outcome(text, r['exit'])
            race = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards displayed \((\d+) errors?, (\d+) warnings?\)', text)
            r['hazards_displayed'] = sum(int(x[0]) for x in race) if race else None
            r['warnings'] = sum(int(x[2]) for x in race) if race else 0
    rows = list({r['key']: r for r in attempts}.values())
    (out / 'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    text = ['# Compute Sanitizer: small cudann parity cases\n\n',
            'Shared fnn-testkit parity test (cup-like) and remainder-batch test (7 samples, batch 3), '
            'each in float and double, with default, queue=graph and blas=tiled. '
            'Each table row is a separate process. A pass requires both pytest success and a zero tool summary. '
            'Timeouts or missing summaries are not passes. Tools check only their supported error classes.\n\n',
            '| case | dtype | options | tool | result | reported errors | warnings | displayed hazards | exit |\n|---|---|---|---|---|---|---|---|---|\n']
    for r in rows:
        text.append(f"| {r['case']} | {r['dtype']} | {r['options']} | {r['tool']} | {r['result']} | {r['errors'] if r['errors'] is not None else 'unknown'} | {r.get('warnings', 0)} | {r.get('hazards_displayed') if r.get('hazards_displayed') is not None else '-'} | {r['exit']} |\n")
    for p in sorted(out.glob('*version.txt')):
        text += [f'\nTool version ({p.name}):\n\n```text\n', p.read_text(), '\n```\n']
    text += ['\nCommands and durations: `cases-<tool>.jsonl`; full diagnostics: `<tool>-<dtype>-<options>-<case>.log`.\n']
    text.append(f'\nCoverage: {len(rows)} of 48 requested case/tool/precision/option combinations recorded. '
                'Unrecorded combinations remain pending; a failed test is not a clean sanitizer result.\n')
    text.append(f'\n{len(attempts)} total attempts; earlier failed setup/collection attempts and their logs are retained.\n')
    prior_errors=[r for r in attempts if r.get('errors')]
    if prior_errors:
        text.append('\nTool errors or hazards occurred in these attempts: '+', '.join(r['key'] for r in prior_errors)+'. A later pass does not erase those observations.\n')
    write_report(out, ''.join(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tool', choices=('memcheck', 'racecheck', 'synccheck', 'initcheck'))
    ap.add_argument('--sanitizer')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=100)
    ap.add_argument('--smoke', action='store_true', help='one float/default remainder case for a <=5-minute access check')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        report(a.out)
        return
    if not a.tool or not a.sanitizer:
        ap.error('--tool and --sanitizer are required unless --report-only is used')
    version = subprocess.run([a.sanitizer, '--version'], capture_output=True, text=True)
    (a.out / 'sanitizer-version.txt').write_text(version.stdout + version.stderr)
    status = a.out / f'cases-{a.tool}.jsonl'
    attempts = [json.loads(l) for l in status.read_text().splitlines()] if status.exists() else []
    done = {r['key'] for r in attempts if r['result'] == 'pass'}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for dtype in ('float', 'double'):
        for option, args in OPTIONS.items():
            for case in CASES:
                if a.smoke and (dtype != 'float' or option != 'default' or case != 'test_remainder_batch'):
                    continue
                key = f'{a.tool}-{dtype}-{option}-{case.split("[")[0]}'
                if key in done:
                    continue
                function, _, parameter = case.partition('[')
                selection = ['-k', parameter.rstrip(']')] if parameter else []
                cmd = [a.sanitizer, '--tool', a.tool, '--error-exitcode', '86', '--target-processes', 'all',
                       'python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--backend', 'cudann', '--device', 'gpu', '--dtype', dtype,
                       *args, *selection, f'../cudann/tests/test_suite.py::{function}']
                attempt = 1 + sum(r['key'] == key for r in attempts)
                log = a.out / (f'{key}.log' if attempt == 1 else f'{key}-attempt-{attempt}.log')
                start = time.monotonic()
                with log.open('w') as fh:
                    fh.write('$ ' + shlex.join(cmd) + '\n'); fh.flush()
                    rc = run_command(cmd, fh, a.timeout, env)

                result, errors = outcome(log.read_text(), rc)
                row = dict(key=key, tool=a.tool, case=case, dtype=dtype, options=option, command=cmd,
                           result=result, errors=errors, exit=rc, seconds=time.monotonic()-start,
                           attempt=attempt, log=log.name)
                attempts.append(row)
                with status.open('a') as fh:
                    fh.write(json.dumps(row) + '\n')
                report(a.out)
                print(key, result, flush=True)
    report(a.out)


if __name__ == '__main__':
    main()
