#!/usr/bin/env python3
"""Read the adopted GPU policy; missing/ambiguous policy fails closed."""
import argparse
from pathlib import Path
import re
import shlex


def prefix(path):
    lines=[s.split('#',1)[0].strip() for s in path.read_text().splitlines()]
    gpu=[s.split(':',1)[1].strip() for s in lines if s.startswith('gpu:')]
    if not gpu:
        gpu=[s for s in lines if s.startswith('numactl ')]
    if len(gpu)!=1:
        raise ValueError('expected exactly one GPU policy')
    if gpu[0]=='none': return ''
    words=shlex.split(gpu[0])
    if not words or words[0]!='numactl' or any(not re.fullmatch(r'[a-zA-Z0-9_+,:=./-]+',s) for s in words):
        raise ValueError('GPU policy must be none or a literal numactl prefix')
    return ' '.join(words)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('policy',type=Path)
    print(prefix(ap.parse_args().policy))
