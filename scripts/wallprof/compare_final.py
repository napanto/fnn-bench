#!/usr/bin/env python3
"""Do the suggested options change the trained network? Same seed, ten epochs, then predictions and a loss in double."""
import argparse
import json
import numpy as np
from fnn_testkit import backends as be, workloads
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--backend', required=True)
ap.add_argument('--option', action='append', default=[])
ap.add_argument('--out', required=True)
a = ap.parse_args()
w = workloads.get('mnist-512-256')
x, y = w.data()
x = np.ascontiguousarray(x[:60000], dtype=np.float32); y = np.ascontiguousarray(y[:60000], dtype=np.float32)
options = {k: {'true': True, 'false': False}.get(v, int(v) if v.isdigit() else v) for k, v in (o.split('=', 1) for o in a.option)}
net = be.NetworkAdapter(a.backend, w.spec, dtype='float', device='gpu', seed=1, **options)
net.train(x, y, 256, 10)
p = np.asarray(net.predict(x, 4096), dtype=np.float64)
t = y.astype(np.float64)
np.save(a.out+'.npy', p.astype(np.float32))
print(json.dumps({'backend': a.backend, 'options': options, 'loss_double': float(0.5*((t-p)**2).sum()/len(t)), 'accuracy': float((p.argmax(1) == t.argmax(1)).mean())}))
