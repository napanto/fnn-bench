"""Device peaks measured *through the backend*: a one-layer network with an
identity activation is a single GEMM (n x n x n), so the profiler's `gemm_ns`
gives the BLAS library's achieved GFLOP/s on this device; the activation kernel
of a wide layer (read net, write net and out: 3 x n x itemsize) gives a
STREAM-like streaming bandwidth of the hand-written kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from fnn_testkit import backends as be
from fnn_testkit.reference import Layer, NetSpec

from . import sysinfo


def measure(backend: str, device: str | None, dtype: str = "float", n: int = 4096, options: dict | None = None,
            repeat: int = 5) -> dict[str, Any]:
    options = dict(options or {})
    options["profile"] = True
    itemsize = 8 if dtype == "double" else 4
    spec = NetSpec((Layer(n, "disabled"), Layer(n, "disabled")), learning_rate=0.1)
    rng = np.random.default_rng(0)
    W = rng.uniform(-0.01, 0.01, (n, n))
    b = np.zeros(n)
    net = be.NetworkAdapter(backend, spec, dtype=dtype, device=device, initial_weights=[W], initial_biases=[b], **options)
    X = np.ascontiguousarray(rng.uniform(-1, 1, (n, n)), dtype=net.np_dtype)
    net.predict(X)  # warm-up (JIT, allocations)
    gemm, act, wall = [], [], []
    for _ in range(repeat):
        net.reset_profile()
        net.predict(X)
        p = net.profile
        gemm.append(p["gemm_ns"])
        act.append(p["act_ns"])
        wall.append(p["wall_ns"])
    flops = 2.0 * n * n * n
    bytes_act = 3.0 * n * n * itemsize
    g = np.median(gemm)
    a = np.median(act)
    return {
        "kind": "peak",
        "backend": backend,
        "device": net.net.device if not net.is_reference else {"name": net.device_name},
        "blas": getattr(net.net, "blas_backend", "auto") if not net.is_reference else "numpy",
        "dtype": dtype,
        "n": n,
        "gemm_gflops": flops / g if g else None,
        "gemm_ms": g / 1e6,
        "act_bandwidth_gbs": bytes_act / a if a else None,
        "act_ms": a / 1e6,
        "wall_ms": float(np.median(wall)) / 1e6,
        "build_info": be.build_info(backend),
        "timestamp": sysinfo.collect()["timestamp"],
        "sysinfo": sysinfo.collect(),
    }
