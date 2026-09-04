"""Run one benchmark configuration and produce one JSONL row.

Protocol : build the network through the fnn-testkit adapter
from fixed initial weights; numerical checks against the NumPy oracle *before*
anything is timed (a tight deterministic check on the untrained forward pass
and a step-count-aware check on the first-epoch loss); `warmup` untimed
train()/predict() calls; then `repeat` timed calls of `epochs` epochs each (one
C++ call = dataset H2D + all epochs + per-epoch loss readback, so the per-epoch
wall time is the whole thing the user pays, amortised over `epochs`); median
and IQR of the per-epoch wall time; the per-phase profile of the timed calls;
the analytic FLOP model; and everything needed to replay the row.

Weights keep training across warm-up and repetitions (only the timing matters
there); the losses are checked to stay finite.  Thread counts are applied by
`cli.py` in a subprocess, because OpenMP/MKL/OpenBLAS runtimes read
OMP_NUM_THREADS once.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import fnn_testkit
from fnn_testkit import backends as be
from fnn_testkit import workloads as wl
from fnn_testkit.plugin import random_params
from fnn_testkit.reference import NetSpec, ReferenceNetwork

from . import gpumon

from . import flops as fl
from . import results, sysinfo

FLOAT_FORWARD_TOL = 2e-4  # same as the parity suite
DOUBLE_FORWARD_TOL = 1e-9
FLOAT_EPOCH_TOL = 5e-2  # first-epoch loss after hundreds of steps: chaotic in the summation order
DOUBLE_EPOCH_TOL = 1e-6


@dataclass
class RunConfig:
    backend: str = "syclnn"
    device: str | None = None
    workload: str = "monk"
    dtype: str = "float"
    batch: int | None = None
    epochs: int = 5
    check_samples: int = 8192  # the oracle check trains on at most this many samples (timing uses all)
    repeat: int = 5
    warmup: int = 1
    mode: str = "train"  # train | infer
    options: dict[str, Any] = field(default_factory=dict)
    threads: int | None = None
    seed: int = 1
    check: bool = True
    samples: int | None = None  # override the workload's sample count (synthetic / truncation)
    layers: list[int] | None = None  # override the workload's layer sizes
    tag: str = ""

    def effective(self) -> dict[str, Any]:
        """The configuration that determines the measurement (used for the run id)."""
        w, X, _, batch = _spec_and_data(self)
        return {
            "backend": self.backend,
            "device": be.resolve_device(self.backend, self.device),
            "workload": self.workload,
            "layers": list(w.spec.sizes),
            "n_samples": int(X.shape[0]),
            "dtype": self.dtype,
            "batch": batch,
            "epochs": self.epochs,
            "mode": self.mode,
            "options": dict(sorted(self.options.items())),
            "threads": self.threads,
            "seed": self.seed,
        }


def _spec_and_data(cfg: RunConfig):
    w = wl.get(cfg.workload)
    if cfg.layers:
        hidden = w.spec.layers[1].activation if len(w.spec.layers) > 2 else "relu"
        w = w.with_layers(cfg.layers, hidden=hidden, output=w.spec.layers[-1].activation)
    kw = dict(w.dataset_kw)
    if cfg.samples and w.dataset in ("synthetic", "cup"):
        kw["n"] = cfg.samples
    X, Y = wl.datasets.load(w.dataset, **kw)
    if cfg.samples and cfg.samples < X.shape[0]:
        X, Y = X[: cfg.samples], Y[: cfg.samples]
    batch = min(cfg.batch or w.batch_size, X.shape[0])
    return w, X, Y, batch


def _ref_cache_path(cfg: RunConfig, spec: NetSpec, n: int, batch: int, shuffle: bool, shuffle_seed: int) -> Path:
    key = json.dumps({"workload": cfg.workload, "layers": list(cfg.layers or []), "samples": cfg.samples, "spec": str(spec),
                      "n": n, "batch": batch, "seed": cfg.seed, "dtype": cfg.dtype, "shuffle": shuffle,
                      "shuffle_seed": shuffle_seed, "testkit": fnn_testkit.__version__}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = Path(os.environ.get("FNN_REF_CACHE", Path.home() / ".cache" / "fnn-bench" / "ref"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def reference_first_epoch_loss(cfg: RunConfig, spec: NetSpec, Ws, bs, X, Y, batch: int) -> float:
    """First-epoch loss of the float64 oracle from the same initial weights (cached)."""
    shuffle = bool(cfg.options.get("shuffle", False))
    shuffle_seed = int(cfg.options.get("shuffle_seed", 1))
    p = _ref_cache_path(cfg, spec, X.shape[0], batch, shuffle, shuffle_seed)
    if p.exists():
        return json.loads(p.read_text())["loss"]
    ref = ReferenceNetwork(spec, Ws, bs, shuffle=shuffle, shuffle_seed=shuffle_seed)
    loss = float(ref.train(X, Y, batch, 1)[0])
    tmp = p.with_suffix(f".{os.getpid()}.tmp")  # the cache is shared between environments: atomic replace
    tmp.write_text(json.dumps({"loss": loss}))
    os.replace(tmp, p)
    return loss


def numerical_check(cfg: RunConfig, spec: NetSpec, Ws, bs, X, Y, batch: int) -> dict[str, Any]:
    """Two-tier check against the oracle, run before anything is timed, on its own
    instance that is released before the timed instance is built (the widest
    double configurations do not fit twice in 11 GB)."""
    out: dict[str, Any] = {"enabled": True}
    is_double = cfg.dtype == "double"
    net = be.NetworkAdapter(cfg.backend, spec, dtype=cfg.dtype, device=cfg.device, initial_weights=Ws,
                            initial_biases=bs, **cfg.options)
    # (1) deterministic: the untrained forward pass on a slice of the data
    n_probe = min(256, X.shape[0])
    ref = ReferenceNetwork(spec, Ws, bs)
    got = net.predict(X[:n_probe]).astype(np.float64)
    exp = ref.predict(X[:n_probe])
    # error relative to the output scale (per-element relative errors explode on
    # near-zero outputs of deep float32 nets whose absolute error is ~1e-6)
    fwd_err = float(np.max(np.abs(got - exp)) / max(float(np.max(np.abs(exp))), 1e-6))
    fwd_tol = DOUBLE_FORWARD_TOL if is_double else FLOAT_FORWARD_TOL
    out.update(forward_rel_err=fwd_err, forward_tol=fwd_tol, forward_ok=bool(fwd_err <= fwd_tol))
    # (2) first-epoch loss (stochastic in the summation order for float32) on a bounded
    # subset: the float64 oracle is single-threaded NumPy and its Adam passes over a
    # 4096x4096 net cost about an hour per epoch on the full W4 dataset, so the
    # check trains on the first `check_samples` samples (a fresh instance with the
    # same initial weights when it is a strict subset; the timing runs use all)
    if cfg.mode == "train":
        n_check = min(X.shape[0], max(int(cfg.check_samples), batch)) if cfg.check_samples > 0 else X.shape[0]
        Xc, Yc = X[:n_check], Y[:n_check]
        loss0 = float(net.train(np.ascontiguousarray(Xc, dtype=net.np_dtype),
                                np.ascontiguousarray(Yc, dtype=net.np_dtype), batch, 1)[0])
        ref0 = reference_first_epoch_loss(cfg, spec, Ws, bs, Xc, Yc, batch)
        rel = abs(loss0 - ref0) / max(abs(ref0), 1e-12)
        tol = DOUBLE_EPOCH_TOL if is_double else FLOAT_EPOCH_TOL
        out.update(samples=int(n_check), loss=loss0, reference_loss=ref0, rel_err=rel, tol=tol, loss_ok=bool(rel <= tol))
    out["ok"] = bool(out["forward_ok"] and out.get("loss_ok", True))
    del net
    gc.collect()  # release the device buffers before the timed instance is built
    return out


def run(cfg: RunConfig, out: str | Path | None = None, verbose: bool = True) -> dict[str, Any]:
    w, X, Y, batch = _spec_and_data(cfg)
    spec = w.spec
    sizes = list(spec.sizes)
    rng = np.random.default_rng(cfg.seed)
    Ws, bs = random_params(spec, rng, scale=0.1)
    check: dict[str, Any] = {"enabled": False}
    if cfg.check:
        check = numerical_check(cfg, spec, Ws, bs, X, Y, batch)
        if verbose and not check["ok"]:
            print(f"!! numerical check FAILED: {json.dumps({k: v for k, v in check.items() if k != 'enabled'})}")
    net = be.NetworkAdapter(cfg.backend, spec, dtype=cfg.dtype, device=cfg.device, initial_weights=Ws,
                            initial_biases=bs, **cfg.options)
    itemsize = 8 if cfg.dtype == "double" else 4
    n = X.shape[0]
    Xd = np.ascontiguousarray(X, dtype=net.np_dtype)
    Yd = np.ascontiguousarray(Y, dtype=net.np_dtype)

    # ---- warm-up ----
    for _ in range(cfg.warmup):
        if cfg.mode == "train":
            net.train(Xd, Yd, batch, 1)
        else:
            net.predict(Xd, batch)
    net.reset_profile()

    # ---- measured repetitions (GPU utilisation / power sampled alongside) ----
    walls = []
    losses = []
    with gpumon.GpuSampler(enabled=cfg.device != "cpu" and not net.is_reference) as sampler:
        for _ in range(cfg.repeat):
            t0 = time.perf_counter()
            if cfg.mode == "train":
                l = net.train(Xd, Yd, batch, cfg.epochs)
                losses.append([float(v) for v in l])
            else:
                for _ in range(cfg.epochs):
                    net.predict(Xd, batch)
            walls.append((time.perf_counter() - t0) / cfg.epochs)
    gpu_samples = sampler.summary()
    finite = bool(np.all(np.isfinite(np.asarray(losses, dtype=np.float64)))) if losses else True
    profiling = bool(cfg.options.get("profile", False)) and not net.is_reference
    prof = (net.profile or {}) if profiling else None

    # ---- derived metrics ----
    median = statistics.median(walls)
    q1, q3 = np.percentile(walls, [25, 75])
    # steady-state epoch: the library's own per-epoch wall (inside the call, after the
    # dataset upload) without the first epoch of each call, which carries the H2D
    # staging, lazy initialisation and (graph mode) the capture. Needs profile=True.
    steady = None
    ew = (prof or {}).get("epoch_wall_ns") or []
    if cfg.mode == "train" and cfg.epochs > 1 and len(ew) == cfg.repeat * cfg.epochs:
        inner = [ew[i] / 1e9 for i in range(len(ew)) if i % cfg.epochs != 0]
        steady = float(statistics.median(inner))
    if cfg.mode == "train":
        step = fl.train_epoch(sizes, n, batch)
        intensity = fl.step_intensity(sizes, batch, itemsize)
    else:
        step = fl.inference(sizes, n)
        intensity = fl.inference_intensity(sizes, batch, itemsize)
    profiled_epochs = (prof or {}).get("epochs") or 0
    per_epoch = None
    if prof:
        div = profiled_epochs if (cfg.mode == "train" and profiled_epochs) else cfg.repeat * cfg.epochs
        per_epoch = {k: v / 1e6 / div for k, v in prof.items() if isinstance(v, (int, float)) and k.endswith("_ns")}
    if prof and prof.get("unprofiled", 0) and verbose:
        print(f"!! {prof['unprofiled']} launches had no profiling information")
    effective_options = None
    if not net.is_reference:
        try:
            effective_options = {k: getattr(net.net.options, k) for k in dir(net.net.options) if not k.startswith("_")}
            effective_options = {k: (v.name if hasattr(v, "name") else v) for k, v in effective_options.items()
                                 if not callable(v)}
        except Exception:
            effective_options = None
    sys_info = sysinfo.collect(repos={"fnn-bench": Path(__file__).resolve().parents[2]})
    row = {
        "id": results.run_id(cfg.effective()),
        "timestamp": sys_info["timestamp"],
        "tag": cfg.tag,
        "backend": cfg.backend,
        "build_info": be.build_info(cfg.backend),
        "device": net.net.device if not net.is_reference else {"name": net.device_name, "type": "cpu", "backend": "numpy"},
        "device_selector": be.resolve_device(cfg.backend, cfg.device),
        "blas": getattr(net.net, "blas_backend", cfg.options.get("blas", "auto")) if not net.is_reference else "numpy",
        "dtype": cfg.dtype,
        "workload": cfg.workload,
        "layers": sizes,
        "n_samples": n,
        "samples_override": cfg.samples,
        "batch": batch,
        "epochs": cfg.epochs,
        "repeat": cfg.repeat,
        "warmup": cfg.warmup,
        "mode": cfg.mode,
        "threads": cfg.threads,
        "threads_effective": sys_info.get("omp_max_threads"),
        "seed": cfg.seed,
        "options": cfg.options,
        "effective_options": effective_options,
        "results": {
            "epoch_wall_s": walls,
            "median_epoch_s": median,
            "iqr_epoch_s": float(q3 - q1),
            "steady_epoch_s": steady,
            "steady_samples_per_s": (n / steady) if steady else None,
            "samples_per_s": n / median,
            "steps_per_s": ((n + batch - 1) // batch) / median,
            "flops_per_epoch": {"gemm": step.gemm, "total": step.total, "forward_gemm": step.forward_gemm,
                                "delta_gemm": step.delta_gemm, "grad_gemm": step.grad_gemm,
                                "elementwise": step.elementwise, "update": step.update},
            "gflops_gemm": step.gemm / median / 1e9,
            "gflops_total": step.total / median / 1e9,
            "intensity_flop_per_byte": intensity,
            "losses": losses,
            "losses_finite": finite,
            "profile": prof,
            "profile_per_epoch_ms": per_epoch,
            "gpu_monitor": gpu_samples,
        },
        "check": check,
        "sysinfo": sys_info,
    }
    if verbose:
        r = row["results"]
        status = "ok" if (check.get("ok", True) and finite) else "FAIL"
        print(f"{cfg.backend:7s} {row['device']['name'][:32]:32s} {cfg.workload:14s} b={batch:<5d} {cfg.dtype:6s} "
              f"{cfg.mode:5s} epoch {median*1e3:10.2f} ms  {r['samples_per_s']:12.0f} samples/s  "
              f"{r['gflops_gemm']:9.1f} GFLOP/s(gemm)  check={status}")
    if out:
        results.append(out, row)
    return row
