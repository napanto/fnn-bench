"""Run one benchmark configuration and produce one JSONL row.

A run = build the network through the fnn-testkit adapter, one warm-up epoch,
then `repeat` measured epochs (each a separate train() call of one epoch on the
same network so that the per-epoch wall time is the whole C++ call), the
profiler read after the measured epochs, the FLOP model, and a numerical sanity
check of the first-epoch loss against the NumPy oracle (cached per configuration).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fnn_testkit import backends as be
from fnn_testkit import workloads as wl
from fnn_testkit.plugin import random_params
from fnn_testkit.reference import NetSpec, ReferenceNetwork

from . import flops as fl
from . import results, sysinfo


@dataclass
class RunConfig:
    backend: str = "syclnn"
    device: str | None = None
    workload: str = "monk"
    dtype: str = "float"
    batch: int | None = None
    epochs: int = 5
    repeat: int = 5
    warmup: int = 1
    mode: str = "train"  # train | infer
    options: dict[str, Any] = field(default_factory=dict)
    threads: int | None = None
    seed: int = 1
    check: bool = True
    check_tol: float = 1e-3
    samples: int | None = None  # override the workload's sample count (synthetic)
    layers: list[int] | None = None  # override the workload's layer sizes
    tag: str = ""

    def key(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("repeat")
        d.pop("warmup")
        d.pop("tag")
        return d


def _spec_and_data(cfg: RunConfig):
    w = wl.get(cfg.workload)
    if cfg.layers:
        w = w.with_layers(cfg.layers, hidden=w.spec.layers[1].activation if len(w.spec.layers) > 2 else "relu",
                          output=w.spec.layers[-1].activation)
    kw = dict(w.dataset_kw)
    if cfg.samples and w.dataset in ("synthetic", "cup"):
        kw["n"] = cfg.samples
    X, Y = wl.datasets.load(w.dataset, **kw)
    if cfg.samples and cfg.samples < X.shape[0]:
        X, Y = X[: cfg.samples], Y[: cfg.samples]
    batch = cfg.batch or w.batch_size
    return w, X, Y, batch


def _ref_cache_path(cfg: RunConfig, spec: NetSpec, n: int, batch: int) -> Path:
    key = json.dumps({"spec": str(spec), "n": n, "batch": batch, "seed": cfg.seed, "dtype": cfg.dtype}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = Path(os.environ.get("FNN_REF_CACHE", Path.home() / ".cache" / "fnn-bench" / "ref"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def reference_first_epoch_loss(cfg: RunConfig, spec: NetSpec, Ws, bs, X, Y, batch: int) -> float:
    """First-epoch loss of the float64 oracle from the same initial weights (cached)."""
    p = _ref_cache_path(cfg, spec, X.shape[0], batch)
    if p.exists():
        return json.loads(p.read_text())["loss"]
    ref = ReferenceNetwork(spec, Ws, bs)
    loss = float(ref.train(X, Y, batch, 1)[0])
    p.write_text(json.dumps({"loss": loss}))
    return loss


def run(cfg: RunConfig, out: str | Path | None = None, verbose: bool = True) -> dict[str, Any]:
    w, X, Y, batch = _spec_and_data(cfg)
    spec = w.spec
    sizes = list(spec.sizes)
    rng = np.random.default_rng(cfg.seed)
    Ws, bs = random_params(spec, rng, scale=0.1)
    if cfg.threads:
        os.environ["OMP_NUM_THREADS"] = str(cfg.threads)
    net = be.NetworkAdapter(cfg.backend, spec, dtype=cfg.dtype, device=cfg.device, initial_weights=Ws,
                            initial_biases=bs, **cfg.options)
    itemsize = 8 if cfg.dtype == "double" else 4
    n = X.shape[0]
    Xd = np.ascontiguousarray(X, dtype=net.np_dtype)
    Yd = np.ascontiguousarray(Y, dtype=net.np_dtype)

    # ---- sanity check: first epoch vs oracle (before warm-up changes the weights) ----
    check: dict[str, Any] = {"enabled": cfg.check}
    if cfg.check and cfg.mode == "train":
        loss0 = float(net.train(Xd, Yd, batch, 1)[0])
        ref0 = reference_first_epoch_loss(cfg, spec, Ws, bs, X, Y, batch)
        rel = abs(loss0 - ref0) / max(abs(ref0), 1e-12)
        tol = cfg.check_tol if cfg.dtype == "float" else cfg.check_tol * 1e-3
        check.update(loss=loss0, reference_loss=ref0, rel_err=rel, tol=tol, ok=bool(rel <= tol))
        if verbose and not check["ok"]:
            print(f"!! numerical check FAILED: loss {loss0} vs reference {ref0} (rel {rel:.2e} > {tol})")

    # ---- warm-up ----
    for _ in range(cfg.warmup):
        if cfg.mode == "train":
            net.train(Xd, Yd, batch, 1)
        else:
            net.predict(Xd, batch)
    net.reset_profile()

    # ---- measured repetitions ----
    walls = []
    losses = []
    for _ in range(cfg.repeat):
        t0 = time.perf_counter()
        if cfg.mode == "train":
            l = net.train(Xd, Yd, batch, cfg.epochs)
            losses.append([float(v) for v in l])
        else:
            for _ in range(cfg.epochs):
                net.predict(Xd, batch)
        walls.append((time.perf_counter() - t0) / cfg.epochs)
    prof = net.profile or {}

    # ---- derived metrics ----
    median = statistics.median(walls)
    q1, q3 = np.percentile(walls, [25, 75])
    step = fl.train_epoch(sizes, n, batch) if cfg.mode == "train" else fl.inference(sizes, n)
    row = {
        "id": results.run_id(cfg.key()),
        "timestamp": sysinfo.collect()["timestamp"],
        "tag": cfg.tag,
        "backend": cfg.backend,
        "build_info": be.build_info(cfg.backend),
        "device": net.net.device if hasattr(net.net, "device") and not net.is_reference else {"name": net.device_name, "type": "cpu"},
        "blas": getattr(net.net, "blas_backend", cfg.options.get("blas", "auto")) if not net.is_reference else "numpy",
        "dtype": cfg.dtype,
        "workload": cfg.workload,
        "layers": sizes,
        "n_samples": n,
        "batch": batch,
        "epochs": cfg.epochs,
        "repeat": cfg.repeat,
        "warmup": cfg.warmup,
        "mode": cfg.mode,
        "threads": cfg.threads or os.environ.get("OMP_NUM_THREADS"),
        "seed": cfg.seed,
        "options": cfg.options,
        "results": {
            "epoch_wall_s": walls,
            "median_epoch_s": median,
            "iqr_epoch_s": float(q3 - q1),
            "samples_per_s": n / median,
            "steps_per_s": ((n + batch - 1) // batch) / median,
            "flops_per_epoch": {"gemm": step.gemm, "total": step.total, "forward_gemm": step.forward_gemm,
                                "delta_gemm": step.delta_gemm, "grad_gemm": step.grad_gemm,
                                "elementwise": step.elementwise, "update": step.update},
            "gflops_gemm": step.gemm / median / 1e9,
            "gflops_total": step.total / median / 1e9,
            "intensity_flop_per_byte": fl.step_intensity(sizes, batch, itemsize),
            "losses": losses,
            "profile": prof,
            "profile_per_epoch_ms": {k: v / 1e6 / (cfg.repeat * cfg.epochs) for k, v in prof.items()
                                     if isinstance(v, (int, float)) and k.endswith("_ns")} if prof else {},
        },
        "check": check,
        "sysinfo": sysinfo.collect(repos={"fnn-bench": Path(__file__).resolve().parents[2]}),
    }
    if verbose:
        r = row["results"]
        print(f"{cfg.backend:7s} {row['device']['name'][:32]:32s} {cfg.workload:10s} b={batch:<5d} {cfg.dtype:6s} "
              f"{cfg.mode:5s} epoch {median*1e3:9.2f} ms  {r['samples_per_s']:12.0f} samples/s  "
              f"{r['gflops_gemm']:9.1f} GFLOP/s(gemm)  check={'ok' if check.get('ok', None) in (True, None) else 'FAIL'}")
    if out:
        results.append(out, row)
    return row
