"""fnnbench command line.

    fnnbench run   --backend syclnn --device cpu --workload mnist --batch 256 --dtype float --epochs 5 --repeat 5 --out results/ws-amd/2026-09-03/
    fnnbench sweep --backend syclnn --device gpu --workload sweep-w1024-d4-b256,mnist --batch 64,256,1024 --dtype float,double --option blas=mklcpu,netlib --out ...
    fnnbench sweep --backend syclnn --device cpu --workload mnist --threads 1,2,4,8,16,32 --out ...     # each thread count in a fresh process
    fnnbench sweep --plan plans/e1_cpu_blas.json --out ...
    fnnbench collect results/ -o results/all.csv
    fnnbench plot results/ws-amd/2026-09-03 -o analysis/figures
    fnnbench replay results/ws-amd/2026-09-03/syclnn.jsonl --id 0123abcd
    fnnbench sysinfo
    fnnbench peak --backend syclnn --device gpu --dtype float --size 8192
    fnnbench workloads
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from . import results, sysinfo
from .runner import RunConfig, run


def _literal(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def _parse_options(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = _literal(v)
    return out


def _parse_option_grid(items: list[str]) -> list[dict]:
    """--option key=v1,v2 --option other=x -> list of option dicts (cross product)."""
    axes = []
    for it in items or []:
        k, _, v = it.partition("=")
        axes.append([(k, _literal(tok)) for tok in v.split(",")])
    if not axes:
        return [{}]
    return [dict(combo) for combo in itertools.product(*axes)]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="syclnn")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--mode", default="train", choices=["train", "infer"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-check", action="store_true", help="skip the numerical sanity check")
    p.add_argument("--check-samples", type=int, default=8192,
                   help="samples used by the first-epoch oracle check (0 = all; timing always uses all)")
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--layers", default=None, help="override layer sizes, e.g. 784,1024,10")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None, help="results directory or .jsonl file")


def _out_path(out: str | None, backend: str) -> Path | None:
    if not out:
        return None
    p = Path(out)
    if p.suffix == ".jsonl":
        return p
    return p / f"{backend}.jsonl"


def _config_from_args(a: argparse.Namespace, **overrides) -> RunConfig:
    kw = dict(backend=a.backend, device=a.device, workload=getattr(a, "workload", "monk"), dtype=a.dtype,
              batch=getattr(a, "batch", None), epochs=a.epochs, repeat=a.repeat, warmup=a.warmup, mode=a.mode,
              options=_parse_options(getattr(a, "option", [])), threads=getattr(a, "threads", None), seed=a.seed,
              check=not a.no_check, samples=a.samples, check_samples=getattr(a, "check_samples", 8192),
              layers=[int(x) for x in a.layers.split(",")] if a.layers else None, tag=a.tag)
    kw.update(overrides)
    return RunConfig(**kw)


def _thread_env(threads: int | None) -> dict[str, str]:
    env = dict(os.environ)
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(threads)
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
    return env


def _run_in_subprocess(cfg: RunConfig, out: Path | None) -> int:
    """Thread counts must be set before the OpenMP/BLAS runtimes initialise, so
    each thread configuration runs in a fresh interpreter."""
    payload = json.dumps({"cfg": cfg.__dict__, "out": str(out) if out else None})
    cmd = [sys.executable, "-m", "fnnbench.cli", "_child", payload]
    return subprocess.run(cmd, env=_thread_env(cfg.threads)).returncode


def cmd_child(a: argparse.Namespace) -> int:
    payload = json.loads(a.payload)
    cfg = RunConfig(**payload["cfg"])
    row = run(cfg, out=payload["out"])
    return 0 if row["check"].get("ok", True) and row["results"]["losses_finite"] else 1


def cmd_run(a: argparse.Namespace) -> int:
    cfg = _config_from_args(a)
    cfg.threads = a.threads
    out = _out_path(a.out, a.backend)
    if cfg.threads:
        return _run_in_subprocess(cfg, out)
    row = run(cfg, out=out)
    return 0 if row["check"].get("ok", True) and row["results"]["losses_finite"] else 1


def _plan_configs(plan_path: str) -> list[RunConfig]:
    plan = json.loads(Path(plan_path).read_text())
    configs = []
    for entry in plan["runs"]:
        base = dict(plan.get("defaults", {}))
        base.update(entry)
        configs.append(RunConfig(**base))
    return configs


def _selected(cfg: RunConfig, selects: list[str]) -> bool:
    """--select key=v1,v2 keeps the configurations whose field matches one value."""
    for it in selects or []:
        k, _, v = it.partition("=")
        wanted = {_literal(tok) for tok in v.split(",")}
        actual = cfg.options.get(k) if k not in cfg.__dict__ else getattr(cfg, k)
        if actual not in wanted and str(actual) not in {str(w) for w in wanted}:
            return False
    return True


def cmd_sweep(a: argparse.Namespace) -> int:
    if a.plan:
        configs = [c for c in _plan_configs(a.plan) if _selected(c, a.select)]
        # explicit --backend / --device override every plan entry (e.g. run the W4 plan,
        # written for syclnn, with cudann)
        for c in configs:
            if a.backend:
                c.backend = a.backend
            if a.device:
                c.device = a.device
    else:
        a.backend = a.backend or "syclnn"
        workloads = a.workload.split(",")
        batches = [int(b) for b in a.batch.split(",")] if a.batch else [None]
        dtypes = a.dtype.split(",")
        threads = [int(t) for t in a.threads.split(",")] if a.threads else [None]
        configs = [
            _config_from_args(a, workload=wl_, batch=b, dtype=dt, threads=th, options=opts)
            for wl_, b, dt, th, opts in itertools.product(workloads, batches, dtypes, threads, _parse_option_grid(a.option))
        ]
    done: dict[Path, set[str]] = {}
    failures = 0
    print(f"{len(configs)} configurations")
    for i, cfg in enumerate(configs, 1):
        out = _out_path(a.out, cfg.backend)
        if out and out not in done:
            done[out] = {r["id"] for r in results.read([out]) if "results" in r} if out.exists() else set()
        try:
            rid = results.run_id(cfg.effective())
        except Exception as exc:
            print(f"[{i}/{len(configs)}] cannot resolve {cfg.workload}: {exc}")
            failures += 1
            continue
        if out and rid in done[out] and not a.rerun:
            continue
        print(f"[{i}/{len(configs)}] ", end="", flush=True)
        try:
            if cfg.threads:
                rc = _run_in_subprocess(cfg, out)
                failures += 1 if rc else 0
            else:
                row = run(cfg, out=out)
                failures += 0 if (row["check"].get("ok", True) and row["results"]["losses_finite"]) else 1
        except Exception as exc:  # a failing configuration is a result too (never marked done)
            print(f"{cfg.backend} {cfg.workload} b={cfg.batch} {cfg.dtype} {cfg.options}: ERROR {exc}")
            if out:
                results.append(out, {"id": rid, "error": str(exc), "backend": cfg.backend, "workload": cfg.workload,
                                     "dtype": cfg.dtype, "batch": cfg.batch, "options": cfg.options,
                                     "threads": cfg.threads, "timestamp": sysinfo.collect()["timestamp"]})
            failures += 1
            if a.fail_fast:
                return 1
    return 1 if failures else 0


def cmd_collect(a: argparse.Namespace) -> int:
    rows = results.dedupe(r for r in results.read(a.paths) if "results" in r)
    n = results.to_csv(rows, a.output)
    print(f"{n} rows -> {a.output}")
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    n = 0
    for row in results.read([a.path]):
        if a.id and row.get("id") != a.id:
            continue
        if "results" not in row:
            continue
        cfg = RunConfig(backend=row["backend"], device=a.device or row.get("device_selector"), workload=row["workload"],
                        dtype=row["dtype"], batch=row["batch"], epochs=row["epochs"], repeat=row["repeat"],
                        warmup=row.get("warmup", 1), mode=row.get("mode", "train"), options=row.get("options", {}),
                        threads=row.get("threads"), seed=row.get("seed", 1), samples=row.get("samples_override"),
                        layers=None, tag=row.get("tag", ""))
        out = _out_path(a.out, row["backend"]) if a.out else None
        if cfg.threads:
            _run_in_subprocess(cfg, out)
        else:
            run(cfg, out=out)
        n += 1
    print(f"replayed {n} rows")
    return 0


def cmd_plot(a: argparse.Namespace) -> int:
    from .plot import plot_all

    return plot_all(a.paths, a.output)


def cmd_sysinfo(a: argparse.Namespace) -> int:
    print(json.dumps(sysinfo.collect(), indent=2))
    return 0


def cmd_workloads(a: argparse.Namespace) -> int:
    from fnn_testkit import workloads as wl

    for name, w in wl.registry().items():
        print(f"{name:24s} {'-'.join(map(str, w.spec.sizes)):22s} batch {w.batch_size:<5d} {w.dataset:10s} {w.description}")
    return 0


def cmd_peak(a: argparse.Namespace) -> int:
    from .peak import measure

    row = measure(a.backend, a.device, a.dtype, a.size, options=_parse_options(a.option))
    print(json.dumps({k: v for k, v in row.items() if k != "sysinfo"}, indent=2))
    if a.out:
        results.append(_out_path(a.out, "peak"), row)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fnnbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one configuration")
    _add_common(r)
    r.add_argument("--workload", default="monk")
    r.add_argument("--batch", type=int, default=None)
    r.add_argument("--threads", type=int, default=None, help="OMP/MKL/OpenBLAS threads (runs in a fresh process)")
    r.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="cross product of workloads x batches x dtypes x threads x options")
    _add_common(s)
    s.add_argument("--workload", default="monk", help="comma-separated")
    s.add_argument("--batch", default=None, help="comma-separated")
    s.add_argument("--threads", default=None, help="comma-separated thread counts (each in a fresh process)")
    s.add_argument("--option", action="append", default=[], metavar="KEY=V1,V2")
    s.add_argument("--plan", default=None, help="JSON plan {defaults:{...}, runs:[{...}]}")
    s.add_argument("--select", action="append", default=[], metavar="KEY=V1,V2",
                   help="keep only plan entries whose field/option matches (e.g. device=cpu, backend=cudann)")
    s.add_argument("--rerun", action="store_true", help="repeat configurations already present in --out")
    s.add_argument("--fail-fast", action="store_true")
    s.set_defaults(func=cmd_sweep, backend=None)  # no default backend: plan entries carry their own

    c = sub.add_parser("collect", help="merge JSONL files into one CSV")
    c.add_argument("paths", nargs="+")
    c.add_argument("-o", "--output", default="results.csv")
    c.set_defaults(func=cmd_collect)

    rp = sub.add_parser("replay", help="re-run rows of a JSONL file")
    rp.add_argument("path")
    rp.add_argument("--id", default=None)
    rp.add_argument("--device", default=None)
    rp.add_argument("--out", default=None)
    rp.set_defaults(func=cmd_replay)

    si = sub.add_parser("sysinfo")
    si.set_defaults(func=cmd_sysinfo)

    pl = sub.add_parser("plot", help="standard figures from JSONL results (needs matplotlib)")
    pl.add_argument("paths", nargs="+")
    pl.add_argument("-o", "--output", default="analysis/figures")
    pl.set_defaults(func=cmd_plot)

    w = sub.add_parser("workloads")
    w.set_defaults(func=cmd_workloads)

    pk = sub.add_parser("peak", help="GEMM peak and streaming bandwidth of a device through the backend")
    pk.add_argument("--backend", default="syclnn")
    pk.add_argument("--device", default=None)
    pk.add_argument("--dtype", default="float")
    pk.add_argument("--size", type=int, default=4096, help="n of the n x n x n GEMM (use 8192 on GPUs)")
    pk.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    pk.add_argument("--out", default=None)
    pk.set_defaults(func=cmd_peak)

    ch = sub.add_parser("_child", help=argparse.SUPPRESS)
    ch.add_argument("payload")
    ch.set_defaults(func=cmd_child)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
