"""fnnbench command line.

    fnnbench run   --backend syclnn --device cpu --workload mnist --batch 256 --dtype float --epochs 5 --repeat 5 --out results/ws-amd/2026-09-03/
    fnnbench sweep --backend syclnn --device gpu --workload sweep-w1024-d4-b256,mnist --batch 64,256,1024 --dtype float,double --option blas=mklcpu,netlib --out ...
    fnnbench sweep --plan plans/e1_cpu_blas.json --out ...
    fnnbench collect results/ -o results/all.csv
    fnnbench replay results/ws-amd/2026-09-03/run.jsonl --id 0123abcd
    fnnbench sysinfo
    fnnbench peak --backend syclnn --device gpu --dtype float
    fnnbench workloads
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from pathlib import Path

from . import results, sysinfo
from .runner import RunConfig, run


def _parse_options(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def _parse_option_grid(items: list[str]) -> list[dict]:
    """--option key=v1,v2 --option other=x -> list of option dicts (cross product)."""
    axes = []
    for it in items or []:
        k, _, v = it.partition("=")
        vals = []
        for tok in v.split(","):
            try:
                vals.append(ast.literal_eval(tok))
            except (ValueError, SyntaxError):
                vals.append(tok)
        axes.append([(k, x) for x in vals])
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
    p.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS for CPU runs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-check", action="store_true", help="skip the numerical sanity check")
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


def cmd_run(a: argparse.Namespace) -> int:
    cfg = RunConfig(backend=a.backend, device=a.device, workload=a.workload, dtype=a.dtype, batch=a.batch,
                    epochs=a.epochs, repeat=a.repeat, warmup=a.warmup, mode=a.mode, options=_parse_options(a.option),
                    threads=a.threads, seed=a.seed, check=not a.no_check, samples=a.samples,
                    layers=[int(x) for x in a.layers.split(",")] if a.layers else None, tag=a.tag)
    row = run(cfg, out=_out_path(a.out, a.backend))
    return 0 if row["check"].get("ok", True) else 1


def cmd_sweep(a: argparse.Namespace) -> int:
    configs: list[RunConfig] = []
    if a.plan:
        plan = json.loads(Path(a.plan).read_text())
        for entry in plan["runs"]:
            base = dict(plan.get("defaults", {}))
            base.update(entry)
            configs.append(RunConfig(**base))
    else:
        workloads = a.workload.split(",")
        batches = [int(b) for b in a.batch.split(",")] if a.batch else [None]
        dtypes = a.dtype.split(",")
        threads = [int(t) for t in a.threads.split(",")] if a.threads else [None]
        for wl_, b, dt, th, opts in itertools.product(workloads, batches, dtypes, threads, _parse_option_grid(a.option)):
            configs.append(RunConfig(backend=a.backend, device=a.device, workload=wl_, dtype=dt, batch=b,
                                     epochs=a.epochs, repeat=a.repeat, warmup=a.warmup, mode=a.mode, options=opts,
                                     threads=th, seed=a.seed, check=not a.no_check, samples=a.samples,
                                     layers=[int(x) for x in a.layers.split(",")] if a.layers else None, tag=a.tag))
    out = _out_path(a.out, a.backend)
    done = set()
    if out and out.exists() and not a.rerun:
        done = {r["id"] for r in results.read([out])}
    failures = 0
    print(f"{len(configs)} configurations, {len(done)} already in {out}" if out else f"{len(configs)} configurations")
    for i, cfg in enumerate(configs, 1):
        rid = results.run_id(cfg.key())
        if rid in done:
            continue
        print(f"[{i}/{len(configs)}] ", end="")
        try:
            row = run(cfg, out=out)
            failures += 0 if row["check"].get("ok", True) else 1
        except Exception as exc:  # a failing configuration is a result too
            print(f"{cfg.backend} {cfg.workload} b={cfg.batch} {cfg.dtype} {cfg.options}: ERROR {exc}")
            if out:
                results.append(out, {"id": rid, "error": str(exc), "backend": cfg.backend, "workload": cfg.workload,
                                     "dtype": cfg.dtype, "batch": cfg.batch, "options": cfg.options,
                                     "timestamp": sysinfo.collect()["timestamp"]})
            failures += 1
            if a.fail_fast:
                return 1
    return 1 if failures else 0


def cmd_collect(a: argparse.Namespace) -> int:
    rows = [r for r in results.read(a.paths) if "results" in r]
    n = results.to_csv(rows, a.output)
    print(f"{n} rows -> {a.output}")
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    for row in results.read([a.path]):
        if a.id and row.get("id") != a.id:
            continue
        if "results" not in row:
            continue
        cfg = RunConfig(backend=row["backend"], device=a.device or row.get("device", {}).get("name"), workload=row["workload"],
                        dtype=row["dtype"], batch=row["batch"], epochs=row["epochs"], repeat=row["repeat"],
                        warmup=row.get("warmup", 1), mode=row.get("mode", "train"), options=row.get("options", {}),
                        threads=row.get("threads"), seed=row.get("seed", 1), layers=row.get("layers"))
        run(cfg, out=_out_path(a.out, row["backend"]) if a.out else None)
    return 0


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
    r.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="cross product of workloads x batches x dtypes x threads x options")
    _add_common(s)
    s.add_argument("--workload", default="monk", help="comma-separated")
    s.add_argument("--batch", default=None, help="comma-separated")
    s.add_argument("--option", action="append", default=[], metavar="KEY=V1,V2")
    s.add_argument("--plan", default=None, help="JSON plan {defaults:{...}, runs:[{...}]}")
    s.add_argument("--rerun", action="store_true", help="repeat configurations already present in --out")
    s.add_argument("--fail-fast", action="store_true")
    s.set_defaults(func=cmd_sweep)

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

    w = sub.add_parser("workloads")
    w.set_defaults(func=cmd_workloads)

    pk = sub.add_parser("peak", help="GEMM peak and streaming bandwidth of a device through the backend")
    pk.add_argument("--backend", default="syclnn")
    pk.add_argument("--device", default=None)
    pk.add_argument("--dtype", default="float")
    pk.add_argument("--size", type=int, default=4096)
    pk.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    pk.add_argument("--out", default=None)
    pk.set_defaults(func=cmd_peak)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
