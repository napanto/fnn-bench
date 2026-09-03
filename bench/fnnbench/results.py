"""JSONL result rows: one row per benchmark run, self-describing and replayable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


class _Encoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dumps(row: dict[str, Any]) -> str:
    return json.dumps(row, cls=_Encoder, sort_keys=True)


def run_id(config: dict[str, Any]) -> str:
    """Stable id of a configuration (backend, device, workload, options...)."""
    return hashlib.sha1(dumps(config).encode()).hexdigest()[:12]


def append(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(dumps(row) + "\n")


def read(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    """One flat dict per row for tables / pandas."""
    r = row.get("results", {})
    prof = r.get("profile", {}) or {}
    flat = {
        "id": row.get("id"),
        "tag": row.get("tag"),
        "timestamp": row.get("timestamp"),
        "host": (row.get("sysinfo") or {}).get("hostname"),
        "backend": row.get("backend"),
        "version": (row.get("build_info") or {}).get("version"),
        "compiler": (row.get("build_info") or {}).get("compiler"),
        "device": (row.get("device") or {}).get("name"),
        "device_type": (row.get("device") or {}).get("type"),
        "blas": row.get("blas"),
        "dtype": row.get("dtype"),
        "workload": row.get("workload"),
        "layers": "-".join(map(str, row.get("layers", []))),
        "batch": row.get("batch"),
        "epochs": row.get("epochs"),
        "mode": row.get("mode"),
        "threads": row.get("threads"),
        "threads_effective": row.get("threads_effective"),
        "n_samples": row.get("n_samples"),
        "seed": row.get("seed"),
        "options": dumps(row.get("options", {})),
        "driver": (row.get("device") or {}).get("driver"),
        "git_bench": ((row.get("sysinfo") or {}).get("git", {}).get("fnn-bench") or {}).get("sha"),
        "git_lib": (row.get("build_info") or {}).get("git_sha"),
        "median_epoch_s": r.get("median_epoch_s"),
        "iqr_epoch_s": r.get("iqr_epoch_s"),
        "steady_epoch_s": r.get("steady_epoch_s"),
        "samples_per_s": r.get("samples_per_s"),
        "steps_per_s": r.get("steps_per_s"),
        "intensity": r.get("intensity_flop_per_byte"),
        "losses_finite": r.get("losses_finite"),
        "gflops_gemm": r.get("gflops_gemm"),
        "gflops_total": r.get("gflops_total"),
        "check_ok": (row.get("check") or {}).get("ok"),
    }
    for k, v in prof.items():
        if k.endswith("_ns"):
            flat[f"prof_{k[:-3]}_ms"] = v / 1e6 if isinstance(v, (int, float)) else None
        elif k in ("launches", "batches", "epochs", "unprofiled", "bytes_h2d", "bytes_d2h"):
            flat[f"prof_{k}"] = v
    for k, v in (r.get("profile_per_epoch_ms") or {}).items():
        flat[f"epoch_{k[:-3]}_ms"] = v
    return flat


def to_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    import csv

    flats = [flatten(r) for r in rows]
    if not flats:
        return 0
    keys = sorted({k for f in flats for k in f})
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for f in flats:
            w.writerow(f)
    return len(flats)
