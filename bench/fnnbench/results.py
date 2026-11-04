"""JSONL result rows: one row per benchmark run, self-describing and replayable."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

class _Encoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        import numpy as np

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
            if f.name == "superseded.jsonl":  # rows explicitly replaced by a re-measurement
                continue
            with f.open() as fh:
                for number, line in enumerate(fh, 1):
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        row["_source"], row["_line"] = str(f), number
                        parts = f.parts
                        if "results" in parts:
                            pos = parts.index("results")
                            if len(parts) > pos + 2:
                                row["_machine"], row["_pass"] = parts[pos + 1:pos + 3]
                        yield row


def environment_key(row: dict[str, Any]) -> str:
    """Recorded execution conditions; unknown values never equal recorded ones."""
    si = row.get("sysinfo") or {}
    prefixes = ("OMP_", "GOMP_", "KMP_", "DPCPP_", "MKL_", "OPENBLAS_", "TBB_", "SYCL_", "ACPP_", "CUDA_", "HIP_")
    return dumps({"machine": row.get("_machine"), "pass": row.get("_pass"),
                  "backend": row.get("backend"), "device": row.get("device"),
                  "cpu_model": (si.get("cpu") or {}).get("model"),
                  "build_info": row.get("build_info"), "libs": sorted(si.get("libs") or []),
                  "package_version": (si.get("packages") or {}).get(row.get("backend")),
                  "env": {k: v for k, v in (si.get("env") or {}).items()
                          if k.startswith(prefixes) or k == "LD_LIBRARY_PATH"},
                  "numa": si.get("numa"), "threads": row.get("threads"),
                  "threads_effective": row.get("threads_effective")})


def configuration_key(row: dict[str, Any]) -> str:
    """Identity for repeated measurements, including the complete timing protocol."""
    opts = dict(row.get("options") or {})
    if opts.get("blas") == "auto":
        opts.pop("blas")
    if opts.get("profile") is False:
        opts.pop("profile")
    fields = ("blas", "workload", "dtype", "batch", "n_samples", "epochs", "repeat", "warmup", "seed", "layers")
    return dumps({**{k: row.get(k) for k in fields}, "environment": environment_key(row),
                  "mode": row.get("mode", "train"), "options": opts,
                  "effective_options": row.get("effective_options")})


def dedupe(rows: Iterable[dict[str, Any]], *, supersede: bool = False) -> list[dict[str, Any]]:
    """Remove copies of the same record, preserving independent repeated sweeps.

    Re-measurement is explicit: archive obsolete rows in superseded.jsonl, or
    request supersede=True when preparing that archive. Analysis never infers
    replacement from timestamps alone. Metadata added by readers is not identity.
    """
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = configuration_key(r) if supersede else dumps({k: v for k, v in r.items() if not k.startswith("_")})
        if key not in best or supersede and (r.get("timestamp") or "") > (best[key].get("timestamp") or ""):
            best[key] = r
    return list(best.values())


def _finite(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_finite(v, positive=positive) for v in value)
    return isinstance(value, (int, float)) and math.isfinite(value) and (not positive or value > 0)


def valid(row: dict[str, Any]) -> bool:
    """Shared validity rule for plots and headline tables; old absent checks pass."""
    r = row.get("results") or {}
    if row.get("error") or not r or (row.get("check") or {}).get("ok") is False:
        return False
    if r.get("losses_finite") is False or not _finite(r.get("losses") or []):
        return False
    if not _finite(r.get("median_epoch_s"), positive=True):
        return False
    return all(r.get(k) is None or _finite(r[k], positive=True)
               for k in ("steady_epoch_s", "steady_epochs_s"))


def epoch_s(row: dict[str, Any], metric: str = "steady") -> float:
    """Inference always uses the complete predict call; training may use steady."""
    if row.get("_replicates"):
        return statistics.median(epoch_s(r, metric) for r in row["_replicates"])
    r = row["results"]
    if row.get("mode", "train") == "train" and metric == "steady":
        if r.get("steady_epochs_s"):
            return statistics.median(r["steady_epochs_s"])
        if r.get("steady_epoch_s"):
            return float(r["steady_epoch_s"])
    return float(r["median_epoch_s"])


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median of valid per-run measurements, with their count and provenance.

    Runs get equal weight; epoch samples from independent processes are never
    pooled. Within-run IQRs are retained on the replicates, not relabelled as an
    IQR of this aggregate. Callers can inspect _replicates for uncertainty.
    """
    groups: dict[str, list] = defaultdict(list)
    for r in dedupe(r for r in rows if valid(r)):
        groups[configuration_key(r)].append(r)
    out = []
    for rs in groups.values():
        row = dict(max(rs, key=lambda r: r.get("timestamp") or ""))
        res = dict(row["results"])
        for key in ("median_epoch_s", "steady_epoch_s", "samples_per_s", "steady_samples_per_s"):
            values = [r["results"][key] for r in rs if r["results"].get(key) is not None]
            if values:
                res[key] = statistics.median(values)
        profiles = [r["results"].get("profile_per_epoch_ms") for r in rs]
        if all(profiles):
            res["profile_per_epoch_ms"] = {k: statistics.median(p.get(k, 0) for p in profiles)
                                           for k in set().union(*profiles)}
        row.update(results=res, aggregate_count=len(rs), _replicates=rs)
        out.append(row)
    return out


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
        "effective_options": dumps(row.get("effective_options")),
        "aggregate_count": row.get("aggregate_count", 1),
        "driver": (row.get("device") or {}).get("driver"),
        "git_bench": ((row.get("sysinfo") or {}).get("git", {}).get("fnn-bench") or {}).get("sha"),
        "git_lib": (row.get("build_info") or {}).get("git_sha"),
        "median_epoch_s": r.get("median_epoch_s"),
        "iqr_epoch_s": r.get("iqr_epoch_s"),
        "steady_epoch_s": r.get("steady_epoch_s"),
        "steady_epochs_s": dumps(r.get("steady_epochs_s")),
        "steady_iqr_s": r.get("steady_iqr_s"),
        "inference_whole_predict_s": r.get("median_epoch_s") if row.get("mode") == "infer" else None,
        "inference_library_internal_predict_s": r.get("steady_epoch_s") if row.get("mode") == "infer" else None,
        "steady_samples_per_s": r.get("steady_samples_per_s"),
        "gpu_util_pct": (r.get("gpu_monitor") or {}).get("util_pct_mean"),
        "gpu_power_w": (r.get("gpu_monitor") or {}).get("power_w_mean"),
        "gpu_energy_j": (r.get("gpu_monitor") or {}).get("energy_j"),
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
