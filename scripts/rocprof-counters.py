#!/usr/bin/env python3
"""Collect available gfx1100 counters on the existing HIP/AdaptiveCpp/amdclang wheels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from experiment_utils import run_command
from gemm_shapes import driver_args, shapes

CANDIDATES = {
    "lds-conflicts": ("LDSBankConflict", "SQ_LDS_BANK_CONFLICT"),
    "lds-instructions": ("LDSInsts", "SQ_INSTS_LDS"),
    "valu-utilization": ("VALUUtilization",),
    "valu-busy": ("VALUBusy",),
    "valu-instructions": ("VALUInsts",),
    "waves": ("Wavefronts", "SQ_WAVES"),
    "occupancy": ("OccupancyPercent", "MeanOccupancyPerCU", "Occupancy"),
}


def plan(available):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", available.read_text())
    has = lambda name: re.search(r"(?<![A-Za-z0-9_])"+re.escape(name)+r"(?![A-Za-z0-9_])", text) is not None
    groups = {group: [next(c for c in names if has(c))] for group, names in CANDIDATES.items() if any(has(c) for c in names)}
    if has("FETCH_SIZE") and has("WRITE_SIZE"):
        groups["memory-read"] = ["FETCH_SIZE"]
        groups["memory-write"] = ["WRITE_SIZE"]
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cudann", "syclnn", "ompnn"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--blas", choices=("auto", "tiled"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--size", choices=("mnist", "large", "both"), default="both")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    groups = plan(a.out / "list-avail.txt")
    extra = None
    if groups.get("lds-instructions") == ["SQ_INSTS_LDS"]:
        # The exposed raw counter has WGP/SA/SE instances; their sum is a count,
        # unlike rates such as LDSBankConflict which must never be summed.
        extra = a.out / "extra-counters.yaml"
        extra.write_text("rocprofiler-sdk:\n  counters-schema-version: 1\n  counters:\n"
                         "    - name: FNN_LDS_INSTS\n      description: Sum of exposed SQ_INSTS_WAVE32_LDS across hardware instances\n"
                         "      properties: []\n      definitions:\n        - architectures: [gfx1100]\n"
                         "          expression: reduce(SQ_INSTS_WAVE32_LDS,sum)\n"
                         "    - name: FNN_LDS_BASE\n      description: Sum of the base SQ_INSTS_LDS counter\n"
                         "      properties: []\n      definitions:\n        - architectures: [gfx1100]\n"
                         "          expression: reduce(SQ_INSTS_LDS,sum)\n")
        groups["lds-instructions"] = ["FNN_LDS_INSTS"]
        groups["lds-instructions-base"] = ["FNN_LDS_BASE"]
    if a.groups:
        groups = {k: v for k, v in groups.items() if k in a.groups}
    if not groups:
        raise SystemExit("No requested counter category is exposed; see list-avail.txt")
    (a.out / "counter-plan.json").write_text(json.dumps(groups, indent=2)+"\n")
    for size in (("mnist", "large") if a.size == "both" else (a.size,)):
        for group, metrics in groups.items():
            target = a.out / f"{a.label}-{a.blas}-{size}" / group
            target.mkdir(parents=True, exist_ok=True)
            if (target / "exit.txt").exists() and (target / "exit.txt").read_text().strip() == "0":
                continue
            cmd = ["rocprofv3", "--marker-trace", "--kernel-trace", "--pmc", *metrics,
                   "--kernel-include-regex", ".*(gemm|Gemm|GEMM|Cijk).*", "--output-format", "csv", "json",
                   "-d", str(target), "-o", "run", "--", *driver_args(a.backend, a.blas, size, target/"driver.json", "hip")]
            if extra:
                cmd[1:1] = ["--extra-counters", str(extra)]
            manifest = dict(command=cmd, backend=a.backend, label=a.label, blas=a.blas, size=size, group=group,
                            counters=metrics, shapes_in_submission_order=shapes(size))
            env = dict(os.environ)
            if a.backend == 'ompnn':
                # Hardware dispatch counters do not need OMPT. Its automatic
                # startup re-enters libomptarget from OpenBLAS's constructor.
                env['OMP_TOOL'] = 'disabled'
                manifest['profiling_environment'] = {'OMP_TOOL': 'disabled'}
            (target/"command.json").write_text(json.dumps(manifest, indent=2)+"\n")
            with (target/"run.log").open("w") as fh:
                fh.write("$ "+shlex.join(cmd)+"\n"); fh.flush()
                code = run_command(cmd, fh, a.timeout, env)
            (target/"exit.txt").write_text(str(code)+"\n")
            print(target, "exit", code, flush=True)
            # Access denial is not fixed by rerunning every model and metric.
            if "permission denied" in (target/"run.log").read_text().lower():
                raise SystemExit("Counter permission denied; retained log, no host settings changed")


if __name__ == "__main__":
    main()
