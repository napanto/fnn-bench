#!/usr/bin/env bash
# CPU cross-validation of the in-library profiler with `perf` (Linux, no root:
# perf_event_paranoid=2 allows user-space sampling of the user's own processes).
#
# Runs one profiled CPU measurement per backend inside its environment, attaches
# `perf record` from the host to the process, and compares the profiler's phase
# shares (GEMM / element-wise / update / other) with perf's symbol-level shares
# (BLAS symbols vs the libraries' own kernels vs runtime). Output:
#   results/ws-amd/perf/<backend>-<workload>.{perf.txt,profile.json,summary.txt}
#
#   fnn-bench/scripts/perf-crosscheck.sh [workload] [batch]      # default mnist-512-256 256
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
WL=${1:-mnist-512-256}
B=${2:-256}
OUT=$HERE/results/ws-amd/perf
mkdir -p "$OUT"
P=$HOME/.local/opt/fnn-rocm

run_one() { # name, venv, backend, extra options
    local name=$1 venv=$2 be=$3; shift 3
    local log=$OUT/$name-$WL
    echo "== $name ($be, $WL b=$B)"
    # the measurement, in the distrobox, with the profiler on; PID exported through a file
    distrobox enter fnn-rocm -- bash -c "
        export PATH=$P/$venv/bin:\$PATH OMP_NUM_THREADS=\${OMP_NUM_THREADS:-24} OMP_PROC_BIND=close OMP_PLACES=cores
        cd $HERE && echo \$\$ > $log.pid && exec fnnbench run --backend $be --device cpu --workload $WL --batch $B --dtype float \
            --epochs 3 --repeat 3 --warmup 1 --option profile=True $* --no-check --out $OUT/rows --tag perf-$name" &
    local job=$!
    sleep 4
    local pid
    pid=$(pgrep -P "$(cat $log.pid)" -f fnnbench || cat $log.pid)
    # user-space samples of the whole process (all threads), 1 kHz, DWARF-less (frame pointers are absent in MKL/rocBLAS: flat profile is enough)
    perf record -e cpu-clock -F 1000 -p "$pid" -o "$log.perf.data" -- sleep 20 >/dev/null 2>&1 || \
        perf record -e cpu-clock -F 1000 -p "$pid" -o "$log.perf.data" -- sleep 20
    wait $job || true
    perf report -i "$log.perf.data" --stdio --sort dso,symbol --percent-limit 0.5 2>/dev/null > "$log.perf.txt" || true
    # aggregate perf shares by DSO class
    python3 - "$log" <<'PY'
import re, sys, json, pathlib
log = sys.argv[1]
txt = pathlib.Path(log + ".perf.txt").read_text()
cls = {"blas": 0.0, "library kernels": 0.0, "runtime": 0.0, "python/numpy": 0.0, "other": 0.0}
for m in re.finditer(r"^\s+([\d.]+)%\s+(\S+)\s+\[.\]\s+(.*)$", txt, re.M):
    pct, dso, sym = float(m.group(1)), m.group(2), m.group(3)
    d = dso.lower()
    if any(k in d for k in ("mkl", "openblas", "blas", "onemath", "rocblas")):
        cls["blas"] += pct
    elif any(k in d for k in ("_syclnn", "_ompnn", "_cudann", "acpp", "hipsycl", "libomp", "libgomp", "libsycl", "libur_", "intelocl", "libtbb", "libamd", "libhsa", "kernel")):
        cls["library kernels" if ("_syclnn" in d or "_ompnn" in d or "kernel" in d or "acpp-jit" in d) else "runtime"] += pct
    elif any(k in d for k in ("python", "numpy", "libc", "ld-linux")):
        cls["python/numpy"] += pct
    else:
        cls["other"] += pct
pathlib.Path(log + ".summary.txt").write_text(json.dumps(cls, indent=1) + "\n")
print("  perf shares:", cls)
PY
}

run_one acpp-cpu venv syclnn
run_one omp-amdclang-cpu venv-amdclang ompnn
echo "profiler rows: $OUT/rows/*.jsonl (results.profile_per_epoch_ms)"
