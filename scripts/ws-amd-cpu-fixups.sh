#!/usr/bin/env bash
# Pass-3 CPU fix-ups on ws-amd (2026-09-05), after three confounders were found in the CPU rows:
#   1. thread-scaling rows (E1 syclnn, E4 ompnn) ran in child processes that inherited the
#      parent's libgomp-pinned single core; the DPC++ OpenCL CPU device also ignores
#      OMP_NUM_THREADS (needs DPCPP_CPU_NUM_CUS)              -> harness fixed, rows re-measured
#   2. ompnn's OpenBLAS wheels resolved libopenblas.so.0 to the distro *pthread* variant
#      although the build pointed at the OpenMP one (two thread pools contend, 1.5x)
#                                                             -> rpath fixed, wheels rebuilt, E4/E5/E7/peaks re-measured
#   3. AdaptiveCpp host-device sweeps ran with 32 threads (no OMP_NUM_THREADS in the generic image)
#      while everything else uses the 16 physical cores       -> re-measured at 16 threads
# Old rows go to superseded.jsonl (kept, skipped by the readers).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATE=${1:-2026-09-05}
RES=results/ws-amd/$DATE
PODMAN="podman run --rm --security-opt=label=disable --device /dev/dri --device /dev/kfd -v $HERE/../syclnn:/work/syclnn -v $HERE/../cudann:/work/cudann -v $HERE/../ompnn:/work/ompnn -v $HERE:/work/fnn-bench"
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

log "cpu fix-ups: supersede the affected rows"
$PY scripts/supersede-stale.py $RES/sycl-cpu-threads --all
for d in omp-cpu-gcc14 omp-cpu-clang18 omp-cpu-clang22 omp-cpu-gcc14mkl e5-infer-omp-cpu-gcc14 e5-breakdown-omp-cpu-gcc14 e7-tiled-omp-cpu-gcc14 sycl-cpu-acpp e5-breakdown-cpu-acpp e7-tiled-cpu-acpp; do
    [ -d $RES/$d ] && $PY scripts/supersede-stale.py $RES/$d --all
done
for d in peaks-omp-gcc14 peaks-omp-clang18 peaks-omp-clang22 peaks-omp-gcc14mkl; do   # peak rows have their own schema: move them by hand
    [ -f $RES/$d/peak.jsonl ] && cat $RES/$d/peak.jsonl >> $RES/$d/superseded.jsonl && rm $RES/$d/peak.jsonl
done

log "1. DPC++ CPU thread scaling (E1) with DPCPP_CPU_NUM_CUS and the full cpuset in the child"
$PODMAN -w /work/fnn-bench localhost/fnn-sycl:dev bash -c "
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    export PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp OMP_PROC_BIND=close OMP_PLACES=cores FNN_REF_CACHE=/work/fnn-bench/.cache/ref
    fnnbench sweep --plan plans/e1_cpu_threads.json --out $RES/sycl-cpu-threads
"

log "2. AdaptiveCpp host device at 16 threads (E2 cpu, E5 breakdown, E7 tiled)"
$PODMAN -w /work/fnn-bench localhost/fnn-sycl-generic:dev bash -c "
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    export PYTHONPATH=/work/fnn-bench/.wheels/sycl-generic OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 FNN_REF_CACHE=/work/fnn-bench/.cache/ref
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --select dtype=float --out $RES/sycl-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp W'
    fnnbench sweep --plan plans/e5_breakdown.json --select device=cpu --select backend=syclnn --select dtype=float --out $RES/e5-breakdown-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp W'
    fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=cpu --select backend=syclnn --out $RES/e7-tiled-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp W'
"

log "3. ompnn host path: rebuild (OpenBLAS rpath) and re-measure E4 + peaks + gcc-14 E5/E7"
bash scripts/matrix-ws-amd.sh $DATE omp-cpu

log "4. supersede older duplicates, analyze, headline"
$PY scripts/supersede-stale.py $RES --dupes
bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py $RES > analysis/headline-ws-amd-$DATE.md && head -20 analysis/headline-ws-amd-$DATE.md
log "cpu fix-ups done"
