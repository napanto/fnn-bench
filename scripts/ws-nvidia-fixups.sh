#!/usr/bin/env bash
# ws-nvidia CPU fix-ups (2026-09-05), the same two defects found on ws-amd (docs/methodology.md,
# "CPU rows: thread counts, affinity and the OpenBLAS variant"): the E4 thread rows ran on one
# inherited core, and the ompnn wheel loaded the pthread OpenBLAS. Re-syncs the harness (child
# affinity fix) and ompnn (rpath fix), rebuilds the clang-22 host wheel, re-measures
# omp-cpu-clang22 (+ a CPU peak), fetches the rows back and re-runs the ws-nvidia analysis.
# Run from ws-amd once every other ws-nvidia step is done.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
HOST=${NVIDIA_HOST:-ws-nvidia}
REMOTE=${NVIDIA_DIR:-$HOME/fnn}
DATE=${1:-2026-09-05}
RUN="podman run --rm --memory=40g --security-opt=label=disable -v $REMOTE/ompnn:/work/ompnn -v $REMOTE/fnn-bench:/work/fnn-bench"
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

log "ws-nvidia fix-ups: fetch ws-nvidia's rows, then sync the harness and ompnn (never the results)"
rsync -a "$HOST:$REMOTE/fnn-bench/results/ws-nvidia/" "$HERE/results/ws-nvidia/"
rsync -a --delete --exclude build --exclude .venv --exclude .wheels --exclude __pycache__ --exclude results --exclude .cache --exclude analysis "$HERE/" "$HOST:$REMOTE/fnn-bench/"
rsync -a --delete --exclude build --exclude __pycache__ "$AC/ompnn/" "$HOST:$REMOTE/ompnn/"

log "ws-nvidia: rebuild the clang-22 host wheel (OpenBLAS rpath), re-measure E4 cpu + peak"
ssh "$HOST" "$RUN -w /work/ompnn localhost/fnn-cuda:dev bash -c '
    export CXX=clang++-22 OMPNN_TARGET=cpu OMPNN_OFFLOAD_ARCH= OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CMAKE_BUILD_PARALLEL_LEVEL=8
    rm -rf /work/fnn-bench/.wheels/ws-nvidia-omp-clang22
    pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-omp-clang22 --config-settings=build-dir=/tmp/b-clang22 . 2>&1 | grep -E \"error:\" || true
    objdump -p /work/fnn-bench/.wheels/ws-nvidia-omp-clang22/ompnn/_ompnn*.so | grep -E \"RPATH|RUNPATH\"
    export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    cd /work/fnn-bench; export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-omp-clang22
    python scripts/supersede-stale.py results/ws-nvidia/$DATE/omp-cpu-clang22 --all
    [ -f results/ws-nvidia/$DATE/peaks-omp-clang22/peak.jsonl ] && cat results/ws-nvidia/$DATE/peaks-omp-clang22/peak.jsonl >> results/ws-nvidia/$DATE/peaks-omp-clang22/superseded.jsonl && rm results/ws-nvidia/$DATE/peaks-omp-clang22/peak.jsonl
    fnnbench sweep --plan plans/e4_omp_cpu.json --out results/ws-nvidia/$DATE/omp-cpu-clang22
    fnnbench peak --backend ompnn --device cpu --dtype float --size 4096 --out results/ws-nvidia/$DATE/peaks-omp-clang22
    python - <<EOF
import json
r = json.loads(open(\"results/ws-nvidia/$DATE/omp-cpu-clang22/ompnn.jsonl\").readlines()[-1])
print(\"libs:\", [l for l in r[\"sysinfo\"].get(\"libs\", []) if \"blas\" in l], \"cpuset:\", r[\"sysinfo\"][\"cpu\"].get(\"cpuset_effective\"))
EOF
    python scripts/supersede-stale.py results/ws-nvidia/$DATE --dupes'"

log "ws-nvidia fix-ups: fetch, analyze, headline"
rsync -a "$HOST:$REMOTE/fnn-bench/results/ws-nvidia/" "$HERE/results/ws-nvidia/"
bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py results/ws-nvidia/$DATE > analysis/headline-ws-nvidia-$DATE.md && head -20 analysis/headline-ws-nvidia-$DATE.md
log "ws-nvidia fix-ups done"
