#!/usr/bin/env bash
# Steps 2-4 of ws-amd-cpu-fixups.sh, after its step 2 had run the "AdaptiveCpp CPU" sweeps in the
# wrong environment (the generic DPC++ image; the AdaptiveCpp host device lives in the fnn-rocm
# distrobox venv). The E1 re-measure (step 1) and the supersede prelude are done.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATE=${1:-2026-09-05}
RES=results/ws-amd/$DATE
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

log "2. AdaptiveCpp host device at 16 threads (E2 cpu, E5 breakdown, E7 tiled), fnn-rocm venv"
distrobox enter fnn-rocm -- bash -c "
    P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --select dtype=float --out $RES/sycl-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    fnnbench sweep --plan plans/e5_breakdown.json --select device=cpu --select backend=syclnn --select dtype=float --out $RES/e5-breakdown-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=cpu --select backend=syclnn --out $RES/e7-tiled-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
"

log "3. ompnn host path: rebuild (OpenBLAS rpath) and re-measure E4 + peaks + gcc-14 E5/E7"
bash scripts/matrix-ws-amd.sh $DATE omp-cpu

log "4. supersede older duplicates, analyze, headline"
$PY scripts/supersede-stale.py $RES --dupes
bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py $RES > analysis/headline-ws-amd-$DATE.md && head -20 analysis/headline-ws-amd-$DATE.md
log "cpu fix-ups done"
