#!/usr/bin/env bash
# Re-measure every GPU-device row of ws-amd's third pass with the host CPU at the fixed 2.8 GHz
# (boost off), so the CPU-device and GPU-device rows share one host clock: the reference
# workloads are launch-bound (docs/methodology.md, "the two regimes") and their absolute times
# depend on the submitting thread's clock. Covers the ROCm block (syclnn/AdaptiveCpp + cudann/HIP:
# E2 gpu, E3, E6 gpu, W4, E7, E5 inference/breakdown, GEMM peaks), the OpenMP offload block
# (amdclang++, gcc-14 amdgcn), the AdaptiveCpp portability GPU rows, the GEMM size probe and
# the ZLUDA cross-check. Old rows -> superseded.jsonl (ZLUDA: rows-stock-clock/, README-stock-clock.md).
# Requires the cap in force and scripts/thermal-guard.sh running (ws-amd-cpu-fixed-clock.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATE=${1:-2026-09-05}
RES=results/ws-amd/$DATE
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
if [ "$cap" != 2800000 ] || [ "$boost" != 0 ]; then echo "CPU cap not in force (max $cap kHz, boost $boost): refusing to run"; exit 1; fi
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running: refusing to run"; exit 1; }

log "fixed clock (GPU rows): supersede every GPU-device row measured with the host at stock clocks"
for d in sycl-gpu-acpp gpu-cuda-vs-sycl sycl-gpu-ablations gpu-w4-sycl gpu-w4-cuda e7-tiled-gpu e5-infer-gpu e5-breakdown-gpu \
         omp-gpu-amdclang omp-gpu-gcc14amd e7-tiled-omp-gpu-amdclang e7-tiled-omp-gpu-gcc14amd e5-infer-omp-gpu-amdclang e5-infer-omp-gpu-gcc14amd \
         e5-breakdown-omp-gpu-amdclang e5-breakdown-omp-gpu-gcc14amd; do
    [ -d $RES/$d ] && $PY scripts/supersede-stale.py $RES/$d --all
done
$PY - <<EOF
import json
from pathlib import Path
def move(path, pred, label):
    p = Path(path)
    if not p.exists(): return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    drop = [r for r in rows if pred(r)]; keep = [r for r in rows if not pred(r)]
    if drop:
        with (p.parent / "superseded.jsonl").open("a") as fh:
            for r in drop: fh.write(json.dumps(r) + "\n")
        p.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"{label}: kept {len(keep)}, superseded {len(drop)}")
gpu = lambda r: (r.get("device") or {}).get("type") == "gpu"
move("$RES/portable-acpp/syclnn.jsonl", gpu, "portable-acpp GPU rows")
for d in ("peaks", "peaks-tiled", "peaks-omp-amdclang", "peaks-omp-gcc14amd", "peaks-tiled-omp-amdclang", "peaks-sizes"):
    move(f"$RES/{d}/peak.jsonl", gpu, f"{d} GPU peak rows")
EOF
if [ -d results/ws-amd/zluda/rows ]; then rm -rf results/ws-amd/zluda/rows-stock-clock; mv results/ws-amd/zluda/rows results/ws-amd/zluda/rows-stock-clock; cp results/ws-amd/zluda/README.md results/ws-amd/zluda/README-stock-clock.md; fi

log "1. RX 7900 XTX: syclnn/AdaptiveCpp + cudann/HIP (E2 gpu, E3, E6 gpu, W4, E7, E5, peaks) and the OpenMP offload block"
bash scripts/matrix-ws-amd.sh $DATE rocm-gpu omp-gpu

log "2. AdaptiveCpp portability GPU rows and the GEMM size probe"
distrobox enter fnn-rocm -- bash -c "
    P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE
    for b in tiled auto; do for wl in 'monk 40' 'cup 40' 'mnist-512-256 256'; do set -- \$wl; fnnbench run --backend syclnn --device gpu --workload \$1 --batch \$2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=\$b --out $RES/portable-acpp 2>&1 | grep -E 'epoch|rror' | tail -1 | cut -c1-150; done; done
    for s in 512 2048 8192; do for be in syclnn cudann; do fnnbench peak --backend \$be --device gpu --dtype float --size \$s --out $RES/peaks-sizes 2>&1 | grep -E 'gemm_gflops|rror' | head -1; done; done
"

log "3. ZLUDA cross-check"
distrobox enter fnn-rocm -- bash -lc "bash $HERE/scripts/zluda-crosscheck.sh" 2>&1 | grep -E 'cudann/ZLUDA|written' | cut -c1-150

log "4. supersede older duplicates, analyze, headline"
$PY scripts/supersede-stale.py $RES --dupes
bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py $RES > results/analysis/headline-ws-amd-$DATE.md && head -20 results/analysis/headline-ws-amd-$DATE.md
log "fixed-clock GPU rows done"
