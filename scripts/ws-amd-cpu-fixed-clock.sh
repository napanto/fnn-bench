#!/usr/bin/env bash
# Re-measure every CPU-device row of ws-amd's third pass at a fixed CPU clock.
#
# Why: the Threadripper 2950X reaches its throttle point (Tdie 68 C / Tctl 95 C, spikes to 99 C)
# within 20 s of any 16-thread load, so the CPU rows measured on 2026-09-05 ran at whatever clock
# kept the die at 68 C (a backend-dependent, uncontrolled clock), and the host froze at 18:22
# under the generic SYCL BLAS sweep. At 2.8 GHz with boost off the same load holds Tctl 75 C
# (docs/methodology.md, "CPU rows at a fixed clock").
#
# Requires the cap in force (volatile, re-apply after a reboot):
#   sudo sh -c 'cpupower frequency-set -u 2.8GHz && echo 0 > /sys/devices/system/cpu/cpufreq/boost'
# and scripts/thermal-guard.sh running. Old rows -> superseded.jsonl.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${1:-2026-09-05}
RES=results/ws-amd/$DATE
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
if [ "$cap" != 2800000 ] || [ "$boost" != 0 ]; then echo "CPU cap not in force (max $cap kHz, boost $boost): refusing to run"; exit 1; fi
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running: refusing to run"; exit 1; }

log "fixed clock: supersede every CPU-device row measured at the thermal ceiling"
for d in sycl-cpu sycl-cpu-threads sycl-cpu-ablations sycl-cpu-generic sycl-cpu-w4 e7-tiled-cpu-dpcpp e5-infer-cpu-dpcpp e5-breakdown-cpu-dpcpp portable-dpcpp \
         sycl-cpu-acpp e5-breakdown-cpu-acpp e7-tiled-cpu-acpp \
         omp-cpu-gcc14 omp-cpu-clang18 omp-cpu-clang22 omp-cpu-gcc14mkl e5-infer-omp-cpu-gcc14 e5-breakdown-omp-cpu-gcc14 e7-tiled-omp-cpu-gcc14; do
    [ -d $RES/$d ] && $PY scripts/supersede-stale.py $RES/$d --all
done
$PY - <<EOF
# mixed directories: the CPU rows of the AdaptiveCpp portability run and the CPU peak rows
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
cpu = lambda r: (r.get("device") or {}).get("type") == "cpu"
move("$RES/portable-acpp/syclnn.jsonl", cpu, "portable-acpp CPU rows")
for d in ("peaks", "peaks-omp-gcc14", "peaks-omp-clang18", "peaks-omp-clang22", "peaks-omp-gcc14mkl"):
    move(f"$RES/{d}/peak.jsonl", cpu, f"{d} CPU peak rows")
EOF

log "1. DPC++ CPU blocks (MKLCPU/Netlib, then the generic SYCL BLAS image)"
bash scripts/matrix-ws-amd.sh $DATE sycl-cpu sycl-generic

log "2. AdaptiveCpp host device (E2 cpu, E5 breakdown, E7 tiled) + its portability CPU rows, fnn-rocm venv"
distrobox enter fnn-rocm -- bash -c "
    P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --select dtype=float --out $RES/sycl-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    fnnbench sweep --plan plans/e5_breakdown.json --select device=cpu --select backend=syclnn --select dtype=float --out $RES/e5-breakdown-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=cpu --select backend=syclnn --out $RES/e7-tiled-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    for b in tiled auto; do for wl in 'monk 40' 'cup 40' 'mnist-512-256 256'; do set -- \$wl; fnnbench run --backend syclnn --device cpu --workload \$1 --batch \$2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=\$b --out $RES/portable-acpp 2>&1 | grep -E 'epoch|rror' | tail -1 | cut -c1-150; done; done
"

log "3. ompnn host path (gcc-14 bound, clang unbound, gcc-14+MKL): E4 + peaks + gcc-14 E5/E7"
bash scripts/matrix-ws-amd.sh $DATE omp-cpu

log "4. portability rows, DPC++ wheel built on ws-nvidia, on ws-amd's CPU"
podman run --rm --memory=20g --security-opt label=disable -v "$AC/syclnn:/work/syclnn" -v "$HERE:/work/fnn-bench" -w /work/fnn-bench localhost/fnn-sycl:dev bash -c '
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_NUM_THREADS=16 DPCPP_CPU_NUM_CUS=16 OMP_PROC_BIND=close OMP_PLACES=cores
    sha256sum /work/fnn-bench/.wheels/ws-nvidia-sycl/syclnn/_syclnn*.so | tee /work/fnn-bench/'"$RES"'/portable-dpcpp.SHA256
    for b in tiled auto; do for wl in "monk 40" "cup 40" "mnist-512-256 256"; do set -- $wl; fnnbench run --backend syclnn --device cpu --workload $1 --batch $2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=$b --out '"$RES"'/portable-dpcpp 2>&1 | grep -E "epoch|rror" | tail -1 | cut -c1-150; done; done
'

log "5. supersede older duplicates, analyze, headline"
$PY scripts/supersede-stale.py $RES --dupes
bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py $RES > results/analysis/headline-ws-amd-$DATE.md && head -20 results/analysis/headline-ws-amd-$DATE.md
log "fixed-clock CPU rows done"
