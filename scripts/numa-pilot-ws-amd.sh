#!/usr/bin/env bash
# NUMA pilot on ws-amd (Threadripper 2950X: 2 NUMA nodes, 0-7,16-23 and 8-15,24-31; the RX 7900 XTX
# hangs off node 0). Replays a few rows of the 2026-09-05 pass under two memory policies, in ABAB
# order, to size the effect of page placement before deciding what to re-measure:
#   CPU rows:  first-touch (the pass: policy default)  vs  numactl --interleave=all
#   GPU rows:  default                                 vs  numactl --cpunodebind=0 --membind=0 (the GPU's node)
# The pass rows ran with the default policy: data first-touched by the master thread, which
# OMP_PROC_BIND=close pins to core 0, so nearly all pages land on node 0 while half of the 16
# worker threads run on node 1. Rows go to results/ws-amd/numa-pilot-<date>/<policy>-r<k>/<env>/;
# each row records the policy in sysinfo.numa.numactl_show.
# Requires the 2.8 GHz cap and scripts/thermal-guard.sh, as every CPU row of ws-amd.
#   scripts/numa-pilot-ws-amd.sh [date]
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${1:-2026-11-03}
P=results/ws-amd/numa-pilot-$DATE
cd "$HERE"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
if [ "$cap" != 2800000 ] || [ "$boost" != 0 ]; then echo "CPU cap not in force (max $cap kHz, boost $boost): refusing to run"; exit 1; fi
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running: refusing to run"; exit 1; }
log() { printf '\n==== [%s] %s\n' "$(date +%T)" "$*"; }
PODMAN="podman run --rm --memory=20g --security-opt label=disable -v $AC/syclnn:/work/syclnn -v $AC/ompnn:/work/ompnn -v $HERE:/work/fnn-bench -w /work/fnn-bench"

cpu_round() {   # $1 = policy name, $2 = numactl prefix (empty for the default policy), $3 = round
    local pol=$1 pre=$2 out=$P/$1-r$3
    log "CPU $pol round $3: DPC++ OpenCL CPU (MKLCPU / Netlib)"
    $PODMAN localhost/fnn-sycl:dev bash -c "
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench 2>&1 | grep -v notice
        export PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=16
        $pre fnnbench replay $P/sel-dpcpp-syclnn.jsonl --out $out/dpcpp 2>&1 | grep -E 'epoch|rror|replayed' | cut -c1-170"
    log "CPU $pol round $3: ompnn gcc-14 + OpenBLAS"
    $PODMAN localhost/fnn-cuda:dev bash -c "
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench 2>&1 | grep -v notice
        export PYTHONPATH=/work/fnn-bench/.wheels/omp-gcc14 FNN_REF_CACHE=/work/fnn-bench/.cache/ref MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
        $pre fnnbench replay $P/sel-omp-gcc14-ompnn.jsonl --out $out/omp-gcc14 2>&1 | grep -E 'epoch|rror|replayed' | cut -c1-170"
    log "CPU $pol round $3: AdaptiveCpp host device"
    distrobox enter fnn-rocm -- bash -c "
        P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores; cd $HERE
        $pre fnnbench replay $P/sel-acpp-cpu-syclnn.jsonl --out $out/acpp-cpu 2>&1 | grep -v 'AdaptiveCpp Warning' | grep -E 'epoch|rror|replayed' | cut -c1-170"
}

gpu_round() {
    local pol=$1 pre=$2 out=$P/$1-r$3
    log "GPU $pol round $3: cudann (HIP) and syclnn (AdaptiveCpp) on the RX 7900 XTX"
    distrobox enter fnn-rocm -- bash -c "
        P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores; cd $HERE
        $pre fnnbench replay $P/sel-gpu-cudann.jsonl --out $out/gpu 2>&1 | grep -v 'AdaptiveCpp Warning' | grep -E 'epoch|rror|replayed' | cut -c1-170
        $pre fnnbench replay $P/sel-gpu-syclnn.jsonl --out $out/gpu 2>&1 | grep -v 'AdaptiveCpp Warning' | grep -E 'epoch|rror|replayed' | cut -c1-170"
}

for k in 1 2; do
    cpu_round first-touch "" $k
    cpu_round interleave "numactl --interleave=all" $k
    gpu_round default "" $k
    gpu_round gpu-node "numactl --cpunodebind=0 --membind=0" $k
done
log "pilot done: $P"
