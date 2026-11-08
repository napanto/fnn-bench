#!/usr/bin/env bash
# ws-amd, CPU, CUP b40, DPC++ with 12 workers on all 32 CPUs: does FIFO's cost depend on the deep idle state C2?
# A root helper enables or disables cpuidle state2 on every CPU on request (<ctl>/want, confirmed in <ctl>/have).
# Arms per round: def/fifo with C2 enabled, def/fifo with C2 disabled; fresh processes, alternated;
# 2.8 GHz cap, boost off, interleaved pages, nice 0.
#   scripts/ws-amd-dpc-fifo-c2.sh <ctl-dir> [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); CTL=$1; ROUNDS=${2:-3}; DATE=${3:-$(date +%F)}
RES=results/ws-amd/dpc-fifo-c2-$DATE; cd "$HERE"; mkdir -p "$RES"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
PODMAN="podman run --rm --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined -v $HERE:/work/fnn-bench -w /work/fnn-bench"
ENV="export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=12 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:/work/fnn-bench/bench:/work/fnn-bench/testkit"
level() { echo "$1" > "$CTL/want"; for i in $(seq 1 100); do [ "$(cat "$CTL/have" 2>/dev/null)" = "$1" ] && return 0; sleep 0.1; done; echo "level $1 not confirmed"; exit 1; }
one() { local v=$1 k=$2 pre=""; local out=$RES/$v/round-$k
    ls "$out"/*.jsonl >/dev/null 2>&1 && return 0; mkdir -p "$out"
    case $v in fifo-*) pre="chrt -f 98 " ;; esac
    level "${v#*-}"; sleep 1
    timeout -s KILL 300 $PODMAN --name fnnc2-$v-$k localhost/fnn-sycl:dev bash -c "$ENV; ${pre}python -m fnnbench.cli run --backend syclnn --device cpu --dtype float --workload cup --batch 40 --epochs 100 --repeat 3 --warmup 1 --seed 1 --samples 1000 --out $out" > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"; podman rm -f fnnc2-$v-$k >/dev/null 2>&1
}
for k in $(seq 1 "$ROUNDS"); do
    list="def-c2on fifo-c2on def-c2off fifo-c2off"; [ $((k % 2)) = 0 ] && list="fifo-c2off def-c2off fifo-c2on def-c2on"
    for v in $list; do one "$v" "$k"; done
done
echo stop > "$CTL/want"
echo "==== [$(date +%T)] done" >> "$RES/run.log"
