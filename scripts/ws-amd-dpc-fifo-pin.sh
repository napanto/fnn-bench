#!/usr/bin/env bash
# ws-amd, CPU, CUP b40, DPC++ with 12 workers. While the process runs, a loop re-applies every 0.3 s the
# affinity of every thread: the main thread alone on CPU 12 (a core whose sibling, CPU 28, nothing uses) or
# on CPU 24 (the SMT sibling of CPU 8, where workers run); every other thread on CPUs 0-11. All four CPUs
# are on NUMA node 1. SMT sharing for the main thread is on or off by construction, and it cannot migrate.
# Arms named dpr-* give the other threads CPUs 0-10 and 16 instead: twelve workers on twelve CPUs, two of
# them on the two hardware threads of core 0.
# Default policy and every thread at SCHED_FIFO 98. Fresh processes, alternated; 2.8 GHz cap, boost off,
# interleaved pages, nice 0. python runs as v-<variant>-r<round> so that its host PID can be found.
#   scripts/ws-amd-dpc-fifo-pin.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-3}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/dpc-fifo-pin-${DATE}${SUFFIX:-b}; cd "$HERE"; mkdir -p "$RES"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
PODMAN="podman run --rm --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined -v $HERE:/work/fnn-bench -w /work/fnn-bench"
ENV="export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=12 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:/work/fnn-bench/bench:/work/fnn-bench/testkit"
one() { local v=$1 k=$2 pre=""; local tag=v-$v-r$k out=$RES/$v/round-$k cpu=${v##*-m} wset=0-11
    case $v in dpr-*) wset=0-10,16 ;; esac
    ls "$out"/*.jsonl >/dev/null 2>&1 && return 0; mkdir -p "$out"
    case $v in fifo-*) pre="chrt -f 98 " ;; esac
    timeout -s KILL 300 $PODMAN --name fnnpin-$v-$k localhost/fnn-sycl:dev bash -c "$ENV; py=\$(command -v python); ln -sf \$py \$(dirname \$py)/$tag && $pre\$(dirname \$py)/$tag -m fnnbench.cli run --backend syclnn --device cpu --dtype float --workload cup --batch 40 --epochs 100 --repeat 3 --warmup 1 --seed 1 --samples 1000 --out $out" > "$out/run.log" 2>&1 &
    local job=$! pid="" n=0
    for i in $(seq 1 300); do pid=$(pgrep -x "$tag" | head -1); [ -n "$pid" ] && break; sleep 0.1; done
    ( while kill -0 "$pid" 2>/dev/null; do
          for t in /proc/$pid/task/*; do tid=${t##*/}; if [ "$tid" = "$pid" ]; then taskset -p -c "$cpu" "$tid"; else taskset -p -c "$wset" "$tid"; fi; done >/dev/null 2>&1
          sleep 0.3
      done ) & local pinner=$!
    sleep 6; for t in /proc/$pid/task/*; do printf '%s %s\n' "${t##*/}" "$(taskset -p ${t##*/} 2>/dev/null | awk '{print $NF}')"; done > "$out/pin.txt"
    wait $job; echo $? > "$out/exit.txt"; kill $pinner 2>/dev/null; podman rm -f fnnpin-$v-$k >/dev/null 2>&1
}
for k in $(seq 1 "$ROUNDS"); do
    list=${VARS:-"def-m12 def-m24 fifo-m12 fifo-m24"}; [ $((k % 2)) = 0 ] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do one "$v" "$k"; done
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
