#!/usr/bin/env bash
# ws-amd, CPU, CUP b40, DPC++: the scheduling behind the SCHED_FIFO cost. Fresh processes, alternated:
# def16/fifo16 (16 workers, default policy or every thread at FIFO 98) and def12/fifo12 (12 workers);
# def12c/fifo12c confine the process to CPUs 0-15, one hardware thread per core (16-31 are the SMT siblings).
# python runs through a symlink named v-<variant>-r<round>, so scripts/sched/runq.bt (run as root
# meanwhile) can attribute every thread. Same conditions as the CPU rows: 2.8 GHz cap, boost off,
# interleaved pages, nice 0.
#   scripts/ws-amd-dpc-fifo-sched.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-3}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/dpc-fifo-sched-$DATE; cd "$HERE"; mkdir -p "$RES"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force"; exit 1; }
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
PODMAN="podman run --rm --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined -v $HERE:/work/fnn-bench -w /work/fnn-bench"
ENV="export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:/work/fnn-bench/bench:/work/fnn-bench/testkit"
one() { local v=$1 k=$2 pre="" n=16; local tag=v-$v-r$k out=$RES/$v/round-$k
    ls "$out"/*.jsonl >/dev/null 2>&1 && return 0; mkdir -p "$out"
    case $v in fifo16) pre="chrt -f 98 " ;; def12) n=12 ;; fifo12) pre="chrt -f 98 "; n=12 ;; def12c) pre="taskset -c 0-15 "; n=12 ;; fifo12c) pre="taskset -c 0-15 chrt -f 98 "; n=12 ;; esac
    { date +%s.%N; cat /proc/uptime; } > "$out/start.txt"
    timeout -s KILL 300 $PODMAN --name fnnsched-$v-$k localhost/fnn-sycl:dev bash -c "$ENV DPCPP_CPU_NUM_CUS=$n OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n; py=\$(command -v python); ln -sf \$py \$(dirname \$py)/$tag && $pre\$(dirname \$py)/$tag -m fnnbench.cli run --backend syclnn --device cpu --dtype float --workload cup --batch 40 --epochs 100 --repeat 3 --warmup 1 --seed 1 --samples 1000 --out $out" > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"; podman rm -f fnnsched-$v-$k >/dev/null 2>&1
}
for k in $(seq 1 "$ROUNDS"); do
    list=${VARS:-"def16 fifo16 def12 fifo12"}; [ $((k % 2)) = 0 ] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do one "$v" "$k"; done
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
