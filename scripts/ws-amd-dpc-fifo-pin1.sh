#!/usr/bin/env bash
# ws-amd, CPU, CUP b40, DPC++ with 12 workers, every thread pinned to one CPU of its own (no migrations).
# Until 8 s into the process the other threads share CPUs 0-11 and the main thread sits on CPU 12; then the
# non-main threads, ordered by CPU time used so far, are pinned one by one:
#   p1-*  CPUs 0 13 1 14 2 3 4 5 6 7 8 9 10: every thread on its own physical core;
#   ps-*  CPUs 0 16 1 17 2 3 4 5 6 7 8 9 10: the four busiest threads in two SMT pairs (cores 0 and 1).
# Threads created later share CPUs 2-10. Default policy (-def) and every thread at SCHED_FIFO 98 (-fifo).
# Fresh processes, alternated; 2.8 GHz cap, boost off, interleaved pages, nice 0.
#   scripts/ws-amd-dpc-fifo-pin1.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-3}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/dpc-fifo-pin1-$DATE; cd "$HERE"; mkdir -p "$RES"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
PODMAN="podman run --rm --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined -v $HERE:/work/fnn-bench -w /work/fnn-bench"
ENV="export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=12 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:/work/fnn-bench/bench:/work/fnn-bench/testkit"
one() { local v=$1 k=$2 pre=""; local tag=v-$v-r$k out=$RES/$v/round-$k
    ls "$out"/*.jsonl >/dev/null 2>&1 && return 0; mkdir -p "$out"
    case $v in *-fifo) pre="chrt -f 98 " ;; esac
    local cpus="0 13 1 14 2 3 4 5 6 7 8 9 10"; case $v in ps-*) cpus="0 16 1 17 2 3 4 5 6 7 8 9 10" ;; esac
    timeout -s KILL 300 $PODMAN --name fnnp1-$v-$k localhost/fnn-sycl:dev bash -c "$ENV; py=\$(command -v python); ln -sf \$py \$(dirname \$py)/$tag && $pre\$(dirname \$py)/$tag -m fnnbench.cli run --backend syclnn --device cpu --dtype float --workload cup --batch 40 --epochs 100 --repeat 3 --warmup 1 --seed 1 --samples 1000 --out $out" > "$out/run.log" 2>&1 &
    local job=$! pid=""
    for i in $(seq 1 300); do pid=$(pgrep -x "$tag" | head -1); [ -n "$pid" ] && break; sleep 0.1; done
    ( start=$(date +%s.%N); declare -A map; assigned=0
      while kill -0 "$pid" 2>/dev/null; do
          now=$(date +%s.%N)
          if [ "$assigned" = 0 ] && [ "$(echo "$now - $start > 8" | bc)" = 1 ]; then
              i=0; set -- $cpus
              for tid in $(for t in /proc/$pid/task/*; do x=${t##*/}; [ "$x" = "$pid" ] && continue; awk -v t=$x '{print $14+$15, t}' $t/stat; done | sort -rn | awk '{print $2}'); do
                  [ $# -gt 0 ] && { map[$tid]=$1; shift; }
              done; assigned=1
              for tid in "${!map[@]}"; do echo "$tid ${map[$tid]}"; done > "$out/map.txt"
          fi
          for t in /proc/$pid/task/*; do tid=${t##*/}
              if [ "$tid" = "$pid" ]; then taskset -p -c 12 "$tid"
              elif [ -n "${map[$tid]:-}" ]; then taskset -p -c "${map[$tid]}" "$tid"
              elif [ "$assigned" = 1 ]; then taskset -p -c 2-10 "$tid"
              else taskset -p -c 0-11 "$tid"; fi
          done >/dev/null 2>&1
          sleep 0.3
      done ) & local pinner=$!
    sleep 12; for t in /proc/$pid/task/*; do printf '%s %s\n' "${t##*/}" "$(taskset -p ${t##*/} 2>/dev/null | awk '{print $NF}')"; done > "$out/pin.txt"
    wait $job; echo $? > "$out/exit.txt"; kill $pinner 2>/dev/null; podman rm -f fnnp1-$v-$k >/dev/null 2>&1
}
for k in $(seq 1 "$ROUNDS"); do
    list=${VARS:-"p1-def ps-def p1-fifo ps-fifo"}; [ $((k % 2)) = 0 ] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do one "$v" "$k"; done
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
