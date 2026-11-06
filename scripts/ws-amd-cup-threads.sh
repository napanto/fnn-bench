#!/usr/bin/env bash
# ws-amd, CPU, CUP b40: per-call thread records of scripts/cup_threads.py, fresh process per round.
# Same conditions as the reported CPU rows (2.8 GHz cap, boost off, interleaved pages, nice 0).
#   scripts/ws-amd-cup-threads.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-6}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/cup-threads-$DATE; cd "$HERE"; mkdir -p "$RES"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force"; exit 1; }
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
COMMON='export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores'
PODMAN="podman run --rm --pid=host --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined --security-opt unmask=ALL -v $HERE:/work/fnn-bench -w /work/fnn-bench"
PP=/work/fnn-bench/bench:/work/fnn-bench/testkit
one() { local be=$1 v=$2 k=$3; local out=$RES/$be-$v-r$k.json; [ -s "$out" ] && return 0
    local env="" pre="" opt="" base=$v spinner="" sp=""
    case $v in *-pause) base=${v%-pause}; spinner=pause ;; *-bash) base=${v%-bash}; spinner=bash ;; *-l1) base=${v%-l1}; spinner=l1 ;; esac
    case $base in
    m[0-9]*)  # main thread on one logical CPU, the 15 other workers on the first thread of their cores
        local c=${base#m}; c=${c%-fifo}; local pl="{$c}" n
        for n in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do [ "$n" = $((c % 16)) ] || pl="$pl,{$n}"; done
        env="export OMP_PLACES='$pl';"; case $base in *-fifo) pre="chrt -f 98 " ;; esac ;;
    iso-first) env="export OMP_PLACES='{8},{9},{10},{11},{12},{13},{14},{15},{0},{1},{2},{3},{4},{5},{6},{7}';" ;;
    hk3-first) env="export OMP_PLACES='{3},{0},{1},{2},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}';" ;;
    hk5-first) env="export OMP_PLACES='{5},{0},{1},{2},{3},{4},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}';" ;;
    iso12-first) env="export OMP_PLACES='{12},{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{13},{14},{15}';" ;;
    single-cpu) env="export OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}';" ;;
    master-fifo1) opt="--master-fifo 1" ;;
    master-fifo60) opt="--master-fifo 60" ;;
    fifo1) pre="chrt -f 1 " ;;
    fifo51) pre="chrt -f 51 " ;;
    fifo98) pre="chrt -f 98 " ;;
    main-fifo98) opt="--master-fifo 98" ;;
    workers-fifo98) pre="chrt -f 98 "; opt="--master-other" ;;
    bound-fifo98) env="export DPCPP_CPU_CU_AFFINITY=close DPCPP_CPU_PLACES=cores;"; pre="chrt -f 98 " ;;
    iso-fifo98) env="export OMP_PLACES='{8},{9},{10},{11},{12},{13},{14},{15},{0},{1},{2},{3},{4},{5},{6},{7}';"; pre="chrt -f 98 " ;;
    bound) env="export DPCPP_CPU_CU_AFFINITY=close DPCPP_CPU_PLACES=cores;" ;;
    esac
    [ -n "${CENSUS:-}" ] && opt="$opt --census --calls 120"
    if [ -n "${PERF:-}" ]; then
        opt="$opt --calls 150 --marker $RES/$be-$v-r$k.pid"
        ( for i in $(seq 1 400); do [ -s "$RES/$be-$v-r$k.pid" ] && break; sleep 0.05; done
          p=$(cat "$RES/$be-$v-r$k.pid"); perf stat -t "$p" -e task-clock,cycles:u,instructions:u,cache-misses:u,L1-dcache-load-misses:u,branch-misses:u -x, -o "$RES/$be-$v-r$k.perf" -- sleep 3 ) &
    fi
    case $spinner in
    pause|l1) taskset -c 16 nice -n 19 "$HERE/.cache/spin" "$spinner" & sp=$! ;;
    bash) taskset -c 16 nice -n 19 bash -c 'while :; do :; done' & sp=$! ;;
    esac
    [ -n "$sp" ] && sleep 0.5
    case $be in
    dpcpp) timeout -s KILL 120 $PODMAN localhost/fnn-sycl:dev bash -c "$COMMON DPCPP_CPU_NUM_CUS=16 PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:$PP; $env $pre python scripts/cup_threads.py --backend syclnn $opt --out $out" ;;
    gcc14) timeout -s KILL 120 $PODMAN localhost/fnn-cuda:dev bash -c "$COMMON MKL_THREADING_LAYER=GNU PYTHONPATH=/work/fnn-bench/.wheels/omp-gcc14:$PP; $env $pre python scripts/cup_threads.py --backend ompnn $opt --out $out" ;;
    esac >> "$RES/run.log" 2>&1
    [ -n "$sp" ] && { kill -9 "$sp" 2>/dev/null; wait "$sp" 2>/dev/null; }
}
[ -x "$HERE/.cache/spin" ] || podman run --rm --security-opt label=disable -v $HERE:/work/fnn-bench -w /work/fnn-bench localhost/fnn-cuda:dev gcc -O2 -static scripts/spin.c -o .cache/spin
LIST=${LIST:-"gcc14:default gcc14:iso-first gcc14:master-fifo1 gcc14:master-fifo60 dpcpp:default dpcpp:fifo1 dpcpp:bound"}
for k in $(seq 1 "$ROUNDS"); do
    list=$LIST; [ $((k % 2)) = 0 ] && list=$(echo "$LIST" | tr ' ' '\n' | tac | tr '\n' ' ')
    echo "==== [$(date +%T)] round $k"
    for item in $list; do one "${item%%:*}" "${item##*:}" "$k"; done
done
echo "==== [$(date +%T)] done"
