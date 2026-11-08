#!/usr/bin/env bash
# ws-amd, CPU, CUP b40, DPC++: where the submitting thread's wall time goes under the default policy and
# with every thread at SCHED_FIFO 98; VARS adds SCHED_RR 98 (rr98) and KMP_BLOCKTIME=0 (fifo98-bt0, default-bt0), and 12 workers on the 16 cores (default-cus12, fifo98-cus12). Same conditions as the reported CPU rows (2.8 GHz cap, boost off,
# interleaved pages, nice 0), fresh process per row, alternated.
#   scripts/ws-amd-dpc-fifo-wallprof.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-3}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/dpc-fifo-wallprof-$DATE; cd "$HERE"; mkdir -p "$RES"
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force"; exit 1; }
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
PODMAN="podman run --rm --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined -v $HERE:/work/fnn-bench -w /work/fnn-bench"
ENV="export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=16 PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:/work/fnn-bench/bench:/work/fnn-bench/testkit"
one() { local v=$1 k=$2 pre=""; local out=$RES/$v-r$k; [ -s "$out.stacks.json" ] && return 0
    local extra=""
    case $v in
    fifo98) pre="chrt -f 98 " ;;
    rr98) pre="chrt -r 98 " ;;
    fifo98-bt0) pre="chrt -f 98 "; extra="export KMP_BLOCKTIME=0;" ;;
    default-bt0) extra="export KMP_BLOCKTIME=0;" ;;
    default-cus12) extra="export DPCPP_CPU_NUM_CUS=12 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12;" ;;
    fifo98-cus12) pre="chrt -f 98 "; extra="export DPCPP_CPU_NUM_CUS=12 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12;" ;;
    esac
    timeout -s KILL 200 $PODMAN --name fnnwp-$v-$k localhost/fnn-sycl:dev bash -c "$ENV; $extra gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o /tmp/wallprof.so && $pre python scripts/wallprof/wallprof_run.py --backend syclnn --device cpu --workload cup --samples 1000 --batch 40 --warm 50 --epochs 300 --delay-ms 300 --period-us 2000 --lib /tmp/wallprof.so --out $out.dump && python scripts/wallprof/wallprof_report.py $out.dump" >> "$RES/run.log" 2>&1
    podman rm -f fnnwp-$v-$k >/dev/null 2>&1
}
for k in $(seq 1 "$ROUNDS"); do
    list=${VARS:-"default fifo98"}; [ $((k % 2)) = 0 ] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do one "$v" "$k"; done
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
