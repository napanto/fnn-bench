#!/usr/bin/env bash
# ws-amd, CPU device, CUP b40 float: why do repeated sweeps of one configuration differ by up to 28.7 %?
# Same row protocol as the reported pass (2.8 GHz cap, boost off, interleaved pages, nice 0), one fresh
# process per row, variants alternated round by round so that drift cannot look like an effect.
#
# Variants inject one known source of interference at a time, or change one control:
#   quiet         nothing added
#   sib-hk        one spinning process at nice 19 on an SMT sibling of a housekeeping core (CPU 16)
#   sib-iso       the same on an SMT sibling of an isolated core (CPU 24)
#   same-cpu      one spinning process at nice 0 on a CPU that also runs a worker (CPU 3)
#   memhog        one memory-writing process at nice 19 on CPU 17
#   bound         DPC++ only: OpenCL CPU workers bound to cores (the pass leaves them unbound)
#   fifo          gcc only: every thread SCHED_FIFO 1
#   fifo-sib-hk   gcc only: SCHED_FIFO 1 with the spinner of sib-hk
#   fifo-same-cpu gcc only: SCHED_FIFO 1 with the spinner of same-cpu
#   iso-first     gcc only: same 16 cores, the master thread on an isolated core
#   noaslr        address-space randomisation off
#
#   scripts/ws-amd-cup-interference.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | head -1)" != "policy: interleave" ]; then exec numactl --interleave=all bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); AC=$(cd "$HERE/.." && pwd)
ROUNDS=${1:-6}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/cup-interference-$DATE
cd "$HERE"; mkdir -p "$RES"
log() { echo "==== [$(date +%T)] $*"; }
cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
[ "$cap" = 2800000 ] && [ "$boost" = 0 ] || { echo "CPU cap not in force (max $cap kHz, boost $boost): refusing to run"; exit 1; }
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running: refusing to run"; exit 1; }
[ "$(nice)" = 0 ] || { echo "not at nice 0: refusing to run"; exit 1; }

RUN="python -m fnnbench.cli run --device cpu --dtype float --workload cup --batch 40 --epochs 5 --repeat 5 --warmup 1 --seed 1 --samples 1000"
COMMON='export FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_PROC_BIND=close OMP_PLACES=cores'
PODMAN="podman run --rm --memory=20g --ulimit rtprio=99:99 --security-opt label=disable --security-opt seccomp=unconfined -v $HERE:/work/fnn-bench -w /work/fnn-bench"
PP=/work/fnn-bench/bench:/work/fnn-bench/testkit

spin=()
inject() { # inject <variant>
    case $1 in
    *sib-hk)   taskset -c 16 nice -n 19 bash -c 'while :; do :; done' & spin+=($!) ;;
    *sib-iso)  taskset -c 24 nice -n 19 bash -c 'while :; do :; done' & spin+=($!) ;;
    *same-cpu) taskset -c 3 bash -c 'while :; do :; done' & spin+=($!) ;;
    *memhog)   taskset -c 17 nice -n 19 dd if=/dev/zero of=/dev/null bs=1G 2>/dev/null & spin+=($!) ;;
    esac
    sleep 0.5
}
clear_injected() { [ ${#spin[@]} -gt 0 ] && kill -9 "${spin[@]}" 2>/dev/null; wait 2>/dev/null; spin=(); }
trap clear_injected EXIT

one() { # one <backend> <variant> <round>
    local be=$1 v=$2 k=$3
    local out=$RES/$v/$be/round-$k
    mkdir -p "$out"
    ls "$out"/*.jsonl >/dev/null 2>&1 && return 0
    { cat /proc/loadavg; grep -E '^cpu[0-9]+ ' /proc/stat; } > "$out/stat-before.txt"
    inject "$v"
    local pre=""; case $v in fifo*) pre="chrt -f 1 " ;; noaslr) pre="setarch x86_64 -R " ;; esac
    case $be in
    dpcpp)
        local env=""; [ "$v" = bound ] && env="export DPCPP_CPU_CU_AFFINITY=close DPCPP_CPU_PLACES=cores;"
        timeout -s KILL 240 $PODMAN localhost/fnn-sycl:dev bash -c "$COMMON DPCPP_CPU_NUM_CUS=16 PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp:$PP; $env $pre$RUN --backend syclnn --out $out" ;;
    gcc14)
        local env=""; [ "$v" = iso-first ] && env="export OMP_PLACES='{8},{9},{10},{11},{12},{13},{14},{15},{0},{1},{2},{3},{4},{5},{6},{7}';"
        timeout -s KILL 240 $PODMAN localhost/fnn-cuda:dev bash -c "$COMMON MKL_THREADING_LAYER=GNU PYTHONPATH=/work/fnn-bench/.wheels/omp-gcc14:$PP; $env $pre$RUN --backend ompnn --out $out" ;;
    acpp)
        P=$HOME/.local/opt/fnn-rocm
        timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores; cd $HERE; $RUN --backend syclnn --out $out" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
    esac > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"
    clear_injected
    { cat /proc/loadavg; grep -E '^cpu[0-9]+ ' /proc/stat; } > "$out/stat-after.txt"
}

VARIANTS=${VARIANTS:-"dpcpp:quiet gcc14:quiet acpp:quiet dpcpp:bound gcc14:fifo dpcpp:sib-hk gcc14:sib-hk acpp:sib-hk dpcpp:sib-iso gcc14:sib-iso acpp:sib-iso dpcpp:same-cpu gcc14:same-cpu acpp:same-cpu dpcpp:memhog gcc14:memhog acpp:memhog gcc14:fifo-sib-hk gcc14:fifo-same-cpu"}
log "memory policy: $(numactl --show | head -1); nice $(nice); rounds $ROUNDS; $RES"
for k in $(seq 1 "$ROUNDS"); do
    list=$VARIANTS; [ $((k % 2)) = 0 ] && list=$(echo "$VARIANTS" | tr ' ' '\n' | tac | tr '\n' ' ')
    log "round $k"
    for item in $list; do one "${item%%:*}" "${item##*:}" "$k"; done
done
log done
