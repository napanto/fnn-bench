#!/usr/bin/env bash
# ws-nvidia: is clang's slow mode the blocking wait of its context? Short rows, fresh processes, GPU free.
# Arms: default; spin (primary context forced to spin before libomptarget starts); one host OpenMP thread.
#   scripts/ws-nvidia-clang-spin.sh [rounds] [date]
set -uo pipefail
D=$HOME/fnn/fnn-bench; K=${1:-8}; DATE=${2:-$(date +%F)}; RES=results/ws-nvidia/clang-spin-$DATE; mkdir -p "$D/$RES"; cd "$D"
RUN="podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl-perf:dev"
ENV='export FNN_REF_CACHE=/work/fnn-bench/.cache/ref LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-} PYTHONPATH=/work/fnn-bench/bench:/work/fnn-bench/testkit:/work/fnn-bench/.wheels/ws-nvidia-omp-clang18nv'
one() { # one <arm> <workload> <samples> <batch> <round>
    local arm=$1 wl=$2 ns=$3 b=$4 k=$5 extra="" env="" info=""
    [ "$arm" = spin ] && extra=--spin
    [ "$arm" = omp1 ] && env="export OMP_NUM_THREADS=1;"
    [ "$k" = 1 ] && info="export LIBOMPTARGET_INFO=-1;"
    timeout -s KILL 60 $RUN bash -c "$ENV; $env $info numactl --cpunodebind=1 --membind=1 python scripts/clock_ramp.py --backend ompnn --workload $wl --samples $ns --batch $b --seconds 6 $extra --out $RES/$arm-$wl-r$k.json" > "$RES/$arm-$wl-r$k.log" 2>&1
}
for k in $(seq 1 "$K"); do
    arms='default spin omp1'; [ $((k % 2)) = 0 ] && arms='omp1 spin default'
    for arm in $arms; do one "$arm" monk 124 40 "$k"; one "$arm" cup 1000 40 "$k"; done
done
echo "==== done $(date +%T)" >> "$RES/run.log"
