#!/usr/bin/env bash
# ws-nvidia, one hour at night: where the host time of an epoch goes, by library and symbol.
# User-space cpu-clock sampling of the training process, inside the image with perf installed.
# Attribution only: these runs carry no timing claim, the GPU is not reserved.
#   scripts/ws-nvidia-perf-night.sh [date]
set -uo pipefail
D=$HOME/fnn/fnn-bench; DATE=${1:-$(date +%F)}; RES=results/ws-nvidia/perf-host-$DATE; mkdir -p "$D/$RES"; cd "$D"
RUN="podman run --rm --memory=40g --ulimit memlock=-1:-1 --device nvidia.com/gpu=all --security-opt=label=disable --security-opt seccomp=unconfined -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl-perf:dev"
ENV='P=$(find /usr/lib -name perf -type f | head -1); export FNN_REF_CACHE=/work/fnn-bench/.cache/ref PYTHONPATH=/work/fnn-bench/bench:/work/fnn-bench/testkit'
NUMA='numactl --cpunodebind=1 --membind=1'
one() { # one <tag> <wheel> <backend> <round> <extra args>
    local tag=$1 w=$2 be=$3 k=$4; shift 4
    local o=$RES/$tag-r$k
    timeout -s KILL 120 $RUN bash -c "$ENV:/work/fnn-bench/.wheels/$w; \$P record -m 32 -F 2999 -e cpu-clock:u -o /tmp/p.data -- $NUMA python scripts/clock_ramp.py --backend $be --seconds 15 --out $o.json $* > $o.record.log 2>&1
        \$P report -i /tmp/p.data --stdio --sort dso --percent-limit 0.2 2>/dev/null > $o.dso.txt
        \$P report -i /tmp/p.data --stdio --sort dso,symbol --percent-limit 0.15 2>/dev/null > $o.symbol.txt" >> "$RES/run.log" 2>&1
}
for k in 1 2 3; do
    one dpcpp-mnist ws-nvidia-sycl syclnn "$k"
    one cuda-mnist ws-nvidia-cuda cudann "$k" --option queue=in_order
    one clang-mnist ws-nvidia-omp-clang18nv ompnn "$k"
done
for k in 1 2 3 4 5 6 7 8; do
    one clang-monk ws-nvidia-omp-clang18nv ompnn "$k" --workload monk --samples 124 --batch 40
done
echo "==== done $(date +%T)" >> "$RES/run.log"
