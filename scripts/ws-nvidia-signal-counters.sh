#!/usr/bin/env bash
# ws-nvidia: what a sampling signal costs DPC++ in user space. Whole-process user counters with the
# sampler at a 200 us period (as a delay, no output) against a period that never fires, alternated.
#   scripts/ws-nvidia-signal-counters.sh [date]
set -uo pipefail
D=$HOME/fnn/fnn-bench; DATE=${1:-$(date +%F)}; RES=results/ws-nvidia/signal-counters-$DATE; mkdir -p "$D/$RES"; cd "$D"
ENV='P=$(find /usr/lib -name perf -type f | head -1); export FNN_REF_CACHE=/work/fnn-bench/.cache/ref PYTHONPATH=/work/fnn-bench/bench:/work/fnn-bench/testkit:/work/fnn-bench/.wheels/ws-nvidia-sycl'
EV=cycles:u,instructions:u,L1-icache-load-misses:u,iTLB-load-misses:u,dTLB-load-misses:u
one() { local tag=$1 p=$2 k=$3 name=fnnsig-$1-$3
    timeout 300 podman run --rm --name $name --memory=40g --ulimit memlock=-1:-1 --device nvidia.com/gpu=all --security-opt=label=disable --security-opt seccomp=unconfined -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl-perf:dev bash -c "$ENV; gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o /tmp/wallprof.so && \$P stat -x, -e $EV -o $RES/$tag-r$k.stat -- numactl --cpunodebind=1 --membind=1 python scripts/wallprof/wallprof_run.py --backend syclnn --epochs 40 --delay-ms 300 --period-us $p --max-samples 400000 --discard --lib /tmp/wallprof.so --out $RES/$tag-r$k.dump" > $RES/$tag-r$k.log 2>&1
    podman rm -f $name >/dev/null 2>&1
}
for k in 1 2 3; do
    one none 100000000 $k
    one p200 200 $k
done
echo "==== done $(date +%T)" >> $RES/run.log
