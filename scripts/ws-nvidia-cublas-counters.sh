#!/usr/bin/env bash
# ws-nvidia: why do the same cuBLAS calls take more host time under DPC++? Hardware counters by library
# (cycles, instructions, L1 instruction-cache and iTLB misses), CUDA in order against DPC++, alternated.
# Ratios per library carry the claim; whole-process counts include initialisation. LW=1 adds DPC++ with
# atomic loss and groups of 256 (same cuBLAS calls, less runtime work).
#   scripts/ws-nvidia-cublas-counters.sh [date]
set -uo pipefail
D=$HOME/fnn/fnn-bench; DATE=${1:-$(date +%F)}; RES=results/ws-nvidia/cublas-counters-$DATE; mkdir -p "$D/$RES"; cd "$D"
RUN="podman run --rm --memory=40g --ulimit memlock=-1:-1 --device nvidia.com/gpu=all --security-opt=label=disable --security-opt seccomp=unconfined -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl-perf:dev"
ENV='P=$(find /usr/lib -name perf -type f | head -1); export FNN_REF_CACHE=/work/fnn-bench/.cache/ref PYTHONPATH=/work/fnn-bench/bench:/work/fnn-bench/testkit'
EV=cycles:u,instructions:u,L1-icache-load-misses:u,iTLB-load-misses:u
one() { # one <tag> <wheel> <backend> <round> [extra args]
    local tag=$1 w=$2 be=$3 k=$4; shift 4
    local o=$RES/$tag-r$k name=fnnperf-$tag-$k
    [ -s "$o.dso.txt" ] && return 0
    timeout 200 ${RUN/podman run --rm/podman run --rm --name $name} bash -c "$ENV:/work/fnn-bench/.wheels/$w; \$P record -m 32 -F 1999 -e $EV -o /tmp/p.data -- numactl --cpunodebind=1 --membind=1 python scripts/clock_ramp.py --backend $be --seconds 20 --out $o.json $* >/dev/null 2>&1
        \$P report -i /tmp/p.data --stdio --sort dso --show-total-period --percent-limit 0.5 2>/dev/null > $o.dso.txt" >> "$RES/run.log" 2>&1
    podman rm -f $name >/dev/null 2>&1
}
for k in 1 2 3; do
    one cuda ws-nvidia-cuda cudann "$k" --option queue=in_order
    one dpcpp ws-nvidia-sycl syclnn "$k"
    [ -n "${LW:-}" ] && one dpcpp-lw ws-nvidia-sycl syclnn "$k" --option loss_reduction=false --option workgroup_size=256
done
echo "==== done $(date +%T)" >> "$RES/run.log"
