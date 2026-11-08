#!/usr/bin/env bash
# ws-nvidia: do cudann and syclnn (through oneMath) call cuBLAS with the same arguments? An LD_PRELOAD
# shim (scripts/cublas-args/cublas_args.c) counts cublasSgemm_v2/cublasSgemv_v2 calls by argument tuple
# during 10 s of MNIST b256 training; CUDA in order, DPC++ default and DPC++ with less runtime work.
#   scripts/ws-nvidia-cublas-args.sh [date]
set -uo pipefail
D=$HOME/fnn/fnn-bench; DATE=${1:-$(date +%F)}; RES=results/ws-nvidia/cublas-args-$DATE; mkdir -p "$D/$RES"; cd "$D"
RUN="podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl-perf:dev"
ENV='export FNN_REF_CACHE=/work/fnn-bench/.cache/ref PYTHONPATH=/work/fnn-bench/bench:/work/fnn-bench/testkit'
one() { # one <tag> <wheel> <backend> [extra args]
    local tag=$1 w=$2 be=$3; shift 3
    timeout 200 ${RUN/podman run --rm/podman run --rm --name fnnargs-$tag} bash -c "$ENV:/work/fnn-bench/.wheels/$w; gcc -O2 -shared -fPIC scripts/cublas-args/cublas_args.c -ldl -o /tmp/cublas_args.so && CUBLAS_ARGS_OUT=$RES/$tag.csv LD_PRELOAD=/tmp/cublas_args.so python scripts/clock_ramp.py --backend $be --seconds 10 --out $RES/$tag.json $*" >> "$RES/run.log" 2>&1
    podman rm -f fnnargs-$tag >/dev/null 2>&1
}
for t in ${ARMS:-cuda dpcpp dpcpp-lw}; do
    case $t in
    cuda) one cuda ws-nvidia-cuda cudann --option queue=in_order ;;
    dpcpp) one dpcpp ws-nvidia-sycl syclnn ;;
    dpcpp-lw) one dpcpp-lw ws-nvidia-sycl syclnn --option loss_reduction=false --option workgroup_size=256 ;;
    esac
done
echo "==== done $(date +%T)" >> "$RES/run.log"
