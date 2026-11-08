#!/bin/bash
# Copied from the run of 2026-11-08 01:44 on ws-nvidia.
# GTX 1080 Ti: the suggested SYCL options against the defaults, checked harness rows, alternated, three processes.
D=$HOME/fnn/fnn-bench; cd $D; RES=results/ws-nvidia/tuned-2026-11-08; mkdir -p $RES
ENV='export FNN_REF_CACHE=/work/fnn-bench/.cache/ref PYTHONPATH=/work/fnn-bench/bench:/work/fnn-bench/testkit'
T="--option loss_reduction=False --option bias_gemv=False"
M="--workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --samples 60000"
one() { local arm=$1 k=$2 w=$3 be=$4; shift 4
  local out=$RES/$arm/mnist/round-$k; mkdir -p $out
  ls $out/*.jsonl >/dev/null 2>&1 && return
  local name=fnnrow-$arm-$k
  timeout 400 podman run --rm --name $name --memory=40g --ulimit memlock=-1:-1 --device nvidia.com/gpu=all --security-opt=label=disable --security-opt seccomp=unconfined -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl-perf:dev bash -c "$ENV:/work/fnn-bench/.wheels/$w; numactl --cpunodebind=1 --membind=1 python -m fnnbench.cli run --backend $be --device gpu --dtype float --seed 1 --warmup 1 $M $* --out $out" > $out/run.log 2>&1
  podman rm -f $name >/dev/null 2>&1
}
for k in 1 2 3; do
  one cuda-inorder $k ws-nvidia-cuda cudann --option queue=in_order
  one dpcpp-default $k ws-nvidia-sycl syclnn
  one dpcpp-tuned $k ws-nvidia-sycl syclnn $T --option workgroup_size=256
  one cuda-tuned $k ws-nvidia-cuda cudann --option queue=in_order $T
done
echo finished
