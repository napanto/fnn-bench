#!/bin/bash
D=$HOME/fnn/fnn-bench; cd $D; RES=results/ws-nvidia/overlap-probe-2026-11-08; mkdir -p $RES
RUN="podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable -v $D:/work/fnn-bench -w /work/fnn-bench localhost/fnn-cuda:dev"
timeout 300 $RUN bash -c "nvcc -O3 -std=c++17 -arch=sm_61 -ccbin g++-13 scripts/launch-probe/probe5_cuda.cu -lcublas -o /tmp/p5 && for r in 1 2 3; do PROBE_R=9 PROBE_REVERSE=\$((r % 2 == 0)) /tmp/p5 > $RES/probe5-r\$r.csv; done" > $RES/run.log 2>&1
echo "==== done" >> $RES/run.log
