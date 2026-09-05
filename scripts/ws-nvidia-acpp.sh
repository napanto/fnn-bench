#!/usr/bin/env bash
# AdaptiveCpp on ws-nvidia's GTX 1080 Ti (optional experiments: SYCL implementation comparison on NVIDIA): build
# syclnn with acpp in the fnn-acpp-cuda image, run the parity suites and the E2/E3/E7 syclnn
# rows into results/ws-nvidia/<date>/sycl-gpu-acpp (the rows carry compiler = AdaptiveCpp).
#   DATE=2026-09-05 fnn-bench/scripts/ws-nvidia-acpp.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
HOST=${NVIDIA_HOST:-ws-nvidia}
REMOTE=${NVIDIA_DIR:-$HOME/fnn}
DATE=${DATE:-$(date +%F)}
RUN="podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v $REMOTE/syclnn:/work/syclnn -v $REMOTE/fnn-bench:/work/fnn-bench"
ssh "$HOST" "$RUN -w /work/syclnn localhost/fnn-acpp-cuda:dev bash -c '
    set -e; export CXX=/opt/acpp/bin/acpp CC=clang-18 SYCLNN_SYCL_IMPL=adaptivecpp SYCLNN_ONEMATH_ROOT=/opt/onemath-acpp CMAKE_PREFIX_PATH=/opt/acpp:/opt/onemath-acpp SYCLNN_CT_BACKENDS=netlib CMAKE_BUILD_PARALLEL_LEVEL=8 ACPP_TARGETS=generic
    rm -rf /work/fnn-bench/.wheels/ws-nvidia-sycl-acpp
    pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-sycl-acpp --config-settings=build-dir=/tmp/b-acpp .
    export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl-acpp
    python -c \"import syclnn; print(syclnn.build_info()); print(syclnn.devices())\"
    for o in \"--dtype double\" \"--dtype float\" \"--option queue=in_order\" \"--option blas_queue=dedicated\" \"--blas tiled --dtype float\"; do printf \"syclnn acpp cuda:gpu %-30s \" \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
    pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | tail -1
    cd /work/fnn-bench
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=gpu --out results/ws-nvidia/$DATE/sycl-gpu-acpp
    fnnbench sweep --plan plans/e3_cuda_vs_sycl_gpu.json --select backend=syclnn --out results/ws-nvidia/$DATE/sycl-gpu-acpp
    fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=syclnn --out results/ws-nvidia/$DATE/sycl-gpu-acpp
    fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=syclnn --out results/ws-nvidia/$DATE/sycl-gpu-acpp
    for dt in float double; do fnnbench peak --backend syclnn --device gpu --dtype \$dt --size 8192 --out results/ws-nvidia/$DATE/peaks-acpp; done'" 2>&1 | grep -v 'AdaptiveCpp Warning'
rsync -a "$HOST:$REMOTE/fnn-bench/results/ws-nvidia/" "$HERE/results/ws-nvidia/"
