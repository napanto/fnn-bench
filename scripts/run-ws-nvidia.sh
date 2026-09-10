#!/usr/bin/env bash
# NVIDIA runs on ws-nvidia (GTX 1080 Ti, Pascal sm_61, CUDA 12.9 images, podman + CDI).
#
# Run from ws-amd with ssh access to ws-nvidia (NVIDIA_HOST, NVIDIA_DIR override the defaults):
#   fnn-bench/scripts/run-ws-nvidia.sh sync      # rsync the four repos + images to ws-nvidia
#   fnn-bench/scripts/run-ws-nvidia.sh tests     # parity suites: syclnn cuda:gpu, cudann, ompnn (clang-18, gcc-14, nvc++)
#   fnn-bench/scripts/run-ws-nvidia.sh matrix    # E2/E3/E4/E6/W4 GPU plans + peaks + nsys traces -> results/ws-nvidia/<date>
#   fnn-bench/scripts/run-ws-nvidia.sh fetch     # copy results back
# Images are transferred with `podman save | ssh podman load` (no registry needed yet).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
HOST=${NVIDIA_HOST:-ws-nvidia}
REMOTE=${NVIDIA_DIR:-$HOME/fnn}
DATE=${DATE:-$(date +%F)}
GPU="--device nvidia.com/gpu=all --security-opt=label=disable"
RUN="podman run --rm --memory=40g $GPU -v $REMOTE/syclnn:/work/syclnn -v $REMOTE/cudann:/work/cudann -v $REMOTE/ompnn:/work/ompnn -v $REMOTE/fnn-bench:/work/fnn-bench"

sync() {
    ssh "$HOST" mkdir -p "$REMOTE"
    for r in syclnn cudann ompnn fnn-bench; do
        # never the results: ws-nvidia's rows are newer than ws-amd's fetched copy
        rsync -a --delete --exclude build --exclude .venv --exclude .wheels --exclude __pycache__ --exclude results --exclude .cache --exclude analysis "$AC/$r/" "$HOST:$REMOTE/$r/"
    done
    for img in fnn-cuda fnn-sycl; do
        if ! ssh "$HOST" podman image exists localhost/$img:dev; then
            podman save localhost/$img:dev | ssh "$HOST" podman load
        fi
    done
}

build_all() {
    # syclnn (DPC++, spir64 + sm_61 + sm_80), cudann (nvcc sm_61;sm_80), ompnn (clang-18 nvptx, gcc-14 nvptx, clang-22 host)
    ssh "$HOST" "$RUN -w /work/syclnn localhost/fnn-sycl:dev bash -c '
        set -e; export CC=clang CXX=clang++ SYCLNN_TARGETS=\"spir64;nvidia_gpu_sm_61\" SYCLNN_ONEMATH_ROOT=/opt/onemath CMAKE_BUILD_PARALLEL_LEVEL=8
        # one CUDA device image only: with nvidia_gpu_sm_61 and nvidia_gpu_sm_80 in the same fat
        # binary the DPC++ CUDA adapter fails to load the program on the 1080 Ti (2026-09-04);
        # the A30 gets its own sm_80 build
        rm -rf /work/fnn-bench/.wheels/ws-nvidia-sycl
        pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-sycl --config-settings=build-dir=/tmp/b .
        sycl-ls'"
    ssh "$HOST" "$RUN -w /work/cudann localhost/fnn-cuda:dev bash -c '
        set -e; export CUDANN_CUDA_ARCHS=\"61;80\" CUDAHOSTCXX=g++-13 CMAKE_BUILD_PARALLEL_LEVEL=8
        rm -rf /work/fnn-bench/.wheels/ws-nvidia-cuda
        pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-cuda --config-settings=build-dir=/tmp/b .
        cd /work/ompnn
        for cfg in \"clang18nv clang++-18 nvidia sm_61\" \"gcc14nv g++-14 nvidia sm_61\" \"clang22 clang++-22 cpu -\"; do
            set -- \$cfg
            export CXX=\$2 OMPNN_TARGET=\$3 OMPNN_OFFLOAD_ARCH=\$([ \$4 = - ] && echo || echo \$4) OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp
            rm -rf /work/fnn-bench/.wheels/ws-nvidia-omp-\$1
            pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-omp-\$1 --config-settings=build-dir=/tmp/b-\$1 . || echo \"BUILD FAILED: \$1\"
        done'"
}

tests() {
    build_all
    ssh "$HOST" "$RUN -w /work/syclnn localhost/fnn-sycl:dev bash -c '
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl
        python -c \"import syclnn; print(syclnn.devices())\"
        for o in \"--dtype double\" \"--dtype float\" \"--option memory=shared\" \"--option queue=in_order\" \"--option fine_deps=False\" \"--blas tiled --dtype double\" \"--blas tiled --dtype float\"; do printf \"syclnn cuda:gpu %-28s \" \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
        pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | tail -1'"
    ssh "$HOST" "$RUN -w /work/cudann localhost/fnn-cuda:dev bash -c '
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-cuda
        python -c \"import cudann; print(cudann.devices())\"
        for o in \"--dtype double\" \"--dtype float\" \"--option memory=shared\" \"--option memory=host\" \"--option queue=in_order\" \"--option queue=graph\" \"--option streams=8\" \"--option queue=graph --option streams=4\" \"--option blas=tiled --dtype double\" \"--option blas=tiled --dtype float\" \"--option blas=tiled --option queue=graph\"; do printf \"cudann %-40s \" \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
        cd /work/ompnn
        for w in clang18nv gcc14nv; do export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-omp-\$w; for o in \"\" \"--option blas=tiled\" \"--option blas=omp --dtype float\"; do printf \"ompnn %-10s gpu %-24s \" \$w \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done; done'"
}

matrix_sycl() {
    ssh "$HOST" "$RUN -w /work/fnn-bench localhost/fnn-sycl:dev bash -c '
        export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl
        fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=gpu --out results/ws-nvidia/$DATE/sycl-gpu
        fnnbench sweep --plan plans/e6_sycl_ablations.json --select device=gpu --out results/ws-nvidia/$DATE/sycl-gpu-ablations
        fnnbench sweep --plan plans/w4_sweep_gpu.json --out results/ws-nvidia/$DATE/gpu-w4-sycl
        fnnbench sweep --plan plans/e3_cuda_vs_sycl_gpu.json --select backend=syclnn --out results/ws-nvidia/$DATE/gpu-cuda-vs-sycl
        fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=syclnn --out results/ws-nvidia/$DATE/e7-tiled-gpu
        fnnbench sweep --plan plans/e5_inference.json --select device=gpu --select backend=syclnn --out results/ws-nvidia/$DATE/e5-infer-gpu
        fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=syclnn --out results/ws-nvidia/$DATE/e5-breakdown-gpu
        for dt in float double; do fnnbench peak --backend syclnn --device gpu --dtype \$dt --size 8192 --out results/ws-nvidia/$DATE/peaks; done
        fnnbench peak --backend syclnn --device gpu --dtype float --size 8192 --option blas=tiled --out results/ws-nvidia/$DATE/peaks-tiled
        nsys profile -o results/ws-nvidia/$DATE/nsys-syclnn-mnist --force-overwrite true fnnbench run --backend syclnn --device gpu --workload mnist-512-256 --batch 256 --dtype float --epochs 1 --repeat 1 --warmup 1 --option profile=True --samples 8192'"
}

matrix_cuda() {
    ssh "$HOST" "$RUN -w /work/fnn-bench localhost/fnn-cuda:dev bash -c '
        export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-cuda
        fnnbench sweep --plan plans/e3_cuda_vs_sycl_gpu.json --select backend=cudann --out results/ws-nvidia/$DATE/gpu-cuda-vs-sycl
        fnnbench sweep --plan plans/w4_sweep_gpu.json --backend cudann --out results/ws-nvidia/$DATE/gpu-w4-cuda
        fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=cudann --out results/ws-nvidia/$DATE/e7-tiled-gpu
        fnnbench sweep --plan plans/e5_inference.json --select device=gpu --select backend=cudann --out results/ws-nvidia/$DATE/e5-infer-gpu
        fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=cudann --out results/ws-nvidia/$DATE/e5-breakdown-gpu
        for dt in float double; do fnnbench peak --backend cudann --device gpu --dtype \$dt --size 8192 --out results/ws-nvidia/$DATE/peaks; done
        nsys profile -o results/ws-nvidia/$DATE/nsys-cudann-mnist --force-overwrite true fnnbench run --backend cudann --device gpu --workload mnist-512-256 --batch 256 --dtype float --epochs 1 --repeat 1 --warmup 1 --option profile=True --samples 8192'"
}

matrix_omp() {
    ssh "$HOST" "$RUN -w /work/fnn-bench localhost/fnn-cuda:dev bash -c '
        export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        for w in clang18nv gcc14nv; do
            export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-omp-\$w
            fnnbench sweep --plan plans/e4_omp_gpu.json --out results/ws-nvidia/$DATE/omp-gpu-\$w
            fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=ompnn --out results/ws-nvidia/$DATE/e7-tiled-omp-gpu-\$w
            fnnbench sweep --plan plans/e5_inference.json --select device=gpu --select backend=ompnn --out results/ws-nvidia/$DATE/e5-infer-omp-gpu-\$w
            fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=ompnn --out results/ws-nvidia/$DATE/e5-breakdown-omp-gpu-\$w
            fnnbench peak --backend ompnn --device gpu --dtype float --size 8192 --out results/ws-nvidia/$DATE/peaks-omp-\$w
        done
        # 12 = physical cores (2 x 6; the image default 16 is ws-amd's); unbound: libomp + OpenBLAS-openmp collapse when bound
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-omp-clang22 OMP_NUM_THREADS=12 OMP_PROC_BIND=false; unset OMP_PLACES
        fnnbench sweep --plan plans/e4_omp_cpu.json --out results/ws-nvidia/$DATE/omp-cpu-clang22
        fnnbench peak --backend ompnn --device cpu --dtype float --size 4096 --out results/ws-nvidia/$DATE/peaks-omp-clang22'"
}

matrix_cuda_omp() {
    matrix_cuda || echo "matrix: cudann block failed"
    matrix_omp || echo "matrix: ompnn block failed"
}

matrix() {
    # each block tolerates the others' failures (a failing peak or nsys run used to abort everything)
    matrix_sycl || echo "matrix: syclnn block failed"
    matrix_cuda_omp || echo "matrix: cudann/ompnn block failed"
}

omp_clang18() {
    # clang-18 nvptx path alone (needs clang-tools-18 in fnn-cuda:dev; the image on ws-nvidia was
    # patched in place on 2026-09-04): build, parity, E4/E7/E5 GPU rows, peak, fetch
    ssh "$HOST" "$RUN -w /work/ompnn localhost/fnn-cuda:dev bash -c '
        export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export CXX=clang++-18 OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH=sm_61 OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CMAKE_BUILD_PARALLEL_LEVEL=8
        rm -rf /work/fnn-bench/.wheels/ws-nvidia-omp-clang18nv
        pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-omp-clang18nv --config-settings=build-dir=/tmp/b-clang18nv . || { echo "BUILD FAILED: clang18nv"; exit 1; }
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-omp-clang18nv
        for o in \"--dtype double\" \"--dtype float\" \"--option blas=tiled --dtype double\" \"--option blas=tiled --dtype float\" \"--option blas=omp --dtype float\"; do printf \"ompnn clang18nv gpu %-34s \" \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
        cd /work/fnn-bench
        fnnbench sweep --plan plans/e4_omp_gpu.json --out results/ws-nvidia/$DATE/omp-gpu-clang18nv
        fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=ompnn --out results/ws-nvidia/$DATE/e7-tiled-omp-gpu-clang18nv
        fnnbench sweep --plan plans/e5_inference.json --select device=gpu --select backend=ompnn --out results/ws-nvidia/$DATE/e5-infer-omp-gpu-clang18nv
        fnnbench peak --backend ompnn --device gpu --dtype float --size 8192 --out results/ws-nvidia/$DATE/peaks-omp-clang18nv'"
    fetch
}

sycl_fix() {
    # syclnn alone: rebuild (sm_61-only image), parity, its matrix block, fetch
    ssh "$HOST" "$RUN -w /work/syclnn localhost/fnn-sycl:dev bash -c '
        set -e; export CC=clang CXX=clang++ SYCLNN_TARGETS=\"spir64;nvidia_gpu_sm_61\" SYCLNN_ONEMATH_ROOT=/opt/onemath CMAKE_BUILD_PARALLEL_LEVEL=8
        rm -rf /work/fnn-bench/.wheels/ws-nvidia-sycl
        pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-sycl --config-settings=build-dir=/tmp/b .
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl
        for o in \"--dtype double\" \"--dtype float\" \"--option memory=shared\" \"--option queue=in_order\" \"--option fine_deps=False\" \"--blas tiled --dtype double\" \"--blas tiled --dtype float\"; do printf \"syclnn cuda:gpu %-28s \" \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
        pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | tail -1'" || echo "sycl_fix: build/tests failed"
    matrix_sycl || echo "sycl_fix: matrix block failed"
    fetch
}

omp_gcc14nv() {
    # gcc-14 nvptx alone: rebuild (tiled kernel change), tiled parity, E7 rows, fetch
    ssh "$HOST" "$RUN -w /work/ompnn localhost/fnn-cuda:dev bash -c '
        export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export CXX=g++-14 OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH=sm_61 OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CMAKE_BUILD_PARALLEL_LEVEL=8
        rm -rf /work/fnn-bench/.wheels/ws-nvidia-omp-gcc14nv
        pip install -q --no-deps --target /work/fnn-bench/.wheels/ws-nvidia-omp-gcc14nv --config-settings=build-dir=/tmp/b-gcc14nv . || { echo \"BUILD FAILED: gcc14nv\"; exit 1; }
        export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-omp-gcc14nv
        for o in \"--option blas=tiled --dtype double\" \"--option blas=tiled --dtype float\" \"--dtype float\"; do printf \"ompnn gcc14nv gpu %-30s \" \"\$o\"; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
        cd /work/fnn-bench
        fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=ompnn --select workload=monk,cup,mnist-512-256,sweep-w256-d4-b256,sweep-w1024-d4-b256 --out results/ws-nvidia/$DATE/e7-tiled-omp-gpu-gcc14nv'"
    fetch
}

fetch() {
    rsync -a "$HOST:$REMOTE/fnn-bench/results/ws-nvidia/" "$HERE/results/ws-nvidia/"
}

case ${1:-} in
sync) sync ;;
build) build_all ;;
tests) tests ;;
matrix) matrix ;;
matrix-cuda-omp) matrix_cuda_omp ;;
matrix-cuda) matrix_cuda ;;
matrix-omp) matrix_omp ;;
matrix-sycl) matrix_sycl ;;
fetch) fetch ;;
omp-clang18) omp_clang18 ;;
sycl-fix) sycl_fix ;;
omp-gcc14nv) omp_gcc14nv ;;
all) sync; build_all; tests; matrix; fetch ;;
*) echo "usage: $0 sync|build|tests|matrix|matrix-sycl|matrix-cuda|matrix-omp|matrix-cuda-omp|fetch|omp-clang18|sycl-fix|omp-gcc14nv|all"; exit 1 ;;
esac
