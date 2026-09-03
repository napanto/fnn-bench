#!/usr/bin/env bash
# The the local runs local experiment matrix on ws-amd (Threadripper 2950X + RX 7900 XTX).
#
# Every environment runs its plans sequentially (CPU sweeps and GPU sweeps must
# not overlap: the BLAS threads would perturb the GPU launch path).  Results go
# to results/ws-amd/<date>/<env>/<backend>.jsonl; re-running skips finished rows.
#
#   fnn-bench/scripts/matrix-ws-amd.sh [date] [env...]     envs: sycl-cpu sycl-generic rocm-gpu omp-cpu omp-gpu
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${1:-$(date +%F)}
shift || true
ENVS=${*:-"sycl-cpu sycl-generic rocm-gpu omp-cpu omp-gpu"}
RES=$HERE/results/ws-amd/$DATE
mkdir -p "$RES" "$HERE/.wheels"
PLANS=$HERE/plans
PODMAN="podman run --rm --memory=20g --security-opt label=disable -v $AC/syclnn:/work/syclnn -v $AC/ompnn:/work/ompnn -v $HERE:/work/fnn-bench"
export OMP_PROC_BIND=close OMP_PLACES=cores

log() { printf '\n==== [%s] %s\n' "$(date +%T)" "$*"; }

sycl_cpu() {
    log "syclnn on opencl:cpu (DPC++, MKLCPU + NETLIB): E1, E2/E6 cpu rows, W4 cpu"
    $PODMAN -w /work/syclnn localhost/fnn-sycl:dev bash -c "
        set -e; export SYCLNN_TARGETS='spir64;nvidia_gpu_sm_61;nvidia_gpu_sm_80' SYCLNN_ONEMATH_ROOT=/opt/onemath CMAKE_BUILD_PARALLEL_LEVEL=8
        pip install -q --no-deps --target /work/fnn-bench/.wheels/sycl-dpcpp --config-settings=build-dir=/tmp/syclnn-build . 2>&1 | grep -E 'error:' || true
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp OMP_PROC_BIND=close OMP_PLACES=cores
        cd /work/fnn-bench
        fnnbench sweep --plan plans/e1_cpu_blas.json --out results/ws-amd/$DATE/sycl-cpu
        fnnbench sweep --plan plans/e1_cpu_threads.json --out results/ws-amd/$DATE/sycl-cpu-threads
        fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --out results/ws-amd/$DATE/sycl-cpu
        fnnbench sweep --plan plans/e6_sycl_ablations.json --select device=cpu --out results/ws-amd/$DATE/sycl-cpu-ablations
        fnnbench sweep --plan plans/w4_sweep_cpu.json --out results/ws-amd/$DATE/sycl-cpu-w4
        fnnbench peak --backend syclnn --device cpu --dtype float --size 4096 --option blas=mklcpu --out results/ws-amd/$DATE/peaks
        fnnbench peak --backend syclnn --device cpu --dtype double --size 4096 --option blas=mklcpu --out results/ws-amd/$DATE/peaks
        fnnbench peak --backend syclnn --device cpu --dtype float --size 4096 --option blas=netlib --out results/ws-amd/$DATE/peaks
    "
}

sycl_generic() {
    log "syclnn on opencl:cpu with the generic SYCL BLAS backend: E1"
    $PODMAN -w /work/syclnn localhost/fnn-sycl-generic:dev bash -c "
        set -e; export SYCLNN_TARGETS='spir64' SYCLNN_ONEMATH_ROOT=/opt/onemath-generic CMAKE_BUILD_PARALLEL_LEVEL=8
        pip install -q --no-deps --target /work/fnn-bench/.wheels/sycl-generic --config-settings=build-dir=/tmp/syclnn-build-generic . 2>&1 | grep -E 'error:' || true
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        export PYTHONPATH=/work/fnn-bench/.wheels/sycl-generic OMP_PROC_BIND=close OMP_PLACES=cores
        cd /work/fnn-bench
        fnnbench sweep --plan plans/e1_cpu_generic.json --out results/ws-amd/$DATE/sycl-cpu-generic
        fnnbench peak --backend syclnn --device cpu --dtype float --size 4096 --option blas=generic --out results/ws-amd/$DATE/peaks
    "
}

rocm_gpu() {
    log "RX 7900 XTX: syclnn (AdaptiveCpp + rocBLAS) and cudann (HIP + hipBLAS): E2 gpu, E3, E6 gpu, W4 gpu"
    distrobox enter fnn-rocm -- bash -c "
        P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH; cd $HERE
        fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=gpu --out results/ws-amd/$DATE/sycl-gpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
        fnnbench sweep --plan plans/e3_cuda_vs_sycl_gpu.json --out results/ws-amd/$DATE/gpu-cuda-vs-sycl 2>&1 | grep -v 'AdaptiveCpp Warning'
        fnnbench sweep --plan plans/e6_sycl_ablations.json --select device=gpu --out results/ws-amd/$DATE/sycl-gpu-ablations 2>&1 | grep -v 'AdaptiveCpp Warning'
        fnnbench sweep --plan plans/w4_sweep_gpu.json --out results/ws-amd/$DATE/gpu-w4-sycl 2>&1 | grep -v 'AdaptiveCpp Warning'
        fnnbench sweep --plan plans/w4_sweep_gpu.json --backend cudann --out results/ws-amd/$DATE/gpu-w4-cuda 2>&1 | grep -v 'AdaptiveCpp Warning'
        for be in syclnn cudann; do for dt in float double; do fnnbench peak --backend \$be --device gpu --dtype \$dt --size 8192 --out results/ws-amd/$DATE/peaks 2>&1 | grep -v Warning | tail -1; done; done
        # the AdaptiveCpp wheel on the CPU (SYCL implementation comparison, E7)
        fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --select dtype=float --out results/ws-amd/$DATE/sycl-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    "
}

omp_cpu() {
    log "ompnn host path: gcc-14, clang-22, clang-18, gcc-14+MKL (fnn-cuda image): E4 cpu"
    $PODMAN -w /work/ompnn localhost/fnn-cuda:dev bash -c "
        pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
        for cfg in 'gcc14 g++-14 openblas /opt/openblas-openmp' 'clang22 clang++-22 openblas /opt/openblas-openmp' 'clang18 clang++-18 openblas /opt/openblas-openmp' 'gcc14mkl g++-14 mkl /opt/venv'; do
            set -- \$cfg
            export CXX=\$2 OMPNN_TARGET=cpu OMPNN_BLAS=\$3 OMPNN_BLAS_ROOT=\$4 CMAKE_BUILD_PARALLEL_LEVEL=8
            pip install -q --no-deps --target /work/fnn-bench/.wheels/omp-\$1 --config-settings=build-dir=/tmp/ompnn-\$1 . 2>&1 | grep -E 'error:' || true
            export PYTHONPATH=/work/fnn-bench/.wheels/omp-\$1 OMP_PROC_BIND=close OMP_PLACES=cores
            (cd /work/fnn-bench && fnnbench sweep --plan plans/e4_omp_cpu.json --out results/ws-amd/$DATE/omp-cpu-\$1 && fnnbench peak --backend ompnn --device cpu --dtype float --size 4096 --out results/ws-amd/$DATE/peaks-omp-\$1)
        done
    "
}

omp_gpu() {
    log "ompnn target path on the RX 7900 XTX: amdclang++, gcc-14-offload-amdgcn: E4 gpu"
    distrobox enter fnn-rocm -- bash -c "
        P=\$HOME/.local/opt/fnn-rocm; cd $HERE
        for v in venv-amdclang venv-gcc14amd; do
            export PATH=\$P/\$v/bin:\$PATH
            fnnbench sweep --plan plans/e4_omp_gpu.json --out results/ws-amd/$DATE/omp-gpu-\${v#venv-}
            fnnbench peak --backend ompnn --device gpu --dtype float --size 8192 --out results/ws-amd/$DATE/peaks-omp-\${v#venv-}
        done
    "
}

for e in $ENVS; do
    case $e in
    sycl-cpu) sycl_cpu ;;
    sycl-generic) sycl_generic ;;
    rocm-gpu) rocm_gpu ;;
    omp-cpu) omp_cpu ;;
    omp-gpu) omp_gpu ;;
    *) echo "unknown env $e" ;;
    esac
done
log "matrix done: $RES"
