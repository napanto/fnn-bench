#!/usr/bin/env bash
# DPC++ (intel/llvm open-source, HIP adapter) on ws-amd's RX 7900 XTX, inside the fnn-rocm
# distrobox (ROCm 7.2.4): the SYCL implementation comparison on AMD (DPC++ vs AdaptiveCpp on
# the same GPU) and the one-wheel portability test (spir64 + nvidia_gpu_sm_61 + amd_gpu_gfx1100).
#   distrobox enter fnn-rocm -- bash fnn-bench/scripts/dpcpp-hip-toolchain.sh
# Installs into ~/.local/opt/fnn-rocm/{dpcpp,onemath-dpcpp,venv-dpcpp-hip}. ~30 min of CPU.
set -euo pipefail
PREFIX=$HOME/.local/opt/fnn-rocm
BUILD=${BUILD:-$HOME/.local/opt/fnn-rocm/build}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SYCL_TAG=${SYCL_TAG:-v7.1.0}
ONEMATH_TAG=${ONEMATH_TAG:-v0.9}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
GFX=${GFX:-gfx1100}
JOBS=${JOBS:-8}
log() { echo "==== [$(date +%T)] $*"; }
mkdir -p "$BUILD" "$PREFIX"

if [ ! -x "$PREFIX/dpcpp/bin/clang++" ]; then
    log "intel/llvm $SYCL_TAG tarball"
    mkdir -p "$PREFIX/dpcpp"
    curl -fsSL "https://github.com/intel/llvm/releases/download/${SYCL_TAG}/sycl_linux.tar.gz" | tar -xz -C "$PREFIX/dpcpp"
fi
export PATH="$PREFIX/dpcpp/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/dpcpp/lib:${LD_LIBRARY_PATH:-}"
clang++ --version | head -1
sycl-ls || true

if [ ! -e "$PREFIX/onemath-dpcpp/lib/cmake/oneMath/oneMathConfig.cmake" ]; then
    log "oneMath $ONEMATH_TAG with DPC++: rocBLAS + NETLIB"
    if [ ! -d "$BUILD/onemath/.git" ]; then
        git clone --depth 1 -b "$ONEMATH_TAG" https://github.com/uxlfoundation/oneMath.git "$BUILD/onemath"
    fi
    if ! grep -q OPENBLAS_VERSION "$BUILD/onemath/src/blas/backends/netlib/netlib_level1.cpp"; then
        patch -p1 -d "$BUILD/onemath" < "$HERE/containers/patches/onemath-netlib-openblas.patch"
    fi
    cmake -S "$BUILD/onemath" -B "$BUILD/onemath/build-dpcpp" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX/onemath-dpcpp" \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_ROCBLAS_BACKEND=ON -DHIP_TARGETS="$GFX" \
        -DENABLE_NETLIB_BACKEND=ON -DREF_BLAS_ROOT="$PREFIX/openblas-openmp" \
        -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF
    cmake --build "$BUILD/onemath/build-dpcpp" -j"$JOBS"
    cmake --install "$BUILD/onemath/build-dpcpp"
fi

if [ ! -x "$PREFIX/venv-dpcpp-hip/bin/python" ]; then
    log "venv-dpcpp-hip"
    python3 -m venv "$PREFIX/venv-dpcpp-hip"
    "$PREFIX/venv-dpcpp-hip/bin/pip" install -q numpy pytest "pybind11>=3.1" "scikit-build-core>=1.0" matplotlib
    "$PREFIX/venv-dpcpp-hip/bin/pip" install -q --no-deps -e "$HERE/testkit" -e "$HERE/bench"
fi
log "syclnn wheel: DPC++ with spir64 + amd_gpu_$GFX (+ nvidia_gpu_sm_61 when SYCLNN_PORTABLE=1)"
targets="spir64;amd_gpu_$GFX"
[ "${SYCLNN_PORTABLE:-0}" = 1 ] && targets="spir64;nvidia_gpu_sm_61;amd_gpu_$GFX"
cd "$HERE/../syclnn"
CC=clang CXX=clang++ SYCLNN_TARGETS="$targets" SYCLNN_ONEMATH_ROOT="$PREFIX/onemath-dpcpp" SYCLNN_CT_BACKENDS="netlib" \
    CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" "$PREFIX/venv-dpcpp-hip/bin/pip" install -q --no-deps --force-reinstall \
    --config-settings=build-dir="$BUILD/syclnn-dpcpp-hip" .
"$PREFIX/venv-dpcpp-hip/bin/python" -c "import syclnn; print(syclnn.build_info()); print(syclnn.devices())"
log "done: PATH=$PREFIX/dpcpp/bin:$PREFIX/venv-dpcpp-hip/bin LD_LIBRARY_PATH=$PREFIX/dpcpp/lib pytest --device gpu"
