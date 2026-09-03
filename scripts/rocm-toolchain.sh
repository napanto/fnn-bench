#!/usr/bin/env bash
# Build the AMD-GPU SYCL toolchain used on ws-amd (RX 7900 XTX, gfx1100):
#   AdaptiveCpp v25.10.0 (SSCP "generic" target, ROCm backend, OpenMP CPU backend)
#   oneMath v0.9 with the rocBLAS + NETLIB(OpenBLAS) BLAS backends, compiled with acpp
#   a Python venv with the packages the libraries and the testkit need
#
# Runs *inside* the `fnn-rocm` distrobox (Ubuntu 24.04 + ROCm 7.2.4, see docs/toolchains.md):
#   distrobox enter fnn-rocm -- bash fnn-bench/scripts/rocm-toolchain.sh
# Idempotent: every stage is skipped when its install prefix already exists.
# Job count is capped (JOBS=8) on purpose: ws-amd has been OOM-frozen by unbounded builds.
set -euo pipefail

PREFIX=${FNN_ROCM_PREFIX:-$HOME/.local/opt/fnn-rocm}
BUILD=${FNN_ROCM_BUILD:-$HOME/.cache/fnn-build/rocm}
JOBS=${JOBS:-8}
ACPP_TAG=${ACPP_TAG:-v25.10.0}
ONEMATH_TAG=${ONEMATH_TAG:-v0.9}
GFX=${GFX:-gfx1100}
LLVM_DIR=${LLVM_DIR:-/usr/lib/llvm-18/cmake}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

mkdir -p "$PREFIX" "$BUILD"
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS

log() { printf '\n==== [%s] %s\n' "$(date +%T)" "$*"; }

# ---------------------------------------------------------------- CBLAS shim
# Ubuntu installs each OpenBLAS variant under openblas-<variant>/; oneMath's
# FindNETLIB wants a root with lib/libcblas.so, lib/libblas.so and include/cblas.h.
for v in openmp pthread; do
    d=$PREFIX/openblas-$v
    if [ ! -e "$d/lib/libcblas.so" ]; then
        log "CBLAS shim for openblas-$v -> $d"
        mkdir -p "$d/lib" "$d/include"
        ln -sf /usr/lib/x86_64-linux-gnu/openblas-$v/libopenblas.so "$d/lib/libcblas.so"
        ln -sf /usr/lib/x86_64-linux-gnu/openblas-$v/libopenblas.so "$d/lib/libblas.so"
        ln -sf /usr/lib/x86_64-linux-gnu/openblas-$v/libopenblas.so "$d/lib/libopenblas.so"
        ln -sf /usr/include/x86_64-linux-gnu/openblas-$v/cblas.h "$d/include/cblas.h"
        ln -sf /usr/include/x86_64-linux-gnu/openblas-$v/openblas_config.h "$d/include/openblas_config.h"
    fi
done

# ---------------------------------------------------------------- AdaptiveCpp
if [ ! -x "$PREFIX/acpp/bin/acpp" ]; then
    log "AdaptiveCpp $ACPP_TAG"
    if [ ! -d "$BUILD/acpp/.git" ]; then
        git clone --depth 1 -b "$ACPP_TAG" https://github.com/AdaptiveCpp/AdaptiveCpp.git "$BUILD/acpp"
    fi
    cmake -S "$BUILD/acpp" -B "$BUILD/acpp/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX/acpp" \
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
        -DLLVM_DIR="$LLVM_DIR" \
        -DWITH_ROCM_BACKEND=ON -DROCM_PATH="$ROCM_PATH" \
        -DWITH_CUDA_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF \
        -DWITH_SSCP_COMPILER=ON -DWITH_ACCELERATED_CPU=ON \
        -DDEFAULT_TARGETS=generic
    ninja -C "$BUILD/acpp/build" -j"$JOBS"
    ninja -C "$BUILD/acpp/build" install
fi
export PATH="$PREFIX/acpp/bin:$PATH"
acpp --acpp-version | head -3 || true

# ---------------------------------------------------------------- oneMath (rocBLAS + NETLIB)
if [ ! -e "$PREFIX/onemath/lib/cmake/oneMath/oneMathConfig.cmake" ]; then
    log "oneMath $ONEMATH_TAG (AdaptiveCpp, rocBLAS + NETLIB)"
    if [ ! -d "$BUILD/onemath/.git" ]; then
        git clone --depth 1 -b "$ONEMATH_TAG" https://github.com/uxlfoundation/oneMath.git "$BUILD/onemath"
    fi
    # oneMath's NETLIB backend redefines cblas_i?amin, which OpenBLAS already declares
    if ! grep -q OPENBLAS_VERSION "$BUILD/onemath/src/blas/backends/netlib/netlib_level1.cpp"; then
        patch -p1 -d "$BUILD/onemath" < "$(dirname "$0")/../containers/patches/onemath-netlib-openblas.patch"
    fi
    cmake -S "$BUILD/onemath" -B "$BUILD/onemath/build-acpp" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX/onemath" \
        -DONEMATH_SYCL_IMPLEMENTATION=adaptivecpp \
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER="$PREFIX/acpp/bin/acpp" \
        -DACPP_TARGETS=generic \
        -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_ROCBLAS_BACKEND=ON -DHIP_TARGETS="$GFX" \
        -DENABLE_NETLIB_BACKEND=ON -DREF_BLAS_ROOT="$PREFIX/openblas-openmp" \
        -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF
    cmake --build "$BUILD/onemath/build-acpp" -j"$JOBS"
    cmake --install "$BUILD/onemath/build-acpp"
fi

# ---------------------------------------------------------------- Python venv
if [ ! -x "$PREFIX/venv/bin/python" ]; then
    log "Python venv"
    python3 -m venv "$PREFIX/venv"
    "$PREFIX/venv/bin/pip" install -q -U pip
    "$PREFIX/venv/bin/pip" install -q numpy==2.5.2 pytest==9.1.1 psutil==7.2.2 \
        pybind11==3.1.0 scikit-build-core==1.0.3 pybind11-stubgen==2.5.5 cmake ninja
fi

log "done: PREFIX=$PREFIX"
ls "$PREFIX"
