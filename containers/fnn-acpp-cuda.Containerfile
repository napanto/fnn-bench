# AdaptiveCpp with the CUDA backend on top of fnn-cuda:dev, for the SYCL implementation
# comparison (E7/optional: DPC++ vs AdaptiveCpp on the same NVIDIA GPU) and the portability
# figure. Mirrors scripts/rocm-toolchain.sh (AdaptiveCpp v25.10.0 SSCP "generic"
# target, oneMath v0.9 built with acpp) with cuBLAS instead of rocBLAS.
#   podman build --memory=40g -f containers/fnn-acpp-cuda.Containerfile -t fnn-acpp-cuda:dev containers/
# Build syclnn inside with:
#   CXX=/opt/acpp/bin/acpp CC=clang-18 SYCLNN_SYCL_IMPL=adaptivecpp SYCLNN_ONEMATH_ROOT=/opt/onemath-acpp \
#   CMAKE_PREFIX_PATH=/opt/acpp:/opt/onemath-acpp pip install .
FROM localhost/fnn-cuda:dev
ARG ACPP_TAG=v25.10.0
ARG ONEMATH_TAG=v0.9
ARG JOBS=8
ENV CUDA_HOME=/usr/local/cuda-12.9
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-context-dev libboost-fiber-dev libboost-filesystem-dev libboost-test-dev git ninja-build \
    && rm -rf /var/lib/apt/lists/*
# AdaptiveCpp: clang-18 host compiler, CUDA backend, SSCP single-pass compiler (generic target)
RUN git clone --depth 1 -b ${ACPP_TAG} https://github.com/AdaptiveCpp/AdaptiveCpp.git /tmp/acpp \
    && cmake -S /tmp/acpp -B /tmp/acpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/acpp \
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DLLVM_DIR=/usr/lib/llvm-18/cmake \
        -DWITH_CUDA_BACKEND=ON -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
        -DWITH_ROCM_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF \
        -DWITH_SSCP_COMPILER=ON -DWITH_ACCELERATED_CPU=ON -DDEFAULT_TARGETS=generic \
    && ninja -C /tmp/acpp/build -j${JOBS} && ninja -C /tmp/acpp/build install && rm -rf /tmp/acpp
# oneMath with acpp: cuBLAS (linked against the driver stub) + NETLIB on OpenBLAS
COPY patches/onemath-netlib-openblas.patch /tmp/onemath-netlib-openblas.patch
RUN git clone --depth 1 -b ${ONEMATH_TAG} https://github.com/uxlfoundation/oneMath.git /tmp/onemath \
    && (grep -q OPENBLAS_VERSION /tmp/onemath/src/blas/backends/netlib/netlib_level1.cpp || patch -p1 -d /tmp/onemath < /tmp/onemath-netlib-openblas.patch) \
    && cmake -S /tmp/onemath -B /tmp/onemath/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/onemath-acpp \
        -DONEMATH_SYCL_IMPLEMENTATION=adaptivecpp \
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=/opt/acpp/bin/acpp -DACPP_TARGETS=generic \
        -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_CUBLAS_BACKEND=ON -DCUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs/libcuda.so \
        -DENABLE_NETLIB_BACKEND=ON -DREF_BLAS_ROOT=/opt/openblas-openmp \
        -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    && cmake --build /tmp/onemath/build -j${JOBS} && cmake --install /tmp/onemath/build && rm -rf /tmp/onemath
ENV PATH=/opt/acpp/bin:$PATH
ENV ACPP_TARGETS=generic
