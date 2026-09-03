# fnn-sycl-generic: fnn-sycl + a second oneMath build with the *generic SYCL BLAS*
# backend (pure SYCL GEMM kernels, the "no vendor library" data point of E1).
# The backend is exclusive (cannot coexist with MKLCPU/NETLIB in one oneMath), so
# it gets its own prefix /opt/onemath-generic and its own syclnn wheel/venv.
#
# GENERIC_BLAS_TUNING_TARGET=INTEL_CPU makes oneMath AOT-compile every kernel for
# spir64_x86_64 (opencl-aot), which takes hours at -j8: build this image in the
# background / in CI, never on the critical path.
#
#   podman build --memory=20g -f containers/fnn-sycl-generic.Containerfile -t fnn-sycl-generic:dev containers/
ARG BASE=localhost/fnn-sycl:dev
FROM ${BASE}
ARG JOBS=8
ARG TUNING=INTEL_CPU

RUN cmake -S /opt/src/onemath -B /tmp/onemath-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/onemath-generic \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_GENERIC_BLAS_BACKEND=ON -DGENERIC_BLAS_TUNING_TARGET=${TUNING} \
        -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    && cmake --build /tmp/onemath-build -j${JOBS} \
    && cmake --install /tmp/onemath-build \
    && rm -rf /tmp/onemath-build

ENV ONEMATH_GENERIC_ROOT=/opt/onemath-generic
LABEL org.opencontainers.image.description="fnn-sycl + oneMath generic SYCL BLAS backend (/opt/onemath-generic)"
