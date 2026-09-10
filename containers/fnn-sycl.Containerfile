# fnn-sycl: fnn-cuda + open-source DPC++ (intel/llvm) + Intel OpenCL CPU runtime + oneMath.
#
#   * intel/llvm v7.1.0 "sycl_linux.tar.gz" (clang 22.1, libsycl 9, oneAPI 2026.1 code base);
#     the release tarball is built with the CUDA adapter, so one wheel runs on
#     opencl:cpu and cuda:gpu.
#   * oclcpuexp 2026-WW28: Intel's OpenCL CPU runtime (runs on AMD CPUs, needs TBB).
#   * oneMath v0.9 in /opt/onemath: MKLCPU + NETLIB(OpenBLAS) + cuBLAS backends.
#     The generic SYCL BLAS backend (exclusive) lives in the optional
#     fnn-sycl-generic image (see fnn-sycl-generic.Containerfile): its CPU AOT
#     compilation takes hours.
#
#   podman build --memory=20g -f containers/fnn-sycl.Containerfile -t fnn-sycl:dev containers/
ARG BASE=localhost/fnn-cuda:dev
FROM ${BASE}
ARG FNN_IMAGE=fnn-sycl:dev
ARG FNN_IMAGE_BUILT=unknown

ARG SYCL_TAG=v7.1.0
ARG OCLCPU_TAG=2026-WW28
ARG OCLCPU_FILE=oclcpuexp-2026.22.6.1.17_160000_rel.tar.gz
ARG ONEMATH_TAG=v0.9
ARG JOBS=8

# --- DPC++ toolchain --------------------------------------------------------
RUN mkdir -p /opt/sycl \
    && curl -fsSL https://github.com/intel/llvm/releases/download/${SYCL_TAG}/sycl_linux.tar.gz | tar -xz -C /opt/sycl \
    && echo "/opt/sycl/lib" > /etc/ld.so.conf.d/sycl.conf && ldconfig \
    && /opt/sycl/bin/clang++ --version | head -1

# --- Intel OpenCL CPU runtime (the opencl:cpu device) ----------------------
RUN mkdir -p /opt/oclcpu /etc/OpenCL/vendors \
    && curl -fsSL https://github.com/intel/llvm/releases/download/${OCLCPU_TAG}/${OCLCPU_FILE} | tar -xz -C /opt/oclcpu \
    && echo /opt/oclcpu/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd \
    && echo "/opt/oclcpu/x64" > /etc/ld.so.conf.d/oclcpu.conf && ldconfig \
    && ldconfig -p | grep -E 'libtbb\.so\.12|libintelocl' \
    && PATH=/opt/sycl/bin:$PATH sycl-ls

ENV PATH=/opt/sycl/bin:${PATH} \
    SYCL_ROOT=/opt/sycl \
    CC=clang CXX=clang++

# --- oneMath: MKLCPU + NETLIB + cuBLAS (run-time and compile-time dispatch) --
COPY patches/onemath-netlib-openblas.patch /opt/src/
RUN git clone --depth 1 -b ${ONEMATH_TAG} https://github.com/uxlfoundation/oneMath.git /opt/src/onemath \
    && patch -p1 -d /opt/src/onemath < /opt/src/onemath-netlib-openblas.patch \
    && cmake -S /opt/src/onemath -B /tmp/onemath-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/onemath \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DMKL_ROOT=/opt/venv \
        -DENABLE_MKLCPU_BACKEND=ON -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_NETLIB_BACKEND=ON -DREF_BLAS_ROOT=/opt/openblas-openmp \
        -DENABLE_CUBLAS_BACKEND=ON -DCUDAToolkit_ROOT=${CUDA_HOME} \
        -DCUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs/libcuda.so \
        -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    && cmake --build /tmp/onemath-build -j${JOBS} \
    && cmake --install /tmp/onemath-build \
    && rm -rf /tmp/onemath-build

ENV ONEMATH_ROOT=/opt/onemath \
    CMAKE_PREFIX_PATH=/opt/onemath \
    SYCL_CACHE_PERSISTENT=1

LABEL org.opencontainers.image.description="fnn-bench SYCL toolchain: intel/llvm v7.1.0 (CUDA adapter), oclcpuexp, oneMath v0.9 (MKLCPU, NETLIB/OpenBLAS, cuBLAS, generic SYCL BLAS)"

# identity of the image, recorded by fnnbench in every row (sysinfo.env / sysinfo.container)
ENV FNN_IMAGE=${FNN_IMAGE} FNN_IMAGE_BUILT=${FNN_IMAGE_BUILT}
LABEL fnn.image="${FNN_IMAGE}" fnn.image.built="${FNN_IMAGE_BUILT}"
