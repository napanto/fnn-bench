# fnn-cuda: NVIDIA-side toolchain image (also the base of fnn-sycl).
#
#   * CUDA 12.9 (nvcc, cuBLAS, NVTX, CUPTI, Nsight Systems)  - cudann, syclnn-on-CUDA
#   * gcc-13 (nvcc host compiler), gcc-14 + nvptx/amdgcn offload - ompnn (CPU + target)
#   * clang-22 + libomp (host OpenMP only: apt.llvm.org ships no offload runtime)
#   * clang-18 (Ubuntu) + libomp-18 with libomptarget nvptx/amdgpu   - ompnn (target)
#     (libomp5-18/libomp-18-dev/libclang-rt-18-dev are unpacked with dpkg -x into
#      /usr/lib/llvm-18 only: the Debian packages "Conflict" with their LLVM-22
#      siblings because of a shared /usr/lib/x86_64-linux-gnu/libomp.so.5 symlink.
#      Programs built with clang-18 must carry an rpath to /usr/lib/llvm-18/lib.)
#   * OpenBLAS (openmp + pthread variants) and Intel oneMKL (pip) - CBLAS backends
#   * Python 3.12 venv (/opt/venv): numpy, pytest, pybind11 3.1, scikit-build-core 1.x
#
# CUDA 12.9 is the last toolkit that still targets Pascal (GTX 1080 Ti); it also
# runs on Ampere (A30). Build with a memory cap:
#   podman build --memory=20g -f containers/fnn-cuda.Containerfile -t fnn-cuda:dev containers/
FROM docker.io/library/ubuntu:24.04

ARG CUDA_PKG_VER=12-9
ARG CUDA_VER=12.9
ARG LLVM_VER=22
ARG WITH_NCU=0

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC LANG=C.UTF-8

# --- base distro toolchain -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg2 wget git xz-utils unzip less vim-tiny \
        build-essential gcc-13 g++-13 gcc-14 g++-14 \
        gcc-14-offload-nvptx gcc-14-offload-amdgcn \
        cmake ninja-build pkg-config \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        libopenblas-openmp-dev libopenblas-pthread-dev \
        libnuma-dev numactl hwloc \
        ocl-icd-opencl-dev ocl-icd-libopencl1 clinfo \
        libtinfo6 libxml2 zlib1g libedit2 libncurses6 libelf1 \
    && rm -rf /var/lib/apt/lists/*

# --- LLVM/clang from apt.llvm.org --------------------------------------------
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VER} main" \
       > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        clang-${LLVM_VER} libomp-${LLVM_VER}-dev libclang-rt-${LLVM_VER}-dev lld-${LLVM_VER} llvm-${LLVM_VER} \
        clang-18 clang-tools-18 \
    && cd /tmp && apt-get download libomp5-18 libomp-18-dev libclang-rt-18-dev \
    && mkdir -p /tmp/llvm18 && for f in /tmp/*.deb; do dpkg -x "$f" /tmp/llvm18; done \
    && cp -a /tmp/llvm18/usr/lib/llvm-18/. /usr/lib/llvm-18/ && rm -rf /tmp/llvm18 /tmp/*.deb \
    && ls /usr/lib/llvm-18/lib/libomptarget-nvptx-sm_61.bc /usr/lib/llvm-18/lib/libomptarget-amdgpu-gfx1100.bc \
          /usr/lib/llvm-18/lib/libomptarget.so.18.1 /usr/lib/llvm-18/lib/libomp.so.5 \
    && rm -rf /var/lib/apt/lists/* \
    && for t in clang clang++ lld ld.lld llvm-config llvm-objdump; do \
         update-alternatives --install /usr/local/bin/$t $t /usr/lib/llvm-${LLVM_VER}/bin/$t 100; done

# --- CUDA 12.9 (network repo, only what the project needs) ------------------
RUN curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_PKG_VER} cuda-cudart-dev-${CUDA_PKG_VER} cuda-driver-dev-${CUDA_PKG_VER} \
        cuda-cccl-${CUDA_PKG_VER} libcublas-dev-${CUDA_PKG_VER} cuda-nvtx-${CUDA_PKG_VER} \
        cuda-nvrtc-dev-${CUDA_PKG_VER} cuda-profiler-api-${CUDA_PKG_VER} cuda-cupti-${CUDA_PKG_VER} \
        cuda-nsight-systems-${CUDA_PKG_VER} \
    && if [ "$WITH_NCU" = "1" ]; then apt-get install -y --no-install-recommends cuda-nsight-compute-${CUDA_PKG_VER}; fi \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-${CUDA_VER} \
    CUDA_PATH=/usr/local/cuda-${CUDA_VER}
ENV PATH=/opt/venv/bin:${CUDA_HOME}/bin:${PATH} \
    LIBRARY_PATH=${CUDA_HOME}/lib64/stubs

# --- Python venv + Intel oneMKL from PyPI (classic + SYCL BLAS) -------------
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        numpy==2.5.2 pytest==9.1.1 psutil==7.2.2 \
        pybind11==3.1.0 scikit-build-core==1.0.3 pybind11-stubgen==2.5.5 \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps \
        mkl==2026.1.0 mkl-devel==2026.1.0 mkl-include==2026.1.0 \
        mkl-devel-dpcpp==2026.1.0 onemkl-sycl-blas==2026.1.0 onemkl-sycl-include==2026.1.0 \
        tbb==2023.1.0 tbb-devel==2023.1.0 intel-openmp==2026.1.1 \
    && echo "/opt/venv/lib" > /etc/ld.so.conf.d/venv.conf \
    && echo "${CUDA_HOME}/lib64" > /etc/ld.so.conf.d/cuda.conf \
    && ldconfig

# CBLAS shim dirs so that plain find_library(cblas)/cblas.h works for every
# OpenBLAS variant (Ubuntu installs them under openblas-<variant>/).
RUN for v in openmp pthread; do \
        d=/opt/openblas-$v; mkdir -p $d/lib $d/include; \
        ln -s /usr/lib/x86_64-linux-gnu/openblas-$v/libopenblas.so $d/lib/libcblas.so; \
        ln -s /usr/lib/x86_64-linux-gnu/openblas-$v/libopenblas.so $d/lib/libblas.so; \
        ln -s /usr/lib/x86_64-linux-gnu/openblas-$v/libopenblas.so $d/lib/libopenblas.so; \
        ln -s /usr/include/x86_64-linux-gnu/openblas-$v/cblas.h $d/include/cblas.h; \
        ln -s /usr/include/x86_64-linux-gnu/openblas-$v/openblas_config.h $d/include/openblas_config.h; \
    done && ls -l /opt/openblas-openmp/lib /opt/openblas-openmp/include

ENV MKLROOT=/opt/venv \
    OPENBLAS_ROOT=/opt/openblas-openmp \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    OMP_NUM_THREADS=16

LABEL org.opencontainers.image.source=https://github.com/napanto/fnn-bench \
      org.opencontainers.image.description="fnn-bench NVIDIA/OpenMP toolchain: CUDA 12.9, gcc-14 offload, clang-22, OpenBLAS, oneMKL"

WORKDIR /work
CMD ["/bin/bash"]
