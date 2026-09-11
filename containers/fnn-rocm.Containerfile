# fnn-rocm: the AMD toolchain of the study (the same stack scripts/rocm-toolchain.sh builds in a
# ROCm distrobox on ws-amd, here as a pullable image):
#   ROCm 7.2.4 (hipcc, amdclang++ 22, rocBLAS / hipBLAS, rocprofv3) from the rocm/dev image,
#   gcc-14 with amdgcn offload, clang-18 (AdaptiveCpp's host compiler), both Ubuntu OpenBLAS variants,
#   AdaptiveCpp v25.10.0 (SSCP generic target, ROCm + OpenMP backends) in /opt/fnn-rocm/acpp,
#   oneMath v0.9 with the rocBLAS and NETLIB backends compiled by acpp in /opt/fnn-rocm/onemath,
#   Python venvs with the packages the libraries and the testkit need (/opt/fnn-rocm/venv for
#   syclnn/cudann-HIP and the harness, venv-amdclang and venv-gcc14amd for the ompnn compiler matrix).
# Serves: syclnn with AdaptiveCpp (RX 7900 XTX and the OpenMP host device), cudann's HIPify build,
# ompnn with amdclang++ and gcc-14 amdgcn offload, the ZLUDA cross-check.
#
# Build from the fnn-bench root (the recipe script and the oneMath patch are copied in):
#   podman build --memory=20g --build-arg IMAGE_NAME=fnn-rocm:dev --build-arg IMAGE_BUILT=$(date -u +%F) \
#       -f containers/fnn-rocm.Containerfile -t fnn-rocm:dev .
# Run with the GPU: podman run --device /dev/kfd --device /dev/dri --group-add keep-groups ...
ARG BASE=docker.io/rocm/dev-ubuntu-24.04:7.2.4-complete
FROM ${BASE}
ARG IMAGE_NAME=fnn-rocm:dev
ARG IMAGE_BUILT=unknown
ARG JOBS=8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 gcc-14-offload-amdgcn \
        clang-18 llvm-18-dev libclang-18-dev libclang-rt-18-dev libomp-18-dev lld-18 \
        libboost-context-dev libboost-fiber-dev libboost-filesystem-dev libboost-test-dev \
        libopenblas-openmp-dev libopenblas-pthread-dev \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        cmake ninja-build git patch curl ca-certificates rsync \
    && rm -rf /var/lib/apt/lists/*

# the same recipe as on the workstation, installed under /opt/fnn-rocm (CMake finds hip/rocBLAS under /opt/rocm)
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm CMAKE_PREFIX_PATH=/opt/rocm
COPY scripts/rocm-toolchain.sh /tmp/fnn-bench/scripts/rocm-toolchain.sh
COPY containers/patches/onemath-netlib-openblas.patch /tmp/fnn-bench/containers/patches/onemath-netlib-openblas.patch
RUN FNN_ROCM_PREFIX=/opt/fnn-rocm FNN_ROCM_BUILD=/tmp/fnn-build JOBS=${JOBS} \
        bash /tmp/fnn-bench/scripts/rocm-toolchain.sh \
    && rm -rf /tmp/fnn-build /tmp/fnn-bench \
    && for v in amdclang gcc14amd; do \
           python3 -m venv /opt/fnn-rocm/venv-$v \
           && /opt/fnn-rocm/venv-$v/bin/pip install -q -U pip \
           && /opt/fnn-rocm/venv-$v/bin/pip install -q numpy==2.5.2 pytest==9.1.1 psutil==7.2.2 \
                  pybind11==3.1.0 scikit-build-core==1.0.3 pybind11-stubgen==2.5.5 cmake ninja; \
       done \
    && /opt/fnn-rocm/acpp/bin/acpp --acpp-version | head -2

ENV PATH=/opt/fnn-rocm/venv/bin:/opt/fnn-rocm/acpp/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH \
    SYCLNN_SYCL_IMPL=adaptivecpp \
    SYCLNN_ONEMATH_ROOT=/opt/fnn-rocm/onemath \
    AdaptiveCpp_DIR=/opt/fnn-rocm/acpp/lib/cmake/AdaptiveCpp \
    ACPP_TARGETS=generic \
    OPENBLAS_ROOT=/opt/fnn-rocm/openblas-openmp \
    OMPNN_BLAS_ROOT=/opt/fnn-rocm/openblas-openmp \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    OMP_NUM_THREADS=16 \
    FNN_IMAGE=${IMAGE_NAME} \
    FNN_IMAGE_BUILT=${IMAGE_BUILT}

LABEL org.opencontainers.image.source=https://github.com/napanto/fnn-bench \
      org.opencontainers.image.description="fnn-bench AMD toolchain: ROCm 7.2.4, AdaptiveCpp 25.10 (generic SSCP), oneMath rocBLAS/NETLIB, gcc-14 amdgcn offload, amdclang++" \
      fnn.image="${IMAGE_NAME}" fnn.image.built="${IMAGE_BUILT}"

WORKDIR /work
CMD ["/bin/bash"]
