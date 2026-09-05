# fnn-sycl + the ROCm pieces DPC++ needs to add an AMD device image (amd_gpu_gfx1100) to the
# same wheel that carries spir64 and nvidia_gpu_sm_61: the one-wheel portability experiment
# (optional experiments). Compile time needs the ROCm device libraries and lld; run time on an AMD host
# needs the HIP runtime (libamdhip64) for DPC++'s HIP adapter, on an NVIDIA host the CUDA
# driver (CDI), on any host the OpenCL CPU runtime already in fnn-sycl.
#   podman build --memory=20g -f containers/fnn-sycl-portable.Containerfile -t fnn-sycl-portable:dev containers/
#   AMD run: podman run --device /dev/kfd --device /dev/dri --group-add keep-groups ...
FROM localhost/fnn-sycl:dev
ARG ROCM_VER=7.2.4
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg2 \
    && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} noble main" > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update && apt-get install -y --no-install-recommends rocm-device-libs hip-runtime-amd hip-dev rocm-llvm \
    && rm -rf /var/lib/apt/lists/*
ENV ROCM_PATH=/opt/rocm
# syclnn portable wheel:
#   CC=clang CXX=clang++ SYCLNN_TARGETS="spir64;nvidia_gpu_sm_61;amd_gpu_gfx1100" SYCLNN_ONEMATH_ROOT=/opt/onemath \
#   SYCLNN_CT_BACKENDS=netlib pip install --target /work/fnn-bench/.wheels/sycl-portable .
# (the oneMath run-time loader picks cuBLAS / rocBLAS / NETLIB by device at run time; the rocBLAS
# backend of oneMath is not built in this image, so blas=tiled is the device-independent path
# used for the portability rows, next to the vendor rows where a backend exists)
