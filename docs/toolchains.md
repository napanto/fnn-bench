# Toolchain matrix

Status of every compiler × device combination used by the study. "OK" means the
shared parity suite (`fnn-testkit`, 117 tests × float/double) passes; "build"
means it compiles but no device to run on; "-" not applicable. Updated as the
the local runs work progresses (see `the study plan`).

Last update: 2026-09-03.

## Images and environments

| Environment | Base | Contents | Built |
|---|---|---|---|
| `localhost/fnn-cuda:dev` (→ `ghcr.io/napanto/fnn-cuda`) | ubuntu 24.04 | CUDA 12.9 (nvcc 12.9.86, cuBLAS, NVTX, CUPTI, Nsight Systems 2025.1.3), gcc-13 (nvcc host), gcc-14 + `gcc-14-offload-nvptx` + `gcc-14-offload-amdgcn`, clang-22 (apt.llvm.org, host OpenMP only), clang-18 (Ubuntu) with `libomptarget` nvptx/amdgpu device runtimes, OpenBLAS 0.3.26 (openmp + pthread), Intel oneMKL 2026.1 from PyPI (`mkl`, `mkl-devel`, `onemkl-sycl-blas`, `onemkl-sycl-include`, `tbb-devel`), Python 3.12 venv (numpy 2.5.2, pytest 9.1, pybind11 3.1.0, scikit-build-core 1.0.3) | 2026-09-03 on ws-amd, 7.6 GB |
| `localhost/fnn-sycl:dev` (→ `ghcr.io/napanto/fnn-sycl`) | fnn-cuda | intel/llvm **v7.1.0** `sycl_linux.tar.gz` (clang 22.1, libsycl 9; adapters: opencl, level_zero, **cuda**), Intel OpenCL CPU runtime **oclcpuexp 2026-WW28** (`sycl-ls` → `[opencl:cpu] AMD Ryzen Threadripper 2950X`), oneMath **v0.9** in `/opt/onemath` (MKLCPU + NETLIB/OpenBLAS + cuBLAS); `CC=clang CXX=clang++` | 2026-09-03 |
| `localhost/fnn-sycl-generic:dev` | fnn-sycl | + oneMath v0.9 with the generic SYCL BLAS backend in `/opt/onemath-generic` (INTEL_CPU tuning = `spir64_x86_64` AOT, ~2 h at -j8) | 2026-09-03 |
| `fnn-rocm` distrobox (ws-amd) | `localhost/rocm-pytorch-distrobox` = Ubuntu 24.04 + ROCm 7.2.4 | hipcc, amdclang++ 22 (ROCm), rocBLAS 5.2 / hipBLAS 3.2, rocprofv3, gcc-13/14 + `gcc-14-offload-amdgcn`, clang-18 + libomp-18 (`libomptarget-amdgpu-gfx1100.bc`), OpenBLAS; `~/.local/opt/fnn-rocm/`: **AdaptiveCpp 25.10.0** (LLVM 18, SSCP `generic` target, ROCm + OpenMP backends), oneMath v0.9 (rocBLAS + NETLIB), Python venv | 2026-09-03, `scripts/rocm-toolchain.sh` |

Build-safety: every image build runs with `podman build --memory=20g`, every
compile with `-j8` (`CMAKE_BUILD_PARALLEL_LEVEL=8`).

## syclnn (SYCL)

| Compiler | Device | BLAS | Status | Notes |
|---|---|---|---|---|
| AdaptiveCpp 25.10 (generic/SSCP) | ws-amd CPU (OpenMP host device) | NETLIB (OpenBLAS-openmp) | **OK** double + float, all 15 ablation switches | `blas=netlib` explicit selection fails on AdaptiveCpp (JIT kernel library load error in oneMath's compile-time dispatch path); `blas=auto` resolves to NETLIB anyway |
| AdaptiveCpp 25.10 (generic/SSCP) | ws-amd RX 7900 XTX (gfx1100, HIP) | rocBLAS 5.2 | **OK** double + float, ablations OK | `memory=host` (zero-copy host USM): 20/118 tests fail with stale reads → not supported on this platform, kept as a result |
| intel/llvm v7.1.0 (DPC++, `-fsycl-targets=spir64,nvidia_gpu_sm_61,nvidia_gpu_sm_80`) | ws-amd CPU (`opencl:cpu`, oclcpuexp 2026-WW28) | MKLCPU 2026.1 (SYCL API, PyPI) | **OK** double + float (`blas=mklcpu` and `auto`) | MKLCPU works with the open-source DPC++ (libsycl 9 ABI = oneAPI 2026.1) |
| intel/llvm v7.1.0 (DPC++) | ws-amd CPU (`opencl:cpu`) | NETLIB (OpenBLAS 0.3.26 openmp) | **OK** double + float (`blas=netlib`) | |
| intel/llvm v7.1.0 (DPC++) | ws-amd CPU (`opencl:cpu`) | generic SYCL BLAS (oneMath `/opt/onemath-generic`, INTEL_CPU tuning) | **OK** double + float (`blas=generic` and `auto`) | separate wheel/venv: the backend is exclusive in oneMath |
| intel/llvm v7.1.0 (DPC++) | ws-nvidia GTX 1080 Ti (`cuda:gpu`, sm_61, CUDA 12.9) | cuBLAS | pending | |
| intel/llvm v7.1.0 (DPC++) | dept. A30 (sm_80) | cuBLAS | the department runs | |

Known AdaptiveCpp note: its persistent JIT cache (`~/.local/share/acpp/apps/<app>/jit-cache`)
is shared by every process of the same executable; running two test sessions
concurrently corrupted it ("could not load shared kernel library"). Run
sequentially, or give each run its own `ACPP_APPDB_DIR`.

## cudann (CUDA)

| Compiler | Device | Status | Notes |
|---|---|---|---|
| nvcc 12.9 | ws-nvidia 1080 Ti (sm_61) | Phase 2 | |
| nvcc 12.9 / 13.x | dept. A30 (sm_80) | the department runs | |
| hipcc 7.2.4 (`CUDANN_HIP=ON`: hipify-perl + hipBLAS 3.2) | ws-amd 7900 XTX (gfx1100) | **OK** double + float; `memory=device/shared/host`, `queue=in_order/out_of_order/graph`, `streams=1/8`, every kernel ablation | graph mode captures from one stream on HIP (multi-stream fork/join capture segfaults in ROCm 7.2); `memory=host` (mapped zero-copy) works here, unlike SYCL host USM through AdaptiveCpp |

## ompnn (OpenMP)

| Compiler | Target | Status | Notes |
|---|---|---|---|
| gcc-14 | ws-amd CPU | Phase 3 | |
| clang-22 | ws-amd CPU | Phase 3 | |
| amdclang++ (ROCm 7.2.4) | ws-amd 7900 XTX (`--offload-arch=gfx1100`) | Phase 3 | |
| clang-18 (Ubuntu) | ws-amd 7900 XTX | Phase 3 | `libomptarget-amdgpu-gfx1100.bc` present |
| gcc-14-offload-amdgcn | ws-amd 7900 XTX | Phase 3 | |
| gcc-14-offload-nvptx | ws-nvidia 1080 Ti | Phase 3 | |
| clang-18 (`--offload-arch=sm_61`) | ws-nvidia 1080 Ti | Phase 3 | clang-22 from apt.llvm.org ships no offload runtime |
| nvc++ 26.5 (`-mp=gpu`) | ws-nvidia 1080 Ti / A30 | Phase 3 / the department runs | V100+ officially; Pascal attempt documented either way |

## Facts worth remembering

- oneMath v0.9's NETLIB backend redefines `cblas_i?amin` with an `int` return
  type; OpenBLAS's `cblas.h` declares them with `CBLAS_INDEX` → compile error.
  `containers/patches/onemath-netlib-openblas.patch` guards the definitions with
  `#ifndef OPENBLAS_VERSION`.
- oneMath's MKLCPU backend uses the *SYCL* oneMKL API (`libmkl_sycl_blas`), so the
  PyPI packages `onemkl-sycl-blas` + `onemkl-sycl-include` + `mkl-devel`
  (+ `tbb-devel` for `MKLConfig.cmake`) replace the multi-GB apt toolkit.
- oneMath's `FindcuBLAS.cmake` uses the legacy `find_package(CUDA)` and needs
  `-DCUDA_CUDA_LIBRARY=$CUDA_HOME/lib64/stubs/libcuda.so` when no driver is present.
- oneMath's exported CMake targets link `ONEMATH::SYCL::SYCL`, which its
  `FindCompiler.cmake` only defines for DPC++; AdaptiveCpp consumers must define
  it themselves (syclnn's CMakeLists does).
- apt.llvm.org's `libomp-22-dev` conflicts with Ubuntu's `libomp-18-dev`
  (`libomp-x.y-dev`), although all files live under versioned directories;
  `fnn-cuda` unpacks the 18 packages with `dpkg -x` into `/usr/lib/llvm-18`.
- AdaptiveCpp needs `libclang-rt-18-dev` (`libclang_rt.builtins-x86_64.a`) for
  its HIP backend library.
- Ubuntu's `libomp-18-dev` ships `libomptarget.rtl.{cuda,amdgpu}.so` and device
  bitcode for `sm_*` and `gfx*` (incl. gfx1100): clang-18 offloads out of the box.
